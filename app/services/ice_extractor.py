"""Extraction structurée des champs d'un certificat ICE (Identifiant Commun de
l'Entreprise) marocain, via un modèle GLM servi par Ollama.

Deux chemins possibles, choisis par le router selon le type de document :
- `extract_ice_fields_from_image`  : image ou page de PDF scanné → modèle
  vision (mêmes mécanismes que la CIN, via `vision_client`).
- `extract_ice_fields_from_text`   : texte natif d'un PDF numérique
  (extrait par `text_extractor.extract_pdf_native_text`) → modèle texte,
  via le client officiel `ollama` (`AsyncClient`).
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ollama import AsyncClient, ResponseError

from app.config import OLLAMA_TEXT_MODEL, OLLAMA_URL, REQUEST_TIMEOUT_SECONDS
from app.schemas import IceData
from app.services.vision_client import (
    VisionExtractionError,
    extract_json_block,
    vision_chat_json,
)

logger = logging.getLogger(__name__)

# Champs communs aux deux prompts (texte et vision) — décrits une seule fois
# pour garder les deux prompts cohérents entre eux.
_FIELDS_DESCRIPTION = """\
{
  "ICE": "<identifiant commun de l'entreprise, 15 chiffres>",
  "Denomination": "<raison sociale / dénomination de l'entreprise>",
  "Identifiant_Fiscal": "<numéro d'identifiant fiscal (IF)>",
  "RC_Numero": "<numéro du registre de commerce UNIQUEMENT, sans la ville>",
  "RC_Ville": "<ville du registre de commerce UNIQUEMENT>",
  "CNSS": "<numéro d'affiliation CNSS>"
}"""

_RULES = """\
Règles strictes :
- Le numéro RC apparaît sous une forme du type "25725/MEKNES" : tu DOIS \
impérativement séparer le numéro ("25725") de la ville ("MEKNES") dans les deux \
champs distincts RC_Numero et RC_Ville.
- Si un champ est illisible ou absent, retourne une chaîne vide "".
- Ne traduis, n'invente et ne reformate aucune valeur : conserve les chiffres exacts.
- Ne retourne rien d'autre que ce JSON."""

_TEXT_SYSTEM_PROMPT = f"""\
Tu es un expert en lecture de documents administratifs marocains, spécialisé dans le \
"Certificat de l'Identifiant Commun de l'Entreprise" (ICE).

Analyse le texte OCR fourni par l'utilisateur et réponds STRICTEMENT avec un objet \
JSON valide, sans aucun texte avant ou après, sans balises markdown, exactement dans \
ce format :
{_FIELDS_DESCRIPTION}

{_RULES}
"""

_VISION_SYSTEM_PROMPT = f"""\
Tu es un expert en lecture de documents administratifs marocains, spécialisé dans le \
"Certificat de l'Identifiant Commun de l'Entreprise" (ICE).

Analyse attentivement l'image fournie et réponds STRICTEMENT avec un objet JSON \
valide, sans aucun texte avant ou après, sans balises markdown, exactement dans ce \
format :
{_FIELDS_DESCRIPTION}

{_RULES}
"""


class IceExtractionError(RuntimeError):
    """Levée quand Ollama / le modèle GLM ne répond pas correctement."""


def _clean(value: Any) -> str:
    return str(value).strip() if value else ""


def _build_ice_data(parsed: dict[str, Any]) -> IceData:
    return IceData(
        ICE=_clean(parsed.get("ICE")),
        Denomination=_clean(parsed.get("Denomination")),
        Identifiant_Fiscal=_clean(parsed.get("Identifiant_Fiscal")),
        RC_Numero=_clean(parsed.get("RC_Numero")),
        RC_Ville=_clean(parsed.get("RC_Ville")),
        CNSS=_clean(parsed.get("CNSS")),
    )


async def extract_ice_fields_from_image(image_bytes: bytes) -> tuple[IceData, float]:
    """Envoie une image (certificat ICE ou page de PDF scanné) au modèle GLM
    vision et retourne les champs structurés.

    Returns:
        (IceData, temps_écoulé_en_ms)
    """
    try:
        parsed, elapsed_ms = await vision_chat_json(
            image_bytes,
            system_prompt=_VISION_SYSTEM_PROMPT,
            user_message="Voici le certificat ICE à analyser.",
        )
    except VisionExtractionError as exc:
        raise IceExtractionError(str(exc)) from exc

    return _build_ice_data(parsed), elapsed_ms


async def extract_ice_fields_from_text(ocr_text: str) -> tuple[IceData, float]:
    """Envoie le texte natif (extrait d'un PDF numérique) au modèle GLM texte
    via le client officiel `ollama` et retourne les champs structurés.

    Returns:
        (IceData, temps_écoulé_en_ms)
    """
    client = AsyncClient(host=OLLAMA_URL, timeout=REQUEST_TIMEOUT_SECONDS)

    started = time.perf_counter()
    try:
        response = await client.chat(
            model=OLLAMA_TEXT_MODEL,
            messages=[
                {"role": "system", "content": _TEXT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Texte OCR extrait du document :\n\n{ocr_text[:6000]}",
                },
            ],
            format="json",
            options={"temperature": 0},
        )
    except ResponseError as exc:
        if "not found" in str(exc).lower():
            raise IceExtractionError(
                f"Modèle '{OLLAMA_TEXT_MODEL}' introuvable sur Ollama. "
                f"Lancez `ollama pull {OLLAMA_TEXT_MODEL}` puis réessayez."
            ) from exc
        raise IceExtractionError(f"Ollama a répondu avec une erreur : {exc}") from exc
    except Exception as exc:
        raise IceExtractionError(
            f"Impossible de contacter Ollama sur {OLLAMA_URL}. "
            "Vérifiez qu'Ollama est démarré (`ollama serve`)."
        ) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000

    raw_content = response.message.content or ""
    if not raw_content:
        raise IceExtractionError("Réponse vide du modèle.")

    parsed = extract_json_block(raw_content)
    return _build_ice_data(parsed), elapsed_ms


async def warmup_model() -> None:
    """Envoie une requête minimale pour précharger le modèle texte en mémoire
    (voir `glm_extractor.warmup_model` pour la justification détaillée)."""
    try:
        client = AsyncClient(host=OLLAMA_URL, timeout=REQUEST_TIMEOUT_SECONDS)
        await client.generate(model=OLLAMA_TEXT_MODEL, prompt="ok", stream=False)
        logger.info("Modèle GLM texte préchargé (%s).", OLLAMA_TEXT_MODEL)
    except Exception as exc:
        logger.warning("Préchargement du modèle GLM texte échoué : %s", exc)
