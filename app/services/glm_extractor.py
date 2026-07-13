"""Extraction structurée des champs d'une CIN via le modèle GLM vision (Ollama).

S'appuie sur `vision_client` pour l'appel bas niveau (image → JSON) — voir ce
module pour la logique de communication avec Ollama.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

import httpx
from PIL import Image

from app.config import OLLAMA_MODEL, OLLAMA_URL, REQUEST_TIMEOUT_SECONDS
from app.schemas import CinData
from app.services.vision_client import VisionExtractionError, vision_chat_json

logger = logging.getLogger(__name__)

# Alias conservé pour compatibilité avec les imports existants (routers/cin.py).
GlmExtractionError = VisionExtractionError

_SYSTEM_PROMPT = """\
Tu es un expert en lecture de cartes d'identité nationales (CIN) marocaines.
Analyse attentivement l'image fournie (recto OU verso de la carte) et extrait \
UNIQUEMENT les informations suivantes.

Réponds STRICTEMENT avec un objet JSON valide, sans texte additionnel, sans balises \
markdown, exactement dans ce format :
{
  "nom": "<nom de famille, en majuscules>",
  "prenom": "<prénom>",
  "cin": "<numéro de la carte, ex: BE123456>",
  "date_naissance": "<date de naissance au format JJ/MM/AAAA>",
  "lieu_naissance": "<lieu de naissance tel qu'il apparaît sur la carte>",
  "date_expiration": "<date d'expiration / de validité de la carte au format JJ/MM/AAAA>",
  "adresse": "<adresse du domicile telle qu'écrite sur la carte>"
}

Règles strictes :
- Si un champ est illisible ou absent de L'IMAGE FOURNIE, retourne une chaîne vide "" \
pour ce champ plutôt que d'inventer une valeur — NE PAS DEVINER.
- Ne traduis aucune valeur, conserve l'orthographe exacte visible sur la carte.
- Sur une CIN marocaine : le RECTO contient nom, prénom, numéro CIN, date et lieu de \
naissance, date d'expiration ; le VERSO contient l'adresse du domicile. Si l'image \
fournie ne montre qu'une seule face, les champs de l'autre face doivent rester "".
- Ne retourne rien d'autre que ce JSON (pas d'explication, pas de commentaire).
"""


def _clean(value: Any) -> str:
    return str(value).strip() if value else ""


async def extract_cin_fields(image_bytes: bytes) -> tuple[CinData, float]:
    """Envoie l'image à Ollama (modèle GLM vision) et retourne les champs structurés.

    Returns:
        (CinData, temps_écoulé_en_ms)
    """
    parsed, elapsed_ms = await vision_chat_json(
        image_bytes,
        system_prompt=_SYSTEM_PROMPT,
        user_message="Voici la carte d'identité nationale à analyser.",
    )

    data = CinData(
        nom=_clean(parsed.get("nom")),
        prenom=_clean(parsed.get("prenom")),
        cin=_clean(parsed.get("cin")).upper(),
        date_naissance=_clean(parsed.get("date_naissance")),
        lieu_naissance=_clean(parsed.get("lieu_naissance")),
        date_expiration=_clean(parsed.get("date_expiration")),
        adresse=_clean(parsed.get("adresse")),
    )
    return data, elapsed_ms


def _filled_count(data: CinData) -> int:
    return sum(1 for value in data.model_dump().values() if value)


def _rotate_jpeg(image_bytes: bytes, degrees: int) -> bytes:
    """Fait tourner l'image de `degrees` (sens horaire) et réencode en JPEG.

    Utilisé en dernier recours quand une face de CIN a été photographiée de
    travers (carte tournée) : le modèle vision lit mal le texte pivoté.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        rotated = img.convert("RGB").rotate(-degrees, expand=True)
        buffer = io.BytesIO()
        rotated.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()


async def extract_cin_fields_auto_rotate(image_bytes: bytes) -> tuple[CinData, float]:
    """Comme `extract_cin_fields`, mais retente automatiquement avec l'image
    pivotée (90°/180°/270°) si le premier essai ne détecte STRICTEMENT rien.

    Cas visé : une face de la carte photographiée de travers (le texte, y
    compris l'adresse au verso, apparaît alors pivoté et devient difficile à
    lire pour le modèle vision dans l'orientation d'origine).
    """
    data, elapsed_ms = await extract_cin_fields(image_bytes)
    if _filled_count(data) > 0:
        return data, elapsed_ms

    total_elapsed = elapsed_ms
    best_data, best_count = data, 0
    for degrees in (90, 180, 270):
        try:
            rotated_bytes = await asyncio.to_thread(_rotate_jpeg, image_bytes, degrees)
            rotated_data, rotated_elapsed = await extract_cin_fields(rotated_bytes)
        except GlmExtractionError as exc:
            logger.warning("Extraction après rotation %s° échouée : %s", degrees, exc)
            continue
        total_elapsed += rotated_elapsed
        count = _filled_count(rotated_data)
        if count > best_count:
            best_data, best_count = rotated_data, count
        if count > 0:
            logger.info("Image récupérée après rotation de %s°.", degrees)
            break

    return best_data, total_elapsed


def merge_cin_sides(*sides: CinData) -> CinData:
    """Fusionne les champs extraits de plusieurs faces d'une même CIN.

    Pour chaque champ, on garde la première valeur non vide rencontrée —
    en pratique le recto est passé en premier, donc ses champs (nom, prénom,
    numéro, dates, lieu de naissance) priment, et l'adresse vient du verso
    dès que le recto ne la fournit pas (ce qui est le cas normal).
    """
    merged: dict[str, str] = {}
    for side in sides:
        for field, value in side.model_dump().items():
            if value and not merged.get(field):
                merged[field] = value
    return CinData(**merged)


async def warmup_model() -> None:
    """Envoie une requête minimale pour précharger le modèle vision en mémoire.

    Sur un serveur Ollama distant (GPU cloud), le premier appel déclenche le
    chargement du modèle depuis le disque, ce qui peut dépasser le timeout
    d'un éventuel reverse proxy. Ce préchargement au démarrage évite que la
    première requête utilisateur n'essuie ce délai.
    """
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": "ok", "stream": False},
            )
        logger.info("Modèle GLM vision préchargé (%s).", OLLAMA_MODEL)
    except Exception as exc:
        logger.warning("Préchargement du modèle GLM vision échoué : %s", exc)
