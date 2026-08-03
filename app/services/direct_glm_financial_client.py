"""Client Ollama GLM Vision : image → JSON financier ciblé (sans Qwen)."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import time
from typing import Any, Type

import httpx
from pydantic import BaseModel, Field, ValidationError

from app.config import (
    DIRECT_FINANCIAL_KEEP_ALIVE,
    DIRECT_FINANCIAL_MAX_ATTEMPTS,
    DIRECT_FINANCIAL_MODEL,
    DIRECT_FINANCIAL_NUM_CTX,
    DIRECT_FINANCIAL_NUM_PREDICT,
    DIRECT_FINANCIAL_TIMEOUT_SECONDS,
    OLLAMA_URL,
)
from app.schemas.direct_financial_extraction import (
    ALLOWED_FIELD_CODES,
    PAGE_TYPE_SCHEMAS,
    PRIORITY_FIELDS,
    FinancialPageType,
    GlmLiteCandidate,
    GlmLitePageOutput,
)

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_PERIOD_ALIASES = {
    "N-1": "N_MINUS_1",
    "N1": "N_MINUS_1",
    "N_1": "N_MINUS_1",
    "EXERCICE_PRECEDENT": "N_MINUS_1",
    "PREV": "N_MINUS_1",
    "CURRENT": "N",
    "EXERCICE": "N",
}

_COLUMN_ALIASES = {
    "NET": "NET_N",
    "NET_EXERCICE": "NET_N",
    "EXERCICE": "EXERCICE_N",
    "EXERCICE_N": "EXERCICE_N",
    "EXERCICE_PRECEDENT": "EXERCICE_N1",
    "N-1": "EXERCICE_N1",
    "TOTALS": "TOTAL_EXERCICE_N",
    "TOTAUX": "TOTAL_EXERCICE_N",
    "3": "TOTAL_EXERCICE_N",
    "4": "EXERCICE_N1",
}


class DirectFinancialExtractionError(RuntimeError):
    pass


class DirectFinancialLengthError(DirectFinancialExtractionError):
    pass


class _PageTypeClassification(BaseModel):
    page_type: FinancialPageType
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reason: str = Field(default="", max_length=120)


_COMMON_SYSTEM_PROMPT = """
Tu extrais des montants visibles sur une page de liasse fiscale marocaine PCGM.

Réponds UNIQUEMENT en JSON : {"candidates":[...]}
Chaque candidat :
{"field_code":"...","raw_value":"...","period":"N|N_MINUS_1",
 "nature":"DETAIL|SUBTOTAL|SECTION_TOTAL|GRAND_TOTAL",
 "confidence":0.0-1.0,
 "evidence":{"raw_label":"...","column_name":null|"...",
 "column_role":"NET_N|EXERCICE_N|TOTAL_EXERCICE_N|EXERCICE_N1|BRUT|AMORT_PROV|UNKNOWN",
 "source_excerpt":"libellé exact | montant"}}

RÈGLES :
1. N'invente rien. Cellule vide ≠ 0. 0 explicite = valeur.
2. raw_value = montant tel qu'affiché (espaces/virgules OK).
3. evidence.raw_label = TEXTE DE LA LIGNE (jamais le montant seul).
4. source_excerpt = "libellé | montant" (pas "ligne|montant").
5. period N = exercice courant ; N_MINUS_1 = exercice précédent.
6. Max 12 candidats. Priorise les champs demandés.
7. Si tu ne vois aucun montant du schéma, retourne {"candidates":[]}.
""".strip()

_SECTION_RULES: dict[str, str] = {
    "IDENTIFICATION": (
        "Textes d'identité : raison sociale, IF, ICE, adresse, dates. "
        "period=N, nature=DETAIL, column_role=IDENTITY_VALUE."
    ),
    "BILAN_ACTIF": (
        "OBLIGATOIRE si visible : TOTAL_ACTIF = ligne "
        "'TOTAL GENERAL I+II+III' (nature=GRAND_TOTAL, column_role=NET_N). "
        "Puis ACTIFS_IMMOBILISES (TOTAL I), ACTIF_CIRCULANT (TOTAL II), "
        "TRESORERIE_ACTIF (TOTAL III), STOCKS, CLIENTS. "
        "Colonnes : Net = N (NET_N) ; Exercice précédent = N-1."
    ),
    "BILAN_PASSIF": (
        "OBLIGATOIRE si visible : TOTAL_PASSIF = 'TOTAL I+II+III' "
        "(nature=GRAND_TOTAL), FONDS_PROPRES = 'TOTAL DES CAPITAUX PROPRES', "
        "DETTES_FINANCIERES = 'TOTAL DES DETTES DE FINANCEMENT', "
        "PASSIF_CIRCULANT, FOURNISSEURS, TRESORERIE_PASSIF, RESULTAT_NET. "
        "column_role=EXERCICE_N pour l'exercice courant."
    ),
    "CPC": (
        "Colonnes Totaux exercice = N (TOTAL_EXERCICE_N). "
        "CHIFFRE_AFFAIRES obligatoire. "
        "RESULTAT_EXPLOITATION = uniquement la ligne "
        "'Résultat d'exploitation' (PAS Produits d'exploitation). "
        "CHARGES_FINANCIERES = TOTAL V / Charges financières. "
        "RESULTAT_NET ou RESULTAT_NET_XVI, RESULTAT_COURANT."
    ),
    "DETAIL_CPC": "Priorité REDEVANCES_CREDIT_BAIL si visible.",
    "RESULTAT_FISCAL": (
        "Uniquement le tableau de passage résultat comptable → fiscal. "
        "RESULTAT_FISCAL, REINTEGRATIONS, DEDUCTIONS, IS_DU, "
        "COTISATION_MINIMALE, REPORT_DEFICITAIRE (libellé avec 'déficit'). "
        "Ne prends PAS les lignes CPC XIV/XVI."
    ),
    "ESG": "CAF, EBE, VALEUR_AJOUTEE si explicitement affichés.",
}


def schema_for_page_type(page_type: str) -> Type[BaseModel]:
    """Schéma strict (tests / validation riche). L'appel GLM utilise le lite."""
    try:
        return PAGE_TYPE_SCHEMAS[page_type]
    except KeyError as exc:
        raise DirectFinancialExtractionError(
            f"Type de page non extractible : {page_type}"
        ) from exc


def lite_schema_for_page_type(page_type: str) -> Type[BaseModel]:
    if page_type not in PAGE_TYPE_SCHEMAS:
        raise DirectFinancialExtractionError(
            f"Type de page non extractible : {page_type}"
        )
    return GlmLitePageOutput


def prompt_for_page_type(page_type: str) -> str:
    rules = _SECTION_RULES.get(page_type, "")
    priority = ", ".join(PRIORITY_FIELDS.get(page_type, ()))
    return (
        f"{_COMMON_SYSTEM_PROMPT}\n\n"
        f"Type de page : {page_type}\n"
        f"Champs prioritaires : {priority}\n"
        f"Règles : {rules}"
    )


def _clean_model_json(raw: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text.startswith("{"):
        match = _JSON_BLOCK_RE.search(text)
        if match:
            text = match.group(0)
    return text.strip()


def _coerce_period(value: Any) -> str:
    text = str(value or "N").strip().upper().replace(" ", "_")
    return _PERIOD_ALIASES.get(text, text if text in {"N", "N_MINUS_1"} else "N")


def _coerce_column_role(value: Any) -> str:
    text = str(value or "UNKNOWN").strip().upper().replace(" ", "_")
    text = _COLUMN_ALIASES.get(text, text)
    allowed = {
        "IDENTITY_VALUE",
        "BRUT",
        "AMORT_PROV",
        "NET_N",
        "EXERCICE_N",
        "TOTAL_EXERCICE_N",
        "EXERCICE_N1",
        "MONTANT_N",
        "MONTANT_N1",
        "UNKNOWN",
    }
    return text if text in allowed else "UNKNOWN"


def _coerce_nature(value: Any) -> str:
    text = str(value or "DETAIL").strip().upper()
    allowed = {"DETAIL", "SUBTOTAL", "SECTION_TOTAL", "GRAND_TOTAL"}
    return text if text in allowed else "DETAIL"


def _normalize_payload(data: Any) -> dict[str, Any]:
    if isinstance(data, list):
        data = {"candidates": data}
    if not isinstance(data, dict):
        return {"candidates": []}
    cands = data.get("candidates")
    if cands is None and "field_code" in data:
        cands = [data]
    if not isinstance(cands, list):
        cands = []

    normalized: list[dict[str, Any]] = []
    for item in cands:
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence") or {}
        if not isinstance(evidence, dict):
            evidence = {}
        raw_label = (
            evidence.get("raw_label")
            or item.get("raw_label")
            or item.get("label")
            or item.get("field_code")
            or "?"
        )
        raw_value = item.get("raw_value")
        if raw_value is None:
            raw_value = item.get("value")
        if raw_value is None:
            continue
        raw_value = str(raw_value).strip()
        if not raw_value:
            continue
        field_code = str(item.get("field_code") or "").strip().upper()
        if not field_code:
            continue
        # Alias utiles
        if field_code == "TOTAL_GENERAL":
            field_code = "TOTAL_ACTIF"
        if field_code in {"CA", "CHIFFRE_D_AFFAIRES", "CHIFFRE_D'AFFAIRES"}:
            field_code = "CHIFFRE_AFFAIRES"
        if field_code in {"RN", "RESULTAT_NET_DE_L_EXERCICE"}:
            field_code = "RESULTAT_NET"
        if field_code == "CAPITAUX_PROPRES":
            field_code = "FONDS_PROPRES"

        normalized.append(
            {
                "field_code": field_code,
                "raw_value": raw_value[:64],
                "period": _coerce_period(item.get("period")),
                "nature": _coerce_nature(item.get("nature")),
                "confidence": float(item.get("confidence") or 0.7),
                "evidence": {
                    "raw_label": str(raw_label)[:120],
                    "column_name": evidence.get("column_name")
                    or item.get("column_name"),
                    "column_role": _coerce_column_role(
                        evidence.get("column_role") or item.get("column_role")
                    ),
                    "source_excerpt": str(
                        evidence.get("source_excerpt")
                        or item.get("source_excerpt")
                        or f"{raw_label}|{raw_value}"
                    )[:160],
                },
                "warnings": list(item.get("warnings") or []),
            }
        )
    return {"candidates": normalized}


def _validate_lite_content(content: str, page_type: str) -> GlmLitePageOutput:
    cleaned = _clean_model_json(content)
    if not cleaned:
        raise DirectFinancialExtractionError("Réponse GLM vide.")
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise DirectFinancialExtractionError(
            "La réponse GLM n'est pas du JSON valide."
        ) from exc

    normalized = _normalize_payload(data)
    allowed = ALLOWED_FIELD_CODES.get(page_type, frozenset())
    kept: list[dict[str, Any]] = []
    for item in normalized["candidates"]:
        code = item["field_code"]
        if allowed and code not in allowed:
            logger.debug("field_code ignoré page=%s code=%s", page_type, code)
            continue
        kept.append(item)
    normalized["candidates"] = kept

    try:
        return GlmLitePageOutput.model_validate(normalized)
    except ValidationError as exc:
        # Garde les candidats individuels valides
        survivors: list[GlmLiteCandidate] = []
        for item in kept:
            try:
                survivors.append(GlmLiteCandidate.model_validate(item))
            except ValidationError:
                continue
        return GlmLitePageOutput(candidates=survivors)


def _validate_content(content: str, schema_model: type[BaseModel]) -> BaseModel:
    cleaned = _clean_model_json(content)
    if not cleaned:
        raise DirectFinancialExtractionError("Réponse GLM vide.")
    try:
        return schema_model.model_validate_json(cleaned)
    except ValidationError:
        try:
            data = json.loads(cleaned)
            return schema_model.model_validate(data)
        except Exception as exc:  # noqa: BLE001
            raise DirectFinancialExtractionError(
                "La réponse GLM ne respecte pas le JSON Schema."
            ) from exc


def _http_timeout() -> httpx.Timeout:
    read = float(DIRECT_FINANCIAL_TIMEOUT_SECONDS)
    return httpx.Timeout(
        connect=30.0,
        read=read,
        write=max(read, 180.0),
        pool=30.0,
    )


def _downscale_for_classify(image_bytes: bytes, max_side: int = 960) -> bytes:
    """Image plus légère pour la classification (évite 504 gateway)."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=75)
        return out.getvalue()


def _image_as_jpeg(image_bytes: bytes, *, max_side: int | None = None, quality: int = 88) -> bytes:
    """Normalise en JPEG lisible (meilleure compat Ollama vision que PNG géant)."""
    import io

    from PIL import Image, ImageEnhance, ImageOps

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            if max_side and max(img.size) > max_side:
                img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            # Léger contraste pour tableaux scannés
            img = ImageOps.autocontrast(img, cutoff=1)
            img = ImageEnhance.Sharpness(img).enhance(1.15)
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=quality, optimize=True)
            return out.getvalue()
    except Exception:  # noqa: BLE001
        # Bytes non-image (tests) : renvoyer tel quel
        return image_bytes


async def warmup_direct_financial_model() -> None:
    """Préchauffe le modèle GLM pour éviter les 504 de cold-start."""
    payload = {
        "model": DIRECT_FINANCIAL_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "keep_alive": DIRECT_FINANCIAL_KEEP_ALIVE,
        "options": {"temperature": 0, "num_predict": 1},
    }
    try:
        async with httpx.AsyncClient(timeout=_http_timeout()) as client:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        logger.info(
            "Warmup GLM direct status=%s model=%s",
            response.status_code,
            DIRECT_FINANCIAL_MODEL,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warmup GLM direct échoué (non bloquant) : %s", exc)


async def classify_page_with_glm(
    image_bytes: bytes,
    *,
    discourage_identification: bool = False,
    force_financial_hint: bool = False,
    previous_page_type: FinancialPageType | None = None,
    page_number: int | None = None,
) -> FinancialPageType:
    """Mini-appel GLM pour classer une page scannée sans texte natif."""
    light = _downscale_for_classify(image_bytes)
    encoded = base64.b64encode(light).decode("ascii")
    schema = _PageTypeClassification.model_json_schema()

    system = (
        "Tu classifies une page de liasse fiscale marocaine PCGM. "
        "Choisis exactement un page_type parmi : "
        "IDENTIFICATION, BILAN_ACTIF, BILAN_PASSIF, CPC, "
        "RESULTAT_FISCAL, ESG, DETAIL_CPC, AUTRE, VIDE.\n"
        "Règles visuelles :\n"
        "- Tableau Brut / Amortissements / Net → BILAN_ACTIF\n"
        "- Capitaux propres / Dettes de financement / Passif circulant "
        "→ BILAN_PASSIF\n"
        "- Compte de produits et charges / Chiffre d'affaires → CPC\n"
        "- Détail des postes du CPC / Redevances → DETAIL_CPC\n"
        "- Passage résultat comptable → fiscal → RESULTAT_FISCAL\n"
        "- État des soldes de gestion / CAF → ESG\n"
        "- IDENTIFICATION uniquement si page d'identité du contribuable "
        "(raison sociale, IF, ICE) SANS grand tableau financier\n"
        "- VIDE = page blanche ; AUTRE = annexe admin non financière\n"
        "Ordre typique d'une liasse : IDENTIFICATION → BILAN_ACTIF → "
        "BILAN_PASSIF → CPC → DETAIL_CPC → RESULTAT_FISCAL / ESG.\n"
        "Ne répète pas le type précédent sauf si la page est clairement "
        "la suite du même état.\n"
        "Réponds uniquement en JSON."
    )
    user = "Quel est le type de cette page ?"
    if page_number is not None:
        user += f" Numéro de page : {page_number}."
    if previous_page_type:
        user += (
            f" Page précédente classée : {previous_page_type}. "
            "Utilise cela seulement comme indice d'ordre, pas comme "
            "copie automatique."
        )
    if discourage_identification:
        user += (
            " Attention : la page précédente était déjà IDENTIFICATION "
            "ou tu es après la page 1. "
            "N'utilise IDENTIFICATION que si c'est vraiment encore la page "
            "d'identité. Si tu vois un tableau financier, choisis BILAN_ACTIF, "
            "BILAN_PASSIF, CPC, DETAIL_CPC, RESULTAT_FISCAL ou ESG."
        )
    if force_financial_hint:
        user += (
            " Cette page contient très probablement un état financier "
            "(bilan ou CPC). Ne réponds PAS IDENTIFICATION."
        )
        if previous_page_type == "IDENTIFICATION":
            user += " Après identification, essaie d'abord BILAN_ACTIF."
        elif previous_page_type == "BILAN_ACTIF":
            user += " Après bilan actif, essaie d'abord BILAN_PASSIF."
        elif previous_page_type == "BILAN_PASSIF":
            user += " Après bilan passif, essaie d'abord CPC."
        elif previous_page_type == "CPC":
            user += " Après CPC, essaie DETAIL_CPC ou RESULTAT_FISCAL."

    payload = {
        "model": DIRECT_FINANCIAL_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": user,
                "images": [encoded],
            },
        ],
        "format": schema,
        "stream": False,
        "keep_alive": DIRECT_FINANCIAL_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_ctx": min(DIRECT_FINANCIAL_NUM_CTX, 4096),
            "num_predict": 128,
        },
    }

    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=_http_timeout()) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json=payload,
                )

            if response.status_code in {500, 502, 503, 504}:
                last_error = DirectFinancialExtractionError(
                    f"Classification GLM HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                logger.warning(
                    "Classif GLM tentative %d/3 : %s",
                    attempt,
                    last_error,
                )
                await asyncio.sleep(min(3 * attempt, 10))
                continue

            if response.status_code >= 400:
                raise DirectFinancialExtractionError(
                    f"Classification GLM HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )

            body = response.json()
            content = ((body.get("message") or {}).get("content") or "").strip()
            parsed = _validate_content(content, _PageTypeClassification)
            return parsed.page_type  # type: ignore[return-value]
        except DirectFinancialExtractionError as exc:
            last_error = exc
            await asyncio.sleep(min(3 * attempt, 10))
        except httpx.HTTPError as exc:
            last_error = DirectFinancialExtractionError(str(exc))
            await asyncio.sleep(min(3 * attempt, 10))

    raise DirectFinancialExtractionError(
        f"Classification GLM impossible : {last_error}"
    )


async def extract_financial_page(
    image_bytes: bytes,
    *,
    page_number: int,
    page_type: str,
    orientation: int,
    schema_model: type[BaseModel] | None = None,
    system_prompt: str | None = None,
    max_attempts: int | None = None,
) -> tuple[GlmLitePageOutput, int]:
    """Envoie l'image à GLM Vision ; valide contre le schéma LITE."""
    del schema_model  # API historique — on force le schéma lite
    used_prompt = system_prompt or prompt_for_page_type(page_type)
    attempts = max_attempts or DIRECT_FINANCIAL_MAX_ATTEMPTS
    # JPEG + contraste : meilleure lecture des tableaux scannés
    from app.config import DIRECT_FINANCIAL_MAX_IMAGE_DIMENSION

    jpeg_bytes = _image_as_jpeg(
        image_bytes,
        max_side=DIRECT_FINANCIAL_MAX_IMAGE_DIMENSION,
        quality=88,
    )
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    timeout = _http_timeout()
    lite_schema = GlmLitePageOutput.model_json_schema()
    priority = ", ".join(PRIORITY_FIELDS.get(page_type, ()))

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        # Tentative 1 : schéma lite ; tentative 2 : format json libre
        use_schema = attempt == 1
        user_content = (
            f"Page {page_number}. Type : {page_type}. Orientation {orientation}°. "
            f"Extrais en priorité : {priority}. "
            "Lis les montants dans les colonnes Net / Totaux exercice / Exercice. "
            'Réponds uniquement JSON {"candidates":[...]}.'
        )
        payload = {
            "model": DIRECT_FINANCIAL_MODEL,
            "messages": [
                {"role": "system", "content": used_prompt},
                {
                    "role": "user",
                    "content": user_content,
                    "images": [encoded],
                },
            ],
            "format": lite_schema if use_schema else "json",
            "stream": False,
            "keep_alive": DIRECT_FINANCIAL_KEEP_ALIVE,
            "options": {
                "temperature": 0,
                "num_ctx": DIRECT_FINANCIAL_NUM_CTX,
                "num_predict": DIRECT_FINANCIAL_NUM_PREDICT,
            },
        }

        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{OLLAMA_URL}/api/chat",
                    json=payload,
                )

            if response.status_code in {500, 502, 503, 504}:
                raise DirectFinancialExtractionError(
                    f"Ollama HTTP {response.status_code}: {response.text[:180]}"
                )
            if response.status_code == 404:
                raise DirectFinancialExtractionError(
                    f"Modèle '{DIRECT_FINANCIAL_MODEL}' introuvable sur Ollama."
                )
            response.raise_for_status()
            body = response.json()

            done_reason = body.get("done_reason")
            logger.info(
                (
                    "GLM direct page=%d type=%s attempt=%d/%d "
                    "done_reason=%r eval_count=%r prompt_eval_count=%r"
                ),
                page_number,
                page_type,
                attempt,
                attempts,
                done_reason,
                body.get("eval_count"),
                body.get("prompt_eval_count"),
            )

            if done_reason == "length":
                raise DirectFinancialLengthError(
                    f"Réponse tronquée page={page_number}, type={page_type}."
                )

            content = (
                (body.get("message") or {}).get("content") or ""
            ).strip()
            thinking = (body.get("message") or {}).get("thinking") or ""
            logger.info(
                "GLM response chars=%d thinking_chars=%d preview=%r",
                len(content),
                len(str(thinking)),
                content[:240],
            )
            if not content:
                raise DirectFinancialExtractionError("Réponse GLM vide.")

            result = _validate_lite_content(content, page_type)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "GLM lite page=%d type=%s → %d candidats (%dms)",
                page_number,
                page_type,
                len(result.candidates),
                elapsed_ms,
            )
            return result, elapsed_ms

        except DirectFinancialLengthError:
            raise
        except httpx.HTTPError as exc:
            last_error = DirectFinancialExtractionError(str(exc))
        except DirectFinancialExtractionError as exc:
            last_error = exc

        logger.warning(
            (
                "GLM direct échec page=%d type=%s "
                "tentative=%d/%d : %s"
            ),
            page_number,
            page_type,
            attempt,
            attempts,
            last_error,
        )
        if attempt < attempts:
            await asyncio.sleep(min(2**attempt, 6))

    raise DirectFinancialExtractionError(
        f"Extraction impossible page={page_number} type={page_type} : {last_error}"
    )
