"""OCR vision — extraction des liasses fiscales scannées (PCGM).

Optimisé pour serveur Ollama distant : images réduites, traitement séquentiel,
retries avec backoff, 2e passe sur les pages en échec, parsing JSON tolérant.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import time
from typing import Any

import fitz  # pymupdf
import httpx
from PIL import Image

from app.config import (
    MAX_IMAGE_DIMENSION,
    OCR_FAILED_PASS_DELAY_SECONDS,
    OCR_MAX_CONCURRENCY,
    OCR_MAX_PAGES,
    OCR_PAGE_DELAY_SECONDS,
    OCR_RETRY_ATTEMPTS,
    OCR_RETRY_DELAY_SECONDS,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    OLLAMA_VISION_MODEL,
    PDF_TO_IMAGE_DPI,
)
from app.schemas.liasse import (
    FinancialElement,
    LiasseExtractionResult,
    RawComponent,
    ScoringInput,
)
from app.services.liasse_extraction import ELEMENTS_19, SCORING_METRICS

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_VISION_CODES = (
    [code for _, code, _, _ in ELEMENTS_19 if code != "TYPE_RESULTAT"]
    + list(SCORING_METRICS)
)
_VALID_SECTIONS = {"BILAN_ACTIF", "BILAN_PASSIF", "CPC"}

_CODE_PAGE_PREFERENCE: dict[str, str] = {
    "ACTIFS_IMMOBILISES": "BILAN_ACTIF",
    "TOTAL_BILAN": "BILAN_ACTIF",
    "ACTIF_CIRCULANT": "BILAN_ACTIF",
    "CREANCES_CLIENTS": "BILAN_ACTIF",
    "TRESORERIE_ACTIF": "BILAN_ACTIF",
    "CAISSE": "BILAN_ACTIF",
    "DETTES_BANCAIRES_MLT": "BILAN_PASSIF",
    "DETTES_BANCAIRES_CT": "BILAN_PASSIF",
    "PASSIF_CIRCULANT": "BILAN_PASSIF",
    "DETTES_FOURNISSEURS": "BILAN_PASSIF",
    "COMPTE_COURANT_ASSOCIES": "BILAN_PASSIF",
    "TRESORERIE_PASSIF": "BILAN_PASSIF",
    "CHIFFRE_AFFAIRES": "CPC",
    "CA_EXPORT": "CPC",
    "ACHATS_REVENDUS": "CPC",
    "AUTRES_CHARGES": "CPC",
    "CHARGES_INTERETS": "CPC",
    "RESULTAT_NET": "CPC",
    "FONDS_PROPRES": "BILAN_PASSIF",
    "CAF": "CPC",
    "FDR": "BILAN_PASSIF",
    "CA_N1": "CPC",
    "AMORTISSEMENTS": "CPC",
    "ENCOURS_LEASING": "BILAN_PASSIF",
    "CMT": "BILAN_PASSIF",
}

_ELEMENTS_JSON_KEYS = ", ".join(f'"{c}": null' for c in _VISION_CODES)

_SYSTEM_PROMPT = f"""Tu es un expert-comptable PCGM (liasse fiscale marocaine).
Analyse cette page et retourne UNIQUEMENT un JSON valide (pas de markdown) :
{{
  "page_type": "BILAN_ACTIF" | "BILAN_PASSIF" | "CPC" | "AUTRE",
  "elements": {{ {_ELEMENTS_JSON_KEYS} }},
  "empty_fields": ["CODE_DU_POSTE"]
}}

Types de page :
- BILAN_ACTIF : tableau bilan actif (immobilisations, actif circulant, trésorerie actif)
- BILAN_PASSIF : tableau bilan passif (capitaux propres, dettes, fournisseurs, trésorerie passif)
- CPC : compte de produits et charges (CA, achats, charges, résultat net)
- AUTRE : page de garde, ESG, annexe

Règles :
- Montants en MAD, nombres JSON (point décimal). Colonne « Net exercice » pour les bilans.
- null si le poste n'est PAS visible sur cette page. Ne devine jamais.
- `empty_fields` contient seulement les codes dont le libellé est clairement
  visible mais dont la case de valeur est réellement vide. Ne mets pas un code
  dans cette liste s'il n'apparaît pas sur la page.
- FONDS_PROPRES = total des capitaux propres ; CAF = capacité d'autofinancement
  si affichée (souvent ESG) ; FDR = fonds de roulement si affiché ; CA_N1 =
  chiffre d'affaires de l'exercice précédent ; AMORTISSEMENTS = dotations aux
  amortissements/provisions ; ENCOURS_LEASING et CMT seulement si explicitement
  affichés. NOUVEAU_FINANCEMENT est externe au document : toujours null.
- Synonymes PCGM à reconnaître :
  - fonds propres : « capitaux propres », « total des capitaux propres »,
    « capitaux propres assimilés » ;
  - dettes financières : « dettes de financement », « emprunts et dettes
    assimilées », « emprunts obligataires », « autres dettes de financement » ;
  - dettes CT : « crédits de trésorerie », « concours bancaires courants »,
    « banques soldes créditeurs » ;
  - fournisseurs : « fournisseurs et comptes rattachés », « fournisseurs » ;
  - clients : « clients et comptes rattachés », « créances de l'actif
    circulant » ;
  - CA : « chiffre d'affaires », « chiffres d'affaires », « ventes de biens
    et services produits » ;
  - achats : « achats revendus de marchandises », « achats consommés de
    matières et fournitures » ;
  - amortissements : « dotations d'exploitation », « dotations aux
    amortissements et provisions » ;
  - FDR : « fonds de roulement », « fonds de roulement fonctionnel ».
"""

_HTTP_TIMEOUT = httpx.Timeout(
    connect=30.0,
    read=OLLAMA_TIMEOUT_SECONDS,
    write=30.0,
    pool=30.0,
)

_model_warmed = False
# Le serveur Ollama distant ne supporte pas de façon fiable plusieurs
# documents vision simultanés. Une file par processus évite que deux uploads
# se provoquent mutuellement des 504.
_OCR_JOB_LOCK = asyncio.Lock()


class VisionOcrError(RuntimeError):
    """Levée quand l'OCR vision (rendu PDF ou appel Ollama) échoue."""


def _downscale_jpeg(data: bytes, max_dimension: int, quality: int = 85) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as img:
            needs_resize = max(img.size) > max_dimension
            if needs_resize:
                img = img.convert("RGB")
                img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
            elif quality >= 85:
                return data
            else:
                img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=quality)
            return buffer.getvalue()
    except Exception as exc:
        logger.warning("Redimensionnement impossible : %s", exc)
        return data


def count_pdf_pages(data: bytes) -> int:
    with fitz.open(stream=io.BytesIO(data), filetype="pdf") as doc:
        return len(doc)


def render_pdf_pages(data: bytes, max_pages: int | None = None) -> list[bytes]:
    limit = max_pages or OCR_MAX_PAGES
    try:
        doc = fitz.open(stream=io.BytesIO(data), filetype="pdf")
        matrix = fitz.Matrix(PDF_TO_IMAGE_DPI / 72, PDF_TO_IMAGE_DPI / 72)
        pages = [
            page.get_pixmap(matrix=matrix).tobytes("jpeg")
            for page in doc[: min(len(doc), limit)]
        ]
        doc.close()
    except Exception as exc:
        raise VisionOcrError(f"Rendu des pages impossible : {exc}") from exc

    if not pages:
        raise VisionOcrError("Le PDF ne contient aucune page.")

    return [_downscale_jpeg(page, MAX_IMAGE_DIMENSION) for page in pages]


async def _warmup_model() -> None:
    """Vérifie le modèle puis le préchauffe avant un document.

    GLM-4.6V peut produire une sortie vide sur un prompt de préchauffage
    minimal ; on vérifie donc l'existence du modèle et la disponibilité
    HTTP, puis les appels réels aux pages appliquent les retries/validations.
    """
    global _model_warmed
    if _model_warmed:
        return

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            tags_response = await client.get(f"{OLLAMA_URL}/api/tags")
            tags_response.raise_for_status()
            models = tags_response.json().get("models", [])
            names = {model.get("name") for model in models if model.get("name")}
            if OLLAMA_VISION_MODEL not in names:
                raise VisionOcrError(
                    f"Le modèle « {OLLAMA_VISION_MODEL} » est absent du serveur Ollama."
                )

            response = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_VISION_MODEL,
                    "prompt": "ok",
                    "stream": False,
                    "options": {"num_predict": 1},
                },
            )
            response.raise_for_status()
        _model_warmed = True
        logger.info("Modèle %s préchauffé.", OLLAMA_VISION_MODEL)
    except VisionOcrError:
        raise
    except httpx.HTTPStatusError as exc:
        raise VisionOcrError(
            f"Pré-vérification Ollama HTTP {exc.response.status_code} : "
            f"{exc.response.text[:160]}"
        ) from exc
    except httpx.HTTPError as exc:
        raise VisionOcrError(
            f"Pré-vérification Ollama impossible : {exc}"
        ) from exc
    except Exception as exc:
        raise VisionOcrError(f"Pré-vérification Ollama impossible : {exc}") from exc


def _parse_json_lenient(raw_content: str, page_num: int) -> dict[str, Any]:
    """Parse JSON du modèle avec fallbacks (réponses tronquées / mal formées)."""
    raw = raw_content.strip()
    if not raw.startswith("{"):
        match = _JSON_BLOCK_RE.search(raw)
        raw = match.group(0) if match else raw

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Fallback : extraction champ par champ
    page_type_m = re.search(r'"page_type"\s*:\s*"([A-Z_]+)"', raw, re.IGNORECASE)
    elements: dict[str, Any] = {}
    for code in _VISION_CODES:
        m = re.search(
            rf'"{code}"\s*:\s*(null|[-]?\d+(?:\.\d+)?)',
            raw,
            re.IGNORECASE,
        )
        if m:
            elements[code] = None if m.group(1).lower() == "null" else m.group(1)

    if page_type_m or elements:
        logger.info("JSON partiel récupéré page %d (%d champs).", page_num, len(elements))
        return {
            "page_type": page_type_m.group(1).upper() if page_type_m else "AUTRE",
            "elements": elements,
        }

    raise VisionOcrError(f"JSON illisible page {page_num}.")


async def _vision_call_once(
    image_bytes: bytes,
    page_num: int,
    total_pages: int,
) -> dict[str, Any]:
    b64_image = base64.b64encode(image_bytes).decode("ascii")
    payload: dict[str, Any] = {
        "model": OLLAMA_VISION_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Page {page_num}/{total_pages} — liasse fiscale PCGM marocaine.",
                "images": [b64_image],
            },
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0},
    }

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise VisionOcrError(
            f"Impossible de contacter Ollama ({OLLAMA_URL}). Vérifiez .env."
        ) from exc
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:200]
        raise VisionOcrError(
            f"Ollama HTTP {exc.response.status_code} page {page_num} : {detail}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise VisionOcrError(f"Timeout Ollama page {page_num}.") from exc

    body = response.json()
    raw_content = (body.get("message") or {}).get("content", "").strip()
    if not raw_content:
        raise VisionOcrError(f"Réponse vide page {page_num}.")

    return _parse_json_lenient(raw_content, page_num)


def _is_retryable_error(exc: VisionOcrError) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "504", "502", "503", "500", "timeout", "vide",
            "illisible", "500 level", "gateway",
        )
    )


async def _vision_call(
    image_bytes: bytes,
    page_num: int,
    total_pages: int,
    *,
    max_attempts: int | None = None,
) -> dict[str, Any]:
    attempts = max_attempts or OCR_RETRY_ATTEMPTS
    last_exc: VisionOcrError | None = None

    for attempt in range(attempts):
        img = image_bytes
        if attempt >= 1:
            img = _downscale_jpeg(image_bytes, max(800, 1100 - attempt * 150), quality=75)

        try:
            return await _vision_call_once(img, page_num, total_pages)
        except VisionOcrError as exc:
            last_exc = exc
            if attempt < attempts - 1 and _is_retryable_error(exc):
                delay = OCR_RETRY_DELAY_SECONDS * (attempt + 1)
                logger.info(
                    "Retry page %d (%d/%d) dans %.0fs : %s",
                    page_num, attempt + 2, attempts, delay, exc,
                )
                await asyncio.sleep(delay)
            else:
                raise

    raise last_exc or VisionOcrError(f"Échec page {page_num}")


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return None if abs(v) > 1e12 else v
    if isinstance(value, str):
        cleaned = value.strip().replace(" ", "").replace("\u00a0", "").replace(",", ".")
        try:
            v = float(cleaned)
            return None if abs(v) > 1e12 else v
        except ValueError:
            return None
    return None


def _merge_page_results(
    page_results: list[tuple[int, dict[str, Any] | None, str | None]],
) -> tuple[dict[str, float], set[str], set[str], list[str], int]:
    best: dict[str, tuple[float, int]] = {}
    sections_found: set[str] = set()
    empty_codes: set[str] = set()
    page_errors: list[str] = []
    pages_used = 0

    for idx, result, error in sorted(page_results, key=lambda r: r[0]):
        if error:
            page_errors.append(f"Page {idx + 1} : {error}")
            continue
        if not result:
            continue
        pages_used += 1
        page_type = (result.get("page_type") or "AUTRE").upper()
        elements = result.get("elements") or {}
        empty_codes.update(
            code for code in result.get("empty_fields", []) if code in _VISION_CODES
        )
        page_has_value = False

        for code, raw_value in elements.items():
            if code not in _VISION_CODES:
                continue
            val = _to_float(raw_value)
            if val is None:
                continue
            page_has_value = True
            preferred = _CODE_PAGE_PREFERENCE.get(code, "")
            priority = 2 if page_type == preferred else (1 if page_type in _VALID_SECTIONS else 0)
            current = best.get(code)
            if current is None or priority > current[1]:
                best[code] = (val, priority)
            elif priority == current[1] and abs(val) > abs(current[0]):
                best[code] = (val, priority)

        if page_has_value and page_type in _VALID_SECTIONS:
            sections_found.add(page_type)

    return (
        {code: pair[0] for code, pair in best.items()},
        sections_found,
        empty_codes,
        page_errors,
        pages_used,
    )


def _build_result(
    values: dict[str, float],
    sections_found: set[str],
    empty_codes: set[str],
    page_errors: list[str],
    pages_used: int,
    pages_total: int,
    elapsed_ms: int,
    filename: str | None,
) -> LiasseExtractionResult:
    financial_elements: list[FinancialElement] = []
    for num, code, label, source in ELEMENTS_19:
        if code == "TYPE_RESULTAT":
            resultat_net = values.get("RESULTAT_NET")
            note = None
            if resultat_net is not None:
                note = (
                    "Bénéficiaire" if resultat_net > 0
                    else ("Déficitaire" if resultat_net < 0 else "Nul")
                )
            financial_elements.append(
                FinancialElement(
                    number=num, code=code, label=label, value=None,
                    source=source, note=note,
                    confidence=0.8 if resultat_net is not None else 0.0,
                    detection_status="derived" if resultat_net is not None else "not_detected",
                )
            )
            continue
        val = values.get(code)
        financial_elements.append(
            FinancialElement(
                number=num, code=code, label=label, value=val,
                source=source, confidence=0.8 if val is not None else 0.0,
                detection_status=(
                    "detected" if val is not None
                    else ("empty" if code in empty_codes else "not_detected")
                ),
            )
        )

    treso_actif = values.get("TRESORERIE_ACTIF")
    treso_passif = values.get("TRESORERIE_PASSIF")
    tresorerie_nette = (
        treso_actif - treso_passif
        if treso_actif is not None and treso_passif is not None
        else None
    )
    raw_components = [
        RawComponent(label=label, value=values[code], source=source, feeds=feeds)
        for code, (label, source, feeds) in SCORING_METRICS.items()
        if values.get(code) is not None
    ]

    scoring_input = ScoringInput(
        chiffre_affaires=values.get("CHIFFRE_AFFAIRES"),
        ca_export=values.get("CA_EXPORT"),
        ca_n1=values.get("CA_N1"),
        total_bilan=values.get("TOTAL_BILAN"),
        fonds_propres=values.get("FONDS_PROPRES"),
        actifs_immobilises=values.get("ACTIFS_IMMOBILISES"),
        actif_circulant=values.get("ACTIF_CIRCULANT"),
        clients=values.get("CREANCES_CLIENTS"),
        fournisseurs=values.get("DETTES_FOURNISSEURS"),
        dettes_financieres=values.get("DETTES_BANCAIRES_MLT"),
        dettes_bancaires_ct=values.get("DETTES_BANCAIRES_CT"),
        passif_circulant=values.get("PASSIF_CIRCULANT"),
        tresorerie_actif=treso_actif,
        tresorerie_passif=treso_passif,
        tresorerie_nette=tresorerie_nette,
        achats=values.get("ACHATS_REVENDUS"),
        frais_financiers=values.get("CHARGES_INTERETS"),
        amortissements=values.get("AMORTISSEMENTS"),
        caf=values.get("CAF"),
        fdr=values.get("FDR"),
        resultat_net=values.get("RESULTAT_NET"),
        compte_courant_associes=values.get("COMPTE_COURANT_ASSOCIES"),
        encours_leasing=values.get("ENCOURS_LEASING"),
        cmt=values.get("CMT"),
        nouveau_financement=values.get("NOUVEAU_FINANCEMENT"),
    )

    sections_completeness = {s: s in sections_found for s in _VALID_SECTIONS}
    complete_count = sum(
        1 for el in financial_elements if el.value is not None and el.confidence > 0
    )
    completeness = round(100.0 * complete_count / len(ELEMENTS_19), 1)

    warnings = [
        f"OCR vision ({OLLAMA_VISION_MODEL}) — {pages_used}/{pages_total} page(s) OK."
    ]
    if pages_total > OCR_MAX_PAGES:
        warnings.append(f"Seules les {OCR_MAX_PAGES} premières pages ont été analysées.")
    for section, ok in sections_completeness.items():
        if not ok:
            warnings.append(f"Section {section} non identifiée.")
    warnings.extend(page_errors)

    return LiasseExtractionResult(
        document_kind="LIASSE_OCR",
        elements=financial_elements,
        raw_components=raw_components,
        scoring_input=scoring_input,
        sections_completeness=sections_completeness,
        completeness_pct=completeness,
        warnings=warnings,
        pages_total=pages_total,
        pages_analyzed=pages_used,
        processing_time_ms=elapsed_ms,
        source_filename=filename,
        document_summary=(
            f"Liasse {filename or ''} — {pages_used}/{pages_total} pages, "
            f"{complete_count}/{len(ELEMENTS_19)} éléments."
        ),
    )


async def _process_all_pages(
    page_images: list[bytes],
    total_to_analyze: int,
) -> list[tuple[int, dict[str, Any] | None, str | None]]:
    """Passe 1 : toutes les pages séquentiellement."""
    results: list[tuple[int, dict[str, Any] | None, str | None]] = []
    semaphore = asyncio.Semaphore(max(1, OCR_MAX_CONCURRENCY))

    for idx, page_bytes in enumerate(page_images):
        async with semaphore:
            try:
                result = await _vision_call(page_bytes, idx + 1, total_to_analyze)
                results.append((idx, result, None))
            except VisionOcrError as exc:
                logger.warning("OCR page %d échouée : %s", idx + 1, exc)
                results.append((idx, None, str(exc)))

            if idx < len(page_images) - 1:
                await asyncio.sleep(OCR_PAGE_DELAY_SECONDS)

    return results


async def _retry_failed_pages(
    page_images: list[bytes],
    page_results: list[tuple[int, dict[str, Any] | None, str | None]],
    total_to_analyze: int,
) -> list[tuple[int, dict[str, Any] | None, str | None]]:
    """Passe 2 : re-tente les pages en échec avec image réduite et longue pause."""
    failed_indices = [idx for idx, _, err in page_results if err is not None]
    if not failed_indices:
        return page_results

    logger.info(
        "2e passe OCR : %d page(s) en échec, pause %.0fs…",
        len(failed_indices), OCR_FAILED_PASS_DELAY_SECONDS,
    )
    await asyncio.sleep(OCR_FAILED_PASS_DELAY_SECONDS)

    updated = list(page_results)
    for idx in failed_indices:
        small_img = _downscale_jpeg(page_images[idx], 800, quality=70)
        try:
            result = await _vision_call(
                small_img, idx + 1, total_to_analyze, max_attempts=OCR_RETRY_ATTEMPTS + 2
            )
            updated[idx] = (idx, result, None)
            logger.info("Page %d récupérée en 2e passe.", idx + 1)
        except VisionOcrError as exc:
            logger.warning("Page %d toujours en échec après 2e passe : %s", idx + 1, exc)
            updated[idx] = (idx, None, str(exc))
        await asyncio.sleep(OCR_PAGE_DELAY_SECONDS)

    return updated


async def extract_liasse_via_vision(
    content: bytes, filename: str | None = None
) -> LiasseExtractionResult:
    """Extraction OCR vision avec pré-vérification, retries et 2e passe."""
    async with _OCR_JOB_LOCK:
        t0 = time.perf_counter()
        await _warmup_model()

        pages_total = count_pdf_pages(content)
        page_images = render_pdf_pages(content)
        total_to_analyze = len(page_images)

        page_results = await _process_all_pages(page_images, total_to_analyze)
        page_results = await _retry_failed_pages(page_images, page_results, total_to_analyze)

        values, sections_found, empty_codes, page_errors, pages_used = _merge_page_results(page_results)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if not values:
            raise VisionOcrError(
                f"Aucun poste identifié ({pages_used}/{total_to_analyze} pages OK)."
            )

        return _build_result(
            values, sections_found, empty_codes, page_errors, pages_used,
            pages_total, elapsed_ms, filename,
        )
