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
    OCR_SOFT_WARMUP,
    OCR_VISION_NUM_PREDICT,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    OLLAMA_VISION_MODEL,
    PDF_TO_IMAGE_DPI,
)
from app.schemas.liasse import LiasseExtractionResult
from app.services.accounting_checks import run_accounting_checks
from app.services.derived_fields import apply_derived_fields
from app.services.field_resolver import (
    extract_document_metadata,
    observations_from_page_payload,
    resolve_all_fields,
)
from app.services.page_preprocessor import crop_region, preprocess_page_image
from app.services.result_builder import build_extraction_result

logger = logging.getLogger(__name__)

_VALID_SECTIONS = {"BILAN_ACTIF", "BILAN_PASSIF", "CPC"}

_SYSTEM_PROMPT = """Tu es un expert-comptable PCGM (liasse fiscale marocaine).
Analyse cette page et retourne UNIQUEMENT un JSON valide (pas de markdown).

Tu dois extraire les OBSERVATIONS BRUTES des tableaux (libellés + colonnes),
SANS choisir toi-même entre brut/net, SANS calculer d'agrégats métier.

Format attendu :
{
  "page_type": "BILAN_ACTIF" | "BILAN_PASSIF" | "CPC" | "ESG" | "IDENTIFICATION" | "AUTRE",
  "table_title": "titre du tableau si visible",
  "columns": ["Brut", "Amortissements et provisions", "Net exercice N", "Net exercice N-1"],
  "rows": [
    {
      "label": "libellé exact de la ligne",
      "values": {
        "Brut": "11 964 530,68",
        "Amortissements et provisions": "1 491 581,62",
        "Net exercice N": "10 472 949,06"
      },
      "empty": false
    }
  ],
  "metadata": {
    "reference": null,
    "entreprise": null,
    "identification_fiscale": null,
    "exercice": null,
    "date_debut_exercice": null,
    "date_fin_exercice": null
  }
}

Règles STRICTES :
- Conserve le libellé EXACT de chaque ligne utile (totaux et postes détaillés).
- Associe chaque montant à SA colonne (en-tête). Ne déplace jamais une valeur entre lignes.
- Pour le CPC, distingue « opérations propres à l'exercice », « exercices précédents »,
  « total de l'exercice » dans les clés de values.
- Cellule vide mais ligne visible : mets "empty": true et values {} ou null.
- Cellule illisible : omets la colonne ou mets null — ne devine pas.
- Extrais TOUTES les lignes utiles même si tu ne sais pas à quel indicateur elles correspondent.
- Pages d'identification : renseigne metadata (raison sociale, IF, référence dépôt, exercice).
- Ne calcule PAS de totaux dérivés (CAF, FDR, autres charges agrégées) : extrais seulement
  les lignes affichées.
- Montants tels qu'affichés (espaces, virgule décimale française autorisée dans les strings).
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
        pages = []
        for page in doc[: min(len(doc), limit)]:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            # PNG avant prétraitement : évite une double compression JPEG
            # avant crop / rotation (ré-encodage JPEG ensuite).
            rendered = pixmap.tobytes("png")
            preprocessed = preprocess_page_image(
                rendered, orientation=int(page.rotation or 0) % 360
            )
            pages.append(preprocessed.image_bytes)
        doc.close()
    except Exception as exc:
        raise VisionOcrError(f"Rendu des pages impossible : {exc}") from exc

    if not pages:
        raise VisionOcrError("Le PDF ne contient aucune page.")

    return [_downscale_jpeg(page, MAX_IMAGE_DIMENSION) for page in pages]


async def _warmup_model(*, soft: bool | None = None) -> dict[str, Any]:
    """Vérifie le modèle distant puis tente un préchauffage.

    Sur un serveur Ollama distant, le préchauffage texte peut renvoyer 504
    alors que le chat vision fonctionne ensuite. En mode soft (défaut),
    on ne bloque pas l'extraction.
    """
    global _model_warmed
    soft_mode = OCR_SOFT_WARMUP if soft is None else soft
    status: dict[str, Any] = {
        "ollama_url": OLLAMA_URL,
        "model": OLLAMA_VISION_MODEL,
        "warmed": _model_warmed,
        "reachable": False,
        "model_present": False,
        "warmup": "skipped" if _model_warmed else "pending",
    }
    if _model_warmed:
        status["reachable"] = True
        status["model_present"] = True
        status["warmup"] = "cached"
        return status

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            tags_response = await client.get(f"{OLLAMA_URL}/api/tags")
            tags_response.raise_for_status()
            status["reachable"] = True
            models = tags_response.json().get("models", [])
            names = {model.get("name") for model in models if model.get("name")}
            if OLLAMA_VISION_MODEL not in names:
                raise VisionOcrError(
                    f"Le modèle « {OLLAMA_VISION_MODEL} » est absent du serveur Ollama "
                    f"({OLLAMA_URL}). Disponibles : {sorted(names)}"
                )
            status["model_present"] = True

            try:
                # Timeout court pour le warmup : un 504 gateway distant
                # ne doit pas bloquer l'API pendant OLLAMA_TIMEOUT (souvent 300s).
                warmup_timeout = httpx.Timeout(connect=15.0, read=25.0, write=15.0, pool=15.0)
                async with httpx.AsyncClient(timeout=warmup_timeout) as warm_client:
                    response = await warm_client.post(
                        f"{OLLAMA_URL}/api/generate",
                        json={
                            "model": OLLAMA_VISION_MODEL,
                            "prompt": "ok",
                            "stream": False,
                            "options": {"num_predict": 1},
                            "keep_alive": "30m",
                        },
                    )
                if response.status_code in {500, 502, 503, 504}:
                    raise VisionOcrError(
                        f"Pré-vérification Ollama HTTP {response.status_code}"
                    )
                response.raise_for_status()
                status["warmup"] = "ok"
            except Exception as warmup_exc:  # noqa: BLE001
                status["warmup"] = f"degraded:{warmup_exc}"
                if not soft_mode:
                    raise VisionOcrError(str(warmup_exc)) from warmup_exc
                logger.warning(
                    "Warmup Ollama dégradé (extraction poursuivie) : %s",
                    warmup_exc,
                )
        _model_warmed = True
        logger.info("Modèle %s prêt (warmup=%s).", OLLAMA_VISION_MODEL, status["warmup"])
        status["warmed"] = True
        return status
    except VisionOcrError:
        raise
    except httpx.HTTPStatusError as exc:
        raise VisionOcrError(
            f"Pré-vérification Ollama HTTP {exc.response.status_code} : "
            f"{exc.response.text[:160]}"
        ) from exc
    except httpx.HTTPError as exc:
        raise VisionOcrError(
            f"Pré-vérification Ollama impossible ({OLLAMA_URL}) : {exc}"
        ) from exc
    except Exception as exc:
        raise VisionOcrError(f"Pré-vérification Ollama impossible : {exc}") from exc


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


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

    # Fallback partiel : page_type + rows si présents
    page_type_m = re.search(r'"page_type"\s*:\s*"([A-Z_]+)"', raw, re.IGNORECASE)
    if page_type_m:
        logger.info("JSON partiel récupéré page %d (page_type seul).", page_num)
        return {
            "page_type": page_type_m.group(1).upper(),
            "rows": [],
            "columns": [],
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
        "keep_alive": "30m",
        "options": {
            "temperature": 0,
            "num_predict": OCR_VISION_NUM_PREDICT,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        if response.status_code in {500, 502, 503, 504}:
            raise VisionOcrError(
                f"Ollama HTTP {response.status_code} page {page_num} : "
                f"{response.text[:180]}"
            )
        response.raise_for_status()
    except VisionOcrError:
        raise
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

    try:
        body = response.json()
    except ValueError as exc:
        raise VisionOcrError(
            f"Réponse non-JSON Ollama page {page_num} (HTTP {response.status_code})."
        ) from exc
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
            "illisible", "500 level", "gateway", "non-json",
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
        if attempt == 0:
            img = image_bytes
        else:
            dim = max(640, 1000 - attempt * 120)
            img = _downscale_jpeg(image_bytes, dim, quality=70)

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


def _merge_region_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "page_type": "AUTRE",
        "table_title": "regional_merge",
        "columns": [],
        "rows": [],
        "metadata": {},
    }
    seen_rows: set[tuple[str, str]] = set()
    for payload in payloads:
        if payload.get("page_type") in {"BILAN_ACTIF", "BILAN_PASSIF", "CPC", "ESG"}:
            merged["page_type"] = payload["page_type"]
        for col in payload.get("columns", []) or []:
            if col not in merged["columns"]:
                merged["columns"].append(col)
        merged["metadata"].update(payload.get("metadata") or {})
        for row in payload.get("rows", []) or []:
            row_key = (row.get("label") or "", json.dumps(row.get("values") or {}, sort_keys=True))
            if row_key in seen_rows:
                continue
            seen_rows.add(row_key)
            merged["rows"].append(row)
    return merged


def _assemble_from_page_results(
    page_results: list[tuple[int, dict[str, Any] | None, str | None]],
    pages_total: int,
    elapsed_ms: int,
    filename: str | None,
) -> LiasseExtractionResult:
    """Niveau 1→5 : observations → résolution → dérivés → validations → JSON."""
    all_observations = []
    payloads: list[dict[str, Any]] = []
    sections_found: set[str] = set()
    page_errors: list[str] = []
    pages_used = 0

    for idx, result, error in sorted(page_results, key=lambda r: r[0]):
        if error:
            page_errors.append(f"Page {idx + 1} : {error}")
            continue
        if not result:
            continue
        obs = observations_from_page_payload(idx + 1, result)
        if not obs:
            page_errors.append(
                f"Page {idx + 1} : OCR sans lignes extraites "
                f"(type={result.get('page_type') or 'AUTRE'})."
            )
            continue
        pages_used += 1
        payloads.append(result)
        page_type = (result.get("page_type") or "AUTRE").upper()
        if page_type in _VALID_SECTIONS:
            # Section détectée si rows ou elements présents
            if result.get("rows") or result.get("elements"):
                sections_found.add(page_type)
        elif page_type == "ESG":
            sections_found.add("CPC")  # CAF souvent dans ESG
        all_observations.extend(obs)

    if not all_observations:
        raise VisionOcrError(
            f"Aucun poste identifié ({pages_used}/{pages_total} pages OK)."
        )

    resolved = resolve_all_fields(all_observations)
    resolved = apply_derived_fields(resolved)
    check_warnings, _, accounting_checks = run_accounting_checks(resolved)
    metadata = extract_document_metadata(all_observations, payloads)

    provenance = {
        code: {
            "selected_value": (
                str(res.selected_value) if res.selected_value is not None else None
            ),
            "detection_status": res.detection_status,
            "confidence": res.confidence,
            "selection_reason": res.selection_reason,
            "validation_status": res.validation_status,
            "candidates": [
                {
                    "value": str(c.value) if c.value is not None else None,
                    "source": c.source,
                    "column": c.column,
                    "score": c.score,
                    "raw_label": c.raw_label,
                }
                for c in (res.candidates or [])[:5]
            ],
        }
        for code, res in resolved.items()
    }

    sections_detected = {s: s in sections_found for s in _VALID_SECTIONS}
    return build_extraction_result(
        resolved=resolved,
        metadata=metadata,
        sections_detected=sections_detected,
        pages_total=pages_total,
        pages_analyzed=pages_used,
        elapsed_ms=elapsed_ms,
        filename=filename,
        extra_warnings=page_errors + check_warnings,
        field_provenance=provenance,
        accounting_checks=accounting_checks,
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
            try:
                preprocessed = preprocess_page_image(small_img)
                region_payloads: list[dict[str, Any]] = []
                for region in preprocessed.regions[1:]:
                    region_img = crop_region(preprocessed.image_bytes, region)
                    region_payloads.append(
                        await _vision_call(
                            region_img,
                            idx + 1,
                            total_to_analyze,
                            max_attempts=max(2, OCR_RETRY_ATTEMPTS),
                        )
                    )
                updated[idx] = (idx, _merge_region_payloads(region_payloads), None)
                logger.info("Page %d récupérée par fallback régional.", idx + 1)
            except VisionOcrError as region_exc:
                logger.warning("Page %d toujours en échec après 2e passe : %s", idx + 1, region_exc)
                updated[idx] = (idx, None, str(region_exc))
        await asyncio.sleep(OCR_PAGE_DELAY_SECONDS)

    return updated


async def extract_liasse_via_vision(
    content: bytes, filename: str | None = None
) -> LiasseExtractionResult:
    """Extraction OCR vision avec pipeline observation → résolution → validation.

    Utilisé par l'API `/api/v1/extraction/liasse` et `/liasse/score` via
    `extract_liasse_document()` — le modèle GLM Flash est appelé sur le
    serveur Ollama distant configuré dans `.env` (pas local).
    """
    async with _OCR_JOB_LOCK:
        t0 = time.perf_counter()
        warmup_status = await _warmup_model(soft=True)

        pages_total = count_pdf_pages(content)
        page_images = render_pdf_pages(content)
        total_to_analyze = len(page_images)

        page_results = await _process_all_pages(page_images, total_to_analyze)
        page_results = await _retry_failed_pages(page_images, page_results, total_to_analyze)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        result = _assemble_from_page_results(
            page_results, pages_total, elapsed_ms, filename
        )
        result.warnings = [
            (
                f"OCR Vision distant : model={OLLAMA_VISION_MODEL} "
                f"url={OLLAMA_URL} warmup={warmup_status.get('warmup')}"
            ),
            *list(result.warnings or []),
        ]
        return result


