"""Extraction du contenu d'un PDF page par page (texte natif ou GLM Vision)."""
from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any

import fitz

from app.config import (
    MIN_NATIVE_TEXT_CHARS,
    OCR_MAX_PAGES,
    OCR_PAGE_DELAY_SECONDS,
    OLLAMA_URL,
    OLLAMA_VISION_MODEL,
)
from app.schemas.pdf_extraction import (
    PdfContentExtractionResult,
    PdfPageExtraction,
    PdfPageTable,
)
from app.services.vision_client import VisionExtractionError, vision_chat_json
from app.services.vision_ocr import (
    VisionOcrError,
    _OCR_JOB_LOCK,
    _warmup_model,
    count_pdf_pages,
    render_pdf_pages,
)

logger = logging.getLogger(__name__)

_CONTENT_SYSTEM_PROMPT = """Tu es un expert OCR multilingue.
Analyse cette page de document PDF et retourne UNIQUEMENT un JSON valide (pas de markdown).

Format attendu :
{
  "page_title": "titre ou en-tête principal si visible, sinon null",
  "content": "transcription complète du texte visible, lignes séparées par \\n",
  "tables": [
    {
      "title": "titre du tableau ou null",
      "headers": ["colonne 1", "colonne 2"],
      "rows": [["valeur ligne 1 col 1", "valeur ligne 1 col 2"]]
    }
  ],
  "language": "fr",
  "notes": "remarques sur la lisibilité si besoin, sinon null"
}

Règles :
- Transcris TOUT le texte lisible dans l'ordre de lecture (haut → bas, gauche → droite).
- Conserve les montants, dates et références exactement tels qu'affichés.
- Pour une portion illisible, écris [illisible].
- Si la page est vide, content = "" et notes = "page vide".
- Ne résume pas : extrais le contenu brut.
"""


def _native_page_text(doc: fitz.Document, page_index: int) -> str:
    page = doc[page_index]
    return " ".join((page.get_text() or "").split())


def _parse_tables(raw_tables: Any) -> list[PdfPageTable]:
    if not isinstance(raw_tables, list):
        return []
    tables: list[PdfPageTable] = []
    for item in raw_tables:
        if not isinstance(item, dict):
            continue
        headers = item.get("headers") or []
        rows = item.get("rows") or []
        tables.append(
            PdfPageTable(
                title=item.get("title"),
                headers=[str(h) for h in headers] if isinstance(headers, list) else [],
                rows=[
                    [str(cell) for cell in row]
                    for row in rows
                    if isinstance(row, list)
                ],
            )
        )
    return tables


def _page_from_vision_payload(
    page_number: int,
    payload: dict[str, Any],
    latency_ms: int | None,
) -> PdfPageExtraction:
    content = (payload.get("content") or "").strip()
    return PdfPageExtraction(
        page_number=page_number,
        status="ok",
        extraction_mode="vision",
        page_title=payload.get("page_title"),
        content=content or None,
        tables=_parse_tables(payload.get("tables")),
        char_count=len(content),
        model_latency_ms=latency_ms,
        raw_model_response=payload,
    )


async def extract_pdf_content_by_page(
    content: bytes,
    filename: str | None = None,
    *,
    max_pages: int | None = None,
    force_vision: bool = False,
) -> PdfContentExtractionResult:
    """Extrait le contenu d'un PDF page par page via texte natif ou GLM Vision."""
    async with _OCR_JOB_LOCK:
        t0 = time.perf_counter()
        warnings: list[str] = []
        warmup = await _warmup_model(soft=True)
        if str(warmup.get("warmup", "")).startswith("degraded"):
            warnings.append(
                f"Warmup Ollama dégradé ({warmup.get('warmup')}) — l'extraction peut échouer."
            )

        pages_total = count_pdf_pages(content)
        limit = min(pages_total, max_pages or OCR_MAX_PAGES)
        if limit < pages_total:
            warnings.append(
                f"Document tronqué : {limit}/{pages_total} page(s) analysée(s)."
            )

        page_images: list[bytes] | None = None
        page_results: list[PdfPageExtraction] = []

        with fitz.open(stream=io.BytesIO(content), filetype="pdf") as doc:
            for idx in range(limit):
                page_number = idx + 1
                native_text = _native_page_text(doc, idx)
                use_native = (
                    not force_vision
                    and len(native_text.strip()) >= MIN_NATIVE_TEXT_CHARS
                )

                if use_native:
                    page_results.append(
                        PdfPageExtraction(
                            page_number=page_number,
                            status="ok",
                            extraction_mode="native",
                            content=native_text,
                            char_count=len(native_text),
                        )
                    )
                    continue

                if page_images is None:
                    try:
                        page_images = render_pdf_pages(content, max_pages=limit)
                    except VisionOcrError as exc:
                        raise VisionExtractionError(str(exc)) from exc

                image_bytes = page_images[idx]
                user_message = (
                    f"Page {page_number}/{limit} — extrais tout le contenu visible."
                )
                try:
                    payload, elapsed_ms = await vision_chat_json(
                        image_bytes,
                        _CONTENT_SYSTEM_PROMPT,
                        user_message,
                        model=OLLAMA_VISION_MODEL,
                        num_predict=4096,
                    )
                    page_results.append(
                        _page_from_vision_payload(
                            page_number, payload, int(elapsed_ms)
                        )
                    )
                except VisionExtractionError as exc:
                    logger.warning("Vision page %d échouée : %s", page_number, exc)
                    page_results.append(
                        PdfPageExtraction(
                            page_number=page_number,
                            status="error",
                            extraction_mode="vision",
                            error=str(exc),
                        )
                    )

                if idx < limit - 1:
                    await asyncio.sleep(OCR_PAGE_DELAY_SECONDS)

        pages_ok = sum(1 for p in page_results if p.status == "ok")
        pages_failed = len(page_results) - pages_ok
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        return PdfContentExtractionResult(
            source_filename=filename,
            pages_total=pages_total,
            pages_processed=len(page_results),
            pages_ok=pages_ok,
            pages_failed=pages_failed,
            model=OLLAMA_VISION_MODEL,
            ollama_url=OLLAMA_URL,
            processing_time_ms=elapsed_ms,
            pages=page_results,
            warnings=[
                f"Modèle={OLLAMA_VISION_MODEL} url={OLLAMA_URL} warmup={warmup.get('warmup')}",
                *warnings,
            ],
        )
