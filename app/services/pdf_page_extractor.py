"""Extraction du contenu d'un PDF page par page (texte natif ou GLM Vision Markdown)."""
from __future__ import annotations

import asyncio
import io
import logging
import time

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
)
from app.services.page_preprocessor import (
    crop_content_regions,
    rotate_image_bytes,
)
from app.services.vision_client import VisionExtractionError, vision_chat_text
from app.services.vision_ocr import (
    VisionOcrError,
    _OCR_JOB_LOCK,
    _warmup_model,
    count_pdf_pages,
    render_pdf_pages,
)

logger = logging.getLogger(__name__)

_CONTENT_SYSTEM_PROMPT = """
Tu es un moteur OCR de documents administratifs, fiscaux et comptables.

Transcris fidèlement la page en Markdown.

Règles :
- Retourne uniquement du Markdown brut, sans JSON et sans bloc de code.
- Ne résume pas, ne calcule pas et n'interprète rien.
- Conserve les titres, paragraphes, dates, références, identifiants et montants.
- Conserve l'ordre de lecture.
- Représente les tableaux en Markdown avec chaque valeur dans la bonne colonne.
- Conserve les lignes à 0 ou 0,00.
- Laisse une cellule vide si elle est vide.
- Écris [illisible] uniquement lorsqu'une zone est réellement illisible.
- Ne répète pas un tableau sous forme de texte après sa transcription.
- Si la page est vide, retourne exactement [PAGE VIDE].
"""


def _native_page_text(doc: fitz.Document, page_index: int) -> str:
    """Extrait le texte natif sans détruire les retours à la ligne."""
    page = doc[page_index]
    text = page.get_text("text", sort=True) or ""

    lines = [line.rstrip() for line in text.replace("\r", "\n").splitlines()]

    cleaned_lines: list[str] = []
    previous_blank = False

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(stripped)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()


def _looks_like_table(page: fitz.Page) -> bool:
    """Détecte approximativement une page contenant un tableau."""
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []

    try:
        words = page.get_text("words") or []
    except Exception:
        words = []

    horizontal_or_vertical_lines = 0
    for drawing in drawings:
        for item in drawing.get("items", []):
            if not item:
                continue
            item_type = item[0]
            if item_type in {"l", "re"}:
                horizontal_or_vertical_lines += 1

    return horizontal_or_vertical_lines >= 8 and len(words) >= 20


def _page_has_large_image(page: fitz.Page) -> bool:
    """Indique si la page PDF contient une image couvrant une grande zone."""
    try:
        images = page.get_images(full=True)
    except Exception:
        return False

    if not images:
        return False

    page_area = max(float(page.rect.width * page.rect.height), 1.0)

    for image in images:
        try:
            xref = image[0]
            rects = page.get_image_rects(xref)
        except Exception:
            continue

        for rect in rects:
            image_area = float(rect.width * rect.height)
            if image_area / page_area >= 0.45:
                return True

    return False


def _should_use_vision(
    page: fitz.Page,
    native_text: str,
    *,
    force_vision: bool,
) -> bool:
    """Décide si une page doit être envoyée au modèle Vision."""
    if force_vision:
        return True

    normalized_chars = len(native_text.replace(" ", "").replace("\n", ""))

    if normalized_chars < MIN_NATIVE_TEXT_CHARS:
        return True

    if _page_has_large_image(page):
        return True

    if _looks_like_table(page):
        return True

    return False


def _normalize_merge_line(line: str) -> str:
    """Normalise une ligne uniquement pour la comparaison."""
    return " ".join(line.strip().lower().split())


def _merge_two_region_markdowns(
    top: str,
    bottom: str,
    *,
    max_overlap_lines: int = 30,
) -> str:
    top_lines = top.strip().splitlines()
    bottom_lines = bottom.strip().splitlines()

    max_size = min(max_overlap_lines, len(top_lines), len(bottom_lines))
    overlap_size = 0

    for size in range(max_size, 0, -1):
        top_suffix = [_normalize_merge_line(line) for line in top_lines[-size:]]
        bottom_prefix = [_normalize_merge_line(line) for line in bottom_lines[:size]]
        if top_suffix == bottom_prefix:
            overlap_size = size
            break

    merged = top_lines + bottom_lines[overlap_size:]
    return "\n".join(merged).strip()


def _merge_markdown_regions(region_contents: list[str]) -> str:
    """Fusionne seulement les chevauchements exacts top/bottom."""
    if not region_contents:
        return ""
    merged = region_contents[0].strip()
    for content in region_contents[1:]:
        merged = _merge_two_region_markdowns(merged, content)
    return merged.strip()


def _markdown_table_quality(markdown: str) -> float:
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if not table_lines:
        return 0.5 if len(lines) >= 4 else 0.0
    column_counts = [max(line.count("|") - 1, 0) for line in table_lines]
    dominant_count = max(set(column_counts), key=column_counts.count)
    consistent_lines = sum(1 for count in column_counts if count == dominant_count)
    return consistent_lines / max(len(column_counts), 1)


def _vision_output_is_insufficient(markdown: str) -> bool:
    """Détecte une sortie Vision manifestement insuffisante."""
    content = markdown.strip()

    if content == "[PAGE VIDE]":
        return False

    if len(content) < 120:
        return True

    lines = [line.strip() for line in content.splitlines() if line.strip()]

    if len(lines) < 4:
        return True

    if _markdown_table_quality(content) < 0.70:
        return True

    return False


def _vision_page_result(
    page_number: int,
    markdown_content: str,
    elapsed_ms: float,
    *,
    rotation_fallback: int | None = None,
    extraction_strategy: str = "full_page",
) -> PdfPageExtraction:
    is_empty = markdown_content.strip() == "[PAGE VIDE]"
    table_quality = _markdown_table_quality(markdown_content) if not is_empty else 0.0
    meta: dict = {
        "output_format": "markdown",
        "layout_preserved": True,
        "table_format": "markdown",
        "extraction_strategy": extraction_strategy,
        "markdown_table_quality": table_quality,
    }
    if is_empty:
        meta["page_empty"] = True
    if rotation_fallback is not None:
        meta["rotation_fallback"] = rotation_fallback

    return PdfPageExtraction(
        page_number=page_number,
        status="ok",
        extraction_mode="vision",
        page_title=None,
        content=markdown_content,
        tables=[],
        char_count=len(markdown_content),
        model_latency_ms=int(elapsed_ms),
        raw_model_response=meta,
    )


async def _extract_page_regions(
    *,
    image_bytes: bytes,
    page_number: int,
    total_pages: int,
) -> tuple[str, float]:
    """Extrait une page par zones haute et basse puis fusionne le Markdown."""
    region_images = crop_content_regions(image_bytes)

    region_contents: list[str] = []
    total_latency_ms = 0.0

    for region_id, region_image in region_images:
        markdown, elapsed_ms = await vision_chat_text(
            image_bytes=region_image,
            system_prompt=_CONTENT_SYSTEM_PROMPT,
            user_message=(
                f"Page {page_number}/{total_pages}, zone {region_id}. "
                "Transcris uniquement le contenu visible dans cette zone. "
                "Conserve les tableaux et ne répète pas les informations "
                "qui ne sont pas visibles dans cette zone."
            ),
            model=OLLAMA_VISION_MODEL,
            num_predict=8192,
            max_attempts=2,
        )

        total_latency_ms += elapsed_ms

        if markdown.strip() != "[PAGE VIDE]":
            region_contents.append(markdown)

    merged = _merge_markdown_regions(region_contents)

    if not merged:
        return "[PAGE VIDE]", total_latency_ms

    return merged, total_latency_ms


async def extract_pdf_content_by_page(
    content: bytes,
    filename: str | None = None,
    *,
    max_pages: int | None = None,
    force_vision: bool = False,
) -> PdfContentExtractionResult:
    """Extrait le contenu d'un PDF page par page (Markdown natif ou Vision)."""
    async with _OCR_JOB_LOCK:
        t0 = time.perf_counter()
        warnings: list[str] = []
        warmup = await _warmup_model(soft=True)
        if str(warmup.get("warmup", "")).startswith("degraded"):
            warnings.append(
                f"Warmup Ollama dégradé ({warmup.get('warmup')}) — l'extraction peut échouer."
            )
        warnings.append(
            "Le contenu Vision est retourné en Markdown afin de préserver le layout."
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
                page = doc[idx]
                native_text = _native_page_text(doc, idx)

                use_vision = _should_use_vision(
                    page,
                    native_text,
                    force_vision=force_vision,
                )
                use_native = not use_vision

                if use_native:
                    page_results.append(
                        PdfPageExtraction(
                            page_number=page_number,
                            status="ok",
                            extraction_mode="native",
                            content=native_text,
                            tables=[],
                            char_count=len(native_text),
                            raw_model_response={
                                "output_format": "plain_text",
                                "layout_preserved": "partial",
                                "extraction_strategy": "native",
                            },
                        )
                    )
                else:
                    if page_images is None:
                        try:
                            page_images = render_pdf_pages(content, max_pages=limit)
                        except VisionOcrError as exc:
                            raise VisionExtractionError(str(exc)) from exc

                    image_bytes = page_images[idx]

                    try:
                        markdown_content, elapsed_ms = await vision_chat_text(
                            image_bytes=image_bytes,
                            system_prompt=_CONTENT_SYSTEM_PROMPT,
                            user_message=(
                                f"Page {page_number}/{limit}. "
                                "Transcris fidèlement tout le contenu visible en Markdown."
                            ),
                            model=OLLAMA_VISION_MODEL,
                            num_predict=12288,
                            max_attempts=2,
                        )

                        if _vision_output_is_insufficient(markdown_content):
                            logger.info(
                                "Sortie pleine page insuffisante page %d, fallback régional.",
                                page_number,
                            )
                            markdown_content, regional_latency = await _extract_page_regions(
                                image_bytes=image_bytes,
                                page_number=page_number,
                                total_pages=limit,
                            )
                            elapsed_ms += regional_latency
                            page_results.append(
                                _vision_page_result(
                                    page_number,
                                    markdown_content,
                                    elapsed_ms,
                                    extraction_strategy="regions",
                                )
                            )
                        else:
                            page_results.append(
                                _vision_page_result(
                                    page_number,
                                    markdown_content,
                                    elapsed_ms,
                                    extraction_strategy="full_page",
                                )
                            )

                    except VisionExtractionError as full_page_exc:
                        logger.warning(
                            "Vision pleine page %d échouée : %s",
                            page_number,
                            full_page_exc,
                        )
                        try:
                            markdown_content, elapsed_ms = await _extract_page_regions(
                                image_bytes=image_bytes,
                                page_number=page_number,
                                total_pages=limit,
                            )
                            page_results.append(
                                _vision_page_result(
                                    page_number,
                                    markdown_content,
                                    elapsed_ms,
                                    extraction_strategy="regions",
                                )
                            )
                        except VisionExtractionError as regional_exc:
                            logger.warning(
                                "Fallback régional page %d échoué : %s",
                                page_number,
                                regional_exc,
                            )
                            try:
                                rotated_image = rotate_image_bytes(image_bytes, 90)
                                markdown_content, elapsed_ms = await vision_chat_text(
                                    image_bytes=rotated_image,
                                    system_prompt=_CONTENT_SYSTEM_PROMPT,
                                    user_message=(
                                        f"Page {page_number}/{limit}. "
                                        "La page a été tournée de 90 degrés. "
                                        "Transcris fidèlement son contenu en Markdown."
                                    ),
                                    model=OLLAMA_VISION_MODEL,
                                    num_predict=12288,
                                    max_attempts=2,
                                )
                                page_results.append(
                                    _vision_page_result(
                                        page_number,
                                        markdown_content,
                                        elapsed_ms,
                                        rotation_fallback=90,
                                        extraction_strategy="rotation_90",
                                    )
                                )
                            except VisionExtractionError as rotated_exc:
                                logger.warning(
                                    "Vision page %d échouée après tous les fallbacks : %s",
                                    page_number,
                                    rotated_exc,
                                )
                                page_results.append(
                                    PdfPageExtraction(
                                        page_number=page_number,
                                        status="error",
                                        extraction_mode="vision",
                                        content=None,
                                        tables=[],
                                        char_count=0,
                                        error=(
                                            "Extraction pleine page, régionale et rotation 90° "
                                            f"échouées : {rotated_exc}"
                                        ),
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
