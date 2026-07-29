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
from app.services.page_preprocessor import rotate_image_bytes
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
Tu es un moteur OCR spécialisé dans les documents administratifs,
comptables, fiscaux et financiers.

Ta mission est de transcrire fidèlement la page fournie sous forme de
Markdown lisible, tout en conservant autant que possible son organisation
visuelle.

RÈGLES OBLIGATOIRES :

1. Retourne uniquement du Markdown brut.
2. Ne retourne aucun JSON.
3. N'ajoute pas de bloc ```markdown.
4. Ne résume pas le document.
5. Ne calcule aucune valeur.
6. Ne corrige aucun montant.
7. N'interprète pas les informations.
8. N'invente aucune information.
9. Respecte l'ordre de lecture de la page.
10. Conserve fidèlement :
    - les titres ;
    - les sous-titres ;
    - les paragraphes ;
    - les libellés ;
    - les montants ;
    - les dates ;
    - les références ;
    - les identifiants ;
    - les numéros fiscaux ;
    - les notes de bas de page.
11. Utilise les titres Markdown :
    # titre principal
    ## section
    ### sous-section
12. Représente les tableaux avec la syntaxe Markdown :
    | Colonne 1 | Colonne 2 |
    |---|---|
    | Valeur 1 | Valeur 2 |
13. Associe chaque valeur à la bonne ligne et à la bonne colonne.
14. Ne déplace jamais une valeur vers une autre ligne.
15. Ne supprime pas les valeurs égales à 0 ou 0,00.
16. Pour une cellule vide, laisse la cellule vide.
17. Pour une zone réellement illisible, écris [illisible].
18. Si une page est tournée, lis-la dans son orientation correcte.
19. Ne répète pas le tableau une seconde fois sous forme de texte.
20. Si la page est vide, retourne exactement :
    [PAGE VIDE]

Pour les tableaux comptables :
- conserve les en-têtes de colonnes ;
- conserve les sections ;
- conserve les sous-totaux ;
- conserve les totaux ;
- conserve les séparateurs de milliers ;
- conserve les virgules décimales ;
- conserve les signes négatifs ;
- ne fusionne pas plusieurs lignes distinctes.
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


def _vision_page_result(
    page_number: int,
    markdown_content: str,
    elapsed_ms: float,
    *,
    rotation_fallback: int | None = None,
) -> PdfPageExtraction:
    is_empty = markdown_content.strip() == "[PAGE VIDE]"
    meta: dict = {
        "output_format": "markdown",
        "layout_preserved": True,
        "table_format": "markdown",
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
                page_has_table = _looks_like_table(page)

                use_native = (
                    not force_vision
                    and not page_has_table
                    and len(native_text.strip()) >= MIN_NATIVE_TEXT_CHARS
                )

                if use_native:
                    page_results.append(
                        PdfPageExtraction(
                            page_number=page_number,
                            status="ok",
                            extraction_mode="native",
                            content=native_text,
                            tables=[],
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
                    f"Page {page_number}/{limit}. "
                    "Transcris tout le contenu visible en Markdown structuré. "
                    "Préserve fidèlement les titres, les paragraphes, "
                    "les tableaux, les colonnes et les lignes."
                )
                try:
                    markdown_content, elapsed_ms = await vision_chat_text(
                        image_bytes=image_bytes,
                        system_prompt=_CONTENT_SYSTEM_PROMPT,
                        user_message=user_message,
                        model=OLLAMA_VISION_MODEL,
                        num_predict=8192,
                        max_attempts=3,
                    )
                    page_results.append(
                        _vision_page_result(
                            page_number, markdown_content, elapsed_ms
                        )
                    )
                except VisionExtractionError as exc:
                    logger.warning("Vision page %d échouée : %s", page_number, exc)
                    try:
                        rotated_image = rotate_image_bytes(image_bytes, 90)
                        markdown_content, elapsed_ms = await vision_chat_text(
                            image_bytes=rotated_image,
                            system_prompt=_CONTENT_SYSTEM_PROMPT,
                            user_message=(
                                f"Page {page_number}/{limit}. "
                                "Cette page peut être tournée. "
                                "Lis-la dans le bon sens et transcris-la en Markdown structuré."
                            ),
                            model=OLLAMA_VISION_MODEL,
                            num_predict=8192,
                            max_attempts=2,
                        )
                        page_results.append(
                            _vision_page_result(
                                page_number,
                                markdown_content,
                                elapsed_ms,
                                rotation_fallback=90,
                            )
                        )
                    except VisionExtractionError as rotated_exc:
                        logger.warning(
                            "Vision page %d échouée même après rotation : %s",
                            page_number,
                            rotated_exc,
                        )
                        page_results.append(
                            PdfPageExtraction(
                                page_number=page_number,
                                status="error",
                                extraction_mode="vision",
                                error=str(rotated_exc),
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
