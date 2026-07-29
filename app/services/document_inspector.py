"""Inspection initiale des documents liasse."""
from __future__ import annotations

from io import BytesIO

import fitz
import pdfplumber

from app.schemas.liasse import DocumentInspection, PageInspection

FORECAST_MARKERS = (
    "prévisionnel",
    "previsionnel",
    "bilan prévisionnel",
    "bilan previsionnel",
    "forecast",
    "budget",
)
INTERIM_MARKERS = ("situation provisoire", "situation intermédiaire", "situation intermediaire")
REPORT_MARKERS = ("rapport des éléments financiers", "elements financiers calcules")

# Seuil bas (aperçu 15 %) : une page réellement blanche a très peu de pixels non blancs.
_VISUAL_SAMPLE_MATRIX = fitz.Matrix(0.15, 0.15)
_MIN_NONWHITE_PIXELS = 150


def _page_has_visual_content(page: fitz.Page) -> bool:
    """Détecte du contenu visuel (scan/image) même sans couche texte PDF."""
    if page.get_images():
        return True
    pix = page.get_pixmap(matrix=_VISUAL_SAMPLE_MATRIX, alpha=False)
    nonwhite = 0
    samples = pix.samples
    for i in range(0, len(samples), 3):
        if samples[i] < 250 or samples[i + 1] < 250 or samples[i + 2] < 250:
            nonwhite += 1
            if nonwhite >= _MIN_NONWHITE_PIXELS:
                return True
    return False


def inspect_document(content: bytes) -> DocumentInspection:
    """Inspecte un PDF page par page avant l'extraction."""
    page_inspections: list[PageInspection] = []
    blank_pages: list[int] = []
    rotated_pages: dict[int, int] = {}
    warnings: list[str] = []
    full_text_parts: list[str] = []

    with fitz.open(stream=BytesIO(content), filetype="pdf") as doc:
        plumber = pdfplumber.open(BytesIO(content))
        try:
            for idx, page in enumerate(doc, start=1):
                text = ""
                try:
                    text = plumber.pages[idx - 1].extract_text() or ""
                except Exception:
                    text = page.get_text() or ""
                cleaned = " ".join(text.split())
                full_text_parts.append(cleaned)
                text_length = len(cleaned)
                has_native_text = text_length >= 20
                rotation = int(page.rotation or 0) % 360
                if rotation:
                    rotated_pages[idx] = rotation
                has_visual = _page_has_visual_content(page)
                # Absence de texte natif ≠ page blanche (liasses scannées).
                is_blank = text_length < 10 and not has_visual
                if is_blank:
                    blank_pages.append(idx)

                page_type = classify_page_text(cleaned, has_visual=has_visual)
                page_warnings: list[str] = []
                if is_blank:
                    page_warnings.append("Page quasi vide.")
                elif text_length < 10 and has_visual:
                    page_warnings.append(
                        "Page scannée (pas de texte natif) — OCR vision requis."
                    )

                page_inspections.append(
                    PageInspection(
                        page_number=idx,
                        text_length=text_length,
                        has_native_text=has_native_text,
                        declared_rotation=rotation,
                        width=float(page.rect.width),
                        height=float(page.rect.height),
                        is_blank=is_blank,
                        page_type=page_type,
                        warnings=page_warnings,
                    )
                )
        finally:
            plumber.close()

    full_text = "\n".join(full_text_parts).lower()
    document_type = "unknown"
    period_type = "unknown"
    confidence = 0.55

    if any(marker in full_text for marker in FORECAST_MARKERS):
        document_type = "forecast_financial_statements"
        period_type = "forecast"
        confidence = 0.9
    elif any(marker in full_text for marker in INTERIM_MARKERS):
        document_type = "interim_financial_statements"
        period_type = "interim"
        confidence = 0.8
    elif any(marker in full_text for marker in REPORT_MARKERS):
        document_type = "extraction_report"
        period_type = "actual"
        confidence = 0.95
    elif "bilan (actif)" in full_text and "bilan (passif)" in full_text:
        document_type = "financial_statements"
        period_type = "actual"
        confidence = 0.8
    elif "liasse fiscale" in full_text:
        document_type = "tax_return"
        period_type = "actual"
        confidence = 0.75

    if blank_pages:
        warnings.append(f"{len(blank_pages)} page(s) quasi blanches détectées.")

    return DocumentInspection(
        pages_total=len(page_inspections),
        pages_with_native_text=sum(1 for p in page_inspections if p.has_native_text),
        pages_scanned=sum(1 for p in page_inspections if not p.has_native_text and not p.is_blank),
        blank_pages=blank_pages,
        rotated_pages=rotated_pages,
        page_inspections=page_inspections,
        document_type=document_type,
        period_type=period_type,
        confidence=confidence,
        warnings=warnings,
    )


def classify_page_text(text: str, *, has_visual: bool = False) -> str:
    lowered = (text or "").lower()
    if not lowered.strip():
        return "SCAN" if has_visual else "BLANCHE"
    if "identification du contribuable" in lowered or "raison sociale" in lowered:
        return "IDENTIFICATION"
    if "bilan (actif)" in lowered:
        return "BILAN_ACTIF"
    if "bilan (passif)" in lowered:
        return "BILAN_PASSIF"
    if "compte de produits" in lowered or "cpc" in lowered:
        return "CPC"
    if "etat des soldes de gestion" in lowered or "capacité d'autofinancement" in lowered or "capacite d autofinancement" in lowered:
        return "ESG"
    if "résultat fiscal" in lowered or "resultat fiscal" in lowered:
        return "RESULTAT_FISCAL"
    if "amortissements" in lowered:
        return "AMORTISSEMENTS"
    if "provisions" in lowered:
        return "PROVISIONS"
    if "credit-bail" in lowered or "crédit-bail" in lowered:
        return "CREDIT_BAIL"
    if "annexe" in lowered:
        return "ANNEXE"
    return "AUTRE"
