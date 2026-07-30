"""Segmentation déterministe des sections financières à partir du Markdown OCR."""
from __future__ import annotations

import re

from app.schemas.financial_mapping import (
    FinancialSection,
    FinancialSectionInput,
)
from app.schemas.pdf_extraction import PdfContentExtractionResult


_SECTION_START_PATTERNS: list[tuple[FinancialSection, re.Pattern[str]]] = [
    (
        "BILAN_ACTIF",
        re.compile(r"\bbilan\s*[-–—]?\s*actif\b", re.IGNORECASE),
    ),
    (
        "BILAN_PASSIF",
        re.compile(r"\bbilan\s*[-–—]?\s*passif\b", re.IGNORECASE),
    ),
    (
        "DETAIL_CPC",
        re.compile(
            r"\bd[eé]tail\s+des\s+postes\s+du\s+c\.?\s*p\.?\s*c\.?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "RESULTAT_FISCAL",
        re.compile(
            (
                r"passage\s+du\s+r[eé]sultat\s+net\s+comptable"
                r".*r[eé]sultat\s+net\s+fiscal"
            ),
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "CPC",
        re.compile(r"\bcompte\s+de\s+produits\s+et\s+charges\b", re.IGNORECASE),
    ),
    (
        "IDENTIFICATION",
        re.compile(
            (
                r"pi[eè]ces\s+annexes\s+[àa]\s+la\s+d[eé]claration"
                r"|identification\s+du\s+contribuable"
            ),
            re.IGNORECASE,
        ),
    ),
]

_CONTINUATION_SECTIONS: set[FinancialSection] = {
    "BILAN_ACTIF",
    "BILAN_PASSIF",
    "CPC",
    "DETAIL_CPC",
    "RESULTAT_FISCAL",
}


def _find_section_starts(markdown: str) -> list[tuple[int, FinancialSection]]:
    starts: list[tuple[int, FinancialSection]] = []

    for section, pattern in _SECTION_START_PATTERNS:
        for match in pattern.finditer(markdown):
            starts.append((match.start(), section))

    starts.sort(key=lambda item: item[0])

    deduplicated: list[tuple[int, FinancialSection]] = []
    for position, section in starts:
        if deduplicated and abs(position - deduplicated[-1][0]) < 10:
            continue
        deduplicated.append((position, section))

    return deduplicated


def split_financial_sections(
    extraction: PdfContentExtractionResult,
) -> list[FinancialSectionInput]:
    """Découpe toutes les sections détectées, y compris multi-sections/page."""
    outputs: list[FinancialSectionInput] = []
    previous_section: FinancialSection | None = None

    for page in extraction.pages:
        if page.status != "ok":
            continue

        markdown = (page.content or "").strip()
        if not markdown or markdown == "[PAGE VIDE]":
            continue

        starts = _find_section_starts(markdown)

        if not starts:
            if previous_section in _CONTINUATION_SECTIONS:
                outputs.append(
                    FinancialSectionInput(
                        section=previous_section,
                        page_number=page.page_number,
                        markdown=markdown,
                    )
                )
            else:
                outputs.append(
                    FinancialSectionInput(
                        section="AUTRE",
                        page_number=page.page_number,
                        markdown=markdown,
                    )
                )
            continue

        first_position = starts[0][0]
        prefix = markdown[:first_position].strip()
        if prefix and len(prefix) >= 50 and previous_section in _CONTINUATION_SECTIONS:
            outputs.append(
                FinancialSectionInput(
                    section=previous_section,
                    page_number=page.page_number,
                    markdown=prefix,
                )
            )

        for index, (start, section) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else len(markdown)
            segment = markdown[start:end].strip()
            if len(segment) < 20:
                continue

            outputs.append(
                FinancialSectionInput(
                    section=section,
                    page_number=page.page_number,
                    markdown=segment,
                )
            )
            if section in _CONTINUATION_SECTIONS:
                previous_section = section

    return outputs
