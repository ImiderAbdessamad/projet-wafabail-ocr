"""Segmentation déterministe des sections financières à partir du Markdown OCR.

V1 : une section détectée par page.
Futur : une page pourra contenir plusieurs sections (ne pas concaténer
toutes les pages dans un seul prompt).
"""
from __future__ import annotations

import re

from app.schemas.financial_mapping import (
    FinancialSection,
    FinancialSectionInput,
)
from app.schemas.pdf_extraction import PdfContentExtractionResult


_SECTION_PATTERNS: list[tuple[FinancialSection, re.Pattern[str]]] = [
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


def detect_financial_section(markdown: str) -> FinancialSection:
    """Détecte la section dominante d'un Markdown de page (déterministe)."""
    content = markdown or ""
    for section, pattern in _SECTION_PATTERNS:
        if pattern.search(content):
            return section
    return "AUTRE"


def split_financial_sections(
    extraction: PdfContentExtractionResult,
) -> list[FinancialSectionInput]:
    """Découpe le résultat OCR en sections (1 page = 1 section en V1).

    Note future : si une page contient plusieurs sections, découper le
    Markdown en plusieurs FinancialSectionInput sans fusionner les pages.
    """
    sections: list[FinancialSectionInput] = []

    for page in extraction.pages:
        if page.status != "ok":
            continue

        markdown = (page.content or "").strip()
        if not markdown or markdown == "[PAGE VIDE]":
            continue

        section = detect_financial_section(markdown)
        sections.append(
            FinancialSectionInput(
                section=section,
                page_number=page.page_number,
                markdown=markdown,
            )
        )

    return sections
