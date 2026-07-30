"""Segmentation déterministe des sections financières à partir du Markdown OCR."""
from __future__ import annotations

import re
import unicodedata

from app.schemas.financial_mapping import (
    FinancialSection,
    FinancialSectionInput,
)
from app.schemas.pdf_extraction import PdfContentExtractionResult
from app.services.financial_normalizer import normalize_label


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

_IMPLICIT_SECTION_MARKERS: dict[FinancialSection, tuple[str, ...]] = {
    "BILAN_ACTIF": (
        "immobilisation en non valeur",
        "immobilisations incorporelles",
        "immobilisations corporelles",
        "creances de l actif circulant",
        "tresorerie actif",
        "stocks f",
    ),
    "BILAN_PASSIF": (
        "capitaux propres",
        "dettes de financement",
        "passif circulant",
        "tresorerie passif",
    ),
    "CPC": (
        "produits d exploitation",
        "charges d exploitation",
        "resultat d exploitation",
        "produits financiers",
        "charges financieres",
    ),
}

_BILAN_ACTIF_SPLIT_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(r"immobilisation\s+en\s+non\s+valeur", re.IGNORECASE),
    re.compile(r"immobilisations\s+incorporelles", re.IGNORECASE),
    re.compile(r"immobilisations\s+corporelles", re.IGNORECASE),
)


def _fold(text: str) -> str:
    normalized = normalize_label(text or "")
    decomposed = unicodedata.normalize("NFKD", normalized)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


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


def infer_implicit_section(markdown: str) -> FinancialSection | None:
    normalized = _fold(markdown)
    scores: dict[str, int] = {}

    for section, markers in _IMPLICIT_SECTION_MARKERS.items():
        scores[section] = sum(1 for marker in markers if marker in normalized)

    best_section = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[best_section] >= 2:
        return best_section  # type: ignore[return-value]
    return None


def _find_implicit_bilan_actif_start(markdown: str) -> int | None:
    positions = [
        match.start()
        for pattern in _BILAN_ACTIF_SPLIT_MARKERS
        for match in pattern.finditer(markdown)
    ]
    return min(positions) if positions else None


def _maybe_split_identification_and_actif(
    markdown: str,
    page_number: int,
    section: FinancialSection,
) -> list[FinancialSectionInput]:
    """Sépare IDENTIFICATION et BILAN_ACTIF sur une même page/sortie."""
    actif_start = _find_implicit_bilan_actif_start(markdown)
    if actif_start is None or actif_start < 20:
        return [
            FinancialSectionInput(
                section=section,
                page_number=page_number,
                markdown=markdown,
            )
        ]

    prefix = markdown[:actif_start].strip()
    actif = markdown[actif_start:].strip()
    prefix_fold = _fold(prefix)
    looks_like_identification = (
        section == "IDENTIFICATION"
        or "identification" in prefix_fold
        or "pieces annexes" in prefix_fold
        or "raison sociale" in prefix_fold
        or "contribuable" in prefix_fold
    )

    if not looks_like_identification and section != "IDENTIFICATION":
        return [
            FinancialSectionInput(
                section=section,
                page_number=page_number,
                markdown=markdown,
            )
        ]

    outputs: list[FinancialSectionInput] = []
    if len(prefix) >= 20:
        outputs.append(
            FinancialSectionInput(
                section="IDENTIFICATION",
                page_number=page_number,
                markdown=prefix,
            )
        )
    if len(actif) >= 20:
        outputs.append(
            FinancialSectionInput(
                section="BILAN_ACTIF",
                page_number=page_number,
                markdown=actif,
            )
        )
    return outputs or [
        FinancialSectionInput(
            section=section,
            page_number=page_number,
            markdown=markdown,
        )
    ]


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
            implicit = infer_implicit_section(markdown)
            if implicit is not None:
                for item in _maybe_split_identification_and_actif(
                    markdown,
                    page.page_number,
                    implicit,
                ):
                    outputs.append(item)
                    if item.section in _CONTINUATION_SECTIONS:
                        previous_section = item.section
            elif previous_section in _CONTINUATION_SECTIONS:
                outputs.append(
                    FinancialSectionInput(
                        section=previous_section,
                        page_number=page.page_number,
                        markdown=markdown,
                    )
                )
            else:
                # Dernier recours : IDENTIFICATION si marqueurs faibles, sinon AUTRE
                folded = _fold(markdown)
                if "identification" in folded or "pieces annexes" in folded:
                    for item in _maybe_split_identification_and_actif(
                        markdown,
                        page.page_number,
                        "IDENTIFICATION",
                    ):
                        outputs.append(item)
                        if item.section in _CONTINUATION_SECTIONS:
                            previous_section = item.section
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

            for item in _maybe_split_identification_and_actif(
                segment,
                page.page_number,
                section,
            ):
                outputs.append(item)
                if item.section in _CONTINUATION_SECTIONS:
                    previous_section = item.section

    return outputs


_PASSIF_SUBSECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "CAPITAUX_PROPRES",
        re.compile(r"\bcapitaux\s+propres\b", re.IGNORECASE),
    ),
    (
        "DETTES_FINANCEMENT",
        re.compile(r"\bdet+es?\s+de\s+financement\b", re.IGNORECASE),
    ),
    (
        "PASSIF_CIRCULANT",
        re.compile(r"\bpassif\s+circulant\b", re.IGNORECASE),
    ),
    (
        "TRESORERIE_PASSIF",
        re.compile(r"\btresorerie\s*[-–—]?\s*passif\b", re.IGNORECASE),
    ),
]

_CPC_SUBSECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "PRODUITS_EXPLOITATION",
        re.compile(r"\bproduits\s+d['\u2019]?\s*exploitation\b", re.IGNORECASE),
    ),
    (
        "CHARGES_EXPLOITATION",
        re.compile(r"\bcharges\s+d['\u2019]?\s*exploitation\b", re.IGNORECASE),
    ),
    (
        "PRODUITS_CHARGES_FINANCIERS",
        re.compile(
            r"\b(?:produits|charges)\s+financi[eè]res?\b",
            re.IGNORECASE,
        ),
    ),
    (
        "RESULTATS",
        re.compile(
            r"\b(?:xiii|xvi)?\s*resultat\s+net\b|\bresultat\s+courant\b",
            re.IGNORECASE,
        ),
    ),
    (
        "NON_COURANT",
        re.compile(r"\b(?:produits|charges)\s+non\s+courants?\b", re.IGNORECASE),
    ),
]


def _split_by_patterns(
    section_input: FinancialSectionInput,
    patterns: list[tuple[str, re.Pattern[str]]],
) -> list[FinancialSectionInput]:
    markdown = section_input.markdown
    starts: list[tuple[int, str]] = []
    for subsection, pattern in patterns:
        for match in pattern.finditer(markdown):
            starts.append((match.start(), subsection))
    starts.sort(key=lambda item: item[0])

    deduped: list[tuple[int, str]] = []
    for position, subsection in starts:
        if deduped and abs(position - deduped[-1][0]) < 20:
            continue
        deduped.append((position, subsection))
    starts = deduped

    if len(starts) < 2:
        return [section_input]

    outputs: list[FinancialSectionInput] = []
    for index, (start, subsection) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(markdown)
        segment = markdown[start:end].strip()
        if not segment:
            continue
        outputs.append(
            FinancialSectionInput(
                section=section_input.section,
                page_number=section_input.page_number,
                markdown=f"SOUS-SECTION : {subsection}\n\n{segment}",
            )
        )
    return outputs or [section_input]


def split_large_financial_section(
    section_input: FinancialSectionInput,
) -> list[FinancialSectionInput]:
    """Découpe métier d'une section trop longue (passif / CPC)."""
    if section_input.section == "BILAN_PASSIF":
        return _split_by_patterns(section_input, _PASSIF_SUBSECTION_PATTERNS)
    if section_input.section == "CPC":
        return _split_by_patterns(section_input, _CPC_SUBSECTION_PATTERNS)
    return [section_input]
