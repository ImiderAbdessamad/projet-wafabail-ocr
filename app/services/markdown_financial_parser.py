"""Parseur Markdown → lignes financières brutes (sans calcul de ratios)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.pdf_extraction import PdfPageExtraction
from app.services.financial_normalizer import normalize_label

_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("BILAN_ACTIF", re.compile(r"bilan\s*[-–]?\s*actif|actif\s+du\s+bilan", re.I)),
    ("BILAN_PASSIF", re.compile(r"bilan\s*[-–]?\s*passif|passif\s+du\s+bilan", re.I)),
    ("CPC", re.compile(r"compte\s+de\s+produits\s+et\s+charges|\bcpc\b", re.I)),
    ("DETAIL_POSTES", re.compile(r"detail\s+des\s+postes|détail\s+des\s+postes", re.I)),
    (
        "RESULTAT_FISCAL",
        re.compile(
            r"passage\s+du\s+resultat|passage\s+du\s+résultat|resultat\s+fiscal",
            re.I,
        ),
    ),
    ("ESG", re.compile(r"etat\s+des\s+soldes|capacité\s+d.?autofinancement|\besg\b", re.I)),
]

_COLUMN_ALIASES = {
    "brut": "brut",
    "amortissements": "amortissements",
    "amortissements et provisions": "amortissements",
    "net": "net_n",
    "net exercice n": "net_n",
    "net n": "net_n",
    "exercice n": "net_n",
    "exercice precedent": "net_n_1",
    "exercice précédent": "net_n_1",
    "net exercice n-1": "net_n_1",
    "net n-1": "net_n_1",
    "n-1": "net_n_1",
    "total de l exercice": "total_exercice",
    "total de l'exercice": "total_exercice",
    "operations propres": "exercice",
    "exercices precedents": "exercices_precedents",
}

_TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")
_SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


@dataclass
class ParsedFinancialRow:
    page_number: int
    section: str
    raw_label: str
    normalized_label: str
    values: dict[str, str | None]
    source_excerpt: str
    warnings: list[str] = field(default_factory=list)


def _detect_section(text: str, current: str) -> str:
    for code, pattern in _SECTION_PATTERNS:
        if pattern.search(text):
            return code
    return current


def _normalize_column(header: str) -> str:
    key = normalize_label(header)
    return _COLUMN_ALIASES.get(key, key or "value")


def _split_md_row(line: str) -> list[str]:
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [cell.strip() for cell in inner.split("|")]


def _parse_markdown_table(
    lines: list[str],
    *,
    page_number: int,
    section: str,
) -> list[ParsedFinancialRow]:
    if len(lines) < 2:
        return []

    headers_raw = _split_md_row(lines[0])
    if len(headers_raw) < 2:
        return []

    # Skip separator
    data_lines = lines[2:] if _SEP_RE.match(lines[1]) else lines[1:]
    headers = [_normalize_column(h) for h in headers_raw]
    # First column is label
    rows: list[ParsedFinancialRow] = []
    for line in data_lines:
        if not _TABLE_ROW_RE.match(line):
            break
        cells = _split_md_row(line)
        if not cells:
            continue
        raw_label = cells[0].strip()
        if not raw_label or set(raw_label) <= {"-", "—", ":"}:
            continue
        values: dict[str, str | None] = {}
        for idx, header in enumerate(headers[1:], start=1):
            raw_cell = cells[idx].strip() if idx < len(cells) else ""
            values[header] = raw_cell if raw_cell else None
        rows.append(
            ParsedFinancialRow(
                page_number=page_number,
                section=section,
                raw_label=raw_label,
                normalized_label=normalize_label(raw_label),
                values=values,
                source_excerpt=line.strip()[:240],
            )
        )
    return rows


def parse_markdown_pages(
    pages: list[PdfPageExtraction],
) -> list[ParsedFinancialRow]:
    """Extrait les lignes financières depuis le Markdown page par page."""
    parsed: list[ParsedFinancialRow] = []
    current_section = "AUTRE"

    for page in pages:
        if page.status != "ok" or not page.content:
            continue
        content = page.content
        current_section = _detect_section(content, current_section)
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            current_section = _detect_section(line, current_section)
            if _TABLE_ROW_RE.match(line):
                block = [line]
                j = i + 1
                while j < len(lines) and (
                    _TABLE_ROW_RE.match(lines[j]) or _SEP_RE.match(lines[j])
                ):
                    block.append(lines[j])
                    j += 1
                parsed.extend(
                    _parse_markdown_table(
                        block,
                        page_number=page.page_number,
                        section=current_section,
                    )
                )
                i = j
                continue
            i += 1

    return parsed
