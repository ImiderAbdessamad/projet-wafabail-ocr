"""Normalisation Decimal des montants et libellés (post-OCR Markdown)."""
from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP

from app.services.amount_parser import parse_amount
from app.services.label_normalizer import normalize_label as _base_normalize_label

ZERO = Decimal("0")
ONE_HUNDRED = Decimal("100")
DAYS_PER_YEAR = Decimal("360")

_OCR_LABEL_FIXES = (
    (re.compile(r"\bdettees\b", re.I), "dettes"),
    (re.compile(r"\bcreances\b", re.I), "creances"),
    (re.compile(r"\bimmobliisations\b", re.I), "immobilisations"),
    (re.compile(r"\bsolid\s+es\s+crediteurs\b", re.I), "soldes crediteurs"),
    (re.compile(r"\bsoldes\s+crediteurs\b", re.I), "soldes crediteurs"),
)


def safe_divide(
    numerator: Decimal | None,
    denominator: Decimal | None,
) -> Decimal | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def quantize_ratio(
    value: Decimal | None,
    decimals: str = "0.01",
) -> Decimal | None:
    if value is None:
        return None
    return value.quantize(Decimal(decimals), rounding=ROUND_HALF_UP)


def parse_decimal_amount(raw: str | None) -> Decimal | None:
    """Parse un montant OCR vers Decimal. Jamais 0 pour une cellule vide."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text in {"-", "−", "–", "—", "n/a", "N/A", ""}:
        return None
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    return parse_amount(text)


def normalize_label(label: str) -> str:
    """Normalise un libellé OCR avec corrections courantes."""
    value = str(label or "")
    for pattern, replacement in _OCR_LABEL_FIXES:
        value = pattern.sub(replacement, value)
    return _base_normalize_label(value)


def is_explicit_zero(raw: str | None) -> bool:
    """True uniquement si la cellule affiche explicitement 0 / 0,00."""
    if raw is None:
        return False
    text = str(raw).strip().replace("\u00a0", " ").replace("\u202f", " ")
    if not text:
        return False
    parsed = parse_decimal_amount(text)
    return parsed is not None and parsed == ZERO
