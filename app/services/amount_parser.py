"""Parsing déterministe des montants financiers marocains / français.

Utilise Decimal — jamais float pour les calculs métier.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional

# Exemples supportés :
# "14 757 502,81" | "14.757.502,81" | "594 733,04-" | "(27 264,00)" | "20,49"
_AMOUNT_CORE = re.compile(
    r"""
    ^\s*
    (?P<paren>\()?
    (?P<sign>[-−–])?
    (?P<body>
        \d{1,3}(?:[\s\u00a0\u202f.]\d{3})*(?:[,.]\d{1,4})?
        |
        \d+(?:[,.]\d{1,4})?
    )
    (?P<trail_sign>[-−–])?
    \)?
    \s*$
    """,
    re.VERBOSE,
)


def parse_amount(raw: str | int | float | Decimal | None) -> Optional[Decimal]:
    """Parse un montant affiché en format FR/MA vers Decimal.

    Retourne None pour cellule vide, tiret seul, ou texte non numérique.
    Les parenthèses et le signe trailing indiquent un négatif.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, Decimal):
        return raw
    if isinstance(raw, (int, float)):
        if abs(float(raw)) > 1e12:
            return None
        return Decimal(str(raw))

    text = str(raw).strip()
    if not text or text in {"-", "−", "–", "—", "n/a", "N/A", "null", "None"}:
        return None

    # Retirer devise / OCR parasites courants
    text = (
        text.replace("MAD", "")
        .replace("Dh", "")
        .replace("DH", "")
        .replace("€", "")
        .replace("*", "")
        .strip()
    )

    match = _AMOUNT_CORE.match(text)
    if not match:
        # Tentative de nettoyage plus agressif (espaces OCR)
        cleaned = re.sub(r"[^\d,.\-−–()]", "", text)
        match = _AMOUNT_CORE.match(cleaned)
        if not match:
            return None

    negative = bool(match.group("paren")) or bool(match.group("sign")) or bool(
        match.group("trail_sign")
    )
    body = match.group("body")

    # Séparateurs : si virgule présente → virgule = décimal ; points = milliers
    if "," in body and "." in body:
        if body.rfind(",") > body.rfind("."):
            body = body.replace(".", "").replace(",", ".")
        else:
            body = body.replace(",", "")
    elif "," in body:
        body = body.replace(" ", "").replace("\u00a0", "").replace("\u202f", "")
        body = body.replace(",", ".")
    else:
        # Points : si plusieurs → milliers ; un seul avec 3 digits après → milliers
        parts = body.split(".")
        if len(parts) > 2:
            body = "".join(parts)
        elif len(parts) == 2 and len(parts[1]) == 3 and len(parts[0]) <= 3:
            # Ambigu : "14.757" vs "14.75" — 3 digits → milliers si partie entière courte
            body = "".join(parts)
        body = body.replace(" ", "").replace("\u00a0", "").replace("\u202f", "")

    body = body.replace(" ", "").replace("\u00a0", "").replace("\u202f", "")
    try:
        value = Decimal(body)
    except InvalidOperation:
        return None

    if abs(value) > Decimal("1e12"):
        return None
    return -value if negative else value


def decimal_to_float(value: Decimal | None) -> float | None:
    """Conversion API rétrocompatible (JSON float)."""
    if value is None:
        return None
    return float(value)


def sum_decimal(*values: Decimal | None) -> Decimal | None:
    """Somme les valeurs non-nulles. None si toutes absentes."""
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present, Decimal("0"))
