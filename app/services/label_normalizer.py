"""Normalisation robuste des libellés comptables PCGM."""
from __future__ import annotations

import re
import unicodedata

_OCR_NOISE = re.compile(r"[^\w\s]")
_MULTI_SPACE = re.compile(r"\s+")


def normalize_label(text: str | None) -> str:
    """Normalise un libellé pour matching (accents, ponctuation, OCR)."""
    if not text:
        return ""
    value = unicodedata.normalize("NFKD", str(text))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = (
        value.replace("œ", "oe")
        .replace("æ", "ae")
        .replace("’", "'")
        .replace("`", "'")
        .replace("´", "'")
    )
    value = _OCR_NOISE.sub(" ", value)
    value = _MULTI_SPACE.sub(" ", value).strip()
    return value


def label_similarity(a: str, b: str) -> float:
    """Score de similarité [0,1] basé sur égalité, inclusion contrôlée et tokens.

    Inclusion asymétrique : on accepte alias ⊂ label observé (label plus riche),
    mais on refuse label observé ⊂ alias long (ex. « CA » ⊂ « CA export »).
    """
    na, nb = normalize_label(a), normalize_label(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0

    # alias (nb) contenu dans observation (na) → bon match
    if nb in na:
        return 0.75 + 0.25 * (len(nb) / max(len(na), 1))

    # observation contenue dans alias : seulement si quasi-égale (éviter CA ⊂ CA export)
    if na in nb:
        ratio = len(na) / max(len(nb), 1)
        if ratio >= 0.9:
            return 0.7 + 0.2 * ratio
        return 0.0

    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return 0.0
    overlap = len(ta & tb) / len(ta | tb)
    return overlap if overlap >= 0.5 else 0.0
