"""Classification des pages de liasses fiscales (lexicale + contexte)."""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Literal

from app.schemas.direct_financial_extraction import FinancialPageType

logger = logging.getLogger(__name__)

_CONTINUATION: dict[str, tuple[str, ...]] = {
    "BILAN_ACTIF": (
        "immobilisation",
        "actif circulant",
        "tresorerie actif",
        "creances",
        "stocks",
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
        "produits financiers",
        "charges financieres",
        "resultat",
    ),
}


def _fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", text or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


_MARKERS: list[tuple[FinancialPageType, tuple[str, ...], int]] = [
    (
        "IDENTIFICATION",
        (
            "identification du contribuable",
            "pieces annexes a la declaration",
            "raison sociale",
            "identifiant fiscal",
        ),
        2,
    ),
    (
        "BILAN_ACTIF",
        (
            "bilan actif",
            "bilan - actif",
            "immobilisations incorporelles",
            "immobilisations corporelles",
            "creances de l actif circulant",
            "tresorerie actif",
            "total general i",
        ),
        2,
    ),
    (
        "BILAN_PASSIF",
        (
            "bilan passif",
            "bilan - passif",
            "capitaux propres",
            "dettes de financement",
            "passif circulant",
            "tresorerie passif",
        ),
        2,
    ),
    (
        "DETAIL_CPC",
        (
            "detail des postes du cpc",
            "detail des postes du c p c",
            "redevances de credit bail",
            "achats consommes de matieres",
        ),
        1,
    ),
    (
        "RESULTAT_FISCAL",
        (
            "passage du resultat net comptable",
            "resultat net fiscal",
            "reintegrations fiscales",
            "deductions fiscales",
        ),
        1,
    ),
    (
        "ESG",
        (
            "etat des soldes de gestion",
            "capacite d autofinancement",
            "tableau de formation des resultats",
            "valeur ajoutee",
        ),
        1,
    ),
    (
        "CPC",
        (
            "compte de produits et charges",
            "produits d exploitation",
            "charges d exploitation",
            "resultat financier",
            "resultat courant",
            "chiffre d affaires",
        ),
        2,
    ),
]


def classify_from_text(
    native_text: str | None,
    *,
    previous_page_type: FinancialPageType | None = None,
) -> FinancialPageType | None:
    """Niveau 1–2 : texte natif + règles lexicales."""
    text = _fold(native_text or "")
    if len(text) < 12:
        return None

    scores: dict[str, int] = {}
    for page_type, markers, _threshold in _MARKERS:
        scores[page_type] = sum(1 for m in markers if m in text)

    # DETAIL_CPC avant CPC si marqueurs détail présents
    if scores.get("DETAIL_CPC", 0) >= 1:
        return "DETAIL_CPC"

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best]
    threshold = 1
    for page_type, _markers, th in _MARKERS:
        if page_type == best:
            threshold = th
            break

    if best_score >= threshold:
        return best  # type: ignore[return-value]

    # Continuation multi-pages
    if previous_page_type and previous_page_type in _CONTINUATION:
        cont_hits = sum(
            1 for m in _CONTINUATION[previous_page_type] if m in text
        )
        if cont_hits >= 1 and best_score >= 1:
            return previous_page_type

    if best_score >= 1:
        return best  # type: ignore[return-value]
    return None


def classify_blank_or_other(
    native_text: str | None,
    *,
    image_byte_size: int | None = None,
) -> FinancialPageType:
    text = (native_text or "").strip()
    if len(text) < 20 and (image_byte_size is None or image_byte_size < 8_000):
        return "VIDE"
    if len(_fold(text)) < 30:
        return "VIDE"
    return "AUTRE"


async def classify_financial_page(
    *,
    image_bytes: bytes | None = None,
    native_text: str | None = None,
    previous_page_type: FinancialPageType | None = None,
    use_glm_fallback: bool = False,
) -> FinancialPageType:
    """Classifie une page. GLM optionnel uniquement si ambigu (désactivé par défaut
    dans les tests / V1 pour rester lexical).
    """
    del image_bytes, use_glm_fallback  # réservé pour mini-appel GLM futur

    lexical = classify_from_text(
        native_text,
        previous_page_type=previous_page_type,
    )
    if lexical is not None:
        return lexical

    if previous_page_type in {
        "BILAN_ACTIF",
        "BILAN_PASSIF",
        "CPC",
        "DETAIL_CPC",
        "RESULTAT_FISCAL",
    }:
        # Page de suite sans marqueur fort
        folded = _fold(native_text or "")
        if previous_page_type in _CONTINUATION:
            if any(m in folded for m in _CONTINUATION[previous_page_type]):
                return previous_page_type

    return classify_blank_or_other(native_text)
