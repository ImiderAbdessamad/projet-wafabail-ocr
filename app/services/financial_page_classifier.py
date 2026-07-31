"""Classification des pages de liasses fiscales (lexicale + image + GLM)."""
from __future__ import annotations

import io
import logging
import re
import unicodedata

from PIL import Image, ImageOps, ImageStat

from app.schemas.direct_financial_extraction import FinancialPageType

logger = logging.getLogger(__name__)

# Types pour lesquels previous_page_type sert de *hint* GLM uniquement
# (plus jamais de continuation aveugle sur scan sans texte).
_FINANCIAL_TYPES: set[str] = {
    "BILAN_ACTIF",
    "BILAN_PASSIF",
    "CPC",
    "DETAIL_CPC",
    "RESULTAT_FISCAL",
    "ESG",
}

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

# Ordre de secours si 0 candidat (liasse marocaine typique)
_FALLBACK_TYPE_ORDER: list[FinancialPageType] = [
    "BILAN_ACTIF",
    "BILAN_PASSIF",
    "CPC",
    "DETAIL_CPC",
    "RESULTAT_FISCAL",
    "ESG",
]


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


def is_mostly_blank_image(image_bytes: bytes | None) -> bool:
    """True si l'image est quasi blanche / sans contenu utile."""
    if not image_bytes or len(image_bytes) < 500:
        return True
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            gray = ImageOps.grayscale(img.convert("RGB"))
            gray.thumbnail((220, 220), Image.Resampling.BILINEAR)
            stats = ImageStat.Stat(gray)
            stddev = float(stats.stddev[0])
            mean = float(stats.mean[0])
            if stddev < 10.0 and mean > 240.0:
                return True
            if stddev < 6.0:
                return True
            return False
    except Exception:  # noqa: BLE001
        return False


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
    image_bytes: bytes | None = None,
) -> FinancialPageType:
    folded = _fold(native_text or "")
    blank_image = is_mostly_blank_image(image_bytes) if image_bytes else True

    if blank_image and len(folded) < 12:
        return "VIDE"

    if len(folded) >= 30 and classify_from_text(native_text) is None:
        return "AUTRE"

    if not blank_image:
        return "AUTRE"

    return "VIDE"


def _has_strong_identification_markers(native_text: str | None) -> bool:
    folded = _fold(native_text or "")
    return any(
        m in folded
        for m in (
            "identification du contribuable",
            "pieces annexes a la declaration",
        )
    )


def next_types_to_try(
    *,
    primary: FinancialPageType,
    previous_page_type: FinancialPageType | None,
) -> list[FinancialPageType]:
    """Ordre d'essai des schémas si l'extraction primaire renvoie 0 candidat."""
    ordered: list[FinancialPageType] = []

    def _add(item: FinancialPageType) -> None:
        if item not in ordered and item in _FINANCIAL_TYPES | {"IDENTIFICATION"}:
            ordered.append(item)

    _add(primary)

    # Enchaînement métier typique d'une liasse
    transitions: dict[str, tuple[FinancialPageType, ...]] = {
        "IDENTIFICATION": ("BILAN_ACTIF", "BILAN_PASSIF", "CPC"),
        "BILAN_ACTIF": ("BILAN_PASSIF", "CPC", "BILAN_ACTIF"),
        "BILAN_PASSIF": ("CPC", "DETAIL_CPC", "BILAN_ACTIF"),
        "CPC": ("DETAIL_CPC", "RESULTAT_FISCAL", "CPC", "BILAN_PASSIF"),
        "DETAIL_CPC": ("RESULTAT_FISCAL", "ESG", "CPC"),
        "RESULTAT_FISCAL": ("ESG", "DETAIL_CPC", "CPC"),
        "ESG": ("DETAIL_CPC", "RESULTAT_FISCAL"),
    }
    for tip in transitions.get(previous_page_type or primary, ()):
        _add(tip)
    for tip in _FALLBACK_TYPE_ORDER:
        _add(tip)
    return ordered


async def classify_financial_page(
    *,
    image_bytes: bytes | None = None,
    native_text: str | None = None,
    previous_page_type: FinancialPageType | None = None,
    use_glm_fallback: bool = True,
    page_number: int | None = None,
) -> FinancialPageType:
    """Classifie une page : texte lexical → GLM (hint previous) — pas de sticky scan."""
    folded = _fold(native_text or "")
    blank = is_mostly_blank_image(image_bytes) if image_bytes else (len(folded) < 12)

    if blank and len(folded) < 12:
        return "VIDE"

    lexical = classify_from_text(
        native_text,
        previous_page_type=previous_page_type,
    )
    if lexical is not None:
        if lexical == "IDENTIFICATION" and page_number and page_number >= 2:
            if not _has_strong_identification_markers(native_text):
                lexical = None
        if lexical is not None:
            return lexical

    # Continuation textuelle uniquement (marqueurs présents dans le texte).
    if previous_page_type in _CONTINUATION:
        if any(m in folded for m in _CONTINUATION[previous_page_type]):
            return previous_page_type  # type: ignore[return-value]

    # IMPORTANT : plus de continuation aveugle sur scan sans texte.
    # Chaque page non classée lexicalement passe par GLM.
    if use_glm_fallback and image_bytes and not blank:
        from app.services.direct_glm_financial_client import classify_page_with_glm

        try:
            discourage_id = previous_page_type == "IDENTIFICATION" or (
                page_number is not None and page_number >= 2
            )
            page_type = await classify_page_with_glm(
                image_bytes,
                discourage_identification=discourage_id,
                previous_page_type=previous_page_type,
                page_number=page_number,
            )

            if (
                page_type == "IDENTIFICATION"
                and discourage_id
                and not _has_strong_identification_markers(native_text)
            ):
                page_type = await classify_page_with_glm(
                    image_bytes,
                    discourage_identification=True,
                    force_financial_hint=True,
                    previous_page_type=previous_page_type,
                    page_number=page_number,
                )
                if page_type == "IDENTIFICATION":
                    # Ne pas forcer BILAN_ACTIF sticky : laisser le pipeline
                    # essayer plusieurs schémas via next_types_to_try.
                    guessed = "BILAN_ACTIF"
                    if previous_page_type == "BILAN_ACTIF":
                        guessed = "BILAN_PASSIF"
                    elif previous_page_type == "BILAN_PASSIF":
                        guessed = "CPC"
                    elif previous_page_type == "CPC":
                        guessed = "DETAIL_CPC"
                    logger.warning(
                        "GLM insiste IDENTIFICATION page=%s — guess %s",
                        page_number,
                        guessed,
                    )
                    page_type = guessed

            logger.info(
                "Classification GLM page=%s → %s (prev=%s)",
                page_number,
                page_type,
                previous_page_type,
            )
            return page_type
        except Exception as exc:  # noqa: BLE001
            logger.warning("Classification GLM échouée : %s", exc)
            if previous_page_type == "IDENTIFICATION":
                return "BILAN_ACTIF"
            if previous_page_type == "BILAN_ACTIF":
                return "BILAN_PASSIF"
            if previous_page_type == "BILAN_PASSIF":
                return "CPC"
            if previous_page_type == "CPC":
                return "DETAIL_CPC"

    return classify_blank_or_other(native_text, image_bytes=image_bytes)


# Compat tests
_EXTRACTABLE_CONTINUATION = _FINANCIAL_TYPES
