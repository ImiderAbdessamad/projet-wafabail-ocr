"""Classification des pages de liasses fiscales (lexicale + image + GLM)."""
from __future__ import annotations

import io
import logging
import re
import unicodedata

from PIL import Image, ImageOps, ImageStat

from app.schemas.direct_financial_extraction import FinancialPageType

logger = logging.getLogger(__name__)

_EXTRACTABLE_CONTINUATION: set[str] = {
    "BILAN_ACTIF",
    "BILAN_PASSIF",
    "CPC",
    "DETAIL_CPC",
    "RESULTAT_FISCAL",
    "ESG",
    "IDENTIFICATION",
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
            # Page blanche : très claire et peu de contraste
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
    """Dernier recours sans GLM.

    Important : un PDF scanné a souvent 0 texte natif mais une image riche.
    Dans ce cas on ne doit PAS renvoyer VIDE (sinon GLM n'est jamais appelé).
    """
    folded = _fold(native_text or "")
    blank_image = is_mostly_blank_image(image_bytes) if image_bytes else True

    if blank_image and len(folded) < 12:
        return "VIDE"

    # Image non vide + peu/pas de texte = scan → laisser le pipeline
    # décider via GLM (appelant) ; ici AUTRE seulement si texte admin clair.
    if len(folded) >= 30 and classify_from_text(native_text) is None:
        return "AUTRE"

    if not blank_image:
        # Signal « inconnu mais pas vide » — le caller doit utiliser GLM.
        # On renvoie AUTRE uniquement si le caller n'a pas de fallback GLM.
        return "AUTRE"

    return "VIDE"


async def classify_financial_page(
    *,
    image_bytes: bytes | None = None,
    native_text: str | None = None,
    previous_page_type: FinancialPageType | None = None,
    use_glm_fallback: bool = True,
) -> FinancialPageType:
    """Classifie une page : texte → image → continuation → GLM."""
    folded = _fold(native_text or "")
    blank = is_mostly_blank_image(image_bytes) if image_bytes else (len(folded) < 12)

    if blank and len(folded) < 12:
        return "VIDE"

    lexical = classify_from_text(
        native_text,
        previous_page_type=previous_page_type,
    )
    if lexical is not None:
        return lexical

    # Suite multi-pages scannées : peu de texte, page précédente financière
    if (
        previous_page_type in _EXTRACTABLE_CONTINUATION
        and image_bytes
        and not blank
        and len(folded) < 40
    ):
        logger.info(
            "Continuation page type=%s (scan sans marqueur texte)",
            previous_page_type,
        )
        return previous_page_type  # type: ignore[return-value]

    if previous_page_type in _CONTINUATION:
        if any(m in folded for m in _CONTINUATION[previous_page_type]):
            return previous_page_type  # type: ignore[return-value]

    if use_glm_fallback and image_bytes and not blank:
        from app.services.direct_glm_financial_client import classify_page_with_glm

        try:
            page_type = await classify_page_with_glm(image_bytes)
            logger.info("Classification GLM → %s", page_type)
            return page_type
        except Exception as exc:  # noqa: BLE001
            logger.warning("Classification GLM échouée : %s", exc)

    return classify_blank_or_other(native_text, image_bytes=image_bytes)
