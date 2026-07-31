"""Détection d'orientation des pages de liasses fiscales (0/90/180/270)."""
from __future__ import annotations

import io
import logging
from typing import Literal

from PIL import Image, ImageFilter, ImageOps, ImageStat

logger = logging.getLogger(__name__)

OrientationDegrees = Literal[0, 90, 180, 270]
_CANDIDATES: tuple[OrientationDegrees, ...] = (0, 90, 180, 270)


def _to_rgb(image_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(image_bytes)) as img:
        return img.convert("RGB")


def _score_orientation(image: Image.Image) -> float:
    """Score heuristique : lignes majoritairement horizontales + contraste."""
    # Réduit pour la vitesse
    sample = image.copy()
    sample.thumbnail((480, 480), Image.Resampling.BILINEAR)
    gray = ImageOps.grayscale(sample)
    edges = gray.filter(ImageFilter.FIND_EDGES)
    # Projection horizontale vs verticale des pixels de bord
    pixels = list(edges.getdata())
    w, h = edges.size
    if w < 8 or h < 8:
        return 0.0

    row_energy = []
    for y in range(h):
        row = pixels[y * w : (y + 1) * w]
        row_energy.append(sum(1 for p in row if p > 40))

    col_energy = []
    for x in range(w):
        col = [pixels[y * w + x] for y in range(h)]
        col_energy.append(sum(1 for p in col if p > 40))

    # Variance des énergies : tables → pics réguliers sur les lignes
    def _var(values: list[int]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)

    h_var = _var(row_energy)
    v_var = _var(col_energy)
    # Préférer une variance ligne > variance colonne (lignes horizontales)
    orientation_bias = h_var / (v_var + 1.0)

    # Contraste global
    contrast = float(ImageStat.Stat(gray).stddev[0])

    # Aspect : pages comptables souvent portrait après correction
    aspect = h / max(w, 1)
    portrait_bonus = 1.15 if aspect >= 1.05 else 0.9

    return orientation_bias * (1.0 + contrast / 80.0) * portrait_bonus


def detect_page_orientation(
    image_bytes: bytes,
    *,
    declared_rotation: int | None = None,
) -> OrientationDegrees:
    """Choisit l'orientation 0/90/180/270 maximisant le score de lisibilité.

    Si le PDF déclare déjà une rotation, elle est testée en priorité puis
    comparée aux autres candidats.
    """
    base = _to_rgb(image_bytes)
    scores: dict[int, float] = {}

    for angle in _CANDIDATES:
        rotated = base if angle == 0 else base.rotate(-angle, expand=True)
        scores[angle] = _score_orientation(rotated)

    if declared_rotation is not None:
        declared = int(declared_rotation) % 360
        if declared in scores:
            # Bonus fort à la rotation PDF : les liasses scannées ont souvent
            # déjà la bonne orientation (declared=0) alors que l'heuristique
            # de lignes peut préférer 90/270 à tort.
            scores[declared] *= 1.35

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    # Si declared=0 et le meilleur non-nul gagne de peu, rester à 0.
    if declared_rotation is not None and int(declared_rotation) % 360 == 0:
        zero = scores.get(0, 0.0)
        if best != 0 and zero > 0 and scores[best] / zero < 1.25:
            best = 0
    logger.debug(
        "Orientation scores=%s → %s",
        {k: round(v, 3) for k, v in scores.items()},
        best,
    )
    return best  # type: ignore[return-value]


def rotate_to_orientation(image_bytes: bytes, orientation: int) -> bytes:
    """Applique une rotation horaire et renvoie du PNG."""
    angle = int(orientation) % 360
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        if angle:
            img = img.rotate(-angle, expand=True)
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()
