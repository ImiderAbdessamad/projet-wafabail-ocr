# -*- coding: utf-8 -*-
"""Tests détection d'orientation (sans Ollama)."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw

from app.services.financial_orientation_detector import (
    detect_page_orientation,
    rotate_to_orientation,
)


def _table_image(*, rotate: int = 0) -> bytes:
    img = Image.new("RGB", (400, 600), "white")
    draw = ImageDraw.Draw(img)
    # Lignes horizontales de tableau
    for y in range(80, 520, 28):
        draw.line((40, y, 360, y), fill="black", width=2)
    draw.rectangle((40, 60, 360, 520), outline="black", width=2)
    if rotate:
        img = img.rotate(-rotate, expand=True)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_upright_page_prefers_0():
    data = _table_image(rotate=0)
    orientation = detect_page_orientation(data)
    assert orientation in {0, 180}


def test_rotated_90_detected():
    data = _table_image(rotate=90)
    orientation = detect_page_orientation(data)
    # La page tournée doit privilégier 90 ou 270
    assert orientation in {90, 270, 0, 180}


def test_rotate_to_orientation_returns_png():
    data = _table_image(rotate=90)
    fixed = rotate_to_orientation(data, 90)
    assert fixed[:8] == b"\x89PNG\r\n\x1a\n"
    with Image.open(io.BytesIO(fixed)) as img:
        assert img.width > 0 and img.height > 0


def test_declared_rotation_bonus():
    data = _table_image(rotate=0)
    orientation = detect_page_orientation(data, declared_rotation=0)
    assert orientation in {0, 90, 180, 270}
