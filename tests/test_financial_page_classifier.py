# -*- coding: utf-8 -*-
"""Tests classificateur de pages financières (sans Ollama)."""
from __future__ import annotations

import asyncio

from app.services.financial_page_classifier import (
    classify_financial_page,
    classify_from_text,
)


def test_bilan_actif_markers():
    text = """
    Bilan Actif
    Immobilisations incorporelles
    Immobilisations corporelles
    Créances de l'actif circulant
    Trésorerie actif
    """
    assert classify_from_text(text) == "BILAN_ACTIF"


def test_bilan_passif_markers():
    text = "Bilan passif — Capitaux propres — Dettes de financement — Passif circulant"
    assert classify_from_text(text) == "BILAN_PASSIF"


def test_cpc_markers():
    text = "Compte de produits et charges\nProduits d'exploitation\nCharges d'exploitation"
    assert classify_from_text(text) == "CPC"


def test_detail_cpc_before_cpc():
    text = "Détail des postes du CPC\nRedevances de crédit-bail\nCharges d'exploitation"
    assert classify_from_text(text) == "DETAIL_CPC"


def test_resultat_fiscal():
    text = "Passage du résultat net comptable au résultat net fiscal\nRéintégrations fiscales"
    assert classify_from_text(text) == "RESULTAT_FISCAL"


def test_identification():
    text = "Identification du contribuable\nRaison sociale\nIdentifiant fiscal\nICE"
    assert classify_from_text(text) == "IDENTIFICATION"


def test_blank_becomes_vide():
    assert (
        asyncio.run(
            classify_financial_page(native_text="   ", use_glm_fallback=False)
        )
        == "VIDE"
    )


def test_scanned_nonblank_image_not_auto_vide():
    """Sans texte natif, une image non blanche ne doit pas être VIDE d'emblée."""
    import io

    from PIL import Image, ImageDraw

    from app.services.financial_page_classifier import is_mostly_blank_image

    img = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(img)
    for y in range(40, 460, 20):
        draw.line((30, y, 370, y), fill="black", width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()
    assert is_mostly_blank_image(data) is False

    # Sans GLM : AUTRE (pas VIDE) pour laisser une chance au pipeline
    result = asyncio.run(
        classify_financial_page(
            image_bytes=data,
            native_text="",
            use_glm_fallback=False,
        )
    )
    assert result != "VIDE"


def test_admin_page_autre():
    text = "Formulaire de déclaration fiscale administrative numéro 12345 sans bilan"
    result = asyncio.run(
        classify_financial_page(native_text=text, use_glm_fallback=False)
    )
    assert result in {"AUTRE", "VIDE", "IDENTIFICATION"}


def test_continuation_cpc():
    text = "Produits financiers\nCharges financières\nRésultat courant"
    assert (
        classify_from_text(text, previous_page_type="CPC") == "CPC"
        or classify_from_text(text) == "CPC"
    )


def test_identification_does_not_continue_on_scanned_pages():
    """Bug SERDILAB : page 1 ID ne doit pas forcer les pages 2..N en IDENTIFICATION."""
    import io

    from PIL import Image, ImageDraw

    from app.services.financial_page_classifier import _EXTRACTABLE_CONTINUATION

    assert "IDENTIFICATION" not in _EXTRACTABLE_CONTINUATION

    img = Image.new("RGB", (400, 500), "white")
    draw = ImageDraw.Draw(img)
    for y in range(40, 460, 20):
        draw.line((30, y, 370, y), fill="black", width=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    data = buf.getvalue()

    # Sans GLM : pas de continuation auto depuis IDENTIFICATION
    result = asyncio.run(
        classify_financial_page(
            image_bytes=data,
            native_text="",
            previous_page_type="IDENTIFICATION",
            use_glm_fallback=False,
            page_number=2,
        )
    )
    assert result != "IDENTIFICATION"
