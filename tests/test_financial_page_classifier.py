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
    assert asyncio.run(classify_financial_page(native_text="   ")) == "VIDE"


def test_admin_page_autre():
    text = "Formulaire de déclaration fiscale administrative numéro 12345 sans bilan"
    result = asyncio.run(classify_financial_page(native_text=text))
    assert result in {"AUTRE", "VIDE", "IDENTIFICATION"}


def test_continuation_cpc():
    text = "Produits financiers\nCharges financières\nRésultat courant"
    assert (
        classify_from_text(text, previous_page_type="CPC") == "CPC"
        or classify_from_text(text) == "CPC"
    )
