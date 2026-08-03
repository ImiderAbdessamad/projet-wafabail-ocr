# -*- coding: utf-8 -*-
"""Tests filet de sécurité texte natif (DMT / négoce)."""
from __future__ import annotations

from decimal import Decimal

from tests.test_direct_financial_resolver import _c
from app.services.direct_financial_resolver import build_dataset_from_direct_candidates
from app.services.financial_ratios import calculate_financial_ratios
from app.services.financial_scoring import calculate_financial_score
from app.services.native_financial_recovery import recover_candidates_from_native_text


DMT_PASSIF = """
Bilan (passif)
Total des capitaux propres (A)
326 072,00
Dettes de financement (C)
Emprunts obligataires
Autres dettes de financement
DETTES DE FINANCEMENT (C)
Provisions durables pour risques et charges (D)
TOTAL I (A+B+C+D+E)
Fournisseurs et comptes rattachés
318 347,65
TOTAL II (F+G+H)
TRESORERIE - PASSIF
TOTAL III
TOTAL III
622 439,00
TOTAL GENERAL I+II+III
"""

DMT_CPC = """
COMPTE DE PRODUITS ET CHARGES (hors taxes)
I. PRODUITS D'EXPLOITATION
*  Ventes de marchandises (en
l'état)
4 746 419,00
4 746 419,00
2 195 641,00
*  Ventes de biens et services
produits
* Chiffres d'affaires
* Achats revendus(2) de
marchandises
24 000,00
14 400,00
* Achats consommés(2) de
matières et fournitures
* Dotations d'exploitation
961,00
4 660 461,00
2 132 961,57
Total II
89 598,00
62 679,43
III. RESULTAT
D'EXPLOITATION (I-II)
"""


def test_dmt_native_recovers_ca_dettes_re():
    glm = [
        _c(
            "RESULTAT_EXPLOITATION",
            "89 598,00",
            page_type="CPC",
            label="Résultat d'exploitation",
            role="TOTAL_EXERCICE_N",
            page=5,
        ),
        _c(
            "RESULTAT_EXPLOITATION",
            "85 958,00",
            page_type="CPC",
            label="VI. (+/-) RESULTAT D'EXPLOITATION",
            role="TOTAL_EXERCICE_N",
            page=9,
        ),
        _c(
            "RESULTAT_NET",
            "70 915,35",
            page_type="BILAN_PASSIF",
            label="Résultat net de l'exercice (2)",
            role="EXERCICE_N",
            page=4,
        ),
        _c(
            "TOTAL_ACTIF",
            "622 439,00",
            page_type="BILAN_ACTIF",
            label="TOTAL GENERAL I+II+III",
            role="NET_N",
            nature="GRAND_TOTAL",
            page=3,
        ),
        _c(
            "TOTAL_PASSIF",
            "622 439,00",
            page_type="BILAN_PASSIF",
            label="TOTAL GENERAL I+II+III",
            role="EXERCICE_N",
            nature="GRAND_TOTAL",
            page=4,
        ),
        _c(
            "FONDS_PROPRES",
            "326 072,00",
            page_type="BILAN_PASSIF",
            label="Total des capitaux propres (A)",
            role="EXERCICE_N",
            nature="SECTION_TOTAL",
            page=4,
        ),
        _c(
            "DOTATIONS_AMORTISSEMENTS",
            "961,00",
            page_type="CPC",
            label="Dotations d'exploitation",
            role="TOTAL_EXERCICE_N",
            page=5,
        ),
    ]
    merged = recover_candidates_from_native_text(
        glm,
        {4: DMT_PASSIF, 5: DMT_CPC},
        page_types={4: "BILAN_PASSIF", 5: "CPC"},
    )
    ds = build_dataset_from_direct_candidates(merged)
    assert ds.chiffre_affaires.value == Decimal("4746419.00")
    assert ds.chiffre_affaires.status == "confirmed"
    assert ds.dettes_financieres.value == Decimal("0.00")
    assert ds.resultat_exploitation.value == Decimal("89598.00")
    assert ds.resultat_exploitation.status == "confirmed"
    assert "RESULTAT_EXPLOITATION conflicting" not in " ".join(ds.warnings)

    ratios = calculate_financial_ratios(ds)
    axis = calculate_financial_score(ds, ratios, scoring_mode="STRICT")
    assert axis.calculable is True
    assert axis.raw_score > 0
