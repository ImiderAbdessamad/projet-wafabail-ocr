# -*- coding: utf-8 -*-
"""Tests resolver GLM direct + fixtures SERDILAB / ADEIS / FDI (sans Ollama)."""
from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.direct_financial_extraction import (
    DirectFinancialCandidate,
    DirectFinancialEvidence,
)
from app.services.direct_financial_resolver import (
    build_dataset_from_direct_candidates,
    is_total_general_label,
    parse_candidate_amount,
    resolve_direct_financial_candidates,
)
from app.services.financial_controls import run_accounting_controls
from app.services.financial_normalizer import parse_decimal_amount


def _c(
    code: str,
    raw: str,
    *,
    page_type: str,
    label: str,
    column: str,
    role: str,
    period: str = "N",
    nature: str = "DETAIL",
    page: int = 1,
    orientation: int = 0,
) -> DirectFinancialCandidate:
    return DirectFinancialCandidate(
        field_code=code,
        raw_value=raw,
        period=period,  # type: ignore[arg-type]
        nature=nature,  # type: ignore[arg-type]
        confidence=0.9,
        evidence=DirectFinancialEvidence(
            page_number=page,
            page_type=page_type,  # type: ignore[arg-type]
            raw_label=label,
            column_name=column,
            column_role=role,  # type: ignore[arg-type]
            source_excerpt=f"| {label} | {raw} |",
            orientation=orientation,  # type: ignore[arg-type]
        ),
    )


def serdilab_candidates() -> list[DirectFinancialCandidate]:
    return [
        _c("TOTAL_ACTIF", "22 303 497,11", page_type="BILAN_ACTIF", label="TOTAL GENERAL I+II+III", column="Net", role="NET_N", nature="GRAND_TOTAL", page=2),
        _c("ACTIFS_IMMOBILISES", "338 562,41", page_type="BILAN_ACTIF", label="TOTAL I", column="Net", role="NET_N", nature="SECTION_TOTAL", page=2),
        _c("ACTIF_CIRCULANT", "21 763 766,88", page_type="BILAN_ACTIF", label="TOTAL II", column="Net", role="NET_N", nature="SECTION_TOTAL", page=2),
        _c("STOCKS", "949 635,00", page_type="BILAN_ACTIF", label="TOTAL STOCKS", column="Net", role="NET_N", page=2),
        _c("TRESORERIE_ACTIF", "201 167,82", page_type="BILAN_ACTIF", label="TOTAL TRESORERIE-ACTIF", column="Net", role="NET_N", nature="SECTION_TOTAL", page=2),
        _c("TOTAL_PASSIF", "22 303 497,11", page_type="BILAN_PASSIF", label="TOTAL I+II+III", column="Exercice", role="EXERCICE_N", nature="GRAND_TOTAL", page=3),
        _c("FONDS_PROPRES", "9 114 715,17", page_type="BILAN_PASSIF", label="TOTAL DES CAPITAUX PROPRES", column="Exercice", role="EXERCICE_N", nature="SECTION_TOTAL", page=3),
        _c("DETTES_FINANCIERES", "133 308,11", page_type="BILAN_PASSIF", label="TOTAL DES DETTES DE FINANCEMENT", column="Exercice", role="EXERCICE_N", nature="SECTION_TOTAL", page=3),
        _c("PASSIF_CIRCULANT", "13 055 473,83", page_type="BILAN_PASSIF", label="TOTAL DU PASSIF CIRCULANT", column="Exercice", role="EXERCICE_N", nature="SECTION_TOTAL", page=3),
        _c("TRESORERIE_PASSIF", "0,00", page_type="BILAN_PASSIF", label="Trésorerie-Passif", column="Exercice", role="EXERCICE_N", page=3),
        _c("CHIFFRE_AFFAIRES", "13 404 177,00", page_type="CPC", label="Chiffre d'affaires", column="Taux du exercice", role="UNKNOWN", page=4),
        _c("CHIFFRE_AFFAIRES", "24 105 417,32", page_type="CPC", label="Chiffre d'affaires", column="Exercice précédent", role="EXERCICE_N1", period="N_MINUS_1", page=4),
        _c("RESULTAT_NET_XIII", "1 179 809,16", page_type="CPC", label="XIII Résultat net", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", nature="SECTION_TOTAL", page=5),
        _c("RESULTAT_NET_XVI", "1 179 809,16", page_type="CPC", label="XVI Résultat net", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", nature="SECTION_TOTAL", page=5),
        _c("RESULTAT_NET", "670 378,06", page_type="CPC", label="Résultat net", column="Exercice précédent", role="EXERCICE_N1", period="N_MINUS_1", page=5),
        _c("ACHATS_REVENDUS", "9 295 560,07", page_type="CPC", label="Achats revendus", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", page=4),
        _c("ACHATS_CONSOMMES", "430 367,29", page_type="CPC", label="Achats consommés", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", page=4),
        _c("CHARGES_INTERETS", "95 394,47", page_type="CPC", label="Charges d'intérêts", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", page=5),
        _c("PRODUITS_FINANCIERS", "7 082,15", page_type="CPC", label="Produits financiers", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", nature="SECTION_TOTAL", page=5),
        _c("CHARGES_FINANCIERES", "200 928,82", page_type="CPC", label="Charges financières", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", nature="SECTION_TOTAL", page=5),
        _c("RESULTAT_COURANT", "1 187 736,60", page_type="CPC", label="Résultat courant", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", nature="SECTION_TOTAL", page=5),
        _c("PRODUITS_NON_COURANTS", "64 027,85", page_type="CPC", label="Produits non courants", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", page=5),
        _c("CHARGES_NON_COURANTES", "71 955,29", page_type="CPC", label="Charges non courantes", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", page=5),
        _c("RESULTAT_NON_COURANT", "-7 927,44", page_type="CPC", label="Résultat non courant", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", nature="SECTION_TOTAL", page=5),
        _c("RESULTAT_AVANT_IMPOT", "1 179 809,16", page_type="CPC", label="Résultat avant impôts", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", nature="SECTION_TOTAL", page=5),
        _c("IMPOT_SUR_RESULTATS", "0,00", page_type="CPC", label="Impôts sur les résultats", column="3 = 1 + 2", role="TOTAL_EXERCICE_N", page=5),
        _c("REDEVANCES_CREDIT_BAIL", "21 729,13", page_type="DETAIL_CPC", label="Redevances de crédit-bail", column="Exercice", role="EXERCICE_N", page=6),
    ]


def test_serdilab_reference_values():
    resolved = resolve_direct_financial_candidates(serdilab_candidates())
    assert resolved["CHIFFRE_AFFAIRES"].value == Decimal("13404177.00")
    assert resolved["CHIFFRE_AFFAIRES_N1"].value == Decimal("24105417.32")
    assert resolved["RESULTAT_NET"].value == Decimal("1179809.16")
    assert resolved["RESULTAT_NET_N1"].value == Decimal("670378.06")
    assert resolved["TOTAL_ACTIF"].value == Decimal("22303497.11")
    assert resolved["TOTAL_PASSIF"].value == Decimal("22303497.11")
    assert resolved["TOTAL_BILAN"].value == Decimal("22303497.11")
    assert resolved["FONDS_PROPRES"].value == Decimal("9114715.17")
    assert resolved["DETTES_FINANCIERES"].value == Decimal("133308.11")
    assert resolved["ACTIFS_IMMOBILISES"].value == Decimal("338562.41")
    assert resolved["ACTIF_CIRCULANT"].value == Decimal("21763766.88")
    assert resolved["PASSIF_CIRCULANT"].value == Decimal("13055473.83")
    assert resolved["STOCKS"].value == Decimal("949635.00")
    assert resolved["TRESORERIE_ACTIF"].value == Decimal("201167.82")
    assert resolved["TRESORERIE_PASSIF"].value == Decimal("0.00")
    assert resolved["ACHATS_REVENDUS"].value == Decimal("9295560.07")
    assert resolved["ACHATS_CONSOMMES"].value == Decimal("430367.29")
    assert resolved["CHARGES_INTERETS"].value == Decimal("95394.47")
    assert resolved["PRODUITS_FINANCIERS"].value == Decimal("7082.15")
    assert resolved["CHARGES_FINANCIERES"].value == Decimal("200928.82")
    assert resolved["RESULTAT_FINANCIER"].value == Decimal("-193846.67")
    assert resolved["RESULTAT_COURANT"].value == Decimal("1187736.60")
    assert resolved["PRODUITS_NON_COURANTS"].value == Decimal("64027.85")
    assert resolved["CHARGES_NON_COURANTES"].value == Decimal("71955.29")
    assert resolved["RESULTAT_NON_COURANT"].value == Decimal("-7927.44")
    assert resolved["RESULTAT_AVANT_IMPOT"].value == Decimal("1179809.16")
    assert resolved["IMPOT_SUR_RESULTATS"].value == Decimal("0.00")
    assert resolved["REDEVANCES_CREDIT_BAIL"].value == Decimal("21729.13")
    assert "ENCOURS_LEASING" not in resolved or resolved["ENCOURS_LEASING"].value is None

    for fv in resolved.values():
        for prov in fv.provenance:
            assert prov.extraction_method == "glm_direct_vision"
            assert "qwen" not in (prov.extraction_method or "").lower()
            assert "deterministic" not in (prov.extraction_method or "").lower()


def test_adeis_invest_actif_net_columns():
    cands = [
        _c("TOTAL_ACTIF", "43 807 944,82", page_type="BILAN_ACTIF", label="TOTAL GENERAL I+II+III", column="Net", role="NET_N", nature="GRAND_TOTAL"),
        _c("TOTAL_ACTIF", "40 074 741,35", page_type="BILAN_ACTIF", label="TOTAL GENERAL I+II+III", column="Exercice précédent", role="EXERCICE_N1", period="N_MINUS_1", nature="GRAND_TOTAL"),
        _c("ACTIFS_IMMOBILISES", "32 512 391,02", page_type="BILAN_ACTIF", label="TOTAL I", column="Net", role="NET_N", nature="SECTION_TOTAL"),
        _c("ACTIF_CIRCULANT", "10 107 666,27", page_type="BILAN_ACTIF", label="TOTAL II", column="Net", role="NET_N", nature="SECTION_TOTAL"),
        _c("TRESORERIE_ACTIF", "1 187 887,53", page_type="BILAN_ACTIF", label="TOTAL TRESORERIE-ACTIF", column="Net", role="NET_N", nature="SECTION_TOTAL"),
        _c("FONDS_PROPRES", "40 460 556,86", page_type="BILAN_PASSIF", label="TOTAL DES CAPITAUX PROPRES", column="Exercice", role="EXERCICE_N", nature="SECTION_TOTAL"),
        _c("RESULTAT_NET", "627 030,56", page_type="BILAN_PASSIF", label="Résultat net", column="Exercice", role="EXERCICE_N"),
        # Brut ne doit pas gagner
        _c("ACTIFS_IMMOBILISES", "99 999 999,00", page_type="BILAN_ACTIF", label="TOTAL I", column="Brut", role="BRUT", nature="SECTION_TOTAL"),
    ]
    resolved = resolve_direct_financial_candidates(cands)
    assert resolved["TOTAL_ACTIF"].value == Decimal("43807944.82")
    assert resolved["TOTAL_BILAN_N1"].value == Decimal("40074741.35")
    assert resolved["ACTIFS_IMMOBILISES"].value == Decimal("32512391.02")
    assert resolved["ACTIF_CIRCULANT"].value == Decimal("10107666.27")
    assert resolved["TRESORERIE_ACTIF"].value == Decimal("1187887.53")
    assert resolved["FONDS_PROPRES"].value == Decimal("40460556.86")
    assert resolved["RESULTAT_NET"].value == Decimal("627030.56")


def test_fdi_invest_minimum():
    cands = [
        _c("TOTAL_ACTIF", "20 509 380,37", page_type="BILAN_ACTIF", label="TOTAL GENERAL I+II+III", column="Net", role="NET_N", nature="GRAND_TOTAL"),
        _c("TOTAL_ACTIF", "17 662 128,51", page_type="BILAN_ACTIF", label="TOTAL GENERAL I+II+III", column="Exercice précédent", role="EXERCICE_N1", period="N_MINUS_1", nature="GRAND_TOTAL"),
        _c("ACTIFS_IMMOBILISES", "14 168 902,06", page_type="BILAN_ACTIF", label="TOTAL I", column="Net", role="NET_N", nature="SECTION_TOTAL"),
        _c("ACTIF_CIRCULANT", "4 458 149,79", page_type="BILAN_ACTIF", label="TOTAL II", column="Net", role="NET_N", nature="SECTION_TOTAL"),
        _c("TRESORERIE_ACTIF", "1 882 328,52", page_type="BILAN_ACTIF", label="TOTAL TRESORERIE-ACTIF", column="Net", role="NET_N", nature="SECTION_TOTAL"),
    ]
    resolved = resolve_direct_financial_candidates(cands)
    assert resolved["TOTAL_ACTIF"].value == Decimal("20509380.37")
    assert resolved["TOTAL_BILAN_N1"].value == Decimal("17662128.51")
    assert resolved["ACTIFS_IMMOBILISES"].value == Decimal("14168902.06")
    assert resolved["ACTIF_CIRCULANT"].value == Decimal("4458149.79")
    assert resolved["TRESORERIE_ACTIF"].value == Decimal("1882328.52")


def test_total_i_rejected_total_general_accepted():
    assert is_total_general_label("TOTAL GENERAL I+II+III")
    assert is_total_general_label("TOTAL I+II+III")
    assert is_total_general_label("TOTAL I II III")
    assert not is_total_general_label("TOTAL I")
    bad = _c(
        "TOTAL_ACTIF",
        "10 000,00",
        page_type="BILAN_ACTIF",
        label="TOTAL I",
        column="Net",
        role="NET_N",
        nature="GRAND_TOTAL",
    )
    resolved = resolve_direct_financial_candidates([bad])
    assert "TOTAL_ACTIF" not in resolved or resolved["TOTAL_ACTIF"].value is None


def test_explicit_zero_and_empty_cell():
    assert parse_decimal_amount("0,00") == Decimal("0.00")
    assert parse_candidate_amount("0,00") == Decimal("0")
    with pytest.raises(ValidationError):
        DirectFinancialCandidate(
            field_code="STOCKS",
            raw_value="",
            period="N",
            nature="DETAIL",
            confidence=0.5,
            evidence=DirectFinancialEvidence(
                page_number=1,
                page_type="BILAN_ACTIF",
                raw_label="Stocks",
                column_role="NET_N",
                source_excerpt="x",
            ),
        )


def test_actif_passif_control_and_dataset():
    dataset = build_dataset_from_direct_candidates(serdilab_candidates())
    checks = run_accounting_controls(dataset)
    codes = {c.code: c.status for c in checks}
    assert any(status == "passed" for status in codes.values())
    assert dataset.encours_leasing.value is None
    assert dataset.tresorerie_nette.value == Decimal("201167.82")
