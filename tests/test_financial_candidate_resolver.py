# -*- coding: utf-8 -*-
"""Tests resolver déterministe des candidats financiers."""
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.schemas.financial_mapping import (
    FinancialCandidate,
    FinancialFieldCode,
    FinancialMappingOutput,
    MappingEvidence,
)
from app.services.financial_candidate_resolver import (
    build_financial_dataset_from_resolved_values,
    candidate_is_eligible,
    candidate_priority,
    clean_qwen_marker,
    is_total_general_candidate,
    resolve_financial_candidates,
)
from app.services.financial_normalizer import parse_decimal_amount
from app.scoring_rules import FINANCIAL_RATIO_RULES


def _default_column_role(
    section: str,
    period: str,
    column: str | None,
) -> str:
    if period == "N_MINUS_1":
        return "EXERCICE_N1"
    if section == "BILAN_ACTIF":
        return "NET_N"
    if section == "BILAN_PASSIF":
        return "EXERCICE_N"
    if section == "CPC":
        return "TOTAL_EXERCICE_N"
    if section == "DETAIL_CPC":
        return "EXERCICE_N"
    return "UNKNOWN"


def _cand(
    field_code: str,
    raw_value: str,
    *,
    section: str = "BILAN_ACTIF",
    period: str = "N",
    nature: str = "DETAIL",
    label: str = "",
    column: str | None = "Net",
    column_role: str | None = None,
    page: int = 1,
    confidence: float = 0.5,
) -> FinancialCandidate:
    role = column_role or _default_column_role(section, period, column)
    return FinancialCandidate(
        field_code=field_code,  # type: ignore[arg-type]
        raw_value=raw_value,
        period=period,  # type: ignore[arg-type]
        nature=nature,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=MappingEvidence(
            page_number=page,
            section=section,  # type: ignore[arg-type]
            raw_label=label or field_code,
            raw_value=raw_value,
            column_name=column,
            column_role=role,  # type: ignore[arg-type]
            source_excerpt=f"| {label} | {raw_value} |",
        ),
    )


def test_stocks_excludes_variation():
    c = _cand(
        "STOCKS",
        "100,00",
        label="Variation des stocks",
        section="BILAN_ACTIF",
    )
    ok, reasons = candidate_is_eligible(c)
    assert ok is False
    assert any("variation" in r.lower() for r in reasons)


def test_clients_excludes_crediteurs():
    c = _cand(
        "CLIENTS",
        "100,00",
        label="Clients créditeurs",
        section="BILAN_ACTIF",
    )
    ok, reasons = candidate_is_eligible(c)
    assert ok is False
    assert any("créditeur" in r.lower() or "crediteur" in r.lower() for r in reasons)


def test_fournisseurs_excludes_debiteurs():
    c = _cand(
        "FOURNISSEURS",
        "100,00",
        label="Fournisseurs débiteurs",
        section="BILAN_PASSIF",
        column="Exercice",
    )
    ok, reasons = candidate_is_eligible(c)
    assert ok is False


def test_dettes_excludes_augmentation():
    c = _cand(
        "DETTES_FINANCIERES",
        "100,00",
        label="Augmentation des dettes de financement",
        section="BILAN_PASSIF",
        column="Exercice",
    )
    ok, _ = candidate_is_eligible(c)
    assert ok is False


def test_resultat_net_excludes_instance_affectation():
    c = _cand(
        "RESULTAT_NET",
        "100,00",
        label="Résultats nets en instance d'affectation",
        section="BILAN_PASSIF",
        column="Exercice",
    )
    ok, _ = candidate_is_eligible(c)
    assert ok is False


def test_encours_excludes_redevance():
    c = _cand(
        "ENCOURS_LEASING",
        "21 729,13",
        label="Redevances de crédit-bail",
        section="DETAIL_CPC",
        column="Totaux de l'exercice",
    )
    ok, reasons = candidate_is_eligible(c)
    assert ok is False
    assert any("redevance" in r.lower() for r in reasons)


def test_charges_interets_excludes_autres():
    c = _cand(
        "CHARGES_INTERETS",
        "100,00",
        label="Autres charges financières",
        section="CPC",
        column="Totaux de l'exercice",
    )
    ok, _ = candidate_is_eligible(c)
    assert ok is False


def test_period_n_required_for_current_fields():
    c = _cand(
        "CLIENTS",
        "100,00",
        label="Clients et comptes rattachés",
        period="N_MINUS_1",
    )
    ok, reasons = candidate_is_eligible(c)
    assert ok is False
    assert any(
        "period=n" in r.lower()
        or "champ courant" in r.lower()
        or "n-1 non supportée" in r.lower()
        for r in reasons
    )


def test_bilan_actif_prefers_net_column():
    net = _cand(
        "CLIENTS",
        "19 097 949,49",
        label="Clients et comptes rattachés",
        column="Net",
        column_role="NET_N",
        confidence=0.1,
    )
    brut = _cand(
        "CLIENTS",
        "20 000 000,00",
        label="Clients et comptes rattachés",
        column="Brut",
        column_role="BRUT",
        confidence=0.99,
    )
    assert candidate_priority(net) > candidate_priority(brut)


def test_bilan_passif_prefers_exercice_column():
    exo = _cand(
        "FOURNISSEURS",
        "4 146 301,83",
        label="Fournisseurs et comptes rattachés",
        section="BILAN_PASSIF",
        column="Exercice",
        column_role="EXERCICE_N",
        confidence=0.2,
    )
    other = _cand(
        "FOURNISSEURS",
        "1,00",
        label="Fournisseurs et comptes rattachés",
        section="BILAN_PASSIF",
        column="Autre",
        column_role="UNKNOWN",
        confidence=0.99,
    )
    assert candidate_priority(exo) > candidate_priority(other)


def test_cpc_prefers_totaux_exercice():
    tot = _cand(
        "CHIFFRE_AFFAIRES",
        "13 404 177,00",
        label="Chiffre d'affaires",
        section="CPC",
        column="Totaux de l'exercice",
        column_role="TOTAL_EXERCICE_N",
        confidence=0.5,
    )
    other = _cand(
        "CHIFFRE_AFFAIRES",
        "1,00",
        label="Chiffre d'affaires",
        section="CPC",
        column="Opérations",
        column_role="UNKNOWN",
        confidence=0.5,
    )
    assert candidate_priority(tot) > candidate_priority(other)


def test_xiii_xvi_identical_confirmed():
    outputs = [
        FinancialMappingOutput(
            section="CPC",
            candidates=[
                _cand(
                    "RESULTAT_NET",
                    "1 179 809,16",
                    label="XIII RESULTAT NET",
                    section="CPC",
                    column="Totaux de l'exercice",
                    nature="SECTION_TOTAL",
                ),
                _cand(
                    "RESULTAT_NET",
                    "1 179 809,16",
                    label="XVI RESULTAT NET",
                    section="CPC",
                    column="Totaux de l'exercice",
                    nature="SECTION_TOTAL",
                ),
            ],
        )
    ]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["RESULTAT_NET"].status == "confirmed"
    assert resolved["RESULTAT_NET"].value == Decimal("1179809.16")


def test_xiii_xvi_divergent_conflicting():
    outputs = [
        FinancialMappingOutput(
            section="CPC",
            candidates=[
                _cand(
                    "RESULTAT_NET",
                    "1 179 809,16",
                    label="XIII RESULTAT NET",
                    section="CPC",
                    column="Totaux de l'exercice",
                ),
                _cand(
                    "RESULTAT_NET",
                    "999 000,00",
                    label="XVI RESULTAT NET",
                    section="CPC",
                    column="Totaux de l'exercice",
                ),
            ],
        )
    ]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["RESULTAT_NET"].status == "conflicting"
    assert resolved["RESULTAT_NET"].value is None


def test_total_actif_passif_agreement():
    outputs = [
        FinancialMappingOutput(
            section="BILAN_ACTIF",
            candidates=[
                _cand(
                    "TOTAL_ACTIF",
                    "22 303 497,11",
                    label="Total général (I + II + III)",
                    section="BILAN_ACTIF",
                    column="Net",
                    nature="GRAND_TOTAL",
                )
            ],
        ),
        FinancialMappingOutput(
            section="BILAN_PASSIF",
            candidates=[
                _cand(
                    "TOTAL_PASSIF",
                    "22 303 497,11",
                    label="Total général (I + II + III)",
                    section="BILAN_PASSIF",
                    column="Exercice",
                    nature="GRAND_TOTAL",
                )
            ],
        ),
    ]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["TOTAL_BILAN"].status == "confirmed"
    assert resolved["TOTAL_BILAN"].value == Decimal("22303497.11")


def test_total_actif_passif_conflict():
    outputs = [
        FinancialMappingOutput(
            section="BILAN_ACTIF",
            candidates=[
                _cand(
                    "TOTAL_ACTIF",
                    "22 303 497,11",
                    label="TOTAL GENERAL I+II+III",
                    section="BILAN_ACTIF",
                    column="Net",
                    nature="GRAND_TOTAL",
                )
            ],
        ),
        FinancialMappingOutput(
            section="BILAN_PASSIF",
            candidates=[
                _cand(
                    "TOTAL_PASSIF",
                    "19 000 000,00",
                    label="TOTAL GENERAL I+II+III",
                    section="BILAN_PASSIF",
                    column="Exercice",
                    nature="GRAND_TOTAL",
                )
            ],
        ),
    ]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["TOTAL_BILAN"].status == "conflicting"
    assert resolved["TOTAL_BILAN"].value is None


def test_absent_remains_none_never_zero():
    outputs = [
        FinancialMappingOutput(
            section="CPC",
            candidates=[
                _cand(
                    "CHIFFRE_AFFAIRES",
                    "13 404 177,00",
                    label="Chiffre d'affaires",
                    section="CPC",
                    column="Totaux de l'exercice",
                )
            ],
        )
    ]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["ENCOURS_LEASING"].value is None
    assert resolved["ENCOURS_LEASING"].status == "missing"
    dataset = build_financial_dataset_from_resolved_values(resolved)
    assert dataset.encours_leasing.value is None
    assert isinstance(dataset.chiffre_affaires.value, Decimal)


def test_redevances_not_encours():
    outputs = [
        FinancialMappingOutput(
            section="DETAIL_CPC",
            candidates=[
                _cand(
                    "REDEVANCES_CREDIT_BAIL",
                    "21 729,13",
                    label="Redevances de crédit-bail",
                    section="DETAIL_CPC",
                    column="Totaux de l'exercice",
                ),
                _cand(
                    "ENCOURS_LEASING",
                    "21 729,13",
                    label="Redevances de crédit-bail",
                    section="DETAIL_CPC",
                    column="Totaux de l'exercice",
                ),
            ],
        )
    ]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["REDEVANCES_CREDIT_BAIL"].value == Decimal("21729.13")
    assert resolved["ENCOURS_LEASING"].value is None
    assert resolved["REDEVANCES_CREDIT_BAIL"].value != resolved["ENCOURS_LEASING"].value


def test_no_float_in_final_amounts():
    outputs = [
        FinancialMappingOutput(
            section="CPC",
            candidates=[
                _cand(
                    "CHIFFRE_AFFAIRES",
                    "13 404 177,00",
                    label="Chiffre d'affaires",
                    section="CPC",
                    column="Totaux de l'exercice",
                )
            ],
        )
    ]
    resolved = resolve_financial_candidates(outputs)
    assert type(resolved["CHIFFRE_AFFAIRES"].value) is Decimal


def test_infer_period_from_column_normalizes_n1():
    from app.services.financial_candidate_resolver import infer_period_from_column

    c = _cand(
        "FONDS_PROPRES",
        "7 934 906,01",
        section="BILAN_PASSIF",
        column="Exercice précédent",
        period="N",
        column_role="EXERCICE_N1",
        label="TOTAL DES CAPITAUX PROPRES",
    )
    fixed = infer_period_from_column(c)
    assert fixed.period == "N_MINUS_1"
    assert any("normalisée" in w.lower() for w in fixed.warnings)


def test_n1_code_canonicalization_before_grouping():
    outputs = [
        FinancialMappingOutput(
            section="BILAN_PASSIF",
            candidates=[
                _cand(
                    "FONDS_PROPRES",
                    "7 934 906,01",
                    section="BILAN_PASSIF",
                    column="Exercice précédent",
                    period="N_MINUS_1",
                    label="TOTAL DES CAPITAUX PROPRES",
                    nature="SECTION_TOTAL",
                )
            ],
        )
    ]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["FONDS_PROPRES_N1"].value == Decimal("7934906.01")
    assert "FONDS_PROPRES" not in resolved


def test_raw_value_none_rejected_by_schema():
    with pytest.raises(ValidationError):
        FinancialCandidate(
            field_code="CHIFFRE_AFFAIRES",
            raw_value=None,  # type: ignore[arg-type]
            period="N",
            nature="DETAIL",
            confidence=0.5,
            evidence=MappingEvidence(
                page_number=1,
                section="CPC",
                raw_label="CA",
                raw_value="",
                column_name="Totaux",
                column_role="TOTAL_EXERCICE_N",
                source_excerpt="x",
            ),
        )


def test_no_n1_codes_in_qwen_field_enum():
    from typing import get_args

    codes = set(get_args(FinancialFieldCode))
    assert "CHIFFRE_AFFAIRES_N1" not in codes
    assert "RESULTAT_NET_N1" not in codes
    assert "TOTAL_BILAN_N1" not in codes
    assert "FONDS_PROPRES_N1" not in codes
    assert "CHIFFRE_AFFAIRES" in codes
    assert "RESULTAT_NET" in codes


def test_detail_cpc_blocks_charges_financieres():
    c = _cand(
        "CHARGES_FINANCIERES",
        "0,00",
        section="DETAIL_CPC",
        column="Exercice",
        label="Charges financières",
    )
    ok, reasons = candidate_is_eligible(c)
    assert ok is False
    assert any("interdit" in r.lower() for r in reasons)


def test_detail_cpc_blocks_charges_non_courantes():
    c = _cand(
        "CHARGES_NON_COURANTES",
        "70 342,89",
        section="DETAIL_CPC",
        column="Exercice",
        label="Charges non courantes",
    )
    ok, _ = candidate_is_eligible(c)
    assert ok is False


def test_detail_cpc_allows_redevances():
    c = _cand(
        "REDEVANCES_CREDIT_BAIL",
        "21 729,13",
        section="DETAIL_CPC",
        column="Exercice",
        label="Redevances de crédit-bail",
    )
    ok, _ = candidate_is_eligible(c)
    assert ok is True


def test_total_i_rejected_total_general_accepted():
    bad = _cand(
        "TOTAL_PASSIF",
        "9 114 715,17",
        section="BILAN_PASSIF",
        column="Exercice",
        label="TOTAL I",
        nature="SECTION_TOTAL",
    )
    assert candidate_is_eligible(bad)[0] is False
    assert is_total_general_candidate(bad) is False

    good = _cand(
        "TOTAL_PASSIF",
        "22 303 497,11",
        section="BILAN_PASSIF",
        column="Exercice",
        label="TOTAL GENERAL I+II+III",
        nature="GRAND_TOTAL",
    )
    assert is_total_general_candidate(good) is True
    assert candidate_is_eligible(good)[0] is True


def test_mixed_comma_ocr_amount():
    amount = parse_decimal_amount("9,116,785,07")
    assert amount == Decimal("9116785.07")
    outputs = [
        FinancialMappingOutput(
            section="BILAN_PASSIF",
            candidates=[
                _cand(
                    "FONDS_PROPRES",
                    "9,116,785,07",
                    section="BILAN_PASSIF",
                    column="Exercice",
                    label="TOTAL DES CAPITAUX PROPRES",
                    nature="SECTION_TOTAL",
                )
            ],
        )
    ]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["FONDS_PROPRES"].value == Decimal("9116785.07")
    assert any(
        "Separateurs OCR normalises." in w
        for w in resolved["FONDS_PROPRES"].warnings
    )


def test_clean_qwen_marker_strips_think():
    assert clean_qwen_marker("1 179 809,16 /think") == "1 179 809,16"
    assert clean_qwen_marker("texte</think>") == "texte"
    c = _cand(
        "CHIFFRE_AFFAIRES",
        "13 404 177,00 /think",
        section="CPC",
        column="3 = 1 + 2",
        label="Chiffre d'affaires",
    )
    c = c.model_copy(
        update={
            "evidence": c.evidence.model_copy(
                update={"source_excerpt": "| CA | 1 | /think"}
            )
        }
    )
    from app.services.financial_candidate_resolver import sanitize_candidate

    cleaned = sanitize_candidate(c)
    assert "/think" not in cleaned.raw_value
    assert "/think" not in cleaned.evidence.source_excerpt
    resolved = resolve_financial_candidates(
        [FinancialMappingOutput(section="CPC", candidates=[cleaned])]
    )
    assert resolved["CHIFFRE_AFFAIRES"].value == Decimal("13404177.00")
    assert all(
        "/think" not in (p.raw_value or "")
        and "/think" not in (p.source_excerpt or "")
        for p in resolved["CHIFFRE_AFFAIRES"].provenance
    )


def test_ca_growth_zero_financial_points():
    assert "ca_growth" not in FINANCIAL_RATIO_RULES
    total = sum(
        (Decimal(str(rule["weight"])) for rule in FINANCIAL_RATIO_RULES.values()),
        Decimal("0"),
    )
    assert total == Decimal("100")


def test_column_role_controls_eligibility():
    bad_role = _cand(
        "CLIENTS",
        "100,00",
        label="Clients et comptes rattachés",
        section="BILAN_ACTIF",
        column="Net",
        column_role="BRUT",
    )
    ok, reasons = candidate_is_eligible(bad_role)
    assert ok is False
    assert any("NET_N" in r for r in reasons)

    good = _cand(
        "CLIENTS",
        "100,00",
        label="Clients et comptes rattachés",
        section="BILAN_ACTIF",
        column="Net",
        column_role="NET_N",
    )
    assert candidate_is_eligible(good)[0] is True


def test_period_maps_to_n1_dataset_field():
    outputs = [
        FinancialMappingOutput(
            section="CPC",
            candidates=[
                _cand(
                    "CHIFFRE_AFFAIRES",
                    "24 105 417,32",
                    section="CPC",
                    column="Exercice précédent",
                    period="N_MINUS_1",
                    label="Chiffre d'affaires",
                )
            ],
        )
    ]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["CHIFFRE_AFFAIRES_N1"].value == Decimal("24105417.32")
    assert "CHIFFRE_AFFAIRES" not in resolved


def test_total_passif_rejects_intermediate_totals():
    c_i = _cand(
        "TOTAL_PASSIF",
        "9 114 715,17",
        section="BILAN_PASSIF",
        column="Exercice",
        label="TOTAL I",
        nature="SECTION_TOTAL",
    )
    ok, reasons = candidate_is_eligible(c_i)
    assert ok is False
    assert any("intermédiaire" in r.lower() for r in reasons)

    c_ok = _cand(
        "TOTAL_PASSIF",
        "22 303 497,11",
        section="BILAN_PASSIF",
        column="Exercice",
        label="TOTAL GENERAL I+II+III",
        nature="GRAND_TOTAL",
    )
    ok2, _ = candidate_is_eligible(c_ok)
    assert ok2 is True


def test_suspected_row_shift_blocks_clients_auto_confirm():
    c = _cand(
        "CLIENTS",
        "19 097 949,49",
        label="Clients et comptes rattachés",
        section="BILAN_ACTIF",
        column="Net",
        confidence=0.4,
    )
    c = c.model_copy(update={"warnings": ["suspected_row_shift"]})
    outputs = [FinancialMappingOutput(section="BILAN_ACTIF", candidates=[c])]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["CLIENTS"].status == "ambiguous"
    assert resolved["CLIENTS"].value is None


def test_serdilab_resultat_financier_derived_from_components():
    outputs = [
        FinancialMappingOutput(
            section="CPC",
            candidates=[
                _cand(
                    "PRODUITS_FINANCIERS",
                    "7 082,15",
                    section="CPC",
                    column="3 = 1 + 2",
                    label="Produits financiers",
                ),
                _cand(
                    "CHARGES_FINANCIERES",
                    "200 928,82",
                    section="CPC",
                    column="3 = 1 + 2",
                    label="Charges financières",
                ),
                _cand(
                    "RESULTAT_FINANCIER",
                    "200 928,82",
                    section="CPC",
                    column="3 = 1 + 2",
                    label="Résultat financier",
                ),
            ],
        )
    ]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["RESULTAT_FINANCIER"].value == Decimal("-193846.67")
    assert resolved["RESULTAT_FINANCIER"].status == "derived"
    assert any(
        "contradictoire" in w.lower() or "contredit" in w.lower()
        for w in resolved["RESULTAT_FINANCIER"].warnings
    )


def test_serdilab_core_resolved_fields():
    outputs = [
        FinancialMappingOutput(
            section="BILAN_PASSIF",
            candidates=[
                _cand(
                    "FONDS_PROPRES",
                    "9 114 715,17",
                    section="BILAN_PASSIF",
                    column="Exercice",
                    label="TOTAL DES CAPITAUX PROPRES",
                    nature="SECTION_TOTAL",
                ),
                _cand(
                    "TOTAL_PASSIF",
                    "22 303 497,11",
                    section="BILAN_PASSIF",
                    column="Exercice",
                    label="TOTAL GENERAL I+II+III",
                    nature="GRAND_TOTAL",
                ),
                _cand(
                    "TOTAL_PASSIF",
                    "9 114 715,17",
                    section="BILAN_PASSIF",
                    column="Exercice",
                    label="TOTAL I",
                    nature="SECTION_TOTAL",
                ),
                _cand(
                    "DETTES_FINANCIERES",
                    "133 308,11",
                    section="BILAN_PASSIF",
                    column="Exercice",
                    label="DETTES DE FINANCEMENT",
                    nature="SECTION_TOTAL",
                ),
                _cand(
                    "PASSIF_CIRCULANT",
                    "13 055 473,83",
                    section="BILAN_PASSIF",
                    column="Exercice",
                    label="Total II Passif circulant",
                    nature="SECTION_TOTAL",
                ),
                _cand(
                    "TRESORERIE_PASSIF",
                    "0,00",
                    section="BILAN_PASSIF",
                    column="Exercice",
                    label="Total III Trésorerie-Passif",
                    nature="SECTION_TOTAL",
                ),
                _cand(
                    "RESULTAT_NET",
                    "1 179 809,16",
                    section="BILAN_PASSIF",
                    column="Exercice",
                    label="Résultat net",
                ),
            ],
        ),
        FinancialMappingOutput(
            section="CPC",
            candidates=[
                _cand(
                    "CHIFFRE_AFFAIRES",
                    "13 404 177,00",
                    section="CPC",
                    column="3 = 1 + 2",
                    label="Chiffre d'affaires",
                ),
                _cand(
                    "CHIFFRE_AFFAIRES",
                    "1,00",
                    section="DETAIL_CPC",
                    column="Exercice",
                    label="Chiffre d'affaires",
                ),
                _cand(
                    "RESULTAT_NET",
                    "1 179 809,16",
                    section="CPC",
                    column="3 = 1 + 2",
                    label="XIII RESULTAT NET",
                    nature="SECTION_TOTAL",
                ),
                _cand(
                    "RESULTAT_NET",
                    "1 179 809,16",
                    section="CPC",
                    column="3 = 1 + 2",
                    label="XVI RESULTAT NET",
                    nature="SECTION_TOTAL",
                ),
                _cand(
                    "CHARGES_INTERETS",
                    "95 394,47",
                    section="CPC",
                    column="3 = 1 + 2",
                    label="Charges d'intérêts",
                ),
                _cand(
                    "PRODUITS_FINANCIERS",
                    "7 082,15",
                    section="CPC",
                    column="3 = 1 + 2",
                    label="Produits financiers",
                ),
                _cand(
                    "CHARGES_FINANCIERES",
                    "200 928,82",
                    section="CPC",
                    column="3 = 1 + 2",
                    label="Charges financières",
                ),
                _cand(
                    "RESULTAT_FINANCIER",
                    "200 928,82",
                    section="CPC",
                    column="3 = 1 + 2",
                    label="Résultat financier",
                ),
            ],
        ),
        FinancialMappingOutput(
            section="BILAN_ACTIF",
            candidates=[
                _cand(
                    "TOTAL_ACTIF",
                    "22 303 497,11",
                    section="BILAN_ACTIF",
                    column="Net",
                    label="TOTAL GENERAL I+II+III",
                    nature="GRAND_TOTAL",
                ),
            ],
        ),
    ]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["FONDS_PROPRES"].value == Decimal("9114715.17")
    assert resolved["RESULTAT_NET"].value == Decimal("1179809.16")
    assert resolved["RESULTAT_NET"].status == "confirmed"
    assert resolved["TOTAL_PASSIF"].value == Decimal("22303497.11")
    assert resolved["CHIFFRE_AFFAIRES"].value == Decimal("13404177.00")
    assert resolved["DETTES_FINANCIERES"].value == Decimal("133308.11")
    assert resolved["PASSIF_CIRCULANT"].value == Decimal("13055473.83")
    assert resolved["TRESORERIE_PASSIF"].value == Decimal("0.00")
    assert resolved["CHARGES_INTERETS"].value == Decimal("95394.47")
    assert resolved["PRODUITS_FINANCIERS"].value == Decimal("7082.15")
    assert resolved["CHARGES_FINANCIERES"].value == Decimal("200928.82")
    assert resolved["RESULTAT_FINANCIER"].value == Decimal("-193846.67")
    assert resolved["RESULTAT_FINANCIER"].status == "derived"
    assert resolved["TOTAL_ACTIF"].value == Decimal("22303497.11")
    assert resolved["TOTAL_BILAN"].value == Decimal("22303497.11")
