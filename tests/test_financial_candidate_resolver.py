# -*- coding: utf-8 -*-
"""Tests resolver déterministe des candidats financiers."""
from decimal import Decimal

from app.schemas.financial_mapping import (
    FinancialCandidate,
    FinancialMappingOutput,
    MappingEvidence,
)
from app.services.financial_candidate_resolver import (
    build_financial_dataset_from_resolved_values,
    candidate_is_eligible,
    candidate_priority,
    resolve_financial_candidates,
)


def _cand(
    field_code: str,
    raw_value: str,
    *,
    section: str = "BILAN_ACTIF",
    period: str = "N",
    nature: str = "DETAIL",
    label: str = "",
    column: str | None = "Net",
    page: int = 1,
    confidence: float = 0.5,
) -> FinancialCandidate:
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
    assert any("période" in r.lower() or "periode" in r.lower() for r in reasons)


def test_bilan_actif_prefers_net_column():
    net = _cand(
        "CLIENTS",
        "19 097 949,49",
        label="Clients et comptes rattachés",
        column="Net",
        confidence=0.1,
    )
    brut = _cand(
        "CLIENTS",
        "20 000 000,00",
        label="Clients et comptes rattachés",
        column="Brut",
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
        confidence=0.2,
    )
    other = _cand(
        "FOURNISSEURS",
        "1,00",
        label="Fournisseurs et comptes rattachés",
        section="BILAN_PASSIF",
        column="Autre",
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
        confidence=0.5,
    )
    other = _cand(
        "CHIFFRE_AFFAIRES",
        "1,00",
        label="Chiffre d'affaires",
        section="CPC",
        column="Opérations",
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
                    label="Total général",
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
                    label="Total général",
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
