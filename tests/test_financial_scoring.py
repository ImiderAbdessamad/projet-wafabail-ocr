# -*- coding: utf-8 -*-
"""Tests scoring axes et décision."""
from decimal import Decimal

from app.schemas.financial_analysis import AxisScore, BehavioralInput
from app.services.behavioral_scoring import calculate_behavioral_score
from app.services.credit_decision import build_credit_decision, calculate_final_score
from app.services.financial_scoring import calculate_financial_score, score_ratio
from app.schemas.financial_analysis import RatioResult
from tests.test_financial_ratios import reference_dataset
from app.services.financial_ratios import calculate_financial_ratios


def test_score_ratio_factors():
    r = RatioResult(
        code="x",
        label="x",
        formula="a/b",
        unit="%",
        status="a_surveiller",
    )
    assert score_ratio(r, Decimal("10")) == Decimal("6.00")


def test_final_score_83():
    axes = [
        AxisScore(
            code="financial",
            label="F",
            raw_score=Decimal("85"),
            weight=Decimal("0.75"),
            weighted_contribution=Decimal("63.75"),
            calculable=True,
        ),
        AxisScore(
            code="behavioral",
            label="B",
            raw_score=Decimal("75"),
            weight=Decimal("0.15"),
            weighted_contribution=Decimal("11.25"),
            calculable=True,
        ),
        AxisScore(
            code="sector",
            label="S",
            raw_score=Decimal("80"),
            weight=Decimal("0.10"),
            weighted_contribution=Decimal("8.00"),
            calculable=True,
        ),
    ]
    final, blocking = calculate_final_score(axes)
    assert blocking == []
    assert final == Decimal("83.00")


def test_decision_grid_a_b_plus():
    d = build_credit_decision(Decimal("83.00"), blocking_reasons=[])
    assert d.risk_class == "A/B+"
    assert "Accord" in d.decision


def test_bam_7_blocks():
    axis, blocking = calculate_behavioral_score(
        BehavioralInput(bam_rating=7, ca_domiciliation_pct=Decimal("90"))
    )
    assert any("BAM" in b for b in blocking)
    assert axis.calculable is False


def test_strict_mode_blocks_missing_essential():
    ds = reference_dataset()
    ds.chiffre_affaires.status = "missing"
    ds.chiffre_affaires.value = None
    ratios = calculate_financial_ratios(ds)
    axis = calculate_financial_score(ds, ratios, scoring_mode="STRICT")
    assert axis.calculable is False
    assert any("chiffre_affaires" in r for r in axis.blocking_reasons)


def test_blocking_prevents_automatic_accord():
    d = build_credit_decision(
        Decimal("90"),
        blocking_reasons=["Cotation BAM bloquante : 8."],
    )
    assert d.blocking_status is not None
    assert d.decision in {"Revue manuelle", "Refus recommandé"}
    assert d.risk_class in {"NON_EVALUABLE", "D/F"}
