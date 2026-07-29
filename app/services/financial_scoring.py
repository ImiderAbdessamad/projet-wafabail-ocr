"""Score axe financier (75 %) — Decimal, modes STRICT / REVIEW."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from app.scoring_rules import (
    AXIS_WEIGHTS,
    ESSENTIAL_FIELDS_FOR_SCORING,
    FINANCIAL_RATIO_RULES,
    USABLE_FIELD_STATUSES,
)
from app.schemas.financial_analysis import (
    AxisScore,
    FinancialDataset,
    RatioResult,
)
from app.services.financial_dataset_builder import usable


def score_ratio(ratio: RatioResult, max_points: Decimal) -> Decimal:
    factors = {
        "conforme": Decimal("1.00"),
        "a_surveiller": Decimal("0.60"),
        "non_conforme": Decimal("0.00"),
        "non_calculable": Decimal("0.00"),
    }
    return (max_points * factors[ratio.status]).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )


def calculate_financial_score(
    dataset: FinancialDataset,
    ratios: list[RatioResult],
    *,
    scoring_mode: str = "STRICT",
) -> AxisScore:
    blocking: list[str] = []

    for field_name in ESSENTIAL_FIELDS_FOR_SCORING:
        fv = getattr(dataset, field_name, None)
        if fv is None or fv.status not in USABLE_FIELD_STATUSES or fv.value is None:
            blocking.append(f"Champ essentiel manquant ou non fiable : {field_name}")

    for check_warning in dataset.warnings:
        if "invalidé" in check_warning.lower() or "conflicting" in check_warning.lower():
            blocking.append(check_warning)

    weighted_ratios = [r for r in ratios if r.code in FINANCIAL_RATIO_RULES]
    total_max = sum(
        (Decimal(str(FINANCIAL_RATIO_RULES[r.code]["weight"])) for r in weighted_ratios),
        Decimal("0"),
    )
    total_points = sum((r.points for r in weighted_ratios), Decimal("0"))

    essential_missing = any(
        r.status == "non_calculable"
        and FINANCIAL_RATIO_RULES.get(r.code, {}).get("essential")
        for r in weighted_ratios
    )
    if essential_missing:
        blocking.append("Au moins un ratio financier essentiel est non calculable.")

    calculable = True
    if scoring_mode.upper() == "STRICT" and blocking:
        calculable = False
        raw_score = Decimal("0")
    elif total_max == 0:
        calculable = False
        raw_score = Decimal("0")
        blocking.append("Aucun ratio financier pondéré disponible.")
    else:
        raw_score = (total_points / total_max * Decimal("100")).quantize(
            Decimal("0.01"),
            rounding=ROUND_HALF_UP,
        )
        if scoring_mode.upper() == "REVIEW" and blocking:
            # Note informative mais axe marqué non pleinement calculable
            calculable = False

    weight = AXIS_WEIGHTS["financial"]
    contribution = (
        (raw_score * weight).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if calculable
        else Decimal("0")
    )

    return AxisScore(
        code="financial",
        label="Ratios financiers",
        raw_score=raw_score,
        weight=weight,
        weighted_contribution=contribution,
        calculable=calculable,
        blocking_reasons=blocking,
    )
