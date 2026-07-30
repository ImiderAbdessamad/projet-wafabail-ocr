"""Score sectoriel (10 %) — comparaison médiane uniquement, pas de percentile inventé."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from app.scoring_rules import AXIS_WEIGHTS
from app.schemas.financial_analysis import AxisScore, RatioResult, SectorBenchmarkInput


def _compare_to_median(
    value: Decimal | None,
    median: Decimal | None,
    *,
    higher_is_better: bool,
) -> tuple[str | None, Decimal | None]:
    if value is None or median is None:
        return None, None
    favorable = value >= median if higher_is_better else value <= median
    comparison = "above_median" if value >= median else "below_median"
    points = Decimal("20") if favorable else Decimal("0")
    return comparison, points


def calculate_sector_score(
    ratios: list[RatioResult],
    sector: SectorBenchmarkInput | None,
) -> AxisScore:
    weight = AXIS_WEIGHTS["sector"]
    if sector is None:
        return AxisScore(
            code="sector",
            label="Positionnement sectoriel",
            raw_score=Decimal("0"),
            weight=weight,
            weighted_contribution=Decimal("0"),
            calculable=False,
            blocking_reasons=["Benchmark sectoriel non fourni."],
        )

    by_code = {r.code: r for r in ratios}
    available_comparisons = 0
    points = Decimal("0")
    max_points = Decimal("0")

    comparisons = [
        (
            "commercial_profitability",
            sector.commercial_profitability_median,
            True,
        ),
        (
            "financial_autonomy",
            sector.financial_autonomy_median,
            True,
        ),
        (
            "debt_ratio",
            sector.debt_ratio_median,
            False,
        ),
        (
            "repayment_capacity",
            sector.repayment_capacity_median,
            False,
        ),
        (
            "ca_growth",
            sector.ca_growth_median,
            True,
        ),
    ]

    for code, median, higher in comparisons:
        ratio = by_code.get(code)
        comparison, comparison_points = _compare_to_median(
            ratio.value if ratio else None,
            median,
            higher_is_better=higher,
        )
        if comparison_points is None:
            continue
        available_comparisons += 1
        points += comparison_points
        max_points += Decimal("20")

    if available_comparisons < 3:
        return AxisScore(
            code="sector",
            label="Positionnement sectoriel",
            raw_score=Decimal("0"),
            weight=weight,
            weighted_contribution=Decimal("0"),
            calculable=False,
            blocking_reasons=["Moins de trois comparaisons sectorielles disponibles."],
        )

    raw = (points / max_points * Decimal("100")).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
    contribution = (raw * weight).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    return AxisScore(
        code="sector",
        label="Positionnement sectoriel",
        raw_score=raw,
        weight=weight,
        weighted_contribution=contribution,
        calculable=True,
        blocking_reasons=[],
    )
