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
) -> tuple[str | None, Decimal]:
    """Retourne (comparison, points 0-20)."""
    if value is None or median is None:
        return None, Decimal("0")
    above = value >= median
    if higher_is_better:
        return (
            "above_median" if above else "below_median",
            Decimal("20") if above else Decimal("8"),
        )
    return (
        "below_median" if value <= median else "above_median",
        Decimal("20") if value <= median else Decimal("8"),
    )


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
    points = Decimal("0")
    max_points = Decimal("0")
    notes: list[str] = []

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
        max_points += Decimal("20")
        ratio = by_code.get(code)
        comparison, pts = _compare_to_median(
            ratio.value if ratio else None,
            median,
            higher_is_better=higher,
        )
        points += pts
        if comparison:
            notes.append(f"{code}={comparison} (percentile=None)")

    if max_points == 0 or points == 0 and not notes:
        return AxisScore(
            code="sector",
            label="Positionnement sectoriel",
            raw_score=Decimal("0"),
            weight=weight,
            weighted_contribution=Decimal("0"),
            calculable=False,
            blocking_reasons=["Médianes sectorielles insuffisantes."],
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
