"""Score comportemental (15 %) — données bancaires externes, jamais inventées."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from app.scoring_rules import (
    AXIS_WEIGHTS,
    BAM_BLOCKING_RATINGS,
    BEHAVIORAL_RULES,
)
from app.schemas.financial_analysis import AxisScore, BehavioralInput


def calculate_behavioral_score(
    behavioral: BehavioralInput | None,
) -> tuple[AxisScore, list[str]]:
    blocking: list[str] = []
    weight = AXIS_WEIGHTS["behavioral"]

    if behavioral is None:
        return (
            AxisScore(
                code="behavioral",
                label="Indicateurs comportementaux",
                raw_score=Decimal("0"),
                weight=weight,
                weighted_contribution=Decimal("0"),
                calculable=False,
                blocking_reasons=["Données comportementales non fournies."],
            ),
            ["Données comportementales non fournies."],
        )

    if behavioral.bam_rating in BAM_BLOCKING_RATINGS:
        blocking.append(
            f"Cotation BAM bloquante : {behavioral.bam_rating}."
        )

    if (behavioral.payment_incidents_24m or 0) > 0:
        blocking.append("Incident(s) de paiement sur 24 mois.")
    if (behavioral.unpaid_bills_24m or 0) > 0:
        blocking.append("Effet(s) impayé(s) sur 24 mois.")
    if (behavioral.rejected_debits_24m or 0) > 0:
        blocking.append("Rejet(s) de prélèvement sur 24 mois.")

    points = Decimal("0")
    max_points = Decimal("0")

    def add(score: Decimal, maximum: Decimal) -> None:
        nonlocal points, max_points
        points += score
        max_points += maximum

    # Domiciliation CA
    max_points += Decimal("20")
    if behavioral.ca_domiciliation_pct is not None:
        if behavioral.ca_domiciliation_pct >= BEHAVIORAL_RULES["domiciliation_good"]:
            points += Decimal("20")
        elif behavioral.ca_domiciliation_pct >= Decimal("60"):
            points += Decimal("12")
        else:
            points += Decimal("4")

    # Découvert
    max_points += Decimal("15")
    if behavioral.overdraft_usage_pct is not None:
        if behavioral.overdraft_usage_pct <= BEHAVIORAL_RULES["overdraft_watch"]:
            points += Decimal("15")
        elif behavioral.overdraft_usage_pct <= Decimal("70"):
            points += Decimal("8")
        else:
            points += Decimal("0")

    # Jours débiteurs
    max_points += Decimal("15")
    if behavioral.debit_position_days is not None:
        if behavioral.debit_position_days <= BEHAVIORAL_RULES["debit_days_watch"]:
            points += Decimal("15")
        elif behavioral.debit_position_days <= 90:
            points += Decimal("8")
        else:
            points += Decimal("0")

    # Écart flux / CA
    max_points += Decimal("15")
    if behavioral.bank_flows_vs_declared_ca_gap_pct is not None:
        gap = abs(behavioral.bank_flows_vs_declared_ca_gap_pct)
        if gap <= BEHAVIORAL_RULES["flow_gap_watch"]:
            points += Decimal("15")
        elif gap <= Decimal("15"):
            points += Decimal("8")
        else:
            points += Decimal("0")

    # Retards leasing
    max_points += Decimal("15")
    if behavioral.leasing_payment_delays_24m is not None:
        if behavioral.leasing_payment_delays_24m == 0:
            points += Decimal("15")
        else:
            points += Decimal("0")
            blocking.append("Retard(s) de paiement leasing.")

    # Incidents / rejets déjà bloquants : 0 point sur ce volet
    max_points += Decimal("20")
    if (
        (behavioral.payment_incidents_24m or 0) == 0
        and (behavioral.rejected_debits_24m or 0) == 0
        and (behavioral.unpaid_bills_24m or 0) == 0
    ):
        points += Decimal("20")

    provided = any(
        [
            behavioral.ca_domiciliation_pct is not None,
            behavioral.overdraft_usage_pct is not None,
            behavioral.debit_position_days is not None,
            behavioral.bank_flows_vs_declared_ca_gap_pct is not None,
            behavioral.leasing_payment_delays_24m is not None,
            behavioral.bam_rating is not None,
            behavioral.payment_incidents_24m is not None,
        ]
    )

    if not provided or max_points == 0:
        return (
            AxisScore(
                code="behavioral",
                label="Indicateurs comportementaux",
                raw_score=Decimal("0"),
                weight=weight,
                weighted_contribution=Decimal("0"),
                calculable=False,
                blocking_reasons=["Indicateurs comportementaux insuffisants."],
            ),
            blocking or ["Indicateurs comportementaux insuffisants."],
        )

    raw = (points / max_points * Decimal("100")).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
    calculable = not blocking
    contribution = (
        (raw * weight).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if calculable
        else Decimal("0")
    )

    return (
        AxisScore(
            code="behavioral",
            label="Indicateurs comportementaux",
            raw_score=raw,
            weight=weight,
            weighted_contribution=contribution,
            calculable=calculable,
            blocking_reasons=blocking,
        ),
        blocking,
    )
