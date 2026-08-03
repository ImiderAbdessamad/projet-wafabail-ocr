"""Décision crédit-bail et score final (Decimal)."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from app.scoring_rules import AXIS_WEIGHTS
from app.schemas.financial_analysis import AxisScore, CreditDecision, DecisionClass


def calculate_final_score(
    axes: list[AxisScore],
    *,
    allow_partial_axes: bool = False,
) -> tuple[Decimal | None, list[str], list[str]]:
    """Retourne (score, hard_blocking, soft_warnings).

    soft_warnings = axes comportemental/sector manquants (liasse seule).
    """
    by_code = {a.code: a for a in axes}
    financial = by_code.get("financial")
    behavioral = by_code.get("behavioral")
    sector = by_code.get("sector")
    hard: list[str] = []
    soft: list[str] = []

    if financial is None or not financial.calculable:
        hard.append(
            f"Axe {financial.code if financial else 'financial'} non calculable."
        )

    for axis, label in (
        (behavioral, "behavioral"),
        (sector, "sector"),
    ):
        if axis is None or not axis.calculable:
            msg = f"Axe {axis.code if axis else label} non calculable."
            if allow_partial_axes:
                soft.append(msg)
            else:
                hard.append(msg)

    if hard:
        return None, hard, soft

    assert financial is not None
    available = [a for a in (financial, behavioral, sector) if a is not None and a.calculable]
    weight_sum = sum((a.weight for a in available), Decimal("0"))
    if weight_sum <= 0:
        return None, ["Pondération d'axes invalide."], soft

    final = (
        sum((a.raw_score * a.weight for a in available), Decimal("0")) / weight_sum
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return final, [], soft


def build_credit_decision(
    final_score: Decimal | None,
    *,
    blocking_reasons: list[str],
    soft_warnings: list[str] | None = None,
) -> CreditDecision:
    soft_warnings = list(soft_warnings or [])
    if blocking_reasons:
        # Critères bloquants : jamais d'accord automatique
        severe = any(
            "BAM" in r or "impayé" in r.lower() or "fraude" in r.lower()
            for r in blocking_reasons
        )
        return CreditDecision(
            score=final_score,
            risk_class="D/F" if severe and final_score is not None and final_score < 50 else "NON_EVALUABLE",
            profile="Non évaluable",
            decision="Revue manuelle" if not severe else "Refus recommandé",
            recommendation="; ".join(blocking_reasons[:5]),
            blocking_status="BLOCKING_CRITERIA" if severe else "INSUFFICIENT_DATA",
        )

    if final_score is None:
        return CreditDecision(
            score=None,
            risk_class="NON_EVALUABLE",
            profile="Non évaluable",
            decision="Revue manuelle",
            recommendation="Score final non calculable.",
            blocking_status="INSUFFICIENT_DATA",
        )

    score = final_score
    risk_class: DecisionClass
    profile: str
    decision: str
    recommendation: str

    if score >= Decimal("90"):
        risk_class, profile = "A+", "Excellent"
        decision, recommendation = "Accord sans condition", "Profil excellent."
    elif score >= Decimal("80"):
        risk_class, profile = "A/B+", "Bon"
        decision, recommendation = (
            "Accord — conditions standards",
            "Profil bon.",
        )
    elif score >= Decimal("65"):
        risk_class, profile = "B/B-", "Moyen"
        decision, recommendation = (
            "Accord avec garanties complémentaires",
            "Garanties complémentaires recommandées.",
        )
    elif score >= Decimal("50"):
        risk_class, profile = "C", "Sensible"
        decision, recommendation = (
            "Accord conditionné ou refus partiel",
            "Dossier sensible.",
        )
    else:
        risk_class, profile = "D/F", "Risqué"
        decision, recommendation = "Refus recommandé", "Profil risqué."

    blocking_status = None
    if soft_warnings:
        # Score financier provisoire — axes externes absents → revue
        decision = "Revue manuelle (score financier provisoire)"
        recommendation = (
            f"{recommendation} " + "; ".join(soft_warnings[:3])
        ).strip()
        blocking_status = "PARTIAL_DATA"

    return CreditDecision(
        score=score,
        risk_class=risk_class,
        profile=profile,
        decision=decision,
        recommendation=recommendation,
        blocking_status=blocking_status,
    )
