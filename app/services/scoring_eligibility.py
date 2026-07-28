"""Règles d'éligibilité du scoring automatique.

Le scoring ne doit utiliser que des champs admissibles et suffisants.
"""
from __future__ import annotations

from typing import Iterable

from pydantic import BaseModel

from app.schemas.liasse import FinancialElement, LiasseExtractionResult
from app.schemas.scoring import BehavioralMetricsInput, FinancialDataInput

MIN_FINANCIAL_RATIOS_CALCULABLE = 8
MIN_FIELD_CONFIDENCE = 0.65
CRITICAL_FIELDS = {
    "CHIFFRE_AFFAIRES": "chiffre d'affaires",
    "TOTAL_BILAN": "total bilan",
    "FONDS_PROPRES": "fonds propres",
    "RESULTAT_NET": "résultat net",
    "CAF": "CAF",
    "FDR": "fonds de roulement",
}
ADMISSIBLE_STATUSES = {"detected", "detected_zero", "derived"}
NON_SCORABLE_STATUSES = {
    "ambiguous",
    "conflicting",
    "incomplete",
    "estimated",
    "invalidated",
    "not_detected",
    "not_present_in_document",
    "page_unreadable",
    "empty",
}


class ScoringEligibilityResult(BaseModel):
    eligible: bool
    mode: str
    blocking_reasons: list[str] = []
    warnings: list[str] = []
    financial_coverage: float = 0.0
    behavioral_coverage: float = 0.0
    sector_coverage: float = 0.0
    ratios_expected: int = 0
    ratios_calculable: int = 0
    ratio_coverage: float = 0.0


def is_field_usable(
    status: str | None,
    confidence: float | None,
    validation_status: str | None,
    *,
    eligible_for_scoring: bool = True,
) -> bool:
    """Détermine si une valeur peut alimenter le scoring."""
    if not eligible_for_scoring:
        return False
    if status not in ADMISSIBLE_STATUSES:
        return False
    if validation_status in {"invalidated", "divergent", "incomplete", "failed"}:
        return False
    if confidence is None or confidence < MIN_FIELD_CONFIDENCE:
        return False
    return True


def evaluate_extraction_eligibility(
    extraction: LiasseExtractionResult,
    ratios: dict[str, dict] | None = None,
    behavioral_data: BehavioralMetricsInput | None = None,
    *,
    scoring_mode: str = "actual",
) -> ScoringEligibilityResult:
    """Évalue l'éligibilité d'une extraction structurée au scoring automatique."""
    elements_by_code = {element.code: element for element in extraction.elements}
    blocking_reasons: list[str] = []
    warnings: list[str] = []

    usable_critical = 0
    for code, label in CRITICAL_FIELDS.items():
        element = elements_by_code.get(code)
        if not element or not is_field_usable(
            element.detection_status,
            element.confidence,
            element.validation.status if element.validation else None,
            eligible_for_scoring=element.eligible_for_scoring,
        ):
            blocking_reasons.append(f"Champ critique indisponible ou non fiable : {label}.")
        else:
            usable_critical += 1

    financial_coverage = usable_critical / max(len(CRITICAL_FIELDS), 1)

    ratios_expected = 0
    ratios_calculable = 0
    if ratios:
        for ratio in ratios.values():
            ratios_expected += 1
            if ratio.get("status") in {"Conforme", "À surveiller", "Non conforme"}:
                ratios_calculable += 1
        if ratios_calculable < MIN_FINANCIAL_RATIOS_CALCULABLE:
            blocking_reasons.append(
                f"Couverture ratios insuffisante : {ratios_calculable}/{ratios_expected} "
                f"(< {MIN_FINANCIAL_RATIOS_CALCULABLE})."
            )

    behavioral_coverage = compute_behavioral_coverage(behavioral_data)
    if behavioral_coverage == 0.0:
        warnings.append("Données comportementales non renseignées.")

    eligible = not blocking_reasons
    if scoring_mode != "actual":
        blocking_reasons.append("Le mode de scoring n'autorise pas une décision réelle.")
        eligible = False

    return ScoringEligibilityResult(
        eligible=eligible,
        mode="automatic" if eligible else "manual_review",
        blocking_reasons=blocking_reasons,
        warnings=warnings,
        financial_coverage=round(financial_coverage, 4),
        behavioral_coverage=round(behavioral_coverage, 4),
        sector_coverage=0.0,
        ratios_expected=ratios_expected,
        ratios_calculable=ratios_calculable,
        ratio_coverage=round(ratios_calculable / ratios_expected, 4) if ratios_expected else 0.0,
    )


def compute_behavioral_coverage(behavioral_data: BehavioralMetricsInput | None) -> float:
    if behavioral_data is None:
        return 0.0
    values = behavioral_data.model_dump()
    provided_fields = set(getattr(behavioral_data, "model_fields_set", set()))
    relevant = [
        "domiciliation_ca_pct",
        "jours_debit",
        "utilisation_decouvert_pct",
        "ecart_flux_ca_pct",
        "engagements_honores",
        "incidents_paiement",
        "rejets_prelevement",
        "effets_impayes",
    ]
    provided = 0
    for key in relevant:
        value = values.get(key)
        if key in provided_fields and value is not None:
            provided += 1
    return provided / len(relevant)


def evaluate_manual_request_eligibility(
    financial_data: FinancialDataInput,
    behavioral_data: BehavioralMetricsInput | None,
    ratios: dict[str, dict],
) -> ScoringEligibilityResult:
    """Éligibilité pour /scoring/evaluate quand seules les données métier sont connues."""
    blocking_reasons: list[str] = []
    usable_critical = 0
    raw = financial_data.model_dump()
    for _, label in CRITICAL_FIELDS.items():
        pass
    for field, label in (
        ("chiffre_affaires", "chiffre d'affaires"),
        ("total_bilan", "total bilan"),
        ("fonds_propres", "fonds propres"),
        ("resultat_net", "résultat net"),
        ("dettes_financieres", "dettes financières"),
        ("caf", "CAF"),
        ("fdr", "fonds de roulement"),
    ):
        value = raw.get(field)
        if value is None:
            blocking_reasons.append(f"Champ critique manquant : {label}.")
        else:
            usable_critical += 1

    ratios_expected = len(ratios)
    ratios_calculable = sum(
        1 for ratio in ratios.values() if ratio.get("status") in {"Conforme", "À surveiller", "Non conforme"}
    )
    if ratios_calculable < MIN_FINANCIAL_RATIOS_CALCULABLE:
        blocking_reasons.append(
            f"Couverture ratios insuffisante : {ratios_calculable}/{ratios_expected} "
            f"(< {MIN_FINANCIAL_RATIOS_CALCULABLE})."
        )

    behavioral_coverage = compute_behavioral_coverage(behavioral_data)
    warnings = ["Données comportementales non renseignées."] if behavioral_coverage == 0.0 else []

    eligible = not blocking_reasons
    return ScoringEligibilityResult(
        eligible=eligible,
        mode="automatic" if eligible else "manual_review",
        blocking_reasons=blocking_reasons,
        warnings=warnings,
        financial_coverage=round(usable_critical / 7, 4),
        behavioral_coverage=round(behavioral_coverage, 4),
        sector_coverage=0.0,
        ratios_expected=ratios_expected,
        ratios_calculable=ratios_calculable,
        ratio_coverage=round(ratios_calculable / ratios_expected, 4) if ratios_expected else 0.0,
    )
