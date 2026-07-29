"""Contrôles de cohérence comptable (Decimal, auditables)."""
from __future__ import annotations

from decimal import Decimal

from app.schemas.financial_analysis import (
    AccountingControlResult,
    FinancialDataset,
    FinancialValue,
)
from app.services.financial_dataset_builder import usable


def _tol(reference: Decimal | None) -> Decimal:
    if reference is None:
        return Decimal("1.00")
    return max(Decimal("1.00"), abs(reference) * Decimal("0.0001"))


def _check(
    code: str,
    expected: Decimal | None,
    observed: Decimal | None,
    affected: list[str],
    message_ok: str,
    message_fail: str,
) -> AccountingControlResult:
    if expected is None or observed is None:
        return AccountingControlResult(
            code=code,
            status="not_testable",
            expected=expected,
            observed=observed,
            affected_fields=affected,
            message="Données insuffisantes pour le contrôle.",
        )
    tolerance = _tol(expected)
    difference = observed - expected
    if abs(difference) <= tolerance:
        return AccountingControlResult(
            code=code,
            status="passed",
            expected=expected,
            observed=observed,
            difference=difference,
            tolerance=tolerance,
            affected_fields=affected,
            message=message_ok,
        )
    return AccountingControlResult(
        code=code,
        status="failed",
        expected=expected,
        observed=observed,
        difference=difference,
        tolerance=tolerance,
        affected_fields=affected,
        message=message_fail,
    )


def run_accounting_controls(
    dataset: FinancialDataset,
) -> list[AccountingControlResult]:
    checks: list[AccountingControlResult] = []

    total_actif = dataset.total_actif.value if usable(dataset.total_actif) else (
        dataset.total_bilan.value if usable(dataset.total_bilan) else None
    )
    total_passif = dataset.total_passif.value if usable(dataset.total_passif) else (
        dataset.total_bilan.value if usable(dataset.total_bilan) else None
    )
    # Si un seul total_bilan : équilibre non testable sans actif/passif séparés
    if usable(dataset.total_actif) and usable(dataset.total_passif):
        checks.append(
            _check(
                "bilan_equilibre",
                dataset.total_actif.value if dataset.total_actif else None,
                dataset.total_passif.value if dataset.total_passif else None,
                ["TOTAL_ACTIF", "TOTAL_PASSIF"],
                "Actif ≈ Passif.",
                "Déséquilibre actif / passif au-delà de la tolérance.",
            )
        )
    else:
        checks.append(
            AccountingControlResult(
                code="bilan_equilibre",
                status="not_testable",
                affected_fields=["TOTAL_BILAN"],
                message="Actif et passif séparés non disponibles.",
            )
        )

    # Trésorerie nette
    if usable(dataset.tresorerie_actif) and usable(dataset.tresorerie_passif):
        expected_tn = dataset.tresorerie_actif.value - dataset.tresorerie_passif.value  # type: ignore[operator]
        observed_tn = (
            dataset.tresorerie_nette.value if usable(dataset.tresorerie_nette) else None
        )
        if observed_tn is not None:
            checks.append(
                _check(
                    "tresorerie_nette",
                    expected_tn,
                    observed_tn,
                    ["TRESORERIE_ACTIF", "TRESORERIE_PASSIF", "TRESORERIE_NETTE"],
                    "Trésorerie nette cohérente.",
                    "Trésorerie nette incohérente avec actif - passif.",
                )
            )

    # Résultat net ≈ résultat avant IS - IS (si dispo)
    if (
        usable(dataset.resultat_avant_impot)
        and usable(dataset.impot_sur_resultats)
        and usable(dataset.resultat_net)
    ):
        expected_rn = (
            dataset.resultat_avant_impot.value - dataset.impot_sur_resultats.value  # type: ignore[operator]
        )
        checks.append(
            _check(
                "resultat_net",
                expected_rn,
                dataset.resultat_net.value,
                ["RESULTAT_AVANT_IMPOT", "IS", "RESULTAT_NET"],
                "Résultat net cohérent.",
                "Résultat net incohérent avec résultat avant IS - IS.",
            )
        )

    return checks


def invalidate_conflicting_fields(
    dataset: FinancialDataset,
    checks: list[AccountingControlResult],
) -> FinancialDataset:
    """Marque les champs impactés par un contrôle échoué comme conflicting."""
    for check in checks:
        if check.status != "failed":
            continue
        for field_code in check.affected_fields:
            attr = field_code.lower()
            # Map common codes
            mapping = {
                "TOTAL_ACTIF": "total_actif",
                "TOTAL_PASSIF": "total_passif",
                "TOTAL_BILAN": "total_bilan",
                "TRESORERIE_NETTE": "tresorerie_nette",
                "RESULTAT_NET": "resultat_net",
            }
            attr_name = mapping.get(field_code, attr)
            fv: FinancialValue | None = getattr(dataset, attr_name, None)
            if fv is None:
                continue
            if fv.status in {"confirmed", "derived"}:
                fv.status = "conflicting"
                fv.warnings.append(check.message)
                fv.value = None
                setattr(dataset, attr_name, fv)
                dataset.warnings.append(
                    f"Champ {field_code} invalidé suite au contrôle {check.code}."
                )
    return dataset
