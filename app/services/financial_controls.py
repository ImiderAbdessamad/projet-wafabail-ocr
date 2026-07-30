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

    # 1. TOTAL ACTIF N = TOTAL PASSIF N
    if usable(dataset.total_actif) and usable(dataset.total_passif):
        checks.append(
            _check(
                "bilan_equilibre",
                dataset.total_actif.value if dataset.total_actif else None,
                dataset.total_passif.value if dataset.total_passif else None,
                ["TOTAL_ACTIF", "TOTAL_PASSIF", "TOTAL_BILAN"],
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

    # 12. Trésorerie nette = actif - passif
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
        elif dataset.tresorerie_nette.status == "missing":
            # Dérivation laissée au builder ; contrôle informatif si absente
            checks.append(
                AccountingControlResult(
                    code="tresorerie_nette",
                    status="not_testable",
                    expected=expected_tn,
                    affected_fields=["TRESORERIE_ACTIF", "TRESORERIE_PASSIF"],
                    message="Trésorerie nette non encore dérivée.",
                )
            )

    # 4. Produits exploitation - charges = résultat exploitation
    if (
        usable(dataset.produits_exploitation)
        and usable(dataset.charges_exploitation)
        and usable(dataset.resultat_exploitation)
    ):
        expected = (
            dataset.produits_exploitation.value - dataset.charges_exploitation.value  # type: ignore[operator]
        )
        checks.append(
            _check(
                "resultat_exploitation",
                expected,
                dataset.resultat_exploitation.value,
                ["PRODUITS_EXPLOITATION", "CHARGES_EXPLOITATION", "RESULTAT_EXPLOITATION"],
                "Résultat d'exploitation cohérent.",
                "Résultat d'exploitation incohérent.",
            )
        )

    # 5. Produits financiers - charges financières = résultat financier
    if (
        usable(dataset.produits_financiers)
        and usable(dataset.charges_financieres)
        and usable(dataset.resultat_financier)
    ):
        expected = (
            dataset.produits_financiers.value - dataset.charges_financieres.value  # type: ignore[operator]
        )
        checks.append(
            _check(
                "resultat_financier",
                expected,
                dataset.resultat_financier.value,
                ["PRODUITS_FINANCIERS", "CHARGES_FINANCIERES", "RESULTAT_FINANCIER"],
                "Résultat financier cohérent.",
                "Résultat financier incohérent.",
            )
        )

    # 6. Résultat exploitation + financier = courant
    if (
        usable(dataset.resultat_exploitation)
        and usable(dataset.resultat_financier)
        and usable(dataset.resultat_courant)
    ):
        expected = (
            dataset.resultat_exploitation.value + dataset.resultat_financier.value  # type: ignore[operator]
        )
        checks.append(
            _check(
                "resultat_courant",
                expected,
                dataset.resultat_courant.value,
                ["RESULTAT_EXPLOITATION", "RESULTAT_FINANCIER", "RESULTAT_COURANT"],
                "Résultat courant cohérent.",
                "Résultat courant incohérent.",
            )
        )

    # 7. Produits NC - charges NC = résultat NC
    if (
        usable(dataset.produits_non_courants)
        and usable(dataset.charges_non_courantes)
        and usable(dataset.resultat_non_courant)
    ):
        expected = (
            dataset.produits_non_courants.value - dataset.charges_non_courantes.value  # type: ignore[operator]
        )
        checks.append(
            _check(
                "resultat_non_courant",
                expected,
                dataset.resultat_non_courant.value,
                [
                    "PRODUITS_NON_COURANTS",
                    "CHARGES_NON_COURANTES",
                    "RESULTAT_NON_COURANT",
                ],
                "Résultat non courant cohérent.",
                "Résultat non courant incohérent.",
            )
        )

    # 8. Courant + non courant = avant impôt
    if (
        usable(dataset.resultat_courant)
        and usable(dataset.resultat_non_courant)
        and usable(dataset.resultat_avant_impot)
    ):
        expected = (
            dataset.resultat_courant.value + dataset.resultat_non_courant.value  # type: ignore[operator]
        )
        checks.append(
            _check(
                "resultat_avant_impot",
                expected,
                dataset.resultat_avant_impot.value,
                ["RESULTAT_COURANT", "RESULTAT_NON_COURANT", "RESULTAT_AVANT_IMPOT"],
                "Résultat avant impôt cohérent.",
                "Résultat avant impôt incohérent.",
            )
        )

    # 9. Résultat avant IS - IS = résultat net
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
                ["RESULTAT_AVANT_IMPOT", "IMPOT_SUR_RESULTATS", "RESULTAT_NET"],
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
