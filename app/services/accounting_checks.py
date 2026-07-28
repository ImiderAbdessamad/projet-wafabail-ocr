"""Contrôles de cohérence comptable indépendants de l'extraction."""
from __future__ import annotations

from decimal import Decimal

from app.schemas.observations import FieldResolution
from app.schemas.liasse import AccountingCheckResult

DEFAULT_TOLERANCE = Decimal("0.01")


def run_accounting_checks(
    resolved: dict[str, FieldResolution],
    *,
    tolerance: Decimal = DEFAULT_TOLERANCE,
) -> tuple[list[str], dict[str, str], list[AccountingCheckResult]]:
    """Retourne (warnings, validation_par_code, contrôles structurés)."""
    warnings: list[str] = []
    status: dict[str, str] = {}
    checks: list[AccountingCheckResult] = []

    def v(code: str) -> Decimal | None:
        item = resolved.get(code)
        return item.selected_value if item else None

    def register(
        check_code: str,
        fields: list[str],
        *,
        status_value: str,
        severity: str,
        expected: Decimal | None,
        actual: Decimal | None,
        diff: Decimal | None,
        message: str,
    ) -> None:
        checks.append(
            AccountingCheckResult(
                check_code=check_code,
                status=status_value,
                severity=severity,
                fields=fields,
                expected_value=float(expected) if expected is not None else None,
                actual_value=float(actual) if actual is not None else None,
                difference=float(diff) if diff is not None else None,
                tolerance=float(tolerance),
                message=message,
            )
        )

    total_actif = v("TOTAL_BILAN")
    total_passif = v("TOTAL_PASSIF")
    if total_actif is not None and total_passif is not None:
        diff = abs(total_actif - total_passif)
        if diff > tolerance:
            msg = f"Total actif net ({total_actif}) ≠ total passif ({total_passif})."
            warnings.append(msg)
            status["TOTAL_BILAN"] = "warning"
            register("TOTAL_ACTIF_EQ_TOTAL_PASSIF", ["TOTAL_BILAN", "TOTAL_PASSIF"], status_value="warning", severity="high", expected=total_passif, actual=total_actif, diff=diff, message=msg)
        else:
            status["TOTAL_BILAN"] = "consistent"
            register("TOTAL_ACTIF_EQ_TOTAL_PASSIF", ["TOTAL_BILAN", "TOTAL_PASSIF"], status_value="passed", severity="info", expected=total_passif, actual=total_actif, diff=diff, message="Bilan équilibré.")
    else:
        register("TOTAL_ACTIF_EQ_TOTAL_PASSIF", ["TOTAL_BILAN", "TOTAL_PASSIF"], status_value="insufficient_data", severity="info", expected=total_passif, actual=total_actif, diff=None, message="Total passif direct indisponible.")

    ta = v("TRESORERIE_ACTIF")
    tp = v("TRESORERIE_PASSIF")
    if ta is not None and tp is not None:
        tn = v("TRESORERIE_NETTE")
        expected = ta - tp
        if tn is not None and abs(tn - expected) > tolerance:
            msg = f"Trésorerie nette incohérente : {tn} ≠ {ta} - {tp}"
            warnings.append(msg)
            status["TRESORERIE_NETTE"] = "invalidated"
            register("TRESORERIE_NETTE_EQ_ACTIF_MINUS_PASSIF", ["TRESORERIE_ACTIF", "TRESORERIE_PASSIF", "TRESORERIE_NETTE"], status_value="failed", severity="high", expected=expected, actual=tn, diff=abs(tn - expected), message=msg)
        elif tn is not None:
            status["TRESORERIE_NETTE"] = "consistent"
            register("TRESORERIE_NETTE_EQ_ACTIF_MINUS_PASSIF", ["TRESORERIE_ACTIF", "TRESORERIE_PASSIF", "TRESORERIE_NETTE"], status_value="passed", severity="info", expected=expected, actual=tn, diff=abs(tn - expected), message="Trésorerie nette cohérente.")
    else:
        register("TRESORERIE_NETTE_EQ_ACTIF_MINUS_PASSIF", ["TRESORERIE_ACTIF", "TRESORERIE_PASSIF", "TRESORERIE_NETTE"], status_value="insufficient_data", severity="info", expected=None, actual=v("TRESORERIE_NETTE"), diff=None, message="Données insuffisantes pour contrôler la trésorerie nette.")

    mlt = v("DETTES_BANCAIRES_MLT")
    ct = v("DETTES_BANCAIRES_CT")
    df = v("DETTES_FINANCIERES")
    if mlt is not None and ct is not None and df is not None:
        if abs(df - (mlt + ct)) > tolerance:
            msg = "Dettes financières ≠ MLT + CT"
            warnings.append(msg)
            status["DETTES_FINANCIERES"] = "invalidated"
            register("DETTES_FIN_EQ_MLT_PLUS_CT", ["DETTES_BANCAIRES_MLT", "DETTES_BANCAIRES_CT", "DETTES_FINANCIERES"], status_value="failed", severity="high", expected=mlt + ct, actual=df, diff=abs(df - (mlt + ct)), message=msg)
        else:
            status["DETTES_FINANCIERES"] = "consistent"
            register("DETTES_FIN_EQ_MLT_PLUS_CT", ["DETTES_BANCAIRES_MLT", "DETTES_BANCAIRES_CT", "DETTES_FINANCIERES"], status_value="passed", severity="info", expected=mlt + ct, actual=df, diff=abs(df - (mlt + ct)), message="Dettes financières cohérentes.")
    else:
        register("DETTES_FIN_EQ_MLT_PLUS_CT", ["DETTES_BANCAIRES_MLT", "DETTES_BANCAIRES_CT", "DETTES_FINANCIERES"], status_value="insufficient_data", severity="info", expected=None, actual=df, diff=None, message="Données insuffisantes pour contrôler les dettes financières.")

    # Trésorerie passif ≈ crédits + banques
    credits = v("CREDITS_TRESORERIE")
    banques = v("BANQUES_SOLDES_CREDITEURS")
    if tp is not None and credits is not None and banques is not None:
        if abs(tp - (credits + banques)) > tolerance:
            msg = f"Trésorerie-passif ({tp}) ≠ crédits+banques ({credits + banques})"
            warnings.append(msg)
            status["TRESORERIE_PASSIF"] = "warning"
            register("TRESO_PASSIF_EQ_COMPONENTS", ["CREDITS_TRESORERIE", "BANQUES_SOLDES_CREDITEURS", "TRESORERIE_PASSIF"], status_value="warning", severity="medium", expected=credits + banques, actual=tp, diff=abs(tp - (credits + banques)), message=msg)
        else:
            status["TRESORERIE_PASSIF"] = "consistent"
            register("TRESO_PASSIF_EQ_COMPONENTS", ["CREDITS_TRESORERIE", "BANQUES_SOLDES_CREDITEURS", "TRESORERIE_PASSIF"], status_value="passed", severity="info", expected=credits + banques, actual=tp, diff=abs(tp - (credits + banques)), message="Trésorerie-passif cohérente.")
    else:
        register("TRESO_PASSIF_EQ_COMPONENTS", ["CREDITS_TRESORERIE", "BANQUES_SOLDES_CREDITEURS", "TRESORERIE_PASSIF"], status_value="insufficient_data", severity="info", expected=None, actual=tp, diff=None, message="Composantes incomplètes de trésorerie-passif.")

    # CAF directe vs recalculée
    caf = resolved.get("CAF")
    if caf and caf.selected_value is not None and caf.calculated_value is not None:
        if abs(caf.selected_value - caf.calculated_value) > Decimal("1"):
            msg = (
                f"CAF directe ({caf.selected_value}) diverge du recalcul "
                f"({caf.calculated_value}) — valeur directe conservée"
            )
            warnings.append(msg)
            status["CAF"] = caf.validation_status or "divergent"
            register("CAF_DIRECTE_EQ_CAF_CALCULEE", ["CAF"], status_value="warning", severity="medium", expected=caf.calculated_value, actual=caf.selected_value, diff=abs(caf.selected_value - caf.calculated_value), message=msg)
        else:
            status["CAF"] = "consistent"
            register("CAF_DIRECTE_EQ_CAF_CALCULEE", ["CAF"], status_value="passed", severity="info", expected=caf.calculated_value, actual=caf.selected_value, diff=abs(caf.selected_value - caf.calculated_value), message="CAF directe cohérente avec le recalcul.")
    elif caf and caf.calculated_value is not None:
        register("CAF_DIRECTE_EQ_CAF_CALCULEE", ["CAF"], status_value="insufficient_data", severity="info", expected=caf.calculated_value, actual=None, diff=None, message="CAF uniquement estimée, non admissible au scoring.")

    # Propager sur les FieldResolution
    for code, st in status.items():
        if code in resolved:
            resolved[code].validation_status = st

    return warnings, status, checks
