"""Contrôles de cohérence comptable indépendants de l'extraction."""
from __future__ import annotations

from decimal import Decimal

from app.schemas.observations import FieldResolution

DEFAULT_TOLERANCE = Decimal("0.01")


def run_accounting_checks(
    resolved: dict[str, FieldResolution],
    *,
    tolerance: Decimal = DEFAULT_TOLERANCE,
) -> tuple[list[str], dict[str, str]]:
    """Retourne (warnings, validation_par_code)."""
    warnings: list[str] = []
    status: dict[str, str] = {}

    def v(code: str) -> Decimal | None:
        item = resolved.get(code)
        return item.selected_value if item else None

    total_actif = v("TOTAL_BILAN")
    # Équilibre : si on a aussi un total passif via fonds+passif+trésorerie
    fp = v("FONDS_PROPRES")
    pc = v("PASSIF_CIRCULANT")
    tp = v("TRESORERIE_PASSIF")
    if total_actif is not None and fp is not None and pc is not None and tp is not None:
        total_passif = fp + pc + tp
        # Approximation : passif peut inclure d'autres postes ; tolérance élargie
        if abs(total_actif - total_passif) > Decimal("1000"):
            # Ne pas invalider si structure incomplète — warning seulement
            warnings.append(
                f"Équilibre bilan approximatif : actif={total_actif} vs "
                f"FP+PC+TP={total_passif} (écart={abs(total_actif - total_passif)})"
            )
            status["TOTAL_BILAN"] = "warning"
        else:
            status["TOTAL_BILAN"] = "consistent"

    ta = v("TRESORERIE_ACTIF")
    if ta is not None and tp is not None:
        tn = v("TRESORERIE_NETTE")
        expected = ta - tp
        if tn is not None and abs(tn - expected) > tolerance:
            warnings.append(
                f"Trésorerie nette incohérente : {tn} ≠ {ta} - {tp}"
            )
            status["TRESORERIE_NETTE"] = "invalidated"
        elif tn is not None:
            status["TRESORERIE_NETTE"] = "consistent"

    mlt = v("DETTES_BANCAIRES_MLT")
    ct = v("DETTES_BANCAIRES_CT")
    df = v("DETTES_FINANCIERES")
    if mlt is not None and ct is not None and df is not None:
        if abs(df - (mlt + ct)) > tolerance:
            warnings.append("Dettes financières ≠ MLT + CT")
            status["DETTES_FINANCIERES"] = "invalidated"
        else:
            status["DETTES_FINANCIERES"] = "consistent"

    # Trésorerie passif ≈ crédits + banques
    credits = v("CREDITS_TRESORERIE")
    banques = v("BANQUES_SOLDES_CREDITEURS")
    if tp is not None and credits is not None and banques is not None:
        if abs(tp - (credits + banques)) > tolerance:
            warnings.append(
                f"Trésorerie-passif ({tp}) ≠ crédits+banques ({credits + banques})"
            )
            status["TRESORERIE_PASSIF"] = "warning"
        else:
            status["TRESORERIE_PASSIF"] = "consistent"

    # CAF directe vs recalculée
    caf = resolved.get("CAF")
    if caf and caf.selected_value is not None and caf.calculated_value is not None:
        if abs(caf.selected_value - caf.calculated_value) > Decimal("1"):
            warnings.append(
                f"CAF directe ({caf.selected_value}) diverge du recalcul "
                f"({caf.calculated_value}) — valeur directe conservée"
            )
            status["CAF"] = caf.validation_status or "divergent"
        else:
            status["CAF"] = "consistent"

    # Propager sur les FieldResolution
    for code, st in status.items():
        if code in resolved:
            resolved[code].validation_status = st

    return warnings, status
