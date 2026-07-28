"""Calcul déterministe des champs dérivés (Decimal uniquement)."""
from __future__ import annotations

from decimal import Decimal

from app.schemas.observations import FieldResolution
from app.services.amount_parser import sum_decimal


def _make_resolution(
    field: str,
    *,
    selected_value: Decimal | None = None,
    calculated_value: Decimal | None = None,
    detection_status: str,
    confidence: float,
    selection_reason: str,
    validation_status: str | None = None,
    calculation_status: str | None = None,
    missing_components: list[str] | None = None,
    eligible_for_scoring: bool = True,
) -> FieldResolution:
    return FieldResolution(
        field=field,
        selected_value=selected_value,
        calculated_value=calculated_value,
        detection_status=detection_status,
        confidence=confidence,
        selection_reason=selection_reason,
        validation_status=validation_status,
        calculation_status=calculation_status,
        missing_components=missing_components or [],
        eligible_for_scoring=eligible_for_scoring,
    )


def apply_derived_fields(
    resolved: dict[str, FieldResolution],
) -> dict[str, FieldResolution]:
    """Calcule dettes_financieres, tresorerie_nette, FDR, TYPE_RESULTAT et CAF.

    Une somme partielle n'alimente jamais selected_value ni le scoring.
    """
    out = dict(resolved)

    mlt = _val(out, "DETTES_BANCAIRES_MLT")
    ct = _val(out, "DETTES_BANCAIRES_CT")
    if mlt is not None and ct is not None:
        dettes = mlt + ct
        out["DETTES_FINANCIERES"] = _make_resolution(
            "DETTES_FINANCIERES",
            selected_value=dettes,
            calculated_value=dettes,
            detection_status="derived",
            confidence=0.9,
            selection_reason="dettes_bancaires_mlt + dettes_bancaires_ct",
            validation_status="consistent",
            calculation_status="complete",
        )
    elif mlt is not None or ct is not None:
        partial = sum_decimal(mlt, ct)
        missing = []
        if mlt is None:
            missing.append("DETTES_BANCAIRES_MLT")
        if ct is None:
            missing.append("DETTES_BANCAIRES_CT")
        out["DETTES_FINANCIERES"] = _make_resolution(
            "DETTES_FINANCIERES",
            selected_value=None,
            calculated_value=partial,
            detection_status="incomplete",
            confidence=0.4,
            selection_reason="Agrégat incomplet : dettes_bancaires_mlt + dettes_bancaires_ct",
            validation_status="not_validated",
            calculation_status="partial",
            missing_components=missing,
            eligible_for_scoring=False,
        )

    actif = _val(out, "TRESORERIE_ACTIF")
    passif = _val(out, "TRESORERIE_PASSIF")
    if actif is not None and passif is not None:
        out["TRESORERIE_NETTE"] = _make_resolution(
            "TRESORERIE_NETTE",
            selected_value=actif - passif,
            calculated_value=actif - passif,
            detection_status="derived",
            confidence=0.92,
            selection_reason="tresorerie_actif - tresorerie_passif",
            validation_status="consistent",
            calculation_status="complete",
        )

    # FDR = capitaux permanents (fonds propres + dettes MLT) - actifs immobilisés
    # Définition métier configurable documentée — utilisée si FDR absent.
    fp = _val(out, "FONDS_PROPRES")
    immo = _val(out, "ACTIFS_IMMOBILISES")
    existing_fdr = out.get("FDR")
    if (
        (existing_fdr is None or existing_fdr.selected_value is None)
        and fp is not None
        and mlt is not None
        and immo is not None
    ):
        capitaux_permanents = sum_decimal(fp, mlt)
        if capitaux_permanents is not None:
            fdr = capitaux_permanents - immo
            out["FDR"] = _make_resolution(
                "FDR",
                selected_value=fdr,
                calculated_value=fdr,
                detection_status="derived",
                confidence=0.85,
                selection_reason=(
                    "FDR = (fonds_propres + dettes_bancaires_mlt) - actifs_immobilises"
                ),
                validation_status="consistent",
                calculation_status="complete",
            )
    elif existing_fdr is None or existing_fdr.selected_value is None:
        missing = []
        if fp is None:
            missing.append("FONDS_PROPRES")
        if mlt is None:
            missing.append("DETTES_BANCAIRES_MLT")
        if immo is None:
            missing.append("ACTIFS_IMMOBILISES")
        out["FDR"] = _make_resolution(
            "FDR",
            selected_value=None,
            calculated_value=None,
            detection_status="incomplete",
            confidence=0.3,
            selection_reason="FDR non calculable sans toutes les composantes métier.",
            validation_status="not_validated",
            calculation_status="unavailable",
            missing_components=missing,
            eligible_for_scoring=False,
        )

    rn = _val(out, "RESULTAT_NET")
    if rn is not None:
        note = "Bénéficiaire" if rn > 0 else ("Déficitaire" if rn < 0 else "Nul")
        out["TYPE_RESULTAT"] = _make_resolution(
            "TYPE_RESULTAT",
            selected_value=None,
            detection_status="derived",
            confidence=0.95,
            selection_reason=note,
            calculation_status="complete",
        )
        out["TYPE_RESULTAT"].source = "Dérivé"

    # CAF : ne jamais utiliser RN + amortissements comme valeur finale scoreable.
    caf = out.get("CAF")
    amort = _val(out, "AMORTISSEMENTS") or _val(out, "DOTATIONS_EXPLOITATION")
    if (caf is None or caf.selected_value is None) and rn is not None and amort is not None:
        calculated_caf = rn + amort
        out["CAF"] = _make_resolution(
            "CAF",
            selected_value=None,
            calculated_value=calculated_caf,
            detection_status="estimated",
            confidence=0.45,
            selection_reason="Estimation informative : resultat_net + dotations_exploitation",
            validation_status="not_validated",
            calculation_status="incomplete_estimate",
            missing_components=["CAF_DIRECTE_ESG"],
            eligible_for_scoring=False,
        )
    elif caf and caf.selected_value is not None and rn is not None and amort is not None:
        calculated_caf = rn + amort
        caf.calculated_value = calculated_caf
        caf.calculation_status = "complete"
        if abs(caf.selected_value - calculated_caf) <= Decimal("1"):
            caf.validation_status = "consistent"
        else:
            # L'écart est fréquent (ESG plus riche) — on garde la valeur directe
            caf.validation_status = "divergent"
            caf.eligible_for_scoring = False
            caf.selection_reason = (
                (caf.selection_reason or "")
                + f" | recalcul RN+amort={calculated_caf}"
            )
        out["CAF"] = caf

    # AMORTISSEMENTS : préférer DOTATIONS_EXPLOITATION si AMORT ambigu
    am = out.get("AMORTISSEMENTS")
    dot = out.get("DOTATIONS_EXPLOITATION")
    if (
        (am is None or am.selected_value is None or am.detection_status == "ambiguous")
        and dot
        and dot.selected_value is not None
    ):
        replacement = dot.model_copy()
        replacement.field = "AMORTISSEMENTS"
        replacement.selection_reason = "Dotation d'exploitation (composante CAF)"
        replacement.detection_status = "detected"
        out["AMORTISSEMENTS"] = replacement

    return out


def _val(resolved: dict[str, FieldResolution], code: str) -> Decimal | None:
    item = resolved.get(code)
    if not item:
        return None
    return item.selected_value
