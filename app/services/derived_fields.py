"""Calcul déterministe des champs dérivés (Decimal uniquement)."""
from __future__ import annotations

from decimal import Decimal

from app.schemas.observations import FieldResolution
from app.services.amount_parser import sum_decimal


def apply_derived_fields(
    resolved: dict[str, FieldResolution],
) -> dict[str, FieldResolution]:
    """Calcule dettes_financieres, tresorerie_nette, FDR, TYPE_RESULTAT, CAF fallback."""
    out = dict(resolved)

    mlt = _val(out, "DETTES_BANCAIRES_MLT")
    ct = _val(out, "DETTES_BANCAIRES_CT")
    if mlt is not None or ct is not None:
        dettes = sum_decimal(mlt or Decimal("0"), ct or Decimal("0"))
        # Si une seule composante absente, on garde quand même la somme partielle
        if mlt is None and ct is not None:
            dettes = ct
        elif ct is None and mlt is not None:
            dettes = mlt
        out["DETTES_FINANCIERES"] = FieldResolution(
            field="DETTES_FINANCIERES",
            selected_value=dettes,
            calculated_value=dettes,
            detection_status="derived",
            confidence=0.92,
            selection_reason="dettes_bancaires_mlt + dettes_bancaires_ct",
            validation_status="consistent",
        )

    actif = _val(out, "TRESORERIE_ACTIF")
    passif = _val(out, "TRESORERIE_PASSIF")
    if actif is not None and passif is not None:
        out["TRESORERIE_NETTE"] = FieldResolution(
            field="TRESORERIE_NETTE",
            selected_value=actif - passif,
            calculated_value=actif - passif,
            detection_status="derived",
            confidence=0.95,
            selection_reason="tresorerie_actif - tresorerie_passif",
            validation_status="consistent",
        )

    # FDR = capitaux permanents (fonds propres + dettes MLT) - actifs immobilisés
    # Définition métier configurable documentée — utilisée si FDR absent.
    fp = _val(out, "FONDS_PROPRES")
    immo = _val(out, "ACTIFS_IMMOBILISES")
    existing_fdr = out.get("FDR")
    if (
        (existing_fdr is None or existing_fdr.selected_value is None)
        and fp is not None
        and immo is not None
    ):
        capitaux_permanents = sum_decimal(fp, mlt or Decimal("0"))
        if capitaux_permanents is not None:
            fdr = capitaux_permanents - immo
            out["FDR"] = FieldResolution(
                field="FDR",
                selected_value=fdr,
                calculated_value=fdr,
                detection_status="derived",
                confidence=0.85,
                selection_reason=(
                    "FDR = (fonds_propres + dettes_bancaires_mlt) - actifs_immobilises"
                ),
                validation_status="consistent",
            )

    rn = _val(out, "RESULTAT_NET")
    if rn is not None:
        note = "Bénéficiaire" if rn > 0 else ("Déficitaire" if rn < 0 else "Nul")
        out["TYPE_RESULTAT"] = FieldResolution(
            field="TYPE_RESULTAT",
            selected_value=None,
            detection_status="derived",
            confidence=0.95,
            selection_reason=note,
            source="Dérivé",
        )

    # CAF : si absente, tenter RN + amortissements (approximation ESG simple)
    caf = out.get("CAF")
    amort = _val(out, "AMORTISSEMENTS") or _val(out, "DOTATIONS_EXPLOITATION")
    if (caf is None or caf.selected_value is None) and rn is not None and amort is not None:
        calculated_caf = rn + amort
        out["CAF"] = FieldResolution(
            field="CAF",
            selected_value=calculated_caf,
            calculated_value=calculated_caf,
            detection_status="derived",
            confidence=0.75,
            selection_reason="CAF ≈ resultat_net + dotations_exploitation (fallback)",
            validation_status="estimated",
        )
    elif caf and caf.selected_value is not None and rn is not None and amort is not None:
        calculated_caf = rn + amort
        caf.calculated_value = calculated_caf
        if abs(caf.selected_value - calculated_caf) <= Decimal("1"):
            caf.validation_status = "consistent"
        else:
            # L'écart est fréquent (ESG plus riche) — on garde la valeur directe
            caf.validation_status = "divergent"
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
        out["AMORTISSEMENTS"] = FieldResolution(
            field="AMORTISSEMENTS",
            selected_value=dot.selected_value,
            detection_status="detected",
            confidence=dot.confidence,
            selection_reason="Dotation d'exploitation (composante CAF)",
            raw_label=dot.raw_label,
            column=dot.column,
            page=dot.page,
            source=dot.source,
            candidates=dot.candidates,
        )

    return out


def _val(resolved: dict[str, FieldResolution], code: str) -> Decimal | None:
    item = resolved.get(code)
    if not item:
        return None
    return item.selected_value
