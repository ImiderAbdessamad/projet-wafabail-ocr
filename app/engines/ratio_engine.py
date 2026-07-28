from decimal import Decimal
from typing import Dict, TypedDict


class RatioResult(TypedDict):
    value: float | None
    status: str  # "Conforme" | "À surveiller" | "Non conforme" | "Non calculable"
    reason: str | None
    numerator: float | None
    denominator: float | None
    validity_status: str


def evaluate_status(
    value: float,
    threshold_conforme: float,
    threshold_surveiller: float,
    lower_is_better: bool = False,
) -> str:
    if lower_is_better:
        if value <= threshold_conforme:
            return "Conforme"
        elif value <= threshold_surveiller:
            return "À surveiller"
        return "Non conforme"
    else:
        if value >= threshold_conforme:
            return "Conforme"
        elif value >= threshold_surveiller:
            return "À surveiller"
        return "Non conforme"


def safe_div(num: Decimal | float | None, den: Decimal | float | None) -> float | None:
    if num is None or den is None:
        return None
    den_f = float(den)
    if den_f == 0.0:
        return None
    return float(num) / den_f


def _result(
    value: float | None,
    status: str,
    *,
    reason: str | None = None,
    numerator: Decimal | float | None = None,
    denominator: Decimal | float | None = None,
    validity_status: str = "valid",
) -> RatioResult:
    return {
        "value": value,
        "status": status,
        "reason": reason,
        "numerator": None if numerator is None else float(numerator),
        "denominator": None if denominator is None else float(denominator),
        "validity_status": validity_status,
    }


# --- 1.1 Structure financière et solvabilité ---

def autonomie_financiere(fp: Decimal | None, total_bilan: Decimal | None) -> RatioResult:
    if fp is None or total_bilan is None:
        return _result(None, "Non calculable", reason="Donnée manquante.", numerator=fp, denominator=total_bilan, validity_status="missing")
    if float(total_bilan) <= 0:
        return _result(None, "Non calculable", reason="Total bilan nul ou négatif.", numerator=fp, denominator=total_bilan, validity_status="invalid_denominator")
    val = safe_div(fp, total_bilan)
    if val is None:
        return _result(None, "Non calculable", reason="Division impossible.", numerator=fp, denominator=total_bilan, validity_status="invalid_denominator")
    status = "Non conforme" if float(fp) < 0 else evaluate_status(val, 0.20, 0.10, False)
    reason = "Fonds propres négatifs." if float(fp) < 0 else None
    return _result(val, status, reason=reason, numerator=fp, denominator=total_bilan)


def ratio_endettement(endettement: Decimal | None, fp: Decimal | None) -> RatioResult:
    """Endettement / FP. Le repère 1,5x est indicatif (référence sectorielle,
    pas un couperet — cf. note de lecture du rapport type) : Conforme ≤ 2,0x,
    À surveiller ≤ 3,0x."""
    if endettement is None or fp is None:
        return _result(None, "Non calculable", reason="Donnée manquante.", numerator=endettement, denominator=fp, validity_status="missing")
    if float(fp) < 0:
        return _result(None, "Non conforme", reason="Fonds propres négatifs.", numerator=endettement, denominator=fp, validity_status="economically_invalid")
    if float(fp) == 0:
        return _result(None, "Non conforme", reason="Fonds propres nuls.", numerator=endettement, denominator=fp, validity_status="invalid_denominator")
    val = safe_div(endettement, fp)
    if val is None:
        return _result(None, "Non calculable", reason="Division impossible.", numerator=endettement, denominator=fp, validity_status="invalid_denominator")
    return _result(val, evaluate_status(val, 2.0, 3.0, True), numerator=endettement, denominator=fp)


def capacite_remboursement(dettes_fin: Decimal | None, caf: Decimal | None) -> RatioResult:
    """Dettes de financement / CAF. Le repère 3 ans est indicatif :
    Conforme ≤ 5 ans, À surveiller ≤ 7 ans (au-delà, capacité insuffisante)."""
    if dettes_fin is None or caf is None:
        return _result(None, "Non calculable", reason="Donnée manquante.", numerator=dettes_fin, denominator=caf, validity_status="missing")
    if float(caf) < 0:
        return _result(None, "Non conforme", reason="CAF négative.", numerator=dettes_fin, denominator=caf, validity_status="economically_invalid")
    if float(caf) == 0:
        return _result(None, "Non conforme", reason="CAF nulle.", numerator=dettes_fin, denominator=caf, validity_status="invalid_denominator")
    val = safe_div(dettes_fin, caf)
    if val is None:
        return _result(None, "Non calculable", reason="Division impossible.", numerator=dettes_fin, denominator=caf, validity_status="invalid_denominator")
    return _result(val, evaluate_status(val, 5.0, 7.0, True), numerator=dettes_fin, denominator=caf)


# --- 1.2 Rentabilité et autofinancement ---

def capacite_autofinancement_ca(caf: Decimal | None, ca: Decimal | None) -> RatioResult:
    if caf is None or ca is None:
        return _result(None, "Non calculable", reason="Donnée manquante.", numerator=caf, denominator=ca, validity_status="missing")
    if float(ca) <= 0:
        return _result(None, "Non calculable", reason="Chiffre d'affaires nul ou négatif.", numerator=caf, denominator=ca, validity_status="invalid_denominator")
    if float(caf) < 0:
        return _result(safe_div(caf, ca), "Non conforme", reason="CAF négative.", numerator=caf, denominator=ca, validity_status="economically_invalid")
    val = safe_div(caf, ca)
    if val is None:
        return _result(None, "Non calculable", reason="Division impossible.", numerator=caf, denominator=ca, validity_status="invalid_denominator")
    return _result(val, evaluate_status(val, 0.05, 0.02, False), numerator=caf, denominator=ca)


def rentabilite_commerciale(rn: Decimal | None, ca: Decimal | None) -> RatioResult:
    if rn is None or ca is None:
        return _result(None, "Non calculable", reason="Donnée manquante.", numerator=rn, denominator=ca, validity_status="missing")
    if float(ca) <= 0:
        return _result(None, "Non calculable", reason="Chiffre d'affaires nul ou négatif.", numerator=rn, denominator=ca, validity_status="invalid_denominator")
    val = safe_div(rn, ca)
    if val is None:
        return _result(None, "Non calculable", reason="Division impossible.", numerator=rn, denominator=ca, validity_status="invalid_denominator")
    return _result(val, evaluate_status(val, 0.05, 0.02, False), numerator=rn, denominator=ca)


def rentabilite_financiere(rn: Decimal | None, fp: Decimal | None) -> RatioResult:
    if rn is None or fp is None:
        return _result(None, "Non calculable", reason="Donnée manquante.", numerator=rn, denominator=fp, validity_status="missing")
    if float(fp) <= 0:
        return _result(None, "Non calculable", reason="Fonds propres nuls ou négatifs.", numerator=rn, denominator=fp, validity_status="economically_invalid")
    val = safe_div(rn, fp)
    if val is None:
        return _result(None, "Non calculable", reason="Division impossible.", numerator=rn, denominator=fp, validity_status="invalid_denominator")
    return _result(val, evaluate_status(val, 0.10, 0.05, False), numerator=rn, denominator=fp)


def rentabilite_economique(
    rn: Decimal | None,
    fp: Decimal | None,
    dettes_fin: Decimal | None,
) -> RatioResult:
    """Rentabilité économique = RN / (Fonds Propres + Dettes financières).

    Both fp and dettes_fin must be known; treating either missing as 0 would
    silently compute a meaningless ratio.
    """
    if rn is None or fp is None or dettes_fin is None:
        return _result(None, "Non calculable", reason="Donnée manquante.", numerator=rn, denominator=None, validity_status="missing")
    den = float(fp) + float(dettes_fin)
    if den <= 0.0:
        return _result(None, "Non calculable", reason="Base économique nulle ou négative.", numerator=rn, denominator=den, validity_status="economically_invalid")
    val = float(rn) / den
    return _result(val, evaluate_status(val, 0.05, 0.02, False), numerator=rn, denominator=den)


# --- 1.3 Liquidité et cycle d'exploitation ---

def fdr_ca(fdr: Decimal | None, ca: Decimal | None) -> RatioResult:
    """FDR / CA.

    FDR > 0 → Conforme (ressources stables couvrent le BFR)
    FDR = 0 → À surveiller
    FDR < 0 → Non conforme (actif immobilisé partiellement financé par passif circulant)
    """
    val = safe_div(fdr, ca)
    if val is None:
        return _result(None, "Non calculable", reason="Donnée manquante ou CA nul.", numerator=fdr, denominator=ca, validity_status="invalid_denominator")
    if val > 0:
        return _result(val, "Conforme", numerator=fdr, denominator=ca)
    if val == 0:
        return _result(val, "À surveiller", numerator=fdr, denominator=ca)
    return _result(val, "Non conforme", numerator=fdr, denominator=ca)


def tresorerie_jours_ca(treso_nette: Decimal | None, ca: Decimal | None) -> RatioResult:
    """Trésorerie nette en jours de CA.

    TN > 0 j → Conforme
    TN = 0 j → À surveiller
    TN < 0 j → Non conforme (découvert structurel)
    """
    val = safe_div(treso_nette, ca)
    if val is None:
        return _result(None, "Non calculable", reason="Donnée manquante ou CA nul.", numerator=treso_nette, denominator=ca, validity_status="invalid_denominator")
    val_j = val * 360
    if val_j > 0:
        return _result(val_j, "Conforme", numerator=treso_nette, denominator=ca)
    if val_j == 0:
        return _result(val_j, "À surveiller", numerator=treso_nette, denominator=ca)
    return _result(val_j, "Non conforme", numerator=treso_nette, denominator=ca)


def delais_clients(clients: Decimal | None, ca: Decimal | None) -> RatioResult:
    if clients is None or ca is None:
        return _result(None, "Non calculable", reason="Donnée manquante.", numerator=clients, denominator=ca, validity_status="missing")
    if float(ca) <= 0:
        return _result(None, "Non calculable", reason="Chiffre d'affaires nul ou négatif.", numerator=clients, denominator=ca, validity_status="invalid_denominator")
    val = safe_div(clients, ca)
    if val is None:
        return _result(None, "Non calculable", reason="Division impossible.", numerator=clients, denominator=ca, validity_status="invalid_denominator")
    val_j = val * 360
    return _result(val_j, evaluate_status(val_j, 60, 90, True), numerator=clients, denominator=ca)


def delais_fournisseurs(
    fournisseurs: Decimal | None,
    achats: Decimal | None,
    clients_j: float | None,
) -> RatioResult:
    if fournisseurs is None or achats is None:
        return _result(None, "Non calculable", reason="Donnée manquante.", numerator=fournisseurs, denominator=achats, validity_status="missing")
    if float(achats) <= 0:
        return _result(None, "Non calculable", reason="Achats nuls ou négatifs.", numerator=fournisseurs, denominator=achats, validity_status="invalid_denominator")
    val = safe_div(fournisseurs, achats)
    if val is None:
        return _result(None, "Non calculable", reason="Division impossible.", numerator=fournisseurs, denominator=achats, validity_status="invalid_denominator")
    val_j = val * 360
    if clients_j is None:
        return _result(val_j, "À surveiller", reason="Délais clients indisponibles.", numerator=fournisseurs, denominator=achats, validity_status="partial_context")
    # Repère indicatif : un léger décalage (< 20 %) sous les délais clients
    # reste Conforme (cf. rapport type : 64 j vs 78 j classé Conforme).
    if val_j >= clients_j * 0.8:
        return _result(val_j, "Conforme", numerator=fournisseurs, denominator=achats)
    if val_j >= clients_j * 0.5:
        return _result(val_j, "À surveiller", numerator=fournisseurs, denominator=achats)
    return _result(val_j, "Non conforme", numerator=fournisseurs, denominator=achats)


# --- Ratios complémentaires (docx : §1.4 activité, §1.5 endettement global) ---

def croissance_ca(ca_n: Decimal | None, ca_n1: Decimal | None) -> RatioResult:
    """Croissance du CA N vs N-1. Conforme ≥ 5 %, à surveiller ≥ 0 %."""
    if ca_n is None or ca_n1 is None or float(ca_n1) == 0.0:
        return _result(None, "Non calculable", reason="CA N-1 absent, nul ou invalide.", numerator=ca_n, denominator=ca_n1, validity_status="invalid_denominator")
    val = (float(ca_n) - float(ca_n1)) / abs(float(ca_n1))
    return _result(val, evaluate_status(val, 0.05, 0.0, False), numerator=float(ca_n) - float(ca_n1), denominator=ca_n1)


def endettement_global_apres_operation(
    encours_leasing: Decimal | None,
    cmt: Decimal | None,
    nouveau_financement: Decimal | None,
    fp: Decimal | None,
) -> RatioResult:
    """Endettement global après opération / Fonds propres (docx §1.5).

    Intègre l'encours leasing, le crédit moyen terme et le nouveau
    financement demandé. Conforme ≤ 2,0x, à surveiller ≤ 3,0x.
    """
    if fp is None:
        return _result(None, "Non calculable", reason="Fonds propres absents.", numerator=None, denominator=fp, validity_status="missing")
    if float(fp) < 0.0:
        return _result(None, "Non conforme", reason="Fonds propres négatifs.", numerator=None, denominator=fp, validity_status="economically_invalid")
    if float(fp) == 0.0:
        return _result(None, "Non conforme", reason="Fonds propres nuls.", numerator=None, denominator=fp, validity_status="invalid_denominator")
    parts = [encours_leasing, cmt, nouveau_financement]
    if all(p is None for p in parts):
        return _result(None, "Non calculable", reason="Aucune composante d'endettement après opération.", numerator=None, denominator=fp, validity_status="missing")
    total = sum(float(p) for p in parts if p is not None)
    val = total / float(fp)
    return _result(val, evaluate_status(val, 2.0, 3.0, True), numerator=total, denominator=fp)


# Métadonnées d'affichage : formule et seuil de référence par ratio (docx tableaux 1.1-1.3)
RATIO_METADATA: Dict[str, dict] = {
    "autonomie_financiere": {"label": "Autonomie financière", "formula": "FP ÷ Total Bilan", "threshold": "≥ 20 %", "unit": "%"},
    "ratio_endettement": {"label": "Ratio d'endettement", "formula": "Endettement ÷ FP", "threshold": "≤ 1,5x (indic. — conforme jusqu'à 2,0x)", "unit": "x"},
    "capacite_remboursement": {"label": "Capacité de remboursement", "formula": "Dettes de fin. ÷ CAF", "threshold": "≤ 3 ans (indic. — conforme jusqu'à 5 ans)", "unit": "ans"},
    "caf_sur_ca": {"label": "Capacité d'autofinancement", "formula": "CAF ÷ CA", "threshold": "≥ 5 %", "unit": "%"},
    "rentabilite_commerciale": {"label": "Rentabilité commerciale", "formula": "RN ÷ CA", "threshold": "≥ 5 %", "unit": "%"},
    "rentabilite_financiere": {"label": "Rentabilité financière", "formula": "RN ÷ FP", "threshold": "> 10 %", "unit": "%"},
    "rentabilite_economique": {"label": "Rentabilité économique", "formula": "RN ÷ (FP + Dettes fin.)", "threshold": "> coût moyen dette (5 %)", "unit": "%"},
    "fdr_sur_ca": {"label": "FDR / CA", "formula": "Fonds de roulement ÷ CA", "threshold": "> 0 %", "unit": "%"},
    "tresorerie_jours_ca": {"label": "Trésorerie en jours de CA", "formula": "Trésorerie nette ÷ CA × 360", "threshold": "> 0 jour", "unit": "j"},
    "delais_clients": {"label": "Délais clients", "formula": "(Clients ÷ CA) × 360", "threshold": "≤ 60 j (à surveiller ≤ 90 j)", "unit": "j"},
    "delais_fournisseurs": {"label": "Délais fournisseurs", "formula": "(Fournisseurs ÷ Achats) × 360", "threshold": "≥ délais clients", "unit": "j"},
    "croissance_ca": {"label": "Croissance du CA", "formula": "(CA N − CA N-1) ÷ CA N-1", "threshold": "≥ 5 %", "unit": "%"},
    "endettement_global_apres_operation": {"label": "Endettement global après opération", "formula": "(Leasing + CMT + Nouveau financement) ÷ FP", "threshold": "≤ 2,0x (à surveiller ≤ 3,0x)", "unit": "x"},
}


def compute_yearly_variations(years: Dict[int, dict]) -> Dict[str, Dict[str, float | None]]:
    """Variations N/N-1 par poste pour une série pluriannuelle (docx §1.4)."""
    variations: Dict[str, Dict[str, float | None]] = {}
    sorted_years = sorted(years)
    postes = {k for data in years.values() for k in data if data[k] is not None}
    for poste in postes:
        per_year: Dict[str, float | None] = {}
        for prev, curr in zip(sorted_years, sorted_years[1:]):
            v_prev, v_curr = years[prev].get(poste), years[curr].get(poste)
            if v_prev in (None, 0) or v_curr is None:
                per_year[f"{curr}/{prev}"] = None
            else:
                per_year[f"{curr}/{prev}"] = (float(v_curr) - float(v_prev)) / abs(float(v_prev))
        if per_year:
            variations[poste] = per_year
    return variations


def calculate_all_ratios(data: dict) -> Dict[str, RatioResult]:
    clients_res = delais_clients(data.get("clients"), data.get("chiffre_affaires"))
    clients_j = clients_res["value"]

    return {
        "autonomie_financiere": autonomie_financiere(data.get("fonds_propres"), data.get("total_bilan")),
        "ratio_endettement": ratio_endettement(data.get("dettes_financieres"), data.get("fonds_propres")),
        "capacite_remboursement": capacite_remboursement(data.get("dettes_financieres"), data.get("caf")),
        "caf_sur_ca": capacite_autofinancement_ca(data.get("caf"), data.get("chiffre_affaires")),
        "rentabilite_commerciale": rentabilite_commerciale(data.get("resultat_net"), data.get("chiffre_affaires")),
        "rentabilite_financiere": rentabilite_financiere(data.get("resultat_net"), data.get("fonds_propres")),
        "rentabilite_economique": rentabilite_economique(
            data.get("resultat_net"), data.get("fonds_propres"), data.get("dettes_financieres")
        ),
        "fdr_sur_ca": fdr_ca(data.get("fdr"), data.get("chiffre_affaires")),
        "tresorerie_jours_ca": tresorerie_jours_ca(data.get("tresorerie_nette"), data.get("chiffre_affaires")),
        "delais_clients": clients_res,
        "delais_fournisseurs": delais_fournisseurs(
            data.get("fournisseurs"), data.get("achats"), clients_j
        ),
    }


def calculate_extended_ratios(data: dict) -> Dict[str, RatioResult]:
    """11 ratios de base + croissance CA + endettement global après opération.

    Champs optionnels attendus en plus : ca_n1, encours_leasing, cmt,
    nouveau_financement.
    """
    ratios = calculate_all_ratios(data)
    ratios["croissance_ca"] = croissance_ca(
        data.get("chiffre_affaires"), data.get("ca_n1")
    )
    ratios["endettement_global_apres_operation"] = endettement_global_apres_operation(
        data.get("encours_leasing"),
        data.get("cmt"),
        data.get("nouveau_financement"),
        data.get("fonds_propres"),
    )
    return ratios
