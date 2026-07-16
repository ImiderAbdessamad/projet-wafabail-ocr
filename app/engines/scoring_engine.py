"""Moteur de scoring crédit-bail à 3 axes pondérés (75 / 15 / 10).

Calibré sur la grille du document d'analyse :
- Axe 1 (financier) : notation par pénalités, reproduit 85/100 pour
  9 ratios conformes + 2 à surveiller.
- Axe 2 (comportemental) : rubrique des 8 métriques bancaires, reproduit
  75/100 pour le cas de référence (41 j de débit, écart flux/CA -4,2 %).
- Axe 3 (sectoriel) : proportion d'indicateurs au-dessus de la médiane
  du panel sectoriel, reproduit 80/100 pour 4 indicateurs sur 5.
"""
from typing import Dict, TypedDict


class DecisionResult(TypedDict):
    score: float
    classe: str  # A+, A/B+, B/B-, C, D/F
    decision: str  # Excellent, Bon, Moyen, Sensible, Risqué
    recommandation: str
    blocking_status: str | None  # "NO_GO" | "MANUAL_REVIEW" | None


# --- Critères bloquants ---

def check_blocking_criteria(bam_cotation: int | None, unresolved_incidents: int = 0) -> str | None:
    """Cotation BAM 7/8/9 → refus automatique ; incidents non résolus → revue manuelle."""
    if bam_cotation is not None and bam_cotation >= 7:
        return "NO_GO"
    if unresolved_incidents > 0:
        return "MANUAL_REVIEW"
    return None


# --- Axe 1 : score financier par pénalités ---

AXE1_PENALTY_SURVEILLER = 7.5
AXE1_PENALTY_NON_CONFORME = 20.0

# Les 11 ratios de la grille Axe 1 (tableaux 1.1 à 1.3 du document).
# croissance_ca et endettement_global_apres_operation sont informatifs.
AXE1_RATIO_KEYS = {
    "autonomie_financiere",
    "ratio_endettement",
    "capacite_remboursement",
    "caf_sur_ca",
    "rentabilite_commerciale",
    "rentabilite_financiere",
    "rentabilite_economique",
    "fdr_sur_ca",
    "tresorerie_jours_ca",
    "delais_clients",
    "delais_fournisseurs",
}


def score_axe1_from_ratios(ratios: Dict[str, dict]) -> dict:
    """Base 100, −7,5 pt par ratio « À surveiller », −20 pt par « Non conforme ».

    Seuls les 11 ratios de la grille Axe 1 sont notés. Reproduit l'exemple
    du document : 9 conformes + 2 à surveiller → 85/100. Les ratios non
    calculables sont exclus de la note mais listés pour signaler la
    qualité de la donnée.
    """
    scored = {k: r for k, r in ratios.items() if k in AXE1_RATIO_KEYS}
    surveiller = [k for k, r in scored.items() if r["status"] == "À surveiller"]
    non_conformes = [k for k, r in scored.items() if r["status"] == "Non conforme"]
    non_calculables = [k for k, r in scored.items() if r["status"] == "Non calculable"]
    ratios = scored
    score = 100.0
    score -= AXE1_PENALTY_SURVEILLER * len(surveiller)
    score -= AXE1_PENALTY_NON_CONFORME * len(non_conformes)
    return {
        "score": max(0.0, round(score, 2)),
        "ratios_conformes": [
            k for k, r in ratios.items() if r["status"] == "Conforme"
        ],
        "ratios_a_surveiller": surveiller,
        "ratios_non_conformes": non_conformes,
        "ratios_non_calculables": non_calculables,
    }


# --- Axe 2 : score comportemental (8 métriques bancaires) ---

def score_axe2_behavioral(
    incidents_paiement: int = 0,
    rejets_prelevement: int = 0,
    effets_impayes: int = 0,
    domiciliation_ca_pct: float | None = None,
    jours_debit: int | None = None,
    utilisation_decouvert_pct: float | None = None,
    ecart_flux_ca_pct: float | None = None,
    engagements_honores: bool | None = None,
) -> dict:
    """Rubrique comportementale calibrée sur le document (§2).

    Cas de référence : 0 incident, domiciliation 96 %, 41 j de débit,
    utilisation découvert 38 %, écart flux/CA −4,2 % → 75/100.
    """
    score = 100.0
    signals: list[str] = []

    if incidents_paiement > 0:
        score -= 40.0
        signals.append(f"{incidents_paiement} incident(s) de paiement déclaré(s)")
    if rejets_prelevement > 0:
        score -= 15.0
        signals.append(f"{rejets_prelevement} rejet(s) de prélèvement")
    if effets_impayes > 0:
        score -= 25.0
        signals.append(f"{effets_impayes} effet(s) impayé(s)")

    if domiciliation_ca_pct is not None and domiciliation_ca_pct < 80.0:
        score -= 10.0
        signals.append(f"Domiciliation du CA insuffisante ({domiciliation_ca_pct:.0f} % < 80 %)")

    if jours_debit is not None and jours_debit > 30:
        score -= 15.0
        signals.append(f"Recours au découvert : {jours_debit} j/an en position débitrice")

    if utilisation_decouvert_pct is not None and utilisation_decouvert_pct > 75.0:
        score -= 10.0
        signals.append(
            f"Utilisation élevée de l'autorisation de découvert ({utilisation_decouvert_pct:.0f} %)"
        )

    if ecart_flux_ca_pct is not None and abs(ecart_flux_ca_pct) > 3.0:
        score -= 10.0
        signals.append(
            f"Écart flux bancaires / CA déclaré de {ecart_flux_ca_pct:+.1f} % (à justifier)"
        )

    if engagements_honores is False:
        score -= 20.0
        signals.append("Retards constatés sur les engagements de leasing en cours")

    return {"score": max(0.0, round(score, 2)), "signaux": signals}


# --- Axe 3 : score sectoriel (comparaison à la médiane du panel) ---

# Médianes par défaut du panel Transport & Logistique (docx §3, n = 1 284)
DEFAULT_SECTOR_MEDIANS = {
    "rentabilite_commerciale": 0.042,   # RN / CA — plus haut = mieux
    "autonomie_financiere": 0.21,       # FP / Total bilan — plus haut = mieux
    "ratio_endettement": 2.3,           # Endettement / FP — plus bas = mieux
    "capacite_remboursement": 4.8,      # Dettes fin. / CAF — plus bas = mieux
    "croissance_ca": 0.06,              # Var. CA — plus haut = mieux
}

_LOWER_IS_BETTER = {"ratio_endettement", "capacite_remboursement"}


def score_axe3_sectoriel(
    ratios: Dict[str, dict],
    sector_medians: Dict[str, float] | None = None,
) -> dict:
    """Proportion d'indicateurs au niveau ou au-dessus de la médiane sectorielle × 100.

    Reproduit l'exemple du document : 4 indicateurs sur 5 au-dessus de la
    médiane → 80/100.
    """
    medians = sector_medians or DEFAULT_SECTOR_MEDIANS
    comparisons: list[dict] = []
    above = 0
    comparable = 0
    for key, median in medians.items():
        ratio = ratios.get(key)
        value = ratio["value"] if ratio else None
        if value is None:
            comparisons.append(
                {"indicateur": key, "valeur": None, "mediane": median, "statut": "Non calculable"}
            )
            continue
        comparable += 1
        better = value <= median if key in _LOWER_IS_BETTER else value >= median
        if better:
            above += 1
        comparisons.append(
            {
                "indicateur": key,
                "valeur": round(value, 4),
                "mediane": median,
                "statut": "Conforme" if better else "À surveiller",
            }
        )
    score = round(100.0 * above / comparable, 2) if comparable else 0.0
    return {"score": score, "comparaisons": comparisons, "indicateurs_compares": comparable}


# --- Agrégation et grille de décision ---

def compute_global_score(axe1_score: float, axe2_score: float, axe3_score: float) -> float:
    """Moyenne pondérée : Axe 1 75 %, Axe 2 15 %, Axe 3 10 %."""
    return (axe1_score * 0.75) + (axe2_score * 0.15) + (axe3_score * 0.10)


DECISION_GRID = [
    (90.0, "A+", "Excellent", "Accord sans condition"),
    (80.0, "A/B+", "Bon", "Accord — conditions standards"),
    (65.0, "B/B-", "Moyen", "Accord avec garanties complémentaires"),
    (50.0, "C", "Sensible", "Accord conditionné ou refus partiel"),
    (0.0, "D/F", "Risqué", "Refus recommandé / systématique"),
]


def map_score_to_decision(score: float) -> dict:
    """Grille à 5 niveaux du document (§5.3)."""
    for threshold, classe, decision, recommandation in DECISION_GRID:
        if score >= threshold:
            return {"classe": classe, "decision": decision, "recommandation": recommandation}
    return {"classe": "D/F", "decision": "Risqué", "recommandation": "Refus recommandé / systématique"}


def evaluate_application(
    bam_cotation: int | None,
    axe1: float,
    axe2: float,
    axe3: float,
    incidents: int = 0,
) -> DecisionResult:
    """Évaluation de bout en bout : critères bloquants puis score pondéré."""
    blocking = check_blocking_criteria(bam_cotation, incidents)
    if blocking == "NO_GO":
        return {
            "score": 0.0,
            "classe": "D/F",
            "decision": "Rejeté (Cotation BAM)",
            "recommandation": "Refus automatique — cotation BAM en zone de refus (7/8/9)",
            "blocking_status": blocking,
        }

    global_score = compute_global_score(axe1, axe2, axe3)
    grid = map_score_to_decision(global_score)
    return {
        "score": round(global_score, 2),
        "classe": grid["classe"],
        "decision": grid["decision"],
        "recommandation": grid["recommandation"],
        "blocking_status": blocking,
    }
