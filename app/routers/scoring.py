"""API de scoring crédit-bail — moteur 3 axes (75 / 15 / 10).

POST /scoring/evaluate : entrée financière + comportementale + sectorielle,
sortie complète : 13 ratios détaillés, scores par axe avec contributions,
décision finale et synthèse (points forts / points de vigilance).
"""
from typing import Dict, List

from fastapi import APIRouter

from app.engines.ratio_engine import (
    RATIO_METADATA,
    calculate_extended_ratios,
    compute_yearly_variations,
)
from app.engines.scoring_engine import (
    evaluate_application,
    score_axe1_from_ratios,
    score_axe2_behavioral,
    score_axe3_sectoriel,
)
from app.schemas.scoring import (
    AxeScore,
    DecisionOutput,
    RatioDetail,
    ScoringEligibilityOutput,
    ScoringRequest,
    ScoringResponse,
    SyntheseOutput,
)
from app.services.scoring_eligibility import evaluate_manual_request_eligibility

router = APIRouter(prefix="/scoring", tags=["Scoring"])

_RATIO_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "autonomie_financiere": ("fonds_propres", "total_bilan"),
    "ratio_endettement": ("dettes_financieres", "fonds_propres"),
    "capacite_remboursement": ("dettes_financieres", "caf"),
    "caf_sur_ca": ("caf", "chiffre_affaires"),
    "rentabilite_commerciale": ("resultat_net", "chiffre_affaires"),
    "rentabilite_financiere": ("resultat_net", "fonds_propres"),
    "rentabilite_economique": ("resultat_net", "fonds_propres", "dettes_financieres"),
    "fdr_sur_ca": ("fdr", "chiffre_affaires"),
    "tresorerie_jours_ca": ("tresorerie_nette", "chiffre_affaires"),
    "delais_clients": ("clients", "chiffre_affaires"),
    "delais_fournisseurs": ("fournisseurs", "achats"),
    "croissance_ca": ("chiffre_affaires", "ca_n1"),
    "endettement_global_apres_operation": (
        "fonds_propres",
        "encours_leasing",
        "cmt",
        "nouveau_financement",
    ),
}

_FIELD_LABELS = {
    "fonds_propres": "fonds propres",
    "total_bilan": "total bilan",
    "dettes_financieres": "dettes financières",
    "caf": "CAF",
    "chiffre_affaires": "chiffre d'affaires",
    "resultat_net": "résultat net",
    "fdr": "fonds de roulement",
    "tresorerie_nette": "trésorerie nette",
    "clients": "créances clients",
    "fournisseurs": "dettes fournisseurs",
    "achats": "achats",
    "ca_n1": "CA N-1",
    "encours_leasing": "encours leasing",
    "cmt": "crédit moyen terme (CMT)",
    "nouveau_financement": "nouveau financement demandé",
}


def _ratio_missing_reason(key: str, raw_data: dict) -> str | None:
    required = _RATIO_REQUIRED_FIELDS.get(key, ())
    missing = [
        _FIELD_LABELS.get(field, field)
        for field in required
        if raw_data.get(field) is None
    ]
    if not missing:
        return None
    return "Non calculable : donnée(s) absente(s) — " + ", ".join(missing) + "."


def _build_ratio_details(
    ratios: Dict[str, dict], raw_data: dict
) -> Dict[str, RatioDetail]:
    details: Dict[str, RatioDetail] = {}
    for key, res in ratios.items():
        meta = RATIO_METADATA.get(key, {})
        details[key] = RatioDetail(
            label=meta.get("label", key),
            formula=meta.get("formula", ""),
            threshold=meta.get("threshold", ""),
            unit=meta.get("unit", ""),
            value=res["value"],
            status=res["status"],
            reason=res.get("reason")
            or (
                _ratio_missing_reason(key, raw_data)
                if res["status"] == "Non calculable"
                else None
            ),
        )
    return details


def _fmt(detail: RatioDetail) -> str:
    if detail.value is None:
        return "n/c"
    if detail.unit == "%":
        return f"{detail.value:.1%}"
    if detail.unit == "j":
        return f"{detail.value:.0f} j"
    if detail.unit in ("x", "ans"):
        return f"{detail.value:.2f}{detail.unit if detail.unit == 'x' else ' ans'}"
    return f"{detail.value:.2f}"


def _build_synthese(
    ratio_details: Dict[str, RatioDetail],
    axe2_details: dict,
    axe3_details: dict,
) -> SyntheseOutput:
    forts: List[str] = []
    vigilance: List[str] = []

    croissance = ratio_details.get("croissance_ca")
    if croissance and croissance.value is not None and croissance.value >= 0.05:
        forts.append(f"Croissance solide du chiffre d'affaires ({croissance.value:+.1%}).")
    autonomie = ratio_details.get("autonomie_financiere")
    if autonomie and autonomie.status == "Conforme" and autonomie.value is not None:
        forts.append(f"Structure financière saine : autonomie financière à {autonomie.value:.1%}.")
    treso = ratio_details.get("tresorerie_jours_ca")
    fdr = ratio_details.get("fdr_sur_ca")
    if treso and fdr and treso.status == "Conforme" and fdr.status == "Conforme":
        forts.append("Trésorerie nette et fonds de roulement positifs.")
    if axe2_details.get("status") == "not_provided":
        vigilance.append("Données comportementales non renseignées.")
    elif not axe2_details.get("signaux"):
        forts.append("Comportement bancaire irréprochable : aucun signal négatif relevé.")
    n_comp = axe3_details.get("indicateurs_compares", 0)
    n_above = sum(
        1 for c in axe3_details.get("comparaisons", []) if c["statut"] == "Conforme"
    )
    if n_comp and n_above / n_comp >= 0.6:
        forts.append(
            f"Bon positionnement sectoriel : au-dessus de la médiane sur {n_above} indicateur(s) sur {n_comp}."
        )

    for key, detail in ratio_details.items():
        if detail.status == "Non conforme":
            vigilance.append(f"{detail.label} non conforme ({_fmt(detail)} — seuil {detail.threshold}).")
        elif detail.status == "À surveiller":
            vigilance.append(f"{detail.label} à surveiller ({_fmt(detail)} — seuil {detail.threshold}).")
    vigilance.extend(axe2_details.get("signaux", []))

    return SyntheseOutput(points_forts=forts, points_vigilance=vigilance)


@router.post("/evaluate", response_model=ScoringResponse)
async def evaluate_scoring(request: ScoringRequest) -> ScoringResponse:
    """Point d'entrée unifié : ratios, 3 axes, critères bloquants, décision, synthèse."""
    raw_data = request.financial_data.model_dump()

    # Si un historique est fourni, dériver ca_n1 et les variations pluriannuelles
    variations: Dict[str, Dict[str, float | None]] = {}
    if request.financial_history:
        years = {
            y.fiscal_year: y.model_dump(exclude={"fiscal_year"})
            for y in request.financial_history
        }
        variations = compute_yearly_variations(years)
        if raw_data.get("ca_n1") is None and len(years) >= 2:
            prev_year = sorted(years)[-2]
            raw_data["ca_n1"] = years[prev_year].get("chiffre_affaires")

    # Axe 1 — ratios financiers
    ratios = calculate_extended_ratios(raw_data)
    axe1_result = score_axe1_from_ratios(ratios)

    # Axe 2 — comportemental
    b = request.behavioral_data
    axe2_result = score_axe2_behavioral(
        incidents_paiement=b.incidents_paiement or 0,
        rejets_prelevement=b.rejets_prelevement or 0,
        effets_impayes=b.effets_impayes or 0,
        domiciliation_ca_pct=b.domiciliation_ca_pct,
        jours_debit=b.jours_debit,
        utilisation_decouvert_pct=b.utilisation_decouvert_pct,
        ecart_flux_ca_pct=b.ecart_flux_ca_pct,
        engagements_honores=b.engagements_honores,
        provided_fields=set(getattr(b, "model_fields_set", set())),
    )

    # Axe 3 — sectoriel
    axe3_result = score_axe3_sectoriel(ratios, request.sector_medians)

    eligibility = evaluate_manual_request_eligibility(
        request.financial_data,
        request.behavioral_data,
        ratios,
    )

    total_incidents = (b.incidents_paiement or 0) + (b.effets_impayes or 0)
    if eligibility.eligible:
        decision = evaluate_application(
            bam_cotation=request.bam_cotation,
            axe1=axe1_result["score"],
            axe2=axe2_result["score"] or 0.0,
            axe3=axe3_result["score"],
            incidents=total_incidents,
        )
    else:
        decision = {
            "score": None,
            "classe": "Non évaluable",
            "decision": "Revue manuelle",
            "recommandation": " ; ".join(eligibility.blocking_reasons or ["Données insuffisantes pour le scoring automatique."]),
            "blocking_status": "INSUFFICIENT_DATA",
        }

    ratio_details = _build_ratio_details(ratios, raw_data)
    synthese = _build_synthese(ratio_details, axe2_result, axe3_result)

    return ScoringResponse(
        ratios=ratio_details,
        variations=variations,
        axe1=AxeScore(
            score=axe1_result["score"],
            ponderation=0.75,
            contribution=round(axe1_result["score"] * 0.75, 2),
            details={k: v for k, v in axe1_result.items() if k != "score"},
        ),
        axe2=AxeScore(
            score=axe2_result["score"] or 0.0,
            ponderation=0.15,
            contribution=round((axe2_result["score"] or 0.0) * 0.15, 2),
            details={k: v for k, v in axe2_result.items() if k != "score"},
        ),
        axe3=AxeScore(
            score=axe3_result["score"],
            ponderation=0.10,
            contribution=round(axe3_result["score"] * 0.10, 2),
            details={k: v for k, v in axe3_result.items() if k != "score"},
        ),
        decision=DecisionOutput(**decision),
        synthese=synthese,
        eligibility=ScoringEligibilityOutput(**eligibility.model_dump()),
    )
