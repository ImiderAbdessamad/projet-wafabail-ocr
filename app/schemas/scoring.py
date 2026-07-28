"""Modèles d'entrée/sortie de l'API de scoring crédit-bail."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class FinancialDataInput(BaseModel):
    # Postes de l'exercice N
    chiffre_affaires: Optional[float] = None
    produits_exploitation: Optional[float] = None
    fonds_propres: Optional[float] = None
    total_bilan: Optional[float] = None
    dettes_financieres: Optional[float] = None
    resultat_net: Optional[float] = None
    caf: Optional[float] = None
    stocks: Optional[float] = None
    clients: Optional[float] = None
    fournisseurs: Optional[float] = None
    frais_financiers: Optional[float] = None
    amortissements: Optional[float] = None
    charges_leasing: Optional[float] = None
    achats: Optional[float] = None  # requis pour délais fournisseurs

    # Métriques dérivées
    fdr: Optional[float] = None
    tresorerie_nette: Optional[float] = None

    # Pluriannuel et endettement après opération
    ca_n1: Optional[float] = None  # CA de l'exercice N-1 (croissance)
    encours_leasing: Optional[float] = None
    cmt: Optional[float] = None  # crédit moyen terme en cours
    nouveau_financement: Optional[float] = None  # montant objet du dossier


class YearlyFinancialData(FinancialDataInput):
    fiscal_year: int


class BehavioralMetricsInput(BaseModel):
    mouvements_confies: Optional[float] = None
    solde_moyen_crediteur: Optional[float] = None
    domiciliation_ca_pct: Optional[float] = None
    jours_debit: Optional[int] = None
    utilisation_decouvert_pct: Optional[float] = None
    incidents_paiement: Optional[int] = 0
    rejets_prelevement: Optional[int] = 0
    effets_impayes: Optional[int] = 0
    ecart_flux_ca_pct: Optional[float] = None
    engagements_honores: Optional[bool] = None


class ScoringEligibilityOutput(BaseModel):
    eligible: bool
    mode: str
    blocking_reasons: List[str] = []
    warnings: List[str] = []
    financial_coverage: float = 0.0
    behavioral_coverage: float = 0.0
    sector_coverage: float = 0.0
    ratios_expected: int = 0
    ratios_calculable: int = 0
    ratio_coverage: float = 0.0


class ScoringRequest(BaseModel):
    bam_cotation: Optional[int] = None
    financial_data: FinancialDataInput
    behavioral_data: BehavioralMetricsInput = BehavioralMetricsInput()
    # Historique optionnel pour le calcul des variations pluriannuelles
    financial_history: Optional[List[YearlyFinancialData]] = None
    # Médianes sectorielles du panel de référence (sinon défaut Transport & Logistique)
    sector_medians: Optional[Dict[str, float]] = None
    secteur: Optional[str] = None


class RatioDetail(BaseModel):
    label: str
    formula: str
    threshold: str
    unit: str
    value: Optional[float] = None
    status: str
    reason: Optional[str] = None


class AxeScore(BaseModel):
    score: float
    ponderation: float
    contribution: float
    details: Dict[str, Any] = {}


class DecisionOutput(BaseModel):
    score: Optional[float] = None
    classe: str
    decision: str
    recommandation: str
    blocking_status: Optional[str] = None


class SyntheseOutput(BaseModel):
    points_forts: List[str] = []
    points_vigilance: List[str] = []


class ScoringResponse(BaseModel):
    ratios: Dict[str, RatioDetail]
    variations: Dict[str, Dict[str, Optional[float]]] = {}
    axe1: AxeScore
    axe2: AxeScore
    axe3: AxeScore
    decision: DecisionOutput
    synthese: SyntheseOutput
    eligibility: Optional[ScoringEligibilityOutput] = None
