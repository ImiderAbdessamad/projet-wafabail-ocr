"""Schemas pour l'extraction intelligente des liasses fiscales marocaines (PCGM).

Structure alignée sur le référentiel des 19 éléments financiers calculés
et des 44 composantes brutes (cf. document « Formules de calcul des champs
financiers » et rapports d'indicateurs générés par le pipeline d'extraction).
"""
from typing import Optional

from pydantic import BaseModel, Field


class RawComponent(BaseModel):
    """Composante brute PCGM extraite de la liasse (1 des 44)."""

    label: str
    value: float = 0.0
    source: str  # "Bilan Actif" | "Bilan Passif" | "CPC" | "Bilan"
    feeds: Optional[str] = None  # élément calculé alimenté par cette composante


class FinancialElement(BaseModel):
    """Élément financier calculé (1 des 19)."""

    number: int = Field(ge=1, le=19)
    code: str
    label: str
    value: Optional[float] = None
    unit: str = "MAD"
    source: str
    note: Optional[str] = None  # ex. "Bénéficiaire"
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    # detected : valeur lue ; empty : libellé visible mais case vide ;
    # not_detected : libellé/valeur non identifié dans le document ;
    # derived : information calculée à partir d'un autre poste.
    detection_status: str = "not_detected"


class ScoringInput(BaseModel):
    """Entrées financières extraites ou dérivées pour le moteur de ratios.

    Les trois derniers postes d'endettement sont généralement externes à la
    liasse fiscale et peuvent être complétés par l'analyste via l'endpoint
    `/liasse/score`.
    """

    chiffre_affaires: Optional[float] = None
    ca_export: Optional[float] = None
    ca_n1: Optional[float] = None
    total_bilan: Optional[float] = None
    fonds_propres: Optional[float] = None
    actifs_immobilises: Optional[float] = None
    actif_circulant: Optional[float] = None
    clients: Optional[float] = None
    fournisseurs: Optional[float] = None
    dettes_financieres: Optional[float] = None  # dettes bancaires MLT
    dettes_bancaires_ct: Optional[float] = None
    passif_circulant: Optional[float] = None
    tresorerie_actif: Optional[float] = None
    tresorerie_passif: Optional[float] = None
    tresorerie_nette: Optional[float] = None
    achats: Optional[float] = None  # achats revendus (délais fournisseurs)
    frais_financiers: Optional[float] = None
    amortissements: Optional[float] = None
    caf: Optional[float] = None
    fdr: Optional[float] = None
    resultat_net: Optional[float] = None
    compte_courant_associes: Optional[float] = None
    encours_leasing: Optional[float] = None
    cmt: Optional[float] = None
    nouveau_financement: Optional[float] = None


class LiasseExtractionResult(BaseModel):
    """Résultat complet d'extraction d'une liasse fiscale."""

    reference: Optional[str] = None
    entreprise: Optional[str] = None
    # "RAPPORT_INDICATEURS" | "LIASSE_OCR" | "LIASSE_NATIVE" | "LIASSE_ECHEC"
    document_kind: str
    elements: list[FinancialElement] = []
    raw_components: list[RawComponent] = []
    scoring_input: ScoringInput = ScoringInput()
    sections_completeness: dict[str, bool] = {}  # {"BILAN_ACTIF": bool, ...}
    completeness_pct: float = 0.0
    warnings: list[str] = []
    document_summary: Optional[str] = None
    # Métadonnées OCR multi-pages
    pages_total: Optional[int] = None
    pages_analyzed: Optional[int] = None
    processing_time_ms: Optional[int] = None
    source_filename: Optional[str] = None
