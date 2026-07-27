"""Schemas pour l'extraction intelligente des liasses fiscales marocaines (PCGM).

Structure alignée sur le référentiel des 19 éléments financiers calculés
et des 44 composantes brutes. Champs de provenance ajoutés en rétrocompatibilité.
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


class RawComponent(BaseModel):
    """Composante brute PCGM extraite de la liasse (1 des 44)."""

    label: str
    value: float = 0.0
    source: str  # "Bilan Actif" | "Bilan Passif" | "CPC" | "Bilan"
    feeds: Optional[str] = None


class FieldValidation(BaseModel):
    status: str = "unknown"  # consistent | divergent | warning | invalidated
    confirmed_by: list[str] = []


class FieldCalculation(BaseModel):
    formula: str
    inputs: dict[str, Optional[float]] = {}


class FinancialElement(BaseModel):
    """Élément financier calculé (1 des 19)."""

    number: int = Field(ge=1, le=19)
    code: str
    label: str
    value: Optional[float] = None
    unit: str = "MAD"
    source: str
    note: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    # detected | detected_zero | derived | ambiguous | conflicting |
    # not_detected | invalidated | empty
    detection_status: str = "not_detected"
    # Provenance (optionnelle, rétrocompatible)
    page: Optional[int] = None
    raw_label: Optional[str] = None
    column: Optional[str] = None
    period: Optional[str] = None
    selection_reason: Optional[str] = None
    validation: Optional[FieldValidation] = None
    calculation: Optional[FieldCalculation] = None


class ScoringInput(BaseModel):
    """Entrées financières extraites ou dérivées pour le moteur de ratios."""

    chiffre_affaires: Optional[float] = None
    ca_export: Optional[float] = None
    ca_n1: Optional[float] = None
    total_bilan: Optional[float] = None
    fonds_propres: Optional[float] = None
    actifs_immobilises: Optional[float] = None
    actif_circulant: Optional[float] = None
    clients: Optional[float] = None
    fournisseurs: Optional[float] = None
    dettes_financieres: Optional[float] = None
    dettes_bancaires_ct: Optional[float] = None
    passif_circulant: Optional[float] = None
    tresorerie_actif: Optional[float] = None
    tresorerie_passif: Optional[float] = None
    tresorerie_nette: Optional[float] = None
    achats: Optional[float] = None
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
    identification_fiscale: Optional[str] = None
    exercice: Optional[str] = None
    date_debut_exercice: Optional[str] = None
    date_fin_exercice: Optional[str] = None
    # "RAPPORT_INDICATEURS" | "LIASSE_OCR" | "LIASSE_NATIVE" | "LIASSE_ECHEC"
    document_kind: str
    elements: list[FinancialElement] = []
    raw_components: list[RawComponent] = []
    scoring_input: ScoringInput = ScoringInput()
    sections_completeness: dict[str, bool] = {}
    sections_detected: dict[str, bool] = {}
    sections_extraction_complete: dict[str, bool] = {}
    sections_validated: dict[str, bool] = {}
    completeness_pct: float = 0.0
    warnings: list[str] = []
    document_summary: Optional[str] = None
    pages_total: Optional[int] = None
    pages_analyzed: Optional[int] = None
    processing_time_ms: Optional[int] = None
    source_filename: Optional[str] = None
    # Debug / audit : résolutions par champ (candidates, scores)
    field_provenance: Optional[dict[str, Any]] = None
