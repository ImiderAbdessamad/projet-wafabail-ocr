"""Schémas Pydantic pour le mapping Markdown → candidats financiers (Qwen)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


FinancialSection = Literal[
    "BILAN_ACTIF",
    "BILAN_PASSIF",
    "CPC",
    "DETAIL_CPC",
    "RESULTAT_FISCAL",
    "IDENTIFICATION",
    "AUTRE",
]


# Codes exposés à Qwen : jamais de suffixe _N1 (period porte N / N-1).
FinancialFieldCode = Literal[
    "EXERCICE",
    "CHIFFRE_AFFAIRES",
    "RESULTAT_NET",
    "RESULTAT_NET_XIII",
    "RESULTAT_NET_XVI",
    "RESULTAT_EXPLOITATION",
    "TOTAL_BILAN",
    "TOTAL_ACTIF",
    "TOTAL_PASSIF",
    "FONDS_PROPRES",
    "ACTIFS_IMMOBILISES",
    "ACTIF_CIRCULANT",
    "PASSIF_CIRCULANT",
    "STOCKS",
    "CLIENTS",
    "FOURNISSEURS",
    "TRESORERIE_ACTIF",
    "TRESORERIE_PASSIF",
    "DETTES_FINANCIERES",
    "DETTES_BANCAIRES_CT",
    "PRODUITS_EXPLOITATION",
    "CHARGES_EXPLOITATION",
    "PRODUITS_FINANCIERS",
    "CHARGES_FINANCIERES",
    "RESULTAT_FINANCIER",
    "RESULTAT_COURANT",
    "PRODUITS_NON_COURANTS",
    "CHARGES_NON_COURANTES",
    "RESULTAT_NON_COURANT",
    "RESULTAT_AVANT_IMPOT",
    "IMPOT_SUR_RESULTATS",
    "ACHATS_REVENDUS",
    "ACHATS_CONSOMMES",
    "ACHATS_TOTAL",
    "CHARGES_INTERETS",
    "DOTATIONS_AMORTISSEMENTS",
    "REPRISES",
    "PRODUITS_CESSION_IMMOBILISATIONS",
    "VALEUR_NETTE_IMMOBILISATIONS_CEDEES",
    "CAF",
    "FDR",
    "BFDR",
    "RESULTAT_FISCAL",
    "REINTEGRATIONS",
    "DEDUCTIONS",
    "IS_DU",
    "COTISATION_MINIMALE",
    "REPORT_DEFICITAIRE",
    "REDEVANCES_CREDIT_BAIL",
    "ENCOURS_LEASING",
    "CMT",
    "NOUVEAU_FINANCEMENT",
    "UNKNOWN",
]


CandidatePeriod = Literal[
    "N",
    "N_MINUS_1",
]


CandidateNature = Literal[
    "DETAIL",
    "SUBTOTAL",
    "SECTION_TOTAL",
    "GRAND_TOTAL",
    "DERIVED_DISPLAYED",
    "UNKNOWN",
]


ColumnRole = Literal[
    "BRUT",
    "AMORT_PROV",
    "NET_N",
    "EXERCICE_N",
    "TOTAL_EXERCICE_N",
    "EXERCICE_N1",
    "UNKNOWN",
]


class MappingEvidence(BaseModel):
    page_number: int
    section: FinancialSection
    raw_label: str
    raw_value: str
    column_name: str | None = None
    column_role: ColumnRole
    source_excerpt: str
    row_index: int | None = None


class FinancialCandidate(BaseModel):
    field_code: FinancialFieldCode
    raw_value: str = Field(min_length=1)
    period: CandidatePeriod
    nature: CandidateNature
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: MappingEvidence
    warnings: list[str] = Field(default_factory=list)


class FinancialMappingOutput(BaseModel):
    section: FinancialSection
    candidates: list[FinancialCandidate] = Field(default_factory=list)
    unresolved_labels: list[str] = Field(default_factory=list)
    document_warnings: list[str] = Field(default_factory=list)


class FinancialSectionInput(BaseModel):
    section: FinancialSection
    page_number: int
    markdown: str


class FinancialMappingBatchResult(BaseModel):
    model: str
    mapped_sections: list[FinancialMappingOutput] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processed_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    failed_sections: list[str] = Field(default_factory=list)


class FinancialMappingAudit(BaseModel):
    """Audit du mapping Qwen exposé par /pdf/analyze."""

    strategy: str = "qwen_only"
    model: str
    sections_detected: int
    sections_processed: int
    sections_skipped: int = 0
    sections_failed: int
    candidates_total: int
    candidates_by_field: dict[str, int] = Field(default_factory=dict)
    resolved_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    conflicting_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
