"""Schémas Pydantic pour le mapping Markdown → candidats financiers (Qwen).

Les schémas exposés à Qwen sont spécifiques par section afin d'interdire
les field_code hors périmètre. FinancialCandidate reste le type interne
commun après conversion.
"""
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


BilanActifFieldCode = Literal[
    "TOTAL_ACTIF",
    "ACTIFS_IMMOBILISES",
    "ACTIF_CIRCULANT",
    "STOCKS",
    "CLIENTS",
    "TRESORERIE_ACTIF",
]

BilanPassifFieldCode = Literal[
    "TOTAL_PASSIF",
    "FONDS_PROPRES",
    "RESULTAT_NET",
    "DETTES_FINANCIERES",
    "PASSIF_CIRCULANT",
    "FOURNISSEURS",
    "TRESORERIE_PASSIF",
]

CpcFieldCode = Literal[
    "CHIFFRE_AFFAIRES",
    "RESULTAT_NET",
    "RESULTAT_NET_XIII",
    "RESULTAT_NET_XVI",
    "RESULTAT_EXPLOITATION",
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
    "CHARGES_INTERETS",
    "DOTATIONS_AMORTISSEMENTS",
    "PRODUITS_CESSION_IMMOBILISATIONS",
    "VALEUR_NETTE_IMMOBILISATIONS_CEDEES",
]

DetailCpcFieldCode = Literal[
    "REDEVANCES_CREDIT_BAIL",
]

FiscalFieldCode = Literal[
    "RESULTAT_FISCAL",
    "REINTEGRATIONS",
    "DEDUCTIONS",
    "IS_DU",
    "COTISATION_MINIMALE",
    "REPORT_DEFICITAIRE",
]


# Union interne (resolver / dataset) — jamais exposée telle quelle à Qwen.
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
    raw_label: str = Field(min_length=1, max_length=180)
    column_name: str | None = Field(default=None, max_length=100)
    column_role: ColumnRole
    source_excerpt: str = Field(min_length=1, max_length=240)
    row_index: int | None = None


class FinancialCandidate(BaseModel):
    """Candidat interne commun (après conversion depuis un schéma de section)."""

    field_code: str
    raw_value: str = Field(min_length=1, max_length=64)
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


# ---------------------------------------------------------------------------
# Schémas Qwen par section
# ---------------------------------------------------------------------------


class _SectionCandidateBase(BaseModel):
    raw_value: str = Field(min_length=1, max_length=64)
    period: CandidatePeriod
    nature: CandidateNature
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: MappingEvidence
    warnings: list[str] = Field(default_factory=list)


class BilanActifCandidate(_SectionCandidateBase):
    field_code: BilanActifFieldCode


class BilanPassifCandidate(_SectionCandidateBase):
    field_code: BilanPassifFieldCode


class CpcCandidate(_SectionCandidateBase):
    field_code: CpcFieldCode


class DetailCpcCandidate(_SectionCandidateBase):
    field_code: DetailCpcFieldCode


class FiscalCandidate(_SectionCandidateBase):
    field_code: FiscalFieldCode


class BilanActifMappingOutput(BaseModel):
    section: Literal["BILAN_ACTIF"]
    candidates: list[BilanActifCandidate] = Field(default_factory=list, max_length=30)
    unresolved_labels: list[str] = Field(default_factory=list)
    document_warnings: list[str] = Field(default_factory=list)


class BilanPassifMappingOutput(BaseModel):
    section: Literal["BILAN_PASSIF"]
    candidates: list[BilanPassifCandidate] = Field(default_factory=list, max_length=30)
    unresolved_labels: list[str] = Field(default_factory=list)
    document_warnings: list[str] = Field(default_factory=list)


class CpcMappingOutput(BaseModel):
    section: Literal["CPC"]
    candidates: list[CpcCandidate] = Field(default_factory=list, max_length=40)
    unresolved_labels: list[str] = Field(default_factory=list)
    document_warnings: list[str] = Field(default_factory=list)


class DetailCpcMappingOutput(BaseModel):
    section: Literal["DETAIL_CPC"]
    candidates: list[DetailCpcCandidate] = Field(default_factory=list, max_length=10)
    unresolved_labels: list[str] = Field(default_factory=list)
    document_warnings: list[str] = Field(default_factory=list)


class FiscalMappingOutput(BaseModel):
    section: Literal["RESULTAT_FISCAL"]
    candidates: list[FiscalCandidate] = Field(default_factory=list, max_length=20)
    unresolved_labels: list[str] = Field(default_factory=list)
    document_warnings: list[str] = Field(default_factory=list)


SECTION_MAPPING_MODELS: dict[str, type[BaseModel]] = {
    "BILAN_ACTIF": BilanActifMappingOutput,
    "BILAN_PASSIF": BilanPassifMappingOutput,
    "CPC": CpcMappingOutput,
    "DETAIL_CPC": DetailCpcMappingOutput,
    "RESULTAT_FISCAL": FiscalMappingOutput,
}


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


def to_common_mapping_output(section_output: BaseModel) -> FinancialMappingOutput:
    """Convertit une sortie de section Qwen vers FinancialMappingOutput interne."""
    data = section_output.model_dump()
    candidates = [
        FinancialCandidate.model_validate(c) for c in data.get("candidates", [])
    ]
    return FinancialMappingOutput(
        section=data["section"],
        candidates=candidates,
        unresolved_labels=list(data.get("unresolved_labels") or []),
        document_warnings=list(data.get("document_warnings") or []),
    )
