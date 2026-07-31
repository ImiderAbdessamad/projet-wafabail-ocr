"""Schémas pour l'extraction financière directe GLM Vision (image → JSON)."""
from __future__ import annotations

from decimal import Decimal
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.financial_analysis import (
    AccountingControlResult,
    AxisScore,
    CreditDecision,
    FinancialDataset,
    RatioResult,
)


FinancialPageType = Literal[
    "IDENTIFICATION",
    "BILAN_ACTIF",
    "BILAN_PASSIF",
    "CPC",
    "RESULTAT_FISCAL",
    "ESG",
    "DETAIL_CPC",
    "AUTRE",
    "VIDE",
]

FinancialPeriod = Literal["N", "N_MINUS_1"]

ColumnRole = Literal[
    "IDENTITY_VALUE",
    "BRUT",
    "AMORT_PROV",
    "NET_N",
    "EXERCICE_N",
    "TOTAL_EXERCICE_N",
    "EXERCICE_N1",
    "MONTANT_N",
    "MONTANT_N1",
    "UNKNOWN",
]

CandidateNature = Literal[
    "DETAIL",
    "SUBTOTAL",
    "SECTION_TOTAL",
    "GRAND_TOTAL",
]

OrientationDegrees = Literal[0, 90, 180, 270]


class DirectFinancialEvidence(BaseModel):
    page_number: int = Field(ge=1)
    page_type: FinancialPageType
    raw_label: str = Field(min_length=1, max_length=180)
    column_name: str | None = Field(default=None, max_length=100)
    column_role: ColumnRole
    source_excerpt: str = Field(min_length=1, max_length=240)
    orientation: OrientationDegrees = 0


class DirectFinancialCandidate(BaseModel):
    field_code: str
    raw_value: str = Field(min_length=1, max_length=64)
    period: FinancialPeriod
    nature: CandidateNature
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: DirectFinancialEvidence
    warnings: list[str] = Field(default_factory=list)


class FinancialPageAudit(BaseModel):
    page_number: int
    detected_type: FinancialPageType
    orientation: OrientationDegrees
    extraction_status: Literal["processed", "skipped", "failed", "empty"]
    extraction_strategy: str
    candidates_count: int = 0
    model_latency_ms: int | None = None
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None


class DirectFinancialExtractionBatch(BaseModel):
    model: str
    pages_total: int
    pages_processed: int
    pages_skipped: int
    pages_failed: int
    candidates: list[DirectFinancialCandidate] = Field(default_factory=list)
    page_audit: list[FinancialPageAudit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Schémas GLM par type de page
# ---------------------------------------------------------------------------

class _PageCandidateBase(BaseModel):
    raw_value: str = Field(min_length=1, max_length=64)
    period: FinancialPeriod
    nature: CandidateNature
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: DirectFinancialEvidence
    warnings: list[str] = Field(default_factory=list)


IdentificationFieldCode = Literal[
    "RAISON_SOCIALE",
    "IDENTIFIANT_FISCAL",
    "ICE",
    "TAXE_PROFESSIONNELLE",
    "ADRESSE",
    "VILLE",
    "DATE_DEBUT_EXERCICE",
    "DATE_FIN_EXERCICE",
    "DATE_DECLARATION",
    "REFERENCE_DECLARATION",
    "TYPE_DOCUMENT",
    "EXERCICE",
]

BilanActifFieldCode = Literal[
    "TOTAL_ACTIF",
    "ACTIFS_IMMOBILISES",
    "ACTIF_CIRCULANT",
    "STOCKS",
    "CLIENTS",
    "TRESORERIE_ACTIF",
    "TOTAL_ACTIF_I",
    "TOTAL_ACTIF_II",
    "TOTAL_ACTIF_III",
]

BilanPassifFieldCode = Literal[
    "TOTAL_PASSIF",
    "FONDS_PROPRES",
    "RESULTAT_NET",
    "DETTES_FINANCIERES",
    "PASSIF_CIRCULANT",
    "FOURNISSEURS",
    "TRESORERIE_PASSIF",
    "CAPITAUX_PERMANENTS",
    "TOTAL_PASSIF_I",
    "TOTAL_PASSIF_II",
    "TOTAL_PASSIF_III",
]

CpcFieldCode = Literal[
    "CHIFFRE_AFFAIRES",
    "RESULTAT_EXPLOITATION",
    "PRODUITS_EXPLOITATION",
    "CHARGES_EXPLOITATION",
    "PRODUITS_FINANCIERS",
    "CHARGES_FINANCIERES",
    "CHARGES_INTERETS",
    "RESULTAT_FINANCIER",
    "RESULTAT_COURANT",
    "PRODUITS_NON_COURANTS",
    "CHARGES_NON_COURANTES",
    "RESULTAT_NON_COURANT",
    "RESULTAT_AVANT_IMPOT",
    "IMPOT_SUR_RESULTATS",
    "RESULTAT_NET_XIII",
    "RESULTAT_NET_XVI",
    "ACHATS_REVENDUS",
    "ACHATS_CONSOMMES",
    "DOTATIONS_AMORTISSEMENTS",
    "PRODUITS_CESSION_IMMOBILISATIONS",
    "VALEUR_NETTE_IMMOBILISATIONS_CEDEES",
]

DetailCpcFieldCode = Literal[
    "REDEVANCES_CREDIT_BAIL",
    "AUTRES_CHARGES_EXTERNES_DETAIL",
    "LOYERS",
    "ENTRETIEN_REPARATIONS",
    "TRANSPORTS",
    "HONORAIRES",
]

FiscalFieldCode = Literal[
    "RESULTAT_COMPTABLE",
    "REINTEGRATIONS",
    "DEDUCTIONS",
    "RESULTAT_FISCAL",
    "IS_DU",
    "COTISATION_MINIMALE",
    "REPORT_DEFICITAIRE",
]

EsgFieldCode = Literal[
    "CAF",
    "AUTOFINANCEMENT",
    "VALEUR_AJOUTEE",
    "EBE",
    "RESULTAT_NET_ESG",
]


class IdentificationCandidate(_PageCandidateBase):
    field_code: IdentificationFieldCode


class BilanActifCandidate(_PageCandidateBase):
    field_code: BilanActifFieldCode


class BilanPassifCandidate(_PageCandidateBase):
    field_code: BilanPassifFieldCode


class CpcCandidate(_PageCandidateBase):
    field_code: CpcFieldCode


class DetailCpcCandidate(_PageCandidateBase):
    field_code: DetailCpcFieldCode


class FiscalCandidate(_PageCandidateBase):
    field_code: FiscalFieldCode


class EsgCandidate(_PageCandidateBase):
    field_code: EsgFieldCode


class IdentificationOutput(BaseModel):
    page_type: Literal["IDENTIFICATION"] = "IDENTIFICATION"
    candidates: list[IdentificationCandidate] = Field(default_factory=list, max_length=20)


class BilanActifOutput(BaseModel):
    page_type: Literal["BILAN_ACTIF"] = "BILAN_ACTIF"
    candidates: list[BilanActifCandidate] = Field(default_factory=list, max_length=30)


class BilanPassifOutput(BaseModel):
    page_type: Literal["BILAN_PASSIF"] = "BILAN_PASSIF"
    candidates: list[BilanPassifCandidate] = Field(default_factory=list, max_length=30)


class CpcOutput(BaseModel):
    page_type: Literal["CPC"] = "CPC"
    candidates: list[CpcCandidate] = Field(default_factory=list, max_length=40)


class DetailCpcOutput(BaseModel):
    page_type: Literal["DETAIL_CPC"] = "DETAIL_CPC"
    candidates: list[DetailCpcCandidate] = Field(default_factory=list, max_length=15)


class FiscalOutput(BaseModel):
    page_type: Literal["RESULTAT_FISCAL"] = "RESULTAT_FISCAL"
    candidates: list[FiscalCandidate] = Field(default_factory=list, max_length=20)


class EsgOutput(BaseModel):
    page_type: Literal["ESG"] = "ESG"
    candidates: list[EsgCandidate] = Field(default_factory=list, max_length=15)


PAGE_TYPE_SCHEMAS: dict[str, type[BaseModel]] = {
    "IDENTIFICATION": IdentificationOutput,
    "BILAN_ACTIF": BilanActifOutput,
    "BILAN_PASSIF": BilanPassifOutput,
    "CPC": CpcOutput,
    "DETAIL_CPC": DetailCpcOutput,
    "RESULTAT_FISCAL": FiscalOutput,
    "ESG": EsgOutput,
}


# ---------------------------------------------------------------------------
# Schémas LEGERS pour Ollama (contrainte JSON) — sans page_number / page_type
# imbriqués. Le pipeline injecte ces métadonnées après coup.
# ---------------------------------------------------------------------------


class GlmLiteEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    raw_label: str = Field(min_length=1, max_length=120)
    column_name: str | None = Field(default=None, max_length=80)
    column_role: ColumnRole = "UNKNOWN"
    source_excerpt: str = Field(default="", max_length=160)


class GlmLiteCandidate(BaseModel):
    """Candidat minimal : field_code en str (filtré ensuite côté Python)."""

    model_config = ConfigDict(extra="ignore")

    field_code: str = Field(min_length=2, max_length=64)
    raw_value: str = Field(min_length=1, max_length=64)
    period: FinancialPeriod = "N"
    nature: CandidateNature = "DETAIL"
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    evidence: GlmLiteEvidence
    warnings: list[str] = Field(default_factory=list)


class GlmLitePageOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidates: list[GlmLiteCandidate] = Field(default_factory=list, max_length=15)


ALLOWED_FIELD_CODES: dict[str, frozenset[str]] = {
    "IDENTIFICATION": frozenset(get_args(IdentificationFieldCode)),
    "BILAN_ACTIF": frozenset(get_args(BilanActifFieldCode)),
    "BILAN_PASSIF": frozenset(get_args(BilanPassifFieldCode)),
    "CPC": frozenset(get_args(CpcFieldCode))
    | frozenset(
        {"RESULTAT_NET", "ACHATS", "FRAIS_FINANCIERS", "AMORTISSEMENTS"}
    ),
    "DETAIL_CPC": frozenset(get_args(DetailCpcFieldCode)),
    "RESULTAT_FISCAL": frozenset(get_args(FiscalFieldCode)),
    "ESG": frozenset(get_args(EsgFieldCode))
    | frozenset({"FDR", "BFDR", "TRESORERIE_NETTE"}),
}

PRIORITY_FIELDS: dict[str, tuple[str, ...]] = {
    "IDENTIFICATION": (
        "RAISON_SOCIALE",
        "IDENTIFIANT_FISCAL",
        "ICE",
        "ADRESSE",
        "DATE_DEBUT_EXERCICE",
        "DATE_FIN_EXERCICE",
        "EXERCICE",
    ),
    "BILAN_ACTIF": (
        "TOTAL_ACTIF",
        "ACTIFS_IMMOBILISES",
        "ACTIF_CIRCULANT",
        "STOCKS",
        "CLIENTS",
        "TRESORERIE_ACTIF",
    ),
    "BILAN_PASSIF": (
        "TOTAL_PASSIF",
        "FONDS_PROPRES",
        "RESULTAT_NET",
        "DETTES_FINANCIERES",
        "PASSIF_CIRCULANT",
        "FOURNISSEURS",
        "TRESORERIE_PASSIF",
    ),
    "CPC": (
        "CHIFFRE_AFFAIRES",
        "RESULTAT_NET",
        "RESULTAT_NET_XVI",
        "RESULTAT_EXPLOITATION",
        "RESULTAT_COURANT",
        "CHARGES_FINANCIERES",
        "ACHATS_REVENDUS",
        "ACHATS_CONSOMMES",
        "DOTATIONS_AMORTISSEMENTS",
    ),
    "DETAIL_CPC": ("REDEVANCES_CREDIT_BAIL",),
    "RESULTAT_FISCAL": (
        "RESULTAT_FISCAL",
        "REINTEGRATIONS",
        "DEDUCTIONS",
        "IS_DU",
        "COTISATION_MINIMALE",
        "REPORT_DEFICITAIRE",
    ),
    "ESG": ("CAF", "EBE", "VALEUR_AJOUTEE"),
}


class CompanyInfo(BaseModel):
    raison_sociale: str | None = None
    identifiant_fiscal: str | None = None
    ice: str | None = None
    adresse: str | None = None
    ville: str | None = None


class ExerciseInfo(BaseModel):
    debut: str | None = None
    fin: str | None = None
    label: str | None = None


class DocumentSummary(BaseModel):
    filename: str
    pages_total: int
    pages_processed: int
    pages_skipped: int
    pages_failed: int
    document_type: str = "LIASSE_FISCALE"
    company: CompanyInfo = Field(default_factory=CompanyInfo)
    exercise: ExerciseInfo = Field(default_factory=ExerciseInfo)


class ExtractionSummary(BaseModel):
    model: str
    page_audit: list[FinancialPageAudit] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FinancialDocumentAnalysisResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    document: DocumentSummary
    extraction: ExtractionSummary
    dataset: FinancialDataset
    accounting_checks: list[AccountingControlResult] = Field(default_factory=list)
    ratios: list[RatioResult] = Field(default_factory=list)
    axes: list[AxisScore] = Field(default_factory=list)
    decision: CreditDecision | None = None
    warnings: list[str] = Field(default_factory=list)
    markdown_pages: list[dict] | None = None


JobStatus = Literal[
    "queued",
    "processing",
    "completed",
    "failed",
]


class FinancialJobProgress(BaseModel):
    job_id: str
    status: JobStatus
    progress_pct: int = 0
    current_step: str = "queued"
    current_page: int | None = None
    pages_total: int | None = None
    pages_financial: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    message: str = ""
    error: str | None = None
    stream_url: str | None = None
    result_url: str | None = None


class FinancialJobCreateResponse(BaseModel):
    job_id: str
    status: JobStatus = "queued"
    stream_url: str
    result_url: str
