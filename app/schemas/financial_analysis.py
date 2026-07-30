"""Schémas Pydantic pour l'analyse financière post-OCR Markdown (Decimal)."""
from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.financial_mapping import FinancialMappingAudit


DataStatus = Literal[
    "confirmed",
    "derived",
    "ambiguous",
    "conflicting",
    "missing",
    "invalid",
]

RatioStatus = Literal[
    "conforme",
    "a_surveiller",
    "non_conforme",
    "non_calculable",
]

DecisionClass = Literal[
    "A+",
    "A/B+",
    "B/B-",
    "C",
    "D/F",
    "NON_EVALUABLE",
]


class ValueProvenance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    page_number: int | None = None
    raw_label: str | None = None
    raw_value: str | None = None
    column_name: str | None = None
    extraction_method: str = "markdown_ocr"
    confidence: Decimal | None = None
    source_excerpt: str | None = None
    # Extensions rétrocompatibles (mapping Qwen)
    section: str | None = None
    nature: str | None = None
    period: str | None = None
    mapping_model: str | None = None


class FinancialValue(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str
    label: str
    value: Decimal | None = None
    unit: str = "MAD"
    status: DataStatus = "missing"
    provenance: list[ValueProvenance] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FinancialDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    exercice: str | None = None

    chiffre_affaires: FinancialValue
    chiffre_affaires_n1: FinancialValue
    resultat_net: FinancialValue
    resultat_net_n1: FinancialValue
    resultat_exploitation: FinancialValue
    total_bilan: FinancialValue
    total_bilan_n1: FinancialValue
    fonds_propres: FinancialValue
    fonds_propres_n1: FinancialValue

    dettes_financieres: FinancialValue
    dettes_financieres_n1: FinancialValue
    dettes_bancaires_ct: FinancialValue
    tresorerie_actif: FinancialValue
    tresorerie_passif: FinancialValue

    actif_circulant: FinancialValue
    passif_circulant: FinancialValue
    stocks: FinancialValue
    clients: FinancialValue
    fournisseurs: FinancialValue

    achats: FinancialValue
    frais_financiers: FinancialValue
    amortissements: FinancialValue
    caf: FinancialValue
    fdr: FinancialValue
    bfdr: FinancialValue
    tresorerie_nette: FinancialValue

    encours_leasing: FinancialValue
    cmt: FinancialValue
    nouveau_financement: FinancialValue

    resultat_fiscal: FinancialValue
    reintegrations: FinancialValue
    deductions: FinancialValue
    is_du: FinancialValue
    cotisation_minimale: FinancialValue
    report_deficitaire: FinancialValue

    # Agrégats optionnels pour contrôles
    actifs_immobilises: FinancialValue | None = None
    produits_exploitation: FinancialValue | None = None
    charges_exploitation: FinancialValue | None = None
    produits_financiers: FinancialValue | None = None
    charges_financieres: FinancialValue | None = None
    resultat_financier: FinancialValue | None = None
    resultat_courant: FinancialValue | None = None
    produits_non_courants: FinancialValue | None = None
    charges_non_courantes: FinancialValue | None = None
    resultat_non_courant: FinancialValue | None = None
    resultat_avant_impot: FinancialValue | None = None
    impot_sur_resultats: FinancialValue | None = None
    total_actif: FinancialValue | None = None
    total_passif: FinancialValue | None = None
    achats_revendus: FinancialValue | None = None
    achats_consommes: FinancialValue | None = None
    redevances_credit_bail: FinancialValue | None = None

    warnings: list[str] = Field(default_factory=list)


class RatioComponent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str
    label: str
    value: Decimal | None
    status: DataStatus


class RatioResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str
    label: str
    formula: str
    value: Decimal | None = None
    unit: str
    components: list[RatioComponent] = Field(default_factory=list)
    threshold: str | None = None
    status: RatioStatus = "non_calculable"
    points: Decimal = Decimal("0")
    max_points: Decimal = Decimal("0")
    warnings: list[str] = Field(default_factory=list)


class AxisScore(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str
    label: str
    raw_score: Decimal
    weight: Decimal
    weighted_contribution: Decimal
    calculable: bool = True
    blocking_reasons: list[str] = Field(default_factory=list)


class CreditDecision(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: Decimal | None = None
    risk_class: DecisionClass
    profile: str
    decision: str
    recommendation: str
    blocking_status: str | None = None


class AccountingControlResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    code: str
    status: Literal["passed", "failed", "not_testable"]
    expected: Decimal | None = None
    observed: Decimal | None = None
    difference: Decimal | None = None
    tolerance: Decimal | None = None
    affected_fields: list[str] = Field(default_factory=list)
    message: str = ""


class BehavioralInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    monthly_credit_movements: list[Decimal] = Field(default_factory=list)
    average_credit_balance: Decimal | None = None
    ca_domiciliation_pct: Decimal | None = None
    debit_position_days: int | None = None
    overdraft_usage_pct: Decimal | None = None
    payment_incidents_24m: int | None = None
    rejected_debits_24m: int | None = None
    unpaid_bills_24m: int | None = None
    leasing_payment_delays_24m: int | None = None
    bank_flows_vs_declared_ca_gap_pct: Decimal | None = None
    bam_rating: int | None = None


class SectorBenchmarkInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sector_name: str | None = None
    sample_size: int | None = None
    commercial_profitability_median: Decimal | None = None
    financial_autonomy_median: Decimal | None = None
    debt_ratio_median: Decimal | None = None
    repayment_capacity_median: Decimal | None = None
    ca_growth_median: Decimal | None = None
    # Quantiles optionnels pour percentile réel
    commercial_profitability_quantiles: list[Decimal] | None = None
    financial_autonomy_quantiles: list[Decimal] | None = None


class FinancialAnalysisResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: FinancialDataset
    accounting_checks: list[AccountingControlResult] = Field(default_factory=list)
    ratios: list[RatioResult] = Field(default_factory=list)
    axes: list[AxisScore] = Field(default_factory=list)
    final_score: Decimal | None = None
    decision: CreditDecision
    warnings: list[str] = Field(default_factory=list)
    scoring_mode: str = "STRICT"
    mapping: FinancialMappingAudit | None = None
