# -*- coding: utf-8 -*-
"""Intégration mapping SERDILAB (candidats Qwen mockés, calculs Python)."""
from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from main import app
from app.schemas.financial_mapping import (
    FinancialCandidate,
    FinancialMappingBatchResult,
    FinancialMappingOutput,
    MappingEvidence,
)
from app.schemas.pdf_extraction import PdfContentExtractionResult, PdfPageExtraction
from app.services.analysis_pipeline import analyze_extracted_pdf
from app.services.financial_candidate_resolver import (
    build_financial_dataset_from_resolved_values,
    resolve_financial_candidates,
)
from app.services.financial_ratios import (
    calculate_commercial_profitability,
    calculate_financial_autonomy,
    calculate_financial_ratios,
)

client = TestClient(app)


def _c(
    code: str,
    raw: str,
    *,
    section: str,
    label: str,
    column: str,
    period: str = "N",
    nature: str = "DETAIL",
    page: int = 1,
) -> FinancialCandidate:
    return FinancialCandidate(
        field_code=code,  # type: ignore[arg-type]
        raw_value=raw,
        period=period,  # type: ignore[arg-type]
        nature=nature,  # type: ignore[arg-type]
        confidence=0.85,
        evidence=MappingEvidence(
            page_number=page,
            section=section,  # type: ignore[arg-type]
            raw_label=label,
            raw_value=raw,
            column_name=column,
            source_excerpt=f"| {label} | {raw} |",
        ),
    )


def serdilab_mapping_outputs() -> list[FinancialMappingOutput]:
    """Candidats Qwen simulés pour les montants SERDILAB attendus."""
    return [
        FinancialMappingOutput(
            section="BILAN_ACTIF",
            candidates=[
                _c(
                    "TOTAL_ACTIF",
                    "22 303 497,11",
                    section="BILAN_ACTIF",
                    label="Total général (I + II + III)",
                    column="Net",
                    nature="GRAND_TOTAL",
                    page=2,
                ),
                _c(
                    "TOTAL_BILAN_N1",
                    "19 500 619,98",
                    section="BILAN_ACTIF",
                    label="Total général (I + II + III)",
                    column="Exercice précédent",
                    period="N_MINUS_1",
                    nature="GRAND_TOTAL",
                    page=2,
                ),
                _c(
                    "ACTIFS_IMMOBILISES",
                    "338 562,41",
                    section="BILAN_ACTIF",
                    label="Total I Actif immobilisé",
                    column="Net",
                    nature="SECTION_TOTAL",
                    page=2,
                ),
                _c(
                    "ACTIF_CIRCULANT",
                    "21 763 766,88",
                    section="BILAN_ACTIF",
                    label="Total II Actif circulant",
                    column="Net",
                    nature="SECTION_TOTAL",
                    page=2,
                ),
                _c(
                    "STOCKS",
                    "949 635,00",
                    section="BILAN_ACTIF",
                    label="STOCKS",
                    column="Net",
                    nature="SECTION_TOTAL",
                    page=2,
                ),
                _c(
                    "CLIENTS",
                    "19 097 949,49",
                    section="BILAN_ACTIF",
                    label="Clients et comptes rattachés",
                    column="Net",
                    page=2,
                ),
                _c(
                    "TRESORERIE_ACTIF",
                    "201 167,82",
                    section="BILAN_ACTIF",
                    label="Total III Trésorerie-Actif",
                    column="Net",
                    nature="SECTION_TOTAL",
                    page=2,
                ),
            ],
        ),
        FinancialMappingOutput(
            section="BILAN_PASSIF",
            candidates=[
                _c(
                    "TOTAL_PASSIF",
                    "22 303 497,11",
                    section="BILAN_PASSIF",
                    label="Total général (I + II + III)",
                    column="Exercice",
                    nature="GRAND_TOTAL",
                    page=3,
                ),
                _c(
                    "FONDS_PROPRES",
                    "9 114 715,17",
                    section="BILAN_PASSIF",
                    label="TOTAL DES CAPITAUX PROPRES",
                    column="Exercice",
                    nature="SECTION_TOTAL",
                    page=3,
                ),
                _c(
                    "FONDS_PROPRES_N1",
                    "7 934 906,01",
                    section="BILAN_PASSIF",
                    label="TOTAL DES CAPITAUX PROPRES",
                    column="Exercice précédent",
                    period="N_MINUS_1",
                    nature="SECTION_TOTAL",
                    page=3,
                ),
                _c(
                    "DETTES_FINANCIERES",
                    "133 308,11",
                    section="BILAN_PASSIF",
                    label="DETTES DE FINANCEMENT",
                    column="Exercice",
                    nature="SECTION_TOTAL",
                    page=3,
                ),
                _c(
                    "PASSIF_CIRCULANT",
                    "13 055 473,83",
                    section="BILAN_PASSIF",
                    label="Total II Passif circulant",
                    column="Exercice",
                    nature="SECTION_TOTAL",
                    page=3,
                ),
                _c(
                    "FOURNISSEURS",
                    "4 146 301,83",
                    section="BILAN_PASSIF",
                    label="Fournisseurs et comptes rattachés",
                    column="Exercice",
                    page=3,
                ),
                _c(
                    "TRESORERIE_PASSIF",
                    "0,00",
                    section="BILAN_PASSIF",
                    label="Total III Trésorerie-Passif",
                    column="Exercice",
                    nature="SECTION_TOTAL",
                    page=3,
                ),
            ],
        ),
        FinancialMappingOutput(
            section="CPC",
            candidates=[
                _c(
                    "CHIFFRE_AFFAIRES",
                    "13 404 177,00",
                    section="CPC",
                    label="Chiffre d'affaires",
                    column="Totaux de l'exercice",
                    page=4,
                ),
                _c(
                    "CHIFFRE_AFFAIRES_N1",
                    "24 105 417,32",
                    section="CPC",
                    label="Chiffre d'affaires",
                    column="Exercice précédent",
                    period="N_MINUS_1",
                    page=4,
                ),
                _c(
                    "RESULTAT_NET",
                    "1 179 809,16",
                    section="CPC",
                    label="XIII RESULTAT NET",
                    column="Totaux de l'exercice",
                    nature="SECTION_TOTAL",
                    page=4,
                ),
                _c(
                    "RESULTAT_NET",
                    "1 179 809,16",
                    section="CPC",
                    label="XVI RESULTAT NET",
                    column="Totaux de l'exercice",
                    nature="SECTION_TOTAL",
                    page=4,
                ),
                _c(
                    "RESULTAT_NET_N1",
                    "670 378,06",
                    section="CPC",
                    label="RESULTAT NET",
                    column="Exercice précédent",
                    period="N_MINUS_1",
                    page=4,
                ),
                _c(
                    "ACHATS_REVENDUS",
                    "9 295 560,07",
                    section="CPC",
                    label="Achats revendus de marchandises",
                    column="Totaux de l'exercice",
                    nature="SECTION_TOTAL",
                    page=4,
                ),
                _c(
                    "ACHATS_CONSOMMES",
                    "430 367,29",
                    section="CPC",
                    label="Achats consommés de matières et fournitures",
                    column="Totaux de l'exercice",
                    page=4,
                ),
                _c(
                    "CHARGES_INTERETS",
                    "95 394,47",
                    section="CPC",
                    label="Charges d'intérêts",
                    column="Totaux de l'exercice",
                    page=4,
                ),
                _c(
                    "PRODUITS_FINANCIERS",
                    "7 082,15",
                    section="CPC",
                    label="Produits financiers",
                    column="Totaux de l'exercice",
                    page=4,
                ),
                _c(
                    "CHARGES_FINANCIERES",
                    "200 928,82",
                    section="CPC",
                    label="Charges financières",
                    column="Totaux de l'exercice",
                    page=4,
                ),
                _c(
                    "RESULTAT_FINANCIER",
                    "200 928,82",
                    section="CPC",
                    label="Résultat financier",
                    column="Totaux de l'exercice",
                    page=4,
                ),
                _c(
                    "RESULTAT_COURANT",
                    "1 187 736,60",
                    section="CPC",
                    label="Résultat courant",
                    column="Totaux de l'exercice",
                    page=4,
                ),
                _c(
                    "PRODUITS_NON_COURANTS",
                    "64 027,85",
                    section="CPC",
                    label="Produits non courants",
                    column="Totaux de l'exercice",
                    page=4,
                ),
                _c(
                    "CHARGES_NON_COURANTES",
                    "71 955,29",
                    section="CPC",
                    label="Charges non courantes",
                    column="Totaux de l'exercice",
                    page=4,
                ),
                _c(
                    "RESULTAT_NON_COURANT",
                    "-7 927,44",
                    section="CPC",
                    label="Résultat non courant",
                    column="Totaux de l'exercice",
                    page=4,
                ),
                _c(
                    "RESULTAT_AVANT_IMPOT",
                    "1 179 809,16",
                    section="CPC",
                    label="Résultat avant impôts",
                    column="Totaux de l'exercice",
                    page=4,
                ),
                _c(
                    "IMPOT_SUR_RESULTATS",
                    "0,00",
                    section="CPC",
                    label="Impôts sur les résultats",
                    column="Totaux de l'exercice",
                    page=4,
                ),
            ],
        ),
        FinancialMappingOutput(
            section="DETAIL_CPC",
            candidates=[
                _c(
                    "REDEVANCES_CREDIT_BAIL",
                    "21 729,13",
                    section="DETAIL_CPC",
                    label="Redevances de crédit-bail",
                    column="Totaux de l'exercice",
                    page=5,
                ),
            ],
        ),
    ]


def test_serdilab_resolved_values():
    resolved = resolve_financial_candidates(serdilab_mapping_outputs())

    assert resolved["CHIFFRE_AFFAIRES"].value == Decimal("13404177.00")
    assert resolved["CHIFFRE_AFFAIRES_N1"].value == Decimal("24105417.32")
    assert resolved["RESULTAT_NET"].value == Decimal("1179809.16")
    assert resolved["RESULTAT_NET_N1"].value == Decimal("670378.06")
    assert resolved["TOTAL_BILAN"].value == Decimal("22303497.11")
    assert resolved["FONDS_PROPRES"].value == Decimal("9114715.17")
    assert resolved["FONDS_PROPRES_N1"].value == Decimal("7934906.01")
    assert resolved["DETTES_FINANCIERES"].value == Decimal("133308.11")
    assert resolved["ACTIFS_IMMOBILISES"].value == Decimal("338562.41")
    assert resolved["ACTIF_CIRCULANT"].value == Decimal("21763766.88")
    assert resolved["PASSIF_CIRCULANT"].value == Decimal("13055473.83")
    assert resolved["STOCKS"].value == Decimal("949635.00")
    assert resolved["CLIENTS"].value == Decimal("19097949.49")
    assert resolved["FOURNISSEURS"].value == Decimal("4146301.83")
    assert resolved["TRESORERIE_ACTIF"].value == Decimal("201167.82")
    assert resolved["TRESORERIE_PASSIF"].value == Decimal("0.00")
    assert resolved["ACHATS_REVENDUS"].value == Decimal("9295560.07")
    assert resolved["ACHATS_CONSOMMES"].value == Decimal("430367.29")
    assert resolved["ACHATS_TOTAL"].value == Decimal("9725927.36")
    assert resolved["CHARGES_INTERETS"].value == Decimal("95394.47")
    assert resolved["PRODUITS_FINANCIERS"].value == Decimal("7082.15")
    assert resolved["CHARGES_FINANCIERES"].value == Decimal("200928.82")
    assert resolved["RESULTAT_FINANCIER"].value == Decimal("-193846.67")
    assert resolved["RESULTAT_FINANCIER"].status == "derived"
    assert resolved["RESULTAT_COURANT"].value == Decimal("1187736.60")
    assert resolved["PRODUITS_NON_COURANTS"].value == Decimal("64027.85")
    assert resolved["CHARGES_NON_COURANTES"].value == Decimal("71955.29")
    assert resolved["RESULTAT_NON_COURANT"].value == Decimal("-7927.44")
    assert resolved["RESULTAT_AVANT_IMPOT"].value == Decimal("1179809.16")
    assert resolved["IMPOT_SUR_RESULTATS"].value == Decimal("0.00")
    assert resolved["REDEVANCES_CREDIT_BAIL"].value == Decimal("21729.13")
    assert resolved["ENCOURS_LEASING"].value is None
    assert resolved["REDEVANCES_CREDIT_BAIL"].value != resolved["ENCOURS_LEASING"].value

    # Provenance conservée
    assert resolved["CHIFFRE_AFFAIRES"].provenance[0].raw_value == "13 404 177,00"
    assert resolved["CHIFFRE_AFFAIRES"].provenance[0].mapping_model == "qwen3:8b"


def test_serdilab_python_ratios_not_llm():
    resolved = resolve_financial_candidates(serdilab_mapping_outputs())
    dataset = build_financial_dataset_from_resolved_values(resolved)

    autonomy = calculate_financial_autonomy(dataset)
    assert autonomy.value is not None
    assert abs(autonomy.value - Decimal("40.87")) < Decimal("0.05")

    commercial = calculate_commercial_profitability(dataset)
    assert commercial.value is not None
    assert abs(commercial.value - Decimal("8.80")) < Decimal("0.05")

    assert dataset.tresorerie_nette.value == Decimal("201167.82")
    assert dataset.achats.value == Decimal("9725927.36")

    ratios = calculate_financial_ratios(dataset)
    assert all(r.value is None or isinstance(r.value, Decimal) for r in ratios)
    assert not any(isinstance(r.value, float) for r in ratios)


def test_pipeline_llm_strategy_with_mocked_qwen():
    extraction = PdfContentExtractionResult(
        source_filename="serdilab.pdf",
        pages_total=1,
        pages_processed=1,
        pages_ok=1,
        pages_failed=0,
        model="glm",
        ollama_url="http://localhost:11434",
        processing_time_ms=1,
        pages=[
            PdfPageExtraction(
                page_number=1,
                status="ok",
                content="# Compte de produits et charges\n| x | 1 |",
                char_count=40,
            )
        ],
    )
    batch = FinancialMappingBatchResult(
        model="qwen3:8b",
        mapped_sections=serdilab_mapping_outputs(),
        warnings=[],
    )

    with patch(
        "app.services.analysis_pipeline.map_financial_sections",
        new=AsyncMock(return_value=batch),
    ):
        result = asyncio.run(
            analyze_extracted_pdf(
                extraction,
                scoring_mode="REVIEW",
            )
        )

    assert result.mapping is not None
    assert result.mapping.strategy == "qwen_only"
    assert result.mapping.model == "qwen3:8b"
    assert result.dataset.chiffre_affaires.value == Decimal("13404177.00")
    assert result.dataset.encours_leasing.value is None
    assert result.dataset.redevances_credit_bail is not None
    assert result.dataset.redevances_credit_bail.value == Decimal("21729.13")
    assert all(
        prov.extraction_method == "qwen_mapping"
        for prov in result.dataset.chiffre_affaires.provenance
    )


def test_bilan_conflict_blocks_strict_score():
    bad = [
        FinancialMappingOutput(
            section="BILAN_ACTIF",
            candidates=[
                _c(
                    "TOTAL_ACTIF",
                    "100,00",
                    section="BILAN_ACTIF",
                    label="Total général",
                    column="Net",
                    nature="GRAND_TOTAL",
                )
            ],
        ),
        FinancialMappingOutput(
            section="BILAN_PASSIF",
            candidates=[
                _c(
                    "TOTAL_PASSIF",
                    "200,00",
                    section="BILAN_PASSIF",
                    label="Total général",
                    column="Exercice",
                    nature="GRAND_TOTAL",
                )
            ],
        ),
    ]
    extraction = PdfContentExtractionResult(
        source_filename="bad.pdf",
        pages_total=1,
        pages_processed=1,
        pages_ok=1,
        pages_failed=0,
        model="glm",
        ollama_url="http://localhost:11434",
        processing_time_ms=1,
        pages=[
            PdfPageExtraction(
                page_number=1,
                status="ok",
                content="# BILAN - ACTIF\n| x | 1 |",
                char_count=20,
            )
        ],
    )
    batch = FinancialMappingBatchResult(
        model="qwen3:8b",
        mapped_sections=bad,
        warnings=[],
    )
    with patch(
        "app.services.analysis_pipeline.map_financial_sections",
        new=AsyncMock(return_value=batch),
    ):
        result = asyncio.run(
            analyze_extracted_pdf(
                extraction,
                scoring_mode="STRICT",
            )
        )
    assert result.dataset.total_bilan.status == "conflicting" or any(
        c.status == "failed" and c.code == "bilan_equilibre"
        for c in result.accounting_checks
    )


def test_pdf_analyze_endpoint_has_no_mapping_strategy_form():
    schema = app.openapi()["paths"]["/api/v1/extraction/pdf/analyze"]["post"]
    assert "mapping_strategy" not in str(schema)
