# -*- coding: utf-8 -*-
"""Tests pipeline d'analyse Qwen-only (sans Ollama)."""
import asyncio
import inspect
from decimal import Decimal
from unittest.mock import AsyncMock, patch

from app.schemas.financial_analysis import BehavioralInput, SectorBenchmarkInput
from app.schemas.financial_mapping import FinancialMappingBatchResult, FinancialMappingOutput
from app.schemas.pdf_extraction import PdfContentExtractionResult, PdfPageExtraction
from app.services.analysis_pipeline import analyze_extracted_pdf
from app.services.financial_ratios import (
    calculate_commercial_profitability,
    calculate_debt_ratio,
    calculate_financial_autonomy,
)
from app.services.sector_scoring import calculate_sector_score
from tests.test_financial_mapping_integration import serdilab_mapping_outputs
from tests.test_financial_ratios import reference_dataset


def _sample_extraction() -> PdfContentExtractionResult:
    return PdfContentExtractionResult(
        source_filename="sample.pdf",
        pages_total=1,
        pages_processed=1,
        pages_ok=1,
        pages_failed=0,
        model="test",
        ollama_url="http://localhost",
        processing_time_ms=1,
        pages=[
            PdfPageExtraction(
                page_number=1,
                status="ok",
                content="# COMPTE DE PRODUITS ET CHARGES\n| x | 1 |",
                char_count=40,
            )
        ],
    )


def test_analysis_pipeline_no_longer_imports_deterministic_parser():
    source = inspect.getsource(__import__("app.services.analysis_pipeline", fromlist=["*"]))
    assert "parse_markdown_pages" not in source
    assert "build_financial_dataset(" not in source
    assert "merge_resolved_values" not in source


def test_analyze_pipeline_blocks_without_behavioral():
    extraction = _sample_extraction()
    batch = FinancialMappingBatchResult(
        model="qwen3:8b",
        mapped_sections=serdilab_mapping_outputs(),
        warnings=[],
    )
    with patch("app.services.analysis_pipeline.map_financial_sections", new=AsyncMock(return_value=batch)):
        result = asyncio.run(analyze_extracted_pdf(extraction, scoring_mode="STRICT"))
    assert result.decision.risk_class == "NON_EVALUABLE"
    assert result.final_score is None
    assert any(isinstance(r.value, Decimal) or r.value is None for r in result.ratios)


def test_sector_no_invented_percentile():
    dataset = reference_dataset()
    ratios = [
        calculate_commercial_profitability(dataset),
        calculate_financial_autonomy(dataset),
        calculate_debt_ratio(dataset),
    ]
    axis = calculate_sector_score(
        ratios,
        SectorBenchmarkInput(
            sector_name="Commerce",
            commercial_profitability_median=Decimal("4.00"),
            financial_autonomy_median=Decimal("20"),
            debt_ratio_median=Decimal("1.5"),
            repayment_capacity_median=Decimal("3"),
            ca_growth_median=Decimal("5"),
        ),
    )
    assert axis.calculable is True
    # Pas de percentile fabriqué dans le code sectoriel
    assert "percentile" not in str(axis.model_dump())


def test_full_axes_with_inputs_can_score():
    extraction = _sample_extraction()
    behavioral = BehavioralInput(
        ca_domiciliation_pct=Decimal("96"),
        debit_position_days=41,
        overdraft_usage_pct=Decimal("38"),
        bank_flows_vs_declared_ca_gap_pct=Decimal("-4.2"),
        leasing_payment_delays_24m=0,
        payment_incidents_24m=0,
        rejected_debits_24m=0,
        unpaid_bills_24m=0,
        bam_rating=3,
    )
    sector = SectorBenchmarkInput(
        sector_name="Services",
        commercial_profitability_median=Decimal("4"),
        financial_autonomy_median=Decimal("20"),
        debt_ratio_median=Decimal("1.5"),
        repayment_capacity_median=Decimal("3"),
        ca_growth_median=Decimal("5"),
    )
    batch = FinancialMappingBatchResult(
        model="qwen3:8b",
        mapped_sections=serdilab_mapping_outputs(),
        warnings=[],
    )
    with patch("app.services.analysis_pipeline.map_financial_sections", new=AsyncMock(return_value=batch)):
        result = asyncio.run(
            analyze_extracted_pdf(
                extraction,
                behavioral_input=behavioral,
                sector_input=sector,
                scoring_mode="STRICT",
            )
        )
    assert result.decision is not None
    assert all(isinstance(a.raw_score, Decimal) for a in result.axes)


def test_qwen_failure_leaves_fields_missing_without_fallback():
    extraction = _sample_extraction()
    batch = FinancialMappingBatchResult(
        model="qwen3:8b",
        mapped_sections=[FinancialMappingOutput(section="CPC", candidates=[])],
        warnings=["Mapping Qwen indisponible : page=1, section=CPC, erreur=timeout"],
    )
    with patch("app.services.analysis_pipeline.map_financial_sections", new=AsyncMock(return_value=batch)):
        result = asyncio.run(analyze_extracted_pdf(extraction, scoring_mode="STRICT"))
    assert result.dataset.chiffre_affaires.value is None
    assert result.dataset.chiffre_affaires.status == "missing"
    assert result.mapping.strategy == "qwen_only"
    assert "deterministic_parser" not in str(result.model_dump())
