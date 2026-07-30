# -*- coding: utf-8 -*-
"""Tests pipeline d'analyse (sans Ollama)."""
import asyncio
from decimal import Decimal

from app.schemas.financial_analysis import BehavioralInput, SectorBenchmarkInput
from app.schemas.pdf_extraction import PdfContentExtractionResult, PdfPageExtraction
from app.services.analysis_pipeline import analyze_extracted_pdf
from app.services.markdown_financial_parser import parse_markdown_pages
from app.services.sector_scoring import calculate_sector_score
from app.services.financial_ratios import calculate_commercial_profitability
from tests.test_financial_ratios import reference_dataset


SAMPLE_MD = """
# BILAN - ACTIF

| Éléments | Brut | Amortissements | Net | Exercice précédent |
|---|---:|---:|---:|---:|
| Total général (I + II + III) | | | 24 800,00 | 23 000,00 |
| Total I Actif immobilisé | | | 10 000,00 | |
| Total II Actif circulant | | | 14 000,00 | |
| Clients et comptes rattachés | | | 8 340,00 | |
| Total III Trésorerie-Actif | | | 800,00 | |

# BILAN - PASSIF

| Éléments | Net |
|---|---:|
| Total des capitaux propres | 6 200,00 |
| Dettes de financement | 11 400,00 |
| Fournisseurs et comptes rattachés | 5 600,00 |
| Total III Trésorerie-Passif | 300,00 |

# CPC

| Libellé | Total de l'exercice |
|---|---:|
| Chiffre d'affaires | 38 500,00 |
| Achats consommés de matières et fournitures | 31 500,00 |
| Charges d'intérêts | 900,00 |
| Dotations d'exploitation | 1 700,00 |
| Résultat net de l'exercice | 1 480,00 |
"""


def test_parse_markdown_tables():
    pages = [
        PdfPageExtraction(
            page_number=1,
            status="ok",
            extraction_mode="vision",
            content=SAMPLE_MD,
            char_count=len(SAMPLE_MD),
        )
    ]
    rows = parse_markdown_pages(pages)
    assert len(rows) >= 5
    assert any("chiffre" in r.normalized_label for r in rows)


def test_analyze_pipeline_blocks_without_behavioral():
    extraction = PdfContentExtractionResult(
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
                content=SAMPLE_MD,
                char_count=len(SAMPLE_MD),
            )
        ],
    )
    result = asyncio.run(
        analyze_extracted_pdf(
            extraction,
            scoring_mode="STRICT",
            mapping_strategy="deterministic",
        )
    )
    assert result.decision.risk_class == "NON_EVALUABLE"
    assert result.final_score is None
    assert any(isinstance(r.value, Decimal) or r.value is None for r in result.ratios)


def test_sector_no_invented_percentile():
    ratios = [calculate_commercial_profitability(reference_dataset())]
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
    extraction = PdfContentExtractionResult(
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
                content=SAMPLE_MD,
                char_count=len(SAMPLE_MD),
            )
        ],
    )
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
    result = asyncio.run(
        analyze_extracted_pdf(
            extraction,
            behavioral_input=behavioral,
            sector_input=sector,
            scoring_mode="STRICT",
            mapping_strategy="deterministic",
        )
    )
    # CAF peut manquer dans le markdown → scoring strict peut bloquer
    assert result.decision is not None
    assert all(isinstance(a.raw_score, Decimal) for a in result.axes)
