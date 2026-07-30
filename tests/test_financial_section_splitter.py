# -*- coding: utf-8 -*-
"""Tests segmentation déterministe des sections financières."""
from app.schemas.pdf_extraction import PdfContentExtractionResult, PdfPageExtraction
from app.services.financial_section_splitter import (
    detect_financial_section,
    split_financial_sections,
)


def test_detect_bilan_actif():
    assert detect_financial_section("# BILAN - ACTIF\n| A | 1 |") == "BILAN_ACTIF"


def test_detect_bilan_passif():
    assert detect_financial_section("Bilan — Passif") == "BILAN_PASSIF"


def test_detect_cpc():
    assert (
        detect_financial_section("Compte de produits et charges\n| CA | 1 |") == "CPC"
    )


def test_detect_detail_cpc():
    assert (
        detect_financial_section("Détail des postes du C.P.C.") == "DETAIL_CPC"
    )


def test_detect_autre():
    assert detect_financial_section("Page de garde") == "AUTRE"


def test_skip_empty_and_page_vide():
    extraction = PdfContentExtractionResult(
        source_filename="t.pdf",
        pages_total=3,
        pages_processed=3,
        pages_ok=3,
        pages_failed=0,
        model="x",
        ollama_url="http://localhost:11434",
        processing_time_ms=1,
        pages=[
            PdfPageExtraction(
                page_number=1,
                status="ok",
                content="[PAGE VIDE]",
                char_count=11,
            ),
            PdfPageExtraction(
                page_number=2,
                status="ok",
                content="",
                char_count=0,
            ),
            PdfPageExtraction(
                page_number=3,
                status="ok",
                content="# BILAN - ACTIF\n| Total | 1 |",
                char_count=30,
            ),
            PdfPageExtraction(
                page_number=4,
                status="error",
                content="# CPC",
                char_count=5,
            ),
        ],
    )
    sections = split_financial_sections(extraction)
    assert len(sections) == 1
    assert sections[0].section == "BILAN_ACTIF"
    assert sections[0].page_number == 3
