# -*- coding: utf-8 -*-
"""Tests segmentation déterministe des sections financières."""
from app.schemas.pdf_extraction import PdfContentExtractionResult, PdfPageExtraction
from app.services.financial_section_splitter import split_financial_sections


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


def test_page_can_contain_passif_then_cpc():
    extraction = PdfContentExtractionResult(
        source_filename="t.pdf",
        pages_total=1,
        pages_processed=1,
        pages_ok=1,
        pages_failed=0,
        model="x",
        ollama_url="http://localhost:11434",
        processing_time_ms=1,
        pages=[
            PdfPageExtraction(
                page_number=1,
                status="ok",
                content=(
                    "# BILAN - PASSIF\n"
                    "| Éléments | Exercice |\n|---|---:|\n| Total | 1 |\n"
                    "# COMPTE DE PRODUITS ET CHARGES\n"
                    "| Libellé | Totaux de l'exercice |\n|---|---:|\n| Chiffre d'affaires | 2 |"
                ),
                char_count=180,
            )
        ],
    )
    sections = split_financial_sections(extraction)
    assert [section.section for section in sections] == ["BILAN_PASSIF", "CPC"]


def test_continuation_page_keeps_previous_section():
    extraction = PdfContentExtractionResult(
        source_filename="t.pdf",
        pages_total=2,
        pages_processed=2,
        pages_ok=2,
        pages_failed=0,
        model="x",
        ollama_url="http://localhost:11434",
        processing_time_ms=1,
        pages=[
            PdfPageExtraction(
                page_number=1,
                status="ok",
                content="# COMPTE DE PRODUITS ET CHARGES\n| Libellé | Totaux de l'exercice |\n|---|---:|\n| Chiffre d'affaires | 2 |",
                char_count=110,
            ),
            PdfPageExtraction(
                page_number=2,
                status="ok",
                content="| Résultat financier | 1 |\n| Résultat net | 1 |",
                char_count=60,
            ),
        ],
    )
    sections = split_financial_sections(extraction)
    assert sections[-1].section == "CPC"
