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


def test_implicit_bilan_actif_without_title():
    from app.services.financial_section_splitter import infer_implicit_section

    markdown = """
| Éléments | Brut | Amort | Net | Exercice précédent |
|---|---:|---:|---:|---:|
| IMMOBILISATION EN NON VALEUR | 1 | 0 | 1 | 1 |
| IMMOBILISATIONS INCORPORELLES | 2 | 0 | 2 | 2 |
| IMMOBILISATIONS CORPORELLES | 3 | 0 | 3 | 3 |
| STOCKS (f) | 4 | 0 | 4 | 4 |
| CREANCES DE L'ACTIF CIRCULANT | 5 | 0 | 5 | 5 |
| TRESORERIE - ACTIF | 6 | 0 | 6 | 6 |
| TOTAL GENERAL I+II+III | 10 | 0 | 10 | 10 |
"""
    assert infer_implicit_section(markdown) == "BILAN_ACTIF"

    extraction = PdfContentExtractionResult(
        source_filename="serdilab.pdf",
        pages_total=1,
        pages_processed=1,
        pages_ok=1,
        pages_failed=0,
        model="x",
        ollama_url="http://localhost:11434",
        processing_time_ms=1,
        pages=[
            PdfPageExtraction(
                page_number=2,
                status="ok",
                content=markdown,
                char_count=len(markdown),
            )
        ],
    )
    sections = split_financial_sections(extraction)
    assert any(s.section == "BILAN_ACTIF" for s in sections)


def test_split_identification_then_bilan_actif():
    markdown = """
Pièces annexes à la déclaration
Identification du contribuable
Raison sociale : SERDILAB SARL
ICE : 001234567000012

IMMOBILISATION EN NON VALEUR | 100 | 0 | 100 |
IMMOBILISATIONS INCORPORELLES | 200 | 0 | 200 |
IMMOBILISATIONS CORPORELLES | 300 | 0 | 300 |
STOCKS (f) | 400 | 0 | 400 |
CREANCES DE L'ACTIF CIRCULANT | 500 | 0 | 500 |
TRESORERIE - ACTIF | 600 | 0 | 600 |
"""
    extraction = PdfContentExtractionResult(
        source_filename="serdilab.pdf",
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
                content=markdown,
                char_count=len(markdown),
            )
        ],
    )
    sections = split_financial_sections(extraction)
    kinds = [s.section for s in sections]
    assert "IDENTIFICATION" in kinds
    assert "BILAN_ACTIF" in kinds
    assert kinds.index("IDENTIFICATION") < kinds.index("BILAN_ACTIF")
