# -*- coding: utf-8 -*-
"""Tests documentaires pipeline GLM direct (mock extraction, sans Ollama)."""
from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import fitz

from app.schemas.direct_financial_extraction import (
    BilanActifOutput,
    BilanPassifOutput,
    CpcOutput,
    DetailCpcOutput,
    DirectFinancialCandidate,
    DirectFinancialEvidence,
)
from app.services.direct_financial_extraction_pipeline import (
    analyze_financial_document,
    count_pdf_pages,
    render_pdf_pages_png,
    validate_pdf,
)
from app.services.financial_page_classifier import classify_from_text
from tests.test_direct_financial_resolver import serdilab_candidates


def _pdf_with_texts(texts: list[str]) -> bytes:
    doc = fitz.open()
    for text in texts:
        page = doc.new_page()
        page.insert_text((50, 72), text[:1200])
    data = doc.tobytes()
    doc.close()
    return data


def test_validate_and_count_pages():
    pdf = _pdf_with_texts(["A"] * 7)
    validate_pdf(pdf)
    assert count_pdf_pages(pdf) == 7


def test_render_pages_png():
    pdf = _pdf_with_texts(["Bilan Actif"] * 3)
    pages = render_pdf_pages_png(pdf, dpi=72, max_pages=3)
    assert len(pages) == 3
    assert pages[0].image_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_long_document_skips_autre_pages():
    texts = (
        ["Formulaire administratif fiscal"] * 5
        + ["Bilan Actif Immobilisations corporelles Trésorerie actif"]
        + ["Bilan Passif Capitaux propres Dettes de financement"]
        + ["Compte de produits et charges Produits d'exploitation"]
        + ["Annexe ESG sans intérêt"] * 20
    )
    assert len(texts) >= 28
    # Classification sans GLM
    types = [classify_from_text(t) or "AUTRE" for t in texts]
    financial = [t for t in types if t in {"BILAN_ACTIF", "BILAN_PASSIF", "CPC"}]
    ignored = [t for t in types if t in {"AUTRE", "VIDE", None}]
    assert len(financial) >= 3
    assert len(ignored) >= 20


def test_multi_page_actif_and_cpc_merge_via_resolver():
    # Deux pages actif + deux pages CPC → candidats fusionnés
    cands = [
        c
        for c in serdilab_candidates()
        if c.evidence.page_type in {"BILAN_ACTIF", "BILAN_PASSIF", "CPC", "DETAIL_CPC"}
    ]
    # Simule répartition multi-pages
    for c in cands:
        if c.field_code == "ACTIF_CIRCULANT":
            c.evidence.page_number = 3
        if c.field_code.startswith("RESULTAT"):
            c.evidence.page_number = 6

    from app.services.direct_financial_resolver import resolve_direct_financial_candidates

    resolved = resolve_direct_financial_candidates(cands)
    assert resolved["TOTAL_ACTIF"].value == Decimal("22303497.11")
    assert resolved["CHIFFRE_AFFAIRES"].value == Decimal("13404177.00")
    assert resolved["RESULTAT_NET"].value == Decimal("1179809.16")


def test_analyze_pipeline_mocked_glm_no_qwen():
    pdf = _pdf_with_texts(
        [
            "Identification du contribuable Raison sociale SERDILAB",
            "Bilan Actif Immobilisations corporelles Trésorerie actif Total général I+II+III",
            "Bilan Passif Capitaux propres Dettes de financement Passif circulant",
            "Compte de produits et charges Chiffre d'affaires Produits d'exploitation",
            "Page blanche",
        ]
    )

    async def fake_extract(image_bytes, **kwargs):
        page_type = kwargs["page_type"]
        page_number = kwargs["page_number"]
        if page_type == "BILAN_ACTIF":
            mapped = BilanActifOutput.model_validate(
                {
                    "page_type": "BILAN_ACTIF",
                    "candidates": [
                        {
                            "field_code": "TOTAL_ACTIF",
                            "raw_value": "22 303 497,11",
                            "period": "N",
                            "nature": "GRAND_TOTAL",
                            "confidence": 0.9,
                            "evidence": {
                                "page_number": page_number,
                                "page_type": "BILAN_ACTIF",
                                "raw_label": "TOTAL GENERAL I+II+III",
                                "column_name": "Net",
                                "column_role": "NET_N",
                                "source_excerpt": "TOTAL|22303497",
                                "orientation": 0,
                            },
                        }
                    ],
                }
            )
            return mapped, 10
        if page_type == "BILAN_PASSIF":
            return (
                BilanPassifOutput.model_validate(
                    {
                        "page_type": "BILAN_PASSIF",
                        "candidates": [
                            {
                                "field_code": "TOTAL_PASSIF",
                                "raw_value": "22 303 497,11",
                                "period": "N",
                                "nature": "GRAND_TOTAL",
                                "confidence": 0.9,
                                "evidence": {
                                    "page_number": page_number,
                                    "page_type": "BILAN_PASSIF",
                                    "raw_label": "TOTAL I+II+III",
                                    "column_name": "Exercice",
                                    "column_role": "EXERCICE_N",
                                    "source_excerpt": "TOTAL|22303497",
                                    "orientation": 0,
                                },
                            },
                            {
                                "field_code": "FONDS_PROPRES",
                                "raw_value": "9 114 715,17",
                                "period": "N",
                                "nature": "SECTION_TOTAL",
                                "confidence": 0.9,
                                "evidence": {
                                    "page_number": page_number,
                                    "page_type": "BILAN_PASSIF",
                                    "raw_label": "TOTAL DES CAPITAUX PROPRES",
                                    "column_name": "Exercice",
                                    "column_role": "EXERCICE_N",
                                    "source_excerpt": "FP|9114715",
                                    "orientation": 0,
                                },
                            },
                        ],
                    }
                ),
                12,
            )
        if page_type == "CPC":
            return (
                CpcOutput.model_validate(
                    {
                        "page_type": "CPC",
                        "candidates": [
                            {
                                "field_code": "CHIFFRE_AFFAIRES",
                                "raw_value": "13 404 177,00",
                                "period": "N",
                                "nature": "DETAIL",
                                "confidence": 0.9,
                                "evidence": {
                                    "page_number": page_number,
                                    "page_type": "CPC",
                                    "raw_label": "Chiffre d'affaires",
                                    "column_name": "Totaux de l'exercice",
                                    "column_role": "TOTAL_EXERCICE_N",
                                    "source_excerpt": "CA|13404177",
                                    "orientation": 0,
                                },
                            }
                        ],
                    }
                ),
                11,
            )
        if page_type == "IDENTIFICATION":
            from app.schemas.direct_financial_extraction import IdentificationOutput

            return (
                IdentificationOutput.model_validate(
                    {
                        "page_type": "IDENTIFICATION",
                        "candidates": [
                            {
                                "field_code": "RAISON_SOCIALE",
                                "raw_value": "SERDILAB",
                                "period": "N",
                                "nature": "DETAIL",
                                "confidence": 0.8,
                                "evidence": {
                                    "page_number": page_number,
                                    "page_type": "IDENTIFICATION",
                                    "raw_label": "Raison sociale",
                                    "column_name": None,
                                    "column_role": "IDENTITY_VALUE",
                                    "source_excerpt": "SERDILAB",
                                    "orientation": 0,
                                },
                            }
                        ],
                    }
                ),
                5,
            )
        raise AssertionError(f"Type inattendu {page_type}")

    with patch(
        "app.services.direct_financial_extraction_pipeline.extract_financial_page",
        new=AsyncMock(side_effect=fake_extract),
    ):
        result = asyncio.run(
            analyze_financial_document(pdf, "mock.pdf", max_pages=5)
        )

    assert result.document.pages_total == 5
    assert result.dataset.total_actif is not None
    assert result.dataset.total_actif.value == Decimal("22303497.11")
    assert result.dataset.chiffre_affaires.value == Decimal("13404177.00")
    assert result.document.company.raison_sociale == "SERDILAB"
    # Provenance GLM direct
    prov = result.dataset.total_actif.provenance[0]
    assert prov.extraction_method == "glm_direct_vision"
