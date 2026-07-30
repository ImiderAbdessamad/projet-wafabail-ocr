# -*- coding: utf-8 -*-
"""Tests prioritaires pipeline qwen_only (sans appel réel à Ollama)."""
from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

from app.config import OLLAMA_URL
from app.schemas.financial_mapping import (
    BilanActifMappingOutput,
    CpcMappingOutput,
    DetailCpcMappingOutput,
    FinancialCandidate,
    FinancialMappingOutput,
    FinancialSectionInput,
    MappingEvidence,
)
from app.services.financial_candidate_resolver import (
    candidate_is_eligible,
    canonicalize_column_role,
    is_total_general_candidate,
    resolve_financial_candidates,
)
from app.services.financial_mapping_client import (
    FinancialMappingLengthError,
    map_financial_sections,
    mapping_schema_for_section,
)
from app.services.financial_section_splitter import split_large_financial_section


def _cand(
    field_code: str,
    raw_value: str,
    *,
    section: str,
    label: str,
    column: str,
    period: str = "N",
    nature: str = "DETAIL",
    column_role: str = "UNKNOWN",
    page: int = 1,
    confidence: float = 0.9,
    excerpt: str | None = None,
) -> FinancialCandidate:
    return FinancialCandidate(
        field_code=field_code,
        raw_value=raw_value,
        period=period,  # type: ignore[arg-type]
        nature=nature,  # type: ignore[arg-type]
        confidence=confidence,
        evidence=MappingEvidence(
            page_number=page,
            section=section,  # type: ignore[arg-type]
            raw_label=label,
            column_name=column,
            column_role=column_role,  # type: ignore[arg-type]
            source_excerpt=excerpt or f"| {label} | {raw_value} |",
        ),
    )


def _passif_markdown() -> str:
    return """
# Bilan - Passif

## Capitaux propres
| Libellé | Exercice | Exercice précédent |
| TOTAL DES CAPITAUX PROPRES | 9 114 715,17 | 7 934 906,01 |

## Dettes de financement
| TOTAL DES DETTES DE FINANCEMENT | 133 308,11 | 150 000,00 |

## Passif circulant
| TOTAL DU PASSIF CIRCULANT | 13 055 473,83 | 12 000 000,00 |

## Trésorerie - Passif
| Trésorerie-Passif | 0,00 | 0,00 |

| TOTAL III | 0,00 | 0,00 |
| TOTAL I+II+III | 22 303 497,11 | 19 500 619,98 |
""".strip()


def _fragment_ok_json(section: str, candidates: list[dict]) -> str:
    return json.dumps(
        {
            "section": section,
            "candidates": candidates,
            "unresolved_labels": [],
            "document_warnings": [],
        }
    )


def _length_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"content": "{" * 100},
            "done": True,
            "done_reason": "length",
            "eval_count": 8192,
            "prompt_eval_count": 2000,
        },
        request=httpx.Request("POST", f"{OLLAMA_URL}/api/chat"),
    )


def _ok_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"content": content, "thinking": ""},
            "done": True,
            "done_reason": "stop",
            "eval_count": 100,
            "prompt_eval_count": 200,
        },
        request=httpx.Request("POST", f"{OLLAMA_URL}/api/chat"),
    )


def test_length_does_not_retry_same_call_then_splits_passif():
    """Test 1 : done_reason=length → pas de retry identique, découpage fragments."""
    markdown = _passif_markdown()
    fragments = split_large_financial_section(
        FinancialSectionInput(
            section="BILAN_PASSIF",
            page_number=3,
            markdown=markdown,
        )
    )
    assert len(fragments) >= 2

    calls: list[str] = []

    async def post(url, json):
        md = json["messages"][1]["content"]
        calls.append(md)
        # Premier appel (section complète) → length
        if "SOUS-SECTION" not in md:
            return _length_response()
        # Fragments → succès minimal
        return _ok_response(
            _fragment_ok_json(
                "BILAN_PASSIF",
                [
                    {
                        "field_code": "FONDS_PROPRES",
                        "raw_value": "9 114 715,17",
                        "period": "N",
                        "nature": "SECTION_TOTAL",
                        "confidence": 0.9,
                        "evidence": {
                            "page_number": 3,
                            "section": "BILAN_PASSIF",
                            "raw_label": "TOTAL DES CAPITAUX PROPRES",
                            "column_name": "Exercice",
                            "column_role": "EXERCICE_N",
                            "source_excerpt": "| TOTAL DES CAPITAUX PROPRES | 9 114 715,17 |",
                        },
                        "warnings": [],
                    }
                ],
            )
        )

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    section = FinancialSectionInput(
        section="BILAN_PASSIF",
        page_number=3,
        markdown=markdown,
    )

    with patch(
        "app.services.financial_mapping_client.httpx.AsyncClient",
        return_value=mock_client,
    ):
        result = asyncio.run(map_financial_sections([section]))

    # Un seul appel sans SOUS-SECTION (pas 3 retries identiques)
    full_calls = [c for c in calls if "SOUS-SECTION" not in c]
    assert len(full_calls) == 1
    # Puis des appels fragmentés
    frag_calls = [c for c in calls if "SOUS-SECTION" in c]
    assert len(frag_calls) >= 2
    assert result.processed_count == 1
    assert any(s.candidates for s in result.mapped_sections)


def test_bilan_actif_schema_forbids_impot():
    """Test 2."""
    schema = mapping_schema_for_section("BILAN_ACTIF")
    assert schema is BilanActifMappingOutput
    with pytest.raises(ValidationError):
        BilanActifMappingOutput.model_validate(
            {
                "section": "BILAN_ACTIF",
                "candidates": [
                    {
                        "field_code": "IMPOT_SUR_RESULTATS",
                        "raw_value": "100",
                        "period": "N",
                        "nature": "DETAIL",
                        "confidence": 0.5,
                        "evidence": {
                            "page_number": 1,
                            "section": "BILAN_ACTIF",
                            "raw_label": "IS",
                            "column_name": "Net",
                            "column_role": "NET_N",
                            "source_excerpt": "IS | 100",
                        },
                    }
                ],
            }
        )


def test_cpc_schema_forbids_total_passif():
    """Test 3."""
    with pytest.raises(ValidationError):
        CpcMappingOutput.model_validate(
            {
                "section": "CPC",
                "candidates": [
                    {
                        "field_code": "TOTAL_PASSIF",
                        "raw_value": "100",
                        "period": "N",
                        "nature": "GRAND_TOTAL",
                        "confidence": 0.5,
                        "evidence": {
                            "page_number": 1,
                            "section": "CPC",
                            "raw_label": "Total",
                            "column_name": "3 = 1 + 2",
                            "column_role": "TOTAL_EXERCICE_N",
                            "source_excerpt": "Total | 100",
                        },
                    }
                ],
            }
        )


def test_detail_cpc_schema_only_redevances():
    """Test 4."""
    with pytest.raises(ValidationError):
        DetailCpcMappingOutput.model_validate(
            {
                "section": "DETAIL_CPC",
                "candidates": [
                    {
                        "field_code": "CHARGES_FINANCIERES",
                        "raw_value": "100",
                        "period": "N",
                        "nature": "DETAIL",
                        "confidence": 0.5,
                        "evidence": {
                            "page_number": 1,
                            "section": "DETAIL_CPC",
                            "raw_label": "Charges",
                            "column_name": "Exercice",
                            "column_role": "EXERCICE_N",
                            "source_excerpt": "Charges | 100",
                        },
                    }
                ],
            }
        )
    ok = DetailCpcMappingOutput.model_validate(
        {
            "section": "DETAIL_CPC",
            "candidates": [
                {
                    "field_code": "REDEVANCES_CREDIT_BAIL",
                    "raw_value": "21 729,13",
                    "period": "N",
                    "nature": "DETAIL",
                    "confidence": 0.9,
                    "evidence": {
                        "page_number": 1,
                        "section": "DETAIL_CPC",
                        "raw_label": "Redevances de crédit-bail",
                        "column_name": "Exercice",
                        "column_role": "EXERCICE_N",
                        "source_excerpt": "Redevances | 21 729,13",
                    },
                }
            ],
        }
    )
    assert ok.candidates[0].field_code == "REDEVANCES_CREDIT_BAIL"


def test_source_excerpt_max_240_rejected():
    """Test 5 : source_excerpt > 240 refusé par le schéma."""
    with pytest.raises(ValidationError):
        MappingEvidence(
            page_number=1,
            section="CPC",
            raw_label="CA",
            column_name="Totaux",
            column_role="TOTAL_EXERCICE_N",
            source_excerpt="x" * 241,
        )


def test_taux_du_exercice_canonicalizes_to_total_exercice_n():
    """Test 6."""
    c = _cand(
        "CHIFFRE_AFFAIRES",
        "13.404.177,00",
        section="CPC",
        label="Chiffre d'affaires",
        column="Taux du exercice",
        column_role="UNKNOWN",
    )
    fixed = canonicalize_column_role(c)
    assert fixed.evidence.column_role == "TOTAL_EXERCICE_N"


def test_ca_taux_du_exercice_accepted():
    """Test 7."""
    c = _cand(
        "CHIFFRE_AFFAIRES",
        "13.404.177,00",
        section="CPC",
        label="Chiffre d'affaires",
        column="Taux du exercice",
        column_role="UNKNOWN",
    )
    fixed = canonicalize_column_role(c)
    ok, reasons = candidate_is_eligible(fixed)
    assert ok is True, reasons
    resolved = resolve_financial_candidates(
        [FinancialMappingOutput(section="CPC", candidates=[c])]
    )
    assert resolved["CHIFFRE_AFFAIRES"].value == Decimal("13404177.00")


def test_resultat_net_xiii_xvi_column_3_equals_accepted():
    """Test 8."""
    for code, label in (
        ("RESULTAT_NET_XIII", "XIII Résultat net"),
        ("RESULTAT_NET_XVI", "XVI Résultat net"),
    ):
        c = _cand(
            code,
            "1 179 809,16",
            section="CPC",
            label=label,
            column="3 = 1 + 2",
            column_role="UNKNOWN",
            nature="SECTION_TOTAL",
        )
        fixed = canonicalize_column_role(c)
        assert fixed.evidence.column_role == "TOTAL_EXERCICE_N"
        ok, reasons = candidate_is_eligible(fixed)
        assert ok is True, reasons


def test_total_general_variants_accepted():
    """Test 9."""
    for label in (
        "TOTAL GENERAL I+II+III",
        "TOTAL I+II+III",
        "TOTAL I II III",
    ):
        c = _cand(
            "TOTAL_ACTIF",
            "22 303 497,11",
            section="BILAN_ACTIF",
            label=label,
            column="Net",
            column_role="NET_N",
            nature="GRAND_TOTAL",
        )
        assert is_total_general_candidate(c) is True, label
        assert candidate_is_eligible(c)[0] is True, label


def test_total_i_rejected_as_total_actif():
    """Test 10."""
    c = _cand(
        "TOTAL_ACTIF",
        "10 000,00",
        section="BILAN_ACTIF",
        label="TOTAL I",
        column="Net",
        column_role="NET_N",
        nature="GRAND_TOTAL",
    )
    assert is_total_general_candidate(c) is False
    assert candidate_is_eligible(c)[0] is False


def test_fragmented_passif_resolves_serdilab_amounts():
    """Test 11."""
    outputs = [
        FinancialMappingOutput(
            section="BILAN_PASSIF",
            candidates=[
                _cand(
                    "FONDS_PROPRES",
                    "9 114 715,17",
                    section="BILAN_PASSIF",
                    label="TOTAL DES CAPITAUX PROPRES",
                    column="Exercice",
                    column_role="EXERCICE_N",
                    nature="SECTION_TOTAL",
                ),
                _cand(
                    "DETTES_FINANCIERES",
                    "133 308,11",
                    section="BILAN_PASSIF",
                    label="TOTAL DES DETTES DE FINANCEMENT",
                    column="Exercice",
                    column_role="EXERCICE_N",
                    nature="SECTION_TOTAL",
                ),
                _cand(
                    "PASSIF_CIRCULANT",
                    "13 055 473,83",
                    section="BILAN_PASSIF",
                    label="TOTAL DU PASSIF CIRCULANT",
                    column="Exercice",
                    column_role="EXERCICE_N",
                    nature="SECTION_TOTAL",
                ),
                _cand(
                    "TRESORERIE_PASSIF",
                    "0,00",
                    section="BILAN_PASSIF",
                    label="Trésorerie-Passif",
                    column="Exercice",
                    column_role="EXERCICE_N",
                ),
                _cand(
                    "TOTAL_PASSIF",
                    "22 303 497,11",
                    section="BILAN_PASSIF",
                    label="TOTAL I+II+III",
                    column="Exercice",
                    column_role="EXERCICE_N",
                    nature="GRAND_TOTAL",
                ),
            ],
        )
    ]
    resolved = resolve_financial_candidates(outputs)
    assert resolved["FONDS_PROPRES"].value == Decimal("9114715.17")
    assert resolved["DETTES_FINANCIERES"].value == Decimal("133308.11")
    assert resolved["PASSIF_CIRCULANT"].value == Decimal("13055473.83")
    assert resolved["TRESORERIE_PASSIF"].value == Decimal("0.00")
    assert resolved["TOTAL_PASSIF"].value == Decimal("22303497.11")


def test_length_error_raised_without_identical_retries():
    """Complément Test 1 : map_financial_section ne retry pas sur length."""
    calls = {"n": 0}

    async def post(url, json):
        calls["n"] += 1
        return _length_response()

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    from app.services.financial_mapping_client import map_financial_section

    section = FinancialSectionInput(
        section="CPC",
        page_number=1,
        markdown="court",
    )
    with patch(
        "app.services.financial_mapping_client.httpx.AsyncClient",
        return_value=mock_client,
    ):
        with pytest.raises(FinancialMappingLengthError):
            asyncio.run(map_financial_section(section, max_attempts=3))

    assert calls["n"] == 1
