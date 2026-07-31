# -*- coding: utf-8 -*-
"""Tests client GLM direct (mock httpx, sans Ollama)."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic import ValidationError

from app.config import DIRECT_FINANCIAL_MODEL, OLLAMA_URL
from app.schemas.direct_financial_extraction import (
    BilanActifOutput,
    CpcOutput,
    DetailCpcOutput,
)
from app.services.direct_glm_financial_client import (
    DirectFinancialLengthError,
    extract_financial_page,
    schema_for_page_type,
)


def _ok_actif_json() -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "field_code": "TOTAL_ACTIF",
                    "raw_value": "22 303 497,11",
                    "period": "N",
                    "nature": "GRAND_TOTAL",
                    "confidence": 0.9,
                    "evidence": {
                        "raw_label": "TOTAL GENERAL I+II+III",
                        "column_name": "Net",
                        "column_role": "NET_N",
                        "source_excerpt": "TOTAL GENERAL | 22 303 497,11",
                    },
                    "warnings": [],
                }
            ],
        }
    )


def _response(content: str, *, done_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"content": content},
            "done": True,
            "done_reason": done_reason,
            "eval_count": 100,
            "prompt_eval_count": 200,
        },
        request=httpx.Request("POST", f"{OLLAMA_URL}/api/chat"),
    )


def test_schema_bilan_actif_forbids_impot():
    with pytest.raises(ValidationError):
        BilanActifOutput.model_validate(
            {
                "page_type": "BILAN_ACTIF",
                "candidates": [
                    {
                        "field_code": "IMPOT_SUR_RESULTATS",
                        "raw_value": "1",
                        "period": "N",
                        "nature": "DETAIL",
                        "confidence": 0.5,
                        "evidence": {
                            "page_number": 1,
                            "page_type": "BILAN_ACTIF",
                            "raw_label": "IS",
                            "column_role": "NET_N",
                            "source_excerpt": "IS|1",
                        },
                    }
                ],
            }
        )


def test_schema_cpc_forbids_total_passif():
    with pytest.raises(ValidationError):
        CpcOutput.model_validate(
            {
                "page_type": "CPC",
                "candidates": [
                    {
                        "field_code": "TOTAL_PASSIF",
                        "raw_value": "1",
                        "period": "N",
                        "nature": "GRAND_TOTAL",
                        "confidence": 0.5,
                        "evidence": {
                            "page_number": 1,
                            "page_type": "CPC",
                            "raw_label": "Total",
                            "column_role": "TOTAL_EXERCICE_N",
                            "source_excerpt": "Total|1",
                        },
                    }
                ],
            }
        )


def test_detail_cpc_forbids_charges_financieres():
    with pytest.raises(ValidationError):
        DetailCpcOutput.model_validate(
            {
                "page_type": "DETAIL_CPC",
                "candidates": [
                    {
                        "field_code": "CHARGES_FINANCIERES",
                        "raw_value": "1",
                        "period": "N",
                        "nature": "DETAIL",
                        "confidence": 0.5,
                        "evidence": {
                            "page_number": 1,
                            "page_type": "DETAIL_CPC",
                            "raw_label": "Charges",
                            "column_role": "EXERCICE_N",
                            "source_excerpt": "Charges|1",
                        },
                    }
                ],
            }
        )


def test_extract_financial_page_payload():
    captured = {}

    async def post(url, json):
        captured["url"] = url
        captured["payload"] = json
        return _response(_ok_actif_json())

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "app.services.direct_glm_financial_client.httpx.AsyncClient",
        return_value=mock_client,
    ):
        result, latency = asyncio.run(
            extract_financial_page(
                b"fake-image",
                page_number=2,
                page_type="BILAN_ACTIF",
                orientation=0,
                max_attempts=1,
            )
        )

    assert latency >= 0
    assert len(result.candidates) == 1
    assert result.candidates[0].field_code == "TOTAL_ACTIF"
    assert captured["url"] == f"{OLLAMA_URL}/api/chat"
    payload = captured["payload"]
    assert payload["model"] == DIRECT_FINANCIAL_MODEL
    assert payload["stream"] is False
    assert "think" not in payload  # aligné sur vision_client CIN/ICE
    assert payload["options"]["temperature"] == 0
    assert "images" in payload["messages"][1]
    assert isinstance(payload["format"], dict)
    # Schéma lite : pas de page_number imbriqué obligatoire
    schema_txt = json.dumps(payload["format"])
    assert "page_number" not in schema_txt or "GlmLite" in str(type(result))
    # Aucun Qwen
    assert "qwen" not in payload["model"].lower()


def test_length_raises_without_identical_retry_loop():
    calls = {"n": 0}

    async def post(url, json):
        calls["n"] += 1
        return _response("{", done_reason="length")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "app.services.direct_glm_financial_client.httpx.AsyncClient",
        return_value=mock_client,
    ):
        with pytest.raises(DirectFinancialLengthError):
            asyncio.run(
                extract_financial_page(
                    b"img",
                    page_number=1,
                    page_type="CPC",
                    orientation=0,
                    max_attempts=3,
                )
            )
    assert calls["n"] == 1


def test_raw_value_required():
    with pytest.raises(ValidationError):
        BilanActifOutput.model_validate(
            {
                "page_type": "BILAN_ACTIF",
                "candidates": [
                    {
                        "field_code": "STOCKS",
                        "raw_value": None,
                        "period": "N",
                        "nature": "DETAIL",
                        "confidence": 0.5,
                        "evidence": {
                            "page_number": 1,
                            "page_type": "BILAN_ACTIF",
                            "raw_label": "Stocks",
                            "column_role": "NET_N",
                            "source_excerpt": "Stocks",
                        },
                    }
                ],
            }
        )


def test_schema_for_page_type():
    assert schema_for_page_type("CPC") is CpcOutput


def test_lite_validate_tolerates_period_alias():
    from app.services.direct_glm_financial_client import _validate_lite_content

    raw = json.dumps(
        {
            "candidates": [
                {
                    "field_code": "CHIFFRE_AFFAIRES",
                    "raw_value": "1 234 567,89",
                    "period": "N-1",
                    "nature": "DETAIL",
                    "evidence": {
                        "raw_label": "Chiffre d'affaires",
                        "column_role": "4",
                    },
                }
            ]
        }
    )
    out = _validate_lite_content(raw, "CPC")
    assert len(out.candidates) == 1
    assert out.candidates[0].period == "N_MINUS_1"
    assert out.candidates[0].evidence.column_role == "EXERCICE_N1"
