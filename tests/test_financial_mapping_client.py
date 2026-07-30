# -*- coding: utf-8 -*-
"""Tests client mapping Qwen (sans appeler Ollama réellement)."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.config import OLLAMA_MAPPING_MODEL, OLLAMA_URL
from app.schemas.financial_mapping import (
    FinancialMappingOutput,
    FinancialSectionInput,
)
from app.services.financial_mapping_client import (
    FinancialMappingError,
    map_financial_section,
    map_financial_sections,
)


def _ok_mapping_json(section: str = "CPC") -> str:
    return json.dumps(
        {
            "section": section,
            "candidates": [
                {
                    "field_code": "CHIFFRE_AFFAIRES",
                    "raw_value": "13 404 177,00",
                    "period": "N",
                    "nature": "DETAIL",
                    "confidence": 0.9,
                    "evidence": {
                        "page_number": 1,
                        "section": section,
                        "raw_label": "Chiffre d'affaires",
                        "column_name": "Totaux de l'exercice",
                        "column_role": "TOTAL_EXERCICE_N",
                        "source_excerpt": "| Chiffre d'affaires | 13 404 177,00 |",
                    },
                    "warnings": [],
                }
            ],
            "unresolved_labels": [],
            "document_warnings": [],
        }
    )


def _make_response(content: str, status_code: int = 200) -> httpx.Response:
    return httpx.Response(
        status_code,
        json={
            "message": {"content": content},
            "done": True,
            "done_reason": "stop",
            "eval_count": 10,
            "prompt_eval_count": 20,
        },
        request=httpx.Request("POST", f"{OLLAMA_URL}/api/chat"),
    )


def test_payload_uses_qwen_model_same_url_no_image_schema_format():
    captured: dict = {}

    async def post(url, json):
        captured["url"] = url
        captured["payload"] = json
        return _make_response(_ok_mapping_json("CPC"))

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    section = FinancialSectionInput(
        section="CPC",
        page_number=1,
        markdown="# Compte de produits et charges\n| CA | 1 |",
    )

    with patch(
        "app.services.financial_mapping_client.httpx.AsyncClient",
        return_value=mock_client,
    ):
        mapped, elapsed = asyncio.run(
            map_financial_section(section, max_attempts=1)
        )

    assert mapped.section == "CPC"
    assert elapsed >= 0
    assert captured["url"] == f"{OLLAMA_URL}/api/chat"
    payload = captured["payload"]
    assert payload["model"] == OLLAMA_MAPPING_MODEL
    assert payload["model"] == "qwen3:8b"
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0
    assert payload["think"] is False
    assert "/no_think" in payload["messages"][1]["content"]
    assert payload["format"] != "json"
    assert isinstance(payload["format"], dict)
    assert (
        "properties" in payload["format"]
        or "$defs" in payload["format"]
        or "title" in payload["format"]
    )
    for message in payload["messages"]:
        assert isinstance(message["content"], str)
        assert "images" not in message
    assert "images" not in payload


def test_invalid_json_triggers_retry():
    calls = {"n": 0}

    async def post(url, json):
        calls["n"] += 1
        if calls["n"] == 1:
            return _make_response("not-json")
        return _make_response(_ok_mapping_json("CPC"))

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    section = FinancialSectionInput(section="CPC", page_number=1, markdown="x")
    with patch(
        "app.services.financial_mapping_client.httpx.AsyncClient",
        return_value=mock_client,
    ):
        mapped, _ = asyncio.run(map_financial_section(section, max_attempts=3))

    assert mapped.section == "CPC"
    assert calls["n"] == 2


def test_empty_response_triggers_retry_then_fails():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_make_response(""))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    section = FinancialSectionInput(section="CPC", page_number=1, markdown="x")
    with patch(
        "app.services.financial_mapping_client.httpx.AsyncClient",
        return_value=mock_client,
    ):
        with pytest.raises(FinancialMappingError):
            asyncio.run(map_financial_section(section, max_attempts=2))

    assert mock_client.post.await_count == 2


def test_wrong_section_rejected():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(
        return_value=_make_response(_ok_mapping_json("BILAN_ACTIF"))
    )
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    section = FinancialSectionInput(section="CPC", page_number=1, markdown="x")
    with patch(
        "app.services.financial_mapping_client.httpx.AsyncClient",
        return_value=mock_client,
    ):
        with pytest.raises(FinancialMappingError, match="JSON Schema"):
            asyncio.run(map_financial_section(section, max_attempts=1))


def test_autre_and_identification_skipped():
    sections = [
        FinancialSectionInput(section="AUTRE", page_number=1, markdown="garde"),
        FinancialSectionInput(
            section="IDENTIFICATION",
            page_number=2,
            markdown="Identification du contribuable",
        ),
    ]
    with patch(
        "app.services.financial_mapping_client.map_financial_section",
        new_callable=AsyncMock,
    ) as mocked:
        result = asyncio.run(map_financial_sections(sections))
    mocked.assert_not_called()
    assert result.mapped_sections == []
    assert result.model == OLLAMA_MAPPING_MODEL
    assert result.skipped_count == 2
    assert result.processed_count == 0
    assert result.failed_count == 0


def test_valid_output_validated_by_pydantic():
    raw = _ok_mapping_json("CPC")
    mapped = FinancialMappingOutput.model_validate_json(raw)
    assert mapped.candidates[0].field_code == "CHIFFRE_AFFAIRES"
    assert mapped.candidates[0].raw_value == "13 404 177,00"
