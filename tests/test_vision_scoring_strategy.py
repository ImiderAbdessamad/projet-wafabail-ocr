# -*- coding: utf-8 -*-
"""Tests de la stratégie Vision liasse/scoring (sans Ollama)."""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.vision_ocr import (
    VisionOcrError,
    _extract_single_page_with_fallbacks,
    _merge_region_payloads,
    _vision_payload_is_insufficient,
    extract_liasse_via_vision,
)


def _bilan_payload(n_rows: int = 5) -> dict:
    return {
        "page_type": "BILAN_ACTIF",
        "columns": ["Brut", "Net"],
        "rows": [
            {
                "label": f"Poste {i}",
                "values": {"Brut": f"{i}00", "Net": f"{i}00"},
            }
            for i in range(n_rows)
        ],
        "metadata": {},
    }


def test_vision_payload_insufficient_financial():
    assert _vision_payload_is_insufficient(None) is True
    assert _vision_payload_is_insufficient(_bilan_payload(1)) is True
    assert _vision_payload_is_insufficient(_bilan_payload(5)) is False


def test_vision_payload_identification():
    empty = {"page_type": "IDENTIFICATION", "rows": [], "metadata": {}}
    assert _vision_payload_is_insufficient(empty) is True
    ok = {
        "page_type": "IDENTIFICATION",
        "rows": [],
        "metadata": {"entreprise": "ACME"},
    }
    assert _vision_payload_is_insufficient(ok) is False


def test_merge_region_payloads_deduplicates():
    a = {
        "page_type": "BILAN_ACTIF",
        "columns": ["Net"],
        "rows": [{"label": "Stocks", "values": {"Net": "100"}}],
        "metadata": {},
    }
    b = {
        "page_type": "BILAN_ACTIF",
        "columns": ["Net"],
        "rows": [
            {"label": "Stocks", "values": {"Net": "100"}},
            {"label": "Clients", "values": {"Net": "50"}},
        ],
        "metadata": {},
    }
    merged = _merge_region_payloads([a, b])
    labels = [row["label"] for row in merged["rows"]]
    assert labels.count("Stocks") == 1
    assert "Clients" in labels
    assert merged["_extraction_strategy"] == "regions"


def test_rotation_only_90_in_fallbacks():
    source = inspect.getsource(_extract_single_page_with_fallbacks)
    assert "rotate_image_bytes(image_bytes, 90)" in source
    assert "180" not in source
    assert "270" not in source


def test_full_page_sufficient_skips_regions():
    vision = AsyncMock(return_value=_bilan_payload(6))
    regions = AsyncMock()

    with (
        patch("app.services.vision_ocr._vision_call", new=vision),
        patch("app.services.vision_ocr._extract_page_regions_json", new=regions),
    ):
        result, error = asyncio.run(
            _extract_single_page_with_fallbacks(b"img", 1, 1)
        )

    assert error is None
    assert result["_extraction_strategy"] == "full_page"
    regions.assert_not_awaited()


def test_insufficient_triggers_regions():
    vision = AsyncMock(return_value=_bilan_payload(1))
    regions = AsyncMock(return_value=_bilan_payload(6))

    with (
        patch("app.services.vision_ocr._vision_call", new=vision),
        patch("app.services.vision_ocr._extract_page_regions_json", new=regions),
    ):
        result, error = asyncio.run(
            _extract_single_page_with_fallbacks(b"img", 1, 1)
        )

    assert error is None
    regions.assert_awaited_once()
    assert result["_extraction_strategy"] == "regions"


def test_full_fail_then_regions_fail_then_rotation_90():
    vision = AsyncMock(
        side_effect=[
            VisionOcrError("full fail"),
            _bilan_payload(6),
        ]
    )
    regions = AsyncMock(side_effect=VisionOcrError("regions fail"))
    rotate = MagicMock(side_effect=lambda data, angle: data)

    with (
        patch("app.services.vision_ocr._vision_call", new=vision),
        patch("app.services.vision_ocr._extract_page_regions_json", new=regions),
        patch("app.services.vision_ocr.rotate_image_bytes", new=rotate),
    ):
        result, error = asyncio.run(
            _extract_single_page_with_fallbacks(b"img", 1, 1)
        )

    assert error is None
    rotate.assert_called_once()
    assert rotate.call_args.args[1] == 90
    assert result["_extraction_strategy"] == "rotation_90"
    assert result["orientation"] == 90


def test_extract_liasse_via_vision_warning_mentions_strategy():
    async def _run():
        with (
            patch(
                "app.services.vision_ocr._warmup_model",
                new=AsyncMock(return_value={"warmup": "ok"}),
            ),
            patch("app.services.vision_ocr.count_pdf_pages", return_value=1),
            patch(
                "app.services.vision_ocr.render_pdf_pages",
                return_value=[b"jpeg"],
            ),
            patch(
                "app.services.vision_ocr._process_all_pages",
                new=AsyncMock(
                    return_value=[
                        (0, {**_bilan_payload(6), "_extraction_strategy": "full_page"}, None)
                    ]
                ),
            ),
            patch(
                "app.services.vision_ocr._assemble_from_page_results",
                return_value=MagicMock(
                    warnings=[],
                ),
            ) as assemble,
        ):
            # assemble returns a MagicMock; we need a real-ish object with warnings list
            from app.schemas.liasse import LiasseExtractionResult

            assembled = LiasseExtractionResult(
                document_kind="LIASSE_OCR",
                warnings=["base"],
            )
            assemble.return_value = assembled
            result = await extract_liasse_via_vision(b"%PDF", "x.pdf")
            return result

    result = asyncio.run(_run())
    assert any("pleine page → régions" in w for w in result.warnings)
    assert any("strategies=" in w for w in result.warnings)
