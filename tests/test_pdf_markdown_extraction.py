# -*- coding: utf-8 -*-
"""Tests unitaires PDF → Markdown (sans appel Ollama)."""
from __future__ import annotations

import asyncio
import inspect
import io
from unittest.mock import AsyncMock, MagicMock, patch

import fitz
from PIL import Image

from app.services import vision_client
from app.services.page_preprocessor import (
    build_content_regions,
    crop_content_regions,
    detect_content_box,
    rotate_image_bytes,
)
from app.services.pdf_page_extractor import (
    _merge_markdown_regions,
    _native_page_text,
    _page_has_large_image,
    _should_use_vision,
    _vision_output_is_insufficient,
    _vision_page_result,
    extract_pdf_content_by_page,
)
from app.services.vision_client import (
    VisionExtractionError,
    _clean_markdown_response,
    vision_chat_json,
    vision_chat_text,
)


def test_clean_markdown_response_strips_fences():
    raw = "```markdown\n# Titre\n\nParagraphe\n```"
    assert _clean_markdown_response(raw) == "# Titre\n\nParagraphe"


def test_clean_markdown_response_plain():
    text = "# BILAN - ACTIF\n\n**Société :** ACME"
    assert _clean_markdown_response(text) == text


def test_vision_chat_text_has_no_format_json():
    source = inspect.getsource(vision_chat_text)
    assert '"format": "json"' not in source
    assert "'format': 'json'" not in source


def test_vision_chat_json_keeps_format_json():
    source = inspect.getsource(vision_chat_json)
    assert '"format": "json"' in source


def _make_text_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def test_native_page_text_preserves_newlines():
    pdf_bytes = _make_text_pdf("Ligne 1\nLigne 2\n\nLigne 3")
    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
        extracted = _native_page_text(doc, 0)
    assert "Ligne 1" in extracted
    assert "Ligne 2" in extracted
    assert "\n" in extracted


def test_page_has_large_image():
    doc = fitz.open()
    page = doc.new_page(width=200, height=300)
    # Image pleine page (scan simulé)
    img = Image.new("RGB", (180, 280), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    page.insert_image(page.rect, stream=buf.getvalue())
    assert _page_has_large_image(page) is True
    doc.close()


def test_should_use_vision_cases():
    pdf_bytes = _make_text_pdf("x")
    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
        page = doc[0]
        assert _should_use_vision(page, "x", force_vision=False) is True
        assert _should_use_vision(page, "x" * 200, force_vision=True) is True

    rich = _make_text_pdf("Paragraphe simple " * 20)
    with fitz.open(stream=io.BytesIO(rich), filetype="pdf") as doc:
        page = doc[0]
        text = _native_page_text(doc, 0)
        # Texte riche sans grande image ni grille → natif
        assert _should_use_vision(page, text, force_vision=False) is False


def test_merge_markdown_regions_deduplicates_overlap():
    top = "# Titre\n\nLigne A\nLigne B\nLigne overlap"
    bottom = "Ligne overlap\nLigne C\nLigne D"
    merged = _merge_markdown_regions([top, bottom])
    assert merged.count("Ligne overlap") == 1
    assert "Ligne A" in merged
    assert "Ligne C" in merged


def test_merge_markdown_regions_keeps_different_amounts():
    top = "| Stocks | 100,00 |"
    bottom = "| Stocks | 200,00 |"
    merged = _merge_markdown_regions([top, bottom])
    assert "100,00" in merged
    assert "200,00" in merged


def test_vision_output_insufficient():
    assert _vision_output_is_insufficient("[PAGE VIDE]") is False
    assert _vision_output_is_insufficient("court") is True
    long_enough = "\n".join(
        [
            "| Col1 | Col2 |",
            "|---|---:|",
            "| Ligne 0 avec du contenu suffisant | 1 |",
            "| Ligne 1 avec du contenu suffisant | 2 |",
            "| Ligne 2 avec du contenu suffisant | 3 |",
            "| Ligne 3 avec du contenu suffisant | 4 |",
        ]
    )
    assert len(long_enough) >= 120
    assert _vision_output_is_insufficient(long_enough) is False

    # Page textuelle d'identification sans vrai tableau : pas insuffisante.
    identification = "\n".join(
        [
            "Pièces annexes à la déclaration",
            "Identification du contribuable",
            "Raison sociale : SERDILAB",
            "ICE : 123456789000012",
            "Adresse : Casablanca Maroc",
            "Exercice clos le 31/12/2024",
        ]
    )
    assert len(identification) >= 120
    assert _vision_output_is_insufficient(identification) is False


def test_page_vide_status_ok():
    result = _vision_page_result(1, "[PAGE VIDE]", 12.0, extraction_strategy="full_page")
    assert result.status == "ok"
    assert result.content == "[PAGE VIDE]"
    assert result.char_count == len("[PAGE VIDE]")
    assert result.raw_model_response["page_empty"] is True
    assert result.raw_model_response["extraction_strategy"] == "full_page"


def test_build_content_regions_two_zones():
    regions = build_content_regions(800, 1000)
    assert len(regions) == 2
    assert regions[0].region_id == "top"
    assert regions[1].region_id == "bottom"
    assert regions[0].bottom > 500
    assert regions[1].top < 500


def test_detect_content_box_rejects_tiny_crop():
    img = Image.new("RGB", (200, 200), color=(255, 255, 255))
    # Petit carré noir en coin → crop trop petit → full box
    for x in range(5):
        for y in range(5):
            img.putpixel((x, y), (0, 0, 0))
    assert detect_content_box(img) == (0, 0, 200, 200)


def test_rotate_image_bytes_90_only_used_in_extractor():
    source = inspect.getsource(extract_pdf_content_by_page)
    assert "rotate_image_bytes(image_bytes, 90)" in source
    assert "rotate_image_bytes(image_bytes, 180)" not in source
    assert "rotate_image_bytes(image_bytes, 270)" not in source


def _jpeg_bytes(size=(120, 160)) -> bytes:
    img = Image.new("RGB", size, color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


SUFFICIENT_MD = "\n".join(
    [
        "# BILAN - ACTIF",
        "",
        "**Société :** ACME",
        "",
        "| Éléments | Brut | Net |",
        "|---|---:|---:|",
        "| Stocks | 100,00 | 100,00 |",
        "| Clients | 50,00 | 50,00 |",
    ]
)


def test_full_page_sufficient_skips_regions():
    pdf_bytes = _make_text_pdf("scan")
    vision_mock = AsyncMock(return_value=(SUFFICIENT_MD, 10.0))
    regions_mock = AsyncMock()

    with (
        patch(
            "app.services.pdf_page_extractor._warmup_model",
            new=AsyncMock(return_value={"warmup": "ok"}),
        ),
        patch(
            "app.services.pdf_page_extractor.render_pdf_pages",
            return_value=[_jpeg_bytes()],
        ),
        patch(
            "app.services.pdf_page_extractor.vision_chat_text",
            new=vision_mock,
        ),
        patch(
            "app.services.pdf_page_extractor._extract_page_regions",
            new=regions_mock,
        ),
        patch(
            "app.services.pdf_page_extractor._should_use_vision",
            return_value=True,
        ),
    ):
        result = asyncio.run(
            extract_pdf_content_by_page(
                pdf_bytes, "t.pdf", max_pages=1, force_vision=True
            )
        )

    assert result.pages_ok == 1
    assert result.pages[0].raw_model_response["extraction_strategy"] == "full_page"
    regions_mock.assert_not_awaited()
    assert vision_mock.await_count == 1


def test_insufficient_triggers_regions():
    short_md = "Trop court"
    merged = SUFFICIENT_MD
    vision_mock = AsyncMock(return_value=(short_md, 5.0))
    regions_mock = AsyncMock(return_value=(merged, 20.0))

    with (
        patch(
            "app.services.pdf_page_extractor._warmup_model",
            new=AsyncMock(return_value={"warmup": "ok"}),
        ),
        patch(
            "app.services.pdf_page_extractor.render_pdf_pages",
            return_value=[_jpeg_bytes()],
        ),
        patch(
            "app.services.pdf_page_extractor.vision_chat_text",
            new=vision_mock,
        ),
        patch(
            "app.services.pdf_page_extractor._extract_page_regions",
            new=regions_mock,
        ),
        patch(
            "app.services.pdf_page_extractor._should_use_vision",
            return_value=True,
        ),
    ):
        result = asyncio.run(
            extract_pdf_content_by_page(
                _make_text_pdf("x"), "t.pdf", max_pages=1, force_vision=True
            )
        )

    regions_mock.assert_awaited_once()
    assert result.pages[0].raw_model_response["extraction_strategy"] == "regions"


def test_full_page_failure_calls_regions_then_rotation_90():
    vision_mock = AsyncMock(
        side_effect=[
            VisionExtractionError("full fail"),
            (SUFFICIENT_MD, 15.0),  # after rotation
        ]
    )
    regions_mock = AsyncMock(side_effect=VisionExtractionError("regions fail"))
    rotate_mock = MagicMock(side_effect=lambda data, angle, **kw: data)

    with (
        patch(
            "app.services.pdf_page_extractor._warmup_model",
            new=AsyncMock(return_value={"warmup": "ok"}),
        ),
        patch(
            "app.services.pdf_page_extractor.render_pdf_pages",
            return_value=[_jpeg_bytes()],
        ),
        patch(
            "app.services.pdf_page_extractor.vision_chat_text",
            new=vision_mock,
        ),
        patch(
            "app.services.pdf_page_extractor._extract_page_regions",
            new=regions_mock,
        ),
        patch(
            "app.services.pdf_page_extractor.rotate_image_bytes",
            new=rotate_mock,
        ),
        patch(
            "app.services.pdf_page_extractor._should_use_vision",
            return_value=True,
        ),
    ):
        result = asyncio.run(
            extract_pdf_content_by_page(
                _make_text_pdf("x"), "t.pdf", max_pages=1, force_vision=True
            )
        )

    regions_mock.assert_awaited_once()
    rotate_mock.assert_called_once()
    assert rotate_mock.call_args.args[1] == 90
    assert result.pages[0].status == "ok"
    assert result.pages[0].raw_model_response["extraction_strategy"] == "rotation_90"
    assert result.pages[0].raw_model_response["rotation_fallback"] == 90


def test_page_vide_does_not_trigger_regions():
    vision_mock = AsyncMock(return_value=("[PAGE VIDE]", 8.0))
    regions_mock = AsyncMock()

    with (
        patch(
            "app.services.pdf_page_extractor._warmup_model",
            new=AsyncMock(return_value={"warmup": "ok"}),
        ),
        patch(
            "app.services.pdf_page_extractor.render_pdf_pages",
            return_value=[_jpeg_bytes()],
        ),
        patch(
            "app.services.pdf_page_extractor.vision_chat_text",
            new=vision_mock,
        ),
        patch(
            "app.services.pdf_page_extractor._extract_page_regions",
            new=regions_mock,
        ),
        patch(
            "app.services.pdf_page_extractor._should_use_vision",
            return_value=True,
        ),
    ):
        result = asyncio.run(
            extract_pdf_content_by_page(
                _make_text_pdf("x"), "t.pdf", max_pages=1, force_vision=True
            )
        )

    regions_mock.assert_not_awaited()
    assert result.pages[0].status == "ok"
    assert result.pages[0].content == "[PAGE VIDE]"


def test_crop_content_regions_returns_two_jpegs():
    parts = crop_content_regions(_jpeg_bytes((200, 400)))
    assert len(parts) == 2
    assert parts[0][0] == "top"
    assert parts[1][0] == "bottom"
    assert parts[0][1][:2] == b"\xff\xd8"
