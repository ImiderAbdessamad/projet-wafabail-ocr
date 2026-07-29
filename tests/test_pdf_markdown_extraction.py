# -*- coding: utf-8 -*-
"""Tests du pipeline d'extraction PDF → Markdown (indépendant du scoring)."""
import io

import fitz
from PIL import Image

from app.services.page_preprocessor import rotate_image_bytes
from app.services.pdf_page_extractor import _looks_like_table, _native_page_text
from app.services.vision_client import _clean_markdown_response


def test_clean_markdown_response_strips_fences():
    raw = "```markdown\n# Titre\n\nParagraphe\n```"
    assert _clean_markdown_response(raw) == "# Titre\n\nParagraphe"


def test_clean_markdown_response_strips_md_and_text_fences():
    assert _clean_markdown_response("```md\nHello\n```") == "Hello"
    assert _clean_markdown_response("```text\nHello\n```") == "Hello"
    assert _clean_markdown_response("```\nHello\n```") == "Hello"


def test_clean_markdown_response_leaves_plain_markdown():
    text = "# BILAN - ACTIF\n\n**Société :** ACME"
    assert _clean_markdown_response(text) == text


def test_clean_markdown_response_empty():
    assert _clean_markdown_response("") == ""
    assert _clean_markdown_response(None) == ""  # type: ignore[arg-type]


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
    # Ne doit PAS aplatir tout le texte sur une seule ligne
    assert extracted != " ".join(extracted.split())

def test_looks_like_table_false_on_plain_text():
    pdf_bytes = _make_text_pdf("Simple paragraph without grid lines.")
    with fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf") as doc:
        assert _looks_like_table(doc[0]) is False


def test_rotate_image_bytes_changes_dimensions():
    img = Image.new("RGB", (200, 100), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    original = buf.getvalue()

    rotated = rotate_image_bytes(original, 90)
    with Image.open(io.BytesIO(rotated)) as out:
        assert out.size == (100, 200)
