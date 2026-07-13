"""Extraction depuis un PDF — texte natif ou rendu en image(s) pour un modèle vision.

Stratégie (zéro dépendance système — aucun binaire externe requis) :
1. Texte natif via `pdfplumber` (rapide, gratuit, pur Python).
2. Si le texte natif est insuffisant (PDF scanné) → rendu des pages en images
   via `PyMuPDF` (également pur Python, aucun binaire externe comme Poppler).
   Les images obtenues sont ensuite envoyées à un modèle vision (voir
   `ice_extractor.extract_ice_fields_from_image`) plutôt qu'à un OCR local.

Les images (JPEG/PNG) uploadées directement ne passent pas par ce module :
elles sont envoyées telles quelles au modèle vision (voir `image_utils.py`).
"""

from __future__ import annotations

import io
import logging
import re

import pdfplumber
from PIL import Image

from app.config import MAX_IMAGE_DIMENSION, MIN_NATIVE_TEXT_CHARS, PDF_TO_IMAGE_DPI

logger = logging.getLogger(__name__)


class TextExtractionError(RuntimeError):
    """Levée quand le PDF ne peut être ni lu nativement, ni rendu en image."""


def clean_text(raw: str) -> str:
    """Normalise les espaces et sauts de ligne du texte natif extrait."""
    text = raw.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_native_text(data: bytes) -> tuple[str, bool]:
    """Extrait le texte natif d'un PDF via pdfplumber.

    Returns:
        (texte_nettoyé, suffisant) — `suffisant` indique si le texte natif
        contient assez de caractères pour être considéré comme un PDF
        numérique (par opposition à un PDF scanné, sans texte natif).
    """
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            pages_text = [page.extract_text() or "" for page in pdf.pages]
        text = clean_text("\n\n".join(pages_text))
    except Exception as exc:
        logger.warning("pdfplumber a échoué : %s", exc)
        return "", False

    char_count = len(text.replace(" ", "").replace("\n", ""))
    sufficient = char_count >= MIN_NATIVE_TEXT_CHARS
    if sufficient:
        logger.info("PDF texte natif (pdfplumber) : %d caractères", len(text))
    return text, sufficient


def _downscale_jpeg(data: bytes, max_dimension: int) -> bytes:
    """Réduit une image JPEG si sa plus grande dimension dépasse `max_dimension`.

    Une page A4 entière rendue à `PDF_TO_IMAGE_DPI` peut dépasser 2000px de
    côté, ce qui consomme énormément de tokens côté modèle vision et peut
    déclencher une erreur "exceed_context_size_error" sur le serveur Ollama.
    """
    try:
        with Image.open(io.BytesIO(data)) as img:
            if max(img.size) <= max_dimension:
                return data
            img = img.convert("RGB")
            img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=88)
            return buffer.getvalue()
    except Exception as exc:
        logger.warning("Redimensionnement de la page PDF impossible : %s", exc)
        return data


def render_pdf_pages_to_images(data: bytes, max_pages: int = 2) -> list[bytes]:
    """Convertit les premières pages d'un PDF (scanné) en images JPEG.

    Utilise PyMuPDF : aucune dépendance système (contrairement à
    `pdf2image`/Poppler), ce qui garde le pipeline ICE aussi simple à
    déployer que le pipeline CIN. Les pages sont ensuite redimensionnées
    (voir `_downscale_jpeg`) pour rester sous la limite de tokens du modèle.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise TextExtractionError(
            "PyMuPDF (pymupdf) n'est pas installé — impossible de traiter un PDF scanné."
        ) from exc

    try:
        doc = fitz.open(stream=data, filetype="pdf")
        matrix = fitz.Matrix(PDF_TO_IMAGE_DPI / 72, PDF_TO_IMAGE_DPI / 72)
        pages = [
            page.get_pixmap(matrix=matrix).tobytes("jpeg")
            for page in doc[:max_pages]
        ]
        doc.close()
    except Exception as exc:
        logger.error("Rendu PDF → image (PyMuPDF) échoué : %s", exc)
        raise TextExtractionError(
            "Impossible de convertir le PDF scanné en image."
        ) from exc

    if not pages:
        raise TextExtractionError("Le PDF ne contient aucune page exploitable.")

    return [_downscale_jpeg(page, MAX_IMAGE_DIMENSION) for page in pages]
