"""Configuration centralisée de l'application (variables d'environnement)."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Ollama / modèle GLM vision -------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "glm4v")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT", "120"))

# --- CORS --------------------------------------------------------------
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

# --- Upload --------------------------------------------------------------
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

# CIN : images classiques + PDF (recto/verso scannés en un seul fichier)
ALLOWED_CIN_CONTENT_TYPES = ALLOWED_CONTENT_TYPES | {"application/pdf"}

# Dimension maximale (px, plus grand côté) d'une image envoyée au modèle
# vision. Au-delà, le nombre de tokens image dépasse souvent le contexte du
# serveur Ollama (ex: erreur "exceed_context_size_error" avec un modèle
# configuré à 4096 tokens de contexte). S'applique aux photos uploadées
# (image_utils.normalize_image) ET aux pages de PDF rendues en image
# (text_extractor.render_pdf_pages_to_images), ces dernières étant souvent
# bien plus grandes (page A4 entière à 200 DPI ≈ 1650×2340px).
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "1600"))

# --- Extraction ICE — modèle GLM texte (aucune dépendance système) ---------
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "glm4")

# Résolution (DPI) utilisée pour convertir en image les pages d'un PDF scanné
# (via PyMuPDF, avant envoi au modèle vision)
PDF_TO_IMAGE_DPI = int(os.getenv("PDF_TO_IMAGE_DPI", "200"))

# Seuil (en caractères) sous lequel un PDF est considéré comme "scanné"
# (texte natif insuffisant) et bascule sur le pipeline vision (image).
MIN_NATIVE_TEXT_CHARS = int(os.getenv("MIN_NATIVE_TEXT_CHARS", "40"))

ALLOWED_ICE_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "application/pdf"}

# --- Répertoires -----------------------------------------------------------
STATIC_DIR = BASE_DIR / "static"
