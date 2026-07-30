"""Configuration centralisée (variables d'environnement) — CIN, ICE, liasse."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Ollama / modèles GLM -------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "glm4v")
# Alias liasse : même modèle vision que CIN/ICE si OLLAMA_VISION_MODEL absent
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", OLLAMA_MODEL)
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "glm4")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT", "180"))
OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT", "300"))

# --- CORS -----------------------------------------------------------------
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    if origin.strip()
]

# --- Upload ---------------------------------------------------------------
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "50"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_CIN_CONTENT_TYPES = ALLOWED_CONTENT_TYPES | {"application/pdf"}
ALLOWED_ICE_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "application/pdf"}

# Dimension max (px) envoyée au modèle vision
MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "1600"))

# --- ICE ------------------------------------------------------------------
PDF_TO_IMAGE_DPI = int(os.getenv("PDF_TO_IMAGE_DPI", "200"))
MIN_NATIVE_TEXT_CHARS = int(os.getenv("MIN_NATIVE_TEXT_CHARS", "40"))

# --- Liasse OCR vision ----------------------------------------------------
OCR_MAX_PAGES = int(os.getenv("OCR_MAX_PAGES", "70"))
OCR_MAX_CONCURRENCY = int(os.getenv("OCR_MAX_CONCURRENCY", "1"))
OCR_RETRY_ATTEMPTS = int(os.getenv("OCR_RETRY_ATTEMPTS", "5"))
OCR_RETRY_DELAY_SECONDS = float(os.getenv("OCR_RETRY_DELAY", "15"))
OCR_PAGE_DELAY_SECONDS = float(os.getenv("OCR_PAGE_DELAY", "4"))
OCR_FAILED_PASS_DELAY_SECONDS = float(os.getenv("OCR_FAILED_PASS_DELAY", "20"))
# Limite tokens de sortie Vision (évite timeouts gateway sur serveur distant)
OCR_VISION_NUM_PREDICT = int(os.getenv("OCR_VISION_NUM_PREDICT", "8192"))
# Warmup : si false, un 504 au préchauffage n'empêche pas l'extraction
OCR_SOFT_WARMUP = os.getenv("OCR_SOFT_WARMUP", "true").lower() in {"1", "true", "yes"}
NATIVE_COMPLETENESS_THRESHOLD = float(os.getenv("NATIVE_COMPLETENESS_THRESHOLD", "15"))
SCORING_MIN_COMPLETENESS_PCT = float(os.getenv("SCORING_MIN_COMPLETENESS_PCT", "50"))

# --- Mapping financier Qwen (même OLLAMA_URL que GLM) ---------------------
OLLAMA_MAPPING_MODEL = os.getenv("OLLAMA_MAPPING_MODEL", "qwen3:8b")
OLLAMA_MAPPING_TIMEOUT_SECONDS = float(
    os.getenv("OLLAMA_MAPPING_TIMEOUT_SECONDS", "300")
)
OLLAMA_MAPPING_NUM_CTX = int(os.getenv("OLLAMA_MAPPING_NUM_CTX", "32768"))
OLLAMA_MAPPING_NUM_PREDICT = int(os.getenv("OLLAMA_MAPPING_NUM_PREDICT", "8192"))
OLLAMA_MAPPING_KEEP_ALIVE = os.getenv("OLLAMA_MAPPING_KEEP_ALIVE", "5m")
OLLAMA_MAPPING_MAX_ATTEMPTS = int(os.getenv("OLLAMA_MAPPING_MAX_ATTEMPTS", "3"))
OLLAMA_MAPPING_MAX_SECTION_CHARS = int(
    os.getenv("OLLAMA_MAPPING_MAX_SECTION_CHARS", "60000")
)

# --- Répertoires ----------------------------------------------------------
STATIC_DIR = BASE_DIR / "static"
