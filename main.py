"""Point d'entrée FastAPI — extraction documents marocains (CIN, ICE, liasse)
et scoring crédit-bail via modèles GLM servis par Ollama.

Lancement :
    uvicorn main:app --reload

Interfaces :
    /        — CIN
    /ice     — ICE
    /liasse  — Liasse fiscale + scoring
"""

from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import (
    ALLOWED_ORIGINS,
    OLLAMA_MODEL,
    OLLAMA_TEXT_MODEL,
    OLLAMA_URL,
    OLLAMA_VISION_MODEL,
    STATIC_DIR,
)
from app.routers import cin, export, extraction, financial_analysis, ice, scoring
from app.services.glm_extractor import warmup_model as warmup_vision_model
from app.services.ice_extractor import warmup_model as warmup_text_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Wafabail OCR API",
    description=(
        "Extraction automatique de documents marocains (CIN, ICE, liasse fiscale PCGM) "
        "et scoring crédit-bail à 3 axes — modèles GLM via Ollama."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cin.router)
app.include_router(ice.router)
app.include_router(extraction.router, prefix="/api/v1")
app.include_router(financial_analysis.router, prefix="/api/v1")
app.include_router(scoring.router, prefix="/api/v1")
app.include_router(export.router, prefix="/api/v1")


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Wafabail OCR API démarrée.")
    asyncio.create_task(warmup_vision_model())
    if OLLAMA_TEXT_MODEL != OLLAMA_MODEL:
        asyncio.create_task(warmup_text_model())


@app.get("/health", tags=["Système"], summary="Vérifier l'état du service")
async def health() -> dict:
    return {"status": "ok", "service": "wafabail-ocr"}


@app.get("/health/ollama", tags=["Système"], summary="Vérifier Ollama / modèle vision")
async def ollama_health() -> dict:
    from app.services.vision_ocr import VisionOcrError, _warmup_model

    try:
        status = await _warmup_model(soft=True)
    except VisionOcrError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    degraded = str(status.get("warmup", "")).startswith("degraded")
    return {
        "status": "degraded" if degraded else "ok",
        **status,
    }


@app.get("/ice", tags=["Système"], include_in_schema=False)
async def ice_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "ice.html")


@app.get("/pdf", tags=["Système"], include_in_schema=False)
async def pdf_extract_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "pdf.html")


@app.get("/liasse", tags=["Système"], include_in_schema=False)
async def liasse_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "liasse.html")


# Interface statique — montée en dernier pour ne pas masquer les routes API.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
