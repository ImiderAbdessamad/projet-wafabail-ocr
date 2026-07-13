"""Point d'entrée FastAPI — extraction de documents marocains (CIN, ICE) via
des modèles GLM servis par Ollama.

Lancement :
    uvicorn main:app --reload

Les interfaces de test (upload + affichage des champs extraits) sont servies
directement depuis le dossier static/ : "/" pour la CIN, "/ice" pour l'ICE.
"""

from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import ALLOWED_ORIGINS, OLLAMA_MODEL, OLLAMA_TEXT_MODEL, STATIC_DIR
from app.routers import cin, ice
from app.services.glm_extractor import warmup_model as warmup_vision_model
from app.services.ice_extractor import warmup_model as warmup_text_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Extractor API",
    description=(
        "Extraction automatique d'informations depuis des documents marocains — "
        "carte d'identité nationale (CIN) et certificat ICE (Identifiant Commun "
        "de l'Entreprise) — via des modèles GLM servis par Ollama."
    ),
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cin.router)
app.include_router(ice.router)


@app.on_event("startup")
async def on_startup() -> None:
    logger.info("Document Extractor API démarrée.")
    # Préchargement asynchrone (non bloquant) des modèles GLM pour éviter
    # qu'une première requête utilisateur n'essuie le délai de chargement
    # à froid (particulièrement sensible sur un serveur Ollama distant).
    asyncio.create_task(warmup_vision_model())
    if OLLAMA_TEXT_MODEL != OLLAMA_MODEL:
        asyncio.create_task(warmup_text_model())


@app.get("/health", tags=["Système"], summary="Vérifier l'état du service")
async def health() -> dict:
    return {"status": "ok", "service": "document-extractor-backend"}


@app.get("/ice", tags=["Système"], include_in_schema=False)
async def ice_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "ice.html")


# Interface de test statique — montée en dernier pour ne pas masquer les routes API.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
