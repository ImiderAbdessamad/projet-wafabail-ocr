"""Router — extraction des informations d'une carte d'identité nationale (CIN).

Supporte :
- Image (JPEG/PNG/WEBP) ou PDF pour chaque face.
- Recto seul, ou recto + verso (deux fichiers séparés).
- Un PDF unique contenant recto ET verso sur 2 pages (le verso est alors
  automatiquement extrait de la 2ᵉ page si aucun fichier `verso` n'est fourni).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import ALLOWED_CIN_CONTENT_TYPES, OLLAMA_MODEL
from app.schemas import ExtractionResponse
from app.services.glm_extractor import (
    GlmExtractionError,
    extract_cin_fields_auto_rotate,
    merge_cin_sides,
)
from app.services.image_utils import read_and_validate, to_image_pages

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cin", tags=["CIN"])


@router.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="Extraire les champs d'une carte d'identité nationale",
)
async def extract_cin(
    recto: UploadFile = File(
        ..., description="Recto de la CIN — image (JPEG/PNG/WEBP) ou PDF"
    ),
    verso: Optional[UploadFile] = File(
        None, description="Verso de la CIN, optionnel — image ou PDF"
    ),
) -> ExtractionResponse:
    """Reçoit une ou deux faces de CIN, les envoie au modèle GLM vision et
    retourne les champs fusionnés : nom, prénom, numéro CIN, date/lieu de
    naissance, date d'expiration et adresse.
    """
    recto_bytes = await read_and_validate(recto, ALLOWED_CIN_CONTENT_TYPES)
    recto_pages = await asyncio.to_thread(
        to_image_pages, recto_bytes, recto.content_type, 2
    )

    verso_image: bytes | None = None
    if verso is not None:
        verso_bytes = await read_and_validate(verso, ALLOWED_CIN_CONTENT_TYPES)
        verso_pages = await asyncio.to_thread(
            to_image_pages, verso_bytes, verso.content_type, 1
        )
        verso_image = verso_pages[0]
    elif len(recto_pages) > 1:
        # PDF unique contenant recto + verso sur 2 pages.
        verso_image = recto_pages[1]

    tasks = [extract_cin_fields_auto_rotate(recto_pages[0])]
    if verso_image is not None:
        tasks.append(extract_cin_fields_auto_rotate(verso_image))

    started = time.perf_counter()
    try:
        results = await asyncio.gather(*tasks)
    except GlmExtractionError as exc:
        logger.error("Extraction CIN échouée : %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000

    sides_data = [side for side, _ in results]
    data = merge_cin_sides(*sides_data)

    warning = None
    if not any([data.nom, data.prenom, data.cin]):
        warning = (
            "Aucune information exploitable détectée. Vérifiez la netteté, "
            "l'éclairage et le cadrage de la photo."
        )
    elif verso_image is None and not data.adresse:
        warning = (
            "Adresse non détectée — elle figure généralement au verso de la "
            "carte. Ajoutez une photo du verso pour l'obtenir."
        )
    elif verso_image is not None and not data.adresse:
        warning = (
            "Adresse non détectée sur le verso fourni. Vérifiez que la carte "
            "est bien droite (non pivotée), nette et bien éclairée sur la photo."
        )

    logger.info(
        "CIN extraite en %.0f ms (%d face(s)) — nom=%r prenom=%r cin=%r",
        elapsed_ms,
        len(tasks),
        data.nom,
        data.prenom,
        data.cin,
    )

    return ExtractionResponse(
        success=True,
        data=data,
        model=OLLAMA_MODEL,
        processing_time_ms=int(elapsed_ms),
        warning=warning,
    )
