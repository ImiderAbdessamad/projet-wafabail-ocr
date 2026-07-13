"""Router — extraction des informations d'un certificat ICE
(Identifiant Commun de l'Entreprise) marocain.

Aucune dépendance système requise (pas de Tesseract/Poppler) :
- Image → envoyée directement au modèle GLM vision (comme pour la CIN).
- PDF numérique → texte natif (pdfplumber) → modèle GLM texte.
- PDF scanné (texte natif insuffisant) → rendu en image (PyMuPDF) → modèle
  GLM vision, comme pour une image classique.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import ALLOWED_ICE_CONTENT_TYPES, MAX_UPLOAD_BYTES, OLLAMA_MODEL, OLLAMA_TEXT_MODEL
from app.schemas import IceExtractionResponse
from app.services.ice_extractor import (
    IceExtractionError,
    extract_ice_fields_from_image,
    extract_ice_fields_from_text,
)
from app.services.image_utils import normalize_image
from app.services.text_extractor import (
    TextExtractionError,
    extract_pdf_native_text,
    render_pdf_pages_to_images,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["ICE"])


@router.post(
    "/extract-ice",
    response_model=IceExtractionResponse,
    summary="Extraire les champs d'un certificat ICE",
)
async def extract_ice(
    file: UploadFile = File(
        ..., description="Certificat ICE : image (JPEG/PNG) ou PDF (natif ou scanné)"
    ),
) -> IceExtractionResponse:
    """Reçoit un certificat ICE et structure les champs via un modèle GLM :
    ICE, dénomination, identifiant fiscal, RC (numéro + ville), CNSS.
    """
    if file.content_type not in ALLOWED_ICE_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Format non supporté ({file.content_type or 'inconnu'}). "
                "Utilisez une image JPEG/PNG ou un fichier PDF."
            ),
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Le fichier envoyé est vide.")
    if len(data) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413, detail=f"Fichier trop volumineux (max {max_mb} Mo)."
        )

    try:
        if file.content_type.startswith("image/"):
            normalized_bytes, _mime = normalize_image(data)
            ice_data, elapsed_ms = await extract_ice_fields_from_image(normalized_bytes)
            ocr_method = "vision"
            model_used = OLLAMA_MODEL
        else:
            # PDF : texte natif (pdfplumber) → sinon rendu image (PyMuPDF) + vision.
            native_text, sufficient = await asyncio.to_thread(extract_pdf_native_text, data)
            if sufficient:
                ice_data, elapsed_ms = await extract_ice_fields_from_text(native_text)
                ocr_method = "pdfplumber"
                model_used = OLLAMA_TEXT_MODEL
            else:
                logger.info("PDF sans texte natif exploitable → rendu image + vision")
                pages = await asyncio.to_thread(render_pdf_pages_to_images, data)
                ice_data, elapsed_ms = await extract_ice_fields_from_image(pages[0])
                ocr_method = "vision-pdf"
                model_used = OLLAMA_MODEL
    except TextExtractionError as exc:
        logger.error("Extraction PDF (ICE) échouée : %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except IceExtractionError as exc:
        logger.error("Extraction ICE (LLM) échouée : %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    warning = None
    if not any([ice_data.ICE, ice_data.Denomination, ice_data.RC_Numero]):
        warning = (
            "Aucune information exploitable détectée. Vérifiez la qualité du "
            "document ou sa lisibilité."
        )

    logger.info(
        "ICE extrait en %.0f ms (méthode: %s) — ICE=%r denomination=%r",
        elapsed_ms,
        ocr_method,
        ice_data.ICE,
        ice_data.Denomination,
    )

    return IceExtractionResponse(
        success=True,
        data=ice_data,
        model=model_used,
        ocr_method=ocr_method,
        processing_time_ms=int(elapsed_ms),
        warning=warning,
    )
