"""Validation et normalisation des images/PDF uploadés pour l'extraction vision.

Objectifs :
- Rejeter les fichiers invalides (mauvais type, trop volumineux, vides).
- Corriger l'orientation EXIF (photos prises au téléphone).
- Réduire la résolution si nécessaire pour accélérer l'appel au modèle vision.
- Convertir un PDF (recto/verso scannés) en image(s) exploitables par le modèle.
"""

from __future__ import annotations

import io
import logging

from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps

from app.config import ALLOWED_CONTENT_TYPES, MAX_IMAGE_DIMENSION, MAX_UPLOAD_BYTES

logger = logging.getLogger(__name__)

# Une CIN n'a pas besoin d'une résolution énorme pour être lisible par le modèle ;
# on limite la plus grande dimension pour garder des temps de réponse raisonnables
# et rester sous la limite de tokens de contexte du serveur Ollama.
MAX_DIMENSION = MAX_IMAGE_DIMENSION


async def read_and_validate(file: UploadFile, allowed_content_types: set[str] | None = None) -> bytes:
    """Lit le fichier uploadé et vérifie son type/sa taille."""
    allowed = allowed_content_types or ALLOWED_CONTENT_TYPES
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Format non supporté ({file.content_type or 'inconnu'}). "
                "Utilisez une image JPEG/PNG/WEBP ou un fichier PDF."
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

    return data


def normalize_image(data: bytes) -> tuple[bytes, str]:
    """Corrige l'orientation, redimensionne si besoin et réencode en JPEG.

    Retombe silencieusement sur les octets bruts si l'image ne peut pas être
    décodée par Pillow (le modèle vision tentera quand même de la lire).
    """
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)

        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")

        if max(image.size) > MAX_DIMENSION:
            image.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue(), "image/jpeg"
    except Exception as exc:  # image corrompue ou format exotique
        logger.warning("Normalisation image impossible, envoi des octets bruts : %s", exc)
        return data, "image/jpeg"


def to_image_pages(data: bytes, content_type: str, max_pages: int = 1) -> list[bytes]:
    """Retourne une liste d'images (JPEG) prêtes pour le modèle vision.

    - Image classique → normalisée, une seule page en sortie.
    - PDF → chaque page (jusqu'à `max_pages`) rendue en image via PyMuPDF
      (aucune dépendance système, contrairement à Poppler/pdf2image). Utile
      pour un PDF recto+verso scanné en un seul fichier de 2 pages.
    """
    if content_type == "application/pdf":
        from app.services.text_extractor import render_pdf_pages_to_images

        return render_pdf_pages_to_images(data, max_pages=max_pages)

    normalized, _mime = normalize_image(data)
    return [normalized]
