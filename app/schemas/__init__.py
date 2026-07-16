"""Schémas Pydantic de l'API (CIN, ICE, liasse, scoring)."""

from app.schemas.documents import (
    CinData,
    ErrorResponse,
    ExtractionResponse,
    IceData,
    IceExtractionResponse,
)

__all__ = [
    "CinData",
    "ErrorResponse",
    "ExtractionResponse",
    "IceData",
    "IceExtractionResponse",
]
