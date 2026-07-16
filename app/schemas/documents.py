"""Schémas Pydantic — extraction CIN et ICE."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class CinData(BaseModel):
    """Informations extraites d'une carte d'identité nationale marocaine.

    `lieu_naissance` figure généralement au recto, `adresse` généralement au
    verso — d'où l'intérêt de fournir les deux faces à l'extraction.
    """

    nom: str = Field(default="", description="Nom de famille")
    prenom: str = Field(default="", description="Prénom")
    cin: str = Field(default="", description="Numéro de la carte d'identité nationale")
    date_naissance: str = Field(default="", description="Date de naissance (JJ/MM/AAAA)")
    lieu_naissance: str = Field(default="", description="Lieu de naissance (recto)")
    date_expiration: str = Field(
        default="", description="Date d'expiration de la carte (JJ/MM/AAAA)"
    )
    adresse: str = Field(default="", description="Adresse du domicile (verso)")


class ExtractionResponse(BaseModel):
    """Réponse renvoyée par /api/cin/extract."""

    success: bool = True
    data: CinData
    model: str
    processing_time_ms: int
    warning: Optional[str] = None


class IceData(BaseModel):
    """Informations extraites d'un certificat ICE (Identifiant Commun de l'Entreprise)."""

    ICE: str = Field(default="", description="Identifiant Commun de l'Entreprise (15 chiffres)")
    Denomination: str = Field(default="", description="Raison sociale / dénomination")
    Identifiant_Fiscal: str = Field(default="", description="Numéro d'identifiant fiscal (IF)")
    RC_Numero: str = Field(default="", description="Numéro du registre de commerce, sans la ville")
    RC_Ville: str = Field(default="", description="Ville du registre de commerce")
    CNSS: str = Field(default="", description="Numéro d'affiliation CNSS")


class IceExtractionResponse(BaseModel):
    """Réponse renvoyée par /api/v1/extract-ice."""

    success: bool = True
    data: IceData
    model: str
    ocr_method: str
    processing_time_ms: int
    warning: Optional[str] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
