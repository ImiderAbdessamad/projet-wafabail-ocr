"""Schémas d'observations brutes et de résolution de champs (niveau 1–3)."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, Field


class RawFinancialObservation(BaseModel):
    """Observation brute issue d'une page Vision / texte."""

    page: int
    section: Optional[str] = None
    table_title: Optional[str] = None
    raw_label: str
    normalized_label: str = ""
    raw_value: Optional[str] = None
    parsed_value: Optional[Decimal] = None
    row_index: Optional[int] = None
    column_name: Optional[str] = None
    column_index: Optional[int] = None
    period: Optional[str] = None
    value_nature: Optional[str] = None  # brut | net_n | total_exercice | ...
    model_confidence: Optional[float] = None
    extraction_method: str = "vision"
    line_present_empty: bool = False  # libellé vu, cellule vide

    class Config:
        arbitrary_types_allowed = True


class FieldCandidate(BaseModel):
    value: Optional[Decimal] = None
    source: Optional[str] = None
    column: Optional[str] = None
    page: Optional[int] = None
    raw_label: Optional[str] = None
    score: float = 0.0
    match_method: str = "alias"
    observation: Optional[dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


class FieldResolution(BaseModel):
    field: str
    candidates: list[FieldCandidate] = Field(default_factory=list)
    selected_value: Optional[Decimal] = None
    selection_reason: Optional[str] = None
    detection_status: str = "not_detected"
    confidence: float = 0.0
    calculated_value: Optional[Decimal] = None
    validation_status: Optional[str] = None
    raw_label: Optional[str] = None
    column: Optional[str] = None
    page: Optional[int] = None
    source: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class DocumentMetadata(BaseModel):
    reference: Optional[str] = None
    entreprise: Optional[str] = None
    identification_fiscale: Optional[str] = None
    exercice: Optional[str] = None
    date_debut_exercice: Optional[str] = None
    date_fin_exercice: Optional[str] = None
