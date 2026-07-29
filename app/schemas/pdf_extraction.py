"""Schémas pour l'extraction de contenu PDF page par page."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class PdfPageTable(BaseModel):
    title: Optional[str] = None
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class PdfPageExtraction(BaseModel):
    page_number: int = Field(ge=1)
    status: str = "ok"  # ok | error
    extraction_mode: str = "native"  # native | vision
    page_title: Optional[str] = None
    content: Optional[str] = None
    tables: list[PdfPageTable] = Field(default_factory=list)
    char_count: int = 0
    model_latency_ms: Optional[int] = None
    raw_model_response: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class PdfContentExtractionResult(BaseModel):
    source_filename: Optional[str] = None
    pages_total: int = 0
    pages_processed: int = 0
    pages_ok: int = 0
    pages_failed: int = 0
    model: str
    ollama_url: str
    processing_time_ms: int = 0
    pages: list[PdfPageExtraction] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
