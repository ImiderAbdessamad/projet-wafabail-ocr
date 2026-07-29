"""API analyse financière post-extraction Markdown.

POST /api/v1/extraction/pdf/analyze
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, ValidationError

from app.schemas.financial_analysis import (
    BehavioralInput,
    FinancialAnalysisResult,
    SectorBenchmarkInput,
)
from app.schemas.pdf_extraction import PdfContentExtractionResult
from app.services.analysis_pipeline import analyze_extracted_pdf
from app.services.pdf_page_extractor import extract_pdf_content_by_page
from app.services.vision_client import VisionExtractionError

# Réutilise la validation d'upload du router extraction
from app.routers.extraction import _read_upload

router = APIRouter(prefix="/extraction", tags=["Analyse financière"])


class PdfFinancialAnalysisResponse(BaseModel):
    extraction: PdfContentExtractionResult
    analysis: FinancialAnalysisResult


@router.post("/pdf/analyze", response_model=PdfFinancialAnalysisResponse)
async def analyze_pdf_financial(
    file: UploadFile = File(..., description="PDF à analyser"),
    max_pages: Optional[int] = Form(None),
    force_vision: bool = Form(False),
    behavioral_data: Optional[str] = Form(
        None,
        description="JSON BehavioralInput (optionnel)",
    ),
    sector_benchmark: Optional[str] = Form(
        None,
        description="JSON SectorBenchmarkInput (optionnel)",
    ),
    scoring_mode: str = Form("STRICT"),
) -> PdfFinancialAnalysisResponse:
    """PDF → Markdown page par page → dataset → ratios Decimal → décision.

    Les calculs sont 100 % Python/Decimal. Le LLM n'intervient que pour l'OCR.
    """
    if scoring_mode.upper() not in {"STRICT", "REVIEW"}:
        raise HTTPException(
            status_code=422,
            detail="scoring_mode doit être STRICT ou REVIEW.",
        )
    if max_pages is not None and max_pages < 1:
        raise HTTPException(status_code=422, detail="max_pages doit être >= 1.")

    content, filename = await _read_upload(file, None)

    behavioral: BehavioralInput | None = None
    sector: SectorBenchmarkInput | None = None
    if behavioral_data:
        try:
            behavioral = BehavioralInput.model_validate(json.loads(behavioral_data))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"behavioral_data invalide : {exc}",
            ) from exc
    if sector_benchmark:
        try:
            sector = SectorBenchmarkInput.model_validate(json.loads(sector_benchmark))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HTTPException(
                status_code=422,
                detail=f"sector_benchmark invalide : {exc}",
            ) from exc

    try:
        extraction = await extract_pdf_content_by_page(
            content,
            filename,
            max_pages=max_pages,
            force_vision=force_vision,
        )
    except VisionExtractionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Extraction PDF impossible : {exc}",
        ) from exc

    analysis = await analyze_extracted_pdf(
        extraction,
        behavioral_input=behavioral,
        sector_input=sector,
        scoring_mode=scoring_mode.upper(),
    )
    return PdfFinancialAnalysisResponse(extraction=extraction, analysis=analysis)
