"""API d'extraction intelligente des liasses fiscales (PCGM).

- POST /extraction/liasse : PDF ou ZIP → 19 éléments financiers via OCR vision
  (méthode principale pour liasses scannées multi-pages).
- POST /extraction/liasse/zip/list : liste les PDF d'une archive ZIP.
- POST /extraction/liasse/score : pipeline complet extraction + scoring 3 axes.
- POST /extraction/pdf/content : PDF → contenu extrait page par page (GLM Vision).
"""
import json
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError

from app.config import MAX_UPLOAD_BYTES, SCORING_MIN_COMPLETENESS_PCT
from app.routers.scoring import evaluate_scoring
from app.schemas.liasse import LiasseExtractionResult
from app.schemas.pdf_extraction import PdfContentExtractionResult
from app.schemas.scoring import (
    BehavioralMetricsInput,
    DecisionOutput,
    FinancialDataInput,
    ScoringRequest,
    ScoringResponse,
)
from app.services.document_loader import (
    DocumentLoadError,
    extract_pdf_from_zip,
    is_zip_content,
    list_zip_pdf_entries,
)
from app.services.liasse_extraction import (
    LiasseExtractionService,
    extract_liasse_document,
    get_liasse_extraction_service,
)
from app.services.pdf_page_extractor import extract_pdf_content_by_page
from app.services.scoring_eligibility import evaluate_extraction_eligibility
from app.services.vision_client import VisionExtractionError

router = APIRouter(prefix="/extraction", tags=["Extraction"])


class ExtractionScoringComplement(BaseModel):
    bam_cotation: Optional[int] = None
    financial_overrides: Optional[FinancialDataInput] = None
    behavioral_data: Optional[BehavioralMetricsInput] = None
    sector_medians: Optional[dict[str, float]] = None


class ExtractAndScoreResponse(BaseModel):
    extraction: LiasseExtractionResult
    scoring: Optional[ScoringResponse] = None
    scoring_skipped_reason: Optional[str] = None
    scoring_warning: Optional[str] = None


class ZipPdfEntryOut(BaseModel):
    path: str
    pages: int
    size_bytes: int
    size_label: str = ""


class ZipListResponse(BaseModel):
    entries: list[ZipPdfEntryOut]
    total: int


def _scoring_block_reason(
    extraction: LiasseExtractionResult,
    financial: FinancialDataInput | None = None,
) -> str | None:
    """Refuse une décision crédit lorsque l'extraction ne permet pas de calculer
    les ratios essentiels.

    Sans ce contrôle, le moteur historique exclut les ratios « Non calculable »
    de ses pénalités et peut artificiellement rendre 90/A+ à une liasse
    incomplète.
    """
    if extraction.document_kind == "LIASSE_ECHEC":
        return (
            "Extraction OCR échouée — aucune donnée exploitable, scoring non lancé. "
            "Vérifiez la qualité du scan et la disponibilité du serveur Ollama."
        )
    if extraction.eligible_for_automatic_scoring is False:
        return extraction.scoring_block_reasons[0] if extraction.scoring_block_reasons else (
            "Scoring non lancé : document non admissible au scoring automatique."
        )

    missing_sections = [
        section
        for section in ("BILAN_ACTIF", "BILAN_PASSIF", "CPC")
        if not extraction.sections_completeness.get(section, False)
    ]
    elements_by_code = {element.code: element for element in extraction.elements}
    data = financial or FinancialDataInput(
        **extraction.scoring_input.model_dump()
    )
    required_values = {
        "chiffre d'affaires": data.chiffre_affaires,
        "total bilan": data.total_bilan,
        "résultat net": data.resultat_net,
        "fonds propres": data.fonds_propres,
        "dettes financières": data.dettes_financieres,
        "CAF": data.caf,
        "fonds de roulement": data.fdr,
    }
    missing_values = [label for label, value in required_values.items() if value is None]
    invalid_values = []
    for code, label in (
        ("CHIFFRE_AFFAIRES", "chiffre d'affaires"),
        ("TOTAL_BILAN", "total bilan"),
        ("RESULTAT_NET", "résultat net"),
        ("FONDS_PROPRES", "fonds propres"),
        ("CAF", "CAF"),
        ("FDR", "fonds de roulement"),
    ):
        element = elements_by_code.get(code)
        if not element:
            continue
        if element.detection_status in {"ambiguous", "conflicting", "incomplete", "estimated"}:
            invalid_values.append(f"{label} ({element.detection_status})")
        elif element.validation and element.validation.status in {"invalidated", "divergent", "failed"}:
            invalid_values.append(f"{label} ({element.validation.status})")

    if (
        extraction.completeness_pct < SCORING_MIN_COMPLETENESS_PCT
        or missing_sections
        or missing_values
        or invalid_values
    ):
        details: list[str] = [
            f"complétude {extraction.completeness_pct:.1f}% "
            f"(< {SCORING_MIN_COMPLETENESS_PCT:.0f}%)"
        ]
        if missing_sections:
            details.append("sections absentes : " + ", ".join(missing_sections))
        if missing_values:
            details.append("données absentes : " + ", ".join(missing_values))
        if invalid_values:
            details.append("données non fiables : " + ", ".join(invalid_values))
        return (
            "Scoring non lancé : extraction insuffisante ("
            + " ; ".join(details)
            + "). Aucune décision de crédit ne peut être déduite de ce résultat."
        )
    return None


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} o"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} Ko"
    return f"{size_bytes / (1024 * 1024):.1f} Mo"


async def _read_upload(
    file: UploadFile,
    pdf_entry: str | None = None,
) -> tuple[bytes, str]:
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux (max {MAX_UPLOAD_BYTES // (1024 * 1024)} Mo).",
        )

    filename = file.filename or "document"

    if is_zip_content(content, filename, file.content_type):
        try:
            pdf_bytes, pdf_name = extract_pdf_from_zip(content, pdf_entry)
            return pdf_bytes, pdf_name
        except DocumentLoadError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=415,
            detail="Format non supporté. Envoyez un PDF ou une archive ZIP contenant des PDF.",
        )
    return content, filename


@router.post("/liasse/zip/list", response_model=ZipListResponse)
async def list_zip_entries(file: UploadFile = File(...)) -> ZipListResponse:
    """Liste les PDF contenus dans une archive ZIP (sans lancer l'extraction)."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Archive trop volumineuse.")
    if not is_zip_content(content, file.filename, file.content_type):
        raise HTTPException(status_code=415, detail="Le fichier n'est pas une archive ZIP.")
    try:
        entries = list_zip_pdf_entries(content)
    except DocumentLoadError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ZipListResponse(
        entries=[
            ZipPdfEntryOut(
                path=e.path,
                pages=e.pages,
                size_bytes=e.size_bytes,
                size_label=_format_size(e.size_bytes),
            )
            for e in entries
        ],
        total=len(entries),
    )


@router.post("/liasse", response_model=LiasseExtractionResult)
async def extract_liasse(
    file: UploadFile = File(..., description="PDF ou archive ZIP contenant des PDF"),
    pdf_entry: Optional[str] = Form(
        None,
        description="Chemin du PDF dans le ZIP (optionnel — sinon le plus complet est choisi)",
    ),
    service: LiasseExtractionService = Depends(get_liasse_extraction_service),
) -> LiasseExtractionResult:
    """Extraction structurée d'une liasse fiscale.

    Pipeline unifié :
    - PDF texte natif → observations natives → résolution → scoring_input
    - PDF scanné → OCR Vision GLM Flash (serveur Ollama distant `.env`) →
      observations → résolution → scoring_input

    Le modèle n'est **pas** local : il est appelé via `OLLAMA_URL` /
    `OLLAMA_VISION_MODEL` (ex. GLM-4.6V-Flash sur NiceGPU).
    """
    content, filename = await _read_upload(file, pdf_entry)
    try:
        return await extract_liasse_document(content, filename, service)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Extraction impossible : {exc}") from exc


@router.post("/liasse/score", response_model=ExtractAndScoreResponse)
async def extract_and_score(
    file: UploadFile = File(...),
    pdf_entry: Optional[str] = Form(None, description="PDF à extraire du ZIP"),
    complement: Optional[str] = Form(
        None,
        description="JSON optionnel : bam_cotation, financial_overrides, behavioral_data…",
    ),
    service: LiasseExtractionService = Depends(get_liasse_extraction_service),
) -> ExtractAndScoreResponse:
    """Pipeline complet API scoring : extraction OCR distant + ratios + 3 axes.

    Même moteur Vision que `/extraction/liasse` (GLM Flash via Ollama distant),
    puis scoring métier.
    """
    content, filename = await _read_upload(file, pdf_entry)
    try:
        extraction = await extract_liasse_document(content, filename, service)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Extraction impossible : {exc}") from exc

    comp = ExtractionScoringComplement()
    if complement:
        try:
            comp = ExtractionScoringComplement.model_validate(json.loads(complement))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=f"Complément invalide : {exc}") from exc

    si = extraction.scoring_input
    financial = FinancialDataInput(
        chiffre_affaires=si.chiffre_affaires,
        ca_n1=si.ca_n1,
        total_bilan=si.total_bilan,
        fonds_propres=si.fonds_propres,
        dettes_financieres=si.dettes_financieres,
        resultat_net=si.resultat_net,
        caf=si.caf,
        clients=si.clients,
        fournisseurs=si.fournisseurs,
        achats=si.achats,
        frais_financiers=si.frais_financiers,
        amortissements=si.amortissements,
        fdr=si.fdr,
        tresorerie_nette=si.tresorerie_nette,
        encours_leasing=si.encours_leasing,
        cmt=si.cmt,
        nouveau_financement=si.nouveau_financement,
    )
    if comp.financial_overrides:
        overrides = comp.financial_overrides.model_dump(exclude_none=True)
        financial = financial.model_copy(update=overrides)

    scoring_block_reason = _scoring_block_reason(extraction, financial)
    if extraction.document_kind == "LIASSE_ECHEC":
        return ExtractAndScoreResponse(
            extraction=extraction,
            scoring_skipped_reason=scoring_block_reason,
        )

    scoring_request = ScoringRequest(
        bam_cotation=comp.bam_cotation,
        financial_data=financial,
        behavioral_data=comp.behavioral_data or BehavioralMetricsInput(),
        sector_medians=comp.sector_medians,
    )
    scoring = await evaluate_scoring(scoring_request)
    eligibility = evaluate_extraction_eligibility(
        extraction,
        {k: v.model_dump() for k, v in scoring.ratios.items()},
        comp.behavioral_data or BehavioralMetricsInput(),
    )
    scoring.eligibility = scoring.eligibility or eligibility

    if scoring_block_reason:
        # Les ratios disponibles restent utiles à l'analyste, mais une
        # décision automatique est interdite tant que les bases financières
        # essentielles restent absentes.
        scoring.decision = DecisionOutput(
            score=None,
            classe="Non évaluable",
            decision="Revue manuelle",
            recommandation=scoring_block_reason,
            blocking_status="INSUFFICIENT_DATA",
        )
        return ExtractAndScoreResponse(
            extraction=extraction,
            scoring=scoring,
            scoring_warning=scoring_block_reason,
        )

    return ExtractAndScoreResponse(extraction=extraction, scoring=scoring)


@router.post("/pdf/content", response_model=PdfContentExtractionResult)
async def extract_pdf_content(
    file: UploadFile = File(..., description="PDF à analyser page par page"),
    max_pages: Optional[int] = Form(
        None,
        description="Nombre max de pages à traiter (défaut : OCR_MAX_PAGES)",
    ),
    force_vision: bool = Form(
        False,
        description="Forcer GLM Vision même si le PDF contient du texte natif",
    ),
) -> PdfContentExtractionResult:
    """Extrait le contenu brut d'un PDF page par page via GLM Vision (ou texte natif).

    Retourne pour chaque page : titre, texte transcrit, tableaux détectés et métadonnées.
    Utile pour prévisualiser l'OCR avant un pipeline métier (liasse, scoring…).
    """
    content, filename = await _read_upload(file, None)
    if max_pages is not None and max_pages < 1:
        raise HTTPException(status_code=422, detail="max_pages doit être >= 1.")
    try:
        return await extract_pdf_content_by_page(
            content,
            filename,
            max_pages=max_pages,
            force_vision=force_vision,
        )
    except VisionExtractionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Extraction PDF impossible : {exc}"
        ) from exc
