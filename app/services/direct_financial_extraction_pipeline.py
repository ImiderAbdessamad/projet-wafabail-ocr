"""Pipeline : PDF → orientation → classification → GLM Vision → resolver → ratios."""
from __future__ import annotations

import asyncio
import io
import logging
from dataclasses import dataclass
from typing import Any, Callable

import fitz
from PIL import Image

from app.config import (
    DIRECT_FINANCIAL_MAX_IMAGE_DIMENSION,
    DIRECT_FINANCIAL_MAX_PAGES,
    DIRECT_FINANCIAL_MODEL,
    DIRECT_FINANCIAL_PAGE_DELAY_SECONDS,
    DIRECT_FINANCIAL_REGION_FALLBACK,
    DIRECT_FINANCIAL_RENDER_DPI,
)
from app.schemas.direct_financial_extraction import (
    CompanyInfo,
    DirectFinancialCandidate,
    DirectFinancialEvidence,
    DirectFinancialExtractionBatch,
    DocumentSummary,
    ExerciseInfo,
    ExtractionSummary,
    FinancialDocumentAnalysisResult,
    FinancialPageAudit,
    FinancialPageType,
)
from app.schemas.financial_analysis import CreditDecision
from app.scoring_rules import SCORING_MODE_DEFAULT
from app.services.credit_decision import build_credit_decision, calculate_final_score
from app.services.direct_financial_resolver import (
    build_dataset_from_direct_candidates,
    dedupe_direct_candidates,
)
from app.services.direct_glm_financial_client import (
    DirectFinancialExtractionError,
    DirectFinancialLengthError,
    extract_financial_page,
    prompt_for_page_type,
    schema_for_page_type,
    warmup_direct_financial_model,
)
from app.services.financial_controls import (
    invalidate_conflicting_fields,
    run_accounting_controls,
)
from app.services.financial_orientation_detector import (
    detect_page_orientation,
    rotate_to_orientation,
)
from app.services.financial_page_classifier import classify_financial_page
from app.services.financial_ratios import calculate_financial_ratios
from app.services.financial_scoring import calculate_financial_score
from app.services.page_preprocessor import (
    crop_content_regions,
    detect_content_box,
)
from app.services.behavioral_scoring import calculate_behavioral_score
from app.services.sector_scoring import calculate_sector_score

logger = logging.getLogger(__name__)

ProgressEmitter = Callable[[str, dict[str, Any]], None]

_EXTRACTABLE: set[str] = {
    "IDENTIFICATION",
    "BILAN_ACTIF",
    "BILAN_PASSIF",
    "CPC",
    "DETAIL_CPC",
    "RESULTAT_FISCAL",
    "ESG",
}


@dataclass
class RenderedPage:
    page_number: int
    image_bytes: bytes
    declared_rotation: int
    native_text: str


def validate_pdf(pdf_bytes: bytes) -> None:
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Le fichier n'est pas un PDF valide (%PDF manquant).")
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"PDF illisible : {exc}") from exc
    try:
        if doc.page_count < 1:
            raise ValueError("PDF sans page.")
    finally:
        doc.close()


def count_pdf_pages(pdf_bytes: bytes) -> int:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return int(doc.page_count)
    finally:
        doc.close()


def _downscale_png(image_bytes: bytes, max_dim: int) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        # Crop prudent
        box = detect_content_box(img)
        if box != (0, 0, img.width, img.height):
            img = img.crop(box)
        out = io.BytesIO()
        img.save(out, format="PNG", optimize=True)
        return out.getvalue()


def render_pdf_pages_png(
    pdf_bytes: bytes,
    *,
    dpi: int = DIRECT_FINANCIAL_RENDER_DPI,
    max_pages: int | None = None,
) -> list[RenderedPage]:
    limit = max_pages or DIRECT_FINANCIAL_MAX_PAGES
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[RenderedPage] = []
    try:
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        for index in range(min(doc.page_count, limit)):
            page = doc[index]
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            png = pixmap.tobytes("png")
            native = page.get_text("text") or ""
            pages.append(
                RenderedPage(
                    page_number=index + 1,
                    image_bytes=png,
                    declared_rotation=int(page.rotation or 0) % 360,
                    native_text=native,
                )
            )
    finally:
        doc.close()
    return pages


def _convert_page_output(
    mapped: Any,
    *,
    page_number: int,
    page_type: FinancialPageType,
    orientation: int,
) -> list[DirectFinancialCandidate]:
    candidates: list[DirectFinancialCandidate] = []
    for item in getattr(mapped, "candidates", []) or []:
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        evidence = data.get("evidence") or {}
        evidence["page_number"] = page_number
        evidence["page_type"] = page_type
        evidence["orientation"] = orientation
        try:
            candidates.append(
                DirectFinancialCandidate(
                    field_code=str(data["field_code"]),
                    raw_value=str(data["raw_value"]),
                    period=data["period"],
                    nature=data["nature"],
                    confidence=float(data.get("confidence", 0.5)),
                    evidence=DirectFinancialEvidence.model_validate(evidence),
                    warnings=list(data.get("warnings") or []),
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Candidat page %d ignoré : %s", page_number, exc)
    return candidates


async def _extract_with_regions(
    oriented_image: bytes,
    *,
    page_number: int,
    page_type: str,
    orientation: int,
) -> tuple[list[DirectFinancialCandidate], int, str]:
    schema = schema_for_page_type(page_type)
    prompt = prompt_for_page_type(page_type)
    regions = crop_content_regions(oriented_image)
    merged: list[DirectFinancialCandidate] = []
    total_latency = 0
    for region_id, region_bytes in regions:
        # crop_content_regions renvoie JPEG — OK pour GLM
        try:
            mapped, latency = await extract_financial_page(
                region_bytes,
                page_number=page_number,
                page_type=page_type,
                orientation=orientation,
                schema_model=schema,
                system_prompt=prompt,
                max_attempts=1,
            )
            total_latency += latency
            merged.extend(
                _convert_page_output(
                    mapped,
                    page_number=page_number,
                    page_type=page_type,  # type: ignore[arg-type]
                    orientation=orientation,
                )
            )
        except DirectFinancialExtractionError as exc:
            logger.warning(
                "Région %s page=%d échouée : %s",
                region_id,
                page_number,
                exc,
            )
    return dedupe_direct_candidates(merged), total_latency, "regional_fallback"


async def analyze_financial_document(
    pdf_bytes: bytes,
    filename: str,
    *,
    include_markdown: bool = False,
    max_pages: int | None = None,
    scoring_mode: str = SCORING_MODE_DEFAULT,
    emit: ProgressEmitter | None = None,
) -> FinancialDocumentAnalysisResult:
    """Pipeline principal d'extraction directe GLM Vision."""

    def _emit(event: str, data: dict[str, Any] | None = None) -> None:
        if emit:
            emit(event, data or {})

    validate_pdf(pdf_bytes)
    _emit("pdf_validated", {"filename": filename})

    # Évite les 504 NiceGPU / cold-start avant la 1ʳᵉ page
    await warmup_direct_financial_model()

    pages_total = count_pdf_pages(pdf_bytes)
    limit = min(max_pages or DIRECT_FINANCIAL_MAX_PAGES, pages_total, DIRECT_FINANCIAL_MAX_PAGES)
    _emit("job_started", {"pages_total": pages_total, "pages_limit": limit})

    rendered = render_pdf_pages_png(
        pdf_bytes,
        dpi=DIRECT_FINANCIAL_RENDER_DPI,
        max_pages=limit,
    )
    _emit("pages_rendered", {"count": len(rendered)})

    page_audit: list[FinancialPageAudit] = []
    all_candidates: list[DirectFinancialCandidate] = []
    warnings: list[str] = []
    previous_type: FinancialPageType | None = None
    pages_processed = 0
    pages_skipped = 0
    pages_failed = 0
    company = CompanyInfo()
    exercise = ExerciseInfo()
    markdown_pages: list[dict] | None = [] if include_markdown else None

    for rendered_page in rendered:
        page_no = rendered_page.page_number
        orientation = detect_page_orientation(
            rendered_page.image_bytes,
            declared_rotation=rendered_page.declared_rotation,
        )
        oriented = rotate_to_orientation(rendered_page.image_bytes, orientation)
        oriented = _downscale_png(oriented, DIRECT_FINANCIAL_MAX_IMAGE_DIMENSION)

        page_type = await classify_financial_page(
            image_bytes=oriented,
            native_text=rendered_page.native_text,
            previous_page_type=previous_type,
            use_glm_fallback=True,
        )
        logger.info(
            "Page %d classifiée=%s orientation=%s native_chars=%d image_bytes=%d",
            page_no,
            page_type,
            orientation,
            len(rendered_page.native_text or ""),
            len(oriented),
        )
        _emit(
            "page_classified",
            {
                "page": page_no,
                "page_type": page_type,
                "orientation": orientation,
            },
        )

        if page_type in {"AUTRE", "VIDE"}:
            pages_skipped += 1
            page_audit.append(
                FinancialPageAudit(
                    page_number=page_no,
                    detected_type=page_type,
                    orientation=orientation,  # type: ignore[arg-type]
                    extraction_status="skipped" if page_type == "AUTRE" else "empty",
                    extraction_strategy="classification",
                    warnings=["Page ignorée (non financière)."],
                )
            )
            _emit(
                "page_skipped",
                {"page": page_no, "page_type": page_type},
            )
            continue

        if page_type not in _EXTRACTABLE:
            pages_skipped += 1
            page_audit.append(
                FinancialPageAudit(
                    page_number=page_no,
                    detected_type=page_type,
                    orientation=orientation,  # type: ignore[arg-type]
                    extraction_status="skipped",
                    extraction_strategy="classification",
                )
            )
            continue

        schema = schema_for_page_type(page_type)
        prompt = prompt_for_page_type(page_type)
        strategy = "full_page"
        latency_ms: int | None = None
        page_candidates: list[DirectFinancialCandidate] = []

        try:
            mapped, latency_ms = await extract_financial_page(
                oriented,
                page_number=page_no,
                page_type=page_type,
                orientation=orientation,
                schema_model=schema,
                system_prompt=prompt,
            )
            page_candidates = _convert_page_output(
                mapped,
                page_number=page_no,
                page_type=page_type,
                orientation=orientation,
            )
        except DirectFinancialLengthError:
            if DIRECT_FINANCIAL_REGION_FALLBACK:
                page_candidates, latency_ms, strategy = await _extract_with_regions(
                    oriented,
                    page_number=page_no,
                    page_type=page_type,
                    orientation=orientation,
                )
            else:
                pages_failed += 1
                page_audit.append(
                    FinancialPageAudit(
                        page_number=page_no,
                        detected_type=page_type,
                        orientation=orientation,  # type: ignore[arg-type]
                        extraction_status="failed",
                        extraction_strategy="length",
                        error="done_reason=length",
                    )
                )
                _emit("page_failed", {"page": page_no, "error": "length"})
                continue
        except DirectFinancialExtractionError as exc:
            if DIRECT_FINANCIAL_REGION_FALLBACK:
                try:
                    page_candidates, latency_ms, strategy = await _extract_with_regions(
                        oriented,
                        page_number=page_no,
                        page_type=page_type,
                        orientation=orientation,
                    )
                except DirectFinancialExtractionError as region_exc:
                    pages_failed += 1
                    page_audit.append(
                        FinancialPageAudit(
                            page_number=page_no,
                            detected_type=page_type,
                            orientation=orientation,  # type: ignore[arg-type]
                            extraction_status="failed",
                            extraction_strategy="failed",
                            error=str(region_exc),
                        )
                    )
                    warnings.append(f"Page {page_no} : {region_exc}")
                    _emit("page_failed", {"page": page_no, "error": str(region_exc)})
                    continue
            else:
                pages_failed += 1
                page_audit.append(
                    FinancialPageAudit(
                        page_number=page_no,
                        detected_type=page_type,
                        orientation=orientation,  # type: ignore[arg-type]
                        extraction_status="failed",
                        extraction_strategy="failed",
                        error=str(exc),
                    )
                )
                warnings.append(f"Page {page_no} : {exc}")
                _emit("page_failed", {"page": page_no, "error": str(exc)})
                continue

        # Métadonnées identification
        if page_type == "IDENTIFICATION":
            for cand in page_candidates:
                code = cand.field_code
                val = cand.raw_value
                if code == "RAISON_SOCIALE" and not company.raison_sociale:
                    company.raison_sociale = val
                elif code == "IDENTIFIANT_FISCAL" and not company.identifiant_fiscal:
                    company.identifiant_fiscal = val
                elif code == "ICE" and not company.ice:
                    company.ice = val
                elif code == "ADRESSE" and not company.adresse:
                    company.adresse = val
                elif code == "VILLE" and not company.ville:
                    company.ville = val
                elif code == "DATE_DEBUT_EXERCICE" and not exercise.debut:
                    exercise.debut = val
                elif code == "DATE_FIN_EXERCICE" and not exercise.fin:
                    exercise.fin = val
                elif code == "EXERCICE" and not exercise.label:
                    exercise.label = val

        all_candidates.extend(page_candidates)
        pages_processed += 1
        previous_type = page_type
        page_audit.append(
            FinancialPageAudit(
                page_number=page_no,
                detected_type=page_type,
                orientation=orientation,  # type: ignore[arg-type]
                extraction_status="processed",
                extraction_strategy=strategy,
                candidates_count=len(page_candidates),
                model_latency_ms=latency_ms,
            )
        )
        _emit(
            "page_extracted",
            {
                "page": page_no,
                "page_type": page_type,
                "candidates": len(page_candidates),
                "latency_ms": latency_ms,
            },
        )

        if DIRECT_FINANCIAL_PAGE_DELAY_SECONDS > 0:
            await asyncio.sleep(DIRECT_FINANCIAL_PAGE_DELAY_SECONDS)

    _emit("resolving_fields", {"candidates": len(all_candidates)})
    dataset = build_dataset_from_direct_candidates(all_candidates)

    _emit("running_controls", {})
    accounting_checks = run_accounting_controls(dataset)
    dataset = invalidate_conflicting_fields(dataset, accounting_checks)
    warnings.extend(dataset.warnings)

    _emit("calculating_ratios", {})
    ratios = calculate_financial_ratios(dataset)
    financial_axis = calculate_financial_score(
        dataset,
        ratios,
        scoring_mode=scoring_mode,
    )
    behavioral_axis, behavioral_blocking = calculate_behavioral_score(None)
    sector_axis = calculate_sector_score(ratios, None)
    axes = [financial_axis, behavioral_axis, sector_axis]
    final_score, final_blocking = calculate_final_score(axes)

    blocking: list[str] = []
    blocking.extend(financial_axis.blocking_reasons)
    blocking.extend(behavioral_blocking)
    blocking.extend(sector_axis.blocking_reasons)
    blocking.extend(final_blocking)
    # Contrôles failed bloquent STRICT
    if scoring_mode.upper() == "STRICT":
        for check in accounting_checks:
            if check.status == "failed":
                blocking.append(f"Contrôle comptable échoué : {check.code}")
    blocking = list(dict.fromkeys(blocking))

    decision: CreditDecision = build_credit_decision(
        final_score,
        blocking_reasons=blocking,
    )

    batch = DirectFinancialExtractionBatch(
        model=DIRECT_FINANCIAL_MODEL,
        pages_total=pages_total,
        pages_processed=pages_processed,
        pages_skipped=pages_skipped,
        pages_failed=pages_failed,
        candidates=dedupe_direct_candidates(all_candidates),
        page_audit=page_audit,
        warnings=warnings,
    )

    result = FinancialDocumentAnalysisResult(
        document=DocumentSummary(
            filename=filename,
            pages_total=pages_total,
            pages_processed=pages_processed,
            pages_skipped=pages_skipped,
            pages_failed=pages_failed,
            company=company,
            exercise=exercise,
        ),
        extraction=ExtractionSummary(
            model=DIRECT_FINANCIAL_MODEL,
            page_audit=page_audit,
            warnings=list(batch.warnings),
        ),
        dataset=dataset,
        accounting_checks=accounting_checks,
        ratios=ratios,
        axes=axes,
        decision=decision,
        warnings=warnings,
        markdown_pages=markdown_pages,
    )
    _emit("result_ready", {"pages_processed": pages_processed})
    return result


async def run_financial_job(
    job_id: str,
    *,
    store: Any,
) -> None:
    """Exécute un job stocké (séquentiel, un seul GLM à la fois)."""
    job = store.get(job_id)
    if job is None or job.pdf_bytes is None:
        return

    store.update(
        job_id,
        status="processing",
        current_step="validating",
        message="Validation du PDF…",
        progress_pct=2,
    )
    store.emit(job_id, "job_started", {"filename": job.filename})

    def emit(event: str, data: dict[str, Any]) -> None:
        # Met à jour la progression approximative
        step_map = {
            "pdf_validated": ("validating", 5, "PDF validé"),
            "pages_rendered": ("rendering", 15, "Pages rendues"),
            "page_classified": ("classifying", None, None),
            "page_extracted": ("extracting_page", None, None),
            "page_skipped": ("extracting_page", None, None),
            "page_failed": ("extracting_page", None, None),
            "resolving_fields": ("resolving", 85, "Résolution des champs"),
            "running_controls": ("controls", 90, "Contrôles comptables"),
            "calculating_ratios": ("ratios", 95, "Calcul des ratios"),
            "result_ready": ("completed", 100, "Analyse terminée"),
        }
        step, pct, message = step_map.get(event, (None, None, None))
        updates: dict[str, Any] = {}
        if step:
            updates["current_step"] = step
        if pct is not None:
            updates["progress_pct"] = pct
        if message:
            updates["message"] = message
        if "page" in data:
            updates["current_page"] = data["page"]
            pages_total = job.pages_total or data.get("pages_total")
            if pages_total:
                updates["progress_pct"] = min(
                    80,
                    15 + int(70 * int(data["page"]) / max(int(pages_total), 1)),
                )
                updates["message"] = (
                    f"Page {data['page']}/{pages_total} — "
                    f"{data.get('page_type', '')}"
                )
        if event == "pages_rendered":
            updates["pages_total"] = data.get("count")
            job.pages_total = data.get("count")
        if event == "page_extracted":
            updates["pages_financial"] = (job.pages_financial or 0) + 1
            job.pages_financial = updates["pages_financial"]
        if event == "page_skipped":
            updates["pages_skipped"] = (job.pages_skipped or 0) + 1
            job.pages_skipped = updates["pages_skipped"]
        if event == "page_failed":
            updates["pages_failed"] = (job.pages_failed or 0) + 1
            job.pages_failed = updates["pages_failed"]
        if updates:
            store.update(job_id, **updates)
        store.emit(job_id, event, data)

    try:
        result = await analyze_financial_document(
            job.pdf_bytes,
            job.filename,
            include_markdown=job.include_markdown,
            max_pages=job.max_pages,
            emit=emit,
        )
        store.update(
            job_id,
            status="completed",
            progress_pct=100,
            current_step="completed",
            message="Analyse terminée",
            result=result,
            pdf_bytes=None,  # libère la mémoire
        )
        store.emit(job_id, "result_ready", {"status": "completed"})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Job financier %s échoué", job_id)
        store.update(
            job_id,
            status="failed",
            current_step="failed",
            message="Échec de l'analyse",
            error=str(exc),
            pdf_bytes=None,
        )
        store.emit(job_id, "job_failed", {"error": str(exc)})
