"""Pipeline principal : Markdown OCR → Qwen mapping → dataset → ratios → scoring."""
from __future__ import annotations

import logging

from app.config import OLLAMA_MAPPING_MODEL
from app.scoring_rules import SCORING_MODE_DEFAULT
from app.schemas.financial_analysis import (
    BehavioralInput,
    FinancialAnalysisResult,
    SectorBenchmarkInput,
)
from app.schemas.financial_mapping import FinancialMappingAudit
from app.schemas.pdf_extraction import PdfContentExtractionResult
from app.services.behavioral_scoring import calculate_behavioral_score
from app.services.credit_decision import build_credit_decision, calculate_final_score
from app.services.financial_candidate_resolver import (
    build_financial_dataset_from_resolved_values,
    resolve_financial_candidates,
)
from app.services.financial_controls import (
    invalidate_conflicting_fields,
    run_accounting_controls,
)
from app.services.financial_mapping_client import map_financial_sections
from app.services.financial_ratios import calculate_financial_ratios
from app.services.financial_scoring import calculate_financial_score
from app.services.financial_section_splitter import split_financial_sections
from app.services.sector_scoring import calculate_sector_score

logger = logging.getLogger(__name__)


async def analyze_extracted_pdf(
    extraction: PdfContentExtractionResult,
    *,
    behavioral_input: BehavioralInput | None = None,
    sector_input: SectorBenchmarkInput | None = None,
    scoring_mode: str = SCORING_MODE_DEFAULT,
) -> FinancialAnalysisResult:
    """Analyse financière de production : Qwen comme unique mapper."""
    warnings: list[str] = list(extraction.warnings or [])
    section_inputs = split_financial_sections(extraction)
    mapping_batch = await map_financial_sections(section_inputs)
    warnings.extend(mapping_batch.warnings)

    resolved_values = resolve_financial_candidates(mapping_batch.mapped_sections)
    dataset = build_financial_dataset_from_resolved_values(resolved_values)
    accounting_checks = run_accounting_controls(dataset)
    dataset = invalidate_conflicting_fields(dataset, accounting_checks)
    warnings.extend(dataset.warnings)

    ratios = calculate_financial_ratios(dataset)
    financial_axis = calculate_financial_score(
        dataset,
        ratios,
        scoring_mode=scoring_mode,
    )
    behavioral_axis, behavioral_blocking = calculate_behavioral_score(
        behavioral_input
    )
    sector_axis = calculate_sector_score(ratios, sector_input)

    axes = [financial_axis, behavioral_axis, sector_axis]
    final_score, final_score_blocking = calculate_final_score(axes)

    blocking_reasons: list[str] = []
    blocking_reasons.extend(financial_axis.blocking_reasons)
    blocking_reasons.extend(behavioral_blocking)
    blocking_reasons.extend(sector_axis.blocking_reasons)
    blocking_reasons.extend(final_score_blocking)
    blocking_reasons = list(dict.fromkeys(blocking_reasons))

    decision = build_credit_decision(
        final_score,
        blocking_reasons=blocking_reasons,
    )

    if scoring_mode.upper() == "REVIEW" and final_score is not None and blocking_reasons:
        warnings.append(
            "Mode REVIEW : score informatif uniquement — décision automatique bloquée."
        )

    candidates_by_field: dict[str, int] = {}
    for section in mapping_batch.mapped_sections:
        for candidate in section.candidates:
            code = str(candidate.field_code)
            candidates_by_field[code] = candidates_by_field.get(code, 0) + 1

    resolved_fields = sorted(
        code
        for code, fv in resolved_values.items()
        if fv.status in {"confirmed", "derived"} and fv.value is not None
    )
    missing_fields = sorted(
        code for code, fv in resolved_values.items() if fv.status == "missing"
    )
    conflicting_fields = sorted(
        code for code, fv in resolved_values.items() if fv.status == "conflicting"
    )

    mapping = FinancialMappingAudit(
        strategy="qwen_only",
        model=mapping_batch.model or OLLAMA_MAPPING_MODEL,
        sections_detected=len(section_inputs),
        sections_processed=len(mapping_batch.mapped_sections),
        sections_failed=max(len(section_inputs) - len(mapping_batch.mapped_sections), 0),
        candidates_total=sum(candidates_by_field.values()),
        candidates_by_field=candidates_by_field,
        resolved_fields=resolved_fields,
        missing_fields=missing_fields,
        conflicting_fields=conflicting_fields,
        warnings=list(mapping_batch.warnings),
    )

    return FinancialAnalysisResult(
        dataset=dataset,
        accounting_checks=accounting_checks,
        ratios=ratios,
        axes=axes,
        final_score=final_score if not blocking_reasons else None,
        decision=decision,
        warnings=warnings,
        scoring_mode=scoring_mode.upper(),
        mapping=mapping,
    )
