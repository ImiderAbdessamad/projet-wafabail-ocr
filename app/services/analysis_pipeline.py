"""Pipeline principal : Markdown OCR → dataset → ratios → scoring → décision."""
from __future__ import annotations

from app.scoring_rules import SCORING_MODE_DEFAULT
from app.schemas.financial_analysis import (
    BehavioralInput,
    FinancialAnalysisResult,
    SectorBenchmarkInput,
)
from app.schemas.pdf_extraction import PdfContentExtractionResult
from app.services.behavioral_scoring import calculate_behavioral_score
from app.services.credit_decision import build_credit_decision, calculate_final_score
from app.services.financial_controls import (
    invalidate_conflicting_fields,
    run_accounting_controls,
)
from app.services.financial_dataset_builder import build_financial_dataset
from app.services.financial_ratios import calculate_financial_ratios
from app.services.financial_scoring import calculate_financial_score
from app.services.markdown_financial_parser import parse_markdown_pages
from app.services.sector_scoring import calculate_sector_score


async def analyze_extracted_pdf(
    extraction: PdfContentExtractionResult,
    *,
    behavioral_input: BehavioralInput | None = None,
    sector_input: SectorBenchmarkInput | None = None,
    scoring_mode: str = SCORING_MODE_DEFAULT,
) -> FinancialAnalysisResult:
    """Analyse financière déterministe à partir du Markdown extrait."""
    warnings: list[str] = list(extraction.warnings or [])

    rows = parse_markdown_pages(extraction.pages)
    if not rows:
        warnings.append(
            "Aucune ligne financière Markdown détectée — dataset vide."
        )

    dataset = build_financial_dataset(rows)
    checks = run_accounting_controls(dataset)
    dataset = invalidate_conflicting_fields(dataset, checks)
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
    blocking: list[str] = []
    blocking.extend(financial_axis.blocking_reasons)
    blocking.extend(behavioral_blocking)
    if not sector_axis.calculable:
        blocking.extend(sector_axis.blocking_reasons)

    final_score, final_blocking = calculate_final_score(axes)
    blocking.extend(final_blocking)
    # Dédupliquer en conservant l'ordre
    seen: set[str] = set()
    unique_blocking: list[str] = []
    for reason in blocking:
        if reason not in seen:
            seen.add(reason)
            unique_blocking.append(reason)

    decision = build_credit_decision(
        final_score,
        blocking_reasons=unique_blocking,
    )

    if scoring_mode.upper() == "REVIEW" and final_score is not None and unique_blocking:
        warnings.append(
            "Mode REVIEW : score informatif uniquement — décision automatique bloquée."
        )

    return FinancialAnalysisResult(
        dataset=dataset,
        accounting_checks=checks,
        ratios=ratios,
        axes=axes,
        final_score=final_score if not unique_blocking else None,
        decision=decision,
        warnings=warnings,
        scoring_mode=scoring_mode.upper(),
    )
