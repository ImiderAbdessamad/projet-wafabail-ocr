"""Pipeline principal : Markdown OCR → mapping → dataset → ratios → scoring."""
from __future__ import annotations

import logging
from typing import Literal

from app.config import OLLAMA_MAPPING_MODEL
from app.scoring_rules import SCORING_MODE_DEFAULT
from app.schemas.financial_analysis import (
    BehavioralInput,
    FinancialAnalysisResult,
    SectorBenchmarkInput,
)
from app.schemas.pdf_extraction import PdfContentExtractionResult
from app.services.behavioral_scoring import calculate_behavioral_score
from app.services.credit_decision import build_credit_decision, calculate_final_score
from app.services.financial_candidate_resolver import (
    build_financial_dataset_from_resolved_values,
    financial_values_from_dataset,
    merge_resolved_values,
    resolve_financial_candidates,
)
from app.services.financial_controls import (
    invalidate_conflicting_fields,
    run_accounting_controls,
)
from app.services.financial_dataset_builder import build_financial_dataset
from app.services.financial_mapping_client import map_financial_sections
from app.services.financial_ratios import calculate_financial_ratios
from app.services.financial_scoring import calculate_financial_score
from app.services.financial_section_splitter import split_financial_sections
from app.services.markdown_financial_parser import parse_markdown_pages
from app.services.sector_scoring import calculate_sector_score

logger = logging.getLogger(__name__)

MappingStrategy = Literal["deterministic", "llm", "hybrid"]


async def analyze_extracted_pdf(
    extraction: PdfContentExtractionResult,
    *,
    behavioral_input: BehavioralInput | None = None,
    sector_input: SectorBenchmarkInput | None = None,
    scoring_mode: str = SCORING_MODE_DEFAULT,
    mapping_strategy: MappingStrategy | str = "hybrid",
) -> FinancialAnalysisResult:
    """Analyse financière : OCR Markdown → (Qwen mapping) → Python/Decimal."""
    strategy = (mapping_strategy or "hybrid").lower().strip()
    if strategy not in {"deterministic", "llm", "hybrid"}:
        strategy = "hybrid"

    warnings: list[str] = list(extraction.warnings or [])
    mapping_warnings: list[str] = []
    sections_processed = 0
    mapping_model: str | None = None
    resolved_values: dict = {}

    if strategy in {"llm", "hybrid"}:
        section_inputs = split_financial_sections(extraction)
        mapping_batch = await map_financial_sections(section_inputs)
        mapping_model = mapping_batch.model or OLLAMA_MAPPING_MODEL
        sections_processed = len(mapping_batch.mapped_sections)
        mapping_warnings.extend(mapping_batch.warnings)
        resolved_values = resolve_financial_candidates(
            mapping_batch.mapped_sections
        )
        logger.info(
            "Mapping strategy=%s model=%s sections=%d candidates_fields=%d",
            strategy,
            mapping_model,
            sections_processed,
            len(resolved_values),
        )

    if strategy in {"deterministic", "hybrid"}:
        rows = parse_markdown_pages(extraction.pages)
        if not rows and strategy == "deterministic":
            warnings.append(
                "Aucune ligne financière Markdown détectée — dataset vide."
            )
        deterministic_dataset = build_financial_dataset(rows)
        deterministic_values = financial_values_from_dataset(deterministic_dataset)

        if strategy == "deterministic":
            resolved_values = deterministic_values
        else:
            resolved_values = merge_resolved_values(
                llm_values=resolved_values,
                deterministic_values=deterministic_values,
            )
            if mapping_warnings and not resolved_values:
                warnings.append(
                    "Mapping Qwen partiellement indisponible — "
                    "repli sur le parser déterministe."
                )

    dataset = build_financial_dataset_from_resolved_values(resolved_values)
    checks = run_accounting_controls(dataset)
    dataset = invalidate_conflicting_fields(dataset, checks)
    warnings.extend(dataset.warnings)
    warnings.extend(mapping_warnings)

    # Conflits comptables essentiels → blocage STRICT
    essential_conflicts = [
        c
        for c in checks
        if c.status == "failed"
        and c.code in {"bilan_equilibre", "resultat_net", "resultat_net_xiii_xvi"}
    ]

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

    if scoring_mode.upper() == "STRICT" and essential_conflicts:
        for check in essential_conflicts:
            blocking.append(f"Contrôle comptable échoué : {check.code} — {check.message}")

    final_score, final_blocking = calculate_final_score(axes)
    blocking.extend(final_blocking)
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

    mapping_audit = {
        "strategy": strategy,
        "model": mapping_model if strategy != "deterministic" else None,
        "sections_processed": sections_processed if strategy != "deterministic" else 0,
        "warnings": mapping_warnings,
    }

    return FinancialAnalysisResult(
        dataset=dataset,
        accounting_checks=checks,
        ratios=ratios,
        axes=axes,
        final_score=final_score if not unique_blocking else None,
        decision=decision,
        warnings=warnings,
        scoring_mode=scoring_mode.upper(),
        mapping=mapping_audit,
    )
