"""Résolution sémantique des champs à partir d'observations brutes."""
from __future__ import annotations

import logging
from decimal import Decimal
from typing import Iterable

from app.schemas.observations import (
    DocumentMetadata,
    FieldCandidate,
    FieldResolution,
    RawFinancialObservation,
)
from app.services.amount_parser import parse_amount, sum_decimal
from app.services.field_definitions import (
    COLUMN_HEADER_MAP,
    FIELD_DEFINITIONS,
    FIELD_EXCLUSIONS,
    METADATA_ALIASES,
    FieldDefinition,
)
from app.services.label_normalizer import label_similarity, normalize_label

logger = logging.getLogger(__name__)

_LABEL_WEIGHT = 0.40
_SECTION_WEIGHT = 0.20
_COLUMN_WEIGHT = 0.25
_PERIOD_WEIGHT = 0.10
_TABLE_WEIGHT = 0.05
_MIN_LABEL_SCORE = 0.55


def classify_column(header: str | None) -> str:
    """Mappe un en-tête de colonne vers une value_nature canonique."""
    if not header:
        return "unknown"
    norm = normalize_label(header)
    if norm in COLUMN_HEADER_MAP:
        return COLUMN_HEADER_MAP[norm]
    for key, nature in COLUMN_HEADER_MAP.items():
        if key in norm or norm in key:
            return nature
    return "unknown"


def observations_from_page_payload(
    page_num: int,
    payload: dict,
    *,
    extraction_method: str = "vision",
) -> list[RawFinancialObservation]:
    """Convertit une réponse Vision (format lignes/colonnes ou legacy flat) en observations."""
    observations: list[RawFinancialObservation] = []
    section = (payload.get("page_type") or payload.get("section") or "AUTRE").upper()
    table_title = payload.get("table_title")
    page_orientation = payload.get("orientation")
    region_id = payload.get("region_id")

    # --- Format structuré (préféré) -----------------------------------------
    rows = payload.get("rows") or []
    columns = payload.get("columns") or []
    for row_idx, row in enumerate(rows):
        label = (row.get("label") or "").strip()
        if not label:
            continue
        values = row.get("values") or {}
        empty = bool(row.get("empty") or row.get("line_present_empty"))
        if not values and empty:
            observations.append(
                RawFinancialObservation(
                    observation_id=f"p{page_num}-r{row_idx}-empty",
                    region_id=region_id,
                    page=page_num,
                    section=section,
                    table_title=table_title,
                    raw_label=label,
                    normalized_label=normalize_label(label),
                    raw_value=None,
                    parsed_value=None,
                    row_index=row_idx,
                    line_present_empty=True,
                    extraction_method=extraction_method,
                    model_confidence=row.get("confidence"),
                    orientation=page_orientation,
                    warnings=row.get("warnings") or [],
                )
            )
            continue
        # values peut être dict colonne→montant ou liste alignée sur columns
        if isinstance(values, list):
            paired = {
                (columns[i] if i < len(columns) else f"col_{i}"): v
                for i, v in enumerate(values)
            }
        else:
            paired = values
        for col_name, raw_val in paired.items():
            nature = classify_column(str(col_name))
            parsed = parse_amount(raw_val)
            observations.append(
                RawFinancialObservation(
                    observation_id=f"p{page_num}-r{row_idx}-c{normalize_label(str(col_name))}",
                    region_id=region_id,
                    page=page_num,
                    section=section,
                    table_title=table_title,
                    raw_label=label,
                    normalized_label=normalize_label(label),
                    raw_value=None if raw_val is None else str(raw_val),
                    parsed_value=parsed,
                    row_index=row_idx,
                    column_name=str(col_name),
                    value_nature=nature,
                    line_present_empty=parsed is None and empty,
                    extraction_method=extraction_method,
                    model_confidence=row.get("confidence"),
                    orientation=page_orientation,
                    warnings=row.get("warnings") or [],
                )
            )

    # --- Métadonnées page identification ------------------------------------
    meta = payload.get("metadata") or {}
    for key, raw in meta.items():
        if raw is None or str(raw).strip() == "":
            continue
        observations.append(
            RawFinancialObservation(
                observation_id=f"p{page_num}-meta-{key}",
                page=page_num,
                section="IDENTIFICATION",
                raw_label=str(key),
                normalized_label=normalize_label(str(key)),
                raw_value=str(raw),
                parsed_value=None,
                value_nature="unknown",
                extraction_method=extraction_method,
                orientation=page_orientation,
            )
        )

    # --- Format legacy flat elements (rétrocompat) --------------------------
    elements = payload.get("elements") or {}
    empty_fields = set(payload.get("empty_fields") or [])
    if elements and not rows:
        from app.services.liasse_extraction import ELEMENTS_19, SCORING_METRICS

        code_labels = {code: label for _, code, label, _ in ELEMENTS_19}
        code_labels.update({c: meta[0] for c, meta in SCORING_METRICS.items()})
        for code, raw_val in elements.items():
            label = code_labels.get(code, code)
            parsed = parse_amount(raw_val)
            observations.append(
                RawFinancialObservation(
                    observation_id=f"p{page_num}-legacy-{code}",
                    page=page_num,
                    section=section,
                    raw_label=label,
                    normalized_label=normalize_label(label),
                    raw_value=None if raw_val is None else str(raw_val),
                    parsed_value=parsed,
                    value_nature="unknown",  # colonne inconnue → confiance ↓
                    line_present_empty=code in empty_fields and parsed is None,
                    extraction_method=f"{extraction_method}_legacy_flat",
                    model_confidence=0.55 if parsed is not None else None,
                    orientation=page_orientation,
                )
            )
            # Annoter le code via table_title pour matching exact code
            if observations:
                observations[-1].table_title = f"CODE:{code}"

    return observations


def _is_excluded(code: str, observation: RawFinancialObservation) -> bool:
    exclusions = FIELD_EXCLUSIONS.get(code, ())
    label = observation.normalized_label
    return any(normalize_label(ex) in label or label == normalize_label(ex) for ex in exclusions)


def _column_score(defn: FieldDefinition, nature: str | None) -> float:
    nature = nature or "unknown"
    if nature in defn.forbidden_columns:
        return -1.0  # rejet
    if not defn.preferred_columns:
        return 0.5
    if nature in defn.preferred_columns:
        # Bonus selon rang de préférence
        rank = defn.preferred_columns.index(nature)
        return 1.0 - 0.1 * rank
    if nature == "unknown":
        return 0.25  # legacy / ambigu
    return 0.0


def _section_score(defn: FieldDefinition, section: str | None) -> float:
    if not section:
        return 0.3
    if section in defn.sections:
        return 1.0
    if section == "AUTRE" and "AUTRE" in defn.sections:
        return 0.8
    return 0.1


def _score_observation(
    defn: FieldDefinition,
    obs: RawFinancialObservation,
) -> tuple[float, str] | None:
    if _is_excluded(defn.code, obs):
        return None

    # Match exact via CODE:XXX (legacy)
    if obs.table_title == f"CODE:{defn.code}":
        label_score = 1.0
        method = "exact_code"
    else:
        best_alias = 0.0
        for alias in defn.aliases:
            # Similarité (observation, alias) — pas l'inverse pour l'inclusion
            best_alias = max(best_alias, label_similarity(obs.raw_label, alias))
        label_score = best_alias
        method = "alias"
        if label_score < _MIN_LABEL_SCORE:
            return None

    col_score = _column_score(defn, obs.value_nature)
    if col_score < 0:
        return None

    sec_score = _section_score(defn, obs.section)
    period_score = 1.0 if obs.value_nature in ("net_n", "total_exercice", "exercice", "unknown") else 0.5
    table_score = 0.8 if obs.table_title else 0.5

    total = (
        _LABEL_WEIGHT * label_score
        + _SECTION_WEIGHT * sec_score
        + _COLUMN_WEIGHT * col_score
        + _PERIOD_WEIGHT * period_score
        + _TABLE_WEIGHT * table_score
    )
    # Pénalité format legacy
    if "legacy" in obs.extraction_method:
        total *= 0.85
    return total, method


def resolve_field(
    code: str,
    observations: Iterable[RawFinancialObservation],
) -> FieldResolution:
    defn = FIELD_DEFINITIONS.get(code)
    if not defn:
        return FieldResolution(field=code)

    candidates: list[FieldCandidate] = []
    empty_line_seen = False
    empty_line_label: str | None = None

    for obs in observations:
        scored = _score_observation(defn, obs)
        if scored is None:
            continue
        score, method = scored
        if obs.line_present_empty and obs.parsed_value is None:
            empty_line_seen = True
            empty_line_label = obs.raw_label
            continue
        if obs.parsed_value is None:
            continue
        candidates.append(
            FieldCandidate(
                value=obs.parsed_value,
                source=obs.section,
                column=obs.value_nature or obs.column_name,
                page=obs.page,
                raw_label=obs.raw_label,
                score=score,
                match_method=method,
                observation=obs.model_dump(mode="json"),
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)

    if not candidates:
        if empty_line_seen and defn.allow_zero_when_empty_line:
            return FieldResolution(
                field=code,
                selected_value=Decimal("0"),
                detection_status="detected_zero",
                confidence=0.7,
                selection_reason="Ligne présente sans montant — zéro comptable",
                raw_label=empty_line_label,
                source=defn.sections[0] if defn.sections else None,
            )
        return FieldResolution(field=code, detection_status="not_detected")

    best = candidates[0]
    # Conflit : deux candidats proches en score, valeurs différentes
    status = "detected"
    reason = (
        f"Label={best.raw_label!r} colonne={best.column} "
        f"score={best.score:.2f} méthode={best.match_method}"
    )
    if len(candidates) > 1:
        second = candidates[1]
        if (
            best.value is not None
            and second.value is not None
            and best.value != second.value
            and abs(best.score - second.score) < 0.08
        ):
            status = "ambiguous"
            reason += f" | conflit avec {second.value} (score={second.score:.2f})"

    confidence = min(0.99, max(0.35, best.score))
    model_confidence = None
    extraction_confidence = best.score
    cross_source_agreement = 1.0 if best.match_method == "exact_code" else 0.65
    conflict_penalty = 0.0
    if status == "ambiguous":
        confidence = min(confidence, 0.55)
        conflict_penalty = 0.25
    if best.column in ("unknown", None):
        confidence = min(confidence, 0.65)
    if best.observation and best.observation.get("model_confidence") is not None:
        model_confidence = float(best.observation["model_confidence"])
        confidence = round((confidence + model_confidence) / 2, 4)

    logger.info(
        "field_resolve code=%s value=%s status=%s reason=%s",
        code,
        best.value,
        status,
        reason,
    )

    return FieldResolution(
        field=code,
        candidates=candidates[:5],
        selected_value=best.value,
        selection_reason=reason,
        detection_status=status if best.value is not None else (
            "detected_zero" if empty_line_seen and defn.allow_zero_when_empty_line else "not_detected"
        ),
        confidence=confidence if best.value is not None else 0.0,
        raw_label=best.raw_label,
        column=best.column,
        page=best.page,
        source=best.source,
        model_confidence=model_confidence,
        extraction_confidence=extraction_confidence,
        cross_source_agreement=cross_source_agreement,
        conflict_penalty=conflict_penalty,
    )


def resolve_all_fields(
    observations: list[RawFinancialObservation],
) -> dict[str, FieldResolution]:
    """Résout tous les champs du registre, puis applique les agrégats sum_components."""
    resolved: dict[str, FieldResolution] = {}
    for code in FIELD_DEFINITIONS:
        resolved[code] = resolve_field(code, observations)

    # Agrégats déterministes (composantes → total)
    for code, defn in FIELD_DEFINITIONS.items():
        if defn.aggregation != "sum_components" or not defn.component_codes:
            continue
        direct = resolved.get(code)
        components = [resolved.get(c) for c in defn.component_codes]
        component_values = [
            c.selected_value for c in components if c and c.selected_value is not None
        ]
        if not component_values:
            continue
        calculated = sum_decimal(*component_values)
        # Préférer le total explicite s'il a un bon score ; sinon somme
        if (
            direct
            and direct.selected_value is not None
            and direct.detection_status in ("detected", "ambiguous")
            and direct.confidence >= 0.7
            and direct.raw_label
            and "total" in normalize_label(direct.raw_label or "")
        ):
            direct.calculated_value = calculated
            if calculated is not None and abs(direct.selected_value - calculated) > Decimal("1"):
                direct.validation_status = "divergent"
                direct.detection_status = "conflicting"
                direct.confidence = min(direct.confidence, 0.6)
            else:
                direct.validation_status = "consistent"
            resolved[code] = direct
        else:
            # Utiliser la somme des composantes
            resolved[code] = FieldResolution(
                field=code,
                candidates=direct.candidates if direct else [],
                selected_value=calculated,
                calculated_value=calculated,
                detection_status="derived",
                confidence=0.9,
                selection_reason=(
                    "Somme des composantes : "
                    + " + ".join(defn.component_codes)
                ),
                validation_status="consistent",
                source=defn.sections[0] if defn.sections else None,
            )

    return resolved


def extract_document_metadata(
    observations: list[RawFinancialObservation],
    page_payloads: list[dict] | None = None,
) -> DocumentMetadata:
    """Consolide les métadonnées d'identification depuis les premières pages."""
    meta = DocumentMetadata()
    # Depuis payloads.metadata
    if page_payloads:
        for payload in page_payloads[:8]:
            block = payload.get("metadata") or {}
            for key in (
                "reference",
                "entreprise",
                "identification_fiscale",
                "exercice",
                "date_debut_exercice",
                "date_fin_exercice",
            ):
                if getattr(meta, key) is None and block.get(key):
                    setattr(meta, key, str(block[key]).strip())

    # Depuis observations IDENTIFICATION / alias
    for field_name, aliases in METADATA_ALIASES.items():
        if getattr(meta, field_name) is not None:
            continue
        for obs in observations:
            if obs.section not in ("IDENTIFICATION", "AUTRE", None):
                if obs.page > 5:
                    continue
            score = max(label_similarity(obs.raw_label, a) for a in aliases)
            if score >= 0.7 and obs.raw_value:
                setattr(meta, field_name, obs.raw_value.strip())
                break
    return meta
