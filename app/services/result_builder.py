"""Construction du LiasseExtractionResult à partir des résolutions de champs."""
from __future__ import annotations

from app.schemas.liasse import (
    FinancialElement,
    FieldValidation,
    LiasseExtractionResult,
    RawComponent,
    ScoringInput,
)
from app.schemas.observations import DocumentMetadata, FieldResolution
from app.services.amount_parser import decimal_to_float
from app.services.field_definitions import FIELD_DEFINITIONS
from app.services.liasse_extraction import ELEMENTS_19, SCORING_METRICS

_VALID_SECTIONS = ("BILAN_ACTIF", "BILAN_PASSIF", "CPC")

_REQUIRED_DIRECT = [
    "ACTIFS_IMMOBILISES",
    "TOTAL_BILAN",
    "CHIFFRE_AFFAIRES",
    "DETTES_BANCAIRES_MLT",
    "DETTES_BANCAIRES_CT",
    "PASSIF_CIRCULANT",
    "DETTES_FOURNISSEURS",
    "COMPTE_COURANT_ASSOCIES",
    "TRESORERIE_PASSIF",
    "ACTIF_CIRCULANT",
    "CREANCES_CLIENTS",
    "TRESORERIE_ACTIF",
    "ACHATS_REVENDUS",
    "AUTRES_CHARGES",
    "CHARGES_INTERETS",
    "RESULTAT_NET",
    "FONDS_PROPRES",
]


def _float(resolution: FieldResolution | None) -> float | None:
    if not resolution:
        return None
    return decimal_to_float(resolution.selected_value)


def build_extraction_result(
    *,
    resolved: dict[str, FieldResolution],
    metadata: DocumentMetadata,
    sections_detected: dict[str, bool],
    pages_total: int,
    pages_analyzed: int,
    elapsed_ms: int,
    filename: str | None,
    extra_warnings: list[str] | None = None,
    field_provenance: dict | None = None,
) -> LiasseExtractionResult:
    financial_elements: list[FinancialElement] = []

    for num, code, label, source in ELEMENTS_19:
        res = resolved.get(code)
        if code == "TYPE_RESULTAT":
            note = res.selection_reason if res else None
            financial_elements.append(
                FinancialElement(
                    number=num,
                    code=code,
                    label=label,
                    value=None,
                    source=source,
                    note=note,
                    confidence=res.confidence if res else 0.0,
                    detection_status=res.detection_status if res else "not_detected",
                    selection_reason=res.selection_reason if res else None,
                )
            )
            continue

        validation = None
        if res and res.validation_status:
            validation = FieldValidation(status=res.validation_status)

        financial_elements.append(
            FinancialElement(
                number=num,
                code=code,
                label=label,
                value=_float(res),
                source=res.source or source if res else source,
                confidence=res.confidence if res else 0.0,
                detection_status=res.detection_status if res else "not_detected",
                page=res.page if res else None,
                raw_label=res.raw_label if res else None,
                column=res.column if res else None,
                selection_reason=res.selection_reason if res else None,
                validation=validation,
            )
        )

    raw_components: list[RawComponent] = []
    for code, (label, source, feeds) in SCORING_METRICS.items():
        res = resolved.get(code)
        if res and res.selected_value is not None:
            raw_components.append(
                RawComponent(
                    label=label,
                    value=float(res.selected_value),
                    source=source,
                    feeds=feeds,
                )
            )

    # Scoring input
    scoring_input = ScoringInput(
        chiffre_affaires=_float(resolved.get("CHIFFRE_AFFAIRES")),
        ca_export=_float(resolved.get("CA_EXPORT")),
        ca_n1=_float(resolved.get("CA_N1")),
        total_bilan=_float(resolved.get("TOTAL_BILAN")),
        fonds_propres=_float(resolved.get("FONDS_PROPRES")),
        actifs_immobilises=_float(resolved.get("ACTIFS_IMMOBILISES")),
        actif_circulant=_float(resolved.get("ACTIF_CIRCULANT")),
        clients=_float(resolved.get("CREANCES_CLIENTS")),
        fournisseurs=_float(resolved.get("DETTES_FOURNISSEURS")),
        dettes_financieres=_float(resolved.get("DETTES_FINANCIERES")),
        dettes_bancaires_ct=_float(resolved.get("DETTES_BANCAIRES_CT")),
        passif_circulant=_float(resolved.get("PASSIF_CIRCULANT")),
        tresorerie_actif=_float(resolved.get("TRESORERIE_ACTIF")),
        tresorerie_passif=_float(resolved.get("TRESORERIE_PASSIF")),
        tresorerie_nette=_float(resolved.get("TRESORERIE_NETTE")),
        achats=_float(resolved.get("ACHATS_REVENDUS")),
        frais_financiers=_float(resolved.get("CHARGES_INTERETS")),
        amortissements=_float(resolved.get("AMORTISSEMENTS")),
        caf=_float(resolved.get("CAF")),
        fdr=_float(resolved.get("FDR")),
        resultat_net=_float(resolved.get("RESULTAT_NET")),
        compte_courant_associes=_float(resolved.get("COMPTE_COURANT_ASSOCIES")),
        encours_leasing=_float(resolved.get("ENCOURS_LEASING")),
        cmt=_float(resolved.get("CMT")),
        nouveau_financement=_float(resolved.get("NOUVEAU_FINANCEMENT")),
    )

    # Complétude : champs directs réellement détectés (pas ambiguous invalide)
    detected_direct = 0
    for code in _REQUIRED_DIRECT:
        res = resolved.get(code)
        if not res:
            continue
        if res.selected_value is not None and res.detection_status in (
            "detected",
            "detected_zero",
            "derived",
        ):
            detected_direct += 1
    # TYPE_RESULTAT
    type_ok = 1 if resolved.get("TYPE_RESULTAT") else 0
    completeness = round(
        100.0 * (detected_direct + type_ok) / (len(_REQUIRED_DIRECT) + 1), 1
    )

    sections_extraction_complete = {}
    sections_validated = {}
    for section in _VALID_SECTIONS:
        codes = [
            c
            for c, d in FIELD_DEFINITIONS.items()
            if section in d.sections and d.element_number is not None
        ]
        if not codes:
            sections_extraction_complete[section] = False
            sections_validated[section] = False
            continue
        filled = sum(
            1
            for c in codes
            if resolved.get(c)
            and resolved[c].selected_value is not None
            and resolved[c].detection_status
            in ("detected", "detected_zero", "derived")
        )
        sections_extraction_complete[section] = filled >= max(1, len(codes) // 2)
        validated = sum(
            1
            for c in codes
            if resolved.get(c) and resolved[c].validation_status == "consistent"
        )
        sections_validated[section] = validated > 0 and validated >= filled // 2

    # Rétrocompat : sections_completeness = extraction réellement utile
    sections_completeness = {
        s: bool(sections_detected.get(s)) and bool(sections_extraction_complete.get(s))
        for s in _VALID_SECTIONS
    }

    warnings = list(extra_warnings or [])
    warnings.insert(
        0,
        f"Pipeline observation/résolution — {pages_analyzed}/{pages_total} page(s).",
    )

    return LiasseExtractionResult(
        reference=metadata.reference,
        entreprise=metadata.entreprise,
        identification_fiscale=metadata.identification_fiscale,
        exercice=metadata.exercice,
        date_debut_exercice=metadata.date_debut_exercice,
        date_fin_exercice=metadata.date_fin_exercice,
        document_kind="LIASSE_OCR",
        elements=financial_elements,
        raw_components=raw_components,
        scoring_input=scoring_input,
        sections_completeness=sections_completeness,
        sections_detected=sections_detected,
        sections_extraction_complete=sections_extraction_complete,
        sections_validated=sections_validated,
        completeness_pct=completeness,
        warnings=warnings,
        pages_total=pages_total,
        pages_analyzed=pages_analyzed,
        processing_time_ms=elapsed_ms,
        source_filename=filename,
        field_provenance=field_provenance,
        document_summary=(
            f"Liasse {filename or ''} — {pages_analyzed}/{pages_total} pages, "
            f"complétude {completeness}%."
        ),
    )
