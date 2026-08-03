"""Résolution Python des candidats GLM Vision → FinancialDataset (Decimal)."""
from __future__ import annotations

import logging
import re
from decimal import Decimal

from app.config import DIRECT_FINANCIAL_MODEL
from app.schemas.direct_financial_extraction import DirectFinancialCandidate
from app.schemas.financial_analysis import FinancialDataset, FinancialValue, ValueProvenance
from app.schemas.financial_mapping import (
    FinancialCandidate,
    FinancialMappingOutput,
    MappingEvidence,
)
from app.services.financial_candidate_resolver import (
    build_financial_dataset_from_resolved_values,
    canonicalize_column_role,
    resolve_financial_candidates,
)
from app.services.financial_normalizer import normalize_label, parse_decimal_amount

logger = logging.getLogger(__name__)

_TOTAL_GENERAL_RE = re.compile(
    r"\btotal\s+(?:general\s+)?i\s*(?:\+|\s)\s*ii\s*(?:\+|\s)\s*iii\b",
    re.IGNORECASE,
)

_AMOUNT_LIKE_RE = re.compile(
    r"^[\s.\u00a0]*-?\d{1,3}(?:[.\s\u00a0]\d{3})*(?:[,.]\d{1,2})?\s*$"
)

# Libellés canoniques quand GLM met le montant dans raw_label
_CANONICAL_LABELS: dict[str, str] = {
    "TOTAL_ACTIF": "TOTAL GENERAL I+II+III",
    "TOTAL_PASSIF": "TOTAL I+II+III",
    "ACTIFS_IMMOBILISES": "TOTAL I",
    "ACTIF_CIRCULANT": "TOTAL II",
    "TRESORERIE_ACTIF": "TOTAL TRESORERIE-ACTIF",
    "TRESORERIE_PASSIF": "TOTAL III TRESORERIE-PASSIF",
    "FONDS_PROPRES": "TOTAL DES CAPITAUX PROPRES",
    "DETTES_FINANCIERES": "TOTAL DES DETTES DE FINANCEMENT",
    "PASSIF_CIRCULANT": "TOTAL DU PASSIF CIRCULANT",
    "STOCKS": "TOTAL STOCKS",
    "CLIENTS": "Clients et comptes rattachés",
    "FOURNISSEURS": "Fournisseurs et comptes rattachés",
    "CHIFFRE_AFFAIRES": "Chiffre d'affaires",
    "RESULTAT_NET": "Résultat net",
    "RESULTAT_EXPLOITATION": "Résultat d'exploitation",
    "CHARGES_FINANCIERES": "Charges financières",
    "RESULTAT_COURANT": "Résultat courant",
}

_DEFAULT_COLUMN_ROLE: dict[str, str] = {
    "IDENTIFICATION": "IDENTITY_VALUE",
    "BILAN_ACTIF": "NET_N",
    "BILAN_PASSIF": "EXERCICE_N",
    "CPC": "TOTAL_EXERCICE_N",
    "DETAIL_CPC": "EXERCICE_N",
    "RESULTAT_FISCAL": "MONTANT_N",
    "ESG": "MONTANT_N",
}

# Mapping page_type → section financial_mapping
_PAGE_TO_SECTION = {
    "IDENTIFICATION": "IDENTIFICATION",
    "BILAN_ACTIF": "BILAN_ACTIF",
    "BILAN_PASSIF": "BILAN_PASSIF",
    "CPC": "CPC",
    "DETAIL_CPC": "DETAIL_CPC",
    "RESULTAT_FISCAL": "RESULTAT_FISCAL",
    "ESG": "AUTRE",
}


def _map_column_role(role: str) -> str:
    allowed = {
        "BRUT",
        "AMORT_PROV",
        "NET_N",
        "EXERCICE_N",
        "TOTAL_EXERCICE_N",
        "EXERCICE_N1",
        "UNKNOWN",
    }
    if role in allowed:
        return role
    if role in {"MONTANT_N", "IDENTITY_VALUE"}:
        return "EXERCICE_N"
    if role == "MONTANT_N1":
        return "EXERCICE_N1"
    return "UNKNOWN"


def _label_is_amount_like(label: str, raw_value: str) -> bool:
    folded = normalize_label(label or "")
    value = normalize_label(raw_value or "")
    if not folded:
        return True
    if value and folded == value:
        return True
    return bool(_AMOUNT_LIKE_RE.match(folded))


def repair_direct_candidate(
    candidate: DirectFinancialCandidate,
) -> DirectFinancialCandidate:
    """Corrige les défauts fréquents de GLM Flash (libellé=montant, role UNKNOWN)."""
    page_type = candidate.evidence.page_type
    field_code = candidate.field_code
    evidence = candidate.evidence
    nature = candidate.nature
    warnings = list(candidate.warnings)
    updates: dict = {}

    role = evidence.column_role
    if role in {"UNKNOWN", "MONTANT_N", "IDENTITY_VALUE"} or not role:
        default_role = _DEFAULT_COLUMN_ROLE.get(page_type)
        if candidate.period == "N_MINUS_1":
            default_role = "EXERCICE_N1"
        if default_role and default_role != role:
            # IDENTITY_VALUE non supporté côté mapping → EXERCICE_N
            mapped = "EXERCICE_N" if default_role == "IDENTITY_VALUE" else default_role
            if mapped == "MONTANT_N":
                mapped = "EXERCICE_N"
            role = mapped  # type: ignore[assignment]
            warnings.append(f"column_role défaut {page_type} → {role}")

    raw_label = evidence.raw_label
    if _label_is_amount_like(raw_label, candidate.raw_value):
        amount = parse_decimal_amount(candidate.raw_value)
        # Ne pas inventer un libellé métier pour un 0,00 ambigu
        # (sinon conflit artificiel avec le vrai TOTAL V).
        if amount == Decimal("0") and field_code in {
            "CHARGES_FINANCIERES",
            "CHARGES_INTERETS",
            "FRAIS_FINANCIERS",
        }:
            pass
        else:
            canon = _CANONICAL_LABELS.get(field_code)
            if canon:
                raw_label = canon
                warnings.append("raw_label reconstitué (GLM avait mis le montant).")

    if field_code in {"TOTAL_ACTIF", "TOTAL_PASSIF"}:
        folded_label = normalize_label(raw_label)
        # TOTAL I / TOTAL II seuls ne doivent PAS devenir le total général.
        is_partial = bool(
            re.search(r"\btotal\s+(i|ii|iii)\b", folded_label, re.I)
            and not _TOTAL_GENERAL_RE.search(folded_label)
        )
        if not is_partial:
            if nature != "GRAND_TOTAL":
                nature = "GRAND_TOTAL"
                warnings.append("nature forcée GRAND_TOTAL pour total général.")
            if _label_is_amount_like(evidence.raw_label, candidate.raw_value):
                raw_label = _CANONICAL_LABELS[field_code]
                warnings.append("libellé total général reconstitué.")

    if field_code in {
        "ACTIFS_IMMOBILISES",
        "ACTIF_CIRCULANT",
        "FONDS_PROPRES",
        "DETTES_FINANCIERES",
        "PASSIF_CIRCULANT",
        "TRESORERIE_ACTIF",
        "TRESORERIE_PASSIF",
    } and nature == "DETAIL":
        # Totaux de section souvent renvoyés en DETAIL par GLM
        nature = "SECTION_TOTAL"
        warnings.append("nature DETAIL → SECTION_TOTAL (total de section).")

    excerpt = evidence.source_excerpt or ""
    if excerpt.strip() in {"", "ligne|montant", "ligne|montant"}:
        excerpt = f"{raw_label}|{candidate.raw_value}"

    if (
        role != evidence.column_role
        or raw_label != evidence.raw_label
        or nature != candidate.nature
        or excerpt != evidence.source_excerpt
    ):
        updates["evidence"] = evidence.model_copy(
            update={
                "column_role": role,
                "raw_label": raw_label[:180],
                "source_excerpt": excerpt[:240],
            }
        )
        updates["nature"] = nature
        updates["warnings"] = warnings
        return candidate.model_copy(update=updates)
    return candidate


def repair_direct_candidates(
    candidates: list[DirectFinancialCandidate],
) -> list[DirectFinancialCandidate]:
    return [repair_direct_candidate(c) for c in candidates]


def to_mapping_candidate(candidate: DirectFinancialCandidate) -> FinancialCandidate | None:
    """Convertit un candidat GLM direct vers le type interne commun."""
    page_type = candidate.evidence.page_type
    if page_type in {"AUTRE", "VIDE", "ESG"}:
        # ESG : on conserve CAF si présent via RESULTAT_FISCAL/AUTRE mapping
        if page_type == "ESG" and candidate.field_code == "CAF":
            section = "RESULTAT_FISCAL"
        elif page_type == "ESG":
            return None
        else:
            return None
    else:
        section = _PAGE_TO_SECTION.get(page_type, "AUTRE")

    # RESULTAT_COMPTABLE → alias RESULTAT_NET si bilan/cpc pas déjà
    field_code = candidate.field_code
    if field_code == "RESULTAT_COMPTABLE":
        field_code = "RESULTAT_NET"
    if field_code == "RESULTAT_NET_ESG":
        field_code = "RESULTAT_NET"
    # Dataset N-1 : TOTAL_ACTIF/PASSIF → TOTAL_BILAN_N1 via mapping existant
    if field_code in {"TOTAL_ACTIF", "TOTAL_PASSIF"} and candidate.period == "N_MINUS_1":
        field_code = "TOTAL_BILAN"

    try:
        evidence = MappingEvidence(
            page_number=candidate.evidence.page_number,
            section=section,  # type: ignore[arg-type]
            raw_label=candidate.evidence.raw_label[:180],
            column_name=candidate.evidence.column_name,
            column_role=_map_column_role(candidate.evidence.column_role),  # type: ignore[arg-type]
            source_excerpt=candidate.evidence.source_excerpt[:240],
        )
        return FinancialCandidate(
            field_code=field_code,
            raw_value=candidate.raw_value,
            period=candidate.period,  # type: ignore[arg-type]
            nature=candidate.nature,  # type: ignore[arg-type]
            confidence=candidate.confidence,
            evidence=evidence,
            warnings=list(candidate.warnings),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Candidat direct ignoré : %s", exc)
        return None


def dedupe_direct_candidates(
    candidates: list[DirectFinancialCandidate],
) -> list[DirectFinancialCandidate]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[DirectFinancialCandidate] = []
    for candidate in candidates:
        key = (
            candidate.field_code,
            candidate.period,
            normalize_label(candidate.evidence.raw_label),
            normalize_label(candidate.raw_value),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _patch_provenance(
    resolved: dict[str, FinancialValue],
    originals: list[DirectFinancialCandidate],
) -> None:
    by_key: dict[tuple[str, str, str], DirectFinancialCandidate] = {}
    for cand in originals:
        by_key[
            (
                cand.field_code,
                normalize_label(cand.evidence.raw_label),
                normalize_label(cand.raw_value),
            )
        ] = cand

    for fv in resolved.values():
        new_prov: list[ValueProvenance] = []
        for prov in fv.provenance:
            match = by_key.get(
                (
                    fv.code if fv.code not in {"RESULTAT_NET_XIII", "RESULTAT_NET_XVI"} else "RESULTAT_NET",
                    normalize_label(prov.raw_label or ""),
                    normalize_label(prov.raw_value or ""),
                )
            )
            # Recherche plus large
            if match is None:
                for cand in originals:
                    if normalize_label(cand.raw_value) == normalize_label(prov.raw_value or ""):
                        if normalize_label(cand.evidence.raw_label) == normalize_label(
                            prov.raw_label or ""
                        ):
                            match = cand
                            break

            orientation = match.evidence.orientation if match else None
            page_type = match.evidence.page_type if match else prov.section
            column_role = (
                match.evidence.column_role if match else None
            )
            new_prov.append(
                prov.model_copy(
                    update={
                        "extraction_method": "glm_direct_vision",
                        "mapping_model": DIRECT_FINANCIAL_MODEL,
                        "page_type": page_type,
                        "orientation": orientation,
                        "column_role": column_role,
                    }
                )
            )
        fv.provenance = new_prov


def resolve_direct_financial_candidates(
    candidates: list[DirectFinancialCandidate],
) -> dict[str, FinancialValue]:
    """Sélectionne les valeurs finales à partir des candidats GLM Vision."""
    cleaned = dedupe_direct_candidates(repair_direct_candidates(candidates))
    mapped: list[FinancialCandidate] = []
    for candidate in cleaned:
        converted = to_mapping_candidate(candidate)
        if converted is None:
            continue
        converted = canonicalize_column_role(converted)
        mapped.append(converted)

    by_section: dict[str, list[FinancialCandidate]] = {}
    for cand in mapped:
        by_section.setdefault(cand.evidence.section, []).append(cand)

    outputs = [
        FinancialMappingOutput(section=section, candidates=items)  # type: ignore[arg-type]
        for section, items in by_section.items()
    ]
    resolved = resolve_financial_candidates(outputs)
    _patch_provenance(resolved, cleaned)
    return resolved


def apply_direct_derived_fields(dataset: FinancialDataset) -> FinancialDataset:
    """Dérivations Decimal complémentaires (FDR / BFDR / trésorerie nette)."""
    warnings = list(dataset.warnings or [])

    ta = dataset.tresorerie_actif
    tp = dataset.tresorerie_passif
    if (
        ta.status in {"confirmed", "derived"}
        and tp.status in {"confirmed", "derived"}
        and ta.value is not None
        and tp.value is not None
    ):
        net = ta.value - tp.value
        if dataset.tresorerie_nette.value is None or dataset.tresorerie_nette.status == "missing":
            dataset.tresorerie_nette = FinancialValue(
                code="TRESORERIE_NETTE",
                label="Trésorerie nette",
                value=net,
                status="derived",
                provenance=ta.provenance + tp.provenance,
                warnings=["Dérivé : tresorerie_actif - tresorerie_passif"],
            )

    # FDR = capitaux permanents - actifs immobilisés (si dispo)
    # Approximation : fonds_propres + dettes_financieres ≈ capitaux permanents
    fp = dataset.fonds_propres
    df = dataset.dettes_financieres
    ai = dataset.actifs_immobilises
    if (
        fp.status in {"confirmed", "derived"}
        and fp.value is not None
        and ai is not None
        and ai.status in {"confirmed", "derived"}
        and ai.value is not None
    ):
        permanents = fp.value
        if df.status in {"confirmed", "derived"} and df.value is not None:
            permanents = fp.value + df.value
        fdr = permanents - ai.value
        if dataset.fdr.value is None or dataset.fdr.status == "missing":
            dataset.fdr = FinancialValue(
                code="FDR",
                label="Fonds de roulement",
                value=fdr,
                status="derived",
                provenance=fp.provenance + (ai.provenance or []),
                warnings=["Dérivé : capitaux_permanents - actifs_immobilises"],
            )

    if (
        dataset.fdr.status in {"confirmed", "derived"}
        and dataset.fdr.value is not None
        and dataset.tresorerie_nette.status in {"confirmed", "derived"}
        and dataset.tresorerie_nette.value is not None
    ):
        if dataset.bfdr.value is None or dataset.bfdr.status == "missing":
            dataset.bfdr = FinancialValue(
                code="BFDR",
                label="Besoin en fonds de roulement",
                value=dataset.fdr.value - dataset.tresorerie_nette.value,
                status="derived",
                provenance=dataset.fdr.provenance + dataset.tresorerie_nette.provenance,
                warnings=["Dérivé : fdr - tresorerie_nette"],
            )

    # Achats = revendus + consommés (si non déjà posé)
    if dataset.achats_revendus and dataset.achats_consommes:
        rev = dataset.achats_revendus
        cons = dataset.achats_consommes
        if (
            rev.status in {"confirmed", "derived"}
            and cons.status in {"confirmed", "derived"}
            and rev.value is not None
            and cons.value is not None
        ):
            if dataset.achats.value is None or dataset.achats.status == "missing":
                dataset.achats = FinancialValue(
                    code="ACHATS_TOTAL",
                    label="Achats",
                    value=rev.value + cons.value,
                    status="derived",
                    provenance=rev.provenance + cons.provenance,
                    warnings=["Dérivé : achats_revendus + achats_consommes"],
                )

    dataset.warnings = warnings
    return dataset


def build_dataset_from_direct_candidates(
    candidates: list[DirectFinancialCandidate],
) -> FinancialDataset:
    resolved = resolve_direct_financial_candidates(candidates)
    dataset = build_financial_dataset_from_resolved_values(resolved)
    return apply_direct_derived_fields(dataset)


def is_total_general_label(label: str) -> bool:
    cleaned = normalize_label(label)
    cleaned = re.sub(r"[*_`#]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return bool(_TOTAL_GENERAL_RE.search(cleaned))


def parse_candidate_amount(raw_value: str) -> Decimal | None:
    return parse_decimal_amount(raw_value)
