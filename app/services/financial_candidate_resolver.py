"""Résolution déterministe des candidats Qwen → FinancialValue.

La confiance LLM n'est jamais le premier critère.
Aucune valeur absente n'est remplacée par zéro.
"""
from __future__ import annotations

import logging
import unicodedata
from collections import defaultdict
from decimal import Decimal

from app.config import OLLAMA_MAPPING_MODEL
from app.schemas.financial_analysis import (
    DataStatus,
    FinancialDataset,
    FinancialValue,
    ValueProvenance,
)
from app.schemas.financial_mapping import (
    FinancialCandidate,
    FinancialMappingOutput,
)
from app.services.financial_dataset_builder import (
    _apply_simple_derived,
    empty_dataset,
)
from app.services.financial_normalizer import (
    is_explicit_zero,
    normalize_label,
    parse_decimal_amount,
)

logger = logging.getLogger(__name__)

_NATURE_PRIORITY: dict[str, int] = {
    "GRAND_TOTAL": 400,
    "SECTION_TOTAL": 300,
    "SUBTOTAL": 200,
    "DETAIL": 100,
    "DERIVED_DISPLAYED": 50,
    "UNKNOWN": 0,
}

# Codes → (attribut FinancialDataset, libellé humain)
FIELD_ATTR_META: dict[str, tuple[str, str]] = {
    "CHIFFRE_AFFAIRES": ("chiffre_affaires", "Chiffre d'affaires"),
    "CHIFFRE_AFFAIRES_N1": ("chiffre_affaires_n1", "Chiffre d'affaires N-1"),
    "RESULTAT_NET": ("resultat_net", "Résultat net"),
    "RESULTAT_NET_N1": ("resultat_net_n1", "Résultat net N-1"),
    "RESULTAT_EXPLOITATION": ("resultat_exploitation", "Résultat d'exploitation"),
    "TOTAL_BILAN": ("total_bilan", "Total bilan"),
    "TOTAL_BILAN_N1": ("total_bilan_n1", "Total bilan N-1"),
    "TOTAL_ACTIF": ("total_actif", "Total actif"),
    "TOTAL_PASSIF": ("total_passif", "Total passif"),
    "FONDS_PROPRES": ("fonds_propres", "Fonds propres"),
    "FONDS_PROPRES_N1": ("fonds_propres_n1", "Fonds propres N-1"),
    "ACTIFS_IMMOBILISES": ("actifs_immobilises", "Actifs immobilisés"),
    "ACTIF_CIRCULANT": ("actif_circulant", "Actif circulant"),
    "PASSIF_CIRCULANT": ("passif_circulant", "Passif circulant"),
    "STOCKS": ("stocks", "Stocks"),
    "CLIENTS": ("clients", "Clients"),
    "FOURNISSEURS": ("fournisseurs", "Fournisseurs"),
    "TRESORERIE_ACTIF": ("tresorerie_actif", "Trésorerie actif"),
    "TRESORERIE_PASSIF": ("tresorerie_passif", "Trésorerie passif"),
    "DETTES_FINANCIERES": ("dettes_financieres", "Dettes financières"),
    "DETTES_BANCAIRES_CT": ("dettes_bancaires_ct", "Dettes bancaires CT"),
    "PRODUITS_EXPLOITATION": ("produits_exploitation", "Produits d'exploitation"),
    "CHARGES_EXPLOITATION": ("charges_exploitation", "Charges d'exploitation"),
    "PRODUITS_FINANCIERS": ("produits_financiers", "Produits financiers"),
    "CHARGES_FINANCIERES": ("charges_financieres", "Charges financières"),
    "RESULTAT_FINANCIER": ("resultat_financier", "Résultat financier"),
    "RESULTAT_COURANT": ("resultat_courant", "Résultat courant"),
    "PRODUITS_NON_COURANTS": ("produits_non_courants", "Produits non courants"),
    "CHARGES_NON_COURANTES": ("charges_non_courantes", "Charges non courantes"),
    "RESULTAT_NON_COURANT": ("resultat_non_courant", "Résultat non courant"),
    "RESULTAT_AVANT_IMPOT": ("resultat_avant_impot", "Résultat avant impôt"),
    "IMPOT_SUR_RESULTATS": ("impot_sur_resultats", "Impôt sur les résultats"),
    "ACHATS_REVENDUS": ("achats_revendus", "Achats revendus"),
    "ACHATS_CONSOMMES": ("achats_consommes", "Achats consommés"),
    "ACHATS_TOTAL": ("achats", "Achats total"),
    "CHARGES_INTERETS": ("frais_financiers", "Charges d'intérêts"),
    "DOTATIONS_AMORTISSEMENTS": ("amortissements", "Dotations amortissements"),
    "CAF": ("caf", "CAF"),
    "FDR": ("fdr", "Fonds de roulement"),
    "BFDR": ("bfdr", "BFDR"),
    "RESULTAT_FISCAL": ("resultat_fiscal", "Résultat fiscal"),
    "REINTEGRATIONS": ("reintegrations", "Réintégrations"),
    "DEDUCTIONS": ("deductions", "Déductions"),
    "IS_DU": ("is_du", "IS dû"),
    "COTISATION_MINIMALE": ("cotisation_minimale", "Cotisation minimale"),
    "REPORT_DEFICITAIRE": ("report_deficitaire", "Report déficitaire"),
    "REDEVANCES_CREDIT_BAIL": ("redevances_credit_bail", "Redevances crédit-bail"),
    "ENCOURS_LEASING": ("encours_leasing", "Encours leasing"),
    "CMT": ("cmt", "CMT"),
    "NOUVEAU_FINANCEMENT": ("nouveau_financement", "Nouveau financement"),
}

_AMOUNT_TOLERANCE = Decimal("1.00")


def _fold(text: str) -> str:
    """Normalise accents pour comparaisons de libellés."""
    normalized = normalize_label(text or "")
    decomposed = unicodedata.normalize("NFKD", normalized)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def validate_candidate_scope(candidate: FinancialCandidate) -> list[str]:
    """Retourne les raisons d'inéligibilité liées au périmètre."""
    _, reasons = candidate_is_eligible(candidate)
    return reasons


def candidate_is_eligible(
    candidate: FinancialCandidate,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    label = _fold(candidate.evidence.raw_label)
    section = candidate.evidence.section
    field_code = candidate.field_code

    if field_code == "UNKNOWN":
        reasons.append("Code UNKNOWN non résolvable.")

    if field_code == "STOCKS":
        if section != "BILAN_ACTIF":
            reasons.append("STOCKS hors BILAN_ACTIF.")
        if "variation" in label:
            reasons.append(
                "Une variation de stocks n'est pas le solde des stocks."
            )

    if field_code == "CLIENTS":
        if section != "BILAN_ACTIF":
            reasons.append("CLIENTS hors BILAN_ACTIF.")
        if "crediteur" in label:
            reasons.append("Clients créditeurs exclus du champ CLIENTS.")

    if field_code == "FOURNISSEURS":
        if section != "BILAN_PASSIF":
            reasons.append("FOURNISSEURS hors BILAN_PASSIF.")
        if "debiteur" in label:
            reasons.append(
                "Fournisseurs débiteurs exclus du champ FOURNISSEURS."
            )

    if field_code == "DETTES_FINANCIERES":
        if section != "BILAN_PASSIF":
            reasons.append("DETTES_FINANCIERES hors BILAN_PASSIF.")
        if "augmentation des dettes" in label:
            reasons.append("Ligne d'écart de conversion exclue.")
        if "diminution des dettes" in label:
            reasons.append("Ligne d'écart de conversion exclue.")
        if "ecart de conversion" in label:
            reasons.append("Ligne d'écart de conversion exclue.")

    if field_code == "RESULTAT_NET":
        if section not in {"BILAN_PASSIF", "CPC"}:
            reasons.append("RESULTAT_NET hors section autorisée.")
        if "instance d affectation" in label or "instance daffectation" in label:
            reasons.append("Résultat en instance d'affectation exclu.")

    if field_code == "ENCOURS_LEASING":
        if "redevance" in label:
            reasons.append(
                "Une redevance de crédit-bail n'est pas un encours."
            )

    if field_code == "CHARGES_INTERETS":
        if "autres charges financieres" in label:
            reasons.append(
                "Autres charges financières non assimilées "
                "automatiquement aux charges d'intérêts."
            )
        if section not in {"CPC", "DETAIL_CPC"}:
            reasons.append("CHARGES_INTERETS hors CPC.")

    if field_code == "CHIFFRE_AFFAIRES":
        if section != "CPC":
            reasons.append("CHIFFRE_AFFAIRES hors CPC.")

    if field_code == "REDEVANCES_CREDIT_BAIL":
        if "encours" in label and "redevance" not in label:
            reasons.append("Encours exclu du champ REDEVANCES_CREDIT_BAIL.")

    # Les champs sans suffixe N1 représentent toujours N.
    if candidate.period != "N" and not str(field_code).endswith("_N1"):
        reasons.append("Période différente de l'exercice courant.")

    if str(field_code).endswith("_N1") and candidate.period == "N":
        reasons.append("Champ N-1 avec période N.")

    return not reasons, reasons


def candidate_priority(candidate: FinancialCandidate) -> float:
    """Priorité déterministe — confiance LLM = bonus secondaire seulement."""
    score = float(_NATURE_PRIORITY.get(candidate.nature, 0))
    score += float(candidate.confidence) * 20.0

    column = _fold(candidate.evidence.column_name or "")
    label = _fold(candidate.evidence.raw_label)
    section = candidate.evidence.section

    # Bonus section attendue
    expected_sections = {
        "STOCKS": {"BILAN_ACTIF"},
        "CLIENTS": {"BILAN_ACTIF"},
        "TRESORERIE_ACTIF": {"BILAN_ACTIF"},
        "ACTIF_CIRCULANT": {"BILAN_ACTIF"},
        "ACTIFS_IMMOBILISES": {"BILAN_ACTIF"},
        "TOTAL_ACTIF": {"BILAN_ACTIF"},
        "FOURNISSEURS": {"BILAN_PASSIF"},
        "DETTES_FINANCIERES": {"BILAN_PASSIF"},
        "FONDS_PROPRES": {"BILAN_PASSIF"},
        "PASSIF_CIRCULANT": {"BILAN_PASSIF"},
        "TRESORERIE_PASSIF": {"BILAN_PASSIF"},
        "TOTAL_PASSIF": {"BILAN_PASSIF"},
        "CHIFFRE_AFFAIRES": {"CPC"},
        "RESULTAT_NET": {"CPC", "BILAN_PASSIF"},
        "CHARGES_INTERETS": {"CPC", "DETAIL_CPC"},
        "ACHATS_REVENDUS": {"CPC", "DETAIL_CPC"},
        "ACHATS_CONSOMMES": {"CPC", "DETAIL_CPC"},
        "REDEVANCES_CREDIT_BAIL": {"DETAIL_CPC", "CPC"},
    }
    expected = expected_sections.get(candidate.field_code)
    if expected and section in expected:
        score += 100.0

    if candidate.period == "N" and not str(candidate.field_code).endswith("_N1"):
        score += 80.0
    if candidate.period == "N_MINUS_1" and str(candidate.field_code).endswith("_N1"):
        score += 80.0

    if section == "BILAN_ACTIF":
        if candidate.period == "N" and "net" in column and "precedent" not in column:
            score += 60.0
        if candidate.period == "N_MINUS_1" and (
            "precedent" in column or "n 1" in column or "n1" in column
        ):
            score += 60.0

    if section == "BILAN_PASSIF":
        if candidate.period == "N" and "exercice" in column and "precedent" not in column:
            score += 60.0
        if candidate.period == "N" and column in {"", "net", "exercice"}:
            score += 30.0

    if section == "CPC":
        if candidate.period == "N" and (
            "totaux" in column
            or "total" in column
            or "3 = 1 + 2" in column
            or "3=1+2" in column.replace(" ", "")
        ):
            score += 60.0
        # Bonus explicite colonne « totaux de l'exercice »
        if "totaux de l exercice" in column or "totaux de l'exercice" in column:
            score += 20.0

    # Libellés exacts attendus
    exact_hints = {
        "CLIENTS": ("clients et comptes rattaches",),
        "FOURNISSEURS": ("fournisseurs et comptes rattaches",),
        "CHIFFRE_AFFAIRES": ("chiffre d affaires",),
        "FONDS_PROPRES": ("total des capitaux propres",),
        "CHARGES_INTERETS": ("charges d interets",),
        "STOCKS": ("stocks",),
        "DETTES_FINANCIERES": ("dettes de financement", "total dettes de financement"),
        "TRESORERIE_ACTIF": ("total iii tresorerie actif", "tresorerie actif"),
        "TRESORERIE_PASSIF": ("total iii tresorerie passif", "tresorerie passif"),
    }
    for hint in exact_hints.get(candidate.field_code, ()):
        if hint in label:
            score += 50.0
            break

    # CPC : préférer la ligne CA plutôt que ventes de marchandises
    if candidate.field_code == "CHIFFRE_AFFAIRES":
        if "chiffre d affaires" in label:
            score += 40.0
        if "ventes de marchandises" in label and "chiffre" not in label:
            score -= 40.0

    # RESULTAT_NET : bonus XIII / XVI CPC
    if candidate.field_code == "RESULTAT_NET" and section == "CPC":
        if "xiii" in label or "xvi" in label:
            score += 50.0
        score += 30.0  # préférence CPC vs bilan

    if candidate.field_code == "RESULTAT_NET" and section == "BILAN_PASSIF":
        score -= 20.0

    return score


def group_candidates_by_field(
    outputs: list[FinancialMappingOutput],
) -> dict[str, list[FinancialCandidate]]:
    grouped: dict[str, list[FinancialCandidate]] = defaultdict(list)
    for output in outputs:
        for candidate in output.candidates:
            if candidate.field_code == "UNKNOWN":
                continue
            grouped[str(candidate.field_code)].append(candidate)
    return grouped


def _parse_candidate_amount(
    candidate: FinancialCandidate,
) -> tuple[Decimal | None, list[str]]:
    warnings: list[str] = []
    raw = candidate.raw_value
    if raw is None or str(raw).strip() == "":
        return None, ["raw_value absent — non converti en zéro."]

    amount = parse_decimal_amount(raw)
    if amount is None:
        if is_explicit_zero(raw):
            return Decimal("0"), []
        warnings.append(f"Montant non parsable : {raw!r}")
        return None, warnings
    return amount, warnings


def _to_provenance(
    candidate: FinancialCandidate,
    *,
    extraction_method: str = "qwen_mapping",
) -> ValueProvenance:
    return ValueProvenance(
        page_number=candidate.evidence.page_number,
        raw_label=candidate.evidence.raw_label,
        raw_value=candidate.raw_value or candidate.evidence.raw_value,
        column_name=candidate.evidence.column_name,
        extraction_method=extraction_method,
        confidence=Decimal(str(candidate.confidence)),
        source_excerpt=candidate.evidence.source_excerpt,
        section=candidate.evidence.section,
        nature=candidate.nature,
        period=candidate.period,
        mapping_model=OLLAMA_MAPPING_MODEL,
    )


def _financial_value(
    code: str,
    value: Decimal | None,
    status: DataStatus,
    provenances: list[ValueProvenance],
    warnings: list[str] | None = None,
) -> FinancialValue:
    meta = FIELD_ATTR_META.get(code)
    label = meta[1] if meta else code
    return FinancialValue(
        code=code,
        label=label,
        value=value,
        status=status,
        provenance=provenances,
        warnings=list(warnings or []),
    )


def _amounts_agree(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= max(_AMOUNT_TOLERANCE, abs(a) * Decimal("0.0001"))


def _eligible_sorted(
    candidates: list[FinancialCandidate],
) -> list[tuple[FinancialCandidate, Decimal]]:
    """Filtre, parse et trie par priorité décroissante."""
    prepared: list[tuple[FinancialCandidate, Decimal, float]] = []
    for candidate in candidates:
        ok, reasons = candidate_is_eligible(candidate)
        if not ok:
            logger.warning(
                "Candidate %s rejected: %s",
                candidate.field_code,
                "; ".join(reasons),
            )
            continue
        amount, parse_warnings = _parse_candidate_amount(candidate)
        if amount is None:
            if parse_warnings:
                logger.warning(
                    "Candidate %s invalid: %s",
                    candidate.field_code,
                    "; ".join(parse_warnings),
                )
            continue
        prepared.append((candidate, amount, candidate_priority(candidate)))
    prepared.sort(key=lambda item: item[2], reverse=True)
    return [(c, a) for c, a, _ in prepared]


def _pick_best(
    candidates: list[FinancialCandidate],
) -> FinancialValue | None:
    ranked = _eligible_sorted(candidates)
    if not ranked:
        return None
    best_c, best_a = ranked[0]
    code = str(best_c.field_code)
    # Conflit si plusieurs montants distincts en tête de priorité proche
    distinct = {amount for _, amount in ranked}
    provenances = [_to_provenance(c) for c, _ in ranked]
    if len(distinct) == 1:
        return _financial_value(code, best_a, "confirmed", provenances[:1])
    # Même nature/priorité avec montants différents → conflicting
    top_priority = candidate_priority(best_c)
    close = [
        (c, a)
        for c, a in ranked
        if abs(candidate_priority(c) - top_priority) < 1.0 or a != best_a
    ]
    close_amounts = {a for _, a in close}
    if len(close_amounts) > 1:
        return _financial_value(
            code,
            None,
            "conflicting",
            [_to_provenance(c) for c, _ in close],
            warnings=["Candidats éligibles avec montants divergents."],
        )
    return _financial_value(code, best_a, "confirmed", [_to_provenance(best_c)])


def _resolve_total_bilan(
    grouped: dict[str, list[FinancialCandidate]],
) -> FinancialValue | None:
    actifs = [
        (c, a)
        for c, a in _eligible_sorted(
            grouped.get("TOTAL_ACTIF", []) + grouped.get("TOTAL_BILAN", [])
        )
        if c.evidence.section == "BILAN_ACTIF"
        and c.period == "N"
        and c.nature in {"GRAND_TOTAL", "SECTION_TOTAL", "UNKNOWN"}
    ]
    # Préférer TOTAL_ACTIF GRAND_TOTAL colonne net
    actifs_pref = [
        (c, a)
        for c, a in actifs
        if c.field_code == "TOTAL_ACTIF" or "total general" in _fold(c.evidence.raw_label)
    ] or actifs

    passifs = [
        (c, a)
        for c, a in _eligible_sorted(
            grouped.get("TOTAL_PASSIF", []) + grouped.get("TOTAL_BILAN", [])
        )
        if c.evidence.section == "BILAN_PASSIF"
        and c.period == "N"
        and c.nature in {"GRAND_TOTAL", "SECTION_TOTAL", "UNKNOWN"}
    ]

    if not actifs_pref and not passifs:
        return None

    if actifs_pref and passifs:
        a_c, a_v = actifs_pref[0]
        p_c, p_v = passifs[0]
        provenances = [_to_provenance(a_c), _to_provenance(p_c)]
        if _amounts_agree(a_v, p_v):
            logger.info(
                "Resolved TOTAL_BILAN=%s from ACTIF/PASSIF agreement",
                a_v,
            )
            return _financial_value(
                "TOTAL_BILAN",
                a_v,
                "confirmed",
                provenances,
            )
        logger.error("TOTAL_ACTIF and TOTAL_PASSIF conflict")
        return _financial_value(
            "TOTAL_BILAN",
            None,
            "conflicting",
            provenances,
            warnings=[
                f"Actif {a_v} ≠ Passif {p_v}",
            ],
        )

    if actifs_pref:
        c, v = actifs_pref[0]
        return _financial_value("TOTAL_BILAN", v, "confirmed", [_to_provenance(c)])
    c, v = passifs[0]
    return _financial_value("TOTAL_BILAN", v, "confirmed", [_to_provenance(c)])


def _resolve_resultat_net(
    grouped: dict[str, list[FinancialCandidate]],
) -> FinancialValue | None:
    cpc = [
        (c, a)
        for c, a in _eligible_sorted(grouped.get("RESULTAT_NET", []))
        if c.evidence.section == "CPC" and c.period == "N"
    ]
    xiii = [
        (c, a)
        for c, a in cpc
        if "xiii" in _fold(c.evidence.raw_label)
    ]
    xvi = [
        (c, a)
        for c, a in cpc
        if "xvi" in _fold(c.evidence.raw_label)
    ]

    if xiii and xvi:
        (_, v13), (c16, v16) = xiii[0], xvi[0]
        c13 = xiii[0][0]
        provenances = [_to_provenance(c13), _to_provenance(c16)]
        if _amounts_agree(v13, v16):
            return _financial_value("RESULTAT_NET", v13, "confirmed", provenances)
        return _financial_value(
            "RESULTAT_NET",
            None,
            "conflicting",
            provenances,
            warnings=[f"XIII {v13} ≠ XVI {v16}"],
        )

    if cpc:
        # Contrôle secondaire éventuel avec bilan passif
        best_c, best_a = cpc[0]
        bilan = [
            (c, a)
            for c, a in _eligible_sorted(grouped.get("RESULTAT_NET", []))
            if c.evidence.section == "BILAN_PASSIF" and c.period == "N"
        ]
        if bilan and not _amounts_agree(best_a, bilan[0][1]):
            return _financial_value(
                "RESULTAT_NET",
                None,
                "conflicting",
                [_to_provenance(best_c), _to_provenance(bilan[0][0])],
                warnings=["Résultat net CPC ≠ bilan passif."],
            )
        return _financial_value(
            "RESULTAT_NET",
            best_a,
            "confirmed",
            [_to_provenance(best_c)],
        )

    bilan_only = [
        (c, a)
        for c, a in _eligible_sorted(grouped.get("RESULTAT_NET", []))
        if c.evidence.section == "BILAN_PASSIF" and c.period == "N"
    ]
    if bilan_only:
        c, a = bilan_only[0]
        return _financial_value("RESULTAT_NET", a, "confirmed", [_to_provenance(c)])
    return None


def _resolve_special(
    code: str,
    grouped: dict[str, list[FinancialCandidate]],
) -> FinancialValue | None:
    if code == "TOTAL_BILAN":
        return _resolve_total_bilan(grouped)
    if code == "RESULTAT_NET":
        return _resolve_resultat_net(grouped)

    candidates = list(grouped.get(code, []))

    # Règles de filtre supplémentaires avant pick
    if code == "CLIENTS":
        candidates = [
            c
            for c in candidates
            if "clients et comptes rattaches" in _fold(c.evidence.raw_label)
            or _fold(c.evidence.raw_label) == "clients"
        ] or candidates
    if code == "FOURNISSEURS":
        candidates = [
            c
            for c in candidates
            if "fournisseurs et comptes rattaches" in _fold(c.evidence.raw_label)
            or _fold(c.evidence.raw_label) == "fournisseurs"
        ] or candidates
    if code == "STOCKS":
        # Préférer SECTION_TOTAL
        section_totals = [c for c in candidates if c.nature == "SECTION_TOTAL"]
        if section_totals:
            candidates = section_totals
    if code == "DETTES_FINANCIERES":
        totals = [
            c
            for c in candidates
            if c.nature in {"SECTION_TOTAL", "SUBTOTAL", "GRAND_TOTAL"}
            or "total" in _fold(c.evidence.raw_label)
            or _fold(c.evidence.raw_label).startswith("dettes de financement")
        ]
        if totals:
            candidates = totals
    if code == "CHIFFRE_AFFAIRES":
        ca_lines = [
            c
            for c in candidates
            if "chiffre d affaires" in _fold(c.evidence.raw_label)
        ]
        if ca_lines:
            candidates = ca_lines
    if code == "ENCOURS_LEASING":
        # Jamais depuis redevance — déjà filtré ; sinon missing
        ranked = _eligible_sorted(candidates)
        if not ranked:
            return _financial_value("ENCOURS_LEASING", None, "missing", [])
        return _pick_best(candidates)

    return _pick_best(candidates)


def resolve_financial_candidates(
    outputs: list[FinancialMappingOutput],
) -> dict[str, FinancialValue]:
    """Résout les candidats Qwen en FinancialValue (autorité Python)."""
    grouped = group_candidates_by_field(outputs)
    resolved: dict[str, FinancialValue] = {}

    # Codes avec résolution spécialisée en premier
    special_codes = {
        "TOTAL_BILAN",
        "TOTAL_ACTIF",
        "TOTAL_PASSIF",
        "RESULTAT_NET",
        "CLIENTS",
        "FOURNISSEURS",
        "STOCKS",
        "DETTES_FINANCIERES",
        "TRESORERIE_ACTIF",
        "TRESORERIE_PASSIF",
        "CHIFFRE_AFFAIRES",
        "CHARGES_INTERETS",
        "ENCOURS_LEASING",
        "REDEVANCES_CREDIT_BAIL",
    }

    for code in special_codes:
        if code in {"TOTAL_ACTIF", "TOTAL_PASSIF"}:
            fv = _pick_best(
                [
                    c
                    for c in grouped.get(code, [])
                    if c.period == "N"
                ]
            )
        else:
            fv = _resolve_special(code, grouped)
        if fv is not None:
            resolved[code] = fv

    # TOTAL_ACTIF / PASSIF séparés pour contrôles
    if "TOTAL_ACTIF" not in resolved:
        actifs = [
            c
            for c in grouped.get("TOTAL_ACTIF", []) + grouped.get("TOTAL_BILAN", [])
            if c.evidence.section == "BILAN_ACTIF" and c.period == "N"
        ]
        fv = _pick_best(actifs)
        if fv is not None:
            fv = _financial_value(
                "TOTAL_ACTIF",
                fv.value,
                fv.status,
                fv.provenance,
                fv.warnings,
            )
            resolved["TOTAL_ACTIF"] = fv
    if "TOTAL_PASSIF" not in resolved:
        passifs = [
            c
            for c in grouped.get("TOTAL_PASSIF", []) + grouped.get("TOTAL_BILAN", [])
            if c.evidence.section == "BILAN_PASSIF" and c.period == "N"
        ]
        fv = _pick_best(passifs)
        if fv is not None:
            fv = _financial_value(
                "TOTAL_PASSIF",
                fv.value,
                fv.status,
                fv.provenance,
                fv.warnings,
            )
            resolved["TOTAL_PASSIF"] = fv

    # Autres champs génériques
    for code, candidates in grouped.items():
        if code in resolved or code in special_codes:
            continue
        if code == "UNKNOWN":
            continue
        fv = _pick_best(candidates)
        if fv is not None:
            resolved[code] = fv

    # ACHATS_TOTAL dérivé
    rev = resolved.get("ACHATS_REVENDUS")
    cons = resolved.get("ACHATS_CONSOMMES")
    if (
        rev
        and cons
        and rev.status in {"confirmed", "derived"}
        and cons.status in {"confirmed", "derived"}
        and rev.value is not None
        and cons.value is not None
    ):
        resolved["ACHATS_TOTAL"] = _financial_value(
            "ACHATS_TOTAL",
            rev.value + cons.value,
            "derived",
            rev.provenance + cons.provenance,
            warnings=["Dérivé : ACHATS_REVENDUS + ACHATS_CONSOMMES"],
        )

    # ENCOURS_LEASING absent → missing explicite (jamais depuis redevance)
    if "ENCOURS_LEASING" not in resolved:
        resolved["ENCOURS_LEASING"] = _financial_value(
            "ENCOURS_LEASING",
            None,
            "missing",
            [],
            warnings=["Aucun encours leasing documenté."],
        )

    # Alias CHARGES_INTERETS → aussi exposé si frais_financiers attendu
    if "CHARGES_INTERETS" in resolved and "FRAIS_FINANCIERS" not in resolved:
        resolved["FRAIS_FINANCIERS"] = resolved["CHARGES_INTERETS"]

    return resolved


def merge_resolved_values(
    llm_values: dict[str, FinancialValue],
    deterministic_values: dict[str, FinancialValue],
) -> dict[str, FinancialValue]:
    """Fusion hybrid : divergence → conflicting, pas d'écrasement silencieux."""
    merged: dict[str, FinancialValue] = {}
    codes = set(llm_values) | set(deterministic_values)

    for code in codes:
        llm = llm_values.get(code)
        det = deterministic_values.get(code)

        def _usable(fv: FinancialValue | None) -> bool:
            return (
                fv is not None
                and fv.status in {"confirmed", "derived"}
                and fv.value is not None
            )

        if _usable(llm) and _usable(det):
            assert llm is not None and det is not None
            assert llm.value is not None and det.value is not None
            if _amounts_agree(llm.value, det.value):
                merged[code] = FinancialValue(
                    code=code,
                    label=llm.label,
                    value=llm.value,
                    status="confirmed",
                    provenance=llm.provenance + det.provenance,
                    warnings=llm.warnings + det.warnings,
                )
            else:
                merged[code] = FinancialValue(
                    code=code,
                    label=llm.label or det.label,
                    value=None,
                    status="conflicting",
                    provenance=llm.provenance + det.provenance,
                    warnings=[
                        f"Divergence hybrid LLM={llm.value} vs déterministe={det.value}"
                    ],
                )
        elif _usable(llm):
            merged[code] = llm  # type: ignore[assignment]
        elif _usable(det):
            merged[code] = det  # type: ignore[assignment]
        elif llm is not None and llm.status == "conflicting":
            merged[code] = llm
        elif det is not None and det.status == "conflicting":
            merged[code] = det
        elif llm is not None:
            merged[code] = llm
        elif det is not None:
            merged[code] = det

    return merged


def financial_values_from_dataset(
    dataset: FinancialDataset,
) -> dict[str, FinancialValue]:
    """Extrait un dict code → FinancialValue depuis un dataset déterministe."""
    out: dict[str, FinancialValue] = {}
    for code, (attr, _label) in FIELD_ATTR_META.items():
        fv = getattr(dataset, attr, None)
        if fv is None:
            continue
        if isinstance(fv, FinancialValue) and fv.status != "missing":
            # Harmoniser extraction_method
            for prov in fv.provenance:
                if not prov.extraction_method or prov.extraction_method == "markdown_ocr":
                    prov.extraction_method = "deterministic_parser"
            out[code] = fv
    return out


def build_financial_dataset_from_resolved_values(
    resolved: dict[str, FinancialValue],
) -> FinancialDataset:
    """Construit un FinancialDataset à partir des valeurs résolues."""
    dataset = empty_dataset()
    warnings: list[str] = []

    for code, fv in resolved.items():
        meta = FIELD_ATTR_META.get(code)
        if not meta:
            continue
        attr, _label = meta
        if not hasattr(dataset, attr):
            continue
        current = getattr(dataset, attr)
        # Optionnels : None jusqu'à affectation
        if current is None or isinstance(current, FinancialValue):
            setattr(dataset, attr, fv)
        if fv.status == "conflicting":
            warnings.append(f"{code} conflicting.")
        if fv.warnings:
            warnings.extend(fv.warnings)

    # ACHATS_TOTAL → achats si présent
    if "ACHATS_TOTAL" in resolved and resolved["ACHATS_TOTAL"].value is not None:
        dataset.achats = resolved["ACHATS_TOTAL"]

    dataset = _apply_simple_derived(dataset)
    dataset.warnings = warnings
    return dataset
