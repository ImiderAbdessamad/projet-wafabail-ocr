"""Résolution déterministe des candidats Qwen → FinancialValue."""
from __future__ import annotations

import logging
import unicodedata
from collections import defaultdict
from decimal import Decimal

from app.config import OLLAMA_MAPPING_MODEL
from app.schemas.financial_analysis import DataStatus, FinancialDataset, FinancialValue, ValueProvenance
from app.schemas.financial_mapping import FinancialCandidate, FinancialMappingOutput
from app.services.financial_dataset_builder import _apply_simple_derived, empty_dataset
from app.services.financial_normalizer import is_explicit_zero, normalize_label, parse_decimal_amount

logger = logging.getLogger(__name__)

_NATURE_PRIORITY: dict[str, int] = {
    "GRAND_TOTAL": 400,
    "SECTION_TOTAL": 300,
    "SUBTOTAL": 200,
    "DETAIL": 100,
    "DERIVED_DISPLAYED": 50,
    "UNKNOWN": 0,
}
_PRIORITY_TIE_MARGIN = 5.0
_AMOUNT_TOLERANCE = Decimal("1.00")
_N1_CODE_MAP = {
    "CHIFFRE_AFFAIRES": "CHIFFRE_AFFAIRES_N1",
    "RESULTAT_NET": "RESULTAT_NET_N1",
    "TOTAL_BILAN": "TOTAL_BILAN_N1",
    "FONDS_PROPRES": "FONDS_PROPRES_N1",
}
_ACTIF_CURRENT_FIELDS = {
    "TOTAL_ACTIF",
    "TOTAL_BILAN",
    "ACTIFS_IMMOBILISES",
    "ACTIF_CIRCULANT",
    "STOCKS",
    "CLIENTS",
    "TRESORERIE_ACTIF",
}
_PASSIF_CURRENT_FIELDS = {
    "TOTAL_PASSIF",
    "TOTAL_BILAN",
    "FONDS_PROPRES",
    "DETTES_FINANCIERES",
    "PASSIF_CIRCULANT",
    "FOURNISSEURS",
    "TRESORERIE_PASSIF",
}
_CPC_CURRENT_FIELDS = {
    "CHIFFRE_AFFAIRES",
    "RESULTAT_NET",
    "RESULTAT_NET_XIII",
    "RESULTAT_NET_XVI",
    "RESULTAT_EXPLOITATION",
    "PRODUITS_EXPLOITATION",
    "CHARGES_EXPLOITATION",
    "PRODUITS_FINANCIERS",
    "CHARGES_FINANCIERES",
    "RESULTAT_FINANCIER",
    "RESULTAT_COURANT",
    "PRODUITS_NON_COURANTS",
    "CHARGES_NON_COURANTES",
    "RESULTAT_NON_COURANT",
    "RESULTAT_AVANT_IMPOT",
    "IMPOT_SUR_RESULTATS",
    "CHARGES_INTERETS",
    "ACHATS_REVENDUS",
    "ACHATS_CONSOMMES",
}

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


def _fold(text: str) -> str:
    normalized = normalize_label(text or "")
    decomposed = unicodedata.normalize("NFKD", normalized)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def validate_candidate_scope(candidate: FinancialCandidate) -> list[str]:
    _, reasons = candidate_is_eligible(candidate)
    return reasons


def canonicalize_candidate_period(candidate: FinancialCandidate) -> FinancialCandidate:
    if candidate.period != "N_MINUS_1":
        return candidate
    mapped_code = _N1_CODE_MAP.get(str(candidate.field_code))
    if mapped_code is None:
        return candidate
    return candidate.model_copy(update={"field_code": mapped_code})


def candidate_is_eligible(candidate: FinancialCandidate) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    label = _fold(candidate.evidence.raw_label)
    column = _fold(candidate.evidence.column_name or "")
    section = candidate.evidence.section
    field_code = str(candidate.field_code)

    if field_code == "UNKNOWN":
        reasons.append("Code UNKNOWN non résolvable.")

    if field_code.endswith("_N1"):
        if candidate.period != "N_MINUS_1":
            reasons.append("Un champ N-1 exige period=N_MINUS_1.")
    else:
        if candidate.period != "N":
            reasons.append("Un champ courant exige period=N.")

    if section == "BILAN_ACTIF" and candidate.period == "N" and field_code in _ACTIF_CURRENT_FIELDS:
        if "net" not in column:
            reasons.append("Le bilan actif courant exige la colonne Net.")
        if "precedent" in column or "brut" in column:
            reasons.append("La colonne Brut ou N-1 est interdite.")

    if section == "BILAN_PASSIF" and candidate.period == "N" and field_code in _PASSIF_CURRENT_FIELDS:
        if "precedent" in column:
            reasons.append("La colonne Exercice précédent est interdite.")

    if section == "CPC" and candidate.period == "N" and field_code in _CPC_CURRENT_FIELDS:
        valid_total_column = any(token in column for token in ("total", "totaux", "3 = 1 + 2", "3=1+2"))
        if not valid_total_column and column != "exercice":
            reasons.append("Le CPC courant exige la colonne Totaux de l'exercice.")

    if field_code == "STOCKS":
        if section != "BILAN_ACTIF":
            reasons.append("STOCKS hors BILAN_ACTIF.")
        if "variation" in label:
            reasons.append("Une variation de stocks n'est pas le solde des stocks.")

    if field_code == "CLIENTS":
        if section != "BILAN_ACTIF":
            reasons.append("CLIENTS hors BILAN_ACTIF.")
        if "crediteur" in label:
            reasons.append("Clients créditeurs exclus du champ CLIENTS.")
        if "clients et comptes rattaches" not in label and label != "clients":
            reasons.append("CLIENTS exige le libellé Clients et comptes rattachés.")

    if field_code == "FOURNISSEURS":
        if section != "BILAN_PASSIF":
            reasons.append("FOURNISSEURS hors BILAN_PASSIF.")
        if "debiteur" in label:
            reasons.append("Fournisseurs débiteurs exclus du champ FOURNISSEURS.")
        if "fournisseurs et comptes rattaches" not in label and label != "fournisseurs":
            reasons.append("FOURNISSEURS exige le libellé Fournisseurs et comptes rattachés.")

    if field_code == "DETTES_FINANCIERES":
        if section != "BILAN_PASSIF":
            reasons.append("DETTES_FINANCIERES hors BILAN_PASSIF.")
        if any(token in label for token in ("augmentation des dettes", "diminution des dettes", "ecart de conversion")):
            reasons.append("Ligne d'écart de conversion exclue.")

    if field_code in {"RESULTAT_NET", "RESULTAT_NET_XIII", "RESULTAT_NET_XVI"}:
        if section not in {"BILAN_PASSIF", "CPC"}:
            reasons.append("RESULTAT_NET hors section autorisée.")
        if "instance d affectation" in label or "instance daffectation" in label:
            reasons.append("Résultat en instance d'affectation exclu.")

    if field_code == "CHARGES_INTERETS":
        if section != "CPC":
            reasons.append("CHARGES_INTERETS hors CPC.")
        if "autres charges financieres" in label:
            reasons.append("Autres charges financières non assimilées automatiquement aux charges d'intérêts.")

    if field_code == "CHIFFRE_AFFAIRES" and section != "CPC":
        reasons.append("CHIFFRE_AFFAIRES hors CPC.")

    if field_code == "ENCOURS_LEASING" and "redevance" in label:
        reasons.append("Une redevance de crédit-bail n'est pas un encours.")

    if field_code == "REDEVANCES_CREDIT_BAIL" and "encours" in label and "redevance" not in label:
        reasons.append("Encours exclu du champ REDEVANCES_CREDIT_BAIL.")

    return not reasons, reasons


def candidate_priority(candidate: FinancialCandidate) -> float:
    score = float(_NATURE_PRIORITY.get(candidate.nature, 0))
    score += float(candidate.confidence) * 20.0
    column = _fold(candidate.evidence.column_name or "")
    label = _fold(candidate.evidence.raw_label)
    section = candidate.evidence.section
    field_code = str(candidate.field_code)

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
        "RESULTAT_NET_XIII": {"CPC"},
        "RESULTAT_NET_XVI": {"CPC"},
        "CHARGES_INTERETS": {"CPC"},
        "ACHATS_REVENDUS": {"CPC", "DETAIL_CPC"},
        "ACHATS_CONSOMMES": {"CPC", "DETAIL_CPC"},
        "REDEVANCES_CREDIT_BAIL": {"DETAIL_CPC", "CPC"},
    }
    expected = expected_sections.get(field_code)
    if expected and section in expected:
        score += 100.0

    if candidate.period == "N" and not field_code.endswith("_N1"):
        score += 80.0
    if candidate.period == "N_MINUS_1" and field_code.endswith("_N1"):
        score += 80.0

    if section == "BILAN_ACTIF":
        if candidate.period == "N" and "net" in column and "precedent" not in column:
            score += 60.0
        if candidate.period == "N_MINUS_1" and ("precedent" in column or "n 1" in column or "n1" in column):
            score += 60.0

    if section == "BILAN_PASSIF":
        if candidate.period == "N" and "exercice" in column and "precedent" not in column:
            score += 60.0

    if section == "CPC" and candidate.period == "N":
        if any(token in column for token in ("totaux", "total", "3 = 1 + 2", "3=1+2")):
            score += 60.0

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
    for hint in exact_hints.get(field_code, ()):
        if hint in label:
            score += 50.0
            break

    if field_code == "CHIFFRE_AFFAIRES":
        if "chiffre d affaires" in label:
            score += 40.0
        if "ventes de marchandises" in label and "chiffre" not in label:
            score -= 40.0

    if field_code in {"RESULTAT_NET", "RESULTAT_NET_XIII", "RESULTAT_NET_XVI"} and section == "CPC":
        if "xiii" in label or "xvi" in label:
            score += 50.0
        score += 30.0

    if field_code == "RESULTAT_NET" and section == "BILAN_PASSIF":
        score -= 20.0

    return score


def group_candidates_by_field(outputs: list[FinancialMappingOutput]) -> dict[str, list[FinancialCandidate]]:
    grouped: dict[str, list[FinancialCandidate]] = defaultdict(list)
    for output in outputs:
        for candidate in output.candidates:
            candidate = canonicalize_candidate_period(candidate)
            if candidate.field_code == "UNKNOWN":
                continue
            if str(candidate.field_code) == "RESULTAT_NET" and candidate.evidence.section == "CPC":
                label = _fold(candidate.evidence.raw_label)
                if "xiii" in label:
                    candidate = candidate.model_copy(update={"field_code": "RESULTAT_NET_XIII"})
                elif "xvi" in label:
                    candidate = candidate.model_copy(update={"field_code": "RESULTAT_NET_XVI"})
            grouped[str(candidate.field_code)].append(candidate)
    return grouped


def _parse_candidate_amount(candidate: FinancialCandidate) -> tuple[Decimal | None, list[str]]:
    raw = candidate.raw_value
    if raw is None or str(raw).strip() == "":
        return None, ["raw_value absent — non converti en zéro."]
    amount = parse_decimal_amount(raw)
    if amount is None:
        if is_explicit_zero(raw):
            return Decimal("0"), []
        return None, [f"Montant non parsable : {raw!r}"]
    return amount, []


def _to_provenance(candidate: FinancialCandidate) -> ValueProvenance:
    return ValueProvenance(
        page_number=candidate.evidence.page_number,
        raw_label=candidate.evidence.raw_label,
        raw_value=candidate.raw_value or candidate.evidence.raw_value,
        column_name=candidate.evidence.column_name,
        extraction_method="qwen_mapping",
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
    label = FIELD_ATTR_META.get(code, (code, code))[1]
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


def _eligible_sorted(candidates: list[FinancialCandidate]) -> list[tuple[FinancialCandidate, Decimal]]:
    prepared: list[tuple[FinancialCandidate, Decimal, float]] = []
    for candidate in candidates:
        ok, reasons = candidate_is_eligible(candidate)
        if not ok:
            logger.warning("Candidate %s rejected: %s", candidate.field_code, "; ".join(reasons))
            continue
        amount, parse_warnings = _parse_candidate_amount(candidate)
        if amount is None:
            logger.warning("Candidate %s invalid: %s", candidate.field_code, "; ".join(parse_warnings))
            continue
        prepared.append((candidate, amount, candidate_priority(candidate)))
    prepared.sort(key=lambda item: item[2], reverse=True)
    return [(c, a) for c, a, _ in prepared]


def _pick_best(candidates: list[FinancialCandidate]) -> FinancialValue | None:
    ranked = _eligible_sorted(candidates)
    if not ranked:
        return None

    best_candidate, best_amount = ranked[0]
    best_priority = candidate_priority(best_candidate)
    tied = [
        (candidate, amount)
        for candidate, amount in ranked
        if abs(candidate_priority(candidate) - best_priority) <= _PRIORITY_TIE_MARGIN
    ]
    tied_amounts = {amount for _, amount in tied}
    code = str(best_candidate.field_code)

    if len(tied_amounts) > 1:
        return _financial_value(
            code,
            None,
            "conflicting",
            [_to_provenance(candidate) for candidate, _ in tied],
            warnings=[
                "Candidats Qwen de priorité équivalente avec montants divergents."
            ],
        )

    warnings: list[str] = []
    if len(ranked) > 1:
        warnings.append(
            f"{len(ranked) - 1} candidat(s) Qwen de priorité inférieure conservé(s) uniquement dans l'audit."
        )
    return _financial_value(
        code,
        best_amount,
        "confirmed",
        [_to_provenance(best_candidate)],
        warnings=warnings,
    )


def _resolve_total_bilan(grouped: dict[str, list[FinancialCandidate]]) -> FinancialValue:
    total_actif = _pick_best(
        [c for c in grouped.get("TOTAL_ACTIF", []) if c.evidence.section == "BILAN_ACTIF" and c.period == "N"]
    )
    total_passif = _pick_best(
        [c for c in grouped.get("TOTAL_PASSIF", []) if c.evidence.section == "BILAN_PASSIF" and c.period == "N"]
    )
    if (
        total_actif is None
        or total_passif is None
        or total_actif.value is None
        or total_passif.value is None
        or total_actif.status not in {"confirmed", "derived"}
        or total_passif.status not in {"confirmed", "derived"}
    ):
        return _financial_value("TOTAL_BILAN", None, "missing", [])

    tolerance = max(Decimal("1.00"), abs(total_actif.value) * Decimal("0.0001"))
    difference = abs(total_actif.value - total_passif.value)
    if difference > tolerance:
        logger.error("TOTAL_ACTIF and TOTAL_PASSIF conflict")
        return _financial_value(
            "TOTAL_BILAN",
            None,
            "conflicting",
            total_actif.provenance + total_passif.provenance,
            warnings=["Total actif et total passif sont différents."],
        )

    logger.info("Resolved TOTAL_BILAN=%s from ACTIF/PASSIF agreement", total_actif.value)
    return _financial_value(
        "TOTAL_BILAN",
        total_actif.value,
        "confirmed",
        total_actif.provenance + total_passif.provenance,
    )


def _resolve_resultat_net(grouped: dict[str, list[FinancialCandidate]]) -> FinancialValue | None:
    xiii = _eligible_sorted(grouped.get("RESULTAT_NET_XIII", []))
    xvi = _eligible_sorted(grouped.get("RESULTAT_NET_XVI", []))
    cpc = _eligible_sorted(grouped.get("RESULTAT_NET", []))
    cpc = [(c, a) for c, a in cpc if c.evidence.section == "CPC" and c.period == "N"]
    bilan = _eligible_sorted(grouped.get("RESULTAT_NET", []))
    bilan = [(c, a) for c, a in bilan if c.evidence.section == "BILAN_PASSIF" and c.period == "N"]

    if xiii and xvi:
        c13, v13 = xiii[0]
        c16, v16 = xvi[0]
        provenances = [_to_provenance(c13), _to_provenance(c16)]
        if _amounts_agree(v13, v16):
            if bilan and not _amounts_agree(v13, bilan[0][1]):
                return _financial_value(
                    "RESULTAT_NET",
                    None,
                    "conflicting",
                    provenances + [_to_provenance(bilan[0][0])],
                    warnings=["Résultat net CPC ≠ bilan passif."],
                )
            return _financial_value("RESULTAT_NET", v13, "confirmed", provenances)
        return _financial_value(
            "RESULTAT_NET",
            None,
            "conflicting",
            provenances,
            warnings=[f"XIII {v13} ≠ XVI {v16}"],
        )

    if cpc:
        best_c, best_a = cpc[0]
        if bilan and not _amounts_agree(best_a, bilan[0][1]):
            return _financial_value(
                "RESULTAT_NET",
                None,
                "conflicting",
                [_to_provenance(best_c), _to_provenance(bilan[0][0])],
                warnings=["Résultat net CPC ≠ bilan passif."],
            )
        return _financial_value("RESULTAT_NET", best_a, "confirmed", [_to_provenance(best_c)])

    if bilan:
        c, a = bilan[0]
        return _financial_value("RESULTAT_NET", a, "confirmed", [_to_provenance(c)])
    return None


def _resolve_special(code: str, grouped: dict[str, list[FinancialCandidate]]) -> FinancialValue | None:
    if code == "RESULTAT_NET":
        return _resolve_resultat_net(grouped)
    candidates = list(grouped.get(code, []))
    if code == "STOCKS":
        candidates = [c for c in candidates if c.nature == "SECTION_TOTAL"] or candidates
    if code == "DETTES_FINANCIERES":
        candidates = [
            c
            for c in candidates
            if c.nature in {"SECTION_TOTAL", "SUBTOTAL", "GRAND_TOTAL"}
            or "total" in _fold(c.evidence.raw_label)
            or _fold(c.evidence.raw_label).startswith("dettes de financement")
        ] or candidates
    if code == "CHIFFRE_AFFAIRES":
        candidates = [c for c in candidates if "chiffre d affaires" in _fold(c.evidence.raw_label)] or candidates
    if code == "ENCOURS_LEASING" and not _eligible_sorted(candidates):
        return _financial_value("ENCOURS_LEASING", None, "missing", [])
    return _pick_best(candidates)


def resolve_financial_candidates(outputs: list[FinancialMappingOutput]) -> dict[str, FinancialValue]:
    grouped = group_candidates_by_field(outputs)
    resolved: dict[str, FinancialValue] = {}

    for code in (
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
    ):
        fv = _resolve_special(code, grouped)
        if fv is not None:
            resolved[code] = fv

    resolved["TOTAL_BILAN"] = _resolve_total_bilan(grouped)

    for code, candidates in grouped.items():
        if code in resolved or code in {"RESULTAT_NET_XIII", "RESULTAT_NET_XVI", "UNKNOWN"}:
            continue
        fv = _pick_best(candidates)
        if fv is not None:
            resolved[code] = fv

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

    if "ENCOURS_LEASING" not in resolved:
        resolved["ENCOURS_LEASING"] = _financial_value(
            "ENCOURS_LEASING",
            None,
            "missing",
            [],
            warnings=["Aucun encours leasing documenté."],
        )
    return resolved


def build_financial_dataset_from_resolved_values(resolved: dict[str, FinancialValue]) -> FinancialDataset:
    dataset = empty_dataset()
    warnings: list[str] = []
    for code, fv in resolved.items():
        meta = FIELD_ATTR_META.get(code)
        if not meta:
            continue
        attr, _label = meta
        if hasattr(dataset, attr):
            setattr(dataset, attr, fv)
        if fv.status == "conflicting":
            warnings.append(f"{code} conflicting.")
        warnings.extend(fv.warnings)

    if "ACHATS_TOTAL" in resolved and resolved["ACHATS_TOTAL"].value is not None:
        dataset.achats = resolved["ACHATS_TOTAL"]

    dataset = _apply_simple_derived(dataset)
    dataset.warnings = warnings
    return dataset
