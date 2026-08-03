"""Résolution déterministe des candidats Qwen → FinancialValue."""
from __future__ import annotations

import logging
import re
import unicodedata
from collections import defaultdict
from decimal import Decimal

from app.config import OLLAMA_MAPPING_MODEL
from app.schemas.financial_analysis import DataStatus, FinancialDataset, FinancialValue, ValueProvenance
from app.schemas.financial_mapping import FinancialCandidate, FinancialMappingOutput
from app.services.financial_dataset_builder import _apply_simple_derived, empty_dataset
from app.services.financial_normalizer import (
    is_explicit_zero,
    is_mixed_comma_ocr_amount,
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
_PRIORITY_TIE_MARGIN = 5.0
_AMOUNT_TOLERANCE = Decimal("1.00")

# Mapping interne période N-1 → code dataset (jamais exposé à Qwen).
_N1_CODE_MAP = {
    "CHIFFRE_AFFAIRES": "CHIFFRE_AFFAIRES_N1",
    "RESULTAT_NET": "RESULTAT_NET_N1",
    "TOTAL_BILAN": "TOTAL_BILAN_N1",
    "FONDS_PROPRES": "FONDS_PROPRES_N1",
    "DETTES_FINANCIERES": "DETTES_FINANCIERES_N1",
}

ALLOWED_FIELDS_BY_SECTION: dict[str, set[str]] = {
    "BILAN_ACTIF": {
        "TOTAL_ACTIF",
        "TOTAL_BILAN",
        "ACTIFS_IMMOBILISES",
        "ACTIF_CIRCULANT",
        "STOCKS",
        "CLIENTS",
        "TRESORERIE_ACTIF",
    },
    "BILAN_PASSIF": {
        "TOTAL_PASSIF",
        "TOTAL_BILAN",
        "FONDS_PROPRES",
        "RESULTAT_NET",
        "DETTES_FINANCIERES",
        "PASSIF_CIRCULANT",
        "FOURNISSEURS",
        "TRESORERIE_PASSIF",
    },
    "CPC": {
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
        "ACHATS_REVENDUS",
        "ACHATS_CONSOMMES",
        "CHARGES_INTERETS",
        "DOTATIONS_AMORTISSEMENTS",
        "PRODUITS_CESSION_IMMOBILISATIONS",
        "VALEUR_NETTE_IMMOBILISATIONS_CEDEES",
    },
    "DETAIL_CPC": {
        "REDEVANCES_CREDIT_BAIL",
    },
    "RESULTAT_FISCAL": {
        "RESULTAT_FISCAL",
        "REINTEGRATIONS",
        "DEDUCTIONS",
        "IS_DU",
        "COTISATION_MINIMALE",
        "REPORT_DEFICITAIRE",
    },
}

_TOTAL_GENERAL_RE = re.compile(
    (
        r"\btotal\s+"
        r"(?:general\s+)?"
        r"i\s*(?:\+|\s)\s*ii\s*(?:\+|\s)\s*iii\b"
    ),
    re.IGNORECASE,
)

SUPPORTED_PERIODS_BY_FIELD: dict[str, set[str]] = {
    "CHIFFRE_AFFAIRES": {"N", "N_MINUS_1"},
    "RESULTAT_NET": {"N", "N_MINUS_1"},
    "FONDS_PROPRES": {"N", "N_MINUS_1"},
    "TOTAL_ACTIF": {"N", "N_MINUS_1"},
    "TOTAL_PASSIF": {"N", "N_MINUS_1"},
    "TOTAL_BILAN": {"N", "N_MINUS_1"},
    "DETTES_FINANCIERES": {"N"},
    "PASSIF_CIRCULANT": {"N"},
    "ACTIF_CIRCULANT": {"N"},
    "ACTIFS_IMMOBILISES": {"N"},
    "STOCKS": {"N"},
    "CLIENTS": {"N"},
    "FOURNISSEURS": {"N"},
    "TRESORERIE_ACTIF": {"N"},
    "TRESORERIE_PASSIF": {"N"},
    "RESULTAT_EXPLOITATION": {"N"},
    "PRODUITS_EXPLOITATION": {"N"},
    "CHARGES_EXPLOITATION": {"N"},
    "PRODUITS_FINANCIERS": {"N"},
    "CHARGES_FINANCIERES": {"N"},
    "RESULTAT_FINANCIER": {"N"},
    "RESULTAT_COURANT": {"N"},
    "PRODUITS_NON_COURANTS": {"N"},
    "CHARGES_NON_COURANTES": {"N"},
    "RESULTAT_NON_COURANT": {"N"},
    "RESULTAT_AVANT_IMPOT": {"N"},
    "IMPOT_SUR_RESULTATS": {"N"},
    "ACHATS_REVENDUS": {"N"},
    "ACHATS_CONSOMMES": {"N"},
    "CHARGES_INTERETS": {"N"},
    "DOTATIONS_AMORTISSEMENTS": {"N"},
    "PRODUITS_CESSION_IMMOBILISATIONS": {"N"},
    "VALEUR_NETTE_IMMOBILISATIONS_CEDEES": {"N"},
    "REDEVANCES_CREDIT_BAIL": {"N"},
    "RESULTAT_NET_XIII": {"N"},
    "RESULTAT_NET_XVI": {"N"},
    "RESULTAT_FISCAL": {"N"},
    "REINTEGRATIONS": {"N"},
    "DEDUCTIONS": {"N"},
    "IS_DU": {"N"},
    "COTISATION_MINIMALE": {"N"},
    "REPORT_DEFICITAIRE": {"N"},
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
    "RESULTAT_NET",
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
    "DOTATIONS_AMORTISSEMENTS",
    "PRODUITS_CESSION_IMMOBILISATIONS",
    "VALEUR_NETTE_IMMOBILISATIONS_CEDEES",
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
    "DETTES_FINANCIERES_N1": ("dettes_financieres_n1", "Dettes financières N-1"),
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


def clean_qwen_marker(text: str) -> str:
    return re.sub(
        r"(?:</?think>|/think)\s*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    ).strip()


def validate_candidate_scope(candidate: FinancialCandidate) -> list[str]:
    _, reasons = candidate_is_eligible(candidate)
    return reasons


def infer_period_from_column(
    candidate: FinancialCandidate,
) -> FinancialCandidate:
    """Normalise period à partir de column_role — sans créer de montant."""
    role = candidate.evidence.column_role

    if role == "EXERCICE_N1":
        inferred = "N_MINUS_1"
    elif role in {"NET_N", "EXERCICE_N", "TOTAL_EXERCICE_N"}:
        inferred = "N"
    else:
        return candidate

    if candidate.period == inferred:
        return candidate

    return candidate.model_copy(
        update={
            "period": inferred,
            "warnings": [
                *candidate.warnings,
                (
                    "Période normalisée par Python "
                    f"à partir de column_role : {inferred}."
                ),
            ],
        }
    )


def canonicalize_candidate_period(candidate: FinancialCandidate) -> FinancialCandidate:
    """Rétrocompat : ne mute plus field_code (interdit dans le schéma Qwen)."""
    return candidate


def dataset_field_code(candidate: FinancialCandidate) -> str:
    """Clé dataset interne : (code métier, period) → attr N ou N-1."""
    code = str(candidate.field_code)
    if candidate.period == "N_MINUS_1":
        return _N1_CODE_MAP.get(code, code)
    return code


def canonicalize_column_role(
    candidate: FinancialCandidate,
) -> FinancialCandidate:
    """Normalise column_role depuis column_name (alias OCR inclus)."""
    name = _fold(candidate.evidence.column_name or "")
    role = candidate.evidence.column_role
    section = candidate.evidence.section

    if section == "CPC":
        current_total_aliases = (
            "3 1 2",
            "3 = 1 + 2",
            "3=1+2",
            "totaux de l exercice",
            "totaux de lexercice",
            "total de l exercice",
            "total de lexercice",
            "taux du exercice",
            "taux de l exercice",
            "taux de lexercice",
        )
        # Forme compacte pour « 3 = 1 + 2 » OCR → « 3 1 2 »
        compact = re.sub(r"[^0-9a-z]+", " ", name)
        compact = re.sub(r"\s+", " ", compact).strip()
        if any(alias in name for alias in current_total_aliases) or "3 1 2" in compact:
            role = "TOTAL_EXERCICE_N"
        elif "exercice precedent" in name or name.strip() == "4":
            role = "EXERCICE_N1"

    elif section == "BILAN_ACTIF":
        if "net" in name and "precedent" not in name:
            role = "NET_N"
        elif "precedent" in name:
            role = "EXERCICE_N1"
        elif "brut" in name:
            role = "BRUT"

    elif section == "BILAN_PASSIF":
        if "precedent" in name:
            role = "EXERCICE_N1"
        elif "exercice" in name:
            role = "EXERCICE_N"

    elif section == "DETAIL_CPC":
        if "precedent" in name:
            role = "EXERCICE_N1"
        elif "exercice" in name or "totaux" in name or "3 1 2" in name:
            role = "EXERCICE_N"

    if role == candidate.evidence.column_role:
        return candidate

    evidence = candidate.evidence.model_copy(update={"column_role": role})
    return candidate.model_copy(
        update={
            "evidence": evidence,
            "warnings": [
                *candidate.warnings,
                (
                    "column_role normalisé depuis "
                    f"column_name={candidate.evidence.column_name!r}."
                ),
            ],
        }
    )


def is_total_general_candidate(candidate: FinancialCandidate) -> bool:
    label = normalize_label(candidate.evidence.raw_label)
    label = re.sub(r"[*_`#]", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    return (
        candidate.nature == "GRAND_TOTAL"
        and bool(_TOTAL_GENERAL_RE.search(label))
    )


def candidate_is_eligible(candidate: FinancialCandidate) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    label = _fold(candidate.evidence.raw_label)
    role = candidate.evidence.column_role
    section = candidate.evidence.section
    field_code = str(candidate.field_code)

    if field_code == "UNKNOWN":
        reasons.append("Code UNKNOWN non résolvable.")

    allowed = ALLOWED_FIELDS_BY_SECTION.get(section, set())
    if field_code not in allowed:
        reasons.append(f"{field_code} interdit dans la section {section}.")

    supported = SUPPORTED_PERIODS_BY_FIELD.get(field_code, {"N"})
    if candidate.period not in supported:
        logger.debug(
            "Candidat ignoré car période non stockée : field=%s period=%s",
            field_code,
            candidate.period,
        )
        return False, ["Période non utilisée dans la version actuelle."]

    if candidate.period == "N_MINUS_1":
        if role != "EXERCICE_N1":
            reasons.append("Un champ N-1 exige column_role=EXERCICE_N1.")
    else:
        if candidate.period != "N":
            reasons.append("Un champ courant exige period=N.")

    if (
        section == "BILAN_ACTIF"
        and candidate.period == "N"
        and field_code in _ACTIF_CURRENT_FIELDS
    ):
        if role != "NET_N":
            reasons.append("Le bilan actif courant exige column_role=NET_N.")

    if (
        section == "BILAN_PASSIF"
        and candidate.period == "N"
        and field_code in _PASSIF_CURRENT_FIELDS
    ):
        if role != "EXERCICE_N":
            reasons.append("Le bilan passif courant exige column_role=EXERCICE_N.")

    if (
        section == "CPC"
        and candidate.period == "N"
        and field_code in _CPC_CURRENT_FIELDS
    ):
        if role != "TOTAL_EXERCICE_N":
            reasons.append("Le CPC courant exige column_role=TOTAL_EXERCICE_N.")

    if field_code in {"RESULTAT_NET_XIII", "RESULTAT_NET_XVI"}:
        if section != "CPC":
            reasons.append("RESULTAT_NET XIII/XVI hors CPC.")
        if role != "TOTAL_EXERCICE_N":
            reasons.append("RESULTAT_NET XIII/XVI exige column_role=TOTAL_EXERCICE_N.")
        if candidate.period != "N":
            reasons.append("RESULTAT_NET XIII/XVI exige period=N.")

    if field_code in {"TOTAL_PASSIF", "TOTAL_ACTIF", "TOTAL_BILAN"}:
        if not is_total_general_candidate(candidate):
            reasons.append(
                "Un total intermédiaire ne peut pas devenir le total général."
            )

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
        if any(
            token in label
            for token in (
                "augmentation des dettes",
                "diminution des dettes",
                "ecart de conversion",
            )
        ):
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
            reasons.append(
                "Autres charges financières non assimilées automatiquement "
                "aux charges d'intérêts."
            )

    if field_code == "CHIFFRE_AFFAIRES":
        if section != "CPC":
            reasons.append("CHIFFRE_AFFAIRES hors CPC.")

    if field_code == "RESULTAT_EXPLOITATION":
        if section != "CPC":
            reasons.append("RESULTAT_EXPLOITATION hors CPC.")
        if "resultat d exploitation" not in label and "resultat dexploitation" not in label:
            reasons.append(
                "RESULTAT_EXPLOITATION exige le libellé Résultat d'exploitation "
                "(pas Produits d'exploitation)."
            )

    if field_code == "RESULTAT_FISCAL":
        if section != "RESULTAT_FISCAL":
            reasons.append("RESULTAT_FISCAL hors section RESULTAT_FISCAL.")
        # Lignes CPC (XIV/XVI) souvent mal classées
        if any(tok in label for tok in ("xiv", "xvi", "xiii", "resultat net (")):
            reasons.append("Ligne CPC résultat net refusée comme RESULTAT_FISCAL.")

    if field_code == "REPORT_DEFICITAIRE":
        # « RESULTAT COURANT ( Report ) » ≠ report déficitaire fiscal
        if "deficit" not in label:
            reasons.append("REPORT_DEFICITAIRE exige un libellé de déficit.")

    if field_code == "ENCOURS_LEASING" and "redevance" in label:
        reasons.append("Une redevance de crédit-bail n'est pas un encours.")

    if (
        field_code == "REDEVANCES_CREDIT_BAIL"
        and "encours" in label
        and "redevance" not in label
    ):
        reasons.append("Encours exclu du champ REDEVANCES_CREDIT_BAIL.")

    return not reasons, reasons

def candidate_priority(candidate: FinancialCandidate) -> float:
    score = float(_NATURE_PRIORITY.get(candidate.nature, 0))
    score += float(candidate.confidence) * 20.0
    role = candidate.evidence.column_role
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
        "REDEVANCES_CREDIT_BAIL": {"DETAIL_CPC", "CPC"},
    }
    expected = expected_sections.get(field_code)
    if expected and section in expected:
        score += 100.0

    if candidate.period == "N":
        score += 80.0
    if candidate.period == "N_MINUS_1":
        score += 80.0

    if role in {"NET_N", "EXERCICE_N", "TOTAL_EXERCICE_N"} and candidate.period == "N":
        score += 60.0
    if role == "EXERCICE_N1" and candidate.period == "N_MINUS_1":
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
        "CHARGES_FINANCIERES": (
            "charges financieres",
            "total v",
            "total des charges financieres",
        ),
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

    if field_code == "CHARGES_FINANCIERES":
        # Pénalise les faux 0,00 issus de pages mal classées
        raw = _fold(candidate.raw_value)
        if raw in {"0", "0 00", "0.00", "0,00"} and "charges" not in label:
            score -= 120.0
        if label.replace(" ", "").replace(",", "").replace(".", "").isdigit():
            score -= 80.0
        # Préférer la 1ʳᵉ page CPC (page_number plus petit souvent = CPC principal)
        score -= min(float(candidate.evidence.page_number), 20.0) * 0.5

    return score


def sanitize_candidate(candidate: FinancialCandidate) -> FinancialCandidate:
    """Nettoie les marqueurs techniques /think sans modifier le métier."""
    raw = clean_qwen_marker(candidate.raw_value)
    evidence = candidate.evidence.model_copy(
        update={
            "source_excerpt": clean_qwen_marker(candidate.evidence.source_excerpt),
            "raw_label": clean_qwen_marker(candidate.evidence.raw_label),
        }
    )
    return candidate.model_copy(update={"raw_value": raw, "evidence": evidence})


def group_candidates_by_field(outputs: list[FinancialMappingOutput]) -> dict[str, list[FinancialCandidate]]:
    grouped: dict[str, list[FinancialCandidate]] = defaultdict(list)
    for output in outputs:
        for candidate in output.candidates:
            candidate = sanitize_candidate(candidate)
            candidate = canonicalize_column_role(candidate)
            candidate = infer_period_from_column(candidate)
            if candidate.field_code == "UNKNOWN":
                continue
            key = dataset_field_code(candidate)
            if key == "RESULTAT_NET" and candidate.evidence.section == "CPC":
                label = _fold(candidate.evidence.raw_label)
                if "xiii" in label:
                    key = "RESULTAT_NET_XIII"
                elif "xvi" in label:
                    key = "RESULTAT_NET_XVI"
            grouped[key].append(candidate)
    return grouped


def _parse_candidate_amount(candidate: FinancialCandidate) -> tuple[Decimal | None, list[str]]:
    raw = clean_qwen_marker(candidate.raw_value)
    if not raw or not str(raw).strip():
        return None, ["raw_value absent — non converti en zéro."]
    warnings: list[str] = []
    if is_mixed_comma_ocr_amount(raw):
        warnings.append("Separateurs OCR normalises.")
    amount = parse_decimal_amount(raw)
    if amount is None:
        if is_explicit_zero(raw):
            return Decimal("0"), warnings
        return None, [f"Montant non parsable : {raw!r}"]
    return amount, warnings


def _to_provenance(candidate: FinancialCandidate) -> ValueProvenance:
    return ValueProvenance(
        page_number=candidate.evidence.page_number,
        raw_label=candidate.evidence.raw_label,
        raw_value=candidate.raw_value,
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


def _eligible_sorted(candidates: list[FinancialCandidate]) -> list[tuple[FinancialCandidate, Decimal, list[str]]]:
    prepared: list[tuple[FinancialCandidate, Decimal, float, list[str]]] = []
    for candidate in candidates:
        ok, reasons = candidate_is_eligible(candidate)
        if not ok:
            period_skip = reasons == [
                "Période non utilisée dans la version actuelle."
            ]
            log_fn = logger.debug if period_skip else logger.warning
            log_fn(
                "Candidate %s rejected: %s",
                candidate.field_code,
                "; ".join(reasons),
            )
            continue
        amount, parse_warnings = _parse_candidate_amount(candidate)
        if amount is None:
            logger.warning("Candidate %s invalid: %s", candidate.field_code, "; ".join(parse_warnings))
            continue
        prepared.append((candidate, amount, candidate_priority(candidate), parse_warnings))
    prepared.sort(key=lambda item: item[2], reverse=True)
    return [(c, a, w) for c, a, _, w in prepared]


def _has_suspected_row_shift(candidate: FinancialCandidate) -> bool:
    return any("suspected_row_shift" in w for w in candidate.warnings)


def _pick_best(candidates: list[FinancialCandidate], *, code: str | None = None) -> FinancialValue | None:
    ranked = _eligible_sorted(candidates)
    if not ranked:
        return None

    best_candidate, best_amount, parse_warnings = ranked[0]
    resolved_code = code or dataset_field_code(best_candidate)
    if (
        str(best_candidate.field_code) == "RESULTAT_NET"
        and best_candidate.evidence.section == "CPC"
    ):
        label = _fold(best_candidate.evidence.raw_label)
        if "xiii" in label:
            resolved_code = "RESULTAT_NET_XIII"
        elif "xvi" in label:
            resolved_code = "RESULTAT_NET_XVI"

    if (
        resolved_code in {"CLIENTS", "FOURNISSEURS", "STOCKS"}
        and _has_suspected_row_shift(best_candidate)
    ):
        return _financial_value(
            resolved_code,
            None,
            "ambiguous",
            [_to_provenance(best_candidate)],
            warnings=[
                "suspected_row_shift : confirmation automatique refusée, "
                "revue manuelle requise."
            ],
        )

    best_priority = candidate_priority(best_candidate)
    tied = [
        (candidate, amount, pw)
        for candidate, amount, pw in ranked
        if abs(candidate_priority(candidate) - best_priority) <= _PRIORITY_TIE_MARGIN
    ]
    tied_amounts = {amount for _, amount, _ in tied}

    if len(tied_amounts) > 1:
        return _financial_value(
            resolved_code,
            None,
            "conflicting",
            [_to_provenance(candidate) for candidate, _, _ in tied],
            warnings=[
                "Candidats de priorité équivalente avec montants divergents."
            ],
        )

    warnings: list[str] = list(parse_warnings)
    if len(ranked) > 1:
        warnings.append(
            f"{len(ranked) - 1} candidat(s) de priorité inférieure conservé(s) "
            "uniquement dans l'audit."
        )
    if _has_suspected_row_shift(best_candidate):
        warnings.append("suspected_row_shift signalé par le modèle.")
    return _financial_value(
        resolved_code,
        best_amount,
        "confirmed",
        [_to_provenance(best_candidate)],
        warnings=warnings,
    )


def _derive_total_actif_from_sections(
    grouped: dict[str, list[FinancialCandidate]],
) -> FinancialValue | None:
    """TOTAL I + TOTAL II + TOTAL III (trésorerie) → TOTAL_ACTIF."""
    parts: list[FinancialValue] = []
    for code in ("ACTIFS_IMMOBILISES", "ACTIF_CIRCULANT", "TRESORERIE_ACTIF"):
        fv = _pick_best(
            [
                c
                for c in grouped.get(code, [])
                if c.evidence.section == "BILAN_ACTIF" and c.period == "N"
            ],
            code=code,
        )
        if not _usable_fv(fv):
            return None
        assert fv is not None
        parts.append(fv)
    total = sum((p.value for p in parts if p.value is not None), Decimal("0"))
    provenances: list[ValueProvenance] = []
    for part in parts:
        provenances.extend(part.provenance)
    return _financial_value(
        "TOTAL_ACTIF",
        total,
        "derived",
        provenances,
        warnings=["Dérivé : ACTIFS_IMMOBILISES + ACTIF_CIRCULANT + TRESORERIE_ACTIF"],
    )


def _resolve_total_bilan(
    grouped: dict[str, list[FinancialCandidate]],
    *,
    resolved_actif: FinancialValue | None = None,
    resolved_passif: FinancialValue | None = None,
) -> FinancialValue:
    total_actif = resolved_actif or _pick_best(
        [c for c in grouped.get("TOTAL_ACTIF", []) if c.evidence.section == "BILAN_ACTIF" and c.period == "N"]
    )
    if not _usable_fv(total_actif):
        total_actif = _derive_total_actif_from_sections(grouped)

    total_passif = resolved_passif or _pick_best(
        [c for c in grouped.get("TOTAL_PASSIF", []) if c.evidence.section == "BILAN_PASSIF" and c.period == "N"]
    )

    if _usable_fv(total_actif) and _usable_fv(total_passif):
        assert total_actif is not None and total_passif is not None
        assert total_actif.value is not None and total_passif.value is not None
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
        logger.info(
            "Resolved TOTAL_BILAN=%s from ACTIF/PASSIF agreement", total_actif.value
        )
        return _financial_value(
            "TOTAL_BILAN",
            total_actif.value,
            "confirmed",
            total_actif.provenance + total_passif.provenance,
        )

    # Un seul côté fiable : retenable pour le scoring (bilan = total actif)
    if _usable_fv(total_actif):
        assert total_actif is not None and total_actif.value is not None
        return _financial_value(
            "TOTAL_BILAN",
            total_actif.value,
            "derived",
            total_actif.provenance,
            warnings=["TOTAL_BILAN dérivé du total actif (passif manquant)."],
        )
    if _usable_fv(total_passif):
        assert total_passif is not None and total_passif.value is not None
        return _financial_value(
            "TOTAL_BILAN",
            total_passif.value,
            "derived",
            total_passif.provenance,
            warnings=["TOTAL_BILAN dérivé du total passif (actif manquant)."],
        )

    return _financial_value("TOTAL_BILAN", None, "missing", [])


def _resolve_resultat_net(grouped: dict[str, list[FinancialCandidate]]) -> FinancialValue | None:
    xiii = _eligible_sorted(grouped.get("RESULTAT_NET_XIII", []))
    xvi = _eligible_sorted(grouped.get("RESULTAT_NET_XVI", []))
    cpc = _eligible_sorted(grouped.get("RESULTAT_NET", []))
    cpc = [
        (c, a, w)
        for c, a, w in cpc
        if c.evidence.section == "CPC"
        and c.period == "N"
        and c.evidence.column_role == "TOTAL_EXERCICE_N"
    ]
    bilan = _eligible_sorted(grouped.get("RESULTAT_NET", []))
    bilan = [
        (c, a, w)
        for c, a, w in bilan
        if c.evidence.section == "BILAN_PASSIF" and c.period == "N"
    ]

    if xiii and xvi:
        c13, v13, w13 = xiii[0]
        c16, v16, w16 = xvi[0]
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
        best_c, best_a, best_w = cpc[0]
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
        c, a, w = bilan[0]
        return _financial_value("RESULTAT_NET", a, "confirmed", [_to_provenance(c)])
    return None


def _resolve_special(code: str, grouped: dict[str, list[FinancialCandidate]]) -> FinancialValue | None:
    if code == "RESULTAT_NET":
        return _resolve_resultat_net(grouped)
    if code == "TOTAL_ACTIF":
        picked = _pick_best(
            [
                c
                for c in grouped.get("TOTAL_ACTIF", [])
                if c.evidence.section == "BILAN_ACTIF"
            ],
            code="TOTAL_ACTIF",
        )
        if _usable_fv(picked):
            return picked
        return _derive_total_actif_from_sections(grouped)
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
    if code == "CHARGES_FINANCIERES":
        # Écarte les 0,00 sans libellé métier (pages mal classées)
        filtered = [
            c
            for c in candidates
            if not (
                _fold(c.raw_value) in {"0", "0 00", "0.00", "0,00"}
                and "charges" not in _fold(c.evidence.raw_label)
                and "total v" not in _fold(c.evidence.raw_label)
            )
        ]
        candidates = filtered or candidates
    if code == "ENCOURS_LEASING" and not _eligible_sorted(candidates):
        return _financial_value("ENCOURS_LEASING", None, "missing", [])
    return _pick_best(candidates, code=code)


def _usable_fv(fv: FinancialValue | None) -> bool:
    return (
        fv is not None
        and fv.status in {"confirmed", "derived"}
        and fv.value is not None
    )


def _prefer_derived_when_ocr_conflicts(
    resolved: dict[str, FinancialValue],
    code: str,
    calculated: Decimal,
    components: list[FinancialValue],
    formula_warning: str,
    conflict_warning: str | None = None,
) -> None:
    provenances: list[ValueProvenance] = []
    for component in components:
        provenances.extend(component.provenance)

    existing = resolved.get(code)
    if (
        existing is not None
        and existing.value is not None
        and existing.status in {"confirmed", "derived"}
        and not _amounts_agree(existing.value, calculated)
    ):
        resolved[code] = _financial_value(
            code,
            calculated,
            "derived",
            existing.provenance + provenances,
            warnings=[
                conflict_warning
                or (
                    f"{code} OCR ({existing.value}) contredit le calcul "
                    f"comptable ({calculated}). Valeur calculée retenue."
                ),
                formula_warning,
            ],
        )
        return

    if existing is None or existing.value is None or existing.status == "missing":
        resolved[code] = _financial_value(
            code,
            calculated,
            "derived",
            provenances,
            warnings=[formula_warning],
        )


def _apply_accounting_derivations(resolved: dict[str, FinancialValue]) -> None:
    """Dérive les agrégats calculables en Decimal ; privilégie le calcul fiable."""
    pf = resolved.get("PRODUITS_FINANCIERS")
    cf = resolved.get("CHARGES_FINANCIERES")
    if _usable_fv(pf) and _usable_fv(cf):
        assert pf is not None and cf is not None
        assert pf.value is not None and cf.value is not None
        _prefer_derived_when_ocr_conflicts(
            resolved,
            "RESULTAT_FINANCIER",
            pf.value - cf.value,
            [pf, cf],
            "Dérivé : PRODUITS_FINANCIERS - CHARGES_FINANCIERES",
            conflict_warning=(
                "Résultat financier OCR contradictoire avec produits - charges."
            ),
        )

    pnc = resolved.get("PRODUITS_NON_COURANTS")
    cnc = resolved.get("CHARGES_NON_COURANTES")
    if _usable_fv(pnc) and _usable_fv(cnc):
        assert pnc is not None and cnc is not None
        assert pnc.value is not None and cnc.value is not None
        _prefer_derived_when_ocr_conflicts(
            resolved,
            "RESULTAT_NON_COURANT",
            pnc.value - cnc.value,
            [pnc, cnc],
            "Dérivé : PRODUITS_NON_COURANTS - CHARGES_NON_COURANTES",
        )

    rc = resolved.get("RESULTAT_COURANT")
    rnc = resolved.get("RESULTAT_NON_COURANT")
    if _usable_fv(rc) and _usable_fv(rnc):
        assert rc is not None and rnc is not None
        assert rc.value is not None and rnc.value is not None
        _prefer_derived_when_ocr_conflicts(
            resolved,
            "RESULTAT_AVANT_IMPOT",
            rc.value + rnc.value,
            [rc, rnc],
            "Dérivé : RESULTAT_COURANT + RESULTAT_NON_COURANT",
        )

    ra = resolved.get("RESULTAT_AVANT_IMPOT")
    impot = resolved.get("IMPOT_SUR_RESULTATS")
    if _usable_fv(ra) and _usable_fv(impot) and "RESULTAT_NET" not in resolved:
        assert ra is not None and impot is not None
        assert ra.value is not None and impot.value is not None
        _prefer_derived_when_ocr_conflicts(
            resolved,
            "RESULTAT_NET",
            ra.value - impot.value,
            [ra, impot],
            "Dérivé : RESULTAT_AVANT_IMPOT - IMPOT_SUR_RESULTATS",
        )

    ta = resolved.get("TRESORERIE_ACTIF")
    tp = resolved.get("TRESORERIE_PASSIF")
    if _usable_fv(ta) and _usable_fv(tp):
        assert ta is not None and tp is not None
        assert ta.value is not None and tp.value is not None
        resolved.setdefault(
            "TRESORERIE_NETTE",
            _financial_value(
                "TRESORERIE_NETTE",
                ta.value - tp.value,
                "derived",
                ta.provenance + tp.provenance,
                warnings=["Dérivé : TRESORERIE_ACTIF - TRESORERIE_PASSIF"],
            ),
        )

    rev = resolved.get("ACHATS_REVENDUS")
    cons = resolved.get("ACHATS_CONSOMMES")
    if (
        _usable_fv(rev)
        and _usable_fv(cons)
        and (
            "ACHATS_TOTAL" not in resolved
            or resolved["ACHATS_TOTAL"].value is None
        )
    ):
        assert rev is not None and cons is not None
        assert rev.value is not None and cons.value is not None
        resolved["ACHATS_TOTAL"] = _financial_value(
            "ACHATS_TOTAL",
            rev.value + cons.value,
            "derived",
            rev.provenance + cons.provenance,
            warnings=["Dérivé : ACHATS_REVENDUS + ACHATS_CONSOMMES"],
        )


def resolve_financial_candidates(outputs: list[FinancialMappingOutput]) -> dict[str, FinancialValue]:
    grouped = group_candidates_by_field(outputs)
    resolved: dict[str, FinancialValue] = {}

    # Composantes bilan avant totaux (pour dérivation I+II+III)
    for code in (
        "ACTIFS_IMMOBILISES",
        "ACTIF_CIRCULANT",
        "TRESORERIE_ACTIF",
        "TRESORERIE_PASSIF",
        "FONDS_PROPRES",
        "PASSIF_CIRCULANT",
    ):
        fv = _pick_best(
            [c for c in grouped.get(code, [])],
            code=code,
        )
        if fv is not None:
            resolved[code] = fv

    for code in (
        "TOTAL_ACTIF",
        "TOTAL_PASSIF",
        "RESULTAT_NET",
        "CLIENTS",
        "FOURNISSEURS",
        "STOCKS",
        "DETTES_FINANCIERES",
        "CHIFFRE_AFFAIRES",
        "CHARGES_INTERETS",
        "CHARGES_FINANCIERES",
        "ENCOURS_LEASING",
        "REDEVANCES_CREDIT_BAIL",
    ):
        fv = _resolve_special(code, grouped)
        if fv is not None:
            resolved[code] = fv

    resolved["TOTAL_BILAN"] = _resolve_total_bilan(
        grouped,
        resolved_actif=resolved.get("TOTAL_ACTIF"),
        resolved_passif=resolved.get("TOTAL_PASSIF"),
    )
    # Propager total_actif dérivé si seulement TOTAL_BILAN l'a produit via sections
    if (
        not _usable_fv(resolved.get("TOTAL_ACTIF"))
        and _usable_fv(resolved.get("TOTAL_BILAN"))
    ):
        derived = _derive_total_actif_from_sections(grouped)
        if derived is not None:
            resolved["TOTAL_ACTIF"] = derived

    for code, candidates in grouped.items():
        if code in resolved or code in {"RESULTAT_NET_XIII", "RESULTAT_NET_XVI", "UNKNOWN"}:
            continue
        fv = _pick_best(candidates, code=code)
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

    _apply_accounting_derivations(resolved)

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
