"""Extraction d'observations depuis texte natif ou rapport indicateurs."""
from __future__ import annotations

import re
from decimal import Decimal

from app.schemas.observations import RawFinancialObservation
from app.services.amount_parser import parse_amount
from app.services.field_definitions import FIELD_DEFINITIONS
from app.services.label_normalizer import normalize_label

_AMOUNT_RE = r"[-−]?\d{1,3}(?:[   .]\d{3})*(?:[,.]\d{1,2})?"


def extract_report_observations(text: str) -> list[RawFinancialObservation]:
    """Convertit un rapport d'indicateurs structuré en observations."""
    observations: list[RawFinancialObservation] = []
    line_re = re.compile(
        rf"^(\d{{1,2}})\s+(.+?)\s+({_AMOUNT_RE})\s*MAD\s*(Bénéficiaire|Déficitaire|Nul)?\s*(Bilan Actif|Bilan Passif|CPC|Bilan|Dérivé)\s*$",
        re.MULTILINE,
    )
    for idx, match in enumerate(line_re.finditer(text), start=1):
        label = match.group(2).strip()
        value = match.group(3)
        observations.append(
            RawFinancialObservation(
                page=1,
                section=_source_to_section(match.group(5)),
                table_title=f"CODE:{_guess_code_from_label(label)}" if _guess_code_from_label(label) else "REPORT_ELEMENTS",
                raw_label=label,
                normalized_label=normalize_label(label),
                raw_value=value,
                parsed_value=parse_amount(value),
                row_index=idx,
                column_name="Valeur (MAD)",
                value_nature="exercice",
                extraction_method="native_report",
            )
        )
    return observations


def extract_native_observations(text: str) -> list[RawFinancialObservation]:
    """Extraction native best-effort vers RawFinancialObservation.

    Phase 2 : on conserve la heuristique historique, mais elle produit désormais
    un format d'observations unique au lieu d'un résultat final direct.
    """
    observations: list[RawFinancialObservation] = []
    patterns = {
        "CHIFFRE_AFFAIRES": [
            r"Chiffres? d'affaires.{0,120}?(" + _AMOUNT_RE + r")",
            r"Ventes de biens et services produits.{0,120}?(" + _AMOUNT_RE + r")",
        ],
        "RESULTAT_NET": [
            r"Résultat net de l'exercice.{0,120}?(" + _AMOUNT_RE + r")",
            r"Résultat net.{0,120}?(" + _AMOUNT_RE + r")",
        ],
        "ACHATS_REVENDUS": [r"Achats revendus.{0,120}?(" + _AMOUNT_RE + r")"],
        "CHARGES_INTERETS": [r"Charges d'intérêts.{0,120}?(" + _AMOUNT_RE + r")"],
        "AUTRES_CHARGES": [r"Autres charges d'exploitation.{0,120}?(" + _AMOUNT_RE + r")"],
        "CA_N1": [r"Chiffres? d'affaires exercice précédent.{0,120}?(" + _AMOUNT_RE + r")"],
        "CAF": [r"Capacité d'autofinancement.{0,120}?(" + _AMOUNT_RE + r")", r"\bCAF\b.{0,120}?(" + _AMOUNT_RE + r")"],
        "AMORTISSEMENTS": [r"Dotations d'exploitation.{0,120}?(" + _AMOUNT_RE + r")"],
        "CREANCES_CLIENTS": [r"Clients et comptes rattachés.{0,120}?(" + _AMOUNT_RE + r")"],
        "DETTES_FOURNISSEURS": [r"Fournisseurs et comptes rattachés.{0,120}?(" + _AMOUNT_RE + r")"],
        "TRESORERIE_ACTIF": [r"Trésorerie-Actif.{0,120}?(" + _AMOUNT_RE + r")"],
        "TRESORERIE_PASSIF": [r"Trésorerie-Passif.{0,120}?(" + _AMOUNT_RE + r")"],
        "CAISSE": [r"Caisse.{0,60}?(" + _AMOUNT_RE + r")"],
        "COMPTE_COURANT_ASSOCIES": [r"Comptes d'associés.{0,120}?(" + _AMOUNT_RE + r")"],
        "ACTIFS_IMMOBILISES": [r"TOTAL I.+?(" + _AMOUNT_RE + r")"],
        "TOTAL_BILAN": [r"TOTAL GENERAL.{0,120}?(" + _AMOUNT_RE + r")"],
        "ACTIF_CIRCULANT": [r"TOTAL II.+?(" + _AMOUNT_RE + r")"],
        "PASSIF_CIRCULANT": [r"Passif circulant.{0,120}?(" + _AMOUNT_RE + r")"],
        "DETTES_BANCAIRES_MLT": [r"Dettes de financement.{0,120}?(" + _AMOUNT_RE + r")"],
        "DETTES_BANCAIRES_CT": [r"Crédits de trésorerie.{0,120}?(" + _AMOUNT_RE + r")"],
        "TRESORERIE_PASSIF": [r"Crédits de trésorerie.{0,120}?(" + _AMOUNT_RE + r")"],
        "FONDS_PROPRES": [r"Total des capitaux propres.{0,120}?(" + _AMOUNT_RE + r")"],
        "FDR": [r"Fonds de roulement.{0,120}?(" + _AMOUNT_RE + r")"],
        "CMT": [r"Crédit moyen terme.{0,120}?(" + _AMOUNT_RE + r")"],
        "ENCOURS_LEASING": [r"Encours leasing.{0,120}?(" + _AMOUNT_RE + r")"],
    }

    for row_index, (code, regexes) in enumerate(patterns.items(), start=1):
        for pattern in regexes:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            value = match.group(1)
            label = FIELD_DEFINITIONS.get(code).label if code in FIELD_DEFINITIONS else code
            observations.append(
                RawFinancialObservation(
                    page=1,
                    section=(FIELD_DEFINITIONS.get(code).sections[0] if code in FIELD_DEFINITIONS and FIELD_DEFINITIONS[code].sections else "AUTRE"),
                    table_title=f"CODE:{code}",
                    raw_label=label,
                    normalized_label=normalize_label(label),
                    raw_value=value,
                    parsed_value=parse_amount(value),
                    row_index=row_index,
                    column_name="native_text",
                    value_nature="unknown",
                    extraction_method="native_text",
                )
            )
            break
    return observations


def extract_metadata_from_native_text(text: str) -> dict[str, str | None]:
    metadata = {
        "reference": None,
        "entreprise": None,
        "identification_fiscale": None,
        "exercice": None,
        "date_debut_exercice": None,
        "date_fin_exercice": None,
    }
    patterns = {
        "reference": [r"Référence liasse\s*:\s*(\S+)", r"N[°o]\s*de dépôt\s*:\s*(\S+)"],
        "entreprise": [r"Entreprise\s*:\s*(.+)", r"Raison sociale\s*:\s*(.+)"],
        "identification_fiscale": [r"Identifiant fiscal\s*:\s*(\S+)", r"IF\s*:\s*(\S+)"],
        "exercice": [r"Exercice\s*:\s*(\d{4})"],
        "date_debut_exercice": [r"du\s*:?[\s]*(\d{2}/\d{2}/\d{4})"],
        "date_fin_exercice": [r"au\s*:?[\s]*(\d{2}/\d{2}/\d{4})"],
    }
    for key, regexes in patterns.items():
        for regex in regexes:
            match = re.search(regex, text, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()
                break
    return metadata


def _guess_code_from_label(label: str) -> str | None:
    normalized = normalize_label(label)
    for code, definition in FIELD_DEFINITIONS.items():
        if normalize_label(definition.label) == normalized:
            return code
        if any(normalize_label(alias) == normalized for alias in definition.aliases):
            return code
    return None


def _source_to_section(source: str) -> str:
    return {
        "Bilan Actif": "BILAN_ACTIF",
        "Bilan Passif": "BILAN_PASSIF",
        "CPC": "CPC",
        "Bilan": "AUTRE",
        "Dérivé": "AUTRE",
    }.get(source, "AUTRE")
