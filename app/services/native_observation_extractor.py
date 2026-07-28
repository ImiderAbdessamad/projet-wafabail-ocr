"""Extraction d'observations depuis texte natif ou rapport indicateurs."""
from __future__ import annotations

import re
from decimal import Decimal

from app.schemas.observations import RawFinancialObservation
from app.services.amount_parser import parse_amount
from app.services.field_definitions import FIELD_DEFINITIONS
from app.services.label_normalizer import normalize_label

# Montants PCGM typiques : "43 807 944,82" (espaces milliers + virgule)
_AMOUNT_RE = r"[-−]?\d{1,3}(?:[   ]\d{3})+(?:,\d{1,2})?|[-−]?\d+,\d{2}"
_AMOUNT_FINDER = re.compile(_AMOUNT_RE)


def extract_report_observations(text: str) -> list[RawFinancialObservation]:
    """Convertit un rapport d'indicateurs structuré en observations."""
    observations: list[RawFinancialObservation] = []
    loose_amount = r"[-−]?\d{1,3}(?:[   .]\d{3})*(?:[,.]\d{1,2})?"
    line_re = re.compile(
        rf"^(\d{{1,2}})\s+(.+?)\s+({loose_amount})\s*MAD\s*"
        rf"(Bénéficiaire|Déficitaire|Nul)?\s*"
        rf"(Bilan Actif|Bilan Passif|CPC|Bilan|Dérivé)\s*$",
        re.MULTILINE,
    )
    for idx, match in enumerate(line_re.finditer(text), start=1):
        label = match.group(2).strip()
        value = match.group(3)
        code = _guess_code_from_label(label)
        observations.append(
            RawFinancialObservation(
                page=1,
                section=_source_to_section(match.group(5)),
                table_title=f"CODE:{code}" if code else "REPORT_ELEMENTS",
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


def _amounts_near_label(
    text: str,
    labels: list[str],
    *,
    before: int = 350,
    after: int = 220,
    pick: str = "net",
    min_abs: float = 1.0,
) -> tuple[str, Decimal] | None:
    """Cherche un montant autour d'un libellé.

    Sur les PDF PCGM natifs, les montants sont souvent *avant* le libellé
    dans le flux texte (rendu colonne par colonne).
    """
    lowered = text.lower()
    for label in labels:
        start = 0
        needle = label.lower()
        while True:
            idx = lowered.find(needle, start)
            if idx < 0:
                break
            window_start = max(0, idx - before)
            window_end = min(len(text), idx + len(label) + after)
            window = text[window_start:window_end]
            raw_amounts = _AMOUNT_FINDER.findall(window)
            parsed: list[tuple[str, Decimal]] = []
            for raw in raw_amounts:
                value = parse_amount(raw)
                if value is None:
                    continue
                if abs(value) < Decimal(str(min_abs)):
                    continue
                # Ignore les numéros de page / indices isolés (1..99 sans décimale)
                if "," not in raw and abs(value) < 100:
                    continue
                parsed.append((raw, value))
            if parsed:
                if pick == "first":
                    return parsed[0]
                if pick == "last":
                    return parsed[-1]
                if pick == "max":
                    return max(parsed, key=lambda item: abs(item[1]))
                # pick == "net" : dans un bloc Brut / Amort / Net N / Net N-1,
                # le Net N est souvent l'avant-dernier ou le 3e montant.
                if len(parsed) >= 4:
                    return parsed[-2]
                if len(parsed) >= 3:
                    return parsed[-2] if abs(parsed[-2][1]) >= abs(parsed[-1][1]) * Decimal("0.01") else parsed[-1]
                if len(parsed) == 2:
                    return parsed[0]
                return parsed[0]
            start = idx + 1
    return None


def _obs(
    code: str,
    raw: str,
    value: Decimal,
    *,
    row_index: int,
) -> RawFinancialObservation:
    definition = FIELD_DEFINITIONS.get(code)
    label = definition.label if definition else code
    section = definition.sections[0] if definition and definition.sections else "AUTRE"
    return RawFinancialObservation(
        page=1,
        section=section,
        table_title=f"CODE:{code}",
        raw_label=label,
        normalized_label=normalize_label(label),
        raw_value=raw,
        parsed_value=value,
        row_index=row_index,
        column_name="native_text",
        value_nature="net_n",
        extraction_method="native_text",
    )


def extract_native_observations(text: str) -> list[RawFinancialObservation]:
    """Extraction native robuste vers RawFinancialObservation."""
    observations: list[RawFinancialObservation] = []
    specs: list[tuple[str, list[str], str, float]] = [
        ("CHIFFRE_AFFAIRES", ["Chiffres d'affaires", "Ventes de biens et services produits"], "max", 10),
        ("CA_EXPORT", ["Ventes à l'export", "CA à l'export"], "first", 0),
        ("RESULTAT_NET", ["Résultat net de l'exercice", "Résultat net"], "first", 1),
        ("ACHATS_REVENDUS", ["Achats revendus de marchandises", "Achats revendus"], "first", 1),
        ("CHARGES_INTERETS", ["Charges d'intérêts"], "first", 1),
        ("AUTRES_CHARGES", ["Autres charges d'exploitation", "Autres charges"], "first", 1),
        ("CAF", ["Capacité d'autofinancement", "CAPACITE D'AUTOFINANCEMENT"], "first", 1),
        ("AMORTISSEMENTS", ["Dotations d'exploitation", "Dotations aux amortissements"], "first", 1),
        ("CREANCES_CLIENTS", ["Clients et comptes rattachés"], "first", 1),
        ("DETTES_FOURNISSEURS", ["Fournisseurs et comptes rattachés"], "first", 1),
        ("TRESORERIE_ACTIF", ["Trésorerie-Actif", "Trésorerie - Actif", "TRESORERIE - ACTIF"], "net", 1),
        ("TRESORERIE_PASSIF", ["Trésorerie-Passif", "Trésorerie - Passif", "TRESORERIE - PASSIF"], "net", 1),
        ("CAISSE", ["Caisse, Régie d'avances", "Caisse"], "first", 1),
        ("COMPTE_COURANT_ASSOCIES", ["Comptes d'associés"], "first", 1),
        ("ACTIFS_IMMOBILISES", ["TOTAL I (A+B+C+D+E)", "TOTAL I"], "net", 100),
        ("TOTAL_BILAN", ["TOTAL GENERAL I+II+III", "TOTAL GENERAL"], "net", 100),
        ("ACTIF_CIRCULANT", ["TOTAL II (F+G+H+I)", "TOTAL II"], "net", 100),
        ("PASSIF_CIRCULANT", ["DETTES DU PASSIF CIRCULANT (F)", "Passif circulant"], "max", 1),
        ("DETTES_BANCAIRES_MLT", ["DETTES DE FINANCEMENT (C)", "Dettes de financement"], "max", 1),
        ("DETTES_BANCAIRES_CT", ["Crédits de trésorerie", "Concours bancaires courants"], "first", 1),
        ("FONDS_PROPRES", ["Total des capitaux propres", "CAPITAUX PROPRES (A)"], "max", 1),
        ("FDR", ["Fonds de roulement", "Fonds de roulement fonctionnel"], "first", 1),
        ("CMT", ["Crédit moyen terme", "Crédits à moyen terme"], "first", 1),
        ("ENCOURS_LEASING", ["Encours leasing", "Crédit-bail"], "first", 1),
        ("CA_N1", ["Chiffres d'affaires"], "last", 10),
    ]

    for row_index, (code, labels, pick, min_abs) in enumerate(specs, start=1):
        if any(obs.table_title == f"CODE:{code}" for obs in observations):
            continue
        hit = _amounts_near_label(text, labels, pick=pick, min_abs=min_abs)
        if not hit:
            continue
        raw, value = hit
        observations.append(_obs(code, raw, value, row_index=row_index))
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
        "reference": [
            r"référence\s+IS[_-]?([A-Za-z0-9]+)",
            r"Etat sous référence\s+(\S+)",
            r"Référence liasse\s*:\s*(\S+)",
        ],
        "entreprise": [
            r"Raison Sociale\s*:?\s*\n+\s*([A-Za-z0-9][^\n]{2,80})",
            r"Entreprise\s*:\s*(.+)",
        ],
        "identification_fiscale": [
            r"Identifiant fiscal\s*:?\s*\n+\s*(\d[\d\s.]{5,})",
            r"IF\s*:\s*(\S+)",
        ],
        "exercice": [r"au titre de la période du\s+\d{2}/\d{2}/(\d{4})"],
        "date_debut_exercice": [r"période du\s+(\d{2}/\d{2}/\d{4})"],
        "date_fin_exercice": [r"période du\s+\d{2}/\d{2}/\d{4}\s+au\s+(\d{2}/\d{2}/\d{4})"],
    }
    # Fallback entreprise : ligne juste après "Raison Sociale"
    for key, regexes in patterns.items():
        for regex in regexes:
            match = re.search(regex, text, re.IGNORECASE)
            if match:
                metadata[key] = re.sub(r"\s+", " ", match.group(1)).strip()
                break

    # Sur ADEIS/FDI le nom est souvent isolé dans le bloc identification.
    if not metadata["entreprise"] or metadata["entreprise"].lower().endswith(":"):
        block = re.search(
            r"Identification du contribuable(.*?)(?:Etat de synthèse|Tableau\s*:)",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if block:
            for line in block.group(1).splitlines():
                candidate = line.strip()
                if (
                    candidate
                    and ":" not in candidate
                    and re.search(r"[A-Za-z]{3,}", candidate)
                    and not re.match(r"^[\d\s.,]+$", candidate)
                    and "conforme" not in candidate.lower()
                    and "adresse" not in candidate.lower()
                    and "ville" not in candidate.lower()
                ):
                    metadata["entreprise"] = candidate
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
