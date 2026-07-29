"""Construction et mapping FinancialDataset depuis lignes Markdown parsées."""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal

from app.schemas.financial_analysis import (
    DataStatus,
    FinancialDataset,
    FinancialValue,
    ValueProvenance,
)
from app.services.financial_normalizer import (
    is_explicit_zero,
    normalize_label,
    parse_decimal_amount,
)
from app.services.label_normalizer import label_similarity
from app.services.markdown_financial_parser import ParsedFinancialRow

FIELD_ALIASES: dict[str, list[str]] = {
    "CHIFFRE_AFFAIRES": [
        "chiffre d affaires",
        "chiffres d affaires",
        "total chiffre d affaires",
        "ventes de biens et services produits",
    ],
    "CHIFFRE_AFFAIRES_N1": [
        "chiffre d affaires n 1",
        "chiffre d affaires exercice precedent",
    ],
    "RESULTAT_NET": [
        "resultat net",
        "resultat net de l exercice",
        "xiii resultat net",
        "xvi resultat net",
    ],
    "RESULTAT_NET_N1": ["resultat net n 1", "resultat net exercice precedent"],
    "RESULTAT_EXPLOITATION": [
        "resultat d exploitation",
        "iii resultat d exploitation",
    ],
    "TOTAL_BILAN": [
        "total general i ii iii",
        "total general",
        "total actif",
        "total passif",
        "total du bilan",
    ],
    "FONDS_PROPRES": [
        "total des capitaux propres",
        "capitaux propres",
        "total i capitaux propres",
    ],
    "DETTES_FINANCIERES": [
        "dettes de financement",
        "total dettes de financement",
        "dettes financieres",
    ],
    "DETTES_BANCAIRES_CT": [
        "credits de tresorerie",
        "dettes bancaires ct",
        "banques soldes crediteurs",
    ],
    "TRESORERIE_ACTIF": [
        "tresorerie actif",
        "total iii tresorerie actif",
        "banques caisses et credit d escompte",
    ],
    "TRESORERIE_PASSIF": [
        "tresorerie passif",
        "total iii tresorerie passif",
        "credits de tresorerie",
    ],
    "ACTIF_CIRCULANT": [
        "actif circulant",
        "total ii actif circulant",
    ],
    "PASSIF_CIRCULANT": [
        "passif circulant",
        "total ii passif circulant",
        "dettes du passif circulant",
    ],
    "STOCKS": ["stocks", "stocks et en cours"],
    "CLIENTS": ["clients et comptes rattaches", "clients"],
    "FOURNISSEURS": ["fournisseurs et comptes rattaches", "fournisseurs"],
    "ACHATS": [
        "achats revendus de marchandises",
        "achats consommes de matieres et fournitures",
        "achats",
    ],
    "FRAIS_FINANCIERS": [
        "charges d interets",
        "frais financiers",
        "charges financieres",
    ],
    "AMORTISSEMENTS": [
        "dotations d exploitation",
        "dotations aux amortissements",
        "dotations d exploitation amortissements et provisions",
    ],
    "CAF": [
        "capacite d autofinancement",
        "caf",
        "autofinancement",
    ],
    "FDR": ["fonds de roulement", "fdr"],
    "BFDR": ["besoin de financement de roulement", "bfr", "bfdr"],
    "TRESORERIE_NETTE": ["tresorerie nette"],
    "ENCOURS_LEASING": ["encours leasing", "credit bail", "encours credit bail"],
    "CMT": ["cmt", "credit moyen terme"],
    "NOUVEAU_FINANCEMENT": ["nouveau financement", "montant demande"],
    "ACTIFS_IMMOBILISES": [
        "actif immobilise",
        "total i actif immobilise",
        "actifs immobilises",
    ],
    "RESULTAT_FISCAL": ["resultat fiscal", "benefice fiscal", "deficit fiscal"],
    "REINTEGRATIONS": ["reintegrations", "total des reintegrations"],
    "DEDUCTIONS": ["deductions", "total des deductions"],
    "IS_DU": ["impot sur les societes", "is du", "impot du"],
    "COTISATION_MINIMALE": ["cotisation minimale"],
    "REPORT_DEFICITAIRE": ["report deficitaire", "deficits anterieurs"],
}

_FIELD_META: dict[str, tuple[str, str]] = {
    "CHIFFRE_AFFAIRES": ("chiffre_affaires", "Chiffre d'affaires"),
    "CHIFFRE_AFFAIRES_N1": ("chiffre_affaires_n1", "Chiffre d'affaires N-1"),
    "RESULTAT_NET": ("resultat_net", "Résultat net"),
    "RESULTAT_NET_N1": ("resultat_net_n1", "Résultat net N-1"),
    "RESULTAT_EXPLOITATION": ("resultat_exploitation", "Résultat d'exploitation"),
    "TOTAL_BILAN": ("total_bilan", "Total bilan"),
    "TOTAL_BILAN_N1": ("total_bilan_n1", "Total bilan N-1"),
    "FONDS_PROPRES": ("fonds_propres", "Fonds propres"),
    "FONDS_PROPRES_N1": ("fonds_propres_n1", "Fonds propres N-1"),
    "DETTES_FINANCIERES": ("dettes_financieres", "Dettes financières"),
    "DETTES_FINANCIERES_N1": ("dettes_financieres_n1", "Dettes financières N-1"),
    "DETTES_BANCAIRES_CT": ("dettes_bancaires_ct", "Dettes bancaires CT"),
    "TRESORERIE_ACTIF": ("tresorerie_actif", "Trésorerie actif"),
    "TRESORERIE_PASSIF": ("tresorerie_passif", "Trésorerie passif"),
    "ACTIF_CIRCULANT": ("actif_circulant", "Actif circulant"),
    "PASSIF_CIRCULANT": ("passif_circulant", "Passif circulant"),
    "STOCKS": ("stocks", "Stocks"),
    "CLIENTS": ("clients", "Clients"),
    "FOURNISSEURS": ("fournisseurs", "Fournisseurs"),
    "ACHATS": ("achats", "Achats"),
    "FRAIS_FINANCIERS": ("frais_financiers", "Frais financiers"),
    "AMORTISSEMENTS": ("amortissements", "Amortissements"),
    "CAF": ("caf", "CAF"),
    "FDR": ("fdr", "Fonds de roulement"),
    "BFDR": ("bfdr", "BFDR"),
    "TRESORERIE_NETTE": ("tresorerie_nette", "Trésorerie nette"),
    "ENCOURS_LEASING": ("encours_leasing", "Encours leasing"),
    "CMT": ("cmt", "CMT"),
    "NOUVEAU_FINANCEMENT": ("nouveau_financement", "Nouveau financement"),
    "RESULTAT_FISCAL": ("resultat_fiscal", "Résultat fiscal"),
    "REINTEGRATIONS": ("reintegrations", "Réintégrations"),
    "DEDUCTIONS": ("deductions", "Déductions"),
    "IS_DU": ("is_du", "IS dû"),
    "COTISATION_MINIMALE": ("cotisation_minimale", "Cotisation minimale"),
    "REPORT_DEFICITAIRE": ("report_deficitaire", "Report déficitaire"),
    "ACTIFS_IMMOBILISES": ("actifs_immobilises", "Actifs immobilisés"),
}

PREFERRED_COLUMNS = ("net_n", "total_exercice", "exercice", "net", "value", "brut")


def _empty_value(code: str, label: str) -> FinancialValue:
    return FinancialValue(code=code, label=label, value=None, status="missing")


def empty_dataset() -> FinancialDataset:
    values = {
        attr: _empty_value(code, label)
        for code, (attr, label) in _FIELD_META.items()
        if attr
        in {
            "chiffre_affaires",
            "chiffre_affaires_n1",
            "resultat_net",
            "resultat_net_n1",
            "resultat_exploitation",
            "total_bilan",
            "total_bilan_n1",
            "fonds_propres",
            "fonds_propres_n1",
            "dettes_financieres",
            "dettes_financieres_n1",
            "dettes_bancaires_ct",
            "tresorerie_actif",
            "tresorerie_passif",
            "actif_circulant",
            "passif_circulant",
            "stocks",
            "clients",
            "fournisseurs",
            "achats",
            "frais_financiers",
            "amortissements",
            "caf",
            "fdr",
            "bfdr",
            "tresorerie_nette",
            "encours_leasing",
            "cmt",
            "nouveau_financement",
            "resultat_fiscal",
            "reintegrations",
            "deductions",
            "is_du",
            "cotisation_minimale",
            "report_deficitaire",
        }
    }
    return FinancialDataset(**values)


def _match_field_code(normalized_label: str) -> tuple[str | None, Decimal]:
    best_code: str | None = None
    best_score = Decimal("0")
    for code, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            score = Decimal(str(label_similarity(normalized_label, alias)))
            if score > best_score:
                best_score = score
                best_code = code
    if best_score < Decimal("0.7"):
        return None, best_score
    return best_code, best_score


def _pick_raw_amount(
    values: dict[str, str | None],
    *,
    prefer_n1: bool = False,
) -> tuple[str | None, str | None]:
    order = (
        ("net_n_1", "exercice_precedent", "n-1")
        if prefer_n1
        else PREFERRED_COLUMNS
    )
    for key in order:
        if key in values and values[key] is not None:
            return values[key], key
    for key, raw in values.items():
        if raw is not None:
            return raw, key
    return None, None


def build_financial_dataset(
    rows: list[ParsedFinancialRow],
) -> FinancialDataset:
    """Construit un FinancialDataset avec statuts et provenance."""
    dataset = empty_dataset()
    candidates: dict[str, list[tuple[Decimal, Decimal, ValueProvenance, str | None]]] = (
        defaultdict(list)
    )
    # code -> list of (value, confidence, provenance, raw)

    for row in rows:
        code, score = _match_field_code(row.normalized_label)
        if not code:
            continue
        prefer_n1 = code.endswith("_N1") or "n1" in code.lower()
        # Also capture N-1 column for CA/TOTAL when matching base field
        raw, col = _pick_raw_amount(row.values, prefer_n1=prefer_n1)
        if raw is None and not any(row.values.values()):
            continue

        # Special: CHIFFRE_AFFAIRES with net_n_1 also fills N1 candidate
        if code == "CHIFFRE_AFFAIRES" and row.values.get("net_n_1"):
            raw_n1 = row.values.get("net_n_1")
            amount_n1 = parse_decimal_amount(raw_n1)
            if amount_n1 is not None or is_explicit_zero(raw_n1):
                amount_n1 = amount_n1 if amount_n1 is not None else Decimal("0")
                candidates["CHIFFRE_AFFAIRES_N1"].append(
                    (
                        amount_n1,
                        score,
                        ValueProvenance(
                            page_number=row.page_number,
                            raw_label=row.raw_label,
                            raw_value=raw_n1,
                            column_name="net_n_1",
                            confidence=score,
                            source_excerpt=row.source_excerpt,
                        ),
                        raw_n1,
                    )
                )

        amount = parse_decimal_amount(raw)
        if amount is None:
            if is_explicit_zero(raw):
                amount = Decimal("0")
            else:
                continue

        provenance = ValueProvenance(
            page_number=row.page_number,
            raw_label=row.raw_label,
            raw_value=raw,
            column_name=col,
            confidence=score,
            source_excerpt=row.source_excerpt,
        )
        candidates[code].append((amount, score, provenance, raw))

    warnings: list[str] = []

    for code, entries in candidates.items():
        meta = _FIELD_META.get(code)
        if not meta:
            continue
        attr, label = meta
        amounts = {entry[0] for entry in entries}
        provenances = [entry[2] for entry in entries]
        fv: FinancialValue

        if len(amounts) == 1:
            value = next(iter(amounts))
            status: DataStatus = "confirmed"
            fv = FinancialValue(
                code=code,
                label=label,
                value=value,
                status=status,
                provenance=provenances,
            )
        elif len(amounts) > 1:
            # Conflit : ne pas choisir arbitrairement
            warnings.append(
                f"{code} : {len(amounts)} valeurs distinctes — marqué conflicting."
            )
            fv = FinancialValue(
                code=code,
                label=label,
                value=None,
                status="conflicting",
                provenance=provenances,
                warnings=["Valeurs OCR divergentes pour ce champ."],
            )
        else:
            continue

        if hasattr(dataset, attr):
            setattr(dataset, attr, fv)

    # Dérivés simples si absents
    dataset = _apply_simple_derived(dataset)
    dataset.warnings = warnings
    return dataset


def _apply_simple_derived(dataset: FinancialDataset) -> FinancialDataset:
    """Dérive TN / FDR / CAF simplifiée uniquement si composants confirmés."""
    ta = dataset.tresorerie_actif
    tp = dataset.tresorerie_passif
    if (
        dataset.tresorerie_nette.status == "missing"
        and ta.status in {"confirmed", "derived"}
        and tp.status in {"confirmed", "derived"}
        and ta.value is not None
        and tp.value is not None
    ):
        dataset.tresorerie_nette = FinancialValue(
            code="TRESORERIE_NETTE",
            label="Trésorerie nette",
            value=ta.value - tp.value,
            status="derived",
            provenance=ta.provenance + tp.provenance,
            warnings=["Dérivée : trésorerie actif - trésorerie passif"],
        )

    ac = dataset.actif_circulant
    pc = dataset.passif_circulant
    if (
        dataset.fdr.status == "missing"
        and all(
            f.status in {"confirmed", "derived"} and f.value is not None
            for f in (ac, ta, pc, tp)
        )
    ):
        assert ac.value is not None and ta.value is not None
        assert pc.value is not None and tp.value is not None
        dataset.fdr = FinancialValue(
            code="FDR",
            label="Fonds de roulement",
            value=(ac.value + ta.value) - (pc.value + tp.value),
            status="derived",
            provenance=ac.provenance + ta.provenance + pc.provenance + tp.provenance,
            warnings=[
                "FDR dérivé : (actif circulant + trésorerie actif) "
                "- (passif circulant + trésorerie passif)"
            ],
        )

    rn = dataset.resultat_net
    am = dataset.amortissements
    if (
        dataset.caf.status == "missing"
        and rn.status in {"confirmed", "derived"}
        and am.status in {"confirmed", "derived"}
        and rn.value is not None
        and am.value is not None
    ):
        dataset.caf = FinancialValue(
            code="CAF",
            label="CAF",
            value=rn.value + am.value,
            status="derived",
            provenance=rn.provenance + am.provenance,
            warnings=["CAF simplifiée, certains composants non disponibles"],
        )

    return dataset


def usable(field: FinancialValue | None) -> bool:
    return (
        field is not None
        and field.status in {"confirmed", "derived"}
        and field.value is not None
    )
