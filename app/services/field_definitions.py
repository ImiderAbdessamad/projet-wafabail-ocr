"""Registre central des définitions métier PCGM (liasse fiscale marocaine).

Aucune valeur document-spécifique ici — uniquement règles génériques.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Aggregation = Literal["direct", "sum_components", "derived"]
ValueNature = Literal[
    "brut",
    "amortissement_provision",
    "net_n",
    "net_n_1",
    "exercice",
    "exercices_precedents",
    "total_exercice",
    "unknown",
]


@dataclass(frozen=True)
class FieldDefinition:
    code: str
    label: str
    sections: tuple[str, ...]
    aliases: tuple[str, ...]
    preferred_columns: tuple[str, ...] = ("net_n", "total_exercice", "exercice")
    forbidden_columns: tuple[str, ...] = ()
    aggregation: Aggregation = "direct"
    component_codes: tuple[str, ...] = ()
    allow_zero_when_empty_line: bool = False
    scoring_key: str | None = None  # clé ScoringInput si différente
    element_number: int | None = None  # 1..19 si élément canonique


# --- Alias normalisés (sans accents côté matching via normalize_label) ------

FIELD_DEFINITIONS: dict[str, FieldDefinition] = {
    "ACTIFS_IMMOBILISES": FieldDefinition(
        code="ACTIFS_IMMOBILISES",
        label="Actifs immobilisés",
        sections=("BILAN_ACTIF",),
        aliases=(
            "total i actif immobilise",
            "total actif immobilise",
            "actif immobilise",
            "actifs immobilises",
            "immobilisations",
        ),
        preferred_columns=("net_n",),
        forbidden_columns=("brut", "amortissement_provision", "net_n_1"),
        element_number=1,
        scoring_key="actifs_immobilises",
    ),
    "TOTAL_BILAN": FieldDefinition(
        code="TOTAL_BILAN",
        label="Total du bilan",
        sections=("BILAN_ACTIF", "BILAN_PASSIF"),
        aliases=(
            "total general",
            "total general i ii iii",
            "total actif",
            "total du bilan",
            "total passif",
        ),
        preferred_columns=("net_n",),
        forbidden_columns=("brut", "amortissement_provision", "net_n_1"),
        element_number=2,
        scoring_key="total_bilan",
    ),
    "CHIFFRE_AFFAIRES": FieldDefinition(
        code="CHIFFRE_AFFAIRES",
        label="Chiffre d'affaires",
        sections=("CPC",),
        aliases=(
            "chiffre d affaires",
            "chiffres d affaires",
            "ventes de biens et services produits",
            "produits d exploitation",
        ),
        preferred_columns=("total_exercice", "exercice", "net_n"),
        element_number=3,
        scoring_key="chiffre_affaires",
    ),
    "CA_EXPORT": FieldDefinition(
        code="CA_EXPORT",
        label="CA à l'export",
        sections=("CPC",),
        aliases=(
            "dont a l export",
            "ventes a l export",
            "chiffre d affaires export",
            "ca a l export",
            "ca export",
        ),
        preferred_columns=("total_exercice", "exercice", "net_n"),
        allow_zero_when_empty_line=True,
        element_number=4,
        scoring_key="ca_export",
    ),
    "CA_N1": FieldDefinition(
        code="CA_N1",
        label="Chiffre d'affaires N-1",
        sections=("CPC", "BILAN_ACTIF", "AUTRE"),
        aliases=(
            "chiffre d affaires n 1",
            "chiffre d affaires exercice precedent",
            "ca n 1",
        ),
        preferred_columns=("net_n_1", "exercice"),
        scoring_key="ca_n1",
    ),
    "DETTES_BANCAIRES_MLT": FieldDefinition(
        code="DETTES_BANCAIRES_MLT",
        label="Dettes bancaires MLT",
        sections=("BILAN_PASSIF",),
        aliases=(
            "dettes de financement",
            "emprunts aupres des etablissements de credit",
            "emprunts bancaires",
            "dettes bancaires moyen et long terme",
            "emprunts et dettes assimilées",
            "autres dettes de financement",
        ),
        preferred_columns=("net_n", "exercice"),
        forbidden_columns=("net_n_1",),
        element_number=5,
        scoring_key="dettes_financieres",  # MLT seul — CT ajouté en dérivé
    ),
    "DETTES_BANCAIRES_CT": FieldDefinition(
        code="DETTES_BANCAIRES_CT",
        label="Dettes bancaires CT",
        sections=("BILAN_PASSIF",),
        aliases=(
            "tresorerie passif",
            "credits de tresorerie",
            "banques soldes crediteurs",
            "concours bancaires courants",
            "dettes bancaires court terme",
        ),
        preferred_columns=("net_n", "exercice"),
        aggregation="sum_components",
        component_codes=("CREDITS_TRESORERIE", "BANQUES_SOLDES_CREDITEURS"),
        element_number=6,
        scoring_key="dettes_bancaires_ct",
    ),
    "CREDITS_TRESORERIE": FieldDefinition(
        code="CREDITS_TRESORERIE",
        label="Crédits de trésorerie",
        sections=("BILAN_PASSIF",),
        aliases=("credits de tresorerie", "credit de tresorerie"),
        preferred_columns=("net_n", "exercice"),
    ),
    "BANQUES_SOLDES_CREDITEURS": FieldDefinition(
        code="BANQUES_SOLDES_CREDITEURS",
        label="Banques, soldes créditeurs",
        sections=("BILAN_PASSIF",),
        aliases=(
            "banques soldes crediteurs",
            "banques creditrices",
            "soldes crediteurs de banques",
        ),
        preferred_columns=("net_n", "exercice"),
    ),
    "PASSIF_CIRCULANT": FieldDefinition(
        code="PASSIF_CIRCULANT",
        label="Passif circulant",
        sections=("BILAN_PASSIF",),
        aliases=(
            "passif circulant",
            "total ii passif circulant",
            "total passif circulant",
        ),
        preferred_columns=("net_n", "exercice"),
        forbidden_columns=("net_n_1",),
        element_number=7,
        scoring_key="passif_circulant",
    ),
    "DETTES_FOURNISSEURS": FieldDefinition(
        code="DETTES_FOURNISSEURS",
        label="Dettes fournisseurs",
        sections=("BILAN_PASSIF",),
        aliases=(
            "fournisseurs et comptes rattaches",
            "fournisseurs",
            "dettes fournisseurs",
        ),
        preferred_columns=("net_n", "exercice"),
        element_number=8,
        scoring_key="fournisseurs",
    ),
    "COMPTE_COURANT_ASSOCIES": FieldDefinition(
        code="COMPTE_COURANT_ASSOCIES",
        label="Compte courant d'associés",
        sections=("BILAN_PASSIF",),
        aliases=(
            "comptes d associes crediteurs",
            "associes comptes courants",
            "comptes courants d associes",
            "associes",
            "compte courant d associes",
        ),
        preferred_columns=("net_n", "exercice"),
        element_number=9,
        scoring_key="compte_courant_associes",
    ),
    "TRESORERIE_PASSIF": FieldDefinition(
        code="TRESORERIE_PASSIF",
        label="Trésorerie au passif",
        sections=("BILAN_PASSIF",),
        aliases=(
            "tresorerie passif",
            "total iii tresorerie passif",
            "tresorerie passif total",
        ),
        preferred_columns=("net_n", "exercice"),
        aggregation="sum_components",
        component_codes=("CREDITS_TRESORERIE", "BANQUES_SOLDES_CREDITEURS"),
        element_number=10,
        scoring_key="tresorerie_passif",
    ),
    "ACTIF_CIRCULANT": FieldDefinition(
        code="ACTIF_CIRCULANT",
        label="Actif circulant",
        sections=("BILAN_ACTIF",),
        aliases=(
            "actif circulant",
            "total ii actif circulant",
            "total actif circulant",
        ),
        preferred_columns=("net_n",),
        forbidden_columns=("brut", "amortissement_provision", "net_n_1"),
        element_number=11,
        scoring_key="actif_circulant",
    ),
    "CREANCES_CLIENTS": FieldDefinition(
        code="CREANCES_CLIENTS",
        label="Créances clients",
        sections=("BILAN_ACTIF",),
        aliases=(
            "clients et comptes rattaches",
            "clients",
            "creances clients",
            "clients debiteurs",
        ),
        preferred_columns=("net_n",),
        forbidden_columns=("brut", "amortissement_provision", "net_n_1"),
        element_number=12,
        scoring_key="clients",
    ),
    "TRESORERIE_ACTIF": FieldDefinition(
        code="TRESORERIE_ACTIF",
        label="Trésorerie à l'actif",
        sections=("BILAN_ACTIF",),
        aliases=(
            "tresorerie actif",
            "total iii tresorerie actif",
            "tresorerie",
        ),
        preferred_columns=("net_n",),
        forbidden_columns=("brut", "net_n_1"),
        element_number=13,
        scoring_key="tresorerie_actif",
    ),
    "CAISSE": FieldDefinition(
        code="CAISSE",
        label="Caisse",
        sections=("BILAN_ACTIF",),
        aliases=(
            "caisse regies d avances et accreditifs",
            "caisse",
            "caisse et regies d avances",
        ),
        preferred_columns=("net_n",),
        forbidden_columns=("brut", "net_n_1"),
        element_number=14,
        scoring_key="caisse",
    ),
    "ACHATS_REVENDUS": FieldDefinition(
        code="ACHATS_REVENDUS",
        label="Achats revendus",
        sections=("CPC",),
        aliases=(
            "achats revendus de marchandises",
            "achats de marchandises",
            "achats revendus",
        ),
        preferred_columns=("total_exercice", "exercice"),
        element_number=15,
        scoring_key="achats",
    ),
    "ACHATS_CONSOMMES": FieldDefinition(
        code="ACHATS_CONSOMMES",
        label="Achats consommés de matières et fournitures",
        sections=("CPC",),
        aliases=(
            "achats consommes de matieres et fournitures",
            "achats consommes",
            "matieres et fournitures",
        ),
        preferred_columns=("total_exercice", "exercice"),
    ),
    "AUTRES_CHARGES_EXTERNES": FieldDefinition(
        code="AUTRES_CHARGES_EXTERNES",
        label="Autres charges externes",
        sections=("CPC",),
        aliases=("autres charges externes",),
        preferred_columns=("total_exercice", "exercice"),
    ),
    "IMPOTS_TAXES": FieldDefinition(
        code="IMPOTS_TAXES",
        label="Impôts et taxes",
        sections=("CPC",),
        aliases=("impots et taxes", "impots taxes"),
        preferred_columns=("total_exercice", "exercice"),
    ),
    "CHARGES_PERSONNEL": FieldDefinition(
        code="CHARGES_PERSONNEL",
        label="Charges de personnel",
        sections=("CPC",),
        aliases=("charges de personnel", "charges personnel"),
        preferred_columns=("total_exercice", "exercice"),
    ),
    "AUTRES_CHARGES_EXPLOITATION": FieldDefinition(
        code="AUTRES_CHARGES_EXPLOITATION",
        label="Autres charges d'exploitation",
        sections=("CPC",),
        aliases=("autres charges d exploitation",),
        preferred_columns=("total_exercice", "exercice"),
    ),
    "DOTATIONS_EXPLOITATION": FieldDefinition(
        code="DOTATIONS_EXPLOITATION",
        label="Dotations d'exploitation",
        sections=("CPC",),
        aliases=(
            "dotations d exploitation",
            "dotations aux amortissements et provisions d exploitation",
        ),
        preferred_columns=("total_exercice", "exercice"),
    ),
    "AUTRES_CHARGES": FieldDefinition(
        code="AUTRES_CHARGES",
        label="Autres charges",
        sections=("CPC",),
        aliases=("autres charges",),
        preferred_columns=("total_exercice", "exercice"),
        aggregation="sum_components",
        component_codes=(
            "ACHATS_CONSOMMES",
            "AUTRES_CHARGES_EXTERNES",
            "IMPOTS_TAXES",
            "CHARGES_PERSONNEL",
            "AUTRES_CHARGES_EXPLOITATION",
            "DOTATIONS_EXPLOITATION",
        ),
        element_number=16,
    ),
    "CHARGES_INTERETS": FieldDefinition(
        code="CHARGES_INTERETS",
        label="Charges d'intérêts",
        sections=("CPC",),
        aliases=(
            "charges d interets",
            "interets des emprunts et dettes",
            "charges financieres d interets",
            "charges d interets et frais assimiles",
        ),
        # Refuser explicitement « autres charges d'exploitation »
        preferred_columns=("total_exercice", "exercice"),
        element_number=17,
        scoring_key="frais_financiers",
    ),
    "RESULTAT_NET": FieldDefinition(
        code="RESULTAT_NET",
        label="Résultat net",
        sections=("CPC", "BILAN_PASSIF", "AUTRE"),
        aliases=(
            "resultat net de l exercice",
            "resultat net",
            "benefice net",
            "perte nette",
        ),
        preferred_columns=("total_exercice", "net_n", "exercice"),
        forbidden_columns=("exercices_precedents",),
        element_number=18,
        scoring_key="resultat_net",
    ),
    "FONDS_PROPRES": FieldDefinition(
        code="FONDS_PROPRES",
        label="Fonds propres",
        sections=("BILAN_PASSIF",),
        aliases=(
            "total des capitaux propres",
            "capitaux propres",
            "capitaux propres assimilés",
            "fonds propres",
            "total i financement permanent",
        ),
        preferred_columns=("net_n", "exercice"),
        scoring_key="fonds_propres",
    ),
    "CAF": FieldDefinition(
        code="CAF",
        label="Capacité d'autofinancement",
        sections=("CPC", "AUTRE"),
        aliases=(
            "capacite d autofinancement",
            "caf",
            "autofinancement",
            "capacite d autofinancement globale",
        ),
        preferred_columns=("total_exercice", "exercice", "net_n"),
        scoring_key="caf",
    ),
    "AMORTISSEMENTS": FieldDefinition(
        code="AMORTISSEMENTS",
        label="Dotations aux amortissements",
        sections=("CPC",),
        aliases=(
            "dotations d exploitation",
            "dotations aux amortissements et provisions",
            "dotations aux amortissements",
        ),
        preferred_columns=("total_exercice", "exercice"),
        # Interdit : cumul bilan (colonne amortissement_provision)
        forbidden_columns=("amortissement_provision", "brut", "net_n"),
        scoring_key="amortissements",
    ),
    "FDR": FieldDefinition(
        code="FDR",
        label="Fonds de roulement",
        sections=("BILAN_PASSIF", "AUTRE"),
        aliases=(
            "fonds de roulement",
            "fonds de roulement fonctionnel",
        ),
        preferred_columns=("net_n", "exercice"),
        aggregation="derived",
        scoring_key="fdr",
    ),
}

# Colonnes : mapping en-têtes Vision → value_nature
COLUMN_HEADER_MAP: dict[str, str] = {
    "brut": "brut",
    "bruts": "brut",
    "amortissements et provisions": "amortissement_provision",
    "amortissements": "amortissement_provision",
    "provisions": "amortissement_provision",
    "net": "net_n",
    "net exercice": "net_n",
    "net exercice n": "net_n",
    "net de l exercice": "net_n",
    "exercice n": "net_n",
    "net exercice n 1": "net_n_1",
    "net n 1": "net_n_1",
    "exercice precedent": "net_n_1",
    "operations propres a l exercice": "exercice",
    "operations concernant les exercices precedents": "exercices_precedents",
    "total de l exercice": "total_exercice",
    "total exercice": "total_exercice",
    "exercice": "exercice",
}

METADATA_ALIASES: dict[str, tuple[str, ...]] = {
    "reference": (
        "reference",
        "numero de depot",
        "n depot",
        "reference du depot",
        "sis",
    ),
    "entreprise": (
        "raison sociale",
        "denomination sociale",
        "entreprise",
        "societe",
        "nom ou raison sociale",
    ),
    "identification_fiscale": (
        "identifiant fiscal",
        "identification fiscale",
        "if",
        "n if",
    ),
    "exercice": ("exercice", "annee", "periode"),
    "date_debut_exercice": (
        "date de debut",
        "du",
        "periode du",
        "date debut exercice",
    ),
    "date_fin_exercice": (
        "date de fin",
        "au",
        "periode au",
        "date fin exercice",
    ),
}

# Alias exclusifs : labels à ne JAMAIS matcher pour un code donné
FIELD_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "CHARGES_INTERETS": (
        "autres charges d exploitation",
        "autres charges externes",
        "charges de personnel",
        "dotations d exploitation",
    ),
    "AMORTISSEMENTS": (
        "amortissements et provisions",  # colonne bilan
        "cumul des amortissements",
    ),
    "TRESORERIE_PASSIF": (
        "passif circulant",
        "total passif circulant",
    ),
    "DETTES_BANCAIRES_CT": (
        "passif circulant",
        "dettes de financement",
    ),
}
