"""Configuration versionnée des seuils et pondérations de scoring (Decimal)."""
from __future__ import annotations

from decimal import Decimal

# Mode par défaut : STRICT = ratio essentiel manquant → axe non calculable
SCORING_MODE_DEFAULT = "STRICT"

AXIS_WEIGHTS = {
    "financial": Decimal("0.75"),
    "behavioral": Decimal("0.15"),
    "sector": Decimal("0.10"),
}

# Points max des ratios financiers (total 100)
FINANCIAL_RATIO_RULES: dict[str, dict] = {
    "financial_autonomy": {
        "direction": "higher_is_better",
        "good": Decimal("20"),
        "watch": Decimal("15"),
        "weight": Decimal("12"),
        "essential": True,
        "threshold_label": ">= 20 %",
    },
    "debt_ratio": {
        "direction": "lower_is_better",
        "good": Decimal("1.50"),
        "watch": Decimal("2.50"),
        "weight": Decimal("10"),
        "essential": True,
        "threshold_label": "<= 1,50",
    },
    "repayment_capacity": {
        "direction": "lower_is_better",
        "good": Decimal("3.00"),
        "watch": Decimal("5.00"),
        "weight": Decimal("12"),
        "essential": True,
        "threshold_label": "<= 3,00 x",
    },
    "caf_margin": {
        "direction": "higher_is_better",
        "good": Decimal("5.00"),
        "watch": Decimal("2.00"),
        "weight": Decimal("10"),
        "essential": True,
        "threshold_label": ">= 5 %",
    },
    "commercial_profitability": {
        "direction": "higher_is_better",
        "good": Decimal("5.00"),
        "watch": Decimal("2.00"),
        "weight": Decimal("8"),
        "essential": False,
        "threshold_label": ">= 5 %",
    },
    "financial_profitability": {
        "direction": "higher_is_better",
        "good": Decimal("10.00"),
        "watch": Decimal("5.00"),
        "weight": Decimal("8"),
        "essential": False,
        "threshold_label": ">= 10 %",
    },
    "economic_profitability": {
        "direction": "higher_is_better",
        "good": Decimal("5.00"),
        "watch": Decimal("2.00"),
        "weight": Decimal("6"),
        "essential": False,
        "threshold_label": ">= 5 %",
    },
    "fdr_ca": {
        "direction": "higher_is_better",
        "good": Decimal("0.00"),
        "watch": Decimal("-5.00"),
        "weight": Decimal("8"),
        "essential": False,
        "threshold_label": ">= 0 %",
    },
    "treasury_days": {
        "direction": "higher_is_better",
        "good": Decimal("0.00"),
        "watch": Decimal("-15.00"),
        "weight": Decimal("8"),
        "essential": False,
        "threshold_label": ">= 0 jours",
    },
    "customer_days": {
        "direction": "lower_is_better",
        "good": Decimal("60"),
        "watch": Decimal("90"),
        "weight": Decimal("10"),
        "essential": False,
        "threshold_label": "<= 60 jours",
    },
    "supplier_days": {
        "direction": "contextual",
        "good": None,
        "watch": None,
        "weight": Decimal("8"),
        "essential": False,
        "threshold_label": ">= délais clients",
    },
}
# ca_growth retiré de l'axe financier (poids 0) — reste ratio informatif / sectoriel.
# Somme des poids actifs = 100.

# Si True : dettes_financieres inclut déjà leasing/CMT → ne pas double-compter
DETTES_FINANCIERES_INCLUDES_LEASING = False
DETTES_FINANCIERES_INCLUDES_CMT = False

BAM_BLOCKING_RATINGS = {7, 8, 9}

BEHAVIORAL_RULES = {
    "domiciliation_good": Decimal("80"),
    "overdraft_watch": Decimal("40"),
    "debit_days_watch": 45,
    "flow_gap_watch": Decimal("5"),
}

ESSENTIAL_FIELDS_FOR_SCORING = (
    "chiffre_affaires",
    "resultat_net",
    "total_bilan",
    "fonds_propres",
)

USABLE_FIELD_STATUSES = frozenset({"confirmed", "derived"})
