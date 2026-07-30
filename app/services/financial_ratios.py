"""Calcul déterministe des ratios financiers (Decimal, formules en code)."""
from __future__ import annotations

from decimal import Decimal

from app.scoring_rules import FINANCIAL_RATIO_RULES
from app.schemas.financial_analysis import (
    FinancialDataset,
    FinancialValue,
    RatioComponent,
    RatioResult,
    RatioStatus,
)
from app.services.financial_dataset_builder import usable
from app.services.financial_normalizer import (
    DAYS_PER_YEAR,
    ONE_HUNDRED,
    quantize_ratio,
    safe_divide,
)


def classify_ratio(
    value: Decimal | None,
    *,
    direction: str,
    good: Decimal | None,
    watch: Decimal | None,
) -> RatioStatus:
    if value is None:
        return "non_calculable"

    if direction == "higher_is_better":
        if good is not None and value >= good:
            return "conforme"
        if watch is not None and value >= watch:
            return "a_surveiller"
        return "non_conforme"

    if direction == "lower_is_better":
        if good is not None and value <= good:
            return "conforme"
        if watch is not None and value <= watch:
            return "a_surveiller"
        return "non_conforme"

    if direction == "contextual":
        return "non_calculable"

    raise ValueError(f"Direction inconnue : {direction}")


def _component(fv: FinancialValue) -> RatioComponent:
    return RatioComponent(
        code=fv.code,
        label=fv.label,
        value=fv.value,
        status=fv.status,
    )


def _non_calculable(
    code: str,
    label: str,
    formula: str,
    unit: str,
    components: list[FinancialValue],
    warnings: list[str] | None = None,
) -> RatioResult:
    rule = FINANCIAL_RATIO_RULES.get(code, {})
    return RatioResult(
        code=code,
        label=label,
        formula=formula,
        value=None,
        unit=unit,
        components=[_component(c) for c in components],
        threshold=rule.get("threshold_label"),
        status="non_calculable",
        points=Decimal("0"),
        max_points=Decimal(str(rule.get("weight", 0))),
        warnings=warnings or ["Composants manquants, ambigus ou invalides."],
    )


def _finalize(
    code: str,
    label: str,
    formula: str,
    value: Decimal | None,
    unit: str,
    components: list[FinancialValue],
    *,
    customer_days: Decimal | None = None,
    supplier_days: Decimal | None = None,
) -> RatioResult:
    rule = FINANCIAL_RATIO_RULES[code]
    max_points = Decimal(str(rule["weight"]))
    if value is None:
        return _non_calculable(code, label, formula, unit, components)

    direction = rule["direction"]
    if direction == "contextual":
        status = _classify_supplier_days(supplier_days, customer_days)
    else:
        status = classify_ratio(
            value,
            direction=direction,
            good=rule.get("good"),
            watch=rule.get("watch"),
        )

    factors = {
        "conforme": Decimal("1.00"),
        "a_surveiller": Decimal("0.60"),
        "non_conforme": Decimal("0.00"),
        "non_calculable": Decimal("0.00"),
    }
    points = (max_points * factors[status]).quantize(Decimal("0.01"))
    return RatioResult(
        code=code,
        label=label,
        formula=formula,
        value=quantize_ratio(value),
        unit=unit,
        components=[_component(c) for c in components],
        threshold=rule.get("threshold_label"),
        status=status,
        points=points,
        max_points=max_points,
    )


def _classify_supplier_days(
    supplier_days: Decimal | None,
    customer_days: Decimal | None,
) -> RatioStatus:
    if supplier_days is None or customer_days is None:
        return "non_calculable"
    gap = supplier_days - customer_days
    if gap >= 0:
        return "conforme"
    if gap >= Decimal("-30"):
        return "a_surveiller"
    return "non_conforme"


def calculate_financial_autonomy(dataset: FinancialDataset) -> RatioResult:
    fp, tb = dataset.fonds_propres, dataset.total_bilan
    if not usable(fp) or not usable(tb):
        return _non_calculable(
            "financial_autonomy",
            "Autonomie financière",
            "fonds_propres / total_bilan * 100",
            "%",
            [fp, tb],
        )
    value = safe_divide(fp.value, tb.value)
    if value is not None:
        value = value * ONE_HUNDRED
    return _finalize(
        "financial_autonomy",
        "Autonomie financière",
        "fonds_propres / total_bilan * 100",
        value,
        "%",
        [fp, tb],
    )


def calculate_debt_ratio(dataset: FinancialDataset) -> RatioResult:
    df, fp = dataset.dettes_financieres, dataset.fonds_propres
    if not usable(df) or not usable(fp):
        return _non_calculable(
            "debt_ratio",
            "Ratio d'endettement",
            "dettes_financieres / fonds_propres",
            "x",
            [df, fp],
        )
    return _finalize(
        "debt_ratio",
        "Ratio d'endettement",
        "dettes_financieres / fonds_propres",
        safe_divide(df.value, fp.value),
        "x",
        [df, fp],
    )


def calculate_repayment_capacity(dataset: FinancialDataset) -> RatioResult:
    df, caf = dataset.dettes_financieres, dataset.caf
    if not usable(df) or not usable(caf):
        return _non_calculable(
            "repayment_capacity",
            "Capacité de remboursement",
            "dettes_financieres / caf",
            "x",
            [df, caf],
        )
    return _finalize(
        "repayment_capacity",
        "Capacité de remboursement",
        "dettes_financieres / caf",
        safe_divide(df.value, caf.value),
        "x",
        [df, caf],
    )


def calculate_caf_margin(dataset: FinancialDataset) -> RatioResult:
    caf, ca = dataset.caf, dataset.chiffre_affaires
    if not usable(caf) or not usable(ca):
        return _non_calculable(
            "caf_margin",
            "CAF / CA",
            "caf / chiffre_affaires * 100",
            "%",
            [caf, ca],
        )
    value = safe_divide(caf.value, ca.value)
    if value is not None:
        value *= ONE_HUNDRED
    return _finalize(
        "caf_margin",
        "CAF / CA",
        "caf / chiffre_affaires * 100",
        value,
        "%",
        [caf, ca],
    )


def calculate_commercial_profitability(dataset: FinancialDataset) -> RatioResult:
    rn, ca = dataset.resultat_net, dataset.chiffre_affaires
    if not usable(rn) or not usable(ca):
        return _non_calculable(
            "commercial_profitability",
            "Rentabilité commerciale",
            "resultat_net / chiffre_affaires * 100",
            "%",
            [rn, ca],
        )
    value = safe_divide(rn.value, ca.value)
    if value is not None:
        value *= ONE_HUNDRED
    return _finalize(
        "commercial_profitability",
        "Rentabilité commerciale",
        "resultat_net / chiffre_affaires * 100",
        value,
        "%",
        [rn, ca],
    )


def calculate_financial_profitability(dataset: FinancialDataset) -> RatioResult:
    rn, fp = dataset.resultat_net, dataset.fonds_propres
    if not usable(rn) or not usable(fp):
        return _non_calculable(
            "financial_profitability",
            "Rentabilité financière",
            "resultat_net / fonds_propres * 100",
            "%",
            [rn, fp],
        )
    value = safe_divide(rn.value, fp.value)
    if value is not None:
        value *= ONE_HUNDRED
    return _finalize(
        "financial_profitability",
        "Rentabilité financière",
        "resultat_net / fonds_propres * 100",
        value,
        "%",
        [rn, fp],
    )


def calculate_economic_profitability(dataset: FinancialDataset) -> RatioResult:
    rn, fp, df = dataset.resultat_net, dataset.fonds_propres, dataset.dettes_financieres
    if not usable(rn) or not usable(fp) or not usable(df):
        return _non_calculable(
            "economic_profitability",
            "Rentabilité économique",
            "resultat_net / (fonds_propres + dettes_financieres) * 100",
            "%",
            [rn, fp, df],
        )
    denom = (fp.value or Decimal("0")) + (df.value or Decimal("0"))
    value = safe_divide(rn.value, denom)
    if value is not None:
        value *= ONE_HUNDRED
    return _finalize(
        "economic_profitability",
        "Rentabilité économique",
        "resultat_net / (fonds_propres + dettes_financieres) * 100",
        value,
        "%",
        [rn, fp, df],
    )


def calculate_fdr_ca(dataset: FinancialDataset) -> RatioResult:
    fdr, ca = dataset.fdr, dataset.chiffre_affaires
    if not usable(fdr) or not usable(ca):
        return _non_calculable(
            "fdr_ca",
            "FDR / CA",
            "fdr / chiffre_affaires * 100",
            "%",
            [fdr, ca],
        )
    value = safe_divide(fdr.value, ca.value)
    if value is not None:
        value *= ONE_HUNDRED
    return _finalize(
        "fdr_ca",
        "FDR / CA",
        "fdr / chiffre_affaires * 100",
        value,
        "%",
        [fdr, ca],
    )


def calculate_treasury_days(dataset: FinancialDataset) -> RatioResult:
    tn, ca = dataset.tresorerie_nette, dataset.chiffre_affaires
    if not usable(tn) or not usable(ca):
        return _non_calculable(
            "treasury_days",
            "Trésorerie en jours de CA",
            "tresorerie_nette / chiffre_affaires * 360",
            "jours",
            [tn, ca],
        )
    value = safe_divide(tn.value, ca.value)
    if value is not None:
        value *= DAYS_PER_YEAR
    return _finalize(
        "treasury_days",
        "Trésorerie en jours de CA",
        "tresorerie_nette / chiffre_affaires * 360",
        value,
        "jours",
        [tn, ca],
    )


def calculate_customer_days(dataset: FinancialDataset) -> RatioResult:
    cl, ca = dataset.clients, dataset.chiffre_affaires
    if not usable(cl) or not usable(ca):
        return _non_calculable(
            "customer_days",
            "Délais clients",
            "clients / chiffre_affaires * 360",
            "jours",
            [cl, ca],
        )
    value = safe_divide(cl.value, ca.value)
    if value is not None:
        value *= DAYS_PER_YEAR
    return _finalize(
        "customer_days",
        "Délais clients",
        "clients / chiffre_affaires * 360",
        value,
        "jours",
        [cl, ca],
    )


def calculate_supplier_days(
    dataset: FinancialDataset,
    *,
    customer_days: Decimal | None = None,
) -> RatioResult:
    fo, ach = dataset.fournisseurs, dataset.achats
    if not usable(fo) or not usable(ach):
        return _non_calculable(
            "supplier_days",
            "Délais fournisseurs",
            "fournisseurs / achats * 360",
            "jours",
            [fo, ach],
        )
    value = safe_divide(fo.value, ach.value)
    if value is not None:
        value *= DAYS_PER_YEAR
    return _finalize(
        "supplier_days",
        "Délais fournisseurs",
        "fournisseurs / achats * 360",
        value,
        "jours",
        [fo, ach],
        customer_days=customer_days,
        supplier_days=quantize_ratio(value),
    )


def calculate_ca_growth(dataset: FinancialDataset) -> RatioResult:
    """Ratio informatif / sectoriel — aucun point dans l'axe financier."""
    ca, ca_n1 = dataset.chiffre_affaires, dataset.chiffre_affaires_n1
    if not usable(ca) or not usable(ca_n1):
        return RatioResult(
            code="ca_growth",
            label="Croissance du CA",
            formula="(ca_n - ca_n1) / abs(ca_n1) * 100",
            value=None,
            unit="%",
            components=[_component(ca), _component(ca_n1)],
            threshold=">= 5 %",
            status="non_calculable",
            points=Decimal("0"),
            max_points=Decimal("0"),
            warnings=["Composants manquants, ambigus ou invalides."],
        )
    assert ca.value is not None and ca_n1.value is not None
    value = safe_divide(ca.value - ca_n1.value, abs(ca_n1.value))
    if value is not None:
        value *= ONE_HUNDRED
    value = quantize_ratio(value)
    status = classify_ratio(
        value,
        direction="higher_is_better",
        good=Decimal("5.00"),
        watch=Decimal("0.00"),
    ) if value is not None else "non_calculable"
    return RatioResult(
        code="ca_growth",
        label="Croissance du CA",
        formula="(ca_n - ca_n1) / abs(ca_n1) * 100",
        value=value,
        unit="%",
        components=[_component(ca), _component(ca_n1)],
        threshold=">= 5 %",
        status=status,
        points=Decimal("0"),
        max_points=Decimal("0"),
        warnings=[],
    )


def calculate_global_debt_ratio(dataset: FinancialDataset) -> RatioResult:
    """Endettement global après opération / fonds propres."""
    parts = [
        dataset.encours_leasing,
        dataset.cmt,
        dataset.nouveau_financement,
    ]
    if not all(usable(p) for p in parts) or not usable(dataset.fonds_propres):
        return RatioResult(
            code="global_debt_ratio",
            label="Endettement global / Fonds propres",
            formula="(encours_leasing + cmt + nouveau_financement) / fonds_propres",
            value=None,
            unit="x",
            components=[_component(p) for p in parts + [dataset.fonds_propres]],
            status="non_calculable",
            points=Decimal("0"),
            max_points=Decimal("0"),
            warnings=["Composants d'endettement global incomplets."],
        )
    total = sum((p.value for p in parts if p.value is not None), Decimal("0"))
    value = safe_divide(total, dataset.fonds_propres.value)
    return RatioResult(
        code="global_debt_ratio",
        label="Endettement global / Fonds propres",
        formula="(encours_leasing + cmt + nouveau_financement) / fonds_propres",
        value=quantize_ratio(value),
        unit="x",
        components=[_component(p) for p in parts + [dataset.fonds_propres]],
        status="conforme" if value is not None and value <= Decimal("3") else (
            "a_surveiller" if value is not None and value <= Decimal("4") else (
                "non_conforme" if value is not None else "non_calculable"
            )
        ),
        points=Decimal("0"),
        max_points=Decimal("0"),
    )


def calculate_financial_ratios(dataset: FinancialDataset) -> list[RatioResult]:
    customer = calculate_customer_days(dataset)
    ratios = [
        calculate_financial_autonomy(dataset),
        calculate_debt_ratio(dataset),
        calculate_repayment_capacity(dataset),
        calculate_caf_margin(dataset),
        calculate_commercial_profitability(dataset),
        calculate_financial_profitability(dataset),
        calculate_economic_profitability(dataset),
        calculate_fdr_ca(dataset),
        calculate_treasury_days(dataset),
        customer,
        calculate_supplier_days(dataset, customer_days=customer.value),
        calculate_ca_growth(dataset),
        calculate_global_debt_ratio(dataset),
    ]
    return ratios
