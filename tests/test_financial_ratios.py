# -*- coding: utf-8 -*-
"""Tests ratios financiers Decimal — fixture document de référence."""
from decimal import Decimal

from app.schemas.financial_analysis import FinancialValue
from app.services.financial_dataset_builder import empty_dataset
from app.services.financial_ratios import (
    calculate_ca_growth,
    calculate_caf_margin,
    calculate_commercial_profitability,
    calculate_customer_days,
    calculate_debt_ratio,
    calculate_economic_profitability,
    calculate_fdr_ca,
    calculate_financial_autonomy,
    calculate_financial_profitability,
    calculate_financial_ratios,
    calculate_global_debt_ratio,
    calculate_repayment_capacity,
    calculate_treasury_days,
    classify_ratio,
)


def _v(code: str, label: str, value: Decimal) -> FinancialValue:
    return FinancialValue(code=code, label=label, value=value, status="confirmed")


def reference_dataset():
    """Données d'exemple du document (KDH)."""
    ds = empty_dataset()
    ds.chiffre_affaires = _v("CHIFFRE_AFFAIRES", "CA", Decimal("38500"))
    ds.chiffre_affaires_n1 = _v("CHIFFRE_AFFAIRES_N1", "CA N-1", Decimal("34200"))
    ds.fonds_propres = _v("FONDS_PROPRES", "FP", Decimal("6200"))
    ds.total_bilan = _v("TOTAL_BILAN", "TB", Decimal("24800"))
    ds.dettes_financieres = _v("DETTES_FINANCIERES", "DF", Decimal("11400"))
    ds.resultat_net = _v("RESULTAT_NET", "RN", Decimal("1480"))
    ds.caf = _v("CAF", "CAF", Decimal("3200"))
    ds.fdr = _v("FDR", "FDR", Decimal("3600"))
    ds.tresorerie_nette = _v("TRESORERIE_NETTE", "TN", Decimal("500"))
    ds.clients = _v("CLIENTS", "Clients", Decimal("8340"))
    ds.fournisseurs = _v("FOURNISSEURS", "Fournisseurs", Decimal("5600"))
    ds.achats = _v("ACHATS", "Achats", Decimal("31500"))
    ds.encours_leasing = _v("ENCOURS_LEASING", "Leasing", Decimal("3200"))
    ds.cmt = _v("CMT", "CMT", Decimal("8200"))
    ds.nouveau_financement = _v("NOUVEAU_FINANCEMENT", "NF", Decimal("4800"))
    return ds


def test_autonomy_25_pct():
    r = calculate_financial_autonomy(reference_dataset())
    assert r.value == Decimal("25.00")
    assert r.status == "conforme"


def test_debt_ratio_1_84_watch():
    r = calculate_debt_ratio(reference_dataset())
    assert r.value == Decimal("1.84")
    assert r.status == "a_surveiller"


def test_repayment_3_56_watch():
    r = calculate_repayment_capacity(reference_dataset())
    assert r.value == Decimal("3.56")
    assert r.status == "a_surveiller"


def test_caf_margin_8_31():
    r = calculate_caf_margin(reference_dataset())
    assert r.value == Decimal("8.31")
    assert r.status == "conforme"


def test_commercial_3_84_watch():
    r = calculate_commercial_profitability(reference_dataset())
    assert r.value == Decimal("3.84")
    assert r.status == "a_surveiller"


def test_financial_profit_23_87():
    r = calculate_financial_profitability(reference_dataset())
    assert r.value == Decimal("23.87")
    assert r.status == "conforme"


def test_economic_8_41():
    r = calculate_economic_profitability(reference_dataset())
    assert r.value == Decimal("8.41")


def test_fdr_ca_9_35():
    r = calculate_fdr_ca(reference_dataset())
    assert r.value == Decimal("9.35")
    assert r.status == "conforme"


def test_treasury_days_4_68():
    r = calculate_treasury_days(reference_dataset())
    assert r.value == Decimal("4.68")
    assert r.status == "conforme"


def test_customer_days_77_98():
    r = calculate_customer_days(reference_dataset())
    assert r.value == Decimal("77.98")
    assert r.status == "a_surveiller"


def test_ca_growth_12_57():
    r = calculate_ca_growth(reference_dataset())
    assert r.value == Decimal("12.57")
    assert r.max_points == Decimal("0")
    assert r.points == Decimal("0")


def test_global_debt_2_61():
    r = calculate_global_debt_ratio(reference_dataset())
    # 3200+8200+4800=16200 / 6200 = 2.6129 → 2.61
    assert r.value == Decimal("2.61")


def test_ambiguous_field_blocks_ratio():
    ds = reference_dataset()
    ds.fonds_propres.status = "ambiguous"
    ds.fonds_propres.value = None
    r = calculate_financial_autonomy(ds)
    assert r.status == "non_calculable"
    assert r.value is None


def test_division_by_zero():
    ds = reference_dataset()
    ds.total_bilan.value = Decimal("0")
    r = calculate_financial_autonomy(ds)
    assert r.status == "non_calculable"


def test_classify_ratio_directions():
    assert classify_ratio(Decimal("25"), direction="higher_is_better", good=Decimal("20"), watch=Decimal("15")) == "conforme"
    assert classify_ratio(Decimal("1.84"), direction="lower_is_better", good=Decimal("1.50"), watch=Decimal("2.50")) == "a_surveiller"


def test_all_ratios_are_decimal_or_none():
    for ratio in calculate_financial_ratios(reference_dataset()):
        assert ratio.value is None or isinstance(ratio.value, Decimal)
        assert isinstance(ratio.points, Decimal)
