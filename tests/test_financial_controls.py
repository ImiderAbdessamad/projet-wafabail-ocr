# -*- coding: utf-8 -*-
"""Tests contrôles comptables."""
from decimal import Decimal

from app.schemas.financial_analysis import FinancialValue
from app.services.financial_controls import (
    invalidate_conflicting_fields,
    run_accounting_controls,
)
from app.services.financial_dataset_builder import empty_dataset


def _fv(code: str, label: str, value: Decimal | None, status: str = "confirmed"):
    return FinancialValue(code=code, label=label, value=value, status=status)


def test_tresorerie_nette_coherence():
    ds = empty_dataset()
    ds.tresorerie_actif = _fv("TRESORERIE_ACTIF", "TA", Decimal("1000"))
    ds.tresorerie_passif = _fv("TRESORERIE_PASSIF", "TP", Decimal("400"))
    ds.tresorerie_nette = _fv("TRESORERIE_NETTE", "TN", Decimal("600"))
    checks = run_accounting_controls(ds)
    tn = next(c for c in checks if c.code == "tresorerie_nette")
    assert tn.status == "passed"


def test_bilan_mismatch_invalidates():
    ds = empty_dataset()
    ds.total_actif = _fv("TOTAL_ACTIF", "Actif", Decimal("1000"))
    ds.total_passif = _fv("TOTAL_PASSIF", "Passif", Decimal("900"))
    checks = run_accounting_controls(ds)
    eq = next(c for c in checks if c.code == "bilan_equilibre")
    assert eq.status == "failed"
    ds2 = invalidate_conflicting_fields(ds, checks)
    assert ds2.total_actif is not None
    assert ds2.total_actif.status == "conflicting"
    assert ds2.total_actif.value is None
