# -*- coding: utf-8 -*-
"""Tests normalizer Decimal (sans Ollama)."""
from decimal import Decimal

from app.services.financial_normalizer import (
    is_explicit_zero,
    parse_decimal_amount,
    quantize_ratio,
    safe_divide,
)


def test_parse_european_thousands_dot():
    assert parse_decimal_amount("22.303.497,11") == Decimal("22303497.11")


def test_parse_negative_dot():
    assert parse_decimal_amount("-193.846,67") == Decimal("-193846.67")


def test_parse_parentheses_negative():
    assert parse_decimal_amount("(193.846,67)") == Decimal("-193846.67")


def test_parse_empty_is_none():
    assert parse_decimal_amount("") is None
    assert parse_decimal_amount(None) is None
    assert parse_decimal_amount("—") is None
    assert parse_decimal_amount("-") is None


def test_explicit_zero():
    assert is_explicit_zero("0,00") is True
    assert is_explicit_zero("0") is True
    assert is_explicit_zero("") is False
    assert is_explicit_zero("—") is False


def test_safe_divide_and_quantize():
    assert safe_divide(Decimal("6200"), Decimal("24800")) == Decimal("6200") / Decimal(
        "24800"
    )
    assert safe_divide(Decimal("1"), Decimal("0")) is None
    assert safe_divide(None, Decimal("1")) is None
    assert quantize_ratio(Decimal("1.838709")) == Decimal("1.84")
