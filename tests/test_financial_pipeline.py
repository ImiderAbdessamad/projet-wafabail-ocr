# -*- coding: utf-8 -*-
"""Tests du pipeline observation → résolution → validation (sans Vision)."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.services.accounting_checks import run_accounting_checks
from app.services.amount_parser import parse_amount
from app.services.document_inspector import classify_page_text
from app.services.derived_fields import apply_derived_fields
from app.services.field_resolver import (
    extract_document_metadata,
    observations_from_page_payload,
    resolve_all_fields,
    resolve_field,
)
from app.services.liasse_extraction import parse_pcgm_native_liasse
from app.services.page_preprocessor import build_regions, crop_region, preprocess_page_image
from app.services.result_builder import build_extraction_result

FIXTURE = Path(__file__).parent / "fixtures" / "20250210.000037.04"


def _load_pages() -> list[dict]:
    return json.loads((FIXTURE / "expected_observations.json").read_text(encoding="utf-8"))


def _load_expected() -> dict[str, Decimal]:
    raw = json.loads((FIXTURE / "expected_financial_fields.json").read_text(encoding="utf-8"))
    return {k: Decimal(v) for k, v in raw.items()}


def _pipeline():
    pages = _load_pages()
    observations = []
    for page in pages:
        observations.extend(observations_from_page_payload(page["page"], page))
    resolved = resolve_all_fields(observations)
    resolved = apply_derived_fields(resolved)
    warnings, _, accounting_checks = run_accounting_checks(resolved)
    meta = extract_document_metadata(observations, pages)
    result = build_extraction_result(
        resolved=resolved,
        metadata=meta,
        sections_detected={"BILAN_ACTIF": True, "BILAN_PASSIF": True, "CPC": True},
        pages_total=39,
        pages_analyzed=5,
        elapsed_ms=1,
        filename="fixture.pdf",
        extra_warnings=warnings,
        accounting_checks=accounting_checks,
    )
    return resolved, result, meta


def test_parse_french_financial_amounts():
    assert parse_amount("14 757 502,81") == Decimal("14757502.81")
    assert parse_amount("594 733,04") == Decimal("594733.04")
    assert parse_amount("(27 264,00)") == Decimal("-27264.00")
    assert parse_amount("594 733,04-") == Decimal("-594733.04")
    assert parse_amount("20,49") == Decimal("20.49")
    assert parse_amount("") is None
    assert parse_amount("-") is None


def test_actifs_immobilises_uses_net_n_not_brut():
    resolved, _, _ = _pipeline()
    r = resolved["ACTIFS_IMMOBILISES"]
    assert r.selected_value == Decimal("10472949.06")
    assert r.column in ("net_n", "Net exercice N") or "net" in (r.column or "")


def test_total_bilan_uses_net_total():
    resolved, _, _ = _pipeline()
    assert resolved["TOTAL_BILAN"].selected_value == Decimal("27710609.88")


def test_resultat_net_uses_total_exercice_column():
    resolved, _, _ = _pipeline()
    assert resolved["RESULTAT_NET"].selected_value == Decimal("594733.04")
    # Ne doit pas prendre 621997.04
    assert resolved["RESULTAT_NET"].selected_value != Decimal("621997.04")


def test_interest_expenses_not_confused_with_other_operating_expenses():
    resolved, _, _ = _pipeline()
    assert resolved["CHARGES_INTERETS"].selected_value == Decimal("141952.39")
    assert resolved["CHARGES_INTERETS"].selected_value != Decimal("20.49")


def test_tresorerie_passif_is_total_of_cash_liabilities():
    resolved, _, _ = _pipeline()
    assert resolved["TRESORERIE_PASSIF"].selected_value == Decimal("6363775.44")
    assert resolved["TRESORERIE_PASSIF"].selected_value != Decimal("14757502.81")


def test_extracts_clients_suppliers_associates_cash_and_mlt_debt():
    resolved, _, _ = _pipeline()
    assert resolved["CREANCES_CLIENTS"].selected_value == Decimal("15263856.64")
    assert resolved["DETTES_FOURNISSEURS"].selected_value == Decimal("2473394.63")
    assert resolved["COMPTE_COURANT_ASSOCIES"].selected_value == Decimal("8251883.28")
    assert resolved["CAISSE"].selected_value == Decimal("20516.39")
    assert resolved["DETTES_BANCAIRES_MLT"].selected_value == Decimal("478812.80")


def test_present_empty_export_sales_line_becomes_zero():
    resolved, _, _ = _pipeline()
    assert resolved["CA_EXPORT"].selected_value == Decimal("0")
    assert resolved["CA_EXPORT"].detection_status == "detected_zero"


def test_other_charges_are_computed_from_configured_components():
    resolved, _, _ = _pipeline()
    assert resolved["AUTRES_CHARGES"].selected_value == Decimal("5251030.43")


def test_caf_is_detected_and_validated_against_components():
    resolved, _, _ = _pipeline()
    assert resolved["CAF"].selected_value == Decimal("330958.05")
    assert resolved["CAF"].validation_status in {"consistent", "divergent"}


def test_dettes_bancaires_ct_equals_tresorerie_components():
    resolved, _, _ = _pipeline()
    assert resolved["DETTES_BANCAIRES_CT"].selected_value == Decimal("6363775.44")


def test_balance_sheet_derived_fields():
    resolved, _, _ = _pipeline()
    assert resolved["DETTES_FINANCIERES"].selected_value == Decimal("6842588.24")
    assert resolved["TRESORERIE_NETTE"].selected_value == Decimal("-5826013.05")


def test_partial_aggregate_is_not_scoring_eligible():
    resolved = {
        "DETTES_BANCAIRES_MLT": resolve_field("DETTES_BANCAIRES_MLT", []),
        "DETTES_BANCAIRES_CT": resolve_field("DETTES_BANCAIRES_CT", []),
    }
    resolved["DETTES_BANCAIRES_MLT"].selected_value = Decimal("10")
    resolved["DETTES_BANCAIRES_MLT"].detection_status = "detected"
    derived = apply_derived_fields(resolved)
    assert derived["DETTES_FINANCIERES"].selected_value is None
    assert derived["DETTES_FINANCIERES"].calculation_status == "partial"
    assert derived["DETTES_FINANCIERES"].eligible_for_scoring is False


def test_caf_fallback_is_estimate_only():
    resolved = {
        "RESULTAT_NET": resolve_field("RESULTAT_NET", []),
        "AMORTISSEMENTS": resolve_field("AMORTISSEMENTS", []),
    }
    resolved["RESULTAT_NET"].selected_value = Decimal("100")
    resolved["RESULTAT_NET"].detection_status = "detected"
    resolved["AMORTISSEMENTS"].selected_value = Decimal("20")
    resolved["AMORTISSEMENTS"].detection_status = "detected"
    derived = apply_derived_fields(resolved)
    assert derived["CAF"].selected_value is None
    assert derived["CAF"].calculated_value == Decimal("120")
    assert derived["CAF"].eligible_for_scoring is False


def test_full_financial_extraction_fixture():
    expected = _load_expected()
    resolved, result, meta = _pipeline()
    assert meta.reference == "SIS2024D09078809"
    assert meta.entreprise == "STE INFRAROUTE SARL"
    assert result.reference == "SIS2024D09078809"
    assert result.entreprise == "STE INFRAROUTE SARL"

    mapping = {
        "actifs_immobilises": "ACTIFS_IMMOBILISES",
        "total_bilan": "TOTAL_BILAN",
        "chiffre_affaires": "CHIFFRE_AFFAIRES",
        "ca_export": "CA_EXPORT",
        "resultat_net": "RESULTAT_NET",
        "frais_financiers": "CHARGES_INTERETS",
        "caf": "CAF",
        "clients": "CREANCES_CLIENTS",
        "fournisseurs": "DETTES_FOURNISSEURS",
        "tresorerie_passif": "TRESORERIE_PASSIF",
        "dettes_bancaires_ct": "DETTES_BANCAIRES_CT",
        "autres_charges": "AUTRES_CHARGES",
        "amortissements": "AMORTISSEMENTS",
        "dettes_financieres": "DETTES_FINANCIERES",
        "tresorerie_nette": "TRESORERIE_NETTE",
    }
    for key, code in mapping.items():
        assert resolved[code].selected_value == expected[key], (
            f"{code}: got {resolved[code].selected_value}, expected {expected[key]}"
        )


def test_no_hardcoded_confidence_080_on_resolved_fields():
    resolved, result, _ = _pipeline()
    # Confiance variable selon score — pas toutes à 0.8
    confidences = {el.confidence for el in result.elements if el.value is not None}
    assert confidences
    assert not confidences.issubset({0.8})


def test_legacy_flat_elements_still_parse():
    payload = {
        "page_type": "BILAN_ACTIF",
        "elements": {"ACTIFS_IMMOBILISES": 10472949.06, "TOTAL_BILAN": 27710609.88},
        "empty_fields": [],
    }
    obs = observations_from_page_payload(1, payload)
    resolved = resolve_all_fields(obs)
    assert resolved["ACTIFS_IMMOBILISES"].selected_value == Decimal("10472949.06")


def test_native_pipeline_goes_through_resolver_and_derivations():
    text = """
    Bilan (actif)
    TOTAL II 1 000,00
    TOTAL GENERAL 5 000,00
    Clients et comptes rattachés 300,00
    Trésorerie-Actif 120,00
    Bilan (passif)
    Total des capitaux propres 800,00
    Dettes de financement 200,00
    Crédits de trésorerie 50,00
    Passif circulant 400,00
    Fournisseurs et comptes rattachés 250,00
    Compte de produits et charges
    Chiffres d'affaires 2 000,00
    Résultat net de l'exercice 100,00
    Dotations d'exploitation 20,00
    """
    result = parse_pcgm_native_liasse(text, "native.pdf", 1)
    assert result.document_kind == "LIASSE_NATIVE"
    assert result.scoring_input.chiffre_affaires == 2000.0
    assert result.scoring_input.tresorerie_nette == 70.0
    assert result.scoring_input.caf is None
    assert result.scoring_input.dettes_financieres == 250.0
    assert result.field_provenance["CAF"]["detection_status"] == "estimated"


def test_classify_page_text_preserves_esg():
    assert (
        classify_page_text("Etat des soldes de gestion et capacité d'autofinancement")
        == "ESG"
    )


@pytest.mark.parametrize("orientation", [90, 180, 270])
def test_page_preprocessor_supports_rotations(orientation):
    image = Image.new("RGB", (200, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 20, 170, 100), fill="black")
    buf = Path(FIXTURE / "tmp.jpg")
    image.save(buf, format="JPEG")
    data = buf.read_bytes()
    buf.unlink()
    preprocessed = preprocess_page_image(data, orientation=orientation)
    assert preprocessed.width > 0
    assert preprocessed.height > 0
    assert preprocessed.orientation == orientation


def test_page_preprocessor_builds_regions_and_crops():
    image = Image.new("RGB", (240, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 30, 220, 370), fill="black")
    tmp = Path(FIXTURE / "tmp2.jpg")
    image.save(tmp, format="JPEG")
    data = tmp.read_bytes()
    tmp.unlink()
    preprocessed = preprocess_page_image(data)
    regions = build_regions(preprocessed.width, preprocessed.height)
    assert [region.region_id for region in regions] == ["full", "top", "bottom"]
    top_crop = crop_region(preprocessed.image_bytes, regions[1])
    bottom_crop = crop_region(preprocessed.image_bytes, regions[2])
    assert top_crop != bottom_crop
