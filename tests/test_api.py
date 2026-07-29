# -*- coding: utf-8 -*-
"""Tests de l'API : calibration scoring (cas de référence du document
d'analyse) et extraction sur les rapports d'indicateurs réels si présents."""
import glob
import json
import os

import pytest
from fastapi.testclient import TestClient

from main import app
from app.routers.extraction import _scoring_block_reason
from app.schemas.liasse import LiasseExtractionResult, ScoringInput

client = TestClient(app)

# Répertoire contenant les PDFs d'exemple (rapports *_indicateurs.pdf)
SAMPLES_DIR = os.environ.get(
    "LIASSE_SAMPLES_DIR", os.path.join(os.path.dirname(__file__), "..", "..")
)

# Cas de référence du document d'analyse : exercice 2024,
# attendu 85 / 75 / 80 -> 83/100, classe A/B+.
REFERENCE_PAYLOAD = {
    "bam_cotation": 3,
    "financial_data": {
        "chiffre_affaires": 38500, "ca_n1": 34200, "fonds_propres": 6200,
        "total_bilan": 24800, "dettes_financieres": 11400, "resultat_net": 1480,
        "caf": 3200, "clients": 8341, "fournisseurs": 4000, "achats": 22500,
        "fdr": 3619, "tresorerie_nette": 535,
        "encours_leasing": 3200, "cmt": 8200, "nouveau_financement": 4800,
    },
    "behavioral_data": {
        "domiciliation_ca_pct": 96, "jours_debit": 41,
        "utilisation_decouvert_pct": 38, "ecart_flux_ca_pct": -4.2,
        "engagements_honores": True,
    },
}


def test_health():
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_scoring_reference_case():
    r = client.post("/api/v1/scoring/evaluate", json=REFERENCE_PAYLOAD)
    assert r.status_code == 200
    d = r.json()
    assert d["axe1"]["score"] == 85.0
    assert d["axe2"]["score"] == 75.0
    assert d["axe3"]["score"] == 80.0
    assert d["decision"]["score"] == 83.0
    assert d["decision"]["classe"] == "A/B+"
    assert d["decision"]["recommandation"] == "Accord — conditions standards"
    assert d["decision"]["blocking_status"] is None
    # Les 2 ratios « à surveiller » du rapport type
    assert sorted(d["axe1"]["details"]["ratios_a_surveiller"]) == [
        "delais_clients", "rentabilite_commerciale",
    ]
    assert d["synthese"]["points_forts"]
    assert d["synthese"]["points_vigilance"]


def test_scoring_blocking_bam():
    payload = dict(REFERENCE_PAYLOAD, bam_cotation=7)
    d = client.post("/api/v1/scoring/evaluate", json=payload).json()
    assert d["decision"]["blocking_status"] == "NO_GO"
    assert d["decision"]["score"] == 0.0


def test_scoring_incidents_manual_review():
    payload = json.loads(json.dumps(REFERENCE_PAYLOAD))
    payload["behavioral_data"]["incidents_paiement"] = 2
    d = client.post("/api/v1/scoring/evaluate", json=payload).json()
    assert d["decision"]["blocking_status"] == "MANUAL_REVIEW"


def test_scoring_without_behavior_data_is_not_treated_as_irreproachable():
    payload = json.loads(json.dumps(REFERENCE_PAYLOAD))
    payload["behavioral_data"] = {}
    d = client.post("/api/v1/scoring/evaluate", json=payload).json()
    assert d["axe2"]["details"]["status"] == "not_provided"
    assert d["eligibility"]["behavioral_coverage"] == 0.0
    assert "irréprochable" not in " ".join(d["synthese"]["points_forts"]).lower()


def test_scoring_empty_payload_does_not_return_a_plus():
    payload = {"financial_data": {}, "behavioral_data": {}}
    d = client.post("/api/v1/scoring/evaluate", json=payload).json()
    assert d["decision"]["classe"] == "Non évaluable"
    assert d["decision"]["blocking_status"] == "INSUFFICIENT_DATA"


def test_negative_caf_does_not_produce_good_repayment_capacity():
    payload = json.loads(json.dumps(REFERENCE_PAYLOAD))
    payload["financial_data"]["caf"] = -100
    d = client.post("/api/v1/scoring/evaluate", json=payload).json()
    assert d["ratios"]["capacite_remboursement"]["status"] == "Non conforme"
    assert "CAF négative" in d["ratios"]["capacite_remboursement"]["reason"]


def test_negative_funds_do_not_produce_good_debt_ratio():
    payload = json.loads(json.dumps(REFERENCE_PAYLOAD))
    payload["financial_data"]["fonds_propres"] = -100
    d = client.post("/api/v1/scoring/evaluate", json=payload).json()
    assert d["ratios"]["ratio_endettement"]["status"] == "Non conforme"
    assert "Fonds propres négatifs" in d["ratios"]["ratio_endettement"]["reason"]


def test_scoring_history_variations():
    payload = dict(
        REFERENCE_PAYLOAD,
        financial_history=[
            {"fiscal_year": 2022, "chiffre_affaires": 31800, "resultat_net": 1140},
            {"fiscal_year": 2023, "chiffre_affaires": 34200, "resultat_net": 1440},
            {"fiscal_year": 2024, "chiffre_affaires": 38500, "resultat_net": 1480},
        ],
    )
    d = client.post("/api/v1/scoring/evaluate", json=payload).json()
    var_ca = d["variations"]["chiffre_affaires"]
    assert round(var_ca["2024/2023"], 4) == round((38500 - 34200) / 34200, 4)


def test_extraction_rejects_non_pdf():
    r = client.post(
        "/api/v1/extraction/liasse",
        files={"file": ("x.pdf", b"not a pdf", "application/pdf")},
    )
    assert r.status_code == 415


def test_partial_extraction_never_produces_credit_score():
    """Une liasse avec seulement le bilan actif ne doit jamais devenir A+."""
    extraction = LiasseExtractionResult(
        document_kind="LIASSE_OCR",
        completeness_pct=36.8,
        sections_completeness={
            "BILAN_ACTIF": True,
            "BILAN_PASSIF": False,
            "CPC": False,
        },
        scoring_input=ScoringInput(
            total_bilan=68_282_077.93,
            resultat_net=4_879_668.56,
        ),
    )
    reason = _scoring_block_reason(extraction)
    assert reason is not None
    assert "Scoring non lancé" in reason
    assert "BILAN_PASSIF" in reason
    assert "chiffre d'affaires" in reason


def test_complete_extraction_with_ratio_inputs_can_be_scored():
    extraction = LiasseExtractionResult(
        document_kind="LIASSE_OCR",
        completeness_pct=75.0,
        sections_completeness={
            "BILAN_ACTIF": True,
            "BILAN_PASSIF": True,
            "CPC": True,
        },
        scoring_input=ScoringInput(
            chiffre_affaires=10_000_000,
            total_bilan=8_000_000,
            resultat_net=900_000,
            fonds_propres=3_000_000,
            dettes_financieres=1_500_000,
            caf=1_100_000,
            fdr=600_000,
        ),
    )
    assert _scoring_block_reason(extraction) is None


def test_forecast_extraction_is_blocked_before_scoring():
    extraction = LiasseExtractionResult(
        document_kind="LIASSE_NATIVE",
        document_type="forecast_financial_statements",
        period_type="forecast",
        eligible_for_automatic_scoring=False,
        scoring_mode="forecast_review",
        scoring_block_reasons=["Document prévisionnel : scoring automatique réel interdit."],
        completeness_pct=90.0,
        sections_completeness={
            "BILAN_ACTIF": True,
            "BILAN_PASSIF": True,
            "CPC": True,
        },
        scoring_input=ScoringInput(
            chiffre_affaires=10_000_000,
            total_bilan=8_000_000,
            resultat_net=900_000,
            fonds_propres=3_000_000,
            caf=1_100_000,
            fdr=600_000,
        ),
    )
    assert _scoring_block_reason(extraction) == (
        "Document prévisionnel : scoring automatique réel interdit."
    )


@pytest.mark.parametrize(
    "pdf_path", sorted(glob.glob(os.path.join(SAMPLES_DIR, "*_indicateurs.pdf")))
)
def test_extraction_indicateurs_reports(pdf_path):
    with open(pdf_path, "rb") as fh:
        r = client.post(
            "/api/v1/extraction/liasse",
            files={"file": (os.path.basename(pdf_path), fh.read(), "application/pdf")},
        )
    assert r.status_code == 200
    d = r.json()
    assert d["document_kind"] == "RAPPORT_INDICATEURS"
    assert d["reference"]
    assert d["completeness_pct"] > 50
    # Le CA et le résultat net (CPC) sont extraits sur tous les échantillons
    assert d["scoring_input"]["chiffre_affaires"] is not None
    assert d["scoring_input"]["resultat_net"] is not None
    # Une section manquante doit générer un avertissement, jamais des zéros silencieux
    missing = [k for k, ok in d["sections_completeness"].items() if not ok]
    if missing:
        assert d["warnings"]


def test_extract_and_score_with_complement():
    samples = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*_indicateurs.pdf")))
    if not samples:
        pytest.skip("Aucun rapport d'indicateurs disponible")
    complement = {
        "bam_cotation": 4,
        "financial_overrides": {"fonds_propres": 8_000_000, "caf": 2_000_000},
        "behavioral_data": {"jours_debit": 10, "domiciliation_ca_pct": 90},
    }
    with open(samples[0], "rb") as fh:
        r = client.post(
            "/api/v1/extraction/liasse/score",
            files={"file": ("ind.pdf", fh.read(), "application/pdf")},
            data={"complement": json.dumps(complement)},
        )
    assert r.status_code == 200
    d = r.json()
    assert d["scoring"] is not None
    assert d["scoring"]["decision"]["classe"]
    # L'override fonds_propres doit débloquer la rentabilité financière
    # (RN extrait du CPC + FP fourni par l'analyste)
    assert d["scoring"]["ratios"]["rentabilite_financiere"]["value"] is not None


def test_pdf_content_extraction_native():
    """PDF texte simple sans tableau : extraction native (layout conservé)."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Titre du document\nParagraphe ligne 1\nParagraphe ligne 2")
    pdf_bytes = doc.tobytes()
    doc.close()

    r = client.post(
        "/api/v1/extraction/pdf/content",
        files={"file": ("sample.pdf", pdf_bytes, "application/pdf")},
        data={"max_pages": "1"},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["pages_processed"] == 1
    assert d["pages_ok"] == 1
    assert d["pages"][0]["extraction_mode"] == "native"
    assert d["pages"][0]["content"]
    assert "\n" in d["pages"][0]["content"]
    assert d["pages"][0]["tables"] == []
    assert d["pages"][0]["raw_model_response"]["extraction_strategy"] == "native"
    assert any("Markdown" in w for w in d["warnings"])
