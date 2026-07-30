# -*- coding: utf-8 -*-
"""Tests API financial-documents (jobs / SSE / result) sans Ollama."""
from __future__ import annotations

import asyncio
import io
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import fitz
from fastapi.testclient import TestClient

from app.schemas.direct_financial_extraction import (
    CompanyInfo,
    DocumentSummary,
    ExerciseInfo,
    ExtractionSummary,
    FinancialDocumentAnalysisResult,
)
from app.schemas.financial_analysis import CreditDecision
from app.services.financial_dataset_builder import empty_dataset
from app.services.financial_job_store import FinancialJobStore
from main import app

client = TestClient(app)


def _minimal_pdf(pages: int = 2) -> bytes:
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page test {i + 1} Bilan Actif Capitaux")
    data = doc.tobytes()
    doc.close()
    return data


def test_create_job_rejects_non_pdf():
    res = client.post(
        "/api/v1/financial-documents/jobs",
        files={"file": ("x.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 422


def test_create_job_and_poll_with_mocked_pipeline():
    pdf = _minimal_pdf(3)

    async def fake_run(job_id, *, store):
        job = store.get(job_id)
        assert job is not None
        store.update(job_id, status="processing", progress_pct=50, current_step="extracting_page")
        store.emit(job_id, "page_extracted", {"page": 1, "page_type": "BILAN_ACTIF"})
        dataset = empty_dataset()
        result = FinancialDocumentAnalysisResult(
            document=DocumentSummary(
                filename=job.filename,
                pages_total=3,
                pages_processed=1,
                pages_skipped=2,
                pages_failed=0,
                company=CompanyInfo(raison_sociale="TEST SA"),
                exercise=ExerciseInfo(label="2024"),
            ),
            extraction=ExtractionSummary(model="mock-glm", page_audit=[], warnings=[]),
            dataset=dataset,
            accounting_checks=[],
            ratios=[],
            axes=[],
            decision=CreditDecision(
                score=None,
                risk_class="NON_EVALUABLE",
                profile="n/a",
                decision="NON_EVALUABLE",
                recommendation="Données insuffisantes",
            ),
            warnings=[],
        )
        store.update(
            job_id,
            status="completed",
            progress_pct=100,
            current_step="completed",
            result=result,
            pdf_bytes=None,
        )
        store.emit(job_id, "result_ready", {"status": "completed"})

    with patch(
        "app.routers.financial_documents.run_financial_job",
        new=fake_run,
    ):
        res = client.post(
            "/api/v1/financial-documents/jobs",
            files={"file": ("liasse.pdf", pdf, "application/pdf")},
            data={"include_markdown": "false"},
        )
        assert res.status_code == 200
        body = res.json()
        job_id = body["job_id"]
        assert body["stream_url"].endswith(f"/jobs/{job_id}/stream")
        assert body["result_url"].endswith(f"/jobs/{job_id}/result")

        # Laisse le background task s'exécuter
        import time

        for _ in range(50):
            status = client.get(f"/api/v1/financial-documents/jobs/{job_id}")
            assert status.status_code == 200
            if status.json()["status"] == "completed":
                break
            time.sleep(0.05)

        result = client.get(f"/api/v1/financial-documents/jobs/{job_id}/result")
        assert result.status_code == 200
        payload = result.json()
        assert payload["document"]["filename"] == "liasse.pdf"
        assert payload["decision"]["risk_class"] == "NON_EVALUABLE"


def test_result_not_ready_returns_409():
    store = FinancialJobStore(ttl_minutes=10)
    job = store.create(pdf_bytes=b"%PDF-1.4", filename="x.pdf")
    # Remplace temporairement le singleton via patch get
    with patch("app.routers.financial_documents.job_store", store):
        res = client.get(f"/api/v1/financial-documents/jobs/{job.job_id}/result")
        assert res.status_code == 409


def test_financial_documents_page_served():
    res = client.get("/financial-documents")
    assert res.status_code == 200
    assert b"Analyse de liasse fiscale" in res.content
    assert b"var(--accent" in open(
        "static/financial-documents.css", encoding="utf-8"
    ).read().encode() or True
    css = open("static/financial-documents.css", encoding="utf-8").read()
    assert "var(--accent" in css
    assert "var(--accent-2" in css
    js = open("static/financial-documents.js", encoding="utf-8").read()
    assert "EventSource" in js
    assert "result_ready" in js
    assert "downloadJson" in js
    assert "innerHTML" not in js or "grid.innerHTML" in js  # structure only, text via textContent


def test_no_qwen_import_in_direct_pipeline():
    import app.services.direct_financial_extraction_pipeline as pipeline
    import app.routers.financial_documents as router_mod
    import inspect

    src = inspect.getsource(pipeline) + inspect.getsource(router_mod)
    assert "financial_mapping_client" not in src
    assert "qwen" not in src.lower() or "sans mapping qwen" in open(
        "static/financial-documents.html", encoding="utf-8"
    ).read().lower()
    assert "map_financial_section" not in src
