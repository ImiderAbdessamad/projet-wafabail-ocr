#!/usr/bin/env python3
"""CLI batch — même pipeline que l'API `/api/v1/extraction/liasse`.

Le modèle GLM Flash reste sur le serveur Ollama distant (`.env` → OLLAMA_URL).
Ce script n'embarque plus de client OCR séparé : il appelle
`extract_liasse_document()` (identique à l'API scoring).

Usage :
  .\\.venv\\Scripts\\python.exe scripts\\extract_liasse_pdfs.py
  .\\.venv\\Scripts\\python.exe scripts\\extract_liasse_pdfs.py --pdf ..\\20250210.000037.04.pdf
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

logger = logging.getLogger("extract_liasse_pdfs")
DEFAULT_PDF_DIR = ROOT.parent
DEFAULT_OUTPUT = ROOT / "output" / "extractions"


def discover_pdfs(pdf_dir: Path, selected: list[str] | None) -> list[Path]:
    if selected:
        paths = []
        for item in selected:
            path = Path(item)
            if not path.is_absolute():
                path = (pdf_dir / item).resolve() if not path.exists() else path.resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            paths.append(path)
        return paths
    names = [
        "20251103.000016.01.pdf",
        "20251210.000011.01.pdf",
        "20251210.000011.02.pdf",
        "20251216.000010.01.pdf",
        "20251224.000028.01.pdf",
        "20251224.000029.01.pdf",
        "20250210.000037.04.pdf",
        "ADEISINVEST-BILAN-2025.pdf",
        "FDINVEST -Bilan 2025.pdf",
    ]
    return [pdf_dir / name for name in names if (pdf_dir / name).exists()]


async def extract_one(pdf_path: Path, out_dir: Path) -> dict:
    from app.services.liasse_extraction import (
        extract_liasse_document,
        get_liasse_extraction_service,
    )

    service = get_liasse_extraction_service()
    content = pdf_path.read_bytes()
    result = await extract_liasse_document(content, pdf_path.name, service)
    payload = result.model_dump(mode="json")

    dest = out_dir / pdf_path.stem
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "result_api.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        "source_filename": pdf_path.name,
        "mode": "api_pipeline",
        "document_kind": result.document_kind,
        "completeness_pct": result.completeness_pct,
        "entreprise": result.entreprise,
        "reference": result.reference,
        "pages_total": result.pages_total,
        "pages_analyzed": result.pages_analyzed,
        "scoring_input": result.scoring_input.model_dump(),
        "warnings": result.warnings[:8],
        "output": str(dest / "result_api.json"),
        "ollama_url": os.getenv("OLLAMA_URL"),
        "model": os.getenv("OLLAMA_VISION_MODEL") or os.getenv("OLLAMA_MODEL"),
    }
    (dest / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


async def async_main(args: argparse.Namespace) -> int:
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = discover_pdfs(Path(args.pdf_dir).resolve(), args.pdf)
    if not pdfs:
        logger.error("Aucun PDF à traiter.")
        return 1

    logger.info(
        "Pipeline API — OLLAMA_URL=%s model=%s pdfs=%d",
        os.getenv("OLLAMA_URL"),
        os.getenv("OLLAMA_VISION_MODEL") or os.getenv("OLLAMA_MODEL"),
        len(pdfs),
    )
    summaries = []
    for pdf in pdfs:
        logger.info("=== %s ===", pdf.name)
        try:
            summary = await extract_one(pdf, out_dir)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Échec %s", pdf.name)
            summary = {"source_filename": pdf.name, "error": str(exc)}
        summaries.append(summary)
        logger.info(
            "RESUME %s | kind=%s | completeness=%s | err=%s",
            summary.get("source_filename"),
            summary.get("document_kind"),
            summary.get("completeness_pct"),
            (summary.get("error") or "")[:120],
        )

    batch = out_dir / f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    batch.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Batch terminé → %s", batch)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extraction liasse via le même pipeline que l'API scoring"
    )
    parser.add_argument("--pdf-dir", default=str(DEFAULT_PDF_DIR))
    parser.add_argument("--pdf", action="append")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    raise SystemExit(asyncio.run(async_main(args)))


if __name__ == "__main__":
    main()
