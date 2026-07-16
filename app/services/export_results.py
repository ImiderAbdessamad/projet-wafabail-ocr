"""Export des résultats d'extraction / scoring (JSON téléchargeable + Excel)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_json_bytes(payload: dict[str, Any]) -> tuple[bytes, str]:
    """Sérialise le résultat calculé en JSON téléchargeable."""
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return body, f"liasse_resultats_{_ts()}.json"


def _autofit(ws, max_width: int = 48) -> None:
    for col_idx, column in enumerate(ws.columns, start=1):
        length = 0
        for cell in column:
            value = "" if cell.value is None else str(cell.value)
            length = max(length, min(len(value), max_width))
        ws.column_dimensions[get_column_letter(col_idx)].width = max(12, length + 2)


def _header_row(ws, headers: list[str], fill: str = "F58220") -> None:
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor=fill)
    for col, title in enumerate(headers, start=1):
        cell = ws.cell(1, col, title)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="left", vertical="center")


def build_excel_bytes(payload: dict[str, Any]) -> tuple[bytes, str]:
    """Construit un classeur Excel des résultats calculés (extraction + scoring)."""
    wb = Workbook()

    extraction = payload.get("extraction") or payload
    scoring = payload.get("scoring")

    # --- Synthèse ---------------------------------------------------------
    ws = wb.active
    ws.title = "Synthese"
    _header_row(ws, ["Champ", "Valeur"])
    rows = [
        ("document_kind", extraction.get("document_kind")),
        ("source_filename", extraction.get("source_filename")),
        ("pages_processed", extraction.get("pages_processed")),
        ("completeness_pct", extraction.get("completeness_pct")),
        ("processing_time_ms", extraction.get("processing_time_ms")),
        ("ocr_method", extraction.get("ocr_method") or extraction.get("method")),
        ("summary", extraction.get("summary")),
    ]
    if scoring and scoring.get("decision"):
        decision = scoring["decision"]
        rows.extend(
            [
                ("score", decision.get("score")),
                ("classe", decision.get("classe")),
                ("decision", decision.get("decision")),
                ("recommandation", decision.get("recommandation")),
                ("blocking_status", decision.get("blocking_status")),
            ]
        )
    if payload.get("scoring_warning"):
        rows.append(("scoring_warning", payload["scoring_warning"]))
    if payload.get("scoring_skipped_reason"):
        rows.append(("scoring_skipped_reason", payload["scoring_skipped_reason"]))

    for i, (key, value) in enumerate(rows, start=2):
        ws.cell(i, 1, key)
        ws.cell(i, 2, value if value is None or isinstance(value, (int, float, str)) else str(value))
    _autofit(ws)

    # --- Éléments PCGM ----------------------------------------------------
    ws_el = wb.create_sheet("Elements_PCGM")
    _header_row(ws_el, ["Code", "Label", "Section", "Valeur", "Statut"])
    elements = extraction.get("elements") or []
    for i, el in enumerate(elements, start=2):
        ws_el.cell(i, 1, el.get("code"))
        ws_el.cell(i, 2, el.get("label"))
        ws_el.cell(i, 3, el.get("section") or el.get("source"))
        ws_el.cell(i, 4, el.get("value"))
        ws_el.cell(i, 5, el.get("detection_status") or el.get("status"))
    _autofit(ws_el)

    # --- Métriques scoring input ------------------------------------------
    ws_in = wb.create_sheet("Donnees_financieres")
    _header_row(ws_in, ["Champ", "Valeur"])
    scoring_input = extraction.get("scoring_input") or {}
    for i, (key, value) in enumerate(sorted(scoring_input.items()), start=2):
        ws_in.cell(i, 1, key)
        ws_in.cell(i, 2, value)
    _autofit(ws_in)

    # --- Ratios & axes ----------------------------------------------------
    if scoring:
        ws_r = wb.create_sheet("Ratios")
        _header_row(ws_r, ["Clé", "Label", "Valeur", "Unité", "Statut", "Seuil", "Formule", "Raison"])
        ratios = scoring.get("ratios") or {}
        for i, (key, detail) in enumerate(ratios.items(), start=2):
            ws_r.cell(i, 1, key)
            ws_r.cell(i, 2, detail.get("label"))
            ws_r.cell(i, 3, detail.get("value"))
            ws_r.cell(i, 4, detail.get("unit"))
            ws_r.cell(i, 5, detail.get("status"))
            ws_r.cell(i, 6, detail.get("threshold"))
            ws_r.cell(i, 7, detail.get("formula"))
            ws_r.cell(i, 8, detail.get("reason"))
        _autofit(ws_r)

        ws_a = wb.create_sheet("Axes")
        _header_row(ws_a, ["Axe", "Score", "Pondération", "Contribution"])
        for i, (name, key) in enumerate(
            [("Axe 1 — Financier", "axe1"), ("Axe 2 — Comportemental", "axe2"), ("Axe 3 — Sectoriel", "axe3")],
            start=2,
        ):
            axe = scoring.get(key) or {}
            ws_a.cell(i, 1, name)
            ws_a.cell(i, 2, axe.get("score"))
            ws_a.cell(i, 3, axe.get("ponderation"))
            ws_a.cell(i, 4, axe.get("contribution"))
        _autofit(ws_a)

        synthese = scoring.get("synthese") or {}
        ws_s = wb.create_sheet("Synthese_analyse")
        _header_row(ws_s, ["Type", "Texte"])
        row = 2
        for text in synthese.get("points_forts") or []:
            ws_s.cell(row, 1, "Point fort")
            ws_s.cell(row, 2, text)
            row += 1
        for text in synthese.get("points_vigilance") or []:
            ws_s.cell(row, 1, "Vigilance")
            ws_s.cell(row, 2, text)
            row += 1
        _autofit(ws_s)

    buffer = BytesIO()
    wb.save(buffer)
    return buffer.getvalue(), f"liasse_resultats_{_ts()}.xlsx"
