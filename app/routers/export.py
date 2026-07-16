"""API d'export des résultats calculés (JSON + Excel)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import Response

from app.services.export_results import build_excel_bytes, build_json_bytes

router = APIRouter(prefix="/export", tags=["Export"])


@router.post("/json")
async def export_json(payload: dict[str, Any]) -> Response:
    """Télécharge le résultat d'extraction/scoring au format JSON."""
    body, filename = build_json_bytes(payload)
    return Response(
        content=body,
        media_type="application/json; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/excel")
async def export_excel(payload: dict[str, Any]) -> Response:
    """Télécharge le résultat calculé (éléments, ratios, décision) en Excel."""
    body, filename = build_excel_bytes(payload)
    return Response(
        content=body,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
