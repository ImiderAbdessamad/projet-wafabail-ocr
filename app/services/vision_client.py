"""Client bas niveau — appels vision à Ollama (image → JSON structuré).

Logique partagée entre l'extraction CIN et l'extraction ICE (lorsqu'un
document ICE est une image ou une page de PDF scanné) : on envoie une image
en base64 au modèle GLM vision et on récupère un objet JSON structuré.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from typing import Any

import httpx

from app.config import (
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    OLLAMA_URL,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class VisionExtractionError(RuntimeError):
    """Levée quand Ollama / le modèle GLM vision ne répond pas correctement."""


def extract_json_block(raw: str) -> dict[str, Any]:
    """Isole et parse le bloc JSON renvoyé par le modèle (tolérant aux textes parasites)."""
    raw = raw.strip()
    if not raw.startswith("{"):
        match = _JSON_BLOCK_RE.search(raw)
        raw = match.group(0) if match else "{}"

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Réponse du modèle non-JSON : %r", raw[:300])
        raise VisionExtractionError(
            "Le modèle a renvoyé une réponse invalide (JSON illisible)."
        ) from exc


async def vision_chat_json(
    image_bytes: bytes,
    system_prompt: str,
    user_message: str = "Analyse cette image.",
    model: str | None = None,
    *,
    timeout_seconds: float | None = None,
    num_predict: int | None = None,
) -> tuple[dict[str, Any], float]:
    """Envoie une image à Ollama (modèle GLM vision) et retourne le JSON structuré.

    Returns:
        (dict_parsé, temps_écoulé_en_ms)
    """
    b64_image = base64.b64encode(image_bytes).decode("ascii")

    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message, "images": [b64_image]},
        ],
        "format": "json",
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": 0},
    }
    if num_predict is not None:
        payload["options"]["num_predict"] = num_predict

    timeout = timeout_seconds or OLLAMA_TIMEOUT_SECONDS or REQUEST_TIMEOUT_SECONDS
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            if response.status_code in {500, 502, 503, 504}:
                raise VisionExtractionError(
                    f"Ollama HTTP {response.status_code} : {response.text[:180]}"
                )
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise VisionExtractionError(
            f"Impossible de contacter Ollama sur {OLLAMA_URL}. "
            "Vérifiez qu'Ollama est démarré (`ollama serve`)."
        ) from exc
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:300]
        used_model = model or OLLAMA_MODEL
        if exc.response.status_code == 404:
            raise VisionExtractionError(
                f"Modèle '{used_model}' introuvable sur Ollama. "
                f"Lancez `ollama pull {used_model}` puis réessayez."
            ) from exc
        raise VisionExtractionError(
            f"Ollama a répondu avec une erreur ({exc.response.status_code}) : {detail}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise VisionExtractionError(
            "Le modèle n'a pas répondu à temps (timeout). Le modèle est peut-être "
            "encore en cours de chargement — réessayez dans quelques secondes."
        ) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000

    body = response.json()
    raw_content = body.get("message", {}).get("content", "")
    if not raw_content:
        raise VisionExtractionError("Réponse vide du modèle.")

    return extract_json_block(raw_content), elapsed_ms
