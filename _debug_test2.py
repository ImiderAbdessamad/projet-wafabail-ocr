import base64
import json

import httpx

from app.services.glm_extractor import _SYSTEM_PROMPT

data = open("test_pdf_page_0.jpg", "rb").read()
b64 = base64.b64encode(data).decode("ascii")

payload = {
    "model": "hf.co/unsloth/GLM-4.6V-Flash-GGUF:Q4_K_M",
    "messages": [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "Voici la carte d'identité nationale à analyser.", "images": [b64]},
    ],
    "format": "json",
    "stream": False,
    "options": {"temperature": 0},
}

r = httpx.post(
    "https://wafa-s7bndh-11434.svc-usw2.nicegpu.com/api/chat",
    json=payload,
    timeout=120,
)
print("status:", r.status_code)
body = r.json()
msg = body.get("message", {})
print("content:", repr(msg.get("content")))
print("thinking length:", len(msg.get("thinking") or ""))
print("thinking tail:", repr((msg.get("thinking") or "")[-500:]))
print("done_reason:", body.get("done_reason"))
print("prompt_eval_count:", body.get("prompt_eval_count"))
print("eval_count:", body.get("eval_count"))
