import base64
import json

import httpx

data = open("test_pdf_page_0.jpg", "rb").read()
b64 = base64.b64encode(data).decode("ascii")

payload = {
    "model": "hf.co/unsloth/GLM-4.6V-Flash-GGUF:Q4_K_M",
    "messages": [
        {"role": "system", "content": 'Tu es un expert. Reponds en JSON valide: {"test": "valeur"}'},
        {"role": "user", "content": "Analyse cette image.", "images": [b64]},
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
print(json.dumps(r.json(), indent=2, ensure_ascii=False)[:4000])
