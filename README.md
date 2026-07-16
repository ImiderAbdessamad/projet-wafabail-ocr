# Wafabail OCR API

API FastAPI unifiée pour l'extraction de documents marocains et le scoring
crédit-bail, via des modèles **GLM** servis par **Ollama** :

| Outil | Document | Endpoint | UI |
|---|---|---|---|
| **CIN** | Carte d'identité (recto ± verso) | `POST /api/cin/extract` | `/` |
| **ICE** | Certificat ICE | `POST /api/v1/extract-ice` | `/ice` |
| **Liasse** | Liasse fiscale PCGM (PDF/ZIP) | `POST /api/v1/extraction/liasse` | `/liasse` |
| **Scoring** | Ratios + 3 axes | `POST /api/v1/scoring/evaluate` | `/liasse` |
| **Pipeline** | Extraction + scoring | `POST /api/v1/extraction/liasse/score` | `/liasse` |
| **Export** | Résultats calculés JSON / Excel | `POST /api/v1/export/json` · `/excel` | boutons UI |

## Architecture

```
projet-wafabail-ocr/
├── main.py                     # Point d'entrée FastAPI
├── requirements.txt
├── .env.example
├── app/
│   ├── config.py               # Env unifié (CIN + ICE + liasse)
│   ├── schemas/                # Pydantic (documents, liasse, scoring)
│   ├── routers/                # cin, ice, extraction, scoring, export
│   ├── engines/                # ratios + scoring crédit-bail
│   └── services/               # OCR vision, extracteurs, export Excel/JSON
└── static/                     # UI CIN / ICE / Liasse
```

## Installation

```bash
cd projet-wafabail-ocr
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Configurer `.env` (réutilise `OLLAMA_URL` / `OLLAMA_MODEL` existants).

## Lancer

```bash
uvicorn main:app --reload --port 8012
```

- CIN : http://localhost:8012/
- ICE : http://localhost:8012/ice
- Liasse + scoring + export : http://localhost:8012/liasse
- Swagger : http://localhost:8012/docs
