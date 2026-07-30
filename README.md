# Wafabail OCR API

API FastAPI unifiée pour l'extraction de documents marocains et le scoring
crédit-bail, via des modèles **GLM** servis par **Ollama** :

| Outil | Document | Endpoint | UI |
|---|---|---|---|
| **CIN** | Carte d'identité (recto ± verso) | `POST /api/cin/extract` | `/` |
| **ICE** | Certificat ICE | `POST /api/v1/extract-ice` | `/ice` |
| **Liasse** | Liasse fiscale PCGM (PDF/ZIP) | `POST /api/v1/extraction/liasse` | `/liasse` |
| **Scoring** | Ratios + 3 axes | `POST /api/v1/scoring/evaluate` | `/liasse` |
| **PDF / Bilans** | Markdown + mapping Qwen (legacy) | `POST /api/v1/extraction/pdf/analyze` | `/pdf` |
| **Liasse GLM direct** | Image → JSON financier (sans Qwen) | `POST /api/v1/financial-documents/jobs` | `/financial-documents` |
| **Export** | Résultats calculés JSON / Excel | `POST /api/v1/export/json` · `/excel` | boutons UI |

## Architecture

```
projet-wafabail-ocr/
├── main.py
├── app/
│   ├── config.py
│   ├── schemas/
│   │   └── direct_financial_extraction.py
│   ├── routers/
│   │   └── financial_documents.py
│   └── services/
│       ├── financial_orientation_detector.py
│       ├── financial_page_classifier.py
│       ├── direct_glm_financial_client.py
│       ├── direct_financial_resolver.py
│       ├── direct_financial_extraction_pipeline.py
│       └── financial_job_store.py
└── static/
    ├── financial-documents.html
    ├── financial-documents.css
    └── financial-documents.js
```

### Pipeline GLM direct (recommandé pour les liasses)

```
PDF → rendu PNG → orientation → classification page
    → GLM Vision (schéma JSON par type de page)
    → candidats → resolver Python Decimal
    → contrôles → ratios → scoring
```

Interdit dans cette API : Qwen, parser Markdown → valeurs financières.
Le Markdown éventuel est uniquement audit/affichage (`include_markdown`).

## Installation

```bash
cd projet-wafabail-ocr
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Configurer `.env` (réutilise `OLLAMA_URL`). Modèle vision :

```
DIRECT_FINANCIAL_MODEL=hf.co/unsloth/GLM-4.6V-Flash-GGUF:Q4_K_M
```

## Lancer

```bash
uvicorn main:app --reload --port 8012
```

- CIN : http://localhost:8012/
- ICE : http://localhost:8012/ice
- Liasse + scoring : http://localhost:8012/liasse
- PDF / Bilans (legacy Qwen) : http://localhost:8012/pdf
- **Liasse GLM direct** : http://localhost:8012/financial-documents
- Swagger : http://localhost:8012/docs

## API jobs (SSE)

Création :

```bash
curl -X POST \
  http://127.0.0.1:8012/api/v1/financial-documents/jobs \
  -F "file=@document.pdf" \
  -F "include_markdown=false"
```

Réponse :

```json
{
  "job_id": "...",
  "status": "queued",
  "stream_url": "/api/v1/financial-documents/jobs/{id}/stream",
  "result_url": "/api/v1/financial-documents/jobs/{id}/result"
}
```

Progression : `GET .../jobs/{id}` ou SSE `GET .../jobs/{id}/stream`

Événements SSE : `job_started`, `pdf_validated`, `pages_rendered`,
`page_classified`, `page_extracted`, `page_skipped`, `page_failed`,
`resolving_fields`, `running_controls`, `calculating_ratios`,
`result_ready`, `job_failed`.

Résultat : `GET .../jobs/{id}/result`

Statuts de champs : `confirmed`, `derived`, `ambiguous`, `conflicting`,
`missing`, `invalid`. Une valeur absente reste `null` / sans candidat.
Une cellule vide ne devient jamais zéro.

## Configuration clé

| Variable | Défaut |
|---|---|
| `OLLAMA_URL` | `http://localhost:11434` |
| `DIRECT_FINANCIAL_MODEL` | GLM Flash GGUF |
| `DIRECT_FINANCIAL_RENDER_DPI` | 220 |
| `DIRECT_FINANCIAL_MAX_PAGES` | 60 |
| `DIRECT_FINANCIAL_JOB_TTL_MINUTES` | 60 |

## Tests

Sans Ollama :

```bash
.venv\Scripts\python.exe -m pytest tests/test_direct_glm_financial_client.py tests/test_direct_financial_resolver.py tests/test_financial_page_classifier.py tests/test_financial_orientation_detector.py tests/test_financial_documents_api.py tests/test_financial_document_fixtures.py -q
```

## Limites connues

- Classification GLM de secours non activée par défaut (lexicale + native).
- Job store mémoire local (TTL) — pas encore Redis.
- Appels GLM séquentiels uniquement (un job à la fois par instance).
- L'ancien pipeline `/pdf` (Qwen) reste disponible mais séparé.
