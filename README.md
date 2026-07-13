# Document Extractor API

API FastAPI qui extrait automatiquement des informations depuis des documents
administratifs marocains, via des modèles **GLM** servis par **Ollama** (local
ou distant) :

| Outil | Document | Champs extraits | Endpoint |
|---|---|---|---|
| **CIN** | Carte d'identité nationale — recto (+ verso optionnel), image ou PDF | `nom`, `prenom`, `cin`, `date_naissance`, `lieu_naissance`, `date_expiration`, `adresse` | `POST /api/cin/extract` |
| **ICE** | Certificat ICE (image ou PDF, natif ou scanné) | `ICE`, `Denomination`, `Identifiant_Fiscal`, `RC_Numero`, `RC_Ville`, `CNSS` | `POST /api/v1/extract-ice` |

Deux interfaces web minimalistes (upload + affichage des résultats) sont
incluses pour tester rapidement, sans rien installer côté frontend :
`/` (CIN) et `/ice` (ICE).

**Aucune dépendance système** (pas de Tesseract, pas de Poppler) : toute
lecture d'image passe directement par le modèle GLM **vision**, exactement
comme pour la CIN. Seul `pip install -r requirements.txt` est nécessaire.

## Architecture

```
backend/
├── main.py                    # Point d'entrée FastAPI
├── requirements.txt
├── .env.example
├── app/
│   ├── config.py              # Variables d'environnement centralisées
│   ├── schemas.py             # Modèles Pydantic (CinData, IceData, ...)
│   ├── routers/
│   │   ├── cin.py             # POST /api/cin/extract
│   │   └── ice.py             # POST /api/v1/extract-ice
│   └── services/
│       ├── image_utils.py     # Validation + normalisation image/PDF → image(s)
│       ├── vision_client.py   # Appel bas niveau Ollama vision (image → JSON)
│       ├── glm_extractor.py   # Extraction CIN (par face) + fusion recto/verso
│       ├── text_extractor.py  # PDF : texte natif (pdfplumber) + rendu image (PyMuPDF)
│       └── ice_extractor.py   # Extraction ICE (texte via `ollama`, image via vision_client)
└── static/                    # UI de test (HTML/CSS/JS vanilla)
    ├── index.html / app.js    # Outil CIN, servi sur "/"
    ├── ice.html   / ice.js    # Outil ICE, servi sur "/ice"
    └── style.css              # Styles partagés
```

### Comment la CIN est traitée (recto / verso / PDF)

```
Recto seul (image ou PDF)                    → 1 appel au modèle GLM vision
Recto + Verso (2 fichiers, image ou PDF)     → 2 appels en parallèle, puis fusion
Recto = PDF de 2 pages (recto+verso scanné)  → page 1 = recto, page 2 = verso (auto)
```

Chaque face est analysée indépendamment (le modèle renvoie `""` pour les
champs non visibles sur cette face), puis les résultats sont **fusionnés** :
pour chaque champ, la première valeur non vide trouvée est retenue (recto
prioritaire). En pratique, le recto fournit `nom`/`prenom`/`cin`/`date_naissance`/
`lieu_naissance`/`date_expiration`, et le verso fournit `adresse`.

### Comment l'ICE est traité selon le type de fichier

```
Image (JPEG/PNG)  ────────────────────────────► modèle GLM vision
PDF numérique      → texte natif (pdfplumber)  → modèle GLM texte
PDF scanné         → rendu image (PyMuPDF)     → modèle GLM vision
```

## 1. Prérequis — Ollama + modèle(s) GLM

1. Installer [Ollama](https://ollama.com/download) puis démarrer le serveur
   (`ollama serve`, souvent lancé automatiquement) — ou utiliser un serveur
   Ollama distant (ex: instance GPU cloud).
2. Télécharger un modèle GLM avec support **vision** (utilisé pour la CIN et
   pour les images/PDF scannés côté ICE) :

   ```bash
   ollama pull hf.co/unsloth/GLM-4.6V-Flash-GGUF:Q4_K_M
   ```

   D'autres variantes existent sur [ollama.com](https://ollama.com/search?q=glm)
   (`glm-4.6v:9b-flash-q4_K_M`, `haervwe/GLM-4.6V-Flash-9B`, …).

3. Ce même modèle GLM vision gère généralement aussi le **texte** (capacités
   `completion` + `vision`) : vous pouvez donc utiliser le même modèle pour
   `OLLAMA_MODEL` et `OLLAMA_TEXT_MODEL` (voir étape 3). Si vous préférez un
   modèle texte dédié plus léger, `ollama pull glm4` fonctionne aussi.

## 2. Installation du backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell)
pip install -r requirements.txt
```

## 3. Configuration

```bash
copy .env.example .env
```

Puis éditez `.env`, par exemple (serveur Ollama local) :

```env
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=hf.co/unsloth/GLM-4.6V-Flash-GGUF:Q4_K_M
OLLAMA_TEXT_MODEL=hf.co/unsloth/GLM-4.6V-Flash-GGUF:Q4_K_M
```

Ou avec un serveur Ollama distant :

```env
OLLAMA_URL=https://votre-serveur-ollama.example.com
OLLAMA_MODEL=hf.co/unsloth/GLM-4.6V-Flash-GGUF:Q4_K_M
OLLAMA_TEXT_MODEL=hf.co/unsloth/GLM-4.6V-Flash-GGUF:Q4_K_M
```

## 4. Lancer le serveur

```bash
uvicorn main:app --reload
```

- Interface CIN : http://localhost:8000
- Interface ICE : http://localhost:8000/ice
- Documentation interactive (Swagger) : http://localhost:8000/docs
- Health check : http://localhost:8000/health

Au démarrage, l'API précharge automatiquement le(s) modèle(s) GLM configurés
(appel silencieux en tâche de fond) pour éviter qu'une première requête
utilisateur n'essuie le délai de chargement à froid — particulièrement
sensible sur un serveur Ollama distant.

## 5. Utiliser l'API directement

### CIN

Recto seul (image ou PDF) :

```bash
curl -X POST http://localhost:8000/api/cin/extract \
  -F "recto=@/chemin/vers/cin_recto.jpg"
```

Recto + verso (recommandé, pour récupérer aussi l'adresse) :

```bash
curl -X POST http://localhost:8000/api/cin/extract \
  -F "recto=@/chemin/vers/cin_recto.jpg" \
  -F "verso=@/chemin/vers/cin_verso.jpg"
```

```json
{
  "success": true,
  "data": {
    "nom": "ALAOUI",
    "prenom": "YASSINE",
    "cin": "BE123456",
    "date_naissance": "12/03/1990",
    "lieu_naissance": "CASABLANCA",
    "date_expiration": "05/11/2029",
    "adresse": "12 RUE DES FLEURS, CASABLANCA"
  },
  "model": "hf.co/unsloth/GLM-4.6V-Flash-GGUF:Q4_K_M",
  "processing_time_ms": 3120,
  "warning": null
}
```

> **Astuce :** l'adresse figure généralement au **verso** de la CIN marocaine
> (le recto ne contient que le *lieu de naissance*). Fournissez `verso` pour
> l'obtenir — si vous ne fournissez que le recto, l'API répond quand même
> mais avec un `warning` indiquant que l'adresse est manquante. Un PDF unique
> de 2 pages (recto+verso scanné) fonctionne aussi : passez-le simplement en
> `recto`, la 2ᵉ page sera automatiquement traitée comme le verso.

### ICE

```bash
curl -X POST http://localhost:8000/api/v1/extract-ice \
  -F "file=@/chemin/vers/certificat_ice.jpg"
```

```json
{
  "success": true,
  "data": {
    "ICE": "001543281000091",
    "Denomination": "STE TALAMT GAZ",
    "Identifiant_Fiscal": "4170282",
    "RC_Numero": "25725",
    "RC_Ville": "MEKNES",
    "CNSS": "7475404"
  },
  "model": "hf.co/unsloth/GLM-4.6V-Flash-GGUF:Q4_K_M",
  "ocr_method": "vision",
  "processing_time_ms": 5800,
  "warning": null
}
```

`ocr_method` indique le chemin suivi : `vision` (image), `pdfplumber` (PDF
numérique) ou `vision-pdf` (PDF scanné, rendu en image puis vision).

## Notes de conception

- **Zéro dépendance système** : ni Tesseract, ni Poppler. Les images sont
  toujours envoyées directement au modèle GLM vision ; seul `pdfplumber`
  (pur Python) et `PyMuPDF` (pur Python, aucun binaire externe) sont utilisés
  pour les PDF.
- **`vision_client.py`** centralise l'appel bas niveau à l'API native
  d'Ollama (`/api/chat`, via `httpx`) avec `format: "json"` — réutilisé par
  la CIN et par l'ICE (image ou PDF scanné).
- **Image normalisée** avant envoi (rotation EXIF corrigée, redimensionnée,
  réencodée en JPEG) pour des appels plus rapides et plus robustes.
- **Traitement en mémoire uniquement** : les fichiers ne sont jamais écrits
  sur disque.
- **Non bloquant** : les traitements PDF synchrones (pdfplumber/PyMuPDF) sont
  déchargés dans un thread (`asyncio.to_thread`) pour ne pas bloquer la
  boucle d'événements FastAPI.
- **Préchargement des modèles** au démarrage (`warmup_model()`) pour éviter
  les timeouts de chargement à froid sur la première requête utilisateur.
- **Recto/verso traités en parallèle** (`asyncio.gather`) puis fusionnés
  champ par champ (`merge_cin_sides`) : pas de latence supplémentaire par
  rapport à un seul appel, même avec les deux faces.
