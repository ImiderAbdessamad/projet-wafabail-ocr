// Interface de test — upload d'un certificat ICE (image ou PDF) et affichage
// des champs extraits. Vanilla JS, aucune dépendance externe.

const dropzone = document.getElementById("dropzone");
const dropzoneEmpty = document.getElementById("dropzoneEmpty");
const previewImg = document.getElementById("previewImg");
const previewFile = document.getElementById("previewFile");
const previewFileName = document.getElementById("previewFileName");
const previewFileSize = document.getElementById("previewFileSize");
const fileInput = document.getElementById("fileInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const errorBanner = document.getElementById("errorBanner");
const warningBanner = document.getElementById("warningBanner");

const stateIdle = document.getElementById("stateIdle");
const stateLoading = document.getElementById("stateLoading");
const stateResult = document.getElementById("stateResult");
const loadingText = document.getElementById("loadingText");

const modelBadge = document.getElementById("modelBadge");
const ocrBadge = document.getElementById("ocrBadge");
const timeBadge = document.getElementById("timeBadge");
const copyJsonBtn = document.getElementById("copyJsonBtn");

const FIELD_MAP = {
  fieldIce: "ICE",
  fieldDenomination: "Denomination",
  fieldIf: "Identifiant_Fiscal",
  fieldCnss: "CNSS",
  fieldRcNumero: "RC_Numero",
  fieldRcVille: "RC_Ville",
};

const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png", "application/pdf"];

let selectedFile = null;
let lastResult = null;

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

function setResultState(state) {
  stateIdle.hidden = state !== "idle";
  stateLoading.hidden = state !== "loading";
  stateResult.hidden = state !== "result";
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} o`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} Ko`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} Mo`;
}

function resetAll() {
  selectedFile = null;
  lastResult = null;
  fileInput.value = "";
  previewImg.hidden = true;
  previewImg.src = "";
  previewFile.hidden = true;
  dropzoneEmpty.hidden = false;
  analyzeBtn.disabled = true;
  resetBtn.hidden = true;
  clearError();
  setResultState("idle");
}

function handleFile(file) {
  clearError();

  if (!file) return;

  if (!ALLOWED_TYPES.includes(file.type)) {
    showError("Format non supporté. Utilisez une image JPEG/PNG ou un fichier PDF.");
    return;
  }

  const maxBytes = 15 * 1024 * 1024;
  if (file.size > maxBytes) {
    showError("Fichier trop volumineux (max 15 Mo).");
    return;
  }

  selectedFile = file;
  dropzoneEmpty.hidden = true;

  if (file.type === "application/pdf") {
    previewImg.hidden = true;
    previewImg.src = "";
    previewFileName.textContent = file.name;
    previewFileSize.textContent = formatFileSize(file.size);
    previewFile.hidden = false;
  } else {
    previewFile.hidden = true;
    const reader = new FileReader();
    reader.onload = (e) => {
      previewImg.src = e.target.result;
      previewImg.hidden = false;
    };
    reader.readAsDataURL(file);
  }

  analyzeBtn.disabled = false;
  resetBtn.hidden = false;
  setResultState("idle");
}

// --- Interactions dropzone ---

dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  handleFile(e.target.files[0]);
});

["dragenter", "dragover"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-active");
  });
});

["dragleave", "drop"].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-active");
  });
});

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  handleFile(file);
});

dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});

resetBtn.addEventListener("click", resetAll);

// --- Analyse ---

function renderField(elementId, value) {
  const el = document.getElementById(elementId);
  if (value && value.trim()) {
    el.textContent = value;
    el.classList.remove("empty");
  } else {
    el.textContent = "Non détecté";
    el.classList.add("empty");
  }
}

function renderResult(payload) {
  lastResult = payload;
  const data = payload.data || {};

  Object.entries(FIELD_MAP).forEach(([elementId, key]) => {
    renderField(elementId, data[key]);
  });

  modelBadge.textContent = payload.model || "modèle inconnu";
  ocrBadge.textContent = payload.ocr_method || "?";
  timeBadge.textContent = `${payload.processing_time_ms ?? "?"} ms`;

  if (payload.warning) {
    warningBanner.textContent = payload.warning;
    warningBanner.hidden = false;
  } else {
    warningBanner.hidden = true;
  }

  setResultState("result");
}

async function analyze() {
  if (!selectedFile) return;

  clearError();
  analyzeBtn.disabled = true;
  loadingText.textContent = "Analyse du document en cours…";
  setResultState("loading");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/api/v1/extract-ice", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      const detail = (payload && payload.detail) || `Erreur HTTP ${response.status}`;
      throw new Error(detail);
    }

    renderResult(payload);
  } catch (err) {
    console.error(err);
    showError(err.message || "Une erreur inattendue est survenue.");
    setResultState("idle");
  } finally {
    analyzeBtn.disabled = false;
  }
}

analyzeBtn.addEventListener("click", analyze);

// --- Copier le JSON ---

copyJsonBtn.addEventListener("click", async () => {
  if (!lastResult) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(lastResult.data, null, 2));
    copyJsonBtn.textContent = "Copié !";
    setTimeout(() => (copyJsonBtn.textContent = "Copier en JSON"), 1500);
  } catch (err) {
    console.error("Copie impossible", err);
  }
});

resetAll();
