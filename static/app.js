// Interface de test — upload du recto (+ verso optionnel) d'une CIN et
// affichage des champs extraits. Vanilla JS, aucune dépendance externe.

const ALLOWED_TYPES = ["image/jpeg", "image/jpg", "image/png", "image/webp", "application/pdf"];
const MAX_BYTES = 15 * 1024 * 1024;

const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const errorBanner = document.getElementById("errorBanner");
const warningBanner = document.getElementById("warningBanner");

const stateIdle = document.getElementById("stateIdle");
const stateLoading = document.getElementById("stateLoading");
const stateResult = document.getElementById("stateResult");

const modelBadge = document.getElementById("modelBadge");
const timeBadge = document.getElementById("timeBadge");
const copyJsonBtn = document.getElementById("copyJsonBtn");

const FIELD_MAP = {
  fieldNom: "nom",
  fieldPrenom: "prenom",
  fieldCin: "cin",
  fieldNaissance: "date_naissance",
  fieldLieuNaissance: "lieu_naissance",
  fieldExpiration: "date_expiration",
  fieldAdresse: "adresse",
};

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

// --- Un "slot" d'upload (recto ou verso), factorisé pour éviter la duplication ---

function createUploadSlot(name, onChange) {
  const dropzone = document.getElementById(`dropzone${name}`);
  const dropzoneEmpty = document.getElementById(`dropzoneEmpty${name}`);
  const previewImg = document.getElementById(`previewImg${name}`);
  const previewFile = document.getElementById(`previewFile${name}`);
  const previewFileName = document.getElementById(`previewFileName${name}`);
  const previewFileSize = document.getElementById(`previewFileSize${name}`);
  const fileInput = document.getElementById(`fileInput${name}`);
  const removeBtn = document.getElementById(`remove${name}`);

  let currentFile = null;

  function showEmpty() {
    dropzoneEmpty.hidden = false;
    previewImg.hidden = true;
    previewImg.src = "";
    previewFile.hidden = true;
    removeBtn.hidden = true;
    currentFile = null;
    fileInput.value = "";
  }

  function showPreview(file) {
    dropzoneEmpty.hidden = true;
    removeBtn.hidden = false;

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
  }

  function handleFile(file) {
    clearError();
    if (!file) return;

    if (!ALLOWED_TYPES.includes(file.type)) {
      showError("Format non supporté. Utilisez une image JPEG/PNG/WEBP ou un PDF.");
      return;
    }
    if (file.size > MAX_BYTES) {
      showError("Fichier trop volumineux (max 15 Mo).");
      return;
    }

    currentFile = file;
    showPreview(file);
    onChange();
  }

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

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
  dropzone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  removeBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    showEmpty();
    onChange();
  });

  return {
    getFile: () => currentFile,
    reset: showEmpty,
  };
}

function handleSlotChange() {
  analyzeBtn.disabled = !rectoSlot.getFile();
  resetBtn.hidden = !(rectoSlot.getFile() || versoSlot.getFile());
  setResultState("idle");
}

const rectoSlot = createUploadSlot("Recto", handleSlotChange);
const versoSlot = createUploadSlot("Verso", handleSlotChange);

resetBtn.addEventListener("click", () => {
  rectoSlot.reset();
  versoSlot.reset();
  lastResult = null;
  clearError();
  handleSlotChange();
});

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
  const rectoFile = rectoSlot.getFile();
  if (!rectoFile) return;

  clearError();
  analyzeBtn.disabled = true;
  setResultState("loading");

  const formData = new FormData();
  formData.append("recto", rectoFile);
  const versoFile = versoSlot.getFile();
  if (versoFile) formData.append("verso", versoFile);

  try {
    const response = await fetch("/api/cin/extract", {
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
    analyzeBtn.disabled = !rectoSlot.getFile();
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

handleSlotChange();
