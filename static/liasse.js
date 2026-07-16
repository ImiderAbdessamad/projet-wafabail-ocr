// Interface de test — liasse fiscale PCGM (PDF / ZIP multi-pages) + scoring optionnel

const dropzone = document.getElementById("dropzone");
const dropzoneEmpty = document.getElementById("dropzoneEmpty");
const previewFile = document.getElementById("previewFile");
const previewFileName = document.getElementById("previewFileName");
const previewFileSize = document.getElementById("previewFileSize");
const fileInput = document.getElementById("fileInput");
const zipPicker = document.getElementById("zipPicker");
const zipEntrySelect = document.getElementById("zipEntrySelect");
const analyzeBtn = document.getElementById("analyzeBtn");
const analyzeBtnLabel = document.getElementById("analyzeBtnLabel");
const resetBtn = document.getElementById("resetBtn");
const errorBanner = document.getElementById("errorBanner");
const scoringOverrides = document.getElementById("scoringOverrides");

const stateIdle = document.getElementById("stateIdle");
const stateLoading = document.getElementById("stateLoading");
const stateResult = document.getElementById("stateResult");
const loadingText = document.getElementById("loadingText");

const kindBadge = document.getElementById("kindBadge");
const completenessBadge = document.getElementById("completenessBadge");
const pagesBadge = document.getElementById("pagesBadge");
const timeBadge = document.getElementById("timeBadge");
const summaryText = document.getElementById("summaryText");
const sectionsBar = document.getElementById("sectionsBar");
const warningsBox = document.getElementById("warningsBox");
const elementsGrid = document.getElementById("elementsGrid");
const scoringMetricsSection = document.getElementById("scoringMetricsSection");
const scoringMetricsGrid = document.getElementById("scoringMetricsGrid");
const scoringPanel = document.getElementById("scoringPanel");
const scoringSkipped = document.getElementById("scoringSkipped");
const decisionCard = document.getElementById("decisionCard");
const axesGrid = document.getElementById("axesGrid");
const ratiosGrid = document.getElementById("ratiosGrid");
const copyJsonBtn = document.getElementById("copyJsonBtn");
const downloadJsonBtn = document.getElementById("downloadJsonBtn");
const downloadExcelBtn = document.getElementById("downloadExcelBtn");

const ALLOWED_TYPES = ["application/pdf", "application/zip", "application/x-zip-compressed"];
const MAX_BYTES = 50 * 1024 * 1024;

let selectedFile = null;
let isZip = false;
let zipEntries = [];
let lastResult = null;

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} o`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} Ko`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} Mo`;
}

function formatAmount(value) {
  if (value == null) return null;
  return new Intl.NumberFormat("fr-MA", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 2,
  }).format(value);
}

function getMode() {
  return document.querySelector('input[name="mode"]:checked')?.value || "extract";
}

function setResultState(state) {
  stateIdle.hidden = state !== "idle";
  stateLoading.hidden = state !== "loading";
  stateResult.hidden = state !== "result";
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

function resetAll() {
  selectedFile = null;
  isZip = false;
  zipEntries = [];
  lastResult = null;
  fileInput.value = "";
  previewFile.hidden = true;
  dropzoneEmpty.hidden = false;
  zipPicker.hidden = true;
  zipEntrySelect.innerHTML = "";
  scoringOverrides.hidden = getMode() !== "score";
  document.querySelectorAll("[data-financial], [data-root]").forEach((input) => {
    input.value = "";
  });
  analyzeBtn.disabled = true;
  resetBtn.hidden = true;
  clearError();
  setResultState("idle");
}

function buildScoringComplement() {
  const financial_overrides = {};
  document.querySelectorAll("[data-financial]").forEach((input) => {
    if (input.value.trim() !== "") {
      financial_overrides[input.dataset.financial] = Number(input.value);
    }
  });
  const complement = {};
  if (Object.keys(financial_overrides).length) {
    complement.financial_overrides = financial_overrides;
  }
  document.querySelectorAll("[data-root]").forEach((input) => {
    if (input.value.trim() !== "") {
      complement[input.dataset.root] = Number(input.value);
    }
  });
  return complement;
}

async function loadZipEntries(file) {
  const formData = new FormData();
  formData.append("file", file);
  const res = await fetch("/api/v1/extraction/liasse/zip/list", {
    method: "POST",
    body: formData,
  });
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error(data?.detail || `Erreur HTTP ${res.status}`);
  return data.entries || [];
}

async function handleFile(file) {
  clearError();
  if (!file) return;

  const isZipFile =
    file.type.includes("zip") || file.name.toLowerCase().endsWith(".zip");
  if (!isZipFile && !ALLOWED_TYPES.includes(file.type) && file.type !== "") {
    showError("Format non supporté. Utilisez un PDF ou une archive ZIP.");
    return;
  }
  if (file.size > MAX_BYTES) {
    showError("Fichier trop volumineux (max 50 Mo).");
    return;
  }

  selectedFile = file;
  isZip = isZipFile;
  dropzoneEmpty.hidden = true;
  previewFileName.textContent = file.name;
  previewFileSize.textContent = formatSize(file.size);
  previewFile.hidden = false;
  analyzeBtn.disabled = false;
  resetBtn.hidden = false;
  setResultState("idle");

  if (isZip) {
    zipPicker.hidden = false;
    zipEntrySelect.innerHTML = '<option value="">Chargement…</option>';
    try {
      zipEntries = await loadZipEntries(file);
      zipEntrySelect.innerHTML = zipEntries
        .map(
          (e) =>
            `<option value="${e.path}">${e.path.split("/").pop()} — ${e.pages} p. (${e.size_label})</option>`
        )
        .join("");
      // Sélectionner le PDF le plus complet (plus de pages)
      const best = zipEntries.reduce((a, b) => (b.pages > a.pages ? b : a), zipEntries[0]);
      if (best) zipEntrySelect.value = best.path;
    } catch (err) {
      showError(err.message);
      zipPicker.hidden = true;
    }
  } else {
    zipPicker.hidden = true;
    zipEntries = [];
  }
}

function renderSections(sections) {
  sectionsBar.innerHTML = "";
  const labels = {
    BILAN_ACTIF: "Bilan Actif",
    BILAN_PASSIF: "Bilan Passif",
    CPC: "CPC",
  };
  Object.entries(sections || {}).forEach(([key, ok]) => {
    const chip = document.createElement("span");
    chip.className = `section-chip ${ok ? "ok" : "missing"}`;
    chip.textContent = `${labels[key] || key} : ${ok ? "✓ capturé" : "✗ manquant"}`;
    sectionsBar.appendChild(chip);
  });
}

function renderElements(elements) {
  elementsGrid.innerHTML = "";
  (elements || []).forEach((el) => {
    const card = document.createElement("div");
    card.className = `element-card ${el.value != null ? "has-value" : ""}`;
    const missingLabels = {
      empty: "Case visible mais vide",
      not_detected: "Non détecté dans le document",
      derived: "Dérivé d'un autre poste",
    };
    const val = el.value != null
      ? `${formatAmount(el.value)} MAD`
      : (missingLabels[el.detection_status] || "Non détecté dans le document");
    card.innerHTML = `
      <div class="element-num">#${el.number} · ${el.source}</div>
      <div class="element-label">${el.label}</div>
      <div class="element-value ${el.value == null ? "empty" : ""}">${val}</div>
      ${el.note ? `<div class="element-meta">${el.note}</div>` : ""}
      ${el.confidence > 0 ? `<div class="element-meta">Confiance ${Math.round(el.confidence * 100)}%</div>` : ""}
    `;
    elementsGrid.appendChild(card);
  });
}

function renderScoringMetrics(components) {
  scoringMetricsGrid.innerHTML = "";
  const metrics = (components || []).filter((component) =>
    component.feeds?.includes("autonomie") ||
    component.feeds?.includes("caf") ||
    component.feeds?.includes("fdr") ||
    component.feeds?.includes("croissance") ||
    component.feeds?.includes("endettement")
  );
  scoringMetricsSection.hidden = metrics.length === 0;
  metrics.forEach((metric) => {
    const card = document.createElement("div");
    card.className = "element-card has-value";
    card.innerHTML = `
      <div class="element-num">${metric.source}</div>
      <div class="element-label">${metric.label}</div>
      <div class="element-value">${formatAmount(metric.value)} MAD</div>
      <div class="element-meta">${metric.feeds || ""}</div>
    `;
    scoringMetricsGrid.appendChild(card);
  });
}

function statusClass(status) {
  if (!status) return "non-calculable";
  return status
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/\s+/g, "-");
}

function renderScoring(scoring) {
  if (!scoring) return;
  scoringPanel.hidden = false;
  scoringSkipped.hidden = true;

  const d = scoring.decision || {};
  decisionCard.innerHTML = `
    <div class="decision-score">${d.score ?? "—"} / 100</div>
    <div class="decision-class">Classe ${d.classe || "—"}</div>
    <div class="decision-text">${d.decision || ""} — ${d.recommandation || ""}</div>
  `;

  axesGrid.innerHTML = "";
  [
    { key: "axe1", label: "Financier (75%)" },
    { key: "axe2", label: "Comportemental (15%)" },
    { key: "axe3", label: "Sectoriel (10%)" },
  ].forEach(({ key, label }) => {
    const axe = scoring[key];
    if (!axe) return;
    const card = document.createElement("div");
    card.className = "axe-card";
    card.innerHTML = `
      <div class="axe-label">${label}</div>
      <div class="axe-score">${axe.score ?? "—"}</div>
      <div class="axe-weight">Contribution ${axe.contribution ?? "—"}</div>
    `;
    axesGrid.appendChild(card);
  });

  ratiosGrid.innerHTML = "";
  Object.values(scoring.ratios || {}).forEach((ratio) => {
    const card = document.createElement("div");
    card.className = "ratio-card";
    const cls = statusClass(ratio.status);
    const displayValue = ratio.value != null ? Number(ratio.value).toFixed(2) : "N/C";
    card.innerHTML = `
      <div class="ratio-label">${ratio.label}</div>
      <span class="ratio-status ${cls}">${ratio.status || "—"}</span>
      <div class="ratio-value">${displayValue} · ${ratio.threshold || ""}</div>
      ${ratio.reason ? `<div class="ratio-value">${ratio.reason}</div>` : ""}
    `;
    ratiosGrid.appendChild(card);
  });
}

function renderExtraction(extraction) {
  const kindLabels = {
    LIASSE_OCR: "OCR Vision",
    LIASSE_NATIVE: "Texte PCGM",
    RAPPORT_INDICATEURS: "Rapport indicateurs",
    LIASSE_ECHEC: "Échec",
  };

  kindBadge.textContent = kindLabels[extraction.document_kind] || extraction.document_kind;
  completenessBadge.textContent = `${extraction.completeness_pct ?? 0}% complet`;

  if (extraction.pages_total) {
    pagesBadge.textContent = `${extraction.pages_analyzed ?? "?"}/${extraction.pages_total} pages`;
    pagesBadge.hidden = false;
  } else {
    pagesBadge.hidden = true;
  }

  if (extraction.processing_time_ms) {
    const sec = (extraction.processing_time_ms / 1000).toFixed(1);
    timeBadge.textContent = `${sec}s`;
    timeBadge.hidden = false;
  } else {
    timeBadge.hidden = true;
  }

  summaryText.textContent = extraction.document_summary || "";
  renderSections(extraction.sections_completeness);
  renderElements(extraction.elements);
  renderScoringMetrics(extraction.raw_components);

  if (extraction.warnings?.length) {
    warningsBox.hidden = false;
    warningsBox.innerHTML = `<strong>Avertissements</strong><ul>${extraction.warnings.map((w) => `<li>${w}</li>`).join("")}</ul>`;
  } else {
    warningsBox.hidden = true;
  }
}

function renderResult(payload) {
  lastResult = payload;
  const extraction = payload.extraction || payload;
  renderExtraction(extraction);

  scoringPanel.hidden = true;
  scoringSkipped.hidden = true;

  if (payload.scoring) {
    renderScoring(payload.scoring);
    if (payload.scoring_warning) {
      scoringSkipped.textContent = payload.scoring_warning;
      scoringSkipped.hidden = false;
    }
  } else if (payload.scoring_skipped_reason) {
    scoringSkipped.textContent = payload.scoring_skipped_reason;
    scoringSkipped.hidden = false;
  }

  setResultState("result");
}

async function analyze() {
  if (!selectedFile) return;
  clearError();

  const mode = getMode();
  const endpoint =
    mode === "score"
      ? "/api/v1/extraction/liasse/score"
      : "/api/v1/extraction/liasse";

  const formData = new FormData();
  formData.append("file", selectedFile);
  if (isZip && zipEntrySelect.value) {
    formData.append("pdf_entry", zipEntrySelect.value);
  }
  if (mode === "score") {
    const complement = buildScoringComplement();
    if (Object.keys(complement).length) {
      formData.append("complement", JSON.stringify(complement));
    }
  }

  analyzeBtn.disabled = true;
  const pagesHint = isZip
    ? zipEntries.find((e) => e.path === zipEntrySelect.value)?.pages
    : null;
  loadingText.textContent = pagesHint
    ? `OCR vision — analyse de ~${pagesHint} pages en cours…`
    : "OCR vision en cours — analyse page par page…";
  setResultState("loading");

  try {
    const response = await fetch(endpoint, { method: "POST", body: formData });
    const payload = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(payload?.detail || `Erreur HTTP ${response.status}`);
    }
    renderResult(payload);
  } catch (err) {
    console.error(err);
    showError(err.message || "Erreur inattendue.");
    setResultState("idle");
  } finally {
    analyzeBtn.disabled = false;
  }
}

// --- Events ---

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

document.querySelectorAll('input[name="mode"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const isScore = getMode() === "score";
    analyzeBtnLabel.textContent = isScore ? "Extraire & scorer" : "Analyser la liasse";
    scoringOverrides.hidden = !isScore;
  });
});

resetBtn.addEventListener("click", resetAll);
analyzeBtn.addEventListener("click", analyze);

copyJsonBtn.addEventListener("click", async () => {
  if (!lastResult) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2));
    copyJsonBtn.textContent = "Copié !";
    setTimeout(() => (copyJsonBtn.textContent = "Copier JSON"), 1500);
  } catch (err) {
    console.error(err);
  }
});

async function downloadExport(format) {
  if (!lastResult) return;
  const btn = format === "excel" ? downloadExcelBtn : downloadJsonBtn;
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Export…";
  try {
    const res = await fetch(`/api/v1/export/${format}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(lastResult),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Export ${format} échoué (${res.status})`);
    }
    const blob = await res.blob();
    const disposition = res.headers.get("Content-Disposition") || "";
    const match = /filename="?([^"]+)"?/.exec(disposition);
    const filename =
      match?.[1] ||
      (format === "excel" ? "liasse_resultats.xlsx" : "liasse_resultats.json");
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (err) {
    showError(err.message || String(err));
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

downloadJsonBtn.addEventListener("click", () => downloadExport("json"));
downloadExcelBtn.addEventListener("click", () => downloadExport("excel"));

resetAll();
