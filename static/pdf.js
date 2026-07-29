const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dropzoneEmpty = document.getElementById("dropzoneEmpty");
const previewFile = document.getElementById("previewFile");
const previewFileName = document.getElementById("previewFileName");
const previewFileSize = document.getElementById("previewFileSize");
const maxPagesInput = document.getElementById("maxPages");
const forceVisionInput = document.getElementById("forceVision");
const extractBtn = document.getElementById("extractBtn");
const errorBox = document.getElementById("errorBox");
const resultPanel = document.getElementById("resultPanel");
const analysisPanel = document.getElementById("analysisPanel");
const analyzeOptions = document.getElementById("analyzeOptions");
const fileBadge = document.getElementById("fileBadge");
const pagesBadge = document.getElementById("pagesBadge");
const timeBadge = document.getElementById("timeBadge");
const modelBadge = document.getElementById("modelBadge");
const warningsBox = document.getElementById("warningsBox");
const pageList = document.getElementById("pageList");
const pageTitle = document.getElementById("pageTitle");
const pageModeBadge = document.getElementById("pageModeBadge");
const pageStatusBadge = document.getElementById("pageStatusBadge");
const pageContent = document.getElementById("pageContent");
const pageTables = document.getElementById("pageTables");
const pageError = document.getElementById("pageError");
const rawJsonBlock = document.getElementById("rawJsonBlock");
const pageRawJson = document.getElementById("pageRawJson");
const copyBtn = document.getElementById("copyBtn");
const decisionCard = document.getElementById("decisionCard");
const axesGrid = document.getElementById("axesGrid");
const ratiosGrid = document.getElementById("ratiosGrid");
const datasetGrid = document.getElementById("datasetGrid");
const analysisWarnings = document.getElementById("analysisWarnings");
const analysisRawJson = document.getElementById("analysisRawJson");
const scoringModeSelect = document.getElementById("scoringMode");

let selectedFile = null;
let lastResult = null;
let activePage = null;

function getMode() {
  return document.querySelector('input[name="pdfMode"]:checked')?.value || "extract";
}

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} o`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} Ko`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} Mo`;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

function showError(message) {
  errorBox.hidden = false;
  errorBox.textContent =
    typeof message === "string"
      ? message
      : Array.isArray(message)
        ? message.map((m) => m.msg || m).join(" ; ")
        : JSON.stringify(message);
}

function escapeHtml(text) {
  return String(text ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function formatAmount(value) {
  if (value == null || value === "") return "—";
  const n = Number(value);
  if (Number.isNaN(n)) return String(value);
  return n.toLocaleString("fr-FR", { maximumFractionDigits: 2 });
}

function setFile(file) {
  if (!file || file.type !== "application/pdf") {
    showError("Veuillez sélectionner un fichier PDF.");
    return;
  }
  selectedFile = file;
  dropzoneEmpty.hidden = true;
  previewFile.hidden = false;
  previewFileName.textContent = file.name;
  previewFileSize.textContent = formatSize(file.size);
  extractBtn.disabled = false;
  updateButtonLabel();
  clearError();
}

function updateButtonLabel() {
  extractBtn.textContent =
    getMode() === "analyze" ? "Extraire + Analyser" : "Extraire le contenu";
  analyzeOptions.hidden = getMode() !== "analyze";
}

document.querySelectorAll('input[name="pdfMode"]').forEach((el) => {
  el.addEventListener("change", updateButtonLabel);
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer?.files?.[0];
  if (file) setFile(file);
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (file) setFile(file);
});

function renderTables(tables) {
  if (!tables?.length) {
    pageTables.hidden = true;
    pageTables.innerHTML = "";
    return;
  }
  pageTables.hidden = false;
  pageTables.innerHTML = tables
    .map((table, idx) => {
      const headers = (table.headers || [])
        .map((h) => `<th>${escapeHtml(h)}</th>`)
        .join("");
      const rows = (table.rows || [])
        .map(
          (row) =>
            `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`
        )
        .join("");
      const title = table.title
        ? `<h3>${escapeHtml(table.title)}</h3>`
        : `<h3>Tableau ${idx + 1}</h3>`;
      return `<div class="pdf-table-card">${title}<table>${headers ? `<thead><tr>${headers}</tr></thead>` : ""}<tbody>${rows}</tbody></table></div>`;
    })
    .join("");
}

function showPage(page) {
  if (!page) return;
  activePage = page;
  pageTitle.textContent = page.page_title || `Page ${page.page_number}`;
  pageModeBadge.textContent =
    page.extraction_mode === "native" ? "Texte natif" : "GLM Vision";
  pageStatusBadge.textContent = page.status === "ok" ? "OK" : "Erreur";
  pageStatusBadge.style.color =
    page.status === "ok" ? "var(--success)" : "var(--danger)";

  if (page.status === "error") {
    pageContent.textContent = "";
    pageError.hidden = false;
    pageError.textContent = page.error || "Erreur inconnue";
    copyBtn.hidden = true;
    pageTables.hidden = true;
    rawJsonBlock.hidden = true;
  } else {
    pageError.hidden = true;
    pageContent.textContent = page.content || "(aucun contenu extrait)";
    copyBtn.hidden = !page.content;
    renderTables(page.tables);
    if (page.raw_model_response) {
      rawJsonBlock.hidden = false;
      pageRawJson.textContent = JSON.stringify(page.raw_model_response, null, 2);
    } else {
      rawJsonBlock.hidden = true;
    }
  }

  pageList.querySelectorAll(".page-tab").forEach((btn) => {
    btn.classList.toggle("active", Number(btn.dataset.page) === page.page_number);
  });
}

function renderExtraction(result) {
  lastResult = result;
  resultPanel.hidden = false;
  fileBadge.textContent = result.source_filename || "document.pdf";
  pagesBadge.textContent = `${result.pages_ok}/${result.pages_processed} pages OK (${result.pages_total} total)`;
  timeBadge.textContent = `${(result.processing_time_ms / 1000).toFixed(1)} s`;
  modelBadge.textContent = (result.model || "").split("/").pop() || result.model;

  if (result.warnings?.length) {
    warningsBox.hidden = false;
    warningsBox.innerHTML = `<strong>Avertissements extraction</strong><ul>${result.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>`;
  } else {
    warningsBox.hidden = true;
  }

  pageList.innerHTML = "";
  (result.pages || []).forEach((page) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "page-tab";
    btn.dataset.page = String(page.page_number);
    const modeLabel = page.extraction_mode === "native" ? "native" : "vision";
    btn.innerHTML = `
      <span>Page ${page.page_number}</span>
      <span class="page-tab-status ${page.status === "ok" ? (page.extraction_mode === "native" ? "native" : "ok") : "error"}">${page.status === "ok" ? modeLabel : "erreur"}</span>
    `;
    btn.addEventListener("click", () => showPage(page));
    pageList.appendChild(btn);
  });

  if (result.pages?.length) showPage(result.pages[0]);
}

function renderAnalysis(analysis) {
  if (!analysis) {
    analysisPanel.hidden = true;
    return;
  }
  analysisPanel.hidden = false;
  const d = analysis.decision || {};
  const score =
    d.score != null
      ? Number(d.score).toFixed(2)
      : analysis.final_score != null
        ? Number(analysis.final_score).toFixed(2)
        : "—";

  decisionCard.innerHTML = `
    <div class="decision-score">${score} / 100</div>
    <div class="decision-class">Classe ${escapeHtml(d.risk_class || "—")}</div>
    <div class="decision-text">${escapeHtml(d.decision || "")} — ${escapeHtml(d.recommendation || "")}</div>
    ${d.blocking_status ? `<div class="decision-text">Blocage : ${escapeHtml(d.blocking_status)}</div>` : ""}
    <div class="decision-text">Mode ${escapeHtml(analysis.scoring_mode || "STRICT")}</div>
  `;

  axesGrid.innerHTML = "";
  (analysis.axes || []).forEach((axe) => {
    const card = document.createElement("div");
    card.className = "axe-card";
    card.innerHTML = `
      <div class="axe-label">${escapeHtml(axe.label)}</div>
      <div class="axe-score">${axe.raw_score != null ? Number(axe.raw_score).toFixed(2) : "—"}</div>
      <div class="axe-weight">Poids ${axe.weight ?? "—"} · Contrib. ${
        axe.weighted_contribution != null ? Number(axe.weighted_contribution).toFixed(2) : "—"
      }</div>
      <div class="axe-weight">${axe.calculable ? "Calculable" : "Non calculable"}</div>
      ${(axe.blocking_reasons || [])
        .slice(0, 2)
        .map((r) => `<div class="axe-weight">${escapeHtml(r)}</div>`)
        .join("")}
    `;
    axesGrid.appendChild(card);
  });

  ratiosGrid.innerHTML = "";
  (analysis.ratios || []).forEach((ratio) => {
    const card = document.createElement("div");
    card.className = "ratio-card";
    const status = ratio.status || "non_calculable";
    card.innerHTML = `
      <div class="ratio-label">${escapeHtml(ratio.label)}</div>
      <span class="ratio-status ${status}">${escapeHtml(status)}</span>
      <div class="ratio-value">${
        ratio.value != null ? Number(ratio.value).toFixed(2) : "N/C"
      } ${escapeHtml(ratio.unit || "")} · ${escapeHtml(ratio.threshold || "")}</div>
      <div class="ratio-value">${escapeHtml(ratio.formula || "")}</div>
      <div class="ratio-value">Points ${ratio.points ?? 0} / ${ratio.max_points ?? 0}</div>
    `;
    ratiosGrid.appendChild(card);
  });

  datasetGrid.innerHTML = "";
  const dataset = analysis.dataset || {};
  Object.entries(dataset).forEach(([key, field]) => {
    if (!field || typeof field !== "object" || !("status" in field)) return;
    const card = document.createElement("div");
    card.className = "dataset-card";
    card.innerHTML = `
      <div class="field-label">${escapeHtml(field.label || key)}</div>
      <div class="field-value">${formatAmount(field.value)}</div>
      <span class="field-status status-${escapeHtml(field.status)}">${escapeHtml(field.status)}</span>
    `;
    datasetGrid.appendChild(card);
  });

  if (analysis.warnings?.length) {
    analysisWarnings.hidden = false;
    analysisWarnings.innerHTML = `<strong>Avertissements analyse</strong><ul>${analysis.warnings
      .map((w) => `<li>${escapeHtml(w)}</li>`)
      .join("")}</ul>`;
  } else {
    analysisWarnings.hidden = true;
  }

  analysisRawJson.textContent = JSON.stringify(analysis, null, 2);
}

function optionalNumber(id) {
  const el = document.getElementById(id);
  const raw = el?.value?.trim();
  if (!raw) return undefined;
  const n = Number(raw);
  return Number.isFinite(n) ? n : undefined;
}

function buildBehavioralJson() {
  const payload = {};
  const map = {
    bam_rating: "bamRating",
    ca_domiciliation_pct: "caDomPct",
    debit_position_days: "debitDays",
    overdraft_usage_pct: "overdraftPct",
    bank_flows_vs_declared_ca_gap_pct: "flowGapPct",
    payment_incidents_24m: "incidents",
    rejected_debits_24m: "rejections",
    unpaid_bills_24m: "unpaid",
    leasing_payment_delays_24m: "leasingDelays",
  };
  Object.entries(map).forEach(([key, id]) => {
    const value = optionalNumber(id);
    if (value !== undefined) payload[key] = value;
  });
  return Object.keys(payload).length ? JSON.stringify(payload) : null;
}

function buildSectorJson() {
  const payload = {};
  const name = document.getElementById("sectorName")?.value?.trim();
  if (name) payload.sector_name = name;
  const map = {
    commercial_profitability_median: "medCommercial",
    financial_autonomy_median: "medAutonomy",
    debt_ratio_median: "medDebt",
    repayment_capacity_median: "medRepay",
    ca_growth_median: "medGrowth",
  };
  Object.entries(map).forEach(([key, id]) => {
    const value = optionalNumber(id);
    if (value !== undefined) payload[key] = value;
  });
  return Object.keys(payload).length ? JSON.stringify(payload) : null;
}

copyBtn.addEventListener("click", async () => {
  if (!activePage?.content) return;
  try {
    await navigator.clipboard.writeText(activePage.content);
    copyBtn.textContent = "Copié !";
    setTimeout(() => {
      copyBtn.textContent = "Copier le texte";
    }, 1500);
  } catch {
    showError("Impossible de copier dans le presse-papiers.");
  }
});

extractBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  clearError();
  extractBtn.disabled = true;
  const mode = getMode();
  extractBtn.textContent =
    mode === "analyze" ? "Analyse en cours…" : "Extraction en cours…";
  analysisPanel.hidden = true;

  const formData = new FormData();
  formData.append("file", selectedFile);
  const maxPages = maxPagesInput.value.trim();
  if (maxPages) formData.append("max_pages", maxPages);
  if (forceVisionInput.checked) formData.append("force_vision", "true");

  let endpoint = "/api/v1/extraction/pdf/content";
  if (mode === "analyze") {
    endpoint = "/api/v1/extraction/pdf/analyze";
    formData.append("scoring_mode", scoringModeSelect.value || "STRICT");
    const behavioral = buildBehavioralJson();
    const sector = buildSectorJson();
    if (behavioral) formData.append("behavioral_data", behavioral);
    if (sector) formData.append("sector_benchmark", sector);
  }

  try {
    const response = await fetch(endpoint, { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Traitement impossible");
    }
    if (mode === "analyze") {
      renderExtraction(payload.extraction);
      renderAnalysis(payload.analysis);
    } else {
      renderExtraction(payload);
      analysisPanel.hidden = true;
    }
  } catch (err) {
    showError(err.message || "Erreur réseau");
  } finally {
    extractBtn.disabled = false;
    updateButtonLabel();
  }
});

updateButtonLabel();
