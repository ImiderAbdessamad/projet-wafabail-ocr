const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const dropzoneEmpty = document.getElementById("dropzoneEmpty");
const fileListPanel = document.getElementById("fileListPanel");
const fileListEl = document.getElementById("fileList");
const clearFilesBtn = document.getElementById("clearFilesBtn");
const maxPagesInput = document.getElementById("maxPages");
const forceVisionInput = document.getElementById("forceVision");
const extractBtn = document.getElementById("extractBtn");
const exportAllJsonBtn = document.getElementById("exportAllJsonBtn");
const exportAllExcelBtn = document.getElementById("exportAllExcelBtn");
const errorBox = document.getElementById("errorBox");
const requestMeta = document.getElementById("requestMeta");
const analyzeOptions = document.getElementById("analyzeOptions");
const scoringModeSelect = document.getElementById("scoringMode");
const resultsContainer = document.getElementById("resultsContainer");
const docTabs = document.getElementById("docTabs");
const docPanels = document.getElementById("docPanels");
const batchSummary = document.getElementById("batchSummary");

/** @type {{ id: string, file: File, status: string, error?: string, extraction?: any, analysis?: any }[]} */
let queue = [];
let activeDocId = null;
let busy = false;

function uid() {
  return `doc_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

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

function updateButtonLabel() {
  const analyze = getMode() === "analyze";
  const n = queue.length;
  if (busy) return;
  extractBtn.textContent = analyze
    ? n > 1
      ? `Extraire + Analyser (${n})`
      : "Extraire + Analyser"
    : n > 1
      ? `Extraire le contenu (${n})`
      : "Extraire le contenu";
  analyzeOptions.hidden = !analyze;
  const modeHint = document.getElementById("modeHint");
  if (modeHint) {
    modeHint.textContent = analyze
      ? "Mode analyse : ratios, axes et décision pour chaque PDF du lot."
      : "Mode extraction seule : Markdown page par page, sans scoring.";
  }
  extractBtn.disabled = queue.length === 0 || busy;
}

function statusLabel(status) {
  return (
    {
      pending: "En attente",
      running: "En cours…",
      done: "Terminé",
      error: "Erreur",
    }[status] || status
  );
}

function renderFileList() {
  if (!queue.length) {
    fileListPanel.hidden = true;
    dropzoneEmpty.hidden = false;
    fileListEl.innerHTML = "";
    updateButtonLabel();
    return;
  }
  dropzoneEmpty.hidden = true;
  fileListPanel.hidden = false;
  fileListEl.innerHTML = queue
    .map(
      (item) => `
      <li class="file-list-item" data-id="${escapeHtml(item.id)}">
        <div class="file-meta">
          <span class="file-name" title="${escapeHtml(item.file.name)}">${escapeHtml(item.file.name)}</span>
          <span class="file-size">${formatSize(item.file.size)}</span>
        </div>
        <span class="file-status ${escapeHtml(item.status)}">${escapeHtml(statusLabel(item.status))}</span>
        <button type="button" class="file-remove" data-remove="${escapeHtml(item.id)}" aria-label="Retirer">&times;</button>
      </li>`
    )
    .join("");
  updateButtonLabel();
}

function addFiles(fileList) {
  const files = Array.from(fileList || []).filter(
    (f) => f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf")
  );
  if (!files.length) {
    showError("Veuillez sélectionner un ou plusieurs fichiers PDF.");
    return;
  }
  clearError();
  for (const file of files) {
    const duplicate = queue.some(
      (q) => q.file.name === file.name && q.file.size === file.size
    );
    if (duplicate) continue;
    queue.push({ id: uid(), file, status: "pending" });
  }
  renderFileList();
}

function removeFile(id) {
  if (busy) return;
  queue = queue.filter((q) => q.id !== id);
  if (activeDocId === id) activeDocId = null;
  renderFileList();
  renderResultsShell();
}

dropzone.addEventListener("click", (e) => {
  if (e.target.closest("[data-remove]") || e.target.closest("#clearFilesBtn")) return;
  fileInput.click();
});
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    fileInput.click();
  }
});
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("drag-active");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-active"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-active");
  if (e.dataTransfer?.files?.length) addFiles(e.dataTransfer.files);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files?.length) addFiles(fileInput.files);
  fileInput.value = "";
});
fileListEl.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-remove]");
  if (!btn) return;
  e.stopPropagation();
  removeFile(btn.getAttribute("data-remove"));
});
clearFilesBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  if (busy) return;
  queue = [];
  activeDocId = null;
  resultsContainer.hidden = true;
  docTabs.innerHTML = "";
  docPanels.innerHTML = "";
  exportAllJsonBtn.hidden = true;
  exportAllExcelBtn.hidden = true;
  renderFileList();
});

document.querySelectorAll('input[name="pdfMode"]').forEach((el) => {
  el.addEventListener("change", updateButtonLabel);
});

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

function formatApiDetail(detail) {
  if (detail == null) return "Traitement impossible";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join(" ; ");
  }
  try {
    return JSON.stringify(detail);
  } catch {
    return String(detail);
  }
}

function buildExportPayload(item) {
  return {
    filename: item.file.name,
    status: item.status,
    extraction: item.extraction || null,
    analysis: item.analysis || null,
    error: item.error || null,
  };
}

async function downloadExport(format, payload, fallbackName) {
  const res = await fetch(`/api/v1/export/${format}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(formatApiDetail(err.detail) || `Export ${format} impossible`);
  }
  const blob = await res.blob();
  const disposition = res.headers.get("Content-Disposition") || "";
  const match = /filename="?([^"]+)"?/i.exec(disposition);
  const filename = match?.[1] || fallbackName;
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function copyText(text, btn, label) {
  try {
    await navigator.clipboard.writeText(text);
    const prev = btn.textContent;
    btn.textContent = "Copié !";
    setTimeout(() => {
      btn.textContent = label || prev;
    }, 1500);
  } catch {
    showError("Impossible de copier dans le presse-papiers.");
  }
}

function renderAnalysisBlock(analysis, root) {
  const decisionCard = root.querySelector("[data-decision]");
  const axesGrid = root.querySelector("[data-axes]");
  const ratiosGrid = root.querySelector("[data-ratios]");
  const datasetGrid = root.querySelector("[data-dataset]");
  const analysisWarnings = root.querySelector("[data-analysis-warnings]");
  const analysisRawJson = root.querySelector("[data-analysis-json]");

  if (!analysis) {
    root.querySelector("[data-analysis-section]").hidden = true;
    return;
  }
  root.querySelector("[data-analysis-section]").hidden = false;

  const d = analysis.decision || {};
  const score =
    d.score != null
      ? Number(d.score).toFixed(2)
      : analysis.final_score != null
        ? Number(analysis.final_score).toFixed(2)
        : "—";
  const ratioCount = (analysis.ratios || []).length;
  const ratioOk = (analysis.ratios || []).filter((r) => r.value != null).length;
  const fieldOk = Object.values(analysis.dataset || {}).filter(
    (f) => f && typeof f === "object" && f.value != null
  ).length;

  decisionCard.innerHTML = `
    <div class="decision-score">${score} / 100</div>
    <div class="decision-class">Classe ${escapeHtml(d.risk_class || "—")}</div>
    <div class="decision-text">${escapeHtml(d.decision || "")} — ${escapeHtml(d.recommendation || "")}</div>
    ${d.blocking_status ? `<div class="decision-text">Blocage : ${escapeHtml(d.blocking_status)}</div>` : ""}
    <div class="decision-text">Mode ${escapeHtml(analysis.scoring_mode || "STRICT")}</div>
    <div class="decision-text">${ratioOk}/${ratioCount} ratios · ${fieldOk} champs dataset</div>
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
    `;
    axesGrid.appendChild(card);
  });

  ratiosGrid.innerHTML = "";
  if (!ratioCount) {
    ratiosGrid.innerHTML = `<p class="analyze-hint">Aucun ratio calculé.</p>`;
  }
  (analysis.ratios || []).forEach((ratio) => {
    const card = document.createElement("div");
    card.className = "ratio-card";
    const status = ratio.status || "non_calculable";
    card.innerHTML = `
      <div class="ratio-label">${escapeHtml(ratio.label)}</div>
      <span class="ratio-status ${status}">${escapeHtml(status)}</span>
      <div class="ratio-value">${
        ratio.value != null ? Number(ratio.value).toFixed(2) : "N/C"
      } ${escapeHtml(ratio.unit || "")}</div>
      <div class="axe-weight">${escapeHtml(ratio.threshold || "")}</div>
      <div class="axe-weight">Points ${ratio.points ?? 0} / ${ratio.max_points ?? 0}</div>
    `;
    ratiosGrid.appendChild(card);
  });

  datasetGrid.innerHTML = "";
  Object.entries(analysis.dataset || {}).forEach(([key, field]) => {
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
    analysisWarnings.innerHTML = `<strong>Avertissements</strong><ul>${analysis.warnings
      .map((w) => `<li>${escapeHtml(w)}</li>`)
      .join("")}</ul>`;
  } else {
    analysisWarnings.hidden = true;
  }

  analysisRawJson.textContent = JSON.stringify(analysis, null, 2);
}

function renderExtractionBlock(extraction, root) {
  const section = root.querySelector("[data-extraction-section]");
  if (!extraction) {
    section.hidden = true;
    return;
  }
  section.hidden = false;
  root.querySelector("[data-file-badge]").textContent =
    extraction.source_filename || "document.pdf";
  root.querySelector("[data-pages-badge]").textContent =
    `${extraction.pages_ok ?? "—"}/${extraction.pages_processed ?? "—"} pages OK`;
  root.querySelector("[data-time-badge]").textContent =
    extraction.processing_time_ms != null
      ? `${(extraction.processing_time_ms / 1000).toFixed(1)} s`
      : "—";
  root.querySelector("[data-model-badge]").textContent =
    (extraction.model || "").split("/").pop() || extraction.model || "—";

  const warningsBox = root.querySelector("[data-extract-warnings]");
  if (extraction.warnings?.length) {
    warningsBox.hidden = false;
    warningsBox.innerHTML = `<strong>Avertissements extraction</strong><ul>${extraction.warnings
      .map((w) => `<li>${escapeHtml(w)}</li>`)
      .join("")}</ul>`;
  } else {
    warningsBox.hidden = true;
  }

  const pageList = root.querySelector("[data-page-list]");
  const pageTitle = root.querySelector("[data-page-title]");
  const pageModeBadge = root.querySelector("[data-page-mode]");
  const pageStatusBadge = root.querySelector("[data-page-status]");
  const pageContent = root.querySelector("[data-page-content]");
  const pageError = root.querySelector("[data-page-error]");
  const copyPageBtn = root.querySelector("[data-copy-page]");

  let activePage = extraction.pages?.[0] || null;

  function showPage(page) {
    if (!page) return;
    activePage = page;
    pageTitle.textContent = page.page_title || `Page ${page.page_number}`;
    pageModeBadge.textContent =
      page.extraction_mode === "native" ? "Texte natif" : "GLM Vision";
    pageStatusBadge.textContent = page.status === "ok" ? "OK" : "Erreur";
    if (page.status === "error") {
      pageContent.textContent = "";
      pageError.hidden = false;
      pageError.textContent = page.error || "Erreur inconnue";
      copyPageBtn.hidden = true;
    } else {
      pageError.hidden = true;
      pageContent.textContent = page.content || "(aucun contenu)";
      copyPageBtn.hidden = !page.content;
    }
    pageList.querySelectorAll(".page-tab").forEach((btn) => {
      btn.classList.toggle(
        "active",
        Number(btn.dataset.page) === page.page_number
      );
    });
  }

  pageList.innerHTML = "";
  (extraction.pages || []).forEach((page) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "page-tab";
    btn.dataset.page = String(page.page_number);
    const modeLabel = page.extraction_mode === "native" ? "native" : "vision";
    btn.innerHTML = `
      <span>Page ${page.page_number}</span>
      <span class="page-tab-status ${
        page.status === "ok"
          ? page.extraction_mode === "native"
            ? "native"
            : "ok"
          : "error"
      }">${page.status === "ok" ? modeLabel : "erreur"}</span>
    `;
    btn.addEventListener("click", () => showPage(page));
    pageList.appendChild(btn);
  });
  if (activePage) showPage(activePage);

  copyPageBtn.onclick = () => {
    if (activePage?.content) {
      copyText(activePage.content, copyPageBtn, "Copier le texte");
    }
  };
}

function buildDocPanel(item) {
  const panel = document.createElement("article");
  panel.className = "doc-panel";
  panel.dataset.docId = item.id;
  panel.hidden = item.id !== activeDocId;

  const shortName = item.file.name;
  panel.innerHTML = `
    <div class="doc-panel-toolbar">
      <button type="button" class="btn-ghost btn-sm" data-copy-json>Copier JSON</button>
      <button type="button" class="btn-ghost btn-sm" data-export-json>Export JSON</button>
      <button type="button" class="btn-ghost btn-sm" data-export-excel>Export Excel</button>
    </div>
    ${
      item.status === "error"
        ? `<p class="error-box">${escapeHtml(item.error || "Erreur")}</p>`
        : ""
    }
    <div data-analysis-section>
      <h2 class="section-title">Analyse — ${escapeHtml(shortName)}</h2>
      <div data-decision class="decision-card"></div>
      <div data-analysis-warnings class="warnings-box" hidden></div>
      <h3 class="section-subtitle">Axes</h3>
      <div data-axes class="axes-grid"></div>
      <h3 class="section-subtitle">Ratios</h3>
      <div data-ratios class="ratios-grid"></div>
      <h3 class="section-subtitle">Champs financiers</h3>
      <div data-dataset class="dataset-grid"></div>
      <details class="pdf-raw">
        <summary>JSON analyse</summary>
        <pre data-analysis-json></pre>
      </details>
    </div>
    <div data-extraction-section>
      <h3 class="section-subtitle">Extraction Markdown</h3>
      <div class="pdf-badges">
        <span data-file-badge class="badge"></span>
        <span data-pages-badge class="badge"></span>
        <span data-time-badge class="badge"></span>
        <span data-model-badge class="badge"></span>
      </div>
      <div data-extract-warnings class="warnings-box" hidden></div>
      <div class="pdf-viewer">
        <aside class="pdf-page-list" data-page-list></aside>
        <article class="pdf-page-detail">
          <div class="pdf-page-header">
            <h3 data-page-title>Sélectionnez une page</h3>
            <div class="pdf-page-meta">
              <span data-page-mode class="badge"></span>
              <span data-page-status class="badge"></span>
              <button type="button" class="btn-secondary" data-copy-page hidden>Copier le texte</button>
            </div>
          </div>
          <pre data-page-content class="pdf-content"></pre>
          <p data-page-error class="error-box" hidden></p>
        </article>
      </div>
    </div>
  `;

  renderAnalysisBlock(item.analysis, panel);
  renderExtractionBlock(item.extraction, panel);

  panel.querySelector("[data-copy-json]").addEventListener("click", (e) => {
    const btn = e.currentTarget;
    copyText(
      JSON.stringify(buildExportPayload(item), null, 2),
      btn,
      "Copier JSON"
    );
  });
  panel.querySelector("[data-export-json]").addEventListener("click", async () => {
    try {
      await downloadExport(
        "json",
        buildExportPayload(item),
        `${item.file.name.replace(/\.pdf$/i, "")}_analyse.json`
      );
    } catch (err) {
      showError(err.message);
    }
  });
  panel.querySelector("[data-export-excel]").addEventListener("click", async () => {
    try {
      await downloadExport(
        "excel",
        buildExportPayload(item),
        `${item.file.name.replace(/\.pdf$/i, "")}_analyse.xlsx`
      );
    } catch (err) {
      showError(err.message);
    }
  });

  return panel;
}

function renderResultsShell() {
  const doneItems = queue.filter((q) => q.status === "done" || q.status === "error");
  if (!doneItems.length) {
    resultsContainer.hidden = true;
    exportAllJsonBtn.hidden = true;
    exportAllExcelBtn.hidden = true;
    return;
  }
  resultsContainer.hidden = false;
  if (!activeDocId || !doneItems.some((d) => d.id === activeDocId)) {
    activeDocId = doneItems[0].id;
  }

  const ok = doneItems.filter((d) => d.status === "done").length;
  const err = doneItems.filter((d) => d.status === "error").length;
  batchSummary.textContent = `${ok} document(s) OK · ${err} erreur(s) · ${queue.length} au total`;

  docTabs.innerHTML = "";
  doneItems.forEach((item) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `doc-tab${item.id === activeDocId ? " active" : ""}${
      item.status === "error" ? " has-error" : ""
    }`;
    btn.innerHTML = `<span class="tab-label" title="${escapeHtml(item.file.name)}">${escapeHtml(
      item.file.name
    )}</span>`;
    btn.addEventListener("click", () => {
      activeDocId = item.id;
      renderResultsShell();
    });
    docTabs.appendChild(btn);
  });

  docPanels.innerHTML = "";
  doneItems.forEach((item) => {
    docPanels.appendChild(buildDocPanel(item));
  });

  const exportable = doneItems.filter((d) => d.status === "done");
  exportAllJsonBtn.hidden = exportable.length === 0;
  exportAllExcelBtn.hidden = exportable.length === 0;
}

async function processOne(item, wantAnalyze) {
  item.status = "running";
  item.error = undefined;
  renderFileList();

  const formData = new FormData();
  formData.append("file", item.file);
  const maxPages = maxPagesInput.value.trim();
  if (maxPages) formData.append("max_pages", maxPages);
  if (forceVisionInput.checked) formData.append("force_vision", "true");

  const endpoint = wantAnalyze
    ? "/api/v1/extraction/pdf/analyze"
    : "/api/v1/extraction/pdf/content";

  if (wantAnalyze) {
    formData.append("scoring_mode", scoringModeSelect?.value || "STRICT");
    const behavioral = buildBehavioralJson();
    const sector = buildSectorJson();
    if (behavioral) formData.append("behavioral_data", behavioral);
    if (sector) formData.append("sector_benchmark", sector);
  }

  const response = await fetch(endpoint, { method: "POST", body: formData });
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error(`Réponse non-JSON (HTTP ${response.status})`);
  }
  if (!response.ok) {
    throw new Error(formatApiDetail(payload.detail));
  }

  item.extraction = payload.extraction || payload;
  item.analysis = payload.analysis || null;
  if (wantAnalyze && !item.analysis) {
    throw new Error("Bloc « analysis » manquant dans la réponse API.");
  }
  item.status = "done";
}

extractBtn.addEventListener("click", async () => {
  if (!queue.length || busy) return;
  clearError();
  busy = true;
  extractBtn.disabled = true;
  const wantAnalyze = getMode() === "analyze";
  const endpoint = wantAnalyze
    ? "/api/v1/extraction/pdf/analyze"
    : "/api/v1/extraction/pdf/content";

  // Reset previous results on re-run
  queue.forEach((q) => {
    q.status = "pending";
    q.error = undefined;
    q.extraction = undefined;
    q.analysis = undefined;
  });
  activeDocId = null;
  renderFileList();
  renderResultsShell();

  if (requestMeta) {
    requestMeta.hidden = false;
    requestMeta.textContent = `Lot de ${queue.length} PDF · ${endpoint}…`;
  }

  let okCount = 0;
  for (let i = 0; i < queue.length; i++) {
    const item = queue[i];
    extractBtn.textContent = `Document ${i + 1}/${queue.length}…`;
    if (requestMeta) {
      requestMeta.textContent = `${i + 1}/${queue.length} · ${item.file.name}`;
    }
    try {
      await processOne(item, wantAnalyze);
      okCount += 1;
      if (!activeDocId) activeDocId = item.id;
    } catch (err) {
      item.status = "error";
      item.error = err.message || "Erreur";
    }
    renderFileList();
    renderResultsShell();
  }

  busy = false;
  updateButtonLabel();
  if (requestMeta) {
    requestMeta.textContent = `Terminé · ${okCount}/${queue.length} OK · ${endpoint}`;
  }
  if (okCount === 0) {
    showError("Aucun document n'a pu être traité.");
  }
});

exportAllJsonBtn.addEventListener("click", async () => {
  const documents = queue
    .filter((q) => q.status === "done")
    .map((q) => buildExportPayload(q));
  try {
    await downloadExport("json", { documents }, "analyse_financiere_lot.json");
  } catch (err) {
    showError(err.message);
  }
});

exportAllExcelBtn.addEventListener("click", async () => {
  const documents = queue
    .filter((q) => q.status === "done")
    .map((q) => buildExportPayload(q));
  try {
    await downloadExport("excel", { documents }, "analyse_financiere_lot.xlsx");
  } catch (err) {
    showError(err.message);
  }
});

updateButtonLabel();
