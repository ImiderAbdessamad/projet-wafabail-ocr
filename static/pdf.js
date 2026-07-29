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

let selectedFile = null;
let lastResult = null;
let activePage = null;

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
  errorBox.textContent = message;
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
  clearError();
}

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

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
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

function renderResult(result) {
  lastResult = result;
  resultPanel.hidden = false;
  fileBadge.textContent = result.source_filename || "document.pdf";
  pagesBadge.textContent = `${result.pages_ok}/${result.pages_processed} pages OK (${result.pages_total} total)`;
  timeBadge.textContent = `${(result.processing_time_ms / 1000).toFixed(1)} s`;
  modelBadge.textContent = result.model.split("/").pop() || result.model;

  if (result.warnings?.length) {
    warningsBox.hidden = false;
    warningsBox.innerHTML = `<strong>Avertissements</strong><ul>${result.warnings.map((w) => `<li>${escapeHtml(w)}</li>`).join("")}</ul>`;
  } else {
    warningsBox.hidden = true;
  }

  pageList.innerHTML = "";
  result.pages.forEach((page) => {
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

  if (result.pages.length) {
    showPage(result.pages[0]);
  }
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
  extractBtn.textContent = "Extraction en cours…";

  const formData = new FormData();
  formData.append("file", selectedFile);
  const maxPages = maxPagesInput.value.trim();
  if (maxPages) formData.append("max_pages", maxPages);
  if (forceVisionInput.checked) formData.append("force_vision", "true");

  try {
    const response = await fetch("/api/v1/extraction/pdf/content", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Extraction impossible");
    }
    renderResult(payload);
  } catch (err) {
    showError(err.message || "Erreur réseau");
  } finally {
    extractBtn.disabled = false;
    extractBtn.textContent = "Extraire le contenu";
  }
});
