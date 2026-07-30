/* Analyse liasse fiscale — GLM Vision direct (SSE) */
(function () {
  "use strict";

  const MAX_BYTES = 50 * 1024 * 1024;
  const API_JOBS = "/api/v1/financial-documents/jobs";

  const state = {
    file: null,
    jobId: null,
    eventSource: null,
    result: null,
    analyzing: false,
  };

  const $ = (id) => document.getElementById(id);

  function text(el, value) {
    if (el) el.textContent = value == null ? "" : String(value);
  }

  function formatBytes(n) {
    if (n < 1024) return `${n} o`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} Ko`;
    return `${(n / (1024 * 1024)).toFixed(1)} Mo`;
  }

  function formatAmount(value) {
    if (value == null || value === "") return "—";
    return String(value);
  }

  function showError(message) {
    const el = $("uploadError");
    if (!el) return;
    text(el, message);
    el.hidden = !message;
  }

  function validateFile(file) {
    if (!file) return "Sélectionnez un fichier PDF.";
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      return "Seuls les fichiers PDF sont acceptés.";
    }
    if (file.size > MAX_BYTES) {
      return `Fichier trop volumineux (max ${formatBytes(MAX_BYTES)}).`;
    }
    return null;
  }

  function selectFile(file) {
    const err = validateFile(file);
    if (err) {
      showError(err);
      return;
    }
    showError("");
    state.file = file;
    $("dropzoneEmpty").hidden = true;
    $("previewFile").hidden = false;
    $("removeFile").hidden = false;
    text($("previewFileName"), file.name);
    text($("previewFileSize"), formatBytes(file.size));
    $("analyzeBtn").disabled = false;
  }

  function clearFile() {
    state.file = null;
    $("fileInput").value = "";
    $("dropzoneEmpty").hidden = false;
    $("previewFile").hidden = true;
    $("removeFile").hidden = true;
    $("analyzeBtn").disabled = true;
    showError("");
  }

  function setAnalyzing(active) {
    state.analyzing = active;
    $("analyzeBtn").disabled = active || !state.file;
    $("resetBtn").hidden = !active && !state.result;
    text($("analyzeBtn"), active ? "Analyse en cours…" : "Lancer l'analyse");
  }

  function updateProgress(payload) {
    $("progressCard").hidden = false;
    const pct = Number(payload.progress_pct || 0);
    $("progressFill").style.width = `${pct}%`;
    $("progressBar").setAttribute("aria-valuenow", String(pct));
    text($("progressMessage"), payload.message || payload.event || "…");
    text($("progressStep"), payload.current_step || payload.event || "—");
    if (payload.current_page && payload.pages_total) {
      text($("progressPage"), `${payload.current_page} / ${payload.pages_total}`);
    } else if (payload.pages_total) {
      text($("progressPage"), `— / ${payload.pages_total}`);
    }
    if (payload.pages_financial != null) text($("progressFinancial"), payload.pages_financial);
    if (payload.pages_skipped != null) text($("progressSkipped"), payload.pages_skipped);
    if (payload.pages_failed != null) text($("progressFailed"), payload.pages_failed);

    const step = payload.current_step || "";
    document.querySelectorAll("#stepsList li").forEach((li) => {
      const key = li.getAttribute("data-step");
      li.classList.toggle("active", key === step);
      const order = [
        "validating",
        "rendering",
        "classifying",
        "extracting_page",
        "resolving",
        "controls",
        "ratios",
        "completed",
      ];
      const idx = order.indexOf(key);
      const cur = order.indexOf(step);
      li.classList.toggle("done", cur >= 0 && idx >= 0 && idx < cur);
    });
  }

  function closeStream() {
    if (state.eventSource) {
      state.eventSource.close();
      state.eventSource = null;
    }
  }

  async function createAnalysisJob() {
    if (!state.file || state.analyzing) return;
    setAnalyzing(true);
    state.result = null;
    $("summaryCard").hidden = true;
    $("resultsCard").hidden = true;
    updateProgress({
      progress_pct: 1,
      current_step: "queued",
      message: "Création du job…",
    });

    const form = new FormData();
    form.append("file", state.file);
    form.append("include_markdown", "false");

    try {
      const res = await fetch(API_JOBS, { method: "POST", body: form });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Erreur HTTP ${res.status}`);
      }
      const data = await res.json();
      state.jobId = data.job_id;
      connectToJobStream(data.job_id);
    } catch (err) {
      showError(err.message || String(err));
      setAnalyzing(false);
      $("resetBtn").hidden = false;
    }
  }

  function connectToJobStream(jobId) {
    closeStream();
    const url = `${API_JOBS}/${jobId}/stream`;
    const es = new EventSource(url);
    state.eventSource = es;

    const forward = (eventName) => {
      es.addEventListener(eventName, (ev) => {
        let payload = {};
        try {
          payload = JSON.parse(ev.data);
        } catch (_) {
          payload = { message: ev.data };
        }
        handleProgressEvent(eventName, payload);
      });
    };

    [
      "job_status",
      "job_started",
      "pdf_validated",
      "pages_rendered",
      "page_classified",
      "page_extracted",
      "page_skipped",
      "page_failed",
      "resolving_fields",
      "running_controls",
      "calculating_ratios",
      "result_ready",
      "job_failed",
    ].forEach(forward);

    es.onerror = () => {
      if (!state.analyzing) return;
      // Reconnexion native EventSource ; si terminé, on ferme
      pollJobStatus(jobId);
    };
  }

  async function pollJobStatus(jobId) {
    try {
      const res = await fetch(`${API_JOBS}/${jobId}`);
      if (!res.ok) return;
      const data = await res.json();
      updateProgress(data);
      if (data.status === "completed") {
        closeStream();
        await loadJobResult(jobId);
      } else if (data.status === "failed") {
        closeStream();
        showError(data.error || "Analyse échouée.");
        setAnalyzing(false);
        $("resetBtn").hidden = false;
      }
    } catch (_) {
      /* ignore */
    }
  }

  function handleProgressEvent(eventName, payload) {
    const merged = { ...payload, event: eventName };
    if (eventName === "job_status") {
      updateProgress(payload);
      return;
    }
    if (eventName === "pages_rendered") {
      updateProgress({
        ...merged,
        current_step: "rendering",
        progress_pct: 15,
        pages_total: payload.count,
        message: `${payload.count || "?"} pages rendues`,
      });
      return;
    }
    if (eventName === "page_classified" || eventName === "page_extracted") {
      updateProgress({
        ...merged,
        current_step: "extracting_page",
        current_page: payload.page,
        message:
          eventName === "page_extracted"
            ? `Extraction page ${payload.page} (${payload.page_type || ""})`
            : `Classification page ${payload.page}`,
      });
      return;
    }
    if (eventName === "resolving_fields") {
      updateProgress({
        current_step: "resolving",
        progress_pct: 85,
        message: "Résolution des champs…",
      });
      return;
    }
    if (eventName === "running_controls") {
      updateProgress({
        current_step: "controls",
        progress_pct: 90,
        message: "Contrôles comptables…",
      });
      return;
    }
    if (eventName === "calculating_ratios") {
      updateProgress({
        current_step: "ratios",
        progress_pct: 95,
        message: "Calcul des ratios…",
      });
      return;
    }
    if (eventName === "result_ready") {
      closeStream();
      updateProgress({
        current_step: "completed",
        progress_pct: 100,
        message: "Analyse terminée",
      });
      loadJobResult(state.jobId);
      return;
    }
    if (eventName === "job_failed") {
      closeStream();
      showError(payload.error || "Analyse échouée.");
      setAnalyzing(false);
      $("resetBtn").hidden = false;
      return;
    }
    updateProgress(merged);
  }

  async function loadJobResult(jobId) {
    try {
      const res = await fetch(`${API_JOBS}/${jobId}/result`);
      if (res.status === 409) {
        setTimeout(() => loadJobResult(jobId), 800);
        return;
      }
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Erreur HTTP ${res.status}`);
      }
      state.result = await res.json();
      renderSummary(state.result);
      renderFinancialFields(state.result);
      renderAccountingChecks(state.result);
      renderRatios(state.result);
      renderPageAudit(state.result);
      renderJson(state.result);
      $("summaryCard").hidden = false;
      $("resultsCard").hidden = false;
      setAnalyzing(false);
      $("resetBtn").hidden = false;
    } catch (err) {
      showError(err.message || String(err));
      setAnalyzing(false);
      $("resetBtn").hidden = false;
    }
  }

  function datasetFields(dataset) {
    if (!dataset) return [];
    const rows = [];
    const n1Map = {
      chiffre_affaires: "chiffre_affaires_n1",
      resultat_net: "resultat_net_n1",
      total_bilan: "total_bilan_n1",
      fonds_propres: "fonds_propres_n1",
      dettes_financieres: "dettes_financieres_n1",
    };
    Object.keys(dataset).forEach((key) => {
      if (key === "warnings" || key.endsWith("_n1")) return;
      const fv = dataset[key];
      if (!fv || typeof fv !== "object" || !("code" in fv)) return;
      const n1 = n1Map[key] ? dataset[n1Map[key]] : null;
      rows.push({ key, fv, n1 });
    });
    return rows;
  }

  function renderSummary(result) {
    const grid = $("kpiGrid");
    grid.innerHTML = "";
    const doc = result.document || {};
    const company = doc.company || {};
    const ds = result.dataset || {};
    const decision = result.decision || {};
    const items = [
      ["Entreprise", company.raison_sociale || "—"],
      ["Exercice", (doc.exercise && (doc.exercise.label || doc.exercise.fin)) || "—"],
      ["Chiffre d'affaires", formatAmount(ds.chiffre_affaires && ds.chiffre_affaires.value)],
      ["Résultat net", formatAmount(ds.resultat_net && ds.resultat_net.value)],
      ["Total bilan", formatAmount(ds.total_bilan && ds.total_bilan.value)],
      ["Fonds propres", formatAmount(ds.fonds_propres && ds.fonds_propres.value)],
      ["Décision", decision.risk_class || "—"],
      [
        "Pages",
        `${doc.pages_processed || 0} traitées / ${doc.pages_skipped || 0} ignorées / ${doc.pages_failed || 0} échouées`,
      ],
    ];
    items.forEach(([label, value]) => {
      const card = document.createElement("div");
      card.className = "fd-kpi";
      const l = document.createElement("span");
      l.className = "label";
      text(l, label);
      const v = document.createElement("span");
      v.className = "value";
      text(v, value);
      card.appendChild(l);
      card.appendChild(v);
      grid.appendChild(card);
    });
  }

  function renderFinancialFields(result) {
    const tbody = $("fieldsTable").querySelector("tbody");
    tbody.innerHTML = "";
    const rows = datasetFields(result.dataset);
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      tr.tabIndex = 0;
      const prov = (row.fv.provenance && row.fv.provenance[0]) || {};
      const conf =
        prov.confidence != null ? Number(prov.confidence).toFixed(2) : "—";
      const cells = [
        row.fv.label || row.fv.code,
        formatAmount(row.fv.value),
        formatAmount(row.n1 && row.n1.value),
        row.fv.status || "—",
        prov.page_number != null ? String(prov.page_number) : "—",
        conf,
      ];
      cells.forEach((c, idx) => {
        const td = document.createElement("td");
        if (idx === 3) {
          const badge = document.createElement("span");
          badge.className = `fd-badge ${row.fv.status || ""}`;
          text(badge, c);
          td.appendChild(badge);
        } else {
          text(td, c);
        }
        tr.appendChild(td);
      });
      tr.addEventListener("click", () => showFieldDetail(row.fv));
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          showFieldDetail(row.fv);
        }
      });
      tbody.appendChild(tr);
    });
  }

  function showFieldDetail(fv) {
    const detail = $("fieldDetail");
    detail.hidden = false;
    const prov = (fv.provenance && fv.provenance[0]) || {};
    const lines = [
      `Champ : ${fv.label || fv.code}`,
      `Statut : ${fv.status}`,
      `Valeur : ${formatAmount(fv.value)}`,
      `Page : ${prov.page_number != null ? prov.page_number : "—"}`,
      `Type page : ${prov.page_type || prov.section || "—"}`,
      `Orientation : ${prov.orientation != null ? prov.orientation + "°" : "—"}`,
      `Libellé source : ${prov.raw_label || "—"}`,
      `Colonne : ${prov.column_name || "—"} (${prov.column_role || "—"})`,
      `Extrait : ${prov.source_excerpt || "—"}`,
      `Méthode : ${prov.extraction_method || "—"}`,
      `Warnings : ${(fv.warnings || []).join(" | ") || "—"}`,
    ];
    text(detail, lines.join("\n"));
  }

  function renderAccountingChecks(result) {
    const root = $("controlsList");
    root.innerHTML = "";
    (result.accounting_checks || []).forEach((check) => {
      const item = document.createElement("div");
      item.className = "fd-check-item";
      const h = document.createElement("h3");
      const badge = document.createElement("span");
      badge.className = `fd-badge ${check.status}`;
      text(badge, check.status);
      text(h, check.code + " ");
      h.appendChild(badge);
      const p1 = document.createElement("p");
      text(p1, check.message || "");
      const p2 = document.createElement("p");
      text(
        p2,
        `Attendu=${formatAmount(check.expected)} · Observé=${formatAmount(check.observed)} · Δ=${formatAmount(check.difference)} · Tol=${formatAmount(check.tolerance)}`
      );
      item.appendChild(h);
      item.appendChild(p1);
      item.appendChild(p2);
      root.appendChild(item);
    });
  }

  function renderRatios(result) {
    const root = $("ratiosList");
    root.innerHTML = "";
    (result.ratios || []).forEach((ratio) => {
      const item = document.createElement("div");
      item.className = "fd-check-item";
      const h = document.createElement("h3");
      const badge = document.createElement("span");
      badge.className = `fd-badge ${ratio.status}`;
      text(badge, ratio.status);
      text(h, `${ratio.label || ratio.code} `);
      h.appendChild(badge);
      const p1 = document.createElement("p");
      text(p1, `Valeur : ${formatAmount(ratio.value)} ${ratio.unit || ""}`);
      const p2 = document.createElement("p");
      text(p2, ratio.formula || "");
      item.appendChild(h);
      item.appendChild(p1);
      item.appendChild(p2);
      root.appendChild(item);
    });
  }

  function renderPageAudit(result) {
    const grid = $("pagesGrid");
    const audit = $("auditList");
    grid.innerHTML = "";
    audit.innerHTML = "";
    const pages = (result.extraction && result.extraction.page_audit) || [];
    pages.forEach((page) => {
      const card = document.createElement("div");
      card.className = "fd-page-card";
      const title = document.createElement("strong");
      text(title, `Page ${page.page_number}`);
      card.appendChild(title);
      [
        `Type : ${page.detected_type}`,
        `Orientation : ${page.orientation}°`,
        `Statut : ${page.extraction_status}`,
        `Candidats : ${page.candidates_count}`,
        `Latence : ${page.model_latency_ms != null ? page.model_latency_ms + " ms" : "—"}`,
      ].forEach((line) => {
        const span = document.createElement("span");
        text(span, line);
        card.appendChild(span);
      });
      grid.appendChild(card);

      (page.warnings || []).forEach((w) => {
        const li = document.createElement("li");
        text(li, `Page ${page.page_number} : ${w}`);
        audit.appendChild(li);
      });
      if (page.error) {
        const li = document.createElement("li");
        text(li, `Page ${page.page_number} erreur : ${page.error}`);
        audit.appendChild(li);
      }
    });
    (result.warnings || []).forEach((w) => {
      const li = document.createElement("li");
      text(li, w);
      audit.appendChild(li);
    });
  }

  function renderJson(result) {
    text($("jsonOutput"), JSON.stringify(result, null, 2));
  }

  async function copyJson() {
    const raw = $("jsonOutput").textContent || "";
    try {
      await navigator.clipboard.writeText(raw);
    } catch (_) {
      showError("Impossible de copier le JSON.");
    }
  }

  function downloadJson() {
    const raw = $("jsonOutput").textContent || "{}";
    const blob = new Blob([raw], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = (state.file && state.file.name
      ? state.file.name.replace(/\.pdf$/i, "")
      : "analyse") + "-financial.json";
    a.click();
    URL.revokeObjectURL(url);
  }

  function resetAnalysis() {
    closeStream();
    state.jobId = null;
    state.result = null;
    showError("");
    setAnalyzing(false);
    $("progressCard").hidden = true;
    $("summaryCard").hidden = true;
    $("resultsCard").hidden = true;
    if (state.file) $("analyzeBtn").disabled = false;
  }

  function bindTabs() {
    document.querySelectorAll(".fd-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        const name = tab.getAttribute("data-tab");
        document.querySelectorAll(".fd-tab").forEach((t) => {
          const active = t === tab;
          t.classList.toggle("active", active);
          t.setAttribute("aria-selected", active ? "true" : "false");
        });
        document.querySelectorAll(".fd-panel").forEach((panel) => {
          const match = panel.id === `panel-${name}`;
          panel.hidden = !match;
          panel.classList.toggle("active", match);
        });
      });
    });
  }

  async function checkOllama() {
    try {
      const res = await fetch("/health/ollama");
      const ok = res.ok;
      $("ollamaDot").style.background = ok ? "#2e7d32" : "#c62828";
      text($("ollamaLabel"), ok ? "Ollama disponible · GLM Vision" : "Ollama indisponible");
    } catch (_) {
      $("ollamaDot").style.background = "#c62828";
      text($("ollamaLabel"), "Ollama indisponible");
    }
  }

  function initDropzone() {
    const dz = $("dropzone");
    const input = $("fileInput");
    $("browseBtn").addEventListener("click", (e) => {
      e.stopPropagation();
      input.click();
    });
    dz.addEventListener("click", () => input.click());
    dz.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        input.click();
      }
    });
    input.addEventListener("change", () => {
      if (input.files && input.files[0]) selectFile(input.files[0]);
    });
    ["dragenter", "dragover"].forEach((ev) => {
      dz.addEventListener(ev, (e) => {
        e.preventDefault();
        dz.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach((ev) => {
      dz.addEventListener(ev, (e) => {
        e.preventDefault();
        dz.classList.remove("dragover");
      });
    });
    dz.addEventListener("drop", (e) => {
      const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (file) selectFile(file);
    });
    $("removeFile").addEventListener("click", (e) => {
      e.stopPropagation();
      clearFile();
    });
  }

  function init() {
    text($("maxSizeLabel"), String(MAX_BYTES / (1024 * 1024)));
    initDropzone();
    bindTabs();
    checkOllama();
    $("analyzeBtn").addEventListener("click", createAnalysisJob);
    $("resetBtn").addEventListener("click", () => {
      resetAnalysis();
      if (state.file) createAnalysisJob();
    });
    $("copyJsonBtn").addEventListener("click", copyJson);
    $("downloadJsonBtn").addEventListener("click", downloadJson);
  }

  // Exports pour tests manuels éventuels
  window.FinancialDocumentsUI = {
    selectFile,
    validateFile,
    createAnalysisJob,
    connectToJobStream,
    handleProgressEvent,
    loadJobResult,
    renderSummary,
    renderFinancialFields,
    renderAccountingChecks,
    renderRatios,
    renderPageAudit,
    renderJson,
    copyJson,
    downloadJson,
    resetAnalysis,
  };

  document.addEventListener("DOMContentLoaded", init);
})();
