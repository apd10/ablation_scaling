const xCol = document.getElementById("x-col");
const yCol = document.getElementById("y-col");
const logX = document.getElementById("log-x");
const enableFit = document.getElementById("enable-fit");
const fitOptions = document.getElementById("fit-options");
const plotBtn = document.getElementById("plot-btn");
const exportBtn = document.getElementById("export-btn");
const statusEl = document.getElementById("status");
const metaEl = document.getElementById("meta");
const plotContent = document.getElementById("plot-content");
const plotImage = document.getElementById("plot-image");
const placeholder = document.getElementById("placeholder");
const predictionTableSection = document.getElementById("prediction-table-section");
const predictionTable = document.getElementById("prediction-table");
const exportModal = document.getElementById("export-modal");
const exportCode = document.getElementById("export-code");
const copyExportBtn = document.getElementById("copy-export-btn");

function selectedValues(selector) {
  return [...document.querySelectorAll(selector)]
    .filter((el) => el.checked)
    .map((el) => el.value);
}

function selectedKValues() {
  return [...document.querySelectorAll(".k-opt")]
    .filter((el) => el.checked)
    .map((el) => Number(el.value));
}

function parseOptionalNumber(input) {
  const raw = input.value.trim();
  if (!raw) return null;
  const num = Number(raw);
  return Number.isFinite(num) ? num : null;
}

function parseNonNegativeInt(input, fallback = 0) {
  const raw = input.value.trim();
  if (!raw) return fallback;
  const num = Number.parseInt(raw, 10);
  return Number.isFinite(num) && num >= 0 ? num : fallback;
}

function fillSelect(select, columns, preferred) {
  select.innerHTML = "";
  for (const col of columns) {
    const opt = document.createElement("option");
    opt.value = col;
    opt.textContent = col;
    select.appendChild(opt);
  }
  if (preferred && columns.includes(preferred)) {
    select.value = preferred;
  }
}

function buildPayload() {
  const kSpace = selectedKValues();
  const familySpace = selectedValues(".family-opt");
  return {
    x_col: xCol.value,
    y_col: yCol.value,
    log_x: logX.checked,
    enable_fit: enableFit.checked,
    kmeans_num_fits_space: kSpace.length ? kSpace : [2, 3],
    function_family_space: familySpace.length ? familySpace : ["log", "linear", "sqrt"],
    anchor_x: parseOptionalNumber(document.getElementById("anchor-x")),
    anchor_y: parseOptionalNumber(document.getElementById("anchor-y")),
    remove_survey_clusters: document.getElementById("remove-survey-clusters").checked,
    outlier_n_remove_x: parseNonNegativeInt(document.getElementById("outlier-n-remove-x")),
    outlier_n_remove_y: parseNonNegativeInt(document.getElementById("outlier-n-remove-y")),
  };
}

function validateFitSelection(payload) {
  if (!payload.enable_fit) {
    return null;
  }
  if (payload.kmeans_num_fits_space.length === 0) {
    return "Select at least one k value in Fit search space.";
  }
  if (payload.function_family_space.length === 0) {
    return "Select at least one function family in Fit search space.";
  }
  return null;
}

function updateExportButton() {
  exportBtn.disabled = !enableFit.checked;
}

function formatTableNumber(value) {
  if (!Number.isFinite(value)) return "—";
  const abs = Math.abs(value);
  if (abs >= 1000 || (abs > 0 && abs < 0.01)) return value.toExponential(4);
  if (abs >= 100) return value.toFixed(2);
  if (abs >= 1) return value.toFixed(4);
  return value.toPrecision(4);
}

function renderPredictionTable(table) {
  if (!table?.rows?.length) {
    predictionTableSection.classList.add("hidden");
    predictionTable.innerHTML = "";
    return;
  }

  const headers = [table.x_col, ...table.y_columns];
  const thead = `<thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead>`;
  const bodyRows = table.rows
    .map((row) => {
      const cells = [row.x, ...table.y_columns.map((col) => row[col])];
      return `<tr>${cells.map((v) => `<td>${formatTableNumber(v)}</td>`).join("")}</tr>`;
    })
    .join("");
  predictionTable.innerHTML = `${thead}<tbody>${bodyRows}</tbody>`;
  predictionTableSection.classList.remove("hidden");
}

async function loadColumns() {
  const res = await fetch("/api/columns");
  if (!res.ok) throw new Error("Failed to load columns");
  const data = await res.json();
  fillSelect(xCol, data.columns, "param_b");
  fillSelect(yCol, data.columns, "num_layers");
}

async function generatePlot() {
  statusEl.textContent = "Generating plot…";
  metaEl.classList.add("hidden");

  const payload = buildPayload();
  const fitError = validateFitSelection(payload);
  if (fitError) {
    statusEl.textContent = fitError;
    return;
  }

  const res = await fetch("/api/plot", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (!res.ok) {
    statusEl.textContent = data.detail || "Plot failed";
    return;
  }

  plotImage.src = data.image;
  plotContent.classList.remove("hidden");
  placeholder.classList.add("hidden");
  renderPredictionTable(data.table);
  statusEl.textContent = `${data.meta.n_fit_points ?? data.meta.n_points} plotted (${data.meta.n_outliers_removed ?? 0} outliers excluded)`;

  if (data.meta.k !== undefined) {
    metaEl.textContent = JSON.stringify(data.meta, null, 2);
    metaEl.classList.remove("hidden");
  } else {
    metaEl.classList.add("hidden");
  }
}

function showExportModal(code) {
  exportCode.value = code;
  copyExportBtn.textContent = "Copy to clipboard";
  exportModal.classList.remove("hidden");
  exportModal.setAttribute("aria-hidden", "false");
  exportCode.focus();
  exportCode.select();
}

function hideExportModal() {
  exportModal.classList.add("hidden");
  exportModal.setAttribute("aria-hidden", "true");
}

async function copyExportCode() {
  const text = exportCode.value;
  try {
    await navigator.clipboard.writeText(text);
  } catch (_) {
    exportCode.focus();
    exportCode.select();
    document.execCommand("copy");
  }
  copyExportBtn.textContent = "Copied!";
}

async function exportPython() {
  const payload = buildPayload();
  if (!payload.enable_fit) {
    statusEl.textContent = "Enable piecewise fit before exporting.";
    return;
  }
  const fitError = validateFitSelection(payload);
  if (fitError) {
    statusEl.textContent = fitError;
    return;
  }

  statusEl.textContent = "Generating export…";
  const res = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    let detail = "Export failed";
    try {
      const err = await res.json();
      detail = err.detail || detail;
    } catch (_) {
      /* ignore */
    }
    statusEl.textContent = detail;
    return;
  }

  const code = await res.text();
  showExportModal(code);
  statusEl.textContent = "Export ready — copy from the dialog.";
}

enableFit.addEventListener("change", () => {
  fitOptions.classList.toggle("hidden", !enableFit.checked);
  updateExportButton();
});

plotBtn.addEventListener("click", () => {
  generatePlot().catch((err) => {
    statusEl.textContent = err.message;
  });
});

exportBtn.addEventListener("click", () => {
  exportPython().catch((err) => {
    statusEl.textContent = err.message;
  });
});

copyExportBtn.addEventListener("click", () => {
  copyExportCode().catch((err) => {
    statusEl.textContent = err.message;
  });
});

for (const el of document.querySelectorAll("[data-close-export]")) {
  el.addEventListener("click", hideExportModal);
}

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !exportModal.classList.contains("hidden")) {
    hideExportModal();
  }
});

updateExportButton();

loadColumns()
  .then(() => generatePlot())
  .catch((err) => {
    statusEl.textContent = err.message;
  });
