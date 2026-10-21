/**
 * CardioPulse FL - Dense Scientific Dashboard Logic
 * Clean, Compact, Modular Hyperparameters & Append-Only Run History Inspector
 */

// Global State
const state = {
  isRunning: false,
  activeStrategy: 'fedbn',
  chartMode: 'strategy', // 'strategy' (semua algoritma), 'client', atau '<strat>_vm'
  activeMetric: 'f1',    // 'f1', 'auroc', 'loss'
  activeLogSource: 'server',
  inspectedRun: null,    // Jika user menginspeksi run historis tertentu
  historicalRuns: [],
  metricsHistory: null,
  benchmarkData: null,
  statusTimer: null,
  logsTimer: null,
  modalMetric: 'f1',
  modalRunData: null,
};

let metricChart = null;
let modalMetricChart = null;

// Color Palette for Algorithms
const STRATEGY_COLORS = {
  fedavg: { border: '#64748b', bg: 'rgba(100, 116, 139, 0.08)' },
  fedprox: { border: '#8b5cf6', bg: 'rgba(139, 92, 246, 0.08)' },
  scaffold: { border: '#06b6d4', bg: 'rgba(6, 182, 212, 0.08)' },
  fedbn: { border: '#2563eb', bg: 'rgba(37, 99, 235, 0.08)' },
  fedbb: { border: '#10b981', bg: 'rgba(16, 185, 129, 0.08)' },
  fedsol: { border: '#f59e0b', bg: 'rgba(245, 158, 11, 0.08)' },
};

const CLIENT_COLORS = {
  ptbxl: { border: '#6366f1', label: 'PTB-XL (VM 2)' },
  chapman: { border: '#f97316', label: 'Chapman (VM 3)' },
  g12ec: { border: '#06b6d4', label: 'G12EC (VM 4)' },
};

// -------------------------------------------------------------
// 1. MODULAR OPTIONS PANEL
// -------------------------------------------------------------
function renderModularOptions(strat) {
  const titleEl = document.getElementById('modularTitle');
  const badgeEl = document.getElementById('modularBadge');
  const bodyEl = document.getElementById('modularBody');
  if (!titleEl || !bodyEl) return;

  strat = strat.toLowerCase();
  state.activeStrategy = strat;

  if (strat === 'fedavg') {
    titleEl.textContent = 'Parameter Khusus: FedAvg';
    badgeEl.textContent = 'Baseline Standard';
    bodyEl.innerHTML = `
      <div class="modular-desc">
        Algoritma standar Federated Averaging tanpa regularisasi tambahan. Agregasi parameter dilakukan berbasis proporsi jumlah data latih klien (<em>n<sub>k</sub> / N</em>).
      </div>
    `;
  } else if (strat === 'fedprox') {
    titleEl.textContent = 'Parameter Khusus: FedProx';
    badgeEl.textContent = 'Proximal Regularizer';
    bodyEl.innerHTML = `
      <div class="modular-row">
        <div class="modular-field">
          <label for="inputProximalMu">Koefisien Proksimal (&mu;):</label>
          <input type="number" id="inputProximalMu" name="proximal_mu" step="0.001" min="0.0001" max="1.0" value="0.01">
        </div>
      </div>
      <div class="modular-desc">
        Menambahkan loss penalty <code>(&mu;/2) * ||w - w<sup>t</sup>||<sup>2</sup></code> pada pelatihan lokal klien untuk mencegah divergensi bobot saat heterogenitas data tinggi.
      </div>
    `;
  } else if (strat === 'scaffold') {
    titleEl.textContent = 'Parameter Khusus: SCAFFOLD';
    badgeEl.textContent = 'Control Variates';
    bodyEl.innerHTML = `
      <div class="modular-desc">
        Menggunakan variat kontrol lokal <code>c<sub>i</sub></code> dan server <code>c</code> untuk mengoreksi drift gradien klien: <code>g = g<sub>i</sub> - c<sub>i</sub> + c</code>. Agregasi Option 2 aktif secara otomatis.
      </div>
    `;
  } else if (strat === 'fedbn') {
    titleEl.textContent = 'Parameter Khusus: FedBN';
    badgeEl.textContent = 'Local BatchNorm';
    bodyEl.innerHTML = `
      <div class="modular-desc">
        Lapisan <code>BatchNorm1d</code> (bobot, bias, running stats) dipertahankan lokal di tiap klien dan tidak dikirimkan ke server. Mengatasi pergeseran fitur sensor (ECG lead hardware heterogeneity).
      </div>
    `;
  } else if (strat === 'fedbb') {
    titleEl.textContent = 'Parameter Khusus: FedBB';
    badgeEl.textContent = 'PNB Loss + CBR';
    bodyEl.innerHTML = `
      <div class="modular-row">
        <div class="modular-field">
          <label for="inputFedbbAlpha">Bobot Class-Balanced Regularizer (&alpha;):</label>
          <input type="number" id="inputFedbbAlpha" name="fedbb_alpha" step="0.05" min="0.0" max="1.0" value="0.5">
        </div>
      </div>
      <div class="modular-desc">
        Mengurangi bias representasi kelas mayoritas menggunakan Positive-Negative Balanced Binary Cross Entropy dan relaksasi classifier balancing.
      </div>
    `;
  } else if (strat === 'fedsol') {
    titleEl.textContent = 'Parameter Khusus: FedSOL';
    badgeEl.textContent = 'Subspace Drift';
    bodyEl.innerHTML = `
      <div class="modular-row">
        <div class="modular-field">
          <label for="inputFedsolMu">Penalti Drift Ortogonal (&mu;):</label>
          <input type="number" id="inputFedsolMu" name="fedsol_mu" step="0.001" min="0.0001" max="1.0" value="0.01">
        </div>
      </div>
      <div class="modular-desc">
        Memproyeksikan update gradien ortogonal terhadap ruang representasi model global untuk mencegah kehancuran representasi fitur laten EKG.
      </div>
    `;
  } else if (strat === 'all') {
    titleEl.textContent = 'Mode Runtut: Multi-Sesi Otomatis';
    badgeEl.textContent = 'Sequential Benchmark';
    bodyEl.innerHTML = `
      <div style="font-size: 0.68rem; color: #475569; margin-bottom: 0.35rem; font-weight: 600;">
        Pilih algoritma yang akan dieksekusi secara berurutan:
      </div>
      <div class="seq-checklist-grid" id="seqChecklistGrid">
        <label class="seq-check-item"><input type="checkbox" name="seq_algo" value="fedavg" checked onchange="updateSeqPreview()"> FedAvg</label>
        <label class="seq-check-item"><input type="checkbox" name="seq_algo" value="fedprox" checked onchange="updateSeqPreview()"> FedProx</label>
        <label class="seq-check-item"><input type="checkbox" name="seq_algo" value="scaffold" checked onchange="updateSeqPreview()"> SCAFFOLD</label>
        <label class="seq-check-item"><input type="checkbox" name="seq_algo" value="fedbn" checked onchange="updateSeqPreview()"> FedBN</label>
        <label class="seq-check-item"><input type="checkbox" name="seq_algo" value="fedbb" checked onchange="updateSeqPreview()"> FedBB</label>
        <label class="seq-check-item"><input type="checkbox" name="seq_algo" value="fedsol" checked onchange="updateSeqPreview()"> FedSOL</label>
      </div>
      <div class="seq-preview-box" id="seqPreviewText">
        Urutan: FEDAVG ➔ FEDPROX ➔ SCAFFOLD ➔ FEDBN ➔ FEDBB ➔ FEDSOL
      </div>
      <div class="modular-row" style="margin-top: 0.5rem;">
        <div class="modular-field">
          <label for="inputProximalMu">FedProx &mu;:</label>
          <input type="number" id="inputProximalMu" name="proximal_mu" step="0.001" min="0.0001" max="1.0" value="0.01">
        </div>
        <div class="modular-field">
          <label for="inputFedbbAlpha">FedBB &alpha;:</label>
          <input type="number" id="inputFedbbAlpha" name="fedbb_alpha" step="0.05" min="0.0" max="1.0" value="0.5">
        </div>
        <div class="modular-field">
          <label for="inputFedsolMu">FedSOL &mu;:</label>
          <input type="number" id="inputFedsolMu" name="fedsol_mu" step="0.001" min="0.0001" max="1.0" value="0.01">
        </div>
      </div>
      <div class="modular-desc">
        Setiap metode akan dijalankan secara otomatis sebanyak ronde yang ditentukan dengan jeda soket 6 detik antar sesi. Setiap sesi disimpan permanen di <code>results/runs/</code>.
      </div>
    `;
  }
}

function updateSeqPreview() {
  const checked = Array.from(document.querySelectorAll('input[name="seq_algo"]:checked')).map(cb => cb.value);
  const previewEl = document.getElementById('seqPreviewText');
  if (previewEl) {
    if (checked.length > 0) {
      previewEl.textContent = 'Urutan: ' + checked.map(c => c.toUpperCase()).join(' ➔ ');
    } else {
      previewEl.textContent = 'Urutan: (Belum ada algoritma dipilih)';
    }
  }
}

// -------------------------------------------------------------
// 2. CHART MANAGEMENT
// -------------------------------------------------------------
function initChart() {
  const ctx = document.getElementById('liveMetricChart');
  if (!ctx) return;

  metricChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [] },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { font: { family: 'Inter', size: 10, weight: '600' }, boxWidth: 12, padding: 8 }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          titleFont: { family: 'Inter', size: 11 },
          bodyFont: { family: 'JetBrains Mono', size: 10 },
          padding: 8
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Ronde Komunikasi (Round)', color: '#64748b', font: { size: 10 } },
          grid: { color: '#f1f5f9' },
          ticks: { color: '#64748b', font: { size: 9, family: 'JetBrains Mono' } }
        },
        y: {
          title: { display: true, text: 'Nilai Metrik', color: '#64748b', font: { size: 10 } },
          grid: { color: '#f1f5f9' },
          ticks: { color: '#64748b', font: { size: 9, family: 'JetBrains Mono' } }
        }
      }
    }
  });
}

function renderChart() {
  if (!metricChart) return;

  // Jika sedang inspect run historis tertentu
  if (state.inspectedRun) {
    renderInspectedRunChart();
    return;
  }

  const h = state.metricsHistory || {};
  let rounds = h.rounds || [];
  Object.values(h.strategies || {}).forEach(s => {
    if (s && Array.isArray(s.rounds) && s.rounds.length > rounds.length) rounds = s.rounds;
  });

  metricChart.data.labels = rounds;
  const metric = state.activeMetric; // 'f1', 'auroc', 'loss'
  const mode = state.chartMode;

  let yTitle = 'Macro-F1 (0.0 - 1.0)';
  if (metric === 'auroc') yTitle = 'AUROC (0.5 - 1.0)';
  if (metric === 'loss') yTitle = 'BCE Validation Loss';
  metricChart.options.scales.y.title.text = yTitle;

  if (mode === 'strategy') {
    // Tampilkan perbandingan global semua strategi
    const datasets = [];
    const metricKey = (metric === 'f1') ? 'f1_macro' : (metric === 'loss' ? 'val_loss' : 'auroc');

    for (const [sKey, color] of Object.entries(STRATEGY_COLORS)) {
      const sData = (h.strategies && h.strategies[sKey]) ? h.strategies[sKey][metricKey] || [] : [];
      if (sData.length > 0) {
        datasets.push({
          label: sKey.toUpperCase(),
          data: sData,
          borderColor: color.border,
          backgroundColor: color.bg,
          borderWidth: 2,
          tension: 0.15,
          fill: false,
          pointRadius: 2.5
        });
      }
    }
    metricChart.data.datasets = datasets;
  } else if (mode.endsWith('_vm')) {
    // Mode per-VM untuk strategi tertentu (misal: fedbn_vm)
    const stratKey = mode.replace('_vm', '');
    const metricKey = (metric === 'f1') ? 'f1_macro' : (metric === 'loss' ? 'val_loss' : 'auroc');
    const stratData = (h.strategies && h.strategies[stratKey]) ? h.strategies[stratKey] : null;

    const datasets = [];
    if (stratData) {
      // Global
      datasets.push({
        label: `${stratKey.toUpperCase()} (Global)`,
        data: stratData[metricKey] || [],
        borderColor: '#0f172a',
        borderWidth: 2.5,
        tension: 0.15,
        pointRadius: 3
      });
      // Per client
      for (const [dsKey, cInfo] of Object.entries(CLIENT_COLORS)) {
        if (stratData[dsKey] && Array.isArray(stratData[dsKey][metricKey])) {
          datasets.push({
            label: cInfo.label,
            data: stratData[dsKey][metricKey],
            borderColor: cInfo.border,
            borderDash: [4, 3],
            borderWidth: 1.8,
            tension: 0.15,
            pointRadius: 2
          });
        }
      }
    }
    metricChart.data.datasets = datasets;
  }

  metricChart.update();
}

function renderInspectedRunChart() {
  const run = state.inspectedRun;
  if (!run || !run.rounds) return;

  const validRounds = run.rounds.filter(r => r.global);
  const rounds = validRounds.map(r => r.round);
  metricChart.data.labels = rounds;

  const metric = state.activeMetric;
  const metricKey = (metric === 'f1') ? 'f1_macro' : (metric === 'loss' ? 'val_loss' : 'auroc');

  let yTitle = 'Macro-F1 (0.0 - 1.0)';
  if (metric === 'auroc') yTitle = 'AUROC (0.5 - 1.0)';
  if (metric === 'loss') yTitle = 'BCE Validation Loss';
  metricChart.options.scales.y.title.text = yTitle;

  const strat = (run.strategy || 'fedavg').toLowerCase();
  const color = STRATEGY_COLORS[strat] || { border: '#2563eb', bg: 'rgba(37, 99, 235, 0.08)' };

  const globalSeries = validRounds.map(r => r.global ? r.global[metricKey] : null);

  const datasets = [
    {
      label: `${run.run_id} (Global)`,
      data: globalSeries,
      borderColor: color.border,
      backgroundColor: color.bg,
      borderWidth: 2.5,
      tension: 0.15,
      pointRadius: 3
    }
  ];

  // Per client curves jika ada
  for (const [dsKey, cInfo] of Object.entries(CLIENT_COLORS)) {
    const clientSeries = validRounds.map(r => {
      const c = r.clients && r.clients[dsKey];
      return c ? c[metricKey] : null;
    });
    if (clientSeries.some(v => v !== null)) {
      datasets.push({
        label: cInfo.label,
        data: clientSeries,
        borderColor: cInfo.border,
        borderDash: [4, 3],
        borderWidth: 1.5,
        tension: 0.15,
        pointRadius: 2
      });
    }
  }

  metricChart.data.datasets = datasets;
  metricChart.update();
}

// -------------------------------------------------------------
// 3. HISTORICAL RUNS TABLE & ACTIONS (ZERO OVERWRITE)
// -------------------------------------------------------------
async function fetchHistoricalRuns() {
  try {
    const res = await fetch('/api/runs');
    if (!res.ok) return;
    const data = await res.json();
    state.historicalRuns = data.runs || [];
    populateRunSelector();
    renderHistoricalRunsTable();

    // Update storage summary di header
    const summaryEl = document.getElementById('storageSummaryText');
    if (summaryEl) {
      summaryEl.textContent = `Storage: ${state.historicalRuns.length} Runs (Append-Only)`;
    }
  } catch (err) {
    console.error('Error fetching historical runs:', err);
  }
}

function populateRunSelector() {
  const selector = document.getElementById('chartRunSelector');
  if (!selector) return;

  const currentVal = selector.value;
  let html = '<option value="latest">⚡ Sesi Terakhir (Live / Default)</option>';

  state.historicalRuns.forEach(r => {
    const strat = (r.strategy || '').toUpperCase();
    const started = r.started_at ? new Date(r.started_at * 1000).toLocaleString('id-ID', {
      day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit'
    }) : '';
    const f1 = (r.best_f1 != null) ? `F1: ${Number(r.best_f1).toFixed(3)}` : '';
    const label = `${r.run_id} [${strat}] ${started ? `(${started})` : ''} ${f1 ? `— ${f1}` : ''}`;
    html += `<option value="${r.run_id}">${label}</option>`;
  });

  selector.innerHTML = html;
  if (state.inspectedRun) {
    selector.value = state.inspectedRun.run_id;
  } else {
    selector.value = currentVal || 'latest';
  }
}

function renderHistoricalRunsTable() {
  const tbody = document.getElementById('historyTableBody');
  if (!tbody) return;

  const filterStrat = (document.getElementById('filterHistoryStrategy')?.value || 'all').toLowerCase();
  const filterStatus = (document.getElementById('filterHistoryStatus')?.value || 'all').toLowerCase();
  let runs = state.historicalRuns;

  if (filterStrat !== 'all') {
    runs = runs.filter(r => (r.strategy || '').toLowerCase() === filterStrat);
  }

  if (filterStatus !== 'all') {
    runs = runs.filter(r => {
      const st = (r.status || '').toLowerCase();
      if (filterStatus === 'failed') {
        return st === 'failed' || st === 'error' || st === 'incomplete';
      }
      return st === filterStatus;
    });
  }

  if (runs.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="9" class="text-center text-muted" style="padding: 1.25rem;">
          Tidak ada riwayat eksperimen yang sesuai dengan filter yang dipilih.
        </td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = runs.map(r => {
    const strat = (r.strategy || 'unknown').toUpperCase();
    const status = (r.status || 'unknown').toLowerCase();
    const started = r.started_at ? new Date(r.started_at * 1000).toLocaleString('id-ID', {
      month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit'
    }) : '-';
    
    const roundsDone = r.rounds_done ?? 0;
    const totalRounds = r.config?.num_rounds ?? 50;

    const f1Str = (r.best_f1 != null) ? Number(r.best_f1).toFixed(4) : '-';
    const lossStr = (r.final_loss != null) ? Number(r.final_loss).toFixed(4) : '-';
    const aurocStr = (r.final_auroc != null) ? Number(r.final_auroc).toFixed(4) : '-';

    const isInspecting = state.inspectedRun && state.inspectedRun.run_id === r.run_id;

    return `
      <tr style="${isInspecting ? 'background: #eff6ff;' : ''}">
        <td class="mono" style="font-weight: 600;">${r.run_id}</td>
        <td>${started}</td>
        <td><span class="mono" style="font-weight: 700; color: #1e293b;">${strat}</span></td>
        <td class="mono">${roundsDone} / ${totalRounds}</td>
        <td><span class="badge-status ${status}">${status}</span></td>
        <td class="f1-highlight">${f1Str}</td>
        <td class="mono">${lossStr}</td>
        <td class="mono">${aurocStr}</td>
        <td>
          <div style="display: flex; gap: 0.3rem; flex-wrap: wrap;">
            <button class="table-btn primary" onclick="openRunDetailModal('${r.run_id}')" title="Buka grafik layar penuh & detail lengkap run ini">
              📈 Grafik
            </button>
            <button class="table-btn" onclick="inspectHistoricalRun('${r.run_id}')" title="Visualisasikan run ini ke grafik dasbor">
              Dasbor
            </button>
            ${r.has_best_model ? `
              <a class="table-btn" href="/api/download-model?run=${r.run_id}&tag=best" download title="Unduh bobot PyTorch terbaik">
                best.pt
              </a>
            ` : ''}
            <a class="table-btn" href="/api/download-metrics?run=${r.run_id}" download title="Unduh metrics JSON lengkap">
              JSON
            </a>
          </div>
        </td>
      </tr>
    `;
  }).join('');
}

async function inspectHistoricalRun(runId) {
  if (runId === 'latest') {
    exitInspect();
    return;
  }
  try {
    const res = await fetch(`/api/runs/${runId}`);
    if (!res.ok) return;
    const runData = await res.json();
    state.inspectedRun = runData;

    // Sync selector
    const selector = document.getElementById('chartRunSelector');
    if (selector) selector.value = runId;

    // Show inspect banner with extra details
    const banner = document.getElementById('inspectBanner');
    const runIdText = document.getElementById('inspectRunId');
    const detailsText = document.getElementById('inspectRunDetails');
    if (banner && runIdText) {
      runIdText.textContent = runId;
      if (detailsText) {
        const strat = (runData.strategy || '').toUpperCase();
        const rounds = (runData.rounds || []).length;
        const bestF1 = runData.models?.best?.f1_macro != null ? Number(runData.models.best.f1_macro).toFixed(4) : '-';
        detailsText.textContent = `• Strategi: ${strat} • ${rounds} Ronde • Best Macro-F1: ${bestF1}`;
      }
      banner.style.display = 'flex';
    }

    renderHistoricalRunsTable();
    renderChart();

    // Scroll to chart smoothly
    document.getElementById('cardMetricsPlots')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    console.error('Error inspecting run:', err);
  }
}

function exitInspect() {
  state.inspectedRun = null;
  const banner = document.getElementById('inspectBanner');
  if (banner) banner.style.display = 'none';
  const selector = document.getElementById('chartRunSelector');
  if (selector) selector.value = 'latest';
  renderHistoricalRunsTable();
  renderChart();
}

// -------------------------------------------------------------
// HISTORICAL RUN MODAL DIALOG & CHART
// -------------------------------------------------------------
async function openRunDetailModal(runId) {
  try {
    const res = await fetch(`/api/runs/${runId}`);
    if (!res.ok) return;
    const runData = await res.json();
    state.modalRunData = runData;

    const modal = document.getElementById('runDetailModal');
    if (!modal) return;

    // Title & subtitle
    document.getElementById('modalRunTitle').textContent = `Inspeksi Grafik: ${runId}`;
    const strat = (runData.strategy || '').toUpperCase();
    const started = runData.started_at ? new Date(runData.started_at * 1000).toLocaleString('id-ID') : '-';
    const durationSec = (runData.finished_at && runData.started_at) ? Math.round(runData.finished_at - runData.started_at) : '-';
    document.getElementById('modalRunSubtitle').textContent = `Strategi ${strat} • Dimulai ${started} • Durasi Total: ${durationSec}s`;

    // Metadata pills
    const bestF1 = runData.models?.best?.f1_macro != null ? Number(runData.models.best.f1_macro).toFixed(4) : '-';
    const lastRound = runData.rounds?.[runData.rounds.length - 1];
    const finalLoss = lastRound?.global?.val_loss != null ? Number(lastRound.global.val_loss).toFixed(4) : '-';
    const finalAuroc = lastRound?.global?.auroc != null ? Number(lastRound.global.auroc).toFixed(4) : '-';

    const metaBar = document.getElementById('modalMetaBar');
    if (metaBar) {
      metaBar.innerHTML = `
        <div class="meta-pill">Algoritma: <strong>${strat}</strong></div>
        <div class="meta-pill">Status: <strong class="badge-status ${runData.status || ''}">${runData.status || '-'}</strong></div>
        <div class="meta-pill">Ronde Selesai: <strong>${(runData.rounds || []).length} / ${runData.config?.num_rounds ?? '-'}</strong></div>
        <div class="meta-pill">Best F1-Macro: <strong style="color: #2563eb;">${bestF1}</strong></div>
        <div class="meta-pill">Final Loss: <strong>${finalLoss}</strong></div>
        <div class="meta-pill">Final AUROC: <strong>${finalAuroc}</strong></div>
      `;
    }

    // Config details
    const cfgBox = document.getElementById('modalConfigDetails');
    if (cfgBox && runData.config) {
      const c = runData.config;
      cfgBox.innerHTML = `
        <div class="modal-config-item"><span>Batch Size</span><strong>${c.batch_size ?? 32}</strong></div>
        <div class="modal-config-item"><span>Local Epochs</span><strong>${c.local_epochs ?? 3}</strong></div>
        <div class="modal-config-item"><span>Learning Rate</span><strong>${c.learning_rate ?? 0.001}</strong></div>
        <div class="modal-config-item"><span>Val Fraction</span><strong>${c.val_fraction ?? 0.2}</strong></div>
        <div class="modal-config-item"><span>Split Seed</span><strong>${c.split_seed ?? 42}</strong></div>
        ${c.proximal_mu != null ? `<div class="modal-config-item"><span>Proximal Mu</span><strong>${c.proximal_mu}</strong></div>` : ''}
        ${c.fedbb_alpha != null || c.alpha != null ? `<div class="modal-config-item"><span>FedBB Alpha</span><strong>${c.fedbb_alpha ?? c.alpha}</strong></div>` : ''}
        ${c.fedsol_mu != null ? `<div class="modal-config-item"><span>FedSOL Mu</span><strong>${c.fedsol_mu}</strong></div>` : ''}
      `;
    }

    // Download buttons
    const dlModel = document.getElementById('modalDownloadModelBtn');
    if (dlModel) {
      if (runData.models?.best) {
        dlModel.href = `/api/download-model?run=${runId}&tag=best`;
        dlModel.style.display = 'inline-flex';
      } else {
        dlModel.style.display = 'none';
      }
    }
    const dlJson = document.getElementById('modalDownloadJsonBtn');
    if (dlJson) {
      dlJson.href = `/api/download-metrics?run=${runId}`;
    }

    modal.style.display = 'flex';
    renderModalChart();
  } catch (err) {
    console.error('Error opening run detail modal:', err);
  }
}

function closeRunDetailModal() {
  const modal = document.getElementById('runDetailModal');
  if (modal) modal.style.display = 'none';
}

function renderModalChart() {
  const run = state.modalRunData;
  if (!run || !run.rounds) return;

  const ctx = document.getElementById('modalMetricChart');
  if (!ctx) return;

  const validRounds = run.rounds.filter(r => r.global);
  const rounds = validRounds.map(r => r.round);

  const metric = state.modalMetric || 'f1';
  const metricKey = (metric === 'f1') ? 'f1_macro' : (metric === 'loss' ? 'val_loss' : 'auroc');

  let yTitle = 'Macro-F1 (0.0 - 1.0)';
  if (metric === 'auroc') yTitle = 'AUROC (0.5 - 1.0)';
  if (metric === 'loss') yTitle = 'BCE Validation Loss';

  const strat = (run.strategy || 'fedavg').toLowerCase();
  const color = STRATEGY_COLORS[strat] || { border: '#2563eb', bg: 'rgba(37, 99, 235, 0.08)' };

  const datasets = [
    {
      label: `Model Global (${strat.toUpperCase()})`,
      data: validRounds.map(r => r.global ? r.global[metricKey] : null),
      borderColor: color.border,
      backgroundColor: color.bg,
      borderWidth: 3,
      tension: 0.15,
      pointRadius: 3
    }
  ];

  // Per-client series
  for (const [dsKey, cInfo] of Object.entries(CLIENT_COLORS)) {
    const clientSeries = validRounds.map(r => {
      const c = r.clients && r.clients[dsKey];
      return c ? c[metricKey] : null;
    });
    if (clientSeries.some(v => v !== null)) {
      datasets.push({
        label: cInfo.label,
        data: clientSeries,
        borderColor: cInfo.border,
        borderDash: [5, 4],
        borderWidth: 2,
        tension: 0.15,
        pointRadius: 2.5
      });
    }
  }

  if (modalMetricChart) {
    modalMetricChart.destroy();
  }

  modalMetricChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: rounds,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: { font: { size: 11, family: 'Inter' }, boxWidth: 14 }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            title: (items) => `Ronde ${items[0].label}`,
            label: (item) => `${item.dataset.label}: ${item.raw != null ? Number(item.raw).toFixed(4) : '-'}`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Ronde Federasi' },
          grid: { color: '#f1f5f9' }
        },
        y: {
          title: { display: true, text: yTitle },
          grid: { color: '#f1f5f9' },
          beginAtZero: metric !== 'auroc',
          min: metric === 'auroc' ? 0.45 : undefined,
          max: (metric === 'f1' || metric === 'auroc') ? 1.0 : undefined
        }
      }
    }
  });
}

// -------------------------------------------------------------
// 4. BENCHMARK COMPARISON TABLE
// -------------------------------------------------------------
async function fetchBenchmark() {
  try {
    const res = await fetch('/api/benchmark');
    if (!res.ok) return;
    const data = await res.json();
    state.benchmarkData = data.benchmark || {};

    const activeBadge = document.getElementById('benchmarkActiveBadge');
    if (activeBadge && data.active_strategy) {
      activeBadge.textContent = `Aktif: ${data.active_strategy.toUpperCase()}`;
    }

    renderBenchmarkTable();
  } catch (err) {
    console.error('Error fetching benchmark:', err);
  }
}

function renderBenchmarkTable() {
  const tbody = document.getElementById('benchmarkTableBody');
  if (!tbody || !state.benchmarkData) return;

  const entries = Object.entries(state.benchmarkData);
  if (entries.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" class="text-center text-muted">Belum ada data benchmark.</td></tr>`;
    return;
  }

  tbody.innerHTML = entries.map(([sKey, item]) => {
    const name = item.name || sKey.toUpperCase();
    const f1 = item.best_f1 != null ? Number(item.best_f1).toFixed(4) : (item.macro_f1 != null ? Number(item.macro_f1).toFixed(4) : '-');
    const auroc = item.auroc != null ? Number(item.auroc).toFixed(4) : '-';
    const loss = item.final_loss != null ? Number(item.final_loss).toFixed(4) : '-';
    const speed = item.speed_sec != null ? `${item.speed_sec} s/rd` : '-';
    const numRuns = item.num_runs ?? 0;
    const status = item.status || 'pending';

    return `
      <tr>
        <td style="font-weight: 700; color: #1e293b;">${name}</td>
        <td style="color: #475569;">${item.mechanism || '-'}</td>
        <td style="color: #64748b; font-size: 0.68rem;">${item.target || '-'}</td>
        <td class="f1-highlight">${f1}</td>
        <td class="mono">${auroc}</td>
        <td class="mono">${loss}</td>
        <td class="mono">${speed}</td>
        <td class="mono" style="text-align: center;">${numRuns}</td>
        <td><span class="badge-status ${status}">${status}</span></td>
      </tr>
    `;
  }).join('');
}

// -------------------------------------------------------------
// 5. CLIENT STATUS & TELEMETRY
// -------------------------------------------------------------
async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const data = await res.json();

    state.isRunning = Boolean(data.is_running ?? (data.status === 'running'));
    updateServerBadge(data.server_status || data.status);
    updateControlButtons(state.isRunning);
    updateClients(data.clients || {}, data.connected_clients ?? data.connected_count ?? 0);
    updateProgress(data);

    // Ambil metrics jika sedang running
    if (state.isRunning) {
      await fetchMetrics();
    }
  } catch (err) {
    console.error('Error fetching status:', err);
  }
}

function updateServerBadge(status) {
  const dot = document.getElementById('serverStatusDot');
  const txt = document.getElementById('serverStatusText');
  if (!dot || !txt) return;

  status = (status || 'idle').toLowerCase();
  dot.className = `dot ${status}`;
  txt.textContent = `Server: ${status.charAt(0).toUpperCase() + status.slice(1)}`;
}

function updateControlButtons(isRunning) {
  const btnStart = document.getElementById('btnStartTraining');
  const btnStop = document.getElementById('btnStopTraining');
  if (btnStart) btnStart.disabled = isRunning;
  if (btnStop) btnStop.disabled = !isRunning;
}

function updateClients(clients, connectedCount) {
  const summaryEl = document.getElementById('clientsConnectedSummary');
  if (summaryEl) {
    summaryEl.textContent = `${connectedCount} / 3 Klien Aktif`;
    summaryEl.className = `badge-status ${connectedCount >= 3 ? 'online' : (connectedCount > 0 ? 'training' : 'offline')}`;
  }

  ['vm2', 'vm3', 'vm4'].forEach(k => {
    const info = clients[k];
    const rowEl = document.getElementById(`clientNode${k.charAt(0).toUpperCase() + k.slice(1)}`);
    const badgeEl = document.getElementById(`badge${k.charAt(0).toUpperCase() + k.slice(1)}`);
    const timingEl = document.getElementById(`timing${k.charAt(0).toUpperCase() + k.slice(1)}`);

    if (!rowEl || !info) return;

    const st = info.status || 'offline';
    rowEl.className = `client-node-row is-${st}`;
    if (badgeEl) {
      badgeEl.className = `badge-status ${st}`;
      badgeEl.textContent = info.status_label || st;
    }

    if (timingEl) {
      if (st === 'training') {
        timingEl.textContent = `R${info.current_round} • Ep ${info.epoch}/${info.total_epochs} • Batch ${info.batch}/${info.total_batches}`;
      } else if (st === 'online') {
        timingEl.textContent = 'Online (Standby gRPC)';
      } else {
        timingEl.textContent = 'Node Offline';
      }
    }
  });
}

function updateProgress(data) {
  const roundText = document.getElementById('progressRoundText');
  const percentText = document.getElementById('progressPercentText');
  const barFill = document.getElementById('progressBarFill');
  const etaText = document.getElementById('progressEtaText');
  const speedText = document.getElementById('progressSpeedText');

  const curRound = data.current_round ?? 0;
  const totalRounds = data.total_rounds ?? 50;
  const pct = totalRounds > 0 ? Math.min(100, Math.round((curRound / totalRounds) * 100)) : 0;
  const isRunning = Boolean(data.is_running ?? (data.status === 'running') ?? state.isRunning);

  if (roundText) {
    roundText.textContent = isRunning
      ? `Ronde Federasi: ${curRound} / ${totalRounds}`
      : (curRound > 0 ? `Selesai Ronde: ${curRound} / ${totalRounds}` : 'Ronde Federasi: Standby');
  }
  if (percentText) percentText.textContent = `${pct}%`;
  if (barFill) barFill.style.width = `${pct}%`;

  if (etaText) {
    if (isRunning) {
      if (data.eta_formatted) {
        etaText.textContent = data.eta_formatted;
      } else if (data.eta_seconds != null && data.eta_seconds > 0) {
        const m = Math.floor(data.eta_seconds / 60);
        const s = data.eta_seconds % 60;
        etaText.textContent = m > 0 ? `~${m}m ${s}s (${Math.max(0, totalRounds - curRound)} rd tersisa)` : `~${s}s`;
      } else {
        etaText.textContent = curRound > 0 ? `Mengukur Ronde ${curRound}...` : 'Menghitung estimasi...';
      }
    } else {
      if (curRound >= totalRounds && totalRounds > 0) {
        etaText.textContent = 'Pelatihan Selesai';
      } else {
        etaText.textContent = data.eta_formatted || 'Siap dimulai';
      }
    }
  }
  if (speedText) {
    if (data.speed_formatted && data.speed_formatted !== '--') {
      speedText.textContent = `Laju: ${data.speed_formatted}`;
    } else if (data.avg_round_duration != null && data.avg_round_duration > 0) {
      speedText.textContent = `Laju: ${data.avg_round_duration} s/rd`;
    } else {
      speedText.textContent = isRunning ? 'Laju: Mengukur...' : 'Kecepatan: --';
    }
  }
}

// -------------------------------------------------------------
// 6. METRICS & LOGS POLLING
// -------------------------------------------------------------
async function fetchMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) return;
    const data = await res.json();
    state.metricsHistory = data;
    renderChart();
  } catch (err) {
    console.error('Error fetching metrics:', err);
  }
}

async function fetchLogs() {
  const box = document.getElementById('logStreamBox');
  if (!box) return;

  try {
    const res = await fetch(`/api/logs?source=${state.activeLogSource}`);
    if (!res.ok) return;
    const text = await res.text();
    if (text) {
      const isScrolledToBottom = box.scrollHeight - box.clientHeight <= box.scrollTop + 40;
      box.textContent = text;
      if (isScrolledToBottom) {
        box.scrollTop = box.scrollHeight;
      }
    }
  } catch (err) {
    console.error('Error fetching logs:', err);
  }
}

// -------------------------------------------------------------
// 7. EXPERIMENT START / STOP / RESET
// -------------------------------------------------------------
async function startTraining() {
  const form = document.getElementById('formExperimentControl');
  if (!form) return;

  const formData = new FormData(form);
  const payload = {};
  formData.forEach((value, key) => {
    if (['num_rounds', 'local_epochs', 'batch_size'].includes(key)) {
      payload[key] = parseInt(value, 10);
    } else if (['learning_rate', 'proximal_mu', 'fedbb_alpha', 'fedsol_mu'].includes(key)) {
      payload[key] = parseFloat(value);
    } else {
      payload[key] = value;
    }
  });

  // Jika mode runtut dipilih, ambil algoritma yang dicentang sesuai urutan
  if (payload.strategy === 'all') {
    const checked = Array.from(document.querySelectorAll('input[name="seq_algo"]:checked')).map(cb => cb.value);
    if (checked.length === 0) {
      alert('Pilih minimal satu algoritma untuk dijalankan dalam mode runtut!');
      return;
    }
    payload.strategy = checked.join(',');
  }

  const btnStart = document.getElementById('btnStartTraining');
  if (btnStart) btnStart.disabled = true;

  try {
    const res = await fetch('/api/experiment/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (res.ok) {
      state.isRunning = true;
      exitInspect();
      fetchStatus();
      fetchBenchmark();
      fetchHistoricalRuns();
    } else {
      alert(`Gagal memulai eksperimen: ${result.message || 'Error server'}`);
      if (btnStart) btnStart.disabled = false;
    }
  } catch (err) {
    alert(`Terjadi kesalahan jaringan: ${err.message}`);
    if (btnStart) btnStart.disabled = false;
  }
}

async function stopTraining() {
  if (!confirm('Hentikan proses pelatihan Flower Server yang sedang berjalan?')) return;
  try {
    await fetch('/api/experiment/stop', { method: 'POST' });
    state.isRunning = false;
    fetchStatus();
    fetchBenchmark();
    fetchHistoricalRuns();
  } catch (err) {
    alert(`Gagal menghentikan pelatihan: ${err.message}`);
  }
}

async function resetExperiment() {
  if (!confirm('Reset data live experiment saat ini? (Data permanen di results/runs/ TIDAK akan terhapus)')) return;
  try {
    await fetch('/api/experiment/reset', { method: 'POST' });
    fetchStatus();
    fetchMetrics();
    fetchBenchmark();
    fetchHistoricalRuns();
  } catch (err) {
    alert(`Gagal reset: ${err.message}`);
  }
}

// -------------------------------------------------------------
// 8. EVENT LISTENERS INITIALIZATION
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initChart();

  // Listeners Radio Strategi
  document.querySelectorAll('input[name="strategy"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      renderModularOptions(e.target.value);
    });
  });
  renderModularOptions('fedbn'); // Default initial

  // Listeners Action Buttons
  document.getElementById('btnStartTraining')?.addEventListener('click', startTraining);
  document.getElementById('btnStopTraining')?.addEventListener('click', stopTraining);
  document.getElementById('btnResetExperiment')?.addEventListener('click', resetExperiment);

  // Listeners Segmented Controls: Mode Chart
  document.querySelectorAll('#chartModeTabs .seg-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('#chartModeTabs .seg-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      state.chartMode = e.target.dataset.mode;
      renderChart();
    });
  });

  // Listeners Segmented Controls: Metric Switcher
  document.querySelectorAll('.metric-control .seg-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.metric-control .seg-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      state.activeMetric = e.target.dataset.metric;
      renderChart();
    });
  });

  // Listeners Segmented Controls: Log Source Tabs
  document.querySelectorAll('#logSourceTabs .seg-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('#logSourceTabs .seg-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      state.activeLogSource = e.target.dataset.source;
      fetchLogs();
    });
  });

  // Chart Run Selector (Pilih Sesi Live atau Arsip Lampau)
  document.getElementById('chartRunSelector')?.addEventListener('change', (e) => {
    if (e.target.value === 'latest') {
      exitInspect();
    } else {
      inspectHistoricalRun(e.target.value);
    }
  });

  // Historical Runs Refresh & Filter
  document.getElementById('btnRefreshHistory')?.addEventListener('click', fetchHistoricalRuns);
  document.getElementById('filterHistoryStrategy')?.addEventListener('change', renderHistoricalRunsTable);
  document.getElementById('filterHistoryStatus')?.addEventListener('change', renderHistoricalRunsTable);
  document.getElementById('btnExitInspect')?.addEventListener('click', exitInspect);
  document.getElementById('btnModalInspect')?.addEventListener('click', () => {
    if (state.inspectedRun?.run_id) {
      openRunDetailModal(state.inspectedRun.run_id);
    }
  });

  // Modal Dialog Controls
  document.getElementById('btnModalClose')?.addEventListener('click', closeRunDetailModal);
  document.getElementById('runDetailModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'runDetailModal') closeRunDetailModal();
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeRunDetailModal();
  });

  document.querySelectorAll('#modalMetricTabs .seg-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('#modalMetricTabs .seg-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      state.modalMetric = e.target.dataset.metric;
      renderModalChart();
    });
  });

  // Benchmark Refresh
  document.getElementById('btnRefreshBenchmark')?.addEventListener('click', fetchBenchmark);

  // Download Model
  document.getElementById('btnDownloadModel')?.addEventListener('click', () => {
    window.location.href = '/api/download-model?tag=best';
  });

  // Initial Fetches
  fetchStatus();
  fetchMetrics();
  fetchBenchmark();
  fetchHistoricalRuns();
  fetchLogs();

  // Polling Loops
  state.statusTimer = setInterval(fetchStatus, 2500);
  state.logsTimer = setInterval(fetchLogs, 2500);
});

// Expose globals for inline HTML onclick attributes
window.inspectHistoricalRun = inspectHistoricalRun;
window.exitInspect = exitInspect;
window.updateSeqPreview = updateSeqPreview;
window.openRunDetailModal = openRunDetailModal;
window.closeRunDetailModal = closeRunDetailModal;
