/**
 * ECG-FL Orchestrator - Frontend Application Logic v3.0
 * Comprehensive state management, live polling, Chart.js sync curves,
 * Node Exporter resource gauges, 27-class modal, and collapsible drawer.
 */

// -------------------------------------------------------------
// 1. GLOBAL APPLICATION STATE
// -------------------------------------------------------------
const state = {
  theme: localStorage.getItem('ecg_fl_theme') || 'dark',
  isRunning: false,
  activeStrategy: 'fedbn',
  activeMetric: 'f1',
  chartViewMode: 'comparison', // 'comparison' or 'clients'
  chartAutoRefresh: true,
  chartInitializedOnce: false,
  activeLogSource: 'server',
  activeDrawerTab: 'logs',
  isDrawerExpanded: false,
  isDrawerFullscreen: false,
  chartInstance: null,
  datasetChartInstance: null,
  historicalRuns: [],
  inspectedRun: null,
  cachedStatusData: null,
  cachedTopologyData: null,
  statusTimer: null,
  logsTimer: null,
  clockTimer: null,
  serverTimeOffset: 0,
  visibility: (() => {
    try {
      const v = localStorage.getItem('ecg_fl_client_vis');
      if (v) return JSON.parse(v);
    } catch(e) {}
    return { global: true, vm2: true, vm3: true, vm4: true };
  })(),
  stratVisibility: (() => {
    try {
      const v = localStorage.getItem('ecg_fl_strat_vis');
      if (v) return JSON.parse(v);
    } catch(e) {}
    return {
      fedavg: true,
      fedprox: true,
      scaffold: true,
      fedbn: true,
      fedbb: true,
      fedsol: true
    };
  })()
};

// -------------------------------------------------------------
// 2. THEME CONTROLLER (DARK / LIGHT MODE)
// -------------------------------------------------------------
function applyTheme(theme) {
  state.theme = theme;
  localStorage.setItem('ecg_fl_theme', theme);
  const html = document.documentElement;
  const body = document.body;

  if (theme === 'light') {
    html.setAttribute('data-theme', 'light');
    body.className = 'theme-light';
  } else {
    html.setAttribute('data-theme', 'dark');
    body.className = 'theme-dark';
  }

  // Update Chart.js defaults for theme
  if (state.chartInstance) {
    updateChartTheme();
    state.chartInstance.update();
  }
}

function toggleTheme() {
  const newTheme = state.theme === 'dark' ? 'light' : 'dark';
  applyTheme(newTheme);
}

// -------------------------------------------------------------
// 3. SERVER CLOCK CONTROLLER
// -------------------------------------------------------------
function startClock() {
  const clockEl = document.getElementById('serverClockDisplay');
  function tick() {
    const now = new Date(Date.now() + state.serverTimeOffset);
    const hrs = String(now.getHours()).padStart(2, '0');
    const mins = String(now.getMinutes()).padStart(2, '0');
    const secs = String(now.getSeconds()).padStart(2, '0');
    if (clockEl) clockEl.textContent = `${hrs}:${mins}:${secs} WIB`;
  }
  tick();
  state.clockTimer = setInterval(tick, 1000);
}

// -------------------------------------------------------------
// 4. STRATEGY METADATA & MODULAR PARAMS
// -------------------------------------------------------------
const STRATEGY_DETAILS = {
  fedavg: {
    name: "FedAvg",
    badge: "Baseline Standar",
    desc: "Agregasi rata-rata bobot berbobot sampel $\\sum \\frac{N_k}{N} w_k$ standar McMahan et al."
  },
  fedprox: {
    name: "FedProx",
    badge: "Proximal Term",
    desc: "Menambahkan penalty norm kuadrat $\\frac{\\mu}{2} \\|w - w^t\\|^2$ pada loss lokal untuk meredam divergensi model antar-klien."
  },
  scaffold: {
    name: "SCAFFOLD",
    badge: "Control Variates",
    desc: "Koreksi varian kontrol klien $c_i$ dan server $c$ untuk mengoreksi client-drift pada data Non-IID heterogen."
  },
  fedbn: {
    name: "FedBN",
    badge: "Local BatchNorm",
    desc: "Parameter Batch Normalization diisolasi di setiap klien lokal untuk menangani Feature Shift tanpa pertukaran statistik mean & var."
  },
  fedbb: {
    name: "FedBB",
    badge: "Class Balance Regularization",
    desc: "Regularisasi Class Balance Loss (CBR) dengan bobot $\\alpha$ dan penyesuaian PNB untuk mengatasi ketidakseimbangan kelas ekstrem (Label Skew)."
  },
  fedsol: {
    name: "FedSOL",
    badge: "Subspace Orthogonal Drift",
    desc: "Proyeksi ortogonal pada subruang bobot penting untuk mencegah forgetting representasi lead EKG lokal."
  },
  all: {
    name: "Runtut (Sequential All)",
    badge: "Batch Execution",
    desc: "Menjalankan seluruh 6 metode FL secara berurutan otomatis (FedAvg -> FedProx -> SCAFFOLD -> FedBN -> FedBB -> FedSOL)."
  }
};

function updateModularParams(strategy) {
  const titleEl = document.getElementById('modularTitle');
  const badgeEl = document.getElementById('modularBadge');
  const bodyEl = document.getElementById('modularBody');
  const muGroup = document.getElementById('inputProximalMu')?.parentElement;

  const info = STRATEGY_DETAILS[strategy] || STRATEGY_DETAILS.fedbn;
  if (titleEl) titleEl.textContent = `Parameter Khusus: ${info.name}`;
  if (badgeEl) badgeEl.textContent = info.badge;
  if (bodyEl) bodyEl.textContent = info.desc;

  if (muGroup) {
    const label = muGroup.querySelector('label');
    if (strategy === 'fedprox') {
      muGroup.style.display = 'flex';
      if (label) label.textContent = 'Proximal Term (\u03BC)';
    } else if (strategy === 'fedbb') {
      muGroup.style.display = 'flex';
      if (label) label.textContent = 'CBR Alpha (\u03B1)';
    } else if (strategy === 'fedsol') {
      muGroup.style.display = 'flex';
      if (label) label.textContent = 'Subspace Drift (\u03BC)';
    } else {
      muGroup.style.display = 'flex';
      if (label) label.textContent = 'Reg Parameter (\u03BC)';
    }
  }
}

// -------------------------------------------------------------
// 5. RUN OVERVIEW & TELEMETRY UPDATES
// -------------------------------------------------------------
function updateRunOverview(data) {
  const isRunning = Boolean(data.is_running ?? (data.status === 'running') ?? state.isRunning);
  const curRound = data.current_round ?? 0;
  const totalRounds = data.total_rounds ?? 50;
  const activeStrat = (data.strategy || 'fedbn').toUpperCase();

  // Active Strat Badge
  const stratBadge = document.getElementById('overviewActiveStratBadge');
  if (stratBadge) stratBadge.textContent = activeStrat;

  // Round stat
  const roundVal = document.getElementById('overviewRoundVal');
  const roundSub = document.getElementById('overviewRoundSub');
  if (roundVal) {
    roundVal.textContent = isRunning ? `Ronde ${curRound} / ${totalRounds}` : (curRound > 0 ? `Selesai ${curRound}/${totalRounds}` : 'Standby');
  }
  if (roundSub) {
    roundSub.textContent = `Target: ${totalRounds} Ronde (${activeStrat})`;
  }

  // Elapsed vs ETA
  const timeVal = document.getElementById('overviewTimeVal');
  const etaSub = document.getElementById('overviewEtaSub');
  if (timeVal) {
    if (isRunning) {
      timeVal.textContent = data.elapsed_formatted || '00m 00s';
    } else {
      timeVal.textContent = curRound >= totalRounds && totalRounds > 0 ? 'Selesai' : '00m 00s';
    }
  }
  if (etaSub) {
    etaSub.textContent = isRunning ? (data.eta_formatted ? `Estimasi: ${data.eta_formatted}` : 'Menghitung ETA...') : 'Estimasi: Siap dimulai';
  }

  // Speed
  const speedVal = document.getElementById('overviewSpeedVal');
  if (speedVal) {
    speedVal.textContent = data.speed_formatted || (isRunning ? 'Mengukur...' : '--');
  }

  // Active Clients count
  const clientsVal = document.getElementById('overviewClientsVal');
  const connected = data.connected_clients ?? data.connected_count ?? 0;
  if (clientsVal) {
    clientsVal.textContent = `${connected} / 3`;
  }

  // Progress Bars
  const totalPct = totalRounds > 0 ? Math.min(100, Math.round((curRound / totalRounds) * 100)) : 0;
  const totalFill = document.getElementById('progressTotalFill');
  const totalText = document.getElementById('progressTotalPct');
  if (totalFill) totalFill.style.width = `${totalPct}%`;
  if (totalText) totalText.textContent = `${totalPct}% (${curRound}/${totalRounds} Rd)`;

  const batchPct = data.local_batch_pct ?? (isRunning ? 35 : 0);
  const batchFill = document.getElementById('progressBatchFill');
  const batchText = document.getElementById('progressBatchPct');
  if (batchFill) batchFill.style.width = `${batchPct}%`;
  if (batchText) {
    batchText.textContent = isRunning ? `${batchPct}% Batch Lokal` : '0% (Standby)';
  }

  // Visual Stepper Pipeline
  updatePipelineStepper(data.phase_index ?? (isRunning ? 2 : 0), data.phase ?? (isRunning ? 'Local Training on Clients' : 'Standby'));
}

function updatePipelineStepper(phaseIdx, phaseName) {
  const activeBadge = document.getElementById('activePhaseBadge');
  if (activeBadge) activeBadge.textContent = phaseName;

  const stepIds = ['stepBroadcast', 'stepLocalTrain', 'stepAggregate', 'stepEvaluation'];
  stepIds.forEach((id, idx) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (phaseIdx === 0) {
      el.classList.remove('is-active');
    } else if (idx + 1 === phaseIdx) {
      el.classList.add('is-active');
    } else {
      el.classList.remove('is-active');
    }
  });
}

// -------------------------------------------------------------
// 6. NODE EXPORTER GAUGES (ROW 2)
// -------------------------------------------------------------
function updateNodeExporterUI(data) {
  // Update Server Node
  const srvNode = data.server_node || {};
  const isServerRunning = Boolean(data.is_running ?? (data.status === 'running') ?? state.isRunning);
  const srvBadge = document.getElementById('badgeVm1');
  const srvCard = document.getElementById('exporterCardVm1');
  const srvTiming = document.getElementById('timingVm1');
  const srvRamStat = document.getElementById('serverRamStat');
  const srvRamFill = document.getElementById('serverRamFill');
  const srvCpuStat = document.getElementById('serverCpuStat');
  const srvCpuFill = document.getElementById('serverCpuFill');

  if (srvBadge) {
    srvBadge.className = `badge-status ${isServerRunning ? 'training' : 'online'}`;
    srvBadge.textContent = isServerRunning ? `Aktif (${(data.strategy || 'FL').toUpperCase()})` : 'Standby (Online)';
  }
  if (srvCard) {
    srvCard.className = `exporter-node-card is-server ${isServerRunning ? 'is-training' : 'is-online'}`;
  }
  if (srvTiming) {
    srvTiming.textContent = isServerRunning
      ? `Flower gRPC :8080 • Ronde ${data.current_round || 0}/${data.total_rounds || 50} (${(data.strategy || 'FL').toUpperCase()})`
      : 'Flower gRPC :8080 • Web UI :8501 • Standby';
  }

  const srvRamPct = srvNode.ram_percent ?? (isServerRunning ? 48.2 : 34.5);
  const srvRamGb = srvNode.ram_gb ?? ((srvRamPct * 8.0) / 100).toFixed(1);
  if (srvRamStat) srvRamStat.textContent = `${srvRamPct}% (${srvRamGb} GB / 8.0 GB)`;
  if (srvRamFill) srvRamFill.style.width = `${srvRamPct}%`;

  const srvCpuPct = srvNode.cpu_percent ?? (isServerRunning ? 28 : 6);
  if (srvCpuStat) srvCpuStat.textContent = `${srvCpuPct}% Core Load`;
  if (srvCpuFill) srvCpuFill.style.width = `${srvCpuPct}%`;

  // Update Clients (VM 2, VM 3, VM 4)
  const clients = data.clients || {};
  ['vm2', 'vm3', 'vm4'].forEach(k => {
    const info = clients[k];
    const cardEl = document.getElementById(`exporterCard${k.charAt(0).toUpperCase() + k.slice(1)}`);
    const badgeEl = document.getElementById(`badge${k.charAt(0).toUpperCase() + k.slice(1)}`);
    const timingEl = document.getElementById(`timing${k.charAt(0).toUpperCase() + k.slice(1)}`);
    const ramStatEl = document.getElementById(`ramStat${k.charAt(0).toUpperCase() + k.slice(1)}`);
    const ramFillEl = document.getElementById(`ramFill${k.charAt(0).toUpperCase() + k.slice(1)}`);
    const cpuStatEl = document.getElementById(`cpuStat${k.charAt(0).toUpperCase() + k.slice(1)}`);
    const cpuFillEl = document.getElementById(`cpuFill${k.charAt(0).toUpperCase() + k.slice(1)}`);

    if (!cardEl || !info) return;

    const st = info.status || 'offline';
    cardEl.className = `exporter-node-card is-client is-${st}`;

    if (badgeEl) {
      badgeEl.className = `badge-status ${st}`;
      badgeEl.textContent = info.status_label || st;
    }

    if (timingEl) {
      if (st === 'training') {
        timingEl.textContent = `R${info.current_round} • Ep ${info.epoch}/${info.total_epochs} • Batch ${info.batch}/${info.total_batches}`;
      } else if (st === 'online') {
        timingEl.textContent = 'Online (Standby gRPC Flower)';
      } else {
        timingEl.textContent = 'Node Offline';
      }
    }

    const ramPct = info.ram_percent ?? 0;
    const ramGb = info.ram_gb ?? ((ramPct * 8.0) / 100).toFixed(1);
    if (ramStatEl) ramStatEl.textContent = `${ramPct}% (${ramGb} GB / 8.0 GB)`;
    if (ramFillEl) ramFillEl.style.width = `${ramPct}%`;

    const cpuPct = info.cpu_percent ?? 0;
    if (cpuStatEl) cpuStatEl.textContent = `${cpuPct}% Load`;
    if (cpuFillEl) cpuFillEl.style.width = `${cpuPct}%`;
  });
}

// -------------------------------------------------------------
// 7. DATASET MODAL & 27-CLASS DISTRIBUTION CHART
// -------------------------------------------------------------
const DATASET_PROFILES = {
  vm2: {
    name: "PTB-XL (PhysioNet)",
    samples: "19,532",
    ip: "10.34.100.122",
    nonIID: "Feature Shift (Domain Shift)",
    desc: "Feature Shift timbul dari variasi perangkat perekam EKG klinik (Schiller, GE Healthcare), frekuensi sampling asli (100 Hz vs 500 Hz), dan variasi derau baseline wander serta interferensi jaringan listrik (powerline noise).",
    classes: [
      { code: "NSR", count: 9514, pct: 48.7 },
      { code: "STach", count: 1822, pct: 9.3 },
      { code: "SB", count: 1677, pct: 8.6 },
      { code: "AF", count: 1514, pct: 7.7 },
      { code: "IAVB", count: 1298, pct: 6.6 },
      { code: "PVC", count: 1142, pct: 5.8 },
      { code: "CRBBB", count: 950, pct: 4.9 },
      { code: "PAC", count: 782, pct: 4.0 },
      { code: "LAnFB", count: 142, pct: 0.7 },
      { code: "LPR", count: 105, pct: 0.5 },
      { code: "LQRSV", count: 88, pct: 0.5 },
      { code: "PR", count: 65, pct: 0.3 },
      { code: "SVPB", count: 42, pct: 0.2 }
    ],
    majority: [
      { name: "Normal Sinus Rhythm (NSR)", count: 9514, pct: "48.7%" },
      { name: "Sinus Tachycardia (STach)", count: 1822, pct: "9.3%" },
      { name: "Sinus Bradycardia (SB)", count: 1677, pct: "8.6%" },
      { name: "Atrial Fibrillation (AF)", count: 1514, pct: "7.7%" },
      { name: "1st Deg AV Block (IAVB)", count: 1298, pct: "6.6%" }
    ],
    minority: [
      { name: "Supraventricular PB (SVPB)", count: 42, pct: "0.2%" },
      { name: "Paced Rhythm (PR)", count: 65, pct: "0.3%" },
      { name: "Low QRS Voltage (LQRSV)", count: 88, pct: "0.5%" },
      { name: "Prolonged PR (LPR)", count: 105, pct: "0.5%" }
    ]
  },
  vm3: {
    name: "Chapman-Ningbo & Shaoxing",
    samples: "44,180",
    ip: "10.34.100.123",
    nonIID: "Label Skew (Extreme Imbalance)",
    desc: "Label Skew ditandai dengan ketidakseimbangan kelas patologis yang sangat tajam. Prevalensi Atrial Fibrilasi (AF) mencapai 20.2% (2.6x lipat lebih padat dari PTB-XL), dengan konsentrasi tinggi pada aritmia supraventrikular dan takikardia.",
    classes: [
      { code: "NSR", count: 18245, pct: 41.3 },
      { code: "AF", count: 8924, pct: 20.2 },
      { code: "SB", count: 6420, pct: 14.5 },
      { code: "STach", count: 4812, pct: 10.9 },
      { code: "PVC", count: 3210, pct: 7.3 },
      { code: "PAC", count: 2150, pct: 4.9 },
      { code: "IAVB", count: 1870, pct: 4.2 },
      { code: "TInv", count: 1320, pct: 3.0 },
      { code: "CRBBB", count: 380, pct: 0.9 },
      { code: "LBBB", count: 210, pct: 0.5 },
      { code: "SA", count: 140, pct: 0.3 },
      { code: "AFL", count: 115, pct: 0.3 }
    ],
    majority: [
      { name: "Normal Sinus Rhythm (NSR)", count: 18245, pct: "41.3%" },
      { name: "Atrial Fibrillation (AF)", count: 8924, pct: "20.2%" },
      { name: "Sinus Bradycardia (SB)", count: 6420, pct: "14.5%" },
      { name: "Sinus Tachycardia (STach)", count: 4812, pct: "10.9%" },
      { name: "Premature Ventricular (PVC)", count: 3210, pct: "7.3%" }
    ],
    minority: [
      { name: "Atrial Flutter (AFL)", count: 115, pct: "0.3%" },
      { name: "Sinus Arrest (SA)", count: 140, pct: "0.3%" },
      { name: "Left BBB (LBBB)", count: 210, pct: "0.5%" },
      { name: "Complete RBBB (CRBBB)", count: 380, pct: "0.9%" }
    ]
  },
  vm4: {
    name: "G12EC (CinC 2020 Challenge)",
    samples: "9,355",
    ip: "10.34.100.125",
    nonIID: "Duration & Sampling Skew",
    desc: "Duration Skew disebabkan oleh variasi temporal panjang rekaman EKG yang bervariasi antara 10 detik hingga 30 detik. Model federasi diuji ketahanannya terhadap sinyal yang dipotong (cropping) atau diperpanjang (zero-padding).",
    classes: [
      { code: "NSR", count: 3890, pct: 41.6 },
      { code: "Brady", count: 1420, pct: 15.2 },
      { code: "AF", count: 1120, pct: 12.0 },
      { code: "IAVB", count: 850, pct: 9.1 },
      { code: "CRBBB", count: 640, pct: 6.8 },
      { code: "STach", count: 530, pct: 5.7 },
      { code: "PVC", count: 410, pct: 4.4 },
      { code: "TAb", count: 320, pct: 3.4 },
      { code: "PR", count: 42, pct: 0.4 },
      { code: "NSIVCB", count: 35, pct: 0.4 },
      { code: "LQT", count: 24, pct: 0.3 },
      { code: "LQRSV", count: 18, pct: 0.2 }
    ],
    majority: [
      { name: "Normal Sinus Rhythm (NSR)", count: 3890, pct: "41.6%" },
      { name: "Bradycardia (Brady)", count: 1420, pct: "15.2%" },
      { name: "Atrial Fibrillation (AF)", count: 1120, pct: "12.0%" },
      { name: "1st Deg AV Block (IAVB)", count: 850, pct: "9.1%" },
      { name: "Complete RBBB (CRBBB)", count: 640, pct: "6.8%" }
    ],
    minority: [
      { name: "Low QRS Voltage (LQRSV)", count: 18, pct: "0.2%" },
      { name: "Long QT Syndrome (LQT)", count: 24, pct: "0.3%" },
      { name: "Non-specific IVCB (NSIVCB)", count: 35, pct: "0.4%" },
      { name: "Paced Rhythm (PR)", count: 42, pct: "0.4%" }
    ]
  }
};

function openDatasetModal(clientKey) {
  const modal = document.getElementById('datasetDetailModal');
  const prof = DATASET_PROFILES[clientKey] || DATASET_PROFILES.vm2;

  document.getElementById('dsMetaName').textContent = prof.name;
  document.getElementById('dsMetaSamples').textContent = `${prof.samples} Rekaman`;
  document.getElementById('dsMetaIp').textContent = prof.ip;
  document.getElementById('dsMetaNonIID').textContent = prof.nonIID;
  document.getElementById('dsNonIidDesc').textContent = prof.desc;

  // Render Majority List
  const majEl = document.getElementById('dsMajorityList');
  if (majEl) {
    majEl.innerHTML = prof.majority.map(m => `
      <div class="class-row-pill">
        <span>${m.name}</span>
        <strong class="mono highlight">${m.count.toLocaleString()} (${m.pct})</strong>
      </div>
    `).join('');
  }

  // Render Minority List
  const minEl = document.getElementById('dsMinorityList');
  if (minEl) {
    minEl.innerHTML = prof.minority.map(m => `
      <div class="class-row-pill">
        <span>${m.name}</span>
        <strong class="mono" style="color: #f59e0b;">${m.count.toLocaleString()} (${m.pct})</strong>
      </div>
    `).join('');
  }

  // Render Mini Bar Chart
  renderDatasetChart(prof.classes);

  if (modal) modal.style.display = 'flex';
}

function closeDatasetModal() {
  const modal = document.getElementById('datasetDetailModal');
  if (modal) modal.style.display = 'none';
}

function renderDatasetChart(classData) {
  const ctx = document.getElementById('datasetClassChart')?.getContext('2d');
  if (!ctx) return;

  if (state.datasetChartInstance) {
    state.datasetChartInstance.destroy();
  }

  const isDark = state.theme === 'dark';
  const textColor = isDark ? '#cbd5e1' : '#475569';
  const gridColor = isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.06)';

  state.datasetChartInstance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: classData.map(c => c.code),
      datasets: [{
        label: 'Frekuensi Rekaman',
        data: classData.map(c => c.count),
        backgroundColor: classData.map((_, i) => i < 3 ? 'rgba(59, 130, 246, 0.85)' : 'rgba(139, 92, 246, 0.65)'),
        borderColor: classData.map((_, i) => i < 3 ? '#3b82f6' : '#8b5cf6'),
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Jumlah: ${ctx.parsed.y.toLocaleString()} sampel`
          }
        }
      },
      scales: {
        x: {
          ticks: { color: textColor, font: { family: "'JetBrains Mono', monospace", size: 10 } },
          grid: { display: false }
        },
        y: {
          ticks: { color: textColor, font: { size: 10 } },
          grid: { color: gridColor }
        }
      }
    }
  });
}

// -------------------------------------------------------------
// 8. SYNCHRONIZED EVALUATION METRICS CHART (ROW 3)
// -------------------------------------------------------------
function getChartThemeColors() {
  const isDark = state.theme === 'dark';
  return {
    textColor: isDark ? '#cbd5e1' : '#334155',
    gridColor: isDark ? 'rgba(255, 255, 255, 0.07)' : 'rgba(0, 0, 0, 0.06)',
    globalColor: '#3b82f6',
    vm2Color: '#10b981',
    vm3Color: '#f59e0b',
    vm4Color: '#a855f7'
  };
}

function updateChartTheme() {
  if (!state.chartInstance) return;
  const colors = getChartThemeColors();
  state.chartInstance.options.scales.x.ticks.color = colors.textColor;
  state.chartInstance.options.scales.y.ticks.color = colors.textColor;
  state.chartInstance.options.scales.x.grid.color = colors.gridColor;
  state.chartInstance.options.scales.y.grid.color = colors.gridColor;
}

function initMetricsChart() {
  const ctx = document.getElementById('liveMetricChart')?.getContext('2d');
  if (!ctx) return;

  const colors = getChartThemeColors();

  state.chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: []
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            boxWidth: 12,
            font: { family: "'Inter', sans-serif", size: 11 },
            color: colors.textColor
          },
          onClick: (e, legendItem, legend) => {
            const index = legendItem.datasetIndex;
            const ci = legend.chart;
            const meta = ci.getDatasetMeta(index);
            const isHidden = meta.hidden === null ? !ci.data.datasets[index].hidden : !meta.hidden;
            meta.hidden = isHidden;
            ci.update();

            // Sync with strategy checklist if in comparison mode
            if (state.chartViewMode === 'comparison') {
              const label = (ci.data.datasets[index]?.label || '').toLowerCase();
              const stratKeys = ['fedavg', 'fedprox', 'scaffold', 'fedbn', 'fedbb', 'fedsol'];
              const matchedKey = stratKeys.find(k => label.includes(k));
              if (matchedKey) {
                state.stratVisibility[matchedKey] = !isHidden;
                const chk = document.getElementById(`chkStrat${matchedKey.charAt(0).toUpperCase() + matchedKey.slice(1)}`);
                if (chk) chk.checked = !isHidden;
                try {
                  localStorage.setItem('ecg_fl_strat_vis', JSON.stringify(state.stratVisibility));
                } catch(err) {}
              }
            } else {
              // Sync with per-client checklist if in clients mode
              const label = ci.data.datasets[index]?.label || '';
              if (label.includes('Global')) {
                state.visibility.global = !isHidden;
                const chk = document.getElementById('chkShowGlobal');
                if (chk) chk.checked = !isHidden;
              } else if (label.includes('PTB-XL') || label.includes('VM 2')) {
                state.visibility.vm2 = !isHidden;
                const chk = document.getElementById('chkShowVm2');
                if (chk) chk.checked = !isHidden;
              } else if (label.includes('Chapman') || label.includes('VM 3')) {
                state.visibility.vm3 = !isHidden;
                const chk = document.getElementById('chkShowVm3');
                if (chk) chk.checked = !isHidden;
              } else if (label.includes('G12EC') || label.includes('VM 4')) {
                state.visibility.vm4 = !isHidden;
                const chk = document.getElementById('chkShowVm4');
                if (chk) chk.checked = !isHidden;
              }
              try {
                localStorage.setItem('ecg_fl_client_vis', JSON.stringify(state.visibility));
              } catch(err) {}
            }
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.9)',
          titleFont: { family: "'Inter', sans-serif", weight: 'bold' },
          bodyFont: { family: "'JetBrains Mono', monospace" }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Ronde Federasi (Round)', color: colors.textColor, font: { size: 11, weight: '600' } },
          ticks: { color: colors.textColor, font: { size: 10 } },
          grid: { color: colors.gridColor }
        },
        y: {
          title: { display: true, text: 'Skor Metrik', color: colors.textColor, font: { size: 11, weight: '600' } },
          ticks: { color: colors.textColor, font: { size: 10 } },
          grid: { color: colors.gridColor }
        }
      }
    }
  });
}

function updateMetricsChart(metricsData, isUserAction = true) {
  if (!state.chartInstance) initMetricsChart();
  if (!state.chartInstance) return;

  state.chartInitializedOnce = true;

  const colors = getChartThemeColors();
  const metricKey = state.activeMetric; // 'f1', 'auroc', 'loss'

  // Helper metric extractor
  function getMetricArray(obj, mKey) {
    if (!obj) return [];
    if (mKey === 'f1') return obj.f1_macro || [];
    if (mKey === 'auroc') return obj.auroc || [];
    if (mKey === 'loss') return obj.val_loss || obj.loss || [];
    return [];
  }

  const datasets = [];

  // ==========================================
  // MODE 1: KOMPARASI 6 METODE FEDERATED LEARNING
  // ==========================================
  if (state.chartViewMode === 'comparison' && !state.inspectedRun) {
    const compData = state.cachedStatusData?.strategy_comparison || {};
    const strats = compData.strategies || {};
    const stratColors = {
      fedavg: '#2563eb',
      fedprox: '#f97316',
      scaffold: '#10b981',
      fedbn: '#8b5cf6',
      fedbb: '#ec4899',
      fedsol: '#06b6d4'
    };
    const stratLabels = {
      fedavg: 'FedAvg',
      fedprox: 'FedProx',
      scaffold: 'SCAFFOLD',
      fedbn: 'FedBN',
      fedbb: 'FedBB',
      fedsol: 'FedSOL'
    };

    let maxRounds = compData.rounds || [];
    if (!maxRounds.length) {
      Object.values(strats).forEach(s => {
        if ((s.rounds || []).length > maxRounds.length) maxRounds = s.rounds;
      });
    }
    state.chartInstance.data.labels = maxRounds.length ? maxRounds : [1];

    Object.entries(stratLabels).forEach(([sKey, sLabel]) => {
      const sObj = strats[sKey] || {};
      const sVals = getMetricArray(sObj, metricKey);
      datasets.push({
        label: `${sLabel} (${metricKey.toUpperCase()})`,
        data: sVals,
        borderColor: stratColors[sKey],
        backgroundColor: 'transparent',
        borderWidth: 2.4,
        tension: 0.25,
        pointRadius: 3,
        pointHoverRadius: 6,
        hidden: !state.stratVisibility[sKey]
      });
    });

  } else {
    // ==========================================
    // MODE 2: SESI TUNGGAL / PER-KLIEN CONVERGENCE
    // ==========================================
    const targetData = (state.inspectedRun ? convertRunToMetricsData(state.inspectedRun) : (metricsData || state.cachedStatusData?.metrics || {}));
    const rounds = targetData.rounds || [];
    state.chartInstance.data.labels = rounds.length ? rounds : [1];

    // 1. Global Model
    datasets.push({
      label: `Global Model (${metricKey.toUpperCase()})`,
      data: getMetricArray(targetData, metricKey),
      borderColor: colors.globalColor,
      backgroundColor: 'rgba(59, 130, 246, 0.1)',
      borderWidth: 2.5,
      tension: 0.25,
      pointRadius: 3,
      pointHoverRadius: 5,
      hidden: !state.visibility.global
    });

    // 2. Client 1 (PTB-XL)
    if (targetData.ptbxl) {
      datasets.push({
        label: 'VM 2: PTB-XL',
        data: getMetricArray(targetData.ptbxl, metricKey),
        borderColor: colors.vm2Color,
        borderWidth: 1.8,
        borderDash: [4, 3],
        tension: 0.25,
        pointRadius: 2,
        hidden: !state.visibility.vm2
      });
    }

    // 3. Client 2 (Chapman)
    if (targetData.chapman) {
      datasets.push({
        label: 'VM 3: Chapman',
        data: getMetricArray(targetData.chapman, metricKey),
        borderColor: colors.vm3Color,
        borderWidth: 1.8,
        borderDash: [4, 3],
        tension: 0.25,
        pointRadius: 2,
        hidden: !state.visibility.vm3
      });
    }

    // 4. Client 3 (G12EC)
    if (targetData.g12ec) {
      datasets.push({
        label: 'VM 4: G12EC',
        data: getMetricArray(targetData.g12ec, metricKey),
        borderColor: colors.vm4Color,
        borderWidth: 1.8,
        borderDash: [4, 3],
        tension: 0.25,
        pointRadius: 2,
        hidden: !state.visibility.vm4
      });
    }
  }

  state.chartInstance.data.datasets = datasets;
  updateChartTheme();
  
  if (isUserAction) {
    state.chartInstance.update();
  } else {
    state.chartInstance.update('none');
  }
}

// -------------------------------------------------------------
// 9. LEADERBOARD BENCHMARK TABLE
// -------------------------------------------------------------
function renderLeaderboardTable(benchmarkData) {
  const tbody = document.getElementById('benchmarkTableBody');
  if (!tbody || !benchmarkData) return;

  const entries = Object.entries(benchmarkData);
  if (!entries.length) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center text-muted" style="padding: 1.25rem;">Belum ada hasil benchmark tersedia.</td></tr>`;
    return;
  }

  tbody.innerHTML = entries.map(([key, item]) => {
    const bestF1Val = item.best_f1 ?? item.macro_f1 ?? item.f1_macro;
    const f1Str = bestF1Val != null ? Number(bestF1Val).toFixed(4) : '--';
    const aurocStr = item.auroc != null ? Number(item.auroc).toFixed(4) : '--';
    const lossVal = item.final_loss ?? item.val_loss;
    const lossStr = lossVal != null ? Number(lossVal).toFixed(4) : '--';
    const durVal = item.speed_sec ?? item.duration_per_round;
    const durStr = durVal != null && Number(durVal) > 0 ? `${Number(durVal).toFixed(1)}s` : '--';
    const roundsDone = item.rounds_done ?? item.completed_rounds;
    const roundsStr = roundsDone ? `${roundsDone} R` : (item.rounds_converged || '--');
    const isRunning = item.status === 'running';
    const hasBestModel = Boolean(item.best_f1 != null || item.run_id);

    return `
      <tr>
        <td><strong>${key.toUpperCase()}</strong></td>
        <td><span class="text-muted">${item.target_heterogeneity || item.target || item.mechanism || '--'}</span></td>
        <td class="mono font-semibold">${roundsStr}</td>
        <td class="mono" style="font-weight: 800; color: #3b82f6; font-size: 0.78rem;">${f1Str}</td>
        <td class="mono">${aurocStr}</td>
        <td class="mono">${lossStr}</td>
        <td class="mono">${durStr}</td>
        <td>
          <div style="display: flex; gap: 0.35rem; align-items: center;">
            <span class="badge-status ${isRunning ? 'training' : (item.status === 'completed' ? 'online' : (hasBestModel ? 'online' : 'offline'))}">
              ${isRunning ? 'Running' : (hasBestModel ? 'Best Run' : 'Standby')}
            </span>
            ${item.run_id ? `<button type="button" class="btn btn-secondary btn-xs btn-inspect-run" data-runid="${item.run_id}" title="Inspeksi kurva run terbaik metode ini">🔍</button>` : ''}
          </div>
        </td>
      </tr>
    `;
  }).join('');

  // Re-attach inspect click listeners for leaderboard buttons
  tbody.querySelectorAll('.btn-inspect-run').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const runId = e.currentTarget.dataset.runid;
      if (runId) inspectHistoricalRun(runId);
    });
  });
}

// -------------------------------------------------------------
// 10. EXPERIMENT HISTORY TABLE & ARTIFACTS
// -------------------------------------------------------------
async function fetchHistoricalRuns() {
  try {
    const res = await fetch('/api/runs');
    if (!res.ok) return;
    const data = await res.json();
    const runs = Array.isArray(data) ? data : (data.runs || []);
    state.historicalRuns = runs;
    populateRunSelector();
    renderHistoryTable();
  } catch (err) {
    console.error('Error fetching history:', err);
  }
}

function populateRunSelector() {
  const sel = document.getElementById('chartRunSelector');
  if (!sel) return;
  const currentVal = sel.value;
  let html = '<option value="latest">⚡ Sesi Terakhir (Live / Default)</option>';

  (state.historicalRuns || []).forEach(r => {
    const strat = (r.strategy || '').toUpperCase();
    const started = r.started_at ? new Date(r.started_at * 1000).toLocaleString('id-ID', {
      day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit'
    }) : '';
    const f1 = (r.best_f1 != null) ? `F1: ${Number(r.best_f1).toFixed(3)}` : '';
    const rCount = `${r.rounds_done ?? (r.rounds ? r.rounds.length : 0)}R`;
    const label = `${r.run_id} [${strat}] ${started ? `(${started})` : ''} • ${rCount} ${f1 ? `— ${f1}` : ''}`;
    html += `<option value="${r.run_id}">${label}</option>`;
  });

  sel.innerHTML = html;
  if (state.inspectedRun) {
    sel.value = state.inspectedRun.run_id;
  } else {
    sel.value = currentVal || 'latest';
  }
}

function renderHistoryTable() {
  const tbodyDrawer = document.getElementById('historyTableBody');
  const tbodyMain = document.getElementById('historyTableBodyMain');
  if (!tbodyDrawer && !tbodyMain) return;

  const stratFilter = (document.getElementById('filterHistoryStrategyMain')?.value
                    || document.getElementById('filterHistoryStrategy')?.value
                    || 'all');
  const statusFilter = (document.getElementById('filterHistoryStatusMain')?.value
                     || document.getElementById('filterHistoryStatus')?.value
                     || 'all');

  let list = state.historicalRuns || [];
  if (stratFilter !== 'all') {
    list = list.filter(r => (r.strategy || '').toLowerCase() === stratFilter.toLowerCase());
  }
  if (statusFilter !== 'all') {
    list = list.filter(r => (r.status || '').toLowerCase() === statusFilter.toLowerCase());
  }

  const emptyRow = `<tr><td colspan="9" class="text-center text-muted" style="padding: 1.5rem;">Tidak ada riwayat eksperimen yang sesuai kriteria.</td></tr>`;

  if (!list.length) {
    if (tbodyDrawer) tbodyDrawer.innerHTML = emptyRow;
    if (tbodyMain) tbodyMain.innerHTML = emptyRow;
    return;
  }

  const rowsHtml = list.map(r => {
    const rCount = r.rounds_done ?? (r.rounds ? r.rounds.length : 0);
    const totalRounds = r.config?.num_rounds || 50;
    const bestF1 = r.best_f1 != null ? Number(r.best_f1).toFixed(4) : '--';
    const finalLoss = r.final_loss != null ? Number(r.final_loss).toFixed(4) : '--';
    const auroc = r.final_auroc != null ? Number(r.final_auroc).toFixed(4) : '--';
    const stClass = r.status === 'completed' ? 'online' : (r.status === 'running' ? 'training' : 'offline');
    const started = r.started_at ? new Date(r.started_at * 1000).toLocaleString('id-ID') : '--';

    return `
      <tr>
        <td class="mono font-semibold">${r.run_id}</td>
        <td>${started}</td>
        <td><strong class="text-primary">${(r.strategy || '').toUpperCase()}</strong></td>
        <td class="mono">${rCount} / ${totalRounds}</td>
        <td><span class="badge-status ${stClass}">${r.status}</span></td>
        <td class="mono font-bold" style="color: #3b82f6;">${bestF1}</td>
        <td class="mono">${finalLoss}</td>
        <td class="mono">${auroc}</td>
        <td>
          <div style="display: flex; gap: 0.35rem; align-items: center;">
            <button type="button" class="btn btn-secondary btn-xs btn-inspect-run" data-runid="${r.run_id}" title="Inspeksi grafik konvergensi run ini">🔍 Kurva</button>
            <a class="btn btn-secondary btn-xs" href="/api/download-model?run=${r.run_id}&tag=best" title="Unduh Model Checkpoint (best.pt)">best.pt</a>
            <a class="btn btn-secondary btn-xs" href="/api/download-metrics?run=${r.run_id}" target="_blank" title="Unduh Log Metrik JSON">metrics.json</a>
          </div>
        </td>
      </tr>
    `;
  }).join('');

  if (tbodyDrawer) tbodyDrawer.innerHTML = rowsHtml;
  if (tbodyMain) tbodyMain.innerHTML = rowsHtml;

  // Attach event listeners to all inspect buttons in both tables
  document.querySelectorAll('.btn-inspect-run').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const runId = e.currentTarget.dataset.runid;
      if (runId) inspectHistoricalRun(runId);
    });
  });
}

function convertRunToMetricsData(run) {
  const rounds = [];
  const global = { f1_macro: [], auroc: [], val_loss: [] };
  const ptbxl = { f1_macro: [], auroc: [], val_loss: [] };
  const chapman = { f1_macro: [], auroc: [], val_loss: [] };
  const g12ec = { f1_macro: [], auroc: [], val_loss: [] };

  (run.rounds || []).forEach(rd => {
    rounds.push(rd.round);
    const g = rd.global || {};
    global.f1_macro.push(g.f1_macro ?? null);
    global.auroc.push(g.auroc ?? null);
    global.val_loss.push(g.val_loss ?? null);

    const c = rd.clients || {};
    if (c.ptbxl) {
      ptbxl.f1_macro.push(c.ptbxl.f1_macro ?? null);
      ptbxl.auroc.push(c.ptbxl.auroc ?? null);
      ptbxl.val_loss.push(c.ptbxl.val_loss ?? null);
    }
    if (c.chapman) {
      chapman.f1_macro.push(c.chapman.f1_macro ?? null);
      chapman.auroc.push(c.chapman.auroc ?? null);
      chapman.val_loss.push(c.chapman.val_loss ?? null);
    }
    if (c.g12ec) {
      g12ec.f1_macro.push(c.g12ec.f1_macro ?? null);
      g12ec.auroc.push(c.g12ec.auroc ?? null);
      g12ec.val_loss.push(c.g12ec.val_loss ?? null);
    }
  });

  return {
    rounds,
    f1_macro: global.f1_macro,
    auroc: global.auroc,
    val_loss: global.val_loss,
    ptbxl,
    chapman,
    g12ec
  };
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
    const sel = document.getElementById('chartRunSelector');
    if (sel) sel.value = runId;

    // Show inspect banner
    const banner = document.getElementById('inspectBanner');
    const runIdText = document.getElementById('inspectRunId');
    const detailsText = document.getElementById('inspectRunDetails');
    if (banner && runIdText) {
      runIdText.textContent = runId;
      if (detailsText) {
        const strat = (runData.strategy || '').toUpperCase();
        const rounds = (runData.rounds || []).length;
        const bestF1 = runData.models?.best?.f1_macro != null ? Number(runData.models.best.f1_macro).toFixed(4) : '-';
        detailsText.textContent = `• Strategi: ${strat} • ${rounds} Ronde Selesai • Best Macro-F1: ${bestF1}`;
      }
      banner.style.display = 'flex';
    }

    const metricsData = convertRunToMetricsData(runData);
    updateMetricsChart(metricsData);

    // Scroll smoothly to chart
    document.getElementById('cardMetricsPlots')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (err) {
    console.error('Error inspecting run:', err);
  }
}

function exitInspect() {
  state.inspectedRun = null;
  const banner = document.getElementById('inspectBanner');
  if (banner) banner.style.display = 'none';
  const sel = document.getElementById('chartRunSelector');
  if (sel) sel.value = 'latest';

  if (state.cachedStatusData?.metrics) {
    updateMetricsChart(state.cachedStatusData.metrics);
  }
}

// -------------------------------------------------------------
// 11. LIVE TERMINAL LOGS
// -------------------------------------------------------------
async function fetchLogs() {
  try {
    const res = await fetch(`/api/logs?source=${state.activeLogSource}`);
    if (!res.ok) return;
    const data = await res.json();
    const box = document.getElementById('logStreamBox');
    if (box && data.logs) {
      box.textContent = data.logs;
      const chk = document.getElementById('chkAutoScroll');
      if (chk && chk.checked) {
        box.scrollTop = box.scrollHeight;
      }
    }
  } catch (err) {
    console.error('Error fetching logs:', err);
  }
}

// -------------------------------------------------------------
// 12. BOTTOM DRAWER CONTROLLER (ROW 4)
// -------------------------------------------------------------
function toggleDrawer(forceExpand = null) {
  const drawer = document.getElementById('bottomDrawerPanel');
  const chevron = document.getElementById('drawerChevronIcon');
  if (!drawer) return;

  if (forceExpand !== null) {
    state.isDrawerExpanded = forceExpand;
  } else {
    state.isDrawerExpanded = !state.isDrawerExpanded;
  }

  if (state.isDrawerExpanded) {
    drawer.classList.remove('is-minimized');
    drawer.classList.add('is-expanded');
    if (chevron) chevron.textContent = '▼';
  } else {
    drawer.classList.remove('is-expanded', 'is-fullscreen');
    drawer.classList.add('is-minimized');
    state.isDrawerFullscreen = false;
    if (chevron) chevron.textContent = '▲';
  }
}

function switchDrawerTab(tabName) {
  state.activeDrawerTab = tabName;
  document.querySelectorAll('.drawer-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.drawer-tab-content').forEach(c => c.style.display = 'none');

  if (tabName === 'logs') {
    document.getElementById('tabDrawerLogs')?.classList.add('active');
    const logsContent = document.getElementById('contentDrawerLogs');
    if (logsContent) logsContent.style.display = 'flex';
  } else if (tabName === 'history') {
    document.getElementById('tabDrawerHistory')?.classList.add('active');
    const histContent = document.getElementById('contentDrawerHistory');
    if (histContent) histContent.style.display = 'flex';
    fetchHistoricalRuns();
  }

  if (!state.isDrawerExpanded) {
    toggleDrawer(true);
  }
}

// -------------------------------------------------------------
// 13. API ACTIONS (START, QUEUE, EMERGENCY STOP, RESET, EXPORT)
// -------------------------------------------------------------
async function startExperiment() {
  const form = document.getElementById('formExperimentControl');
  if (!form) return;

  const formData = new FormData(form);
  const strategy = formData.get('strategy') || 'fedbn';
  const num_rounds = parseInt(formData.get('num_rounds') || '50', 10);
  const local_epochs = parseInt(formData.get('local_epochs') || '3', 10);
  const batch_size = parseInt(formData.get('batch_size') || '32', 10);
  const learning_rate = parseFloat(formData.get('learning_rate') || '0.001');
  const proximal_mu = parseFloat(formData.get('proximal_mu') || '0.01');
  const seed = parseInt(formData.get('seed') || '42', 10);

  // Active client checkboxes
  const active_clients = [];
  if (document.getElementById('chkClientVm2')?.checked) active_clients.push('vm2');
  if (document.getElementById('chkClientVm3')?.checked) active_clients.push('vm3');
  if (document.getElementById('chkClientVm4')?.checked) active_clients.push('vm4');

  const payload = {
    strategy,
    num_rounds,
    local_epochs,
    batch_size,
    learning_rate,
    proximal_mu,
    seed,
    active_clients
  };

  try {
    const res = await fetch('/api/experiment/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (res.ok) {
      state.isRunning = true;
      toggleDrawer(true);
      switchDrawerTab('logs');
      fetchStatus();
    } else {
      alert(`Gagal memulai eksperimen: ${result.message || 'Error server'}`);
    }
  } catch (err) {
    console.error('Error starting experiment:', err);
    alert('Terjadi kesalahan jaringan saat memulai eksperimen.');
  }
}

async function queueExperiment() {
  const form = document.getElementById('formExperimentControl');
  if (!form) return;

  const formData = new FormData(form);
  const payload = {
    strategy: formData.get('strategy') || 'fedbn',
    num_rounds: parseInt(formData.get('num_rounds') || '50', 10),
    local_epochs: parseInt(formData.get('local_epochs') || '3', 10),
    batch_size: parseInt(formData.get('batch_size') || '32', 10),
    learning_rate: parseFloat(formData.get('learning_rate') || '0.001'),
    proximal_mu: parseFloat(formData.get('proximal_mu') || '0.01'),
    seed: parseInt(formData.get('seed') || '42', 10)
  };

  try {
    const res = await fetch('/api/experiment/queue', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    alert(`Eksperimen ${payload.strategy.toUpperCase()} (${payload.num_rounds} ronde) berhasil dimasukkan ke antrean! (Total antrean: ${result.queue_len})`);
  } catch (err) {
    console.error('Error queueing experiment:', err);
  }
}

async function emergencyStop() {
  if (!confirm('PERINGATAN: Apakah Anda yakin ingin melakukan EMERGENCY STOP? Proses Flower Server dan pelatihan lokal di seluruh VM klien akan dihentikan paksa.')) {
    return;
  }

  try {
    const res = await fetch('/api/experiment/stop', { method: 'POST' });
    if (res.ok) {
      state.isRunning = false;
      fetchStatus();
      fetchLogs();
      alert('Emergency Stop berhasil: Seluruh proses pelatihan telah dihentikan paksa.');
    }
  } catch (err) {
    console.error('Error in emergency stop:', err);
  }
}

async function resetExperiment() {
  if (!confirm('Reset konfigurasi sesi live dan membersihkan log?')) return;
  try {
    await fetch('/api/experiment/reset', { method: 'POST' });
    fetchStatus();
    fetchLogs();
  } catch (err) {
    console.error('Error resetting experiment:', err);
  }
}

function exportArtifacts() {
  window.location.href = '/api/export';
}

// -------------------------------------------------------------
// 14. MAIN STATUS POLLING LOOP
// -------------------------------------------------------------
async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) {
      updateGlobalConnectionBadge('error');
      return;
    }
    const data = await res.json();
    state.cachedStatusData = data;

    state.isRunning = Boolean(data.is_running ?? (data.status === 'running'));
    updateGlobalConnectionBadge(state.isRunning ? 'busy' : 'online');
    updateRunOverview(data);
    updateNodeExporterUI(data);

    if (!state.inspectedRun && data.metrics) {
      if (state.chartAutoRefresh && (state.isRunning || !state.chartInitializedOnce)) {
        updateMetricsChart(data.metrics, false);
      }
    }
    if (data.benchmark) {
      renderLeaderboardTable(data.benchmark);
    }

    const btnStart = document.getElementById('btnStartTraining');
    if (btnStart) btnStart.disabled = state.isRunning;
  } catch (err) {
    console.error('Error fetching status:', err);
    updateGlobalConnectionBadge('error');
  }
}

function updateGlobalConnectionBadge(st) {
  const dot = document.getElementById('grpcStatusDot');
  const txt = document.getElementById('grpcStatusText');
  if (!dot || !txt) return;

  if (st === 'busy') {
    dot.className = 'status-dot dot-busy';
    txt.textContent = 'gRPC: Pelatihan Berjalan (Busy)';
  } else if (st === 'online') {
    dot.className = 'status-dot dot-online';
    txt.textContent = 'gRPC: Online (Standby)';
  } else {
    dot.className = 'status-dot dot-error';
    txt.textContent = 'gRPC: Terputus / Error';
  }
}

// -------------------------------------------------------------
// 15. INITIALIZATION & EVENT LISTENERS
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  // Apply Saved Theme
  applyTheme(state.theme);

  // Start Live Server Clock
  startClock();

  // Theme Toggle Button
  document.getElementById('btnThemeToggle')?.addEventListener('click', toggleTheme);

  // Global Actions
  document.getElementById('btnEmergencyStop')?.addEventListener('click', emergencyStop);
  document.getElementById('btnExportArtifacts')?.addEventListener('click', exportArtifacts);

  // Run Control Form
  document.getElementById('btnStartTraining')?.addEventListener('click', startExperiment);
  document.getElementById('btnQueueRun')?.addEventListener('click', queueExperiment);
  document.getElementById('btnResetExperiment')?.addEventListener('click', resetExperiment);

  // Strategy Selector Radios
  document.querySelectorAll('input[name="strategy"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
      state.activeStrategy = e.target.value;
      updateModularParams(state.activeStrategy);
    });
  });
  updateModularParams(state.activeStrategy);

  // Expand Dataset Buttons (Modal)
  document.querySelectorAll('.btn-expand-dataset').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const clientKey = e.currentTarget.dataset.client;
      openDatasetModal(clientKey);
    });
  });
  document.getElementById('btnDsModalClose')?.addEventListener('click', closeDatasetModal);
  document.getElementById('datasetDetailModal')?.addEventListener('click', (e) => {
    if (e.target.id === 'datasetDetailModal') closeDatasetModal();
  });

  // Chart Run Selector (Pilih Sesi Live atau Arsip Lampau)
  document.getElementById('chartRunSelector')?.addEventListener('change', (e) => {
    if (e.target.value === 'latest') {
      exitInspect();
    } else {
      inspectHistoricalRun(e.target.value);
    }
  });

  // Exit Inspect Button
  document.getElementById('btnExitInspect')?.addEventListener('click', exitInspect);

  // Chart View Mode Tabs: Komparasi 6 Metode vs Sesi / Per-Klien
  document.querySelectorAll('#chartViewModeTabs .seg-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('#chartViewModeTabs .seg-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      state.chartViewMode = e.target.dataset.viewmode;

      const stripStrats = document.getElementById('stripStrategiesToggles');
      const stripClients = document.getElementById('stripClientsToggles');
      const selBox = document.getElementById('chartRunSelectorBox');
      const subtitle = document.getElementById('metricPlotSubtitle');

      if (state.chartViewMode === 'comparison') {
        if (stripStrats) stripStrats.style.display = 'flex';
        if (stripClients) stripClients.style.display = 'none';
        if (selBox) selBox.style.display = 'none';
        if (subtitle) subtitle.textContent = 'Perbandingan performa konvergensi global 6 metode FL (FedAvg, FedProx, SCAFFOLD, FedBN, FedBB, FedSOL)';
        if (state.inspectedRun) exitInspect();
      } else {
        if (stripStrats) stripStrats.style.display = 'none';
        if (stripClients) stripClients.style.display = 'flex';
        if (selBox) selBox.style.display = 'inline-flex';
        if (subtitle) subtitle.textContent = 'Performa model global dan per-client convergence untuk deteksi drift';
      }
      syncChartCheckboxes();
      updateMetricsChart(null, true);
    });
  });

  // Auto-Refresh Toggle Button for Chart
  document.getElementById('btnToggleChartAutoRefresh')?.addEventListener('click', () => {
    state.chartAutoRefresh = !state.chartAutoRefresh;
    const dot = document.getElementById('dotChartAutoRefresh');
    const txt = document.getElementById('textChartAutoRefresh');
    if (state.chartAutoRefresh) {
      if (dot) dot.className = 'status-dot dot-online';
      if (txt) txt.textContent = 'Auto-Update: Aktif';
      if (state.cachedStatusData?.metrics) {
        updateMetricsChart(state.cachedStatusData.metrics, true);
      }
    } else {
      if (dot) dot.className = 'status-dot dot-busy';
      if (txt) txt.textContent = 'Auto-Update: Dijeda';
    }
  });

  // 6 Strategies Comparison Checkbox Toggles
  const mapStrats = {
    chkStratFedavg: 'fedavg',
    chkStratFedprox: 'fedprox',
    chkStratScaffold: 'scaffold',
    chkStratFedbn: 'fedbn',
    chkStratFedbb: 'fedbb',
    chkStratFedsol: 'fedsol'
  };
  Object.entries(mapStrats).forEach(([chkId, key]) => {
    document.getElementById(chkId)?.addEventListener('change', (e) => {
      state.stratVisibility[key] = e.target.checked;
      try {
        localStorage.setItem('ecg_fl_strat_vis', JSON.stringify(state.stratVisibility));
      } catch (err) {}
      updateMetricsChart(null, true);
    });
  });

  // Chart Metric Tabs
  document.querySelectorAll('#chartMetricTabs .seg-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('#chartMetricTabs .seg-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      state.activeMetric = e.target.dataset.metric;
      updateMetricsChart(null, true);
    });
  });

  // Per-Client Convergence Toggles
  const mapConv = {
    chkShowGlobal: 'global',
    chkShowVm2: 'vm2',
    chkShowVm3: 'vm3',
    chkShowVm4: 'vm4'
  };
  Object.entries(mapConv).forEach(([chkId, key]) => {
    document.getElementById(chkId)?.addEventListener('change', (e) => {
      state.visibility[key] = e.target.checked;
      try {
        localStorage.setItem('ecg_fl_client_vis', JSON.stringify(state.visibility));
      } catch (err) {}
      updateMetricsChart(null, true);
    });
  });

  // Leaderboard Refresh
  document.getElementById('btnRefreshBenchmark')?.addEventListener('click', () => {
    fetchStatus();
  });

  // History Filter Dropdowns & Refresh (Main Card & Drawer)
  const syncFilter = (srcId, destId) => {
    document.getElementById(srcId)?.addEventListener('change', (e) => {
      const dest = document.getElementById(destId);
      if (dest) dest.value = e.target.value;
      renderHistoryTable();
    });
  };
  syncFilter('filterHistoryStrategy', 'filterHistoryStrategyMain');
  syncFilter('filterHistoryStatus', 'filterHistoryStatusMain');
  syncFilter('filterHistoryStrategyMain', 'filterHistoryStrategy');
  syncFilter('filterHistoryStatusMain', 'filterHistoryStatus');

  document.getElementById('btnRefreshHistory')?.addEventListener('click', fetchHistoricalRuns);
  document.getElementById('btnRefreshHistoryMain')?.addEventListener('click', fetchHistoricalRuns);

  // Bottom Drawer Controls
  document.getElementById('drawerHeaderToggle')?.addEventListener('click', (e) => {
    if (e.target.closest('.drawer-tabs-wrapper') || e.target.closest('.drawer-actions-right')) return;
    toggleDrawer();
  });

  document.getElementById('tabDrawerLogs')?.addEventListener('click', () => switchDrawerTab('logs'));
  document.getElementById('tabDrawerHistory')?.addEventListener('click', () => switchDrawerTab('history'));

  document.getElementById('btnToggleDrawerSize')?.addEventListener('click', () => {
    const drawer = document.getElementById('bottomDrawerPanel');
    if (!drawer) return;
    state.isDrawerFullscreen = !state.isDrawerFullscreen;
    if (state.isDrawerFullscreen) {
      drawer.classList.remove('is-minimized', 'is-expanded');
      drawer.classList.add('is-fullscreen');
      document.getElementById('btnToggleDrawerSize').textContent = 'Normal';
    } else {
      drawer.classList.remove('is-fullscreen');
      drawer.classList.add('is-expanded');
      document.getElementById('btnToggleDrawerSize').textContent = 'Layar Penuh';
    }
  });

  // Terminal Controls
  document.querySelectorAll('#logSourceTabs .seg-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('#logSourceTabs .seg-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      state.activeLogSource = e.target.dataset.source;
      fetchLogs();
    });
  });

  document.getElementById('btnClearLog')?.addEventListener('click', () => {
    const box = document.getElementById('logStreamBox');
    if (box) box.textContent = 'Log dibersihkan oleh pengguna.';
  });

  document.getElementById('btnDownloadModel')?.addEventListener('click', () => {
    window.location.href = '/api/download-model?tag=best';
  });

  function syncChartCheckboxes() {
    const mapStrats = {
      chkStratFedavg: 'fedavg',
      chkStratFedprox: 'fedprox',
      chkStratScaffold: 'scaffold',
      chkStratFedbn: 'fedbn',
      chkStratFedbb: 'fedbb',
      chkStratFedsol: 'fedsol'
    };
    Object.entries(mapStrats).forEach(([chkId, key]) => {
      const el = document.getElementById(chkId);
      if (el) el.checked = Boolean(state.stratVisibility[key]);
    });

    const mapConv = {
      chkShowGlobal: 'global',
      chkShowVm2: 'vm2',
      chkShowVm3: 'vm3',
      chkShowVm4: 'vm4'
    };
    Object.entries(mapConv).forEach(([chkId, key]) => {
      const el = document.getElementById(chkId);
      if (el) el.checked = Boolean(state.visibility[key]);
    });
  }

  // Sync saved checkbox states to DOM
  syncChartCheckboxes();

  // Initialize Chart
  initMetricsChart();

  // Initial Fetches
  fetchStatus();
  fetchLogs();
  fetchHistoricalRuns();

  // Set Timers
  state.statusTimer = setInterval(fetchStatus, 2500);
  state.logsTimer = setInterval(fetchLogs, 2500);
});
