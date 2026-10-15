/**
 * CardioPulse FL - Light Theme Experiment Control & Live Monitor
 * Manages Start/Stop commands, live metric plots, client status, and log streaming.
 */

// Application State
const appState = {
  isRunning: false,
  activeMetric: 'loss',
  activeLogSource: 'server',
  currentRound: 0,
  totalRounds: 50,
  metricsHistory: {
    rounds: [],
    loss: [],
    val_loss: [],
    f1_macro: [],
    auroc: []
  }
};

let metricChart = null;

// Initialize Chart.js
function initChart() {
  const ctx = document.getElementById('liveMetricChart');
  if (!ctx) return;

  metricChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Global (Agregat)',
          data: [],
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.08)',
          borderWidth: 2.5,
          tension: 0.3,
          fill: true
        },
        {
          label: 'PTB-XL (VM 2)',
          data: [],
          borderColor: '#6366f1',
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [3, 3],
          tension: 0.3
        },
        {
          label: 'Chapman (VM 3)',
          data: [],
          borderColor: '#f97316',
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 3],
          tension: 0.3
        },
        {
          label: 'G12EC (VM 4)',
          data: [],
          borderColor: '#06b6d4',
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [6, 4],
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { font: { family: 'Inter', size: 11, weight: '600' }, boxWidth: 14, padding: 12 }
        },
        tooltip: {
          backgroundColor: '#0f172a',
          titleFont: { family: 'Inter' },
          bodyFont: { family: 'JetBrains Mono' }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Ronde Komunikasi (Round)', color: '#64748b' },
          grid: { color: '#f1f5f9' },
          ticks: { color: '#64748b' }
        },
        y: {
          title: { display: true, text: 'BCE Loss', color: '#64748b' },
          grid: { color: '#f1f5f9' },
          ticks: { color: '#64748b' }
        }
      }
    }
  });
}

// Switch Metric Tab (Loss / F1 / AUROC) dengan dukungan multi-klien (Global + PTB-XL + Chapman + G12EC)
function switchMetricTab(metricName) {
  appState.activeMetric = metricName;
  document.querySelectorAll('.chart-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.metric === metricName);
  });

  if (!metricChart) return;

  const h = appState.metricsHistory || {};
  const rounds = h.rounds || [];

  const getClientSeries = (clientKey, metric, fallback) => {
    if (h[clientKey] && Array.isArray(h[clientKey][metric]) && h[clientKey][metric].length > 0) {
      return h[clientKey][metric];
    }
    return fallback || [];
  };

  if (metricName === 'loss') {
    metricChart.data.labels = rounds;
    metricChart.data.datasets = [
      {
        label: 'Global (Agregat)',
        data: h.loss || [],
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37, 99, 235, 0.08)',
        borderWidth: 2.5,
        tension: 0.3,
        fill: true
      },
      {
        label: 'PTB-XL (VM 2)',
        data: getClientSeries('ptbxl', 'loss', h.loss),
        borderColor: '#6366f1',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [3, 3],
        tension: 0.3
      },
      {
        label: 'Chapman (VM 3)',
        data: getClientSeries('chapman', 'loss', h.loss),
        borderColor: '#f97316',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [5, 3],
        tension: 0.3
      },
      {
        label: 'G12EC (VM 4)',
        data: getClientSeries('g12ec', 'loss', h.loss),
        borderColor: '#06b6d4',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [6, 4],
        tension: 0.3
      }
    ];
    metricChart.options.scales.y.title.text = 'BCE Loss';
  } else if (metricName === 'f1') {
    metricChart.data.labels = rounds;
    metricChart.data.datasets = [
      {
        label: 'Global F1-Macro',
        data: h.f1_macro || [],
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37, 99, 235, 0.08)',
        borderWidth: 2.5,
        tension: 0.3,
        fill: true
      },
      {
        label: 'PTB-XL F1 (VM 2)',
        data: getClientSeries('ptbxl', 'f1_macro', h.f1_macro),
        borderColor: '#6366f1',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [3, 3],
        tension: 0.3
      },
      {
        label: 'Chapman F1 (VM 3)',
        data: getClientSeries('chapman', 'f1_macro', h.f1_macro),
        borderColor: '#f97316',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [5, 3],
        tension: 0.3
      },
      {
        label: 'G12EC F1 (VM 4)',
        data: getClientSeries('g12ec', 'f1_macro', h.f1_macro),
        borderColor: '#06b6d4',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [6, 4],
        tension: 0.3
      }
    ];
    metricChart.options.scales.y.title.text = 'Macro-F1 (0.0 - 1.0)';
  } else if (metricName === 'auroc') {
    metricChart.data.labels = rounds;
    metricChart.data.datasets = [
      {
        label: 'Global AUROC',
        data: h.auroc || [],
        borderColor: '#d97706',
        backgroundColor: 'rgba(217, 119, 6, 0.08)',
        borderWidth: 2.5,
        tension: 0.3,
        fill: true
      },
      {
        label: 'PTB-XL AUROC (VM 2)',
        data: getClientSeries('ptbxl', 'auroc', h.auroc),
        borderColor: '#6366f1',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [3, 3],
        tension: 0.3
      },
      {
        label: 'Chapman AUROC (VM 3)',
        data: getClientSeries('chapman', 'auroc', h.auroc),
        borderColor: '#f97316',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [5, 3],
        tension: 0.3
      },
      {
        label: 'G12EC AUROC (VM 4)',
        data: getClientSeries('g12ec', 'auroc', h.auroc),
        borderColor: '#06b6d4',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [6, 4],
        tension: 0.3
      }
    ];
    metricChart.options.scales.y.title.text = 'Macro-AUROC (0.5 - 1.0)';
  }

  metricChart.update();
}

// UI Event Handlers
function setupEventListeners() {
  // Chart Tabs
  document.querySelectorAll('.chart-tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      switchMetricTab(e.target.dataset.metric);
    });
  });

  // Start Training Button
  const btnStart = document.getElementById('btnStartTraining');
  btnStart.addEventListener('click', async () => {
    const form = document.getElementById('formExperimentControl');
    const formData = new FormData(form);

    const payload = {
      strategy: formData.get('strategy'),
      model_arch: formData.get('model_arch'),
      num_rounds: parseInt(formData.get('num_rounds')),
      local_epochs: parseInt(formData.get('local_epochs')),
      learning_rate: parseFloat(formData.get('learning_rate')),
      batch_size: parseInt(formData.get('batch_size'))
    };

    btnStart.disabled = true;
    btnStart.textContent = 'Memulai Server...';

    try {
      const res = await fetch('/api/experiment/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (res.ok) {
        setServerRunningState(true);
        appState.totalRounds = payload.num_rounds;
      } else {
        alert(`Gagal memulai: ${data.message || 'Error server'}`);
        btnStart.disabled = false;
        btnStart.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start Training`;
      }
    } catch (err) {
      alert(`Koneksi error: ${err.message}`);
      btnStart.disabled = false;
      btnStart.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start Training`;
    }
  });

  // Stop Training Button
  const btnStop = document.getElementById('btnStopTraining');
  btnStop.addEventListener('click', async () => {
    if (!confirm('Apakah Anda yakin ingin menghentikan sesi pelatihan yang sedang berjalan?')) return;

    btnStop.disabled = true;
    btnStop.textContent = 'Menghentikan...';

    try {
      await fetch('/api/experiment/stop', { method: 'POST' });
      setServerRunningState(false);
    } catch (err) {
      alert(`Error menghentikan: ${err.message}`);
    }
  });

  // Reset Button
  const btnReset = document.getElementById('btnResetExperiment');
  btnReset.addEventListener('click', async () => {
    if (appState.isRunning) {
      const confirmReset = confirm('Pelatihan sedang berjalan. Apakah Anda yakin ingin menghentikan dan mereset seluruh sesi & UI?');
      if (!confirmReset) return;
    }

    btnReset.disabled = true;
    btnReset.textContent = 'Mereset...';

    try {
      await fetch('/api/experiment/reset', { method: 'POST' });
    } catch (e) {
      console.warn('Reset request error:', e);
    }

    // 1. Reset State
    setServerRunningState(false);
    appState.currentRound = 0;
    appState.metricsHistory = {
      rounds: [],
      loss: [],
      val_loss: [],
      f1_macro: [],
      auroc: [],
      ptbxl: { loss: [], f1_macro: [], auroc: [] },
      chapman: { loss: [], f1_macro: [], auroc: [] },
      g12ec: { loss: [], f1_macro: [], auroc: [] }
    };

    // 2. Reset Chart Visuals
    if (metricChart) {
      metricChart.data.labels = [];
      metricChart.data.datasets.forEach(ds => {
        ds.data = [];
      });
      metricChart.update();
    }

    // 3. Reset Progress Bar
    updateProgressBar(0, appState.totalRounds);
    const roundText = document.getElementById('progressRoundText');
    const percentText = document.getElementById('progressPercentText');
    const barFill = document.getElementById('progressBarFill');
    if (roundText) roundText.textContent = 'Ronde Federasi: Menunggu Memulai...';
    if (percentText) percentText.textContent = '0%';
    if (barFill) barFill.style.width = '0%';

    // 4. Reset Live ETA & Speed Text
    const etaEl = document.getElementById('progressEtaText');
    const speedEl = document.getElementById('progressSpeedText');
    if (etaEl) etaEl.textContent = 'Siap dimulai';
    if (speedEl) speedEl.textContent = 'Laju: --';

    // 5. Reset Timing pada Tiap Kartu Node Klien
    ['timingVm2', 'timingVm3', 'timingVm4'].forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.textContent = 'Menunggu Ronde 1...';
        el.style.color = '#64748b';
      }
    });

    // 6. Reset Log Box & Button Text
    const logBox = document.getElementById('logStreamBox');
    if (logBox) logBox.textContent = 'Sesi telah di-reset bersih. Siap memulai eksperimen baru.';

    btnReset.disabled = false;
    btnReset.textContent = 'Reset';
  });

  // Download Model Checkpoint
  const btnDownload = document.getElementById('btnDownloadModel');
  btnDownload.addEventListener('click', () => {
    window.location.href = '/api/download-model';
  });

  // Log Source Tabs (Server vs VM2 vs VM3 vs VM4 vs Semua)
  const logTabs = document.querySelectorAll('.log-tab-btn');
  logTabs.forEach(btn => {
    btn.addEventListener('click', () => {
      logTabs.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      appState.activeLogSource = btn.dataset.source;
      const box = document.getElementById('logStreamBox');
      if (box) box.textContent = `Memuat log dari [${btn.textContent}]...`;
      pollLogs();
    });
  });
}

function setServerRunningState(isRunning) {
  appState.isRunning = isRunning;
  const btnStart = document.getElementById('btnStartTraining');
  const btnStop = document.getElementById('btnStopTraining');
  const statusDot = document.getElementById('serverStatusDot');
  const statusText = document.getElementById('serverStatusText');

  if (isRunning) {
    btnStart.disabled = true;
    btnStart.textContent = 'Training Berjalan...';
    btnStop.disabled = false;
    btnStop.textContent = 'Stop / Abort';
    statusDot.className = 'dot running';
    statusText.textContent = 'Server: Training Active';
  } else {
    btnStart.disabled = false;
    btnStart.innerHTML = `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg> Start Training`;
    btnStop.disabled = true;
    btnStop.textContent = 'Stop / Abort';
    statusDot.className = 'dot idle';
    statusText.textContent = 'Server: Idle';
  }
}

function updateClientStatuses(clients, connectedCount, totalClients) {
  const summaryEl = document.getElementById('clientsConnectedSummary');
  if (summaryEl) {
    summaryEl.textContent = `${connectedCount} / ${totalClients} Klien Terhubung`;
    if (connectedCount === totalClients) {
      summaryEl.className = 'badge-status online';
    } else if (connectedCount > 0) {
      summaryEl.className = 'badge-status idle';
    } else {
      summaryEl.className = 'badge-status offline';
    }
  }

  const clientMap = {
    'vm2': { badge: 'badgeVm2', card: 'clientNodeVm2', timing: 'timingVm2' },
    'vm3': { badge: 'badgeVm3', card: 'clientNodeVm3', timing: 'timingVm3' },
    'vm4': { badge: 'badgeVm4', card: 'clientNodeVm4', timing: 'timingVm4' }
  };

  for (const [key, els] of Object.entries(clientMap)) {
    const badge = document.getElementById(els.badge);
    const card = document.getElementById(els.card);
    const timing = document.getElementById(els.timing);
    const client = clients ? clients[key] : null;

    if (!client || client.status === 'offline') {
      if (badge) {
        badge.className = 'badge-status offline';
        badge.textContent = 'Offline';
      }
      if (card) {
        card.classList.add('is-offline');
        card.classList.remove('is-training');
      }
      if (timing) {
        timing.textContent = 'Offline';
        timing.style.color = '#94a3b8';
      }
    } else if (client.status === 'training' && appState.isRunning) {
      if (badge) {
        badge.className = 'badge-status training';
        badge.textContent = client.status_label || 'Training';
      }
      if (card) {
        card.classList.remove('is-offline');
        card.classList.add('is-training');
      }
      if (timing) {
        const elapsed = Math.round(client.elapsed_in_round || 0);
        timing.innerHTML = `<span style="color:#2563eb;font-weight:600;">Sedang berjalan: ${elapsed}s...</span>`;
      }
    } else {
      if (badge) {
        badge.className = 'badge-status online';
        badge.textContent = appState.isRunning ? (client.status_label || 'Online') : 'Online (Standby)';
      }
      if (card) {
        card.classList.remove('is-offline');
        card.classList.remove('is-training');
      }
      if (timing) {
        if (appState.isRunning && client.measured_duration && client.measured_duration > 0) {
          const s = Math.round(client.measured_duration);
          const m = (client.measured_duration / 60).toFixed(1);
          timing.innerHTML = `Terukur: <span style="color:#059669;font-weight:700;">${s}s</span> (${m} mnt/ronde)`;
        } else {
          timing.textContent = 'Menunggu Ronde 1...';
          timing.style.color = '#64748b';
        }
      }
    }
  }
}

function updateProgressBar(current, total) {
  const percent = total > 0 ? Math.min(100, Math.round((current / total) * 100)) : 0;
  const fill = document.getElementById('progressBarFill');
  const txtRound = document.getElementById('progressRoundText');
  const txtPercent = document.getElementById('progressPercentText');

  if (fill) fill.style.width = `${percent}%`;
  if (txtPercent) txtPercent.textContent = `${percent}%`;
  if (txtRound) {
    txtRound.textContent = current > 0 
      ? `Ronde Federasi: ${current} dari ${total}` 
      : 'Ronde Federasi: Menunggu Memulai...';
  }
}

// Polling Status & Metrics from Backend
async function pollBackend() {
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      const data = await res.json();
      const isServerActive = (data.status === 'running' || data.status === 'training');
      
      if (isServerActive !== appState.isRunning) {
        setServerRunningState(isServerActive);
      }

      // Update real status of each client
      if (data.clients) {
        updateClientStatuses(data.clients, data.connected_count || 0, data.total_clients || 3);
      }

      if (data.current_round !== undefined) {
        appState.currentRound = data.current_round;
        appState.totalRounds = data.total_rounds || appState.totalRounds;
        updateProgressBar(appState.currentRound, appState.totalRounds);
      }

      // Update Benchmark Comparison Table
      if (data.benchmark) {
        renderBenchmarkTable(data.benchmark, data.strategy || 'fedbn');
      }

      // Update Live ETA & Speed
      const etaEl = document.getElementById('progressEtaText');
      const speedEl = document.getElementById('progressSpeedText');
      if (etaEl && data.eta_formatted) {
        etaEl.textContent = data.eta_formatted;
      }
      if (speedEl && data.speed_formatted) {
        speedEl.textContent = `Laju: ${data.speed_formatted}`;
      }

      // Update metrics if available
      if (data.metrics && data.metrics.rounds && data.metrics.rounds.length > 0) {
        appState.metricsHistory = data.metrics;
        switchMetricTab(appState.activeMetric);
      }
    }
  } catch (err) {
    // Offline or server temporarily busy
  }
}

// Render Benchmark Comparison Table (FedBN vs FedBB vs FedSOL)
function renderBenchmarkTable(benchmark, activeStrategy) {
  const tbody = document.getElementById('benchmarkTableBody');
  const badge = document.getElementById('benchmarkActiveBadge');
  if (!tbody || !benchmark) return;

  if (badge) {
    badge.textContent = `Aktif: ${(activeStrategy || 'FEDBN').toUpperCase()}`;
  }

  const stratKeys = ['fedbn', 'fedbb', 'fedsol'];
  
  // Hitung best score untuk highlight visual
  let maxF1 = 0;
  let maxAuroc = 0;
  stratKeys.forEach(k => {
    if (benchmark[k]) {
      if (benchmark[k].macro_f1 > maxF1) maxF1 = benchmark[k].macro_f1;
      if (benchmark[k].auroc > maxAuroc) maxAuroc = benchmark[k].auroc;
    }
  });

  let html = '';
  stratKeys.forEach(key => {
    const item = benchmark[key];
    if (!item) return;

    const isActive = (key.toLowerCase() === (activeStrategy || '').toLowerCase());
    const isTopF1 = (item.macro_f1 >= maxF1 && maxF1 > 0);
    const isTopAuroc = (item.auroc >= maxAuroc && maxAuroc > 0);

    let statusBadge = '<span class="badge-status idle">Siap Uji</span>';
    if (item.status === 'running') {
      statusBadge = '<span class="badge-status training" style="animation: pulse 1.5s infinite;">Pelatihan (R' + (item.rounds_done || 1) + ')</span>';
    } else if (item.status === 'completed') {
      statusBadge = '<span class="badge-status online">Selesai Evaluasi</span>';
    }

    const f1Percent = Math.min(100, Math.round(item.macro_f1 * 100));
    const aurocPercent = Math.min(100, Math.round(item.auroc * 100));

    html += `
      <tr class="${isActive ? 'active-strat-row' : ''}">
        <td>
          <div class="strat-badge-cell">
            <span class="strat-pill ${key}">${item.name}</span>
            <div style="font-size: 0.72rem; color: var(--text-muted);">${item.full_name || ''}</div>
          </div>
        </td>
        <td style="max-width: 200px;">
          <div style="font-weight: 500;">${item.mechanism}</div>
          <span class="benchmark-subtext">${item.note || ''}</span>
        </td>
        <td style="max-width: 190px;">
          <span style="color: var(--text-main); font-weight: 500;">${item.target}</span>
        </td>
        <td>
          <div class="score-cell ${isTopF1 ? 'highlight-top' : ''}">
            ${item.macro_f1.toFixed(3)}
            <span class="score-micro-bar">
              <span class="score-micro-fill ${isTopF1 ? 'green' : ''}" style="width: ${f1Percent}%;"></span>
            </span>
          </div>
        </td>
        <td>
          <div class="score-cell ${isTopAuroc ? 'highlight-top' : ''}">
            ${item.auroc.toFixed(3)}
            <span class="score-micro-bar">
              <span class="score-micro-fill ${isTopAuroc ? 'amber' : ''}" style="width: ${aurocPercent}%;"></span>
            </span>
          </div>
        </td>
        <td>
          <span class="score-cell" style="color: var(--text-sub);">${item.final_loss.toFixed(4)}</span>
        </td>
        <td>
          <span style="font-family: var(--font-mono); font-size: 0.78rem; font-weight: 600;">
            ${item.speed_sec ? item.speed_sec + 's' : '--'}
          </span>
        </td>
        <td>${statusBadge}</td>
        <td>
          <button type="button" 
                  class="btn-select-strat ${isActive ? 'active-strat' : ''}" 
                  onclick="selectStrategyDirectly('${key}')">
            ${isActive ? '✓ Terpilih' : 'Pilih'}
          </button>
        </td>
      </tr>
    `;
  });

  tbody.innerHTML = html;
}

// Quick selector from table row to radio button in Experiment Control Panel
window.selectStrategyDirectly = function(strategyKey) {
  const radio = document.querySelector(`input[name="strategy"][value="${strategyKey}"]`);
  if (radio) {
    radio.checked = true;
    // Dispatch change event
    radio.dispatchEvent(new Event('change', { bubbles: true }));
    // Update badge immediately
    const badge = document.getElementById('benchmarkActiveBadge');
    if (badge) badge.textContent = `Aktif: ${strategyKey.toUpperCase()}`;
    // Re-render table to update active row styling
    fetch('/api/benchmark')
      .then(r => r.json())
      .then(d => {
        if (d.benchmark) renderBenchmarkTable(d.benchmark, strategyKey);
      })
      .catch(() => {});
  }
};

// Polling Logs
async function pollLogs() {
  try {
    const source = appState.activeLogSource || 'server';
    const res = await fetch(`/api/logs?source=${source}`);
    if (res.ok) {
      const data = await res.json();
      if (data.logs) {
        const box = document.getElementById('logStreamBox');
        box.textContent = data.logs;
        box.scrollTop = box.scrollHeight; // Auto-scroll to bottom
      }
    }
  } catch (err) {
    // ignore
  }
}

// Fetch initial benchmark
async function fetchInitialBenchmark() {
  try {
    const res = await fetch('/api/benchmark');
    if (res.ok) {
      const data = await res.json();
      if (data.benchmark) {
        renderBenchmarkTable(data.benchmark, data.active_strategy || 'fedbn');
      }
    }
  } catch (err) {
    // ignore
  }
}

document.addEventListener('DOMContentLoaded', () => {
  initChart();
  setupEventListeners();
  fetchInitialBenchmark();
  
  const refreshBtn = document.getElementById('btnRefreshBenchmark');
  if (refreshBtn) {
    refreshBtn.addEventListener('click', fetchInitialBenchmark);
  }

  setInterval(pollBackend, 2000);
  setInterval(pollLogs, 2500);
});

