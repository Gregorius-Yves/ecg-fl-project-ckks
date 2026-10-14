/**
 * CardioPulse FL - Light Theme Experiment Control & Live Monitor
 * Manages Start/Stop commands, live metric plots, client status, and log streaming.
 */

// Application State
const appState = {
  isRunning: false,
  activeMetric: 'loss',
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
          label: 'Train Loss',
          data: [],
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.08)',
          borderWidth: 2,
          tension: 0.3,
          fill: true
        },
        {
          label: 'Val Loss',
          data: [],
          borderColor: '#10b981',
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 5],
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
          labels: { font: { family: 'Inter', size: 12 }, boxWidth: 16 }
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
          title: { display: true, text: 'Nilai Metrik', color: '#64748b' },
          grid: { color: '#f1f5f9' },
          ticks: { color: '#64748b' }
        }
      }
    }
  });
}

// Switch Metric Tab (Loss / F1 / AUROC)
function switchMetricTab(metricName) {
  appState.activeMetric = metricName;
  document.querySelectorAll('.chart-tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.metric === metricName);
  });

  if (!metricChart) return;

  const rounds = appState.metricsHistory.rounds;

  if (metricName === 'loss') {
    metricChart.data.labels = rounds;
    metricChart.data.datasets = [
      {
        label: 'Train Loss',
        data: appState.metricsHistory.loss,
        borderColor: '#2563eb',
        backgroundColor: 'rgba(37, 99, 235, 0.08)',
        borderWidth: 2,
        tension: 0.3,
        fill: true
      },
      {
        label: 'Val Loss',
        data: appState.metricsHistory.val_loss,
        borderColor: '#10b981',
        backgroundColor: 'transparent',
        borderWidth: 2,
        borderDash: [5, 5],
        tension: 0.3
      }
    ];
    metricChart.options.scales.y.title.text = 'BCE Loss';
  } else if (metricName === 'f1') {
    metricChart.data.labels = rounds;
    metricChart.data.datasets = [
      {
        label: 'Macro-F1 Score',
        data: appState.metricsHistory.f1_macro,
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.08)',
        borderWidth: 2.5,
        tension: 0.3,
        fill: true
      }
    ];
    metricChart.options.scales.y.title.text = 'Macro-F1 (0.0 - 1.0)';
  } else if (metricName === 'auroc') {
    metricChart.data.labels = rounds;
    metricChart.data.datasets = [
      {
        label: 'Macro-AUROC',
        data: appState.metricsHistory.auroc,
        borderColor: '#f59e0b',
        backgroundColor: 'rgba(245, 158, 11, 0.08)',
        borderWidth: 2.5,
        tension: 0.3,
        fill: true
      }
    ];
    metricChart.options.scales.y.title.text = 'AUROC (0.5 - 1.0)';
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
  btnReset.addEventListener('click', () => {
    if (appState.isRunning) {
      alert('Hentikan pelatihan terlebih dahulu sebelum melakukan reset!');
      return;
    }
    appState.metricsHistory = { rounds: [], loss: [], val_loss: [], f1_macro: [], auroc: [] };
    appState.currentRound = 0;
    updateProgressBar(0, appState.totalRounds);
    switchMetricTab(appState.activeMetric);
    document.getElementById('logStreamBox').textContent = 'Sesi di-reset. Siap memulai eksperimen baru.';
  });

  // Download Model Checkpoint
  const btnDownload = document.getElementById('btnDownloadModel');
  btnDownload.addEventListener('click', () => {
    window.location.href = '/api/download-model';
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
    'vm2': { badge: 'badgeVm2', card: 'clientNodeVm2' },
    'vm3': { badge: 'badgeVm3', card: 'clientNodeVm3' },
    'vm4': { badge: 'badgeVm4', card: 'clientNodeVm4' }
  };

  for (const [key, els] of Object.entries(clientMap)) {
    const badge = document.getElementById(els.badge);
    const card = document.getElementById(els.card);
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
    } else if (client.status === 'training') {
      if (badge) {
        badge.className = 'badge-status training';
        badge.textContent = client.status_label || 'Training';
      }
      if (card) {
        card.classList.remove('is-offline');
        card.classList.add('is-training');
      }
    } else {
      if (badge) {
        badge.className = 'badge-status online';
        badge.textContent = client.status_label || 'Online';
      }
      if (card) {
        card.classList.remove('is-offline');
        card.classList.remove('is-training');
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

// Polling Logs
async function pollLogs() {
  try {
    const res = await fetch('/api/logs');
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

document.addEventListener('DOMContentLoaded', () => {
  initChart();
  setupEventListeners();
  setInterval(pollBackend, 2000);
  setInterval(pollLogs, 2500);
});
