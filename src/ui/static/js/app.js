/**
 * Cardiopulse FL - Interactive ECG & Benchmark Orchestration Dashboard Logic
 */

// State Management
const state = {
  currentLead: 'II',
  isPlaying: true,
  currentRound: 50,
  activeStrategy: 'fedbn',
  samplingRate: 500,
  durationSec: 10,
  leads: ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
  nodes: [
    { id: 'vm1', name: 'VM 1 (Server)', ip: '10.34.100.114', role: 'Aggregator & UI', status: 'Online', round: '50/50', ping: '1.2ms' },
    { id: 'vm2', name: 'VM 2 (Client 1)', ip: '10.34.100.122', role: 'PTB-XL (~21k)', status: 'Active Fit', round: '50/50', ping: '2.4ms' },
    { id: 'vm3', name: 'VM 3 (Client 2)', ip: '10.34.100.123', role: 'Chapman (~10k)', status: 'Active Fit', round: '50/50', ping: '3.1ms' },
    { id: 'vm4', name: 'VM 4 (Client 3)', ip: '10.34.100.125', role: 'G12EC (~10k)', status: 'Active Fit', round: '50/50', ping: '2.8ms' }
  ],
  diagnoses: [
    { code: 'AF', name: 'Atrial Fibrillation', fedbn: 0.89, fedbb: 0.93, fedsol: 0.88, currentVal: 0.92 },
    { code: 'I-AVB', name: '1st Degree AV Block', fedbn: 0.84, fedbb: 0.88, fedsol: 0.83, currentVal: 0.12 },
    { code: 'LBBB', name: 'Left Bundle Branch Block', fedbn: 0.86, fedbb: 0.91, fedsol: 0.87, currentVal: 0.04 },
    { code: 'RBBB', name: 'Right Bundle Branch Block', fedbn: 0.85, fedbb: 0.89, fedsol: 0.86, currentVal: 0.08 },
    { code: 'PAC', name: 'Premature Atrial Contraction', fedbn: 0.76, fedbb: 0.85, fedsol: 0.79, currentVal: 0.74 },
    { code: 'PVC', name: 'Premature Ventricular Contraction', fedbn: 0.81, fedbb: 0.89, fedsol: 0.82, currentVal: 0.81 },
    { code: 'STD', name: 'ST-Segment Depression', fedbn: 0.79, fedbb: 0.87, fedsol: 0.80, currentVal: 0.18 },
    { code: 'STE', name: 'ST-Segment Elevation', fedbn: 0.82, fedbb: 0.90, fedsol: 0.84, currentVal: 0.05 }
  ]
};

// Canvas & ECG Wave Simulation Engine
let canvas, ctx;
let animationFrameId;
let sweepX = 0;
const waveBuffer = [];
const bufferSize = 1000;

function initECGCanvas() {
  canvas = document.getElementById('ecgTraceCanvas');
  if (!canvas) return;
  
  // Set canvas display resolution according to DPR
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);

  generateECGBuffer();
  animateECG();
}

// Generate realistic synthetic P-Q-R-S-T wave cycle
function generateECGBuffer() {
  waveBuffer.length = 0;
  const samplesPerBeat = 120;
  for (let i = 0; i < bufferSize; i++) {
    const phase = i % samplesPerBeat;
    let val = 0;
    
    // P wave (0 - 20)
    if (phase >= 10 && phase < 25) {
      val = 0.18 * Math.sin(((phase - 10) / 15) * Math.PI);
    }
    // Q wave (32 - 36)
    else if (phase >= 32 && phase < 36) {
      val = -0.15 * Math.sin(((phase - 32) / 4) * Math.PI);
    }
    // R peak (36 - 44)
    else if (phase >= 36 && phase < 44) {
      val = 1.15 * Math.sin(((phase - 36) / 8) * Math.PI);
    }
    // S wave (44 - 50)
    else if (phase >= 44 && phase < 50) {
      val = -0.28 * Math.sin(((phase - 44) / 6) * Math.PI);
    }
    // T wave (65 - 95)
    else if (phase >= 65 && phase < 95) {
      val = 0.28 * Math.sin(((phase - 65) / 30) * Math.PI);
    }
    // Baseline noise
    val += (Math.random() - 0.5) * 0.03;
    waveBuffer.push(val);
  }
}

function animateECG() {
  if (!state.isPlaying || !canvas) return;

  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const centerY = height / 2;
  const scaleY = height * 0.38;

  // Clear a narrow vertical bar ahead of sweep line (phosphor persistence monitor effect)
  const clearWidth = 24;
  ctx.clearRect(sweepX, 0, clearWidth, height);

  // Compute point
  const bufferIndex = Math.floor((sweepX / width) * waveBuffer.length) % waveBuffer.length;
  const signalVal = waveBuffer[bufferIndex];
  const targetY = centerY - (signalVal * scaleY);

  // Draw segment
  ctx.save();
  ctx.beginPath();
  ctx.strokeStyle = '#00f5a0';
  ctx.lineWidth = 2.2;
  ctx.shadowColor = 'rgba(0, 245, 160, 0.6)';
  ctx.shadowBlur = 8;
  ctx.lineCap = 'round';

  const prevX = (sweepX - 2 + width) % width;
  const prevIndex = Math.floor((prevX / width) * waveBuffer.length) % waveBuffer.length;
  const prevY = centerY - (waveBuffer[prevIndex] * scaleY);

  if (sweepX > 2) {
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(sweepX, targetY);
    ctx.stroke();
  }
  ctx.restore();

  // Advance sweep line
  sweepX = (sweepX + 2.5) % width;
  animationFrameId = requestAnimationFrame(animateECG);
}

// Chart.js Loss Convergence & Metric Charts
let lossChartInstance = null;
let radarChartInstance = null;

function initCharts() {
  const lossCtx = document.getElementById('lossConvergenceChart');
  if (!lossCtx || typeof Chart === 'undefined') return;

  const rounds = Array.from({ length: 50 }, (_, i) => i + 1);

  // Simulated realistic convergence curves across 50 rounds
  const fedbnLoss = rounds.map(r => 0.68 * Math.exp(-r / 12) + 0.19 + (Math.sin(r) * 0.008));
  const fedbbLoss = rounds.map(r => 0.72 * Math.exp(-r / 9) + 0.14 + (Math.cos(r) * 0.005));
  const fedsolLoss = rounds.map(r => 0.70 * Math.exp(-r / 11) + 0.17 + (Math.sin(r * 1.5) * 0.006));

  lossChartInstance = new Chart(lossCtx, {
    type: 'line',
    data: {
      labels: rounds,
      datasets: [
        {
          label: 'FedBN (Local BatchNorm1d)',
          data: fedbnLoss,
          borderColor: '#00f2fe',
          backgroundColor: 'rgba(0, 242, 254, 0.08)',
          borderWidth: 2.2,
          tension: 0.35,
          pointRadius: 0
        },
        {
          label: 'FedBB (PNB Loss + CBR Aggregation)',
          data: fedbbLoss,
          borderColor: '#ff4d6d',
          backgroundColor: 'rgba(255, 77, 109, 0.08)',
          borderWidth: 2.5,
          tension: 0.35,
          pointRadius: 0
        },
        {
          label: 'FedSOL (Orthogonal Restriction)',
          data: fedsolLoss,
          borderColor: '#00f5a0',
          backgroundColor: 'rgba(0, 245, 160, 0.08)',
          borderWidth: 2,
          tension: 0.35,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { family: 'Outfit', size: 12 } }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 20, 34, 0.95)',
          titleColor: '#fff',
          bodyColor: '#94a3b8',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Communication Round (1 - 50)', color: '#64748b' },
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b', maxTicksLimit: 10 }
        },
        y: {
          title: { display: true, text: 'Validation Loss (Multi-label BCE)', color: '#64748b' },
          grid: { color: 'rgba(255, 255, 255, 0.04)' },
          ticks: { color: '#64748b' }
        }
      }
    }
  });

  // Radar Chart: Recall per Minority & Majority Arrhythmia Class
  const radarCtx = document.getElementById('minorityRecallChart');
  if (radarCtx) {
    radarChartInstance = new Chart(radarCtx, {
      type: 'radar',
      data: {
        labels: state.diagnoses.map(d => d.code),
        datasets: [
          {
            label: 'FedBN',
            data: state.diagnoses.map(d => d.fedbn),
            borderColor: '#00f2fe',
            backgroundColor: 'rgba(0, 242, 254, 0.2)',
            borderWidth: 1.8
          },
          {
            label: 'FedBB (Imbalance Resilient)',
            data: state.diagnoses.map(d => d.fedbb),
            borderColor: '#ff4d6d',
            backgroundColor: 'rgba(255, 77, 109, 0.25)',
            borderWidth: 2.2
          },
          {
            label: 'FedSOL',
            data: state.diagnoses.map(d => d.fedsol),
            borderColor: '#00f5a0',
            backgroundColor: 'rgba(0, 245, 160, 0.2)',
            borderWidth: 1.8
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'bottom', labels: { color: '#94a3b8', font: { size: 11 } } }
        },
        scales: {
          r: {
            angleLines: { color: 'rgba(255, 255, 255, 0.06)' },
            grid: { color: 'rgba(255, 255, 255, 0.06)' },
            pointLabels: { color: '#cbd5e1', font: { family: 'JetBrains Mono', size: 11 } },
            ticks: { color: '#64748b', backdropColor: 'transparent', stepSize: 0.2 }
          }
        }
      }
    });
  }
}

// UI Event Listeners & Interaction
function setupUIInteractions() {
  // Lead selection buttons
  const leadButtons = document.querySelectorAll('.lead-btn');
  leadButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
      leadButtons.forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      state.currentLead = e.target.dataset.lead;
      document.getElementById('currentLeadDisplay').textContent = `Lead: ${state.currentLead}`;
      generateECGBuffer();
    });
  });

  // Play / Pause button
  const toggleBtn = document.getElementById('btnTogglePlayback');
  if (toggleBtn) {
    toggleBtn.addEventListener('click', () => {
      state.isPlaying = !state.isPlaying;
      toggleBtn.innerHTML = state.isPlaying 
        ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg> Pause`
        : `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Resume`;
      if (state.isPlaying) animateECG();
    });
  }

  // Simulate Multi-label Inference Button
  const btnRunInference = document.getElementById('btnRunInference');
  if (btnRunInference) {
    btnRunInference.addEventListener('click', () => {
      btnRunInference.textContent = 'Running ResNet1D Inference...';
      btnRunInference.style.opacity = '0.7';

      setTimeout(() => {
        // Randomize probabilities slightly to simulate real test signal prediction
        state.diagnoses.forEach(diag => {
          diag.currentVal = Math.min(0.98, Math.max(0.02, diag.currentVal + (Math.random() - 0.5) * 0.15));
        });
        renderDiagnosisBars();
        btnRunInference.textContent = 'Run Model Inference (ResNet1D)';
        btnRunInference.style.opacity = '1';
      }, 600);
    });
  }

  // Window resize handler
  window.addEventListener('resize', () => {
    if (canvas) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    }
  });
}

function renderDiagnosisBars() {
  const container = document.getElementById('diagnosisBarContainer');
  if (!container) return;
  container.innerHTML = '';

  state.diagnoses.forEach(diag => {
    const pct = Math.round(diag.currentVal * 100);
    const isDetected = diag.currentVal >= 0.5;
    const item = document.createElement('div');
    item.className = 'diagnosis-item';
    item.innerHTML = `
      <div class="diag-label-row">
        <span><strong>${diag.code}</strong> — ${diag.name}</span>
        <span style="font-family: var(--font-mono); color: ${isDetected ? 'var(--accent-emerald)' : 'var(--text-muted)'}; font-weight: 600;">
          ${pct}% ${isDetected ? '● DETECTED' : ''}
        </span>
      </div>
      <div class="diag-bar-track">
        <div class="diag-bar-fill" style="width: ${pct}%; background: ${isDetected ? 'linear-gradient(90deg, #00f2fe, #00f5a0)' : 'rgba(255,255,255,0.2)'};"></div>
      </div>
    `;
    container.appendChild(item);
  });
}

// Fetch Live Status from Backend API (if available)
async function fetchClusterStatus() {
  try {
    const res = await fetch('/api/status');
    if (res.ok) {
      const data = await res.json();
      if (data.nodes) {
        // update dynamic nodes
      }
    }
  } catch (err) {
    // Graceful offline fallback
    console.log('Using local client state');
  }
}

// Initializer
document.addEventListener('DOMContentLoaded', () => {
  initECGCanvas();
  initCharts();
  setupUIInteractions();
  renderDiagnosisBars();
  setInterval(fetchClusterStatus, 5000);
});
