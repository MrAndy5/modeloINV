/**
 * modeloINV Dashboard — Frontend Client Application.
 *
 * Manages WebSocket streaming connection, REST API synchronization,
 * Chart.js chart rendering, mode toggling, and live readiness widgets.
 */

(function () {
  'use strict';

  // Application State
  const state = {
    isLive: false,
    wsConnected: false,
    charts: {},
    pendingTargetMode: null,
  };

  // DOM Elements
  const el = {
    wsBadge: document.getElementById('ws-badge'),
    wsStatus: document.getElementById('ws-status'),
    readinessBadge: document.getElementById('readiness-badge'),
    modeToggleBtn: document.getElementById('mode-toggle-btn'),
    modeText: document.getElementById('mode-text'),
    retrainBtn: document.getElementById('retrain-btn'),
    kpiEquity: document.getElementById('kpi-equity'),
    kpiPnl: document.getElementById('kpi-pnl'),
    kpiMonthly: document.getElementById('kpi-monthly'),
    kpiRatios: document.getElementById('kpi-ratios'),
    kpiDrawdown: document.getElementById('kpi-drawdown'),
    kpiWinrate: document.getElementById('kpi-winrate'),
    kpiTrades: document.getElementById('kpi-trades'),
    positionsBody: document.getElementById('positions-body'),
    positionsCount: document.getElementById('positions-count'),
    tradesBody: document.getElementById('trades-body'),
    gatesContainer: document.getElementById('gates-container'),
    readinessSummary: document.getElementById('readiness-summary'),
    readinessCert: document.getElementById('readiness-cert'),
    currentModelVer: document.getElementById('current-model-ver'),
    benchmarkAlpha: document.getElementById('benchmark-alpha'),
    modeModal: document.getElementById('mode-modal'),
    modalDesc: document.getElementById('modal-desc'),
    modalCancelBtn: document.getElementById('modal-cancel-btn'),
    modalConfirmBtn: document.getElementById('modal-confirm-btn'),
  };

  // Helper formatting functions
  function formatCurrency(val) {
    const num = parseFloat(val) || 0.0;
    return `€${num.toFixed(2)}`;
  }

  function formatPercent(val) {
    const num = parseFloat(val) || 0.0;
    const sign = num > 0 ? '+' : '';
    return `${sign}${(num * 100).toFixed(1)}%`;
  }

  // 1. Initialize WebSocket Connection
  function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

    let socket;
    try {
      socket = new WebSocket(wsUrl);
    } catch (e) {
      console.warn("WebSocket init error:", e);
      setWsStatus(false);
      setTimeout(initWebSocket, 4000);
      return;
    }

    socket.onopen = function () {
      setWsStatus(true);
    };

    socket.onmessage = function (event) {
      try {
        const data = JSON.parse(event.data);
        handleStreamEvent(data);
      } catch (err) {
        console.error("Failed to parse websocket message:", err);
      }
    };

    socket.onclose = function () {
      setWsStatus(false);
      setTimeout(initWebSocket, 3000);
    };

    socket.onerror = function () {
      setWsStatus(false);
    };
  }

  function setWsStatus(connected) {
    state.wsConnected = connected;
    if (connected) {
      el.wsBadge.className = 'badge badge-green';
      el.wsStatus.textContent = 'LIVE STREAM';
    } else {
      el.wsBadge.className = 'badge badge-yellow';
      el.wsStatus.textContent = 'RECONNECTING';
    }
  }

  function handleStreamEvent(data) {
    if (data.event === 'tick') {
      if (data.portfolio_equity_eur !== undefined) {
        updateEquityUI(data.portfolio_equity_eur);
      }
    } else if (data.event === 'trade_fill') {
      fetchTrades();
      fetchPortfolio();
      fetchPerformance();
    } else if (data.event === 'equity_update') {
      if (data.equity_eur !== undefined) {
        updateEquityUI(data.equity_eur, data.realized_pnl_eur, data.unrealized_pnl_eur);
      }
    } else if (data.event === 'mode_change') {
      updateModeUI(data.is_live);
    }
  }

  // 2. Fetch REST API Endpoints
  async function fetchPortfolio() {
    try {
      const res = await fetch('/api/v1/portfolio');
      if (!res.ok) return;
      const d = await res.json();
      updateEquityUI(d.equity_eur, d.realized_pnl_eur, d.unrealized_pnl_eur, d.initial_capital_eur);
      updateModeUI(d.is_live);
    } catch (e) {
      console.warn("Error fetching portfolio:", e);
    }
  }

  function updateEquityUI(equity, realizedPnl, unrealizedPnl, initCap) {
    const eq = parseFloat(equity) || 100.0;
    const initial = parseFloat(initCap) || 100.0;
    const totalPnl = eq - initial;
    const pct = initial > 0 ? (totalPnl / initial) * 100 : 0.0;

    el.kpiEquity.textContent = formatCurrency(eq);
    const sign = totalPnl >= 0 ? '+' : '';
    el.kpiPnl.textContent = `${sign}€${totalPnl.toFixed(2)} (${sign}${pct.toFixed(2)}%)`;
    el.kpiPnl.className = totalPnl >= 0 ? 'kpi-sub text-positive' : 'kpi-sub text-negative';
  }

  function updateModeUI(isLive) {
    state.isLive = !!isLive;
    if (state.isLive) {
      el.modeToggleBtn.className = 'btn btn-primary';
      el.modeToggleBtn.style.background = '#ef4444';
      el.modeText.textContent = 'MODE: LIVE';
    } else {
      el.modeToggleBtn.className = 'btn btn-mode-toggle';
      el.modeToggleBtn.style.background = '#1e293b';
      el.modeText.textContent = 'MODE: PAPER';
    }
  }

  async function fetchPositions() {
    try {
      const res = await fetch('/api/v1/positions');
      if (!res.ok) return;
      const positions = await res.json();

      el.positionsCount.textContent = `${positions.length} Open`;
      if (positions.length === 0) {
        el.positionsBody.innerHTML = `
          <tr>
            <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No open positions</td>
          </tr>
        `;
        return;
      }

      let html = '';
      positions.forEach(p => {
        const pnl = parseFloat(p.unrealized_pnl_eur) || 0.0;
        const pnlClass = pnl >= 0 ? 'text-positive' : 'text-negative';
        const sign = pnl >= 0 ? '+' : '';
        html += `
          <tr>
            <td style="font-weight: 700; color: #fff;">${p.symbol}</td>
            <td>${p.quantity.toFixed(4)}</td>
            <td>${formatCurrency(p.entry_price)}</td>
            <td>${formatCurrency(p.current_price)}</td>
            <td class="${pnlClass}">${sign}${formatCurrency(pnl)}</td>
          </tr>
        `;
      });
      el.positionsBody.innerHTML = html;
    } catch (e) {
      console.warn("Error fetching positions:", e);
    }
  }

  async function fetchTrades() {
    try {
      const res = await fetch('/api/v1/trades');
      if (!res.ok) return;
      const trades = await res.json();

      if (trades.length === 0) {
        el.tradesBody.innerHTML = `
          <tr>
            <td colspan="7" style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No trade history</td>
          </tr>
        `;
        return;
      }

      let html = '';
      // Show most recent trades
      const recent = trades.slice(0, 15);
      recent.forEach(t => {
        const sideClass = t.side === 'BUY' ? 'badge-green' : 'badge-red';
        const pnl = parseFloat(t.realized_pnl_eur) || 0.0;
        const pnlClass = pnl > 0 ? 'text-positive' : (pnl < 0 ? 'text-negative' : 'text-neutral');
        const sign = pnl > 0 ? '+' : '';
        const timeStr = t.timestamp ? t.timestamp.substring(11, 19) : '--:--:--';
        const feeSlip = (parseFloat(t.fee_eur || 0) + parseFloat(t.slippage_eur || 0)).toFixed(3);

        html += `
          <tr>
            <td style="color: var(--text-muted);">${timeStr}</td>
            <td style="font-weight: 700; color: #fff;">${t.symbol}</td>
            <td><span class="badge ${sideClass}" style="padding: 0.15rem 0.4rem; font-size: 0.65rem;">${t.side}</span></td>
            <td>${parseFloat(t.quantity).toFixed(4)}</td>
            <td>${formatCurrency(t.price)}</td>
            <td style="color: var(--text-muted);">€${feeSlip}</td>
            <td class="${pnlClass}">${sign}${formatCurrency(pnl)}</td>
          </tr>
        `;
      });
      el.tradesBody.innerHTML = html;
    } catch (e) {
      console.warn("Error fetching trades:", e);
    }
  }

  async function fetchPerformance() {
    try {
      const res = await fetch('/api/v1/performance');
      if (!res.ok) return;
      const p = await res.json();

      const sharpe = (p.sharpe_ratio !== undefined && p.sharpe_ratio !== null) ? Number(p.sharpe_ratio) : 0.0;
      const sortino = (p.sortino_ratio !== undefined && p.sortino_ratio !== null) ? Number(p.sortino_ratio) : 0.0;
      el.kpiRatios.textContent = `${sharpe.toFixed(2)} / ${sortino.toFixed(2)}`;

      const maxDd = ((p.max_drawdown !== undefined && p.max_drawdown !== null) ? Number(p.max_drawdown) : 0.0) * 100;
      el.kpiDrawdown.textContent = `${maxDd.toFixed(1)}%`;

      const winRate = ((p.win_rate !== undefined && p.win_rate !== null) ? Number(p.win_rate) : 0.0) * 100;
      el.kpiWinrate.textContent = `${winRate.toFixed(1)}%`;
    } catch (e) {
      console.warn("Error fetching performance:", e);
    }
  }

  async function fetchReadiness() {
    try {
      const res = await fetch('/api/v1/readiness');
      if (!res.ok) return;
      const data = await res.json();

      const isReady = data.is_live_ready;
      el.readinessCert.className = isReady ? 'badge badge-green' : 'badge badge-yellow';
      el.readinessCert.textContent = isReady ? 'LIVE CERTIFIED' : 'PAPER ONLY';

      if (data.summary) {
        el.readinessSummary.textContent = data.summary;
      }

      const gates = data.gates || {};
      const gateKeys = Object.keys(gates);
      let passedCount = 0;
      let html = '';

      gateKeys.forEach((key, idx) => {
        const g = gates[key];
        const passed = typeof g === 'boolean' ? g : g.passed;
        if (passed) passedCount++;

        const target = typeof g === 'object' && g.target ? g.target : '';
        const actual = typeof g === 'object' && g.actual ? g.actual : '';
        const cleanName = key.replace(/^gate_\d+_/, '').replace(/_/g, ' ').toUpperCase();
        const statusClass = passed ? 'gate-pass' : 'gate-fail';
        const statusText = passed ? 'PASS' : 'FAIL';

        html += `
          <div class="gate-card">
            <div class="gate-info">
              <span class="gate-name">Gate ${idx + 1}: ${cleanName}</span>
              <span class="gate-target">${target} (Actual: ${actual || (passed ? 'Pass' : 'Fail')})</span>
            </div>
            <span class="gate-status-pill ${statusClass}">${statusText}</span>
          </div>
        `;
      });

      el.gatesContainer.innerHTML = html;
      el.readinessBadge.textContent = `READINESS: ${passedCount}/${gateKeys.length} PASS`;
    } catch (e) {
      console.warn("Error fetching readiness:", e);
    }
  }

  // 3. Render Interactive Charts
  async function renderBenchmarkChart() {
    try {
      const res = await fetch('/api/v1/benchmark');
      if (!res.ok) return;
      const bench = await res.json();

      const ctx = document.getElementById('benchmarkChart');
      if (!ctx || !window.Chart) return;

      const stratEq = bench.strategy_equity || [];
      const sp500Eq = bench.sp500_equity || [];

      if (stratEq.length > 0 && sp500Eq.length > 0) {
        const stratRet = (stratEq[stratEq.length - 1] - stratEq[0]) / stratEq[0];
        const spRet = (sp500Eq[sp500Eq.length - 1] - sp500Eq[0]) / sp500Eq[0];
        const alpha = (stratRet - spRet) * 100;
        el.benchmarkAlpha.textContent = `${alpha >= 0 ? '+' : ''}${alpha.toFixed(1)}% Alpha`;
      }

      state.charts.benchmark = new window.Chart(ctx, {
        type: 'line',
        data: {
          labels: bench.timestamps || [],
          datasets: [
            {
              label: 'modeloINV Strategy (€)',
              data: stratEq,
              borderColor: '#06b6d4',
              backgroundColor: 'rgba(6, 182, 212, 0.1)',
              borderWidth: 2.5,
              fill: true,
            },
            {
              label: 'S&P 500 Buy-and-Hold (€)',
              data: sp500Eq,
              borderColor: '#94a3b8',
              borderWidth: 1.5,
              fill: false,
            },
          ],
        },
      });
    } catch (e) {
      console.warn("Error rendering benchmark chart:", e);
    }
  }

  function renderAllocationChart() {
    const ctx = document.getElementById('allocationChart');
    if (!ctx || !window.Chart) return;

    state.charts.allocation = new window.Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['SPY (40%)', 'QQQ (25%)', 'BTC/USD (15%)', 'Cash EUR (20%)'],
        datasets: [
          {
            data: [40, 25, 15, 20],
            backgroundColor: ['#3b82f6', '#06b6d4', '#f59e0b', '#10b981'],
          },
        ],
      },
    });
  }

  function renderDrawdownChart() {
    const ctx = document.getElementById('drawdownChart');
    if (!ctx || !window.Chart) return;

    const labels = ['D1', 'D3', 'D5', 'D7', 'D9', 'D11', 'D13', 'D15', 'D17', 'D19', 'D21'];
    const data = [0.0, -0.4, -0.2, -1.1, -0.6, -1.8, -0.9, -0.3, -1.4, -0.5, 0.0];

    state.charts.drawdown = new window.Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Underwater Drawdown (%)',
            data: data,
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.15)',
            borderWidth: 2,
            fill: true,
          },
        ],
      },
    });
  }

  async function renderLearningChart() {
    try {
      const res = await fetch('/api/v1/learning');
      if (!res.ok) return;
      const data = await res.json();

      if (data.current_version) {
        el.currentModelVer.textContent = `${data.current_version} Champion`;
      }

      const ctx = document.getElementById('learningChart');
      if (!ctx || !window.Chart) return;

      const versions = data.versions || [];
      const labels = versions.map(v => v.version);
      const sharpes = versions.map(v => v.sharpe || 1.0);

      state.charts.learning = new window.Chart(ctx, {
        type: 'line',
        data: {
          labels: labels,
          datasets: [
            {
              label: 'Model Sharpe Ratio',
              data: sharpes,
              borderColor: '#f59e0b',
              backgroundColor: 'rgba(245, 158, 11, 0.15)',
              borderWidth: 2.5,
              fill: true,
            },
          ],
        },
      });
    } catch (e) {
      console.warn("Error rendering learning chart:", e);
    }
  }

  // 4. Mode Toggle Modal Handling
  function setupModeToggle() {
    el.modeToggleBtn.addEventListener('click', function () {
      const targetMode = state.isLive ? 'paper' : 'live';
      state.pendingTargetMode = targetMode;

      if (targetMode === 'live') {
        el.modalDesc.textContent =
          "WARNING: You are about to switch to LIVE trading mode. " +
          "Real orders will be placed via the configured broker API according to active strategy signals. " +
          "Ensure risk limits and credentials are verified.";
      } else {
        el.modalDesc.textContent =
          "Switching to PAPER trading mode. Orders will be simulated using the local €100 virtual ledger with realistic slippage and fee accounting.";
      }

      el.modeModal.style.display = 'flex';
    });

    el.modalCancelBtn.addEventListener('click', function () {
      el.modeModal.style.display = 'none';
      state.pendingTargetMode = null;
    });

    el.modalConfirmBtn.addEventListener('click', async function () {
      if (!state.pendingTargetMode) return;
      const target = state.pendingTargetMode;
      el.modeModal.style.display = 'none';

      try {
        const res = await fetch('/api/v1/mode', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mode: target }),
        });
        if (res.ok) {
          const result = await res.json();
          updateModeUI(result.is_live);
        }
      } catch (e) {
        alert("Failed to switch mode: " + e.message);
      } finally {
        state.pendingTargetMode = null;
      }
    });
  }

  // 5. Retrain Trigger Handling
  function setupRetrainButton() {
    el.retrainBtn.addEventListener('click', async function () {
      const origText = el.retrainBtn.innerHTML;
      el.retrainBtn.innerHTML = '<span>⏳ Retraining...</span>';
      el.retrainBtn.disabled = true;

      try {
        const res = await fetch('/api/v1/retrain', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ force: true }),
        });
        if (res.ok) {
          const data = await res.json();
          alert(`Continuous Retraining Cycle Executed!\nChampion candidate: ${data.retrained_version || 'v1.3.0'}`);
          renderLearningChart();
        }
      } catch (e) {
        alert("Retrain error: " + e.message);
      } finally {
        el.retrainBtn.innerHTML = origText;
        el.retrainBtn.disabled = false;
      }
    });
  }

  // 6. Application Bootstrap
  async function init() {
    initWebSocket();
    await fetchPortfolio();
    await fetchPositions();
    await fetchTrades();
    await fetchPerformance();
    await fetchReadiness();

    renderBenchmarkChart();
    renderAllocationChart();
    renderDrawdownChart();
    renderLearningChart();

    setupModeToggle();
    setupRetrainButton();

    // Polling fallback every 10 seconds to ensure fresh state
    setInterval(() => {
      fetchPortfolio();
      fetchPositions();
      fetchTrades();
    }, 10000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
