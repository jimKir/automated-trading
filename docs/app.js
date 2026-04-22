/* ── Dashboard App ────────────────────────────────────────────────── */
(function () {
  "use strict";

  const DATA_URL = "data/snapshot.json";

  /* ── Helpers ───────────────────────────────────────────────────── */
  function $(id) { return document.getElementById(id); }

  function fmt(n, decimals) {
    if (n == null) return "\u2014";
    return n.toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  }

  function fmtDollar(n) {
    if (n == null) return "\u2014";
    return "$" + fmt(Math.abs(n), 2);
  }

  function fmtPct(n, showSign) {
    if (n == null) return "\u2014";
    var s = fmt(Math.abs(n), 2) + "%";
    if (showSign !== false) s = (n >= 0 ? "+" : "\u2212") + s;
    return s;
  }

  function fmtPP(n) {
    if (n == null) return "\u2014";
    return (n >= 0 ? "+" : "\u2212") + fmt(Math.abs(n), 2) + " pp";
  }

  function colorClass(n) {
    if (n == null || n === 0) return "";
    return n > 0 ? "positive" : "negative";
  }

  function relativeTime(iso) {
    var d = new Date(iso);
    var diff = Math.floor((Date.now() - d.getTime()) / 1000);
    if (diff < 60) return "Updated just now";
    if (diff < 3600) return "Updated " + Math.floor(diff / 60) + " min ago";
    if (diff < 86400) return "Updated " + Math.floor(diff / 3600) + " hr ago";
    return "Updated " + d.toLocaleDateString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  }

  function formatOrderTime(iso) {
    if (!iso) return "\u2014";
    var d = new Date(iso);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" }) +
      " " + d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  }

  /* ── KPI Cards ────────────────────────────────────────────────── */
  function renderKPIs(data) {
    var a = data.account;
    var b = data.benchmarks;
    var m = data.metrics;

    var eqEl = $("kpi-equity");
    eqEl.textContent = "$" + fmt(a.equity, 2);

    var dcEl = $("kpi-day-change");
    dcEl.textContent = (a.day_change >= 0 ? "+" : "\u2212") + "$" + fmt(Math.abs(a.day_change), 2) + " today";
    dcEl.className = "sub " + colorClass(a.day_change);

    var retEl = $("kpi-return");
    retEl.textContent = fmtPct(a.total_return_pct);
    retEl.className = "value " + colorClass(a.total_return_pct);

    var spyEl = $("kpi-vs-spy");
    spyEl.textContent = fmtPP(b.vs_spy_pp);
    spyEl.className = "value " + colorClass(b.vs_spy_pp);

    var qqqEl = $("kpi-vs-qqq");
    qqqEl.textContent = fmtPP(b.vs_qqq_pp);
    qqqEl.className = "value " + colorClass(b.vs_qqq_pp);

    var ddEl = $("kpi-drawdown");
    ddEl.textContent = fmtPct(m.max_drawdown_pct, false);
    ddEl.className = "value negative";

    $("kpi-positions").textContent = data.positions.length;
  }

  /* ── Chart ─────────────────────────────────────────────────────── */
  var chartInstance = null;

  function renderChart(curve) {
    var labels = curve.map(function (p) { return p.date; });
    var botData = curve.map(function (p) { return p.bot; });
    var spyData = curve.map(function (p) { return p.spy; });
    var qqqData = curve.map(function (p) { return p.qqq; });

    var ctx = $("perf-chart").getContext("2d");
    if (chartInstance) chartInstance.destroy();

    chartInstance = new Chart(ctx, {
      type: "line",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Bot",
            data: botData,
            borderColor: "#58a6ff",
            backgroundColor: "rgba(88,166,255,0.08)",
            fill: true,
            tension: 0.2,
            pointRadius: 0,
            pointHitRadius: 8,
            borderWidth: 2,
          },
          {
            label: "SPY",
            data: spyData,
            borderColor: "#8b949e",
            backgroundColor: "transparent",
            fill: false,
            tension: 0.2,
            pointRadius: 0,
            pointHitRadius: 8,
            borderWidth: 1.5,
            borderDash: [4, 3],
          },
          {
            label: "QQQ",
            data: qqqData,
            borderColor: "#d29922",
            backgroundColor: "transparent",
            fill: false,
            tension: 0.2,
            pointRadius: 0,
            pointHitRadius: 8,
            borderWidth: 1.5,
            borderDash: [4, 3],
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            labels: { color: "#c9d1d9", usePointStyle: true, padding: 16 },
          },
          tooltip: {
            backgroundColor: "#161b22",
            titleColor: "#c9d1d9",
            bodyColor: "#c9d1d9",
            borderColor: "#30363d",
            borderWidth: 1,
            callbacks: {
              label: function (ctx) {
                var val = ctx.parsed.y;
                var change = val - 100;
                return ctx.dataset.label + ": " + val.toFixed(2) +
                  " (" + (change >= 0 ? "+" : "") + change.toFixed(2) + "%)";
              },
            },
          },
        },
        scales: {
          x: {
            ticks: {
              color: "#8b949e",
              maxTicksLimit: 12,
              maxRotation: 0,
            },
            grid: { color: "rgba(48,54,61,0.4)" },
          },
          y: {
            ticks: { color: "#8b949e" },
            grid: { color: "rgba(48,54,61,0.4)" },
          },
        },
      },
    });
  }

  /* ── Positions Table ───────────────────────────────────────────── */
  function renderPositions(positions) {
    var tbody = $("positions-body");
    if (!positions.length) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;color:var(--text-muted)">No open positions</td></tr>';
      return;
    }
    tbody.innerHTML = positions.map(function (p) {
      var plClass = colorClass(p.unrealized_pl);
      return '<tr>' +
        '<td>' + p.symbol + '</td>' +
        '<td class="right">' + fmt(p.qty, 2) + '</td>' +
        '<td class="right">' + fmtDollar(p.market_value) + '</td>' +
        '<td class="right">$' + fmt(p.avg_entry_price, 2) + '</td>' +
        '<td class="right">$' + fmt(p.current_price, 2) + '</td>' +
        '<td class="right ' + plClass + '">' + (p.unrealized_pl >= 0 ? '+' : '\u2212') + '$' + fmt(Math.abs(p.unrealized_pl), 2) + '</td>' +
        '<td class="right ' + plClass + '">' + fmtPct(p.unrealized_pl_pct) + '</td>' +
        '</tr>';
    }).join("");
  }

  /* ── Orders Table ──────────────────────────────────────────────── */
  function renderOrders(orders) {
    var tbody = $("orders-body");
    if (!orders.length) {
      tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;color:var(--text-muted)">No recent orders</td></tr>';
      return;
    }
    tbody.innerHTML = orders.map(function (o) {
      var sideClass = o.side === "buy" ? "positive" : "negative";
      return '<tr>' +
        '<td>' + formatOrderTime(o.submitted_at) + '</td>' +
        '<td>' + o.symbol + '</td>' +
        '<td class="' + sideClass + '">' + o.side.toUpperCase() + '</td>' +
        '<td class="right">' + fmt(o.qty, 2) + '</td>' +
        '<td>' + o.status + '</td>' +
        '<td class="right">' + (o.filled_avg_price ? '$' + fmt(o.filled_avg_price, 2) : '\u2014') + '</td>' +
        '</tr>';
    }).join("");
  }

  /* ── Load & Render ─────────────────────────────────────────────── */
  function load() {
    fetch(DATA_URL + "?t=" + Date.now())
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (data) {
        $("last-updated").textContent = relativeTime(data.last_updated);
        renderKPIs(data);
        renderChart(data.equity_curve);
        renderPositions(data.positions);
        renderOrders(data.recent_orders);
      })
      .catch(function (err) {
        $("last-updated").textContent = "Failed to load data";
        $("last-updated").classList.add("error");
        console.error("Dashboard load error:", err);
      });
  }

  load();
})();
