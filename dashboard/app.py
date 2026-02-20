"""
모의투자 웹 대시보드
- Flask 기반 단일 파일
- 실시간 포트폴리오 현황, 매매 내역, 성과 차트, 후회 로그
"""
import sys
import os
import json
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nn_config import DB_PATH, DASHBOARD_PORT, DASHBOARD_HOST


def run_dashboard():
    try:
        from flask import Flask, render_template_string, jsonify
    except ImportError:
        print("Flask가 필요합니다: pip3 install --break-system-packages flask")
        return

    from db import Database

    app = Flask(__name__)
    db = Database(str(DB_PATH))

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/portfolio")
    def api_portfolio():
        latest = db.get_latest_portfolio("paper")
        if not latest:
            return jsonify({"error": "No data"})
        return jsonify(latest)

    @app.route("/api/portfolio/history")
    def api_portfolio_history():
        rows = db.get_portfolio_snapshots("paper")
        data = []
        for r in reversed(rows):  # 오름차순
            data.append({
                "date": r["date"],
                "total_value": r["total_value"],
                "cumulative_return": r.get("cumulative_return", 0),
                "daily_return": r.get("daily_return", 0),
                "max_drawdown": r.get("max_drawdown", 0),
                "sharpe_ratio": r.get("sharpe_ratio", 0),
            })
        return jsonify(data)

    @app.route("/api/trades")
    def api_trades():
        trades = db.get_trades(trade_type="paper")
        return jsonify(trades[-100:])  # 최근 100건

    @app.route("/api/regrets")
    def api_regrets():
        regrets = db.execute_raw(
            "SELECT * FROM regret_log ORDER BY id DESC LIMIT 50"
        )
        return jsonify(regrets)

    @app.route("/api/predictions")
    def api_predictions():
        today = datetime.now().strftime("%Y%m%d")
        preds = db.get_predictions(date=today, model_name="ensemble")
        return jsonify(preds)

    @app.route("/api/predictions/forecast")
    def api_forecast():
        """auto_forecast + human_forecast 예측 현황."""
        rows = db.execute_raw(
            """SELECT p.stock_code, p.date, p.model_name, p.predicted_class,
                      p.confidence, p.actual_class, p.actual_return
               FROM predictions p
               WHERE p.model_name IN ('auto_forecast', 'human_forecast')
               ORDER BY p.id DESC LIMIT 30""")
        return jsonify(rows)

    @app.route("/api/live_prices")
    def api_live_prices():
        """보유 종목 실시간 가격 (장중용)."""
        latest = db.get_latest_portfolio("paper")
        if not latest or not latest.get("positions"):
            return jsonify([])
        try:
            from kiwoom_api import KiwoomAPI
            api = KiwoomAPI()
            result = []
            for pos in latest["positions"]:
                try:
                    info = api.get_current_price(pos["stock_code"])
                    price = info.get("price", 0)
                    pnl = (price - pos["avg_price"]) / pos["avg_price"] * 100 if pos["avg_price"] else 0
                    result.append({
                        "stock_code": pos["stock_code"],
                        "stock_name": pos.get("stock_name", pos["stock_code"]),
                        "quantity": pos["quantity"],
                        "avg_price": pos["avg_price"],
                        "current_price": price,
                        "pnl": round(pnl, 2),
                        "market_value": price * pos["quantity"],
                        "change_rate": info.get("change_rate", 0),
                    })
                except Exception:
                    pass
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)})

    @app.route("/api/nxt")
    def api_nxt():
        """최근 NXT 가격 데이터."""
        rows = db.execute_raw(
            """SELECT stock_code, date, time, nxt_price, krx_close,
                      nxt_vs_close_pct, spread_pct
               FROM nxt_prices
               ORDER BY date DESC, time DESC LIMIT 100""")
        return jsonify(rows)

    @app.route("/api/stats")
    def api_stats():
        stats = db.get_db_stats()
        return jsonify(stats)

    print(f"대시보드 시작: http://localhost:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=False)


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NN Trading Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #0f0f23; color: #e0e0e0; }
.header { background: #1a1a3e; padding: 20px 30px; border-bottom: 2px solid #333; }
.header h1 { font-size: 24px; color: #ffd700; }
.header .sub { color: #888; font-size: 14px; margin-top: 4px; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 20px; }
.card { background: #1a1a3e; border-radius: 12px; padding: 20px; border: 1px solid #333; }
.card h3 { color: #aaa; font-size: 13px; text-transform: uppercase; margin-bottom: 8px; }
.card .value { font-size: 28px; font-weight: bold; }
.card .value.positive { color: #00e676; }
.card .value.negative { color: #ff5252; }
.chart-container { background: #1a1a3e; border-radius: 12px; padding: 20px; border: 1px solid #333; margin-bottom: 20px; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; padding: 10px; color: #888; border-bottom: 1px solid #333; font-size: 13px; }
td { padding: 10px; border-bottom: 1px solid #222; font-size: 14px; }
.positive { color: #00e676; }
.negative { color: #ff5252; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
.tag-buy { background: #1b5e20; color: #69f0ae; }
.tag-sell { background: #b71c1c; color: #ff8a80; }
.section-title { font-size: 18px; color: #ffd700; margin: 20px 0 12px; }
#status { color: #4caf50; font-size: 12px; }
</style>
</head>
<body>
<div class="header">
  <h1>Neural Trading System</h1>
  <div class="sub">후회 기반 학습 모의투자 대시보드 <span id="status">Loading...</span></div>
</div>
<div class="container">
  <div class="grid" id="summary-cards"></div>
  <div class="chart-container">
    <canvas id="equityChart" height="300"></canvas>
  </div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px;">
    <div class="card">
      <div class="section-title">최근 매매</div>
      <table id="trades-table"><thead><tr><th>시간</th><th>종목</th><th>매매</th><th>수량</th><th>가격</th></tr></thead><tbody></tbody></table>
    </div>
    <div class="card">
      <div class="section-title">후회 로그</div>
      <table id="regrets-table"><thead><tr><th>날짜</th><th>교훈</th><th>후회</th></tr></thead><tbody></tbody></table>
    </div>
  </div>
  <div class="card" style="margin-top:16px;">
    <div class="section-title">보유 종목 <button onclick="loadLivePrices()" style="background:#333;color:#ffd700;border:1px solid #555;padding:4px 12px;border-radius:6px;cursor:pointer;font-size:12px;margin-left:10px;">실시간 갱신</button> <span id="live-status" style="color:#666;font-size:12px;"></span></div>
    <table id="positions-table"><thead><tr><th>종목</th><th>수량</th><th>평균가</th><th>현재가</th><th>수익률</th><th>평가금</th><th>오늘</th></tr></thead><tbody></tbody></table>
  </div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px;">
    <div class="card">
      <div class="section-title">예측 검증 (forecast)</div>
      <table id="forecast-table"><thead><tr><th>날짜</th><th>종목</th><th>예측</th><th>실제</th><th>수익률</th><th>적중</th></tr></thead><tbody></tbody></table>
    </div>
    <div class="card">
      <div class="section-title">NXT 야간가격</div>
      <table id="nxt-table"><thead><tr><th>종목</th><th>시간</th><th>NXT가</th><th>종가대비</th><th>스프레드</th></tr></thead><tbody></tbody></table>
    </div>
  </div>
</div>
<script>
let equityChart = null;

async function fetchJSON(url) {
  const r = await fetch(url);
  return r.json();
}

function fmt(n) { return n ? n.toLocaleString() : '0'; }

async function loadDashboard() {
  try {
    const [portfolio, history, trades, regrets] = await Promise.all([
      fetchJSON('/api/portfolio'),
      fetchJSON('/api/portfolio/history'),
      fetchJSON('/api/trades'),
      fetchJSON('/api/regrets'),
    ]);

    document.getElementById('status').textContent =
      portfolio.error ? 'No data' : 'Updated: ' + new Date().toLocaleTimeString();

    // Summary cards
    if (!portfolio.error) {
      const cards = document.getElementById('summary-cards');
      const cum = portfolio.cumulative_return || 0;
      cards.innerHTML = `
        <div class="card"><h3>Total Value</h3><div class="value">${fmt(portfolio.total_value)}원</div></div>
        <div class="card"><h3>Cumulative Return</h3><div class="value ${cum >= 0 ? 'positive' : 'negative'}">${cum >= 0 ? '+' : ''}${cum.toFixed(2)}%</div></div>
        <div class="card"><h3>Cash</h3><div class="value">${fmt(portfolio.cash)}원</div></div>
        <div class="card"><h3>Positions</h3><div class="value">${portfolio.num_positions || 0}종목</div></div>
        <div class="card"><h3>Sharpe Ratio</h3><div class="value">${(portfolio.sharpe_ratio || 0).toFixed(2)}</div></div>
        <div class="card"><h3>Max Drawdown</h3><div class="value negative">${(portfolio.max_drawdown || 0).toFixed(1)}%</div></div>
        <div class="card"><h3>Win Rate</h3><div class="value">${((portfolio.win_rate || 0) * 100).toFixed(1)}%</div></div>
      `;
      // Positions
      const pTbody = document.querySelector('#positions-table tbody');
      pTbody.innerHTML = (portfolio.positions || []).map(p => `
        <tr>
          <td>${p.stock_name}</td><td>${fmt(p.quantity)}</td>
          <td>${fmt(Math.round(p.avg_price))}</td><td>${fmt(Math.round(p.current_price))}</td>
          <td class="${p.pnl >= 0 ? 'positive' : 'negative'}">${p.pnl >= 0 ? '+' : ''}${p.pnl.toFixed(1)}%</td>
          <td>${fmt(p.market_value)}</td>
        </tr>
      `).join('');
    }

    // Equity chart
    if (history.length > 0) {
      const labels = history.map(h => h.date);
      const values = history.map(h => h.cumulative_return);
      if (equityChart) equityChart.destroy();
      equityChart = new Chart(document.getElementById('equityChart'), {
        type: 'line',
        data: {
          labels,
          datasets: [{
            label: 'Cumulative Return (%)',
            data: values,
            borderColor: '#ffd700',
            backgroundColor: 'rgba(255,215,0,0.1)',
            fill: true, tension: 0.3, pointRadius: 0,
          }]
        },
        options: {
          responsive: true, maintainAspectRatio: false,
          scales: {
            x: { ticks: { color: '#666', maxTicksLimit: 20 }, grid: { color: '#222' } },
            y: { ticks: { color: '#666' }, grid: { color: '#222' } }
          },
          plugins: { legend: { labels: { color: '#ccc' } } }
        }
      });
    }

    // Trades
    const tTbody = document.querySelector('#trades-table tbody');
    tTbody.innerHTML = trades.slice(-20).reverse().map(t => `
      <tr>
        <td>${t.executed_at || ''}</td>
        <td>${t.stock_name}</td>
        <td><span class="tag ${t.side === 'buy' ? 'tag-buy' : 'tag-sell'}">${t.side.toUpperCase()}</span></td>
        <td>${fmt(t.quantity)}</td>
        <td>${fmt(t.price)}</td>
      </tr>
    `).join('');

    // Regrets
    const rTbody = document.querySelector('#regrets-table tbody');
    rTbody.innerHTML = regrets.slice(0, 15).map(r => `
      <tr>
        <td>${r.trade_date}</td>
        <td>${r.lesson || ''}</td>
        <td class="negative">${(r.regret_score || 0).toFixed(2)}</td>
      </tr>
    `).join('');

  } catch (e) {
    document.getElementById('status').textContent = 'Error: ' + e.message;
  }
}

const CLASS_NAMES = ['하락','보합','상승'];

async function loadLivePrices() {
  document.getElementById('live-status').textContent = '갱신중...';
  try {
    const data = await fetchJSON('/api/live_prices');
    if (data.error) { document.getElementById('live-status').textContent = data.error; return; }
    const tbody = document.querySelector('#positions-table tbody');
    let totalValue = 0;
    tbody.innerHTML = data.map(p => {
      totalValue += p.market_value;
      return `<tr>
        <td>${p.stock_name}</td><td>${fmt(p.quantity)}</td>
        <td>${fmt(p.avg_price)}</td><td>${fmt(p.current_price)}</td>
        <td class="${p.pnl >= 0 ? 'positive' : 'negative'}">${p.pnl >= 0 ? '+' : ''}${p.pnl.toFixed(1)}%</td>
        <td>${fmt(p.market_value)}</td>
        <td class="${p.change_rate >= 0 ? 'positive' : 'negative'}">${p.change_rate >= 0 ? '+' : ''}${p.change_rate.toFixed(1)}%</td>
      </tr>`;
    }).join('');
    document.getElementById('live-status').textContent = 'Updated ' + new Date().toLocaleTimeString();
  } catch(e) { document.getElementById('live-status').textContent = e.message; }
}

async function loadForecast() {
  try {
    const data = await fetchJSON('/api/predictions/forecast');
    const tbody = document.querySelector('#forecast-table tbody');
    tbody.innerHTML = data.map(r => {
      const hit = r.actual_class === null ? '-' : (r.predicted_class === r.actual_class ? 'O' : 'X');
      const hitClass = hit === 'O' ? 'positive' : (hit === 'X' ? 'negative' : '');
      return `<tr>
        <td>${r.date}</td><td>${r.stock_code}</td>
        <td>${CLASS_NAMES[r.predicted_class] || '?'} (${(r.confidence*100).toFixed(0)}%)</td>
        <td>${r.actual_class !== null ? CLASS_NAMES[r.actual_class] : '대기'}</td>
        <td class="${(r.actual_return||0) >= 0 ? 'positive' : 'negative'}">${r.actual_return !== null ? (r.actual_return >= 0 ? '+' : '') + r.actual_return.toFixed(1) + '%' : '-'}</td>
        <td class="${hitClass}">${hit}</td>
      </tr>`;
    }).join('');
  } catch(e) {}
}

async function loadNxt() {
  try {
    const data = await fetchJSON('/api/nxt');
    const tbody = document.querySelector('#nxt-table tbody');
    // 종목별 최신만
    const seen = new Set();
    const unique = data.filter(r => { if (seen.has(r.stock_code)) return false; seen.add(r.stock_code); return true; });
    tbody.innerHTML = unique.slice(0, 15).map(r => `<tr>
      <td>${r.stock_code}</td><td>${r.date} ${r.time}</td>
      <td>${fmt(r.nxt_price)}</td>
      <td class="${(r.nxt_vs_close_pct||0) >= 0 ? 'positive' : 'negative'}">${(r.nxt_vs_close_pct||0) >= 0 ? '+' : ''}${(r.nxt_vs_close_pct||0).toFixed(1)}%</td>
      <td>${(r.spread_pct||0).toFixed(2)}%</td>
    </tr>`).join('');
  } catch(e) {}
}

loadDashboard();
loadForecast();
loadNxt();
setInterval(loadDashboard, 30000);
setInterval(loadForecast, 60000);
setInterval(loadNxt, 120000);
</script>
</body>
</html>"""
