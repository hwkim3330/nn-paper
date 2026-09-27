# nn-paper — ML 기반 주식 모의투자 연구 (ML paper-trading research for Korean stocks)

A research system for Korean equities (KOSPI200 / KOSDAQ150 + watchlist) that collects market data, builds features, trains models and runs them in a **paper-trading** loop. It is meant for experiments and simulated trading, not financial advice.

KOSPI/KOSDAQ 종목 데이터를 수집해 피처를 만들고, 모델을 학습해 **모의투자**로 검증하는 연구용 시스템입니다.

## Pipeline

1. **Collect** (`collector/`) — prices, order book, investor flows, sector, rankings via the Kiwoom REST API; scheduled jobs; SQLite storage (`db/`).
2. **Features** (`features/`) — technical, market, flow and order-book features with normalization (v2: 93 features).
3. **Labels & models** (`models/`) — 3-class triple-barrier labels (down / flat / up); LightGBM, an attention-LSTM over 60-day sequences, and an ensemble; evaluation tools.
4. **Trading** (`trading/`) — paper-trading engine (1-day and 3-day swing mix, fees and slippage, daily loss limit), portfolio, risk manager, and a "regret" log that records what the better trade would have been 5 days later. `live_engine.py` exists but real trading is gated behind paper results.
5. **Dashboard** (`dashboard/app.py`) — Flask dashboard (port 8502); Telegram reports.

## Usage

```bash
pip install torch lightgbm numpy flask
# create .env with KIWOOM_APP_KEY, KIWOOM_APP_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
./run.sh setup         # sync → features → train
./run.sh paper         # start paper trading
./run.sh backtest --days 60
./run.sh dashboard
```

Some modules (`kiwoom_rest_api`, `telegram_bot`, `market_news`) and the price database come from a companion `stock-bot` project that is not included here (`scripts/sync_from_stockbot.py` syncs from it).

Credentials are read from environment variables / `.env` only (see `config.py`); `.env` is git-ignored.

**Tech:** Python, PyTorch, LightGBM, SQLite, Flask, Kiwoom REST API.

**Status:** personal research project, v2 in progress.
