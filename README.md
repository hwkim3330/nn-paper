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

## TimesFM-3 (optional)

`models/timesfm_forecaster.py` wraps Google's [TimesFM 3.0](https://github.com/google-research/timesfm) time-series foundation model ([`google/timesfm-3.0-pytorch`](https://huggingface.co/google/timesfm-3.0-pytorch), 330M). It runs zero-shot and needs no training. It provides:

- quantile forecasts (q10…q90) of 1/3/5/15-day log returns and 5/20-day realized volatility, with optional past-only covariates (KOSPI, volume);
- `tfm_*` features for the existing pipeline, such as `tfm_p_up_1d`, `tfm_exp_ret_5d`, `tfm_spread_5d`, `tfm_rv_5d` and `tfm_tb_p_up`;
- a standalone triple-barrier 3-class signal in `PaperEngine`'s prediction format.

```bash
# Apple Silicon → MLX backend; elsewhere → PyTorch (CUDA/CPU). Tested with Python 3.12 via uv.
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements-timesfm.txt
```

```python
from models.timesfm_forecaster import TimesFMForecaster
fc = TimesFMForecaster()                    # downloads ~1.3 GB weights on first use
feats = fc.features_for_stock(prices)       # prices = db.get_daily_prices(code) (≥ 512 bars recommended)
sig = fc.signal(prices, stock_code=code)    # {predicted_class, confidence, buy_prob, ...}
```

The pipeline does not need TimesFM. Nothing imports it at module load time, and without it everything else runs as before (`python -m unittest tests.test_timesfm_optional`).

**License restriction:** the TimesFM 3.0 weights are under `timesfm-non-commercial-license-v1.0`, which allows **non-commercial, non-production use only**. Use them for research and paper trading only. `trading/live_engine.py` marks the process as live, and from then on it refuses to load TimesFM and refuses models with `tfm_*` features.

**Evaluation:** [`docs/timesfm_eval.md`](docs/timesfm_eval.md) is a walk-forward test on 50 KOSPI large caps using public data, 2019–2026, with 89,642 test origins. Zero-shot TimesFM-3 **did not beat naive baselines on returns**: its MAE is worse than a zero-return forecast at every horizon, its CRPS is worse or tied, and its directional accuracy is about 50%. Adding its features to LightGBM did not improve 3-class macro-F1 or the cost-aware backtest. It is competitive on **volatility**: it had the best 5-day RV MAE, slightly ahead of HAR-RV, but is not better on QLIKE or at 20 days. So it is useful as an uncertainty/volatility input, not as a return signal.

**Tech:** Python, PyTorch, LightGBM, SQLite, Flask, Kiwoom REST API.

**Status:** personal research project, v2 in progress.
