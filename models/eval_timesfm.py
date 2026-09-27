"""
TimesFM-3 walk-forward 평가 (공개 일봉 데이터, look-ahead 없음)

비교 대상
  - 나이브: zero-return, last-return, historical-drift (+ always-up 방향)
  - LightGBM (repo LGBMModel + repo 기술적 지표 피처 + KOSPI 컨텍스트 = 축소 피처셋)
  - TimesFM-3 zero-shot (단변량 / KOSPI·거래량 공변량 / 수익률 시계열 입력 ablation)
  - LightGBM + TimesFM 파생 피처
  - 변동성: persistence, 60일 역사적, EWMA(0.94), HAR-RV, TimesFM-3

지표: 방향 정확도, AUC, MAE, CRPS(분위수 기반), rank IC, 3-class macro-F1,
      비용 반영 long-only 백테스트 (Sharpe, MDD) — 연도별 창 + 블록 부트스트랩 95% CI

No look-ahead 규칙
  - 예측 시점 t의 입력은 C[..t], 거래량[..t], KOSPI[..t] 만 사용
  - 타깃은 t+1..t+h
  - 학습 모델(LightGBM, HAR, 보정 LR)은 test 연도 시작 전에 라벨 구간(t+15)이 끝난 표본만 사용 (purge)
  - 백테스트는 t 종가 신호 → t+1 시가 체결 (open-to-open 수익률)

사용:
  .venv/bin/python -m models.eval_timesfm                # 전체 (데이터 fetch → 예측 캐시 → 평가)
  .venv/bin/python -m models.eval_timesfm --tickers 8 --no-cov   # 빠른 확인
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.timesfm_forecaster import (  # noqa: E402
    QUANTILE_LEVELS, barrier_probs, crps_from_quantiles, prob_above,
)

logger = logging.getLogger("eval_timesfm")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "public"
DOCS = ROOT / "docs"

# KOSPI 대형주 50개 (config.KOSPI200_TOP 40개 + 10개). 현재 시점 대형주 → 생존편향 있음.
EXTRA_TICKERS = {
    "000100": "유한양행", "011170": "롯데케미칼", "010140": "삼성중공업",
    "024110": "기업은행", "047050": "포스코인터내셔널", "009830": "한화솔루션",
    "042660": "한화오션", "012450": "한화에어로스페이스", "267250": "HD현대",
    "138040": "메리츠금융지주",
}
INDEX_CODE = "KS11"

CONTEXT_LEN = 512
MIN_INDEX = CONTEXT_LEN + 25     # RV20 컨텍스트까지 NaN 없게
TB_H = 15                        # triple barrier max_holding_days (nn_config와 동일)
Z90 = 1.2815515655446004


# ============================================================
# 데이터
# ============================================================

def universe() -> dict:
    from config import KOSPI200_TOP
    u = dict(KOSPI200_TOP)
    u.update(EXTRA_TICKERS)
    return u


def fetch(codes: list[str], refresh: bool = False) -> dict:
    """FinanceDataReader 일봉 (수정주가) → data/public/prices/*.pkl 캐시."""
    import pandas as pd
    import FinanceDataReader as fdr
    out = {}
    d = CACHE / "prices"
    d.mkdir(parents=True, exist_ok=True)
    for code in codes + [INDEX_CODE]:
        p = d / f"{code}.pkl"
        if p.exists() and not refresh:
            out[code] = pd.read_pickle(p)
            continue
        for attempt in range(3):
            try:
                df = fdr.DataReader(code, "2010-01-01")
                break
            except Exception as e:  # 네트워크
                logger.warning("fetch %s 실패 (%s) 재시도", code, e)
                time.sleep(2)
        else:
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.to_pickle(p)
        out[code] = df
    return out


def build_panel(raw: dict, codes: list[str]) -> dict:
    """종목별 배열 (KOSPI와 날짜 교집합). 0/음수 가격, 거래정지(거래량 0) 행 제거."""
    idx = raw[INDEX_CODE]
    panel = {}
    for code in codes:
        if code not in raw:
            continue
        df = raw[code]
        df = df[(df["Close"] > 0) & (df["Open"] > 0) & (df["Volume"] > 0)]
        j = df.join(idx[["Close"]].rename(columns={"Close": "Index"}), how="inner")
        if len(j) < MIN_INDEX + 300:
            logger.info("%s: 이력 부족 (%d일) — 제외", code, len(j))
            continue
        panel[code] = {
            "dates": np.array([d.strftime("%Y-%m-%d") for d in j.index]),
            "open": j["Open"].to_numpy(float), "high": j["High"].to_numpy(float),
            "low": j["Low"].to_numpy(float), "close": j["Close"].to_numpy(float),
            "volume": j["Volume"].to_numpy(float), "index": j["Index"].to_numpy(float),
        }
    return panel


class _PanelDB:
    """FeatureEngine.get_target_labels 재사용을 위한 최소 DB 어댑터."""

    def __init__(self, s):
        self.rows = [{"date": s["dates"][i], "open": s["open"][i], "high": s["high"][i],
                      "low": s["low"][i], "close": s["close"][i], "volume": s["volume"][i]}
                     for i in range(len(s["dates"]))]

    def get_daily_prices(self, stock_code, end_date=None, limit=None):
        return self.rows


def _trail_rv(close, w):
    r2 = np.concatenate([[np.nan], np.diff(np.log(close)) ** 2])
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(r2))])
    out = np.full(len(close), np.nan)
    i = np.arange(w, len(close))
    out[i] = np.sqrt((cs[i + 1] - cs[i + 1 - w]) / w)
    return out


def build_samples(panel: dict) -> dict:
    """
    (종목, t) 표본 + 타깃 + 나이브 입력 + LightGBM 피처.
    t 조건: t ≥ MIN_INDEX, t+TB_H < n (15일 라벨 완결).
    """
    from features.engine import FeatureEngine
    from features.technical import compute_technical_features

    rows = []
    feats = []
    for code, s in panel.items():
        c, n = s["close"], len(s["close"])
        lr = np.concatenate([[np.nan], np.diff(np.log(c))])
        rv5, rv20, rv22, rv60 = (_trail_rv(c, w) for w in (5, 20, 22, 60))
        ix = s["index"]
        ilr = np.concatenate([[np.nan], np.diff(np.log(ix))])
        # EWMA(0.94) 분산 (t까지)
        ew = np.full(n, np.nan)
        v = np.nanvar(lr[1:61])
        for t in range(1, n):
            v = 0.94 * v + 0.06 * lr[t] ** 2
            ew[t] = v
        eng = FeatureEngine(_PanelDB(s))
        ts = list(range(MIN_INDEX, n - TB_H))
        labels = eng.get_target_labels(code, [s["dates"][t] for t in ts])
        prices = _PanelDB(s).rows
        for t, lab in zip(ts, labels):
            if lab is None:
                continue
            hist = lr[t - 249:t + 1]
            # ATR(14) 단순평균 — repo triple barrier와 동일
            tr = [max(s["high"][k] - s["low"][k], abs(s["high"][k] - c[k - 1]),
                      abs(s["low"][k] - c[k - 1])) for k in range(t - 13, t + 1)]
            barrier = math.log1p(1.5 * (sum(tr) / 14) / c[t])
            fut = lr[t + 1:t + TB_H + 1]
            rv20_f = math.sqrt(np.mean(lr[t + 1:t + 21] ** 2)) if t + 20 < n else np.nan
            rows.append({
                "code": code, "t": t, "date": s["dates"][t],
                "y1": fut[0], "y3": fut[:3].sum(), "y5": fut[:5].sum(), "y15": fut.sum(),
                "rv5_f": math.sqrt(np.mean(fut[:5] ** 2)), "rv20_f": rv20_f,
                "tb": int(lab), "barrier": barrier,
                "mu": float(np.mean(hist)), "sd": float(np.std(hist, ddof=1)),
                "last1": lr[t], "last3": lr[t - 2:t + 1].sum(), "last5": lr[t - 4:t + 1].sum(),
                "last15": lr[t - 14:t + 1].sum(),
                "rv_d": abs(lr[t]), "rv5": rv5[t], "rv20": rv20[t], "rv22": rv22[t],
                "rv60": rv60[t], "ewma_sd": math.sqrt(ew[t]),
            })
            f = compute_technical_features(prices[t - 149:t + 1])
            f.update({
                "mkt_ret_1d": ilr[t], "mkt_ret_5d": ilr[t - 4:t + 1].sum(),
                "mkt_ret_20d": ilr[t - 19:t + 1].sum(),
                "mkt_rv_20d": float(np.sqrt(np.mean(ilr[t - 19:t + 1] ** 2))),
                "rel_ret_20d": lr[t - 19:t + 1].sum() - ilr[t - 19:t + 1].sum(),
            })
            feats.append(f)
        logger.info("%s: 표본 %d", code, len(ts))

    names = sorted(set().union(*[f.keys() for f in feats]))
    X = np.array([[f.get(k, 0.0) for k in names] for f in feats], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    cols = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    cols["X"], cols["feature_names"] = X, names
    return cols


# ============================================================
# TimesFM 예측 (캐시)
# ============================================================

def run_timesfm(panel: dict, S: dict, variant: str, mask: np.ndarray, fc, batch: int = 256):
    """variant: ret (로그가격, h=15) | ret_cov (공변량) | retser (수익률 시계열, h=5) | rv5 | rv20"""
    from models.timesfm_forecaster import log_price_context
    L = CONTEXT_LEN
    idx = np.where(mask)[0]
    H = {"ret": TB_H, "ret_cov": TB_H, "retser": 5, "rv5": 5, "rv20": 20}[variant]
    out = np.full((len(S["t"]), H, 9), np.nan, dtype=np.float32)
    rv_cache = {}
    t0 = time.time()
    for s0 in range(0, len(idx), batch):
        sel = idx[s0:s0 + batch]
        ctx, cov = [], []
        for i in sel:
            s, t = panel[S["code"][i]], int(S["t"][i])
            c = s["close"][:t + 1]
            if variant in ("ret", "ret_cov"):
                ctx.append(log_price_context(c, L))
                if variant == "ret_cov":
                    ix = np.log(s["index"][t - L + 1:t + 1]); ix -= ix[-1]
                    lv = np.log1p(s["volume"][t - L + 1:t + 1])
                    lv = (lv - lv.mean()) / (lv.std() + 1e-9)
                    cov.append(np.stack([ix, lv]).astype(np.float32))
            elif variant == "retser":
                ctx.append((np.diff(np.log(s["close"][t - L:t + 1])) * 100).astype(np.float32))
            else:
                w = 5 if variant == "rv5" else 20
                key = (S["code"][i], w)
                if key not in rv_cache:
                    rv_cache[key] = _trail_rv(s["close"], w)
                ctx.append(rv_cache[key][t - L + 1:t + 1].astype(np.float32))
        q = fc.predict_quantiles(ctx, H, cov if variant == "ret_cov" else None)
        out[sel] = q
        if (s0 // batch) % 20 == 0:
            done = s0 + len(sel)
            logger.info("TimesFM[%s] %d/%d (%.0f/s)", variant, done, len(idx),
                        done / max(time.time() - t0, 1e-9))
    if variant == "retser":
        out = out / 100.0
    return out, time.time() - t0


# ============================================================
# 지표
# ============================================================

def gauss_q(mu, sd):
    z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0, 0.2533, 0.5244, 0.8416, 1.2816])
    return mu[:, None] + sd[:, None] * z[None, :]


def _block_indices(n, n_boot, block, seed):
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    return (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]


def block_bootstrap(per_date: np.ndarray, n_boot: int = 1000, block: int = 20, seed: int = 0):
    """날짜별 값(시간순)의 평균에 대한 moving-block bootstrap 95% CI."""
    x = np.asarray(per_date, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < block * 2:
        return (float("nan"), float("nan"))
    means = x[_block_indices(n, n_boot, block, seed)].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def per_date_mean(dates, values):
    order = np.argsort(dates, kind="stable")
    d, v = dates[order], values[order]
    uniq, start = np.unique(d, return_index=True)
    return uniq, np.add.reduceat(v, start) / np.diff(np.append(start, len(d)))


def per_date_ic(dates, score, y):
    """날짜별 횡단면 Spearman rank IC (종목 10개 이상인 날짜만)."""
    from scipy.stats import rankdata
    order = np.argsort(dates, kind="stable")
    d, sc, yy = dates[order], score[order], y[order]
    uniq, start = np.unique(d, return_index=True)
    ends = np.append(start[1:], len(d))
    ics = np.full(len(uniq), np.nan)
    for k, (a, b) in enumerate(zip(start, ends)):
        if b - a >= 10 and np.std(sc[a:b]) > 0:
            ics[k] = np.corrcoef(rankdata(sc[a:b]), rankdata(yy[a:b]))[0, 1]
    ok = ~np.isnan(ics)
    return uniq[ok], ics[ok]


def sharpe(r, rf=0.035):
    r = np.asarray(r)
    ex = r - rf / 252
    return float(ex.mean() / ex.std(ddof=1) * math.sqrt(252)) if ex.std() > 0 else 0.0


def sharpe_ci(r, n_boot=1000, block=20, seed=0, rf=0.035):
    ex = np.asarray(r) - rf / 252
    b = ex[_block_indices(len(ex), n_boot, block, seed)]
    vals = b.mean(1) / b.std(1, ddof=1) * math.sqrt(252)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def macro_f1(y, p):
    from sklearn.metrics import f1_score
    return float(f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0))


# ============================================================
# 학습 모델 (walk-forward, 연도별 재학습)
# ============================================================

def fold_masks(S, year: int, panel: dict):
    """test = year의 origin, train = label 종료일(t+15)이 test 시작 전인 origin."""
    test = np.array([d[:4] == str(year) for d in S["date"]])
    if not test.any():
        return None, None
    start = f"{year}-01-01"
    lab_end = np.array([panel[c]["dates"][t + TB_H] for c, t in zip(S["code"], S["t"])])
    train = lab_end < start
    return train, test


def train_lgbm(X, y, dates, tr, te):
    from models.lgbm_model import LGBMModel
    from nn_config import LGBM_PARAMS
    params = dict(LGBM_PARAMS)
    params["n_jobs"] = 12
    tr_idx = np.where(tr)[0]
    d_tr = dates[tr_idx]
    val_start = np.unique(d_tr)[-60]
    fit = tr_idx[d_tr < val_start]
    val = tr_idx[d_tr >= val_start]
    # repo Trainer의 class-uniqueness 가중치 (후회/시간감쇠는 DB 의존이라 생략)
    cnt = np.bincount(y[fit], minlength=3).astype(float)
    w = (len(fit) / (3 * np.maximum(cnt, 1)))[y[fit]]
    m = LGBMModel(params)
    m.train(X[fit], y[fit], sample_weight=w, X_val=X[val], y_val=y[val])
    return m.predict_proba(X[te])


def fit_lr(F, y, tr, te):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(F[tr], y[tr])
    return lr.predict_proba(F[te])


def fit_har(S, tr, te, target):
    """HAR-RV (Corsi 2009), 수준 OLS: RV_h(t+) ~ 1 + |r_t| + RV5 + RV22."""
    A = np.column_stack([np.ones(len(S["t"])), S["rv_d"], S["rv5"], S["rv22"]])
    ok = tr & ~np.isnan(S[target])
    beta, *_ = np.linalg.lstsq(A[ok], S[target][ok], rcond=None)
    return np.maximum(A[te] @ beta, 1e-5)


# ============================================================
# 백테스트
# ============================================================

def backtest(panel, S, score: np.ndarray, mask: np.ndarray, hold: int, k: int = 10,
             threshold: float = 0.0):
    """
    long-only, hold일마다 재조정, 점수 상위 k개 (score > threshold), 동일가중 1/k (나머지 현금).
    t 종가 신호 → t+1 시가 체결, open-to-open 일수익률. 비용: nn_config 모의투자 설정.
    """
    from nn_config import (PAPER_COMMISSION_BUY, PAPER_COMMISSION_SELL,
                           PAPER_TAX_RATE, PAPER_SLIPPAGE_BPS)
    c_buy = PAPER_COMMISSION_BUY + PAPER_SLIPPAGE_BPS / 1e4
    c_sell = PAPER_COMMISSION_SELL + PAPER_TAX_RATE + PAPER_SLIPPAGE_BPS / 1e4

    dates = np.unique(S["date"][mask])
    # 종목별 open-to-open 수익률 (date → t+1 시가 기준 다음날)
    o2o = {}
    for code, s in panel.items():
        o = s["open"]
        o2o[code] = {s["dates"][t]: o[t + 2] / o[t + 1] - 1 for t in range(len(o) - 2)}
    by_date = {}
    for i in np.where(mask)[0]:
        by_date.setdefault(S["date"][i], []).append(i)

    w = {}
    daily = []
    for j, d in enumerate(dates):
        cost = 0.0
        if j % hold == 0:
            cand = [i for i in by_date.get(d, []) if score[i] > threshold]
            cand.sort(key=lambda i: -score[i])
            new = {S["code"][i]: 1.0 / k for i in cand[:k]}
            for code in set(w) | set(new):
                dw = new.get(code, 0.0) - w.get(code, 0.0)
                cost += dw * c_buy if dw > 0 else -dw * c_sell
            w = new
        r = sum(wt * o2o[code].get(d, 0.0) for code, wt in w.items())
        daily.append(r - cost)
        # 가중치 drift
        if w:
            tot = 1.0 + r
            w = {code: wt * (1 + o2o[code].get(d, 0.0)) / tot for code, wt in w.items()}
    daily = np.array(daily)
    eq = np.cumprod(1 + daily)
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min() * 100)
    years = len(daily) / 252
    return {
        "sharpe": sharpe(daily), "sharpe_ci": sharpe_ci(daily),
        "cagr_pct": float((eq[-1] ** (1 / years) - 1) * 100), "mdd_pct": mdd,
        "total_return_pct": float((eq[-1] - 1) * 100), "n_days": len(daily),
    }, daily


# ============================================================
# 메인
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=0, help="앞에서 N개 종목만 (0=전체)")
    ap.add_argument("--test-start", type=int, default=2019)
    ap.add_argument("--no-cov", action="store_true", help="공변량 변형 생략")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=str(DOCS / "timesfm_eval_results.json"))
    ap.add_argument("--report", action="store_true", help="기존 결과 JSON → 마크다운 표 출력만")
    args = ap.parse_args()
    if args.report:
        R = json.loads(Path(args.out).read_text())
        # JSON은 연도 키를 문자열로 저장
        print(render_markdown(R))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    t_start = time.time()
    timings = {}

    uni = universe()
    codes = list(uni)[: args.tickers or None]
    raw = fetch(codes, args.refresh)
    panel = build_panel(raw, codes)
    tag = f"n{len(panel)}"
    sp = CACHE / f"samples_{tag}.pkl"
    if sp.exists() and not args.refresh:
        S = pickle.loads(sp.read_bytes())
    else:
        t0 = time.time()
        S = build_samples(panel)
        sp.write_bytes(pickle.dumps(S))
        timings["features_s"] = time.time() - t0
    N = len(S["t"])
    dates = S["date"]
    years = sorted({int(d[:4]) for d in dates if int(d[:4]) >= args.test_start})
    test_all = np.array([int(d[:4]) >= args.test_start for d in dates])
    logger.info("표본 %d, 종목 %d, test 연도 %s", N, len(panel), years)

    # ---- TimesFM ----
    variants = ["ret", "rv5", "rv20", "retser"] + ([] if args.no_cov else ["ret_cov"])
    Q = {}
    fc = None
    for v in variants:
        cp = CACHE / f"tfm_{v}_{tag}.npy"
        m = np.ones(N, bool) if v in ("ret", "rv5") else test_all
        if cp.exists() and not args.refresh:
            Q[v] = np.load(cp)
            continue
        if fc is None:
            from models.timesfm_forecaster import TimesFMForecaster
            t0 = time.time()
            fc = TimesFMForecaster(batch_size=256)
            timings["tfm_load_s"] = time.time() - t0
        Q[v], el = run_timesfm(panel, S, v, m, fc)
        timings[f"tfm_{v}_s"] = el
        timings[f"tfm_{v}_n"] = int(m.sum())
        np.save(cp, Q[v])

    R = {"setup": {}, "returns": {}, "returns_by_year": {}, "vol": {}, "vol_by_year": {},
         "cls": {}, "cls_by_year": {}, "backtest": {}, "backtest_by_year": {}}

    # ---- 학습 모델: LightGBM / 보정 / HAR (연도별 walk-forward) ----
    qret = Q["ret"]
    X, names = S["X"], S["feature_names"]
    y_tb = S["tb"]
    tfm_feats = []
    for h in (1, 3, 5, 15):
        qh = qret[:, h - 1, :]
        tfm_feats += [prob_above(qh, 0.0), qh[:, 4], qh[:, -1] - qh[:, 0],
                      (qh[:, -1] - qh[:, 4]) - (qh[:, 4] - qh[:, 0])]
    tb_tfm = barrier_probs(qret, S["barrier"])
    tb_hist = barrier_probs(
        np.stack([gauss_q(S["mu"] * k, S["sd"] * math.sqrt(k)) for k in range(1, TB_H + 1)], 1),
        S["barrier"])
    rv5_med = Q["rv5"][:, 4, 4]
    tfm_feats += [tb_tfm[:, 0], tb_tfm[:, 2], rv5_med]
    F_tfm = np.column_stack(tfm_feats).astype(np.float32)
    X_plus = np.hstack([X, F_tfm])

    P = {k: np.full((N, 3), np.nan) for k in
         ("lgbm", "lgbm_tfm", "tfm_calib", "hist_calib")}
    har5 = np.full(N, np.nan)
    har20 = np.full(N, np.nan)
    majority = np.full(N, -1)
    t0 = time.time()
    for Y in years:
        tr, tst = fold_masks(S, Y, panel)
        if tst is None or tr.sum() < 1000:
            continue
        P["lgbm"][tst] = train_lgbm(X, y_tb, dates, tr, tst)
        P["lgbm_tfm"][tst] = train_lgbm(X_plus, y_tb, dates, tr, tst)
        F1 = np.column_stack([tb_tfm[:, 0], tb_tfm[:, 2], F_tfm[:, 12], F_tfm[:, 14]])
        P["tfm_calib"][tst] = fit_lr(F1, y_tb, tr, tst)
        F2 = np.column_stack([tb_hist[:, 0], tb_hist[:, 2], S["mu"], S["sd"]])
        P["hist_calib"][tst] = fit_lr(F2, y_tb, tr, tst)
        majority[tst] = np.bincount(y_tb[tr], minlength=3).argmax()
        # HAR: 타깃 구간이 test 시작 전에 끝난 표본만
        tr20 = tr & np.array([
            panel[c]["dates"][min(t + 20, len(panel[c]["dates"]) - 1)] < f"{Y}-01-01"
            for c, t in zip(S["code"], S["t"])])
        har5[tst] = fit_har(S, tr, tst, "rv5_f")
        har20[tst] = fit_har(S, tr20, tst, "rv20_f")
        logger.info("fold %d: train %d, test %d", Y, tr.sum(), tst.sum())
    timings["walkforward_fit_s"] = time.time() - t0
    ev = ~np.isnan(P["lgbm"][:, 0])   # 평가 가능한 test 표본

    # ---- 수익률 예측 ----
    te = ev
    for h in (1, 3, 5, 15):
        y = S[f"y{h}"]
        mu, sd = S["mu"], S["sd"]
        models = {
            "zero": (np.zeros(N), gauss_q(np.zeros(N), sd * math.sqrt(h))),
            "last_return": (S[f"last{h}"], gauss_q(S[f"last{h}"], sd * math.sqrt(h))),
            "hist_drift": (mu * h, gauss_q(mu * h, sd * math.sqrt(h))),
            "timesfm": (qret[:, h - 1, 4], qret[:, h - 1, :]),
        }
        if "ret_cov" in Q:
            models["timesfm_cov"] = (Q["ret_cov"][:, h - 1, 4], Q["ret_cov"][:, h - 1, :])
        if h <= 5:
            cum = np.cumsum(Q["retser"][:, :h, 4], axis=1)[:, -1]
            models["timesfm_retseries"] = (cum, None)
        # 학습 모델: 3-class 확률 → 방향 점수 P(상승)-P(하락) (크기 예측이 아니므로 MAE/CRPS 제외)
        models["lgbm"] = (P["lgbm"][:, 2] - P["lgbm"][:, 0], None)
        models["lgbm_plus_timesfm"] = (P["lgbm_tfm"][:, 2] - P["lgbm_tfm"][:, 0], None)
        res = {}
        yr = {}
        for name, (point, q) in models.items():
            m = te & ~np.isnan(point)
            is_score = name.startswith("lgbm")
            ent = {"n": int(m.sum())}
            if not is_score:
                ud, loss = per_date_mean(dates[m], np.abs(point[m] - y[m]))
                ent.update({"mae": float(loss.mean()), "mae_ci": block_bootstrap(loss)})
            if q is not None:
                crps = crps_from_quantiles(q[m], y[m])
                _, cd = per_date_mean(dates[m], crps)
                ent.update({"crps": float(cd.mean()), "crps_ci": block_bootstrap(cd)})
                pup = prob_above(q[m], 0.0)
            else:
                pup = None
            if name != "zero":
                nz = y[m] != 0
                corr = (np.sign(point[m]) == np.sign(y[m])).astype(float)
                _, da = per_date_mean(dates[m][nz], corr[nz])
                ent.update({"dir_acc": float(corr[nz].mean()), "dir_acc_ci": block_bootstrap(da)})
                from sklearn.metrics import roc_auc_score
                sc = pup if pup is not None and np.std(pup) > 0 else point[m]
                ent["auc"] = float(roc_auc_score(y[m][nz] > 0, sc[nz]))
                icd, ic = per_date_ic(dates[m], point[m], y[m])
                ent.update({"ic": float(np.nanmean(ic)), "ic_ci": block_bootstrap(ic)})
                yr[name] = {}
                for Y in years:
                    my = np.array([d.startswith(str(Y)) for d in dates[m]])
                    yr[name][Y] = {
                        "dir_acc": float(corr[nz & my].mean()),
                        "ic": float(np.nanmean(ic[np.array([d.startswith(str(Y)) for d in icd])]))
                        if len(icd) else float("nan"),
                        "mae": float("nan") if is_score else float(np.abs(point[m][my] - y[m][my]).mean()),
                    }
            res[name] = ent
        # TimesFM − zero-return 의 날짜별 손실 차이 (음수 = TimesFM 우세)
        m = te
        for nm in ("timesfm", "timesfm_cov"):
            if nm in models:
                pt, qq = models[nm]
                _, dm = per_date_mean(dates[m], np.abs(pt[m] - y[m]) - np.abs(y[m]))
                _, dc = per_date_mean(dates[m], crps_from_quantiles(qq[m], y[m])
                                      - crps_from_quantiles(models["zero"][1][m], y[m]))
                res[f"{nm}_minus_zero"] = {"mae": float(dm.mean()), "mae_ci": block_bootstrap(dm),
                                           "crps": float(dc.mean()), "crps_ci": block_bootstrap(dc)}
        # always-up
        m = te & (y != 0)
        res["always_up"] = {"dir_acc": float((y[m] > 0).mean())}
        R["returns"][f"h{h}"] = res
        R["returns_by_year"][f"h{h}"] = yr

    # ---- 3-class ----
    cls_models = {
        "majority_class": majority,
        "hist_drift_barrier": tb_hist.argmax(1),
        "hist_drift_calibrated": np.nan_to_num(P["hist_calib"]).argmax(1),
        "timesfm_barrier_raw": tb_tfm.argmax(1),
        "timesfm_calibrated": np.nan_to_num(P["tfm_calib"]).argmax(1),
        "lgbm": np.nan_to_num(P["lgbm"]).argmax(1),
        "lgbm_plus_timesfm": np.nan_to_num(P["lgbm_tfm"]).argmax(1),
    }
    R["cls"]["class_dist_test"] = np.bincount(y_tb[ev], minlength=3).tolist()
    for name, pred in cls_models.items():
        from sklearn.metrics import f1_score
        per_date = []
        R["cls"][name] = {
            "macro_f1": macro_f1(y_tb[ev], pred[ev]),
            "accuracy": float((y_tb[ev] == pred[ev]).mean()),
            "pred_dist": np.bincount(pred[ev], minlength=3).tolist(),
        }
        # 연도별 macro-F1 → 창 간 분포
        R["cls_by_year"][name] = {}
        for Y in years:
            my = ev & np.array([d.startswith(str(Y)) for d in dates])
            if my.any():
                R["cls_by_year"][name][Y] = macro_f1(y_tb[my], pred[my])
        # bootstrap CI (월 블록)
        rng = np.random.default_rng(0)
        months = np.array([d[:7] for d in dates[ev]])
        um = np.unique(months)
        yt, yp = y_tb[ev], pred[ev]
        grp = {mm: np.where(months == mm)[0] for mm in um}
        vals = []
        for _ in range(300):
            pick = rng.integers(0, len(um) - 2, size=len(um) // 3)
            ii = np.concatenate([np.concatenate([grp[um[p]], grp[um[p + 1]], grp[um[p + 2]]])
                                 for p in pick])
            vals.append(f1_score(yt[ii], yp[ii], average="macro", labels=[0, 1, 2], zero_division=0))
        R["cls"][name]["macro_f1_ci"] = (float(np.percentile(vals, 2.5)),
                                         float(np.percentile(vals, 97.5)))

    # ---- 변동성 ----
    for w, tgt, har in ((5, "rv5_f", har5), (20, "rv20_f", har20)):
        yv = S[tgt]
        vm = {
            "persistence": S[f"rv{w}"], "hist_60d": S["rv60"], "ewma_094": S["ewma_sd"],
            "har_rv": har, "timesfm": Q[f"rv{w}"][:, w - 1, 4],
            # 분위수 평균 ≈ 예측분포 평균 (median은 우측 꼬리 분포에서 과소추정 경향)
            "timesfm_qmean": Q[f"rv{w}"][:, w - 1, :].mean(axis=1),
        }
        res, yr = {}, {}
        base = ev & ~np.isnan(yv)
        for name, pv in vm.items():
            m = base & ~np.isnan(pv)
            p, t_ = np.maximum(pv[m], 1e-5), yv[m]
            _, ae = per_date_mean(dates[m], np.abs(p - t_))
            ql = t_ ** 2 / p ** 2 - np.log(t_ ** 2 / p ** 2) - 1
            ql = np.where(t_ > 0, ql, np.nan)
            _, qd = per_date_mean(dates[m][~np.isnan(ql)], ql[~np.isnan(ql)])
            res[name] = {
                "n": int(m.sum()), "mae": float(ae.mean()), "mae_ci": block_bootstrap(ae),
                "rmse": float(np.sqrt(np.mean((p - t_) ** 2))),
                "qlike": float(qd.mean()), "qlike_ci": block_bootstrap(qd),
                "mz_r2": float(np.corrcoef(p, t_)[0, 1] ** 2),
            }
            yr[name] = {Y: float(np.abs(p - t_)[np.array([d.startswith(str(Y)) for d in dates[m]])].mean())
                        for Y in years}
        # TimesFM − 베이스라인 paired 차이 (날짜별 평균, 음수 = TimesFM 우세)
        def _loss(pv, kind):
            p = np.maximum(pv, 1e-5)
            if kind == "mae":
                return np.abs(p - yv)
            r = yv ** 2 / p ** 2
            return np.where(yv > 0, r - np.log(np.where(yv > 0, r, 1.0)) - 1, np.nan)
        m = base & ~np.isnan(har)
        for tf in ("timesfm", "timesfm_qmean"):
            for bl in ("har_rv", "ewma_094", "hist_60d"):
                for kind in ("mae", "qlike"):
                    dl = _loss(vm[tf], kind) - _loss(vm[bl], kind)
                    mm = m & ~np.isnan(dl)
                    _, dd = per_date_mean(dates[mm], dl[mm])
                    res[f"{tf}_minus_{bl}_{kind}"] = {"mean": float(dd.mean()), "ci": block_bootstrap(dd)}
        R["vol"][f"rv{w}"] = res
        R["vol_by_year"][f"rv{w}"] = yr

    # ---- 백테스트 ----
    scores = {
        "timesfm": {1: qret[:, 0, 4], 5: qret[:, 4, 4]},
        "hist_drift": {1: S["mu"], 5: S["mu"]},
        "last_return": {1: S["last1"], 5: S["last5"]},
        "lgbm": {1: P["lgbm"][:, 2] - P["lgbm"][:, 0], 5: P["lgbm"][:, 2] - P["lgbm"][:, 0]},
        "lgbm_plus_timesfm": {1: P["lgbm_tfm"][:, 2] - P["lgbm_tfm"][:, 0],
                              5: P["lgbm_tfm"][:, 2] - P["lgbm_tfm"][:, 0]},
    }
    if "ret_cov" in Q:
        scores["timesfm_cov"] = {1: Q["ret_cov"][:, 0, 4], 5: Q["ret_cov"][:, 4, 4]}
    t0 = time.time()
    for hold in (1, 5):
        bt = {}
        yr = {}
        # 동일가중 유니버스 (신호 없음, 전 종목)
        bt["equal_weight_universe"], d_ew = backtest(panel, S, np.ones(N), ev, hold, k=len(panel),
                                                     threshold=0.0)
        for name, sc in scores.items():
            s_h = np.nan_to_num(sc[hold], nan=-1e9)
            # rank: 항상 상위 10종목 보유 (순수 종목선택 능력)
            bt[name], dly = backtest(panel, S, s_h, ev, hold, threshold=-1e8)
            # gated: 점수 > 0 인 종목만 (시장 타이밍 포함, 나머지 현금)
            bt[name + "_gated"], _ = backtest(panel, S, s_h, ev, hold, threshold=0.0)
            yd = np.unique(dates[ev])
            yr[name] = {Y: sharpe(dly[np.array([d.startswith(str(Y)) for d in yd])]) for Y in years}
        yd = np.unique(dates[ev])
        yr["equal_weight_universe"] = {Y: sharpe(d_ew[np.array([d.startswith(str(Y)) for d in yd])])
                                       for Y in years}
        R["backtest"][f"hold{hold}"] = bt
        R["backtest_by_year"][f"hold{hold}"] = yr
    # KOSPI 지수 (종가 기준, 비용 없음) 동일 기간
    idx = raw[INDEX_CODE]["Close"]
    yd = np.unique(dates[ev])
    ki = idx[(idx.index >= str(yd[0])) & (idx.index <= str(yd[-1]))].pct_change().dropna().to_numpy()
    keq = np.cumprod(1 + ki)
    R["backtest"]["kospi_index"] = {"sharpe": sharpe(ki), "mdd_pct": float((keq / np.maximum.accumulate(keq) - 1).min() * 100),
                                    "total_return_pct": float((keq[-1] - 1) * 100)}
    timings["backtest_s"] = time.time() - t0

    import importlib.metadata as md
    vers = {}
    for p in ("timesfm", "mlx", "lightgbm", "numpy", "scikit-learn", "finance-datareader", "scipy"):
        try:
            vers[p] = md.version(p)
        except Exception:
            vers[p] = None
    R["setup"] = {
        "tickers": sorted(panel), "n_tickers": len(panel), "n_samples": N,
        "n_test_samples": int(ev.sum()), "test_years": years,
        "first_date": str(min(dates)), "last_date": str(max(dates)),
        "context_len": CONTEXT_LEN, "backend": getattr(fc, "backend", "cached"),
        "python": platform.python_version(), "machine": platform.machine(),
        "platform": platform.platform(), "versions": vers,
        "timings_s": timings, "total_runtime_s": time.time() - t_start,
        "n_lgbm_features": len(names), "lgbm_features": names,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(R, ensure_ascii=False, indent=1, default=float))
    write_csv(R, Path(args.out).with_suffix(".csv"))
    logger.info("완료 %.0fs → %s", time.time() - t_start, args.out)


def write_csv(R, path):
    lines = ["section,horizon,model,metric,value"]
    for sec in ("returns", "vol", "backtest"):
        for h, models in R[sec].items():
            if not isinstance(models, dict):
                continue
            for mname, ent in models.items():
                if not isinstance(ent, dict):
                    continue
                for k, v in ent.items():
                    if isinstance(v, (int, float)):
                        lines.append(f"{sec},{h},{mname},{k},{v:.6g}")
    for mname, ent in R["cls"].items():
        if isinstance(ent, dict):
            for k in ("macro_f1", "accuracy"):
                lines.append(f"cls,tb15,{mname},{k},{ent[k]:.6g}")
    path.write_text("\n".join(lines) + "\n")


def _ci(ci, fmt):
    return f"[{fmt.format(ci[0])}, {fmt.format(ci[1])}]" if ci and ci[0] == ci[0] else ""


def render_markdown(R) -> str:
    """결과 JSON → 마크다운 표 (docs/timesfm_eval.md에 그대로 붙여넣는 용도)."""
    L = []
    for h in ("h1", "h5", "h15"):
        L.append(f"\n#### Returns, horizon {h[1:]}d (test origins {R['setup']['n_test_samples']:,})\n")
        L.append("| model | dir. acc [95% CI] | AUC | rank IC [95% CI] | MAE ×100 | CRPS ×100 |")
        L.append("|---|---|---|---|---|---|")
        for m, e in R["returns"][h].items():
            if m.endswith("_minus_zero"):
                continue
            da = f"{e['dir_acc']:.4f} {_ci(e.get('dir_acc_ci'), '{:.4f}')}" if "dir_acc" in e else "—"
            ic = f"{e['ic']:+.4f} {_ci(e.get('ic_ci'), '{:+.4f}')}" if "ic" in e else "—"
            auc = f"{e['auc']:.4f}" if "auc" in e else "—"
            mae = f"{e['mae'] * 100:.4f}" if "mae" in e else "—"
            crps = f"{e['crps'] * 100:.4f}" if "crps" in e else "—"
            L.append(f"| {m} | {da} | {auc} | {ic} | {mae} | {crps} |")
        for m, e in R["returns"][h].items():
            if m.endswith("_minus_zero"):
                L.append(f"\n`{m}` (per-date paired, ×100; negative = TimesFM better): "
                         f"MAE {e['mae'] * 100:+.4f} {_ci([x * 100 for x in e['mae_ci']], '{:+.4f}')}, "
                         f"CRPS {e['crps'] * 100:+.4f} {_ci([x * 100 for x in e['crps_ci']], '{:+.4f}')}")
    L.append("\n#### Direction accuracy by test year (horizon 5d)\n")
    yr = R["returns_by_year"]["h5"]
    years = list(next(iter(yr.values())).keys())
    L.append("| model | " + " | ".join(str(y) for y in years) + " |")
    L.append("|---|" + "---|" * len(years))
    for m, d in yr.items():
        L.append(f"| {m} | " + " | ".join(f"{d[y]['dir_acc']:.3f}" for y in years) + " |")
    L.append("\n#### Rank IC by test year (horizon 5d)\n")
    L.append("| model | " + " | ".join(str(y) for y in years) + " |")
    L.append("|---|" + "---|" * len(years))
    for m, d in yr.items():
        L.append(f"| {m} | " + " | ".join(f"{d[y]['ic']:+.3f}" for y in years) + " |")

    c = R["cls"]
    L.append(f"\n#### 3-class triple barrier (±1.5·ATR14, 15d) — test class counts "
             f"down/flat/up = {c['class_dist_test']}\n")
    L.append("| model | macro-F1 [95% CI] | accuracy | predicted down/flat/up |")
    L.append("|---|---|---|---|")
    for m, e in c.items():
        if isinstance(e, dict):
            L.append(f"| {m} | {e['macro_f1']:.4f} {_ci(e['macro_f1_ci'], '{:.4f}')} | "
                     f"{e['accuracy']:.4f} | {e['pred_dist']} |")
    L.append("\nmacro-F1 by test year:\n")
    cy = R["cls_by_year"]
    L.append("| model | " + " | ".join(str(y) for y in years) + " |")
    L.append("|---|" + "---|" * len(years))
    for m, d in cy.items():
        L.append(f"| {m} | " + " | ".join(f"{d[y]:.3f}" if y in d else "—" for y in years) + " |")

    for w in ("rv5", "rv20"):
        L.append(f"\n#### Volatility, {w[2:]}-day realized vol (daily units)\n")
        L.append("| model | MAE ×100 [95% CI] | RMSE ×100 | QLIKE [95% CI] | MZ R² |")
        L.append("|---|---|---|---|---|")
        for m, e in R["vol"][w].items():
            if "_minus_" in m:
                continue
            L.append(f"| {m} | {e['mae'] * 100:.4f} {_ci([x * 100 for x in e['mae_ci']], '{:.4f}')} | "
                     f"{e['rmse'] * 100:.4f} | {e['qlike']:.4f} {_ci(e['qlike_ci'], '{:.4f}')} | "
                     f"{e['mz_r2']:.4f} |")
        L.append("\nPaired per-date loss differences (negative = TimesFM better):\n")
        L.append("| comparison | mean | 95% CI |")
        L.append("|---|---|---|")
        for m, e in R["vol"][w].items():
            if "_minus_" in m:
                sc = 100 if m.endswith("_mae") else 1
                L.append(f"| {m} | {e['mean'] * sc:+.5f} | {_ci([x * sc for x in e['ci']], '{:+.5f}')} |")

    for hk in ("hold1", "hold5"):
        L.append(f"\n#### Backtest, rebalance every {hk[4:]}d, top-10 equal weight, costs on\n")
        L.append("| strategy | Sharpe [95% CI] | CAGR % | MDD % | total % |")
        L.append("|---|---|---|---|---|")
        for m, e in R["backtest"][hk].items():
            L.append(f"| {m} | {e['sharpe']:.2f} {_ci(e['sharpe_ci'], '{:.2f}')} | {e['cagr_pct']:.1f} | "
                     f"{e['mdd_pct']:.1f} | {e['total_return_pct']:.0f} |")
    k = R["backtest"]["kospi_index"]
    L.append(f"\nKOSPI index (close-to-close, no costs, same period): Sharpe {k['sharpe']:.2f}, "
             f"MDD {k['mdd_pct']:.1f}%, total {k['total_return_pct']:.0f}%")
    L.append("\nSharpe by test year (rebalance 5d, rank variant):\n")
    by = R["backtest_by_year"]["hold5"]
    L.append("| strategy | " + " | ".join(str(y) for y in years) + " |")
    L.append("|---|" + "---|" * len(years))
    for m, d in by.items():
        L.append(f"| {m} | " + " | ".join(f"{d[y]:.2f}" for y in years) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
