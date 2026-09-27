"""
TimesFM-3 시계열 파운데이션 모델 래퍼 (선택 기능, 연구/모의투자 전용)

- Google TimesFM 3.0 (330M, google/timesfm-3.0-pytorch) zero-shot 예측
- 백엔드: Apple Silicon이면 MLX, 아니면 PyTorch(CUDA/CPU)
- 제공 기능
  (a) 분위수 예측: h일 누적 로그수익률 / 실현변동성 (9개 분위수 q10..q90)
  (b) 기존 파이프라인용 파생 피처: tfm_p_up_1d, tfm_exp_ret_5d, tfm_spread_5d, tfm_rv_5d ...
  (c) 단독 신호: Triple Barrier 3-class (하락/보합/상승) 확률 → paper_engine 예측 dict 형식

라이선스: TimesFM 3.0 가중치는 timesfm-non-commercial-license-v1.0 —
비상업·비프로덕션 용도만 허용. 따라서 실거래(trading/live_engine.py)에서는
절대 사용할 수 없으며, 이 모듈은 실거래 프로세스에서 로드를 거부한다.

timesfm 패키지가 없어도 이 모듈의 import 자체는 실패하지 않는다
(무거운 의존성은 TimesFMForecaster 생성 시점에만 import).
설치: pip install "timesfm[mlx]" (Apple Silicon) 또는 pip install "timesfm[torch]"
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np

logger = logging.getLogger(__name__)

TIMESFM_CHECKPOINT = "google/timesfm-3.0-pytorch"
TIMESFM_LICENSE = "timesfm-non-commercial-license-v1.0"
QUANTILE_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
FEATURE_PREFIX = "tfm_"

# 실거래 프로세스 표시 (LiveEngine이 설정). 설정되면 TimesFM 로드/사용 거부.
_LIVE_ENV = "NN_PAPER_LIVE_TRADING"
_live_process = False

# Φ^-1(0.9) - Φ^-1(0.1) — q10..q90 폭을 정규분포 σ로 환산
_Z_SPAN_10_90 = 2.5631031310892007


class TimesFMLicenseError(RuntimeError):
    """실거래 경로에서 비상업 라이선스 모델을 쓰려 할 때."""


def mark_live_process():
    """실거래 프로세스임을 표시 — 이후 TimesFM 사용이 전부 거부된다."""
    global _live_process
    _live_process = True
    os.environ[_LIVE_ENV] = "1"


def is_live_process() -> bool:
    return _live_process or os.environ.get(_LIVE_ENV, "") not in ("", "0")


def assert_research_use():
    if is_live_process():
        raise TimesFMLicenseError(
            "TimesFM-3 가중치는 비상업·비프로덕션 라이선스(%s)입니다 — "
            "실거래(live) 경로에서는 사용할 수 없습니다." % TIMESFM_LICENSE
        )


def contains_timesfm_features(feature_names) -> bool:
    return any(str(n).startswith(FEATURE_PREFIX) for n in (feature_names or []))


def is_available() -> bool:
    """timesfm3 패키지 설치 여부 (가중치 다운로드는 확인하지 않음)."""
    try:
        import importlib.util
        return importlib.util.find_spec("timesfm3") is not None
    except Exception:
        return False


# ============================================================
# 분위수 → 확률 유틸 (numpy만 사용, 백엔드 불필요)
# ============================================================

def _norm_cdf(z):
    z = np.asarray(z, dtype=np.float64)
    try:
        from scipy.special import ndtr
        return ndtr(z)
    except ImportError:
        from math import erf
        return 0.5 * (1.0 + np.vectorize(erf)(z / math.sqrt(2.0)))


def prob_above(quantiles: np.ndarray, x) -> np.ndarray:
    """
    분위수 예측으로 P(Y > x) 근사.
    quantiles: (..., 9) — q10..q90 (정렬됨)
    q10..q90 사이는 선형보간, 바깥 꼬리는 정규분포(중앙=q50, σ=(q90-q10)/2.563)로 외삽.
    """
    q = np.asarray(quantiles, dtype=np.float64)
    x = np.broadcast_to(np.asarray(x, dtype=np.float64), q.shape[:-1])
    lo, hi, med = q[..., 0], q[..., -1], q[..., 4]
    sigma = np.maximum((hi - lo) / _Z_SPAN_10_90, 1e-9)

    # 내부: 인접 분위수 사이 선형보간 (벡터화)
    nq = q.shape[-1]
    k = np.clip((q <= x[..., None]).sum(axis=-1), 1, nq - 1)   # x가 속한 구간 [k-1, k]
    q_lo = np.take_along_axis(q, (k - 1)[..., None], axis=-1)[..., 0]
    q_hi = np.take_along_axis(q, k[..., None], axis=-1)[..., 0]
    t_lo, t_hi = QUANTILE_LEVELS[k - 1], QUANTILE_LEVELS[k]
    w = np.clip((x - q_lo) / np.maximum(q_hi - q_lo, 1e-12), 0, 1)
    cdf_in = t_lo + w * (t_hi - t_lo)

    cdf_tail = _norm_cdf((x - med) / sigma)
    cdf = np.where(x < lo, np.minimum(cdf_tail, 0.1),
                   np.where(x > hi, np.maximum(cdf_tail, 0.9), cdf_in))
    return 1.0 - cdf


def barrier_probs(path_quantiles: np.ndarray, barrier: np.ndarray) -> np.ndarray:
    """
    Triple Barrier 3-class 확률 근사 (하락, 보합, 상승).
    path_quantiles: (N, H, 9) — 각 k일 누적 로그수익률 분위수
    barrier: (N,) — 로그수익률 단위 barrier 폭 (예: log(1 + 1.5*ATR/close))

    분위수는 경로가 아닌 주변분포이므로 반사원리(드리프트 없는 브라운운동:
    P(max_{s≤H} X_s ≥ b) = 2·P(X_H ≥ b))로 barrier 터치 확률을 근사하고,
    드리프트가 있는 경우를 위해 각 k 중 최대값을 사용한다.
    """
    pq = np.asarray(path_quantiles, dtype=np.float64)
    b = np.asarray(barrier, dtype=np.float64)[:, None]
    b = np.broadcast_to(b, pq.shape[:2])
    p_up_k = prob_above(pq, b)                  # (N, H)
    p_dn_k = 1.0 - prob_above(pq, -b)
    p_up = np.clip(2.0 * p_up_k[:, -1], 0, 1)
    p_dn = np.clip(2.0 * p_dn_k[:, -1], 0, 1)
    p_up = np.maximum(p_up, p_up_k.max(axis=1))
    p_dn = np.maximum(p_dn, p_dn_k.max(axis=1))
    total = p_up + p_dn
    scale = np.where(total > 1.0, 1.0 / np.maximum(total, 1e-12), 1.0)
    p_up, p_dn = p_up * scale, p_dn * scale
    p_flat = np.clip(1.0 - p_up - p_dn, 0, 1)
    return np.stack([p_dn, p_flat, p_up], axis=1)


def crps_from_quantiles(quantiles: np.ndarray, y: np.ndarray) -> np.ndarray:
    """분위수 손실 기반 CRPS 근사: 2 · mean_τ pinball_τ(y, q_τ) (9개 분위)."""
    q = np.asarray(quantiles, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)[..., None]
    diff = y - q
    pinball = np.maximum(QUANTILE_LEVELS * diff, (QUANTILE_LEVELS - 1) * diff)
    return 2.0 * pinball.mean(axis=-1)


# ============================================================
# 컨텍스트 구성
# ============================================================

def log_price_context(closes: np.ndarray, context_len: int) -> np.ndarray:
    """마지막 종가 기준 로그가격 (마지막 값 = 0). 예측값 = 누적 로그수익률."""
    c = np.asarray(closes, dtype=np.float64)[-context_len:]
    return (np.log(c) - np.log(c[-1])).astype(np.float32)


def rolling_rv(closes: np.ndarray, window: int) -> np.ndarray:
    """trailing window일 실현변동성 sqrt(mean r²) (일간, 로그수익률). 앞부분 NaN."""
    c = np.asarray(closes, dtype=np.float64)
    r2 = np.concatenate([[np.nan], np.diff(np.log(c)) ** 2])
    out = np.full(len(c), np.nan)
    cs = np.nancumsum(np.nan_to_num(r2))
    for i in range(window, len(c)):
        out[i] = math.sqrt((cs[i] - cs[i - window]) / window)
    return out


# ============================================================
# Forecaster
# ============================================================

class TimesFMForecaster:
    """
    TimesFM-3 배치 예측기.

    사용 예:
        fc = TimesFMForecaster()                       # 가중치 자동 다운로드 (~1.3GB)
        q = fc.forecast_log_returns([closes1, closes2], horizon=15)   # (N, 15, 9)
        feats = fc.features_for_stock(prices)          # {tfm_p_up_1d: ..., ...}
        sig = fc.signal(prices, stock_code="005930")   # paper_engine 예측 dict
    """

    def __init__(self, backend: str = "auto", context_len: int = 512,
                 batch_size: int = 64, checkpoint: str = TIMESFM_CHECKPOINT,
                 device: str | None = None):
        assert_research_use()
        self.context_len = context_len
        self.batch_size = batch_size
        self.checkpoint = checkpoint
        self.backend = None
        self.model = None
        self._load(backend, device)

    def _load(self, backend: str, device: str | None):
        errors = []
        order = ["mlx", "torch"] if backend == "auto" else [backend]
        for name in order:
            try:
                if name == "mlx":
                    from timesfm3.mlx import TimesFM3Forecaster  # Apple Silicon 전용
                    self.model = TimesFM3Forecaster.from_pretrained(
                        self.checkpoint, per_core_batch_size=self.batch_size)
                elif name == "torch":
                    from timesfm3.torch import TimesFM3Forecaster
                    self.model = TimesFM3Forecaster.from_pretrained(
                        self.checkpoint, device=device)
                else:
                    raise ValueError(f"unknown backend {name}")
                self.backend = name
                logger.info("TimesFM-3 로드 완료 (backend=%s, license=%s — 연구/모의투자 전용)",
                            name, TIMESFM_LICENSE)
                return
            except Exception as e:  # ImportError, 다운로드 실패 등
                errors.append(f"{name}: {e!r}")
        raise ImportError(
            "TimesFM-3 백엔드를 불러올 수 없습니다. "
            "pip install 'timesfm[mlx]' 또는 'timesfm[torch]' — " + "; ".join(errors))

    # ---------------- 저수준 ----------------
    def predict_quantiles(self, contexts: list[np.ndarray], horizon: int,
                          past_covariates: list[np.ndarray] | None = None) -> np.ndarray:
        """contexts: 1D 배열 리스트 → (N, horizon, 9) 분위수 (정렬됨)."""
        assert_research_use()
        out = np.empty((len(contexts), horizon, len(QUANTILE_LEVELS)), dtype=np.float32)
        for s in range(0, len(contexts), self.batch_size):
            ctx = [np.asarray(c, dtype=np.float32) for c in contexts[s:s + self.batch_size]]
            kw = {}
            if past_covariates is not None:
                kw["past_only_covariates"] = [np.asarray(p, dtype=np.float32)
                                              for p in past_covariates[s:s + self.batch_size]]
            res = self.model.predict_batch(contexts=ctx, horizon=horizon,
                                           return_quantiles=True, **kw)
            for j, r in enumerate(res):
                out[s + j] = r.quantiles
        return out

    # ---------------- (a) 분위수 예측 ----------------
    def forecast_log_returns(self, close_series: list[np.ndarray], horizon: int = 15,
                             covariates: list[np.ndarray] | None = None) -> np.ndarray:
        """
        각 종목의 종가 시계열(오름차순, 마지막 = 예측 시점 t)로
        k=1..horizon일 누적 로그수익률 log(C[t+k]/C[t])의 분위수 예측.
        covariates: 종목별 (n_cov, len) 과거 전용 공변량 (예: 지수 로그가격, 거래량) — 선택.
        Returns: (N, horizon, 9)
        """
        L = self.context_len
        ctx = [log_price_context(c, L) for c in close_series]
        cov = None
        if covariates is not None:
            cov = [np.atleast_2d(np.asarray(cv, dtype=np.float32))[:, -len(x):]
                   for cv, x in zip(covariates, ctx)]
        return self.predict_quantiles(ctx, horizon, cov)

    def forecast_volatility(self, close_series: list[np.ndarray], window: int = 5) -> np.ndarray:
        """
        향후 window일 실현변동성(일간) 분위수 예측.
        컨텍스트 = trailing window일 RV 시계열 → window 스텝 후 값이 곧 t+1..t+window RV.
        Returns: (N, 9)
        """
        ctx = []
        for c in close_series:
            rv = rolling_rv(c, window)
            rv = rv[~np.isnan(rv)][-self.context_len:]
            ctx.append(rv.astype(np.float32))
        q = self.predict_quantiles(ctx, window)
        return np.maximum(q[:, window - 1, :], 0.0)

    # ---------------- (b) 파생 피처 ----------------
    @staticmethod
    def derive_features(ret_q: np.ndarray, rv5_q: np.ndarray | None = None,
                        barrier: np.ndarray | None = None) -> list[dict]:
        """
        분위수 → 파이프라인 피처 (접두어 tfm_).
        ret_q: (N, H, 9) 누적 로그수익률 분위수 (H ≥ 5 권장)
        rv5_q: (N, 9) 5일 RV 분위수 (선택)
        barrier: (N,) triple-barrier 폭 (로그수익률, 선택)
        """
        H = ret_q.shape[1]
        feats = [dict() for _ in range(ret_q.shape[0])]
        for h in (1, 3, 5, 15):
            if h > H:
                continue
            q = ret_q[:, h - 1, :]
            p_up = prob_above(q, 0.0)
            for i in range(len(feats)):
                feats[i][f"tfm_p_up_{h}d"] = float(p_up[i])
                feats[i][f"tfm_exp_ret_{h}d"] = float(q[i, 4])
                feats[i][f"tfm_spread_{h}d"] = float(q[i, -1] - q[i, 0])
                feats[i][f"tfm_skew_{h}d"] = float((q[i, -1] - q[i, 4]) - (q[i, 4] - q[i, 0]))
        if rv5_q is not None:
            for i in range(len(feats)):
                feats[i]["tfm_rv_5d"] = float(rv5_q[i, 4])
        if barrier is not None:
            probs = barrier_probs(ret_q, barrier)
            for i in range(len(feats)):
                feats[i]["tfm_tb_p_down"] = float(probs[i, 0])
                feats[i]["tfm_tb_p_up"] = float(probs[i, 2])
        return feats

    # ---------------- (c) 단독 신호 ----------------
    @staticmethod
    def signal_from_quantiles(ret_q: np.ndarray, barrier: np.ndarray,
                              stock_codes: list[str] | None = None) -> list[dict]:
        """
        Triple Barrier 3-class 신호 (0 하락 / 1 보합 / 2 상승).
        반환 형식은 PaperEngine.execute_signals의 predictions 원소와 동일.
        """
        from nn_config import CLASS_NAMES
        probs = barrier_probs(ret_q, barrier)
        out = []
        for i, p in enumerate(probs):
            cls = int(np.argmax(p))
            out.append({
                "stock_code": stock_codes[i] if stock_codes else None,
                "predicted_class": cls,
                "confidence": float(p[cls]),
                "buy_prob": float(p[2]),
                "sell_prob": float(p[0]),
                "probabilities": [float(x) for x in p],
                "signal": CLASS_NAMES[cls],
                "source": "timesfm3",
            })
        return out

    # ---------------- 파이프라인 편의 함수 ----------------
    @staticmethod
    def barrier_from_prices(prices: list[dict]) -> float:
        """repo의 Triple Barrier와 동일: ±atr_multiplier × ATR(단순평균 14) / 종가 → 로그수익률."""
        from nn_config import TRIPLE_BARRIER
        period = TRIPLE_BARRIER["atr_period"]
        mult = TRIPLE_BARRIER["atr_multiplier"]
        trs = []
        for i in range(len(prices) - period, len(prices)):
            h, l, pc = prices[i]["high"], prices[i]["low"], prices[i - 1]["close"]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        atr = sum(trs) / len(trs)
        return math.log1p(mult * atr / prices[-1]["close"])

    def features_batch(self, prices_list: list[list[dict]]) -> list[dict]:
        """일봉 리스트([{date, open, high, low, close, volume}], 오름차순)들 → tfm_ 피처 dict들."""
        from nn_config import TRIPLE_BARRIER
        H = TRIPLE_BARRIER["max_holding_days"]
        closes = [np.array([p["close"] for p in pr], dtype=np.float64) for pr in prices_list]
        ret_q = self.forecast_log_returns(closes, horizon=H)
        rv_q = self.forecast_volatility(closes, window=5)
        barrier = np.array([self.barrier_from_prices(pr) for pr in prices_list])
        return self.derive_features(ret_q, rv_q, barrier)

    def features_for_stock(self, prices: list[dict]) -> dict:
        return self.features_batch([prices])[0]

    def signal(self, prices: list[dict], stock_code: str | None = None) -> dict:
        from nn_config import TRIPLE_BARRIER
        closes = np.array([p["close"] for p in prices], dtype=np.float64)
        ret_q = self.forecast_log_returns([closes], horizon=TRIPLE_BARRIER["max_holding_days"])
        barrier = np.array([self.barrier_from_prices(prices)])
        return self.signal_from_quantiles(ret_q, barrier, [stock_code])[0]
