"""
기술적 지표 피처 (40개)
- MA, EMA, RSI, MACD, BB, ATR, ADX, Stochastic, CCI, OBV, Williams%R 등
- 입력: 일봉 리스트 [{date, open, high, low, close, volume}, ...] (오름차순)
"""
import math
import logging

logger = logging.getLogger(__name__)


def compute_technical_features(prices: list[dict]) -> dict:
    """
    기술적 지표 40개 계산.
    prices: 일봉 리스트 (오름차순, 최소 120개 권장)
    Returns: {feature_name: value, ...}
    """
    if len(prices) < 20:
        return {}

    closes = [p["close"] for p in prices]
    highs = [p["high"] for p in prices]
    lows = [p["low"] for p in prices]
    opens = [p["open"] for p in prices]
    volumes = [p["volume"] for p in prices]

    f = {}

    # === 이동평균 (MA) ===
    for period in [5, 10, 20, 60, 120]:
        ma = _sma(closes, period)
        if ma is not None:
            f[f"ma_{period}"] = ma
            f[f"close_ma_{period}_ratio"] = closes[-1] / ma if ma > 0 else 0

    # EMA
    for period in [12, 26]:
        ema = _ema(closes, period)
        if ema is not None:
            f[f"ema_{period}"] = ema

    # MA 배열 상태 (정배열=1, 역배열=-1, 혼조=0)
    ma5 = f.get("ma_5")
    ma10 = f.get("ma_10")
    ma20 = f.get("ma_20")
    if ma5 and ma10 and ma20:
        if ma5 > ma10 > ma20:
            f["ma_alignment"] = 1
        elif ma5 < ma10 < ma20:
            f["ma_alignment"] = -1
        else:
            f["ma_alignment"] = 0

    # 골든크로스 / 데드크로스
    if len(closes) >= 11:
        prev_ma5 = _sma(closes[:-1], 5)
        prev_ma10 = _sma(closes[:-1], 10)
        if prev_ma5 and prev_ma10 and ma5 and ma10:
            f["golden_cross"] = 1 if (prev_ma5 <= prev_ma10 and ma5 > ma10) else 0
            f["dead_cross"] = 1 if (prev_ma5 >= prev_ma10 and ma5 < ma10) else 0

    # === RSI ===
    rsi = _rsi(closes, 14)
    if rsi is not None:
        f["rsi_14"] = rsi
        f["rsi_oversold"] = 1 if rsi < 30 else 0
        f["rsi_overbought"] = 1 if rsi > 70 else 0

    # === MACD ===
    macd, signal, hist = _macd(closes, 12, 26, 9)
    if macd is not None:
        f["macd"] = macd
        f["macd_signal"] = signal
        f["macd_histogram"] = hist
        f["macd_cross_up"] = 1 if hist > 0 else 0

    # === Bollinger Bands ===
    bb_upper, bb_middle, bb_lower = _bollinger(closes, 20, 2)
    if bb_upper is not None:
        f["bb_upper"] = bb_upper
        f["bb_middle"] = bb_middle
        f["bb_lower"] = bb_lower
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
        f["bb_width"] = bb_width
        f["bb_position"] = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) > 0 else 0.5

    # === ATR (Average True Range) ===
    atr = _atr(highs, lows, closes, 14)
    if atr is not None:
        f["atr_14"] = atr
        f["atr_pct"] = atr / closes[-1] * 100 if closes[-1] > 0 else 0

    # === Stochastic ===
    k, d = _stochastic(highs, lows, closes, 14, 3)
    if k is not None:
        f["stoch_k"] = k
        f["stoch_d"] = d

    # === ADX ===
    adx = _adx(highs, lows, closes, 14)
    if adx is not None:
        f["adx_14"] = adx
        f["strong_trend"] = 1 if adx > 25 else 0

    # === CCI ===
    cci = _cci(highs, lows, closes, 20)
    if cci is not None:
        f["cci_20"] = cci

    # === Williams %R ===
    wr = _williams_r(highs, lows, closes, 14)
    if wr is not None:
        f["williams_r"] = wr

    # === OBV (On-Balance Volume) ===
    obv = _obv(closes, volumes)
    if obv is not None:
        f["obv"] = obv
        obv_ma = _sma_list(_obv_series(closes, volumes), 20)
        if obv_ma:
            f["obv_ma_ratio"] = obv / obv_ma if obv_ma != 0 else 0

    # === 가격 변화율 ===
    for period in [1, 3, 5, 10, 20]:
        if len(closes) > period and closes[-1 - period] > 0:
            ret = (closes[-1] - closes[-1 - period]) / closes[-1 - period] * 100
            f[f"return_{period}d"] = round(ret, 4)

    # === 거래량 ===
    vol_avg_10 = _sma_list(volumes, 10)
    vol_avg_20 = _sma_list(volumes, 20)
    if vol_avg_10 and vol_avg_10 > 0:
        f["volume_ratio_10"] = volumes[-1] / vol_avg_10
    if vol_avg_20 and vol_avg_20 > 0:
        f["volume_ratio_20"] = volumes[-1] / vol_avg_20

    # 연속 상승/하락일
    consec = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] > closes[i - 1]:
            if consec >= 0:
                consec += 1
            else:
                break
        elif closes[i] < closes[i - 1]:
            if consec <= 0:
                consec -= 1
            else:
                break
        else:
            break
    f["consecutive_days"] = consec

    # 캔들 패턴
    if opens[-1] > 0 and closes[-1] > 0:
        body = closes[-1] - opens[-1]
        range_ = highs[-1] - lows[-1]
        f["candle_body_ratio"] = body / range_ if range_ > 0 else 0
        f["upper_shadow"] = (highs[-1] - max(opens[-1], closes[-1])) / range_ if range_ > 0 else 0
        f["lower_shadow"] = (min(opens[-1], closes[-1]) - lows[-1]) / range_ if range_ > 0 else 0

    # === 새 피처: vol_ratio_zscore ===
    if vol_avg_20 and vol_avg_20 > 0 and len(volumes) >= 20:
        vol_ratios = [volumes[-i] / vol_avg_20 for i in range(1, min(21, len(volumes)))]
        vr_mean = sum(vol_ratios) / len(vol_ratios)
        vr_std = math.sqrt(sum((x - vr_mean) ** 2 for x in vol_ratios) / len(vol_ratios)) if len(vol_ratios) > 1 else 1
        f["vol_ratio_zscore"] = (volumes[-1] / vol_avg_20 - vr_mean) / vr_std if vr_std > 0 else 0
    else:
        f["vol_ratio_zscore"] = 0

    # === 새 피처: close_vs_high_pct (종가위치) ===
    if highs[-1] > 0 and lows[-1] < highs[-1]:
        f["close_vs_high_pct"] = (closes[-1] - lows[-1]) / (highs[-1] - lows[-1])
    else:
        f["close_vs_high_pct"] = 0.5

    # === 새 피처: rv_ratio (단기/장기 변동성 비율) ===
    if len(closes) >= 20:
        rets_5 = [(closes[-i] - closes[-i - 1]) / closes[-i - 1] for i in range(1, min(6, len(closes)))]
        rets_20 = [(closes[-i] - closes[-i - 1]) / closes[-i - 1] for i in range(1, min(21, len(closes)))]
        rv_5 = math.sqrt(sum(r ** 2 for r in rets_5) / len(rets_5)) if rets_5 else 0
        rv_20 = math.sqrt(sum(r ** 2 for r in rets_20) / len(rets_20)) if rets_20 else 0
        f["rv_ratio"] = rv_5 / rv_20 if rv_20 > 0 else 1.0
    else:
        f["rv_ratio"] = 1.0

    return f


# ============================================================
# 내부 계산 함수
# ============================================================

def _sma(data: list, period: int) -> float | None:
    if len(data) < period:
        return None
    return sum(data[-period:]) / period


def _sma_list(data: list, period: int) -> float | None:
    if len(data) < period:
        return None
    return sum(data[-period:]) / period


def _ema(data: list, period: int) -> float | None:
    if len(data) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(data[:period]) / period
    for val in data[period:]:
        ema = val * k + ema * (1 - k)
    return ema


def _rsi(closes: list, period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    # Wilder's smoothing
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(closes: list, fast: int = 12, slow: int = 26, signal: int = 9):
    if len(closes) < slow + signal:
        return None, None, None
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    if ema_fast is None or ema_slow is None:
        return None, None, None

    # MACD line series
    macd_series = []
    k_fast = 2 / (fast + 1)
    k_slow = 2 / (slow + 1)
    ef = sum(closes[:fast]) / fast
    es = sum(closes[:slow]) / slow
    for i in range(slow, len(closes)):
        if i >= fast:
            ef = closes[i] * k_fast + ef * (1 - k_fast)
        es = closes[i] * k_slow + es * (1 - k_slow)
        macd_series.append(ef - es)

    if len(macd_series) < signal:
        return None, None, None

    sig = _ema(macd_series, signal)
    macd_val = macd_series[-1]
    hist = macd_val - sig if sig is not None else 0
    return macd_val, sig, hist


def _bollinger(closes: list, period: int = 20, mult: float = 2.0):
    if len(closes) < period:
        return None, None, None
    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = math.sqrt(variance)
    return middle + mult * std, middle, middle - mult * std


def _atr(highs: list, lows: list, closes: list, period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period - 1) + trs[i]) / period
    return atr


def _stochastic(highs, lows, closes, k_period=14, d_period=3):
    if len(closes) < k_period:
        return None, None
    k_values = []
    for i in range(k_period - 1, len(closes)):
        h = max(highs[i - k_period + 1:i + 1])
        l = min(lows[i - k_period + 1:i + 1])
        if h == l:
            k_values.append(50.0)
        else:
            k_values.append((closes[i] - l) / (h - l) * 100)
    k = k_values[-1]
    d = sum(k_values[-d_period:]) / d_period if len(k_values) >= d_period else k
    return k, d


def _adx(highs, lows, closes, period=14) -> float | None:
    if len(closes) < period * 2:
        return None
    plus_dm, minus_dm, tr_list = [], [], []
    for i in range(1, len(closes)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0)
        minus_dm.append(down if down > up and down > 0 else 0)
        tr_list.append(max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        ))

    if len(tr_list) < period:
        return None

    # Smoothed values
    atr = sum(tr_list[:period])
    plus_di_s = sum(plus_dm[:period])
    minus_di_s = sum(minus_dm[:period])

    dx_list = []
    for i in range(period, len(tr_list)):
        atr = atr - atr / period + tr_list[i]
        plus_di_s = plus_di_s - plus_di_s / period + plus_dm[i]
        minus_di_s = minus_di_s - minus_di_s / period + minus_dm[i]

        if atr == 0:
            continue
        plus_di = 100 * plus_di_s / atr
        minus_di = 100 * minus_di_s / atr
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_list.append(0)
        else:
            dx_list.append(abs(plus_di - minus_di) / di_sum * 100)

    if len(dx_list) < period:
        return None
    adx = sum(dx_list[:period]) / period
    for i in range(period, len(dx_list)):
        adx = (adx * (period - 1) + dx_list[i]) / period
    return adx


def _cci(highs, lows, closes, period=20) -> float | None:
    if len(closes) < period:
        return None
    tp = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
    tp_sma = sum(tp[-period:]) / period
    mean_dev = sum(abs(tp[-period + i] - tp_sma) for i in range(period)) / period
    if mean_dev == 0:
        return 0
    return (tp[-1] - tp_sma) / (0.015 * mean_dev)


def _williams_r(highs, lows, closes, period=14) -> float | None:
    if len(closes) < period:
        return None
    h = max(highs[-period:])
    l = min(lows[-period:])
    if h == l:
        return -50.0
    return (h - closes[-1]) / (h - l) * -100


def _obv(closes, volumes) -> float | None:
    if len(closes) < 2:
        return None
    obv = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]
    return obv


def _obv_series(closes, volumes) -> list:
    series = [0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            series.append(series[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            series.append(series[-1] - volumes[i])
        else:
            series.append(series[-1])
    return series
