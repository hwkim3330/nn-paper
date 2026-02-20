"""
시장 컨텍스트 피처 (25개)
- US 지수, 환율, VIX, 업종 강도, 시장 심리 등
"""
import logging

logger = logging.getLogger(__name__)


def compute_market_features(market_data: list[dict],
                            sector_data: list[dict],
                            stock_sector: str = "") -> dict:
    """
    시장 컨텍스트 피처 계산.
    market_data: 당일 시장 데이터 [{symbol, price, change_pct, ...}]
    sector_data: 업종 데이터 [{sector_code, close, change_rate, ...}]
    stock_sector: 해당 종목의 섹터 코드
    """
    f = {}

    # market_data를 symbol→dict 매핑
    md = {r.get("symbol", ""): r for r in market_data}

    # === US 지수 ===
    for key, feat_name in [
        ("NASDAQ", "us_nasdaq_pct"),
        ("SP500", "us_sp500_pct"),
        ("DOW", "us_dow_pct"),
        ("SOX", "us_sox_pct"),
        ("VIX", "us_vix"),
        ("USDKRW", "usd_krw"),
    ]:
        d = md.get(key, {})
        if key == "VIX":
            f[feat_name] = d.get("price", 0)
        elif key == "USDKRW":
            f[feat_name] = d.get("price", 0)
        else:
            f[feat_name] = d.get("change_pct", 0)

    # 미국 개별종목
    for sym in ["NVDA", "MU", "TSLA", "AAPL"]:
        d = md.get(f"US_{sym}", {})
        f[f"us_{sym.lower()}_pct"] = d.get("change_pct", 0)

    # VIX 레벨 (원핫)
    vix = f.get("us_vix", 0)
    f["vix_low"] = 1 if vix < 15 else 0      # 안정
    f["vix_mid"] = 1 if 15 <= vix < 25 else 0 # 보통
    f["vix_high"] = 1 if 25 <= vix < 35 else 0 # 불안
    f["vix_extreme"] = 1 if vix >= 35 else 0   # 공포

    # 환율 레벨
    krw = f.get("usd_krw", 0)
    f["krw_high"] = 1 if krw >= 1400 else 0
    f["krw_low"] = 1 if krw <= 1250 else 0

    # === 업종 ===
    # KOSPI 종합
    kospi = _find_sector(sector_data, "001")
    f["kospi_change"] = kospi.get("change_rate", 0) if kospi else 0

    # 해당 종목 업종 강도
    if stock_sector:
        sec = _find_sector(sector_data, stock_sector)
        if sec:
            f["sector_change"] = sec.get("change_rate", 0)
            f["sector_advance_ratio"] = (
                sec.get("advance_count", 0) /
                max(sec.get("advance_count", 0) + sec.get("decline_count", 1), 1)
            )
        else:
            f["sector_change"] = 0
            f["sector_advance_ratio"] = 0.5
    else:
        f["sector_change"] = 0
        f["sector_advance_ratio"] = 0.5

    # 전기전자 (삼성전자 등 반도체 업종)
    elec = _find_sector(sector_data, "013")
    f["sector_elec_change"] = elec.get("change_rate", 0) if elec else 0

    # 금융업
    fin = _find_sector(sector_data, "021")
    f["sector_finance_change"] = fin.get("change_rate", 0) if fin else 0

    # 업종 대비 상대강도
    kospi_chg = f.get("kospi_change", 0)
    sector_chg = f.get("sector_change", 0)
    f["sector_relative_strength"] = sector_chg - kospi_chg

    # 시장 심리 종합 (간단 합산)
    sentiment = 0
    nasdaq_pct = f.get("us_nasdaq_pct", 0)
    if nasdaq_pct >= 1.0:
        sentiment += 2
    elif nasdaq_pct >= 0.3:
        sentiment += 1
    elif nasdaq_pct <= -1.0:
        sentiment -= 2
    elif nasdaq_pct <= -0.3:
        sentiment -= 1

    if vix >= 30:
        sentiment -= 1
    elif vix <= 15:
        sentiment += 1

    f["market_sentiment"] = sentiment

    return f


def _find_sector(sector_data: list[dict], code: str) -> dict | None:
    """업종 데이터에서 특정 코드 찾기."""
    for s in sector_data:
        if s.get("sector_code") == code:
            return s
    return None
