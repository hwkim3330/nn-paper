"""
수급 피처 (23개)
- 외국인/기관/개인 순매수, 프로그램 매매, 연속 매수일, 보유율 변화 등
- 외국인 z-score, 모멘텀, 업종 상대강도
"""
import math
import logging

logger = logging.getLogger(__name__)


def compute_flow_features(investor_flows: list[dict],
                          program_trades: list[dict]) -> dict:
    """
    수급 피처 계산.
    investor_flows: 일별 투자자 매매동향 리스트 (오름차순)
    program_trades: 일별 프로그램 매매 리스트 (오름차순)
    Returns: {feature_name: value, ...}
    """
    f = {}

    # === 외국인/기관 ===
    if investor_flows:
        latest = investor_flows[-1]
        f["foreign_net"] = latest.get("foreign_net", 0)
        f["institution_net"] = latest.get("institution_net", 0)
        f["individual_net"] = latest.get("individual_net", 0)
        f["foreign_hold_ratio"] = latest.get("foreign_hold_ratio", 0)

        # 5일/20일 누적
        recent5 = investor_flows[-5:] if len(investor_flows) >= 5 else investor_flows
        recent20 = investor_flows[-20:] if len(investor_flows) >= 20 else investor_flows

        f["foreign_net_5d"] = sum(r.get("foreign_net", 0) for r in recent5)
        f["foreign_net_20d"] = sum(r.get("foreign_net", 0) for r in recent20)
        f["institution_net_5d"] = sum(r.get("institution_net", 0) for r in recent5)
        f["institution_net_20d"] = sum(r.get("institution_net", 0) for r in recent20)

        # 외국인 연속 매수/매도일
        f["foreign_consecutive"] = _consecutive_sign(
            [r.get("foreign_net", 0) for r in investor_flows]
        )
        f["institution_consecutive"] = _consecutive_sign(
            [r.get("institution_net", 0) for r in investor_flows]
        )

        # 외국인 보유율 변화 (5일 전 대비)
        if len(investor_flows) >= 5:
            f["foreign_hold_change_5d"] = (
                latest.get("foreign_hold_ratio", 0) -
                investor_flows[-5].get("foreign_hold_ratio", 0)
            )
        else:
            f["foreign_hold_change_5d"] = 0

        # 수급 합산 (외국인+기관 vs 개인)
        f["smart_money_net"] = latest.get("foreign_net", 0) + latest.get("institution_net", 0)
        f["smart_money_5d"] = f["foreign_net_5d"] + f["institution_net_5d"]

        # === 새 피처: foreign_flow_zscore (20일 대비 이례적인 수준) ===
        foreign_nets = [r.get("foreign_net", 0) for r in recent20]
        if len(foreign_nets) >= 5:
            fn_mean = sum(foreign_nets) / len(foreign_nets)
            fn_std = math.sqrt(sum((x - fn_mean) ** 2 for x in foreign_nets) / len(foreign_nets))
            f["foreign_flow_zscore"] = (foreign_nets[-1] - fn_mean) / fn_std if fn_std > 0 else 0
        else:
            f["foreign_flow_zscore"] = 0

        # === 새 피처: foreign_flow_momentum (5일평균 - 20일평균) ===
        fn_5d_avg = sum(r.get("foreign_net", 0) for r in recent5) / len(recent5) if recent5 else 0
        fn_20d_avg = sum(r.get("foreign_net", 0) for r in recent20) / len(recent20) if recent20 else 0
        f["foreign_flow_momentum"] = fn_5d_avg - fn_20d_avg

        # === 새 피처: sector_relative_strength (업종 대비 상대강도는 market_features에서 계산) ===

    else:
        for k in ["foreign_net", "institution_net", "individual_net",
                   "foreign_hold_ratio", "foreign_net_5d", "foreign_net_20d",
                   "institution_net_5d", "institution_net_20d",
                   "foreign_consecutive", "institution_consecutive",
                   "foreign_hold_change_5d", "smart_money_net", "smart_money_5d",
                   "foreign_flow_zscore", "foreign_flow_momentum"]:
            f[k] = 0

    # === 프로그램 매매 ===
    if program_trades:
        latest_p = program_trades[-1]
        f["program_net"] = latest_p.get("program_net", 0)
        f["program_buy_ratio"] = (
            latest_p.get("program_buy", 0) /
            (latest_p.get("program_buy", 0) + latest_p.get("program_sell", 1))
        )

        recent5_p = program_trades[-5:] if len(program_trades) >= 5 else program_trades
        f["program_net_5d"] = sum(r.get("program_net", 0) for r in recent5_p)

        # 차익/비차익
        f["arbitrage_net"] = (
            latest_p.get("arbitrage_buy", 0) - latest_p.get("arbitrage_sell", 0)
        )
        f["non_arbitrage_net"] = (
            latest_p.get("non_arbitrage_buy", 0) - latest_p.get("non_arbitrage_sell", 0)
        )
    else:
        for k in ["program_net", "program_buy_ratio", "program_net_5d",
                   "arbitrage_net", "non_arbitrage_net"]:
            f[k] = 0

    return f


def _consecutive_sign(values: list) -> int:
    """마지막부터 같은 부호 연속 일수. 양수=매수 연속, 음수=매도 연속."""
    if not values:
        return 0
    consec = 0
    last_sign = 1 if values[-1] > 0 else (-1 if values[-1] < 0 else 0)
    if last_sign == 0:
        return 0
    for v in reversed(values):
        sign = 1 if v > 0 else (-1 if v < 0 else 0)
        if sign == last_sign:
            consec += 1
        else:
            break
    return consec * last_sign
