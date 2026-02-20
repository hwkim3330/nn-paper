"""
리스크 관리 — ML 기반 청산
- 손절/익절을 고정 %가 아니라 ML 신호로 판단
- 극단적 손실(-5%)만 절대 안전장치로 유지
- 일일 손실 제한, 연속 손실 시 축소
"""
import math
import logging

from trading.portfolio import Portfolio

logger = logging.getLogger(__name__)

# 절대 안전장치 (ML 판단과 무관하게 강제 청산)
CATASTROPHIC_STOP = -5.0  # -5% 이하는 무조건 손절


class RiskManager:

    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        from nn_config import (
            MAX_POSITIONS, MAX_POSITION_PCT, MAX_SECTOR_PCT,
            TRAILING_STOP_PCT,
            KELLY_FRACTION, MIN_POSITION_SIZE,
            DAILY_LOSS_LIMIT, CONSECUTIVE_LOSS_DAYS, CONSECUTIVE_LOSS_SCALE,
        )
        self.max_positions = MAX_POSITIONS
        self.max_position_pct = MAX_POSITION_PCT
        self.max_sector_pct = MAX_SECTOR_PCT
        self.trailing_stop = TRAILING_STOP_PCT
        self.kelly_fraction = KELLY_FRACTION
        self.min_position_size = MIN_POSITION_SIZE
        self.daily_loss_limit = DAILY_LOSS_LIMIT
        self.consecutive_loss_days = CONSECUTIVE_LOSS_DAYS
        self.consecutive_loss_scale = CONSECUTIVE_LOSS_SCALE
        self._daily_pnl = 0.0
        self._trading_halted = False

    def reset_daily(self):
        """일일 리셋 (매일 장 시작 전 호출)."""
        self._daily_pnl = 0.0
        self._trading_halted = False

    def record_trade_pnl(self, pnl_pct: float):
        """매매 수익률 기록 (일일 손실 추적용)."""
        self._daily_pnl += pnl_pct
        if self._daily_pnl <= self.daily_loss_limit:
            self._trading_halted = True
            logger.warning("일일 손실 한도 초과: %.2f%% <= %.1f%% — 당일 매매 중단",
                          self._daily_pnl, self.daily_loss_limit)

    def is_trading_halted(self) -> bool:
        return self._trading_halted

    def _get_position_scale(self) -> float:
        """연속 손실 시 포지션 축소 배율."""
        daily_rets = self.portfolio.daily_returns
        if len(daily_rets) < self.consecutive_loss_days:
            return 1.0

        recent = daily_rets[-self.consecutive_loss_days:]
        if all(r < 0 for r in recent):
            logger.info("연속 %d일 손실 — 포지션 사이즈 %.0f%% 축소",
                        self.consecutive_loss_days, (1 - self.consecutive_loss_scale) * 100)
            return self.consecutive_loss_scale
        return 1.0

    def can_buy(self, stock_code: str, sector: str = "") -> bool:
        """매수 가능 여부 체크."""
        if self._trading_halted:
            return False

        if self.portfolio.num_positions >= self.max_positions:
            if stock_code not in self.portfolio.positions:
                return False

        if sector:
            exposure = self.portfolio.get_sector_exposure()
            if exposure.get(sector, 0) >= self.max_sector_pct:
                return False

        return True

    def compute_position_size(self, stock_code: str, price: int,
                               confidence: float, win_prob: float) -> int:
        """Kelly Criterion 기반 포지션 크기 계산."""
        total = self.portfolio.total_value
        if total <= 0 or price <= 0:
            return 0

        kelly = confidence * self.kelly_fraction
        kelly = max(self.min_position_size, min(self.max_position_pct, kelly))

        scale = self._get_position_scale()
        kelly *= scale

        existing_value = 0
        if stock_code in self.portfolio.positions:
            existing_value = self.portfolio.positions[stock_code].market_value

        target_value = int(total * kelly)
        available_value = target_value - existing_value

        if available_value <= 0:
            return 0

        available_value = min(available_value, self.portfolio.cash)

        quantity = available_value // price
        return max(0, quantity)

    def check_stop_conditions(self) -> list[dict]:
        """
        절대 안전장치만 체크 (ML과 무관).
        - 극단적 손실 -5% 이하: 무조건 청산
        - 트레일링 스톱: 고점 대비 -5% (수익 보호)
        나머지(일반 손절/익절)는 ML이 판단.
        """
        actions = []

        for code, pos in list(self.portfolio.positions.items()):
            pnl = pos.pnl

            # 극단적 손실 — 무조건 손절 (안전장치)
            if pnl <= CATASTROPHIC_STOP:
                actions.append({
                    "stock_code": code,
                    "stock_name": pos.stock_name,
                    "reason": f"긴급 손절 ({pnl:.1f}% <= {CATASTROPHIC_STOP}%)",
                    "action": "sell_all",
                    "pnl": pnl,
                })
                continue

            # 트레일링 스톱 (고점 대비 — 수익 보호)
            if pos.highest_price > 0 and pnl > 0:
                from_high = (pos.current_price - pos.highest_price) / pos.highest_price * 100
                if from_high <= self.trailing_stop:
                    actions.append({
                        "stock_code": code,
                        "stock_name": pos.stock_name,
                        "reason": f"트레일링 스톱 (고점 대비 {from_high:.1f}%, 수익 {pnl:.1f}%)",
                        "action": "sell_all",
                        "pnl": pnl,
                    })

        return actions

    def should_sell_on_signal(self, stock_code: str, predicted_class: int,
                                confidence: float) -> bool:
        """
        ML 신호 기반 매도 판단 (핵심 청산 로직).
        3-class: 0=하락, 1=보합, 2=상승

        - 하락(0) + 확신 40%↑ → 즉시 매도
        - 보합(1) + 확신 60%↑ + 수익 중 → 익절 (상승 모멘텀 끝)
        - 상승(2) → 보유 유지
        """
        if stock_code not in self.portfolio.positions:
            return False

        pos = self.portfolio.positions[stock_code]
        pnl = pos.pnl

        # ML이 하락 예측 → 매도
        if predicted_class == 0 and confidence > 0.4:
            logger.info("ML 하락 매도: %s (확신 %.0f%%, pnl %.1f%%)",
                       pos.stock_name, confidence * 100, pnl)
            return True

        # ML이 보합 예측 + 높은 확신 + 수익 중 → 모멘텀 소진, 익절
        if predicted_class == 1 and confidence > 0.6 and pnl > 1.0:
            logger.info("ML 보합→익절: %s (확신 %.0f%%, pnl %.1f%%)",
                       pos.stock_name, confidence * 100, pnl)
            return True

        # ML이 보합 예측 + 손실 중 + 오래 보유 → 기회비용, 청산
        if predicted_class == 1 and confidence > 0.5 and pnl < -1.0:
            # 보유일 체크
            import datetime
            try:
                entry_dt = datetime.datetime.strptime(pos.entry_date, "%Y%m%d")
                today_dt = datetime.datetime.now()
                holding_days = (today_dt - entry_dt).days
                if holding_days >= 3:
                    logger.info("ML 보합+손실→청산: %s (보유%d일, pnl %.1f%%)",
                               pos.stock_name, holding_days, pnl)
                    return True
            except (ValueError, AttributeError):
                pass

        return False
