"""
실거래 엔진 (Phase 5: 모의투자 기준 충족 후)
- kt10000/kt10001: 주문 실행
- 서킷브레이커, 긴급 청산
- 단계적 자금 투입
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LiveEngine:
    """
    실거래 엔진 (미구현 — Phase 5에서 활성화)
    모의투자 60일+ & 기준 충족 시에만 사용.
    """

    def __init__(self, db, portfolio):
        self.db = db
        self.portfolio = portfolio
        self.enabled = False
        self.capital_stage = 0  # 0~4
        logger.warning("LiveEngine 초기화 — 아직 비활성 상태")

    def is_ready(self) -> bool:
        """실전 전환 준비 상태."""
        from models.evaluator import Evaluator
        metrics = Evaluator.trading_metrics(self.portfolio.daily_returns)
        check = Evaluator.check_live_criteria(metrics)
        return check["overall_passed"]

    def execute_order(self, stock_code: str, side: str, quantity: int, price: int):
        """실제 주문 실행 (미구현)."""
        if not self.enabled:
            logger.error("LiveEngine이 비활성 상태입니다")
            return None
        # TODO: kt10000 (시장가) / kt10001 (지정가) 구현
        raise NotImplementedError("실거래 주문은 Phase 5에서 구현됩니다")

    def emergency_liquidate(self):
        """긴급 전량 청산."""
        if not self.enabled:
            return
        logger.critical("🚨 긴급 청산 실행!")
        # TODO: 모든 보유 종목 시장가 매도
        raise NotImplementedError("긴급 청산은 Phase 5에서 구현됩니다")
