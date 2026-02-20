"""
수집 스케줄러 — 장중/장전/장후 자동 수집
"""
import logging
import time
from datetime import datetime

from .price_collector import PriceCollector
from .orderbook_collector import OrderbookCollector
from .flow_collector import FlowCollector
from .ranking_collector import RankingCollector
from .sector_collector import SectorCollector
from .market_collector import MarketCollector

logger = logging.getLogger(__name__)


class CollectorScheduler:
    """데이터 수집 스케줄 관리."""

    def __init__(self, api, db, stock_codes: dict):
        self.api = api
        self.db = db
        self.stock_codes = stock_codes

        self.price = PriceCollector(api, db)
        self.orderbook = OrderbookCollector(api, db)
        self.flow = FlowCollector(api, db)
        self.ranking = RankingCollector(api, db)
        self.sector = SectorCollector(api, db)
        self.market = MarketCollector(api, db)

        self._last_orderbook = 0
        self._last_daily = ""

    @staticmethod
    def is_market_hours() -> bool:
        """장중 여부 (09:00~15:30 평일)."""
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        t = now.strftime("%H:%M")
        return "09:00" <= t <= "15:30"

    @staticmethod
    def is_market_day() -> bool:
        return datetime.now().weekday() < 5

    def run_pre_market(self):
        """장전 수집 (08:00) — US 시장 + 뉴스."""
        logger.info("=== 장전 수집 시작 ===")
        self.market.collect_us_market()

    def run_post_market(self):
        """장후 수집 (16:00~17:00) — 일봉, 수급, 프로그램, 업종, 순위."""
        logger.info("=== 장후 수집 시작 ===")
        today = datetime.now().strftime("%Y%m%d")

        if self._last_daily == today:
            logger.info("오늘 이미 장후 수집 완료")
            return

        # 일봉 (증분)
        self.price.collect_daily_all(self.stock_codes, days=5)

        # 투자자 매매동향
        self.flow.collect_investor_flow_all(self.stock_codes, days=5)

        # 프로그램 매매
        self.flow.collect_program_trading_all(self.stock_codes)

        # 업종
        self.sector.collect_all_sectors()

        # 순위
        self.ranking.collect_all_rankings()

        # 체결강도
        from config import WATCHLIST
        for code in WATCHLIST:
            try:
                self.flow.collect_execution_strength(code)
            except Exception as e:
                logger.error("%s 체결강도 실패: %s", code, e)

        self._last_daily = today
        logger.info("=== 장후 수집 완료 ===")

    def run_intraday(self):
        """장중 수집 — 호가 스냅샷 (매분)."""
        now = time.time()
        if now - self._last_orderbook < 60:
            return
        self._last_orderbook = now

        from config import WATCHLIST
        self.orderbook.collect_all_snapshots(WATCHLIST)

    def run_backfill(self, days: int = 600):
        """과거 데이터 백필 (최초 1회)."""
        logger.info("=== 백필 시작: %d일 ===", days)

        # 일봉
        self.price.collect_daily_all(self.stock_codes, days=days)

        # 투자자 매매동향 (최대 60일)
        self.flow.collect_investor_flow_all(self.stock_codes, days=min(days, 60))

        # 프로그램 매매
        self.flow.collect_program_trading_all(self.stock_codes)

        # 업종
        self.sector.collect_all_sectors()

        # 시장
        self.market.collect_us_market()

        # 순위
        self.ranking.collect_all_rankings()

        # 체결강도 (관심종목만)
        from config import WATCHLIST
        for code in WATCHLIST:
            try:
                self.flow.collect_execution_strength(code)
            except Exception as e:
                logger.error("%s 체결강도 실패: %s", code, e)

        stats = self.db.get_db_stats()
        logger.info("=== 백필 완료 ===")
        logger.info("DB 통계: %s", stats)
        return stats
