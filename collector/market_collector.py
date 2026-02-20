"""
시장 데이터 수집 — US 지수, 환율, VIX 등
- 네이버 금융 API 활용 (market_news.py 재사용)
"""
import logging
from datetime import datetime

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class MarketCollector(BaseCollector):

    def collect_us_market(self) -> int:
        """미국 시장 + 환율 수집 → market_data 테이블."""
        # 기존 market_news.py 활용
        try:
            from market_news import get_us_market
            us_data = get_us_market()
        except Exception as e:
            logger.error("US 시장 데이터 수집 실패: %s", e)
            return 0

        today = datetime.now().strftime("%Y%m%d")
        rows = []

        # 주요 지수
        for key in ["dow", "sp500", "nasdaq", "sox", "vix", "usdkrw"]:
            d = us_data.get(key, {})
            if not d or not d.get("price"):
                continue
            rows.append({
                "date": today,
                "symbol": key.upper(),
                "name": d.get("name", key),
                "price": d["price"],
                "change": d.get("change", 0),
                "change_pct": d.get("change_pct", 0),
            })

        # 미국 개별종목
        us_stocks = us_data.get("us_stocks", {})
        for symbol, info in us_stocks.items():
            rows.append({
                "date": today,
                "symbol": f"US_{symbol}",
                "name": info.get("name", symbol),
                "price": info.get("price", 0),
                "change": 0,
                "change_pct": info.get("change_pct", 0),
            })

        if rows:
            self.db.upsert_market_data(rows)
            logger.info("시장 데이터 %d건 저장", len(rows))
        return len(rows)
