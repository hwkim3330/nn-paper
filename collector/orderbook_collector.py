"""
호가 데이터 수집 (ka10004)
- 장중 매분 스냅샷 → DB 저장
"""
import json
import logging
from datetime import datetime

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class OrderbookCollector(BaseCollector):

    def collect_snapshot(self, stock_code: str) -> dict | None:
        """
        현재 호가 스냅샷 수집 → DB 저장.
        Returns: 저장된 데이터 dict
        """
        raw = self._api_call(self.api.get_orderbook, stock_code)
        if not raw or (not raw.get("asks") and not raw.get("bids")):
            return None

        total_ask = raw.get("total_ask_volume", 0)
        total_bid = raw.get("total_bid_volume", 0)
        ratio = total_bid / total_ask if total_ask > 0 else 0

        ask_prices = [a["price"] for a in raw.get("asks", [])]
        bid_prices = [b["price"] for b in raw.get("bids", [])]

        spread_pct = 0.0
        if ask_prices and bid_prices and bid_prices[0] > 0:
            spread_pct = round((ask_prices[0] - bid_prices[0]) / bid_prices[0] * 100, 4)

        data = {
            "stock_code": stock_code,
            "datetime": datetime.now().strftime("%Y%m%d%H%M%S"),
            "total_ask_volume": total_ask,
            "total_bid_volume": total_bid,
            "ask_bid_ratio": round(ratio, 4),
            "spread_pct": spread_pct,
            "ask_prices": ask_prices,
            "ask_volumes": [a["volume"] for a in raw.get("asks", [])],
            "bid_prices": bid_prices,
            "bid_volumes": [b["volume"] for b in raw.get("bids", [])],
        }

        self.db.insert_orderbook(data)
        return data

    def collect_all_snapshots(self, stock_codes: dict) -> int:
        """전체 관심종목 호가 스냅샷 수집."""
        count = 0
        for code in stock_codes:
            try:
                result = self.collect_snapshot(code)
                if result:
                    count += 1
            except Exception as e:
                logger.error("%s 호가 수집 실패: %s", code, e)
        logger.debug("호가 스냅샷 %d종목 수집", count)
        return count
