"""
순위 데이터 수집 — 거래량/등락률 순위
- ka10024: 거래량 상위
"""
import logging
from datetime import datetime

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class RankingCollector(BaseCollector):

    def collect_volume_ranking(self) -> int:
        """거래량 상위 종목 수집 (ka10024)."""
        result = self._api_call_raw(
            self.api.stock_info.trading_volume_update_request_ka10024,
            market_type="0",           # 전체
            cycle_type="1",            # 당일
            trade_quantity_type="0",   # 거래량
            stock_exchange_type="0",   # 전체거래소
        )
        if not result:
            return 0

        rows_data = result.get("trde_qty_updt", [])
        today = datetime.now().strftime("%Y%m%d")
        rows = []
        for i, r in enumerate(rows_data[:50]):
            code = r.get("stk_cd", "").strip()
            if not code:
                continue
            rows.append({
                "date": today,
                "ranking_type": "volume",
                "rank_num": i + 1,
                "stock_code": code,
                "stock_name": r.get("stk_nm", "").strip(),
                "price": abs(int(r.get("cur_prc", "0").replace("+", "").replace("-", "").replace(",", "") or 0)),
                "change_rate": float(r.get("flu_rt", "0").replace("+", "") or 0),
                "volume": abs(int(r.get("now_trde_qty", "0").replace(",", "") or 0)),
                "amount": 0,
            })

        self.db.upsert_ranking_data(rows)
        logger.info("거래량 순위 %d건 저장", len(rows))
        return len(rows)

    def collect_all_rankings(self) -> int:
        """전체 순위 수집."""
        total = self.collect_volume_ranking()
        return total
