"""
가격 데이터 수집 — 일봉 (ka10081), 분봉 (ka10080), 주봉 (ka10082)
"""
import logging
from datetime import datetime, timedelta

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class PriceCollector(BaseCollector):

    def collect_daily(self, stock_code: str, days: int = 600) -> int:
        """
        일봉 데이터 수집 후 DB 저장.
        이미 있는 데이터는 최신분만 추가 수집.
        Returns: 저장된 행 수
        """
        # 이미 DB에 있는 최근 날짜 확인
        latest = self.db.get_latest_daily_date(stock_code)
        if latest:
            # 최근 날짜 이후만 수집 (여유분 +5일)
            from datetime import datetime as dt
            latest_dt = dt.strptime(latest, "%Y%m%d")
            today = dt.now()
            delta = (today - latest_dt).days
            if delta <= 1:
                logger.debug("%s: 이미 최신 데이터 존재 (%s)", stock_code, latest)
                return 0
            days = min(days, delta + 5)

        raw = self._api_call(self.api.get_daily_chart, stock_code, days)
        if not raw:
            return 0

        rows = []
        for i, r in enumerate(raw):
            change_rate = 0.0
            if i < len(raw) - 1 and raw[i + 1]["close"] > 0:
                prev_close = raw[i + 1]["close"]
                change_rate = round((r["close"] - prev_close) / prev_close * 100, 2)

            rows.append({
                "stock_code": stock_code,
                "date": r["date"],
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r["volume"],
                "change_rate": change_rate,
            })

        self.db.upsert_daily_prices(rows)
        logger.info("%s: 일봉 %d건 저장", stock_code, len(rows))
        return len(rows)

    def collect_daily_all(self, stock_codes: dict, days: int = 600) -> int:
        """전체 종목 일봉 수집."""
        total = 0
        for i, (code, name) in enumerate(stock_codes.items()):
            try:
                n = self.collect_daily(code, days)
                total += n
                if (i + 1) % 10 == 0:
                    logger.info("일봉 수집 진행: %d/%d 종목", i + 1, len(stock_codes))
            except Exception as e:
                logger.error("%s(%s) 일봉 수집 실패: %s", name, code, e)
        logger.info("일봉 수집 완료: 총 %d건 (%d종목)", total, len(stock_codes))
        return total

    def collect_minute(self, stock_code: str, interval: int = 5) -> int:
        """분봉 데이터 수집 후 DB 저장."""
        raw = self._api_call(self.api.get_minute_chart, stock_code, interval)
        if not raw:
            return 0

        rows = []
        for r in raw:
            rows.append({
                "stock_code": stock_code,
                "datetime": r["datetime"],
                "interval_min": interval,
                "open": r["open"],
                "high": r["high"],
                "low": r["low"],
                "close": r["close"],
                "volume": r["volume"],
            })

        self.db.upsert_minute_prices(rows)
        logger.debug("%s: %d분봉 %d건 저장", stock_code, interval, len(rows))
        return len(rows)

    def collect_weekly(self, stock_code: str) -> list[dict]:
        """주봉 데이터 수집 (피처용, DB 저장 안 함)."""
        try:
            self.rate_limiter.wait()
            result = self.api.chart.stock_weekly_chart_request_ka10082(
                stk_cd=stock_code,
                base_dt=datetime.now().strftime("%Y%m%d"),
                upd_stkpc_tp="1",
            )
            rows = result.get("stk_dt_pole_chart_qry", [])
            data = []
            for r in rows:
                data.append({
                    "date": r.get("dt", ""),
                    "open": abs(int(r.get("open_pric", 0))),
                    "high": abs(int(r.get("high_pric", 0))),
                    "low": abs(int(r.get("low_pric", 0))),
                    "close": abs(int(r.get("cur_prc", 0))),
                    "volume": abs(int(r.get("trde_qty", 0))),
                })
            return data
        except Exception as e:
            logger.error("%s 주봉 조회 실패: %s", stock_code, e)
            return []
