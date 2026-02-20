"""
업종 데이터 수집
- ka20006: 업종 일봉 차트 (Chart 클래스)
"""
import logging
from datetime import datetime

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

# 주요 업종 코드
MAJOR_SECTORS = {
    "001": "종합(KOSPI)",
    "002": "대형주",
    "003": "중형주",
    "004": "소형주",
    "005": "음식료업",
    "006": "섬유의복",
    "007": "종이목재",
    "008": "화학",
    "009": "의약품",
    "010": "비금속광물",
    "011": "철강금속",
    "012": "기계",
    "013": "전기전자",
    "014": "의료정밀",
    "015": "운수장비",
    "016": "유통업",
    "017": "전기가스업",
    "018": "건설업",
    "019": "운수창고",
    "020": "통신업",
    "021": "금융업",
    "022": "은행",
    "024": "증권",
    "025": "보험",
    "026": "서비스업",
    "027": "제조업",
    "028": "코스닥",
}


class SectorCollector(BaseCollector):

    def collect_sector_daily(self, sector_code: str, sector_name: str) -> int:
        """업종 일봉 수집 (ka20006, Chart 클래스)."""
        result = self._api_call_raw(
            self.api.chart.industry_daily_chart_request_ka20006,
            inds_cd=sector_code,
            base_dt=datetime.now().strftime("%Y%m%d"),
        )
        if not result:
            return 0

        rows_data = result.get("inds_dt_pole_chart_qry", [])
        if not rows_data:
            # 다른 키 시도
            for k, v in result.items():
                if isinstance(v, list) and v and isinstance(v[0], dict) and "dt" in v[0]:
                    rows_data = v
                    break

        rows = []
        for r in rows_data[:120]:
            date = r.get("dt", "")
            if not date:
                continue

            close_val = r.get("jisu", r.get("cur_prc", "0"))
            close_str = str(close_val).replace("+", "").replace(",", "").replace("-", "")
            try:
                close = float(close_str)
            except ValueError:
                close = 0

            flu_str = str(r.get("flu_rt", "0")).replace("+", "")
            try:
                change_rate = float(flu_str)
            except ValueError:
                change_rate = 0

            rows.append({
                "sector_code": sector_code,
                "date": date,
                "sector_name": sector_name,
                "close": close,
                "change_rate": change_rate,
                "volume": abs(int(str(r.get("trde_qty", "0")).replace(",", "") or 0)),
                "market_cap": 0,
                "advance_count": 0,
                "decline_count": 0,
            })

        if rows:
            self.db.upsert_sector_data(rows)
            logger.debug("업종 %s: %d건 저장", sector_name, len(rows))
        return len(rows)

    def collect_all_sectors(self) -> int:
        """전체 주요 업종 수집."""
        total = 0
        for code, name in MAJOR_SECTORS.items():
            try:
                total += self.collect_sector_daily(code, name)
            except Exception as e:
                logger.error("업종 %s(%s) 수집 실패: %s", name, code, e)
        logger.info("업종 데이터 수집 완료: %d건", total)
        return total
