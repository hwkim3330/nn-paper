"""
수급 데이터 수집 — 외국인/기관/프로그램/체결강도
- ka10060: 종목별 투자자 기관별 추이 (일별, Chart 클래스)
- ka90013: 종목별 프로그램 매매 추이 (일별)
- ka10047: 일별 체결강도
"""
import logging
from datetime import datetime

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


def _parse_signed(val) -> int:
    """'+7800' / '--294626' / '-100' 형태 파싱."""
    if not val:
        return 0
    s = str(val).replace(",", "").strip()
    s = s.lstrip("+")
    # '--' 은 음수
    if s.startswith("--"):
        s = "-" + s[2:]
    try:
        return int(s)
    except ValueError:
        return 0


class FlowCollector(BaseCollector):

    def collect_investor_flow(self, stock_code: str, days: int = 60) -> int:
        """
        외국인/기관 매매동향 수집 (ka10060).
        Returns: 저장된 행 수
        """
        result = self._api_call_raw(
            self.api.chart.stockwise_investor_institution_chart_request_ka10060,
            dt=datetime.now().strftime("%Y%m%d"),
            stk_cd=stock_code,
            amt_qty_tp="2",   # 수량
            trde_tp="0",      # 순매수
            unit_tp="1",      # 주
        )
        if not result:
            return 0

        rows_data = result.get("stk_invsr_orgn_chart", [])
        if not rows_data:
            return 0

        rows = []
        for r in rows_data[:days]:
            date = r.get("dt", "")
            if not date:
                continue

            foreign_net = _parse_signed(r.get("frgnr_invsr", 0))
            institution_net = _parse_signed(r.get("orgn", 0))
            individual_net = _parse_signed(r.get("ind_invsr", 0))

            rows.append({
                "stock_code": stock_code,
                "date": date,
                "foreign_buy": max(foreign_net, 0),
                "foreign_sell": abs(min(foreign_net, 0)),
                "foreign_net": foreign_net,
                "institution_buy": max(institution_net, 0),
                "institution_sell": abs(min(institution_net, 0)),
                "institution_net": institution_net,
                "individual_net": individual_net,
                "foreign_hold_ratio": 0,  # ka10060에는 보유율 없음
            })

        self.db.upsert_investor_flow(rows)
        logger.info("%s: 투자자 매매동향 %d건 저장", stock_code, len(rows))
        return len(rows)

    def collect_investor_flow_all(self, stock_codes: dict, days: int = 60) -> int:
        """전체 종목 투자자 매매동향 수집."""
        total = 0
        for i, (code, name) in enumerate(stock_codes.items()):
            try:
                n = self.collect_investor_flow(code, days)
                total += n
                if (i + 1) % 10 == 0:
                    logger.info("투자자 매매동향 진행: %d/%d", i + 1, len(stock_codes))
            except Exception as e:
                logger.error("%s(%s) 투자자 동향 실패: %s", name, code, e)
        return total

    def collect_program_trading(self, stock_code: str) -> int:
        """
        종목별 프로그램 매매 수집 (ka90013).
        Returns: 저장된 행 수
        """
        result = self._api_call_raw(
            self.api.market.stockwise_program_trading_by_day_request_ka90013,
            stock_code=stock_code,
            amount_quantity_type="2",  # 수량
            date=datetime.now().strftime("%Y%m%d"),
        )
        if not result:
            return 0

        rows_data = result.get("stk_daly_prm_trde_trnsn", [])
        rows = []
        for r in rows_data:
            date = r.get("dt", "")
            if not date:
                continue

            buy_qty = abs(_parse_signed(r.get("prm_buy_qty", 0)))
            sell_qty = abs(_parse_signed(r.get("prm_sell_qty", 0)))
            net_qty = _parse_signed(r.get("prm_netprps_qty", 0))

            rows.append({
                "stock_code": stock_code,
                "date": date,
                "program_buy": buy_qty,
                "program_sell": sell_qty,
                "program_net": net_qty,
                "arbitrage_buy": 0,
                "arbitrage_sell": 0,
                "non_arbitrage_buy": buy_qty,
                "non_arbitrage_sell": sell_qty,
            })

        if rows:
            self.db.upsert_program_trading(rows)
            logger.debug("%s: 프로그램 매매 %d건 저장", stock_code, len(rows))
        return len(rows)

    def collect_program_trading_all(self, stock_codes: dict) -> int:
        total = 0
        for i, (code, name) in enumerate(stock_codes.items()):
            try:
                total += self.collect_program_trading(code)
                if (i + 1) % 10 == 0:
                    logger.info("프로그램 매매 진행: %d/%d", i + 1, len(stock_codes))
            except Exception as e:
                logger.error("%s 프로그램 매매 실패: %s", code, e)
        return total

    def collect_execution_strength(self, stock_code: str) -> int:
        """
        체결강도 수집 (ka10047 일별).
        체결강도 = 매수체결량 / 매도체결량 * 100 (100 이상 = 매수우위)
        """
        result = self._api_call_raw(
            self.api.market.execution_strength_by_day_request_ka10047,
            stock_code=stock_code,
        )
        if not result:
            return 0

        rows_data = result.get("cntr_str_daly", [])
        # 체결강도는 features에서 직접 사용 — DB에는 market_data로 저장
        today = datetime.now().strftime("%Y%m%d")
        rows = []
        for r in rows_data[:60]:
            date = r.get("dt", "")
            if not date:
                continue
            strength = float(r.get("cntr_str", 0) or 0)
            rows.append({
                "date": date,
                "symbol": f"EXEC_{stock_code}",
                "name": f"체결강도_{stock_code}",
                "price": strength,
                "change": 0,
                "change_pct": float(r.get("flu_rt", "0").replace("+", "") or 0),
            })

        if rows:
            self.db.upsert_market_data(rows)
        return len(rows)
