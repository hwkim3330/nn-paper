"""
키움 REST API 래퍼
- kiwoom-rest-api 공식 패키지 활용
- 일봉 차트 (ka10081), 호가 (ka10004), 현재가 (ka10001)
"""
import os
import logging
from datetime import datetime

# config.py가 .env를 자동 로딩한 후 패키지용 환경변수 설정
from config import KIWOOM_APP_KEY, KIWOOM_APP_SECRET, CHART_DAYS
os.environ["KIWOOM_API_KEY"] = KIWOOM_APP_KEY
os.environ["KIWOOM_API_SECRET"] = KIWOOM_APP_SECRET

from kiwoom_rest_api.auth.token import TokenManager
from kiwoom_rest_api.koreanstock.chart import Chart
from kiwoom_rest_api.koreanstock.market_condition import MarketCondition
from kiwoom_rest_api.koreanstock.stockinfo import StockInfo

logger = logging.getLogger(__name__)


class KiwoomAPI:
    def __init__(self):
        base_url = "https://api.kiwoom.com"
        self.token_mgr = TokenManager()
        self.chart = Chart(base_url=base_url, token_manager=self.token_mgr)
        self.market = MarketCondition(base_url=base_url, token_manager=self.token_mgr)
        self.stock_info = StockInfo(base_url=base_url, token_manager=self.token_mgr)
        logger.info("키움 API 초기화 완료")

    # ----------------------------------------------------------
    # 일봉 차트 (ka10081)
    # ----------------------------------------------------------
    def get_daily_chart(self, stock_code: str, days: int = None) -> list[dict]:
        """
        일봉 차트 데이터 조회.
        반환: [{date, open, high, low, close, volume}, ...]
        최신 날짜가 첫 번째.
        """
        if days is None:
            days = CHART_DAYS
        base_dt = datetime.now().strftime("%Y%m%d")

        all_data = []
        next_key = ""
        cont_yn = "N"

        while len(all_data) < days:
            try:
                result = self.chart.stock_daily_chart_request_ka10081(
                    stk_cd=stock_code,
                    base_dt=base_dt,
                    upd_stkpc_tp="1",
                    cont_yn=cont_yn,
                    next_key=next_key,
                )
            except Exception as e:
                logger.error("일봉 조회 실패 %s: %s", stock_code, e)
                break

            rows = result.get("stk_dt_pole_chart_qry", [])
            if not rows:
                break

            for row in rows:
                all_data.append({
                    "date": row.get("dt", ""),
                    "open": abs(int(row.get("open_pric", 0))),
                    "high": abs(int(row.get("high_pric", 0))),
                    "low": abs(int(row.get("low_pric", 0))),
                    "close": abs(int(row.get("cur_prc", 0))),
                    "volume": abs(int(row.get("trde_qty", 0))),
                })

            # 연속 조회
            next_key = result.get("next-key", result.get("next_key", ""))
            if result.get("cont-yn", result.get("cont_yn", "N")) == "Y" and next_key:
                cont_yn = "Y"
            else:
                break

        return all_data[:days]

    # ----------------------------------------------------------
    # 호가 (ka10004)
    # ----------------------------------------------------------
    def get_orderbook(self, stock_code: str) -> dict:
        """
        호가 데이터 조회 (10단계).
        반환: {asks, bids, total_ask_volume, total_bid_volume}
        """
        try:
            result = self.market.stock_quote_request_ka10004(
                stock_code=stock_code,
            )
        except Exception as e:
            logger.error("호가 조회 실패 %s: %s", stock_code, e)
            return {"asks": [], "bids": [], "total_ask_volume": 0, "total_bid_volume": 0}

        asks = []
        bids = []
        # 1차 호가: fpr (first price)
        ask1 = abs(int(result.get("sel_fpr_bid", 0) or 0))
        ask1_vol = abs(int(result.get("sel_fpr_req", 0) or 0))
        bid1 = abs(int(result.get("buy_fpr_bid", 0) or 0))
        bid1_vol = abs(int(result.get("buy_fpr_req", 0) or 0))
        if ask1:
            asks.append({"price": ask1, "volume": ask1_vol})
        if bid1:
            bids.append({"price": bid1, "volume": bid1_vol})
        # 2~10차 호가: Nth_pre
        for i in range(2, 11):
            ask_price = abs(int(result.get(f"sel_{i}th_pre_bid", 0) or 0))
            ask_vol = abs(int(result.get(f"sel_{i}th_pre_req", 0) or 0))
            bid_price = abs(int(result.get(f"buy_{i}th_pre_bid", 0) or 0))
            bid_vol = abs(int(result.get(f"buy_{i}th_pre_req", 0) or 0))
            if ask_price:
                asks.append({"price": ask_price, "volume": ask_vol})
            if bid_price:
                bids.append({"price": bid_price, "volume": bid_vol})

        total_ask = abs(int(result.get("tot_sel_req", 0) or 0))
        total_bid = abs(int(result.get("tot_buy_req", 0) or 0))

        return {
            "asks": asks,
            "bids": bids,
            "total_ask_volume": total_ask,
            "total_bid_volume": total_bid,
        }

    # ----------------------------------------------------------
    # 현재가 (ka10001)
    # ----------------------------------------------------------
    def get_current_price(self, stock_code: str) -> dict:
        """
        현재가 정보 조회.
        반환: {price, change, change_rate, volume, ...}
        """
        try:
            result = self.stock_info.basic_stock_information_request_ka10001(
                stock_code=stock_code,
            )
        except Exception as e:
            logger.error("현재가 조회 실패 %s: %s", stock_code, e)
            return {}

        return {
            "price": abs(int(result.get("cur_prc", 0) or 0)),
            "change": int(result.get("flu_amt", 0) or 0),
            "change_rate": float(result.get("flu_rt", 0) or 0),
            "volume": abs(int(result.get("trde_qty", 0) or 0)),
            "high": abs(int(result.get("high_pric", 0) or 0)),
            "low": abs(int(result.get("low_pric", 0) or 0)),
            "open": abs(int(result.get("open_pric", 0) or 0)),
        }

    # ----------------------------------------------------------
    # 분봉 (ka10080)
    # ----------------------------------------------------------
    def get_minute_chart(self, stock_code: str, interval: int = 5) -> list[dict]:
        """분봉 데이터 조회."""
        try:
            result = self.chart.stock_minute_chart_request_ka10080(
                stk_cd=stock_code,
                base_dt=datetime.now().strftime("%Y%m%d"),
                tic_scope=str(interval),
                upd_stkpc_tp="1",
            )
        except Exception as e:
            logger.error("분봉 조회 실패 %s: %s", stock_code, e)
            return []

        rows = result.get("stk_dt_pole_chart_qry", [])
        data = []
        for row in rows:
            data.append({
                "datetime": row.get("dt", ""),
                "open": abs(int(row.get("open_pric", 0))),
                "high": abs(int(row.get("high_pric", 0))),
                "low": abs(int(row.get("low_pric", 0))),
                "close": abs(int(row.get("cur_prc", 0))),
                "volume": abs(int(row.get("trde_qty", 0))),
            })
        return data

    # ----------------------------------------------------------
    # NXT (넥스트레이드) 조회
    # ----------------------------------------------------------
    @staticmethod
    def _nxt_code(stock_code: str) -> str:
        """종목코드를 NXT 형식으로 변환."""
        base = stock_code.split("_")[0]  # 이미 _NX 등이 붙어있으면 제거
        return f"{base}_NX"

    @staticmethod
    def _sor_code(stock_code: str) -> str:
        """종목코드를 SOR(통합) 형식으로 변환."""
        base = stock_code.split("_")[0]
        return f"{base}_AL"

    def get_nxt_price(self, stock_code: str) -> dict:
        """
        NXT(넥스트레이드) 현재가 조회.
        반환: {price, change_rate, volume, high, low, ...}
        """
        nxt_code = self._nxt_code(stock_code)
        try:
            result = self.stock_info.basic_stock_information_request_ka10001(
                stock_code=nxt_code,
            )
        except Exception as e:
            logger.error("NXT 현재가 조회 실패 %s: %s", nxt_code, e)
            return {}

        price = abs(int(result.get("cur_prc", 0) or 0))
        if not price:
            return {}
        return {
            "market": "NXT",
            "code": nxt_code,
            "price": price,
            "change": int(result.get("pred_pre", 0) or 0),
            "change_rate": float(result.get("flu_rt", 0) or 0),
            "volume": abs(int(result.get("trde_qty", 0) or 0)),
            "high": abs(int(result.get("high_pric", 0) or 0)),
            "low": abs(int(result.get("low_pric", 0) or 0)),
            "open": abs(int(result.get("open_pric", 0) or 0)),
        }

    def get_sor_price(self, stock_code: str) -> dict:
        """
        SOR(통합 KRX+NXT) 현재가 조회.
        반환: {price, change_rate, volume, ...}
        """
        sor_code = self._sor_code(stock_code)
        try:
            result = self.stock_info.basic_stock_information_request_ka10001(
                stock_code=sor_code,
            )
        except Exception as e:
            logger.error("SOR 현재가 조회 실패 %s: %s", sor_code, e)
            return {}

        price = abs(int(result.get("cur_prc", 0) or 0))
        if not price:
            return {}
        return {
            "market": "SOR",
            "code": sor_code,
            "price": price,
            "change": int(result.get("pred_pre", 0) or 0),
            "change_rate": float(result.get("flu_rt", 0) or 0),
            "volume": abs(int(result.get("trde_qty", 0) or 0)),
            "high": abs(int(result.get("high_pric", 0) or 0)),
            "low": abs(int(result.get("low_pric", 0) or 0)),
            "open": abs(int(result.get("open_pric", 0) or 0)),
        }

    def get_nxt_orderbook(self, stock_code: str) -> dict:
        """NXT 호가 조회."""
        return self.get_orderbook(self._nxt_code(stock_code))

    def get_multi_market_price(self, stock_code: str) -> dict:
        """
        KRX + NXT + SOR 3개 시장 현재가 비교.
        반환: {krx: {...}, nxt: {...}, sor: {...}, spread, spread_pct}
        """
        krx = self.get_current_price(stock_code)
        nxt = self.get_nxt_price(stock_code)
        sor = self.get_sor_price(stock_code)

        result = {"krx": krx, "nxt": nxt, "sor": sor, "spread": 0, "spread_pct": 0.0}

        krx_price = krx.get("price", 0)
        nxt_price = nxt.get("price", 0)
        if krx_price and nxt_price:
            result["spread"] = nxt_price - krx_price
            result["spread_pct"] = round((nxt_price - krx_price) / krx_price * 100, 3)

        return result
