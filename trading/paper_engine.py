"""
모의투자 엔진
- 1일 단타 + 3일 스윙 혼합 전략
- 수수료/슬리피지 반영
- 후회 분석 (매매 후 5일 뒤 "이렇게 했어야" 기록)
- 일일 손실 제한
"""
import logging
from datetime import datetime

from trading.portfolio import Portfolio
from trading.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class PaperEngine:

    def __init__(self, db, portfolio: Portfolio):
        self.db = db
        self.portfolio = portfolio
        self.risk = RiskManager(portfolio)

        from nn_config import (
            PAPER_COMMISSION_BUY, PAPER_COMMISSION_SELL,
            PAPER_TAX_RATE, PAPER_SLIPPAGE_BPS, ALL_STOCKS, SECTOR_MAP,
        )
        self.comm_buy = PAPER_COMMISSION_BUY
        self.comm_sell = PAPER_COMMISSION_SELL
        self.tax_rate = PAPER_TAX_RATE
        self.slippage_bps = PAPER_SLIPPAGE_BPS
        self.stock_names = ALL_STOCKS
        self.sector_map = SECTOR_MAP

    def execute_signals(self, predictions: list[dict], price_map: dict[str, int]):
        """
        예측 신호에 따라 매매 실행.
        predictions: [{stock_code, predicted_class, confidence, buy_prob, sell_prob, ...}]
        price_map: {stock_code: current_price}

        3-class 전략:
        - predicted_class 2 (상승) + conf > 0.7 → 1일 단타 (당일 매수 → 다음날 매도)
        - predicted_class 2 (상승) + conf 0.5~0.7 → 3일 스윙
        - predicted_class 0 (하락) → 보유 종목 매도
        """
        now = datetime.now().strftime("%Y%m%d %H:%M:%S")

        # 일일 리셋 (장 시작)
        self.risk.reset_daily()

        # 1. 현재가 업데이트
        self.portfolio.update_prices(price_map)

        # 2. 보유 기간 초과 종목 매도 (회전율 높이기)
        self._sell_expired_positions(price_map, now)

        # 3. 안전장치 체크 (긴급 손절 -5%, 트레일링 스톱)
        stop_actions = self.risk.check_stop_conditions()
        for action in stop_actions:
            code = action["stock_code"]
            if code in price_map:
                pnl = self._sell(code, price_map[code], reason=action["reason"], now=now)
                if pnl is not None:
                    self.risk.record_trade_pnl(pnl)

        # 4. ML 신호 기반 청산 (핵심: 손절/익절을 ML이 판단)
        for pred in predictions:
            code = pred["stock_code"]
            if self.risk.should_sell_on_signal(code, pred["predicted_class"], pred["confidence"]):
                if code in price_map:
                    signal = pred.get('signal', '')
                    conf = pred.get('confidence', 0)
                    pnl = self._sell(code, price_map[code],
                                     reason=f"ML {signal} ({conf:.0%})", now=now)
                    if pnl is not None:
                        self.risk.record_trade_pnl(pnl)

        # 5. 일일 손실 한도 체크
        if self.risk.is_trading_halted():
            logger.warning("일일 손실 한도 초과 — 매수 중단")
            return

        # 6. 매수 신호 처리 (상승 신호만)
        buy_candidates = [
            p for p in predictions
            if p["predicted_class"] == 2 and p["confidence"] > 0.4
        ]
        buy_candidates.sort(key=lambda x: -x["confidence"])

        for pred in buy_candidates:
            code = pred["stock_code"]
            if code not in price_map or price_map[code] <= 0:
                continue

            sector = self.sector_map.get(code, "")
            if not self.risk.can_buy(code, sector):
                continue

            price = price_map[code]

            # 신호 강도에 따른 보유 전략 결정
            conf = pred["confidence"]
            if conf > 0.7:
                hold_days = 1   # 단타
            elif conf > 0.5:
                hold_days = 3   # 스윙
            else:
                hold_days = 5   # 여유

            quantity = self.risk.compute_position_size(
                code, price, pred["confidence"],
                pred.get("buy_prob", 0.5),
            )

            if quantity <= 0:
                continue

            self._buy(code, price, quantity, sector=sector,
                       predicted_class=pred["predicted_class"],
                       confidence=pred["confidence"],
                       hold_days=hold_days, now=now)

    def _sell_expired_positions(self, price_map: dict, now: str):
        """보유 기간 초과 종목 자동 매도 (회전율 유지)."""
        today = datetime.now().strftime("%Y%m%d")

        for code, pos in list(self.portfolio.positions.items()):
            if not pos.entry_date or code not in price_map:
                continue

            try:
                entry_dt = datetime.strptime(pos.entry_date, "%Y%m%d")
                today_dt = datetime.strptime(today, "%Y%m%d")
                holding_days = (today_dt - entry_dt).days
            except ValueError:
                continue

            # hold_days 메타데이터가 있으면 그 값, 없으면 5일 기본
            max_hold = getattr(pos, '_hold_days', 5)

            if holding_days >= max_hold:
                pnl = self._sell(
                    code, price_map[code],
                    reason=f"보유기간 만료 ({holding_days}일 >= {max_hold}일)",
                    now=now,
                )
                if pnl is not None:
                    self.risk.record_trade_pnl(pnl)

    def _buy(self, stock_code: str, price: int, quantity: int,
             sector: str = "", predicted_class: int = None,
             confidence: float = None, hold_days: int = 3, now: str = ""):
        """매수 실행 (수수료/슬리피지 반영)."""
        # 슬리피지 (매수 시 불리하게)
        slippage_amount = int(price * quantity * self.slippage_bps / 10000)
        effective_price = price + int(price * self.slippage_bps / 10000)

        # 수수료
        commission = int(effective_price * quantity * self.comm_buy)

        name = self.stock_names.get(stock_code, stock_code)

        success = self.portfolio.buy(
            stock_code, name, quantity, effective_price,
            commission=commission, slippage=slippage_amount,
            sector=sector,
        )

        if success:
            # hold_days 메타데이터 저장
            if stock_code in self.portfolio.positions:
                self.portfolio.positions[stock_code]._hold_days = hold_days

            from nn_config import CLASS_NAMES
            signal = CLASS_NAMES[predicted_class] if predicted_class is not None else ""
            hold_label = {1: "단타", 3: "스윙", 5: "중기"}.get(hold_days, f"{hold_days}일")
            self.db.insert_trade({
                "trade_type": "paper",
                "stock_code": stock_code,
                "stock_name": name,
                "side": "buy",
                "quantity": quantity,
                "price": effective_price,
                "amount": effective_price * quantity,
                "commission": commission,
                "slippage": slippage_amount,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "reason": f"모델: {signal} ({confidence:.0%}) [{hold_label}]" if confidence else "",
                "executed_at": now or datetime.now().strftime("%Y%m%d %H:%M:%S"),
            })

    def _sell(self, stock_code: str, price: int, reason: str = "", now: str = "") -> float | None:
        """전량 매도 (수수료/세금/슬리피지 반영). Returns 수익률 %."""
        pos = self.portfolio.positions.get(stock_code)
        if not pos:
            return None

        quantity = pos.quantity

        # 슬리피지 (매도 시 불리하게)
        effective_price = price - int(price * self.slippage_bps / 10000)

        # 수수료 + 세금
        commission = int(effective_price * quantity * self.comm_sell)
        tax = int(effective_price * quantity * self.tax_rate)

        pnl = self.portfolio.sell(
            stock_code, quantity, effective_price,
            commission=commission, tax=tax,
        )

        self.db.insert_trade({
            "trade_type": "paper",
            "stock_code": stock_code,
            "stock_name": pos.stock_name,
            "side": "sell",
            "quantity": quantity,
            "price": effective_price,
            "amount": effective_price * quantity,
            "commission": commission,
            "tax": tax,
            "reason": reason,
            "executed_at": now or datetime.now().strftime("%Y%m%d %H:%M:%S"),
        })

        logger.info("매도 완료: %s (수익률 %.2f%%) — %s", pos.stock_name, pnl or 0, reason)
        return pnl

    def analyze_regret(self):
        """
        후회 분석: 5일 이상 지난 매매의 사후 평가.
        "이 매매를 안 했으면?" / "반대로 했으면?" 을 분석.
        """
        from nn_config import REGRET_LOOKBACK, CLASS_NAMES

        pending_trades = self.db.get_trades(trade_type="paper", pending_regret=True)
        if not pending_trades:
            return

        logger.info("후회 분석 대상: %d건", len(pending_trades))

        for trade in pending_trades:
            code = trade["stock_code"]
            trade_date = trade["executed_at"][:8]

            # 매매 후 5일간 가격 변화
            prices = self.db.get_daily_prices(code, start_date=trade_date)
            if len(prices) < REGRET_LOOKBACK + 1:
                continue

            trade_price = trade["price"]
            future_prices = [p["close"] for p in prices[1:REGRET_LOOKBACK + 1]]

            if not future_prices or trade_price <= 0:
                continue

            # 5일 후 수익률
            actual_return_5d = (future_prices[-1] - trade_price) / trade_price * 100

            # 후회 분석
            side = trade["side"]
            if side == "buy":
                if actual_return_5d < -2:
                    regret_score = abs(actual_return_5d) / 10
                    optimal = "관망"
                    lesson = f"매수 후 {actual_return_5d:.1f}% 하락 — 매수 타이밍 불량"
                elif actual_return_5d > 3:
                    regret_score = 0
                    optimal = "매수 유지"
                    lesson = f"매수 후 {actual_return_5d:.1f}% 상승 — 좋은 진입"
                else:
                    regret_score = 0.1
                    optimal = "관망"
                    lesson = f"매수 후 {actual_return_5d:.1f}% — 보합"
            else:
                if actual_return_5d > 2:
                    regret_score = actual_return_5d / 10
                    optimal = "보유 유지"
                    lesson = f"매도 후 {actual_return_5d:.1f}% 상승 — 조기 매도"
                elif actual_return_5d < -3:
                    regret_score = 0
                    optimal = "매도 (잘함)"
                    lesson = f"매도 후 {actual_return_5d:.1f}% 하락 — 좋은 매도"
                else:
                    regret_score = 0.1
                    optimal = "매도"
                    lesson = f"매도 후 {actual_return_5d:.1f}% — 보합"

            regret_score = min(1.0, max(0.0, regret_score))

            # DB 저장
            self.db.update_trade_regret(
                trade["id"], regret_score,
                lesson, actual_return_5d, optimal,
            )

            if regret_score > 0.3:
                self.db.insert_regret({
                    "trade_id": trade["id"],
                    "stock_code": code,
                    "trade_date": trade_date,
                    "action_taken": f"{side} {trade['quantity']}주 @ {trade_price}",
                    "optimal_action": optimal,
                    "regret_score": regret_score,
                    "return_if_optimal": -actual_return_5d if side == "buy" else actual_return_5d,
                    "return_actual": actual_return_5d if side == "buy" else -actual_return_5d,
                    "lesson": lesson,
                })

        logger.info("후회 분석 완료: %d건 처리", len(pending_trades))

    def update_prices(self, price_map: dict):
        """현재가 업데이트."""
        self.portfolio.update_prices(price_map)

    def daily_close(self):
        """일일 마감 처리."""
        self.portfolio.record_daily()
        snapshot = self.portfolio.snapshot()
        self.db.insert_portfolio_snapshot(snapshot)
        logger.info(
            "일일 마감: 총평가 %s원, 수익률 %+.2f%%",
            f"{snapshot['total_value']:,}", snapshot['cumulative_return'],
        )
        return snapshot

    def generate_report(self) -> str:
        """텔레그램용 일일 리포트 텍스트."""
        from models.evaluator import Evaluator

        now = datetime.now()
        metrics = Evaluator.trading_metrics(self.portfolio.daily_returns)
        live_check = Evaluator.check_live_criteria(metrics)

        lines = [
            f"🤖 [모의투자 일일 리포트] {now.strftime('%Y-%m-%d')}",
            "",
            self.portfolio.summary_text(),
            "",
            f"📊 성과 지표",
            f"  Sharpe: {metrics.get('sharpe_ratio', 0):.2f}",
            f"  MDD: {metrics.get('max_drawdown', 0):.1f}%",
            f"  승률: {self.portfolio.win_rate:.1%}",
            f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}",
            "",
        ]

        # 실전 전환 체크
        if live_check["overall_passed"]:
            lines.append("✅ 실전 전환 기준 충족!")
        else:
            lines.append("📋 실전 전환 기준")
            for name, check in live_check["checks"].items():
                emoji = "✅" if check["passed"] else "❌"
                lines.append(f"  {emoji} {name}: {check['value']} (기준: {check['threshold']})")

        # 오늘 매매 내역
        today = now.strftime("%Y%m%d")
        today_trades = self.db.get_trades(trade_type="paper", start_date=today)
        if today_trades:
            lines.append("")
            lines.append(f"📝 오늘 매매 ({len(today_trades)}건)")
            for t in today_trades:
                side_emoji = "🔴" if t["side"] == "sell" else "🟢"
                lines.append(
                    f"  {side_emoji} {t['stock_name']} {t['side'].upper()} "
                    f"{t['quantity']}주 @ {t['price']:,}원"
                )

        # 후회 (최근 큰 후회)
        recent_regrets = self.db.execute_raw(
            "SELECT * FROM regret_log WHERE regret_score > 0.3 ORDER BY id DESC LIMIT 3"
        )
        if recent_regrets:
            lines.append("")
            lines.append("😔 최근 후회 (학습 반영 예정)")
            for r in recent_regrets:
                lines.append(f"  - {r.get('lesson', '')}")

        return "\n".join(lines)
