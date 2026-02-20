"""
포트폴리오 상태 관리
- 보유 종목, 현금, 평가금, 수익률 추적
- 일별 스냅샷 → DB 저장
"""
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class Position:
    stock_code: str
    stock_name: str
    quantity: int
    avg_price: float        # 평균 매수가
    current_price: float    # 현재가
    highest_price: float    # 보유 중 최고가 (트레일링 스톱용)
    sector: str = ""
    entry_date: str = ""

    @property
    def market_value(self) -> int:
        return int(self.quantity * self.current_price)

    @property
    def pnl(self) -> float:
        """손익률 (%)."""
        if self.avg_price <= 0:
            return 0
        return (self.current_price - self.avg_price) / self.avg_price * 100

    @property
    def pnl_amount(self) -> int:
        """손익금 (원)."""
        return int((self.current_price - self.avg_price) * self.quantity)

    def to_dict(self) -> dict:
        return {
            "stock_code": self.stock_code,
            "stock_name": self.stock_name,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "current_price": self.current_price,
            "highest_price": self.highest_price,
            "market_value": self.market_value,
            "pnl": round(self.pnl, 2),
            "pnl_amount": self.pnl_amount,
            "sector": self.sector,
            "entry_date": self.entry_date,
        }


class Portfolio:

    def __init__(self, initial_capital: int, trade_type: str = "paper"):
        self.trade_type = trade_type
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}  # code → Position
        self.daily_returns: list[float] = []
        self._prev_total = initial_capital
        self._peak_value = initial_capital
        self._trade_count = 0
        self._win_count = 0

    @property
    def total_value(self) -> int:
        """총 평가금 (현금 + 주식)."""
        stock_val = sum(p.market_value for p in self.positions.values())
        return self.cash + stock_val

    @property
    def stock_value(self) -> int:
        return sum(p.market_value for p in self.positions.values())

    @property
    def num_positions(self) -> int:
        return len(self.positions)

    @property
    def cumulative_return(self) -> float:
        """누적 수익률 (%)."""
        return (self.total_value - self.initial_capital) / self.initial_capital * 100

    @property
    def max_drawdown(self) -> float:
        """최대 낙폭 (%)."""
        if self._peak_value <= 0:
            return 0
        return (self.total_value - self._peak_value) / self._peak_value * 100

    @property
    def win_rate(self) -> float:
        if self._trade_count == 0:
            return 0
        return self._win_count / self._trade_count

    def buy(self, stock_code: str, stock_name: str, quantity: int,
            price: int, commission: int = 0, slippage: int = 0,
            sector: str = "") -> bool:
        """매수."""
        total_cost = price * quantity + commission + slippage
        if total_cost > self.cash:
            logger.warning("자금 부족: 필요 %d, 보유 %d", total_cost, self.cash)
            return False

        self.cash -= total_cost

        if stock_code in self.positions:
            pos = self.positions[stock_code]
            total_qty = pos.quantity + quantity
            pos.avg_price = (pos.avg_price * pos.quantity + price * quantity) / total_qty
            pos.quantity = total_qty
            pos.current_price = price
            pos.highest_price = max(pos.highest_price, price)
        else:
            self.positions[stock_code] = Position(
                stock_code=stock_code,
                stock_name=stock_name,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                highest_price=price,
                sector=sector,
                entry_date=datetime.now().strftime("%Y%m%d"),
            )

        logger.info("매수: %s %d주 @ %d원 (수수료 %d)", stock_name, quantity, price, commission)
        return True

    def sell(self, stock_code: str, quantity: int, price: int,
             commission: int = 0, tax: int = 0, slippage: int = 0) -> float | None:
        """
        매도. Returns: 실현 수익률 (%) or None.
        """
        if stock_code not in self.positions:
            logger.warning("보유하지 않은 종목: %s", stock_code)
            return None

        pos = self.positions[stock_code]
        if quantity > pos.quantity:
            quantity = pos.quantity

        total_revenue = price * quantity - commission - tax - slippage
        self.cash += total_revenue

        # 수익률 계산
        pnl_pct = (price - pos.avg_price) / pos.avg_price * 100 if pos.avg_price > 0 else 0

        self._trade_count += 1
        if pnl_pct > 0:
            self._win_count += 1

        pos.quantity -= quantity
        if pos.quantity <= 0:
            del self.positions[stock_code]

        logger.info("매도: %s %d주 @ %d원 (수익률 %.2f%%)", pos.stock_name, quantity, price, pnl_pct)
        return pnl_pct

    def update_prices(self, price_map: dict[str, int]):
        """
        현재가 업데이트.
        price_map: {stock_code: current_price}
        """
        for code, price in price_map.items():
            if code in self.positions:
                pos = self.positions[code]
                pos.current_price = price
                pos.highest_price = max(pos.highest_price, price)

    def record_daily(self):
        """일별 수익률 기록."""
        total = self.total_value
        if self._prev_total > 0:
            daily_ret = (total - self._prev_total) / self._prev_total
            self.daily_returns.append(daily_ret)
        self._peak_value = max(self._peak_value, total)
        self._prev_total = total

    def get_sector_exposure(self) -> dict[str, float]:
        """섹터별 비중."""
        total = self.total_value
        if total <= 0:
            return {}
        exposure = {}
        for pos in self.positions.values():
            sector = pos.sector or "기타"
            exposure[sector] = exposure.get(sector, 0) + pos.market_value / total
        return exposure

    def snapshot(self) -> dict:
        """현재 상태 스냅샷 (DB 저장용)."""
        from models.evaluator import Evaluator

        metrics = Evaluator.trading_metrics(self.daily_returns)
        return {
            "date": datetime.now().strftime("%Y%m%d"),
            "trade_type": self.trade_type,
            "total_value": self.total_value,
            "cash": self.cash,
            "stock_value": self.stock_value,
            "daily_return": self.daily_returns[-1] if self.daily_returns else 0,
            "cumulative_return": self.cumulative_return,
            "positions": [p.to_dict() for p in self.positions.values()],
            "num_positions": self.num_positions,
            "max_drawdown": metrics.get("max_drawdown", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "win_rate": self.win_rate,
        }

    def load_from_snapshot(self, snapshot: dict):
        """스냅샷에서 포트폴리오 복원."""
        self.cash = snapshot.get("cash", self.initial_capital)
        self.positions = {}
        for p in snapshot.get("positions", []):
            self.positions[p["stock_code"]] = Position(
                stock_code=p["stock_code"],
                stock_name=p.get("stock_name", ""),
                quantity=p["quantity"],
                avg_price=p["avg_price"],
                current_price=p["current_price"],
                highest_price=p.get("highest_price", p["current_price"]),
                sector=p.get("sector", ""),
                entry_date=p.get("entry_date", ""),
            )

    def summary_text(self) -> str:
        """텔레그램 리포트용 텍스트."""
        lines = []
        lines.append(f"총 평가금: {self.total_value:,}원")
        lines.append(f"현금: {self.cash:,}원 | 주식: {self.stock_value:,}원")
        lines.append(f"누적 수익률: {self.cumulative_return:+.2f}%")
        lines.append(f"보유 종목: {self.num_positions}개")

        if self.positions:
            lines.append("")
            for pos in sorted(self.positions.values(), key=lambda p: -abs(p.pnl)):
                emoji = "+" if pos.pnl >= 0 else ""
                lines.append(
                    f"  {pos.stock_name}: {pos.quantity}주 "
                    f"@ {pos.avg_price:,.0f}원 → {pos.current_price:,.0f}원 "
                    f"({emoji}{pos.pnl:.1f}%)"
                )

        return "\n".join(lines)
