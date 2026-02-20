"""
모델 평가 지표
- Accuracy, Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor
"""
import math
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Evaluator:

    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                class_names: list[str] = None) -> dict:
        """분류 성능 지표."""
        accuracy = np.mean(y_true == y_pred)

        # 클래스별 정확도
        class_acc = {}
        for cls in sorted(set(y_true)):
            mask = y_true == cls
            if mask.sum() > 0:
                name = class_names[cls] if class_names and cls < len(class_names) else str(cls)
                class_acc[name] = float(np.mean(y_pred[mask] == cls))

        # Adjacent accuracy (1 클래스 차이 허용)
        adj_acc = np.mean(np.abs(y_true.astype(int) - y_pred.astype(int)) <= 1)

        return {
            "accuracy": float(accuracy),
            "adjacent_accuracy": float(adj_acc),
            "class_accuracy": class_acc,
        }

    @staticmethod
    def trading_metrics(returns: list[float], risk_free_rate: float = 0.035) -> dict:
        """
        트레이딩 성과 지표.
        returns: 일별 수익률 리스트 (예: [0.01, -0.005, 0.003, ...])
        """
        if not returns or len(returns) < 2:
            return {
                "total_return": 0, "sharpe_ratio": 0,
                "max_drawdown": 0, "win_rate": 0,
                "profit_factor": 0, "avg_return": 0,
            }

        r = np.array(returns)
        n = len(r)

        # 총 수익률
        cumulative = np.cumprod(1 + r)
        total_return = float(cumulative[-1] - 1)

        # Sharpe Ratio (연환산)
        daily_rf = risk_free_rate / 252
        excess = r - daily_rf
        avg_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)
        sharpe = float(avg_excess / std_excess * math.sqrt(252)) if std_excess > 0 else 0

        # Max Drawdown
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        max_dd = float(drawdown.min() * 100)

        # Win Rate
        wins = np.sum(r > 0)
        losses = np.sum(r < 0)
        total_trades = wins + losses
        win_rate = float(wins / total_trades) if total_trades > 0 else 0

        # Profit Factor
        total_profit = np.sum(r[r > 0])
        total_loss = abs(np.sum(r[r < 0]))
        profit_factor = float(total_profit / total_loss) if total_loss > 0 else float('inf')

        # 평균 수익률
        avg_return = float(np.mean(r) * 100)

        # Calmar Ratio
        calmar = float(total_return / abs(max_dd) * 100) if max_dd != 0 else 0

        return {
            "total_return": round(total_return * 100, 2),  # %
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(max_dd, 2),  # %
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "avg_daily_return": round(avg_return, 4),
            "calmar_ratio": round(calmar, 4),
            "num_days": n,
        }

    @staticmethod
    def check_live_criteria(metrics: dict) -> dict:
        """
        실전 전환 기준 체크.
        Returns: {criteria_name: {value, threshold, passed}, overall_passed}
        """
        from nn_config import LIVE_CRITERIA

        checks = {}
        checks["min_days"] = {
            "value": metrics.get("num_days", 0),
            "threshold": LIVE_CRITERIA["min_days"],
            "passed": metrics.get("num_days", 0) >= LIVE_CRITERIA["min_days"],
        }
        checks["cumulative_return"] = {
            "value": metrics.get("total_return", 0),
            "threshold": LIVE_CRITERIA["min_cumulative_return"],
            "passed": metrics.get("total_return", 0) > LIVE_CRITERIA["min_cumulative_return"],
        }
        checks["sharpe_ratio"] = {
            "value": metrics.get("sharpe_ratio", 0),
            "threshold": LIVE_CRITERIA["min_sharpe"],
            "passed": metrics.get("sharpe_ratio", 0) >= LIVE_CRITERIA["min_sharpe"],
        }
        checks["max_drawdown"] = {
            "value": metrics.get("max_drawdown", 0),
            "threshold": LIVE_CRITERIA["max_drawdown"],
            "passed": metrics.get("max_drawdown", 0) >= LIVE_CRITERIA["max_drawdown"],
        }
        checks["win_rate"] = {
            "value": metrics.get("win_rate", 0),
            "threshold": LIVE_CRITERIA["min_win_rate"],
            "passed": metrics.get("win_rate", 0) >= LIVE_CRITERIA["min_win_rate"],
        }

        overall = all(c["passed"] for c in checks.values())

        return {"checks": checks, "overall_passed": overall}
