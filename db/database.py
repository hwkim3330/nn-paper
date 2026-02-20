"""
SQLite Database 클래스 — 모든 CRUD 연산
- Thread-safe (check_same_thread=False)
- WAL mode for concurrent reads
- Bulk insert 지원
"""
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path

from .schema import create_tables

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        create_tables(self.db_path)
        self._setup_pragmas()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _setup_pragmas(self):
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB
            conn.commit()
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 일봉 데이터
    # ----------------------------------------------------------
    def upsert_daily_prices(self, rows: list[dict]):
        """일봉 데이터 bulk upsert. rows: [{stock_code, date, open, high, low, close, volume, ...}]"""
        if not rows:
            return
        conn = self._connect()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO daily_prices
                   (stock_code, date, open, high, low, close, volume, change_rate)
                   VALUES (:stock_code, :date, :open, :high, :low, :close, :volume, :change_rate)""",
                rows,
            )
            conn.commit()
            logger.debug("daily_prices upserted %d rows", len(rows))
        finally:
            conn.close()

    def get_daily_prices(self, stock_code: str, start_date: str = None,
                         end_date: str = None, limit: int = None) -> list[dict]:
        """일봉 조회. 날짜 오름차순."""
        conn = self._connect()
        try:
            sql = "SELECT * FROM daily_prices WHERE stock_code = ?"
            params = [stock_code]
            if start_date:
                sql += " AND date >= ?"
                params.append(start_date)
            if end_date:
                sql += " AND date <= ?"
                params.append(end_date)
            sql += " ORDER BY date ASC"
            if limit:
                sql += f" LIMIT {limit}"
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_latest_daily_date(self, stock_code: str) -> str | None:
        """해당 종목의 가장 최근 일봉 날짜."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT MAX(date) as max_date FROM daily_prices WHERE stock_code = ?",
                (stock_code,),
            ).fetchone()
            return row["max_date"] if row else None
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 분봉 데이터
    # ----------------------------------------------------------
    def upsert_minute_prices(self, rows: list[dict]):
        if not rows:
            return
        conn = self._connect()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO minute_prices
                   (stock_code, datetime, interval_min, open, high, low, close, volume)
                   VALUES (:stock_code, :datetime, :interval_min, :open, :high, :low, :close, :volume)""",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 호가 스냅샷
    # ----------------------------------------------------------
    def insert_orderbook(self, data: dict):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO orderbook_snapshots
                   (stock_code, datetime, total_ask_volume, total_bid_volume,
                    ask_bid_ratio, spread_pct, ask_prices, ask_volumes, bid_prices, bid_volumes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["stock_code"],
                    data["datetime"],
                    data.get("total_ask_volume", 0),
                    data.get("total_bid_volume", 0),
                    data.get("ask_bid_ratio", 0),
                    data.get("spread_pct", 0),
                    json.dumps(data.get("ask_prices", [])),
                    json.dumps(data.get("ask_volumes", [])),
                    json.dumps(data.get("bid_prices", [])),
                    json.dumps(data.get("bid_volumes", [])),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_orderbook_snapshots(self, stock_code: str, date: str) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """SELECT * FROM orderbook_snapshots
                   WHERE stock_code = ? AND datetime LIKE ?
                   ORDER BY datetime ASC""",
                (stock_code, f"{date}%"),
            ).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                for k in ("ask_prices", "ask_volumes", "bid_prices", "bid_volumes"):
                    if d.get(k):
                        d[k] = json.loads(d[k])
                result.append(d)
            return result
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 투자자 매매동향
    # ----------------------------------------------------------
    def upsert_investor_flow(self, rows: list[dict]):
        if not rows:
            return
        conn = self._connect()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO investor_flow
                   (stock_code, date, foreign_buy, foreign_sell, foreign_net,
                    institution_buy, institution_sell, institution_net,
                    individual_net, foreign_hold_ratio)
                   VALUES (:stock_code, :date, :foreign_buy, :foreign_sell, :foreign_net,
                           :institution_buy, :institution_sell, :institution_net,
                           :individual_net, :foreign_hold_ratio)""",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def get_investor_flow(self, stock_code: str, start_date: str = None,
                          end_date: str = None) -> list[dict]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM investor_flow WHERE stock_code = ?"
            params = [stock_code]
            if start_date:
                sql += " AND date >= ?"
                params.append(start_date)
            if end_date:
                sql += " AND date <= ?"
                params.append(end_date)
            sql += " ORDER BY date ASC"
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 프로그램 매매
    # ----------------------------------------------------------
    def upsert_program_trading(self, rows: list[dict]):
        if not rows:
            return
        conn = self._connect()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO program_trading
                   (stock_code, date, program_buy, program_sell, program_net,
                    arbitrage_buy, arbitrage_sell, non_arbitrage_buy, non_arbitrage_sell)
                   VALUES (:stock_code, :date, :program_buy, :program_sell, :program_net,
                           :arbitrage_buy, :arbitrage_sell, :non_arbitrage_buy, :non_arbitrage_sell)""",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def get_program_trading(self, stock_code: str, start_date: str = None,
                            end_date: str = None) -> list[dict]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM program_trading WHERE stock_code = ?"
            params = [stock_code]
            if start_date:
                sql += " AND date >= ?"
                params.append(start_date)
            if end_date:
                sql += " AND date <= ?"
                params.append(end_date)
            sql += " ORDER BY date ASC"
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 업종 데이터
    # ----------------------------------------------------------
    def upsert_sector_data(self, rows: list[dict]):
        if not rows:
            return
        conn = self._connect()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO sector_data
                   (sector_code, date, sector_name, close, change_rate, volume,
                    market_cap, advance_count, decline_count)
                   VALUES (:sector_code, :date, :sector_name, :close, :change_rate,
                           :volume, :market_cap, :advance_count, :decline_count)""",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def get_sector_data(self, sector_code: str = None, date: str = None) -> list[dict]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM sector_data WHERE 1=1"
            params = []
            if sector_code:
                sql += " AND sector_code = ?"
                params.append(sector_code)
            if date:
                sql += " AND date = ?"
                params.append(date)
            sql += " ORDER BY date ASC"
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 시장 데이터 (US, 환율 등)
    # ----------------------------------------------------------
    def upsert_market_data(self, rows: list[dict]):
        if not rows:
            return
        conn = self._connect()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO market_data
                   (date, symbol, name, price, change, change_pct)
                   VALUES (:date, :symbol, :name, :price, :change, :change_pct)""",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def get_market_data(self, symbol: str = None, start_date: str = None,
                        end_date: str = None) -> list[dict]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM market_data WHERE 1=1"
            params = []
            if symbol:
                sql += " AND symbol = ?"
                params.append(symbol)
            if start_date:
                sql += " AND date >= ?"
                params.append(start_date)
            if end_date:
                sql += " AND date <= ?"
                params.append(end_date)
            sql += " ORDER BY date ASC"
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 피처 벡터
    # ----------------------------------------------------------
    def upsert_features(self, stock_code: str, date: str, features: dict, version: int = 1):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO features
                   (stock_code, date, feature_vector, feature_version)
                   VALUES (?, ?, ?, ?)""",
                (stock_code, date, json.dumps(features), version),
            )
            conn.commit()
        finally:
            conn.close()

    def get_features(self, stock_code: str, start_date: str = None,
                     end_date: str = None) -> list[dict]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM features WHERE stock_code = ?"
            params = [stock_code]
            if start_date:
                sql += " AND date >= ?"
                params.append(start_date)
            if end_date:
                sql += " AND date <= ?"
                params.append(end_date)
            sql += " ORDER BY date ASC"
            rows = conn.execute(sql, params).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                d["feature_vector"] = json.loads(d["feature_vector"])
                result.append(d)
            return result
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 예측
    # ----------------------------------------------------------
    def insert_prediction(self, data: dict):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO predictions
                   (stock_code, date, model_name, predicted_class,
                    class_probabilities, confidence)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    data["stock_code"],
                    data["date"],
                    data["model_name"],
                    data["predicted_class"],
                    json.dumps(data.get("class_probabilities", [])),
                    data.get("confidence", 0),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def update_prediction_actual(self, pred_id: int, actual_class: int, actual_return: float):
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE predictions SET actual_class = ?, actual_return = ? WHERE id = ?",
                (actual_class, actual_return, pred_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_predictions(self, stock_code: str = None, date: str = None,
                        model_name: str = None) -> list[dict]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM predictions WHERE 1=1"
            params = []
            if stock_code:
                sql += " AND stock_code = ?"
                params.append(stock_code)
            if date:
                sql += " AND date = ?"
                params.append(date)
            if model_name:
                sql += " AND model_name = ?"
                params.append(model_name)
            sql += " ORDER BY date ASC, id ASC"
            rows = conn.execute(sql, params).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                if d.get("class_probabilities"):
                    d["class_probabilities"] = json.loads(d["class_probabilities"])
                result.append(d)
            return result
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 매매 내역
    # ----------------------------------------------------------
    def insert_trade(self, data: dict) -> int:
        conn = self._connect()
        try:
            cursor = conn.execute(
                """INSERT INTO trades
                   (trade_type, stock_code, stock_name, side, quantity, price, amount,
                    commission, tax, slippage, predicted_class, confidence, reason, executed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["trade_type"],
                    data["stock_code"],
                    data.get("stock_name", ""),
                    data["side"],
                    data["quantity"],
                    data["price"],
                    data["amount"],
                    data.get("commission", 0),
                    data.get("tax", 0),
                    data.get("slippage", 0),
                    data.get("predicted_class"),
                    data.get("confidence"),
                    data.get("reason", ""),
                    data["executed_at"],
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def update_trade_regret(self, trade_id: int, regret_score: float,
                            regret_analysis: str, actual_return_5d: float,
                            optimal_action: str):
        conn = self._connect()
        try:
            conn.execute(
                """UPDATE trades SET regret_score = ?, regret_analysis = ?,
                   actual_return_5d = ?, optimal_action = ? WHERE id = ?""",
                (regret_score, regret_analysis, actual_return_5d, optimal_action, trade_id),
            )
            conn.commit()
        finally:
            conn.close()

    def get_trades(self, trade_type: str = None, stock_code: str = None,
                   start_date: str = None, end_date: str = None,
                   pending_regret: bool = False) -> list[dict]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM trades WHERE 1=1"
            params = []
            if trade_type:
                sql += " AND trade_type = ?"
                params.append(trade_type)
            if stock_code:
                sql += " AND stock_code = ?"
                params.append(stock_code)
            if start_date:
                sql += " AND executed_at >= ?"
                params.append(start_date)
            if end_date:
                sql += " AND executed_at <= ?"
                params.append(end_date)
            if pending_regret:
                sql += " AND regret_score IS NULL AND executed_at <= date('now', '-5 days')"
            sql += " ORDER BY executed_at ASC"
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 포트폴리오 스냅샷
    # ----------------------------------------------------------
    def insert_portfolio_snapshot(self, data: dict):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO portfolio_snapshots
                   (date, trade_type, total_value, cash, stock_value,
                    daily_return, cumulative_return, positions, num_positions,
                    max_drawdown, sharpe_ratio, win_rate)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["date"],
                    data["trade_type"],
                    data["total_value"],
                    data["cash"],
                    data["stock_value"],
                    data.get("daily_return", 0),
                    data.get("cumulative_return", 0),
                    json.dumps(data.get("positions", []), ensure_ascii=False),
                    data.get("num_positions", 0),
                    data.get("max_drawdown", 0),
                    data.get("sharpe_ratio", 0),
                    data.get("win_rate", 0),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_portfolio_snapshots(self, trade_type: str = "paper",
                                start_date: str = None, limit: int = None) -> list[dict]:
        conn = self._connect()
        try:
            sql = "SELECT * FROM portfolio_snapshots WHERE trade_type = ?"
            params = [trade_type]
            if start_date:
                sql += " AND date >= ?"
                params.append(start_date)
            sql += " ORDER BY date DESC"
            if limit:
                sql += f" LIMIT {limit}"
            rows = conn.execute(sql, params).fetchall()
            result = []
            for r in rows:
                d = dict(r)
                if d.get("positions"):
                    d["positions"] = json.loads(d["positions"])
                result.append(d)
            return result
        finally:
            conn.close()

    def get_latest_portfolio(self, trade_type: str = "paper") -> dict | None:
        rows = self.get_portfolio_snapshots(trade_type, limit=1)
        return rows[0] if rows else None

    # ----------------------------------------------------------
    # 학습 로그
    # ----------------------------------------------------------
    def insert_training_log(self, data: dict):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO training_log
                   (model_name, train_start_date, train_end_date,
                    valid_start_date, valid_end_date, metrics, model_path,
                    feature_importance, training_time_sec)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["model_name"],
                    data.get("train_start_date"),
                    data.get("train_end_date"),
                    data.get("valid_start_date"),
                    data.get("valid_end_date"),
                    json.dumps(data.get("metrics", {})),
                    data.get("model_path", ""),
                    json.dumps(data.get("feature_importance", {})),
                    data.get("training_time_sec", 0),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 후회 로그
    # ----------------------------------------------------------
    def insert_regret(self, data: dict):
        conn = self._connect()
        try:
            conn.execute(
                """INSERT INTO regret_log
                   (trade_id, stock_code, trade_date, action_taken,
                    optimal_action, regret_score, return_if_optimal,
                    return_actual, lesson)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data.get("trade_id"),
                    data["stock_code"],
                    data["trade_date"],
                    data["action_taken"],
                    data["optimal_action"],
                    data["regret_score"],
                    data.get("return_if_optimal", 0),
                    data.get("return_actual", 0),
                    data.get("lesson", ""),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_unprocessed_regrets(self) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM regret_log WHERE applied_to_training = 0 ORDER BY trade_date ASC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def mark_regret_applied(self, regret_ids: list[int]):
        if not regret_ids:
            return
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(regret_ids))
            conn.execute(
                f"UPDATE regret_log SET applied_to_training = 1 WHERE id IN ({placeholders})",
                regret_ids,
            )
            conn.commit()
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 순위 데이터
    # ----------------------------------------------------------
    def upsert_ranking_data(self, rows: list[dict]):
        if not rows:
            return
        conn = self._connect()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO ranking_data
                   (date, ranking_type, rank_num, stock_code, stock_name,
                    price, change_rate, volume, amount)
                   VALUES (:date, :ranking_type, :rank_num, :stock_code, :stock_name,
                           :price, :change_rate, :volume, :amount)""",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    # ----------------------------------------------------------
    # 유틸리티
    # ----------------------------------------------------------
    def get_all_stock_codes_in_db(self) -> list[str]:
        conn = self._connect()
        try:
            rows = conn.execute("SELECT DISTINCT stock_code FROM daily_prices").fetchall()
            return [r["stock_code"] for r in rows]
        finally:
            conn.close()

    def get_date_range(self, stock_code: str) -> tuple[str | None, str | None]:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT MIN(date) as min_d, MAX(date) as max_d FROM daily_prices WHERE stock_code = ?",
                (stock_code,),
            ).fetchone()
            return (row["min_d"], row["max_d"]) if row else (None, None)
        finally:
            conn.close()

    def execute_raw(self, sql: str, params: tuple = ()) -> list[dict]:
        conn = self._connect()
        try:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_db_stats(self) -> dict:
        conn = self._connect()
        try:
            stats = {}
            tables = [
                "daily_prices", "minute_prices", "orderbook_snapshots",
                "investor_flow", "program_trading", "sector_data",
                "market_data", "features", "predictions", "trades",
                "portfolio_snapshots", "training_log", "regret_log", "ranking_data",
            ]
            for t in tables:
                try:
                    row = conn.execute(f"SELECT COUNT(*) as cnt FROM {t}").fetchone()
                    stats[t] = row["cnt"]
                except Exception:
                    stats[t] = 0
            return stats
        finally:
            conn.close()
