"""
FeatureEngine — 전체 피처 벡터 생성 파이프라인
- 기술적 지표 (40) + 호가 (15) + 수급 (20) + 시장 (25) + 기타 = ~143개
- DB에서 데이터 로드 → 피처 계산 → 정규화 → 저장
"""
import logging
from datetime import datetime, timedelta

from .technical import compute_technical_features
from .orderbook_features import compute_orderbook_features
from .flow_features import compute_flow_features
from .market_features import compute_market_features
from .normalizer import FeatureNormalizer

logger = logging.getLogger(__name__)


class FeatureEngine:

    def __init__(self, db):
        self.db = db
        self.normalizer = FeatureNormalizer()

    def compute_features_for_stock(self, stock_code: str, date: str,
                                    sector_code: str = "") -> dict:
        """
        특정 종목/날짜에 대한 전체 피처 벡터 계산.
        Returns: {feature_name: raw_value, ...}
        """
        features = {}

        # 1. 기술적 지표 (일봉 기반)
        prices = self.db.get_daily_prices(stock_code, end_date=date, limit=150)
        if prices:
            tech = compute_technical_features(prices)
            features.update(tech)

        # 2. 호가 피처
        ob_snaps = self.db.get_orderbook_snapshots(stock_code, date)
        ob = compute_orderbook_features(ob_snaps)
        features.update(ob)

        # 3. 수급 피처
        inv_flows = self.db.get_investor_flow(stock_code, end_date=date)
        prog_trades = self.db.get_program_trading(stock_code, end_date=date)
        flow = compute_flow_features(inv_flows[-30:], prog_trades[-30:])
        features.update(flow)

        # 4. 시장 컨텍스트
        market_rows = self.db.get_market_data(start_date=date, end_date=date)
        sector_rows = self.db.get_sector_data(date=date)
        market = compute_market_features(market_rows, sector_rows, sector_code)
        features.update(market)

        return features

    def compute_features_batch(self, stock_code: str,
                                dates: list[str] = None,
                                sector_code: str = "") -> list[dict]:
        """
        여러 날짜에 대한 피처 벡터 일괄 계산 (학습용).
        Returns: [{date: ..., features: {...}}, ...]
        """
        all_prices = self.db.get_daily_prices(stock_code)
        if not all_prices:
            return []

        if dates is None:
            dates = [p["date"] for p in all_prices]

        # 사전 로드 (효율성)
        inv_flows = self.db.get_investor_flow(stock_code)
        prog_trades = self.db.get_program_trading(stock_code)

        inv_by_date = {r["date"]: r for r in inv_flows}
        prog_by_date = {r["date"]: r for r in prog_trades}

        results = []
        for i, date in enumerate(dates):
            # 해당 날짜까지의 가격 (최소 20일)
            prices_up_to = [p for p in all_prices if p["date"] <= date]
            if len(prices_up_to) < 20:
                continue

            features = {}

            # 기술적 지표
            tech = compute_technical_features(prices_up_to[-150:])
            features.update(tech)

            # 호가 (배치에서는 스킵 가능 — 과거 호가 없을 수 있음)
            ob_snaps = self.db.get_orderbook_snapshots(stock_code, date)
            ob = compute_orderbook_features(ob_snaps)
            features.update(ob)

            # 수급
            inv_up_to = [r for r in inv_flows if r["date"] <= date]
            prog_up_to = [r for r in prog_trades if r["date"] <= date]
            flow = compute_flow_features(inv_up_to[-30:], prog_up_to[-30:])
            features.update(flow)

            # 시장
            market_rows = self.db.get_market_data(start_date=date, end_date=date)
            sector_rows = self.db.get_sector_data(date=date)
            market = compute_market_features(market_rows, sector_rows, sector_code)
            features.update(market)

            results.append({"date": date, "features": features})

            if (i + 1) % 50 == 0:
                logger.info("%s: 피처 계산 %d/%d", stock_code, i + 1, len(dates))

        return results

    def compute_and_save(self, stock_code: str, sector_code: str = "") -> int:
        """피처 계산 후 DB 저장."""
        batch = self.compute_features_batch(stock_code, sector_code=sector_code)
        for item in batch:
            self.db.upsert_features(stock_code, item["date"], item["features"])
        logger.info("%s: %d일 피처 저장", stock_code, len(batch))
        return len(batch)

    def get_feature_matrix(self, stock_codes: list[str],
                           start_date: str = None,
                           end_date: str = None) -> tuple[list[list[float]], list[str], list[str]]:
        """
        학습용 피처 행렬 반환.
        Returns: (X_matrix, feature_names, labels)
          - X_matrix: [[feature_values], ...]
          - feature_names: [feature1, feature2, ...]
          - labels: [(stock_code, date), ...]
        """
        all_features = []
        all_labels = []

        for code in stock_codes:
            rows = self.db.get_features(code, start_date, end_date)
            for r in rows:
                all_features.append(r["feature_vector"])
                all_labels.append((code, r["date"]))

        if not all_features:
            return [], [], []

        # 통일된 피처 이름 (알파벳순)
        all_keys = set()
        for fd in all_features:
            all_keys.update(fd.keys())
        feature_names = sorted(all_keys)

        # 행렬 구성
        X = []
        for fd in all_features:
            row = [fd.get(k, 0) for k in feature_names]
            X.append(row)

        return X, feature_names, all_labels

    def get_feature_sequence(self, stock_code: str, date: str,
                             feature_names: list[str],
                             seq_len: int = 20) -> list[list[float]] | None:
        """최근 N일의 피처 시퀀스 반환 (LSTM 실시간 예측용).

        DB에서 date 이전 seq_len일의 피처를 가져와 시계열 형태로 반환.
        Returns: [[feat_day1], [feat_day2], ..., [feat_dayN]] or None
        """
        rows = self.db.get_features(stock_code, end_date=date)
        if not rows:
            return None

        # 날짜순 정렬, 최근 seq_len개
        rows.sort(key=lambda r: r['date'])
        recent = rows[-seq_len:]

        if len(recent) < 3:  # 최소 3일은 있어야
            return None

        sequence = []
        for r in recent:
            fv = r.get('feature_vector', {})
            vec = [fv.get(k, 0) for k in feature_names]
            sequence.append(vec)

        # seq_len보다 적으면 앞에 zero-pad
        if len(sequence) < seq_len:
            pad_count = seq_len - len(sequence)
            n_feat = len(feature_names)
            padding = [[0.0] * n_feat for _ in range(pad_count)]
            sequence = padding + sequence

        return sequence

    def get_target_labels(self, stock_code: str, dates: list[str],
                           horizon: int = 5) -> list[int | None]:
        """
        Triple Barrier 기반 3-class 타겟 라벨 생성.
        ATR 기반 동적 barrier → 상승(2) / 보합(1) / 하락(0)
        Returns: [class_label or None, ...]
        """
        from nn_config import TRIPLE_BARRIER

        all_prices = self.db.get_daily_prices(stock_code)
        if not all_prices:
            return [None] * len(dates)

        date_list = [p["date"] for p in all_prices]
        close_map = {p["date"]: p["close"] for p in all_prices}
        high_map = {p["date"]: p["high"] for p in all_prices}
        low_map = {p["date"]: p["low"] for p in all_prices}

        atr_period = TRIPLE_BARRIER["atr_period"]
        atr_mult = TRIPLE_BARRIER["atr_multiplier"]
        max_hold = TRIPLE_BARRIER["max_holding_days"]

        labels = []
        for date in dates:
            if date not in close_map:
                labels.append(None)
                continue

            try:
                idx = date_list.index(date)
            except ValueError:
                labels.append(None)
                continue

            # ATR 계산 (entry 시점)
            if idx < atr_period:
                labels.append(None)
                continue

            atr = self._compute_atr(all_prices, idx, atr_period)
            if atr <= 0:
                labels.append(None)
                continue

            entry_price = close_map[date]
            if entry_price <= 0:
                labels.append(None)
                continue

            profit_barrier = entry_price + atr * atr_mult
            stop_barrier = entry_price - atr * atr_mult

            # 미래 max_hold일 동안 barrier 체크
            if idx + 1 >= len(date_list):
                labels.append(None)
                continue

            label = 1  # 보합 (기본: timeout)
            end_idx = min(idx + max_hold, len(date_list) - 1)

            for j in range(idx + 1, end_idx + 1):
                high_j = high_map[date_list[j]]
                low_j = low_map[date_list[j]]

                hit_profit = high_j >= profit_barrier
                hit_stop = low_j <= stop_barrier

                if hit_profit and hit_stop:
                    # 둘 다 닿으면 종가 기준
                    close_j = close_map[date_list[j]]
                    label = 2 if close_j >= entry_price else 0
                    break
                elif hit_profit:
                    label = 2  # 상승
                    break
                elif hit_stop:
                    label = 0  # 하락
                    break

            labels.append(label)

        return labels

    @staticmethod
    def _compute_atr(prices: list[dict], idx: int, period: int) -> float:
        """ATR 계산 (idx 시점 기준)."""
        if idx < period:
            return 0
        trs = []
        for i in range(idx - period + 1, idx + 1):
            h = prices[i]["high"]
            l = prices[i]["low"]
            pc = prices[i - 1]["close"]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else 0
