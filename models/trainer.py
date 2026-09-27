"""
Walk-forward 학습 + 후회 기반 샘플 가중치
- 시계열 데이터의 정보 누수 방지를 위한 expanding window
- 후회 기록 + 시간 감쇠 + uniqueness weighting
"""
import time
import math
import logging
import numpy as np

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(self, db, feature_engine, lgbm_model, lstm_model):
        self.db = db
        self.engine = feature_engine
        self.lgbm = lgbm_model
        self.lstm = lstm_model

    def prepare_data(self, stock_codes: list[str],
                     start_date: str = None, end_date: str = None,
                     horizon: int = 5) -> dict:
        """
        학습 데이터 준비.
        Returns: {X, y, feature_names, labels, sequences}
        """
        from nn_config import SEQUENCE_LENGTH

        X_all, feature_names, labels = self.engine.get_feature_matrix(
            stock_codes, start_date, end_date
        )

        if not X_all:
            logger.warning("피처 데이터 없음")
            return None

        X = np.array(X_all, dtype=np.float32)

        # 타겟 라벨 생성 (triple barrier)
        y_list = []
        for code, date in labels:
            target = self.engine.get_target_labels(code, [date], horizon)
            y_list.append(target[0] if target else None)

        # None 제거
        valid_mask = [i for i, y in enumerate(y_list) if y is not None]
        X = X[valid_mask]
        y = np.array([y_list[i] for i in valid_mask], dtype=np.int64)
        labels = [labels[i] for i in valid_mask]

        # NaN/Inf 처리
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # LSTM 시퀀스 구성
        sequences = self._make_sequences(X, SEQUENCE_LENGTH)

        logger.info("데이터 준비: %d samples, %d features, seq_shape=%s",
                     len(X), len(feature_names), sequences.shape if sequences is not None else "None")

        # 클래스 분포 로깅
        if len(y) > 0:
            from nn_config import CLASS_NAMES
            for cls in range(len(CLASS_NAMES)):
                cnt = int(np.sum(y == cls))
                logger.info("  class %d (%s): %d (%.1f%%)", cls, CLASS_NAMES[cls], cnt, cnt / len(y) * 100)

        return {
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "labels": labels,
            "sequences": sequences,
        }

    def _make_sequences(self, X: np.ndarray, seq_len: int) -> np.ndarray | None:
        """시퀀스 데이터 생성 (LSTM용)."""
        n = len(X)
        if n < seq_len:
            return None

        sequences = []
        for i in range(seq_len - 1, n):
            seq = X[i - seq_len + 1:i + 1]
            sequences.append(seq)

        # 앞쪽 패딩 (seq_len-1개는 zero-pad)
        pad_seqs = []
        for i in range(seq_len - 1):
            pad = np.zeros((seq_len, X.shape[1]), dtype=np.float32)
            pad[-(i + 1):] = X[:i + 1]
            pad_seqs.append(pad)

        return np.array(pad_seqs + sequences, dtype=np.float32)

    def walk_forward_train(self, data: dict,
                            train_window: int = 400,
                            valid_window: int = 60) -> list[dict]:
        """
        Walk-forward 교차 검증 학습.
        Returns: [{fold, metrics, ...}, ...]
        """
        X, y, sequences = data["X"], data["y"], data["sequences"]
        feature_names = data["feature_names"]

        n = len(X)
        results = []
        fold = 0

        # 후회 + 시간 감쇠 + uniqueness 가중치
        sample_weights = self._compute_sample_weights(data["labels"], y)

        start_idx = train_window
        while start_idx + valid_window <= n:
            fold += 1
            train_end = start_idx
            val_end = start_idx + valid_window

            X_train, y_train = X[:train_end], y[:train_end]
            X_val, y_val = X[train_end:val_end], y[train_end:val_end]
            w_train = sample_weights[:train_end] if sample_weights is not None else None

            t0 = time.time()

            # LightGBM 학습
            lgbm_result = self.lgbm.train(
                X_train, y_train,
                feature_names=feature_names,
                sample_weight=w_train,
                X_val=X_val, y_val=y_val,
            )

            # LSTM 학습 (시퀀스 있을 때만)
            lstm_result = {}
            if sequences is not None:
                seq_train = sequences[:train_end]
                seq_val = sequences[train_end:val_end]
                lstm_result = self.lstm.train(
                    seq_train, y_train,
                    X_val_seq=seq_val, y_val=y_val,
                    sample_weight=w_train,
                )

            elapsed = time.time() - t0

            fold_result = {
                "fold": fold,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "lgbm": lgbm_result,
                "lstm": lstm_result,
                "training_time": elapsed,
            }
            results.append(fold_result)

            logger.info(
                "Fold %d: train=%d, val=%d, lgbm_acc=%.4f, time=%.1fs",
                fold, len(X_train), len(X_val),
                lgbm_result.get("val_accuracy", 0), elapsed,
            )

            start_idx += valid_window

        return results

    def train_final(self, data: dict) -> dict:
        """
        최종 학습 (전체 데이터, 검증 없음).
        모의투자/실전 배포용.
        """
        X, y, sequences = data["X"], data["y"], data["sequences"]
        feature_names = data["feature_names"]

        sample_weights = self._compute_sample_weights(data["labels"], y)

        # 학습/검증 분할 (마지막 60일 검증)
        split = max(len(X) - 60, int(len(X) * 0.85))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        w_train = sample_weights[:split] if sample_weights is not None else None

        t0 = time.time()

        lgbm_result = self.lgbm.train(
            X_train, y_train,
            feature_names=feature_names,
            sample_weight=w_train,
            X_val=X_val, y_val=y_val,
        )

        lstm_result = {}
        if sequences is not None:
            seq_train = sequences[:split]
            seq_val = sequences[split:]
            lstm_result = self.lstm.train(
                seq_train, y_train,
                X_val_seq=seq_val, y_val=y_val,
                sample_weight=w_train,
            )

        elapsed = time.time() - t0

        return {
            "lgbm": lgbm_result,
            "lstm": lstm_result,
            "total_samples": len(X),
            "training_time": elapsed,
        }

    def _compute_sample_weights(self, labels: list[tuple],
                                 y: np.ndarray) -> np.ndarray | None:
        """
        복합 샘플 가중치 계산:
        1. 후회 기반 부스트 (5x)
        2. 시간 감쇠 (최신 데이터에 높은 가중치)
        3. Uniqueness weighting (라벨 오버랩 보정)
        """
        from nn_config import REGRET_WEIGHT_BOOST, TIME_DECAY_HALFLIFE

        n = len(labels)
        weights = np.ones(n, dtype=np.float32)

        # 1. 후회 기반 부스트
        regrets = self.db.get_unprocessed_regrets()
        regret_map = {}
        if regrets:
            for r in regrets:
                key = (r["stock_code"], r["trade_date"])
                regret_map[key] = r["regret_score"]

        boosted = 0
        for i, (code, date) in enumerate(labels):
            score = regret_map.get((code, date), 0)
            if score > 0:
                weights[i] *= 1.0 + score * REGRET_WEIGHT_BOOST
                boosted += 1

        # 2. 시간 감쇠 (최신 데이터 = 높은 가중치)
        # 나이는 행 순서가 아니라 거래일 기준: 행이 종목별로 묶여 있어서
        # 행 번호를 쓰면 마지막 종목들이 "최신"으로 취급된다.
        if n > 1:
            halflife = TIME_DECAY_HALFLIFE
            dates = sorted({str(date) for _, date in labels})
            newest = len(dates) - 1
            day_index = {d: k for k, d in enumerate(dates)}
            for i, (_, date) in enumerate(labels):
                age = newest - day_index[str(date)]  # 0=최신 거래일
                weights[i] *= math.pow(0.5, age / halflife)

        # 3. Uniqueness weighting (소수 클래스에 더 높은 가중치)
        if len(y) == n:
            from nn_config import NUM_CLASSES
            class_counts = np.bincount(y, minlength=NUM_CLASSES)
            total = class_counts.sum()
            for cls in range(NUM_CLASSES):
                if class_counts[cls] > 0:
                    cls_weight = total / (NUM_CLASSES * class_counts[cls])
                    mask = y == cls
                    weights[mask] *= cls_weight

        # 정규화 (평균 1.0)
        if weights.sum() > 0:
            weights *= n / weights.sum()

        if boosted > 0:
            logger.info("샘플 가중치: %d/%d 후회부스트, 시간감쇠 halflife=%d, uniqueness 적용",
                        boosted, n, TIME_DECAY_HALFLIFE)
        else:
            logger.info("샘플 가중치: 시간감쇠 + uniqueness 적용 (%d samples)", n)

        return weights
