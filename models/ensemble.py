"""
앙상블 모델 — LightGBM + LSTM
- 3-class 분류 (하락/보합/상승)
- 최근 30일 정확도 기반 동적 가중치 조정
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EnsembleModel:

    def __init__(self, lgbm_model, lstm_model, weights: dict = None):
        from nn_config import ENSEMBLE_WEIGHTS, NUM_CLASSES
        self.lgbm = lgbm_model
        self.lstm = lstm_model
        self.weights = weights or ENSEMBLE_WEIGHTS.copy()
        self.num_classes = NUM_CLASSES

    def predict_proba(self, X_flat: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
        """
        앙상블 확률 예측.
        X_flat: (n_samples, n_features) — LightGBM용
        X_seq: (n_samples, seq_len, n_features) — LSTM용
        Returns: (n_samples, num_classes)
        """
        nc = self.num_classes
        w_lgbm = self.weights.get("lgbm", 0.6)
        w_lstm = self.weights.get("lstm", 0.4)

        probs = np.zeros((X_flat.shape[0], nc))

        # LightGBM
        try:
            lgbm_probs = self.lgbm.predict_proba(X_flat)
            probs += w_lgbm * lgbm_probs
        except Exception as e:
            logger.error("LightGBM predict failed: %s", e)
            probs += w_lgbm * (1.0 / nc)

        # LSTM
        try:
            lstm_probs = self.lstm.predict_proba(X_seq)
            probs += w_lstm * lstm_probs
        except Exception as e:
            logger.warning("LSTM predict failed (using LGBM only): %s", e)
            try:
                probs = self.lgbm.predict_proba(X_flat)
            except Exception:
                probs = np.full((X_flat.shape[0], nc), 1.0 / nc)

        # 정규화 (합계=1)
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        probs = probs / row_sums

        return probs

    def predict(self, X_flat: np.ndarray, X_seq: np.ndarray) -> np.ndarray:
        """앙상블 클래스 예측."""
        probs = self.predict_proba(X_flat, X_seq)
        return probs.argmax(axis=1)

    def predict_single(self, X_flat: np.ndarray, X_seq: np.ndarray) -> dict:
        """
        단일 종목 예측 (상세 결과).
        Returns: {predicted_class, probabilities, confidence, signal}
        """
        from nn_config import CLASS_NAMES

        probs = self.predict_proba(X_flat, X_seq)
        pred_class = int(probs[0].argmax())
        confidence = float(probs[0].max())

        return {
            "predicted_class": pred_class,
            "class_probabilities": probs[0].tolist(),
            "confidence": confidence,
            "signal": CLASS_NAMES[pred_class],
            "buy_prob": float(probs[0][2]),     # 상승 확률
            "sell_prob": float(probs[0][0]),     # 하락 확률
        }

    def update_weights(self, db):
        """
        최근 30일 예측 정확도 기반 동적 가중치 조정.
        각 모델별 예측 기록 필요 — prediction_log 테이블에서.
        """
        from nn_config import ENSEMBLE_DYNAMIC
        if not ENSEMBLE_DYNAMIC:
            return

        try:
            rows = db.execute_raw(
                "SELECT model_name, correct FROM prediction_log "
                "WHERE created_at >= date('now', '-30 days') "
                "ORDER BY created_at DESC LIMIT 500"
            )
        except Exception:
            return

        if not rows or len(rows) < 20:
            return

        lgbm_correct = [r["correct"] for r in rows if r.get("model_name") == "lgbm"]
        lstm_correct = [r["correct"] for r in rows if r.get("model_name") == "lstm"]

        if not lgbm_correct or not lstm_correct:
            return

        lgbm_acc = sum(lgbm_correct) / len(lgbm_correct)
        lstm_acc = sum(lstm_correct) / len(lstm_correct)

        total_acc = lgbm_acc + lstm_acc
        if total_acc <= 0:
            return

        # 정확도 비례 가중치 (최소 0.2, 최대 0.8)
        w_lgbm = max(0.2, min(0.8, lgbm_acc / total_acc))
        w_lstm = 1.0 - w_lgbm

        old = self.weights.copy()
        self.weights["lgbm"] = round(w_lgbm, 3)
        self.weights["lstm"] = round(w_lstm, 3)

        logger.info(
            "앙상블 가중치 조정: lgbm %.3f→%.3f, lstm %.3f→%.3f (acc: lgbm=%.3f, lstm=%.3f)",
            old["lgbm"], w_lgbm, old["lstm"], w_lstm, lgbm_acc, lstm_acc,
        )
