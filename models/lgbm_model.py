"""
LightGBM 3-class 분류기
- Triple Barrier 기반 3-class (하락/보합/상승)
- CPU 최적화, 수초 내 학습
- 샘플 가중치 지원 (후회 + 시간 감쇠)
"""
import json
import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class LGBMModel:

    def __init__(self, params: dict = None):
        from nn_config import LGBM_PARAMS
        self.params = params or LGBM_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.feature_importance_ = None

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: list[str] = None,
              sample_weight: np.ndarray = None,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> dict:
        """
        LightGBM 학습.
        Returns: {accuracy, loss, ...}
        """
        import lightgbm as lgb

        self.feature_names = feature_names

        # 파라미터 분리
        fit_params = {}
        model_params = {k: v for k, v in self.params.items()
                        if k not in ("n_estimators",)}

        n_estimators = self.params.get("n_estimators", 500)

        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            **{k: v for k, v in model_params.items() if k != "n_estimators"},
        )

        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        callbacks = []
        if eval_set:
            callbacks.append(lgb.early_stopping(50, verbose=False))
            callbacks.append(lgb.log_evaluation(period=0))

        self.model.fit(
            X, y,
            sample_weight=sample_weight,
            eval_set=eval_set if eval_set else None,
            callbacks=callbacks if callbacks else None,
        )

        # 피처 중요도
        if feature_names:
            imp = self.model.feature_importances_
            self.feature_importance_ = dict(zip(feature_names, imp.tolist()))

        # 학습 결과
        train_pred = self.model.predict(X)
        train_acc = np.mean(train_pred == y)

        result = {"train_accuracy": float(train_acc)}
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = np.mean(val_pred == y_val)
            result["val_accuracy"] = float(val_acc)

        logger.info("LightGBM 학습 완료: train_acc=%.4f", train_acc)
        return result

    def incremental_train(self, X: np.ndarray, y: np.ndarray,
                          sample_weight: np.ndarray = None,
                          n_new_trees: int = 100) -> dict:
        """기존 모델에서 이어서 학습 (온라인 학습).

        매 거래 세션 후 새 데이터로 모델을 점진적으로 업데이트.
        전체 재학습 없이 빠르게 적응.
        """
        import lightgbm as lgb

        if self.model is None:
            logger.warning("기존 모델 없음 — 전체 학습으로 전환")
            return self.train(X, y, sample_weight=sample_weight)

        # 기존 모델에서 이어서 학습
        old_n_estimators = self.model.n_estimators
        self.model.n_estimators = old_n_estimators + n_new_trees

        self.model.fit(
            X, y,
            sample_weight=sample_weight,
            init_model=self.model,
            callbacks=[lgb.log_evaluation(period=0)],
        )

        train_pred = self.model.predict(X)
        train_acc = np.mean(train_pred == y)

        logger.info("온라인 학습 완료: +%d trees (총 %d), acc=%.4f",
                    n_new_trees, self.model.n_estimators, train_acc)
        return {"train_accuracy": float(train_acc), "n_trees": self.model.n_estimators}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """예측 클래스 반환."""
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """클래스별 확률 반환. shape: (n_samples, num_classes)"""
        if self.model is None:
            raise RuntimeError("모델이 학습되지 않았습니다")
        return self.model.predict_proba(X)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_names": self.feature_names,
                "feature_importance": self.feature_importance_,
                "params": self.params,
            }, f)
        logger.info("LightGBM 모델 저장: %s", path)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data.get("feature_names")
        self.feature_importance_ = data.get("feature_importance")
        self.params = data.get("params", self.params)
        logger.info("LightGBM 모델 로드: %s", path)

    def get_top_features(self, n: int = 20) -> list[tuple[str, float]]:
        """상위 N개 중요 피처."""
        if not self.feature_importance_:
            return []
        sorted_feat = sorted(self.feature_importance_.items(), key=lambda x: -x[1])
        return sorted_feat[:n]
