"""
피처 정규화
- Z-score normalization (학습 시 통계 저장, 추론 시 적용)
- 이상치 클리핑
"""
import json
import math
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """Z-score 정규화 + 이상치 클리핑."""

    def __init__(self):
        self.stats = {}  # {feature_name: {"mean": m, "std": s}}

    def fit(self, feature_dicts: list[dict]):
        """학습 데이터에서 통계 계산."""
        if not feature_dicts:
            return

        # 모든 피처 이름 수집
        all_keys = set()
        for fd in feature_dicts:
            all_keys.update(fd.keys())

        for key in all_keys:
            values = [fd.get(key, 0) for fd in feature_dicts if fd.get(key) is not None]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = math.sqrt(variance) if variance > 0 else 1.0
            self.stats[key] = {"mean": mean, "std": std}

        logger.info("Normalizer fit: %d features", len(self.stats))

    def transform(self, features: dict) -> dict:
        """Z-score 정규화 적용."""
        result = {}
        for key, value in features.items():
            if value is None:
                result[key] = 0.0
                continue
            stat = self.stats.get(key)
            if stat and stat["std"] > 0:
                z = (value - stat["mean"]) / stat["std"]
                # 클리핑 [-5, 5]
                z = max(-5.0, min(5.0, z))
                result[key] = round(z, 6)
            else:
                result[key] = 0.0
        return result

    def fit_transform(self, feature_dicts: list[dict]) -> list[dict]:
        """fit + transform."""
        self.fit(feature_dicts)
        return [self.transform(fd) for fd in feature_dicts]

    def save(self, path: str):
        """통계 저장."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.stats, f, indent=2)
        logger.info("Normalizer saved: %s (%d features)", path, len(self.stats))

    def load(self, path: str):
        """통계 로드."""
        with open(path) as f:
            self.stats = json.load(f)
        logger.info("Normalizer loaded: %s (%d features)", path, len(self.stats))

    def get_feature_names(self) -> list[str]:
        """정렬된 피처 이름 목록."""
        return sorted(self.stats.keys())
