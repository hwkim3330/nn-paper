"""
호가 피처 — 비활성화
호가 데이터 수집이 안 되어 전부 0 → 노이즈만 추가하므로 제거.
피처 인터페이스는 유지하되 빈 dict 반환.
"""
import logging

logger = logging.getLogger(__name__)


def compute_orderbook_features(snapshots: list[dict]) -> dict:
    """호가 피처 비활성화 — 빈 dict 반환."""
    return {}
