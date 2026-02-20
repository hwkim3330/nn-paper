"""
Base Collector — Rate-limited API 래퍼
- 키움 API: 초당 5회 제한
- 에러 시 재시도 (최대 3회, exponential backoff)
"""
import time
import logging
import threading
from functools import wraps

logger = logging.getLogger(__name__)


class RateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, calls_per_sec: int = 5):
        self.calls_per_sec = calls_per_sec
        self.interval = 1.0 / calls_per_sec
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self.interval:
                sleep_time = self.interval - elapsed
                time.sleep(sleep_time)
            self._last_call = time.time()


# 싱글턴 rate limiter
_rate_limiter = RateLimiter(calls_per_sec=5)


class BaseCollector:
    """모든 Collector의 베이스 클래스."""

    def __init__(self, api, db):
        """
        Args:
            api: KiwoomAPI 인스턴스
            db: Database 인스턴스
        """
        self.api = api
        self.db = db
        self.rate_limiter = _rate_limiter

    def _api_call(self, func, *args, max_retries: int = 3, **kwargs):
        """
        Rate-limited API 호출 with retry.
        Returns: API 결과 또는 None (실패 시)
        """
        for attempt in range(max_retries):
            try:
                self.rate_limiter.wait()
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    "API call failed (attempt %d/%d): %s - %s",
                    attempt + 1, max_retries, func.__name__ if hasattr(func, '__name__') else str(func), e,
                )
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s
                else:
                    logger.error("API call failed after %d retries: %s", max_retries, e)
                    return None

    def _api_call_raw(self, method, **kwargs):
        """
        kiwoom-rest-api 패키지 메서드 직접 호출 (rate limited).
        """
        for attempt in range(3):
            try:
                self.rate_limiter.wait()
                return method(**kwargs)
            except Exception as e:
                logger.warning("Raw API call failed (attempt %d): %s", attempt + 1, e)
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None
