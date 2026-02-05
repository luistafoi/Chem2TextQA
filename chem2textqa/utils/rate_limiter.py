import threading
import time


class RateLimiter:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, requests_per_second: float):
        self._rate = requests_per_second
        self._min_interval = 1.0 / requests_per_second
        self._last_request_time = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until a request is allowed."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request_time = time.monotonic()
