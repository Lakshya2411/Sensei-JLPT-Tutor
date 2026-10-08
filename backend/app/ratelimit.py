"""Per-IP sliding-window rate limiter (in memory).

Stops one visitor from burning the whole day's free LLM quota. In-memory is
fine for a single small server; use Redis if you ever run several instances.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self.max = max_requests
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] >= self.window:
                q.popleft()
            if len(q) >= self.max:
                return False
            q.append(now)
            return True
