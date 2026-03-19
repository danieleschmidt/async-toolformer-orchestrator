"""Token-bucket rate limiter — one per tool, fully async."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class TokenBucket:
    """
    Leaky-token-bucket rate limiter.

    Capacity = calls_per_min tokens.  One token consumed per call.
    Tokens refill continuously at `calls_per_min / 60` tokens/second.

    Callers `await acquire()` — they block until a token is available.
    No fairness guarantees (asyncio.Lock gives FIFO though).
    """

    calls_per_min: float
    """Maximum sustained rate in calls per minute."""

    _tokens: float = field(init=False)
    _last_refill: float = field(init=False)
    _lock: asyncio.Lock = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.calls_per_min)  # start full
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    @property
    def _rate_per_sec(self) -> float:
        return self.calls_per_min / 60.0

    @property
    def _capacity(self) -> float:
        return float(self.calls_per_min)

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_sec)
        self._last_refill = now

    def _wait_time(self) -> float:
        """Seconds until 1 token will be available (0 if already available)."""
        self._refill()
        if self._tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self._tokens
        return deficit / self._rate_per_sec

    async def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        async with self._lock:
            wait = self._wait_time()
            if wait > 0:
                await asyncio.sleep(wait)
                self._refill()
            self._tokens -= 1.0


class RateLimiterRegistry:
    """Per-tool token buckets, lazily created."""

    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}

    def register(self, tool_name: str, calls_per_min: float) -> None:
        self._buckets[tool_name] = TokenBucket(calls_per_min=calls_per_min)

    def get(self, tool_name: str) -> TokenBucket | None:
        return self._buckets.get(tool_name)

    async def acquire(self, tool_name: str) -> None:
        """Acquire a token for *tool_name*.  No-op if no bucket registered."""
        bucket = self._buckets.get(tool_name)
        if bucket is not None:
            await bucket.acquire()
