"""Advanced rate limiting system with Redis support and backpressure."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from .config import BackpressureStrategy, RateLimitConfig
from .exceptions import RateLimitError
from .metrics import get_metrics_collector, track_metric

logger = logging.getLogger(__name__)


@dataclass
class RateLimitInfo:
    """Information about current rate limit status."""

    service: str
    limit_type: str
    current_count: int
    limit: int
    window_seconds: int
    reset_time: float
    retry_after: float | None = None

    @property
    def is_exceeded(self) -> bool:
        """Check if rate limit is exceeded."""
        return self.current_count >= self.limit

    @property
    def remaining(self) -> int:
        """Get remaining requests in current window."""
        return max(0, self.limit - self.current_count)

    @property
    def usage_percentage(self) -> float:
        """Get usage as percentage of limit."""
        return (self.current_count / self.limit) * 100 if self.limit > 0 else 0


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> RateLimitInfo:
        """Check if request is allowed and return rate limit info."""
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a specific key."""
        pass

    @abstractmethod
    async def get_info(self, key: str) -> RateLimitInfo | None:
        """Get current rate limit information."""
        pass


class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter implementation."""

    def __init__(self, capacity: int, refill_rate: float, refill_interval: float = 1.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_interval = refill_interval
        self._buckets: dict[str, dict[str, float]] = {}
        self._lock = asyncio.Lock()

    async def _refill_bucket(self, key: str) -> None:
        """Refill tokens in bucket based on elapsed time."""
        now = time.time()
        bucket = self._buckets.setdefault(key, {
            'tokens': float(self.capacity),
            'last_refill': now
        })

        elapsed = now - bucket['last_refill']
        tokens_to_add = elapsed * self.refill_rate / self.refill_interval

        bucket['tokens'] = min(self.capacity, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> RateLimitInfo:
        """Check if request is allowed using token bucket algorithm."""
        async with self._lock:
            await self._refill_bucket(key)
            bucket = self._buckets[key]

            current_tokens = int(bucket['tokens'])

            if bucket['tokens'] >= 1.0:
                bucket['tokens'] -= 1.0
                allowed = True
            else:
                allowed = False

            # Calculate retry after based on refill rate
            retry_after = None
            if not allowed:
                retry_after = (1.0 - bucket['tokens']) / (self.refill_rate / self.refill_interval)

            return RateLimitInfo(
                service=key.split(':')[0] if ':' in key else 'unknown',
                limit_type='token_bucket',
                current_count=self.capacity - current_tokens,
                limit=self.capacity,
                window_seconds=int(self.refill_interval),
                reset_time=time.time() + retry_after if retry_after else time.time(),
                retry_after=retry_after if not allowed else None
            )

    async def reset(self, key: str) -> None:
        """Reset token bucket for key."""
        async with self._lock:
            if key in self._buckets:
                self._buckets[key]['tokens'] = float(self.capacity)
                self._buckets[key]['last_refill'] = time.time()

    async def get_info(self, key: str) -> RateLimitInfo | None:
        """Get current bucket information."""
        async with self._lock:
            if key not in self._buckets:
                return None

            await self._refill_bucket(key)
            bucket = self._buckets[key]
            current_tokens = int(bucket['tokens'])

            return RateLimitInfo(
                service=key.split(':')[0] if ':' in key else 'unknown',
                limit_type='token_bucket',
                current_count=self.capacity - current_tokens,
                limit=self.capacity,
                window_seconds=int(self.refill_interval),
                reset_time=time.time(),
            )


class SlidingWindowLimiter(RateLimiter):
    """Sliding window rate limiter implementation."""

    def __init__(self, redis_client: Any | None = None):
        self.redis_client = redis_client
        self._local_windows: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> RateLimitInfo:
        """Check if request is allowed using sliding window algorithm."""
        if self.redis_client:
            return await self._redis_sliding_window(key, limit, window_seconds)
        else:
            return await self._local_sliding_window(key, limit, window_seconds)

    async def _local_sliding_window(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> RateLimitInfo:
        """Local sliding window implementation."""
        async with self._lock:
            now = time.time()
            window_start = now - window_seconds

            # Clean old entries and count current requests
            if key not in self._local_windows:
                self._local_windows[key] = []

            window = self._local_windows[key]
            window[:] = [ts for ts in window if ts > window_start]

            current_count = len(window)
            allowed = current_count < limit

            if allowed:
                window.append(now)
                current_count += 1

            # Calculate when window will reset
            reset_time = window[0] + window_seconds if window else now + window_seconds
            retry_after = reset_time - now if not allowed else None

            return RateLimitInfo(
                service=key.split(':')[0] if ':' in key else 'unknown',
                limit_type='sliding_window',
                current_count=current_count,
                limit=limit,
                window_seconds=window_seconds,
                reset_time=reset_time,
                retry_after=retry_after
            )

    async def _redis_sliding_window(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> RateLimitInfo:
        """Redis-based sliding window implementation."""
        now = time.time()
        window_start = now - window_seconds

        pipeline = self.redis_client.pipeline()

        # Remove old entries
        pipeline.zremrangebyscore(key, 0, window_start)

        # Count current entries
        pipeline.zcard(key)

        # Add current request
        pipeline.zadd(key, {str(now): now})

        # Set expiry
        pipeline.expire(key, window_seconds + 1)

        results = await pipeline.execute()
        current_count = results[1]  # Count before adding current request

        allowed = current_count < limit

        if not allowed:
            # Remove the request we just added since it's not allowed
            await self.redis_client.zrem(key, str(now))
        else:
            current_count += 1

        # Calculate reset time from oldest entry
        oldest_entries = await self.redis_client.zrange(key, 0, 0, withscores=True)
        if oldest_entries:
            reset_time = oldest_entries[0][1] + window_seconds
        else:
            reset_time = now + window_seconds

        retry_after = reset_time - now if not allowed else None

        return RateLimitInfo(
            service=key.split(':')[0] if ':' in key else 'unknown',
            limit_type='sliding_window',
            current_count=current_count,
            limit=limit,
            window_seconds=window_seconds,
            reset_time=reset_time,
            retry_after=retry_after
        )

    async def reset(self, key: str) -> None:
        """Reset sliding window for key."""
        if self.redis_client:
            await self.redis_client.delete(key)
        else:
            async with self._lock:
                self._local_windows.pop(key, None)

    async def get_info(self, key: str) -> RateLimitInfo | None:
        """Get current sliding window information."""
        # This would need window_seconds and limit parameters
        # For now, return None as we can't determine without those params
        return None


class AdaptiveRateLimiter(RateLimiter):
    """Adaptive rate limiter that adjusts based on system load."""

    def __init__(
        self,
        base_limiter: RateLimiter,
        load_factor_callback: Any | None = None
    ):
        self.base_limiter = base_limiter
        self.load_factor_callback = load_factor_callback or self._default_load_factor
        self._load_history: list[float] = []
        self._adjustment_factor = 1.0

    async def _default_load_factor(self) -> float:
        """Default load factor based on system metrics."""
        # This would integrate with system monitoring
        # For now, return a static value
        return 1.0

    async def _calculate_adjustment(self) -> float:
        """Calculate rate limit adjustment based on load."""
        current_load = await self.load_factor_callback()
        self._load_history.append(current_load)

        # Keep only recent history
        if len(self._load_history) > 100:
            self._load_history.pop(0)

        # Calculate adjustment factor
        if len(self._load_history) >= 5:
            avg_load = sum(self._load_history[-5:]) / 5
            if avg_load > 0.8:  # High load
                self._adjustment_factor = max(0.5, self._adjustment_factor * 0.95)
            elif avg_load < 0.5:  # Low load
                self._adjustment_factor = min(2.0, self._adjustment_factor * 1.05)

        return self._adjustment_factor

    async def is_allowed(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> RateLimitInfo:
        """Check if request is allowed with adaptive limits."""
        adjustment = await self._calculate_adjustment()
        adjusted_limit = int(limit * adjustment)

        info = await self.base_limiter.is_allowed(key, adjusted_limit, window_seconds)

        # Update info to reflect original limit for transparency
        info.limit = limit
        info.current_count = int(info.current_count / adjustment) if adjustment != 0 else info.current_count

        return info

    async def reset(self, key: str) -> None:
        """Reset adaptive rate limiter."""
        await self.base_limiter.reset(key)

    async def get_info(self, key: str) -> RateLimitInfo | None:
        """Get adaptive rate limiter info."""
        return await self.base_limiter.get_info(key)


class RateLimitManager:
    """Main rate limit manager that coordinates multiple limiters."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._limiters: dict[str, RateLimiter] = {}
        self._redis_client: Any | None = None
        self._metrics = get_metrics_collector()

        # Initialize Redis if configured
        if config.use_redis and REDIS_AVAILABLE and config.redis_url:
            self._setup_redis()

        # Setup default limiters
        self._setup_limiters()

    def _setup_redis(self) -> None:
        """Setup Redis connection for distributed rate limiting."""
        try:
            self._redis_client = redis.from_url(
                self.config.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            logger.info("Redis client initialized for rate limiting")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            self._redis_client = None

    def _setup_limiters(self) -> None:
        """Setup rate limiters based on configuration."""
        # Global limiter
        if self.config.global_max > 0:
            if self._redis_client:
                self._limiters['global'] = SlidingWindowLimiter(self._redis_client)
            else:
                self._limiters['global'] = TokenBucketLimiter(
                    capacity=self.config.global_max,
                    refill_rate=self.config.global_max,
                    refill_interval=1.0
                )

        # Service-specific limiters
        for service, limits in self.config.service_limits.items():
            if self._redis_client:
                limiter = SlidingWindowLimiter(self._redis_client)
            else:
                # Use token bucket with service-specific capacity
                calls_limit = limits.get('calls', 100)
                limiter = TokenBucketLimiter(
                    capacity=calls_limit,
                    refill_rate=calls_limit,
                    refill_interval=1.0
                )

            # Wrap with adaptive limiter if needed
            if self.config.backpressure_strategy == BackpressureStrategy.ADAPTIVE:
                limiter = AdaptiveRateLimiter(limiter)

            self._limiters[service] = limiter

    @track_metric("rate_limit_check", "histogram")
    async def check_rate_limit(
        self,
        service: str,
        identifier: str,
        limit_type: str = "calls"
    ) -> RateLimitInfo:
        """Check rate limit for a service and identifier."""
        # Build rate limit key
        key = f"{service}:{identifier}:{limit_type}"

        # Get service configuration
        service_config = self.config.service_limits.get(service, {})
        limit = service_config.get(limit_type, self.config.global_max)
        window = service_config.get('window', 60)  # Default 60 seconds

        # Check global limit first
        if 'global' in self._limiters:
            global_info = await self._limiters['global'].is_allowed(
                f"global:{identifier}",
                self.config.global_max,
                60
            )

            if global_info.is_exceeded:
                self._metrics.increment_counter(
                    'rate_limit_hits_total',
                    labels={'service': 'global', 'limit_type': limit_type}
                )

                if self.config.backpressure_strategy == BackpressureStrategy.FAIL_FAST:
                    raise RateLimitError(
                        service='global',
                        limit_type=limit_type,
                        retry_after=global_info.retry_after
                    )

                return global_info

        # Check service-specific limit
        limiter = self._limiters.get(service)
        if not limiter:
            # Create default limiter if service not configured
            limiter = TokenBucketLimiter(
                capacity=limit,
                refill_rate=limit,
                refill_interval=1.0
            )
            self._limiters[service] = limiter

        info = await limiter.is_allowed(key, limit, window)

        if info.is_exceeded:
            self._metrics.increment_counter(
                'rate_limit_hits_total',
                labels={'service': service, 'limit_type': limit_type}
            )

            if self.config.backpressure_strategy == BackpressureStrategy.FAIL_FAST:
                raise RateLimitError(
                    service=service,
                    limit_type=limit_type,
                    retry_after=info.retry_after
                )

        return info

    @asynccontextmanager
    async def rate_limited_execution(
        self,
        service: str,
        identifier: str,
        limit_type: str = "calls"
    ):
        """Context manager for rate-limited execution."""
        info = await self.check_rate_limit(service, identifier, limit_type)

        if info.is_exceeded:
            if self.config.backpressure_strategy == BackpressureStrategy.QUEUE:
                # Wait and retry
                if info.retry_after:
                    await asyncio.sleep(info.retry_after)
                    info = await self.check_rate_limit(service, identifier, limit_type)

            if info.is_exceeded:
                raise RateLimitError(
                    service=service,
                    limit_type=limit_type,
                    retry_after=info.retry_after
                )

        try:
            yield info
        except Exception:
            # Could implement compensation logic here
            raise

    async def reset_rate_limit(self, service: str, identifier: str) -> None:
        """Reset rate limit for a specific service and identifier."""
        limiter = self._limiters.get(service)
        if limiter:
            key = f"{service}:{identifier}"
            await limiter.reset(key)

    async def get_rate_limit_info(
        self,
        service: str,
        identifier: str
    ) -> RateLimitInfo | None:
        """Get current rate limit information."""
        limiter = self._limiters.get(service)
        if limiter:
            key = f"{service}:{identifier}"
            return await limiter.get_info(key)
        return None

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._redis_client:
            await self._redis_client.close()

        logger.info("Rate limit manager cleanup completed")
