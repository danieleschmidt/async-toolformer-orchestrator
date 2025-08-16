"""Redis manager for distributed rate limiting and caching."""

import asyncio
import json
import logging
import pickle
import time
from typing import Any

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    aioredis = None

logger = logging.getLogger(__name__)


class RedisManager:
    """
    Manages Redis connections for distributed operations.

    Features:
    - Distributed rate limiting
    - Result caching
    - Distributed locks
    - Pub/sub for coordination
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str | None = None,
        db: int = 0,
        ssl: bool = False,
        cluster_mode: bool = False,
        key_prefix: str = "orchestrator",
        ttl_seconds: int = 3600,
    ):
        """
        Initialize Redis manager.

        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            db: Redis database number
            ssl: Use SSL connection
            cluster_mode: Use Redis cluster
            key_prefix: Prefix for all keys
            ttl_seconds: Default TTL for cached items
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis package not installed. Install with: pip install redis")

        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.ssl = ssl
        self.cluster_mode = cluster_mode
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds

        self._client: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._lock_registry: dict[str, asyncio.Lock] = {}

        # Metrics
        self._metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "rate_limit_checks": 0,
            "rate_limit_exceeded": 0,
            "distributed_locks": 0,
        }

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            if self.cluster_mode:
                # Redis cluster connection
                from redis.asyncio.cluster import RedisCluster
                self._client = await RedisCluster(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    ssl=self.ssl,
                    decode_responses=False,
                )
            else:
                # Single Redis instance
                self._client = await aioredis.from_url(
                    f"{'rediss' if self.ssl else 'redis'}://{self.host}:{self.port}/{self.db}",
                    password=self.password,
                    decode_responses=False,
                )

            # Test connection
            await self._client.ping()
            logger.info(f"Connected to Redis at {self.host}:{self.port}")

            # Setup pub/sub
            self._pubsub = self._client.pubsub()

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._pubsub:
            await self._pubsub.close()

        if self._client:
            await self._client.close()

        logger.info("Disconnected from Redis")

    def _make_key(self, key: str) -> str:
        """Create a namespaced key."""
        return f"{self.key_prefix}:{key}"

    # ========================================
    # Caching Operations
    # ========================================

    async def cache_get(self, key: str) -> Any | None:
        """
        Get a cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self._client:
            return None

        full_key = self._make_key(f"cache:{key}")

        try:
            data = await self._client.get(full_key)
            if data:
                self._metrics["cache_hits"] += 1
                return pickle.loads(data)
            else:
                self._metrics["cache_misses"] += 1
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    async def cache_set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> bool:
        """
        Set a cached value.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: TTL in seconds

        Returns:
            Success status
        """
        if not self._client:
            return False

        full_key = self._make_key(f"cache:{key}")
        ttl = ttl_seconds or self.ttl_seconds

        try:
            data = pickle.dumps(value)
            await self._client.setex(full_key, ttl, data)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    async def cache_delete(self, key: str) -> bool:
        """Delete a cached value."""
        if not self._client:
            return False

        full_key = self._make_key(f"cache:{key}")

        try:
            await self._client.delete(full_key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    async def cache_clear_pattern(self, pattern: str) -> int:
        """
        Clear all cache entries matching a pattern.

        Args:
            pattern: Key pattern (e.g., "user:*")

        Returns:
            Number of keys deleted
        """
        if not self._client:
            return 0

        full_pattern = self._make_key(f"cache:{pattern}")

        try:
            # Use SCAN to avoid blocking
            deleted = 0
            async for key in self._client.scan_iter(match=full_pattern):
                await self._client.delete(key)
                deleted += 1
            return deleted
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0

    # ========================================
    # Rate Limiting Operations
    # ========================================

    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Check if rate limit is exceeded using sliding window.

        Args:
            key: Rate limit key (e.g., "api:openai")
            max_requests: Maximum requests in window
            window_seconds: Time window in seconds

        Returns:
            (allowed, remaining_requests)
        """
        if not self._client:
            return True, max_requests

        self._metrics["rate_limit_checks"] += 1

        full_key = self._make_key(f"ratelimit:{key}")
        now = time.time()
        window_start = now - window_seconds

        pipe = self._client.pipeline()

        try:
            # Remove old entries
            pipe.zremrangebyscore(full_key, 0, window_start)

            # Count current entries
            pipe.zcard(full_key)

            # Add current request
            pipe.zadd(full_key, {str(now): now})

            # Set expiry
            pipe.expire(full_key, window_seconds + 1)

            results = await pipe.execute()
            current_count = results[1]

            if current_count >= max_requests:
                self._metrics["rate_limit_exceeded"] += 1
                return False, 0

            return True, max_requests - current_count - 1

        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True, max_requests

    async def get_rate_limit_info(
        self,
        key: str,
        window_seconds: int,
    ) -> dict[str, Any]:
        """
        Get detailed rate limit information.

        Args:
            key: Rate limit key
            window_seconds: Time window

        Returns:
            Rate limit information
        """
        if not self._client:
            return {"available": True, "requests_in_window": 0}

        full_key = self._make_key(f"ratelimit:{key}")
        now = time.time()
        window_start = now - window_seconds

        try:
            # Remove old entries
            await self._client.zremrangebyscore(full_key, 0, window_start)

            # Get current entries
            entries = await self._client.zrange(full_key, 0, -1, withscores=True)

            return {
                "key": key,
                "requests_in_window": len(entries),
                "window_seconds": window_seconds,
                "oldest_request": min([score for _, score in entries]) if entries else None,
                "newest_request": max([score for _, score in entries]) if entries else None,
            }
        except Exception as e:
            logger.error(f"Rate limit info error: {e}")
            return {"available": True, "requests_in_window": 0}

    # ========================================
    # Distributed Lock Operations
    # ========================================

    async def acquire_lock(
        self,
        key: str,
        timeout_seconds: int = 10,
        blocking: bool = True,
        blocking_timeout: float = None,
    ) -> bool:
        """
        Acquire a distributed lock.

        Args:
            key: Lock key
            timeout_seconds: Lock timeout
            blocking: Whether to block waiting for lock
            blocking_timeout: Maximum time to wait

        Returns:
            Success status
        """
        if not self._client:
            return True  # No Redis, allow operation

        full_key = self._make_key(f"lock:{key}")
        self._metrics["distributed_locks"] += 1

        try:
            # Try to acquire lock
            lock_id = f"{asyncio.current_task().get_name()}_{time.time()}"

            if blocking:
                # Blocking acquire with retry
                start_time = time.time()
                while True:
                    acquired = await self._client.set(
                        full_key,
                        lock_id,
                        nx=True,
                        ex=timeout_seconds,
                    )

                    if acquired:
                        self._lock_registry[key] = lock_id
                        return True

                    if blocking_timeout and (time.time() - start_time) > blocking_timeout:
                        return False

                    await asyncio.sleep(0.1)
            else:
                # Non-blocking acquire
                acquired = await self._client.set(
                    full_key,
                    lock_id,
                    nx=True,
                    ex=timeout_seconds,
                )

                if acquired:
                    self._lock_registry[key] = lock_id
                    return True
                return False

        except Exception as e:
            logger.error(f"Lock acquire error: {e}")
            return False

    async def release_lock(self, key: str) -> bool:
        """
        Release a distributed lock.

        Args:
            key: Lock key

        Returns:
            Success status
        """
        if not self._client:
            return True

        full_key = self._make_key(f"lock:{key}")

        try:
            # Only release if we own the lock
            if key in self._lock_registry:
                lock_id = self._lock_registry[key]

                # Use Lua script to ensure atomic check-and-delete
                lua_script = """
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("del", KEYS[1])
                else
                    return 0
                end
                """

                result = await self._client.eval(
                    lua_script,
                    1,
                    full_key,
                    lock_id,
                )

                if result:
                    del self._lock_registry[key]
                    return True

            return False

        except Exception as e:
            logger.error(f"Lock release error: {e}")
            return False

    # ========================================
    # Pub/Sub Operations
    # ========================================

    async def publish(self, channel: str, message: Any) -> int:
        """
        Publish a message to a channel.

        Args:
            channel: Channel name
            message: Message to publish

        Returns:
            Number of subscribers that received the message
        """
        if not self._client:
            return 0

        full_channel = self._make_key(f"channel:{channel}")

        try:
            data = json.dumps(message) if not isinstance(message, str | bytes) else message
            return await self._client.publish(full_channel, data)
        except Exception as e:
            logger.error(f"Publish error: {e}")
            return 0

    async def subscribe(self, channel: str) -> None:
        """Subscribe to a channel."""
        if not self._pubsub:
            return

        full_channel = self._make_key(f"channel:{channel}")
        await self._pubsub.subscribe(full_channel)

    async def get_message(self, timeout: float = 1.0) -> dict[str, Any] | None:
        """Get a message from subscribed channels."""
        if not self._pubsub:
            return None

        try:
            message = await self._pubsub.get_message(timeout=timeout)
            if message and message["type"] == "message":
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                try:
                    return json.loads(data)
                except:
                    return {"raw": data}
            return None
        except Exception as e:
            logger.error(f"Get message error: {e}")
            return None

    # ========================================
    # Metrics and Monitoring
    # ========================================

    async def increment_counter(self, key: str, amount: int = 1) -> int:
        """Increment a counter."""
        if not self._client:
            return 0

        full_key = self._make_key(f"counter:{key}")

        try:
            return await self._client.incrby(full_key, amount)
        except Exception as e:
            logger.error(f"Counter increment error: {e}")
            return 0

    async def get_counter(self, key: str) -> int:
        """Get a counter value."""
        if not self._client:
            return 0

        full_key = self._make_key(f"counter:{key}")

        try:
            value = await self._client.get(full_key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Counter get error: {e}")
            return 0

    def get_metrics(self) -> dict[str, Any]:
        """Get Redis manager metrics."""
        cache_total = self._metrics["cache_hits"] + self._metrics["cache_misses"]
        hit_rate = (
            self._metrics["cache_hits"] / cache_total if cache_total > 0 else 0
        )

        return {
            **self._metrics,
            "cache_hit_rate": hit_rate,
            "connected": self._client is not None,
            "active_locks": len(self._lock_registry),
        }

    async def health_check(self) -> bool:
        """Check Redis connection health."""
        if not self._client:
            return False

        try:
            await self._client.ping()
            return True
        except Exception:
            return False
