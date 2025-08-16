"""Advanced caching system for tool results."""

import asyncio
import contextlib
import hashlib
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .simple_structured_logging import get_logger

logger = get_logger(__name__)


class CompressionType(Enum):
    """Compression algorithms."""
    NONE = "none"
    ZLIB = "zlib"
    # Could add more: GZIP, LZ4, etc.


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live only
    ADAPTIVE = "adaptive"  # Intelligent eviction


@dataclass
class CacheEntry:
    """Enhanced cache entry with compression and metadata."""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: float | None = None
    compressed: bool = False
    compression_type: CompressionType = CompressionType.NONE
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at

    @property
    def compression_ratio(self) -> float:
        """Get compression ratio."""
        if self.original_size_bytes == 0:
            return 1.0
        return self.compressed_size_bytes / self.original_size_bytes

    @property
    def access_frequency(self) -> float:
        """Get access frequency (accesses per hour)."""
        if self.age_seconds == 0:
            return 0.0
        hours = self.age_seconds / 3600
        return self.access_count / hours if hours > 0 else self.access_count

    def touch(self) -> None:
        """Update access information."""
        self.accessed_at = time.time()
        self.access_count += 1

    def get_priority_score(self) -> float:
        """Get priority score for eviction (higher = keep longer)."""
        # Combine access frequency, recency, and size efficiency
        frequency_score = min(self.access_frequency, 10.0) / 10.0  # Cap at 10/hour
        recency_score = max(0, 1.0 - (self.age_seconds / 3600))  # Decay over 1 hour
        size_score = self.compression_ratio  # Better compression = higher score

        return (frequency_score * 0.5) + (recency_score * 0.3) + (size_score * 0.2)


class CacheBackend(ABC):
    """Abstract cache backend."""

    @abstractmethod
    async def get(self, key: str) -> CacheEntry | None:
        """Get cache entry by key."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set cache entry."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete cache entry."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache backend with LRU eviction."""

    def __init__(self, max_size: int = 1000, default_ttl: float | None = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0,
        }

    async def get(self, key: str) -> CacheEntry | None:
        """Get cache entry by key."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats['misses'] += 1
                return None

            if entry.is_expired:
                await self._remove_entry(key)
                self._stats['misses'] += 1
                return None

            # Update access info and LRU order
            entry.touch()
            self._access_order.remove(key)
            self._access_order.append(key)

            self._stats['hits'] += 1
            return entry

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Set cache entry."""
        async with self._lock:
            now = time.time()
            ttl = ttl or self.default_ttl

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                accessed_at=now,
                ttl=ttl
            )

            # Remove existing entry if present
            if key in self._cache:
                self._access_order.remove(key)

            # Add new entry
            self._cache[key] = entry
            self._access_order.append(key)

            # Evict if necessary
            while len(self._cache) > self.max_size:
                await self._evict_lru()

            self._stats['sets'] += 1

    async def delete(self, key: str) -> None:
        """Delete cache entry."""
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                self._stats['deletes'] += 1

    async def clear(self) -> None:
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total_requests) if total_requests > 0 else 0

            return {
                **self._stats,
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': hit_rate,
                'total_requests': total_requests,
            }

    async def _remove_entry(self, key: str) -> None:
        """Remove entry from cache and access order."""
        if key in self._cache:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order[0]
            await self._remove_entry(lru_key)
            self._stats['evictions'] += 1


class ToolResultCache:
    """High-level cache for tool execution results."""

    def __init__(
        self,
        backend: CacheBackend,
        enable_compression: bool = False,
        hash_args: bool = True
    ):
        self.backend = backend
        self.enable_compression = enable_compression
        self.hash_args = hash_args
        self._enabled = True

    def _generate_cache_key(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> str:
        """Generate cache key for tool execution."""
        if self.hash_args:
            # Create a deterministic hash of arguments
            args_str = str(sorted(args.items()))
            args_hash = hashlib.md5(args_str.encode()).hexdigest()[:8]
            key = f"tool:{tool_name}:{args_hash}"
        else:
            # Use simple concatenation (less robust but more readable)
            args_str = "_".join(f"{k}={v}" for k, v in sorted(args.items()))
            key = f"tool:{tool_name}:{args_str}"

        if context:
            context_hash = hashlib.md5(str(sorted(context.items())).encode()).hexdigest()[:8]
            key += f":{context_hash}"

        return key

    async def get_cached_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any] | None = None
    ) -> Any | None:
        """Get cached result for tool execution."""
        if not self._enabled:
            return None

        key = self._generate_cache_key(tool_name, args, context)

        try:
            entry = await self.backend.get(key)
            if entry:
                logger.debug(f"Cache hit for {tool_name}: {key}")
                return entry.value

            logger.debug(f"Cache miss for {tool_name}: {key}")
            return None

        except Exception as e:
            logger.warning(f"Cache get error for {key}: {e}")
            return None

    async def cache_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: Any,
        ttl: float | None = None,
        context: dict[str, Any] | None = None
    ) -> None:
        """Cache result of tool execution."""
        if not self._enabled:
            return

        key = self._generate_cache_key(tool_name, args, context)

        try:
            # Optionally compress large results
            cached_result = result
            if self.enable_compression:
                # Simple compression using pickle
                try:
                    serialized = pickle.dumps(result)
                    if len(serialized) > 1024:  # Only compress if > 1KB
                        import gzip
                        compressed = gzip.compress(serialized)
                        if len(compressed) < len(serialized):
                            cached_result = {'_compressed': True, 'data': compressed}
                except Exception:
                    # Fall back to uncompressed if compression fails
                    pass

            await self.backend.set(key, cached_result, ttl)
            logger.debug(f"Cached result for {tool_name}: {key}")

        except Exception as e:
            logger.warning(f"Cache set error for {key}: {e}")

    async def invalidate_tool(self, tool_name: str) -> None:
        """Invalidate all cached results for a tool."""
        # This is a simplified implementation
        # In practice, you'd need a more sophisticated approach
        logger.info(f"Cache invalidation requested for tool: {tool_name}")

    async def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return await self.backend.get_stats()

    def enable(self) -> None:
        """Enable caching."""
        self._enabled = True

    def disable(self) -> None:
        """Disable caching."""
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """Check if caching is enabled."""
        return self._enabled


# Utility functions for cache configuration
def create_memory_cache(
    max_size: int = 1000,
    default_ttl: float | None = 3600,  # 1 hour default
    enable_compression: bool = False
) -> ToolResultCache:
    """Create a memory-based tool result cache."""
    backend = MemoryCache(max_size=max_size, default_ttl=default_ttl)
    return ToolResultCache(
        backend=backend,
        enable_compression=enable_compression
    )


# Cache decorators for easy integration
def cached_tool(
    cache: ToolResultCache,
    ttl: float | None = None,
    cache_key_context: Any | None = None
):
    """Decorator for caching tool results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract tool name
            tool_name = getattr(func, '__name__', 'unknown')

            # Generate context if provided
            context = None
            if cache_key_context:
                with contextlib.suppress(Exception):
                    context = cache_key_context(*args, **kwargs)

            # Try to get cached result
            cached = await cache.get_cached_result(tool_name, kwargs, context)
            if cached is not None:
                # Handle decompression if needed
                if isinstance(cached, dict) and cached.get('_compressed'):
                    import gzip
                    import pickle
                    try:
                        decompressed = gzip.decompress(cached['data'])
                        return pickle.loads(decompressed)
                    except Exception:
                        pass  # Fall through to execute function
                else:
                    return cached

            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.cache_result(tool_name, kwargs, result, ttl, context)

            return result

        return wrapper
    return decorator
