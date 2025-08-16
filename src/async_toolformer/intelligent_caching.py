"""Intelligent caching system for Generation 3 optimization."""

import asyncio
import hashlib
import logging
import pickle
import time
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Different caching strategies."""
    LRU = auto()           # Least Recently Used
    LFU = auto()           # Least Frequently Used
    TTL = auto()           # Time To Live
    ADAPTIVE = auto()      # Adaptive based on usage patterns
    INTELLIGENT = auto()   # AI-driven caching decisions


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = auto()     # In-memory cache (fastest)
    L2_COMPRESSED = auto() # Compressed memory cache
    L3_DISK = auto()       # Disk-based cache (persistent)


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: float | None = None
    size_bytes: int = 0
    compression_ratio: float = 1.0
    hit_rate: float = 0.0
    computation_cost: float = 0.0  # Cost to recompute
    cache_level: CacheLevel = CacheLevel.L1_MEMORY
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    def get_age(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_at

    def update_access(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistics for cache performance."""

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    avg_computation_cost_saved: float = 0.0
    hit_rate: float = 0.0

    # Level-specific stats
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0

    # Time-based stats
    avg_retrieval_time_ms: float = 0.0
    avg_storage_time_ms: float = 0.0

    def calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_requests > 0:
            self.hit_rate = self.cache_hits / self.total_requests
        return self.hit_rate


class IntelligentCache:
    """Intelligent multi-level caching system with adaptive strategies."""

    def __init__(
        self,
        max_memory_mb: int = 100,
        max_disk_mb: int = 1000,
        default_ttl_seconds: float = 3600,
        strategy: CacheStrategy = CacheStrategy.INTELLIGENT
    ):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        self.default_ttl = default_ttl_seconds
        self.strategy = strategy

        # Multi-level cache storage
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()  # Memory cache
        self.l2_cache: dict[str, bytes] = {}  # Compressed cache
        self.l3_cache: dict[str, str] = {}   # Disk cache (file paths)

        # Cache metadata
        self.stats = CacheStats()
        self.access_patterns: dict[str, list[float]] = {}
        self.computation_costs: dict[str, float] = {}

        # Intelligent features
        self.learning_enabled = True
        self.prediction_enabled = True
        self.auto_optimization = True

        # Performance tracking
        self.performance_history: list[tuple[float, CacheStats]] = []

        # Lock for thread safety
        self.cache_lock = asyncio.Lock()

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache with intelligent retrieval."""
        async with self.cache_lock:
            start_time = time.time()
            self.stats.total_requests += 1

            try:
                # Try L1 cache first (memory)
                if key in self.l1_cache:
                    entry = self.l1_cache[key]

                    if entry.is_expired():
                        await self._remove_entry(key)
                        self.stats.cache_misses += 1
                        return default

                    # Update access pattern
                    entry.update_access()
                    self._move_to_front(key)  # LRU behavior
                    self._record_access_pattern(key)

                    self.stats.cache_hits += 1
                    self.stats.l1_hits += 1

                    # Update retrieval time
                    retrieval_time = (time.time() - start_time) * 1000
                    self._update_avg_retrieval_time(retrieval_time)

                    return entry.value

                # Try L2 cache (compressed)
                if key in self.l2_cache:
                    compressed_data = self.l2_cache[key]
                    value = await self._decompress_value(compressed_data)

                    # Promote to L1 if frequently accessed
                    if await self._should_promote_to_l1(key):
                        await self._promote_to_l1(key, value)

                    self.stats.cache_hits += 1
                    self.stats.l2_hits += 1
                    self._record_access_pattern(key)

                    retrieval_time = (time.time() - start_time) * 1000
                    self._update_avg_retrieval_time(retrieval_time)

                    return value

                # Try L3 cache (disk) - simplified for demo
                if key in self.l3_cache:
                    # In real implementation, would read from disk
                    self.stats.cache_hits += 1
                    self.stats.l3_hits += 1
                    self._record_access_pattern(key)
                    return default  # Placeholder

                # Cache miss
                self.stats.cache_misses += 1
                return default

            except Exception as e:
                logger.error(f"Cache get error for key {key}: {e}")
                self.stats.cache_misses += 1
                return default

    async def get_cached_result(self, tool_name: str, args: dict) -> Any | None:
        """Get cached result for a tool call (compatibility method)."""
        cache_key = self._generate_key(tool_name, args)
        return await self.get(cache_key)

    async def cache_result(self, tool_name: str, args: dict, result: Any, ttl: float = 3600) -> None:
        """Cache a tool result (compatibility method)."""
        cache_key = self._generate_key(tool_name, args)
        await self.put(cache_key, result, ttl)

    def _generate_key(self, tool_name: str, args: dict) -> str:
        """Generate cache key from tool name and arguments."""
        import hashlib
        import json
        args_str = json.dumps(args, sort_keys=True, default=str)
        key_data = f"{tool_name}:{args_str}"
        return hashlib.md5(key_data.encode()).hexdigest()

    async def put(
        self,
        key: str,
        value: Any,
        ttl_seconds: float | None = None,
        computation_cost: float = 1.0
    ) -> None:
        """Put value in cache with intelligent placement strategy."""
        async with self.cache_lock:
            start_time = time.time()

            try:
                ttl = ttl_seconds or self.default_ttl
                value_size = await self._calculate_size(value)

                # Determine optimal cache level
                cache_level = await self._determine_cache_level(key, value, value_size, computation_cost)

                # Create cache entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    ttl_seconds=ttl,
                    size_bytes=value_size,
                    computation_cost=computation_cost,
                    cache_level=cache_level
                )

                if cache_level == CacheLevel.L1_MEMORY:
                    await self._store_l1(entry)
                elif cache_level == CacheLevel.L2_COMPRESSED:
                    await self._store_l2(key, value)
                elif cache_level == CacheLevel.L3_DISK:
                    await self._store_l3(key, value)

                # Record computation cost for learning
                self.computation_costs[key] = computation_cost

                # Update storage time
                storage_time = (time.time() - start_time) * 1000
                self._update_avg_storage_time(storage_time)

            except Exception as e:
                logger.error(f"Cache put error for key {key}: {e}")

    async def _determine_cache_level(
        self,
        key: str,
        value: Any,
        size_bytes: int,
        computation_cost: float
    ) -> CacheLevel:
        """Intelligently determine which cache level to use."""

        # Check access patterns
        access_frequency = await self._get_access_frequency(key)
        recency = await self._get_access_recency(key)

        # Decision factors
        factors = {
            'size': size_bytes,
            'frequency': access_frequency,
            'recency': recency,
            'computation_cost': computation_cost,
            'current_l1_usage': self._get_l1_usage_ratio(),
        }

        # Intelligent cache level selection
        if self.strategy == CacheStrategy.INTELLIGENT:
            return await self._ai_cache_level_decision(factors)
        else:
            # Simple heuristic-based decision
            if size_bytes < 1024 and access_frequency > 0.1:  # Small and frequently accessed
                return CacheLevel.L1_MEMORY
            elif size_bytes < 10240 and computation_cost > 0.5:  # Medium size, expensive to compute
                return CacheLevel.L2_COMPRESSED
            else:
                return CacheLevel.L3_DISK

    async def _ai_cache_level_decision(self, factors: dict[str, float]) -> CacheLevel:
        """AI-driven cache level decision (simplified ML-like approach)."""

        # Scoring system for each cache level
        l1_score = 0.0
        l2_score = 0.0
        l3_score = 0.0

        # L1 (Memory) scoring
        if factors['size'] < 1024:  # Small size bonus
            l1_score += 0.3
        if factors['frequency'] > 0.2:  # High frequency bonus
            l1_score += 0.4
        if factors['recency'] < 60:  # Recently accessed bonus
            l1_score += 0.2
        if factors['current_l1_usage'] < 0.8:  # L1 not full bonus
            l1_score += 0.1

        # L2 (Compressed) scoring
        if factors['size'] < 10240:  # Medium size bonus
            l2_score += 0.2
        if factors['computation_cost'] > 0.5:  # High computation cost bonus
            l2_score += 0.3
        if factors['frequency'] > 0.05:  # Moderate frequency bonus
            l2_score += 0.3
        if factors['current_l1_usage'] > 0.7:  # L1 getting full bonus
            l2_score += 0.2

        # L3 (Disk) scoring - default for large/infrequent items
        l3_score += 0.1  # Base score
        if factors['size'] > 10240:  # Large size bonus
            l3_score += 0.3
        if factors['frequency'] < 0.02:  # Low frequency bonus
            l3_score += 0.3

        # Choose level with highest score
        scores = [(CacheLevel.L1_MEMORY, l1_score), (CacheLevel.L2_COMPRESSED, l2_score), (CacheLevel.L3_DISK, l3_score)]
        best_level = max(scores, key=lambda x: x[1])[0]

        return best_level

    async def _store_l1(self, entry: CacheEntry) -> None:
        """Store entry in L1 memory cache."""

        # Check if we need to evict entries
        while (self._get_l1_size_bytes() + entry.size_bytes > self.max_memory_bytes and
               len(self.l1_cache) > 0):
            await self._evict_l1_entry()

        self.l1_cache[entry.key] = entry
        self.stats.total_size_bytes += entry.size_bytes

    async def _store_l2(self, key: str, value: Any) -> None:
        """Store value in L2 compressed cache."""

        compressed_value = await self._compress_value(value)
        self.l2_cache[key] = compressed_value

    async def _store_l3(self, key: str, value: Any) -> None:
        """Store value in L3 disk cache."""

        # Simplified - in real implementation would write to disk
        file_path = f"/tmp/cache_{hashlib.md5(key.encode()).hexdigest()}"
        self.l3_cache[key] = file_path

    async def _evict_l1_entry(self) -> None:
        """Evict entry from L1 cache based on strategy."""

        if not self.l1_cache:
            return

        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used (oldest in OrderedDict)
            key_to_remove = next(iter(self.l1_cache))
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            key_to_remove = min(self.l1_cache.keys(),
                               key=lambda k: self.l1_cache[k].access_count)
        else:
            # Intelligent eviction
            key_to_remove = await self._intelligent_eviction_choice()

        await self._remove_entry(key_to_remove)
        self.stats.evictions += 1

    async def _intelligent_eviction_choice(self) -> str:
        """Make intelligent choice for cache eviction."""

        # Score each entry for eviction (lower score = better candidate for eviction)
        eviction_scores = {}

        for key, entry in self.l1_cache.items():
            score = 0.0

            # Recency factor (newer = higher score)
            age_hours = entry.get_age() / 3600
            score += max(0, 10 - age_hours)  # Decays over 10 hours

            # Frequency factor (more accesses = higher score)
            score += min(entry.access_count, 10) * 0.5

            # Computation cost factor (expensive to recompute = higher score)
            score += entry.computation_cost * 2

            # Size factor (smaller = slightly higher score for memory efficiency)
            score += max(0, 2 - (entry.size_bytes / 1024))

            eviction_scores[key] = score

        # Choose entry with lowest score for eviction
        key_to_evict = min(eviction_scores.keys(), key=lambda k: eviction_scores[k])
        return key_to_evict

    async def _remove_entry(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            self.stats.total_size_bytes -= entry.size_bytes
            del self.l1_cache[key]

        if key in self.l2_cache:
            del self.l2_cache[key]

        if key in self.l3_cache:
            del self.l3_cache[key]

    async def _should_promote_to_l1(self, key: str) -> bool:
        """Determine if key should be promoted to L1 cache."""
        frequency = await self._get_access_frequency(key)
        return frequency > 0.1 and self._get_l1_usage_ratio() < 0.9

    async def _promote_to_l1(self, key: str, value: Any) -> None:
        """Promote entry to L1 cache."""
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            last_accessed=time.time(),
            cache_level=CacheLevel.L1_MEMORY
        )
        await self._store_l1(entry)

    def _move_to_front(self, key: str) -> None:
        """Move key to front of LRU cache."""
        if key in self.l1_cache:
            self.l1_cache.move_to_end(key)

    def _record_access_pattern(self, key: str) -> None:
        """Record access pattern for learning."""
        if key not in self.access_patterns:
            self.access_patterns[key] = []

        self.access_patterns[key].append(time.time())

        # Trim old access records (keep last 100)
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]

    async def _get_access_frequency(self, key: str) -> float:
        """Calculate access frequency for key."""
        if key not in self.access_patterns:
            return 0.0

        accesses = self.access_patterns[key]
        if len(accesses) < 2:
            return 0.0

        # Calculate accesses per hour over last 24 hours
        current_time = time.time()
        recent_accesses = [t for t in accesses if current_time - t < 86400]  # 24 hours

        if len(recent_accesses) < 2:
            return 0.0

        time_span_hours = (recent_accesses[-1] - recent_accesses[0]) / 3600
        if time_span_hours > 0:
            return len(recent_accesses) / time_span_hours

        return 0.0

    async def _get_access_recency(self, key: str) -> float:
        """Get time since last access in seconds."""
        if key not in self.access_patterns or not self.access_patterns[key]:
            return float('inf')

        return time.time() - self.access_patterns[key][-1]

    def _get_l1_usage_ratio(self) -> float:
        """Get L1 cache usage ratio."""
        if self.max_memory_bytes == 0:
            return 1.0
        return self._get_l1_size_bytes() / self.max_memory_bytes

    def _get_l1_size_bytes(self) -> int:
        """Get current L1 cache size in bytes."""
        return sum(entry.size_bytes for entry in self.l1_cache.values())

    async def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, int | float):
                return 8  # Approximate size
            elif isinstance(value, list | dict):
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            else:
                return len(str(value).encode('utf-8'))
        except Exception:
            return 1024  # Default size estimate

    async def _compress_value(self, value: Any) -> bytes:
        """Compress value for L2 storage."""
        try:
            serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = zlib.compress(serialized, level=6)
            return compressed
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)

    async def _decompress_value(self, compressed_data: bytes) -> Any:
        """Decompress value from L2 storage."""
        try:
            decompressed = zlib.decompress(compressed_data)
            value = pickle.loads(decompressed)
            return value
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return pickle.loads(compressed_data)

    def _update_avg_retrieval_time(self, retrieval_time_ms: float) -> None:
        """Update average retrieval time."""
        current_avg = self.stats.avg_retrieval_time_ms
        total_requests = self.stats.total_requests

        if total_requests > 1:
            self.stats.avg_retrieval_time_ms = (
                (current_avg * (total_requests - 1) + retrieval_time_ms) / total_requests
            )
        else:
            self.stats.avg_retrieval_time_ms = retrieval_time_ms

    def _update_avg_storage_time(self, storage_time_ms: float) -> None:
        """Update average storage time."""
        # Simple exponential moving average
        alpha = 0.1  # Smoothing factor
        if self.stats.avg_storage_time_ms == 0:
            self.stats.avg_storage_time_ms = storage_time_ms
        else:
            self.stats.avg_storage_time_ms = (
                alpha * storage_time_ms + (1 - alpha) * self.stats.avg_storage_time_ms
            )

    async def clear(self) -> None:
        """Clear all caches."""
        async with self.cache_lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.l3_cache.clear()
            self.stats = CacheStats()
            self.access_patterns.clear()

    async def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        async with self.cache_lock:
            self.stats.calculate_hit_rate()
            return self.stats

    async def get_cache_info(self) -> dict[str, Any]:
        """Get detailed cache information."""
        async with self.cache_lock:
            return {
                "l1_entries": len(self.l1_cache),
                "l2_entries": len(self.l2_cache),
                "l3_entries": len(self.l3_cache),
                "l1_size_bytes": self._get_l1_size_bytes(),
                "l1_usage_ratio": self._get_l1_usage_ratio(),
                "max_memory_bytes": self.max_memory_bytes,
                "max_disk_bytes": self.max_disk_bytes,
                "strategy": self.strategy.name,
                "learning_enabled": self.learning_enabled,
                "prediction_enabled": self.prediction_enabled,
                "tracked_patterns": len(self.access_patterns),
            }

    async def optimize_cache(self) -> dict[str, Any]:
        """Perform cache optimization."""
        if not self.auto_optimization:
            return {"message": "Auto-optimization disabled"}

        async with self.cache_lock:
            optimizations_made = []

            # Optimize cache levels based on access patterns
            promoted_count = 0
            demoted_count = 0

            # Check for promotion candidates in L2
            for key in list(self.l2_cache.keys()):
                frequency = await self._get_access_frequency(key)
                if frequency > 0.2 and self._get_l1_usage_ratio() < 0.8:
                    # Promote to L1
                    compressed_data = self.l2_cache[key]
                    value = await self._decompress_value(compressed_data)
                    await self._promote_to_l1(key, value)
                    del self.l2_cache[key]
                    promoted_count += 1

            # Check for demotion candidates in L1
            for key, entry in list(self.l1_cache.items()):
                frequency = await self._get_access_frequency(key)
                if frequency < 0.01 and entry.get_age() > 3600:  # Low frequency and old
                    # Demote to L2
                    await self._store_l2(key, entry.value)
                    await self._remove_entry(key)
                    demoted_count += 1

            if promoted_count > 0:
                optimizations_made.append(f"Promoted {promoted_count} entries to L1")
            if demoted_count > 0:
                optimizations_made.append(f"Demoted {demoted_count} entries from L1")

            # Clean up expired entries
            expired_count = 0
            for key, entry in list(self.l1_cache.items()):
                if entry.is_expired():
                    await self._remove_entry(key)
                    expired_count += 1

            if expired_count > 0:
                optimizations_made.append(f"Removed {expired_count} expired entries")

            return {
                "optimizations_made": optimizations_made,
                "cache_info": await self.get_cache_info(),
                "stats": await self.get_stats()
            }


# Global intelligent cache instance
intelligent_cache = IntelligentCache()
