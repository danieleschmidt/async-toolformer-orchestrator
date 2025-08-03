"""Memory management for high-concurrency tool execution."""

import asyncio
import gc
import logging
import os
import pickle
import psutil
import tempfile
import time
import zlib
from typing import Any, Dict, Optional, Set
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    max_memory_gb: float = 8.0
    gc_threshold_gb: float = 6.0
    compress_results: bool = True
    compression_threshold_bytes: int = 10 * 1024  # 10KB
    swap_to_disk: bool = True
    disk_path: Optional[str] = None
    disk_threshold_gb: float = 4.0
    cache_ttl_seconds: int = 300


class MemoryManager:
    """
    Manages memory usage for high-concurrency tool execution.
    
    Features:
    - Memory monitoring and limits
    - Result compression
    - Disk swapping for large results
    - Garbage collection triggers
    - Memory pressure handling
    """
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the memory manager.
        
        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        
        # Set up disk swap directory
        if self.config.swap_to_disk:
            if self.config.disk_path:
                self.disk_path = Path(self.config.disk_path)
            else:
                self.disk_path = Path(tempfile.mkdtemp(prefix="orchestrator_swap_"))
            self.disk_path.mkdir(parents=True, exist_ok=True)
        else:
            self.disk_path = None
        
        # Track memory usage
        self._process = psutil.Process()
        self._initial_memory = self._get_memory_usage_gb()
        
        # Result storage
        self._memory_store: Dict[str, Any] = {}
        self._compressed_store: Dict[str, bytes] = {}
        self._disk_keys: Set[str] = set()
        
        # Metrics
        self._metrics = {
            "peak_memory_gb": self._initial_memory,
            "gc_triggers": 0,
            "compressions": 0,
            "disk_swaps": 0,
            "cache_evictions": 0,
        }
        
        # Start monitoring task
        self._monitor_task = asyncio.create_task(self._monitor_memory())
    
    def _get_memory_usage_gb(self) -> float:
        """Get current memory usage in GB."""
        return self._process.memory_info().rss / (1024 ** 3)
    
    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB."""
        return psutil.virtual_memory().available / (1024 ** 3)
    
    async def store_result(
        self, key: str, data: Any, compress: Optional[bool] = None
    ) -> None:
        """
        Store a result with memory management.
        
        Args:
            key: Result key
            data: Data to store
            compress: Whether to compress (None = auto)
        """
        # Check memory pressure
        await self._check_memory_pressure()
        
        # Estimate size
        size = self._estimate_size(data)
        
        # Decide on storage strategy
        if compress is None:
            compress = (
                self.config.compress_results
                and size > self.config.compression_threshold_bytes
            )
        
        if compress:
            # Compress the data
            compressed = self._compress_data(data)
            self._compressed_store[key] = compressed
            self._metrics["compressions"] += 1
            logger.debug(
                f"Compressed {key}: {size} bytes -> {len(compressed)} bytes"
            )
        elif self._should_swap_to_disk(size):
            # Swap to disk
            await self._swap_to_disk(key, data)
            self._metrics["disk_swaps"] += 1
        else:
            # Store in memory
            self._memory_store[key] = data
    
    async def get_result(self, key: str) -> Optional[Any]:
        """
        Retrieve a stored result.
        
        Args:
            key: Result key
            
        Returns:
            Stored data or None
        """
        # Check memory store
        if key in self._memory_store:
            return self._memory_store[key]
        
        # Check compressed store
        if key in self._compressed_store:
            return self._decompress_data(self._compressed_store[key])
        
        # Check disk
        if key in self._disk_keys:
            return await self._load_from_disk(key)
        
        return None
    
    async def delete_result(self, key: str) -> None:
        """Delete a stored result."""
        self._memory_store.pop(key, None)
        self._compressed_store.pop(key, None)
        
        if key in self._disk_keys:
            await self._delete_from_disk(key)
            self._disk_keys.discard(key)
    
    async def _monitor_memory(self) -> None:
        """Monitor memory usage and trigger cleanup as needed."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                current_memory = self._get_memory_usage_gb()
                
                # Update peak memory
                if current_memory > self._metrics["peak_memory_gb"]:
                    self._metrics["peak_memory_gb"] = current_memory
                
                # Check if we need to trigger GC
                if current_memory > self.config.gc_threshold_gb:
                    await self._trigger_gc()
                
                # Check if we're approaching limits
                if current_memory > self.config.max_memory_gb * 0.9:
                    await self._handle_memory_pressure()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")
    
    async def _check_memory_pressure(self) -> None:
        """Check and handle memory pressure before allocation."""
        current_memory = self._get_memory_usage_gb()
        
        if current_memory > self.config.max_memory_gb * 0.8:
            # Try to free memory
            await self._trigger_gc()
            
            # If still high, start evicting
            if current_memory > self.config.max_memory_gb * 0.9:
                await self._evict_old_results()
    
    async def _trigger_gc(self) -> None:
        """Trigger garbage collection."""
        logger.info("Triggering garbage collection")
        gc.collect()
        self._metrics["gc_triggers"] += 1
        
        # Log memory after GC
        new_memory = self._get_memory_usage_gb()
        logger.info(f"Memory after GC: {new_memory:.2f} GB")
    
    async def _handle_memory_pressure(self) -> None:
        """Handle high memory pressure."""
        logger.warning("High memory pressure detected")
        
        # First, try GC
        await self._trigger_gc()
        
        # Compress uncompressed results
        await self._compress_all_results()
        
        # Move large results to disk
        await self._swap_large_results_to_disk()
        
        # If still high, evict old results
        current_memory = self._get_memory_usage_gb()
        if current_memory > self.config.max_memory_gb * 0.95:
            await self._evict_old_results()
    
    async def _compress_all_results(self) -> None:
        """Compress all uncompressed results."""
        keys_to_compress = list(self._memory_store.keys())
        
        for key in keys_to_compress:
            if key in self._memory_store:
                data = self._memory_store.pop(key)
                compressed = self._compress_data(data)
                self._compressed_store[key] = compressed
                self._metrics["compressions"] += 1
    
    async def _swap_large_results_to_disk(self) -> None:
        """Swap large results to disk."""
        if not self.config.swap_to_disk:
            return
        
        # Find large results
        large_results = []
        for key, data in self._memory_store.items():
            size = self._estimate_size(data)
            if size > 1024 * 1024:  # > 1MB
                large_results.append((key, data, size))
        
        # Sort by size (largest first)
        large_results.sort(key=lambda x: x[2], reverse=True)
        
        # Swap to disk
        for key, data, _ in large_results[:10]:  # Swap top 10
            await self._swap_to_disk(key, data)
            self._memory_store.pop(key, None)
            self._metrics["disk_swaps"] += 1
    
    async def _evict_old_results(self, fraction: float = 0.2) -> None:
        """
        Evict old results to free memory.
        
        Args:
            fraction: Fraction of results to evict
        """
        logger.warning(f"Evicting {fraction * 100:.0f}% of cached results")
        
        # Calculate how many to evict
        total_items = (
            len(self._memory_store)
            + len(self._compressed_store)
            + len(self._disk_keys)
        )
        to_evict = int(total_items * fraction)
        
        # Evict from memory first
        evicted = 0
        for key in list(self._memory_store.keys())[:to_evict]:
            del self._memory_store[key]
            evicted += 1
        
        # Then compressed store
        remaining = to_evict - evicted
        for key in list(self._compressed_store.keys())[:remaining]:
            del self._compressed_store[key]
            evicted += 1
        
        # Finally disk (just remove references)
        remaining = to_evict - evicted
        for key in list(self._disk_keys)[:remaining]:
            self._disk_keys.discard(key)
            evicted += 1
        
        self._metrics["cache_evictions"] += evicted
        logger.info(f"Evicted {evicted} results")
    
    def _should_swap_to_disk(self, size_bytes: int) -> bool:
        """Check if data should be swapped to disk."""
        if not self.config.swap_to_disk:
            return False
        
        current_memory = self._get_memory_usage_gb()
        return (
            current_memory > self.config.disk_threshold_gb
            or size_bytes > 10 * 1024 * 1024  # > 10MB
        )
    
    async def _swap_to_disk(self, key: str, data: Any) -> None:
        """Swap data to disk."""
        if not self.disk_path:
            return
        
        file_path = self.disk_path / f"{key}.pkl"
        
        # Run disk I/O in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: file_path.write_bytes(pickle.dumps(data))
        )
        
        self._disk_keys.add(key)
        logger.debug(f"Swapped {key} to disk")
    
    async def _load_from_disk(self, key: str) -> Optional[Any]:
        """Load data from disk."""
        if not self.disk_path:
            return None
        
        file_path = self.disk_path / f"{key}.pkl"
        
        if not file_path.exists():
            return None
        
        # Run disk I/O in executor
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(
            None,
            lambda: pickle.loads(file_path.read_bytes())
        )
        
        return data
    
    async def _delete_from_disk(self, key: str) -> None:
        """Delete data from disk."""
        if not self.disk_path:
            return
        
        file_path = self.disk_path / f"{key}.pkl"
        
        if file_path.exists():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, file_path.unlink)
    
    def _compress_data(self, data: Any) -> bytes:
        """Compress data using zlib."""
        pickled = pickle.dumps(data)
        return zlib.compress(pickled, level=6)
    
    def _decompress_data(self, compressed: bytes) -> Any:
        """Decompress data."""
        decompressed = zlib.decompress(compressed)
        return pickle.loads(decompressed)
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate the size of data in bytes."""
        try:
            return len(pickle.dumps(data))
        except Exception:
            # Fallback for unpickleable objects
            return 1024  # Assume 1KB
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get memory manager metrics."""
        current_memory = self._get_memory_usage_gb()
        available_memory = self._get_available_memory_gb()
        
        return {
            **self._metrics,
            "current_memory_gb": current_memory,
            "available_memory_gb": available_memory,
            "memory_usage_percent": (current_memory / self.config.max_memory_gb) * 100,
            "items_in_memory": len(self._memory_store),
            "items_compressed": len(self._compressed_store),
            "items_on_disk": len(self._disk_keys),
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Cancel monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Clear stores
        self._memory_store.clear()
        self._compressed_store.clear()
        
        # Clean up disk files
        if self.disk_path and self.disk_path.exists():
            for file in self.disk_path.glob("*.pkl"):
                file.unlink()
            
            # Try to remove directory (only if empty)
            try:
                self.disk_path.rmdir()
            except OSError:
                pass  # Directory not empty or in use
        
        logger.info("Memory manager cleanup completed")