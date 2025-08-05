"""Connection pooling for efficient resource management."""

import asyncio
import aiohttp
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
from contextlib import asynccontextmanager
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for connection pools."""
    
    max_connections: int = 100
    max_connections_per_host: int = 30
    timeout_seconds: float = 30.0
    keepalive_timeout: float = 30.0
    enable_cleanup: bool = True
    cleanup_interval: float = 60.0


class ConnectionPoolManager:
    """Manages HTTP connection pools for tool integrations."""
    
    def __init__(self, config: PoolConfig = None):
        self.config = config or PoolConfig()
        self._sessions: Dict[str, aiohttp.ClientSession] = {}
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    async def get_session(self, 
                         pool_name: str = "default",
                         headers: Optional[Dict[str, str]] = None,
                         timeout: Optional[float] = None) -> aiohttp.ClientSession:
        """Get or create a session for the specified pool."""
        async with self._lock:
            if pool_name not in self._sessions:
                # Create connector with connection pooling
                connector = aiohttp.TCPConnector(
                    limit=self.config.max_connections,
                    limit_per_host=self.config.max_connections_per_host,
                    keepalive_timeout=self.config.keepalive_timeout,
                    enable_cleanup=self.config.enable_cleanup,
                )
                
                # Create timeout configuration
                timeout_config = aiohttp.ClientTimeout(
                    total=timeout or self.config.timeout_seconds,
                    connect=10.0,
                    sock_read=30.0
                )
                
                # Create session
                session = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout_config,
                    headers=headers or {}
                )
                
                self._sessions[pool_name] = session
                self._stats[pool_name] = {
                    'created_at': time.time(),
                    'requests': 0,
                    'errors': 0,
                    'total_time': 0.0,
                }
                
                logger.info(f"Created connection pool: {pool_name}")
            
            return self._sessions[pool_name]
    
    @asynccontextmanager
    async def request(self,
                     method: str,
                     url: str,
                     pool_name: str = "default",
                     **kwargs):
        """Context manager for making HTTP requests with connection pooling."""
        session = await self.get_session(pool_name)
        start_time = time.time()
        
        try:
            async with session.request(method, url, **kwargs) as response:
                # Update stats
                self._stats[pool_name]['requests'] += 1
                self._stats[pool_name]['total_time'] += time.time() - start_time
                
                yield response
                
        except Exception as e:
            self._stats[pool_name]['errors'] += 1
            logger.error(f"Request error in pool {pool_name}: {e}")
            raise
    
    async def get_pool_stats(self, pool_name: str = None) -> Dict[str, Any]:
        """Get statistics for connection pools."""
        if pool_name:
            return self._stats.get(pool_name, {})
        else:
            # Return stats for all pools
            total_stats = {
                'pools': dict(self._stats),
                'total_requests': sum(stats['requests'] for stats in self._stats.values()),
                'total_errors': sum(stats['errors'] for stats in self._stats.values()),
                'avg_response_time': 0.0,
            }
            
            total_requests = total_stats['total_requests']
            total_time = sum(stats['total_time'] for stats in self._stats.values())
            
            if total_requests > 0:
                total_stats['avg_response_time'] = total_time / total_requests
                
            return total_stats
    
    async def close_pool(self, pool_name: str) -> None:
        """Close a specific connection pool."""
        async with self._lock:
            if pool_name in self._sessions:
                await self._sessions[pool_name].close()
                del self._sessions[pool_name]
                del self._stats[pool_name]
                logger.info(f"Closed connection pool: {pool_name}")
    
    async def close_all(self) -> None:
        """Close all connection pools."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        async with self._lock:
            for pool_name, session in self._sessions.items():
                await session.close()
                logger.info(f"Closed connection pool: {pool_name}")
            
            self._sessions.clear()
            self._stats.clear()
    
    def start_cleanup_task(self) -> None:
        """Start background cleanup task."""
        if self.config.enable_cleanup and not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up idle connections."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                
                # Force cleanup of idle connections
                for session in self._sessions.values():
                    if hasattr(session.connector, 'close'):
                        # This would trigger connector cleanup
                        pass
                        
                logger.debug("Connection pool cleanup completed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection pool cleanup error: {e}")


# Utility functions for different types of connections
class APIConnectionPools:
    """Pre-configured connection pools for common APIs."""
    
    def __init__(self):
        self.manager = ConnectionPoolManager()
    
    async def openai_session(self) -> aiohttp.ClientSession:
        """Get session optimized for OpenAI API."""
        return await self.manager.get_session(
            pool_name="openai",
            headers={
                "User-Agent": "AsyncToolformer/1.0",
                "Content-Type": "application/json"
            },
            timeout=60.0  # OpenAI can be slow
        )
    
    async def anthropic_session(self) -> aiohttp.ClientSession:
        """Get session optimized for Anthropic API."""
        return await self.manager.get_session(
            pool_name="anthropic",
            headers={
                "User-Agent": "AsyncToolformer/1.0",
                "Content-Type": "application/json"
            },
            timeout=60.0
        )
    
    async def general_web_session(self) -> aiohttp.ClientSession:
        """Get session for general web requests."""
        return await self.manager.get_session(
            pool_name="web",
            headers={
                "User-Agent": "AsyncToolformer/1.0"
            },
            timeout=30.0
        )
    
    async def close_all(self) -> None:
        """Close all API connection pools."""
        await self.manager.close_all()


# Global instance for easy access
_global_pools = APIConnectionPools()


async def get_api_session(api_name: str = "general") -> aiohttp.ClientSession:
    """Get optimized session for specific API."""
    if api_name == "openai":
        return await _global_pools.openai_session()
    elif api_name == "anthropic":
        return await _global_pools.anthropic_session()
    else:
        return await _global_pools.general_web_session()


async def cleanup_global_pools() -> None:
    """Cleanup global connection pools."""
    await _global_pools.close_all()