"""Simple rate limiter implementation for basic functionality."""

import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass

from .config import RateLimitConfig
from .exceptions import RateLimitError


@dataclass
class RateLimitState:
    """Simple rate limit state tracking."""
    count: int = 0
    window_start: float = 0.0
    

class SimpleRateLimiter:
    """Simple memory-based rate limiter."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._states: Dict[str, RateLimitState] = {}
        self._lock = asyncio.Lock()
    
    async def check_limit(self, service: str, identifier: str = "default") -> bool:
        """Check if request is within rate limits."""
        async with self._lock:
            key = f"{service}:{identifier}"
            now = time.time()
            
            # Get service configuration
            service_config = self.config.service_limits.get(service, {})
            limit = service_config.get('calls', self.config.global_max)
            window = service_config.get('window', 60)  # Default 60 seconds
            
            # Get or create state
            state = self._states.get(key)
            if not state:
                state = RateLimitState(count=0, window_start=now)
                self._states[key] = state
            
            # Reset window if expired
            if now - state.window_start >= window:
                state.count = 0
                state.window_start = now
            
            # Check if limit exceeded
            if state.count >= limit:
                retry_after = window - (now - state.window_start)
                raise RateLimitError(
                    service=service,
                    limit_type="calls",
                    retry_after=retry_after
                )
            
            # Increment count
            state.count += 1
            return True
    
    async def reset_limit(self, service: str, identifier: str = "default") -> None:
        """Reset rate limit for a service."""
        async with self._lock:
            key = f"{service}:{identifier}"
            if key in self._states:
                del self._states[key]
    
    def get_remaining(self, service: str, identifier: str = "default") -> int:
        """Get remaining requests in current window."""
        key = f"{service}:{identifier}"
        state = self._states.get(key)
        if not state:
            return self.config.global_max
        
        service_config = self.config.service_limits.get(service, {})
        limit = service_config.get('calls', self.config.global_max)
        
        return max(0, limit - state.count)