"""Advanced error recovery and resilience patterns."""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from dataclasses import dataclass
from enum import Enum

from .exceptions import OrchestratorError, ToolExecutionError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RecoveryStrategy(Enum):
    """Strategies for error recovery."""
    FAIL_FAST = "fail_fast"
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class RecoveryPolicy:
    """Policy for error recovery."""
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    max_retries: int = 3
    backoff_factor: float = 1.5
    timeout_ms: int = 5000
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_time: int = 60
    fallback_function: Optional[Callable] = None
    
    
class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 reset_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.expected_exception = expected_exception
        
        self._failure_count = 0
        self._last_failure_time = None
        self._state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self._state == "OPEN":
            if time.time() - self._last_failure_time > self.reset_timeout:
                self._state = "HALF_OPEN"
                return False
            return True
        return False
    
    async def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        if self.is_open():
            raise OrchestratorError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Reset on successful call."""
        self._failure_count = 0
        self._state = "CLOSED"
    
    def _on_failure(self):
        """Handle failure."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._failure_count >= self.failure_threshold:
            self._state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self._failure_count} failures")


class ErrorRecoveryManager:
    """Manages error recovery across the orchestrator."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.recovery_policies: Dict[str, RecoveryPolicy] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_errors: Dict[str, Exception] = {}
        
    def register_policy(self, component: str, policy: RecoveryPolicy):
        """Register a recovery policy for a component."""
        self.recovery_policies[component] = policy
        
        if policy.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            self.circuit_breakers[component] = CircuitBreaker(
                failure_threshold=policy.circuit_breaker_threshold,
                reset_timeout=policy.circuit_breaker_reset_time
            )
        
        logger.info(f"Registered recovery policy for {component}: {policy.strategy.value}")
    
    async def execute_with_recovery(self,
                                  component: str,
                                  func: Callable,
                                  *args,
                                  **kwargs) -> Any:
        """Execute function with recovery policy."""
        policy = self.recovery_policies.get(component, RecoveryPolicy())
        
        if policy.strategy == RecoveryStrategy.FAIL_FAST:
            return await func(*args, **kwargs)
        
        elif policy.strategy == RecoveryStrategy.RETRY:
            return await self._execute_with_retry(component, func, policy, *args, **kwargs)
        
        elif policy.strategy == RecoveryStrategy.CIRCUIT_BREAKER:
            return await self._execute_with_circuit_breaker(component, func, *args, **kwargs)
        
        elif policy.strategy == RecoveryStrategy.FALLBACK:
            return await self._execute_with_fallback(component, func, policy, *args, **kwargs)
        
        elif policy.strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
            return await self._execute_with_degradation(component, func, policy, *args, **kwargs)
        
        else:
            return await func(*args, **kwargs)
    
    async def _execute_with_retry(self,
                                component: str,
                                func: Callable,
                                policy: RecoveryPolicy,
                                *args,
                                **kwargs) -> Any:
        """Execute with retry logic."""
        last_exception = None
        
        for attempt in range(policy.max_retries + 1):
            try:
                if attempt > 0:
                    delay = policy.backoff_factor ** (attempt - 1)
                    logger.debug(f"{component}: Retry attempt {attempt} after {delay}s delay")
                    await asyncio.sleep(delay)
                
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=policy.timeout_ms / 1000.0
                )
                
                # Reset error count on success
                self.error_counts[component] = 0
                return result
                
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"{component}: Timeout on attempt {attempt + 1}")
                
            except Exception as e:
                last_exception = e
                self.error_counts[component] = self.error_counts.get(component, 0) + 1
                self.last_errors[component] = e
                logger.warning(f"{component}: Error on attempt {attempt + 1}: {e}")
        
        # All retries exhausted
        logger.error(f"{component}: All {policy.max_retries} retries exhausted")
        raise ToolExecutionError(
            tool_name=component,
            message=f"All retries exhausted: {last_exception}",
            original_error=last_exception
        )
    
    async def _execute_with_circuit_breaker(self,
                                          component: str,
                                          func: Callable,
                                          *args,
                                          **kwargs) -> Any:
        """Execute with circuit breaker."""
        circuit_breaker = self.circuit_breakers[component]
        return await circuit_breaker.call(func, *args, **kwargs)
    
    async def _execute_with_fallback(self,
                                   component: str,
                                   func: Callable,
                                   policy: RecoveryPolicy,
                                   *args,
                                   **kwargs) -> Any:
        """Execute with fallback on failure."""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"{component}: Primary function failed, trying fallback: {e}")
            
            if policy.fallback_function:
                try:
                    return await policy.fallback_function(*args, **kwargs)
                except Exception as fallback_error:
                    logger.error(f"{component}: Fallback also failed: {fallback_error}")
                    raise ToolExecutionError(
                        tool_name=component,
                        message=f"Both primary and fallback failed: {e}, {fallback_error}",
                        original_error=e
                    )
            else:
                raise
    
    async def _execute_with_degradation(self,
                                      component: str,
                                      func: Callable,
                                      policy: RecoveryPolicy,
                                      *args,
                                      **kwargs) -> Any:
        """Execute with graceful degradation."""
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"{component}: Function failed, returning degraded result: {e}")
            
            # Return a simplified/cached result instead of failing
            return {
                "status": "degraded",
                "error": str(e),
                "timestamp": time.time(),
                "component": component
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all components."""
        status = {
            "overall": "healthy",
            "components": {},
            "circuit_breakers": {},
            "error_counts": dict(self.error_counts),
            "last_errors": {k: str(v) for k, v in self.last_errors.items()}
        }
        
        # Check circuit breakers
        for name, cb in self.circuit_breakers.items():
            cb_status = {
                "state": cb._state,
                "failure_count": cb._failure_count,
                "last_failure_time": cb._last_failure_time
            }
            status["circuit_breakers"][name] = cb_status
            
            if cb._state == "OPEN":
                status["overall"] = "degraded"
        
        # Check error counts
        for component, count in self.error_counts.items():
            if count > 10:  # Arbitrary threshold
                status["overall"] = "degraded"
                status["components"][component] = "unhealthy"
            elif count > 5:
                status["components"][component] = "degraded"
            else:
                status["components"][component] = "healthy"
        
        return status
    
    def reset_component(self, component: str):
        """Reset error tracking for a component."""
        self.error_counts.pop(component, None)
        self.last_errors.pop(component, None)
        
        if component in self.circuit_breakers:
            cb = self.circuit_breakers[component]
            cb._failure_count = 0
            cb._state = "CLOSED"
        
        logger.info(f"Reset error tracking for component: {component}")


# Global error recovery manager instance
error_recovery = ErrorRecoveryManager()