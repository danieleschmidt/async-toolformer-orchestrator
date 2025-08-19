"""
Advanced Error Recovery System - Generation 2 Implementation.

Provides sophisticated error recovery, circuit breaker patterns, and
self-healing capabilities for the Async Toolformer Orchestrator.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open" # Testing if the service has recovered


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAILOVER = "failover"


class ErrorCategory(Enum):
    """Error categories for recovery decisions."""
    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class RecoveryConfig:
    """Configuration for error recovery."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    jitter: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    recovery_timeout: float = 30.0
    enable_fallback: bool = True


@dataclass
class ErrorContext:
    """Context information about an error."""
    error: Exception
    category: ErrorCategory
    attempt: int
    operation_id: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAttempt:
    """Represents a recovery attempt."""
    strategy: RecoveryStrategy
    timestamp: float
    success: bool
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying half-open state
            success_threshold: Number of successes needed to close circuit in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._last_attempt_time: Optional[float] = None
        
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
        
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (allowing requests)."""
        return self._state == CircuitState.CLOSED
        
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self._state == CircuitState.OPEN
        
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN
        
    def record_success(self) -> None:
        """Record a successful operation."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._close_circuit()
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0  # Reset failure count on success
            
    def record_failure(self) -> None:
        """Record a failed operation."""
        current_time = time.time()
        self._last_failure_time = current_time
        
        if self._state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._open_circuit()
        elif self._state == CircuitState.HALF_OPEN:
            self._open_circuit()
            
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        current_time = time.time()
        
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            # Check if enough time has passed to try half-open
            if (self._last_failure_time and 
                current_time - self._last_failure_time >= self.recovery_timeout):
                self._half_open_circuit()
                return True
            return False
        elif self._state == CircuitState.HALF_OPEN:
            return True
            
        return False
        
    def _open_circuit(self) -> None:
        """Open the circuit."""
        logger.warning("Circuit breaker opened", circuit=self.name)
        self._state = CircuitState.OPEN
        self._success_count = 0
        
    def _close_circuit(self) -> None:
        """Close the circuit."""
        logger.info("Circuit breaker closed", circuit=self.name)
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        
    def _half_open_circuit(self) -> None:
        """Move circuit to half-open state."""
        logger.info("Circuit breaker half-open", circuit=self.name)
        self._state = CircuitState.HALF_OPEN
        self._success_count = 0


class RetryManager:
    """Manages retry logic with exponential backoff and jitter."""
    
    def __init__(self, config: RecoveryConfig):
        self.config = config
        
    async def retry_async(
        self,
        operation: Callable[..., Any],
        operation_name: str,
        error_categories: List[ErrorCategory] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Retry an async operation with exponential backoff."""
        error_categories = error_categories or [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT]
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self._calculate_delay(attempt)
                    logger.info(
                        "Retrying operation",
                        operation=operation_name,
                        attempt=attempt,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                    
                result = await operation(*args, **kwargs)
                
                if attempt > 0:
                    logger.info(
                        "Operation succeeded after retry",
                        operation=operation_name,
                        attempt=attempt,
                    )
                    
                return result
                
            except Exception as e:
                last_error = e
                error_category = self._categorize_error(e)
                
                logger.warning(
                    "Operation failed",
                    operation=operation_name,
                    attempt=attempt,
                    error=str(e),
                    error_category=error_category.value,
                )
                
                # Check if this error should be retried
                if error_category not in error_categories or attempt >= self.config.max_retries:
                    break
                    
        logger.error(
            "Operation failed after all retries",
            operation=operation_name,
            max_retries=self.config.max_retries,
            final_error=str(last_error),
        )
        
        raise last_error
        
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter."""
        delay = min(
            self.config.base_delay * (self.config.backoff_factor ** (attempt - 1)),
            self.config.max_delay,
        )
        
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add jitter: 50-100% of calculated delay
            
        return delay
        
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error for recovery decisions."""
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()
        
        if "timeout" in error_type or "timeout" in error_message:
            return ErrorCategory.TIMEOUT
        elif any(net_error in error_type for net_error in ["connection", "network", "socket"]):
            return ErrorCategory.NETWORK
        elif "rate" in error_message or "limit" in error_message:
            return ErrorCategory.RATE_LIMIT
        elif any(auth_error in error_type for auth_error in ["auth", "permission", "unauthorized"]):
            return ErrorCategory.AUTHENTICATION
        elif "validation" in error_type or "invalid" in error_message:
            return ErrorCategory.VALIDATION
        elif any(sys_error in error_type for sys_error in ["system", "os", "memory"]):
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.UNKNOWN


class FallbackManager:
    """Manages fallback operations when primary operations fail."""
    
    def __init__(self):
        self._fallback_handlers: Dict[str, Callable] = {}
        
    def register_fallback(self, operation_name: str, fallback_handler: Callable) -> None:
        """Register a fallback handler for an operation."""
        self._fallback_handlers[operation_name] = fallback_handler
        logger.info("Registered fallback handler", operation=operation_name)
        
    async def execute_fallback(
        self, operation_name: str, original_error: Exception, *args, **kwargs
    ) -> Any:
        """Execute fallback for a failed operation."""
        if operation_name not in self._fallback_handlers:
            logger.warning("No fallback handler available", operation=operation_name)
            raise original_error
            
        fallback_handler = self._fallback_handlers[operation_name]
        
        try:
            logger.info("Executing fallback", operation=operation_name)
            result = await fallback_handler(*args, **kwargs)
            logger.info("Fallback succeeded", operation=operation_name)
            return result
            
        except Exception as fallback_error:
            logger.error(
                "Fallback failed",
                operation=operation_name,
                original_error=str(original_error),
                fallback_error=str(fallback_error),
            )
            raise fallback_error


class AdvancedErrorRecovery:
    """Advanced error recovery system with multiple strategies."""
    
    def __init__(self, config: RecoveryConfig = None):
        self.config = config or RecoveryConfig()
        self.retry_manager = RetryManager(self.config)
        self.fallback_manager = FallbackManager()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._recovery_history: List[RecoveryAttempt] = []
        
    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=self.config.circuit_breaker_threshold,
                recovery_timeout=self.config.circuit_breaker_timeout,
            )
        return self._circuit_breakers[name]
        
    @asynccontextmanager
    async def protected_operation(
        self,
        operation_name: str,
        circuit_breaker_name: Optional[str] = None,
        fallback_name: Optional[str] = None,
    ):
        """Context manager for protected operations with error recovery."""
        circuit_breaker = None
        if circuit_breaker_name:
            circuit_breaker = self.get_circuit_breaker(circuit_breaker_name)
            
            if not circuit_breaker.can_execute():
                logger.warning(
                    "Circuit breaker is open, failing fast",
                    circuit=circuit_breaker_name,
                )
                raise Exception(f"Circuit breaker {circuit_breaker_name} is open")
                
        start_time = time.time()
        success = False
        
        try:
            yield
            success = True
            
            if circuit_breaker:
                circuit_breaker.record_success()
                
        except Exception as e:
            if circuit_breaker:
                circuit_breaker.record_failure()
                
            # Try fallback if available
            if fallback_name and self.config.enable_fallback:
                try:
                    await self.fallback_manager.execute_fallback(fallback_name, e)
                    success = True
                except Exception:
                    pass  # Fallback failed, continue with original exception
                    
            if not success:
                raise
                
        finally:
            duration = time.time() - start_time
            self._record_recovery_attempt(
                strategy=RecoveryStrategy.CIRCUIT_BREAKER if circuit_breaker else RecoveryStrategy.FALLBACK,
                success=success,
                duration=duration,
                metadata={
                    "operation": operation_name,
                    "circuit_breaker": circuit_breaker_name,
                    "fallback": fallback_name,
                },
            )
            
    async def execute_with_recovery(
        self,
        operation: Callable[..., Any],
        operation_name: str,
        strategies: List[RecoveryStrategy] = None,
        circuit_breaker_name: Optional[str] = None,
        fallback_name: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Execute operation with comprehensive error recovery."""
        strategies = strategies or [
            RecoveryStrategy.RETRY,
            RecoveryStrategy.CIRCUIT_BREAKER,
            RecoveryStrategy.FALLBACK,
        ]
        
        last_error = None
        
        # Try circuit breaker protection
        if RecoveryStrategy.CIRCUIT_BREAKER in strategies and circuit_breaker_name:
            async with self.protected_operation(operation_name, circuit_breaker_name, fallback_name):
                if RecoveryStrategy.RETRY in strategies:
                    return await self.retry_manager.retry_async(
                        operation, operation_name, *args, **kwargs
                    )
                else:
                    return await operation(*args, **kwargs)
        
        # Try retry strategy
        if RecoveryStrategy.RETRY in strategies:
            try:
                return await self.retry_manager.retry_async(
                    operation, operation_name, *args, **kwargs
                )
            except Exception as e:
                last_error = e
                
        # Try fallback strategy
        if RecoveryStrategy.FALLBACK in strategies and fallback_name:
            try:
                return await self.fallback_manager.execute_fallback(
                    fallback_name, last_error or Exception("Unknown error"), *args, **kwargs
                )
            except Exception as e:
                last_error = e
                
        # If all strategies failed, raise the last error
        if last_error:
            raise last_error
        else:
            # This shouldn't happen, but just in case
            raise Exception(f"All recovery strategies failed for operation {operation_name}")
            
    def _record_recovery_attempt(
        self,
        strategy: RecoveryStrategy,
        success: bool,
        duration: float,
        metadata: Dict[str, Any],
    ) -> None:
        """Record a recovery attempt for analysis."""
        attempt = RecoveryAttempt(
            strategy=strategy,
            timestamp=time.time(),
            success=success,
            duration=duration,
            metadata=metadata,
        )
        
        self._recovery_history.append(attempt)
        
        # Keep only recent attempts (last 1000)
        if len(self._recovery_history) > 1000:
            self._recovery_history = self._recovery_history[-1000:]
            
    def get_recovery_stats(self) -> Dict[str, Any]:
        """Get recovery statistics."""
        if not self._recovery_history:
            return {"total_attempts": 0}
            
        total_attempts = len(self._recovery_history)
        successful_attempts = sum(1 for attempt in self._recovery_history if attempt.success)
        
        strategy_stats = {}
        for strategy in RecoveryStrategy:
            strategy_attempts = [a for a in self._recovery_history if a.strategy == strategy]
            if strategy_attempts:
                strategy_stats[strategy.value] = {
                    "total": len(strategy_attempts),
                    "successful": sum(1 for a in strategy_attempts if a.success),
                    "success_rate": sum(1 for a in strategy_attempts if a.success) / len(strategy_attempts),
                    "avg_duration": sum(a.duration for a in strategy_attempts) / len(strategy_attempts),
                }
                
        circuit_breaker_stats = {}
        for name, cb in self._circuit_breakers.items():
            circuit_breaker_stats[name] = {
                "state": cb.state.value,
                "failure_count": cb._failure_count,
                "success_count": cb._success_count,
            }
            
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": successful_attempts / total_attempts,
            "strategy_stats": strategy_stats,
            "circuit_breaker_stats": circuit_breaker_stats,
            "recent_attempts": [
                {
                    "strategy": a.strategy.value,
                    "success": a.success,
                    "duration": a.duration,
                    "timestamp": a.timestamp,
                }
                for a in self._recovery_history[-10:]  # Last 10 attempts
            ],
        }


# Utility decorators
def with_circuit_breaker(circuit_name: str, recovery_system: AdvancedErrorRecovery):
    """Decorator to add circuit breaker protection to a function."""
    
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            async with recovery_system.protected_operation(
                operation_name=func.__name__,
                circuit_breaker_name=circuit_name,
            ):
                return await func(*args, **kwargs)
                
        return wrapper
        
    return decorator


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    error_categories: List[ErrorCategory] = None,
):
    """Decorator to add retry logic to a function."""
    
    def decorator(func: Callable) -> Callable:
        async def wrapper(*args, **kwargs):
            config = RecoveryConfig(
                max_retries=max_retries,
                base_delay=base_delay,
            )
            retry_manager = RetryManager(config)
            
            return await retry_manager.retry_async(
                func,
                func.__name__,
                error_categories or [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT],
                *args,
                **kwargs,
            )
            
        return wrapper
        
    return decorator


def create_advanced_error_recovery(config: RecoveryConfig = None) -> AdvancedErrorRecovery:
    """Create an advanced error recovery system with default configuration."""
    return AdvancedErrorRecovery(config or RecoveryConfig())