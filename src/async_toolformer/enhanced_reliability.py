"""Enhanced reliability features for Generation 2 robustness."""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class ReliabilityLevel(Enum):
    """Reliability levels for different operations."""
    BASIC = auto()
    ENHANCED = auto()
    MAXIMUM = auto()


class FailurePattern(Enum):
    """Common failure patterns detected by the system."""
    INTERMITTENT = auto()
    CASCADING = auto()
    TIMEOUT_SPIRAL = auto()
    RESOURCE_EXHAUSTION = auto()
    DEPENDENCY_FAILURE = auto()


@dataclass
class ReliabilityMetrics:
    """Metrics for tracking system reliability."""

    uptime_seconds: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    error_rate: float = 0.0
    availability: float = 1.0
    mtbf_seconds: float = 0.0  # Mean Time Between Failures
    mttr_seconds: float = 0.0  # Mean Time To Recovery
    failure_patterns: dict[FailurePattern, int] = field(default_factory=dict)

    def calculate_derived_metrics(self) -> None:
        """Calculate derived reliability metrics."""
        if self.total_requests > 0:
            self.error_rate = self.failed_requests / self.total_requests
            self.availability = self.successful_requests / self.total_requests

        # Calculate MTBF if we have failure data
        if self.failed_requests > 0 and self.uptime_seconds > 0:
            self.mtbf_seconds = self.uptime_seconds / self.failed_requests


@dataclass
class FailureContext:
    """Context information about a failure."""

    operation: str
    error_type: str
    error_message: str
    timestamp: float
    execution_time_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    retry_attempt: int = 0
    failure_pattern: FailurePattern | None = None


class ReliabilityManager:
    """Manager for enhanced system reliability."""

    def __init__(self, reliability_level: ReliabilityLevel = ReliabilityLevel.ENHANCED):
        self.reliability_level = reliability_level
        self.metrics = ReliabilityMetrics()
        self.failure_history: list[FailureContext] = []
        self.circuit_breakers: dict[str, CircuitBreaker] = {}
        self.start_time = time.time()
        self.last_failure_time = 0.0

        # Pattern detection
        self.failure_patterns: dict[str, list[float]] = {}
        self.pattern_detection_window = 300  # 5 minutes

    async def record_success(self, operation: str, execution_time_ms: float) -> None:
        """Record a successful operation."""
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1

        # Update average response time
        current_avg = self.metrics.average_response_time_ms
        total_requests = self.metrics.total_requests
        self.metrics.average_response_time_ms = (
            (current_avg * (total_requests - 1) + execution_time_ms) / total_requests
        )

        # Update circuit breaker
        if operation in self.circuit_breakers:
            await self.circuit_breakers[operation].record_success()

    async def record_failure(
        self,
        operation: str,
        error_type: str,
        error_message: str,
        execution_time_ms: float,
        metadata: dict[str, Any] | None = None,
        retry_attempt: int = 0
    ) -> FailureContext:
        """Record a failed operation."""

        failure_context = FailureContext(
            operation=operation,
            error_type=error_type,
            error_message=error_message,
            timestamp=time.time(),
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
            retry_attempt=retry_attempt
        )

        # Detect failure patterns
        pattern = await self._detect_failure_pattern(failure_context)
        failure_context.failure_pattern = pattern

        self.failure_history.append(failure_context)
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        self.last_failure_time = time.time()

        # Update circuit breaker
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreaker(operation)
        await self.circuit_breakers[operation].record_failure()

        # Trim old failure history
        self._trim_failure_history()

        logger.warning(
            f"Operation failed: {operation}",
            extra={
                "error_type": error_type,
                "execution_time_ms": execution_time_ms,
                "failure_pattern": pattern.name if pattern else None,
                "retry_attempt": retry_attempt
            }
        )

        return failure_context

    async def _detect_failure_pattern(self, failure_context: FailureContext) -> FailurePattern | None:
        """Detect failure patterns from recent history."""

        operation = failure_context.operation
        current_time = time.time()

        # Get recent failures for this operation
        recent_failures = [
            f for f in self.failure_history
            if f.operation == operation and
            current_time - f.timestamp < self.pattern_detection_window
        ]

        if len(recent_failures) < 2:
            return None

        # Intermittent failures - alternating success/failure
        if self._is_intermittent_pattern(recent_failures):
            return FailurePattern.INTERMITTENT

        # Timeout spiral - increasing timeouts
        if self._is_timeout_spiral_pattern(recent_failures):
            return FailurePattern.TIMEOUT_SPIRAL

        # Cascading failures - multiple operations failing together
        if self._is_cascading_pattern(recent_failures, current_time):
            return FailurePattern.CASCADING

        # Resource exhaustion - specific error patterns
        if self._is_resource_exhaustion_pattern(recent_failures):
            return FailurePattern.RESOURCE_EXHAUSTION

        return None

    def _is_intermittent_pattern(self, failures: list[FailureContext]) -> bool:
        """Check if failures show intermittent pattern."""
        if len(failures) < 3:
            return False

        # Look for regular intervals between failures
        intervals = []
        for i in range(1, len(failures)):
            interval = failures[i].timestamp - failures[i-1].timestamp
            intervals.append(interval)

        # Check if intervals are relatively consistent
        if len(intervals) < 2:
            return False

        avg_interval = sum(intervals) / len(intervals)
        variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)

        # Low variance indicates regular intermittent pattern
        return variance < (avg_interval * 0.5) ** 2

    def _is_timeout_spiral_pattern(self, failures: list[FailureContext]) -> bool:
        """Check if failures show increasing timeout pattern."""
        timeout_failures = [
            f for f in failures
            if 'timeout' in f.error_type.lower() or 'timeout' in f.error_message.lower()
        ]

        if len(timeout_failures) < 3:
            return False

        # Check if execution times are increasing
        times = [f.execution_time_ms for f in timeout_failures[-3:]]
        return all(times[i] < times[i+1] for i in range(len(times)-1))

    def _is_cascading_pattern(self, failures: list[FailureContext], current_time: float) -> bool:
        """Check if multiple operations are failing together."""
        # Get all recent failures across operations
        all_recent_failures = [
            f for f in self.failure_history
            if current_time - f.timestamp < 60  # Last minute
        ]

        if len(all_recent_failures) < 3:
            return False

        # Count unique operations failing
        failing_operations = {f.operation for f in all_recent_failures}

        # Cascading if multiple operations failing simultaneously
        return len(failing_operations) >= 3

    def _is_resource_exhaustion_pattern(self, failures: list[FailureContext]) -> bool:
        """Check if failures indicate resource exhaustion."""
        resource_keywords = ['memory', 'connection', 'pool', 'limit', 'quota', 'throttle']

        for failure in failures[-3:]:  # Check last 3 failures
            error_text = (failure.error_type + " " + failure.error_message).lower()
            if any(keyword in error_text for keyword in resource_keywords):
                return True

        return False

    def _trim_failure_history(self) -> None:
        """Trim old entries from failure history."""
        current_time = time.time()
        cutoff_time = current_time - (self.pattern_detection_window * 2)  # Keep 2x window

        self.failure_history = [
            f for f in self.failure_history
            if f.timestamp > cutoff_time
        ]

    async def get_reliability_metrics(self) -> ReliabilityMetrics:
        """Get current reliability metrics."""
        # Update uptime
        self.metrics.uptime_seconds = time.time() - self.start_time

        # Calculate MTTR if we have recent failures
        if self.failure_history:
            recent_failures = [
                f for f in self.failure_history
                if time.time() - f.timestamp < 3600  # Last hour
            ]
            if recent_failures:
                recovery_times = []
                for _i, failure in enumerate(recent_failures[:-1]):
                    next_success_time = failure.timestamp + failure.execution_time_ms / 1000
                    recovery_times.append(next_success_time - failure.timestamp)

                if recovery_times:
                    self.metrics.mttr_seconds = sum(recovery_times) / len(recovery_times)

        # Calculate pattern frequencies
        pattern_counts = {}
        for failure in self.failure_history[-100:]:  # Last 100 failures
            if failure.failure_pattern:
                pattern_counts[failure.failure_pattern] = pattern_counts.get(failure.failure_pattern, 0) + 1

        self.metrics.failure_patterns = pattern_counts

        # Calculate derived metrics
        self.metrics.calculate_derived_metrics()

        return self.metrics

    async def should_retry(self, operation: str, attempt: int, failure_context: FailureContext) -> bool:
        """Determine if operation should be retried based on reliability analysis."""

        # Check circuit breaker
        if operation in self.circuit_breakers:
            if not await self.circuit_breakers[operation].allow_request():
                return False

        # Check failure pattern
        if failure_context.failure_pattern:
            if failure_context.failure_pattern == FailurePattern.CASCADING:
                # Don't retry during cascading failures
                return False
            elif failure_context.failure_pattern == FailurePattern.RESOURCE_EXHAUSTION:
                # Wait longer before retry
                await asyncio.sleep(min(attempt * 2, 30))
            elif failure_context.failure_pattern == FailurePattern.TIMEOUT_SPIRAL:
                # Exponential backoff for timeout spirals
                await asyncio.sleep(min(2 ** attempt, 60))

        # Standard retry logic based on reliability level
        max_attempts = {
            ReliabilityLevel.BASIC: 2,
            ReliabilityLevel.ENHANCED: 3,
            ReliabilityLevel.MAXIMUM: 5
        }

        return attempt < max_attempts[self.reliability_level]

    async def get_health_status(self) -> dict[str, Any]:
        """Get system health status based on reliability metrics."""
        metrics = await self.get_reliability_metrics()

        # Determine health status
        if metrics.availability >= 0.99:
            status = "healthy"
        elif metrics.availability >= 0.95:
            status = "degraded"
        else:
            status = "unhealthy"

        # Check for concerning patterns
        warnings = []
        if FailurePattern.CASCADING in metrics.failure_patterns:
            warnings.append("Cascading failures detected")
        if FailurePattern.TIMEOUT_SPIRAL in metrics.failure_patterns:
            warnings.append("Timeout spiral pattern detected")
        if metrics.error_rate > 0.05:  # 5% error rate
            warnings.append("High error rate")

        return {
            "status": status,
            "availability": metrics.availability,
            "error_rate": metrics.error_rate,
            "uptime_seconds": metrics.uptime_seconds,
            "mtbf_seconds": metrics.mtbf_seconds,
            "mttr_seconds": metrics.mttr_seconds,
            "warnings": warnings,
            "circuit_breakers": {
                name: await cb.get_status()
                for name, cb in self.circuit_breakers.items()
            }
        }


class CircuitBreakerState(Enum):
    """States for circuit breaker pattern."""
    CLOSED = auto()    # Normal operation
    OPEN = auto()      # Failing fast
    HALF_OPEN = auto() # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5      # Failures before opening
    timeout_seconds: float = 60.0   # Time before trying half-open
    success_threshold: int = 3      # Successes needed to close
    window_seconds: float = 300.0   # Time window for failure counting


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, name: str, config: CircuitBreakerConfig | None = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.failure_times: list[float] = []

    async def allow_request(self) -> bool:
        """Check if request should be allowed through."""
        current_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            # Check if timeout has passed
            if current_time - self.last_failure_time > self.config.timeout_seconds:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        else:  # HALF_OPEN
            return True

    async def record_success(self) -> None:
        """Record successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.failure_times.clear()

    async def record_failure(self) -> None:
        """Record failed operation."""
        current_time = time.time()
        self.last_failure_time = current_time
        self.failure_times.append(current_time)

        # Remove old failures outside window
        cutoff_time = current_time - self.config.window_seconds
        self.failure_times = [t for t in self.failure_times if t > cutoff_time]

        self.failure_count = len(self.failure_times)

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failed while testing, go back to open
            self.state = CircuitBreakerState.OPEN
        elif self.failure_count >= self.config.failure_threshold:
            # Too many failures, open circuit
            self.state = CircuitBreakerState.OPEN

    async def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status."""
        return {
            "state": self.state.name,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "timeout_seconds": self.config.timeout_seconds,
                "success_threshold": self.config.success_threshold,
            }
        }


# Global reliability manager instance
reliability_manager = ReliabilityManager()


async def with_reliability_tracking(
    operation: str,
    func: Callable,
    *args,
    max_retries: int = 3,
    **kwargs
) -> Any:
    """Execute function with reliability tracking and retry logic."""

    for attempt in range(max_retries + 1):
        start_time = time.time()

        try:
            # Check if request should proceed
            if not await reliability_manager.circuit_breakers.get(operation, CircuitBreaker("dummy")).allow_request():
                raise Exception(f"Circuit breaker open for {operation}")

            result = await func(*args, **kwargs)
            execution_time_ms = (time.time() - start_time) * 1000

            await reliability_manager.record_success(operation, execution_time_ms)
            return result

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000

            failure_context = await reliability_manager.record_failure(
                operation=operation,
                error_type=type(e).__name__,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
                retry_attempt=attempt
            )

            # Check if we should retry
            if attempt < max_retries:
                should_retry = await reliability_manager.should_retry(
                    operation, attempt + 1, failure_context
                )
                if should_retry:
                    continue

            # No more retries, re-raise the exception
            raise e
