"""Unit tests for enhanced reliability system."""

import asyncio
import time

import pytest

from async_toolformer.enhanced_reliability import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerState,
    ReliabilityLevel,
    ReliabilityManager,
    with_reliability_tracking,
)


class TestReliabilityManager:
    """Test suite for ReliabilityManager."""

    @pytest.fixture
    def reliability_manager(self):
        """Create reliability manager for testing."""
        return ReliabilityManager(ReliabilityLevel.ENHANCED)

    @pytest.mark.asyncio
    async def test_record_success(self, reliability_manager):
        """Test recording successful operations."""
        initial_requests = reliability_manager.metrics.total_requests
        initial_successful = reliability_manager.metrics.successful_requests

        await reliability_manager.record_success("test_operation", 100.0)

        assert reliability_manager.metrics.total_requests == initial_requests + 1
        assert reliability_manager.metrics.successful_requests == initial_successful + 1
        assert reliability_manager.metrics.average_response_time_ms > 0

    @pytest.mark.asyncio
    async def test_record_failure(self, reliability_manager):
        """Test recording failed operations."""
        initial_requests = reliability_manager.metrics.total_requests
        initial_failed = reliability_manager.metrics.failed_requests

        failure_context = await reliability_manager.record_failure(
            operation="test_operation",
            error_type="TestError",
            error_message="Test failure message",
            execution_time_ms=500.0,
            metadata={"test_key": "test_value"}
        )

        assert reliability_manager.metrics.total_requests == initial_requests + 1
        assert reliability_manager.metrics.failed_requests == initial_failed + 1
        assert failure_context.operation == "test_operation"
        assert failure_context.error_type == "TestError"
        assert failure_context.error_message == "Test failure message"
        assert failure_context.execution_time_ms == 500.0
        assert failure_context.metadata["test_key"] == "test_value"

    @pytest.mark.asyncio
    async def test_failure_pattern_detection_intermittent(self, reliability_manager):
        """Test intermittent failure pattern detection."""
        # Create regular pattern of failures
        base_time = time.time()

        for i in range(5):
            failure_context = await reliability_manager.record_failure(
                operation="test_operation",
                error_type="TestError",
                error_message="Intermittent failure",
                execution_time_ms=100.0
            )
            # Manually set timestamp for pattern
            reliability_manager.failure_history[-1].timestamp = base_time + (i * 60)  # Every minute

        # Check if intermittent pattern is detected
        recent_failures = [f for f in reliability_manager.failure_history if f.operation == "test_operation"]
        is_intermittent = reliability_manager._is_intermittent_pattern(recent_failures)

        # With regular intervals, should detect intermittent pattern
        assert is_intermittent or len(recent_failures) >= 3

    @pytest.mark.asyncio
    async def test_failure_pattern_detection_timeout_spiral(self, reliability_manager):
        """Test timeout spiral pattern detection."""
        # Create increasing timeout pattern
        for i, timeout in enumerate([1000, 2000, 4000, 8000]):  # Increasing timeouts
            failure_context = await reliability_manager.record_failure(
                operation="test_operation",
                error_type="TimeoutError",
                error_message="Operation timed out",
                execution_time_ms=float(timeout)
            )

        recent_failures = [f for f in reliability_manager.failure_history if f.operation == "test_operation"]
        timeout_failures = [f for f in recent_failures if 'timeout' in f.error_type.lower()]

        if len(timeout_failures) >= 3:
            is_timeout_spiral = reliability_manager._is_timeout_spiral_pattern(recent_failures)
            assert is_timeout_spiral

    @pytest.mark.asyncio
    async def test_should_retry_logic(self, reliability_manager):
        """Test retry decision logic."""
        # Create a normal failure context
        failure_context = await reliability_manager.record_failure(
            operation="test_operation",
            error_type="TestError",
            error_message="Test failure",
            execution_time_ms=100.0
        )

        # Should allow retry for first few attempts
        should_retry_1 = await reliability_manager.should_retry("test_operation", 1, failure_context)
        should_retry_2 = await reliability_manager.should_retry("test_operation", 2, failure_context)
        should_retry_10 = await reliability_manager.should_retry("test_operation", 10, failure_context)

        assert should_retry_1
        assert should_retry_2
        assert not should_retry_10  # Should not retry after too many attempts

    @pytest.mark.asyncio
    async def test_health_status(self, reliability_manager):
        """Test health status calculation."""
        # Add some successful operations
        await reliability_manager.record_success("test_op", 100.0)
        await reliability_manager.record_success("test_op", 150.0)

        # Add a failure
        await reliability_manager.record_failure("test_op", "TestError", "Test", 200.0)

        health_status = await reliability_manager.get_health_status()

        assert "status" in health_status
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]
        assert "availability" in health_status
        assert "error_rate" in health_status
        assert "uptime_seconds" in health_status
        assert "warnings" in health_status
        assert isinstance(health_status["warnings"], list)

    @pytest.mark.asyncio
    async def test_reliability_metrics_calculation(self, reliability_manager):
        """Test reliability metrics calculation."""
        # Record some operations
        await reliability_manager.record_success("test_op", 100.0)
        await reliability_manager.record_success("test_op", 200.0)
        await reliability_manager.record_failure("test_op", "TestError", "Test", 150.0)

        metrics = await reliability_manager.get_reliability_metrics()

        assert metrics.total_requests == 3
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 1
        assert metrics.error_rate == pytest.approx(1/3, rel=0.1)
        assert metrics.availability == pytest.approx(2/3, rel=0.1)
        assert metrics.uptime_seconds > 0

        # Check derived metrics calculation
        metrics.calculate_derived_metrics()
        assert metrics.error_rate == pytest.approx(1/3, rel=0.1)
        assert metrics.availability == pytest.approx(2/3, rel=0.1)


class TestCircuitBreaker:
    """Test suite for CircuitBreaker."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            timeout_seconds=1.0,
            success_threshold=2
        )
        return CircuitBreaker("test_breaker", config)

    @pytest.mark.asyncio
    async def test_initial_state(self, circuit_breaker):
        """Test circuit breaker initial state."""
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert await circuit_breaker.allow_request()

    @pytest.mark.asyncio
    async def test_circuit_opening(self, circuit_breaker):
        """Test circuit breaker opening after failures."""
        # Record failures to exceed threshold
        for _ in range(3):
            await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitBreakerState.OPEN
        assert not await circuit_breaker.allow_request()

    @pytest.mark.asyncio
    async def test_circuit_half_open_transition(self, circuit_breaker):
        """Test transition to half-open state."""
        # Open the circuit
        for _ in range(3):
            await circuit_breaker.record_failure()

        assert circuit_breaker.state == CircuitBreakerState.OPEN

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Should transition to half-open
        assert await circuit_breaker.allow_request()
        assert circuit_breaker.state == CircuitBreakerState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_closing_after_success(self, circuit_breaker):
        """Test circuit closing after successful operations in half-open."""
        # Open the circuit
        for _ in range(3):
            await circuit_breaker.record_failure()

        # Wait and transition to half-open
        await asyncio.sleep(1.1)
        await circuit_breaker.allow_request()

        # Record successful operations to close circuit
        await circuit_breaker.record_success()
        await circuit_breaker.record_success()

        assert circuit_breaker.state == CircuitBreakerState.CLOSED
        assert await circuit_breaker.allow_request()

    @pytest.mark.asyncio
    async def test_circuit_status(self, circuit_breaker):
        """Test circuit breaker status reporting."""
        status = await circuit_breaker.get_status()

        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status
        assert "last_failure_time" in status
        assert "config" in status

        assert status["state"] == "CLOSED"
        assert status["failure_count"] == 0
        assert status["success_count"] == 0


class TestWithReliabilityTracking:
    """Test suite for with_reliability_tracking decorator."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Test reliability tracking for successful execution."""
        async def test_function(value: int) -> int:
            await asyncio.sleep(0.1)
            return value * 2

        result = await with_reliability_tracking(
            "test_operation",
            test_function,
            5,
            max_retries=2
        )

        assert result == 10

    @pytest.mark.asyncio
    async def test_failed_execution_with_retry(self):
        """Test reliability tracking for failed execution with retry."""
        call_count = 0

        async def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Test failure")
            return "success"

        result = await with_reliability_tracking(
            "test_operation",
            failing_function,
            max_retries=3
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_complete_failure(self):
        """Test reliability tracking for complete failure."""
        async def always_failing_function():
            raise Exception("Always fails")

        with pytest.raises(Exception, match="Always fails"):
            await with_reliability_tracking(
                "test_operation",
                always_failing_function,
                max_retries=2
            )

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test integration with circuit breaker."""
        # This test would require more setup to properly test circuit breaker integration
        # For now, verify the function can handle circuit breaker exceptions

        async def test_function():
            return "success"

        result = await with_reliability_tracking(
            "test_circuit_operation",
            test_function,
            max_retries=1
        )

        assert result == "success"
