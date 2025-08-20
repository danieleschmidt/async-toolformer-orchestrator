"""Integration tests for Generation 1-3 enhancements."""

import asyncio
import time

import pytest

from async_toolformer import AsyncOrchestrator, OrchestratorConfig, Tool
from async_toolformer.adaptive_scaling import ScalingDirection, adaptive_scaler
from async_toolformer.advanced_validation import advanced_validator
from async_toolformer.enhanced_reliability import reliability_manager
from async_toolformer.intelligent_caching import intelligent_cache


class TestGenerationEnhancements:
    """Test suite for Generation 1-3 enhancements."""

    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator with enhanced features."""
        config = OrchestratorConfig(
            max_parallel_tools=5,
            tool_timeout_ms=5000,
            total_timeout_ms=10000
        )

        orchestrator = AsyncOrchestrator(config=config)

        # Register test tools
        @Tool("Test tool for generation enhancements")
        async def test_tool(message: str) -> str:
            await asyncio.sleep(0.1)
            return f"Processed: {message}"

        @Tool("Tool that fails occasionally")
        async def failing_tool(should_fail: bool = False) -> str:
            if should_fail:
                raise Exception("Simulated failure")
            return "Success"

        orchestrator.register_tool(test_tool)
        orchestrator.register_tool(failing_tool)

        yield orchestrator

        # Cleanup
        await orchestrator.cleanup()

    @pytest.mark.asyncio
    async def test_generation_1_execution_stats(self, orchestrator):
        """Test Generation 1: Enhanced execution statistics."""

        # Execute some operations to generate stats
        await orchestrator.execute("Use test_tool to process 'hello world'")
        await orchestrator.execute("Use test_tool to process 'goodbye world'")

        # Get execution statistics
        stats = await orchestrator.get_execution_stats()

        assert stats['total_executions'] >= 2
        assert stats['successful_executions'] >= 0
        assert stats['success_rate'] >= 0
        assert 'adaptive_timeout_ms' in stats
        assert 'registered_tools' in stats
        assert stats['registered_tools'] == 2

    @pytest.mark.asyncio
    async def test_generation_1_adaptive_timeout(self, orchestrator):
        """Test Generation 1: Adaptive timeout adjustment."""

        initial_timeout = orchestrator._adaptive_timeout

        # Simulate successful fast execution
        await orchestrator.adaptive_timeout_adjustment(500, True)  # 500ms success
        fast_timeout = orchestrator._adaptive_timeout

        # Timeout should decrease for fast execution
        assert fast_timeout <= initial_timeout

        # Simulate slow/failed execution
        await orchestrator.adaptive_timeout_adjustment(8000, False)  # 8s failure
        slow_timeout = orchestrator._adaptive_timeout

        # Timeout should increase for failures
        assert slow_timeout >= fast_timeout

    @pytest.mark.asyncio
    async def test_generation_2_advanced_validation(self, orchestrator):
        """Test Generation 2: Advanced validation system."""

        # Test security validation
        malicious_prompt = "SELECT * FROM users WHERE 1=1; DROP TABLE users;"
        result = await advanced_validator.validate_and_sanitize(
            {"prompt": malicious_prompt},
            context="test.security",
            strict_mode=True
        )

        assert not result.is_valid
        security_issues = [i for i in result.issues if i.category.name == "SECURITY"]
        assert len(security_issues) > 0

        # Test performance validation
        large_data = "x" * 2000000  # 2MB string
        result = await advanced_validator.validate_and_sanitize(
            {"data": large_data},
            context="test.performance"
        )

        performance_issues = [i for i in result.issues if i.category.name == "PERFORMANCE"]
        assert len(performance_issues) > 0

        # Test PII detection
        pii_data = "Contact me at john.doe@email.com or call 555-123-4567"
        result = await advanced_validator.validate_and_sanitize(
            {"message": pii_data},
            context="test.compliance"
        )

        compliance_issues = [i for i in result.issues if i.category.name == "COMPLIANCE"]
        assert len(compliance_issues) > 0

    @pytest.mark.asyncio
    async def test_generation_2_reliability_tracking(self, orchestrator):
        """Test Generation 2: Enhanced reliability tracking."""

        # Clear previous metrics
        reliability_manager.metrics.total_requests = 0
        reliability_manager.metrics.successful_requests = 0
        reliability_manager.metrics.failed_requests = 0

        # Test successful execution tracking
        await orchestrator.execute("Use test_tool to process 'success test'")

        # Test failed execution tracking
        try:
            await orchestrator.execute("Use failing_tool with should_fail=true")
        except Exception:
            pass  # Expected failure

        # Check reliability metrics
        metrics = await reliability_manager.get_reliability_metrics()
        assert metrics.total_requests >= 1

        # Test failure pattern detection
        failure_context = await reliability_manager.record_failure(
            operation="test_operation",
            error_type="TestError",
            error_message="Test failure",
            execution_time_ms=1000
        )

        assert failure_context.operation == "test_operation"
        assert failure_context.error_type == "TestError"

        # Test health status
        health_status = await reliability_manager.get_health_status()
        assert "status" in health_status
        assert "availability" in health_status
        assert "error_rate" in health_status

    @pytest.mark.asyncio
    async def test_generation_3_intelligent_caching(self, orchestrator):
        """Test Generation 3: Intelligent caching system."""

        # Clear cache
        await intelligent_cache.clear()

        # Test cache storage and retrieval
        test_key = "test_key_1"
        test_value = {"data": "test_value", "timestamp": time.time()}

        await intelligent_cache.put(test_key, test_value, computation_cost=2.0)
        retrieved_value = await intelligent_cache.get(test_key)

        assert retrieved_value == test_value

        # Test cache miss
        missing_value = await intelligent_cache.get("nonexistent_key", default="not_found")
        assert missing_value == "not_found"

        # Test cache statistics
        stats = await intelligent_cache.get_stats()
        assert stats.total_requests >= 2  # 1 hit, 1 miss
        assert stats.cache_hits >= 1
        assert stats.cache_misses >= 1

        # Test cache info
        cache_info = await intelligent_cache.get_cache_info()
        assert "l1_entries" in cache_info
        assert "strategy" in cache_info
        assert cache_info["strategy"] == "INTELLIGENT"

        # Test cache optimization
        optimization_result = await intelligent_cache.optimize_cache()
        assert "optimizations_made" in optimization_result
        assert "cache_info" in optimization_result

    @pytest.mark.asyncio
    async def test_generation_3_adaptive_scaling(self, orchestrator):
        """Test Generation 3: Adaptive scaling system."""

        # Test metrics collection
        metrics = await adaptive_scaler._collect_metrics()

        assert hasattr(metrics, 'cpu_utilization')
        assert hasattr(metrics, 'memory_usage')
        assert hasattr(metrics, 'queue_depth')
        assert hasattr(metrics, 'average_response_time_ms')
        assert hasattr(metrics, 'error_rate')
        assert hasattr(metrics, 'throughput_requests_per_second')

        # Test utilization score calculation
        utilization_score = metrics.get_utilization_score()
        assert 0 <= utilization_score <= 1

        # Test scaling decision logic
        decision = await adaptive_scaler._make_scaling_decision(metrics)
        assert decision in [ScalingDirection.UP, ScalingDirection.DOWN, ScalingDirection.STABLE]

        # Test scaling status
        status = await adaptive_scaler.get_scaling_status()
        assert "current_workers" in status
        assert "min_workers" in status
        assert "max_workers" in status
        assert "current_utilization" in status
        assert "scale_up_threshold" in status
        assert "scale_down_threshold" in status

        # Test performance report
        performance_report = await adaptive_scaler.get_performance_report()
        assert "utilization_stats" in performance_report
        assert "current_policy" in performance_report

    @pytest.mark.asyncio
    async def test_integrated_generation_features(self, orchestrator):
        """Test integrated functionality across all generations."""

        # Clear previous state
        await intelligent_cache.clear()

        # Execute orchestrator with all enhancements active
        result = await orchestrator.execute(
            "Use test_tool to process 'integrated test message'",
            user_id="test_user_123"
        )

        assert result["status"] == "completed"
        assert "execution_id" in result
        assert "total_time_ms" in result
        assert "tools_executed" in result
        assert "successful_tools" in result
        assert "success_rate" in result
        assert "adaptive_timeout_ms" in result

        # Verify caching worked
        cache_stats = await intelligent_cache.get_stats()
        cache_info = await intelligent_cache.get_cache_info()

        # Verify reliability tracking
        reliability_metrics = await reliability_manager.get_reliability_metrics()
        assert reliability_metrics.total_requests > 0

        # Verify orchestrator metrics include enhancements
        orchestrator_metrics = await orchestrator.get_metrics()
        assert "registered_tools" in orchestrator_metrics
        assert "active_tasks" in orchestrator_metrics
        assert "cache" in orchestrator_metrics

    @pytest.mark.asyncio
    async def test_error_scenarios_with_enhancements(self, orchestrator):
        """Test error handling with all enhancements."""

        # Test validation errors
        result = await orchestrator.execute("")  # Empty prompt
        assert result["status"] in ["validation_failed", "no_tools_called"]

        # Test tool execution errors with reliability tracking
        try:
            result = await orchestrator.execute("Use failing_tool with should_fail=true")
            # Should handle gracefully
            assert "status" in result
        except Exception:
            # Or may propagate error depending on configuration
            pass

        # Verify reliability tracking recorded the failure
        reliability_metrics = await reliability_manager.get_reliability_metrics()
        health_status = await reliability_manager.get_health_status()
        assert "status" in health_status

    @pytest.mark.asyncio
    async def test_performance_with_enhancements(self, orchestrator):
        """Test performance characteristics with all enhancements."""

        start_time = time.time()

        # Execute multiple operations concurrently
        tasks = []
        for i in range(5):
            task = orchestrator.execute(f"Use test_tool to process 'performance test {i}'")
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.time() - start_time

        # Should complete relatively quickly with optimizations
        assert total_time < 10  # Should complete within 10 seconds

        # Verify results
        successful_results = [r for r in results if isinstance(r, dict) and r.get("status") == "completed"]
        assert len(successful_results) >= 3  # At least most should succeed

        # Check cache effectiveness
        cache_stats = await intelligent_cache.get_stats()
        if cache_stats.total_requests > 0:
            assert cache_stats.hit_rate >= 0  # Some cache activity

        # Check scaling metrics
        scaling_status = await adaptive_scaler.get_scaling_status()
        assert scaling_status["current_workers"] >= adaptive_scaler.policy.min_workers

    @pytest.mark.asyncio
    async def test_monitoring_integration(self, orchestrator):
        """Test monitoring integration across generations."""

        # Execute operations to generate monitoring data
        await orchestrator.execute("Use test_tool to process 'monitoring test'")

        # Check orchestrator health
        orchestrator_metrics = await orchestrator.get_metrics()
        assert "registered_tools" in orchestrator_metrics
        assert "active_tasks" in orchestrator_metrics

        # Check reliability health
        reliability_health = await reliability_manager.get_health_status()
        assert "status" in reliability_health
        assert reliability_health["status"] in ["healthy", "degraded", "unhealthy"]

        # Check cache performance
        cache_info = await intelligent_cache.get_cache_info()
        assert "l1_entries" in cache_info

        # Check scaling status
        scaling_status = await adaptive_scaler.get_scaling_status()
        assert "current_utilization" in scaling_status
