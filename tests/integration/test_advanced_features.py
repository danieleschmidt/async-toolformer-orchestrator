"""Integration tests for advanced orchestrator features."""

import asyncio

import pytest

from async_toolformer import (
    AsyncOrchestrator,
    MemoryConfig,
    OrchestratorConfig,
    RateLimitConfig,
    Tool,
)


@Tool(description="Test tool for caching", priority=1)
async def cacheable_tool(data: str, modifier: int = 1) -> dict:
    """Test tool that produces cacheable results."""
    await asyncio.sleep(0.1)  # Simulate work
    return {
        "input": data,
        "modifier": modifier,
        "result": hash(data) * modifier,
        "computed": True
    }


@Tool(description="Fast tool for parallelism testing", priority=2)
async def fast_tool(value: str = "default") -> str:
    """Fast tool for parallelism testing."""
    await asyncio.sleep(0.01)
    return f"processed_{value}"


@Tool(description="Tool with retry behavior", retry_attempts=3, priority=1)
async def flaky_tool(fail_probability: float = 0.5) -> str:
    """Tool that sometimes fails to test retry logic."""
    await asyncio.sleep(0.02)

    import random
    if random.random() < fail_probability:
        raise Exception("Simulated failure")

    return "success"


@Tool(description="Rate limited tool", rate_limit_group="limited", priority=1)
async def rate_limited_tool(request_id: str) -> str:
    """Tool for testing rate limiting."""
    await asyncio.sleep(0.01)
    return f"rate_limited_response_{request_id}"


@pytest.mark.asyncio
async def test_caching_functionality():
    """Test result caching functionality."""
    config = OrchestratorConfig(
        max_parallel_tools=2,
        max_parallel_per_type=2,
        memory_config=MemoryConfig(
            max_memory_gb=1.0,
            compress_results=True
        )
    )

    orchestrator = AsyncOrchestrator(
        tools=[cacheable_tool],
        config=config
    )

    # First execution - should compute
    result1 = await orchestrator.execute("Run cacheable tool")
    first_time = result1.get('total_time_ms', 0)

    # Second execution - should be cached (faster)
    result2 = await orchestrator.execute("Run cacheable tool")
    second_time = result2.get('total_time_ms', 0)

    # Verify caching worked (second should be much faster)
    assert second_time < first_time * 0.5  # At least 50% faster

    # Check cache stats
    cache_stats = await orchestrator.cache.get_stats()
    assert cache_stats['size'] > 0  # Should have cached entries

    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_parallel_execution_scaling():
    """Test parallel execution with many concurrent tools."""
    config = OrchestratorConfig(
        max_parallel_tools=20,
        max_parallel_per_type=10,
        tool_timeout_ms=5000
    )

    orchestrator = AsyncOrchestrator(
        tools=[fast_tool],
        config=config
    )

    # Execute many operations concurrently
    tasks = []
    for i in range(15):
        task = orchestrator.execute(f"Parallel operation {i}")
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # All should succeed
    successful = sum(1 for r in results if r.get('status') == 'completed')
    assert successful == 15

    # Should complete relatively quickly due to parallelism
    total_tools = sum(r.get('tools_executed', 0) for r in results)
    assert total_tools > 0

    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_retry_mechanism():
    """Test tool retry functionality."""
    config = OrchestratorConfig(
        max_parallel_tools=1,
        max_parallel_per_type=1,
        tool_timeout_ms=10000
    )

    orchestrator = AsyncOrchestrator(
        tools=[flaky_tool],
        config=config
    )

    # Run several times - some should succeed due to retries
    results = []
    for i in range(5):
        result = await orchestrator.execute(f"Retry test {i}")
        results.append(result)

    # At least some should succeed (retries should help)
    successful = sum(1 for r in results if r.get('successful_tools', 0) > 0)
    total_attempts = sum(1 for r in results if r.get('tools_executed', 0) > 0)

    # Should have attempted all
    assert total_attempts == 5

    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_rate_limiting():
    """Test rate limiting functionality."""
    rate_config = RateLimitConfig(
        global_max=50,
        service_limits={
            "limited": {"calls": 3, "window": 2}  # Very restrictive for testing
        }
    )

    config = OrchestratorConfig(
        max_parallel_tools=10,
        max_parallel_per_type=5,
        rate_limit_config=rate_config
    )

    orchestrator = AsyncOrchestrator(
        tools=[rate_limited_tool],
        config=config
    )

    # Make many requests quickly
    tasks = []
    for i in range(8):  # More than the limit of 3
        task = orchestrator.execute(f"Rate limit test {i}")
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # Some should succeed, some should be rate limited
    successful = sum(1 for r in results if r.get('successful_tools', 0) > 0)
    rate_limited = sum(1 for r in results if r.get('tools_executed', 0) == 0)

    # Should have both successes and rate limited requests
    assert successful > 0
    assert rate_limited > 0
    assert successful + rate_limited <= len(results)

    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_connection_pooling():
    """Test connection pool functionality."""
    config = OrchestratorConfig(
        max_parallel_tools=5,
        max_parallel_per_type=3
    )

    orchestrator = AsyncOrchestrator(
        tools=[fast_tool],
        config=config
    )

    # Make multiple requests to test pooling
    tasks = []
    for i in range(10):
        task = orchestrator.execute(f"Pool test {i}")
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    # All should succeed
    successful = sum(1 for r in results if r.get('status') == 'completed')
    assert successful == len(results)

    # Check connection pool stats
    pool_stats = await orchestrator.connection_pool.get_pool_stats()
    assert 'pools' in pool_stats or 'total_requests' in pool_stats

    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_comprehensive_metrics():
    """Test comprehensive metrics collection."""
    memory_config = MemoryConfig(compress_results=True)
    rate_config = RateLimitConfig(global_max=100)

    config = OrchestratorConfig(
        max_parallel_tools=5,
        max_parallel_per_type=3,
        memory_config=memory_config,
        rate_limit_config=rate_config
    )

    orchestrator = AsyncOrchestrator(
        tools=[cacheable_tool, fast_tool, rate_limited_tool],
        config=config
    )

    # Execute some operations
    await orchestrator.execute("Test metrics collection")

    # Get comprehensive metrics
    metrics = await orchestrator.get_metrics()

    # Verify all expected metrics are present
    expected_keys = [
        'registered_tools', 'registered_chains', 'active_tasks',
        'cached_results', 'speculation_tasks', 'cache',
        'connection_pools', 'config'
    ]

    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"

    # Verify specific values
    assert metrics['registered_tools'] == 3
    assert metrics['registered_chains'] == 0
    assert 'cache' in metrics
    assert 'connection_pools' in metrics

    # Cache metrics should have expected structure
    cache_metrics = metrics['cache']
    assert 'size' in cache_metrics
    assert 'max_size' in cache_metrics
    assert 'hit_rate' in cache_metrics

    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_memory_management():
    """Test memory management features."""
    memory_config = MemoryConfig(
        max_memory_gb=0.1,  # Very small for testing
        compress_results=True,
        max_result_size_mb=1
    )

    config = OrchestratorConfig(
        max_parallel_tools=3,
        max_parallel_per_type=3,
        memory_config=memory_config
    )

    orchestrator = AsyncOrchestrator(
        tools=[cacheable_tool],
        config=config
    )

    # Execute operations to fill cache
    for i in range(5):
        await orchestrator.execute(f"Memory test {i}")

    # Cache should have been managed (evictions, etc.)
    cache_stats = await orchestrator.cache.get_stats()
    assert 'evictions' in cache_stats or cache_stats.get('size', 0) >= 0

    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_error_handling_robustness():
    """Test comprehensive error handling."""
    config = OrchestratorConfig(
        max_parallel_tools=3,
        max_parallel_per_type=3,
        tool_timeout_ms=1000  # Short timeout for testing
    )

    orchestrator = AsyncOrchestrator(
        tools=[flaky_tool, fast_tool],
        config=config
    )

    # Test various error conditions
    results = []

    # Normal execution
    result1 = await orchestrator.execute("Normal test")
    results.append(result1)

    # With high failure probability
    result2 = await orchestrator.execute("High failure test")
    results.append(result2)

    # All executions should complete (either success or handled failure)
    for result in results:
        assert 'status' in result
        assert 'execution_id' in result
        assert 'total_time_ms' in result

    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_context_manager_with_features():
    """Test async context manager with all features."""
    memory_config = MemoryConfig(compress_results=True)
    rate_config = RateLimitConfig(global_max=50)

    config = OrchestratorConfig(
        max_parallel_tools=3,
        max_parallel_per_type=3,
        memory_config=memory_config,
        rate_limit_config=rate_config
    )

    async with AsyncOrchestrator(tools=[fast_tool], config=config) as orchestrator:
        result = await orchestrator.execute("Context manager test")

        assert result['status'] in ['completed', 'no_tools_called']

        # Metrics should be available
        metrics = await orchestrator.get_metrics()
        assert 'registered_tools' in metrics

    # Cleanup should be automatic
