"""Performance benchmark tests for async-toolformer-orchestrator."""

import asyncio
import time
from typing import Dict, List
from unittest.mock import AsyncMock

import pytest


@pytest.mark.benchmark
@pytest.mark.performance
class TestParallelExecutionBenchmarks:
    """Benchmark tests for parallel tool execution performance."""

    async def test_sequential_vs_parallel_execution(
        self,
        mock_orchestrator: AsyncMock,
        sample_tools: List[AsyncMock],
        benchmark_config: Dict
    ):
        """Benchmark sequential vs parallel execution."""
        tool_count = 10
        expected_tool_duration = 0.1  # 100ms per tool
        
        # Create tools with predictable timing
        tools = []
        for i in range(tool_count):
            tool = AsyncMock()
            tool.__name__ = f"tool_{i}"
            tool.side_effect = lambda: asyncio.sleep(expected_tool_duration) or f"result_{i}"
            tools.append(tool)
        
        # Benchmark sequential execution
        start_time = time.perf_counter()
        sequential_results = []
        for tool in tools:
            result = await tool()
            sequential_results.append(result)
        sequential_duration = time.perf_counter() - start_time
        
        # Benchmark parallel execution
        start_time = time.perf_counter()
        parallel_results = await asyncio.gather(*[tool() for tool in tools])
        parallel_duration = time.perf_counter() - start_time
        
        # Calculate speedup
        speedup = sequential_duration / parallel_duration
        
        # Assertions
        assert len(sequential_results) == tool_count
        assert len(parallel_results) == tool_count
        assert sequential_duration >= tool_count * expected_tool_duration * 0.9  # Allow 10% tolerance
        assert parallel_duration <= expected_tool_duration * 1.5  # Should be close to single tool time
        assert speedup >= 5.0  # Should be at least 5x faster
        
        print(f"Sequential duration: {sequential_duration:.3f}s")
        print(f"Parallel duration: {parallel_duration:.3f}s")
        print(f"Speedup: {speedup:.1f}x")

    async def test_scalability_with_tool_count(
        self,
        mock_orchestrator: AsyncMock,
        benchmark_config: Dict
    ):
        """Test performance scaling with increasing tool count."""
        tool_counts = benchmark_config["tool_counts"]
        tool_duration = 0.05  # 50ms per tool
        results = {}
        
        for count in tool_counts:
            # Create tools
            tools = []
            for i in range(count):
                tool = AsyncMock()
                tool.__name__ = f"tool_{i}"
                tool.side_effect = lambda: asyncio.sleep(tool_duration) or f"result_{i}"
                tools.append(tool)
            
            # Benchmark parallel execution
            start_time = time.perf_counter()
            await asyncio.gather(*[tool() for tool in tools])
            duration = time.perf_counter() - start_time
            
            results[count] = {
                "duration": duration,
                "throughput": count / duration,
                "efficiency": (count * tool_duration) / duration
            }
            
            print(f"Count: {count:2d}, Duration: {duration:.3f}s, "
                  f"Throughput: {results[count]['throughput']:.1f} tools/s, "
                  f"Efficiency: {results[count]['efficiency']:.2f}")
        
        # Verify scaling characteristics
        assert results[1]["duration"] >= tool_duration * 0.9
        assert results[max(tool_counts)]["efficiency"] >= 0.8  # At least 80% efficiency
        
        # Check that duration doesn't increase linearly with tool count
        duration_ratio = results[max(tool_counts)]["duration"] / results[1]["duration"]
        assert duration_ratio <= 2.0  # Should not double when tools 25x increase

    async def test_rate_limiting_performance_impact(
        self,
        mock_orchestrator: AsyncMock,
        mock_redis: AsyncMock
    ):
        """Benchmark the performance impact of rate limiting."""
        tool_count = 20
        tool_duration = 0.02  # 20ms per tool
        
        # Create tools
        tools = []
        for i in range(tool_count):
            tool = AsyncMock()
            tool.__name__ = f"tool_{i}"
            tool.side_effect = lambda: asyncio.sleep(tool_duration) or f"result_{i}"
            tools.append(tool)
        
        # Benchmark without rate limiting
        start_time = time.perf_counter()
        await asyncio.gather(*[tool() for tool in tools])
        no_limit_duration = time.perf_counter() - start_time
        
        # Simulate rate limiting overhead
        async def rate_limited_execution():
            tasks = []
            for tool in tools:
                # Simulate rate limit check
                await mock_redis.incr("rate_limit_key")
                await asyncio.sleep(0.001)  # 1ms rate limit overhead
                tasks.append(tool())
            return await asyncio.gather(*tasks)
        
        # Benchmark with rate limiting
        start_time = time.perf_counter()
        await rate_limited_execution()
        rate_limited_duration = time.perf_counter() - start_time
        
        # Calculate overhead
        overhead = (rate_limited_duration - no_limit_duration) / no_limit_duration
        
        print(f"No rate limiting: {no_limit_duration:.3f}s")
        print(f"With rate limiting: {rate_limited_duration:.3f}s")
        print(f"Overhead: {overhead:.1%}")
        
        # Rate limiting overhead should be minimal
        assert overhead <= 0.20  # No more than 20% overhead
        assert rate_limited_duration <= no_limit_duration * 1.25

    async def test_memory_usage_scaling(
        self,
        mock_orchestrator: AsyncMock,
        benchmark_config: Dict
    ):
        """Test memory usage scaling with parallel execution."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        tool_counts = [10, 50, 100, 200]
        memory_usage = {}
        
        for count in tool_counts:
            # Create memory-intensive tools
            tools = []
            for i in range(count):
                tool = AsyncMock()
                tool.__name__ = f"memory_tool_{i}"
                # Simulate some memory usage
                tool.side_effect = lambda: asyncio.sleep(0.01) or "x" * 1000  # 1KB result
                tools.append(tool)
            
            # Execute and measure memory
            await asyncio.gather(*[tool() for tool in tools])
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage[count] = current_memory - baseline_memory
            
            print(f"Tools: {count:3d}, Memory usage: {memory_usage[count]:5.1f} MB")
        
        # Memory usage should scale reasonably
        memory_per_tool = memory_usage[max(tool_counts)] / max(tool_counts)
        assert memory_per_tool <= 1.0  # No more than 1MB per tool on average
        
        # Memory growth should be sublinear
        growth_ratio = memory_usage[200] / memory_usage[10]
        tools_ratio = 200 / 10
        assert growth_ratio <= tools_ratio * 0.5  # Memory should grow slower than tool count

    async def test_error_handling_performance(
        self,
        mock_orchestrator: AsyncMock
    ):
        """Benchmark performance when some tools fail."""
        success_count = 15
        error_count = 5
        tool_duration = 0.03  # 30ms per tool
        
        # Create successful tools
        success_tools = []
        for i in range(success_count):
            tool = AsyncMock()
            tool.__name__ = f"success_tool_{i}"
            tool.side_effect = lambda: asyncio.sleep(tool_duration) or f"success_{i}"
            success_tools.append(tool)
        
        # Create error tools
        error_tools = []
        for i in range(error_count):
            tool = AsyncMock()
            tool.__name__ = f"error_tool_{i}"
            tool.side_effect = Exception(f"Error {i}")
            error_tools.append(tool)
        
        all_tools = success_tools + error_tools
        
        # Benchmark execution with mixed success/failure
        start_time = time.perf_counter()
        results = await asyncio.gather(*[tool() for tool in all_tools], return_exceptions=True)
        duration = time.perf_counter() - start_time
        
        # Count results
        success_results = [r for r in results if not isinstance(r, Exception)]
        error_results = [r for r in results if isinstance(r, Exception)]
        
        print(f"Duration: {duration:.3f}s")
        print(f"Successful: {len(success_results)}, Errors: {len(error_results)}")
        
        # Verify results
        assert len(success_results) == success_count
        assert len(error_results) == error_count
        assert duration <= tool_duration * 1.5  # Should be parallel, not sequential


@pytest.mark.benchmark
@pytest.mark.performance
class TestSpeculationBenchmarks:
    """Benchmark tests for speculation engine performance."""

    async def test_speculation_hit_rate_performance(
        self,
        mock_orchestrator: AsyncMock
    ):
        """Benchmark performance with different speculation hit rates."""
        total_tools = 20
        speculation_latency = 0.01  # 10ms speculation overhead
        tool_duration = 0.05  # 50ms per tool
        
        hit_rates = [0.0, 0.25, 0.5, 0.75, 1.0]
        results = {}
        
        for hit_rate in hit_rates:
            hits = int(total_tools * hit_rate)
            misses = total_tools - hits
            
            # Simulate speculation results
            hit_tools = []
            for i in range(hits):
                tool = AsyncMock()
                tool.__name__ = f"hit_tool_{i}"
                # Pre-fetched result (faster)
                tool.side_effect = lambda: asyncio.sleep(speculation_latency) or f"hit_{i}"
                hit_tools.append(tool)
            
            miss_tools = []
            for i in range(misses):
                tool = AsyncMock()
                tool.__name__ = f"miss_tool_{i}"
                # Full execution time
                tool.side_effect = lambda: asyncio.sleep(tool_duration) or f"miss_{i}"
                miss_tools.append(tool)
            
            all_tools = hit_tools + miss_tools
            
            # Benchmark execution
            start_time = time.perf_counter()
            await asyncio.gather(*[tool() for tool in all_tools])
            duration = time.perf_counter() - start_time
            
            results[hit_rate] = duration
            print(f"Hit rate: {hit_rate:.0%}, Duration: {duration:.3f}s")
        
        # Higher hit rates should result in better performance
        assert results[1.0] <= results[0.0] * 0.3  # 100% hit rate should be much faster
        assert results[0.75] <= results[0.25]  # Higher hit rate should be faster


@pytest.mark.benchmark
@pytest.mark.performance
class TestRateLimitingBenchmarks:
    """Benchmark tests for rate limiting performance."""

    async def test_redis_rate_limiter_performance(
        self,
        mock_redis: AsyncMock
    ):
        """Benchmark Redis-based rate limiter performance."""
        request_count = 100
        rate_limit = 50  # 50 requests per second
        
        # Setup Redis responses for rate limiting
        mock_redis.pipeline.return_value = mock_redis
        mock_redis.execute.return_value = [1, True, 1] * request_count
        
        # Benchmark rate limit checks
        start_time = time.perf_counter()
        
        tasks = []
        for i in range(request_count):
            async def check_rate_limit():
                # Simulate rate limit check
                await mock_redis.incr(f"rate_limit_key_{i}")
                await mock_redis.expire(f"rate_limit_key_{i}", 60)
                return True
            
            tasks.append(check_rate_limit())
        
        await asyncio.gather(*tasks)
        duration = time.perf_counter() - start_time
        
        throughput = request_count / duration
        
        print(f"Rate limit checks: {request_count}")
        print(f"Duration: {duration:.3f}s")
        print(f"Throughput: {throughput:.0f} checks/s")
        
        # Rate limit checks should be fast
        assert duration <= 1.0  # Should complete within 1 second
        assert throughput >= 500  # At least 500 checks per second

    async def test_backpressure_performance(
        self,
        mock_orchestrator: AsyncMock,
        mock_redis: AsyncMock
    ):
        """Benchmark performance under rate limit backpressure."""
        tool_count = 30
        rate_limit = 10  # 10 tools per second
        tool_duration = 0.02  # 20ms per tool
        
        # Create tools with rate limiting
        tools = []
        for i in range(tool_count):
            tool = AsyncMock()
            tool.__name__ = f"rate_limited_tool_{i}"
            
            async def rate_limited_execution():
                # Simulate rate limit wait
                await asyncio.sleep(1.0 / rate_limit)  # Wait based on rate limit
                await asyncio.sleep(tool_duration)  # Actual tool execution
                return f"result_{i}"
            
            tool.side_effect = rate_limited_execution
            tools.append(tool)
        
        # Benchmark execution with backpressure
        start_time = time.perf_counter()
        results = await asyncio.gather(*[tool() for tool in tools])
        duration = time.perf_counter() - start_time
        
        effective_rate = len(results) / duration
        
        print(f"Tools executed: {len(results)}")
        print(f"Duration: {duration:.3f}s")
        print(f"Effective rate: {effective_rate:.1f} tools/s")
        
        # Should respect rate limit
        assert effective_rate <= rate_limit * 1.2  # Allow 20% tolerance
        assert len(results) == tool_count  # All tools should complete


@pytest.mark.benchmark
@pytest.mark.performance
@pytest.mark.slow
class TestEndToEndBenchmarks:
    """End-to-end performance benchmark tests."""

    async def test_realistic_workflow_performance(
        self,
        mock_orchestrator: AsyncMock,
        benchmark_config: Dict
    ):
        """Benchmark a realistic workflow with mixed tool types."""
        # Simulate a realistic research workflow
        fast_tools_count = 10  # Quick data fetches
        medium_tools_count = 5  # API calls
        slow_tools_count = 2   # Complex processing
        
        fast_duration = 0.01   # 10ms
        medium_duration = 0.1  # 100ms
        slow_duration = 0.5    # 500ms
        
        # Create tool mix
        all_tools = []
        
        # Fast tools
        for i in range(fast_tools_count):
            tool = AsyncMock()
            tool.__name__ = f"fast_tool_{i}"
            tool.side_effect = lambda: asyncio.sleep(fast_duration) or f"fast_result_{i}"
            all_tools.append(tool)
        
        # Medium tools
        for i in range(medium_tools_count):
            tool = AsyncMock()
            tool.__name__ = f"medium_tool_{i}"
            tool.side_effect = lambda: asyncio.sleep(medium_duration) or f"medium_result_{i}"
            all_tools.append(tool)
        
        # Slow tools
        for i in range(slow_tools_count):
            tool = AsyncMock()
            tool.__name__ = f"slow_tool_{i}"
            tool.side_effect = lambda: asyncio.sleep(slow_duration) or f"slow_result_{i}"
            all_tools.append(tool)
        
        total_tools = len(all_tools)
        
        # Benchmark realistic workflow
        start_time = time.perf_counter()
        results = await asyncio.gather(*[tool() for tool in all_tools])
        duration = time.perf_counter() - start_time
        
        # Calculate expected sequential time
        expected_sequential = (
            fast_tools_count * fast_duration +
            medium_tools_count * medium_duration +
            slow_tools_count * slow_duration
        )
        
        speedup = expected_sequential / duration
        
        print(f"Total tools: {total_tools}")
        print(f"Parallel duration: {duration:.3f}s")
        print(f"Expected sequential: {expected_sequential:.3f}s")
        print(f"Speedup: {speedup:.1f}x")
        
        # Verify performance
        assert len(results) == total_tools
        assert duration <= slow_duration * 1.2  # Should be dominated by slowest tool
        assert speedup >= 3.0  # Should achieve significant speedup