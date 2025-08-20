"""
Performance tests for Quantum-Enhanced AsyncOrchestrator.

This module provides comprehensive performance tests:
- Performance optimization validation
- Resource scaling behavior
- Concurrency coordination performance
- Throughput and latency benchmarks
- Memory and resource usage tests
"""

import asyncio
import gc
import statistics
import time
from typing import Any

import psutil
import pytest

from async_toolformer import (
    OptimizationStrategy,
    QuantumPerformanceOptimizer,
    Tool,
    create_quantum_orchestrator,
)


# Performance test tools with varying characteristics
@Tool(description="Lightweight computation")
async def lightweight_task(value: int) -> int:
    """Very fast computation task."""
    return value * 2


@Tool(description="CPU intensive computation")
async def cpu_intensive_task(iterations: int = 1000) -> dict[str, Any]:
    """CPU-intensive task for load testing."""
    start_time = time.time()

    # Simulate CPU-intensive work
    result = 0
    for i in range(iterations):
        result += i ** 2
        if i % 100 == 0:
            await asyncio.sleep(0.001)  # Yield control periodically

    execution_time = (time.time() - start_time) * 1000

    return {
        "result": result,
        "iterations": iterations,
        "execution_time_ms": execution_time,
        "cpu_intensive": True,
    }


@Tool(description="Memory intensive computation")
async def memory_intensive_task(size_mb: int = 10) -> dict[str, Any]:
    """Memory-intensive task for load testing."""
    start_time = time.time()

    # Allocate memory
    data = bytearray(size_mb * 1024 * 1024)  # size_mb MB

    # Do some work with the data
    for i in range(0, len(data), 1000):
        data[i] = i % 256

    # Small delay
    await asyncio.sleep(0.1)

    execution_time = (time.time() - start_time) * 1000
    data_checksum = sum(data[::1000]) % 1000000

    # Clean up memory
    del data
    gc.collect()

    return {
        "size_mb": size_mb,
        "checksum": data_checksum,
        "execution_time_ms": execution_time,
        "memory_intensive": True,
    }


@Tool(description="IO simulation task")
async def io_simulation_task(delay_ms: int = 100) -> dict[str, Any]:
    """IO simulation task."""
    start_time = time.time()

    # Simulate IO wait
    await asyncio.sleep(delay_ms / 1000.0)

    execution_time = (time.time() - start_time) * 1000

    return {
        "delay_ms": delay_ms,
        "execution_time_ms": execution_time,
        "io_simulation": True,
    }


@Tool(description="Variable duration task")
async def variable_duration_task(min_ms: int = 50, max_ms: int = 500) -> dict[str, Any]:
    """Task with variable execution time."""
    import random

    duration_ms = random.randint(min_ms, max_ms)
    start_time = time.time()

    await asyncio.sleep(duration_ms / 1000.0)

    actual_time = (time.time() - start_time) * 1000

    return {
        "requested_duration_ms": duration_ms,
        "actual_duration_ms": actual_time,
        "variable_duration": True,
    }


class TestQuantumPerformanceOptimizer:
    """Test the QuantumPerformanceOptimizer functionality."""

    @pytest.fixture
    def performance_optimizer(self):
        """Create a performance optimizer for testing."""
        return QuantumPerformanceOptimizer(
            optimization_strategy=OptimizationStrategy.EFFICIENCY,
            enable_auto_scaling=True,
            performance_history_size=100,
            monitoring_interval_seconds=0.1,  # Fast for tests
        )

    @pytest.mark.asyncio
    async def test_performance_monitoring(self, performance_optimizer):
        """Test performance monitoring functionality."""
        # Start monitoring
        await performance_optimizer.start_monitoring()

        try:
            # Record some task executions
            for i in range(10):
                performance_optimizer.record_task_execution(
                    task_id=f"test_task_{i}",
                    duration_ms=100 + (i * 10),
                    success=True,
                    worker_id=f"worker_{i % 3}",
                )

            # Wait for monitoring to collect data
            await asyncio.sleep(0.5)

            # Get current metrics
            metrics = performance_optimizer.get_current_metrics()

            assert metrics.total_tasks_completed == 10
            assert metrics.successful_tasks == 10
            assert metrics.failed_tasks == 0
            assert metrics.average_task_duration_ms > 0

            # Get history
            history = performance_optimizer.get_performance_history()
            assert len(history) > 0

        finally:
            await performance_optimizer.stop_monitoring()

    @pytest.mark.asyncio
    async def test_auto_scaling(self, performance_optimizer):
        """Test automatic resource scaling."""
        await performance_optimizer.start_monitoring()

        try:
            # Simulate high CPU usage scenarios
            for i in range(20):
                performance_optimizer.record_task_execution(
                    task_id=f"cpu_task_{i}",
                    duration_ms=1000,  # Long duration to simulate high usage
                    success=True,
                    worker_id=f"worker_{i % 2}",
                    resource_usage={"cpu": 0.9, "memory": 0.7},
                )

            # Wait for scaling decisions
            await asyncio.sleep(1.0)

            # Check scaling status
            scaling_status = performance_optimizer.get_resource_scaling_status()

            assert "current_scales" in scaling_status
            assert "scaling_rules" in scaling_status
            assert scaling_status["auto_scaling_enabled"] is True

            # Should have scaling rules for different resources
            assert len(scaling_status["scaling_rules"]) > 0

        finally:
            await performance_optimizer.stop_monitoring()

    def test_optimization_recommendations(self, performance_optimizer):
        """Test optimization recommendations generation."""
        # Simulate performance history with various patterns

        # High CPU utilization scenario
        for i in range(10):
            performance_optimizer.record_task_execution(
                task_id=f"high_cpu_{i}",
                duration_ms=2000,  # Long tasks
                success=True,
                resource_usage={"cpu": 0.95},
            )

        # Update metrics manually for testing
        performance_optimizer._current_metrics.cpu_utilization = 0.95

        # Get recommendations
        recommendations = performance_optimizer.get_optimization_recommendations()

        # Should have recommendations for high CPU usage
        cpu_recommendations = [
            rec for rec in recommendations
            if rec["metric"] == "cpu_utilization"
        ]

        assert len(cpu_recommendations) > 0
        high_cpu_rec = cpu_recommendations[0]
        assert high_cpu_rec["priority"] == "high"
        assert "scaling" in high_cpu_rec["recommendation"].lower()

    def test_cache_optimization(self, performance_optimizer):
        """Test cache access pattern optimization."""
        # Record cache access patterns
        for i in range(20):
            cache_key = f"frequent_key_{i % 3}"  # Create some frequent keys
            hit = i % 4 != 0  # 75% hit rate
            performance_optimizer.record_cache_access(cache_key, hit)

        # Should track access patterns
        assert len(performance_optimizer._cache_access_patterns) > 0

        # Should have recorded frequent accesses
        frequent_key_accesses = performance_optimizer._cache_access_patterns.get("frequent_key_0", [])
        assert len(frequent_key_accesses) > 5  # Should have multiple accesses

    @pytest.mark.asyncio
    async def test_strategy_switching(self, performance_optimizer):
        """Test optimization strategy switching."""
        original_strategy = performance_optimizer.optimization_strategy
        original_interval = performance_optimizer.monitoring_interval

        # Switch to latency optimization
        performance_optimizer.set_optimization_strategy(OptimizationStrategy.LATENCY)

        assert performance_optimizer.optimization_strategy == OptimizationStrategy.LATENCY
        assert performance_optimizer.monitoring_interval < original_interval  # Should be faster

        # Switch to throughput optimization
        performance_optimizer.set_optimization_strategy(OptimizationStrategy.THROUGHPUT)

        assert performance_optimizer.optimization_strategy == OptimizationStrategy.THROUGHPUT
        assert performance_optimizer.monitoring_interval > 0.5  # Should be slower for throughput

        # Switch to quantum coherent
        performance_optimizer.set_optimization_strategy(OptimizationStrategy.QUANTUM_COHERENT)

        assert performance_optimizer.optimization_strategy == OptimizationStrategy.QUANTUM_COHERENT
        assert performance_optimizer.monitoring_interval == 0.1  # Very frequent


class TestQuantumOrchestratorPerformance:
    """Test performance aspects of QuantumAsyncOrchestrator."""

    @pytest.fixture
    async def performance_orchestrator(self):
        """Create an orchestrator optimized for performance testing."""
        orchestrator = create_quantum_orchestrator(
            tools=[
                lightweight_task,
                cpu_intensive_task,
                memory_intensive_task,
                io_simulation_task,
                variable_duration_task,
            ],
            quantum_config={
                "max_parallel_tasks": 20,
                "optimization_iterations": 25,  # Reduced for testing
                "performance": {
                    "strategy": OptimizationStrategy.EFFICIENCY,
                    "auto_scaling": True,
                    "monitoring_interval": 0.1,
                },
                "concurrency": {
                    "quantum_sync": True,
                    "deadlock_detection": True,
                },
            }
        )

        yield orchestrator
        await orchestrator.cleanup()

    @pytest.mark.asyncio
    async def test_parallel_execution_performance(self, performance_orchestrator):
        """Test performance of parallel execution."""
        num_tasks = 10
        start_time = time.time()

        # Execute multiple tasks in parallel
        tasks = []
        for i in range(num_tasks):
            task = asyncio.create_task(
                performance_orchestrator.quantum_execute(
                    f"Run lightweight task with value {i * 10}",
                    optimize_plan=True,
                )
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Verify all completed successfully
        successful_count = sum(1 for r in results if r["status"] == "completed")
        assert successful_count == num_tasks

        # Verify parallel execution was efficient
        # Should be much faster than sequential execution
        expected_sequential_time = num_tasks * 0.2  # Estimate
        assert total_time < expected_sequential_time * 0.7  # At least 30% faster

        print(f"Parallel execution: {num_tasks} tasks in {total_time:.2f}s")
        print(f"Throughput: {num_tasks / total_time:.2f} tasks/second")

    @pytest.mark.asyncio
    async def test_resource_intensive_performance(self, performance_orchestrator):
        """Test performance with resource-intensive tasks."""
        # Monitor system resources before test
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        # Execute resource-intensive tasks
        result = await performance_orchestrator.quantum_execute(
            "Run CPU intensive task with 2000 iterations and memory intensive task with 50 MB",
            optimize_plan=True,
            timeout_ms=15000,  # 15 second timeout
        )

        execution_time = time.time() - start_time

        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        assert result["status"] == "completed"
        assert execution_time < 12.0  # Should complete within reasonable time

        # Memory should not increase excessively (good cleanup)
        assert memory_increase < 100  # Less than 100MB increase

        print(f"Resource intensive test: {execution_time:.2f}s")
        print(f"Memory increase: {memory_increase:.1f}MB")

    @pytest.mark.asyncio
    async def test_performance_optimization_effectiveness(self, performance_orchestrator):
        """Test effectiveness of performance optimization."""
        # Test without optimization
        start_time = time.time()
        unoptimized_result = await performance_orchestrator.quantum_execute(
            "Run multiple variable duration tasks",
            optimize_plan=False,
        )
        unoptimized_time = time.time() - start_time

        # Test with optimization
        start_time = time.time()
        optimized_result = await performance_orchestrator.quantum_execute(
            "Run multiple variable duration tasks",
            optimize_plan=True,
        )
        optimized_time = time.time() - start_time

        # Both should complete successfully
        assert unoptimized_result["status"] == "completed"
        assert optimized_result["status"] == "completed"

        # Get performance metrics
        opt_metrics = optimized_result["quantum_metrics"]
        unopt_metrics = unoptimized_result["quantum_metrics"]

        # Optimization should provide measurable benefits
        opt_score = opt_metrics["optimization_score"]
        unopt_score = unopt_metrics["optimization_score"]

        print(f"Unoptimized: {unoptimized_time:.2f}s, score: {unopt_score:.3f}")
        print(f"Optimized: {optimized_time:.2f}s, score: {opt_score:.3f}")

        # Optimized version should have better or equal optimization score
        assert opt_score >= unopt_score - 0.1  # Allow small variance

    @pytest.mark.asyncio
    async def test_concurrent_execution_scaling(self, performance_orchestrator):
        """Test scaling behavior with concurrent executions."""
        concurrency_levels = [1, 2, 5, 10]
        results = {}

        for concurrency in concurrency_levels:
            start_time = time.time()

            # Create concurrent executions
            tasks = []
            for i in range(concurrency):
                task = asyncio.create_task(
                    performance_orchestrator.quantum_execute(
                        "Run IO simulation task with delay 200",
                        optimize_plan=True,
                    )
                )
                tasks.append(task)

            # Wait for completion
            await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            throughput = concurrency / total_time
            results[concurrency] = {
                "time": total_time,
                "throughput": throughput,
            }

            print(f"Concurrency {concurrency}: {total_time:.2f}s, throughput: {throughput:.2f}")

        # Verify scaling behavior
        # Higher concurrency should generally have higher throughput
        assert results[10]["throughput"] > results[1]["throughput"] * 2
        assert results[5]["throughput"] > results[2]["throughput"]

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, performance_orchestrator):
        """Test memory usage stability over multiple executions."""
        process = psutil.Process()
        memory_samples = []

        # Baseline memory usage
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024

        # Execute multiple tasks and monitor memory
        for i in range(20):
            await performance_orchestrator.quantum_execute(
                "Run memory intensive task with size 5 MB",
                optimize_plan=True,
            )

            # Force garbage collection and sample memory
            gc.collect()
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory - initial_memory)

            # Small delay between executions
            await asyncio.sleep(0.1)

        # Analyze memory usage pattern
        avg_memory_increase = statistics.mean(memory_samples)
        max_memory_increase = max(memory_samples)
        final_memory_increase = memory_samples[-1]

        print(f"Memory usage - Avg: {avg_memory_increase:.1f}MB, "
              f"Max: {max_memory_increase:.1f}MB, Final: {final_memory_increase:.1f}MB")

        # Memory usage should be stable (not continuously growing)
        assert avg_memory_increase < 50  # Average increase less than 50MB
        assert final_memory_increase < max_memory_increase * 1.5  # No major leak

        # Memory should not grow linearly with number of executions
        early_avg = statistics.mean(memory_samples[:5])
        late_avg = statistics.mean(memory_samples[-5:])
        growth_ratio = late_avg / max(early_avg, 1.0)

        assert growth_ratio < 3.0  # Less than 3x growth from early to late


class TestPerformanceBenchmarks:
    """Performance benchmarks and stress tests."""

    @pytest.fixture
    async def benchmark_orchestrator(self):
        """Create orchestrator for benchmarking."""
        orchestrator = create_quantum_orchestrator(
            tools=[
                lightweight_task,
                cpu_intensive_task,
                io_simulation_task,
                variable_duration_task,
            ],
            quantum_config={
                "max_parallel_tasks": 50,
                "optimization_iterations": 100,
                "performance": {
                    "strategy": OptimizationStrategy.THROUGHPUT,
                    "auto_scaling": True,
                    "monitoring_interval": 0.05,
                },
            }
        )

        yield orchestrator
        await orchestrator.cleanup()

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_throughput_benchmark(self, benchmark_orchestrator):
        """Benchmark maximum throughput."""
        num_executions = 50
        batch_size = 10

        start_time = time.time()
        total_successful = 0

        # Execute in batches to avoid overwhelming the system
        for batch_start in range(0, num_executions, batch_size):
            batch_end = min(batch_start + batch_size, num_executions)
            batch_tasks = []

            for i in range(batch_start, batch_end):
                task = asyncio.create_task(
                    benchmark_orchestrator.quantum_execute(
                        f"Run lightweight task with value {i}",
                        optimize_plan=True,
                    )
                )
                batch_tasks.append(task)

            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # Count successful executions
            batch_successful = sum(
                1 for r in batch_results
                if not isinstance(r, Exception) and r.get("status") == "completed"
            )
            total_successful += batch_successful

        total_time = time.time() - start_time
        throughput = total_successful / total_time

        print("\nThroughput Benchmark Results:")
        print(f"Total Executions: {num_executions}")
        print(f"Successful: {total_successful}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} executions/second")
        print(f"Success Rate: {(total_successful/num_executions)*100:.1f}%")

        # Performance assertions
        assert total_successful >= num_executions * 0.9  # At least 90% success
        assert throughput > 5.0  # At least 5 executions per second

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_latency_benchmark(self, benchmark_orchestrator):
        """Benchmark execution latency."""
        num_samples = 30
        latencies = []

        # Warm up
        for _ in range(5):
            await benchmark_orchestrator.quantum_execute(
                "Run lightweight task with value 1",
                optimize_plan=True,
            )

        # Measure latencies
        for i in range(num_samples):
            start_time = time.time()

            result = await benchmark_orchestrator.quantum_execute(
                f"Run lightweight task with value {i}",
                optimize_plan=True,
            )

            latency = time.time() - start_time
            latencies.append(latency)

            assert result["status"] == "completed"

        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

        print("\nLatency Benchmark Results:")
        print(f"Samples: {num_samples}")
        print(f"Average: {avg_latency*1000:.1f}ms")
        print(f"Median: {median_latency*1000:.1f}ms")
        print(f"Min: {min_latency*1000:.1f}ms")
        print(f"Max: {max_latency*1000:.1f}ms")
        print(f"P95: {p95_latency*1000:.1f}ms")

        # Performance assertions
        assert avg_latency < 1.0  # Average latency under 1 second
        assert p95_latency < 2.0  # 95th percentile under 2 seconds
        assert min_latency < 0.5  # Best case under 500ms

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_stress_test(self, benchmark_orchestrator):
        """Stress test with high load."""
        duration_seconds = 10
        concurrent_executions = 20

        start_time = time.time()
        completed_executions = 0
        failed_executions = 0

        async def stress_worker(worker_id: int):
            nonlocal completed_executions, failed_executions

            while time.time() - start_time < duration_seconds:
                try:
                    result = await benchmark_orchestrator.quantum_execute(
                        "Run variable duration task with min 50 and max 200",
                        optimize_plan=True,
                    )

                    if result["status"] == "completed":
                        completed_executions += 1
                    else:
                        failed_executions += 1

                except Exception as e:
                    failed_executions += 1
                    print(f"Worker {worker_id} error: {e}")

                # Small delay to avoid overwhelming
                await asyncio.sleep(0.01)

        # Start stress workers
        workers = [
            asyncio.create_task(stress_worker(i))
            for i in range(concurrent_executions)
        ]

        # Wait for test duration
        await asyncio.sleep(duration_seconds + 1)

        # Cancel workers
        for worker in workers:
            worker.cancel()

        await asyncio.gather(*workers, return_exceptions=True)

        total_executions = completed_executions + failed_executions
        success_rate = completed_executions / max(total_executions, 1)
        throughput = completed_executions / duration_seconds

        print("\nStress Test Results:")
        print(f"Duration: {duration_seconds}s")
        print(f"Concurrent Workers: {concurrent_executions}")
        print(f"Total Executions: {total_executions}")
        print(f"Completed: {completed_executions}")
        print(f"Failed: {failed_executions}")
        print(f"Success Rate: {success_rate*100:.1f}%")
        print(f"Throughput: {throughput:.2f} executions/second")

        # Stress test assertions
        assert success_rate > 0.7  # At least 70% success under stress
        assert throughput > 2.0    # At least 2 executions/second under stress
        assert total_executions > 50  # Should have attempted substantial work


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s", "-m", "not benchmark"])

    # Run benchmarks separately if requested
    print("\nTo run benchmarks, use: pytest -m benchmark")
