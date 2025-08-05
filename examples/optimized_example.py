#!/usr/bin/env python3
"""Optimized example showing caching, connection pooling, and performance features."""

import asyncio
import random
import time
from async_toolformer import (
    AsyncOrchestrator, Tool, OrchestratorConfig, 
    RateLimitConfig, MemoryConfig
)


@Tool(description="Expensive computation that benefits from caching", priority=1)
async def expensive_computation(input_data: str, complexity: int = 5) -> dict:
    """Simulate an expensive computation that should be cached."""
    # Simulate expensive work
    await asyncio.sleep(complexity * 0.1)
    
    result = {
        "input": input_data,
        "complexity": complexity,
        "result": hash(input_data) % 10000,
        "computed_at": time.time(),
        "computation_time": complexity * 0.1
    }
    
    return result


@Tool(description="Fast data retrieval", priority=2)
async def fast_data_retrieval(key: str) -> dict:
    """Simulate fast data retrieval."""
    await asyncio.sleep(0.02)
    return {
        "key": key,
        "value": f"data_{key}_{random.randint(1, 100)}",
        "retrieved_at": time.time()
    }


@Tool(description="Network API call simulation", priority=1)
async def network_api_call(endpoint: str, timeout: float = 1.0) -> dict:
    """Simulate a network API call."""
    await asyncio.sleep(min(timeout, 0.5))  # Simulate network latency
    
    return {
        "endpoint": endpoint,
        "status": 200,
        "data": f"response_from_{endpoint}",
        "latency_ms": timeout * 1000
    }


@Tool(description="Batch processor", priority=3)
async def batch_processor(items: list, batch_size: int = 10) -> dict:
    """Process items in batches."""
    # Simulate batch processing
    num_batches = len(items) // batch_size + (1 if len(items) % batch_size else 0)
    await asyncio.sleep(num_batches * 0.05)
    
    return {
        "total_items": len(items),
        "batch_size": batch_size,
        "num_batches": num_batches,
        "processed": True
    }


async def demonstrate_caching():
    """Demonstrate caching effectiveness."""
    print("💾 Caching Demonstration")
    print("-" * 30)
    
    # Configure for optimal caching
    memory_config = MemoryConfig(
        max_memory_gb=1.0,
        compress_results=True,
        max_result_size_mb=10
    )
    
    config = OrchestratorConfig(
        max_parallel_tools=4,
        max_parallel_per_type=4,
        memory_config=memory_config
    )
    
    orchestrator = AsyncOrchestrator(
        tools=[expensive_computation],
        config=config
    )
    
    # First execution - should be slow
    print("First execution (no cache):")
    start_time = time.time()
    result1 = await orchestrator.execute("Run expensive computation on test_data")
    first_time = time.time() - start_time
    print(f"  Time: {first_time:.3f}s")
    
    # Second execution - should be fast (cached)
    print("Second execution (cached):")
    start_time = time.time()
    result2 = await orchestrator.execute("Run expensive computation on test_data")
    second_time = time.time() - start_time
    print(f"  Time: {second_time:.3f}s")
    
    # Show cache effectiveness
    speedup = first_time / second_time if second_time > 0 else float('inf')
    print(f"  Speedup: {speedup:.1f}x")
    
    # Check if result was cached
    if result2.get('results'):
        for tool_result in result2['results']:
            if tool_result.metadata.get('cached'):
                print("  ✅ Result served from cache")
            else:
                print("  ❌ Result not cached")
    
    cache_stats = await orchestrator.cache.get_stats()
    print(f"  Cache hit rate: {cache_stats.get('hit_rate', 0):.1%}")
    
    await orchestrator.cleanup()
    print()


async def demonstrate_parallel_optimization():
    """Demonstrate parallel execution optimization."""
    print("⚡ Parallel Optimization")
    print("-" * 30)
    
    config = OrchestratorConfig(
        max_parallel_tools=20,  # High parallelism
        max_parallel_per_type=10,
        tool_timeout_ms=5000
    )
    
    orchestrator = AsyncOrchestrator(
        tools=[fast_data_retrieval, network_api_call, batch_processor],
        config=config
    )
    
    # Test with many parallel operations
    print("Executing 15 parallel operations:")
    
    tasks = []
    start_time = time.time()
    
    for i in range(15):
        task = orchestrator.execute(f"Process batch operation {i}")
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    successful = sum(1 for r in results if r.get('status') == 'completed')
    total_tools = sum(r.get('tools_executed', 0) for r in results)
    
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Operations: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Total tools executed: {total_tools}")
    print(f"  Average time per operation: {total_time/len(results):.3f}s")
    
    await orchestrator.cleanup()
    print()


async def demonstrate_connection_pooling():
    """Demonstrate connection pooling benefits."""
    print("🔗 Connection Pooling")
    print("-" * 30)
    
    config = OrchestratorConfig(
        max_parallel_tools=10,
        max_parallel_per_type=5
    )
    
    orchestrator = AsyncOrchestrator(
        tools=[network_api_call],
        config=config
    )
    
    # Make multiple network calls
    print("Making 10 concurrent network calls:")
    
    start_time = time.time()
    tasks = [
        orchestrator.execute(f"Call API endpoint_{i}")
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Concurrent calls: {len(results)}")
    
    # Show connection pool stats
    pool_stats = await orchestrator.connection_pool.get_pool_stats()
    print(f"  Total requests: {pool_stats.get('total_requests', 0)}")
    print(f"  Average response time: {pool_stats.get('avg_response_time', 0):.3f}s")
    
    await orchestrator.cleanup()
    print()


async def performance_benchmark():
    """Run a comprehensive performance benchmark."""
    print("🏁 Performance Benchmark")
    print("-" * 30)
    
    # Optimized configuration
    memory_config = MemoryConfig(
        max_memory_gb=2.0,
        compress_results=True
    )
    
    rate_config = RateLimitConfig(
        global_max=1000,  # Very high limits for benchmarking
        service_limits={
            "benchmark": {"calls": 500, "window": 10}
        }
    )
    
    config = OrchestratorConfig(
        max_parallel_tools=50,
        max_parallel_per_type=25,
        tool_timeout_ms=10000,
        memory_config=memory_config,
        rate_limit_config=rate_config
    )
    
    orchestrator = AsyncOrchestrator(
        tools=[expensive_computation, fast_data_retrieval, network_api_call, batch_processor],
        config=config
    )
    
    print("Running comprehensive benchmark...")
    
    # Mix of different operation types
    benchmark_tasks = []
    start_time = time.time()
    
    # Add compute-heavy tasks
    for i in range(10):
        task = orchestrator.execute(f"Expensive computation {i}")
        benchmark_tasks.append(task)
    
    # Add fast tasks
    for i in range(20):
        task = orchestrator.execute(f"Fast retrieval {i}")
        benchmark_tasks.append(task)
    
    # Add network tasks
    for i in range(15):
        task = orchestrator.execute(f"Network call {i}")
        benchmark_tasks.append(task)
    
    # Execute all tasks
    results = await asyncio.gather(*benchmark_tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    # Analyze results
    successful = sum(1 for r in results if not isinstance(r, Exception) and r.get('status') == 'completed')
    errors = sum(1 for r in results if isinstance(r, Exception))
    total_operations = len(results)
    
    print(f"  Total operations: {total_operations}")
    print(f"  Successful: {successful}")
    print(f"  Errors: {errors}")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Operations per second: {total_operations/total_time:.1f}")
    
    # Get comprehensive metrics
    metrics = await orchestrator.get_metrics()
    
    print("\n📊 Final Metrics:")
    print(f"  Cache hit rate: {metrics['cache'].get('hit_rate', 0):.1%}")
    print(f"  Cache size: {metrics['cache'].get('size', 0)} entries")
    print(f"  Connection pools: {len(metrics.get('connection_pools', {}).get('pools', {}))}")
    print(f"  Total requests: {metrics.get('connection_pools', {}).get('total_requests', 0)}")
    
    await orchestrator.cleanup()
    print()


async def main():
    """Run all optimization demonstrations."""
    print("🚀 AsyncOrchestrator Optimization Demo")
    print("=" * 50)
    
    await demonstrate_caching()
    await demonstrate_parallel_optimization()
    await demonstrate_connection_pooling()
    await performance_benchmark()
    
    print("🎉 Optimization demo completed!")


if __name__ == "__main__":
    # Set up logging for performance monitoring
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())