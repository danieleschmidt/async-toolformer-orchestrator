#!/usr/bin/env python3
"""Robust example showing error handling, rate limiting, and retry logic."""

import asyncio
import random
from async_toolformer import AsyncOrchestrator, Tool, OrchestratorConfig, RateLimitConfig


@Tool(description="Search the web with error simulation", retry_attempts=3)
async def web_search_with_errors(query: str) -> str:
    """Simulate web search with potential errors."""
    await asyncio.sleep(0.1)
    
    # Simulate random failures for demonstration
    if random.random() < 0.3:  # 30% failure rate
        raise Exception(f"Network error while searching for: {query}")
    
    return f"Search results for: {query}"


@Tool(description="Analyze code with rate limiting", rate_limit_group="analysis")
async def analyze_code_limited(filename: str) -> dict:
    """Analyze code with rate limiting."""
    await asyncio.sleep(0.2)
    return {
        "file": filename,
        "complexity": random.randint(1, 10),
        "issues": random.randint(0, 5),
        "score": round(random.uniform(5.0, 10.0), 1)
    }


@Tool(description="High-frequency tool", rate_limit_group="high_freq")
async def high_frequency_tool(data: str = "default") -> str:
    """Tool that gets called frequently."""
    await asyncio.sleep(0.05)
    return f"Processed: {data}"


@Tool(description="Slow tool with timeout", timeout_ms=2000)
async def slow_tool(delay: float = 1.0) -> str:
    """Tool that may timeout."""
    await asyncio.sleep(delay)
    return f"Completed after {delay}s"


async def main():
    """Demonstrate robust orchestration features."""
    print("🛡️ AsyncOrchestrator Robust Example")
    print("=" * 50)
    
    # Configure rate limiting
    rate_config = RateLimitConfig(
        global_max=20,  # 20 requests per minute globally
        service_limits={
            "analysis": {"calls": 5, "window": 10},  # 5 calls per 10 seconds
            "high_freq": {"calls": 10, "window": 5},  # 10 calls per 5 seconds
        }
    )
    
    # Configure orchestrator with robustness features
    config = OrchestratorConfig(
        max_parallel_tools=8,
        max_parallel_per_type=5,  # Must be <= max_parallel_tools
        tool_timeout_ms=5000,
        total_timeout_ms=15000,
        retry_attempts=3,
        rate_limit_config=rate_config
    )
    
    orchestrator = AsyncOrchestrator(
        tools=[web_search_with_errors, analyze_code_limited, high_frequency_tool, slow_tool],
        config=config
    )
    
    print(f"Configured orchestrator with:")
    print(f"  - Max parallel tools: {config.max_parallel_tools}")
    print(f"  - Tool timeout: {config.tool_timeout_ms}ms")
    print(f"  - Global rate limit: {rate_config.global_max} calls/minute")
    print()
    
    # Example 1: Error handling and retries
    print("🔄 Example 1: Error Handling & Retries")
    print("-" * 40)
    
    for i in range(3):
        print(f"Attempt {i+1}:")
        result = await orchestrator.execute(
            "Search for Python async best practices"
        )
        
        print(f"  Status: {result['status']}")
        print(f"  Tools executed: {result.get('tools_executed', 0)}")
        print(f"  Successful: {result.get('successful_tools', 0)}")
        print(f"  Time: {result['total_time_ms']:.1f}ms")
        
        if result.get('results'):
            for j, tool_result in enumerate(result['results']):
                status = "✅" if tool_result.success else "❌"
                data_preview = str(tool_result.data)[:50] if tool_result.data else "None"
                print(f"    {status} {tool_result.tool_name}: {data_preview}...")
        print()
        
        await asyncio.sleep(0.5)  # Brief delay between attempts
    
    # Example 2: Rate limiting demonstration
    print("⏱️ Example 2: Rate Limiting")
    print("-" * 40)
    
    print("Testing high-frequency tool (should hit rate limits):")
    
    tasks = []
    for i in range(15):  # Try to exceed the 10 calls/5 seconds limit
        task = orchestrator.execute(f"Process batch {i}")
        tasks.append(task)
    
    # Execute all at once to trigger rate limiting
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    success_count = 0
    rate_limited_count = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"  Batch {i}: Exception - {result}")
        elif result.get('status') == 'completed':
            success_count += 1
        else:
            # Check if any tools were rate limited
            if result.get('results'):
                for tool_result in result['results']:
                    if not tool_result.success and 'rate limit' in str(tool_result.data).lower():
                        rate_limited_count += 1
                        break
    
    print(f"Results: {success_count} successful, {rate_limited_count} rate limited")
    print()
    
    # Example 3: Timeout handling
    print("⏰ Example 3: Timeout Handling")
    print("-" * 40)
    
    print("Testing tool with various delays:")
    test_delays = [0.5, 1.0, 1.5, 3.0]  # Last one should timeout (tool timeout is 2s)
    
    for delay in test_delays:
        print(f"  Testing {delay}s delay:")
        result = await orchestrator.execute(f"Run slow operation with {delay}s delay")
        
        if result.get('results'):
            for tool_result in result['results']:
                status = "✅" if tool_result.success else "❌"
                print(f"    {status} Time: {tool_result.execution_time_ms:.1f}ms")
                if not tool_result.success:
                    print(f"        Error: {tool_result.data}")
        print()
    
    # Example 4: Metrics and monitoring
    print("📊 Example 4: Metrics")
    print("-" * 40)
    
    metrics = await orchestrator.get_metrics()
    print("Orchestrator Metrics:")
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    print("\nLLM Integration Metrics:")
    llm_metrics = orchestrator.llm_integration.get_metrics()
    for key, value in llm_metrics.items():
        print(f"  {key}: {value}")
    
    print("\nRate Limiter Status:")
    for tool_name in ["analysis", "high_freq"]:
        remaining = orchestrator.rate_limiter.get_remaining(tool_name)
        print(f"  {tool_name}: {remaining} requests remaining")
    
    print()
    print("🎉 Robust demo completed!")
    
    # Cleanup
    await orchestrator.cleanup()


if __name__ == "__main__":
    # Set up logging to see what's happening
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(main())