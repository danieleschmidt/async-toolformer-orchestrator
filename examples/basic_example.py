#!/usr/bin/env python3
"""Basic example of AsyncOrchestrator usage."""

import asyncio
import aiohttp
from async_toolformer import AsyncOrchestrator, Tool


@Tool(description="Search the web for information about a topic")
async def web_search(query: str) -> str:
    """Simulate web search."""
    await asyncio.sleep(0.5)  # Simulate API call
    return f"Search results for: {query}"


@Tool(description="Analyze code in a file for complexity")
async def analyze_code(filename: str) -> dict:
    """Simulate code analysis."""
    await asyncio.sleep(0.3)
    return {
        "file": filename,
        "complexity": 42,
        "issues": ["Missing docstring", "Too many parameters"],
        "score": 7.5
    }


@Tool(description="Fetch weather information for a city")
async def get_weather(city: str) -> dict:
    """Simulate weather API call."""
    await asyncio.sleep(0.2)
    return {
        "city": city,
        "temperature": 72,
        "condition": "Sunny",
        "humidity": 45
    }


@Tool(description="Calculate mathematical expressions")
async def calculate(expression: str) -> float:
    """Safe calculator for basic math."""
    await asyncio.sleep(0.1)
    try:
        # Safe evaluation of basic math expressions
        allowed_chars = "0123456789+-*/(). "
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return float(result)
        else:
            return 0.0
    except:
        return 0.0


async def main():
    """Demonstrate AsyncOrchestrator capabilities."""
    print("🚀 AsyncOrchestrator Basic Example")
    print("=" * 50)
    
    # Create orchestrator with tools
    orchestrator = AsyncOrchestrator(
        tools=[web_search, analyze_code, get_weather, calculate],
        max_parallel=10,
        enable_speculation=False  # Keep it simple for demo
    )
    
    print(f"Registered {len(orchestrator.registry._tools)} tools")
    print(f"Tools: {list(orchestrator.registry._tools.keys())}")
    print()
    
    # Example 1: Research task
    print("📊 Example 1: Research Task")
    print("-" * 30)
    
    result1 = await orchestrator.execute(
        "I need to research Python async programming. "
        "Search for information about asyncio and analyze async patterns."
    )
    
    print(f"Execution ID: {result1['execution_id']}")
    print(f"Total time: {result1['total_time_ms']:.1f}ms")
    print(f"Tools executed: {result1['tools_executed']}")
    print(f"Successful: {result1['successful_tools']}")
    print()
    
    if result1.get('results'):
        for i, result in enumerate(result1['results']):
            print(f"Tool {i+1}: {result.tool_name}")
            print(f"  Success: {result.success}")
            print(f"  Time: {result.execution_time_ms:.1f}ms")
            print(f"  Data: {result.data}")
            print()
    
    # Example 2: Streaming execution
    print("📡 Example 2: Streaming Results")
    print("-" * 30)
    
    print("Streaming results as they complete...")
    async for partial_result in orchestrator.stream_execute(
        "Get weather for San Francisco and New York, "
        "then calculate the average temperature."
    ):
        print(f"✅ {partial_result.tool_name} completed in {partial_result.execution_time_ms:.1f}ms")
        print(f"   Result: {partial_result.data}")
    
    print()
    
    # Example 3: Show metrics
    print("📈 Example 3: Orchestrator Metrics")
    print("-" * 30)
    
    metrics = orchestrator.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    print()
    print("🎉 Demo completed!")
    
    # Cleanup
    await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())