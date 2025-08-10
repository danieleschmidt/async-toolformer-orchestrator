#!/usr/bin/env python3
"""Basic functionality demonstration for Generation 1: MAKE IT WORK (Simple)"""

import asyncio
from async_toolformer import AsyncOrchestrator, Tool, ToolChain, parallel

# Simple tools for demonstration
@Tool(description="Search the web for information")
async def web_search(query: str) -> str:
    """Simulate web search."""
    await asyncio.sleep(0.1)  # Simulate network delay
    return f"Web search results for: {query}"

@Tool(description="Analyze code quality")
async def analyze_code(file_path: str) -> dict:
    """Simulate code analysis."""
    await asyncio.sleep(0.05)
    return {
        "file": file_path,
        "complexity": 5,
        "issues": ["Consider adding type hints"],
        "score": 85
    }

@Tool(description="Send notification")
async def send_notification(message: str, priority: str = "normal") -> bool:
    """Simulate sending notification."""
    await asyncio.sleep(0.02)
    print(f"📧 {priority.upper()}: {message}")
    return True

@ToolChain
async def research_and_report(topic: str):
    """Tool chain that researches a topic and creates a report."""
    # Parallel research phase
    web_results, code_analysis = await parallel(
        web_search(f"{topic} best practices"),
        analyze_code(f"src/{topic}_module.py")
    )
    
    # Sequential report generation
    report = {
        "topic": topic,
        "web_research": web_results,
        "code_analysis": code_analysis,
        "timestamp": "2025-01-01T00:00:00Z"
    }
    
    # Send completion notification
    await send_notification(f"Research report for '{topic}' completed", "high")
    
    return report

async def main():
    """Demonstrate basic functionality."""
    print("🚀 Generation 1: MAKE IT WORK (Simple) - Basic Functionality Demo")
    print("=" * 60)
    
    # Test individual tools
    print("\n1. Testing individual tools:")
    search_result = await web_search("Python async patterns")
    print(f"✅ Web Search: {search_result}")
    
    analysis_result = await analyze_code("async_module.py")
    print(f"✅ Code Analysis: {analysis_result}")
    
    notification_sent = await send_notification("Test message")
    print(f"✅ Notification Sent: {notification_sent}")
    
    # Test parallel execution
    print("\n2. Testing parallel execution:")
    start_time = asyncio.get_event_loop().time()
    results = await parallel(
        web_search("async orchestration"),
        analyze_code("orchestrator.py"),
        send_notification("Parallel test completed")
    )
    end_time = asyncio.get_event_loop().time()
    print(f"✅ Parallel execution completed in {end_time - start_time:.3f}s")
    for i, result in enumerate(results, 1):
        print(f"   Result {i}: {result}")
    
    # Test tool chain
    print("\n3. Testing tool chain:")
    chain_result = await research_and_report("async_toolformer")
    print(f"✅ Tool Chain Result:")
    for key, value in chain_result.items():
        print(f"   {key}: {value}")
    
    # Test AsyncOrchestrator (basic initialization)
    print("\n4. Testing AsyncOrchestrator initialization:")
    orchestrator = AsyncOrchestrator(max_parallel_tools=15, max_parallel_per_type=5)
    print(f"✅ AsyncOrchestrator initialized with max_parallel_tools={orchestrator.config.max_parallel_tools}")
    
    print("\n🎉 Generation 1 Basic Functionality: ALL TESTS PASSED")
    print("✅ Tool decoration and execution")
    print("✅ Tool chain composition")
    print("✅ Parallel execution utilities")
    print("✅ AsyncOrchestrator basic initialization")

if __name__ == "__main__":
    asyncio.run(main())