#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK - Simple Demo
========================================

This demonstrates basic functionality with enhanced error handling
and adaptive timeout management.
"""

import asyncio
import time
from typing import Dict, Any, List

# Simple mock classes for demonstration without external dependencies
class SimpleTool:
    """Simple tool implementation for demonstration."""
    
    def __init__(self, name: str, func, description: str = ""):
        self.name = name
        self.func = func
        self.description = description
        self.call_count = 0
        self.total_time = 0.0
        
    async def execute(self, *args, **kwargs):
        """Execute the tool with basic tracking."""
        start_time = time.time()
        try:
            self.call_count += 1
            result = await self.func(*args, **kwargs)
            success = True
        except Exception as e:
            result = {"error": str(e)}
            success = False
            
        execution_time = time.time() - start_time
        self.total_time += execution_time
        
        return {
            "result": result,
            "success": success,
            "execution_time_ms": execution_time * 1000,
            "tool_name": self.name
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool execution statistics."""
        avg_time = self.total_time / self.call_count if self.call_count > 0 else 0
        return {
            "call_count": self.call_count,
            "total_time_seconds": self.total_time,
            "average_time_ms": avg_time * 1000,
            "success_rate": 1.0  # Simplified for demo
        }


class SimpleOrchestrator:
    """
    Generation 1: Simple orchestrator with enhanced features.
    
    Features:
    - Basic parallel execution
    - Adaptive timeout management
    - Enhanced execution statistics
    - Simple error recovery
    """
    
    def __init__(self, max_parallel: int = 5):
        self.max_parallel = max_parallel
        self.tools: Dict[str, SimpleTool] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.adaptive_timeouts: Dict[str, float] = {}
        self.default_timeout = 5.0
        
        print(f"🚀 Generation 1 Simple Orchestrator initialized")
        print(f"   Max parallel tools: {max_parallel}")
        
    def register_tool(self, name: str, func, description: str = ""):
        """Register a tool for orchestrated execution."""
        tool = SimpleTool(name, func, description)
        self.tools[name] = tool
        self.adaptive_timeouts[name] = self.default_timeout
        print(f"   ✅ Tool registered: {name}")
        
    async def execute_tools_parallel(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple tools in parallel with adaptive timeout management.
        
        Generation 1 Enhancement: Adaptive timeouts based on historical performance.
        """
        print(f"\n🔄 Executing {len(tool_calls)} tools in parallel...")
        
        # Create semaphore to limit parallelism
        semaphore = asyncio.Semaphore(self.max_parallel)
        
        async def execute_single_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                tool_name = tool_call["tool"]
                args = tool_call.get("args", [])
                kwargs = tool_call.get("kwargs", {})
                
                if tool_name not in self.tools:
                    return {
                        "error": f"Tool '{tool_name}' not found",
                        "success": False,
                        "tool_name": tool_name
                    }
                
                tool = self.tools[tool_name]
                timeout = self.adaptive_timeouts.get(tool_name, self.default_timeout)
                
                try:
                    # Execute with adaptive timeout
                    result = await asyncio.wait_for(
                        tool.execute(*args, **kwargs),
                        timeout=timeout
                    )
                    
                    # Adaptive timeout adjustment based on success
                    await self._adaptive_timeout_adjustment(
                        tool_name, 
                        result.get("execution_time_ms", 0),
                        result.get("success", False)
                    )
                    
                    return result
                    
                except asyncio.TimeoutError:
                    # Increase timeout for slow tools
                    self.adaptive_timeouts[tool_name] *= 1.5
                    return {
                        "error": f"Tool '{tool_name}' timed out after {timeout}s",
                        "success": False,
                        "tool_name": tool_name,
                        "timeout_adjusted": True
                    }
                except Exception as e:
                    return {
                        "error": f"Tool '{tool_name}' failed: {str(e)}",
                        "success": False,
                        "tool_name": tool_name
                    }
        
        # Execute all tools in parallel
        start_time = time.time()
        results = await asyncio.gather(*[
            execute_single_tool(call) for call in tool_calls
        ], return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Process results and track execution
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "error": str(result),
                    "success": False,
                    "tool_name": "unknown"
                })
            else:
                processed_results.append(result)
        
        # Track execution statistics
        self.execution_history.append({
            "timestamp": time.time(),
            "tool_count": len(tool_calls),
            "total_time_seconds": total_time,
            "success_count": sum(1 for r in processed_results if r.get("success", False)),
            "timeout_adjustments": sum(1 for r in processed_results if r.get("timeout_adjusted", False))
        })
        
        successful = sum(1 for r in processed_results if r.get("success", False))
        print(f"✅ Execution complete: {successful}/{len(tool_calls)} tools successful")
        print(f"   Total time: {total_time:.3f}s")
        
        return processed_results
    
    async def _adaptive_timeout_adjustment(self, tool_name: str, execution_time_ms: float, success: bool):
        """
        Generation 1 Enhancement: Adaptively adjust timeouts based on execution patterns.
        """
        current_timeout = self.adaptive_timeouts.get(tool_name, self.default_timeout)
        execution_time_s = execution_time_ms / 1000.0
        
        if success and execution_time_s < current_timeout * 0.5:
            # Tool is consistently fast, reduce timeout slightly
            self.adaptive_timeouts[tool_name] = max(1.0, current_timeout * 0.9)
        elif success and execution_time_s > current_timeout * 0.8:
            # Tool is using most of timeout, increase it
            self.adaptive_timeouts[tool_name] = min(30.0, current_timeout * 1.1)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Generation 1 Enhancement: Enhanced execution statistics.
        """
        if not self.execution_history:
            return {"message": "No executions recorded"}
        
        total_executions = len(self.execution_history)
        total_tools = sum(h["tool_count"] for h in self.execution_history)
        total_time = sum(h["total_time_seconds"] for h in self.execution_history)
        total_successes = sum(h["success_count"] for h in self.execution_history)
        
        tool_stats = {}
        for name, tool in self.tools.items():
            tool_stats[name] = tool.get_stats()
        
        return {
            "total_executions": total_executions,
            "total_tools_called": total_tools,
            "total_execution_time_seconds": total_time,
            "overall_success_rate": total_successes / total_tools if total_tools > 0 else 0,
            "average_execution_time_seconds": total_time / total_executions if total_executions > 0 else 0,
            "tool_statistics": tool_stats,
            "adaptive_timeouts": dict(self.adaptive_timeouts)
        }


# Demo tools for testing
async def fast_tool(query: str) -> str:
    """A fast tool that completes quickly."""
    await asyncio.sleep(0.1)  # 100ms
    return f"Fast result for: {query}"

async def medium_tool(data: str) -> Dict[str, Any]:
    """A medium-speed tool."""
    await asyncio.sleep(0.5)  # 500ms
    return {
        "processed_data": data.upper(),
        "processing_time": "500ms",
        "status": "complete"
    }

async def slow_tool(complex_task: str) -> Dict[str, Any]:
    """A slower tool that simulates complex processing."""
    await asyncio.sleep(1.0)  # 1000ms
    return {
        "analysis": f"Complex analysis of: {complex_task}",
        "complexity_score": 42,
        "recommendations": ["optimize", "refactor", "test"]
    }

async def error_prone_tool(input_data: str) -> str:
    """A tool that sometimes fails."""
    if "error" in input_data.lower():
        raise ValueError("Simulated error condition")
    await asyncio.sleep(0.3)
    return f"Processed: {input_data}"


async def main():
    """Demonstrate Generation 1 functionality."""
    print("=" * 60)
    print("🚀 GENERATION 1: MAKE IT WORK - SIMPLE DEMONSTRATION")
    print("=" * 60)
    
    # Create orchestrator
    orchestrator = SimpleOrchestrator(max_parallel=3)
    
    # Register tools
    orchestrator.register_tool("fast_search", fast_tool, "Fast search tool")
    orchestrator.register_tool("data_processor", medium_tool, "Data processing tool")
    orchestrator.register_tool("complex_analyzer", slow_tool, "Complex analysis tool")
    orchestrator.register_tool("error_prone", error_prone_tool, "Error-prone tool for testing")
    
    print("\n📋 Test 1: Basic parallel execution")
    tool_calls = [
        {"tool": "fast_search", "kwargs": {"query": "Python async patterns"}},
        {"tool": "data_processor", "kwargs": {"data": "sample data"}},
        {"tool": "complex_analyzer", "kwargs": {"complex_task": "code optimization"}},
    ]
    
    results = await orchestrator.execute_tools_parallel(tool_calls)
    
    print("\n📊 Results:")
    for i, result in enumerate(results):
        status = "✅" if result.get("success", False) else "❌"
        tool_name = result.get("tool_name", "unknown")
        exec_time = result.get("execution_time_ms", 0)
        print(f"  {status} {tool_name}: {exec_time:.1f}ms")
    
    print("\n📋 Test 2: Error handling and adaptive timeouts")
    tool_calls_with_errors = [
        {"tool": "error_prone", "kwargs": {"input_data": "normal data"}},
        {"tool": "error_prone", "kwargs": {"input_data": "error data"}},  # Will fail
        {"tool": "fast_search", "kwargs": {"query": "error handling"}},
        {"tool": "nonexistent_tool", "kwargs": {"data": "test"}},  # Will fail
    ]
    
    results = await orchestrator.execute_tools_parallel(tool_calls_with_errors)
    
    print("\n📊 Error Handling Results:")
    for result in results:
        status = "✅" if result.get("success", False) else "❌"
        tool_name = result.get("tool_name", "unknown")
        if not result.get("success", False):
            error = result.get("error", "Unknown error")
            print(f"  {status} {tool_name}: {error}")
        else:
            exec_time = result.get("execution_time_ms", 0)
            print(f"  {status} {tool_name}: {exec_time:.1f}ms")
    
    print("\n📈 GENERATION 1 EXECUTION STATISTICS")
    print("=" * 50)
    stats = orchestrator.get_execution_stats()
    
    print(f"Total Executions: {stats['total_executions']}")
    print(f"Total Tools Called: {stats['total_tools_called']}")
    print(f"Overall Success Rate: {stats['overall_success_rate']:.1%}")
    print(f"Average Execution Time: {stats['average_execution_time_seconds']:.3f}s")
    
    print(f"\n🔧 Adaptive Timeouts:")
    for tool, timeout in stats['adaptive_timeouts'].items():
        print(f"  {tool}: {timeout:.1f}s")
    
    print(f"\n📊 Tool Statistics:")
    for tool_name, tool_stats in stats['tool_statistics'].items():
        calls = tool_stats['call_count']
        avg_time = tool_stats['average_time_ms']
        print(f"  {tool_name}: {calls} calls, {avg_time:.1f}ms avg")
    
    print("\n✅ Generation 1: MAKE IT WORK - COMPLETED SUCCESSFULLY")
    print("   Key Features Demonstrated:")
    print("   • Basic parallel tool execution")
    print("   • Adaptive timeout management")
    print("   • Enhanced execution statistics")
    print("   • Error recovery and handling")
    print("   • Resource management with semaphores")


if __name__ == "__main__":
    asyncio.run(main())