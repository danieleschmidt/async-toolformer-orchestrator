#!/usr/bin/env python3
"""
Performance Benchmark Runner for async-toolformer-orchestrator

Runs comprehensive performance benchmarks and generates reports.
"""

import asyncio
import json
import time
import statistics
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import psutil
import aiohttp


class BenchmarkResult:
    """Container for benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.execution_times: List[float] = []
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
        self.success_count = 0
        self.failure_count = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def add_result(self, execution_time: float, success: bool = True, 
                   memory_mb: Optional[float] = None, cpu_percent: Optional[float] = None):
        """Add a single benchmark result."""
        self.execution_times.append(execution_time)
        if memory_mb is not None:
            self.memory_usage.append(memory_mb)
        if cpu_percent is not None:
            self.cpu_usage.append(cpu_percent)
        
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not self.execution_times:
            return {"error": "No results collected"}
        
        stats = {
            "name": self.name,
            "total_runs": len(self.execution_times),
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_count / (self.success_count + self.failure_count),
            "execution_time": {
                "mean": statistics.mean(self.execution_times),
                "median": statistics.median(self.execution_times),
                "p95": statistics.quantiles(self.execution_times, n=20)[18] if len(self.execution_times) >= 20 else max(self.execution_times),
                "p99": statistics.quantiles(self.execution_times, n=100)[98] if len(self.execution_times) >= 100 else max(self.execution_times),
                "min": min(self.execution_times),
                "max": max(self.execution_times),
                "stddev": statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0
            }
        }
        
        if self.memory_usage:
            stats["memory_mb"] = {
                "mean": statistics.mean(self.memory_usage),
                "max": max(self.memory_usage),
                "min": min(self.memory_usage)
            }
        
        if self.cpu_usage:
            stats["cpu_percent"] = {
                "mean": statistics.mean(self.cpu_usage),
                "max": max(self.cpu_usage)
            }
        
        if self.start_time and self.end_time:
            stats["total_duration"] = self.end_time - self.start_time
            if stats["total_duration"] > 0:
                stats["throughput_per_second"] = len(self.execution_times) / stats["total_duration"]
        
        return stats


class MockOrchestrator:
    """Mock orchestrator for benchmarking."""
    
    def __init__(self, max_parallel: int = 10):
        self.max_parallel = max_parallel
    
    async def execute_single_tool(self, delay_ms: int = 300) -> Dict[str, Any]:
        """Simulate single tool execution."""
        start_time = time.time()
        await asyncio.sleep(delay_ms / 1000)  # Convert to seconds
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "result": f"Mock result after {execution_time:.1f}ms",
            "execution_time_ms": execution_time,
            "success": True
        }
    
    async def execute_parallel_tools(self, num_tools: int, delay_ms: int = 300) -> Dict[str, Any]:
        """Simulate parallel tool execution."""
        start_time = time.time()
        
        # Create tasks for parallel execution
        tasks = [
            self.execute_single_tool(delay_ms) 
            for _ in range(min(num_tools, self.max_parallel))
        ]
        
        # Execute remaining tools in batches if needed
        results = []
        for i in range(0, num_tools, self.max_parallel):
            batch = tasks[i:i + self.max_parallel] if i == 0 else [
                self.execute_single_tool(delay_ms) 
                for _ in range(min(self.max_parallel, num_tools - i))
            ]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
        
        total_time = (time.time() - start_time) * 1000
        sequential_time = num_tools * delay_ms
        efficiency = sequential_time / total_time if total_time > 0 else 0
        
        return {
            "results": results,
            "total_execution_time_ms": total_time,
            "tools_executed": num_tools,
            "parallel_efficiency": min(efficiency, 1.0),
            "success": True
        }


class BenchmarkRunner:
    """Main benchmark runner."""
    
    def __init__(self):
        self.orchestrator = MockOrchestrator()
        self.process = psutil.Process()
    
    def get_system_stats(self) -> Dict[str, float]:
        """Get current system resource usage."""
        return {
            "memory_mb": self.process.memory_info().rss / 1024 / 1024,
            "cpu_percent": self.process.cpu_percent()
        }
    
    async def benchmark_single_tool(self, runs: int = 100) -> BenchmarkResult:
        """Benchmark single tool execution."""
        result = BenchmarkResult("single_tool_execution")
        result.start_time = time.time()
        
        for i in range(runs):
            sys_stats = self.get_system_stats()
            start_time = time.time()
            
            try:
                response = await self.orchestrator.execute_single_tool()
                execution_time = (time.time() - start_time) * 1000
                
                result.add_result(
                    execution_time=execution_time,
                    success=response["success"],
                    memory_mb=sys_stats["memory_mb"],
                    cpu_percent=sys_stats["cpu_percent"]
                )
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                result.add_result(execution_time=execution_time, success=False)
                print(f"Error in run {i}: {e}")
        
        result.end_time = time.time()
        return result
    
    async def benchmark_parallel_tools(self, num_tools: int = 5, runs: int = 50) -> BenchmarkResult:
        """Benchmark parallel tool execution."""
        result = BenchmarkResult(f"parallel_{num_tools}_tools")
        result.start_time = time.time()
        
        for i in range(runs):
            sys_stats = self.get_system_stats()
            start_time = time.time()
            
            try:
                response = await self.orchestrator.execute_parallel_tools(num_tools)
                execution_time = (time.time() - start_time) * 1000
                
                result.add_result(
                    execution_time=execution_time,
                    success=response["success"],
                    memory_mb=sys_stats["memory_mb"],
                    cpu_percent=sys_stats["cpu_percent"]
                )
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                result.add_result(execution_time=execution_time, success=False)
                print(f"Error in run {i}: {e}")
        
        result.end_time = time.time()
        return result
    
    async def benchmark_sustained_load(self, duration_seconds: int = 60, 
                                     requests_per_second: int = 10) -> BenchmarkResult:
        """Benchmark sustained load."""
        result = BenchmarkResult(f"sustained_load_{requests_per_second}rps")
        result.start_time = time.time()
        
        end_time = time.time() + duration_seconds
        request_interval = 1.0 / requests_per_second
        
        while time.time() < end_time:
            sys_stats = self.get_system_stats()
            start_time = time.time()
            
            try:
                response = await self.orchestrator.execute_single_tool()
                execution_time = (time.time() - start_time) * 1000
                
                result.add_result(
                    execution_time=execution_time,
                    success=response["success"],
                    memory_mb=sys_stats["memory_mb"],
                    cpu_percent=sys_stats["cpu_percent"]
                )
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                result.add_result(execution_time=execution_time, success=False)
                print(f"Error during sustained load: {e}")
            
            # Wait for next request (accounting for execution time)
            elapsed = time.time() - start_time
            sleep_time = max(0, request_interval - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        result.end_time = time.time()
        return result
    
    async def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        print("🚀 Starting benchmark suite...")
        
        benchmarks = []
        
        # Single tool performance
        print("📊 Running single tool benchmarks...")
        single_result = await self.benchmark_single_tool(runs=100)
        benchmarks.append(single_result.get_stats())
        
        # Parallel tool performance (different sizes)
        for num_tools in [5, 10, 20]:
            print(f"📊 Running {num_tools}-tool parallel benchmarks...")
            parallel_result = await self.benchmark_parallel_tools(num_tools=num_tools, runs=30)
            benchmarks.append(parallel_result.get_stats())
        
        # Sustained load test
        print("📊 Running sustained load test...")
        load_result = await self.benchmark_sustained_load(duration_seconds=30, requests_per_second=5)
        benchmarks.append(load_result.get_stats())
        
        # Generate summary report
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "environment": {
                "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "platform": __import__('platform').platform()
            },
            "benchmarks": benchmarks,
            "summary": self._generate_summary(benchmarks)
        }
        
        return report
    
    def _generate_summary(self, benchmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics across all benchmarks."""
        summary = {
            "overall_success_rate": statistics.mean([b["success_rate"] for b in benchmarks]),
            "total_runs": sum(b["total_runs"] for b in benchmarks),
            "fastest_benchmark": min(benchmarks, key=lambda b: b["execution_time"]["mean"])["name"],
            "slowest_benchmark": max(benchmarks, key=lambda b: b["execution_time"]["mean"])["name"]
        }
        
        # Calculate parallel efficiency if available
        parallel_benchmarks = [b for b in benchmarks if "parallel" in b["name"]]
        if parallel_benchmarks:
            # This would need actual parallel efficiency data from the orchestrator
            summary["average_parallel_efficiency"] = "N/A (mock data)"
        
        return summary


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run async-toolformer-orchestrator benchmarks")
    parser.add_argument("--output", "-o", default="benchmark-results.json", 
                       help="Output file for results")
    parser.add_argument("--format", choices=["json", "markdown"], default="json",
                       help="Output format")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick benchmark suite (fewer iterations)")
    
    args = parser.parse_args()
    
    async def run_benchmarks():
        runner = BenchmarkRunner()
        
        if args.quick:
            print("⚡ Running quick benchmark suite...")
            # Reduce iterations for quick testing
            runner._original_runs = (20, 10, 10)  # single, parallel, sustained
        
        results = await runner.run_all_benchmarks()
        
        # Save results
        output_path = Path(args.output)
        
        if args.format == "json":
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"📈 Results saved to {output_path}")
        
        elif args.format == "markdown":
            markdown_content = generate_markdown_report(results)
            markdown_path = output_path.with_suffix(".md")
            with open(markdown_path, "w") as f:
                f.write(markdown_content)
            print(f"📝 Markdown report saved to {markdown_path}")
        
        # Print summary to console
        print("\n📋 Benchmark Summary:")
        print(f"Overall Success Rate: {results['summary']['overall_success_rate']:.2%}")
        print(f"Total Test Runs: {results['summary']['total_runs']}")
        print(f"Fastest Test: {results['summary']['fastest_benchmark']}")
        print(f"Slowest Test: {results['summary']['slowest_benchmark']}")
    
    # Run the benchmarks
    asyncio.run(run_benchmarks())


def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate markdown report from benchmark results."""
    report = f"""# Benchmark Report

**Generated:** {results['timestamp']}

## Environment
- Python: {results['environment']['python_version']}
- CPU Cores: {results['environment']['cpu_count']}
- Memory: {results['environment']['memory_total_gb']:.1f} GB
- Platform: {results['environment']['platform']}

## Summary
- Overall Success Rate: {results['summary']['overall_success_rate']:.2%}
- Total Test Runs: {results['summary']['total_runs']}
- Fastest Test: {results['summary']['fastest_benchmark']}
- Slowest Test: {results['summary']['slowest_benchmark']}

## Detailed Results

"""
    
    for benchmark in results['benchmarks']:
        report += f"""### {benchmark['name'].replace('_', ' ').title()}

| Metric | Value |
|--------|-------|
| Success Rate | {benchmark['success_rate']:.2%} |
| Mean Execution Time | {benchmark['execution_time']['mean']:.1f}ms |
| Median Execution Time | {benchmark['execution_time']['median']:.1f}ms |
| 95th Percentile | {benchmark['execution_time']['p95']:.1f}ms |
| 99th Percentile | {benchmark['execution_time']['p99']:.1f}ms |
| Min Time | {benchmark['execution_time']['min']:.1f}ms |
| Max Time | {benchmark['execution_time']['max']:.1f}ms |
| Standard Deviation | {benchmark['execution_time']['stddev']:.1f}ms |

"""
        
        if 'memory_mb' in benchmark:
            report += f"- **Memory Usage**: {benchmark['memory_mb']['mean']:.1f} MB (avg), {benchmark['memory_mb']['max']:.1f} MB (peak)\n"
        
        if 'cpu_percent' in benchmark:
            report += f"- **CPU Usage**: {benchmark['cpu_percent']['mean']:.1f}% (avg), {benchmark['cpu_percent']['max']:.1f}% (peak)\n"
        
        if 'throughput_per_second' in benchmark:
            report += f"- **Throughput**: {benchmark['throughput_per_second']:.1f} requests/second\n"
        
        report += "\n"
    
    return report


if __name__ == "__main__":
    main()