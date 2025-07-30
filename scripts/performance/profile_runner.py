#!/usr/bin/env python3
"""Performance profiling runner for the Async Toolformer Orchestrator."""

import asyncio
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from async_toolformer import AsyncOrchestrator, Tool, OrchestratorConfig
from async_toolformer.profiling import AsyncProfiler, PerformanceOptimizer, get_profiler
from async_toolformer.metrics import initialize_observability

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Example tools for profiling
@Tool(description="CPU-intensive computation tool")
async def cpu_intensive_tool(iterations: int = 1000000) -> Dict[str, Any]:
    """Simulate CPU-intensive work."""
    result = 0
    for i in range(iterations):
        result += i ** 0.5
    
    await asyncio.sleep(0.001)  # Small async operation
    return {"result": result, "iterations": iterations}


@Tool(description="Memory-intensive tool")
async def memory_intensive_tool(size_mb: int = 10) -> Dict[str, Any]:
    """Simulate memory-intensive work."""
    # Allocate large data structure
    data = [i for i in range(size_mb * 1024 * 100)]  # Rough MB calculation
    
    # Process data
    processed = [x * 2 for x in data[-1000:]]  # Process last 1000 items
    
    await asyncio.sleep(0.01)
    
    return {
        "data_size": len(data),
        "processed_size": len(processed),
        "sample_result": sum(processed[:10])
    }


@Tool(description="IO simulation tool")
async def io_simulation_tool(delay_ms: int = 100) -> Dict[str, Any]:
    """Simulate IO-bound work."""
    start_time = asyncio.get_event_loop().time()
    
    # Simulate network/disk IO
    await asyncio.sleep(delay_ms / 1000.0)
    
    end_time = asyncio.get_event_loop().time()
    actual_delay = (end_time - start_time) * 1000
    
    return {
        "requested_delay_ms": delay_ms,
        "actual_delay_ms": actual_delay,
        "accuracy": abs(delay_ms - actual_delay) / delay_ms * 100
    }


@Tool(description="Mixed workload tool")
async def mixed_workload_tool(cpu_work: int = 10000, io_delay: int = 50) -> Dict[str, Any]:
    """Tool with mixed CPU and IO work."""
    # CPU work
    cpu_result = sum(i ** 2 for i in range(cpu_work))
    
    # IO work
    await asyncio.sleep(io_delay / 1000.0)
    
    # More CPU work
    final_result = cpu_result % 1000000
    
    return {
        "cpu_work": cpu_work,
        "io_delay": io_delay,
        "final_result": final_result
    }


class ProfileRunner:
    """Main profiling runner."""
    
    def __init__(self, config_overrides: Dict[str, Any] = None):
        self.config_overrides = config_overrides or {}
        self.profiler = get_profiler()
        self.optimizer = PerformanceOptimizer()
        self.results: List[Dict[str, Any]] = []
    
    def create_orchestrator(self, test_config: Dict[str, Any]) -> AsyncOrchestrator:
        """Create orchestrator with test configuration."""
        config = OrchestratorConfig(
            max_parallel_tools=test_config.get('max_parallel', 10),
            tool_timeout_ms=test_config.get('tool_timeout_ms', 30000),
            total_timeout_ms=test_config.get('total_timeout_ms', 60000),
        )
        
        # Apply overrides
        for key, value in self.config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        orchestrator = AsyncOrchestrator(config=config)
        
        # Register test tools
        orchestrator.register_tool(cpu_intensive_tool)
        orchestrator.register_tool(memory_intensive_tool)
        orchestrator.register_tool(io_simulation_tool)
        orchestrator.register_tool(mixed_workload_tool)
        
        return orchestrator
    
    async def run_basic_performance_test(self) -> Dict[str, Any]:
        """Run basic performance test."""
        logger.info("Running basic performance test...")
        
        config = {
            'max_parallel': 5,
            'tool_timeout_ms': 10000,
            'total_timeout_ms': 30000,
        }
        
        orchestrator = self.create_orchestrator(config)
        
        async with self.profiler.profile("basic_test"):
            result = await orchestrator.execute(
                "Run basic performance test with mixed workloads"
            )
        
        profile_result = self.profiler.get_profile("basic_test")
        recommendations = self.optimizer.analyze_profile(profile_result)
        
        return {
            "test_name": "basic_performance",
            "config": config,
            "execution_result": result,
            "performance_metrics": profile_result.metrics.to_dict(),
            "recommendations": recommendations,
            "top_functions": profile_result.get_top_functions(5),
        }
    
    async def run_parallel_scaling_test(self) -> Dict[str, Any]:
        """Test performance with different parallel limits."""
        logger.info("Running parallel scaling test...")
        
        scaling_results = []
        parallel_limits = [1, 2, 5, 10, 20, 50]
        
        for limit in parallel_limits:
            logger.info(f"Testing with parallel limit: {limit}")
            
            config = {
                'max_parallel': limit,
                'tool_timeout_ms': 5000,
                'total_timeout_ms': 20000,
            }
            
            orchestrator = self.create_orchestrator(config)
            test_name = f"parallel_scaling_{limit}"
            
            async with self.profiler.profile(test_name):
                result = await orchestrator.execute(
                    f"Test parallel scaling with limit {limit}"
                )
            
            profile_result = self.profiler.get_profile(test_name)
            
            scaling_results.append({
                "parallel_limit": limit,
                "execution_time": profile_result.metrics.execution_time,
                "cpu_usage": profile_result.metrics.cpu_usage,
                "memory_usage": profile_result.metrics.memory_usage.get('current_mb', 0),
                "successful_tools": result.get('successful_tools', 0),
                "tools_executed": result.get('tools_executed', 0),
            })
        
        return {
            "test_name": "parallel_scaling",
            "results": scaling_results,
            "optimal_parallel": self._find_optimal_parallel(scaling_results),
        }
    
    async def run_memory_stress_test(self) -> Dict[str, Any]:
        """Test memory usage under stress."""
        logger.info("Running memory stress test...")
        
        config = {
            'max_parallel': 10,
            'tool_timeout_ms': 15000,
            'total_timeout_ms': 45000,
        }
        
        orchestrator = self.create_orchestrator(config)
        
        # Run with different memory loads
        memory_results = []
        memory_sizes = [1, 5, 10, 25, 50]  # MB
        
        for size in memory_sizes:
            test_name = f"memory_stress_{size}mb"
            logger.info(f"Testing memory usage with {size}MB allocation")
            
            async with self.profiler.profile(test_name):
                # Override tool behavior for this test
                result = await orchestrator.execute(
                    f"Memory stress test with {size}MB allocation"
                )
            
            profile_result = self.profiler.get_profile(test_name)
            memory_hotspots = profile_result.get_memory_hotspots(3)
            
            memory_results.append({
                "memory_size_mb": size,
                "execution_time": profile_result.metrics.execution_time,
                "peak_memory_mb": profile_result.metrics.memory_usage.get('peak_mb', 0),
                "current_memory_mb": profile_result.metrics.memory_usage.get('current_mb', 0),
                "memory_hotspots": memory_hotspots,
            })
        
        return {
            "test_name": "memory_stress",
            "results": memory_results,
            "memory_recommendations": self.optimizer.analyze_profile(
                self.profiler.get_profile(f"memory_stress_{memory_sizes[-1]}mb")
            ),
        }
    
    async def run_latency_benchmark(self) -> Dict[str, Any]:
        """Benchmark latency characteristics."""
        logger.info("Running latency benchmark...")
        
        config = {
            'max_parallel': 1,  # Sequential for accurate latency measurement
            'tool_timeout_ms': 5000,
            'total_timeout_ms': 15000,
        }
        
        orchestrator = self.create_orchestrator(config)
        
        # Test different IO delays
        latency_results = []
        io_delays = [10, 50, 100, 250, 500, 1000]  # milliseconds
        
        for delay in io_delays:
            test_name = f"latency_benchmark_{delay}ms"
            logger.info(f"Testing latency with {delay}ms IO delay")
            
            # Run multiple iterations for statistical accuracy
            times = []
            for iteration in range(5):
                async with self.profiler.profile(f"{test_name}_iter_{iteration}"):
                    result = await orchestrator.execute(
                        f"Latency test with {delay}ms delay"
                    )
                
                profile_result = self.profiler.get_profile(f"{test_name}_iter_{iteration}")
                times.append(profile_result.metrics.execution_time)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            latency_results.append({
                "io_delay_ms": delay,
                "avg_execution_time": avg_time,
                "min_execution_time": min_time,
                "max_execution_time": max_time,
                "overhead_ms": (avg_time * 1000) - delay,
                "iterations": len(times),
            })
        
        return {
            "test_name": "latency_benchmark",
            "results": latency_results,
            "latency_analysis": self._analyze_latency(latency_results),
        }
    
    def _find_optimal_parallel(self, scaling_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find optimal parallel limit from scaling results."""
        # Calculate efficiency (successful tools per second per CPU%)
        best_efficiency = 0
        optimal_config = None
        
        for result in scaling_results:
            if result['cpu_usage'] > 0 and result['execution_time'] > 0:
                efficiency = (
                    result['successful_tools'] / result['execution_time']
                ) / result['cpu_usage'] * 100
                
                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    optimal_config = result
        
        return {
            "optimal_parallel_limit": optimal_config['parallel_limit'] if optimal_config else 10,
            "efficiency_score": best_efficiency,
            "reasoning": "Based on successful tools per second per CPU percentage",
        }
    
    def _analyze_latency(self, latency_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze latency benchmark results."""
        overheads = [r['overhead_ms'] for r in latency_results]
        avg_overhead = sum(overheads) / len(overheads)
        
        return {
            "average_overhead_ms": avg_overhead,
            "overhead_trend": "increasing" if overheads[-1] > overheads[0] else "stable",
            "recommendation": (
                "Low overhead - good for latency-sensitive applications"
                if avg_overhead < 10 else
                "Consider optimization for latency-sensitive workloads"
            ),
        }
    
    async def generate_comprehensive_report(self, output_file: str) -> None:
        """Generate comprehensive performance report."""
        logger.info("Generating comprehensive performance report...")
        
        # Run all tests
        basic_result = await self.run_basic_performance_test()
        scaling_result = await self.run_parallel_scaling_test()
        memory_result = await self.run_memory_stress_test()
        latency_result = await self.run_latency_benchmark()
        
        # Get system recommendations
        system_recommendations = self.optimizer.get_system_recommendations()
        
        # Compile comprehensive report
        report = {
            "profile_summary": {
                "timestamp": asyncio.get_event_loop().time(),
                "total_tests": 4,
                "total_profiles": len(self.profiler.get_all_profiles()),
            },
            "test_results": {
                "basic_performance": basic_result,
                "parallel_scaling": scaling_result,
                "memory_stress": memory_result,
                "latency_benchmark": latency_result,
            },
            "system_recommendations": system_recommendations,
            "overall_recommendations": self._generate_overall_recommendations([
                basic_result, scaling_result, memory_result, latency_result
            ]),
        }
        
        # Save report
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive report saved to: {output_path}")
        
        # Also generate text report
        text_report = self._generate_text_report(report)
        text_path = output_path.with_suffix('.txt')
        text_path.write_text(text_report)
        
        logger.info(f"Text report saved to: {text_path}")
    
    def _generate_overall_recommendations(self, test_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate overall recommendations from all test results."""
        recommendations = []
        
        # Extract optimal parallel limit
        scaling_result = next((r for r in test_results if r.get('test_name') == 'parallel_scaling'), None)
        if scaling_result:
            optimal_parallel = scaling_result.get('optimal_parallel', {}).get('optimal_parallel_limit', 10)
            recommendations.append({
                'type': 'configuration',
                'priority': 'high',
                'message': f'Use optimal parallel limit of {optimal_parallel} for best performance',
                'impact': 'performance improvement',
            })
        
        # Memory usage recommendations
        memory_result = next((r for r in test_results if r.get('test_name') == 'memory_stress'), None)
        if memory_result:
            max_memory = max(r['peak_memory_mb'] for r in memory_result['results'])
            if max_memory > 500:  # > 500MB
                recommendations.append({
                    'type': 'memory',
                    'priority': 'high',
                    'message': f'Peak memory usage of {max_memory:.1f}MB detected',
                    'impact': 'potential memory issues under load',
                })
        
        return recommendations
    
    def _generate_text_report(self, report: Dict[str, Any]) -> str:
        """Generate human-readable text report."""
        lines = [
            "Async Toolformer Orchestrator - Performance Report",
            "=" * 55,
            "",
            f"Report generated at: {report['profile_summary']['timestamp']}",
            f"Total tests run: {report['profile_summary']['total_tests']}",
            f"Total profiles created: {report['profile_summary']['total_profiles']}",
            "",
            "EXECUTIVE SUMMARY",
            "-" * 20,
        ]
        
        # Add overall recommendations
        for rec in report.get('overall_recommendations', []):
            lines.append(f"• {rec['message']} ({rec['priority']} priority)")
        
        lines.extend([
            "",
            "DETAILED RESULTS",
            "-" * 20,
        ])
        
        # Add test results summary
        for test_name, test_result in report['test_results'].items():
            lines.append(f"\n{test_name.upper()}:")
            
            if test_name == 'basic_performance':
                metrics = test_result['performance_metrics']
                lines.extend([
                    f"  Execution time: {metrics['execution_time']:.3f}s",
                    f"  CPU usage: {metrics['cpu_usage']:.1f}%",
                    f"  Memory usage: {metrics.get('memory_usage', {}).get('current_mb', 0):.1f}MB",
                ])
            
            elif test_name == 'parallel_scaling':
                optimal = test_result.get('optimal_parallel', {})
                lines.extend([
                    f"  Optimal parallel limit: {optimal.get('optimal_parallel_limit', 'N/A')}",
                    f"  Efficiency score: {optimal.get('efficiency_score', 0):.2f}",
                ])
        
        return "\n".join(lines)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Profile Async Toolformer Orchestrator")
    parser.add_argument(
        '--test', 
        choices=['basic', 'scaling', 'memory', 'latency', 'all'],
        default='all',
        help='Type of test to run'
    )
    parser.add_argument(
        '--output', 
        default='performance_report.json',
        help='Output file for the report'
    )
    parser.add_argument(
        '--max-parallel',
        type=int,
        help='Override max parallel tools'
    )
    parser.add_argument(
        '--enable-metrics',
        action='store_true',
        help='Enable metrics collection'
    )
    
    args = parser.parse_args()
    
    # Initialize observability if requested
    if args.enable_metrics:
        initialize_observability(enable_metrics=True, enable_tracing=False)
    
    # Setup config overrides
    config_overrides = {}
    if args.max_parallel:
        config_overrides['max_parallel_tools'] = args.max_parallel
    
    # Create runner
    runner = ProfileRunner(config_overrides)
    
    try:
        if args.test == 'all':
            await runner.generate_comprehensive_report(args.output)
        elif args.test == 'basic':
            result = await runner.run_basic_performance_test()
            print(json.dumps(result, indent=2, default=str))
        elif args.test == 'scaling':
            result = await runner.run_parallel_scaling_test()
            print(json.dumps(result, indent=2, default=str))
        elif args.test == 'memory':
            result = await runner.run_memory_stress_test()
            print(json.dumps(result, indent=2, default=str))
        elif args.test == 'latency':
            result = await runner.run_latency_benchmark()
            print(json.dumps(result, indent=2, default=str))
        
        logger.info("Profiling completed successfully")
        
    except Exception as e:
        logger.error(f"Profiling failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())