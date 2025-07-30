"""Performance profiling and optimization utilities."""

import asyncio
import cProfile
import pstats
import time
import tracemalloc
import psutil
import gc
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from functools import wraps
import logging
import threading
from pathlib import Path

try:
    import uvloop
    UVLOOP_AVAILABLE = True
except ImportError:
    UVLOOP_AVAILABLE = False

try:
    import memory_profiler
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

from .metrics import get_metrics_collector, track_metric

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    execution_time: float
    memory_usage: Dict[str, int] = field(default_factory=dict)
    cpu_usage: float = 0.0
    gc_collections: Dict[str, int] = field(default_factory=dict)
    async_task_count: int = 0
    context_switches: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'execution_time': self.execution_time,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'gc_collections': self.gc_collections,
            'async_task_count': self.async_task_count,
            'context_switches': self.context_switches,
        }


@dataclass
class ProfilingResult:
    """Result from profiling session."""
    
    function_name: str
    metrics: PerformanceMetrics
    profile_data: Optional[pstats.Stats] = None
    memory_trace: Optional[tracemalloc.Snapshot] = None
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_top_functions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get top functions by execution time."""
        if not self.profile_data:
            return []
        
        stats = []
        for func, (cc, nc, tt, ct, callers) in self.profile_data.stats.items():
            stats.append({
                'function': f"{func[0]}:{func[1]}({func[2]})",
                'calls': nc,
                'total_time': tt,
                'cumulative_time': ct,
                'per_call': tt / nc if nc > 0 else 0,
            })
        
        return sorted(stats, key=lambda x: x['total_time'], reverse=True)[:count]
    
    def get_memory_hotspots(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get memory allocation hotspots."""
        if not self.memory_trace:
            return []
        
        top_stats = self.memory_trace.statistics('lineno')
        hotspots = []
        
        for stat in top_stats[:count]:
            hotspots.append({
                'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count,
            })
        
        return hotspots


class AsyncProfiler:
    """Profiler for async functions and coroutines."""
    
    def __init__(self, enable_memory_tracing: bool = True):
        self.enable_memory_tracing = enable_memory_tracing
        self._profiles: Dict[str, ProfilingResult] = {}
        self._active_profiles: Dict[str, Dict[str, Any]] = {}
        
        if enable_memory_tracing:
            try:
                tracemalloc.start()
            except RuntimeError:
                pass  # Already started
    
    @asynccontextmanager
    async def profile(self, name: str):
        """Context manager for profiling async code."""
        # Start profiling
        profiler = cProfile.Profile()
        start_time = time.time()
        
        # Memory tracking
        if self.enable_memory_tracing:
            tracemalloc.start()
            gc.collect()  # Clean start
            start_memory = tracemalloc.get_traced_memory()
        
        # System metrics
        process = psutil.Process()
        start_cpu_times = process.cpu_times()
        start_ctx_switches = process.num_ctx_switches()
        
        profiler.enable()
        
        try:
            yield
        finally:
            profiler.disable()
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Collect metrics
            end_cpu_times = process.cpu_times()
            end_ctx_switches = process.num_ctx_switches()
            
            cpu_usage = (
                (end_cpu_times.user - start_cpu_times.user) + 
                (end_cpu_times.system - start_cpu_times.system)
            ) / execution_time * 100
            
            context_switches = (
                end_ctx_switches.voluntary - start_ctx_switches.voluntary +
                end_ctx_switches.involuntary - start_ctx_switches.involuntary
            )
            
            # Memory metrics
            memory_usage = {}
            memory_trace = None
            if self.enable_memory_tracing:
                end_memory = tracemalloc.get_traced_memory()
                memory_usage = {
                    'current_mb': end_memory[0] / 1024 / 1024,
                    'peak_mb': end_memory[1] / 1024 / 1024,
                    'allocated_mb': (end_memory[0] - start_memory[0]) / 1024 / 1024,
                }
                memory_trace = tracemalloc.take_snapshot()
            
            # GC stats
            gc_stats = {f'gen_{i}': gc.get_count()[i] for i in range(3)}
            
            # Async task count
            try:
                current_task = asyncio.current_task()
                all_tasks = asyncio.all_tasks()
                task_count = len(all_tasks)
            except RuntimeError:
                task_count = 0
            
            # Create metrics
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_usage=cpu_usage,
                gc_collections=gc_stats,
                async_task_count=task_count,
                context_switches=context_switches,
            )
            
            # Create profiling result
            stats = pstats.Stats(profiler)
            result = ProfilingResult(
                function_name=name,
                metrics=metrics,
                profile_data=stats,
                memory_trace=memory_trace,
            )
            
            self._profiles[name] = result
            
            # Log performance metrics
            logger.info(
                f"Profile '{name}': {execution_time:.3f}s, "
                f"CPU: {cpu_usage:.1f}%, "
                f"Memory: {memory_usage.get('current_mb', 0):.1f}MB"
            )
    
    def get_profile(self, name: str) -> Optional[ProfilingResult]:
        """Get profiling result by name."""
        return self._profiles.get(name)
    
    def get_all_profiles(self) -> Dict[str, ProfilingResult]:
        """Get all profiling results."""
        return self._profiles.copy()
    
    def clear_profiles(self) -> None:
        """Clear all stored profiles."""
        self._profiles.clear()
    
    def generate_report(self, name: str, output_file: Optional[str] = None) -> str:
        """Generate detailed performance report."""
        result = self._profiles.get(name)
        if not result:
            return f"No profile found for '{name}'"
        
        report_lines = [
            f"Performance Report for '{name}'",
            "=" * 50,
            f"Execution Time: {result.metrics.execution_time:.3f}s",
            f"CPU Usage: {result.metrics.cpu_usage:.1f}%",
            f"Context Switches: {result.metrics.context_switches}",
            f"Async Tasks: {result.metrics.async_task_count}",
            "",
            "Memory Usage:",
        ]
        
        for key, value in result.metrics.memory_usage.items():
            report_lines.append(f"  {key}: {value:.2f} MB")
        
        report_lines.extend([
            "",
            "Top Functions by Time:",
            "-" * 30,
        ])
        
        for func in result.get_top_functions():
            report_lines.append(
                f"  {func['function']}: {func['total_time']:.3f}s "
                f"({func['calls']} calls, {func['per_call']:.4f}s/call)"
            )
        
        if result.memory_trace:
            report_lines.extend([
                "",
                "Memory Hotspots:",
                "-" * 20,
            ])
            
            for hotspot in result.get_memory_hotspots():
                report_lines.append(
                    f"  {hotspot['file']}: {hotspot['size_mb']:.2f}MB "
                    f"({hotspot['count']} allocations)"
                )
        
        report = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report)
        
        return report


class PerformanceOptimizer:
    """Automatic performance optimization recommendations."""
    
    def __init__(self):
        self.recommendations: List[Dict[str, Any]] = []
        self.metrics_collector = get_metrics_collector()
    
    def analyze_profile(self, result: ProfilingResult) -> List[Dict[str, Any]]:
        """Analyze profile and generate optimization recommendations."""
        recommendations = []
        metrics = result.metrics
        
        # Memory usage recommendations
        current_memory = metrics.memory_usage.get('current_mb', 0)
        peak_memory = metrics.memory_usage.get('peak_mb', 0)
        
        if current_memory > 1000:  # > 1GB
            recommendations.append({
                'type': 'memory_optimization',
                'severity': 'high',
                'message': f'High memory usage: {current_memory:.1f}MB',
                'suggestion': 'Consider implementing result streaming or pagination',
                'impact': 'high'
            })
        
        if peak_memory > current_memory * 2:
            recommendations.append({
                'type': 'memory_spike',
                'severity': 'medium',
                'message': f'Memory spike detected: peak {peak_memory:.1f}MB vs current {current_memory:.1f}MB',
                'suggestion': 'Check for memory leaks or inefficient data structures',
                'impact': 'medium'
            })
        
        # CPU usage recommendations
        if metrics.cpu_usage > 80:
            recommendations.append({
                'type': 'cpu_optimization',
                'severity': 'high',
                'message': f'High CPU usage: {metrics.cpu_usage:.1f}%',
                'suggestion': 'Consider async optimization or parallel processing',
                'impact': 'high'
            })
        
        # Context switching recommendations
        if metrics.context_switches > 1000:
            recommendations.append({
                'type': 'context_switching',
                'severity': 'medium',
                'message': f'High context switches: {metrics.context_switches}',
                'suggestion': 'Reduce thread contention or async task granularity',
                'impact': 'medium'
            })
        
        # Async task recommendations
        if metrics.async_task_count > 100:
            recommendations.append({
                'type': 'async_tasks',
                'severity': 'medium',
                'message': f'High async task count: {metrics.async_task_count}',
                'suggestion': 'Consider task pooling or batch processing',
                'impact': 'medium'
            })
        
        # Function-level recommendations
        top_functions = result.get_top_functions(5)
        for func in top_functions:
            if func['per_call'] > 0.1:  # > 100ms per call
                recommendations.append({
                    'type': 'slow_function',
                    'severity': 'high',
                    'message': f'Slow function: {func["function"]} takes {func["per_call"]:.3f}s per call',
                    'suggestion': 'Optimize this function or consider caching',
                    'impact': 'high'
                })
        
        self.recommendations.extend(recommendations)
        return recommendations
    
    def get_system_recommendations(self) -> List[Dict[str, Any]]:
        """Get system-level optimization recommendations."""
        recommendations = []
        
        # Event loop recommendations
        try:
            loop = asyncio.get_running_loop()
            if not isinstance(loop, (uvloop.Loop if UVLOOP_AVAILABLE else type(None))):
                recommendations.append({
                    'type': 'event_loop',
                    'severity': 'medium',
                    'message': 'Using default asyncio event loop',
                    'suggestion': 'Consider using uvloop for better performance',
                    'impact': 'medium'
                })
        except RuntimeError:
            pass
        
        # Memory recommendations
        if MEMORY_PROFILER_AVAILABLE:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            if memory_mb > 2000:  # > 2GB
                recommendations.append({
                    'type': 'system_memory',
                    'severity': 'high',
                    'message': f'High system memory usage: {memory_mb:.1f}MB',
                    'suggestion': 'Consider memory optimization or scaling horizontally',
                    'impact': 'high'
                })
        
        return recommendations
    
    def optimize_event_loop(self) -> bool:
        """Optimize the current event loop if possible."""
        if not UVLOOP_AVAILABLE:
            logger.warning("uvloop not available for event loop optimization")
            return False
        
        try:
            current_loop = asyncio.get_running_loop()
            if isinstance(current_loop, uvloop.Loop):
                logger.info("Already using uvloop")
                return True
            
            # Can't change running loop, but can recommend
            logger.info("Consider restarting with uvloop policy for better performance")
            return False
            
        except RuntimeError:
            # No running loop, can set policy
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("Set uvloop as default event loop policy")
            return True
    
    def optimize_gc(self) -> None:
        """Optimize garbage collection settings."""
        # Tune GC thresholds for better performance
        current_thresholds = gc.get_threshold()
        
        # Increase thresholds to reduce GC frequency for async workloads
        new_thresholds = (
            current_thresholds[0] * 2,  # Generation 0
            current_thresholds[1] * 2,  # Generation 1
            current_thresholds[2] * 2,  # Generation 2
        )
        
        gc.set_threshold(*new_thresholds)
        logger.info(f"Optimized GC thresholds: {current_thresholds} -> {new_thresholds}")
    
    def get_all_recommendations(self) -> List[Dict[str, Any]]:
        """Get all accumulated recommendations."""
        return self.recommendations
    
    def clear_recommendations(self) -> None:
        """Clear all recommendations."""
        self.recommendations.clear()


# Decorators for easy profiling
def profile_async(name: Optional[str] = None, profiler: Optional[AsyncProfiler] = None):
    """Decorator to profile async functions."""
    def decorator(func):
        func_name = name or f"{func.__module__}.{func.__name__}"
        _profiler = profiler or AsyncProfiler()
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with _profiler.profile(func_name):
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def benchmark_async(iterations: int = 1, warmup: int = 0):
    """Decorator to benchmark async functions."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Warmup
            for _ in range(warmup):
                await func(*args, **kwargs)
            
            # Benchmark
            times = []
            for i in range(iterations):
                start = time.perf_counter()
                result = await func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
                
                # Return result on last iteration
                if i == iterations - 1:
                    final_result = result
            
            # Calculate statistics
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            logger.info(
                f"Benchmark {func.__name__}: "
                f"avg={avg_time:.4f}s, min={min_time:.4f}s, max={max_time:.4f}s "
                f"({iterations} iterations)"
            )
            
            return final_result
        
        return wrapper
    return decorator


# Global profiler instance
_global_profiler: Optional[AsyncProfiler] = None


def get_profiler() -> AsyncProfiler:
    """Get global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = AsyncProfiler()
    return _global_profiler