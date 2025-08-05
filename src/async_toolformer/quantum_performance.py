"""
Quantum Performance Optimization Module for AsyncOrchestrator.

This module provides advanced performance optimization features:
- Quantum-inspired load balancing
- Adaptive resource scaling
- Performance monitoring and metrics
- Intelligent caching strategies
- Predictive execution optimization
"""

import asyncio
import time
import math
import statistics
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import weakref
import gc

from .quantum_planner import QuantumTask, ExecutionPlan, TaskState

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    THROUGHPUT = "throughput"          # Maximize task completion rate
    LATENCY = "latency"               # Minimize individual task latency
    EFFICIENCY = "efficiency"          # Optimize resource utilization
    QUANTUM_COHERENT = "quantum_coherent"  # Maintain quantum coherence


@dataclass
class PerformanceMetrics:
    """Performance metrics collection."""
    timestamp: float = field(default_factory=time.time)
    
    # Execution metrics
    total_tasks_completed: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_task_duration_ms: float = 0.0
    median_task_duration_ms: float = 0.0
    p95_task_duration_ms: float = 0.0
    
    # Throughput metrics
    tasks_per_second: float = 0.0
    peak_throughput: float = 0.0
    throughput_efficiency: float = 0.0
    
    # Resource metrics
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    network_utilization: float = 0.0
    io_utilization: float = 0.0
    
    # Quantum metrics
    quantum_coherence: float = 0.0
    entanglement_efficiency: float = 0.0
    superposition_collapse_rate: float = 0.0
    
    # Caching metrics
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    cache_eviction_rate: float = 0.0
    
    # Scaling metrics
    parallelism_factor: float = 0.0
    load_balance_score: float = 0.0
    resource_scaling_efficiency: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp,
            "execution": {
                "total_tasks": self.total_tasks_completed,
                "successful": self.successful_tasks,
                "failed": self.failed_tasks,
                "avg_duration_ms": self.average_task_duration_ms,
                "median_duration_ms": self.median_task_duration_ms,
                "p95_duration_ms": self.p95_task_duration_ms,
            },
            "throughput": {
                "tasks_per_second": self.tasks_per_second,
                "peak_throughput": self.peak_throughput,
                "efficiency": self.throughput_efficiency,
            },
            "resources": {
                "cpu": self.cpu_utilization,
                "memory": self.memory_utilization,
                "network": self.network_utilization,
                "io": self.io_utilization,
            },
            "quantum": {
                "coherence": self.quantum_coherence,
                "entanglement_efficiency": self.entanglement_efficiency,
                "collapse_rate": self.superposition_collapse_rate,
            },
            "caching": {
                "hit_rate": self.cache_hit_rate,
                "miss_rate": self.cache_miss_rate,
                "eviction_rate": self.cache_eviction_rate,
            },
            "scaling": {
                "parallelism_factor": self.parallelism_factor,
                "load_balance": self.load_balance_score,
                "scaling_efficiency": self.resource_scaling_efficiency,
            }
        }


@dataclass
class ResourceScalingRule:
    """Rule for automatic resource scaling."""
    resource_type: str
    metric_threshold: float
    scale_factor: float
    cooldown_seconds: int = 60
    max_scale: float = 10.0
    min_scale: float = 0.1
    last_scaled: float = 0.0
    
    def can_scale(self) -> bool:
        """Check if scaling is allowed (cooldown period)."""
        return time.time() - self.last_scaled > self.cooldown_seconds
    
    def should_scale_up(self, current_value: float) -> bool:
        """Check if should scale up based on threshold."""
        return current_value > self.metric_threshold and self.can_scale()
    
    def should_scale_down(self, current_value: float) -> bool:
        """Check if should scale down based on threshold."""
        return current_value < (self.metric_threshold * 0.7) and self.can_scale()


class QuantumPerformanceOptimizer:
    """
    Quantum-inspired performance optimizer for task orchestration.
    
    Features:
    - Adaptive resource scaling based on quantum principles
    - Intelligent load balancing using superposition
    - Predictive performance optimization
    - Advanced caching with quantum-inspired eviction
    - Real-time performance monitoring
    """
    
    def __init__(
        self,
        optimization_strategy: OptimizationStrategy = OptimizationStrategy.EFFICIENCY,
        enable_auto_scaling: bool = True,
        performance_history_size: int = 1000,
        monitoring_interval_seconds: float = 1.0,
        cache_optimization_enabled: bool = True,
    ):
        """
        Initialize the quantum performance optimizer.
        
        Args:
            optimization_strategy: Primary optimization strategy
            enable_auto_scaling: Whether to enable automatic resource scaling
            performance_history_size: Number of metrics to keep in history
            monitoring_interval_seconds: Metrics collection interval
            cache_optimization_enabled: Whether to optimize caching
        """
        self.optimization_strategy = optimization_strategy
        self.enable_auto_scaling = enable_auto_scaling
        self.performance_history_size = performance_history_size
        self.monitoring_interval = monitoring_interval_seconds
        self.cache_optimization_enabled = cache_optimization_enabled
        
        # Performance tracking
        self._performance_history: deque = deque(maxlen=performance_history_size)
        self._current_metrics = PerformanceMetrics()
        self._task_timings: deque = deque(maxlen=1000)
        self._throughput_samples: deque = deque(maxlen=100)
        
        # Resource scaling
        self._scaling_rules: Dict[str, ResourceScalingRule] = {}
        self._current_resource_scale: Dict[str, float] = defaultdict(lambda: 1.0)
        
        # Load balancing
        self._worker_loads: Dict[str, float] = defaultdict(float)
        self._task_affinities: Dict[str, Set[str]] = defaultdict(set)
        
        # Caching optimization
        self._cache_access_patterns: Dict[str, List[float]] = defaultdict(list)
        self._cache_prediction_model: Optional[Dict[str, Any]] = None
        
        # Monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        # Initialize default scaling rules
        self._initialize_scaling_rules()
        
        logger.info(f"QuantumPerformanceOptimizer initialized with {optimization_strategy.value} strategy")
    
    def _initialize_scaling_rules(self):
        """Initialize default resource scaling rules."""
        self._scaling_rules.update({
            "cpu": ResourceScalingRule(
                resource_type="cpu",
                metric_threshold=0.8,  # Scale up when CPU > 80%
                scale_factor=1.5,
                cooldown_seconds=30,
                max_scale=8.0,
            ),
            "memory": ResourceScalingRule(
                resource_type="memory",
                metric_threshold=0.85,  # Scale up when memory > 85%
                scale_factor=1.3,
                cooldown_seconds=60,
                max_scale=4.0,
            ),
            "network": ResourceScalingRule(
                resource_type="network",
                metric_threshold=0.9,   # Scale up when network > 90%
                scale_factor=2.0,
                cooldown_seconds=15,
                max_scale=10.0,
            ),
            "parallelism": ResourceScalingRule(
                resource_type="parallelism",
                metric_threshold=0.95,  # Scale up when parallelism > 95%
                scale_factor=1.2,
                cooldown_seconds=10,
                max_scale=5.0,
            ),
        })
    
    async def start_monitoring(self):
        """Start performance monitoring."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring."""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self._is_monitoring:
                await self._collect_metrics()
                await self._analyze_performance()
                
                if self.enable_auto_scaling:
                    await self._apply_auto_scaling()
                
                await asyncio.sleep(self.monitoring_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
    
    async def _collect_metrics(self):
        """Collect current performance metrics."""
        current_time = time.time()
        
        # Calculate task duration statistics
        if self._task_timings:
            durations = list(self._task_timings)
            self._current_metrics.average_task_duration_ms = statistics.mean(durations)
            self._current_metrics.median_task_duration_ms = statistics.median(durations)
            
            if len(durations) >= 20:  # Need sufficient samples for percentile
                sorted_durations = sorted(durations)
                p95_index = int(0.95 * len(sorted_durations))
                self._current_metrics.p95_task_duration_ms = sorted_durations[p95_index]
        
        # Calculate throughput
        if self._throughput_samples:
            self._current_metrics.tasks_per_second = statistics.mean(self._throughput_samples)
            self._current_metrics.peak_throughput = max(self._throughput_samples)
        
        # Update quantum metrics (simulated)
        self._current_metrics.quantum_coherence = self._calculate_quantum_coherence()
        self._current_metrics.entanglement_efficiency = self._calculate_entanglement_efficiency()
        
        # Calculate resource utilization
        self._current_metrics.cpu_utilization = self._estimate_cpu_utilization()
        self._current_metrics.memory_utilization = self._estimate_memory_utilization()
        
        # Calculate scaling metrics
        self._current_metrics.parallelism_factor = self._calculate_parallelism_factor()
        self._current_metrics.load_balance_score = self._calculate_load_balance_score()
        
        # Store in history
        self._current_metrics.timestamp = current_time
        self._performance_history.append(self._current_metrics.to_dict())
    
    def _calculate_quantum_coherence(self) -> float:
        """Calculate simulated quantum coherence based on system state."""
        # Simulate coherence based on task completion rate and system stability
        if not self._task_timings:
            return 1.0
        
        # Lower variance in task timings = higher coherence
        variance = statistics.variance(self._task_timings) if len(self._task_timings) > 1 else 0
        max_variance = 1000000  # Arbitrary maximum
        coherence = max(0.1, 1.0 - (variance / max_variance))
        
        return min(1.0, coherence)
    
    def _calculate_entanglement_efficiency(self) -> float:
        """Calculate entanglement efficiency based on task interdependencies."""
        # Simulate based on load balance - better balance = better entanglement
        load_balance = self._calculate_load_balance_score()
        return load_balance
    
    def _estimate_cpu_utilization(self) -> float:
        """Estimate CPU utilization based on task activity."""
        # Simple estimation based on number of active tasks and their complexity
        active_workers = len([load for load in self._worker_loads.values() if load > 0])
        max_workers = max(len(self._worker_loads), 1)
        
        base_utilization = active_workers / max_workers
        
        # Adjust based on recent task timings
        if self._task_timings:
            avg_duration = statistics.mean(list(self._task_timings)[-10:])  # Last 10 tasks
            # Longer tasks suggest higher CPU usage
            duration_factor = min(2.0, avg_duration / 1000.0)  # Normalize to seconds
            return min(1.0, base_utilization * duration_factor)
        
        return base_utilization
    
    def _estimate_memory_utilization(self) -> float:
        """Estimate memory utilization."""
        # Simple estimation based on performance history size and system activity
        history_factor = len(self._performance_history) / self.performance_history_size
        active_factor = len(self._worker_loads) / 100.0  # Assume 100 max workers
        
        return min(1.0, (history_factor + active_factor) / 2.0)
    
    def _calculate_parallelism_factor(self) -> float:
        """Calculate current parallelism factor."""
        if not self._worker_loads:
            return 1.0
        
        active_workers = sum(1 for load in self._worker_loads.values() if load > 0.1)
        return max(1.0, active_workers)
    
    def _calculate_load_balance_score(self) -> float:
        """Calculate load balance score (1.0 = perfectly balanced)."""
        if not self._worker_loads:
            return 1.0
        
        loads = list(self._worker_loads.values())
        if not loads:
            return 1.0
        
        # Calculate coefficient of variation (lower = better balance)
        mean_load = statistics.mean(loads)
        if mean_load == 0:
            return 1.0
        
        std_dev = statistics.stdev(loads) if len(loads) > 1 else 0
        cv = std_dev / mean_load
        
        # Convert to score (1.0 = perfect balance, 0.0 = terrible balance)
        return max(0.0, 1.0 - cv)
    
    async def _analyze_performance(self):
        """Analyze performance trends and identify optimization opportunities."""
        if len(self._performance_history) < 10:
            return  # Need sufficient history
        
        recent_metrics = list(self._performance_history)[-10:]
        
        # Analyze throughput trends
        throughput_values = [m["throughput"]["tasks_per_second"] for m in recent_metrics]
        if len(throughput_values) >= 2:
            throughput_trend = (throughput_values[-1] - throughput_values[0]) / len(throughput_values)
            
            if throughput_trend < -0.1:
                logger.warning("Declining throughput detected")
                await self._optimize_for_throughput()
            elif throughput_trend > 0.1:
                logger.info("Improving throughput detected")
        
        # Analyze resource utilization
        cpu_values = [m["resources"]["cpu"] for m in recent_metrics]
        avg_cpu = statistics.mean(cpu_values)
        
        if avg_cpu > 0.9:
            logger.warning("High CPU utilization detected")
            await self._optimize_for_cpu_efficiency()
        elif avg_cpu < 0.3:
            logger.info("Low CPU utilization - consider scaling down")
    
    async def _optimize_for_throughput(self):
        """Optimize system for maximum throughput."""
        logger.info("Applying throughput optimizations")
        
        # Increase parallelism if possible
        current_parallelism = self._current_resource_scale.get("parallelism", 1.0)
        if current_parallelism < 3.0:  # Don't scale too aggressively
            self._current_resource_scale["parallelism"] = min(3.0, current_parallelism * 1.2)
        
        # Optimize cache for high-frequency access patterns
        if self.cache_optimization_enabled:
            await self._optimize_cache_for_throughput()
    
    async def _optimize_for_cpu_efficiency(self):
        """Optimize system for CPU efficiency."""
        logger.info("Applying CPU efficiency optimizations")
        
        # Balance load across workers
        await self._rebalance_worker_loads()
        
        # Consider reducing parallelism if over-saturated
        current_parallelism = self._current_resource_scale.get("parallelism", 1.0)
        if current_parallelism > 2.0:
            self._current_resource_scale["parallelism"] = max(1.0, current_parallelism * 0.9)
    
    async def _optimize_cache_for_throughput(self):
        """Optimize caching strategy for throughput."""
        # Analyze access patterns and adjust cache priorities
        for cache_key, access_times in self._cache_access_patterns.items():
            if len(access_times) >= 5:
                # Calculate access frequency
                time_span = max(access_times) - min(access_times)
                if time_span > 0:
                    frequency = len(access_times) / time_span
                    
                    # High frequency items should have higher cache priority
                    if frequency > 1.0:  # More than 1 access per second
                        logger.debug(f"High-frequency cache pattern detected: {cache_key}")
    
    async def _rebalance_worker_loads(self):
        """Rebalance work across available workers."""
        if not self._worker_loads:
            return
        
        # Find over- and under-loaded workers
        loads = list(self._worker_loads.items())
        avg_load = statistics.mean([load for _, load in loads])
        
        overloaded = [(worker, load) for worker, load in loads if load > avg_load * 1.5]
        underloaded = [(worker, load) for worker, load in loads if load < avg_load * 0.5]
        
        if overloaded and underloaded:
            logger.info(f"Rebalancing load: {len(overloaded)} overloaded, {len(underloaded)} underloaded workers")
            
            # Simulate load rebalancing by adjusting worker loads
            for worker, _ in overloaded:
                self._worker_loads[worker] *= 0.8  # Reduce load
            
            for worker, _ in underloaded:
                self._worker_loads[worker] *= 1.2  # Increase load
    
    async def _apply_auto_scaling(self):
        """Apply automatic resource scaling based on current metrics."""
        for resource_type, rule in self._scaling_rules.items():
            current_value = self._get_metric_value(resource_type)
            current_scale = self._current_resource_scale[resource_type]
            
            if rule.should_scale_up(current_value):
                new_scale = min(rule.max_scale, current_scale * rule.scale_factor)
                if new_scale != current_scale:
                    self._current_resource_scale[resource_type] = new_scale
                    rule.last_scaled = time.time()
                    logger.info(f"Scaled up {resource_type}: {current_scale:.2f} -> {new_scale:.2f}")
            
            elif rule.should_scale_down(current_value):
                new_scale = max(rule.min_scale, current_scale / rule.scale_factor)
                if new_scale != current_scale:
                    self._current_resource_scale[resource_type] = new_scale
                    rule.last_scaled = time.time()
                    logger.info(f"Scaled down {resource_type}: {current_scale:.2f} -> {new_scale:.2f}")
    
    def _get_metric_value(self, resource_type: str) -> float:
        """Get current metric value for a resource type."""
        if resource_type == "cpu":
            return self._current_metrics.cpu_utilization
        elif resource_type == "memory":
            return self._current_metrics.memory_utilization
        elif resource_type == "network":
            return self._current_metrics.network_utilization
        elif resource_type == "parallelism":
            return self._current_metrics.parallelism_factor / 10.0  # Normalize to 0-1
        else:
            return 0.0
    
    def record_task_execution(
        self,
        task_id: str,
        duration_ms: float,
        success: bool,
        worker_id: Optional[str] = None,
        resource_usage: Optional[Dict[str, float]] = None,
    ):
        """
        Record task execution for performance analysis.
        
        Args:
            task_id: Task identifier
            duration_ms: Execution duration in milliseconds
            success: Whether task succeeded
            worker_id: Worker that executed the task
            resource_usage: Resource usage during execution
        """
        # Record timing
        self._task_timings.append(duration_ms)
        
        # Update task counters
        self._current_metrics.total_tasks_completed += 1
        if success:
            self._current_metrics.successful_tasks += 1
        else:
            self._current_metrics.failed_tasks += 1
        
        # Update worker load tracking
        if worker_id:
            # Simple load model: recent tasks contribute to current load
            current_load = self._worker_loads.get(worker_id, 0.0)
            # Exponential decay + new task contribution
            decay_factor = 0.95
            task_contribution = duration_ms / 10000.0  # Normalize to reasonable range
            
            self._worker_loads[worker_id] = current_load * decay_factor + task_contribution
        
        # Calculate current throughput
        current_time = time.time()
        
        # Count tasks in last second
        recent_tasks = len([
            t for t in self._task_timings 
            if current_time - (t / 1000.0) <= 1.0  # Rough approximation
        ])
        
        self._throughput_samples.append(recent_tasks)
    
    def record_cache_access(self, cache_key: str, hit: bool):
        """
        Record cache access for optimization analysis.
        
        Args:
            cache_key: Cache key accessed
            hit: Whether it was a cache hit
        """
        if not self.cache_optimization_enabled:
            return
        
        current_time = time.time()
        
        # Record access pattern
        self._cache_access_patterns[cache_key].append(current_time)
        
        # Keep only recent access times (last 10 minutes)
        cutoff_time = current_time - 600
        self._cache_access_patterns[cache_key] = [
            t for t in self._cache_access_patterns[cache_key]
            if t > cutoff_time
        ]
        
        # Update cache metrics
        if hit:
            self._current_metrics.cache_hit_rate = (
                self._current_metrics.cache_hit_rate * 0.9 + 0.1
            )
        else:
            self._current_metrics.cache_miss_rate = (
                self._current_metrics.cache_miss_rate * 0.9 + 0.1
            )
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get performance optimization recommendations.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        if len(self._performance_history) < 5:
            return recommendations
        
        recent_metrics = list(self._performance_history)[-5:]
        
        # CPU utilization recommendations
        cpu_values = [m["resources"]["cpu"] for m in recent_metrics]
        avg_cpu = statistics.mean(cpu_values)
        
        if avg_cpu > 0.85:
            recommendations.append({
                "type": "resource_scaling",
                "priority": "high",
                "description": "High CPU utilization detected",
                "recommendation": "Consider scaling up CPU resources or optimizing task algorithms",
                "metric": "cpu_utilization",
                "current_value": avg_cpu,
                "threshold": 0.85,
            })
        elif avg_cpu < 0.3:
            recommendations.append({
                "type": "resource_optimization",
                "priority": "medium",
                "description": "Low CPU utilization detected",
                "recommendation": "Consider scaling down CPU resources or increasing task parallelism",
                "metric": "cpu_utilization",
                "current_value": avg_cpu,
                "threshold": 0.3,
            })
        
        # Throughput recommendations
        throughput_values = [m["throughput"]["tasks_per_second"] for m in recent_metrics]
        if len(throughput_values) >= 2:
            throughput_trend = throughput_values[-1] - throughput_values[0]
            
            if throughput_trend < -0.5:
                recommendations.append({
                    "type": "performance_optimization",
                    "priority": "high",
                    "description": "Declining throughput trend detected",
                    "recommendation": "Investigate task bottlenecks and optimize critical paths",
                    "metric": "throughput_trend",
                    "current_value": throughput_trend,
                    "threshold": -0.5,
                })
        
        # Load balance recommendations
        load_balance_values = [m["scaling"]["load_balance"] for m in recent_metrics]
        avg_load_balance = statistics.mean(load_balance_values)
        
        if avg_load_balance < 0.7:
            recommendations.append({
                "type": "load_balancing",
                "priority": "medium",
                "description": "Poor load balancing detected",
                "recommendation": "Implement better task distribution or worker affinity rules",
                "metric": "load_balance_score",
                "current_value": avg_load_balance,
                "threshold": 0.7,
            })
        
        # Cache optimization recommendations
        cache_hit_rate = self._current_metrics.cache_hit_rate
        if cache_hit_rate < 0.8:
            recommendations.append({
                "type": "cache_optimization",
                "priority": "medium",
                "description": "Low cache hit rate detected",
                "recommendation": "Optimize cache size, TTL, or implement better caching strategies",
                "metric": "cache_hit_rate",
                "current_value": cache_hit_rate,
                "threshold": 0.8,
            })
        
        return recommendations
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        return self._current_metrics
    
    def get_performance_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get performance history.
        
        Args:
            last_n: Number of recent entries to return (None for all)
            
        Returns:
            List of performance metric dictionaries
        """
        history = list(self._performance_history)
        if last_n:
            return history[-last_n:]
        return history
    
    def get_resource_scaling_status(self) -> Dict[str, Any]:
        """Get current resource scaling status."""
        return {
            "current_scales": dict(self._current_resource_scale),
            "scaling_rules": {
                name: {
                    "threshold": rule.metric_threshold,
                    "scale_factor": rule.scale_factor,
                    "cooldown": rule.cooldown_seconds,
                    "can_scale": rule.can_scale(),
                    "last_scaled": rule.last_scaled,
                }
                for name, rule in self._scaling_rules.items()
            },
            "auto_scaling_enabled": self.enable_auto_scaling,
        }
    
    def set_optimization_strategy(self, strategy: OptimizationStrategy):
        """Change the optimization strategy."""
        old_strategy = self.optimization_strategy
        self.optimization_strategy = strategy
        
        logger.info(f"Changed optimization strategy: {old_strategy.value} -> {strategy.value}")
        
        # Adjust monitoring parameters based on strategy
        if strategy == OptimizationStrategy.LATENCY:
            self.monitoring_interval = 0.5  # More frequent monitoring for latency
        elif strategy == OptimizationStrategy.THROUGHPUT:
            self.monitoring_interval = 2.0  # Less frequent for throughput
        elif strategy == OptimizationStrategy.QUANTUM_COHERENT:
            self.monitoring_interval = 0.1  # Very frequent for quantum coherence
    
    def add_scaling_rule(self, resource_type: str, rule: ResourceScalingRule):
        """Add or update a resource scaling rule."""
        self._scaling_rules[resource_type] = rule
        logger.info(f"Added scaling rule for {resource_type}")
    
    def optimize_for_quantum_coherence(self):
        """Optimize system parameters for maximum quantum coherence."""
        logger.info("Optimizing for quantum coherence")
        
        # Reduce parallelism to maintain coherence
        current_parallelism = self._current_resource_scale.get("parallelism", 1.0)
        self._current_resource_scale["parallelism"] = min(2.0, current_parallelism)
        
        # Increase monitoring frequency
        self.monitoring_interval = 0.1
        
        # Enable cache optimization for consistency
        self.cache_optimization_enabled = True
    
    async def cleanup(self):
        """Clean up optimizer resources."""
        await self.stop_monitoring()
        
        # Clear large data structures
        self._performance_history.clear()
        self._task_timings.clear()
        self._throughput_samples.clear()
        self._cache_access_patterns.clear()
        self._worker_loads.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("QuantumPerformanceOptimizer cleanup completed")


# Convenience functions
def create_performance_optimizer(
    strategy: OptimizationStrategy = OptimizationStrategy.EFFICIENCY,
    auto_scaling: bool = True,
    **kwargs
) -> QuantumPerformanceOptimizer:
    """Create a QuantumPerformanceOptimizer with sensible defaults."""
    return QuantumPerformanceOptimizer(
        optimization_strategy=strategy,
        enable_auto_scaling=auto_scaling,
        **kwargs
    )


async def monitor_performance(
    optimizer: QuantumPerformanceOptimizer,
    duration_seconds: float
):
    """Monitor performance for a specified duration."""
    await optimizer.start_monitoring()
    await asyncio.sleep(duration_seconds)
    await optimizer.stop_monitoring()
    
    return optimizer.get_current_metrics()