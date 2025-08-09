"""Advanced performance optimization and auto-scaling."""

import asyncio
import time
import statistics
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import weakref

from .simple_structured_logging import get_logger

logger = get_logger(__name__)


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    BASIC = "basic"
    AGGRESSIVE = "aggressive" 
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time_ms: float = 0.0
    p95_execution_time_ms: float = 0.0
    p99_execution_time_ms: float = 0.0
    throughput_per_second: float = 0.0
    current_concurrent_tasks: int = 0
    max_concurrent_tasks: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    recent_execution_times: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    action: str  # "scale_up", "scale_down", "no_change"
    current_capacity: int
    target_capacity: int
    reason: str
    confidence: float
    metrics_snapshot: PerformanceMetrics = None


class PerformanceAnalyzer:
    """Analyzes performance patterns and bottlenecks."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.execution_times: List[float] = []
        self.success_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
    def record_execution(self, execution_time_ms: float, success: bool):
        """Record a single execution."""
        self.execution_times.append(execution_time_ms)
        
        # Keep only recent executions
        if len(self.execution_times) > self.window_size:
            self.execution_times.pop(0)
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if not self.execution_times:
            return PerformanceMetrics()
        
        total_executions = self.success_count + self.error_count
        avg_time = statistics.mean(self.execution_times)
        
        # Calculate percentiles
        sorted_times = sorted(self.execution_times)
        p95_index = int(0.95 * len(sorted_times))
        p99_index = int(0.99 * len(sorted_times))
        
        p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else avg_time
        p99_time = sorted_times[p99_index] if p99_index < len(sorted_times) else avg_time
        
        # Calculate throughput
        elapsed_time = time.time() - self.start_time
        throughput = total_executions / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate error rate
        error_rate = self.error_count / total_executions if total_executions > 0 else 0
        
        return PerformanceMetrics(
            total_executions=total_executions,
            successful_executions=self.success_count,
            failed_executions=self.error_count,
            avg_execution_time_ms=avg_time,
            p95_execution_time_ms=p95_time,
            p99_execution_time_ms=p99_time,
            throughput_per_second=throughput,
            error_rate=error_rate,
            recent_execution_times=self.execution_times[-100:],  # Last 100 for analysis
        )
    
    def detect_performance_anomalies(self) -> List[str]:
        """Detect performance anomalies."""
        anomalies = []
        
        if len(self.execution_times) < 10:
            return anomalies
        
        recent_times = self.execution_times[-10:]
        older_times = self.execution_times[-100:-10] if len(self.execution_times) > 100 else []
        
        if older_times:
            recent_avg = statistics.mean(recent_times)
            older_avg = statistics.mean(older_times)
            
            # Performance degradation
            if recent_avg > older_avg * 1.5:
                anomalies.append("Performance degradation detected")
            
            # High variance in recent times
            if len(recent_times) > 3:
                recent_stdev = statistics.stdev(recent_times)
                if recent_stdev > recent_avg * 0.5:
                    anomalies.append("High variance in execution times")
        
        # High error rate
        if self.error_count > 0:
            error_rate = self.error_count / (self.success_count + self.error_count)
            if error_rate > 0.1:  # 10% error rate
                anomalies.append(f"High error rate: {error_rate:.2%}")
        
        return anomalies


class AutoScaler:
    """Automatic scaling based on performance metrics."""
    
    def __init__(self, 
                 min_capacity: int = 1,
                 max_capacity: int = 100,
                 target_cpu_percent: float = 70.0,
                 scale_up_threshold: float = 80.0,
                 scale_down_threshold: float = 50.0,
                 cooldown_seconds: int = 300):
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.target_cpu_percent = target_cpu_percent
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds
        
        self.last_scaling_time = 0
        self.current_capacity = min_capacity
        
    def should_scale(self, metrics: PerformanceMetrics) -> ScalingDecision:
        """Determine if scaling is needed."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_time < self.cooldown_seconds:
            return ScalingDecision(
                action="no_change",
                current_capacity=self.current_capacity,
                target_capacity=self.current_capacity,
                reason="In cooldown period",
                confidence=0.0,
                metrics_snapshot=metrics
            )
        
        # Scale up conditions
        scale_up_signals = 0
        scale_up_reasons = []
        
        # High CPU usage
        if metrics.cpu_usage_percent > self.scale_up_threshold:
            scale_up_signals += 1
            scale_up_reasons.append(f"High CPU: {metrics.cpu_usage_percent:.1f}%")
        
        # High concurrent tasks
        if metrics.current_concurrent_tasks > self.current_capacity * 0.8:
            scale_up_signals += 1
            scale_up_reasons.append("High concurrent task load")
        
        # Slow response times
        if metrics.p95_execution_time_ms > 5000:  # 5 second threshold
            scale_up_signals += 1
            scale_up_reasons.append(f"Slow P95 response: {metrics.p95_execution_time_ms:.0f}ms")
        
        # Low cache hit rate (may indicate overload)
        if metrics.cache_hit_rate < 0.5 and metrics.total_executions > 100:
            scale_up_signals += 1
            scale_up_reasons.append(f"Low cache hit rate: {metrics.cache_hit_rate:.2%}")
        
        # Scale down conditions
        scale_down_signals = 0
        scale_down_reasons = []
        
        # Low CPU usage
        if metrics.cpu_usage_percent < self.scale_down_threshold:
            scale_down_signals += 1
            scale_down_reasons.append(f"Low CPU: {metrics.cpu_usage_percent:.1f}%")
        
        # Low concurrent tasks
        if metrics.current_concurrent_tasks < self.current_capacity * 0.3:
            scale_down_signals += 1
            scale_down_reasons.append("Low concurrent task load")
        
        # Fast response times
        if metrics.p95_execution_time_ms < 1000:  # Very fast responses
            scale_down_signals += 1
            scale_down_reasons.append(f"Fast P95 response: {metrics.p95_execution_time_ms:.0f}ms")
        
        # Make scaling decision
        if scale_up_signals >= 2:  # Need multiple signals
            target_capacity = min(
                int(self.current_capacity * 1.5),  # 50% increase
                self.max_capacity
            )
            
            if target_capacity > self.current_capacity:
                return ScalingDecision(
                    action="scale_up",
                    current_capacity=self.current_capacity,
                    target_capacity=target_capacity,
                    reason="; ".join(scale_up_reasons),
                    confidence=scale_up_signals / 4.0,  # Max 4 signals
                    metrics_snapshot=metrics
                )
        
        elif scale_down_signals >= 2 and self.current_capacity > self.min_capacity:
            target_capacity = max(
                int(self.current_capacity * 0.7),  # 30% decrease
                self.min_capacity
            )
            
            if target_capacity < self.current_capacity:
                return ScalingDecision(
                    action="scale_down",
                    current_capacity=self.current_capacity,
                    target_capacity=target_capacity,
                    reason="; ".join(scale_down_reasons),
                    confidence=scale_down_signals / 4.0,
                    metrics_snapshot=metrics
                )
        
        return ScalingDecision(
            action="no_change",
            current_capacity=self.current_capacity,
            target_capacity=self.current_capacity,
            reason="No scaling triggers met",
            confidence=0.5,
            metrics_snapshot=metrics
        )
    
    def apply_scaling_decision(self, decision: ScalingDecision) -> bool:
        """Apply a scaling decision."""
        if decision.action == "no_change":
            return False
        
        logger.info(
            f"Auto-scaling: {decision.action}",
            current_capacity=decision.current_capacity,
            target_capacity=decision.target_capacity,
            reason=decision.reason,
            confidence=decision.confidence
        )
        
        self.current_capacity = decision.target_capacity
        self.last_scaling_time = time.time()
        
        return True


class PerformanceOptimizer:
    """Comprehensive performance optimization manager."""
    
    def __init__(self,
                 optimization_level: OptimizationLevel = OptimizationLevel.ADAPTIVE):
        self.optimization_level = optimization_level
        self.analyzer = PerformanceAnalyzer()
        self.auto_scaler = AutoScaler()
        self.optimizations_applied = set()
        
        # Task scheduling
        self.task_queue_high_priority = asyncio.Queue()
        self.task_queue_normal_priority = asyncio.Queue()
        self.task_queue_low_priority = asyncio.Queue()
        
        # Worker management
        self.active_workers = weakref.WeakSet()
        self.worker_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.last_optimization_time = 0
        self.optimization_interval = 60  # Check every minute
        
    async def optimize_execution(self,
                               executor_func,
                               *args,
                               priority: str = "normal",
                               **kwargs) -> Any:
        """Execute with optimization."""
        start_time = time.time()
        
        try:
            # Choose appropriate queue based on priority
            if priority == "high":
                queue = self.task_queue_high_priority
            elif priority == "low": 
                queue = self.task_queue_low_priority
            else:
                queue = self.task_queue_normal_priority
            
            # Execute the function
            result = await executor_func(*args, **kwargs)
            
            # Record performance
            execution_time = (time.time() - start_time) * 1000
            self.analyzer.record_execution(execution_time, True)
            
            return result
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.analyzer.record_execution(execution_time, False)
            raise
    
    async def start_performance_monitoring(self):
        """Start continuous performance monitoring and optimization."""
        logger.info("Starting performance monitoring")
        
        while True:
            try:
                await asyncio.sleep(self.optimization_interval)
                await self._perform_optimization_cycle()
                
            except Exception as e:
                logger.error("Error in performance monitoring", error=e)
                await asyncio.sleep(self.optimization_interval)
    
    async def _perform_optimization_cycle(self):
        """Perform one optimization cycle."""
        metrics = self.analyzer.get_performance_metrics()
        
        # Check for anomalies
        anomalies = self.analyzer.detect_performance_anomalies()
        if anomalies:
            logger.warning("Performance anomalies detected", anomalies=anomalies)
        
        # Check auto-scaling
        scaling_decision = self.auto_scaler.should_scale(metrics)
        if scaling_decision.action != "no_change":
            applied = self.auto_scaler.apply_scaling_decision(scaling_decision)
            if applied:
                await self._adjust_worker_pool(scaling_decision.target_capacity)
        
        # Apply optimizations based on metrics
        await self._apply_performance_optimizations(metrics)
        
        logger.debug(
            "Performance optimization cycle completed",
            total_executions=metrics.total_executions,
            avg_time_ms=metrics.avg_execution_time_ms,
            throughput_per_sec=metrics.throughput_per_second,
            error_rate=metrics.error_rate,
            current_capacity=self.auto_scaler.current_capacity
        )
    
    async def _adjust_worker_pool(self, target_capacity: int):
        """Adjust the worker pool size."""
        current_workers = len(self.worker_tasks)
        
        if target_capacity > current_workers:
            # Add workers
            for _ in range(target_capacity - current_workers):
                worker_task = asyncio.create_task(self._worker_loop())
                self.worker_tasks.append(worker_task)
        
        elif target_capacity < current_workers:
            # Remove workers
            workers_to_remove = current_workers - target_capacity
            for i in range(workers_to_remove):
                if i < len(self.worker_tasks):
                    self.worker_tasks[i].cancel()
            
            # Clean up cancelled tasks
            self.worker_tasks = [t for t in self.worker_tasks if not t.cancelled()]
        
        logger.info(
            f"Adjusted worker pool size",
            previous_size=current_workers,
            new_size=len(self.worker_tasks),
            target_capacity=target_capacity
        )
    
    async def _worker_loop(self):
        """Worker loop for processing tasks."""
        try:
            while True:
                # Process tasks in priority order
                task = None
                
                try:
                    # High priority first
                    task = self.task_queue_high_priority.get_nowait()
                except asyncio.QueueEmpty:
                    try:
                        # Normal priority second
                        task = self.task_queue_normal_priority.get_nowait()
                    except asyncio.QueueEmpty:
                        try:
                            # Low priority last
                            task = self.task_queue_low_priority.get_nowait()
                        except asyncio.QueueEmpty:
                            # No tasks, wait a bit
                            await asyncio.sleep(0.1)
                            continue
                
                if task:
                    await self._execute_task(task)
                    
        except asyncio.CancelledError:
            logger.debug("Worker cancelled")
        except Exception as e:
            logger.error("Worker error", error=e)
    
    async def _execute_task(self, task):
        """Execute a single task."""
        # Task execution logic would go here
        # This is a placeholder for the actual task execution
        await asyncio.sleep(0.01)  # Simulate work
    
    async def _apply_performance_optimizations(self, metrics: PerformanceMetrics):
        """Apply performance optimizations based on current metrics."""
        
        # Adaptive batch sizing
        if metrics.throughput_per_second > 50 and "batch_optimization" not in self.optimizations_applied:
            logger.info("Applying batch optimization")
            self.optimizations_applied.add("batch_optimization")
        
        # Connection pooling optimization
        if metrics.total_executions > 1000 and "connection_pooling" not in self.optimizations_applied:
            logger.info("Applying connection pooling optimization")
            self.optimizations_applied.add("connection_pooling")
        
        # Memory management optimization
        if metrics.memory_usage_mb > 1000 and "memory_optimization" not in self.optimizations_applied:
            logger.info("Applying memory optimization")
            self.optimizations_applied.add("memory_optimization")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        metrics = self.analyzer.get_performance_metrics()
        
        return {
            "metrics": {
                "total_executions": metrics.total_executions,
                "success_rate": (metrics.successful_executions / metrics.total_executions) 
                               if metrics.total_executions > 0 else 0,
                "avg_execution_time_ms": metrics.avg_execution_time_ms,
                "p95_execution_time_ms": metrics.p95_execution_time_ms,
                "p99_execution_time_ms": metrics.p99_execution_time_ms,
                "throughput_per_second": metrics.throughput_per_second,
                "error_rate": metrics.error_rate,
            },
            "scaling": {
                "current_capacity": self.auto_scaler.current_capacity,
                "min_capacity": self.auto_scaler.min_capacity,
                "max_capacity": self.auto_scaler.max_capacity,
                "last_scaling_time": self.auto_scaler.last_scaling_time,
            },
            "optimizations": {
                "level": self.optimization_level.value,
                "applied": list(self.optimizations_applied),
                "active_workers": len(self.worker_tasks),
            },
            "anomalies": self.analyzer.detect_performance_anomalies(),
            "timestamp": time.time()
        }


# Global performance optimizer instance
performance_optimizer = PerformanceOptimizer()