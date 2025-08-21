"""
Generation 3 Enhancement: Auto-Scaling and Load Balancing System
Implements intelligent auto-scaling with predictive load balancing.
"""

import asyncio
import time
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque
import concurrent.futures

from .simple_structured_logging import get_logger

logger = get_logger(__name__)

class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RESPONSE_TIME = "weighted_response_time"
    PREDICTIVE = "predictive"

@dataclass
class WorkerMetrics:
    worker_id: str
    active_tasks: int
    completed_tasks: int
    avg_response_time_ms: float
    cpu_utilization: float
    memory_usage_mb: float
    error_count: int
    last_updated: float

@dataclass
class ScalingEvent:
    timestamp: float
    direction: ScalingDirection
    from_workers: int
    to_workers: int
    reason: str
    load_metrics: Dict[str, float]

class AutoScaler:
    """
    Generation 3: Intelligent auto-scaling system with predictive algorithms,
    load balancing, and resource optimization.
    """
    
    def __init__(self, 
                 min_workers: int = 3,
                 max_workers: int = 50,
                 target_cpu_threshold: float = 70.0,
                 scale_up_threshold: float = 80.0,
                 scale_down_threshold: float = 40.0):
        
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.target_cpu_threshold = target_cpu_threshold
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        # Worker management
        self._workers: Dict[str, WorkerMetrics] = {}
        self._worker_pool = concurrent.futures.ThreadPoolExecutor(max_workers=min_workers)
        self._current_worker_count = min_workers
        
        # Load balancing
        self._load_balancing_strategy = LoadBalancingStrategy.PREDICTIVE
        self._round_robin_index = 0
        
        # Metrics and monitoring
        self._load_history = deque(maxlen=100)  # Last 100 load measurements
        self._scaling_events: List[ScalingEvent] = []
        self._performance_metrics = deque(maxlen=1000)
        
        # Predictive scaling
        self._prediction_model = {
            'hourly_patterns': {},
            'trend_coefficient': 0.0,
            'seasonal_adjustments': {}
        }
        
        self._auto_scaling_enabled = True
        self._last_scaling_time = 0
        self._scaling_cooldown = 300  # 5 minutes between scaling events
        
        logger.info(f"Auto-scaler initialized: {min_workers}-{max_workers} workers, "
                   f"CPU thresholds: {scale_down_threshold}%-{scale_up_threshold}%")
    
    async def submit_task(self, task_func: Callable, *args, **kwargs) -> Any:
        """Submit a task to the auto-scaled worker pool."""
        # Select best worker using load balancing strategy
        worker_id = await self._select_worker()
        
        start_time = time.time()
        
        # Update worker metrics before execution
        if worker_id in self._workers:
            self._workers[worker_id].active_tasks += 1
        
        try:
            # Execute task in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._worker_pool, 
                self._execute_task_wrapper,
                task_func, args, kwargs, worker_id
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Update worker metrics after execution
            if worker_id in self._workers:
                worker = self._workers[worker_id]
                worker.active_tasks = max(0, worker.active_tasks - 1)
                worker.completed_tasks += 1
                # Update running average of response time
                worker.avg_response_time_ms = (
                    (worker.avg_response_time_ms * (worker.completed_tasks - 1) + execution_time) /
                    worker.completed_tasks
                )
                worker.last_updated = time.time()
            
            # Record performance metrics
            self._performance_metrics.append({
                'timestamp': time.time(),
                'execution_time_ms': execution_time,
                'worker_id': worker_id,
                'success': True
            })
            
            return result
            
        except Exception as e:
            # Update error count
            if worker_id in self._workers:
                self._workers[worker_id].error_count += 1
                self._workers[worker_id].active_tasks = max(0, self._workers[worker_id].active_tasks - 1)
            
            # Record failed execution
            self._performance_metrics.append({
                'timestamp': time.time(),
                'execution_time_ms': (time.time() - start_time) * 1000,
                'worker_id': worker_id,
                'success': False,
                'error': str(e)
            })
            
            raise
    
    def _execute_task_wrapper(self, task_func: Callable, args: tuple, kwargs: dict, worker_id: str) -> Any:
        """Wrapper function for task execution with metrics collection."""
        try:
            return task_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Task execution failed in worker {worker_id}: {e}")
            raise
    
    async def _select_worker(self) -> str:
        """Select the best worker based on load balancing strategy."""
        if not self._workers:
            # Initialize workers if none exist
            await self._initialize_workers()
        
        if self._load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            worker_ids = list(self._workers.keys())
            selected_id = worker_ids[self._round_robin_index % len(worker_ids)]
            self._round_robin_index += 1
            return selected_id
            
        elif self._load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Select worker with fewest active tasks
            return min(self._workers.keys(), 
                      key=lambda w_id: self._workers[w_id].active_tasks)
            
        elif self._load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_RESPONSE_TIME:
            # Select worker with best response time, weighted by current load
            def score_worker(w_id: str) -> float:
                worker = self._workers[w_id]
                load_factor = 1 + (worker.active_tasks * 0.2)  # Penalty for active tasks
                response_factor = worker.avg_response_time_ms / 1000  # Convert to seconds
                return load_factor * response_factor
            
            return min(self._workers.keys(), key=score_worker)
            
        elif self._load_balancing_strategy == LoadBalancingStrategy.PREDICTIVE:
            # Use predictive algorithm considering multiple factors
            return await self._predictive_worker_selection()
        
        # Fallback to round-robin
        return await self._select_worker()
    
    async def _predictive_worker_selection(self) -> str:
        """Advanced predictive worker selection using multiple factors."""
        current_time = time.time()
        scores = {}
        
        for worker_id, worker in self._workers.items():
            # Base score factors
            load_score = worker.active_tasks / 10.0  # Normalize active tasks
            response_score = worker.avg_response_time_ms / 1000.0  # Response time in seconds
            error_score = worker.error_count / max(1, worker.completed_tasks)  # Error rate
            
            # Time-based factors
            time_since_update = current_time - worker.last_updated
            freshness_score = min(1.0, time_since_update / 300)  # Penalty for stale metrics
            
            # Predictive factor: estimate future load
            predicted_load = await self._predict_worker_load(worker_id)
            
            # Combined score (lower is better)
            combined_score = (
                load_score * 0.3 +
                response_score * 0.25 +
                error_score * 0.2 +
                freshness_score * 0.15 +
                predicted_load * 0.1
            )
            
            scores[worker_id] = combined_score
        
        # Select worker with lowest score
        best_worker = min(scores.keys(), key=lambda k: scores[k])
        
        logger.debug(f"Predictive selection: {best_worker} (score: {scores[best_worker]:.3f})")
        return best_worker
    
    async def _predict_worker_load(self, worker_id: str) -> float:
        """Predict future load for a worker based on patterns."""
        # Simple prediction based on recent task completion rate
        current_time = time.time()
        recent_completions = [
            m for m in self._performance_metrics
            if (m['worker_id'] == worker_id and 
                current_time - m['timestamp'] < 300 and  # Last 5 minutes
                m['success'])
        ]
        
        if len(recent_completions) < 2:
            return 0.5  # Default prediction
        
        # Calculate completion rate (tasks per minute)
        time_span = max(300, recent_completions[-1]['timestamp'] - recent_completions[0]['timestamp'])
        completion_rate = len(recent_completions) / (time_span / 60)
        
        # Predict load as inverse of completion rate (normalized)
        predicted_load = max(0.0, min(1.0, 1.0 / max(0.1, completion_rate)))
        
        return predicted_load
    
    async def _initialize_workers(self):
        """Initialize worker metrics tracking."""
        for i in range(self._current_worker_count):
            worker_id = f"worker_{i}"
            self._workers[worker_id] = WorkerMetrics(
                worker_id=worker_id,
                active_tasks=0,
                completed_tasks=0,
                avg_response_time_ms=100.0,  # Default
                cpu_utilization=30.0,  # Default
                memory_usage_mb=256.0,  # Default
                error_count=0,
                last_updated=time.time()
            )
        
        logger.info(f"Initialized {len(self._workers)} workers")
    
    async def check_scaling_conditions(self):
        """Check if scaling is needed and execute scaling decisions."""
        if not self._auto_scaling_enabled:
            return
        
        current_time = time.time()
        
        # Cooldown check
        if current_time - self._last_scaling_time < self._scaling_cooldown:
            return
        
        # Calculate current load metrics
        load_metrics = await self._calculate_load_metrics()
        
        # Make scaling decision
        scaling_decision = await self._make_scaling_decision(load_metrics)
        
        if scaling_decision != ScalingDirection.STABLE:
            await self._execute_scaling(scaling_decision, load_metrics)
    
    async def _calculate_load_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive load metrics."""
        current_time = time.time()
        
        # Worker-based metrics
        if self._workers:
            avg_cpu = statistics.mean([w.cpu_utilization for w in self._workers.values()])
            avg_memory = statistics.mean([w.memory_usage_mb for w in self._workers.values()])
            total_active_tasks = sum([w.active_tasks for w in self._workers.values()])
            avg_response_time = statistics.mean([w.avg_response_time_ms for w in self._workers.values()])
            total_errors = sum([w.error_count for w in self._workers.values()])
            total_completed = sum([w.completed_tasks for w in self._workers.values()])
        else:
            avg_cpu = avg_memory = total_active_tasks = avg_response_time = 0
            total_errors = total_completed = 0
        
        # Performance-based metrics
        recent_metrics = [
            m for m in self._performance_metrics
            if current_time - m['timestamp'] < 300  # Last 5 minutes
        ]
        
        if recent_metrics:
            throughput = len(recent_metrics) / 5.0  # Tasks per minute
            success_rate = sum(1 for m in recent_metrics if m['success']) / len(recent_metrics)
            recent_avg_time = statistics.mean([m['execution_time_ms'] for m in recent_metrics])
        else:
            throughput = success_rate = recent_avg_time = 0
        
        load_metrics = {
            'avg_cpu_utilization': avg_cpu,
            'avg_memory_usage_mb': avg_memory,
            'total_active_tasks': total_active_tasks,
            'avg_response_time_ms': avg_response_time,
            'throughput_tasks_per_minute': throughput,
            'success_rate': success_rate,
            'error_rate': total_errors / max(1, total_completed),
            'worker_count': len(self._workers),
            'recent_avg_response_time_ms': recent_avg_time,
            'queue_depth': total_active_tasks / max(1, len(self._workers))
        }
        
        # Store in load history
        self._load_history.append({
            'timestamp': current_time,
            **load_metrics
        })
        
        return load_metrics
    
    async def _make_scaling_decision(self, load_metrics: Dict[str, float]) -> ScalingDirection:
        """Make intelligent scaling decision based on multiple factors."""
        cpu_util = load_metrics['avg_cpu_utilization']
        response_time = load_metrics['recent_avg_response_time_ms']
        queue_depth = load_metrics['queue_depth']
        success_rate = load_metrics['success_rate']
        
        # Scale up conditions
        scale_up_reasons = []
        if cpu_util > self.scale_up_threshold:
            scale_up_reasons.append(f"CPU utilization: {cpu_util:.1f}%")
        if response_time > 2000:  # 2 seconds
            scale_up_reasons.append(f"High response time: {response_time:.0f}ms")
        if queue_depth > 5:  # More than 5 tasks per worker on average
            scale_up_reasons.append(f"High queue depth: {queue_depth:.1f}")
        if success_rate < 0.9:  # Less than 90% success rate
            scale_up_reasons.append(f"Low success rate: {success_rate:.1%}")
        
        # Scale down conditions
        scale_down_reasons = []
        if (cpu_util < self.scale_down_threshold and 
            response_time < 500 and  # Less than 500ms
            queue_depth < 1 and  # Less than 1 task per worker
            success_rate > 0.95):  # Greater than 95% success
            scale_down_reasons.append("All metrics indicate low utilization")
        
        # Decision logic with hysteresis
        if scale_up_reasons and len(self._workers) < self.max_workers:
            logger.info(f"Scale up decision: {'; '.join(scale_up_reasons)}")
            return ScalingDirection.UP
        elif scale_down_reasons and len(self._workers) > self.min_workers:
            logger.info(f"Scale down decision: {'; '.join(scale_down_reasons)}")
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE
    
    async def _execute_scaling(self, direction: ScalingDirection, load_metrics: Dict[str, float]):
        """Execute scaling decision."""
        current_workers = len(self._workers)
        
        if direction == ScalingDirection.UP:
            # Scale up by 25% or at least 1 worker
            scale_amount = max(1, int(current_workers * 0.25))
            new_worker_count = min(self.max_workers, current_workers + scale_amount)
        else:  # ScalingDirection.DOWN
            # Scale down by 20% or at least 1 worker
            scale_amount = max(1, int(current_workers * 0.2))
            new_worker_count = max(self.min_workers, current_workers - scale_amount)
        
        if new_worker_count != current_workers:
            await self._adjust_worker_count(new_worker_count, direction)
            
            # Record scaling event
            scaling_event = ScalingEvent(
                timestamp=time.time(),
                direction=direction,
                from_workers=current_workers,
                to_workers=new_worker_count,
                reason=f"Load metrics triggered {direction.value} scaling",
                load_metrics=load_metrics.copy()
            )
            self._scaling_events.append(scaling_event)
            self._last_scaling_time = time.time()
            
            logger.info(f"Scaling {direction.value}: {current_workers} → {new_worker_count} workers")
    
    async def _adjust_worker_count(self, target_count: int, direction: ScalingDirection):
        """Adjust the actual worker count."""
        current_count = len(self._workers)
        
        if direction == ScalingDirection.UP:
            # Add new workers
            for i in range(current_count, target_count):
                worker_id = f"worker_{i}"
                self._workers[worker_id] = WorkerMetrics(
                    worker_id=worker_id,
                    active_tasks=0,
                    completed_tasks=0,
                    avg_response_time_ms=100.0,
                    cpu_utilization=30.0,
                    memory_usage_mb=256.0,
                    error_count=0,
                    last_updated=time.time()
                )
            
            # Expand thread pool
            self._worker_pool._max_workers = target_count
            
        elif direction == ScalingDirection.DOWN:
            # Remove excess workers (prefer workers with no active tasks)
            workers_to_remove = current_count - target_count
            idle_workers = [w_id for w_id, w in self._workers.items() if w.active_tasks == 0]
            
            # Remove idle workers first
            for worker_id in idle_workers[:workers_to_remove]:
                del self._workers[worker_id]
                workers_to_remove -= 1
            
            # If more workers need to be removed, remove those with least active tasks
            if workers_to_remove > 0:
                sorted_workers = sorted(self._workers.items(), key=lambda x: x[1].active_tasks)
                for worker_id, _ in sorted_workers[:workers_to_remove]:
                    del self._workers[worker_id]
            
            # Note: ThreadPoolExecutor doesn't support dynamic shrinking
            # In production, you'd implement a more sophisticated worker management system
        
        self._current_worker_count = len(self._workers)
        logger.debug(f"Adjusted worker count to {self._current_worker_count}")
    
    async def get_scaling_metrics(self) -> Dict[str, Any]:
        """Get comprehensive scaling and performance metrics."""
        current_time = time.time()
        load_metrics = await self._calculate_load_metrics()
        
        # Recent scaling events
        recent_events = [
            event for event in self._scaling_events
            if current_time - event.timestamp < 3600  # Last hour
        ]
        
        # Performance trends
        recent_load = [entry for entry in self._load_history if current_time - entry['timestamp'] < 3600]
        
        if recent_load:
            cpu_trend = statistics.mean([entry['avg_cpu_utilization'] for entry in recent_load[-10:]])
            response_trend = statistics.mean([entry['avg_response_time_ms'] for entry in recent_load[-10:]])
        else:
            cpu_trend = response_trend = 0
        
        return {
            "current_state": {
                "worker_count": len(self._workers),
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "auto_scaling_enabled": self._auto_scaling_enabled,
                "load_balancing_strategy": self._load_balancing_strategy.value
            },
            "load_metrics": load_metrics,
            "performance_trends": {
                "cpu_trend_percent": cpu_trend,
                "response_time_trend_ms": response_trend,
                "scaling_events_last_hour": len(recent_events)
            },
            "scaling_history": [
                {
                    "timestamp": event.timestamp,
                    "direction": event.direction.value,
                    "from_workers": event.from_workers,
                    "to_workers": event.to_workers,
                    "reason": event.reason
                }
                for event in recent_events
            ],
            "optimization_features": {
                "predictive_load_balancing": True,
                "intelligent_scaling_decisions": True,
                "multi_factor_worker_selection": True,
                "adaptive_thresholds": False  # Could be implemented
            }
        }

# Global auto-scaler instance
auto_scaler = AutoScaler()