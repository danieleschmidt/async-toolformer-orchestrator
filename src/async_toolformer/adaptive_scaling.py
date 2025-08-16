"""Adaptive scaling system for Generation 3 optimization."""

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Direction of scaling operations."""
    UP = auto()
    DOWN = auto()
    STABLE = auto()


class ScalingTrigger(Enum):
    """Triggers for scaling decisions."""
    CPU_UTILIZATION = auto()
    MEMORY_USAGE = auto()
    QUEUE_DEPTH = auto()
    RESPONSE_TIME = auto()
    ERROR_RATE = auto()
    THROUGHPUT = auto()


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""

    cpu_utilization: float = 0.0
    memory_usage: float = 0.0
    queue_depth: int = 0
    average_response_time_ms: float = 0.0
    error_rate: float = 0.0
    throughput_requests_per_second: float = 0.0
    active_workers: int = 0
    pending_tasks: int = 0
    timestamp: float = field(default_factory=time.time)

    def get_utilization_score(self) -> float:
        """Calculate overall utilization score (0-1)."""
        weights = {
            'cpu': 0.3,
            'memory': 0.2,
            'queue': 0.2,
            'response_time': 0.2,
            'error_rate': 0.1
        }

        # Normalize metrics to 0-1 scale
        normalized_cpu = min(self.cpu_utilization, 1.0)
        normalized_memory = min(self.memory_usage, 1.0)
        normalized_queue = min(self.queue_depth / 100, 1.0)  # Assume max queue of 100
        normalized_response = min(self.average_response_time_ms / 5000, 1.0)  # 5s max
        normalized_error = min(self.error_rate, 1.0)

        score = (
            weights['cpu'] * normalized_cpu +
            weights['memory'] * normalized_memory +
            weights['queue'] * normalized_queue +
            weights['response_time'] * normalized_response +
            weights['error_rate'] * normalized_error
        )

        return score


@dataclass
class ScalingPolicy:
    """Policy for scaling decisions."""

    name: str
    scale_up_threshold: float = 0.7      # Scale up when utilization > 70%
    scale_down_threshold: float = 0.3     # Scale down when utilization < 30%
    min_workers: int = 2
    max_workers: int = 50
    scale_up_increment: int = 2           # Add 2 workers when scaling up
    scale_down_increment: int = 1         # Remove 1 worker when scaling down
    cooldown_seconds: float = 30.0        # Wait 30s between scaling operations
    evaluation_window: int = 5            # Evaluate over 5 metric samples

    # Advanced thresholds
    emergency_scale_threshold: float = 0.95  # Immediate scaling trigger
    error_rate_threshold: float = 0.1        # Scale up if error rate > 10%
    response_time_threshold_ms: float = 2000  # Scale up if response time > 2s


@dataclass
class ScalingEvent:
    """Record of a scaling event."""

    timestamp: float
    direction: ScalingDirection
    trigger: ScalingTrigger
    workers_before: int
    workers_after: int
    metrics: ScalingMetrics
    reason: str
    success: bool = True


class AdaptiveScaler:
    """Adaptive scaling system for dynamic resource management."""

    def __init__(self, policy: ScalingPolicy | None = None):
        self.policy = policy or ScalingPolicy(name="default")
        self.metrics_history: list[ScalingMetrics] = []
        self.scaling_events: list[ScalingEvent] = []
        self.last_scaling_time = 0.0
        self.current_workers = self.policy.min_workers
        self.worker_pool: list[asyncio.Task] = []
        self.task_queue = asyncio.Queue()
        self.metrics_lock = asyncio.Lock()

        # Performance tracking
        self.performance_baseline: ScalingMetrics | None = None
        self.learning_enabled = True
        self.adaptation_factor = 0.1  # How quickly to adapt thresholds

        # Predictive scaling
        self.prediction_enabled = True
        self.trend_window = 10  # Minutes for trend analysis

    async def start_monitoring(self) -> None:
        """Start the scaling monitoring loop."""
        asyncio.create_task(self._monitoring_loop())

    async def _monitoring_loop(self) -> None:
        """Continuous monitoring and scaling loop."""
        while True:
            try:
                # Collect current metrics
                metrics = await self._collect_metrics()

                async with self.metrics_lock:
                    self.metrics_history.append(metrics)
                    self._trim_metrics_history()

                # Make scaling decision
                scaling_decision = await self._make_scaling_decision(metrics)

                if scaling_decision != ScalingDirection.STABLE:
                    await self._execute_scaling(scaling_decision, metrics)

                # Adaptive learning
                if self.learning_enabled:
                    await self._adapt_thresholds()

                # Predictive scaling
                if self.prediction_enabled:
                    await self._predictive_scaling()

            except Exception as e:
                logger.error(f"Scaling monitoring loop error: {e}")

            # Check every 10 seconds
            await asyncio.sleep(10)

    async def _collect_metrics(self) -> ScalingMetrics:
        """Collect current system metrics."""

        # Simulate metric collection (in real implementation, gather from system)
        current_time = time.time()

        # Calculate queue depth
        queue_depth = self.task_queue.qsize()

        # Calculate response time from recent history
        recent_metrics = self.metrics_history[-5:] if self.metrics_history else []
        avg_response_time = 0.0
        if recent_metrics:
            avg_response_time = statistics.mean(m.average_response_time_ms for m in recent_metrics)

        # Calculate throughput
        throughput = 0.0
        if len(recent_metrics) >= 2:
            time_diff = recent_metrics[-1].timestamp - recent_metrics[0].timestamp
            if time_diff > 0:
                throughput = len(recent_metrics) / time_diff

        # Estimate CPU and memory usage based on active workers and queue
        cpu_utilization = min((self.current_workers + queue_depth) / self.policy.max_workers, 1.0)
        memory_usage = cpu_utilization * 0.8  # Assume memory scales with CPU

        # Calculate error rate (from recent scaling events)
        error_rate = 0.0
        recent_events = [e for e in self.scaling_events if current_time - e.timestamp < 300]
        if recent_events:
            failed_events = sum(1 for e in recent_events if not e.success)
            error_rate = failed_events / len(recent_events)

        return ScalingMetrics(
            cpu_utilization=cpu_utilization,
            memory_usage=memory_usage,
            queue_depth=queue_depth,
            average_response_time_ms=avg_response_time,
            error_rate=error_rate,
            throughput_requests_per_second=throughput,
            active_workers=self.current_workers,
            pending_tasks=queue_depth,
            timestamp=current_time
        )

    async def _make_scaling_decision(self, metrics: ScalingMetrics) -> ScalingDirection:
        """Make intelligent scaling decision based on metrics."""

        # Check cooldown period
        if time.time() - self.last_scaling_time < self.policy.cooldown_seconds:
            return ScalingDirection.STABLE

        # Get utilization score
        utilization_score = metrics.get_utilization_score()

        # Emergency scaling - immediate action required
        if utilization_score >= self.policy.emergency_scale_threshold:
            logger.warning(f"Emergency scaling triggered: utilization={utilization_score:.2f}")
            return ScalingDirection.UP

        # Error rate scaling
        if metrics.error_rate >= self.policy.error_rate_threshold:
            logger.warning(f"Scaling up due to high error rate: {metrics.error_rate:.2f}")
            return ScalingDirection.UP

        # Response time scaling
        if metrics.average_response_time_ms >= self.policy.response_time_threshold_ms:
            logger.warning(f"Scaling up due to high response time: {metrics.average_response_time_ms:.0f}ms")
            return ScalingDirection.UP

        # Standard threshold-based scaling
        if utilization_score >= self.policy.scale_up_threshold:
            if self.current_workers < self.policy.max_workers:
                logger.info(f"Scaling up: utilization={utilization_score:.2f}")
                return ScalingDirection.UP
        elif utilization_score <= self.policy.scale_down_threshold:
            if self.current_workers > self.policy.min_workers:
                # Ensure we don't scale down too aggressively
                if len(self.metrics_history) >= self.policy.evaluation_window:
                    recent_scores = [m.get_utilization_score() for m in self.metrics_history[-self.policy.evaluation_window:]]
                    avg_recent_score = statistics.mean(recent_scores)

                    if avg_recent_score <= self.policy.scale_down_threshold:
                        logger.info(f"Scaling down: avg_utilization={avg_recent_score:.2f}")
                        return ScalingDirection.DOWN

        return ScalingDirection.STABLE

    async def _execute_scaling(self, direction: ScalingDirection, metrics: ScalingMetrics) -> None:
        """Execute the scaling operation."""

        workers_before = self.current_workers
        trigger = self._determine_scaling_trigger(metrics)

        try:
            if direction == ScalingDirection.UP:
                new_workers = min(
                    self.current_workers + self.policy.scale_up_increment,
                    self.policy.max_workers
                )
                await self._scale_up_to(new_workers)

            elif direction == ScalingDirection.DOWN:
                new_workers = max(
                    self.current_workers - self.policy.scale_down_increment,
                    self.policy.min_workers
                )
                await self._scale_down_to(new_workers)

            # Record scaling event
            event = ScalingEvent(
                timestamp=time.time(),
                direction=direction,
                trigger=trigger,
                workers_before=workers_before,
                workers_after=self.current_workers,
                metrics=metrics,
                reason=f"Utilization: {metrics.get_utilization_score():.2f}",
                success=True
            )

            self.scaling_events.append(event)
            self.last_scaling_time = time.time()

            logger.info(
                f"Scaling {direction.name.lower()} completed",
                extra={
                    "workers_before": workers_before,
                    "workers_after": self.current_workers,
                    "trigger": trigger.name,
                    "utilization": metrics.get_utilization_score()
                }
            )

        except Exception as e:
            logger.error(f"Scaling operation failed: {e}")

            # Record failed scaling event
            event = ScalingEvent(
                timestamp=time.time(),
                direction=direction,
                trigger=trigger,
                workers_before=workers_before,
                workers_after=workers_before,  # No change due to failure
                metrics=metrics,
                reason=f"Failed: {str(e)}",
                success=False
            )
            self.scaling_events.append(event)

    def _determine_scaling_trigger(self, metrics: ScalingMetrics) -> ScalingTrigger:
        """Determine what triggered the scaling decision."""

        if metrics.error_rate >= self.policy.error_rate_threshold:
            return ScalingTrigger.ERROR_RATE
        elif metrics.average_response_time_ms >= self.policy.response_time_threshold_ms:
            return ScalingTrigger.RESPONSE_TIME
        elif metrics.queue_depth > 20:  # Arbitrary threshold
            return ScalingTrigger.QUEUE_DEPTH
        elif metrics.cpu_utilization >= 0.8:
            return ScalingTrigger.CPU_UTILIZATION
        elif metrics.memory_usage >= 0.8:
            return ScalingTrigger.MEMORY_USAGE
        else:
            return ScalingTrigger.THROUGHPUT

    async def _scale_up_to(self, target_workers: int) -> None:
        """Scale up worker pool to target size."""

        workers_to_add = target_workers - self.current_workers

        for _ in range(workers_to_add):
            worker_task = asyncio.create_task(self._worker_loop())
            self.worker_pool.append(worker_task)

        self.current_workers = target_workers

    async def _scale_down_to(self, target_workers: int) -> None:
        """Scale down worker pool to target size."""

        workers_to_remove = self.current_workers - target_workers

        # Cancel excess workers
        for _ in range(workers_to_remove):
            if self.worker_pool:
                worker = self.worker_pool.pop()
                worker.cancel()

        self.current_workers = target_workers

    async def _worker_loop(self) -> None:
        """Worker loop for processing tasks."""
        while True:
            try:
                # Wait for task from queue
                task = await self.task_queue.get()

                # Process task (placeholder implementation)
                time.time()

                if callable(task):
                    await task()
                else:
                    # Simulate processing
                    await asyncio.sleep(0.1)

                self.task_queue.task_done()

            except asyncio.CancelledError:
                # Worker being shut down
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")

    async def _adapt_thresholds(self) -> None:
        """Adapt scaling thresholds based on performance history."""

        if len(self.scaling_events) < 10:
            return  # Need more data

        # Analyze recent scaling events for effectiveness
        recent_events = self.scaling_events[-10:]

        # Count successful vs failed scaling events
        successful_scale_ups = sum(1 for e in recent_events
                                  if e.direction == ScalingDirection.UP and e.success)
        total_scale_ups = sum(1 for e in recent_events
                             if e.direction == ScalingDirection.UP)

        # Adapt scale-up threshold based on success rate
        if total_scale_ups > 0:
            success_rate = successful_scale_ups / total_scale_ups

            if success_rate < 0.7:  # Less than 70% success rate
                # Make scaling more conservative (higher threshold)
                adjustment = self.adaptation_factor * 0.1
                self.policy.scale_up_threshold = min(0.9, self.policy.scale_up_threshold + adjustment)
            elif success_rate > 0.9:  # More than 90% success rate
                # Make scaling more aggressive (lower threshold)
                adjustment = self.adaptation_factor * 0.1
                self.policy.scale_up_threshold = max(0.5, self.policy.scale_up_threshold - adjustment)

    async def _predictive_scaling(self) -> None:
        """Predictive scaling based on trends."""

        if len(self.metrics_history) < 20:
            return  # Need more historical data

        # Analyze trends in the last trend_window minutes
        current_time = time.time()
        trend_cutoff = current_time - (self.trend_window * 60)

        trend_metrics = [m for m in self.metrics_history if m.timestamp >= trend_cutoff]

        if len(trend_metrics) < 5:
            return

        # Calculate trend in utilization
        [m.timestamp for m in trend_metrics]
        utilizations = [m.get_utilization_score() for m in trend_metrics]

        # Simple linear trend calculation
        if len(utilizations) >= 3:
            recent_trend = utilizations[-1] - utilizations[-3]

            # Predict future utilization
            predicted_utilization = utilizations[-1] + recent_trend * 2  # Predict 2 steps ahead

            # Pre-emptive scaling if trend suggests it
            if predicted_utilization > self.policy.scale_up_threshold and recent_trend > 0.1:
                if self.current_workers < self.policy.max_workers:
                    logger.info(f"Predictive scaling up: trend={recent_trend:.3f}, predicted={predicted_utilization:.3f}")
                    await self._scale_up_to(self.current_workers + 1)

    def _trim_metrics_history(self) -> None:
        """Trim old metrics to prevent memory growth."""
        max_history = 1000
        if len(self.metrics_history) > max_history:
            self.metrics_history = self.metrics_history[-max_history:]

    async def add_task(self, task: Any) -> None:
        """Add task to the processing queue."""
        await self.task_queue.put(task)

    async def get_scaling_status(self) -> dict[str, Any]:
        """Get current scaling status and statistics."""

        if self.metrics_history:
            current_metrics = self.metrics_history[-1]
            utilization = current_metrics.get_utilization_score()
        else:
            utilization = 0.0

        # Calculate scaling statistics
        total_scaling_events = len(self.scaling_events)
        successful_events = sum(1 for e in self.scaling_events if e.success)

        recent_events = [e for e in self.scaling_events if time.time() - e.timestamp < 3600]

        return {
            "current_workers": self.current_workers,
            "min_workers": self.policy.min_workers,
            "max_workers": self.policy.max_workers,
            "current_utilization": utilization,
            "queue_depth": self.task_queue.qsize(),
            "scale_up_threshold": self.policy.scale_up_threshold,
            "scale_down_threshold": self.policy.scale_down_threshold,
            "total_scaling_events": total_scaling_events,
            "successful_scaling_events": successful_events,
            "recent_scaling_events": len(recent_events),
            "last_scaling_time": self.last_scaling_time,
            "time_since_last_scaling": time.time() - self.last_scaling_time,
            "learning_enabled": self.learning_enabled,
            "prediction_enabled": self.prediction_enabled,
        }

    async def get_performance_report(self) -> dict[str, Any]:
        """Generate performance and scaling effectiveness report."""

        if not self.scaling_events:
            return {"message": "No scaling events recorded yet"}

        # Analyze scaling effectiveness
        scale_up_events = [e for e in self.scaling_events if e.direction == ScalingDirection.UP]
        scale_down_events = [e for e in self.scaling_events if e.direction == ScalingDirection.DOWN]

        # Calculate average time between scaling events
        if len(self.scaling_events) > 1:
            time_diffs = []
            for i in range(1, len(self.scaling_events)):
                time_diff = self.scaling_events[i].timestamp - self.scaling_events[i-1].timestamp
                time_diffs.append(time_diff)

            avg_time_between_scaling = statistics.mean(time_diffs)
        else:
            avg_time_between_scaling = 0

        # Calculate utilization statistics
        if self.metrics_history:
            utilizations = [m.get_utilization_score() for m in self.metrics_history]
            avg_utilization = statistics.mean(utilizations)
            min_utilization = min(utilizations)
            max_utilization = max(utilizations)
        else:
            avg_utilization = min_utilization = max_utilization = 0

        return {
            "total_scaling_events": len(self.scaling_events),
            "scale_up_events": len(scale_up_events),
            "scale_down_events": len(scale_down_events),
            "successful_events": sum(1 for e in self.scaling_events if e.success),
            "failed_events": sum(1 for e in self.scaling_events if not e.success),
            "avg_time_between_scaling_seconds": avg_time_between_scaling,
            "utilization_stats": {
                "average": avg_utilization,
                "minimum": min_utilization,
                "maximum": max_utilization
            },
            "current_policy": {
                "scale_up_threshold": self.policy.scale_up_threshold,
                "scale_down_threshold": self.policy.scale_down_threshold,
                "min_workers": self.policy.min_workers,
                "max_workers": self.policy.max_workers,
            }
        }


# Global adaptive scaler instance
adaptive_scaler = AdaptiveScaler()
