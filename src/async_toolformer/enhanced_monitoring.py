"""
Enhanced Monitoring and Observability System - Generation 2 Implementation.

Provides comprehensive monitoring, alerting, and observability capabilities
for the Async Toolformer Orchestrator with real-time health tracking.
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Union

import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Represents a health check configuration."""
    name: str
    check_function: Callable[[], bool]
    interval_seconds: float
    timeout_seconds: float
    failure_threshold: int = 3
    recovery_threshold: int = 2
    critical: bool = False


@dataclass
class Alert:
    """Represents a system alert."""
    id: str
    severity: AlertSeverity
    message: str
    source: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_timestamp: Optional[float] = None


@dataclass
class MetricValue:
    """Represents a metric value with timestamp."""
    value: Union[int, float]
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self, retention_period: int = 3600):
        """
        Initialize metrics collector.
        
        Args:
            retention_period: How long to retain metrics in seconds
        """
        self.retention_period = retention_period
        self._metrics: Dict[str, Deque[MetricValue]] = defaultdict(lambda: deque(maxlen=1000))
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = defaultdict(float)
        
    def record_counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a counter metric."""
        self._counters[name] += value
        self._record_metric(name, value, labels or {})
        
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a gauge metric."""
        self._gauges[name] = value
        self._record_metric(name, value, labels or {})
        
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a histogram metric."""
        self._record_metric(f"{name}_value", value, labels or {})
        
    def _record_metric(self, name: str, value: float, labels: Dict[str, str]) -> None:
        """Internal method to record a metric."""
        metric_value = MetricValue(value=value, timestamp=time.time(), labels=labels)
        self._metrics[name].append(metric_value)
        self._cleanup_old_metrics()
        
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics beyond retention period."""
        cutoff_time = time.time() - self.retention_period
        
        for metric_name, values in self._metrics.items():
            while values and values[0].timestamp < cutoff_time:
                values.popleft()
                
    def get_metric_history(self, name: str, duration: int = 300) -> List[MetricValue]:
        """Get metric history for the specified duration."""
        cutoff_time = time.time() - duration
        return [v for v in self._metrics[name] if v.timestamp >= cutoff_time]
        
    def get_current_value(self, name: str) -> Optional[float]:
        """Get the most recent value for a metric."""
        if name not in self._metrics or not self._metrics[name]:
            return None
        return self._metrics[name][-1].value
        
    def get_average_value(self, name: str, duration: int = 300) -> Optional[float]:
        """Get average value over the specified duration."""
        history = self.get_metric_history(name, duration)
        if not history:
            return None
        return sum(v.value for v in history) / len(history)


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self._alerts: Dict[str, Alert] = {}
        self._alert_handlers: List[Callable[[Alert], None]] = []
        self._suppression_rules: Dict[str, float] = {}  # Alert ID -> suppression end time
        
    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert handler function."""
        self._alert_handlers.append(handler)
        
    def create_alert(
        self,
        id: str,
        severity: AlertSeverity,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Create and process a new alert."""
        # Check if alert is suppressed
        if id in self._suppression_rules:
            if time.time() < self._suppression_rules[id]:
                return self._alerts.get(id)  # Return existing suppressed alert
            else:
                del self._suppression_rules[id]  # Remove expired suppression
                
        alert = Alert(
            id=id,
            severity=severity,
            message=message,
            source=source,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        
        self._alerts[id] = alert
        
        # Notify handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error("Alert handler failed", handler=handler, error=str(e))
                
        return alert
        
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self._alerts:
            return False
            
        alert = self._alerts[alert_id]
        alert.resolved = True
        alert.resolution_timestamp = time.time()
        return True
        
    def suppress_alert(self, alert_id: str, duration_seconds: int) -> bool:
        """Suppress an alert for the specified duration."""
        if alert_id not in self._alerts:
            return False
            
        self._suppression_rules[alert_id] = time.time() + duration_seconds
        return True
        
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get all active alerts, optionally filtered by severity."""
        alerts = [a for a in self._alerts.values() if not a.resolved]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
            
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
        
    def get_alert_summary(self) -> Dict[str, int]:
        """Get summary of alerts by severity."""
        active_alerts = self.get_active_alerts()
        summary = {severity.value: 0 for severity in AlertSeverity}
        
        for alert in active_alerts:
            summary[alert.severity.value] += 1
            
        return summary


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        self.metrics = metrics_collector
        self.alerts = alert_manager
        self._health_checks: Dict[str, HealthCheck] = {}
        self._check_results: Dict[str, Deque[bool]] = defaultdict(lambda: deque(maxlen=10))
        self._health_status = HealthStatus.HEALTHY
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
    def register_health_check(self, health_check: HealthCheck) -> None:
        """Register a health check."""
        self._health_checks[health_check.name] = health_check
        
    def register_basic_health_checks(self) -> None:
        """Register basic health checks for the orchestrator."""
        
        def check_memory_usage() -> bool:
            """Check if memory usage is within acceptable limits."""
            # Simplified memory check - in production, use psutil
            import gc
            return len(gc.get_objects()) < 100000
            
        def check_error_rate() -> bool:
            """Check if error rate is acceptable."""
            error_rate = self.metrics.get_average_value("error_rate", 300)
            return error_rate is None or error_rate < 0.05  # Less than 5% error rate
            
        def check_response_time() -> bool:
            """Check if average response time is acceptable."""
            avg_response = self.metrics.get_average_value("response_time", 300)
            return avg_response is None or avg_response < 2.0  # Less than 2 seconds
            
        def check_resource_utilization() -> bool:
            """Check if resource utilization is reasonable."""
            cpu_usage = self.metrics.get_current_value("cpu_utilization")
            return cpu_usage is None or cpu_usage < 0.9  # Less than 90% CPU
            
        # Register health checks
        health_checks = [
            HealthCheck("memory_usage", check_memory_usage, 30.0, 5.0, critical=True),
            HealthCheck("error_rate", check_error_rate, 60.0, 10.0, critical=True),
            HealthCheck("response_time", check_response_time, 30.0, 5.0),
            HealthCheck("resource_utilization", check_resource_utilization, 60.0, 10.0),
        ]
        
        for check in health_checks:
            self.register_health_check(check)
            
    async def start_monitoring(self) -> None:
        """Start the health monitoring loop."""
        if self._running:
            return
            
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
        
    async def stop_monitoring(self) -> None:
        """Stop the health monitoring loop."""
        if not self._running:
            return
            
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Health monitoring stopped")
        
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                # Run all health checks
                check_tasks = []
                for check in self._health_checks.values():
                    task = asyncio.create_task(self._run_health_check(check))
                    check_tasks.append(task)
                    
                if check_tasks:
                    await asyncio.gather(*check_tasks, return_exceptions=True)
                    
                # Update overall health status
                await self._update_health_status()
                
                # Wait before next iteration
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in monitoring loop", error=str(e))
                await asyncio.sleep(30.0)  # Wait longer on error
                
    async def _run_health_check(self, check: HealthCheck) -> None:
        """Run a single health check."""
        try:
            # Run the check with timeout
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, check.check_function),
                timeout=check.timeout_seconds
            )
            
            # Record result
            self._check_results[check.name].append(result)
            
            # Check if we need to create/resolve alerts
            recent_results = list(self._check_results[check.name])[-check.failure_threshold:]
            
            if len(recent_results) >= check.failure_threshold and all(not r for r in recent_results):
                # Create failure alert
                severity = AlertSeverity.CRITICAL if check.critical else AlertSeverity.ERROR
                self.alerts.create_alert(
                    id=f"health_check_{check.name}",
                    severity=severity,
                    message=f"Health check '{check.name}' is failing",
                    source="health_monitor",
                    metadata={"check_name": check.name, "failure_count": len(recent_results)},
                )
            elif len(recent_results) >= check.recovery_threshold and all(r for r in recent_results[-check.recovery_threshold:]):
                # Resolve failure alert
                self.alerts.resolve_alert(f"health_check_{check.name}")
                
        except asyncio.TimeoutError:
            logger.warning("Health check timed out", check_name=check.name)
            self._check_results[check.name].append(False)
        except Exception as e:
            logger.error("Health check failed", check_name=check.name, error=str(e))
            self._check_results[check.name].append(False)
            
    async def _update_health_status(self) -> None:
        """Update the overall health status based on active alerts."""
        active_alerts = self.alerts.get_active_alerts()
        
        if not active_alerts:
            new_status = HealthStatus.HEALTHY
        elif any(a.severity == AlertSeverity.CRITICAL for a in active_alerts):
            new_status = HealthStatus.CRITICAL
        elif any(a.severity == AlertSeverity.ERROR for a in active_alerts):
            new_status = HealthStatus.UNHEALTHY
        elif any(a.severity == AlertSeverity.WARNING for a in active_alerts):
            new_status = HealthStatus.DEGRADED
        else:
            new_status = HealthStatus.HEALTHY
            
        if new_status != self._health_status:
            old_status = self._health_status
            self._health_status = new_status
            
            logger.info(
                "Health status changed",
                old_status=old_status.value,
                new_status=new_status.value,
                active_alerts=len(active_alerts),
            )
            
            # Record status change metric
            self.metrics.record_gauge("health_status", self._get_health_score())
            
    def _get_health_score(self) -> float:
        """Convert health status to a numeric score."""
        scores = {
            HealthStatus.HEALTHY: 1.0,
            HealthStatus.DEGRADED: 0.7,
            HealthStatus.UNHEALTHY: 0.3,
            HealthStatus.CRITICAL: 0.0,
        }
        return scores.get(self._health_status, 0.5)
        
    def get_health_status(self) -> HealthStatus:
        """Get current health status."""
        return self._health_status
        
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "overall_status": self._health_status.value,
            "health_score": self._get_health_score(),
            "active_alerts": len(self.alerts.get_active_alerts()),
            "alert_summary": self.alerts.get_alert_summary(),
            "health_checks": {
                name: {
                    "last_result": self._check_results[name][-1] if self._check_results[name] else None,
                    "success_rate": sum(self._check_results[name]) / len(self._check_results[name])
                    if self._check_results[name] else 0.0,
                }
                for name in self._health_checks.keys()
            },
            "timestamp": time.time(),
        }


class EnhancedLogger:
    """Enhanced structured logger with context management."""
    
    def __init__(self, name: str):
        self._logger = structlog.get_logger(name)
        self._context_stack: List[Dict[str, Any]] = []
        
    @asynccontextmanager
    async def operation_context(self, operation: str, **context):
        """Context manager for operation logging."""
        operation_id = f"{operation}_{int(time.time())}"
        full_context = {"operation": operation, "operation_id": operation_id, **context}
        
        self._context_stack.append(full_context)
        start_time = time.time()
        
        try:
            self._logger.info("Operation started", **full_context)
            yield operation_id
        except Exception as e:
            duration = time.time() - start_time
            error_context = {**full_context, "duration": duration, "error": str(e)}
            self._logger.error("Operation failed", **error_context)
            raise
        else:
            duration = time.time() - start_time
            success_context = {**full_context, "duration": duration}
            self._logger.info("Operation completed", **success_context)
        finally:
            self._context_stack.pop()
            
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        context = self._get_current_context()
        self._logger.info(message, **context, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        context = self._get_current_context()
        self._logger.warning(message, **context, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        context = self._get_current_context()
        self._logger.error(message, **context, **kwargs)
        
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        context = self._get_current_context()
        self._logger.debug(message, **context, **kwargs)
        
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current logging context."""
        context = {}
        for ctx in self._context_stack:
            context.update(ctx)
        return context


def default_alert_handler(alert: Alert) -> None:
    """Default alert handler that logs alerts."""
    logger = structlog.get_logger("alert_handler")
    
    log_method = {
        AlertSeverity.INFO: logger.info,
        AlertSeverity.WARNING: logger.warning,
        AlertSeverity.ERROR: logger.error,
        AlertSeverity.CRITICAL: logger.error,
    }.get(alert.severity, logger.info)
    
    log_method(
        f"Alert: {alert.message}",
        alert_id=alert.id,
        severity=alert.severity.value,
        source=alert.source,
        metadata=alert.metadata,
    )


def create_enhanced_monitoring_system() -> tuple[MetricsCollector, AlertManager, HealthMonitor]:
    """Create a complete enhanced monitoring system."""
    metrics_collector = MetricsCollector()
    alert_manager = AlertManager()
    health_monitor = HealthMonitor(metrics_collector, alert_manager)
    
    # Add default alert handler
    alert_manager.add_alert_handler(default_alert_handler)
    
    # Register basic health checks
    health_monitor.register_basic_health_checks()
    
    return metrics_collector, alert_manager, health_monitor


# Decorators for automatic monitoring
def monitor_execution(metric_name: str, metrics_collector: MetricsCollector):
    """Decorator to monitor function execution time and success rate."""
    
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            
            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                metrics_collector.record_counter(f"{metric_name}_errors")
                raise
            finally:
                execution_time = time.time() - start_time
                metrics_collector.record_histogram(f"{metric_name}_duration", execution_time)
                metrics_collector.record_counter(
                    f"{metric_name}_executions",
                    labels={"success": str(success).lower()}
                )
                
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = False
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                metrics_collector.record_counter(f"{metric_name}_errors")
                raise
            finally:
                execution_time = time.time() - start_time
                metrics_collector.record_histogram(f"{metric_name}_duration", execution_time)
                metrics_collector.record_counter(
                    f"{metric_name}_executions",
                    labels={"success": str(success).lower()}
                )
                
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
    return decorator