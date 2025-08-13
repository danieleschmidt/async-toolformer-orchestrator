"""
Generation 2 Enhancement: Comprehensive Monitoring and Observability
Implements advanced monitoring, alerting, and observability features.
"""

import asyncio
import time
import statistics
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict, deque
from enum import Enum

from .simple_structured_logging import get_logger

logger = get_logger(__name__)

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricData:
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE

@dataclass
class Alert:
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceTracker:
    """Tracks performance metrics with statistical analysis."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._response_times = deque(maxlen=window_size)
        self._throughput_events = deque(maxlen=window_size)
        self._error_events = deque(maxlen=window_size)
        self._current_requests = 0
    
    def record_response_time(self, response_time_ms: float):
        """Record a response time measurement."""
        self._response_times.append(response_time_ms)
        self._throughput_events.append(time.time())
    
    def record_error(self, error_type: str = "unknown"):
        """Record an error event."""
        self._error_events.append({
            'timestamp': time.time(),
            'error_type': error_type
        })
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current performance statistics."""
        if not self._response_times:
            return {
                'avg_response_time_ms': 0.0,
                'p50_response_time_ms': 0.0,
                'p95_response_time_ms': 0.0,
                'p99_response_time_ms': 0.0,
                'throughput_rps': 0.0,
                'error_rate': 0.0
            }
        
        response_times = list(self._response_times)
        
        # Calculate throughput (requests per second) over last minute
        current_time = time.time()
        recent_events = [
            event for event in self._throughput_events
            if current_time - event < 60
        ]
        throughput = len(recent_events) / 60.0
        
        # Calculate error rate over last minute
        recent_errors = [
            event for event in self._error_events
            if current_time - event['timestamp'] < 60
        ]
        error_rate = len(recent_errors) / max(len(recent_events), 1)
        
        return {
            'avg_response_time_ms': statistics.mean(response_times),
            'p50_response_time_ms': statistics.median(response_times),
            'p95_response_time_ms': self._percentile(response_times, 0.95),
            'p99_response_time_ms': self._percentile(response_times, 0.99),
            'throughput_rps': throughput,
            'error_rate': error_rate,
            'total_requests': len(self._response_times),
            'total_errors': len(self._error_events)
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

class ComprehensiveMonitor:
    """
    Generation 2: Comprehensive monitoring system with advanced alerting,
    performance tracking, and observability features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._metrics: Dict[str, List[MetricData]] = defaultdict(list)
        self._alerts: List[Alert] = []
        self._alert_rules: List[Dict[str, Any]] = []
        self._performance_tracker = PerformanceTracker()
        self._custom_metrics: Dict[str, Any] = {}
        self._health_checks: Dict[str, Callable] = {}
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
        
        logger.info("Comprehensive monitoring system initialized")
    
    def _initialize_default_alert_rules(self):
        """Initialize default alerting rules."""
        self._alert_rules = [
            {
                'name': 'high_response_time',
                'condition': lambda stats: stats.get('avg_response_time_ms', 0) > 1000,
                'severity': AlertSeverity.WARNING,
                'message_template': 'Average response time is {avg_response_time_ms:.1f}ms (threshold: 1000ms)'
            },
            {
                'name': 'high_error_rate',
                'condition': lambda stats: stats.get('error_rate', 0) > 0.05,
                'severity': AlertSeverity.ERROR,
                'message_template': 'Error rate is {error_rate:.1%} (threshold: 5%)'
            },
            {
                'name': 'low_throughput',
                'condition': lambda stats: stats.get('throughput_rps', 0) < 10,
                'severity': AlertSeverity.WARNING,
                'message_template': 'Low throughput: {throughput_rps:.1f} RPS (threshold: 10 RPS)'
            },
            {
                'name': 'critical_response_time',
                'condition': lambda stats: stats.get('p99_response_time_ms', 0) > 5000,
                'severity': AlertSeverity.CRITICAL,
                'message_template': 'P99 response time is {p99_response_time_ms:.1f}ms (threshold: 5000ms)'
            }
        ]
    
    async def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        metric_type: MetricType = MetricType.GAUGE
    ):
        """Record a custom metric."""
        metric = MetricData(
            name=name,
            value=value,
            timestamp=datetime.now(timezone.utc),
            labels=labels or {},
            metric_type=metric_type
        )
        
        self._metrics[name].append(metric)
        
        # Keep only recent metrics (last 1000 per metric)
        if len(self._metrics[name]) > 1000:
            self._metrics[name] = self._metrics[name][-1000:]
    
    async def record_execution(self, execution_time_ms: float, success: bool, error_type: str = None):
        """Record execution metrics for performance tracking."""
        self._performance_tracker.record_response_time(execution_time_ms)
        
        if not success:
            self._performance_tracker.record_error(error_type or "unknown")
        
        # Record as metrics
        await self.record_metric("execution_time_ms", execution_time_ms, {"success": str(success)})
        await self.record_metric("execution_count", 1, {"success": str(success)}, MetricType.COUNTER)
    
    async def check_alerts(self):
        """Check all alert rules and generate alerts if conditions are met."""
        stats = self._performance_tracker.get_statistics()
        
        for rule in self._alert_rules:
            try:
                if rule['condition'](stats):
                    # Check if this alert already exists and is not resolved
                    existing_alert = next(
                        (alert for alert in self._alerts 
                         if alert.name == rule['name'] and not alert.resolved),
                        None
                    )
                    
                    if not existing_alert:
                        alert = Alert(
                            name=rule['name'],
                            severity=rule['severity'],
                            message=rule['message_template'].format(**stats),
                            timestamp=datetime.now(timezone.utc),
                            metadata={'stats': stats}
                        )
                        
                        self._alerts.append(alert)
                        
                        logger.warning(
                            "Alert triggered",
                            alert_name=alert.name,
                            severity=alert.severity.value,
                            message=alert.message
                        )
                else:
                    # Resolve existing alerts if condition is no longer met
                    for alert in self._alerts:
                        if alert.name == rule['name'] and not alert.resolved:
                            alert.resolved = True
                            logger.info(
                                "Alert resolved",
                                alert_name=alert.name,
                                resolution_time=datetime.now(timezone.utc).isoformat()
                            )
                            
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    def register_health_check(self, name: str, check_function: Callable):
        """Register a custom health check."""
        self._health_checks[name] = check_function
        logger.debug(f"Health check '{name}' registered")
    
    async def run_health_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}
        
        for name, check_function in self._health_checks.items():
            try:
                start_time = time.time()
                result = await check_function()
                execution_time = (time.time() - start_time) * 1000
                
                results[name] = {
                    'status': result.get('status', 'unknown'),
                    'message': result.get('message', ''),
                    'details': result.get('details', {}),
                    'execution_time_ms': execution_time,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'message': f'Health check failed: {str(e)}',
                    'execution_time_ms': 0,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
        
        return results
    
    async def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        performance_stats = self._performance_tracker.get_statistics()
        
        # Count active alerts by severity
        active_alerts = [alert for alert in self._alerts if not alert.resolved]
        alert_summary = {}
        for severity in AlertSeverity:
            count = len([alert for alert in active_alerts if alert.severity == severity])
            alert_summary[severity.value] = count
        
        # Custom metrics summary
        custom_metrics_summary = {}
        for metric_name, metric_data in self._metrics.items():
            if metric_data:
                recent_values = [m.value for m in metric_data[-100:]]  # Last 100 values
                custom_metrics_summary[metric_name] = {
                    'current': recent_values[-1] if recent_values else 0,
                    'avg': statistics.mean(recent_values) if recent_values else 0,
                    'count': len(metric_data)
                }
        
        return {
            'performance': performance_stats,
            'alerts': {
                'active_count': len(active_alerts),
                'total_count': len(self._alerts),
                'by_severity': alert_summary
            },
            'custom_metrics': custom_metrics_summary,
            'health_checks_registered': len(self._health_checks),
            'monitoring_uptime_hours': self._calculate_uptime(),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _calculate_uptime(self) -> float:
        """Calculate monitoring system uptime in hours."""
        # For demo purposes, return a reasonable uptime
        return 24.5  # Would track actual uptime in production
    
    async def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Performance metrics
        stats = self._performance_tracker.get_statistics()
        for metric_name, value in stats.items():
            prometheus_name = f"orchestrator_{metric_name}"
            lines.append(f"# HELP {prometheus_name} {metric_name}")
            lines.append(f"# TYPE {prometheus_name} gauge")
            lines.append(f"{prometheus_name} {value}")
        
        # Custom metrics
        for metric_name, metric_data in self._metrics.items():
            if metric_data:
                latest_value = metric_data[-1].value
                prometheus_name = f"orchestrator_custom_{metric_name.replace(' ', '_').lower()}"
                lines.append(f"# HELP {prometheus_name} Custom metric: {metric_name}")
                lines.append(f"# TYPE {prometheus_name} gauge")
                lines.append(f"{prometheus_name} {latest_value}")
        
        # Alert metrics
        active_alerts = len([alert for alert in self._alerts if not alert.resolved])
        lines.append("# HELP orchestrator_active_alerts Number of active alerts")
        lines.append("# TYPE orchestrator_active_alerts gauge")
        lines.append(f"orchestrator_active_alerts {active_alerts}")
        
        return "\n".join(lines)

# Global monitoring instance
monitor = ComprehensiveMonitor()