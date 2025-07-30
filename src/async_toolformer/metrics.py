"""Metrics collection and monitoring for the Async Toolformer Orchestrator."""

import time
from typing import Dict, Any, Optional, List
from functools import wraps
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import asyncio
import logging

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Info, start_http_server,
        CollectorRegistry, CONTENT_TYPE_LATEST, generate_latest
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    from opentelemetry import trace, metrics as otel_metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Container for metric values."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Centralized metrics collection system."""
    
    def __init__(self, registry: Optional[Any] = None):
        self.registry = registry or (CollectorRegistry() if PROMETHEUS_AVAILABLE else None)
        self._metrics: Dict[str, Any] = {}
        self._custom_metrics: List[MetricValue] = []
        self._enabled = PROMETHEUS_AVAILABLE
        
        if self._enabled:
            self._setup_default_metrics()
    
    def _setup_default_metrics(self) -> None:
        """Setup default Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            return
            
        # Core orchestrator metrics
        self._metrics['tool_executions_total'] = Counter(
            'async_orchestrator_tool_executions_total',
            'Total number of tool executions',
            ['tool_name', 'status'],
            registry=self.registry
        )
        
        self._metrics['tool_duration_seconds'] = Histogram(
            'async_orchestrator_tool_duration_seconds',
            'Tool execution duration in seconds',
            ['tool_name'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self._metrics['parallel_executions'] = Gauge(
            'async_orchestrator_parallel_executions',
            'Current number of parallel tool executions',
            registry=self.registry
        )
        
        self._metrics['rate_limit_hits_total'] = Counter(
            'async_orchestrator_rate_limit_hits_total',
            'Total number of rate limit hits',
            ['service', 'limit_type'],
            registry=self.registry
        )
        
        self._metrics['speculation_outcomes_total'] = Counter(
            'async_orchestrator_speculation_outcomes_total',
            'Speculation execution outcomes',
            ['outcome'],  # hit, miss, timeout, error
            registry=self.registry
        )
        
        self._metrics['active_orchestrators'] = Gauge(
            'async_orchestrator_active_instances',
            'Number of active orchestrator instances',
            registry=self.registry
        )
        
        self._metrics['memory_usage_bytes'] = Gauge(
            'async_orchestrator_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        self._metrics['queue_size'] = Gauge(
            'async_orchestrator_queue_size',
            'Size of various internal queues',
            ['queue_type'],
            registry=self.registry
        )
        
        # System info
        self._metrics['info'] = Info(
            'async_orchestrator_info',
            'Information about the orchestrator',
            registry=self.registry
        )
    
    def increment_counter(self, name: str, labels: Optional[Dict[str, str]] = None, value: float = 1.0) -> None:
        """Increment a counter metric."""
        if not self._enabled or name not in self._metrics:
            return
        
        metric = self._metrics[name]
        if labels:
            metric.labels(**labels).inc(value)
        else:
            metric.inc(value)
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value in a histogram metric."""
        if not self._enabled or name not in self._metrics:
            return
        
        metric = self._metrics[name]
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value."""
        if not self._enabled or name not in self._metrics:
            return
        
        metric = self._metrics[name]
        if labels:
            metric.labels(**labels).set(value)
        else:
            metric.set(value)
    
    def set_info(self, name: str, info: Dict[str, str]) -> None:
        """Set info metric."""
        if not self._enabled or name not in self._metrics:
            return
        
        self._metrics[name].info(info)
    
    def add_custom_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Add a custom metric value."""
        self._custom_metrics.append(MetricValue(
            name=name,
            value=value,
            labels=labels or {},
        ))
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format."""
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus not available\n"
        
        return generate_latest(self.registry).decode('utf-8')
    
    def get_custom_metrics(self) -> List[MetricValue]:
        """Get custom metrics and clear the buffer."""
        metrics = self._custom_metrics.copy()
        self._custom_metrics.clear()
        return metrics


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def track_metric(metric_name: str, metric_type: str = "counter", labels: Optional[Dict[str, str]] = None):
    """Decorator to track function execution metrics."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                if metric_type == "counter":
                    collector.increment_counter(f"{metric_name}_total", labels)
                elif metric_type == "histogram":
                    collector.observe_histogram(f"{metric_name}_duration_seconds", duration, labels)
                elif metric_type == "custom":
                    collector.add_custom_metric(metric_name, duration, labels)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                error_labels = (labels or {}).copy()
                error_labels['error'] = type(e).__name__
                
                collector.increment_counter(f"{metric_name}_errors_total", error_labels)
                collector.observe_histogram(f"{metric_name}_duration_seconds", duration, error_labels)
                
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if metric_type == "counter":
                    collector.increment_counter(f"{metric_name}_total", labels)
                elif metric_type == "histogram":
                    collector.observe_histogram(f"{metric_name}_duration_seconds", duration, labels)
                elif metric_type == "custom":
                    collector.add_custom_metric(metric_name, duration, labels)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                error_labels = (labels or {}).copy()
                error_labels['error'] = type(e).__name__
                
                collector.increment_counter(f"{metric_name}_errors_total", error_labels)
                collector.observe_histogram(f"{metric_name}_duration_seconds", duration, error_labels)
                
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


@asynccontextmanager
async def measure_time(metric_name: str, labels: Optional[Dict[str, str]] = None):
    """Context manager to measure execution time."""
    collector = get_metrics_collector()
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        collector.observe_histogram(f"{metric_name}_duration_seconds", duration, labels)


class MetricsServer:
    """HTTP server for exposing Prometheus metrics."""
    
    def __init__(self, port: int = 8000, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.collector = get_metrics_collector()
        self._server = None
        self._running = False
    
    async def start(self) -> None:
        """Start the metrics server."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available, metrics server not started")
            return
        
        try:
            # Start Prometheus HTTP server in a thread
            start_http_server(self.port, self.host, registry=self.collector.registry)
            self._running = True
            logger.info(f"Metrics server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def stop(self) -> None:
        """Stop the metrics server."""
        if self._server:
            self._server.close()
            self._running = False
            logger.info("Metrics server stopped")
    
    def is_running(self) -> bool:
        """Check if the metrics server is running."""
        return self._running


class OpenTelemetrySetup:
    """Setup OpenTelemetry tracing and metrics."""
    
    def __init__(self, service_name: str = "async-toolformer", service_version: str = "0.1.0"):
        self.service_name = service_name
        self.service_version = service_version
        self._tracer = None
        self._meter = None
        
        if OPENTELEMETRY_AVAILABLE:
            self._setup_tracing()
            self._setup_metrics()
    
    def _setup_tracing(self) -> None:
        """Setup distributed tracing."""
        try:
            # Configure the tracer provider
            trace.set_tracer_provider(
                trace.TracerProvider(
                    resource=trace.Resource.create({
                        "service.name": self.service_name,
                        "service.version": self.service_version,
                    })
                )
            )
            
            self._tracer = trace.get_tracer(__name__)
            logger.info("OpenTelemetry tracing configured")
            
        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry tracing: {e}")
    
    def _setup_metrics(self) -> None:
        """Setup OpenTelemetry metrics."""
        try:
            # Setup metrics provider with Prometheus reader
            reader = PrometheusMetricReader()
            provider = MeterProvider(metric_readers=[reader])
            otel_metrics.set_meter_provider(provider)
            
            self._meter = otel_metrics.get_meter(__name__)
            logger.info("OpenTelemetry metrics configured")
            
        except Exception as e:
            logger.error(f"Failed to setup OpenTelemetry metrics: {e}")
    
    def get_tracer(self):
        """Get the configured tracer."""
        return self._tracer
    
    def get_meter(self):
        """Get the configured meter."""
        return self._meter
    
    @asynccontextmanager
    async def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a traced span."""
        if not self._tracer:
            yield
            return
        
        with self._tracer.start_as_current_span(name) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            yield span


# Global OpenTelemetry setup
_otel_setup: Optional[OpenTelemetrySetup] = None


def get_otel_setup() -> OpenTelemetrySetup:
    """Get the global OpenTelemetry setup."""
    global _otel_setup
    if _otel_setup is None:
        _otel_setup = OpenTelemetrySetup()
    return _otel_setup


def initialize_observability(
    service_name: str = "async-toolformer",
    metrics_port: int = 8000,
    enable_tracing: bool = True,
    enable_metrics: bool = True,
) -> Dict[str, Any]:
    """Initialize complete observability stack."""
    components = {}
    
    if enable_metrics:
        # Setup metrics collector
        collector = get_metrics_collector()
        components['metrics_collector'] = collector
        
        # Start metrics server
        if PROMETHEUS_AVAILABLE:
            server = MetricsServer(port=metrics_port)
            components['metrics_server'] = server
        
        # Set initial info
        collector.set_info('info', {
            'version': '0.1.0',
            'service': service_name,
            'python_version': '3.11+',
        })
    
    if enable_tracing:
        # Setup OpenTelemetry
        otel_setup = get_otel_setup()
        components['otel_setup'] = otel_setup
    
    logger.info(f"Observability initialized with components: {list(components.keys())}")
    return components