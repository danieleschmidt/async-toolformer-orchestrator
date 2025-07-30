# Operational Excellence Framework

Comprehensive operational practices for the Async Toolformer Orchestrator, designed for production-grade reliability, monitoring, and incident management.

## Operational Philosophy

**Reliability First**: Zero-downtime deployments, graceful degradation, and comprehensive monitoring.

**Observability-Driven**: Every component instrumented with metrics, logs, and traces.

**Automation-Centric**: Manual operations minimized through intelligent automation.

## Service Level Objectives (SLOs)

### Performance SLOs
- **Availability**: 99.9% uptime (8.76 hours downtime/year maximum)
- **Latency**: P95 response time < 2 seconds for tool execution
- **Throughput**: Support 1000+ concurrent tool executions
- **Error Rate**: < 0.1% of requests result in internal errors

### Business SLOs  
- **API Key Management**: Zero key exposure incidents
- **Cost Efficiency**: Maintain cost per request within 10% variance
- **Compliance**: 100% compliance with data retention policies

## Monitoring and Observability

### 1. Comprehensive Metrics

#### Application Metrics
Enhanced Prometheus metrics for deep observability:

```python
# src/async_toolformer/observability/metrics.py
"""Comprehensive application metrics."""

from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from functools import wraps
from typing import Callable, Any

# Core orchestrator metrics
orchestrator_requests_total = Counter(
    'orchestrator_requests_total',
    'Total orchestrator requests',
    ['method', 'provider', 'status']
)

orchestrator_request_duration = Histogram(
    'orchestrator_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'provider'],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

active_tool_executions = Gauge(
    'active_tool_executions',
    'Number of currently executing tools',
    ['tool_type']
)

# Resource utilization metrics
memory_usage_bytes = Gauge(
    'memory_usage_bytes',
    'Memory usage in bytes',
    ['component']
)

cpu_usage_percent = Gauge(
    'cpu_usage_percent', 
    'CPU usage percentage',
    ['component']
)

# Business metrics
api_costs_total = Counter(
    'api_costs_total',
    'Total API costs',
    ['provider', 'model']
)

speculation_accuracy = Histogram(
    'speculation_accuracy_ratio',
    'Accuracy of speculative execution',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Custom metrics decorator
def track_execution_metrics(operation: str):
    """Decorator to track execution metrics."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                raise
            finally:
                duration = time.time() - start_time
                orchestrator_request_duration.labels(
                    method=operation,
                    provider=kwargs.get('provider', 'unknown')
                ).observe(duration)
                
                orchestrator_requests_total.labels(
                    method=operation,
                    provider=kwargs.get('provider', 'unknown'),
                    status=status
                ).inc()
        
        return wrapper
    return decorator
```

#### Infrastructure Metrics
System-level monitoring:

```yaml
# config/monitoring/node-exporter.yml
# Node exporter configuration for infrastructure metrics
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "infrastructure_rules.yml"
  - "application_rules.yml"

scrape_configs:
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 5s
    metrics_path: /metrics
    
  - job_name: 'async-toolformer'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5s
    metrics_path: /metrics
```

### 2. Structured Logging

#### Advanced Log Configuration
Production-ready logging with correlation:

```python
# src/async_toolformer/observability/logging.py
"""Advanced structured logging configuration."""

import structlog
import logging
import sys
from typing import Dict, Any
from datetime import datetime
import json

def configure_logging(
    log_level: str = "INFO",
    service_name: str = "async-toolformer",
    environment: str = "production"
) -> None:
    """Configure structured logging for production."""
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            add_service_metadata,
            add_request_correlation,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer() if environment == "development" 
            else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def add_service_metadata(logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add service metadata to log entries."""
    event_dict.update({
        "service": "async-toolformer-orchestrator",
        "version": "0.1.0",
        "environment": "production"
    })
    return event_dict

def add_request_correlation(logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add request correlation ID to log entries."""
    # Implementation to extract correlation ID from context
    correlation_id = getattr(logger, '_correlation_id', None)
    if correlation_id:
        event_dict['correlation_id'] = correlation_id
    return event_dict

# Usage example
logger = structlog.get_logger()

@track_execution_metrics("orchestrator_execute")
async def execute_with_logging(orchestrator, prompt: str) -> Dict[str, Any]:
    """Execute orchestrator with comprehensive logging."""
    logger.info(
        "Starting orchestrator execution",
        prompt_length=len(prompt),
        max_parallel=orchestrator.max_parallel
    )
    
    try:
        result = await orchestrator.execute(prompt)
        logger.info(
            "Orchestrator execution completed successfully",
            tools_executed=len(result.get('tools', [])),
            duration=result.get('duration', 0),
            parallel_efficiency=result.get('parallel_efficiency', 0)
        )
        return result
        
    except Exception as e:
        logger.error(
            "Orchestrator execution failed",
            error=str(e),
            error_type=type(e).__name__,
            prompt_length=len(prompt)
        )
        raise
```

### 3. Distributed Tracing

#### OpenTelemetry Integration
End-to-end request tracing:

```python
# src/async_toolformer/observability/tracing.py
"""Distributed tracing with OpenTelemetry."""

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
import asyncio
from functools import wraps
from typing import Dict, Any, Optional

class TracingManager:
    """Manages distributed tracing configuration."""
    
    def __init__(
        self,
        service_name: str = "async-toolformer",
        jaeger_endpoint: str = "http://localhost:14268/api/traces",
        sample_rate: float = 0.1
    ):
        self.service_name = service_name
        self.tracer_provider = TracerProvider()
        trace.set_tracer_provider(self.tracer_provider)
        
        # Configure Jaeger exporter
        jaeger_exporter = JaegerExporter(
            endpoint=jaeger_endpoint,
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        self.tracer_provider.add_span_processor(span_processor)
        
        # Get tracer
        self.tracer = trace.get_tracer(service_name)
        
        # Auto-instrument libraries
        AioHttpClientInstrumentor().instrument()
        RedisInstrumentor().instrument()
    
    def trace_async_function(self, operation_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Decorator to trace async functions."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(operation_name) as span:
                    # Add custom attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, value)
                    
                    # Add function arguments as attributes
                    span.set_attribute("function.args_count", len(args))
                    span.set_attribute("function.kwargs_count", len(kwargs))
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("function.success", True)
                        return result
                    except Exception as e:
                        span.set_attribute("function.success", False)
                        span.set_attribute("function.error", str(e))
                        span.record_exception(e)
                        raise
            return wrapper
        return decorator

# Global tracing manager
tracing_manager = TracingManager()

# Usage decorators
trace_tool_execution = tracing_manager.trace_async_function("tool_execution")
trace_llm_request = tracing_manager.trace_async_function("llm_request")
trace_orchestrator_execute = tracing_manager.trace_async_function("orchestrator_execute")
```

## Alerting and Incident Management

### 1. Alert Rules

#### Prometheus Alert Rules
Comprehensive alerting for all critical conditions:

```yaml
# config/monitoring/alert-rules.yml
groups:
  - name: async-toolformer-alerts
    rules:
      # High error rate alert
      - alert: HighErrorRate
        expr: rate(orchestrator_requests_total{status="error"}[5m]) > 0.01
        for: 2m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
          
      # High latency alert  
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(orchestrator_request_duration_seconds_bucket[5m])) > 2.0
        for: 5m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High latency detected"
          description: "95th percentile latency is {{ $value }} seconds"
          
      # Memory usage alert
      - alert: HighMemoryUsage
        expr: memory_usage_bytes / 1024 / 1024 / 1024 > 1.0
        for: 3m
        labels:
          severity: warning
          team: platform
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}GB"
          
      # API cost alert
      - alert: HighAPICosts
        expr: increase(api_costs_total[1h]) > 100
        for: 0m
        labels:
          severity: warning
          team: product
        annotations:
          summary: "High API costs detected"
          description: "API costs have increased by ${{ $value }} in the last hour"
          
      # Service down alert
      - alert: ServiceDown
        expr: up{job="async-toolformer"} == 0
        for: 1m
        labels:
          severity: critical
          team: platform
        annotations:
          summary: "Service is down"
          description: "Async Toolformer service is not responding"
```

### 2. Incident Response

#### Automated Incident Response
Intelligent automated responses to common issues:

```python
# src/async_toolformer/operations/incident_response.py
"""Automated incident response system."""

import asyncio
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()

@dataclass
class IncidentRule:
    """Incident response rule definition."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    actions: List[Callable[[Dict[str, Any]], Any]]
    cooldown_minutes: int = 5
    max_executions_per_hour: int = 10

class IncidentResponseManager:
    """Manages automated incident response."""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.execution_history: Dict[str, List[datetime]] = {}
    
    async def process_alert(self, alert_data: Dict[str, Any]) -> None:
        """Process incoming alert and execute appropriate responses."""
        logger.info("Processing alert", alert=alert_data)
        
        for rule in self.rules:
            if rule.condition(alert_data):
                if await self._can_execute_rule(rule.name):
                    await self._execute_rule(rule, alert_data)
                else:
                    logger.warning(
                        "Rule execution skipped due to rate limiting",
                        rule=rule.name
                    )
    
    async def _execute_rule(self, rule: IncidentRule, alert_data: Dict[str, Any]) -> None:
        """Execute incident response rule."""
        logger.info("Executing incident response rule", rule=rule.name)
        
        for action in rule.actions:
            try:
                await action(alert_data)
                logger.info("Action executed successfully", action=action.__name__)
            except Exception as e:
                logger.error(
                    "Action execution failed",
                    action=action.__name__,
                    error=str(e)
                )
        
        # Record execution
        if rule.name not in self.execution_history:
            self.execution_history[rule.name] = []
        self.execution_history[rule.name].append(datetime.utcnow())
    
    def _initialize_rules(self) -> List[IncidentRule]:
        """Initialize incident response rules."""
        return [
            IncidentRule(
                name="high_error_rate_response",
                condition=lambda alert: alert.get("alertname") == "HighErrorRate",
                actions=[
                    self._scale_up_replicas,
                    self._enable_circuit_breaker,
                    self._notify_on_call_engineer
                ]
            ),
            IncidentRule(
                name="high_memory_response", 
                condition=lambda alert: alert.get("alertname") == "HighMemoryUsage",
                actions=[
                    self._trigger_garbage_collection,
                    self._scale_up_replicas,
                    self._enable_request_throttling
                ]
            ),
            IncidentRule(
                name="service_down_response",
                condition=lambda alert: alert.get("alertname") == "ServiceDown",
                actions=[
                    self._restart_service,
                    self._check_dependencies,
                    self._escalate_to_oncall
                ]
            )
        ]
    
    async def _scale_up_replicas(self, alert_data: Dict[str, Any]) -> None:
        """Scale up service replicas."""
        logger.info("Scaling up replicas in response to incident")
        # Implementation for scaling (e.g., Kubernetes API call)
    
    async def _enable_circuit_breaker(self, alert_data: Dict[str, Any]) -> None:
        """Enable circuit breaker for failing services."""
        logger.info("Enabling circuit breaker")
        # Implementation for circuit breaker
    
    async def _notify_on_call_engineer(self, alert_data: Dict[str, Any]) -> None:
        """Notify on-call engineer."""
        logger.info("Notifying on-call engineer")
        # Implementation for notification (PagerDuty, Slack, etc.)
```

## Capacity Planning

### 1. Resource Forecasting

#### Predictive Scaling
ML-based capacity planning:

```python
# src/async_toolformer/operations/capacity_planning.py
"""Capacity planning and predictive scaling."""

import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()

class CapacityPlanner:
    """Handles capacity planning and resource forecasting."""
    
    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector
        self.models = {}
    
    async def forecast_resource_needs(
        self,
        forecast_horizon_hours: int = 24
    ) -> Dict[str, Any]:
        """Forecast resource needs based on historical patterns."""
        
        # Collect historical metrics
        historical_data = await self._collect_historical_metrics()
        
        # Generate forecasts
        forecasts = {}
        for metric_name, data in historical_data.items():
            model = self._train_forecasting_model(data)
            forecast = self._generate_forecast(model, forecast_horizon_hours)
            forecasts[metric_name] = forecast
        
        # Generate scaling recommendations
        recommendations = self._generate_scaling_recommendations(forecasts)
        
        logger.info(
            "Generated capacity forecast",
            horizon_hours=forecast_horizon_hours,
            recommendations=len(recommendations)
        )
        
        return {
            "forecasts": forecasts,
            "recommendations": recommendations,
            "generated_at": datetime.utcnow(),
            "horizon_hours": forecast_horizon_hours
        }
    
    def _train_forecasting_model(self, data: List[Tuple[datetime, float]]) -> LinearRegression:
        """Train simple forecasting model."""
        if len(data) < 10:
            logger.warning("Insufficient data for forecasting")
            return None
        
        # Convert timestamps to numerical features
        X = np.array([[
            ts.hour,
            ts.weekday(),
            (ts - data[0][0]).total_seconds() / 3600
        ] for ts, _ in data])
        
        y = np.array([value for _, value in data])
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model
    
    def _generate_scaling_recommendations(
        self,
        forecasts: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate scaling recommendations based on forecasts."""
        recommendations = []
        
        # CPU-based recommendations
        if "cpu_usage" in forecasts:
            max_cpu = max(forecasts["cpu_usage"])
            if max_cpu > 80:
                recommendations.append({
                    "type": "scale_up",
                    "resource": "cpu",
                    "current_threshold": 80,
                    "predicted_max": max_cpu,
                    "action": "Increase CPU allocation by 50%"
                })
        
        # Memory-based recommendations
        if "memory_usage" in forecasts:
            max_memory = max(forecasts["memory_usage"])
            if max_memory > 1000:  # 1GB
                recommendations.append({
                    "type": "scale_up", 
                    "resource": "memory",
                    "current_threshold": 1000,
                    "predicted_max": max_memory,
                    "action": "Increase memory allocation by 25%"
                })
        
        return recommendations
```

### 2. Cost Optimization

#### Intelligent Cost Management
Automated cost optimization strategies:

```python
# src/async_toolformer/operations/cost_optimization.py
"""Cost optimization and resource efficiency."""

from typing import Dict, Any, List
import structlog
from datetime import datetime, timedelta

logger = structlog.get_logger()

class CostOptimizer:
    """Manages cost optimization strategies."""
    
    def __init__(self):
        self.optimization_strategies = [
            self._optimize_api_usage,
            self._optimize_speculation_threshold,
            self._optimize_caching_strategy,
            self._optimize_resource_allocation
        ]
    
    async def run_cost_optimization(self) -> Dict[str, Any]:
        """Run all cost optimization strategies."""
        optimization_results = []
        total_savings = 0
        
        for strategy in self.optimization_strategies:
            try:
                result = await strategy()
                optimization_results.append(result)
                total_savings += result.get("estimated_savings", 0)
                
                logger.info(
                    "Cost optimization strategy completed",
                    strategy=strategy.__name__,
                    savings=result.get("estimated_savings", 0)
                )
                
            except Exception as e:
                logger.error(
                    "Cost optimization strategy failed",
                    strategy=strategy.__name__,
                    error=str(e)
                )
        
        return {
            "results": optimization_results,
            "total_estimated_savings": total_savings,
            "optimization_timestamp": datetime.utcnow()
        }
    
    async def _optimize_api_usage(self) -> Dict[str, Any]:
        """Optimize API usage patterns."""
        # Analyze API usage patterns
        # Suggest model changes for better cost efficiency
        # Implement intelligent caching
        
        return {
            "strategy": "api_usage_optimization",
            "estimated_savings": 150.0,
            "actions_taken": [
                "Switched 20% of requests to gpt-4o-mini",
                "Increased cache hit rate by 15%"
            ]
        }
    
    async def _optimize_speculation_threshold(self) -> Dict[str, Any]:
        """Optimize speculative execution threshold."""
        # Analyze speculation accuracy vs cost
        # Adjust confidence thresholds
        
        return {
            "strategy": "speculation_optimization", 
            "estimated_savings": 80.0,
            "actions_taken": [
                "Increased speculation threshold to 0.85",
                "Reduced false positive speculation by 25%"
            ]
        }
```

## Disaster Recovery

### 1. Backup and Recovery Strategy

#### Automated Backup System
Comprehensive backup strategy:

```python
# src/async_toolformer/operations/backup_recovery.py
"""Backup and disaster recovery management."""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
import structlog

logger = structlog.get_logger()

class BackupManager:
    """Manages automated backups and recovery procedures."""
    
    def __init__(self, storage_backend):
        self.storage_backend = storage_backend
        self.backup_schedule = {
            "configuration": timedelta(hours=6),
            "metrics": timedelta(hours=1),
            "logs": timedelta(hours=12),
            "user_data": timedelta(hours=24)
        }
    
    async def create_full_backup(self) -> Dict[str, Any]:
        """Create comprehensive system backup."""
        backup_id = f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        backup_manifest = {
            "backup_id": backup_id,
            "timestamp": datetime.utcnow(),
            "components": []
        }
        
        # Backup configuration
        config_backup = await self._backup_configuration()
        backup_manifest["components"].append(config_backup)
        
        # Backup metrics
        metrics_backup = await self._backup_metrics()
        backup_manifest["components"].append(metrics_backup)
        
        # Backup application state
        state_backup = await self._backup_application_state()
        backup_manifest["components"].append(state_backup)
        
        # Store backup manifest
        await self.storage_backend.store(
            f"{backup_id}/manifest.json",
            json.dumps(backup_manifest)
        )
        
        logger.info(
            "Full backup completed",
            backup_id=backup_id,
            components=len(backup_manifest["components"])
        )
        
        return backup_manifest
    
    async def restore_from_backup(self, backup_id: str) -> Dict[str, Any]:
        """Restore system from backup."""
        logger.info("Starting system restore", backup_id=backup_id)
        
        # Load backup manifest
        manifest_data = await self.storage_backend.retrieve(f"{backup_id}/manifest.json")
        manifest = json.loads(manifest_data)
        
        # Restore components
        restore_results = []
        for component in manifest["components"]:
            result = await self._restore_component(backup_id, component)
            restore_results.append(result)
        
        logger.info(
            "System restore completed",
            backup_id=backup_id,
            components_restored=len(restore_results)
        )
        
        return {
            "backup_id": backup_id,
            "restore_timestamp": datetime.utcnow(),
            "results": restore_results
        }
```

### 2. High Availability Setup

#### Multi-Region Deployment
Disaster recovery across regions:

```yaml
# k8s/high-availability/multi-region-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: async-toolformer
  labels:
    app: async-toolformer
    tier: production
spec:
  replicas: 6
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  selector:
    matchLabels:
      app: async-toolformer
  template:
    metadata:
      labels:
        app: async-toolformer
    spec:
      # Anti-affinity to spread across nodes/zones
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - async-toolformer
              topologyKey: kubernetes.io/hostname
      containers:
      - name: async-toolformer
        image: async-toolformer:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

This operational excellence framework provides production-grade operations appropriate for a maturing system requiring high reliability and comprehensive observability.