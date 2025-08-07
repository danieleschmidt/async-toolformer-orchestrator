"""Monitoring, metrics, and observability for sentiment analysis."""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import structlog

from .metrics import track_metric
from .sentiment_analyzer import SentimentPolarity, SentimentResult, BatchSentimentResult

logger = structlog.get_logger(__name__)


@dataclass
class SentimentMetrics:
    """Sentiment analysis metrics."""
    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    total_processing_time_ms: float = 0.0
    avg_processing_time_ms: float = 0.0
    min_processing_time_ms: float = float('inf')
    max_processing_time_ms: float = 0.0
    
    # Sentiment distribution
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    mixed_count: int = 0
    
    # Quality metrics
    avg_confidence: float = 0.0
    low_confidence_count: int = 0
    high_confidence_count: int = 0
    
    # Temporal metrics
    analyses_last_hour: int = 0
    analyses_last_day: int = 0
    peak_throughput: float = 0.0  # analyses per second
    
    def update_from_result(self, result: SentimentResult, processing_time_ms: float = None):
        """Update metrics from a sentiment result."""
        self.total_analyses += 1
        self.successful_analyses += 1
        
        # Processing time
        time_ms = processing_time_ms or result.processing_time_ms
        self.total_processing_time_ms += time_ms
        self.avg_processing_time_ms = self.total_processing_time_ms / self.total_analyses
        self.min_processing_time_ms = min(self.min_processing_time_ms, time_ms)
        self.max_processing_time_ms = max(self.max_processing_time_ms, time_ms)
        
        # Sentiment distribution
        if result.sentiment.polarity == SentimentPolarity.POSITIVE:
            self.positive_count += 1
        elif result.sentiment.polarity == SentimentPolarity.NEGATIVE:
            self.negative_count += 1
        elif result.sentiment.polarity == SentimentPolarity.NEUTRAL:
            self.neutral_count += 1
        else:
            self.mixed_count += 1
        
        # Confidence metrics
        confidence = result.sentiment.confidence
        total_confidence = (self.avg_confidence * (self.total_analyses - 1) + confidence)
        self.avg_confidence = total_confidence / self.total_analyses
        
        if confidence < 0.5:
            self.low_confidence_count += 1
        elif confidence > 0.8:
            self.high_confidence_count += 1
    
    def record_failure(self):
        """Record a failed analysis."""
        self.total_analyses += 1
        self.failed_analyses += 1
    
    def get_success_rate(self) -> float:
        """Get analysis success rate."""
        if self.total_analyses == 0:
            return 0.0
        return self.successful_analyses / self.total_analyses
    
    def get_sentiment_distribution(self) -> Dict[str, float]:
        """Get sentiment distribution percentages."""
        if self.successful_analyses == 0:
            return {"positive": 0.0, "negative": 0.0, "neutral": 0.0, "mixed": 0.0}
        
        return {
            "positive": self.positive_count / self.successful_analyses * 100,
            "negative": self.negative_count / self.successful_analyses * 100,
            "neutral": self.neutral_count / self.successful_analyses * 100,
            "mixed": self.mixed_count / self.successful_analyses * 100
        }


@dataclass
class PerformanceAlert:
    """Performance alert definition."""
    timestamp: datetime
    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    metric_value: float
    threshold: float
    suggested_action: Optional[str] = None


class SentimentMonitor:
    """Comprehensive monitoring for sentiment analysis operations."""
    
    def __init__(
        self,
        enable_detailed_logging: bool = True,
        alert_thresholds: Optional[Dict[str, float]] = None,
        max_history_size: int = 10000
    ):
        """Initialize sentiment monitor."""
        self.enable_detailed_logging = enable_detailed_logging
        self.max_history_size = max_history_size
        
        # Metrics
        self.metrics = SentimentMetrics()
        self.quantum_metrics = defaultdict(float)
        
        # Performance tracking
        self.recent_analyses = deque(maxlen=max_history_size)
        self.performance_history = deque(maxlen=1000)
        self.alerts = deque(maxlen=100)
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            "avg_processing_time_ms": 5000.0,  # 5 seconds
            "success_rate": 0.95,  # 95%
            "low_confidence_rate": 0.3,  # 30%
            "throughput_degradation": 0.5,  # 50% decrease
            "memory_usage_mb": 1000.0,  # 1GB
        }
        
        # State tracking
        self.start_time = datetime.utcnow()
        self.last_alert_check = datetime.utcnow()
        
        logger.info("Sentiment monitor initialized", 
                   thresholds=self.alert_thresholds,
                   max_history=max_history_size)
    
    async def record_analysis(
        self, 
        result: Union[SentimentResult, Exception],
        text: str,
        processing_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a sentiment analysis operation."""
        timestamp = datetime.utcnow()
        
        # Record in history
        analysis_record = {
            "timestamp": timestamp,
            "success": isinstance(result, SentimentResult),
            "processing_time_ms": processing_time_ms,
            "text_length": len(text),
            "metadata": metadata or {}
        }
        
        if isinstance(result, SentimentResult):
            analysis_record.update({
                "sentiment_score": result.sentiment.score,
                "sentiment_polarity": result.sentiment.polarity.value,
                "confidence": result.sentiment.confidence,
                "emotions_count": len(result.emotions),
                "keywords_count": len(result.keywords)
            })
            
            # Update metrics
            self.metrics.update_from_result(result, processing_time_ms)
            
            # Detailed logging
            if self.enable_detailed_logging:
                logger.info("Sentiment analysis completed",
                           polarity=result.sentiment.polarity.value,
                           score=result.sentiment.score,
                           confidence=result.sentiment.confidence,
                           processing_time_ms=processing_time_ms,
                           text_length=len(text))
        else:
            # Record failure
            self.metrics.record_failure()
            analysis_record["error"] = str(result)
            
            logger.error("Sentiment analysis failed",
                        error=str(result),
                        processing_time_ms=processing_time_ms,
                        text_length=len(text))
        
        self.recent_analyses.append(analysis_record)
        
        # Check for alerts
        await self._check_alerts()
        
        # Update performance metrics
        self._update_performance_metrics(processing_time_ms)
    
    async def record_batch_analysis(
        self, 
        batch_result: Union[BatchSentimentResult, Exception],
        batch_size: int,
        processing_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a batch sentiment analysis operation."""
        timestamp = datetime.utcnow()
        
        if isinstance(batch_result, BatchSentimentResult):
            # Record successful batch
            for result in batch_result.results:
                await self.record_analysis(
                    result, 
                    result.text,
                    result.processing_time_ms,
                    {**(metadata or {}), "batch": True}
                )
            
            # Calculate throughput
            throughput = batch_size / (processing_time_ms / 1000) if processing_time_ms > 0 else 0
            self.quantum_metrics["batch_throughput"] = throughput
            
            logger.info("Batch sentiment analysis completed",
                       batch_size=batch_size,
                       successful=len(batch_result.results),
                       total_processing_time_ms=processing_time_ms,
                       throughput_per_sec=throughput)
        else:
            # Record batch failure
            self.metrics.failed_analyses += batch_size
            
            logger.error("Batch sentiment analysis failed",
                        error=str(batch_result),
                        batch_size=batch_size,
                        processing_time_ms=processing_time_ms)
    
    async def record_quantum_metrics(self, quantum_data: Dict[str, Any]):
        """Record quantum-specific metrics."""
        for key, value in quantum_data.items():
            if isinstance(value, (int, float)):
                self.quantum_metrics[key] = value
        
        # Log quantum performance
        if self.enable_detailed_logging:
            logger.info("Quantum metrics updated", **quantum_data)
    
    def _update_performance_metrics(self, processing_time_ms: float):
        """Update performance tracking."""
        now = datetime.utcnow()
        
        # Add to performance history
        self.performance_history.append({
            "timestamp": now,
            "processing_time_ms": processing_time_ms
        })
        
        # Calculate recent throughput
        recent_window = now - timedelta(minutes=1)
        recent_count = sum(1 for record in self.recent_analyses 
                          if record["timestamp"] > recent_window)
        current_throughput = recent_count / 60.0  # per second
        
        # Update peak throughput
        if current_throughput > self.metrics.peak_throughput:
            self.metrics.peak_throughput = current_throughput
        
        # Update temporal metrics
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        self.metrics.analyses_last_hour = sum(1 for record in self.recent_analyses 
                                            if record["timestamp"] > hour_ago)
        self.metrics.analyses_last_day = sum(1 for record in self.recent_analyses 
                                           if record["timestamp"] > day_ago)
    
    async def _check_alerts(self):
        """Check for performance alerts."""
        now = datetime.utcnow()
        
        # Only check alerts every minute to avoid spam
        if now - self.last_alert_check < timedelta(minutes=1):
            return
        
        self.last_alert_check = now
        
        # Check average processing time
        if (self.metrics.avg_processing_time_ms > 
            self.alert_thresholds["avg_processing_time_ms"]):
            await self._create_alert(
                "high_processing_time",
                "high",
                f"Average processing time {self.metrics.avg_processing_time_ms:.2f}ms "
                f"exceeds threshold {self.alert_thresholds['avg_processing_time_ms']:.2f}ms",
                self.metrics.avg_processing_time_ms,
                self.alert_thresholds["avg_processing_time_ms"],
                "Consider optimizing analysis algorithms or increasing resources"
            )
        
        # Check success rate
        success_rate = self.metrics.get_success_rate()
        if success_rate < self.alert_thresholds["success_rate"]:
            await self._create_alert(
                "low_success_rate",
                "critical",
                f"Success rate {success_rate:.2%} below threshold "
                f"{self.alert_thresholds['success_rate']:.2%}",
                success_rate,
                self.alert_thresholds["success_rate"],
                "Investigate error patterns and input validation"
            )
        
        # Check low confidence rate
        if self.metrics.successful_analyses > 0:
            low_confidence_rate = (self.metrics.low_confidence_count / 
                                 self.metrics.successful_analyses)
            if low_confidence_rate > self.alert_thresholds["low_confidence_rate"]:
                await self._create_alert(
                    "high_low_confidence_rate",
                    "medium",
                    f"Low confidence rate {low_confidence_rate:.2%} exceeds threshold "
                    f"{self.alert_thresholds['low_confidence_rate']:.2%}",
                    low_confidence_rate,
                    self.alert_thresholds["low_confidence_rate"],
                    "Review input text quality or adjust confidence thresholds"
                )
    
    async def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metric_value: float,
        threshold: float,
        suggested_action: Optional[str] = None
    ):
        """Create a performance alert."""
        alert = PerformanceAlert(
            timestamp=datetime.utcnow(),
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_value=metric_value,
            threshold=threshold,
            suggested_action=suggested_action
        )
        
        self.alerts.append(alert)
        
        logger.warning("Performance alert created",
                      alert_type=alert_type,
                      severity=severity,
                      message=message,
                      metric_value=metric_value,
                      threshold=threshold)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        now = datetime.utcnow()
        uptime = now - self.start_time
        
        # Recent performance (last 10 minutes)
        recent_window = now - timedelta(minutes=10)
        recent_analyses = [r for r in self.recent_analyses if r["timestamp"] > recent_window]
        
        avg_recent_time = (sum(r["processing_time_ms"] for r in recent_analyses) / 
                          len(recent_analyses) if recent_analyses else 0)
        
        # Calculate trends
        sentiment_trends = self._calculate_sentiment_trends()
        performance_trends = self._calculate_performance_trends()
        
        return {
            "system_info": {
                "uptime_seconds": uptime.total_seconds(),
                "start_time": self.start_time.isoformat(),
                "current_time": now.isoformat()
            },
            "metrics": {
                "total_analyses": self.metrics.total_analyses,
                "success_rate": self.metrics.get_success_rate(),
                "avg_processing_time_ms": self.metrics.avg_processing_time_ms,
                "recent_avg_processing_time_ms": avg_recent_time,
                "peak_throughput": self.metrics.peak_throughput,
                "analyses_last_hour": self.metrics.analyses_last_hour,
                "analyses_last_day": self.metrics.analyses_last_day
            },
            "sentiment_distribution": self.metrics.get_sentiment_distribution(),
            "quality_metrics": {
                "avg_confidence": self.metrics.avg_confidence,
                "low_confidence_rate": (self.metrics.low_confidence_count / 
                                      max(self.metrics.successful_analyses, 1)),
                "high_confidence_rate": (self.metrics.high_confidence_count / 
                                       max(self.metrics.successful_analyses, 1))
            },
            "quantum_metrics": dict(self.quantum_metrics),
            "trends": {
                "sentiment": sentiment_trends,
                "performance": performance_trends
            },
            "recent_alerts": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message
                }
                for alert in list(self.alerts)[-5:]  # Last 5 alerts
            ],
            "health_status": self._get_health_status()
        }
    
    def _calculate_sentiment_trends(self) -> Dict[str, Any]:
        """Calculate sentiment trends over time."""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        recent_results = [
            r for r in self.recent_analyses 
            if r["timestamp"] > hour_ago and r["success"]
        ]
        
        if len(recent_results) < 2:
            return {"trend": "insufficient_data"}
        
        # Split into two halves
        mid_point = len(recent_results) // 2
        first_half = recent_results[:mid_point]
        second_half = recent_results[mid_point:]
        
        def avg_score(results):
            scores = [r.get("sentiment_score", 0) for r in results]
            return sum(scores) / len(scores) if scores else 0
        
        first_avg = avg_score(first_half)
        second_avg = avg_score(second_half)
        
        trend_direction = "stable"
        if second_avg > first_avg + 0.1:
            trend_direction = "improving"
        elif second_avg < first_avg - 0.1:
            trend_direction = "declining"
        
        return {
            "trend": trend_direction,
            "first_half_avg": first_avg,
            "second_half_avg": second_avg,
            "change": second_avg - first_avg,
            "sample_size": len(recent_results)
        }
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends."""
        if len(self.performance_history) < 10:
            return {"trend": "insufficient_data"}
        
        recent = list(self.performance_history)[-10:]
        times = [r["processing_time_ms"] for r in recent]
        
        # Simple linear trend
        n = len(times)
        x_sum = sum(range(n))
        y_sum = sum(times)
        xy_sum = sum(i * times[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        
        trend_direction = "stable"
        if slope > 100:  # Increasing by more than 100ms per sample
            trend_direction = "degrading"
        elif slope < -100:  # Decreasing by more than 100ms per sample
            trend_direction = "improving"
        
        return {
            "trend": trend_direction,
            "slope_ms_per_sample": slope,
            "avg_recent_time_ms": sum(times) / len(times),
            "sample_size": len(times)
        }
    
    def _get_health_status(self) -> str:
        """Get overall system health status."""
        issues = 0
        
        # Check recent alerts
        recent_alerts = [a for a in self.alerts 
                        if a.timestamp > datetime.utcnow() - timedelta(minutes=30)]
        critical_alerts = [a for a in recent_alerts if a.severity == "critical"]
        high_alerts = [a for a in recent_alerts if a.severity == "high"]
        
        if critical_alerts:
            issues += 3
        if high_alerts:
            issues += 2
        
        # Check success rate
        if self.metrics.get_success_rate() < 0.9:
            issues += 2
        
        # Check processing time
        if self.metrics.avg_processing_time_ms > 3000:
            issues += 1
        
        if issues == 0:
            return "healthy"
        elif issues <= 2:
            return "warning"
        else:
            return "critical"
    
    async def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format."""
        if format == "prometheus":
            return self._export_prometheus_metrics()
        elif format == "json":
            import json
            return json.dumps(self.get_dashboard_data(), indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Basic metrics
        lines.append(f"# HELP sentiment_total_analyses Total number of sentiment analyses")
        lines.append(f"# TYPE sentiment_total_analyses counter")
        lines.append(f"sentiment_total_analyses {self.metrics.total_analyses}")
        
        lines.append(f"# HELP sentiment_success_rate Success rate of sentiment analyses")
        lines.append(f"# TYPE sentiment_success_rate gauge")
        lines.append(f"sentiment_success_rate {self.metrics.get_success_rate()}")
        
        lines.append(f"# HELP sentiment_avg_processing_time_ms Average processing time in milliseconds")
        lines.append(f"# TYPE sentiment_avg_processing_time_ms gauge")
        lines.append(f"sentiment_avg_processing_time_ms {self.metrics.avg_processing_time_ms}")
        
        # Sentiment distribution
        distribution = self.metrics.get_sentiment_distribution()
        for sentiment, percentage in distribution.items():
            lines.append(f"# HELP sentiment_distribution_{sentiment} Percentage of {sentiment} sentiments")
            lines.append(f"# TYPE sentiment_distribution_{sentiment} gauge")
            lines.append(f"sentiment_distribution_{sentiment} {percentage}")
        
        # Quantum metrics
        for metric, value in self.quantum_metrics.items():
            lines.append(f"# HELP quantum_{metric} Quantum metric: {metric}")
            lines.append(f"# TYPE quantum_{metric} gauge")
            lines.append(f"quantum_{metric} {value}")
        
        return "\n".join(lines)


# Global monitor instance
_global_monitor = None


def get_sentiment_monitor() -> SentimentMonitor:
    """Get global sentiment monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = SentimentMonitor()
    return _global_monitor


def set_sentiment_monitor(monitor: SentimentMonitor):
    """Set global sentiment monitor instance."""
    global _global_monitor
    _global_monitor = monitor