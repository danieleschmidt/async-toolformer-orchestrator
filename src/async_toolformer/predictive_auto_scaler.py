"""
Predictive Auto Scaler - Generation 3 Implementation.

Advanced auto-scaling system with ML-based prediction, quantum optimization,
and multi-dimensional resource management for the Async Toolformer Orchestrator.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ScalingDirection(Enum):
    """Scaling direction for auto-scaling decisions."""
    UP = "up"
    DOWN = "down"
    STEADY = "steady"


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    COMPUTE = "compute"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    CONNECTIONS = "connections"


class PredictionModel(Enum):
    """Prediction models for auto-scaling."""
    LINEAR_REGRESSION = "linear_regression"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    SEASONAL_ARIMA = "seasonal_arima"
    NEURAL_NETWORK = "neural_network"
    QUANTUM_INSPIRED = "quantum_inspired"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    cpu_usage: float
    memory_usage: float
    network_io: float
    disk_io: float
    connection_count: int
    request_rate: float
    error_rate: float
    response_time: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class ScalingEvent:
    """Represents a scaling event."""
    timestamp: float
    direction: ScalingDirection
    resource_type: ResourceType
    old_capacity: float
    new_capacity: float
    trigger_metric: str
    trigger_value: float
    prediction_confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of a resource demand prediction."""
    predicted_demand: float
    confidence: float
    prediction_horizon: float
    model_used: PredictionModel
    contributing_factors: dict[str, float]
    uncertainty_bounds: tuple[float, float]
    timestamp: float = field(default_factory=time.time)


class TimeSeriesPredictor:
    """Advanced time series prediction for resource demands."""

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._metric_history: dict[str, deque] = {}
        self._seasonal_patterns: dict[str, list[float]] = {}
        self._trend_coefficients: dict[str, float] = {}

    def add_metrics(self, metrics: ResourceMetrics) -> None:
        """Add new metrics to the prediction history."""

        metric_dict = {
            "cpu_usage": metrics.cpu_usage,
            "memory_usage": metrics.memory_usage,
            "network_io": metrics.network_io,
            "disk_io": metrics.disk_io,
            "connection_count": metrics.connection_count,
            "request_rate": metrics.request_rate,
            "error_rate": metrics.error_rate,
            "response_time": metrics.response_time,
            "timestamp": metrics.timestamp,
        }

        for metric_name, value in metric_dict.items():
            if metric_name not in self._metric_history:
                self._metric_history[metric_name] = deque(maxlen=self.history_size)

            self._metric_history[metric_name].append({
                "value": value,
                "timestamp": metrics.timestamp,
            })

        # Update seasonal patterns and trends
        self._update_patterns()

    def predict(
        self,
        metric_name: str,
        horizon_minutes: float = 15.0,
        model: PredictionModel = PredictionModel.QUANTUM_INSPIRED,
    ) -> PredictionResult:
        """Predict future value for a metric."""

        if metric_name not in self._metric_history:
            return self._default_prediction(metric_name, horizon_minutes, model)

        history = list(self._metric_history[metric_name])

        if len(history) < 10:  # Need minimum history
            return self._default_prediction(metric_name, horizon_minutes, model)

        if model == PredictionModel.LINEAR_REGRESSION:
            return self._linear_regression_predict(metric_name, history, horizon_minutes)
        elif model == PredictionModel.EXPONENTIAL_SMOOTHING:
            return self._exponential_smoothing_predict(metric_name, history, horizon_minutes)
        elif model == PredictionModel.SEASONAL_ARIMA:
            return self._seasonal_arima_predict(metric_name, history, horizon_minutes)
        elif model == PredictionModel.NEURAL_NETWORK:
            return self._neural_network_predict(metric_name, history, horizon_minutes)
        elif model == PredictionModel.QUANTUM_INSPIRED:
            return self._quantum_inspired_predict(metric_name, history, horizon_minutes)
        else:
            return self._default_prediction(metric_name, horizon_minutes, model)

    def _linear_regression_predict(
        self, metric_name: str, history: list[dict], horizon_minutes: float
    ) -> PredictionResult:
        """Simple linear regression prediction."""

        if len(history) < 2:
            return self._default_prediction(metric_name, horizon_minutes, PredictionModel.LINEAR_REGRESSION)

        # Calculate trend
        x_values = list(range(len(history)))
        y_values = [point["value"] for point in history]

        n = len(history)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values, strict=False))
        sum_x2 = sum(x * x for x in x_values)

        # Linear regression coefficients
        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
        else:
            slope = 0
            intercept = sum_y / n

        # Predict future value
        future_x = len(history) + (horizon_minutes / 5)  # Assuming 5-minute intervals
        predicted_value = slope * future_x + intercept

        # Calculate confidence based on R-squared
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in y_values)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values, strict=False))

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = max(0.1, min(0.9, r_squared))

        # Calculate uncertainty bounds
        std_error = math.sqrt(ss_res / max(n - 2, 1))
        margin = 1.96 * std_error  # 95% confidence interval

        return PredictionResult(
            predicted_demand=max(0.0, predicted_value),
            confidence=confidence,
            prediction_horizon=horizon_minutes,
            model_used=PredictionModel.LINEAR_REGRESSION,
            contributing_factors={"trend": slope, "base": intercept},
            uncertainty_bounds=(
                max(0.0, predicted_value - margin),
                predicted_value + margin
            ),
        )

    def _exponential_smoothing_predict(
        self, metric_name: str, history: list[dict], horizon_minutes: float
    ) -> PredictionResult:
        """Exponential smoothing prediction with trend and seasonality."""

        values = [point["value"] for point in history]

        # Simple exponential smoothing parameters
        alpha = 0.3  # Level smoothing
        beta = 0.1   # Trend smoothing
        gamma = 0.1  # Seasonality smoothing

        # Initialize
        level = values[0]
        trend = 0.0

        # Apply exponential smoothing
        for i in range(1, len(values)):
            new_level = alpha * values[i] + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend

            level = new_level
            trend = new_trend

        # Predict future value
        periods_ahead = max(1, int(horizon_minutes / 5))
        predicted_value = level + trend * periods_ahead

        # Add seasonal component if available
        seasonal_pattern = self._seasonal_patterns.get(metric_name, [])
        if seasonal_pattern:
            current_hour = int(time.time() % 86400 // 3600)
            future_hour = (current_hour + int(horizon_minutes / 60)) % 24
            seasonal_factor = seasonal_pattern[future_hour] if future_hour < len(seasonal_pattern) else 1.0
            predicted_value *= seasonal_factor

        # Calculate confidence based on recent prediction accuracy
        confidence = self._calculate_exponential_smoothing_confidence(values, alpha, beta)

        return PredictionResult(
            predicted_demand=max(0.0, predicted_value),
            confidence=confidence,
            prediction_horizon=horizon_minutes,
            model_used=PredictionModel.EXPONENTIAL_SMOOTHING,
            contributing_factors={
                "level": level,
                "trend": trend,
                "seasonal": seasonal_pattern[future_hour] if seasonal_pattern and future_hour < len(seasonal_pattern) else 1.0,
            },
            uncertainty_bounds=(
                max(0.0, predicted_value * 0.8),
                predicted_value * 1.2
            ),
        )

    def _quantum_inspired_predict(
        self, metric_name: str, history: list[dict], horizon_minutes: float
    ) -> PredictionResult:
        """Quantum-inspired prediction using superposition and entanglement concepts."""

        values = [point["value"] for point in history]
        timestamps = [point["timestamp"] for point in history]

        if len(values) < 5:
            return self._default_prediction(metric_name, horizon_minutes, PredictionModel.QUANTUM_INSPIRED)

        # Quantum superposition: Multiple prediction states
        prediction_states = []

        # State 1: Trend-based prediction
        recent_trend = self._calculate_trend(values[-10:])
        trend_prediction = values[-1] + recent_trend * (horizon_minutes / 5)
        prediction_states.append(("trend", trend_prediction, 0.3))

        # State 2: Seasonal pattern prediction
        seasonal_prediction = self._apply_seasonal_pattern(metric_name, values[-1], horizon_minutes)
        prediction_states.append(("seasonal", seasonal_prediction, 0.25))

        # State 3: Mean reversion prediction
        mean_value = sum(values[-20:]) / min(len(values), 20)
        reversion_factor = 0.1 * (mean_value - values[-1])
        reversion_prediction = values[-1] + reversion_factor
        prediction_states.append(("reversion", reversion_prediction, 0.2))

        # State 4: Momentum prediction
        momentum = self._calculate_momentum(values[-5:])
        momentum_prediction = values[-1] * (1 + momentum * horizon_minutes / 60)
        prediction_states.append(("momentum", momentum_prediction, 0.15))

        # State 5: Volatility-adjusted prediction
        volatility = self._calculate_volatility(values[-10:])
        volatility_adjustment = volatility * 0.1 * (horizon_minutes / 15)
        volatility_prediction = values[-1] + volatility_adjustment
        prediction_states.append(("volatility", volatility_prediction, 0.1))

        # Quantum interference: Combine states with phase relationships
        final_prediction = 0.0
        total_weight = 0.0

        for i, (name, prediction, weight) in enumerate(prediction_states):
            # Apply quantum phase based on historical accuracy
            phase = self._calculate_quantum_phase(metric_name, name)
            quantum_weight = weight * math.cos(phase)  # Constructive/destructive interference

            final_prediction += quantum_weight * prediction
            total_weight += abs(quantum_weight)

        if total_weight > 0:
            final_prediction /= total_weight
        else:
            final_prediction = values[-1]

        # Quantum entanglement: Consider correlated metrics
        entanglement_adjustment = self._apply_metric_entanglement(metric_name, final_prediction)
        final_prediction += entanglement_adjustment

        # Calculate confidence using quantum uncertainty principle
        prediction_variance = sum((pred - final_prediction) ** 2 for _, pred, _ in prediction_states) / len(prediction_states)
        quantum_uncertainty = math.sqrt(prediction_variance) / final_prediction if final_prediction > 0 else 0.1
        confidence = max(0.1, min(0.95, 1.0 - quantum_uncertainty))

        # Contributing factors
        factors = {name: weight for name, _, weight in prediction_states}
        factors["quantum_interference"] = total_weight / len(prediction_states)
        factors["entanglement_effect"] = abs(entanglement_adjustment) / max(final_prediction, 0.1)

        return PredictionResult(
            predicted_demand=max(0.0, final_prediction),
            confidence=confidence,
            prediction_horizon=horizon_minutes,
            model_used=PredictionModel.QUANTUM_INSPIRED,
            contributing_factors=factors,
            uncertainty_bounds=(
                max(0.0, final_prediction - quantum_uncertainty * final_prediction),
                final_prediction + quantum_uncertainty * final_prediction
            ),
        )

    def _calculate_trend(self, values: list[float]) -> float:
        """Calculate recent trend in values."""
        if len(values) < 2:
            return 0.0

        # Simple linear trend
        n = len(values)
        x_sum = n * (n - 1) / 2
        y_sum = sum(values)
        xy_sum = sum(i * v for i, v in enumerate(values))
        x2_sum = n * (n - 1) * (2 * n - 1) / 6

        denominator = n * x2_sum - x_sum * x_sum
        if abs(denominator) < 1e-10:
            return 0.0

        trend = (n * xy_sum - x_sum * y_sum) / denominator
        return trend

    def _calculate_momentum(self, values: list[float]) -> float:
        """Calculate momentum (rate of change acceleration)."""
        if len(values) < 3:
            return 0.0

        # Calculate recent rate of change
        recent_change = (values[-1] - values[-2]) / max(values[-2], 0.1)
        previous_change = (values[-2] - values[-3]) / max(values[-3], 0.1)

        momentum = recent_change - previous_change
        return momentum

    def _calculate_volatility(self, values: list[float]) -> float:
        """Calculate volatility (standard deviation of returns)."""
        if len(values) < 2:
            return 0.0

        returns = []
        for i in range(1, len(values)):
            if values[i-1] > 0:
                returns.append((values[i] - values[i-1]) / values[i-1])

        if not returns:
            return 0.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)

        return math.sqrt(variance)

    def _apply_seasonal_pattern(self, metric_name: str, current_value: float, horizon_minutes: float) -> float:
        """Apply seasonal pattern to prediction."""

        seasonal_pattern = self._seasonal_patterns.get(metric_name, [])
        if not seasonal_pattern:
            return current_value

        current_hour = int(time.time() % 86400 // 3600)
        future_hour = (current_hour + int(horizon_minutes / 60)) % 24

        if future_hour < len(seasonal_pattern):
            seasonal_factor = seasonal_pattern[future_hour]
            return current_value * seasonal_factor

        return current_value

    def _calculate_quantum_phase(self, metric_name: str, state_name: str) -> float:
        """Calculate quantum phase for interference effects."""

        # Use historical accuracy to determine phase
        # In a real implementation, this would be based on backtest results
        phase_map = {
            "trend": 0.0,      # Base phase
            "seasonal": math.pi / 4,   # 45 degrees
            "reversion": math.pi / 2,  # 90 degrees
            "momentum": 3 * math.pi / 4,  # 135 degrees
            "volatility": math.pi,     # 180 degrees
        }

        return phase_map.get(state_name, 0.0)

    def _apply_metric_entanglement(self, metric_name: str, prediction: float) -> float:
        """Apply quantum entanglement effects from correlated metrics."""

        # Define metric correlations (entanglements)
        entanglements = {
            "cpu_usage": {"memory_usage": 0.6, "response_time": 0.7},
            "memory_usage": {"cpu_usage": 0.6, "request_rate": 0.5},
            "request_rate": {"cpu_usage": 0.8, "memory_usage": 0.5, "connection_count": 0.9},
            "response_time": {"cpu_usage": 0.7, "error_rate": 0.6},
            "error_rate": {"response_time": 0.6, "cpu_usage": 0.4},
        }

        if metric_name not in entanglements:
            return 0.0

        entanglement_effect = 0.0

        for correlated_metric, strength in entanglements[metric_name].items():
            if correlated_metric in self._metric_history:
                recent_values = list(self._metric_history[correlated_metric])[-5:]
                if recent_values:
                    correlated_trend = self._calculate_trend([v["value"] for v in recent_values])
                    entanglement_effect += strength * correlated_trend * 0.1

        return entanglement_effect

    def _update_patterns(self) -> None:
        """Update seasonal patterns and trends from historical data."""

        for metric_name, history in self._metric_history.items():
            if len(history) < 24:  # Need at least 24 hours of data
                continue

            # Update seasonal pattern (24-hour)
            hourly_values = [[] for _ in range(24)]

            for point in history:
                timestamp = point["timestamp"]
                hour = int(timestamp % 86400 // 3600)
                hourly_values[hour].append(point["value"])

            # Calculate average for each hour
            seasonal_pattern = []
            overall_mean = sum(point["value"] for point in history) / len(history)

            for hour_values in hourly_values:
                if hour_values:
                    hour_mean = sum(hour_values) / len(hour_values)
                    seasonal_factor = hour_mean / overall_mean if overall_mean > 0 else 1.0
                else:
                    seasonal_factor = 1.0

                seasonal_pattern.append(seasonal_factor)

            self._seasonal_patterns[metric_name] = seasonal_pattern

            # Update trend coefficient
            recent_values = [point["value"] for point in list(history)[-20:]]
            self._trend_coefficients[metric_name] = self._calculate_trend(recent_values)

    def _default_prediction(
        self, metric_name: str, horizon_minutes: float, model: PredictionModel
    ) -> PredictionResult:
        """Default prediction when insufficient data is available."""

        if metric_name in self._metric_history and self._metric_history[metric_name]:
            last_value = self._metric_history[metric_name][-1]["value"]
        else:
            last_value = 0.5  # Default assumption

        return PredictionResult(
            predicted_demand=last_value,
            confidence=0.3,  # Low confidence due to insufficient data
            prediction_horizon=horizon_minutes,
            model_used=model,
            contributing_factors={"insufficient_data": 1.0},
            uncertainty_bounds=(last_value * 0.5, last_value * 1.5),
        )

    def _calculate_exponential_smoothing_confidence(
        self, values: list[float], alpha: float, beta: float
    ) -> float:
        """Calculate confidence for exponential smoothing predictions."""

        if len(values) < 5:
            return 0.3

        # Simulate prediction accuracy on historical data
        errors = []
        level = values[0]
        trend = 0.0

        for i in range(1, len(values) - 1):
            # Predict next value
            predicted = level + trend
            actual = values[i + 1]

            error = abs(predicted - actual) / max(actual, 0.1)
            errors.append(error)

            # Update smoothing
            new_level = alpha * values[i] + (1 - alpha) * (level + trend)
            new_trend = beta * (new_level - level) + (1 - beta) * trend
            level = new_level
            trend = new_trend

        if errors:
            mean_error = sum(errors) / len(errors)
            confidence = max(0.1, min(0.9, 1.0 - mean_error))
        else:
            confidence = 0.5

        return confidence


class AutoScalingEngine:
    """Advanced auto-scaling engine with predictive capabilities."""

    def __init__(
        self,
        min_capacity: float = 1.0,
        max_capacity: float = 100.0,
        target_utilization: float = 0.7,
        scale_up_threshold: float = 0.8,
        scale_down_threshold: float = 0.3,
        cooldown_period: float = 300.0,  # 5 minutes
    ):
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period

        self.current_capacity = min_capacity
        self.last_scaling_time = 0.0

        self.predictor = TimeSeriesPredictor()
        self.scaling_history: list[ScalingEvent] = []
        self.resource_metrics: deque = deque(maxlen=1000)

    async def add_metrics(self, metrics: ResourceMetrics) -> None:
        """Add new resource metrics for scaling decisions."""

        self.resource_metrics.append(metrics)
        self.predictor.add_metrics(metrics)

        logger.debug(
            "Metrics added",
            cpu=metrics.cpu_usage,
            memory=metrics.memory_usage,
            connections=metrics.connection_count,
        )

    async def evaluate_scaling_decision(
        self,
        prediction_horizon: float = 15.0,
    ) -> ScalingEvent | None:
        """Evaluate whether scaling is needed based on current metrics and predictions."""

        current_time = time.time()

        # Check cooldown period
        if current_time - self.last_scaling_time < self.cooldown_period:
            logger.debug("Scaling decision skipped due to cooldown period")
            return None

        if not self.resource_metrics:
            logger.debug("No metrics available for scaling decision")
            return None

        current_metrics = self.resource_metrics[-1]

        # Get predictions for key metrics
        predictions = {}
        for metric_name in ["cpu_usage", "memory_usage", "request_rate", "connection_count"]:
            prediction = self.predictor.predict(
                metric_name,
                prediction_horizon,
                PredictionModel.QUANTUM_INSPIRED
            )
            predictions[metric_name] = prediction

        # Calculate overall utilization and predicted utilization
        current_utilization = self._calculate_utilization(current_metrics)
        predicted_utilization = self._calculate_predicted_utilization(predictions)

        logger.debug(
            "Utilization analysis",
            current=current_utilization,
            predicted=predicted_utilization,
            threshold_up=self.scale_up_threshold,
            threshold_down=self.scale_down_threshold,
        )

        # Make scaling decision
        scaling_event = None

        if predicted_utilization > self.scale_up_threshold and current_utilization > 0.6:
            # Scale up
            scaling_event = await self._create_scale_up_event(
                current_metrics, predictions, predicted_utilization
            )
        elif predicted_utilization < self.scale_down_threshold and current_utilization < 0.5:
            # Scale down
            scaling_event = await self._create_scale_down_event(
                current_metrics, predictions, predicted_utilization
            )

        if scaling_event:
            await self._execute_scaling_event(scaling_event)

        return scaling_event

    def _calculate_utilization(self, metrics: ResourceMetrics) -> float:
        """Calculate overall resource utilization."""

        # Weighted combination of different resource metrics
        cpu_weight = 0.4
        memory_weight = 0.3
        connection_weight = 0.2
        response_time_weight = 0.1

        # Normalize connection count (assume max 1000 connections per unit)
        connection_utilization = min(metrics.connection_count / 1000.0, 1.0)

        # Normalize response time (assume 2 seconds is high)
        response_utilization = min(metrics.response_time / 2.0, 1.0)

        overall_utilization = (
            cpu_weight * metrics.cpu_usage +
            memory_weight * metrics.memory_usage +
            connection_weight * connection_utilization +
            response_time_weight * response_utilization
        )

        return min(overall_utilization, 1.0)

    def _calculate_predicted_utilization(self, predictions: dict[str, PredictionResult]) -> float:
        """Calculate predicted utilization from predictions."""

        predicted_cpu = predictions.get("cpu_usage", PredictionResult(0.5, 0.5, 15, PredictionModel.LINEAR_REGRESSION, {}, (0, 1))).predicted_demand
        predicted_memory = predictions.get("memory_usage", PredictionResult(0.5, 0.5, 15, PredictionModel.LINEAR_REGRESSION, {}, (0, 1))).predicted_demand
        predicted_connections = predictions.get("connection_count", PredictionResult(500, 0.5, 15, PredictionModel.LINEAR_REGRESSION, {}, (0, 1000))).predicted_demand

        # Normalize and combine
        connection_utilization = min(predicted_connections / 1000.0, 1.0)

        predicted_utilization = (
            0.4 * min(predicted_cpu, 1.0) +
            0.3 * min(predicted_memory, 1.0) +
            0.3 * connection_utilization
        )

        return min(predicted_utilization, 1.0)

    async def _create_scale_up_event(
        self,
        current_metrics: ResourceMetrics,
        predictions: dict[str, PredictionResult],
        predicted_utilization: float,
    ) -> ScalingEvent:
        """Create a scale-up event."""

        # Calculate optimal new capacity
        utilization_ratio = predicted_utilization / self.target_utilization
        optimal_capacity = self.current_capacity * utilization_ratio

        # Apply safety margins and constraints
        new_capacity = min(
            max(optimal_capacity, self.current_capacity * 1.2),  # At least 20% increase
            self.max_capacity
        )

        # Determine primary trigger
        trigger_metric = "cpu_usage"
        trigger_value = current_metrics.cpu_usage

        for metric_name in ["cpu_usage", "memory_usage", "connection_count"]:
            if metric_name in predictions:
                pred = predictions[metric_name]
                normalized_pred = pred.predicted_demand / 1000.0 if metric_name == "connection_count" else pred.predicted_demand
                if normalized_pred > trigger_value:
                    trigger_metric = metric_name
                    trigger_value = normalized_pred

        # Calculate confidence based on prediction confidence
        avg_confidence = sum(pred.confidence for pred in predictions.values()) / len(predictions)

        return ScalingEvent(
            timestamp=time.time(),
            direction=ScalingDirection.UP,
            resource_type=ResourceType.COMPUTE,
            old_capacity=self.current_capacity,
            new_capacity=new_capacity,
            trigger_metric=trigger_metric,
            trigger_value=trigger_value,
            prediction_confidence=avg_confidence,
            metadata={
                "predicted_utilization": predicted_utilization,
                "target_utilization": self.target_utilization,
                "predictions": {
                    name: {
                        "value": pred.predicted_demand,
                        "confidence": pred.confidence,
                        "model": pred.model_used.value,
                    }
                    for name, pred in predictions.items()
                },
            },
        )

    async def _create_scale_down_event(
        self,
        current_metrics: ResourceMetrics,
        predictions: dict[str, PredictionResult],
        predicted_utilization: float,
    ) -> ScalingEvent:
        """Create a scale-down event."""

        # Calculate optimal new capacity
        utilization_ratio = predicted_utilization / self.target_utilization
        optimal_capacity = self.current_capacity * utilization_ratio

        # Apply safety margins and constraints
        new_capacity = max(
            min(optimal_capacity, self.current_capacity * 0.8),  # At most 20% decrease
            self.min_capacity
        )

        # More conservative scaling down
        if self.current_capacity - new_capacity < self.current_capacity * 0.1:
            new_capacity = self.current_capacity  # Skip small scale-downs

        trigger_metric = "overall_utilization"
        trigger_value = predicted_utilization

        avg_confidence = sum(pred.confidence for pred in predictions.values()) / len(predictions)

        return ScalingEvent(
            timestamp=time.time(),
            direction=ScalingDirection.DOWN,
            resource_type=ResourceType.COMPUTE,
            old_capacity=self.current_capacity,
            new_capacity=new_capacity,
            trigger_metric=trigger_metric,
            trigger_value=trigger_value,
            prediction_confidence=avg_confidence,
            metadata={
                "predicted_utilization": predicted_utilization,
                "target_utilization": self.target_utilization,
                "scale_down_safety_check": True,
            },
        )

    async def _execute_scaling_event(self, event: ScalingEvent) -> None:
        """Execute a scaling event."""

        if event.old_capacity == event.new_capacity:
            logger.debug("Skipping scaling event - no capacity change needed")
            return

        logger.info(
            "Executing scaling event",
            direction=event.direction.value,
            old_capacity=event.old_capacity,
            new_capacity=event.new_capacity,
            trigger=event.trigger_metric,
            confidence=event.prediction_confidence,
        )

        # Update current capacity
        self.current_capacity = event.new_capacity
        self.last_scaling_time = event.timestamp

        # Record the event
        self.scaling_history.append(event)

        # Keep only recent history
        if len(self.scaling_history) > 100:
            self.scaling_history = self.scaling_history[-100:]

    def get_scaling_statistics(self) -> dict[str, Any]:
        """Get comprehensive scaling statistics."""

        if not self.scaling_history:
            return {
                "total_events": 0,
                "current_capacity": self.current_capacity,
                "avg_confidence": 0.0,
            }

        scale_up_events = [e for e in self.scaling_history if e.direction == ScalingDirection.UP]
        scale_down_events = [e for e in self.scaling_history if e.direction == ScalingDirection.DOWN]

        avg_confidence = sum(e.prediction_confidence for e in self.scaling_history) / len(self.scaling_history)

        return {
            "total_events": len(self.scaling_history),
            "scale_up_events": len(scale_up_events),
            "scale_down_events": len(scale_down_events),
            "current_capacity": self.current_capacity,
            "min_capacity": self.min_capacity,
            "max_capacity": self.max_capacity,
            "avg_prediction_confidence": avg_confidence,
            "recent_events": [
                {
                    "timestamp": e.timestamp,
                    "direction": e.direction.value,
                    "old_capacity": e.old_capacity,
                    "new_capacity": e.new_capacity,
                    "trigger": e.trigger_metric,
                    "confidence": e.prediction_confidence,
                }
                for e in self.scaling_history[-10:]  # Last 10 events
            ],
            "capacity_utilization_target": self.target_utilization,
            "scale_up_threshold": self.scale_up_threshold,
            "scale_down_threshold": self.scale_down_threshold,
        }

    async def simulate_scaling_scenarios(
        self,
        scenarios: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Simulate different scaling scenarios for planning."""

        simulation_results = {}

        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get("name", f"scenario_{i}")

            # Create temporary metrics based on scenario
            simulated_metrics = ResourceMetrics(
                cpu_usage=scenario.get("cpu_usage", 0.5),
                memory_usage=scenario.get("memory_usage", 0.5),
                network_io=scenario.get("network_io", 0.5),
                disk_io=scenario.get("disk_io", 0.5),
                connection_count=scenario.get("connection_count", 500),
                request_rate=scenario.get("request_rate", 100),
                error_rate=scenario.get("error_rate", 0.01),
                response_time=scenario.get("response_time", 0.2),
            )

            # Simulate scaling decision
            original_capacity = self.current_capacity
            original_metrics = list(self.resource_metrics)

            # Temporarily add simulated metrics
            await self.add_metrics(simulated_metrics)

            # Get scaling decision
            scaling_event = await self.evaluate_scaling_decision(
                prediction_horizon=scenario.get("prediction_horizon", 15.0)
            )

            # Restore original state
            self.current_capacity = original_capacity
            self.resource_metrics = deque(original_metrics, maxlen=1000)

            # Record simulation results
            if scaling_event:
                simulation_results[scenario_name] = {
                    "scaling_needed": True,
                    "direction": scaling_event.direction.value,
                    "old_capacity": scaling_event.old_capacity,
                    "new_capacity": scaling_event.new_capacity,
                    "confidence": scaling_event.prediction_confidence,
                    "trigger_metric": scaling_event.trigger_metric,
                    "trigger_value": scaling_event.trigger_value,
                }
            else:
                simulation_results[scenario_name] = {
                    "scaling_needed": False,
                    "current_capacity": self.current_capacity,
                    "utilization": self._calculate_utilization(simulated_metrics),
                }

        return simulation_results


def create_predictive_auto_scaler(
    min_capacity: float = 1.0,
    max_capacity: float = 100.0,
    target_utilization: float = 0.7,
) -> AutoScalingEngine:
    """Create a predictive auto-scaler with default configuration."""

    auto_scaler = AutoScalingEngine(
        min_capacity=min_capacity,
        max_capacity=max_capacity,
        target_utilization=target_utilization,
    )

    logger.info(
        "Predictive auto-scaler created",
        min_capacity=min_capacity,
        max_capacity=max_capacity,
        target_utilization=target_utilization,
    )

    return auto_scaler
