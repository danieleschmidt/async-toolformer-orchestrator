"""
Generation 4 Enhancement: ML-Driven Predictive Optimization

Advanced machine learning capabilities for intelligent orchestration with
predictive task planning, adaptive resource allocation, and pattern recognition.
"""

import statistics
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

# Optional ML dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import scikit_learn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

ML_AVAILABLE = NUMPY_AVAILABLE and SKLEARN_AVAILABLE


@dataclass
class MLMetrics:
    """Machine learning performance metrics."""

    prediction_accuracy: float = 0.0
    model_confidence: float = 0.0
    learning_rate: float = 0.001
    training_samples: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    feature_importance: dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of ML prediction."""

    predicted_value: float
    confidence: float
    features_used: list[str]
    model_type: str
    timestamp: datetime = field(default_factory=datetime.now)


class SimpleMLModel:
    """Lightweight ML model for orchestration optimization."""

    def __init__(self, model_type: str = "linear_regression"):
        self.model_type = model_type
        self.weights = {}
        self.bias = 0.0
        self.learning_rate = 0.01
        self.training_data = deque(maxlen=1000)
        self.feature_stats = defaultdict(lambda: {"mean": 0.0, "std": 1.0})

    def normalize_features(self, features: dict[str, float]) -> dict[str, float]:
        """Normalize feature values using running statistics."""
        normalized = {}
        for key, value in features.items():
            stats = self.feature_stats[key]
            normalized[key] = (value - stats["mean"]) / max(stats["std"], 0.001)
        return normalized

    def update_feature_stats(self, features: dict[str, float]) -> None:
        """Update running feature statistics."""
        for key, value in features.items():
            if key not in self.feature_stats:
                self.feature_stats[key] = {"values": deque(maxlen=100), "mean": 0.0, "std": 1.0}

            self.feature_stats[key]["values"].append(value)
            values = list(self.feature_stats[key]["values"])

            if len(values) > 1:
                self.feature_stats[key]["mean"] = statistics.mean(values)
                self.feature_stats[key]["std"] = max(statistics.stdev(values), 0.001)

    def predict(self, features: dict[str, float]) -> PredictionResult:
        """Make prediction using simple linear model."""
        normalized_features = self.normalize_features(features)

        # Simple linear prediction
        prediction = self.bias
        for feature, value in normalized_features.items():
            weight = self.weights.get(feature, 0.0)
            prediction += weight * value

        # Calculate confidence based on feature coverage
        covered_features = set(features.keys()) & set(self.weights.keys())
        confidence = len(covered_features) / max(len(self.weights), 1) if self.weights else 0.0

        return PredictionResult(
            predicted_value=max(0.0, prediction),  # Ensure non-negative
            confidence=confidence,
            features_used=list(covered_features),
            model_type=self.model_type
        )

    def train(self, features: dict[str, float], target: float) -> None:
        """Train model with new data point using gradient descent."""
        self.update_feature_stats(features)
        normalized_features = self.normalize_features(features)

        # Store training sample
        self.training_data.append((normalized_features.copy(), target))

        # Simple gradient descent update
        current_prediction = self.bias
        for feature, value in normalized_features.items():
            current_prediction += self.weights.get(feature, 0.0) * value

        error = target - current_prediction

        # Update weights and bias
        self.bias += self.learning_rate * error
        for feature, value in normalized_features.items():
            if feature not in self.weights:
                self.weights[feature] = 0.0
            self.weights[feature] += self.learning_rate * error * value


class MLPredictiveOptimizer:
    """
    Machine Learning-driven predictive optimizer for orchestration.
    
    Uses historical execution patterns to predict optimal configurations,
    resource needs, and potential bottlenecks before they occur.
    """

    def __init__(self, enable_online_learning: bool = True):
        self.enable_online_learning = enable_online_learning
        self.models = {
            "execution_time": SimpleMLModel("execution_time_predictor"),
            "resource_usage": SimpleMLModel("resource_predictor"),
            "failure_probability": SimpleMLModel("failure_predictor"),
            "optimal_parallelism": SimpleMLModel("parallelism_optimizer")
        }
        self.metrics = MLMetrics()
        self.execution_history = deque(maxlen=10000)
        self.prediction_cache = {}
        self.cache_ttl = timedelta(minutes=5)

    def extract_features(self, context: dict[str, Any]) -> dict[str, float]:
        """Extract numerical features from execution context."""
        features = {}

        # Basic context features
        features["tool_count"] = float(context.get("tool_count", 1))
        features["parallel_limit"] = float(context.get("parallel_limit", 10))
        features["timeout_ms"] = float(context.get("timeout_ms", 10000))
        features["retry_count"] = float(context.get("retry_count", 0))

        # Historical performance features
        if self.execution_history:
            recent_executions = list(self.execution_history)[-100:]
            features["avg_execution_time"] = statistics.mean(
                [ex.get("execution_time_ms", 0) for ex in recent_executions]
            )
            features["avg_success_rate"] = statistics.mean(
                [ex.get("success_rate", 1.0) for ex in recent_executions]
            )
            features["recent_failure_rate"] = 1.0 - features["avg_success_rate"]
        else:
            features["avg_execution_time"] = 1000.0  # Default estimate
            features["avg_success_rate"] = 0.9
            features["recent_failure_rate"] = 0.1

        # Time-based features
        now = datetime.now()
        features["hour_of_day"] = float(now.hour)
        features["day_of_week"] = float(now.weekday())
        features["minute_of_hour"] = float(now.minute)

        # System load indicators (simulated)
        features["system_load"] = min(len(self.execution_history) / 1000.0, 1.0)

        return features

    def predict_execution_time(self, context: dict[str, Any]) -> PredictionResult:
        """Predict execution time for given context."""
        cache_key = f"exec_time_{hash(str(sorted(context.items())))}"

        # Check cache
        if cache_key in self.prediction_cache:
            cached_result, timestamp = self.prediction_cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_result

        features = self.extract_features(context)
        result = self.models["execution_time"].predict(features)

        # Cache result
        self.prediction_cache[cache_key] = (result, datetime.now())

        return result

    def predict_optimal_parallelism(self, context: dict[str, Any]) -> PredictionResult:
        """Predict optimal parallelism level for given context."""
        features = self.extract_features(context)
        result = self.models["optimal_parallelism"].predict(features)

        # Clamp to reasonable bounds
        result.predicted_value = max(1.0, min(result.predicted_value, 50.0))

        return result

    def predict_failure_probability(self, context: dict[str, Any]) -> PredictionResult:
        """Predict probability of execution failure."""
        features = self.extract_features(context)
        result = self.models["failure_probability"].predict(features)

        # Clamp to [0, 1] probability range
        result.predicted_value = max(0.0, min(result.predicted_value, 1.0))

        return result

    def predict_resource_usage(self, context: dict[str, Any]) -> PredictionResult:
        """Predict resource usage (memory, CPU) for given context."""
        features = self.extract_features(context)
        result = self.models["resource_usage"].predict(features)

        # Ensure non-negative resource prediction
        result.predicted_value = max(0.0, result.predicted_value)

        return result

    def learn_from_execution(self, context: dict[str, Any], outcome: dict[str, Any]) -> None:
        """Learn from completed execution to improve predictions."""
        if not self.enable_online_learning:
            return

        features = self.extract_features(context)

        # Record execution for history
        execution_record = {
            **context,
            **outcome,
            "timestamp": datetime.now().isoformat(),
            "features": features
        }
        self.execution_history.append(execution_record)

        # Train models with actual outcomes
        if "execution_time_ms" in outcome:
            self.models["execution_time"].train(features, outcome["execution_time_ms"])

        if "success_rate" in outcome:
            failure_rate = 1.0 - outcome["success_rate"]
            self.models["failure_probability"].train(features, failure_rate)

        if "memory_usage_mb" in outcome:
            self.models["resource_usage"].train(features, outcome["memory_usage_mb"])

        # Train parallelism optimizer based on efficiency
        if "parallel_efficiency" in outcome:
            optimal_parallelism = context.get("parallel_limit", 10) * outcome["parallel_efficiency"]
            self.models["optimal_parallelism"].train(features, optimal_parallelism)

        # Update metrics
        self.metrics.training_samples += 1
        self.metrics.last_updated = datetime.now()

    def get_optimization_recommendations(self, context: dict[str, Any]) -> dict[str, Any]:
        """Get AI-driven optimization recommendations."""
        recommendations = {}

        # Predict optimal settings
        parallelism_pred = self.predict_optimal_parallelism(context)
        failure_pred = self.predict_failure_probability(context)
        time_pred = self.predict_execution_time(context)
        resource_pred = self.predict_resource_usage(context)

        recommendations["optimal_parallelism"] = {
            "value": int(parallelism_pred.predicted_value),
            "confidence": parallelism_pred.confidence,
            "reasoning": f"ML model predicts {parallelism_pred.predicted_value:.1f} optimal parallel workers"
        }

        recommendations["failure_risk"] = {
            "probability": failure_pred.predicted_value,
            "confidence": failure_pred.confidence,
            "risk_level": "high" if failure_pred.predicted_value > 0.3 else "medium" if failure_pred.predicted_value > 0.1 else "low"
        }

        recommendations["estimated_duration"] = {
            "milliseconds": time_pred.predicted_value,
            "confidence": time_pred.confidence,
            "category": "fast" if time_pred.predicted_value < 1000 else "medium" if time_pred.predicted_value < 5000 else "slow"
        }

        recommendations["resource_requirements"] = {
            "estimated_memory_mb": resource_pred.predicted_value,
            "confidence": resource_pred.confidence
        }

        # Generate actionable insights
        insights = []

        if failure_pred.predicted_value > 0.2:
            insights.append("Consider reducing parallelism or adding retry logic due to high failure risk")

        if time_pred.predicted_value > 10000:
            insights.append("Execution may be slow - consider optimizing tool selection or adding caching")

        if parallelism_pred.confidence > 0.7:
            insights.append(f"High confidence recommendation: use {int(parallelism_pred.predicted_value)} parallel workers")

        recommendations["insights"] = insights
        recommendations["model_confidence"] = statistics.mean([
            parallelism_pred.confidence,
            failure_pred.confidence,
            time_pred.confidence,
            resource_pred.confidence
        ])

        return recommendations

    def get_performance_metrics(self) -> MLMetrics:
        """Get current ML performance metrics."""
        # Calculate feature importance across models
        feature_importance = defaultdict(float)
        for model_name, model in self.models.items():
            for feature, weight in model.weights.items():
                feature_importance[feature] += abs(weight) / len(self.models)

        self.metrics.feature_importance = dict(feature_importance)
        self.metrics.training_samples = len(self.execution_history)

        return self.metrics

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self.prediction_cache.clear()

    def export_model_data(self) -> dict[str, Any]:
        """Export model data for analysis or persistence."""
        return {
            "models": {
                name: {
                    "weights": model.weights,
                    "bias": model.bias,
                    "learning_rate": model.learning_rate,
                    "feature_stats": dict(model.feature_stats)
                }
                for name, model in self.models.items()
            },
            "metrics": {
                "prediction_accuracy": self.metrics.prediction_accuracy,
                "model_confidence": self.metrics.model_confidence,
                "training_samples": self.metrics.training_samples,
                "feature_importance": self.metrics.feature_importance
            },
            "execution_history_size": len(self.execution_history)
        }


# Global ML optimizer instance
ml_optimizer = MLPredictiveOptimizer()
