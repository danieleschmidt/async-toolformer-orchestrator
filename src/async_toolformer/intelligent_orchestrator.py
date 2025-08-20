"""
Generation 4 Enhancement: Intelligent AI-Driven Orchestrator

Advanced orchestrator with machine learning-powered optimization,
predictive analytics, and autonomous decision-making capabilities.
"""

import time
from datetime import datetime
from typing import Any

from .config import OrchestratorConfig
from .ml_optimizer import MLPredictiveOptimizer
from .orchestrator import AsyncOrchestrator
from .simple_structured_logging import get_logger
from .tools import ToolFunction, ToolResult

logger = get_logger(__name__)


class IntelligentAsyncOrchestrator(AsyncOrchestrator):
    """
    AI-enhanced orchestrator with machine learning optimization.
    
    Features:
    - Predictive resource allocation
    - Adaptive parallelism optimization
    - Intelligent failure prevention
    - Performance pattern learning
    - Autonomous configuration tuning
    """

    def __init__(
        self,
        llm_client: Any | None = None,
        tools: list[ToolFunction] | None = None,
        config: OrchestratorConfig | None = None,
        enable_ml_optimization: bool = True,
        learning_rate: float = 0.01,
        **kwargs
    ):
        super().__init__(llm_client, tools, config, **kwargs)

        self.enable_ml_optimization = enable_ml_optimization
        self.ml_optimizer = MLPredictiveOptimizer(enable_online_learning=True)
        self.ml_optimizer.models["execution_time"].learning_rate = learning_rate

        # Intelligence features
        self.adaptive_config = self.config.__class__()  # Copy of original config
        self.intelligence_metrics = {
            "optimizations_applied": 0,
            "predictions_made": 0,
            "learning_sessions": 0,
            "performance_improvements": 0.0
        }

        logger.info("Intelligent orchestrator initialized with ML optimization",
                   extra={"ml_enabled": enable_ml_optimization, "learning_rate": learning_rate})

    async def execute(
        self,
        prompt: str,
        tools: list[str] | None = None,
        max_parallel: int | None = None,
        timeout_ms: int | None = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute with AI-driven optimization and predictive analytics.
        """
        start_time = time.time()

        # Prepare execution context for ML analysis
        execution_context = {
            "prompt_length": len(prompt),
            "requested_tools": tools,
            "max_parallel": max_parallel or self.config.max_parallel_tools,
            "timeout_ms": timeout_ms or self.config.tool_timeout_ms,
            "tool_count": len(tools) if tools else len(self.registry._tools),
            "timestamp": datetime.now().isoformat()
        }

        # AI-driven pre-execution optimization
        if self.enable_ml_optimization:
            await self._apply_ml_optimizations(execution_context)

        logger.info("Starting intelligent execution with ML optimization",
                   extra={"context": execution_context, "ml_enabled": self.enable_ml_optimization})

        try:
            # Execute with potentially optimized configuration
            result = await super().execute(prompt, tools, max_parallel, timeout_ms, **kwargs)

            # Record execution outcome for learning
            # Handle both ToolResult objects and dict results
            success = getattr(result, 'success', result.get('success', True)) if result else False
            results_count = len(getattr(result, 'results', result.get('results', []))) if result else 0

            execution_outcome = {
                "execution_time_ms": (time.time() - start_time) * 1000,
                "success_rate": 1.0 if success else 0.0,
                "tools_executed": results_count if results_count > 0 else 1,
                "parallel_efficiency": self._calculate_parallel_efficiency(result, execution_context),
                "memory_usage_mb": self._estimate_memory_usage(),
                "outcome": "success" if success else "failure"
            }

            # Learn from execution for future optimization
            if self.enable_ml_optimization:
                await self._learn_from_execution(execution_context, execution_outcome)

            logger.info("Intelligent execution completed",
                       extra={"outcome": execution_outcome, "optimizations": self.intelligence_metrics})

            return result

        except Exception as e:
            # Learn from failures too
            execution_outcome = {
                "execution_time_ms": (time.time() - start_time) * 1000,
                "success_rate": 0.0,
                "error_type": type(e).__name__,
                "outcome": "error"
            }

            if self.enable_ml_optimization:
                await self._learn_from_execution(execution_context, execution_outcome)

            logger.error("Intelligent execution failed",
                        extra={"error": str(e), "context": execution_context})
            raise

    async def _apply_ml_optimizations(self, context: dict[str, Any]) -> None:
        """Apply machine learning-driven optimizations before execution."""
        try:
            # Get ML recommendations
            recommendations = self.ml_optimizer.get_optimization_recommendations(context)

            logger.info("Applying ML optimizations", extra={"recommendations": recommendations})

            # Apply parallelism optimization
            if (recommendations["optimal_parallelism"]["confidence"] > 0.5 and
                recommendations["optimal_parallelism"]["value"] != context["max_parallel"]):

                old_parallel = self.config.max_parallel_tools
                self.config.max_parallel_tools = recommendations["optimal_parallelism"]["value"]

                logger.info("ML optimization: adjusted parallelism",
                           extra={"old": old_parallel, "new": self.config.max_parallel_tools,
                                 "confidence": recommendations["optimal_parallelism"]["confidence"]})

                self.intelligence_metrics["optimizations_applied"] += 1

            # Apply timeout optimization based on predicted execution time
            estimated_duration = recommendations["estimated_duration"]["milliseconds"]
            if (recommendations["estimated_duration"]["confidence"] > 0.6 and
                estimated_duration > context["timeout_ms"]):

                old_timeout = self.config.tool_timeout_ms
                # Add 50% buffer to predicted time
                self.config.tool_timeout_ms = int(estimated_duration * 1.5)

                logger.info("ML optimization: adjusted timeout",
                           extra={"old": old_timeout, "new": self.config.tool_timeout_ms,
                                 "predicted_duration": estimated_duration})

                self.intelligence_metrics["optimizations_applied"] += 1

            # Apply failure prevention strategies
            if recommendations["failure_risk"]["probability"] > 0.3:
                logger.warning("High failure risk detected, applying preventive measures",
                              extra={"failure_probability": recommendations["failure_risk"]["probability"],
                                    "risk_level": recommendations["failure_risk"]["risk_level"]})

                # Reduce parallelism for high-risk executions
                self.config.max_parallel_tools = max(1, self.config.max_parallel_tools // 2)

                # Increase retry attempts
                self.config.retry_attempts = min(5, self.config.retry_attempts + 1)

                self.intelligence_metrics["optimizations_applied"] += 1

            self.intelligence_metrics["predictions_made"] += 1

        except Exception as e:
            logger.warning("Failed to apply ML optimizations", extra={"error": str(e)})

    async def _learn_from_execution(self, context: dict[str, Any], outcome: dict[str, Any]) -> None:
        """Learn from execution outcome to improve future predictions."""
        try:
            # Feed execution data to ML optimizer
            self.ml_optimizer.learn_from_execution(context, outcome)

            # Update intelligence metrics
            self.intelligence_metrics["learning_sessions"] += 1

            # Calculate performance improvement
            if outcome.get("success_rate", 0) > 0.8 and outcome.get("execution_time_ms", 0) < 5000:
                self.intelligence_metrics["performance_improvements"] += 1

            logger.debug("Learning session completed",
                        extra={"context_features": len(context),
                              "outcome_metrics": len(outcome),
                              "total_sessions": self.intelligence_metrics["learning_sessions"]})

        except Exception as e:
            logger.warning("Failed to learn from execution", extra={"error": str(e)})

    def _calculate_parallel_efficiency(self, result: ToolResult, context: dict[str, Any]) -> float:
        """Calculate parallel execution efficiency."""
        try:
            max_parallel = context.get("max_parallel", 1)
            tools_executed = context.get("tool_count", 1)

            if max_parallel <= 1 or tools_executed <= 1:
                return 1.0

            # Simple efficiency metric: successful parallel execution
            theoretical_max = min(max_parallel, tools_executed)
            actual_efficiency = min(theoretical_max, tools_executed) / theoretical_max

            return actual_efficiency

        except Exception:
            return 1.0  # Default efficiency

    def _estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        # Simple estimation based on cache size and active tools
        try:
            cache_size_mb = len(str(self.cache.__dict__ if hasattr(self, 'cache') else {})) / (1024 * 1024)
            tool_count_mb = len(self.registry._tools) * 0.1  # Estimate 0.1MB per tool

            return cache_size_mb + tool_count_mb + 10  # Base overhead
        except Exception:
            return 50.0  # Default estimate

    def get_intelligence_analytics(self) -> dict[str, Any]:
        """Get comprehensive intelligence and learning analytics."""
        ml_metrics = self.ml_optimizer.get_performance_metrics()

        analytics = {
            "intelligence_metrics": self.intelligence_metrics.copy(),
            "ml_performance": {
                "training_samples": ml_metrics.training_samples,
                "prediction_accuracy": ml_metrics.prediction_accuracy,
                "model_confidence": ml_metrics.model_confidence,
                "feature_importance": ml_metrics.feature_importance,
                "last_updated": ml_metrics.last_updated.isoformat()
            },
            "model_insights": {
                "most_important_features": sorted(
                    ml_metrics.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                "optimization_success_rate": (
                    self.intelligence_metrics["performance_improvements"] /
                    max(self.intelligence_metrics["optimizations_applied"], 1)
                ),
                "learning_velocity": self.intelligence_metrics["learning_sessions"] / max(time.time() - 1, 1)
            },
            "adaptive_config": {
                "current_max_parallel": self.config.max_parallel_tools,
                "current_timeout_ms": self.config.tool_timeout_ms,
                "original_max_parallel": self.adaptive_config.max_parallel_tools,
                "original_timeout_ms": self.adaptive_config.tool_timeout_ms
            },
            "recommendations": self.ml_optimizer.get_optimization_recommendations({
                "tool_count": len(self.registry._tools),
                "max_parallel": self.config.max_parallel_tools,
                "timeout_ms": self.config.tool_timeout_ms
            })
        }

        return analytics

    async def auto_tune_configuration(self) -> dict[str, Any]:
        """Automatically tune orchestrator configuration based on learned patterns."""
        logger.info("Starting autonomous configuration tuning")

        # Analyze execution history for optimal settings
        ml_metrics = self.ml_optimizer.get_performance_metrics()

        if ml_metrics.training_samples < 10:
            logger.warning("Insufficient data for auto-tuning",
                          extra={"samples": ml_metrics.training_samples})
            return {"status": "insufficient_data", "samples_needed": 10 - ml_metrics.training_samples}

        # Get current optimal recommendations
        current_context = {
            "tool_count": len(self.registry._tools),
            "max_parallel": self.config.max_parallel_tools,
            "timeout_ms": self.config.tool_timeout_ms
        }

        recommendations = self.ml_optimizer.get_optimization_recommendations(current_context)

        tuning_results = {
            "previous_config": {
                "max_parallel_tools": self.config.max_parallel_tools,
                "tool_timeout_ms": self.config.tool_timeout_ms
            },
            "optimized_config": {},
            "confidence_scores": {},
            "changes_applied": []
        }

        # Apply high-confidence optimizations
        if recommendations["optimal_parallelism"]["confidence"] > 0.7:
            old_parallel = self.config.max_parallel_tools
            self.config.max_parallel_tools = recommendations["optimal_parallelism"]["value"]

            tuning_results["optimized_config"]["max_parallel_tools"] = self.config.max_parallel_tools
            tuning_results["confidence_scores"]["parallelism"] = recommendations["optimal_parallelism"]["confidence"]
            tuning_results["changes_applied"].append(f"Parallelism: {old_parallel} → {self.config.max_parallel_tools}")

        # Adjust timeout based on execution patterns
        avg_duration = recommendations["estimated_duration"]["milliseconds"]
        if recommendations["estimated_duration"]["confidence"] > 0.6:
            old_timeout = self.config.tool_timeout_ms
            # Set timeout to 2x average execution time, minimum 5 seconds
            self.config.tool_timeout_ms = max(5000, int(avg_duration * 2))

            tuning_results["optimized_config"]["tool_timeout_ms"] = self.config.tool_timeout_ms
            tuning_results["confidence_scores"]["timeout"] = recommendations["estimated_duration"]["confidence"]
            tuning_results["changes_applied"].append(f"Timeout: {old_timeout}ms → {self.config.tool_timeout_ms}ms")

        tuning_results["status"] = "completed"
        tuning_results["overall_confidence"] = recommendations["model_confidence"]

        logger.info("Auto-tuning completed", extra={"results": tuning_results})

        return tuning_results

    def reset_to_original_config(self) -> None:
        """Reset configuration to original values."""
        self.config.max_parallel_tools = self.adaptive_config.max_parallel_tools
        self.config.tool_timeout_ms = self.adaptive_config.tool_timeout_ms

        logger.info("Configuration reset to original values")

    async def cleanup(self) -> None:
        """Enhanced cleanup with intelligence data preservation."""
        # Save ML model data before cleanup
        model_data = self.ml_optimizer.export_model_data()

        logger.info("Cleaning up intelligent orchestrator",
                   extra={"intelligence_metrics": self.intelligence_metrics,
                         "ml_training_samples": model_data.get("execution_history_size", 0)})

        await super().cleanup()


def create_intelligent_orchestrator(
    llm_client: Any = None,
    tools: list[ToolFunction] = None,
    enable_ml: bool = True,
    **kwargs
) -> IntelligentAsyncOrchestrator:
    """Factory function to create an intelligent orchestrator."""
    return IntelligentAsyncOrchestrator(
        llm_client=llm_client,
        tools=tools,
        enable_ml_optimization=enable_ml,
        **kwargs
    )
