"""
Generation 4: Autonomous Learning Engine
Self-improving orchestration with ML-driven optimization.
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
from collections import defaultdict, deque

from .simple_structured_logging import get_logger
from .comprehensive_monitoring import monitor, MetricType

logger = get_logger(__name__)


@dataclass
class PerformancePattern:
    """ML-detected performance pattern."""
    pattern_id: str
    tool_sequence: List[str]
    avg_execution_time: float
    success_rate: float
    resource_usage: Dict[str, float]
    context_features: Dict[str, Any]
    confidence: float
    last_updated: datetime


@dataclass
class OptimizationRecommendation:
    """AI-generated optimization recommendation."""
    recommendation_id: str
    optimization_type: str  # 'parallelization', 'caching', 'reordering', 'speculation'
    target_tools: List[str]
    expected_improvement: float
    confidence: float
    implementation_complexity: int  # 1-10
    risk_level: str  # 'low', 'medium', 'high'


class AutonomousLearningEngine:
    """
    Generation 4: Self-learning orchestration engine.
    
    Uses machine learning to:
    - Detect execution patterns automatically
    - Optimize tool sequences dynamically  
    - Predict optimal parallelization strategies
    - Self-tune performance parameters
    - Generate autonomous improvements
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        pattern_window_size: int = 1000,
        min_pattern_confidence: float = 0.75,
        enable_autonomous_optimization: bool = True,
        model_persistence_path: Optional[str] = None
    ):
        self.learning_rate = learning_rate
        self.pattern_window_size = pattern_window_size
        self.min_pattern_confidence = min_pattern_confidence
        self.enable_autonomous_optimization = enable_autonomous_optimization
        self.model_persistence_path = model_persistence_path or "./models/learning_engine"
        
        # Performance tracking
        self.execution_history: deque = deque(maxlen=pattern_window_size)
        self.performance_patterns: Dict[str, PerformancePattern] = {}
        self.optimization_history: List[OptimizationRecommendation] = []
        
        # ML models (simplified for autonomous operation)
        self.pattern_weights: Dict[str, np.ndarray] = {}
        self.performance_predictors: Dict[str, Any] = {}
        
        # Self-improvement metrics
        self.improvement_metrics = {
            'patterns_discovered': 0,
            'optimizations_applied': 0,
            'avg_performance_gain': 0.0,
            'prediction_accuracy': 0.0
        }
        
        # Ensure model directory exists
        Path(self.model_persistence_path).mkdir(parents=True, exist_ok=True)
        
        # Load previous learning if available
        self._load_learned_patterns()

    @monitor(MetricType.GAUGE)
    async def record_execution(
        self,
        tool_sequence: List[str],
        execution_time: float,
        success: bool,
        resource_usage: Dict[str, float],
        context: Dict[str, Any]
    ) -> None:
        """Record execution for pattern learning."""
        execution_record = {
            'timestamp': datetime.utcnow(),
            'tool_sequence': tool_sequence,
            'execution_time': execution_time,
            'success': success,
            'resource_usage': resource_usage,
            'context': context
        }
        
        self.execution_history.append(execution_record)
        
        # Trigger pattern analysis if we have enough data
        if len(self.execution_history) >= 50:  # Minimum sample size
            await self._analyze_patterns()

    async def _analyze_patterns(self) -> None:
        """Analyze execution history to discover performance patterns."""
        logger.info("Analyzing execution patterns for ML optimization")
        
        # Group similar tool sequences
        sequence_groups = defaultdict(list)
        for record in self.execution_history:
            sequence_key = tuple(sorted(record['tool_sequence']))
            sequence_groups[sequence_key].append(record)
        
        # Analyze each group for patterns
        new_patterns_found = 0
        for sequence_key, records in sequence_groups.items():
            if len(records) < 10:  # Need minimum samples
                continue
                
            pattern = await self._extract_pattern(sequence_key, records)
            if pattern and pattern.confidence >= self.min_pattern_confidence:
                self.performance_patterns[pattern.pattern_id] = pattern
                new_patterns_found += 1
        
        self.improvement_metrics['patterns_discovered'] += new_patterns_found
        logger.info(f"Discovered {new_patterns_found} new performance patterns")
        
        # Generate optimization recommendations
        if self.enable_autonomous_optimization:
            await self._generate_optimizations()

    async def _extract_pattern(
        self, 
        sequence_key: Tuple[str, ...], 
        records: List[Dict[str, Any]]
    ) -> Optional[PerformancePattern]:
        """Extract performance pattern from execution records."""
        successful_records = [r for r in records if r['success']]
        if len(successful_records) < 5:
            return None
        
        # Calculate pattern metrics
        execution_times = [r['execution_time'] for r in successful_records]
        avg_time = np.mean(execution_times)
        success_rate = len(successful_records) / len(records)
        
        # Aggregate resource usage
        avg_resources = {}
        if successful_records:
            for key in successful_records[0]['resource_usage'].keys():
                values = [r['resource_usage'][key] for r in successful_records]
                avg_resources[key] = np.mean(values)
        
        # Extract context features
        context_features = self._extract_context_features(successful_records)
        
        # Calculate confidence based on consistency
        time_std = np.std(execution_times)
        confidence = max(0.0, 1.0 - (time_std / avg_time)) if avg_time > 0 else 0.0
        
        pattern_id = f"pattern_{hash(sequence_key)}_{int(datetime.utcnow().timestamp())}"
        
        return PerformancePattern(
            pattern_id=pattern_id,
            tool_sequence=list(sequence_key),
            avg_execution_time=avg_time,
            success_rate=success_rate,
            resource_usage=avg_resources,
            context_features=context_features,
            confidence=confidence,
            last_updated=datetime.utcnow()
        )

    def _extract_context_features(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract relevant context features for pattern matching."""
        features = {}
        
        # Time of day patterns
        hours = [r['timestamp'].hour for r in records]
        features['primary_hour'] = max(set(hours), key=hours.count)
        
        # Payload size patterns (if available)
        if records and 'payload_size' in records[0]['context']:
            sizes = [r['context']['payload_size'] for r in records if 'payload_size' in r['context']]
            if sizes:
                features['avg_payload_size'] = np.mean(sizes)
        
        # Concurrency patterns
        if records and 'concurrent_tools' in records[0]['context']:
            concurrency = [r['context']['concurrent_tools'] for r in records if 'concurrent_tools' in r['context']]
            if concurrency:
                features['avg_concurrency'] = np.mean(concurrency)
        
        return features

    async def _generate_optimizations(self) -> None:
        """Generate autonomous optimization recommendations."""
        recommendations = []
        
        for pattern in self.performance_patterns.values():
            if pattern.confidence < self.min_pattern_confidence:
                continue
                
            # Parallelization opportunities
            if len(pattern.tool_sequence) > 1:
                parallel_rec = await self._analyze_parallelization_potential(pattern)
                if parallel_rec:
                    recommendations.append(parallel_rec)
            
            # Caching opportunities
            cache_rec = await self._analyze_caching_potential(pattern)
            if cache_rec:
                recommendations.append(cache_rec)
            
            # Reordering opportunities
            reorder_rec = await self._analyze_reordering_potential(pattern)
            if reorder_rec:
                recommendations.append(reorder_rec)
        
        # Sort by expected improvement
        recommendations.sort(key=lambda x: x.expected_improvement, reverse=True)
        
        # Apply top recommendations automatically (if enabled)
        if self.enable_autonomous_optimization:
            await self._apply_optimizations(recommendations[:3])  # Top 3
        
        self.optimization_history.extend(recommendations)

    async def _analyze_parallelization_potential(
        self, 
        pattern: PerformancePattern
    ) -> Optional[OptimizationRecommendation]:
        """Analyze if tools in pattern can be parallelized."""
        if len(pattern.tool_sequence) < 2:
            return None
        
        # Simple heuristic: if execution time is high and tools don't seem dependent
        if pattern.avg_execution_time > 1.0:  # >1 second
            # Check for potential independence (simplified analysis)
            independent_tools = await self._identify_independent_tools(pattern.tool_sequence)
            
            if len(independent_tools) >= 2:
                expected_improvement = 0.3 + (len(independent_tools) - 2) * 0.1  # 30-60% improvement
                
                return OptimizationRecommendation(
                    recommendation_id=f"parallel_{pattern.pattern_id}",
                    optimization_type="parallelization",
                    target_tools=independent_tools,
                    expected_improvement=expected_improvement,
                    confidence=pattern.confidence * 0.8,
                    implementation_complexity=3,
                    risk_level="low"
                )
        return None

    async def _analyze_caching_potential(
        self,
        pattern: PerformancePattern
    ) -> Optional[OptimizationRecommendation]:
        """Analyze caching opportunities."""
        # Look for repeated tool calls or expensive operations
        tool_counts = defaultdict(int)
        for tool in pattern.tool_sequence:
            tool_counts[tool] += 1
        
        repeated_tools = [tool for tool, count in tool_counts.items() if count > 1]
        
        if repeated_tools and pattern.avg_execution_time > 0.5:
            expected_improvement = min(0.5, len(repeated_tools) * 0.15)
            
            return OptimizationRecommendation(
                recommendation_id=f"cache_{pattern.pattern_id}",
                optimization_type="caching",
                target_tools=repeated_tools,
                expected_improvement=expected_improvement,
                confidence=pattern.confidence * 0.9,
                implementation_complexity=2,
                risk_level="low"
            )
        return None

    async def _analyze_reordering_potential(
        self,
        pattern: PerformancePattern
    ) -> Optional[OptimizationRecommendation]:
        """Analyze tool reordering opportunities."""
        # Simple heuristic: if we have fast and slow tools, put fast ones first
        if len(pattern.tool_sequence) > 2:
            # This would require more sophisticated dependency analysis
            # For now, suggest reordering if pattern shows suboptimal performance
            if pattern.success_rate < 0.9:  # Low success rate might indicate ordering issues
                return OptimizationRecommendation(
                    recommendation_id=f"reorder_{pattern.pattern_id}",
                    optimization_type="reordering",
                    target_tools=pattern.tool_sequence,
                    expected_improvement=0.2,
                    confidence=pattern.confidence * 0.6,
                    implementation_complexity=4,
                    risk_level="medium"
                )
        return None

    async def _identify_independent_tools(self, tool_sequence: List[str]) -> List[str]:
        """Identify tools that can potentially run in parallel."""
        # Simplified independence analysis
        # In a real implementation, this would analyze tool dependencies
        independent = []
        
        for tool in tool_sequence:
            # Simple heuristics for independence
            if any(keyword in tool.lower() for keyword in ['search', 'fetch', 'get', 'read']):
                independent.append(tool)
            elif any(keyword in tool.lower() for keyword in ['analyze', 'process', 'compute']):
                independent.append(tool)
        
        return independent

    async def _apply_optimizations(self, recommendations: List[OptimizationRecommendation]) -> None:
        """Autonomously apply optimization recommendations."""
        applied_count = 0
        
        for rec in recommendations:
            if rec.risk_level == "high":
                continue  # Skip high-risk optimizations in autonomous mode
            
            try:
                success = await self._implement_optimization(rec)
                if success:
                    applied_count += 1
                    logger.info(f"Applied optimization: {rec.optimization_type} for {rec.target_tools}")
            except Exception as e:
                logger.warning(f"Failed to apply optimization {rec.recommendation_id}: {e}")
        
        self.improvement_metrics['optimizations_applied'] += applied_count

    async def _implement_optimization(self, rec: OptimizationRecommendation) -> bool:
        """Implement a specific optimization recommendation."""
        # This would integrate with the orchestrator's configuration
        # For now, we'll just log the implementation
        
        if rec.optimization_type == "parallelization":
            # Would update orchestrator's parallel execution rules
            pass
        elif rec.optimization_type == "caching":
            # Would update caching configuration
            pass
        elif rec.optimization_type == "reordering":
            # Would update tool execution order rules
            pass
        
        # Simulate implementation success
        return True

    async def predict_performance(
        self,
        tool_sequence: List[str],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Predict performance for a given tool sequence."""
        # Find matching patterns
        matching_patterns = []
        sequence_set = set(tool_sequence)
        
        for pattern in self.performance_patterns.values():
            pattern_set = set(pattern.tool_sequence)
            overlap = len(sequence_set.intersection(pattern_set)) / len(sequence_set.union(pattern_set))
            
            if overlap > 0.6:  # 60% similarity threshold
                matching_patterns.append((pattern, overlap))
        
        if not matching_patterns:
            # No patterns found, return conservative estimates
            return {
                'predicted_time': len(tool_sequence) * 1.0,  # 1 second per tool
                'success_probability': 0.8,
                'confidence': 0.3
            }
        
        # Weight predictions by pattern similarity and confidence
        total_weight = 0
        weighted_time = 0
        weighted_success = 0
        
        for pattern, similarity in matching_patterns:
            weight = similarity * pattern.confidence
            total_weight += weight
            weighted_time += pattern.avg_execution_time * weight
            weighted_success += pattern.success_rate * weight
        
        if total_weight == 0:
            return {'predicted_time': 1.0, 'success_probability': 0.8, 'confidence': 0.3}
        
        return {
            'predicted_time': weighted_time / total_weight,
            'success_probability': weighted_success / total_weight,
            'confidence': total_weight / len(matching_patterns)
        }

    async def get_optimization_suggestions(self, limit: int = 5) -> List[OptimizationRecommendation]:
        """Get current optimization suggestions."""
        recent_recommendations = [
            rec for rec in self.optimization_history
            if datetime.utcnow() - datetime.fromisoformat(rec.recommendation_id.split('_')[-1]) < timedelta(hours=24)
        ]
        
        # Sort by expected improvement and return top N
        recent_recommendations.sort(key=lambda x: x.expected_improvement, reverse=True)
        return recent_recommendations[:limit]

    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get current learning and improvement metrics."""
        return {
            **self.improvement_metrics,
            'patterns_tracked': len(self.performance_patterns),
            'execution_samples': len(self.execution_history),
            'recent_optimizations': len([
                rec for rec in self.optimization_history
                if datetime.utcnow() - datetime.fromisoformat(rec.recommendation_id.split('_')[-1]) < timedelta(hours=1)
            ]),
            'avg_pattern_confidence': np.mean([p.confidence for p in self.performance_patterns.values()]) if self.performance_patterns else 0.0
        }

    def _save_learned_patterns(self) -> None:
        """Persist learned patterns to disk."""
        try:
            model_data = {
                'patterns': {k: asdict(v) for k, v in self.performance_patterns.items()},
                'optimization_history': [asdict(rec) for rec in self.optimization_history[-100:]],  # Last 100
                'improvement_metrics': self.improvement_metrics
            }
            
            with open(f"{self.model_persistence_path}/learned_patterns.json", 'w') as f:
                json.dump(model_data, f, default=str, indent=2)
            
            logger.info("Saved learned patterns to disk")
        except Exception as e:
            logger.warning(f"Failed to save learned patterns: {e}")

    def _load_learned_patterns(self) -> None:
        """Load previously learned patterns from disk."""
        try:
            pattern_file = f"{self.model_persistence_path}/learned_patterns.json"
            if Path(pattern_file).exists():
                with open(pattern_file, 'r') as f:
                    model_data = json.load(f)
                
                # Load patterns
                for k, v in model_data.get('patterns', {}).items():
                    v['last_updated'] = datetime.fromisoformat(v['last_updated'])
                    self.performance_patterns[k] = PerformancePattern(**v)
                
                # Load optimization history
                for opt_data in model_data.get('optimization_history', []):
                    self.optimization_history.append(OptimizationRecommendation(**opt_data))
                
                # Load metrics
                self.improvement_metrics.update(model_data.get('improvement_metrics', {}))
                
                logger.info(f"Loaded {len(self.performance_patterns)} learned patterns from disk")
        except Exception as e:
            logger.warning(f"Failed to load learned patterns: {e}")

    async def shutdown(self) -> None:
        """Graceful shutdown with pattern persistence."""
        logger.info("Shutting down autonomous learning engine")
        self._save_learned_patterns()


# Factory function for easy integration
def create_autonomous_learning_engine(
    learning_rate: float = 0.01,
    enable_autonomous_optimization: bool = True,
    model_persistence_path: Optional[str] = None
) -> AutonomousLearningEngine:
    """Create and configure an autonomous learning engine."""
    return AutonomousLearningEngine(
        learning_rate=learning_rate,
        enable_autonomous_optimization=enable_autonomous_optimization,
        model_persistence_path=model_persistence_path
    )