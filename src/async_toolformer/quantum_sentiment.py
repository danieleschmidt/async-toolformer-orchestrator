"""Quantum-Enhanced Sentiment Analysis using the Quantum Orchestrator."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from .quantum_orchestrator import QuantumAsyncOrchestrator, create_quantum_orchestrator
from .quantum_security import SecurityContext, SecurityLevel
from .sentiment_analyzer import (
    analyze_batch_sentiment,
    analyze_social_media_sentiment,
    analyze_text_sentiment,
    compare_sentiment_sources,
    SentimentResult,
    BatchSentimentResult
)
from .tools import Tool
from .exceptions import OrchestratorError

logger = structlog.get_logger(__name__)


@dataclass
class QuantumSentimentConfig:
    """Configuration for quantum-enhanced sentiment analysis."""
    max_parallel_analyses: int = 20
    enable_speculation: bool = True
    coherence_threshold: float = 0.8
    superposition_depth: int = 3
    security_level: SecurityLevel = SecurityLevel.HIGH
    enable_quantum_optimization: bool = True
    enable_multi_source_fusion: bool = True
    social_media_boost: bool = True


class QuantumSentimentAnalyzer:
    """Quantum-enhanced sentiment analyzer using the orchestrator."""
    
    def __init__(
        self, 
        config: Optional[QuantumSentimentConfig] = None,
        orchestrator: Optional[QuantumAsyncOrchestrator] = None
    ):
        """Initialize the quantum sentiment analyzer."""
        self.config = config or QuantumSentimentConfig()
        
        if orchestrator is None:
            # Create quantum orchestrator with sentiment-optimized settings
            self.orchestrator = create_quantum_orchestrator(
                max_parallel=self.config.max_parallel_analyses,
                enable_speculation=self.config.enable_speculation,
                enable_quantum_optimization=self.config.enable_quantum_optimization,
                coherence_threshold=self.config.coherence_threshold,
                superposition_depth=self.config.superposition_depth,
                security_level=self.config.security_level
            )
        else:
            self.orchestrator = orchestrator
            
        # Register sentiment analysis tools
        self.orchestrator.register_tool(analyze_text_sentiment)
        self.orchestrator.register_tool(analyze_batch_sentiment)
        self.orchestrator.register_tool(analyze_social_media_sentiment)
        self.orchestrator.register_tool(compare_sentiment_sources)
        self.orchestrator.register_tool(self._quantum_multi_approach_analysis)
        self.orchestrator.register_tool(self._quantum_sentiment_fusion)
        self.orchestrator.register_tool(self._quantum_temporal_sentiment)
        
    @Tool(
        name="quantum_multi_approach_sentiment",
        description="Analyze sentiment using multiple approaches simultaneously with quantum optimization"
    )
    async def _quantum_multi_approach_analysis(
        self,
        text: str,
        approaches: List[str] = None,
        include_social_analysis: bool = True
    ) -> Dict[str, Any]:
        """Perform multi-approach sentiment analysis with quantum optimization."""
        if approaches is None:
            approaches = ["rule_based", "social_media", "emotion_focused"]
            
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create analysis tasks for parallel execution
            tasks = []
            
            # Rule-based analysis
            if "rule_based" in approaches:
                tasks.append(
                    self._create_rule_based_task(text)
                )
            
            # Social media analysis
            if "social_media" in approaches and include_social_analysis:
                tasks.append(
                    self._create_social_media_task(text)
                )
                
            # Emotion-focused analysis
            if "emotion_focused" in approaches:
                tasks.append(
                    self._create_emotion_focused_task(text)
                )
                
            # Execute all approaches in quantum superposition
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"Analysis approach {i} failed", error=str(result))
                else:
                    successful_results.append(result)
                    
            if not successful_results:
                raise OrchestratorError("All sentiment analysis approaches failed")
                
            # Quantum fusion of results
            fused_result = await self._quantum_fuse_results(successful_results)
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "fused_sentiment": fused_result,
                "individual_results": successful_results,
                "analysis_approaches": approaches,
                "quantum_coherence": self._calculate_result_coherence(successful_results),
                "processing_time_ms": processing_time,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error("Quantum multi-approach analysis failed", error=str(e))
            raise OrchestratorError(f"Quantum sentiment analysis failed: {e}")
    
    async def _create_rule_based_task(self, text: str) -> SentimentResult:
        """Create rule-based analysis task."""
        return await analyze_text_sentiment(
            text=text,
            include_emotions=True,
            include_keywords=True
        )
    
    async def _create_social_media_task(self, text: str) -> SentimentResult:
        """Create social media analysis task."""
        return await analyze_social_media_sentiment(
            text=text,
            platform="twitter",  # Can be parameterized
            extract_hashtags=True,
            extract_mentions=True
        )
    
    async def _create_emotion_focused_task(self, text: str) -> SentimentResult:
        """Create emotion-focused analysis task."""
        # Enhanced emotion analysis version
        return await analyze_text_sentiment(
            text=text,
            include_emotions=True,
            include_keywords=False  # Focus on emotions
        )
    
    async def _quantum_fuse_results(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Quantum fusion of multiple sentiment analysis results."""
        if not results:
            return {}
            
        # Calculate weighted sentiment based on confidence
        total_confidence = sum(result.sentiment.confidence for result in results)
        
        if total_confidence == 0:
            return {"error": "No confident results to fuse"}
            
        weighted_score = sum(
            result.sentiment.score * result.sentiment.confidence 
            for result in results
        ) / total_confidence
        
        weighted_confidence = total_confidence / len(results)
        
        # Determine consensus polarity
        polarities = [result.sentiment.polarity for result in results]
        polarity_counts = {}
        for polarity in polarities:
            polarity_counts[polarity.value] = polarity_counts.get(polarity.value, 0) + 1
        
        dominant_polarity = max(polarity_counts.items(), key=lambda x: x[1])[0]
        
        # Fuse emotions
        all_emotions = []
        for result in results:
            all_emotions.extend(result.emotions)
            
        # Aggregate emotions by type
        emotion_aggregation = {}
        for emotion in all_emotions:
            emotion_type = emotion.emotion
            if emotion_type not in emotion_aggregation:
                emotion_aggregation[emotion_type] = []
            emotion_aggregation[emotion_type].append(emotion)
            
        fused_emotions = []
        for emotion_type, emotion_list in emotion_aggregation.items():
            avg_intensity = sum(e.intensity for e in emotion_list) / len(emotion_list)
            avg_confidence = sum(e.confidence for e in emotion_list) / len(emotion_list)
            fused_emotions.append({
                "emotion": emotion_type.value,
                "intensity": avg_intensity,
                "confidence": avg_confidence
            })
            
        # Fuse keywords
        all_keywords = []
        for result in results:
            all_keywords.extend(result.keywords)
            
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "sentiment": {
                "polarity": dominant_polarity,
                "score": weighted_score,
                "confidence": weighted_confidence
            },
            "emotions": sorted(fused_emotions, key=lambda x: x["intensity"], reverse=True),
            "keywords": [keyword for keyword, count in top_keywords],
            "fusion_method": "quantum_weighted_consensus",
            "source_count": len(results),
            "polarity_consensus": polarity_counts
        }
    
    def _calculate_result_coherence(self, results: List[SentimentResult]) -> float:
        """Calculate quantum coherence between results."""
        if len(results) < 2:
            return 1.0
            
        scores = [result.sentiment.score for result in results]
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Coherence is inversely related to variance
        coherence = 1.0 / (1.0 + variance)
        return min(1.0, coherence)
    
    @Tool(
        name="quantum_sentiment_fusion",
        description="Fuse sentiment analysis from multiple text sources using quantum entanglement principles"
    )
    async def _quantum_sentiment_fusion(
        self,
        text_sources: Dict[str, str],
        fusion_method: str = "quantum_superposition",
        enable_cross_correlation: bool = True
    ) -> Dict[str, Any]:
        """Advanced quantum fusion of multiple text sources."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Analyze each source
            source_tasks = []
            for source_name, text in text_sources.items():
                if text.strip():
                    task = self._quantum_multi_approach_analysis(text)
                    source_tasks.append((source_name, task))
            
            # Execute in quantum superposition
            source_results = {}
            for source_name, task in source_tasks:
                try:
                    result = await task
                    source_results[source_name] = result
                except Exception as e:
                    logger.warning(f"Source {source_name} analysis failed", error=str(e))
            
            if not source_results:
                raise OrchestratorError("No sources could be analyzed")
            
            # Quantum entanglement analysis
            if enable_cross_correlation:
                cross_correlations = self._calculate_cross_correlations(source_results)
            else:
                cross_correlations = {}
            
            # Fusion based on quantum principles
            if fusion_method == "quantum_superposition":
                fused_sentiment = self._superposition_fusion(source_results)
            elif fusion_method == "quantum_entanglement":
                fused_sentiment = self._entanglement_fusion(source_results, cross_correlations)
            else:
                fused_sentiment = self._coherent_fusion(source_results)
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "fused_sentiment": fused_sentiment,
                "source_analyses": source_results,
                "cross_correlations": cross_correlations,
                "fusion_method": fusion_method,
                "quantum_metrics": {
                    "coherence": self._calculate_system_coherence(source_results),
                    "entanglement_strength": len(cross_correlations),
                    "superposition_states": len(source_results)
                },
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            logger.error("Quantum sentiment fusion failed", error=str(e))
            raise OrchestratorError(f"Quantum sentiment fusion failed: {e}")
    
    def _calculate_cross_correlations(
        self, 
        source_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate cross-correlations between sentiment sources."""
        correlations = {}
        sources = list(source_results.keys())
        
        for i, source1 in enumerate(sources):
            for source2 in sources[i+1:]:
                try:
                    result1 = source_results[source1]["fused_sentiment"]["sentiment"]
                    result2 = source_results[source2]["fused_sentiment"]["sentiment"]
                    
                    score1 = result1["score"]
                    score2 = result2["score"]
                    
                    # Simple correlation calculation
                    correlation = 1.0 - abs(score1 - score2)
                    correlations[f"{source1}__{source2}"] = correlation
                    
                except (KeyError, TypeError) as e:
                    logger.warning(f"Could not calculate correlation between {source1} and {source2}", error=str(e))
                    
        return correlations
    
    def _superposition_fusion(self, source_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Quantum superposition-based fusion."""
        all_scores = []
        all_confidences = []
        
        for result in source_results.values():
            try:
                sentiment = result["fused_sentiment"]["sentiment"]
                all_scores.append(sentiment["score"])
                all_confidences.append(sentiment["confidence"])
            except (KeyError, TypeError):
                continue
        
        if not all_scores:
            return {"error": "No valid scores for fusion"}
        
        # Superposition: weighted by quantum probability amplitudes
        weights = [conf ** 2 for conf in all_confidences]  # Probability amplitudes squared
        total_weight = sum(weights)
        
        if total_weight == 0:
            fused_score = sum(all_scores) / len(all_scores)
            fused_confidence = 0.5
        else:
            fused_score = sum(score * weight for score, weight in zip(all_scores, weights)) / total_weight
            fused_confidence = sum(all_confidences) / len(all_confidences)
        
        # Determine polarity
        if fused_score > 0.1:
            polarity = "positive"
        elif fused_score < -0.1:
            polarity = "negative"
        else:
            polarity = "neutral"
        
        return {
            "sentiment": {
                "polarity": polarity,
                "score": fused_score,
                "confidence": fused_confidence
            },
            "fusion_type": "quantum_superposition",
            "quantum_states": len(all_scores)
        }
    
    def _entanglement_fusion(
        self, 
        source_results: Dict[str, Dict[str, Any]], 
        correlations: Dict[str, float]
    ) -> Dict[str, Any]:
        """Quantum entanglement-based fusion with correlation weighting."""
        # Start with superposition fusion as base
        base_fusion = self._superposition_fusion(source_results)
        
        # Apply entanglement corrections based on correlations
        if correlations:
            avg_correlation = sum(correlations.values()) / len(correlations)
            entanglement_factor = avg_correlation  # Use correlation as entanglement measure
            
            # Boost confidence if sources are highly entangled (correlated)
            base_fusion["sentiment"]["confidence"] *= (1.0 + entanglement_factor * 0.3)
            base_fusion["sentiment"]["confidence"] = min(1.0, base_fusion["sentiment"]["confidence"])
            
            base_fusion["entanglement_factor"] = entanglement_factor
        
        base_fusion["fusion_type"] = "quantum_entanglement"
        return base_fusion
    
    def _coherent_fusion(self, source_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Coherent quantum fusion maintaining phase relationships."""
        # Calculate phase coherence between sources
        coherence = self._calculate_system_coherence(source_results)
        
        # Use superposition as base, but weight by coherence
        fusion = self._superposition_fusion(source_results)
        
        # Apply coherence correction
        fusion["sentiment"]["confidence"] *= coherence
        fusion["coherence"] = coherence
        fusion["fusion_type"] = "quantum_coherent"
        
        return fusion
    
    def _calculate_system_coherence(self, source_results: Dict[str, Dict[str, Any]]) -> float:
        """Calculate overall system coherence."""
        scores = []
        for result in source_results.values():
            try:
                score = result["fused_sentiment"]["sentiment"]["score"]
                scores.append(score)
            except (KeyError, TypeError):
                continue
        
        if len(scores) < 2:
            return 1.0
        
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Coherence decreases with variance
        coherence = 1.0 / (1.0 + variance)
        return min(1.0, coherence)
    
    @Tool(
        name="quantum_temporal_sentiment", 
        description="Analyze sentiment evolution over time using quantum temporal dynamics"
    )
    async def _quantum_temporal_sentiment(
        self,
        temporal_texts: List[Dict[str, Any]],  # [{"text": str, "timestamp": datetime}, ...]
        time_decay_factor: float = 0.9,
        enable_momentum: bool = True
    ) -> Dict[str, Any]:
        """Analyze temporal sentiment evolution with quantum dynamics."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Sort by timestamp
            sorted_texts = sorted(temporal_texts, key=lambda x: x.get("timestamp", datetime.min))
            
            # Analyze each temporal slice
            temporal_results = []
            for i, item in enumerate(sorted_texts):
                result = await self._quantum_multi_approach_analysis(item["text"])
                result["temporal_index"] = i
                result["timestamp"] = item.get("timestamp", datetime.utcnow())
                temporal_results.append(result)
            
            # Calculate temporal momentum
            momentum = self._calculate_temporal_momentum(temporal_results) if enable_momentum else None
            
            # Apply temporal weighting (recent texts have more influence)
            temporal_weights = []
            for i in range(len(temporal_results)):
                weight = time_decay_factor ** (len(temporal_results) - 1 - i)
                temporal_weights.append(weight)
            
            # Weighted fusion
            weighted_sentiment = self._temporal_weighted_fusion(temporal_results, temporal_weights)
            
            # Quantum temporal metrics
            temporal_coherence = self._calculate_temporal_coherence(temporal_results)
            sentiment_volatility = self._calculate_sentiment_volatility(temporal_results)
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "temporal_sentiment": weighted_sentiment,
                "temporal_analysis": temporal_results,
                "temporal_momentum": momentum,
                "quantum_temporal_metrics": {
                    "coherence": temporal_coherence,
                    "volatility": sentiment_volatility,
                    "temporal_span": len(temporal_results),
                    "time_decay_factor": time_decay_factor
                },
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            logger.error("Quantum temporal sentiment analysis failed", error=str(e))
            raise OrchestratorError(f"Quantum temporal sentiment analysis failed: {e}")
    
    def _calculate_temporal_momentum(self, temporal_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate sentiment momentum over time."""
        if len(temporal_results) < 2:
            return {"momentum": 0.0, "acceleration": 0.0}
        
        scores = [result["fused_sentiment"]["sentiment"]["score"] for result in temporal_results]
        
        # Calculate velocity (rate of change)
        velocities = []
        for i in range(1, len(scores)):
            velocity = scores[i] - scores[i-1]
            velocities.append(velocity)
        
        # Calculate acceleration (rate of change of velocity)
        accelerations = []
        for i in range(1, len(velocities)):
            acceleration = velocities[i] - velocities[i-1]
            accelerations.append(acceleration)
        
        avg_momentum = sum(velocities) / len(velocities) if velocities else 0.0
        avg_acceleration = sum(accelerations) / len(accelerations) if accelerations else 0.0
        
        return {
            "momentum": avg_momentum,
            "acceleration": avg_acceleration,
            "velocity_trend": "increasing" if avg_momentum > 0.1 else "decreasing" if avg_momentum < -0.1 else "stable"
        }
    
    def _temporal_weighted_fusion(
        self, 
        temporal_results: List[Dict[str, Any]], 
        weights: List[float]
    ) -> Dict[str, Any]:
        """Fuse temporal results with time-based weighting."""
        total_weight = sum(weights)
        
        if total_weight == 0:
            return {"error": "No valid weights for temporal fusion"}
        
        weighted_score = 0
        weighted_confidence = 0
        
        for result, weight in zip(temporal_results, weights):
            try:
                sentiment = result["fused_sentiment"]["sentiment"]
                weighted_score += sentiment["score"] * weight
                weighted_confidence += sentiment["confidence"] * weight
            except (KeyError, TypeError):
                continue
        
        final_score = weighted_score / total_weight
        final_confidence = weighted_confidence / total_weight
        
        # Determine polarity
        if final_score > 0.1:
            polarity = "positive"
        elif final_score < -0.1:
            polarity = "negative"
        else:
            polarity = "neutral"
        
        return {
            "sentiment": {
                "polarity": polarity,
                "score": final_score,
                "confidence": final_confidence
            },
            "fusion_method": "quantum_temporal_weighted"
        }
    
    def _calculate_temporal_coherence(self, temporal_results: List[Dict[str, Any]]) -> float:
        """Calculate coherence of sentiment over time."""
        if len(temporal_results) < 2:
            return 1.0
        
        scores = []
        for result in temporal_results:
            try:
                score = result["fused_sentiment"]["sentiment"]["score"]
                scores.append(score)
            except (KeyError, TypeError):
                continue
        
        if len(scores) < 2:
            return 1.0
        
        # Calculate variance over time
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Coherence is inversely related to temporal variance
        coherence = 1.0 / (1.0 + variance)
        return min(1.0, coherence)
    
    def _calculate_sentiment_volatility(self, temporal_results: List[Dict[str, Any]]) -> float:
        """Calculate sentiment volatility (how much sentiment changes over time)."""
        if len(temporal_results) < 2:
            return 0.0
        
        scores = []
        for result in temporal_results:
            try:
                score = result["fused_sentiment"]["sentiment"]["score"]
                scores.append(score)
            except (KeyError, TypeError):
                continue
        
        if len(scores) < 2:
            return 0.0
        
        # Calculate absolute differences between consecutive scores
        differences = []
        for i in range(1, len(scores)):
            diff = abs(scores[i] - scores[i-1])
            differences.append(diff)
        
        # Average volatility
        return sum(differences) / len(differences) if differences else 0.0


# Factory function for easy instantiation
def create_quantum_sentiment_analyzer(
    max_parallel: int = 20,
    enable_speculation: bool = True,
    security_level: SecurityLevel = SecurityLevel.HIGH
) -> QuantumSentimentAnalyzer:
    """Create a quantum sentiment analyzer with default settings."""
    config = QuantumSentimentConfig(
        max_parallel_analyses=max_parallel,
        enable_speculation=enable_speculation,
        security_level=security_level
    )
    return QuantumSentimentAnalyzer(config=config)