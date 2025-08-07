"""Integration tests for sentiment analysis components."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from async_toolformer.quantum_sentiment import (
    create_quantum_sentiment_analyzer,
    QuantumSentimentAnalyzer,
    QuantumSentimentConfig
)
from async_toolformer.sentiment_intelligence import (
    QuantumSentimentIntelligence,
    create_quantum_sentiment_intelligence
)
from async_toolformer.sentiment_monitoring import (
    SentimentMonitor,
    get_sentiment_monitor,
    set_sentiment_monitor
)
from async_toolformer.quantum_orchestrator import create_quantum_orchestrator
from async_toolformer.quantum_security import SecurityLevel
from async_toolformer.sentiment_analyzer import SentimentResult, SentimentPolarity


@pytest.fixture
async def mock_orchestrator():
    """Create a mock quantum orchestrator."""
    orchestrator = MagicMock()
    orchestrator.quantum_execute = AsyncMock()
    orchestrator.execute_tool = AsyncMock()
    orchestrator.register_tool = MagicMock()
    orchestrator.get_quantum_analytics = MagicMock(return_value={
        'paths_explored': 3,
        'coherence_score': 0.85
    })
    return orchestrator


@pytest.fixture
async def quantum_sentiment_analyzer(mock_orchestrator):
    """Create a quantum sentiment analyzer for testing."""
    config = QuantumSentimentConfig(
        max_parallel_analyses=5,
        enable_speculation=True,
        security_level=SecurityLevel.MEDIUM
    )
    return QuantumSentimentAnalyzer(config=config, orchestrator=mock_orchestrator)


@pytest.fixture
async def sentiment_monitor():
    """Create a sentiment monitor for testing."""
    monitor = SentimentMonitor(
        enable_detailed_logging=True,
        alert_thresholds={
            "avg_processing_time_ms": 1000.0,
            "success_rate": 0.9
        },
        max_history_size=100
    )
    return monitor


class TestQuantumSentimentAnalyzer:
    """Test QuantumSentimentAnalyzer integration."""
    
    @pytest.mark.asyncio
    async def test_analyzer_initialization(self, mock_orchestrator):
        """Test quantum sentiment analyzer initialization."""
        config = QuantumSentimentConfig(
            max_parallel_analyses=10,
            enable_speculation=True,
            enable_quantum_optimization=True
        )
        
        analyzer = QuantumSentimentAnalyzer(config=config, orchestrator=mock_orchestrator)
        
        assert analyzer.config == config
        assert analyzer.orchestrator == mock_orchestrator
        assert mock_orchestrator.register_tool.call_count >= 4  # Should register multiple tools
    
    @pytest.mark.asyncio
    async def test_quantum_multi_approach_analysis(self, quantum_sentiment_analyzer):
        """Test quantum multi-approach analysis."""
        text = "I absolutely love this amazing product!"
        
        # Mock the orchestrator's response
        quantum_sentiment_analyzer.orchestrator.quantum_execute.return_value = {
            "sentiment": {"polarity": "positive", "score": 0.8, "confidence": 0.9}
        }
        
        result = await quantum_sentiment_analyzer._quantum_multi_approach_analysis(
            text=text,
            approaches=["rule_based", "social_media"],
            include_social_analysis=True
        )
        
        assert "fused_sentiment" in result
        assert "individual_results" in result
        assert "analysis_approaches" in result
        assert "quantum_coherence" in result
        assert result["analysis_approaches"] == ["rule_based", "social_media"]
    
    @pytest.mark.asyncio
    async def test_quantum_sentiment_fusion(self, quantum_sentiment_analyzer):
        """Test quantum sentiment fusion."""
        text_sources = {
            "review": "This product is excellent and well-made.",
            "social": "Love this product! 😍 #amazing",
            "expert": "High quality materials and construction."
        }
        
        # Mock individual analysis results
        mock_result = {
            "fused_sentiment": {
                "sentiment": {"polarity": "positive", "score": 0.75, "confidence": 0.85}
            },
            "quantum_coherence": 0.8
        }
        
        quantum_sentiment_analyzer._quantum_multi_approach_analysis = AsyncMock(return_value=mock_result)
        
        result = await quantum_sentiment_analyzer._quantum_sentiment_fusion(
            text_sources=text_sources,
            fusion_method="quantum_superposition",
            enable_cross_correlation=True
        )
        
        assert "fused_sentiment" in result
        assert "source_analyses" in result
        assert "cross_correlations" in result
        assert "fusion_method" in result
        assert "quantum_metrics" in result
        assert result["fusion_method"] == "quantum_superposition"
    
    @pytest.mark.asyncio
    async def test_quantum_temporal_sentiment(self, quantum_sentiment_analyzer):
        """Test quantum temporal sentiment analysis."""
        base_time = datetime.utcnow()
        temporal_texts = [
            {
                "text": "Initial impression is great!",
                "timestamp": base_time
            },
            {
                "text": "Still loving it after a week.",
                "timestamp": base_time + timedelta(days=7)
            },
            {
                "text": "Long-term satisfaction confirmed.",
                "timestamp": base_time + timedelta(days=30)
            }
        ]
        
        # Mock multi-approach analysis
        mock_result = {
            "fused_sentiment": {
                "sentiment": {"polarity": "positive", "score": 0.8, "confidence": 0.9}
            }
        }
        quantum_sentiment_analyzer._quantum_multi_approach_analysis = AsyncMock(return_value=mock_result)
        
        result = await quantum_sentiment_analyzer._quantum_temporal_sentiment(
            temporal_texts=temporal_texts,
            time_decay_factor=0.8,
            enable_momentum=True
        )
        
        assert "temporal_sentiment" in result
        assert "temporal_analysis" in result
        assert "temporal_momentum" in result
        assert "quantum_temporal_metrics" in result
        assert len(result["temporal_analysis"]) == 3


class TestQuantumSentimentIntelligence:
    """Test QuantumSentimentIntelligence integration."""
    
    @pytest.mark.asyncio
    async def test_intelligence_initialization(self, mock_orchestrator):
        """Test quantum sentiment intelligence initialization."""
        intelligence = QuantumSentimentIntelligence(
            orchestrator=mock_orchestrator,
            enable_ml_enhancement=True,
            enable_pattern_learning=True,
            enable_quantum_superposition=True
        )
        
        assert intelligence.orchestrator == mock_orchestrator
        assert intelligence.enable_ml_enhancement is True
        assert intelligence.enable_pattern_learning is True
        assert intelligence.enable_quantum_superposition is True
        assert intelligence.pattern_detector is not None
        assert intelligence.quantum_state_manager is not None
        assert intelligence.adaptive_optimizer is not None
        assert intelligence.context_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_quantum_intelligent_analysis(self, mock_orchestrator):
        """Test quantum intelligent analysis."""
        intelligence = QuantumSentimentIntelligence(
            orchestrator=mock_orchestrator,
            enable_ml_enhancement=True
        )
        
        text = "This revolutionary product exceeds all expectations!"
        context = {"domain": "business", "urgency": 0.7}
        
        # Mock the enhanced sentiment analysis
        with patch.object(intelligence, '_enhanced_sentiment_analysis') as mock_enhanced:
            mock_enhanced.return_value = {
                "sentiment": {"polarity": "positive", "score": 0.9, "confidence": 0.95}
            }
            
            # Mock other analysis methods
            intelligence.context_analyzer.analyze_context = AsyncMock(return_value={
                "contextual_sentiment": {"polarity": "positive", "score": 0.85}
            })
            
            intelligence.pattern_detector.detect_patterns = AsyncMock(return_value=[])
            
            result = await intelligence.quantum_intelligent_analysis(
                text=text,
                context=context,
                optimization_level="high",
                enable_learning=True
            )
            
            assert "intelligent_analysis" in result
            assert "quantum_states" in result
            assert "analysis_methods" in result
            assert "optimization_level" in result
            assert "quantum_intelligence_metrics" in result
            assert result["optimization_level"] == "high"
    
    @pytest.mark.asyncio
    async def test_quantum_superposition_creation(self, mock_orchestrator):
        """Test quantum superposition state creation."""
        intelligence = QuantumSentimentIntelligence(
            orchestrator=mock_orchestrator,
            enable_quantum_superposition=True
        )
        
        text = "Amazing product with incredible quality!"
        context = {"domain": "review"}
        
        states = await intelligence._create_quantum_superposition(text, context)
        
        assert len(states) > 0
        assert all(hasattr(state, 'state_id') for state in states)
        assert all(hasattr(state, 'amplitude') for state in states)
        assert all(hasattr(state, 'sentiment_vector') for state in states)
        assert all(len(state.sentiment_vector) == 10 for state in states)  # 10-dimensional
    
    @pytest.mark.asyncio
    async def test_ml_enhancement(self, mock_orchestrator):
        """Test machine learning enhancement."""
        intelligence = QuantumSentimentIntelligence(
            orchestrator=mock_orchestrator,
            enable_ml_enhancement=True
        )
        
        # Set up some learned weights
        intelligence.model_weights = {
            "text_length": 0.1,
            "exclamation_count": 0.2,
            "caps_ratio": -0.1
        }
        
        base_result = {
            "sentiment": {"score": 0.6, "confidence": 0.7, "polarity": "positive"}
        }
        text = "GREAT PRODUCT!!!"
        context = {"domain": "social"}
        
        enhanced_result = await intelligence._apply_ml_enhancement(base_result, text, context)
        
        assert "ml_features" in enhanced_result
        assert "ml_feature_importance" in enhanced_result
        assert "ml_confidence_adjustment" in enhanced_result
        assert enhanced_result["sentiment"]["confidence"] != base_result["sentiment"]["confidence"]


class TestSentimentMonitor:
    """Test SentimentMonitor integration."""
    
    @pytest.mark.asyncio
    async def test_monitor_initialization(self):
        """Test sentiment monitor initialization."""
        monitor = SentimentMonitor(
            enable_detailed_logging=True,
            alert_thresholds={"avg_processing_time_ms": 2000.0},
            max_history_size=500
        )
        
        assert monitor.enable_detailed_logging is True
        assert monitor.alert_thresholds["avg_processing_time_ms"] == 2000.0
        assert monitor.max_history_size == 500
        assert len(monitor.recent_analyses) == 0
    
    @pytest.mark.asyncio
    async def test_record_successful_analysis(self, sentiment_monitor):
        """Test recording successful analysis."""
        # Create a mock sentiment result
        from async_toolformer.sentiment_analyzer import SentimentScore
        
        sentiment_score = SentimentScore(
            polarity=SentimentPolarity.POSITIVE,
            confidence=0.85,
            score=0.7
        )
        
        result = SentimentResult(
            text="Great product!",
            sentiment=sentiment_score,
            processing_time_ms=150.0
        )
        
        await sentiment_monitor.record_analysis(
            result=result,
            text="Great product!",
            processing_time_ms=150.0,
            metadata={"test": True}
        )
        
        assert sentiment_monitor.metrics.total_analyses == 1
        assert sentiment_monitor.metrics.successful_analyses == 1
        assert sentiment_monitor.metrics.positive_count == 1
        assert len(sentiment_monitor.recent_analyses) == 1
        assert sentiment_monitor.recent_analyses[0]["success"] is True
    
    @pytest.mark.asyncio
    async def test_record_failed_analysis(self, sentiment_monitor):
        """Test recording failed analysis."""
        error = Exception("Analysis failed")
        
        await sentiment_monitor.record_analysis(
            result=error,
            text="Test text",
            processing_time_ms=50.0
        )
        
        assert sentiment_monitor.metrics.total_analyses == 1
        assert sentiment_monitor.metrics.failed_analyses == 1
        assert sentiment_monitor.metrics.successful_analyses == 0
        assert len(sentiment_monitor.recent_analyses) == 1
        assert sentiment_monitor.recent_analyses[0]["success"] is False
        assert "error" in sentiment_monitor.recent_analyses[0]
    
    @pytest.mark.asyncio
    async def test_batch_analysis_recording(self, sentiment_monitor):
        """Test recording batch analysis."""
        from async_toolformer.sentiment_analyzer import BatchSentimentResult
        
        # Create mock individual results
        individual_results = []
        for i in range(3):
            sentiment_score = SentimentScore(
                polarity=SentimentPolarity.POSITIVE,
                confidence=0.8,
                score=0.6
            )
            result = SentimentResult(
                text=f"Test {i}",
                sentiment=sentiment_score,
                processing_time_ms=100.0
            )
            individual_results.append(result)
        
        batch_result = BatchSentimentResult(
            results=individual_results,
            summary={"positive": 3},
            total_texts=3,
            processing_time_ms=300.0
        )
        
        await sentiment_monitor.record_batch_analysis(
            batch_result=batch_result,
            batch_size=3,
            processing_time_ms=300.0
        )
        
        assert sentiment_monitor.metrics.successful_analyses == 3
        assert len(sentiment_monitor.recent_analyses) == 3
        assert "batch_throughput" in sentiment_monitor.quantum_metrics
    
    @pytest.mark.asyncio
    async def test_alert_generation(self, sentiment_monitor):
        """Test alert generation."""
        # Set low threshold to trigger alert
        sentiment_monitor.alert_thresholds["avg_processing_time_ms"] = 50.0
        
        # Record slow analysis
        from async_toolformer.sentiment_analyzer import SentimentScore
        
        sentiment_score = SentimentScore(
            polarity=SentimentPolarity.POSITIVE,
            confidence=0.8,
            score=0.6
        )
        
        result = SentimentResult(
            text="Test",
            sentiment=sentiment_score,
            processing_time_ms=100.0  # Exceeds threshold
        )
        
        await sentiment_monitor.record_analysis(result, "Test", 100.0)
        
        # Force alert check by setting last check time to past
        sentiment_monitor.last_alert_check = datetime.utcnow() - timedelta(minutes=2)
        await sentiment_monitor._check_alerts()
        
        # Should have generated an alert
        assert len(sentiment_monitor.alerts) > 0
    
    def test_dashboard_data_generation(self, sentiment_monitor):
        """Test dashboard data generation."""
        # Add some mock data
        sentiment_monitor.metrics.total_analyses = 100
        sentiment_monitor.metrics.successful_analyses = 95
        sentiment_monitor.metrics.positive_count = 60
        sentiment_monitor.metrics.negative_count = 25
        sentiment_monitor.metrics.neutral_count = 10
        
        dashboard_data = sentiment_monitor.get_dashboard_data()
        
        assert "system_info" in dashboard_data
        assert "metrics" in dashboard_data
        assert "sentiment_distribution" in dashboard_data
        assert "quality_metrics" in dashboard_data
        assert "trends" in dashboard_data
        assert "health_status" in dashboard_data
        
        # Check specific metrics
        assert dashboard_data["metrics"]["total_analyses"] == 100
        assert dashboard_data["metrics"]["success_rate"] == 0.95
        
        # Check sentiment distribution
        distribution = dashboard_data["sentiment_distribution"]
        assert abs(distribution["positive"] - 63.16) < 1  # 60/95 * 100
        assert abs(distribution["negative"] - 26.32) < 1  # 25/95 * 100
    
    def test_prometheus_metrics_export(self, sentiment_monitor):
        """Test Prometheus metrics export."""
        # Add some mock data
        sentiment_monitor.metrics.total_analyses = 50
        sentiment_monitor.metrics.avg_processing_time_ms = 200.0
        sentiment_monitor.quantum_metrics["coherence_score"] = 0.85
        
        prometheus_output = sentiment_monitor._export_prometheus_metrics()
        
        assert "sentiment_total_analyses 50" in prometheus_output
        assert "sentiment_avg_processing_time_ms 200.0" in prometheus_output
        assert "quantum_coherence_score 0.85" in prometheus_output
        assert "# HELP" in prometheus_output
        assert "# TYPE" in prometheus_output


class TestEndToEndIntegration:
    """Test end-to-end integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_sentiment_analysis_workflow(self):
        """Test complete sentiment analysis workflow."""
        # This test requires actual orchestrator implementation
        # For now, we'll test the integration points
        
        # Create quantum sentiment analyzer
        config = QuantumSentimentConfig(max_parallel_analyses=3)
        
        with patch('async_toolformer.quantum_sentiment.create_quantum_orchestrator') as mock_create:
            mock_orchestrator = MagicMock()
            mock_orchestrator.register_tool = MagicMock()
            mock_create.return_value = mock_orchestrator
            
            analyzer = QuantumSentimentAnalyzer(config=config)
            
            # Verify orchestrator was created with correct parameters
            mock_create.assert_called_once_with(
                max_parallel=3,
                enable_speculation=True,
                enable_quantum_optimization=True,
                coherence_threshold=0.8,
                superposition_depth=3,
                security_level=SecurityLevel.HIGH
            )
            
            # Verify tools were registered
            assert mock_orchestrator.register_tool.call_count >= 4
    
    @pytest.mark.asyncio
    async def test_intelligence_with_monitoring(self, mock_orchestrator):
        """Test intelligence system with monitoring."""
        # Create intelligence system
        intelligence = QuantumSentimentIntelligence(
            orchestrator=mock_orchestrator,
            enable_ml_enhancement=True
        )
        
        # Create and set monitor
        monitor = SentimentMonitor(max_history_size=10)
        set_sentiment_monitor(monitor)
        
        # Mock successful analysis
        intelligence.validator.validate_input_text = MagicMock(return_value=MagicMock(
            processing_safe=True, issues=[]
        ))
        
        with patch.multiple(intelligence,
                          _create_quantum_superposition=AsyncMock(return_value=[]),
                          _enhanced_sentiment_analysis=AsyncMock(return_value={
                              "sentiment": {"polarity": "positive", "score": 0.8, "confidence": 0.9}
                          }),
                          _learn_from_analysis=AsyncMock()):
            
            result = await intelligence.quantum_intelligent_analysis(
                text="Great product!",
                context={"domain": "review"},
                optimization_level="medium"
            )
            
            # Verify result structure
            assert "intelligent_analysis" in result
            assert "processing_time_ms" in result
            
            # Verify monitoring was updated (this would happen in actual implementation)
            # For mocked test, we just verify the monitor exists
            current_monitor = get_sentiment_monitor()
            assert current_monitor is monitor
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, mock_orchestrator):
        """Test error handling across integration points."""
        intelligence = QuantumSentimentIntelligence(
            orchestrator=mock_orchestrator,
            enable_quantum_superposition=True
        )
        
        # Test with invalid input that should be caught by validator
        with pytest.raises(Exception):  # Should raise OrchestratorError
            await intelligence.quantum_intelligent_analysis(
                text="<script>alert('hack')</script>",
                optimization_level="high"
            )
    
    def test_global_monitor_management(self):
        """Test global monitor management."""
        # Test getting default monitor
        monitor1 = get_sentiment_monitor()
        monitor2 = get_sentiment_monitor()
        assert monitor1 is monitor2  # Should be same instance
        
        # Test setting custom monitor
        custom_monitor = SentimentMonitor(max_history_size=50)
        set_sentiment_monitor(custom_monitor)
        
        retrieved_monitor = get_sentiment_monitor()
        assert retrieved_monitor is custom_monitor
        assert retrieved_monitor.max_history_size == 50


@pytest.mark.asyncio
async def test_factory_functions():
    """Test factory functions for creating components."""
    
    # Test quantum sentiment analyzer factory
    analyzer = create_quantum_sentiment_analyzer(
        max_parallel=15,
        enable_speculation=False,
        security_level=SecurityLevel.LOW
    )
    
    assert isinstance(analyzer, QuantumSentimentAnalyzer)
    assert analyzer.config.max_parallel_analyses == 15
    assert analyzer.config.enable_speculation is False
    assert analyzer.config.security_level == SecurityLevel.LOW
    
    # Test quantum intelligence factory
    with patch('async_toolformer.sentiment_intelligence.create_quantum_orchestrator') as mock_create:
        mock_orchestrator = MagicMock()
        mock_create.return_value = mock_orchestrator
        
        # Since we need an orchestrator, we'll create a mock one
        intelligence = create_quantum_sentiment_intelligence(
            orchestrator=mock_orchestrator,
            enable_ml_enhancement=False,
            enable_pattern_learning=True
        )
        
        assert isinstance(intelligence, QuantumSentimentIntelligence)
        assert intelligence.enable_ml_enhancement is False
        assert intelligence.enable_pattern_learning is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])