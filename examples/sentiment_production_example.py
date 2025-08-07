"""Production-ready sentiment analysis example with full feature demonstration."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any

from async_toolformer.quantum_sentiment import create_quantum_sentiment_analyzer, QuantumSentimentConfig
from async_toolformer.sentiment_intelligence import create_quantum_sentiment_intelligence
from async_toolformer.sentiment_monitoring import SentimentMonitor, set_sentiment_monitor
from async_toolformer.sentiment_validation import SentimentValidator, SentimentValidationConfig
from async_toolformer.sentiment_globalization import global_sentiment_analysis, get_supported_languages
from async_toolformer.quantum_orchestrator import create_quantum_orchestrator
from async_toolformer.quantum_security import SecurityLevel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ProductionSentimentService:
    """Production-ready sentiment analysis service."""
    
    def __init__(self):
        """Initialize production service."""
        self.setup_monitoring()
        self.setup_validation()
        self.setup_quantum_analyzer()
        self.setup_intelligence_system()
        
    def setup_monitoring(self):
        """Setup comprehensive monitoring."""
        self.monitor = SentimentMonitor(
            enable_detailed_logging=True,
            alert_thresholds={
                "avg_processing_time_ms": 3000.0,  # 3 second alert threshold
                "success_rate": 0.95,  # 95% success rate threshold
                "low_confidence_rate": 0.25,  # 25% low confidence threshold
                "throughput_degradation": 0.4,  # 40% throughput degradation
            },
            max_history_size=50000  # Large history for production
        )
        set_sentiment_monitor(self.monitor)
        logger.info("🔍 Monitoring system initialized")
    
    def setup_validation(self):
        """Setup input validation."""
        validation_config = SentimentValidationConfig(
            min_text_length=2,  # Allow very short texts
            max_text_length=50000,  # Large texts for enterprise
            min_confidence_threshold=0.2,  # Lower threshold for production
            max_batch_size=5000,  # Large batches
            allowed_languages=get_supported_languages(),
            block_suspicious_patterns=True,
            require_ascii_printable=False,  # Support unicode
            enable_toxicity_detection=True,
            enable_spam_detection=True
        )
        self.validator = SentimentValidator(validation_config)
        logger.info("🛡️  Validation system initialized")
    
    def setup_quantum_analyzer(self):
        """Setup quantum-enhanced analyzer."""
        config = QuantumSentimentConfig(
            max_parallel_analyses=50,  # High parallelism for production
            enable_speculation=True,
            coherence_threshold=0.7,  # Slightly lower for better throughput
            superposition_depth=4,  # Deeper analysis
            security_level=SecurityLevel.HIGH,
            enable_quantum_optimization=True,
            enable_multi_source_fusion=True,
            social_media_boost=True
        )
        
        self.quantum_analyzer = create_quantum_sentiment_analyzer(
            max_parallel=config.max_parallel_analyses,
            enable_speculation=config.enable_speculation,
            security_level=config.security_level
        )
        logger.info("⚡ Quantum analyzer initialized")
    
    def setup_intelligence_system(self):
        """Setup AI intelligence system."""
        orchestrator = create_quantum_orchestrator(
            max_parallel=100,  # Very high for intelligence system
            enable_speculation=True,
            enable_quantum_optimization=True,
            coherence_threshold=0.8,
            security_level=SecurityLevel.HIGH
        )
        
        self.intelligence = create_quantum_sentiment_intelligence(
            orchestrator=orchestrator,
            enable_ml_enhancement=True,
            enable_pattern_learning=True,
            enable_quantum_superposition=True
        )
        logger.info("🧠 Intelligence system initialized")
    
    async def analyze_single_text(
        self, 
        text: str, 
        language: str = None,
        context: Dict[str, Any] = None,
        compliance_region: str = "global"
    ) -> Dict[str, Any]:
        """Analyze single text with full production pipeline."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Input validation
            validation = self.validator.validate_input_text(text)
            if not validation.processing_safe:
                raise ValueError(f"Input validation failed: {validation.issues[0].message}")
            
            # Global analysis with compliance
            global_result = await global_sentiment_analysis(
                text=text,
                language=language,
                compliance_region=compliance_region,
                user_consent=True,  # Assume consent in production
                cultural_context=context
            )
            
            # Intelligence-enhanced analysis
            intelligent_result = await self.intelligence.quantum_intelligent_analysis(
                text=text,
                context=context,
                optimization_level="high",
                enable_learning=True
            )
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Combine results
            combined_result = {
                "global_analysis": global_result,
                "intelligent_analysis": intelligent_result,
                "validation": {
                    "confidence_score": validation.confidence_score,
                    "issues": len(validation.issues),
                    "warnings": len(validation.warnings)
                },
                "performance": {
                    "total_processing_time_ms": processing_time,
                    "quantum_optimization_used": True,
                    "intelligence_enhancement_used": True
                },
                "metadata": {
                    "service_version": "production-1.0.0",
                    "timestamp": datetime.utcnow(),
                    "request_id": f"req_{hash(text) % 10000}"
                }
            }
            
            # Record for monitoring
            await self.monitor.record_analysis(
                result=combined_result,
                text=text,
                processing_time_ms=processing_time,
                metadata={"production": True, "enhanced": True}
            )
            
            return combined_result
            
        except Exception as e:
            await self.monitor.record_analysis(
                result=e,
                text=text,
                processing_time_ms=0,
                metadata={"production": True, "error": True}
            )
            raise
    
    async def analyze_batch(
        self, 
        texts: List[str],
        language: str = None,
        context: Dict[str, Any] = None,
        max_parallel: int = 25
    ) -> Dict[str, Any]:
        """Analyze batch of texts with production optimizations."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Batch validation
            validation, invalid_indices = self.validator.validate_batch_input(texts)
            
            # Filter out invalid texts
            valid_texts = [text for i, text in enumerate(texts) if i not in invalid_indices]
            
            if not valid_texts:
                raise ValueError("No valid texts to analyze")
            
            # Create semaphore for controlled concurrency
            semaphore = asyncio.Semaphore(max_parallel)
            
            async def analyze_single(text: str, index: int) -> Dict[str, Any]:
                async with semaphore:
                    try:
                        result = await self.analyze_single_text(
                            text=text,
                            language=language,
                            context=context
                        )
                        return {"index": index, "success": True, "result": result}
                    except Exception as e:
                        return {"index": index, "success": False, "error": str(e)}
            
            # Process all texts in parallel
            tasks = [analyze_single(text, i) for i, text in enumerate(valid_texts)]
            results = await asyncio.gather(*tasks)
            
            # Separate successful and failed results
            successful_results = [r for r in results if r["success"]]
            failed_results = [r for r in results if not r["success"]]
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            batch_summary = {
                "batch_info": {
                    "total_texts": len(texts),
                    "valid_texts": len(valid_texts),
                    "invalid_texts": len(invalid_indices),
                    "successful_analyses": len(successful_results),
                    "failed_analyses": len(failed_results)
                },
                "results": successful_results,
                "failures": failed_results,
                "performance": {
                    "total_processing_time_ms": processing_time,
                    "average_processing_time_ms": processing_time / len(valid_texts) if valid_texts else 0,
                    "throughput_per_second": len(valid_texts) / (processing_time / 1000) if processing_time > 0 else 0,
                    "parallel_processing_used": True
                },
                "validation_summary": {
                    "validation_confidence": validation.confidence_score,
                    "invalid_indices": invalid_indices,
                    "validation_issues": len(validation.issues)
                },
                "metadata": {
                    "service_version": "production-1.0.0",
                    "timestamp": datetime.utcnow(),
                    "batch_id": f"batch_{hash(str(texts)) % 10000}"
                }
            }
            
            # Record batch metrics
            await self.monitor.record_batch_analysis(
                batch_result=batch_summary,
                batch_size=len(texts),
                processing_time_ms=processing_time,
                metadata={"production": True, "batch": True}
            )
            
            return batch_summary
            
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            raise
    
    async def analyze_temporal_sentiment(
        self,
        temporal_data: List[Dict[str, Any]],
        time_decay_factor: float = 0.8
    ) -> Dict[str, Any]:
        """Analyze sentiment evolution over time."""
        try:
            result = await self.quantum_analyzer._quantum_temporal_sentiment(
                temporal_texts=temporal_data,
                time_decay_factor=time_decay_factor,
                enable_momentum=True
            )
            
            return {
                "temporal_analysis": result,
                "metadata": {
                    "analysis_type": "temporal",
                    "data_points": len(temporal_data),
                    "time_span_days": self._calculate_time_span(temporal_data),
                    "timestamp": datetime.utcnow()
                }
            }
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            raise
    
    async def multi_source_fusion_analysis(
        self,
        sources: Dict[str, str],
        fusion_method: str = "quantum_entanglement"
    ) -> Dict[str, Any]:
        """Analyze sentiment from multiple sources with fusion."""
        try:
            result = await self.quantum_analyzer._quantum_sentiment_fusion(
                text_sources=sources,
                fusion_method=fusion_method,
                enable_cross_correlation=True
            )
            
            return {
                "fusion_analysis": result,
                "metadata": {
                    "analysis_type": "multi_source_fusion",
                    "source_count": len(sources),
                    "fusion_method": fusion_method,
                    "timestamp": datetime.utcnow()
                }
            }
            
        except Exception as e:
            logger.error(f"Multi-source fusion analysis failed: {e}")
            raise
    
    def _calculate_time_span(self, temporal_data: List[Dict[str, Any]]) -> float:
        """Calculate time span in days for temporal data."""
        if len(temporal_data) < 2:
            return 0.0
        
        timestamps = [item.get("timestamp", datetime.utcnow()) for item in temporal_data]
        timestamps.sort()
        
        time_span = timestamps[-1] - timestamps[0]
        return time_span.total_seconds() / (24 * 3600)  # Convert to days
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive service health status."""
        dashboard_data = self.monitor.get_dashboard_data()
        
        # Get quantum analytics
        quantum_analytics = self.quantum_analyzer.orchestrator.get_quantum_analytics()
        
        return {
            "service_status": {
                "status": dashboard_data["health_status"],
                "uptime_seconds": dashboard_data["system_info"]["uptime_seconds"],
                "version": "production-1.0.0"
            },
            "performance_metrics": dashboard_data["metrics"],
            "sentiment_distribution": dashboard_data["sentiment_distribution"],
            "quality_metrics": dashboard_data["quality_metrics"],
            "quantum_metrics": {
                "paths_explored": quantum_analytics.get("paths_explored", 0),
                "coherence_score": quantum_analytics.get("coherence_score", 0.0),
                "optimization_active": True
            },
            "recent_alerts": dashboard_data["recent_alerts"],
            "supported_features": {
                "languages": len(get_supported_languages()),
                "compliance_regions": ["eu", "us", "ca", "uk", "sg", "br", "au", "jp", "kr", "global"],
                "quantum_enhancement": True,
                "intelligence_system": True,
                "real_time_monitoring": True,
                "batch_processing": True,
                "temporal_analysis": True,
                "multi_source_fusion": True
            }
        }
    
    async def export_metrics(self, format: str = "prometheus") -> str:
        """Export metrics in specified format."""
        return await self.monitor.export_metrics(format)


async def production_demo():
    """Demonstrate production sentiment analysis service."""
    print("🚀 Initializing Production Sentiment Analysis Service")
    print("=" * 70)
    
    # Initialize service
    service = ProductionSentimentService()
    
    print("✅ Service initialized successfully!")
    print("\n📊 Service Health Check")
    print("-" * 30)
    
    # Health check
    health = await service.get_service_health()
    print(f"Status: {health['service_status']['status'].upper()}")
    print(f"Uptime: {health['service_status']['uptime_seconds']:.1f} seconds")
    print(f"Supported Languages: {health['supported_features']['languages']}")
    print(f"Quantum Enhancement: {'✅' if health['supported_features']['quantum_enhancement'] else '❌'}")
    
    print("\n🔬 Single Text Analysis")
    print("-" * 30)
    
    # Single text analysis
    single_result = await service.analyze_single_text(
        text="This revolutionary quantum-enhanced product absolutely exceeds all my expectations! Amazing innovation! 🚀",
        language="en",
        context={"domain": "technology", "urgency": 0.8},
        compliance_region="global"
    )
    
    global_sentiment = single_result["global_analysis"]["sentiment_analysis"]
    print(f"Text: {global_sentiment['text'][:50]}...")
    print(f"Sentiment: {global_sentiment['sentiment']['polarity']} (score: {global_sentiment['sentiment']['score']:.2f})")
    print(f"Confidence: {global_sentiment['sentiment']['confidence']:.2f}")
    print(f"Language Detected: {single_result['global_analysis']['language_detected']}")
    print(f"Processing Time: {single_result['performance']['total_processing_time_ms']:.1f}ms")
    
    print("\n📊 Batch Analysis")
    print("-" * 30)
    
    # Batch analysis
    batch_texts = [
        "I absolutely love this product! Best purchase ever!",
        "Terrible quality, waste of money. Very disappointed.",
        "It's okay, nothing special but does the job.",
        "¡Increíble! Me encanta este producto fantástico! 😍",
        "C'est un produit magnifique, je le recommande vivement!",
        "このプロダクトは素晴らしいです！とても気に入りました。",
        "Dieses Produkt ist ausgezeichnet! Sehr zufrieden.",
        "Amazing features and excellent customer service!",
        "Not worth the price, poor build quality.",
        "Perfect for my needs, highly recommend!"
    ]
    
    batch_result = await service.analyze_batch(
        texts=batch_texts,
        language=None,  # Auto-detect
        max_parallel=5
    )
    
    print(f"Total Texts: {batch_result['batch_info']['total_texts']}")
    print(f"Successful Analyses: {batch_result['batch_info']['successful_analyses']}")
    print(f"Failed Analyses: {batch_result['batch_info']['failed_analyses']}")
    print(f"Average Processing Time: {batch_result['performance']['average_processing_time_ms']:.1f}ms")
    print(f"Throughput: {batch_result['performance']['throughput_per_second']:.1f} texts/second")
    
    print("\n⏰ Temporal Analysis")
    print("-" * 30)
    
    # Temporal analysis
    base_time = datetime.utcnow()
    temporal_data = [
        {
            "text": "First impressions are amazing! Love the new features.",
            "timestamp": base_time - timedelta(days=30)
        },
        {
            "text": "After a week of use, still very satisfied with performance.",
            "timestamp": base_time - timedelta(days=23)
        },
        {
            "text": "Some minor issues appeared, but overall still positive.",
            "timestamp": base_time - timedelta(days=15)
        },
        {
            "text": "Updates fixed the issues. Back to loving it!",
            "timestamp": base_time - timedelta(days=7)
        },
        {
            "text": "Long-term review: Excellent product, highly recommended.",
            "timestamp": base_time
        }
    ]
    
    temporal_result = await service.analyze_temporal_sentiment(
        temporal_data=temporal_data,
        time_decay_factor=0.8
    )
    
    temporal_sentiment = temporal_result["temporal_analysis"]["temporal_sentiment"]["sentiment"]
    temporal_momentum = temporal_result["temporal_analysis"]["temporal_momentum"]
    
    print(f"Temporal Sentiment: {temporal_sentiment['polarity']} (score: {temporal_sentiment['score']:.2f})")
    print(f"Time Span: {temporal_result['metadata']['time_span_days']:.1f} days")
    print(f"Momentum: {temporal_momentum['velocity_trend']} (value: {temporal_momentum['momentum']:.3f})")
    
    print("\n🔗 Multi-Source Fusion")
    print("-" * 30)
    
    # Multi-source fusion
    sources = {
        "product_review": "Comprehensive review: This product delivers exceptional value with innovative features and reliable performance.",
        "social_media": "OMG! This product is AMAZING! 🤩 Best purchase this year! #love #recommended #musthave",
        "customer_support": "Customer reported high satisfaction with product quality and our responsive support team.",
        "expert_opinion": "From a technical standpoint, this product represents a significant advancement in the field.",
        "user_forum": "Been using this for 3 months. Solid build quality, great features, worth every penny."
    }
    
    fusion_result = await service.multi_source_fusion_analysis(
        sources=sources,
        fusion_method="quantum_entanglement"
    )
    
    fused_sentiment = fusion_result["fusion_analysis"]["fused_sentiment"]["sentiment"]
    quantum_metrics = fusion_result["fusion_analysis"]["quantum_metrics"]
    
    print(f"Fused Sentiment: {fused_sentiment['polarity']} (score: {fused_sentiment['score']:.2f})")
    print(f"System Coherence: {quantum_metrics['coherence']:.2f}")
    print(f"Source Count: {fusion_result['metadata']['source_count']}")
    print(f"Entanglement Strength: {quantum_metrics['entanglement_strength']}")
    
    print("\n📈 Final Health Check")
    print("-" * 30)
    
    # Final health check after processing
    final_health = await service.get_service_health()
    metrics = final_health["performance_metrics"]
    
    print(f"Total Analyses: {metrics['total_analyses']}")
    print(f"Success Rate: {metrics['success_rate']:.1%}")
    print(f"Average Processing Time: {metrics['avg_processing_time_ms']:.1f}ms")
    print(f"Peak Throughput: {metrics['peak_throughput']:.1f} analyses/second")
    
    # Export metrics
    print("\n📊 Exporting Prometheus Metrics")
    print("-" * 30)
    
    prometheus_metrics = await service.export_metrics("prometheus")
    print(f"Exported {len(prometheus_metrics.split('\\n'))} metric lines")
    print("First few metrics:")
    for line in prometheus_metrics.split('\n')[:5]:
        if line.strip() and not line.startswith('#'):
            print(f"  {line}")
    
    print("\n🎉 Production Demo Completed Successfully!")
    print("=" * 70)
    print("🚀 Quantum-Enhanced Sentiment Analysis is ready for production use!")


if __name__ == "__main__":
    # Set up event loop optimization
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        print("🔧 Using uvloop for enhanced performance")
    except ImportError:
        print("🔧 Using standard asyncio (uvloop not available)")
    
    # Run the production demo
    asyncio.run(production_demo())