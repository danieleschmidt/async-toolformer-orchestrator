#!/usr/bin/env python3
"""
Generation 4 Demo: AI-Driven Intelligent Orchestration

Demonstrates advanced machine learning-powered optimization with predictive
analytics, adaptive configuration tuning, and autonomous decision-making.
"""

import asyncio
import time
import json
from datetime import datetime

from src.async_toolformer.intelligent_orchestrator import IntelligentAsyncOrchestrator, create_intelligent_orchestrator
from src.async_toolformer.tools import Tool
from src.async_toolformer.config import OrchestratorConfig
from src.async_toolformer.simple_structured_logging import get_logger

logger = get_logger(__name__)


# Demo tools for intelligent orchestration
@Tool(description="AI-enhanced web search with learning capabilities")
async def intelligent_web_search(query: str, context: str = "general") -> dict:
    """Smart web search that adapts based on usage patterns."""
    # Simulate variable execution time based on query complexity
    complexity_factor = len(query.split()) / 10.0
    base_time = 0.1 + (complexity_factor * 0.2)
    
    await asyncio.sleep(base_time)
    
    return {
        "query": query,
        "context": context,
        "results": [
            f"AI result 1 for: {query}",
            f"AI result 2 for: {query}",
            f"AI result 3 for: {query}"
        ],
        "relevance_score": min(0.95, 0.7 + complexity_factor),
        "execution_time": base_time,
        "ai_enhanced": True
    }


@Tool(description="Predictive data analysis with pattern recognition")
async def predictive_data_analyzer(data_source: str, analysis_type: str = "trend") -> dict:
    """Analyze data with predictive ML capabilities."""
    # Simulate different analysis complexities
    complexity_map = {
        "simple": 0.05,
        "trend": 0.15,
        "advanced": 0.3,
        "ml_prediction": 0.5
    }
    
    execution_time = complexity_map.get(analysis_type, 0.15)
    await asyncio.sleep(execution_time)
    
    # Simulate varying success rates
    success_rate = 0.95 if analysis_type in ["simple", "trend"] else 0.85
    
    if success_rate < 0.9:
        # Occasionally simulate analysis challenges
        import random
        if random.random() > success_rate:
            raise ValueError(f"Analysis failed for {analysis_type} on {data_source}")
    
    return {
        "data_source": data_source,
        "analysis_type": analysis_type,
        "insights": [
            f"Pattern detected in {data_source}",
            f"Trend analysis: {analysis_type}",
            f"Confidence: {success_rate:.2%}"
        ],
        "confidence_score": success_rate,
        "execution_time": execution_time,
        "patterns_found": 3 if analysis_type == "ml_prediction" else 2
    }


@Tool(description="Adaptive computation with resource optimization")
async def adaptive_computation(task_type: str, data_size: int = 1000) -> dict:
    """Perform computation that adapts based on learned resource patterns."""
    # Simulate computation time based on data size
    computation_time = (data_size / 10000.0) * 0.2
    
    await asyncio.sleep(computation_time)
    
    # Simulate resource usage
    memory_usage = data_size * 0.001  # MB
    cpu_usage = min(100.0, computation_time * 50)
    
    return {
        "task_type": task_type,
        "data_size": data_size,
        "result": f"Processed {data_size} items with {task_type}",
        "execution_time": computation_time,
        "memory_usage_mb": memory_usage,
        "cpu_usage_percent": cpu_usage,
        "optimization_applied": True
    }


async def demonstrate_intelligent_orchestration():
    """Demonstrate Generation 4 intelligent orchestration capabilities."""
    print("🤖 Generation 4: AI-Driven Intelligent Orchestration Demo")
    print("=" * 70)
    
    # Create intelligent orchestrator with ML optimization
    config = OrchestratorConfig(
        max_parallel_tools=5,
        max_parallel_per_type=3,  # Must be <= max_parallel_tools
        tool_timeout_ms=8000,
        retry_attempts=2
    )
    
    orchestrator = create_intelligent_orchestrator(
        tools=[intelligent_web_search, predictive_data_analyzer, adaptive_computation],
        config=config,
        enable_ml=True,
        learning_rate=0.02
    )
    
    print("\n1. Initial ML-Optimized Execution:")
    print("=" * 50)
    
    start_time = time.time()
    result1 = await orchestrator.execute(
        "Analyze market trends and optimize computational processes"
    )
    execution_time1 = (time.time() - start_time) * 1000
    
    print(f"✅ Initial execution completed: {execution_time1:.1f}ms")
    success1 = getattr(result1, 'success', result1.get('success', True)) if result1 else False
    print(f"   Success rate: {1.0 if success1 else 0.0:.2%}")
    
    # Get initial analytics
    analytics1 = orchestrator.get_intelligence_analytics()
    print(f"   ML predictions made: {analytics1['intelligence_metrics']['predictions_made']}")
    print(f"   Optimizations applied: {analytics1['intelligence_metrics']['optimizations_applied']}")
    
    print("\n2. Learning Phase - Multiple Executions:")
    print("=" * 50)
    
    # Execute multiple times to build learning data
    execution_times = []
    success_rates = []
    
    test_scenarios = [
        "Quick data analysis and simple web search",
        "Complex predictive analysis with advanced computation",
        "Multi-source research with trend analysis",
        "High-volume data processing with optimization",
        "Comprehensive analysis with pattern recognition"
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        try:
            start_time = time.time()
            result = await orchestrator.execute(scenario)
            exec_time = (time.time() - start_time) * 1000
            
            execution_times.append(exec_time)
            success = getattr(result, 'success', result.get('success', True)) if result else False
            success_rates.append(1.0 if success else 0.0)
            
            print(f"   Scenario {i}: {exec_time:.1f}ms | Success: {'✅' if success else '❌'}")
            
        except Exception as e:
            execution_times.append(0)
            success_rates.append(0.0)
            print(f"   Scenario {i}: Failed - {str(e)[:50]}...")
    
    print(f"\n📊 Learning Phase Results:")
    print(f"   Average execution time: {sum(execution_times)/len(execution_times):.1f}ms")
    print(f"   Overall success rate: {sum(success_rates)/len(success_rates):.2%}")
    
    print("\n3. Auto-Tuning Configuration:")
    print("=" * 50)
    
    # Demonstrate autonomous configuration tuning
    tuning_results = await orchestrator.auto_tune_configuration()
    
    print(f"✅ Auto-tuning completed: {tuning_results['status']}")
    print(f"   Overall confidence: {tuning_results.get('overall_confidence', 0):.2%}")
    
    if tuning_results.get('changes_applied'):
        print("   Configuration changes:")
        for change in tuning_results['changes_applied']:
            print(f"     • {change}")
    else:
        print("   No configuration changes needed")
    
    print("\n4. Optimized Performance Test:")
    print("=" * 50)
    
    # Test performance after optimization
    optimized_times = []
    
    for i in range(3):
        start_time = time.time()
        result = await orchestrator.execute(
            "Optimized execution with learned patterns and auto-tuned configuration"
        )
        exec_time = (time.time() - start_time) * 1000
        optimized_times.append(exec_time)
        
        print(f"   Optimized run {i+1}: {exec_time:.1f}ms")
    
    avg_optimized_time = sum(optimized_times) / len(optimized_times)
    improvement = ((execution_time1 - avg_optimized_time) / execution_time1) * 100
    
    print(f"\n✅ Performance improvement: {improvement:.1f}%")
    print(f"   Initial: {execution_time1:.1f}ms → Optimized: {avg_optimized_time:.1f}ms")
    
    print("\n5. Advanced Intelligence Analytics:")
    print("=" * 50)
    
    # Get comprehensive analytics
    final_analytics = orchestrator.get_intelligence_analytics()
    
    print("📊 Machine Learning Performance:")
    ml_perf = final_analytics['ml_performance']
    print(f"   Training samples: {ml_perf['training_samples']}")
    print(f"   Model confidence: {ml_perf['model_confidence']:.2%}")
    print(f"   Prediction accuracy: {ml_perf['prediction_accuracy']:.2%}")
    
    print("\n🧠 Intelligence Metrics:")
    intel_metrics = final_analytics['intelligence_metrics']
    print(f"   Total predictions: {intel_metrics['predictions_made']}")
    print(f"   Optimizations applied: {intel_metrics['optimizations_applied']}")
    print(f"   Learning sessions: {intel_metrics['learning_sessions']}")
    print(f"   Performance improvements: {intel_metrics['performance_improvements']}")
    
    print("\n🔍 Model Insights:")
    insights = final_analytics['model_insights']
    print(f"   Optimization success rate: {insights['optimization_success_rate']:.2%}")
    print(f"   Learning velocity: {insights['learning_velocity']:.3f} sessions/sec")
    
    if insights['most_important_features']:
        print("   Most important features:")
        for feature, importance in insights['most_important_features']:
            print(f"     • {feature}: {importance:.3f}")
    
    print("\n📈 Current Recommendations:")
    recommendations = final_analytics['recommendations']
    
    opt_parallel = recommendations['optimal_parallelism']
    print(f"   Optimal parallelism: {opt_parallel['value']} (confidence: {opt_parallel['confidence']:.2%})")
    
    failure_risk = recommendations['failure_risk']
    print(f"   Failure risk: {failure_risk['probability']:.2%} ({failure_risk['risk_level']} risk)")
    
    duration_est = recommendations['estimated_duration']
    print(f"   Estimated duration: {duration_est['milliseconds']:.0f}ms ({duration_est['category']})")
    
    if recommendations.get('insights'):
        print("   AI Insights:")
        for insight in recommendations['insights']:
            print(f"     • {insight}")
    
    print("\n🎯 Generation 4 Intelligence Features Demonstrated:")
    print("✅ Machine learning-powered predictive optimization")
    print("✅ Adaptive configuration tuning based on learned patterns")
    print("✅ Intelligent failure prevention and risk assessment")
    print("✅ Performance pattern recognition and auto-optimization")
    print("✅ Real-time analytics and actionable insights")
    print("✅ Autonomous decision-making with confidence scoring")
    
    # Cleanup
    await orchestrator.cleanup()
    
    print(f"\n📊 Session Summary - Correlation ID: {datetime.now().strftime('%Y%m%d_%H%M%S')}")


async def main():
    """Main demonstration function."""
    logger.info("Starting Generation 4 AI-driven demonstration")
    
    try:
        await demonstrate_intelligent_orchestration()
        print("\n🎉 AI-Driven Intelligence: ALL FEATURES DEMONSTRATED")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())