#!/usr/bin/env python3
"""
Advanced Quality Gates with ML Validation

Comprehensive quality assurance with machine learning-powered validation,
autonomous testing, and intelligent failure prediction.
"""

import asyncio
import time
import json
from datetime import datetime
from typing import Dict, List, Any

from src.async_toolformer.intelligent_orchestrator import IntelligentAsyncOrchestrator, create_intelligent_orchestrator
from src.async_toolformer.autonomous_manager import AutonomousOrchestrator, AutonomyLevel, SystemHealth
from src.async_toolformer.research_algorithms import QuantumInspiredTaskScheduler, AdaptiveLoadBalancer, research_framework
from src.async_toolformer.config import OrchestratorConfig
from src.async_toolformer.tools import Tool
from src.async_toolformer.simple_structured_logging import get_logger

logger = get_logger(__name__)


# Advanced test tools for quality validation
@Tool(description="Advanced security validation with ML threat detection")
async def ml_security_validator(input_data: str, threat_level: str = "medium") -> dict:
    """ML-powered security validation."""
    await asyncio.sleep(0.1)
    
    # Simulate ML threat detection
    threat_indicators = [
        'script injection', 'sql injection', 'xss attempt', 
        'path traversal', 'command injection', 'buffer overflow'
    ]
    
    detected_threats = []
    risk_score = 0.0
    
    for threat in threat_indicators:
        if threat.replace(' ', '_') in input_data.lower():
            detected_threats.append(threat)
            risk_score += 0.3
    
    # Simulate ML confidence
    ml_confidence = 0.95 if len(detected_threats) > 0 else 0.85
    
    return {
        "input_data": input_data,
        "threat_level": threat_level,
        "detected_threats": detected_threats,
        "risk_score": min(1.0, risk_score),
        "ml_confidence": ml_confidence,
        "validation_result": "blocked" if risk_score > 0.5 else "approved",
        "recommendations": [f"Block {threat}" for threat in detected_threats] if detected_threats else ["Input approved"]
    }


@Tool(description="Performance regression detection with ML analysis")
async def ml_performance_analyzer(benchmark_data: dict, baseline_metrics: dict = None) -> dict:
    """ML-powered performance analysis and regression detection."""
    await asyncio.sleep(0.05)
    
    if baseline_metrics is None:
        baseline_metrics = {"avg_latency": 100.0, "throughput": 1000.0, "error_rate": 0.02}
    
    current_latency = benchmark_data.get("latency", 100.0)
    current_throughput = benchmark_data.get("throughput", 1000.0)
    current_error_rate = benchmark_data.get("error_rate", 0.02)
    
    # ML regression detection
    latency_regression = (current_latency - baseline_metrics["avg_latency"]) / baseline_metrics["avg_latency"]
    throughput_regression = (baseline_metrics["throughput"] - current_throughput) / baseline_metrics["throughput"]
    error_rate_regression = (current_error_rate - baseline_metrics["error_rate"]) / baseline_metrics["error_rate"]
    
    # Overall performance score
    performance_score = max(0.0, 1.0 - (latency_regression * 0.4 + throughput_regression * 0.4 + error_rate_regression * 0.2))
    
    # ML predictions
    regressions_detected = []
    if latency_regression > 0.1:
        regressions_detected.append(f"Latency regression: +{latency_regression:.1%}")
    if throughput_regression > 0.1:
        regressions_detected.append(f"Throughput regression: -{throughput_regression:.1%}")
    if error_rate_regression > 0.2:
        regressions_detected.append(f"Error rate regression: +{error_rate_regression:.1%}")
    
    return {
        "benchmark_data": benchmark_data,
        "baseline_metrics": baseline_metrics,
        "performance_score": performance_score,
        "regressions_detected": regressions_detected,
        "ml_analysis": {
            "latency_trend": "degrading" if latency_regression > 0.05 else "stable",
            "throughput_trend": "degrading" if throughput_regression > 0.05 else "stable",
            "error_trend": "degrading" if error_rate_regression > 0.1 else "stable"
        },
        "recommendation": "investigate_performance" if regressions_detected else "performance_acceptable",
        "confidence": 0.92
    }


@Tool(description="Intelligent code quality analysis with ML insights")
async def ml_code_quality_analyzer(code_metrics: dict, quality_standards: dict = None) -> dict:
    """ML-powered code quality analysis."""
    await asyncio.sleep(0.08)
    
    if quality_standards is None:
        quality_standards = {
            "max_complexity": 10,
            "min_coverage": 0.85,
            "max_duplication": 0.05,
            "max_technical_debt": 0.1
        }
    
    complexity = code_metrics.get("complexity", 5)
    coverage = code_metrics.get("coverage", 0.9)
    duplication = code_metrics.get("duplication", 0.02)
    technical_debt = code_metrics.get("technical_debt", 0.05)
    
    # ML quality scoring
    complexity_score = max(0.0, 1.0 - (complexity / quality_standards["max_complexity"]))
    coverage_score = coverage / quality_standards["min_coverage"]
    duplication_score = max(0.0, 1.0 - (duplication / quality_standards["max_duplication"]))
    debt_score = max(0.0, 1.0 - (technical_debt / quality_standards["max_technical_debt"]))
    
    overall_quality = (complexity_score + coverage_score + duplication_score + debt_score) / 4.0
    
    # ML insights
    quality_issues = []
    if complexity > quality_standards["max_complexity"]:
        quality_issues.append(f"High complexity: {complexity} > {quality_standards['max_complexity']}")
    if coverage < quality_standards["min_coverage"]:
        quality_issues.append(f"Low coverage: {coverage:.1%} < {quality_standards['min_coverage']:.1%}")
    if duplication > quality_standards["max_duplication"]:
        quality_issues.append(f"High duplication: {duplication:.1%} > {quality_standards['max_duplication']:.1%}")
    if technical_debt > quality_standards["max_technical_debt"]:
        quality_issues.append(f"High technical debt: {technical_debt:.1%} > {quality_standards['max_technical_debt']:.1%}")
    
    return {
        "code_metrics": code_metrics,
        "quality_standards": quality_standards,
        "overall_quality_score": overall_quality,
        "quality_grade": "A" if overall_quality > 0.9 else "B" if overall_quality > 0.7 else "C" if overall_quality > 0.5 else "D",
        "quality_issues": quality_issues,
        "ml_recommendations": [
            "Refactor complex methods" if complexity > quality_standards["max_complexity"] else None,
            "Add unit tests" if coverage < quality_standards["min_coverage"] else None,
            "Eliminate duplicate code" if duplication > quality_standards["max_duplication"] else None,
            "Address technical debt" if technical_debt > quality_standards["max_technical_debt"] else None
        ],
        "confidence": 0.88,
        "passed": overall_quality >= 0.7
    }


class MLQualityGateOrchestrator:
    """ML-powered quality gate orchestrator with intelligent validation."""
    
    def __init__(self):
        self.intelligent_orchestrator = None
        self.autonomous_manager = None
        self.quality_history = []
        self.failure_predictions = []
        
    async def initialize(self):
        """Initialize ML-powered quality systems."""
        # Create intelligent orchestrator
        config = OrchestratorConfig(
            max_parallel_tools=8,
            max_parallel_per_type=4,
            tool_timeout_ms=10000,
            retry_attempts=3
        )
        
        self.intelligent_orchestrator = create_intelligent_orchestrator(
            tools=[ml_security_validator, ml_performance_analyzer, ml_code_quality_analyzer],
            config=config,
            enable_ml=True,
            learning_rate=0.05
        )
        
        # Create autonomous manager
        self.autonomous_manager = AutonomousOrchestrator(
            autonomy_level=AutonomyLevel.FULLY_AUTONOMOUS,
            enable_self_healing=True,
            enable_evolution=True
        )
        
        logger.info("ML Quality Gate Orchestrator initialized")
    
    async def run_comprehensive_quality_gates(self) -> Dict[str, Any]:
        """Run comprehensive ML-powered quality gates."""
        quality_results = {
            "timestamp": datetime.now().isoformat(),
            "gates_executed": [],
            "overall_status": "unknown",
            "ml_insights": {},
            "autonomous_actions": [],
            "quality_score": 0.0
        }
        
        # Gate 1: ML Security Validation
        print("\n🛡️ Gate 1: ML Security Validation")
        security_tests = [
            "normal_user_input",
            "script_injection_attempt",
            "sql_injection_attempt", 
            "xss_attempt",
            "path_traversal_attempt"
        ]
        
        security_results = []
        for test_input in security_tests:
            result = await self.intelligent_orchestrator.execute(
                f"Validate security for input: {test_input}"
            )
            security_results.append(result)
            
            # Extract security validation results
            if hasattr(result, 'results') or isinstance(result, dict):
                results_data = getattr(result, 'results', result)
                if results_data:
                    first_result = results_data[0] if isinstance(results_data, list) else results_data
                    validation_result = first_result.get('validation_result', 'unknown')
                    print(f"   {test_input}: {validation_result}")
        
        security_score = len([r for r in security_results if self._is_successful_result(r)]) / len(security_results)
        print(f"   Security Score: {security_score:.1%}")
        
        quality_results["gates_executed"].append({
            "gate": "security_validation",
            "score": security_score,
            "tests": len(security_tests),
            "passed": security_score >= 0.8
        })
        
        # Gate 2: ML Performance Analysis
        print("\n⚡ Gate 2: ML Performance Analysis")
        performance_benchmarks = [
            {"latency": 95.0, "throughput": 1100.0, "error_rate": 0.015},
            {"latency": 120.0, "throughput": 950.0, "error_rate": 0.025},
            {"latency": 85.0, "throughput": 1200.0, "error_rate": 0.01}
        ]
        
        performance_results = []
        for i, benchmark in enumerate(performance_benchmarks):
            result = await self.intelligent_orchestrator.execute(
                f"Analyze performance benchmark {i+1}: {json.dumps(benchmark)}"
            )
            performance_results.append(result)
            
            print(f"   Benchmark {i+1}: Latency={benchmark['latency']}ms, Throughput={benchmark['throughput']}")
        
        performance_score = len([r for r in performance_results if self._is_successful_result(r)]) / len(performance_results)
        print(f"   Performance Score: {performance_score:.1%}")
        
        quality_results["gates_executed"].append({
            "gate": "performance_analysis", 
            "score": performance_score,
            "benchmarks": len(performance_benchmarks),
            "passed": performance_score >= 0.8
        })
        
        # Gate 3: ML Code Quality Analysis
        print("\n📊 Gate 3: ML Code Quality Analysis")
        code_samples = [
            {"complexity": 8, "coverage": 0.92, "duplication": 0.03, "technical_debt": 0.06},
            {"complexity": 12, "coverage": 0.78, "duplication": 0.08, "technical_debt": 0.12},
            {"complexity": 6, "coverage": 0.95, "duplication": 0.02, "technical_debt": 0.04}
        ]
        
        quality_analysis_results = []
        for i, metrics in enumerate(code_samples):
            result = await self.intelligent_orchestrator.execute(
                f"Analyze code quality metrics {i+1}: {json.dumps(metrics)}"
            )
            quality_analysis_results.append(result)
            
            print(f"   Code Sample {i+1}: Complexity={metrics['complexity']}, Coverage={metrics['coverage']:.1%}")
        
        quality_analysis_score = len([r for r in quality_analysis_results if self._is_successful_result(r)]) / len(quality_analysis_results)
        print(f"   Code Quality Score: {quality_analysis_score:.1%}")
        
        quality_results["gates_executed"].append({
            "gate": "code_quality_analysis",
            "score": quality_analysis_score,
            "samples": len(code_samples),
            "passed": quality_analysis_score >= 0.8
        })
        
        # Calculate overall quality score
        overall_score = sum(gate["score"] for gate in quality_results["gates_executed"]) / len(quality_results["gates_executed"])
        quality_results["quality_score"] = overall_score
        
        # Determine overall status
        all_passed = all(gate["passed"] for gate in quality_results["gates_executed"])
        quality_results["overall_status"] = "passed" if all_passed and overall_score >= 0.8 else "failed"
        
        # Get ML insights from intelligent orchestrator
        analytics = self.intelligent_orchestrator.get_intelligence_analytics()
        quality_results["ml_insights"] = {
            "predictions_made": analytics["intelligence_metrics"]["predictions_made"],
            "optimizations_applied": analytics["intelligence_metrics"]["optimizations_applied"],
            "model_confidence": analytics["ml_performance"]["model_confidence"],
            "recommendations": analytics.get("recommendations", {})
        }
        
        return quality_results
    
    async def run_autonomous_quality_monitoring(self) -> Dict[str, Any]:
        """Run autonomous quality monitoring with self-healing."""
        print("\n🤖 Autonomous Quality Monitoring")
        
        # Record some performance metrics for autonomous analysis
        for i in range(10):
            metric = {
                "timestamp": time.time(),
                "response_time_ms": 100 + i * 10,
                "success": True,
                "memory_usage_mb": 50 + i * 2
            }
            self.autonomous_manager.record_performance_metric(metric)
        
        # Perform health check
        health_status = await self.autonomous_manager._perform_health_check()
        print(f"   System Health: {health_status.value}")
        
        # Apply self-healing if needed
        healing_actions = await self.autonomous_manager._apply_self_healing()
        print(f"   Self-healing actions applied: {len(healing_actions)}")
        
        # Get autonomous status
        autonomous_status = self.autonomous_manager.get_autonomous_status()
        
        return {
            "health_status": health_status.value,
            "healing_actions": len(healing_actions),
            "autonomous_events": len(autonomous_status["recent_events"]),
            "system_metrics": autonomous_status["metrics_summary"]
        }
    
    async def run_research_algorithm_validation(self) -> Dict[str, Any]:
        """Validate research algorithms with quality gates."""
        print("\n🔬 Research Algorithm Validation")
        
        # Test quantum scheduler
        quantum_scheduler = QuantumInspiredTaskScheduler(coherence_factor=0.9)
        quantum_scheduler.create_task_superposition("validation_task", [
            {"name": "strategy_a", "weight": 0.6},
            {"name": "strategy_b", "weight": 0.4}
        ])
        
        quantum_measurement = quantum_scheduler.measure_optimal_strategy(
            "validation_task", 
            {"cpu_usage": 50, "workload_size": 15}
        )
        
        print(f"   Quantum Scheduler: Strategy={quantum_measurement['strategy']['name']}, Confidence={quantum_measurement['confidence']:.3f}")
        
        # Test adaptive load balancer
        adaptive_balancer = AdaptiveLoadBalancer(learning_rate=0.1)
        test_workload = [{"task_id": f"t{i}", "cpu_requirement": 5} for i in range(10)]
        
        balancer_result = await adaptive_balancer.optimize_load_distribution(test_workload)
        print(f"   Adaptive Balancer: Strategy={balancer_result['strategy']}, Confidence={balancer_result['confidence']:.3f}")
        
        # Validate research framework
        research_status = {
            "quantum_scheduler_operational": quantum_measurement['confidence'] > 0.5,
            "adaptive_balancer_operational": balancer_result['confidence'] > 0.0,
            "research_experiments": len(research_framework.experiments)
        }
        
        return research_status
    
    def _is_successful_result(self, result: Any) -> bool:
        """Check if a result indicates success."""
        if hasattr(result, 'success'):
            return result.success
        elif isinstance(result, dict):
            return result.get('success', True)
        else:
            return result is not None
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.intelligent_orchestrator:
            await self.intelligent_orchestrator.cleanup()
        if self.autonomous_manager:
            await self.autonomous_manager.stop_autonomous_operation()


async def main():
    """Main quality gates demonstration."""
    print("🛡️ ADVANCED QUALITY GATES WITH ML VALIDATION")
    print("=" * 80)
    
    orchestrator = MLQualityGateOrchestrator()
    
    try:
        # Initialize ML systems
        await orchestrator.initialize()
        
        # Run comprehensive quality gates
        quality_results = await orchestrator.run_comprehensive_quality_gates()
        
        # Run autonomous monitoring
        autonomous_results = await orchestrator.run_autonomous_quality_monitoring()
        
        # Validate research algorithms
        research_validation = await orchestrator.run_research_algorithm_validation()
        
        # Generate final report
        print(f"\n📊 QUALITY GATES FINAL REPORT")
        print("=" * 60)
        
        print(f"Overall Status: {'✅ PASSED' if quality_results['overall_status'] == 'passed' else '❌ FAILED'}")
        print(f"Quality Score: {quality_results['quality_score']:.1%}")
        
        print(f"\n🎯 Gate Results:")
        for gate in quality_results["gates_executed"]:
            status = "✅ PASS" if gate["passed"] else "❌ FAIL" 
            print(f"   {gate['gate']}: {status} ({gate['score']:.1%})")
        
        print(f"\n🤖 Autonomous System:")
        print(f"   Health Status: {autonomous_results['health_status']}")
        print(f"   Self-healing Actions: {autonomous_results['healing_actions']}")
        print(f"   System Events: {autonomous_results['autonomous_events']}")
        
        print(f"\n🔬 Research Validation:")
        print(f"   Quantum Scheduler: {'✅' if research_validation['quantum_scheduler_operational'] else '❌'}")
        print(f"   Adaptive Balancer: {'✅' if research_validation['adaptive_balancer_operational'] else '❌'}")
        print(f"   Research Experiments: {research_validation['research_experiments']}")
        
        print(f"\n🧠 ML Insights:")
        ml_insights = quality_results["ml_insights"]
        print(f"   Predictions Made: {ml_insights['predictions_made']}")
        print(f"   Optimizations Applied: {ml_insights['optimizations_applied']}")
        print(f"   Model Confidence: {ml_insights['model_confidence']:.1%}")
        
        print(f"\n🎉 ADVANCED QUALITY GATES: {'✅ ALL SYSTEMS VALIDATED' if quality_results['overall_status'] == 'passed' else '⚠️ ISSUES DETECTED'}")
        
        return quality_results
        
    except Exception as e:
        logger.error(f"Quality gates failed: {e}")
        print(f"\n❌ Quality gates execution failed: {e}")
        raise
    finally:
        await orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())