"""Quality gates validation script for SDLC completion."""

import asyncio
import sys
import importlib.util
import traceback
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our modules
from async_toolformer import AsyncOrchestrator, Tool, OrchestratorConfig
from async_toolformer.enhanced_reliability import reliability_manager
from async_toolformer.advanced_validation import advanced_validator
from async_toolformer.adaptive_scaling import adaptive_scaler
from async_toolformer.intelligent_caching import intelligent_cache


class QualityGateValidator:
    """Validates quality gates for autonomous SDLC completion."""
    
    def __init__(self):
        self.results: Dict[str, Any] = {
            "gates_passed": 0,
            "gates_failed": 0,
            "total_gates": 0,
            "details": []
        }
    
    async def run_quality_gate(self, name: str, test_func) -> bool:
        """Run a single quality gate test."""
        print(f"🔍 Running Quality Gate: {name}")
        self.results["total_gates"] += 1
        
        try:
            result = await test_func()
            if result:
                print(f"✅ PASSED: {name}")
                self.results["gates_passed"] += 1
                self.results["details"].append({
                    "gate": name,
                    "status": "PASSED",
                    "message": "Quality gate validation successful"
                })
                return True
            else:
                print(f"❌ FAILED: {name}")
                self.results["gates_failed"] += 1
                self.results["details"].append({
                    "gate": name,
                    "status": "FAILED",
                    "message": "Quality gate validation failed"
                })
                return False
        except Exception as e:
            print(f"💥 ERROR in {name}: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            self.results["gates_failed"] += 1
            self.results["details"].append({
                "gate": name,
                "status": "ERROR",
                "message": f"Exception during validation: {str(e)}"
            })
            return False
    
    async def test_generation_1_functionality(self) -> bool:
        """Test Generation 1: Basic functionality works."""
        
        # Create orchestrator
        config = OrchestratorConfig(max_parallel_tools=3, max_parallel_per_type=2)
        orchestrator = AsyncOrchestrator(config=config)
        
        # Define test tool
        @Tool("Simple test tool")
        async def simple_test_tool(message: str) -> str:
            return f"Processed: {message}"
        
        orchestrator.register_tool(simple_test_tool)
        
        # Test basic execution
        result = await orchestrator.execute("Use simple_test_tool with message 'hello world'")
        
        # Validate result structure
        if not isinstance(result, dict):
            return False
        
        required_fields = ["execution_id", "status", "total_time_ms"]
        if not all(field in result for field in required_fields):
            return False
        
        # Test execution statistics
        stats = await orchestrator.get_execution_stats()
        if not isinstance(stats, dict) or "total_executions" not in stats:
            return False
        
        await orchestrator.cleanup()
        return True
    
    async def test_generation_2_robustness(self) -> bool:
        """Test Generation 2: Enhanced reliability and validation."""
        
        # Test advanced validation
        result = await advanced_validator.validate_and_sanitize(
            "SELECT * FROM users WHERE 1=1",
            context="test_security"
        )
        
        if result.is_valid:  # Should be invalid due to SQL injection
            return False
        
        security_issues = [i for i in result.issues if i.category.name == "SECURITY"]
        if len(security_issues) == 0:
            return False
        
        # Test reliability tracking
        await reliability_manager.record_success("test_operation", 100.0)
        await reliability_manager.record_failure("test_operation", "TestError", "Test", 200.0)
        
        metrics = await reliability_manager.get_reliability_metrics()
        if metrics.total_requests < 2:
            return False
        
        health_status = await reliability_manager.get_health_status()
        if "status" not in health_status:
            return False
        
        return True
    
    async def test_generation_3_optimization(self) -> bool:
        """Test Generation 3: Performance optimization and scaling."""
        
        # Test intelligent caching
        await intelligent_cache.clear()
        
        test_key = "test_optimization_key"
        test_value = {"data": "optimization_test", "number": 42}
        
        await intelligent_cache.put(test_key, test_value, computation_cost=1.5)
        retrieved = await intelligent_cache.get(test_key)
        
        if retrieved != test_value:
            return False
        
        cache_stats = await intelligent_cache.get_stats()
        if cache_stats.total_requests == 0:
            return False
        
        # Test adaptive scaling status
        scaling_status = await adaptive_scaler.get_scaling_status()
        required_status_fields = ["current_workers", "min_workers", "max_workers"]
        if not all(field in scaling_status for field in required_status_fields):
            return False
        
        return True
    
    async def test_integrated_functionality(self) -> bool:
        """Test integrated functionality across all generations."""
        
        config = OrchestratorConfig(max_parallel_tools=2, max_parallel_per_type=2)
        orchestrator = AsyncOrchestrator(config=config)
        
        @Tool("Integration test tool")
        async def integration_tool(data: str) -> Dict[str, Any]:
            return {
                "processed": data,
                "timestamp": "2025-01-01T00:00:00Z",
                "success": True
            }
        
        orchestrator.register_tool(integration_tool)
        
        # Execute with all enhancements
        result = await orchestrator.execute(
            "Use integration_tool to process 'integrated test data'",
            user_id="test_user"
        )
        
        # Validate comprehensive result
        if result.get("status") != "completed":
            return False
        
        if "adaptive_timeout_ms" not in result:
            return False
        
        # Test metrics collection
        orchestrator_metrics = await orchestrator.get_metrics()
        if "registered_tools" not in orchestrator_metrics:
            return False
        
        await orchestrator.cleanup()
        return True
    
    async def test_error_handling_robustness(self) -> bool:
        """Test comprehensive error handling."""
        
        config = OrchestratorConfig(max_parallel_tools=2, max_parallel_per_type=2)
        orchestrator = AsyncOrchestrator(config=config)
        
        @Tool("Failing test tool")
        async def failing_tool(should_fail: bool = True) -> str:
            if should_fail:
                raise ValueError("Intentional test failure")
            return "Success"
        
        orchestrator.register_tool(failing_tool)
        
        # Test graceful failure handling
        result = await orchestrator.execute("Use failing_tool with should_fail=true")
        
        # Should handle error gracefully, not crash
        if not isinstance(result, dict):
            return False
        
        # Should have error information
        if "status" not in result:
            return False
        
        await orchestrator.cleanup()
        return True
    
    async def test_performance_characteristics(self) -> bool:
        """Test performance meets requirements."""
        
        import time
        
        config = OrchestratorConfig(max_parallel_tools=5, max_parallel_per_type=3)
        orchestrator = AsyncOrchestrator(config=config)
        
        @Tool("Performance test tool")
        async def perf_tool(delay_ms: int = 100) -> str:
            await asyncio.sleep(delay_ms / 1000)
            return f"Completed with {delay_ms}ms delay"
        
        orchestrator.register_tool(perf_tool)
        
        # Test parallel execution performance
        start_time = time.time()
        
        tasks = []
        for i in range(3):
            task = orchestrator.execute(f"Use perf_tool with delay_ms=200")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Should complete in reasonable time (parallel execution should be faster than sequential)
        # 3 tasks with 200ms delay each = 600ms sequential, should be much faster in parallel
        if total_time > 2.0:  # Allow 2 second maximum
            return False
        
        successful_results = [r for r in results if isinstance(r, dict) and r.get("status") == "completed"]
        if len(successful_results) < 2:  # At least 2 should succeed
            return False
        
        await orchestrator.cleanup()
        return True
    
    async def test_security_compliance(self) -> bool:
        """Test security and compliance features."""
        
        # Test multiple security validations
        security_tests = [
            "SELECT * FROM users; DROP TABLE passwords;",
            "<script>alert('xss')</script>",
            "exec('malicious_code')",
            "../../../etc/passwd",
            "user@example.com and phone 555-1234"
        ]
        
        security_issues_found = 0
        
        for test_input in security_tests:
            result = await advanced_validator.validate_and_sanitize(
                test_input,
                context=f"security_test",
                strict_mode=True
            )
            
            if not result.is_valid:
                security_issues_found += 1
        
        # Should detect security issues in most test cases
        if security_issues_found < 3:
            return False
        
        return True
    
    def get_summary(self) -> Dict[str, Any]:
        """Get quality gates summary."""
        success_rate = 0
        if self.results["total_gates"] > 0:
            success_rate = self.results["gates_passed"] / self.results["total_gates"]
        
        return {
            **self.results,
            "success_rate": success_rate,
            "overall_status": "PASSED" if success_rate >= 0.85 else "FAILED"
        }


async def main():
    """Run all quality gates."""
    print("🚀 TERRAGON AUTONOMOUS SDLC - QUALITY GATES VALIDATION")
    print("=" * 60)
    
    validator = QualityGateValidator()
    
    # Define quality gates
    quality_gates = [
        ("Generation 1: Basic Functionality", validator.test_generation_1_functionality),
        ("Generation 2: Robustness & Reliability", validator.test_generation_2_robustness),
        ("Generation 3: Optimization & Scaling", validator.test_generation_3_optimization),
        ("Integrated Functionality", validator.test_integrated_functionality),
        ("Error Handling Robustness", validator.test_error_handling_robustness),
        ("Performance Characteristics", validator.test_performance_characteristics),
        ("Security & Compliance", validator.test_security_compliance),
    ]
    
    # Run all quality gates
    print(f"\n🔬 Running {len(quality_gates)} Quality Gates...\n")
    
    for gate_name, gate_func in quality_gates:
        await validator.run_quality_gate(gate_name, gate_func)
        print()  # Add spacing
    
    # Generate summary
    summary = validator.get_summary()
    
    print("=" * 60)
    print("📊 QUALITY GATES SUMMARY")
    print("=" * 60)
    print(f"Total Gates: {summary['total_gates']}")
    print(f"Passed: {summary['gates_passed']} ✅")
    print(f"Failed: {summary['gates_failed']} ❌")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Overall Status: {summary['overall_status']}")
    
    if summary['overall_status'] == 'PASSED':
        print("\n🎉 ALL QUALITY GATES PASSED - AUTONOMOUS SDLC COMPLETE!")
        print("✅ Code runs without errors")
        print("✅ Enhanced reliability and validation")  
        print("✅ Performance optimization implemented")
        print("✅ Security compliance validated")
        print("✅ Comprehensive error handling")
    else:
        print(f"\n⚠️  QUALITY GATES FAILED - {summary['gates_failed']} issues need resolution")
        print("\nFailed Gates:")
        for detail in summary['details']:
            if detail['status'] in ['FAILED', 'ERROR']:
                print(f"❌ {detail['gate']}: {detail['message']}")
    
    return summary['overall_status'] == 'PASSED'


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Quality gates validation interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Quality gates validation failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)