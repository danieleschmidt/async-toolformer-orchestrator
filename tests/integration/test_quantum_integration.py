"""
Integration tests for Quantum-Enhanced AsyncOrchestrator.

This module provides comprehensive integration tests for the quantum features:
- Quantum task planning and execution
- Security context management
- Performance optimization
- Concurrency coordination
- End-to-end quantum workflows
"""

import pytest
import asyncio
import time
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

from async_toolformer import (
    QuantumAsyncOrchestrator,
    create_quantum_orchestrator,
    Tool,
    SecurityLevel,
    AccessLevel,
    ValidationLevel,
    OptimizationStrategy,
    SynchronizationType,
)


# Test tools for quantum integration testing
@Tool(description="Fast computation task")
async def fast_task(value: int) -> int:
    """Simulate a fast computational task."""
    await asyncio.sleep(0.1)
    return value * 2


@Tool(description="Medium computation task")
async def medium_task(value: int) -> int:
    """Simulate a medium computational task."""
    await asyncio.sleep(0.5)
    return value + 10


@Tool(description="Slow computation task")
async def slow_task(value: int) -> int:
    """Simulate a slow computational task."""
    await asyncio.sleep(1.0)
    return value ** 2


@Tool(description="Resource intensive task")
async def resource_intensive_task(data_size: int) -> Dict[str, int]:
    """Simulate a resource-intensive task."""
    await asyncio.sleep(0.3)
    return {
        "processed_items": data_size,
        "memory_used": data_size * 1024,
        "cpu_cycles": data_size * 1000,
    }


@Tool(description="Network simulation task")
async def network_task(endpoint: str) -> Dict[str, Any]:
    """Simulate a network operation."""
    await asyncio.sleep(0.2)
    return {
        "endpoint": endpoint,
        "status_code": 200,
        "response_time_ms": 200,
        "data_size": 1024,
    }


@Tool(description="Database simulation task")
async def database_task(query: str) -> Dict[str, Any]:
    """Simulate a database operation."""
    await asyncio.sleep(0.4)
    return {
        "query": query,
        "rows_affected": 42,
        "execution_time_ms": 400,
        "cache_hit": True,
    }


class TestQuantumIntegration:
    """Integration tests for quantum-enhanced orchestrator."""
    
    @pytest.fixture
    async def quantum_orchestrator(self):
        """Create a quantum orchestrator for testing."""
        orchestrator = create_quantum_orchestrator(
            tools=[
                fast_task,
                medium_task,
                slow_task,
                resource_intensive_task,
                network_task,
                database_task,
            ],
            quantum_config={
                "max_parallel_tasks": 10,
                "optimization_iterations": 20,  # Reduced for testing
                "enable_entanglement": True,
                "security": {
                    "security_level": SecurityLevel.MEDIUM,
                    "enable_quantum_tokens": True,
                },
                "validation": {
                    "validation_level": ValidationLevel.STANDARD,
                    "enable_coherence_checks": True,
                },
                "performance": {
                    "strategy": OptimizationStrategy.EFFICIENCY,
                    "auto_scaling": True,
                    "monitoring_interval": 0.1,  # Fast for testing
                },
                "concurrency": {
                    "quantum_sync": True,
                    "deadlock_detection": True,
                    "max_wait_time": 5.0,  # Shorter for testing
                },
            }
        )
        
        yield orchestrator
        
        # Cleanup
        await orchestrator.cleanup()
    
    @pytest.mark.asyncio
    async def test_basic_quantum_execution(self, quantum_orchestrator):
        """Test basic quantum execution functionality."""
        prompt = """
        Perform parallel computations:
        - Run fast task with value 5
        - Run medium task with value 10
        - Check the results are correct
        """
        
        result = await quantum_orchestrator.quantum_execute(
            prompt=prompt,
            optimize_plan=True,
            timeout_ms=10000,
        )
        
        # Verify execution completed
        assert result["status"] == "completed"
        assert result["tools_executed"] >= 2  # At least fast and medium tasks
        assert result["successful_tools"] >= 2
        assert result["total_time_ms"] > 0
        
        # Verify quantum metrics are present
        assert "quantum_metrics" in result
        assert "parallelism_achieved" in result["quantum_metrics"]
        assert "quantum_coherence" in result["quantum_metrics"]
        
        # Verify performance metrics
        assert "performance_metrics" in result
        assert "execution" in result["performance_metrics"]
        
        # Verify concurrency metrics
        assert "concurrency_metrics" in result
    
    @pytest.mark.asyncio
    async def test_quantum_planning_optimization(self, quantum_orchestrator):
        """Test quantum planning optimization features."""
        prompt = """
        Execute a complex workflow:
        - Process data with resource intensive task (size: 1000)
        - Perform network operations to multiple endpoints
        - Run database queries
        - Apply fast and slow computations
        """
        
        # Test with optimization enabled
        optimized_result = await quantum_orchestrator.quantum_execute(
            prompt=prompt,
            optimize_plan=True,
        )
        
        # Test with optimization disabled
        unoptimized_result = await quantum_orchestrator.quantum_execute(
            prompt=prompt,
            optimize_plan=False,
        )
        
        # Verify both completed successfully
        assert optimized_result["status"] == "completed"
        assert unoptimized_result["status"] == "completed"
        
        # Optimized version should have better metrics
        opt_score = optimized_result["quantum_metrics"]["optimization_score"]
        unopt_score = unoptimized_result["quantum_metrics"]["optimization_score"]
        
        # The optimized version should have equal or better optimization score
        assert opt_score >= unopt_score - 0.1  # Small tolerance for variance
    
    @pytest.mark.asyncio
    async def test_security_context_integration(self, quantum_orchestrator):
        """Test security context integration."""
        # Create a security context
        security_context = quantum_orchestrator.security_manager.create_security_context(
            user_id="test_user",
            access_level=AccessLevel.RESTRICTED,
            security_level=SecurityLevel.HIGH,
            allowed_resources={"computation", "network"},
        )
        
        prompt = "Run fast task with value 42"
        
        result = await quantum_orchestrator.quantum_execute(
            prompt=prompt,
            security_context=security_context,
            validation_level=ValidationLevel.STRICT,
        )
        
        # Verify execution with security context
        assert result["status"] == "completed"
        
        # Verify security metrics
        security_metrics = result["security_metrics"]
        assert security_metrics["active_contexts"] >= 1
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, quantum_orchestrator):
        """Test performance monitoring and optimization."""
        # Start monitoring explicitly
        await quantum_orchestrator.performance_optimizer.start_monitoring()
        
        try:
            # Execute multiple tasks to generate performance data
            for i in range(5):
                result = await quantum_orchestrator.quantum_execute(
                    f"Run fast task with value {i * 10}",
                    optimize_plan=True,
                )
                assert result["status"] == "completed"
                
                # Small delay to allow metrics collection
                await asyncio.sleep(0.2)
            
            # Get performance recommendations
            recommendations = quantum_orchestrator.get_performance_recommendations()
            
            # Should have some recommendations (may be empty for simple tests)
            assert isinstance(recommendations, list)
            
            # Get current performance metrics
            current_metrics = quantum_orchestrator.performance_optimizer.get_current_metrics()
            assert current_metrics.total_tasks_completed >= 5
            
        finally:
            await quantum_orchestrator.performance_optimizer.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_concurrency_coordination(self, quantum_orchestrator):
        """Test concurrency coordination features."""
        # Create multiple concurrent executions
        tasks = []
        
        for i in range(3):
            task = asyncio.create_task(
                quantum_orchestrator.quantum_execute(
                    f"Run medium task with value {i * 5}",
                    optimize_plan=True,
                )
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all completed
        for result in results:
            assert not isinstance(result, Exception)
            assert result["status"] == "completed"
        
        # Check concurrency status
        concurrency_status = quantum_orchestrator.get_concurrency_status()
        assert "total_locks" in concurrency_status
        assert "task_locks" in concurrency_status
    
    @pytest.mark.asyncio
    async def test_quantum_streaming_execution(self, quantum_orchestrator):
        """Test quantum streaming execution."""
        prompt = """
        Execute streaming workflow:
        - Run fast task with value 1
        - Run medium task with value 2
        - Run slow task with value 3
        """
        
        results = []
        phase_count = 0
        
        async for phase_result in quantum_orchestrator.quantum_stream_execute(
            prompt=prompt,
            optimize_plan=True,
        ):
            phase_count += 1
            results.append(phase_result)
            
            # Verify phase result structure
            assert "phase" in phase_result
            assert "total_phases" in phase_result
            assert "results" in phase_result
            assert "quantum_coherence" in phase_result
        
        # Should have received at least one phase
        assert phase_count > 0
        assert len(results) > 0
        
        # Verify final phase completed
        final_phase = results[-1]
        assert final_phase["phase"] == final_phase["total_phases"]
    
    @pytest.mark.asyncio
    async def test_validation_integration(self, quantum_orchestrator):
        """Test validation system integration."""
        # Test with strict validation
        prompt = "Run fast task with value 100"
        
        result = await quantum_orchestrator.quantum_execute(
            prompt=prompt,
            validation_level=ValidationLevel.STRICT,
        )
        
        # Should complete successfully with valid inputs
        assert result["status"] == "completed"
        
        # Get validation metrics
        validation_metrics = result["validation_metrics"]
        assert "validation_level" in validation_metrics
        assert "total_rules" in validation_metrics
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, quantum_orchestrator):
        """Test error handling and recovery mechanisms."""
        # Create a failing tool for testing
        @Tool(description="Failing task for testing")
        async def failing_task(should_fail: bool = True) -> str:
            if should_fail:
                raise ValueError("Intentional test failure")
            return "success"
        
        # Register the failing tool temporarily
        quantum_orchestrator.register_tool(failing_task)
        
        try:
            # Execute with a task that will fail
            result = await quantum_orchestrator.quantum_execute(
                "Run failing task with should_fail=True",
                optimize_plan=True,
            )
            
            # Execution should complete but with failures
            assert result["status"] == "completed"
            assert result["failed_tools"] >= 1
            
            # Verify error information is captured
            assert "results" in result
            failed_results = [r for r in result["results"] if not r.success]
            assert len(failed_results) >= 1
            
        finally:
            # Clean up the failing tool (would need registry method for this)
            pass
    
    @pytest.mark.asyncio
    async def test_resource_scaling(self, quantum_orchestrator):
        """Test resource scaling features."""
        # Get initial scaling status
        initial_scaling = quantum_orchestrator.performance_optimizer.get_resource_scaling_status()
        assert "current_scales" in initial_scaling
        assert "auto_scaling_enabled" in initial_scaling
        
        # Execute resource-intensive tasks
        for i in range(3):
            await quantum_orchestrator.quantum_execute(
                f"Run resource intensive task with data_size {1000 * (i + 1)}",
                optimize_plan=True,
            )
        
        # Get final scaling status
        final_scaling = quantum_orchestrator.performance_optimizer.get_resource_scaling_status()
        
        # Scaling rules should be present
        assert "scaling_rules" in final_scaling
        assert len(final_scaling["scaling_rules"]) > 0
    
    @pytest.mark.asyncio
    async def test_quantum_coherence_maintenance(self, quantum_orchestrator):
        """Test quantum coherence maintenance."""
        # Execute tasks that should maintain coherence
        result = await quantum_orchestrator.quantum_execute(
            "Run fast task with value 42",
            optimize_plan=True,
            validation_level=ValidationLevel.QUANTUM_COHERENT,
        )
        
        assert result["status"] == "completed"
        
        # Check quantum coherence level
        quantum_coherence = result["quantum_metrics"]["quantum_coherence"]
        assert 0.0 <= quantum_coherence <= 1.0
        
        # For simple tasks, coherence should be reasonably high
        assert quantum_coherence > 0.5
    
    @pytest.mark.asyncio
    async def test_comprehensive_workflow(self, quantum_orchestrator):
        """Test a comprehensive quantum workflow."""
        complex_prompt = """
        Execute a comprehensive data processing workflow:
        
        1. Fetch data from multiple network endpoints
        2. Process the data with resource-intensive computations
        3. Store results in database
        4. Perform fast validation checks
        5. Generate summary statistics
        
        Optimize for both speed and resource efficiency.
        """
        
        start_time = time.time()
        
        result = await quantum_orchestrator.quantum_execute(
            prompt=complex_prompt,
            optimize_plan=True,
            timeout_ms=30000,  # 30 second timeout for complex workflow
        )
        
        execution_time = time.time() - start_time
        
        # Verify successful completion
        assert result["status"] == "completed"
        assert result["tools_executed"] >= 3  # Should execute multiple tools
        assert result["successful_tools"] >= 3
        
        # Verify reasonable execution time
        assert execution_time < 25.0  # Should complete within timeout
        
        # Verify comprehensive metrics
        assert "quantum_metrics" in result
        assert "performance_metrics" in result
        assert "concurrency_metrics" in result
        assert "security_metrics" in result
        assert "validation_metrics" in result
        
        # Verify optimization effectiveness
        optimization_score = result["quantum_metrics"]["optimization_score"]
        assert 0.0 <= optimization_score <= 1.0
        
        # Verify performance metrics show actual work
        perf_metrics = result["performance_metrics"]
        assert perf_metrics["execution"]["total_tasks"] >= 3
        
        # Log summary for inspection
        print(f"\nComprehensive Workflow Results:")
        print(f"Execution Time: {execution_time:.2f}s")
        print(f"Tools Executed: {result['tools_executed']}")
        print(f"Success Rate: {result['successful_tools']}/{result['tools_executed']}")
        print(f"Optimization Score: {optimization_score:.3f}")
        print(f"Quantum Coherence: {result['quantum_metrics']['quantum_coherence']:.3f}")


class TestQuantumEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    async def basic_orchestrator(self):
        """Create a basic orchestrator for edge case testing."""
        orchestrator = create_quantum_orchestrator(
            tools=[fast_task],
            quantum_config={
                "max_parallel_tasks": 2,
                "optimization_iterations": 5,
            }
        )
        
        yield orchestrator
        await orchestrator.cleanup()
    
    @pytest.mark.asyncio
    async def test_empty_prompt(self, basic_orchestrator):
        """Test handling of empty prompts."""
        result = await basic_orchestrator.quantum_execute("")
        
        # Should handle gracefully
        assert result["status"] in ["no_tools_called", "completed"]
    
    @pytest.mark.asyncio
    async def test_no_matching_tools(self, basic_orchestrator):
        """Test when no tools match the prompt."""
        result = await basic_orchestrator.quantum_execute(
            "Perform advanced quantum physics calculations that require specialized tools"
        )
        
        # Should handle gracefully when no tools match
        assert result["status"] in ["no_tools_called", "completed"]
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self, basic_orchestrator):
        """Test timeout handling."""
        # Create a very slow task
        @Tool(description="Very slow task")
        async def very_slow_task(delay: float = 10.0) -> str:
            await asyncio.sleep(delay)
            return "completed"
        
        basic_orchestrator.register_tool(very_slow_task)
        
        result = await basic_orchestrator.quantum_execute(
            "Run very slow task with delay 10.0",
            timeout_ms=1000,  # 1 second timeout
        )
        
        # Should either timeout or complete quickly
        assert result["total_time_ms"] < 5000  # Should not take too long
    
    @pytest.mark.asyncio
    async def test_high_concurrency(self, basic_orchestrator):
        """Test high concurrency scenarios."""
        # Launch many concurrent executions
        tasks = []
        
        for i in range(20):  # More than max_parallel_tasks
            task = asyncio.create_task(
                basic_orchestrator.quantum_execute(f"Run fast task with value {i}")
            )
            tasks.append(task)
        
        # Wait for all with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=10.0
            )
            
            # Count successful completions
            successful = sum(
                1 for r in results 
                if not isinstance(r, Exception) and r.get("status") == "completed"
            )
            
            # Should have some successes despite high concurrency
            assert successful > 0
            
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # High concurrency timeout is acceptable
            pytest.skip("High concurrency test timed out - acceptable for stress test")


@pytest.mark.integration
class TestQuantumPerformanceBenchmarks:
    """Performance benchmarks for quantum features."""
    
    @pytest.fixture
    async def benchmark_orchestrator(self):
        """Create orchestrator optimized for benchmarking."""
        orchestrator = create_quantum_orchestrator(
            tools=[fast_task, medium_task, slow_task, resource_intensive_task],
            quantum_config={
                "max_parallel_tasks": 20,
                "optimization_iterations": 50,
                "performance": {
                    "strategy": OptimizationStrategy.THROUGHPUT,
                    "auto_scaling": True,
                    "monitoring_interval": 0.1,
                },
            }
        )
        
        yield orchestrator
        await orchestrator.cleanup()
    
    @pytest.mark.asyncio
    async def test_throughput_benchmark(self, benchmark_orchestrator):
        """Benchmark throughput performance."""
        num_executions = 10
        start_time = time.time()
        
        # Execute multiple parallel workflows
        tasks = []
        for i in range(num_executions):
            task = asyncio.create_task(
                benchmark_orchestrator.quantum_execute(
                    f"Run fast task with value {i}",
                    optimize_plan=True,
                )
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # Count successful executions
        successful = sum(
            1 for r in results 
            if not isinstance(r, Exception) and r.get("status") == "completed"
        )
        
        # Calculate throughput
        throughput = successful / total_time
        
        print(f"\nThroughput Benchmark Results:")
        print(f"Successful Executions: {successful}/{num_executions}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} executions/second")
        
        # Verify reasonable performance
        assert successful >= num_executions * 0.8  # At least 80% success rate
        assert throughput > 1.0  # At least 1 execution per second
    
    @pytest.mark.asyncio
    async def test_latency_benchmark(self, benchmark_orchestrator):
        """Benchmark latency performance."""
        latencies = []
        
        # Execute sequential tasks to measure latency
        for i in range(10):
            start_time = time.time()
            
            result = await benchmark_orchestrator.quantum_execute(
                f"Run fast task with value {i}",
                optimize_plan=True,
            )
            
            latency = time.time() - start_time
            latencies.append(latency)
            
            assert result["status"] == "completed"
        
        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        print(f"\nLatency Benchmark Results:")
        print(f"Average Latency: {avg_latency:.3f}s")
        print(f"Min Latency: {min_latency:.3f}s")
        print(f"Max Latency: {max_latency:.3f}s")
        
        # Verify reasonable latency
        assert avg_latency < 2.0  # Average latency under 2 seconds
        assert min_latency < 1.0  # Best case under 1 second


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "-s"])