"""
Advanced tests for Next Generation Orchestrator with autonomous capabilities.

Tests cover:
- Autonomous hypothesis testing
- Self-healing mechanisms  
- Research experiment execution
- Global deployment optimization
- Real-time adaptation
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from async_toolformer.config import OrchestratorConfig
from async_toolformer.next_gen_orchestrator import (
    AutoHypothesis,
    NextGenOrchestrator,
    create_next_gen_orchestrator,
)
from async_toolformer.tools import Tool


@pytest.fixture
async def mock_llm_client():
    """Mock LLM client for testing."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=MagicMock(
        choices=[MagicMock(message=MagicMock(content="test response", tool_calls=[]))]
    ))
    return client


@pytest.fixture
async def sample_tools():
    """Sample tools for testing."""

    @Tool(description="Test search tool")
    async def test_search(query: str) -> str:
        await asyncio.sleep(0.1)
        return f"Search results for: {query}"

    @Tool(description="Test analysis tool")
    async def test_analysis(data: str) -> dict:
        await asyncio.sleep(0.05)
        return {"analysis": f"Analyzed: {data}", "score": 0.85}

    return [test_search, test_analysis]


@pytest.fixture
async def next_gen_orchestrator(mock_llm_client, sample_tools):
    """Next generation orchestrator for testing."""
    config = OrchestratorConfig()
    config.max_parallel_tools = 10

    orchestrator = NextGenOrchestrator(
        llm_client=mock_llm_client,
        tools=sample_tools,
        config=config,
        research_mode=True,
        global_regions=["us-east-1", "eu-west-1", "ap-southeast-1"],
    )

    await orchestrator.initialize_autonomous_systems()
    return orchestrator


class TestAutoHypothesis:
    """Test autonomous hypothesis system."""

    def test_hypothesis_creation(self):
        """Test hypothesis creation with success criteria."""

        hypothesis = AutoHypothesis(
            hypothesis="Performance will improve with parallel execution",
            success_criteria={
                "execution_time": {"improvement": 0.2},  # 20% improvement
                "throughput": {"min_value": 10.0},
            },
            baseline_metrics={"execution_time": 1.0, "throughput": 8.0},
        )

        assert hypothesis.hypothesis == "Performance will improve with parallel execution"
        assert not hypothesis.validated
        assert len(hypothesis.test_results) == 0

    def test_hypothesis_validation_success(self):
        """Test successful hypothesis validation."""

        hypothesis = AutoHypothesis(
            hypothesis="Test hypothesis",
            success_criteria={
                "metric1": {"improvement": 0.1},  # 10% improvement
                "metric2": {"min_value": 5.0},
            },
            baseline_metrics={"metric1": 10.0, "metric2": 4.0},
        )

        # Add test results that meet criteria
        hypothesis.add_test_result({"metric1": 11.5, "metric2": 6.0})  # 15% improvement, above min
        hypothesis.add_test_result({"metric1": 11.2, "metric2": 5.5})  # 12% improvement, above min
        validated = hypothesis.add_test_result({"metric1": 11.0, "metric2": 5.2})  # 10% improvement, above min

        assert validated
        assert hypothesis.validated
        assert len(hypothesis.test_results) == 3

    def test_hypothesis_validation_failure(self):
        """Test failed hypothesis validation."""

        hypothesis = AutoHypothesis(
            hypothesis="Test hypothesis",
            success_criteria={
                "metric1": {"improvement": 0.2},  # 20% improvement required
            },
            baseline_metrics={"metric1": 10.0},
        )

        # Add test results that don't meet criteria
        hypothesis.add_test_result({"metric1": 10.5})  # Only 5% improvement
        hypothesis.add_test_result({"metric1": 10.8})  # Only 8% improvement
        validated = hypothesis.add_test_result({"metric1": 10.3})  # Only 3% improvement

        assert not validated
        assert not hypothesis.validated


class TestNextGenOrchestrator:
    """Test Next Generation Orchestrator functionality."""

    @pytest.mark.asyncio
    async def test_initialization(self, next_gen_orchestrator):
        """Test orchestrator initialization."""

        assert next_gen_orchestrator.research_mode
        assert len(next_gen_orchestrator.global_regions) == 3
        assert "us-east-1" in next_gen_orchestrator.global_regions
        assert len(next_gen_orchestrator.active_hypotheses) == 0
        assert len(next_gen_orchestrator.performance_baselines) > 0

    @pytest.mark.asyncio
    async def test_performance_baseline_establishment(self, next_gen_orchestrator):
        """Test performance baseline establishment."""

        baselines = next_gen_orchestrator.performance_baselines

        assert "simple_execution" in baselines
        assert "parallel_execution" in baselines
        assert "resource_utilization" in baselines

        # All baselines should be positive numbers
        for name, value in baselines.items():
            assert isinstance(value, (int, float))
            assert value > 0

    @pytest.mark.asyncio
    async def test_autonomous_execution_basic(self, next_gen_orchestrator):
        """Test basic autonomous execution."""

        result = await next_gen_orchestrator.execute_with_autonomous_optimization(
            prompt="Test autonomous execution",
            enable_hypothesis_testing=False,
        )

        assert "result" in result
        assert "metrics" in result
        assert "execution_time" in result
        assert "region" in result
        assert "autonomous_insights" in result

        # Check metrics structure
        metrics = result["metrics"]
        assert "execution_time" in metrics
        assert "throughput" in metrics
        assert "resource_utilization" in metrics
        assert "success_rate" in metrics

        assert metrics["success_rate"] == 1.0  # Should succeed

    @pytest.mark.asyncio
    async def test_hypothesis_testing_flow(self, next_gen_orchestrator):
        """Test autonomous hypothesis testing flow."""

        result = await next_gen_orchestrator.execute_with_autonomous_optimization(
            prompt="Complex task requiring optimization",
            enable_hypothesis_testing=True,
        )

        # Should have created a hypothesis
        assert len(next_gen_orchestrator.active_hypotheses) > 0

        # Check hypothesis validation status
        assert "hypothesis_validated" in result

        # Get the created hypothesis
        execution_id = list(next_gen_orchestrator.active_hypotheses.keys())[0]
        hypothesis = next_gen_orchestrator.active_hypotheses[execution_id]

        assert hypothesis.hypothesis is not None
        assert len(hypothesis.test_results) == 1  # One test result added

    @pytest.mark.asyncio
    async def test_region_selection(self, next_gen_orchestrator):
        """Test optimal region selection."""

        # Set up different performance metrics for regions
        next_gen_orchestrator.region_performance["us-east-1"] = {
            "latency": 0.1,
            "availability": 0.99,
            "load": 0.3,
            "active": True,
        }
        next_gen_orchestrator.region_performance["eu-west-1"] = {
            "latency": 0.2,
            "availability": 0.95,
            "load": 0.8,
            "active": True,
        }
        next_gen_orchestrator.region_performance["ap-southeast-1"] = {
            "latency": 0.15,
            "availability": 0.97,
            "load": 0.1,
            "active": True,
        }

        optimal_region = await next_gen_orchestrator._select_optimal_region()

        # Should select the best performing region (likely us-east-1 or ap-southeast-1)
        assert optimal_region in ["us-east-1", "ap-southeast-1"]

    @pytest.mark.asyncio
    async def test_self_healing_mechanism(self, next_gen_orchestrator):
        """Test self-healing when errors occur."""

        # Mock a failing operation
        with patch.object(next_gen_orchestrator, '_execute_with_monitoring', side_effect=Exception("Test error")):
            # Should trigger self-healing attempt
            await next_gen_orchestrator._attempt_self_healing(
                Exception("Test error"),
                "test prompt",
                None
            )

            # Check that adaptation history was updated
            assert len(next_gen_orchestrator.adaptation_history) > 0

            last_adaptation = next_gen_orchestrator.adaptation_history[-1]
            assert "healing_strategy" in last_adaptation
            assert "error_type" in last_adaptation

    @pytest.mark.asyncio
    async def test_research_experiment_execution(self, next_gen_orchestrator):
        """Test research experiment execution."""

        # Create test dataset
        dataset = [
            {"task": "search", "complexity": 1},
            {"task": "analysis", "complexity": 3},
            {"task": "synthesis", "complexity": 5},
        ]

        # Run experiment
        results = await next_gen_orchestrator.run_research_experiment(
            "quantum_optimization",
            dataset
        )

        assert "experiment_name" in results
        assert "baseline_results" in results
        assert "novel_results" in results
        assert "statistical_analysis" in results
        assert "dataset_size" in results

        # Check statistical analysis structure
        stats = results["statistical_analysis"]
        assert isinstance(stats, dict)

        # Should have some metrics analyzed
        if stats:
            for metric_name, analysis in stats.items():
                assert "baseline_mean" in analysis
                assert "novel_mean" in analysis
                assert "improvement" in analysis
                assert "statistically_significant" in analysis

    @pytest.mark.asyncio
    async def test_autonomous_adaptation(self, next_gen_orchestrator):
        """Test autonomous system adaptation."""

        # Simulate poor performance metrics
        poor_metrics = {
            "throughput": 0.1,  # Much lower than baseline
            "resource_utilization": 0.9,
            "success_rate": 0.8,
        }

        original_parallelism = next_gen_orchestrator.config.max_parallel_tools

        await next_gen_orchestrator._autonomous_adaptation(poor_metrics)

        # Should have adapted the system
        assert len(next_gen_orchestrator.adaptation_history) > 0

        # Check if parallelism was adjusted
        # (This depends on the specific adaptation logic)
        adaptation = next_gen_orchestrator.adaptation_history[-1]
        assert "adaptation" in adaptation
        assert "trigger_value" in adaptation

    @pytest.mark.asyncio
    async def test_autonomous_status_reporting(self, next_gen_orchestrator):
        """Test autonomous system status reporting."""

        # Add some test data
        await next_gen_orchestrator.execute_with_autonomous_optimization(
            "test task",
            enable_hypothesis_testing=True,
        )

        status = await next_gen_orchestrator.get_autonomous_status()

        assert "system_health" in status
        assert "active_hypotheses" in status
        assert "validated_hypotheses" in status
        assert "adaptation_count" in status
        assert "recent_adaptations" in status
        assert "performance_baselines" in status
        assert "global_regions" in status
        assert "research_experiments" in status

        # Check data types
        assert isinstance(status["active_hypotheses"], int)
        assert isinstance(status["validated_hypotheses"], int)
        assert isinstance(status["adaptation_count"], int)
        assert isinstance(status["recent_adaptations"], list)
        assert isinstance(status["performance_baselines"], dict)
        assert isinstance(status["global_regions"], dict)

    @pytest.mark.asyncio
    async def test_multiple_hypothesis_management(self, next_gen_orchestrator):
        """Test management of multiple concurrent hypotheses."""

        # Execute multiple tasks with hypothesis testing
        tasks = [
            "Simple task requiring basic optimization",
            "Complex task requiring advanced optimization",
            "Resource-intensive task requiring scaling",
        ]

        results = []
        for task in tasks:
            result = await next_gen_orchestrator.execute_with_autonomous_optimization(
                prompt=task,
                enable_hypothesis_testing=True,
            )
            results.append(result)

        # Should have created multiple hypotheses
        assert len(next_gen_orchestrator.active_hypotheses) == len(tasks)

        # Each result should have hypothesis information
        for result in results:
            assert "hypothesis_validated" in result
            assert "autonomous_insights" in result

    @pytest.mark.asyncio
    async def test_insights_generation(self, next_gen_orchestrator):
        """Test autonomous insights generation."""

        # Test with various metric scenarios
        test_scenarios = [
            {
                "execution_time": 0.5,
                "throughput": 2.0,
                "resource_utilization": 0.2,
                "resource_efficiency": 0.9,
            },
            {
                "execution_time": 3.0,
                "throughput": 0.3,
                "resource_utilization": 0.9,
                "resource_efficiency": 0.4,
            },
        ]

        for metrics in test_scenarios:
            insights = await next_gen_orchestrator._generate_insights(metrics)

            assert "performance_category" in insights
            assert "resource_efficiency" in insights
            assert "recommendations" in insights

            assert insights["performance_category"] in ["optimal", "suboptimal"]
            assert insights["resource_efficiency"] in ["high", "medium", "low"]
            assert isinstance(insights["recommendations"], list)

    @pytest.mark.asyncio
    async def test_error_recovery_with_adaptation(self, next_gen_orchestrator, sample_tools):
        """Test error recovery with system adaptation."""

        # Create a tool that fails initially
        failure_count = 0

        @Tool(description="Failing tool for testing")
        async def failing_tool(data: str) -> str:
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise Exception(f"Simulated failure {failure_count}")
            return f"Success after {failure_count} attempts: {data}"

        # Add the failing tool
        next_gen_orchestrator.registry.register(failing_tool)

        # Execute with the failing tool - should trigger self-healing
        with patch.object(next_gen_orchestrator.llm_integration, 'generate_response') as mock_generate:
            mock_generate.return_value = {
                "content": "I'll use the failing tool",
                "tool_calls": [{"name": "failing_tool", "arguments": {"data": "test"}}]
            }

            result = await next_gen_orchestrator.execute_with_autonomous_optimization(
                prompt="Use the failing tool",
                tools=[failing_tool],
            )

            # Should have adaptation history from error handling
            assert len(next_gen_orchestrator.adaptation_history) > 0


class TestCreateNextGenOrchestrator:
    """Test next generation orchestrator creation function."""

    def test_create_with_defaults(self):
        """Test creation with default parameters."""

        orchestrator = create_next_gen_orchestrator()

        assert isinstance(orchestrator, NextGenOrchestrator)
        assert not orchestrator.research_mode  # Default is False
        assert orchestrator.config.max_parallel_tools == 50
        assert orchestrator.config.enable_caching
        assert orchestrator.config.enable_speculation

    def test_create_with_research_mode(self):
        """Test creation with research mode enabled."""

        orchestrator = create_next_gen_orchestrator(
            research_mode=True,
            global_regions=["us-west-2", "eu-central-1"],
        )

        assert orchestrator.research_mode
        assert orchestrator.global_regions == ["us-west-2", "eu-central-1"]

    def test_create_with_custom_config(self):
        """Test creation with custom configuration."""

        orchestrator = create_next_gen_orchestrator(
            max_parallel_tools=20,
            enable_caching=False,
        )

        assert orchestrator.config.max_parallel_tools == 20
        assert not orchestrator.config.enable_caching


@pytest.mark.asyncio
async def test_integration_autonomous_workflow(mock_llm_client, sample_tools):
    """Integration test for complete autonomous workflow."""

    # Create orchestrator with research mode
    orchestrator = create_next_gen_orchestrator(
        research_mode=True,
        global_regions=["us-east-1", "eu-west-1"],
    )

    # Initialize with mock client and tools
    orchestrator.llm_integration = MagicMock()
    orchestrator.llm_integration.generate_response = AsyncMock(return_value={
        "content": "I'll search and analyze the data",
        "tool_calls": [
            {"name": "test_search", "arguments": {"query": "autonomous systems"}},
            {"name": "test_analysis", "arguments": {"data": "search results"}},
        ]
    })

    for tool in sample_tools:
        orchestrator.registry.register(tool)

    await orchestrator.initialize_autonomous_systems()

    # Execute autonomous workflow
    result = await orchestrator.execute_with_autonomous_optimization(
        prompt="Research and analyze autonomous system capabilities",
        enable_hypothesis_testing=True,
    )

    # Verify complete workflow execution
    assert result["result"] is not None
    assert result["execution_time"] > 0
    assert result["region"] in orchestrator.global_regions
    assert "autonomous_insights" in result

    # Verify hypothesis was created and tested
    assert len(orchestrator.active_hypotheses) > 0

    # Verify system status is comprehensive
    status = await orchestrator.get_autonomous_status()
    assert status["active_hypotheses"] > 0
    assert len(status["performance_baselines"]) > 0
    assert len(status["global_regions"]) == 2

    # Run research experiment
    dataset = [{"complexity": i} for i in range(5)]
    experiment_results = await orchestrator.run_research_experiment(
        "quantum_optimization",
        dataset
    )

    assert experiment_results["experiment_name"] == "quantum_optimization"
    assert experiment_results["dataset_size"] == 5
    assert "statistical_analysis" in experiment_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
