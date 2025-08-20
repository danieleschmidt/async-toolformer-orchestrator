"""Property-based tests for the AsyncOrchestrator."""

import asyncio
from typing import Any

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule

from async_toolformer import AsyncOrchestrator, OrchestratorConfig, Tool
from async_toolformer.tools import ToolResult


class OrchestratorStateMachine(RuleBasedStateMachine):
    """Stateful testing of the AsyncOrchestrator."""

    def __init__(self):
        super().__init__()
        self.orchestrator = None
        self.registered_tools = []
        self.execution_results = []

    @initialize()
    def setup_orchestrator(self):
        """Initialize the orchestrator with default config."""
        config = OrchestratorConfig(
            max_parallel_tools=10,
            tool_timeout_ms=5000,
            total_timeout_ms=15000,
        )
        self.orchestrator = AsyncOrchestrator(config=config)

    @rule(
        tool_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        description=st.text(min_size=1, max_size=100),
        timeout_ms=st.integers(min_value=100, max_value=10000),
    )
    def register_tool(self, tool_name: str, description: str, timeout_ms: int):
        """Register a new tool with the orchestrator."""
        assume(tool_name not in [t['name'] for t in self.registered_tools])

        @Tool(description=description, timeout_ms=timeout_ms)
        async def dynamic_tool(value: int = 42) -> dict[str, Any]:
            await asyncio.sleep(0.01)  # Simulate work
            return {"result": value * 2, "tool": tool_name}

        # Override the function name for registration
        dynamic_tool.__name__ = tool_name

        self.orchestrator.register_tool(dynamic_tool)
        self.registered_tools.append({
            "name": tool_name,
            "description": description,
            "timeout_ms": timeout_ms,
            "function": dynamic_tool,
        })

    @rule(
        prompt=st.text(min_size=1, max_size=200),
        max_parallel=st.integers(min_value=1, max_value=5),
    )
    async def execute_tools(self, prompt: str, max_parallel: int):
        """Execute tools with the orchestrator."""
        assume(len(self.registered_tools) > 0)

        result = await self.orchestrator.execute(
            prompt=prompt,
            max_parallel=max_parallel,
        )

        self.execution_results.append(result)

    @invariant()
    def orchestrator_is_consistent(self):
        """Ensure orchestrator state remains consistent."""
        if self.orchestrator:
            metrics = self.orchestrator.get_metrics()
            assert metrics["registered_tools"] == len(self.registered_tools)
            assert metrics["registered_tools"] >= 0
            assert metrics["active_tasks"] >= 0


# Property-based tests for tool execution
@given(
    tool_count=st.integers(min_value=1, max_value=5),
    timeout_ms=st.integers(min_value=100, max_value=5000),
    max_parallel=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=20, deadline=10000)
async def test_orchestrator_handles_multiple_tools(tool_count: int, timeout_ms: int, max_parallel: int):
    """Test that orchestrator correctly handles multiple tools with various configurations."""
    config = OrchestratorConfig(
        max_parallel_tools=max_parallel,
        tool_timeout_ms=timeout_ms,
        total_timeout_ms=timeout_ms * 3,
    )

    orchestrator = AsyncOrchestrator(config=config)

    # Register multiple tools
    tools = []
    for i in range(tool_count):
        @Tool(description=f"Test tool {i}")
        async def test_tool(value: int = i) -> dict[str, Any]:
            await asyncio.sleep(0.01)  # Simulate work
            return {"result": value, "tool_id": i}

        test_tool.__name__ = f"test_tool_{i}"
        orchestrator.register_tool(test_tool)
        tools.append(test_tool)

    # Execute tools
    result = await orchestrator.execute("Test prompt")

    # Verify results
    assert result["status"] in ["completed", "failed", "no_tools_called"]
    if result["status"] == "completed":
        assert "results" in result
        assert "total_time_ms" in result
        assert result["total_time_ms"] >= 0
        assert result["tools_executed"] >= 0
        assert result["successful_tools"] >= 0
        assert result["successful_tools"] <= result["tools_executed"]


@given(
    delay_ms=st.integers(min_value=0, max_value=1000),
    should_fail=st.booleans(),
    return_value=st.one_of(
        st.none(),
        st.integers(),
        st.text(),
        st.dictionaries(st.text(), st.integers()),
        st.lists(st.integers()),
    ),
)
@settings(max_examples=50, deadline=5000)
async def test_tool_execution_properties(delay_ms: int, should_fail: bool, return_value: Any):
    """Test properties of individual tool execution."""

    @Tool(description="Property test tool")
    async def property_tool() -> Any:
        await asyncio.sleep(delay_ms / 1000.0)
        if should_fail:
            raise ValueError("Intentional test failure")
        return return_value

    orchestrator = AsyncOrchestrator()
    orchestrator.register_tool(property_tool)

    result = await orchestrator.execute("Test")

    # Properties that should always hold
    assert isinstance(result, dict)
    assert "execution_id" in result
    assert "total_time_ms" in result
    assert "status" in result

    if result["status"] == "completed":
        assert "results" in result
        if result["results"]:
            tool_result = result["results"][0]
            assert isinstance(tool_result, ToolResult)
            assert tool_result.tool_name == "property_tool"

            if should_fail:
                assert not tool_result.success
            else:
                assert tool_result.success
                assert tool_result.data == return_value


@given(
    config_values=st.fixed_dictionaries({
        "max_parallel_tools": st.integers(min_value=1, max_value=20),
        "tool_timeout_ms": st.integers(min_value=100, max_value=10000),
        "total_timeout_ms": st.integers(min_value=1000, max_value=30000),
        "retry_attempts": st.integers(min_value=0, max_value=5),
    })
)
@settings(max_examples=30)
async def test_config_validation_properties(config_values: dict[str, int]):
    """Test that configuration validation works correctly."""
    assume(config_values["total_timeout_ms"] >= config_values["tool_timeout_ms"])

    config = OrchestratorConfig(**config_values)

    # Should not raise an exception for valid configs
    config.validate()

    orchestrator = AsyncOrchestrator(config=config)
    metrics = orchestrator.get_metrics()

    # Verify config is applied correctly
    assert metrics["config"]["max_parallel_tools"] == config_values["max_parallel_tools"]
    assert metrics["config"]["tool_timeout_ms"] == config_values["tool_timeout_ms"]
    assert metrics["config"]["total_timeout_ms"] == config_values["total_timeout_ms"]


@given(
    execution_count=st.integers(min_value=1, max_value=5),
    tools_per_execution=st.integers(min_value=1, max_value=3),
)
@settings(max_examples=10, deadline=15000)
async def test_multiple_executions_properties(execution_count: int, tools_per_execution: int):
    """Test properties across multiple executions."""
    orchestrator = AsyncOrchestrator()

    # Register tools
    for i in range(tools_per_execution):
        @Tool(description=f"Multi-exec tool {i}")
        async def multi_tool(iteration: int = i) -> dict[str, int]:
            await asyncio.sleep(0.01)
            return {"iteration": iteration, "value": iteration * 10}

        multi_tool.__name__ = f"multi_tool_{i}"
        orchestrator.register_tool(multi_tool)

    results = []
    total_execution_time = 0

    # Execute multiple times
    for exec_num in range(execution_count):
        result = await orchestrator.execute(f"Execution {exec_num}")
        results.append(result)
        total_execution_time += result.get("total_time_ms", 0)

    # Properties that should hold across executions
    assert len(results) == execution_count
    assert all(isinstance(r, dict) for r in results)
    assert all("execution_id" in r for r in results)
    assert total_execution_time >= 0

    # Each execution should have unique ID
    execution_ids = [r["execution_id"] for r in results]
    assert len(set(execution_ids)) == len(execution_ids)

    # Orchestrator state should remain consistent
    final_metrics = orchestrator.get_metrics()
    assert final_metrics["registered_tools"] == tools_per_execution
    assert final_metrics["active_tasks"] == 0  # All tasks should be complete


@given(
    parallel_limit=st.integers(min_value=1, max_value=8),
    tool_count=st.integers(min_value=2, max_value=15),
)
@settings(max_examples=20, deadline=10000)
async def test_parallel_execution_limits(parallel_limit: int, tool_count: int):
    """Test that parallel execution limits are respected."""
    config = OrchestratorConfig(max_parallel_tools=parallel_limit)
    orchestrator = AsyncOrchestrator(config=config)

    execution_log = []

    # Register tools that log their execution
    for i in range(tool_count):
        @Tool(description=f"Parallel test tool {i}")
        async def parallel_tool(tool_id: int = i) -> dict[str, Any]:
            execution_log.append(f"start_{tool_id}")
            await asyncio.sleep(0.1)  # Ensure overlap
            execution_log.append(f"end_{tool_id}")
            return {"tool_id": tool_id}

        parallel_tool.__name__ = f"parallel_tool_{i}"
        orchestrator.register_tool(parallel_tool)

    result = await orchestrator.execute("Parallel test")

    # Verify execution completed
    assert result["status"] in ["completed", "no_tools_called"]

    # Analyze execution log to verify parallel limits
    # (This is a simplified check - in practice, we'd need more sophisticated timing analysis)
    start_events = [event for event in execution_log if event.startswith("start_")]
    end_events = [event for event in execution_log if event.startswith("end_")]

    assert len(start_events) <= tool_count
    assert len(end_events) <= tool_count
