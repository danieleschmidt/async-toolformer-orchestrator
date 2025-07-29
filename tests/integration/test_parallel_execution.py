"""Integration tests for parallel tool execution."""

import pytest
import asyncio
import time
from unittest.mock import AsyncMock

from async_toolformer import AsyncOrchestrator, Tool


@pytest.mark.integration
class TestParallelExecution:
    """Test parallel execution behavior."""

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance(self, mock_llm_client):
        """Test that parallel execution is faster than sequential."""
        @Tool(description="Slow task")
        async def slow_task(delay: float = 0.1) -> str:
            await asyncio.sleep(delay)
            return f"Task completed after {delay}s"

        # Mock LLM response for parallel calls
        mock_llm_client.chat.completions.create.return_value = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"id": "1", "function": {"name": "slow_task", "arguments": '{"delay": 0.1}'}},
                        {"id": "2", "function": {"name": "slow_task", "arguments": '{"delay": 0.1}'}},
                        {"id": "3", "function": {"name": "slow_task", "arguments": '{"delay": 0.1}'}},
                    ]
                }
            }]
        }

        orchestrator = AsyncOrchestrator(
            llm_client=mock_llm_client,
            tools=[slow_task],
            max_parallel=10
        )

        start_time = time.time()
        await orchestrator.execute("Run three slow tasks")
        parallel_time = time.time() - start_time

        # Parallel execution should be much faster than 0.3s (3 * 0.1s)
        assert parallel_time < 0.25, f"Parallel execution took {parallel_time}s, expected < 0.25s"

    @pytest.mark.asyncio
    async def test_parallel_limit_enforcement(self, mock_llm_client):
        """Test that parallel limits are enforced."""
        execution_tracker = []

        @Tool(description="Tracked task")
        async def tracked_task(task_id: int) -> str:
            execution_tracker.append(f"start_{task_id}")
            await asyncio.sleep(0.1)
            execution_tracker.append(f"end_{task_id}")
            return f"Task {task_id} done"

        # Create many parallel calls
        mock_llm_client.chat.completions.create.return_value = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"id": str(i), "function": {"name": "tracked_task", "arguments": f'{{"task_id": {i}}}'}}
                        for i in range(10)
                    ]
                }
            }]
        }

        orchestrator = AsyncOrchestrator(
            llm_client=mock_llm_client,
            tools=[tracked_task],
            max_parallel=3  # Limit to 3 concurrent
        )

        await orchestrator.execute("Run ten tasks")

        # Check that no more than 3 tasks were running simultaneously
        concurrent_count = 0
        max_concurrent = 0
        
        for event in execution_tracker:
            if event.startswith("start_"):
                concurrent_count += 1
                max_concurrent = max(max_concurrent, concurrent_count)
            elif event.startswith("end_"):
                concurrent_count -= 1

        assert max_concurrent <= 3, f"Max concurrent was {max_concurrent}, expected <= 3"

    @pytest.mark.asyncio
    async def test_error_isolation(self, mock_llm_client):
        """Test that errors in one tool don't affect others."""
        results = []

        @Tool(description="Success task")
        async def success_task(task_id: int) -> str:
            await asyncio.sleep(0.05)
            result = f"Success {task_id}"
            results.append(result)
            return result

        @Tool(description="Failing task")
        async def failing_task(task_id: int) -> str:
            await asyncio.sleep(0.05)
            raise ValueError(f"Task {task_id} failed")

        mock_llm_client.chat.completions.create.return_value = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {"id": "1", "function": {"name": "success_task", "arguments": '{"task_id": 1}'}},
                        {"id": "2", "function": {"name": "failing_task", "arguments": '{"task_id": 2}'}},
                        {"id": "3", "function": {"name": "success_task", "arguments": '{"task_id": 3}'}},
                    ]
                }
            }]
        }

        orchestrator = AsyncOrchestrator(
            llm_client=mock_llm_client,
            tools=[success_task, failing_task]
        )

        # Should not raise exception, but handle errors gracefully
        await orchestrator.execute("Run mixed success/failure tasks")

        # Successful tasks should still complete
        assert "Success 1" in results
        assert "Success 3" in results
        assert len(results) == 2  # Only successful tasks