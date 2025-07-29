"""Shared test configuration and fixtures."""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock

import pytest_asyncio
from async_toolformer import AsyncOrchestrator, Tool


@pytest_asyncio.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    client = Mock()
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def sample_tool():
    """Sample tool for testing."""
    @Tool(description="Test tool that returns a greeting")
    async def test_tool(name: str = "World") -> str:
        await asyncio.sleep(0.01)  # Simulate async work
        return f"Hello, {name}!"
    
    return test_tool


@pytest.fixture
def orchestrator(mock_llm_client, sample_tool):
    """Basic orchestrator instance for testing."""
    return AsyncOrchestrator(
        llm_client=mock_llm_client,
        tools=[sample_tool],
        max_parallel=5
    )


@pytest.fixture
def slow_tool():
    """Slow tool for timeout testing."""
    @Tool(description="Slow test tool")
    async def slow_test_tool() -> str:
        await asyncio.sleep(2.0)
        return "Finally done!"
    
    return slow_test_tool


# Test data fixtures
@pytest.fixture
def sample_llm_response():
    """Sample LLM response with tool calls."""
    return {
        "choices": [{
            "message": {
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "test_tool",
                            "arguments": '{"name": "Alice"}'
                        }
                    },
                    {
                        "id": "call_2", 
                        "function": {
                            "name": "test_tool",
                            "arguments": '{"name": "Bob"}'
                        }
                    }
                ]
            }
        }]
    }