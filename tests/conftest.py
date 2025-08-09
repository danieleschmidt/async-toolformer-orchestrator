"""Shared test configuration and fixtures."""

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from _pytest.logging import LogCaptureFixture

# Configure test logging
logging.getLogger("async_toolformer").setLevel(logging.DEBUG)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def log_capture(caplog: LogCaptureFixture) -> LogCaptureFixture:
    """Capture logs during test execution."""
    caplog.set_level(logging.DEBUG, logger="async_toolformer")
    return caplog


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Mock OpenAI client for testing."""
    client = AsyncMock()
    
    # Mock chat completions response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                tool_calls=[
                    MagicMock(
                        id="call_123",
                        function=MagicMock(
                            name="test_tool",
                            arguments='{"arg1": "value1"}'
                        )
                    )
                ]
            )
        )
    ]
    
    client.chat.completions.create.return_value = mock_response
    return client


@pytest.fixture
def mock_anthropic_client() -> AsyncMock:
    """Mock Anthropic client for testing."""
    client = AsyncMock()
    
    # Mock message response
    mock_response = MagicMock()
    mock_response.content = [
        MagicMock(
            type="tool_use",
            id="toolu_123",
            name="test_tool",
            input={"arg1": "value1"}
        )
    ]
    
    client.messages.create.return_value = mock_response
    return client


@pytest.fixture
def mock_llm_client(mock_openai_client) -> AsyncMock:
    """Generic mock LLM client for backward compatibility."""
    return mock_openai_client


@pytest.fixture
def mock_redis() -> AsyncMock:
    """Mock Redis client for testing."""
    redis = AsyncMock()
    
    # Mock rate limiting methods
    redis.get.return_value = None
    redis.set.return_value = True
    redis.expire.return_value = True
    redis.incr.return_value = 1
    redis.pipeline.return_value = redis
    redis.execute.return_value = [True, True, 1]
    
    return redis


@pytest.fixture
async def sample_tools() -> List[MagicMock]:
    """Create sample tools for testing."""
    tools = []
    
    # Fast tool
    fast_tool = AsyncMock()
    fast_tool.__name__ = "fast_tool"
    fast_tool.return_value = "fast_result"
    fast_tool.description = "A fast test tool"
    tools.append(fast_tool)
    
    # Slow tool
    slow_tool = AsyncMock()
    slow_tool.__name__ = "slow_tool"
    slow_tool.side_effect = lambda: asyncio.sleep(0.1) or "slow_result"
    slow_tool.description = "A slow test tool"
    tools.append(slow_tool)
    
    # Error tool
    error_tool = AsyncMock()
    error_tool.__name__ = "error_tool"
    error_tool.side_effect = Exception("Test error")
    error_tool.description = "A tool that raises errors"
    tools.append(error_tool)
    
    return tools


@pytest.fixture
def orchestrator_config() -> Dict:
    """Default orchestrator configuration for testing."""
    return {
        "max_parallel_tools": 5,
        "tool_timeout_ms": 1000,
        "total_timeout_ms": 5000,
        "enable_speculation": False,
        "enable_tracing": False,
        "debug": True
    }


@pytest.fixture
async def mock_orchestrator(
    mock_openai_client: AsyncMock,
    mock_redis: AsyncMock,
    sample_tools: List[MagicMock],
    orchestrator_config: Dict
) -> AsyncMock:
    """Create a mock orchestrator for testing."""
    from async_toolformer import AsyncOrchestrator
    
    orchestrator = AsyncMock(spec=AsyncOrchestrator)
    orchestrator.llm_client = mock_openai_client
    orchestrator.redis_client = mock_redis
    orchestrator.tools = sample_tools
    orchestrator.config = orchestrator_config
    
    # Mock execution methods
    orchestrator.execute.return_value = {
        "results": ["fast_result", "slow_result"],
        "execution_time": 0.15,
        "parallel_count": 2
    }
    
    orchestrator.stream_execute.return_value = AsyncMock()
    
    return orchestrator


@pytest.fixture
def sample_api_responses() -> Dict:
    """Sample API responses for testing."""
    return {
        "openai_chat_completion": {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query": "Python asyncio"}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        },
        "anthropic_message": {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_abc123",
                    "name": "web_search",
                    "input": {"query": "Python asyncio"}
                }
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50
            }
        }
    }


@pytest.fixture
def performance_metrics() -> Dict:
    """Expected performance metrics for testing."""
    return {
        "tool_execution_rate": 10.5,
        "average_duration": 0.25,
        "error_rate": 0.05,
        "speculation_hit_rate": 0.75,
        "parallel_efficiency": 0.85,
        "memory_usage_mb": 128.5,
        "rate_limit_hits": 2
    }


@pytest.fixture
def test_environment() -> str:
    """Get test environment from environment variable."""
    return os.getenv("TEST_ENV", "testing")


@pytest.fixture
def api_keys() -> Dict[str, Optional[str]]:
    """API keys for E2E testing (if available)."""
    return {
        "openai": os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY")
    }


@pytest.fixture
def skip_if_no_api_keys(api_keys: Dict[str, Optional[str]]):
    """Skip test if API keys are not available."""
    if not any(api_keys.values()):
        pytest.skip("API keys not available for E2E testing")


@pytest.fixture
async def cleanup_redis(mock_redis: AsyncMock):
    """Clean up Redis state after tests."""
    yield
    if hasattr(mock_redis, 'flushdb'):
        await mock_redis.flushdb()


@pytest.fixture
def benchmark_config() -> Dict:
    """Configuration for benchmark tests."""
    return {
        "iterations": 10,
        "warmup_iterations": 2,
        "timeout_seconds": 30,
        "parallel_workers": [1, 2, 5, 10],
        "tool_counts": [1, 5, 10, 25],
        "expected_speedup": 4.0
    }


# Marks for test categorization
pytestmark = [
    pytest.mark.asyncio,
]


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )
    config.addinivalue_line(
        "markers", "security: marks tests as security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on location."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
            item.add_marker(pytest.mark.slow)
        
        # Add performance marker for benchmark tests
        if "benchmark" in item.name or "perf" in item.name:
            item.add_marker(pytest.mark.benchmark)
            item.add_marker(pytest.mark.performance)
        
        # Add security marker for security tests
        if "security" in item.name or "auth" in item.name:
            item.add_marker(pytest.mark.security)


@pytest.fixture(autouse=True)
def setup_test_logging(caplog: LogCaptureFixture):
    """Automatically set up logging for all tests."""
    caplog.set_level(logging.DEBUG, logger="async_toolformer")
    
    # Suppress noisy third-party loggers
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_tool_call(
        name: str = "test_tool",
        arguments: Dict = None,
        call_id: str = "call_123"
    ) -> Dict:
        """Create a mock tool call."""
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments or {"arg": "value"})
            }
        }
    
    @staticmethod
    def create_tool_result(
        call_id: str = "call_123",
        result: str = "test_result",
        execution_time: float = 0.1,
        success: bool = True
    ) -> Dict:
        """Create a mock tool result."""
        return {
            "call_id": call_id,
            "result": result,
            "execution_time": execution_time,
            "success": success,
            "error": None if success else "Test error"
        }


@pytest.fixture
def test_data_factory() -> TestDataFactory:
    """Provide test data factory."""
    return TestDataFactory()


# Global test configuration
pytest_plugins = [
    "pytest_asyncio",
    "pytest_mock",
    "pytest_cov",
]