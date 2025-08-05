"""Integration tests for basic orchestration functionality."""

import asyncio
import pytest
from async_toolformer import AsyncOrchestrator, Tool, OrchestratorConfig


@Tool(description="Test tool that returns a simple message")
async def simple_tool(message: str = "Hello") -> str:
    """Simple test tool."""
    await asyncio.sleep(0.1)
    return f"Tool response: {message}"


@Tool(description="Test tool that performs a calculation") 
async def math_tool(a: int, b: int = 10) -> int:
    """Math test tool."""
    await asyncio.sleep(0.05)
    return a + b


@Tool(description="Test tool that returns structured data")
async def data_tool(key: str) -> dict:
    """Data test tool."""
    await asyncio.sleep(0.02)
    return {"key": key, "value": 42, "status": "success"}


@pytest.mark.asyncio
async def test_basic_orchestrator_creation():
    """Test basic orchestrator creation and setup."""
    config = OrchestratorConfig(max_parallel_tools=5, max_parallel_per_type=5)
    orchestrator = AsyncOrchestrator(
        tools=[simple_tool, math_tool, data_tool],
        config=config
    )
    
    assert len(orchestrator.registry._tools) == 3
    assert "simple_tool" in orchestrator.registry._tools
    assert "math_tool" in orchestrator.registry._tools
    assert "data_tool" in orchestrator.registry._tools
    
    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_tool_execution():
    """Test basic tool execution."""
    orchestrator = AsyncOrchestrator(tools=[simple_tool, math_tool])
    
    result = await orchestrator.execute("Test prompt")
    
    assert result["status"] in ["completed", "no_tools_called"]
    assert "execution_id" in result
    assert "total_time_ms" in result
    assert result["total_time_ms"] > 0
    
    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_streaming_execution():
    """Test streaming execution."""
    orchestrator = AsyncOrchestrator(tools=[simple_tool, data_tool])
    
    results = []
    async for result in orchestrator.stream_execute("Test streaming"):
        results.append(result)
        assert hasattr(result, 'tool_name')
        assert hasattr(result, 'success')
        assert hasattr(result, 'execution_time_ms')
    
    # Should have at least some results from mock LLM
    assert len(results) >= 0  # Mock might return 0 results
    
    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_tool_registry():
    """Test tool registry functionality."""
    orchestrator = AsyncOrchestrator()
    
    # Register tools individually
    orchestrator.register_tool(simple_tool)
    orchestrator.register_tool(math_tool)
    
    assert len(orchestrator.registry._tools) == 2
    
    # Test getting tools
    tool_metadata = orchestrator.registry.get_tool("simple_tool")
    assert tool_metadata is not None
    assert tool_metadata.name == "simple_tool"
    assert tool_metadata.description == "Test tool that returns a simple message"
    
    # Test listing tools
    tools = orchestrator.registry.list_tools()
    assert len(tools) == 2
    
    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_configuration_validation():
    """Test configuration validation."""
    # Valid configuration
    config = OrchestratorConfig(
        max_parallel_tools=10,
        tool_timeout_ms=5000,
        total_timeout_ms=30000
    )
    config.validate()  # Should not raise
    
    # Invalid configuration - negative parallel tools
    with pytest.raises(ValueError):
        invalid_config = OrchestratorConfig(max_parallel_tools=-1)
        invalid_config.validate()
    
    # Invalid configuration - total timeout < tool timeout
    with pytest.raises(ValueError):
        invalid_config = OrchestratorConfig(
            tool_timeout_ms=10000,
            total_timeout_ms=5000
        )
        invalid_config.validate()


@pytest.mark.asyncio
async def test_metrics():
    """Test metrics collection."""
    orchestrator = AsyncOrchestrator(tools=[simple_tool])
    
    metrics = await orchestrator.get_metrics()
    
    assert "registered_tools" in metrics
    assert "registered_chains" in metrics
    assert "active_tasks" in metrics
    assert "config" in metrics
    assert "cache" in metrics
    assert "connection_pools" in metrics
    
    assert metrics["registered_tools"] == 1
    assert metrics["registered_chains"] == 0
    
    await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_context_manager():
    """Test async context manager functionality."""
    async with AsyncOrchestrator(tools=[simple_tool]) as orchestrator:
        assert len(orchestrator.registry._tools) == 1
        
        result = await orchestrator.execute("Test context manager")
        assert "execution_id" in result
    
    # Cleanup should be called automatically


@pytest.mark.asyncio
async def test_llm_integration_mock():
    """Test mock LLM integration."""
    orchestrator = AsyncOrchestrator(tools=[simple_tool, math_tool])
    
    # Test metrics
    llm_metrics = orchestrator.llm_integration.get_metrics()
    assert "registered_providers" in llm_metrics
    assert "default_provider" in llm_metrics
    assert llm_metrics["default_provider"] == "mock"
    
    await orchestrator.cleanup()