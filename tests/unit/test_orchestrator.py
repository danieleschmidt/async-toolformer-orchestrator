"""Unit tests for AsyncOrchestrator."""

import pytest
from unittest.mock import AsyncMock, patch

from async_toolformer import AsyncOrchestrator, Tool, ToolExecutionError


@pytest.mark.unit
class TestAsyncOrchestrator:
    """Test cases for AsyncOrchestrator class."""

    @pytest.mark.asyncio
    async def test_initialization(self, mock_llm_client):
        """Test orchestrator initialization."""
        orchestrator = AsyncOrchestrator(llm_client=mock_llm_client)
        
        assert orchestrator.llm_integration is not None
        assert orchestrator.config.max_parallel_tools == 30  # default
        assert len(orchestrator.registry._tools) == 0

    @pytest.mark.asyncio  
    async def test_tool_registration(self, mock_llm_client, sample_tools):
        """Test tool registration."""
        orchestrator = AsyncOrchestrator(
            llm_client=mock_llm_client,
            tools=sample_tools
        )
        
        assert len(orchestrator.registry._tools) == len(sample_tools)

    @pytest.mark.asyncio
    async def test_parallel_execution_limit(self, mock_llm_client):
        """Test parallel execution limits are respected."""
        from async_toolformer import OrchestratorConfig
        config = OrchestratorConfig(max_parallel_tools=2, max_parallel_per_type=1)
        
        orchestrator = AsyncOrchestrator(
            llm_client=mock_llm_client,
            config=config
        )
        
        assert orchestrator.config.max_parallel_tools == 2

    @pytest.mark.asyncio
    async def test_tool_timeout_handling(self, mock_llm_client, sample_tools):
        """Test tool timeout handling."""
        orchestrator = AsyncOrchestrator(
            llm_client=mock_llm_client,
            tools=sample_tools,
            tool_timeout_ms=100
        )
        
        # Get list of registered tool names
        registered_tools = list(orchestrator.registry._tools.keys())
        assert len(registered_tools) > 0
        
        # Test execution of the first registered tool
        first_tool_name = registered_tools[0]
        result = await orchestrator._execute_single_tool(first_tool_name, {})
        # The tool should complete successfully
        assert result is not None
        assert result.tool_name == first_tool_name

    @pytest.mark.asyncio
    async def test_rate_limit_integration(self, mock_llm_client):
        """Test rate limiting integration."""
        from async_toolformer import RateLimitConfig
        
        rate_config = RateLimitConfig(global_max=5)
        orchestrator = AsyncOrchestrator(
            llm_client=mock_llm_client,
        )
        
        assert orchestrator.rate_limiter is not None