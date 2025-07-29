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
        
        assert orchestrator.llm_client == mock_llm_client
        assert orchestrator.max_parallel == 10  # default
        assert orchestrator.tools == []

    @pytest.mark.asyncio  
    async def test_tool_registration(self, mock_llm_client, sample_tool):
        """Test tool registration."""
        orchestrator = AsyncOrchestrator(
            llm_client=mock_llm_client,
            tools=[sample_tool]
        )
        
        assert len(orchestrator.tools) == 1
        assert orchestrator.tools[0] == sample_tool

    @pytest.mark.asyncio
    async def test_parallel_execution_limit(self, mock_llm_client):
        """Test parallel execution limits are respected."""
        orchestrator = AsyncOrchestrator(
            llm_client=mock_llm_client,
            max_parallel=2
        )
        
        assert orchestrator.max_parallel == 2

    @pytest.mark.asyncio
    async def test_tool_timeout_handling(self, mock_llm_client, slow_tool):
        """Test tool timeout handling."""
        orchestrator = AsyncOrchestrator(
            llm_client=mock_llm_client,
            tools=[slow_tool],
            tool_timeout_ms=100
        )
        
        with pytest.raises(ToolExecutionError):
            await orchestrator._execute_tool(slow_tool, {})

    @pytest.mark.asyncio
    async def test_rate_limit_integration(self, mock_llm_client):
        """Test rate limiting integration."""
        from async_toolformer import RateLimitConfig
        
        rate_config = RateLimitConfig(global_max=5)
        orchestrator = AsyncOrchestrator(
            llm_client=mock_llm_client,
            rate_limit_config=rate_config
        )
        
        assert orchestrator.rate_limiter is not None