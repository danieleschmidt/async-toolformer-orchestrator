"""Unit tests for Tool and ToolChain decorators."""

import asyncio

import pytest

from async_toolformer import Tool, ToolChain


@pytest.mark.unit
class TestTool:
    """Test cases for Tool decorator."""

    def test_tool_decoration(self):
        """Test basic tool decoration."""
        @Tool(description="Test tool")
        async def test_func(x: int) -> int:
            return x * 2

        assert hasattr(test_func, '_tool_description')
        assert test_func._tool_description == "Test tool"
        assert hasattr(test_func, '_tool_schema')

    def test_tool_schema_generation(self):
        """Test tool schema generation from function signature."""
        @Tool(description="Multiply by factor")
        async def multiply(value: int, factor: int = 2) -> int:
            return value * factor

        schema = multiply._tool_schema
        assert 'value' in schema['parameters']['properties']
        assert 'factor' in schema['parameters']['properties']
        assert schema['parameters']['properties']['factor']['default'] == 2

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test tool execution."""
        @Tool(description="Add numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(2, 3)
        assert result == 5

    def test_tool_metadata_preservation(self):
        """Test that tool metadata is preserved."""
        @Tool(description="Test function", timeout_ms=5000)
        async def test_func() -> str:
            """Original docstring."""
            return "test"

        assert test_func.__doc__ == "Original docstring."
        assert test_func._tool_timeout_ms == 5000


@pytest.mark.unit
class TestToolChain:
    """Test cases for ToolChain decorator."""

    def test_chain_decoration(self):
        """Test chain decoration."""
        @ToolChain
        async def test_chain():
            return "chain result"

        assert hasattr(test_chain, '_is_tool_chain')
        assert test_chain._is_tool_chain is True

    @pytest.mark.asyncio
    async def test_chain_execution(self):
        """Test chain execution with dependencies."""
        @ToolChain
        async def process_data(data: str):
            # Simulate multi-step processing
            step1 = data.upper()
            await asyncio.sleep(0.01)
            step2 = step1.replace(" ", "_")
            return step2

        result = await process_data("hello world")
        assert result == "HELLO_WORLD"
