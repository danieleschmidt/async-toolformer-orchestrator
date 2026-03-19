"""Tests for ToolRegistry and ToolSpec."""

import asyncio
import pytest

from async_toolformer import ToolRegistry, ToolResult, ToolSpec
from async_toolformer.exceptions import ToolNotFoundError


def test_register_via_decorator():
    registry = ToolRegistry()

    @registry.tool("greet someone", timeout_s=1, calls_per_min=60)
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    assert len(registry) == 1
    spec = registry.get("greet")
    assert spec.name == "greet"
    assert spec.description == "greet someone"
    assert spec.timeout_s == 1
    assert spec.calls_per_min == 60


def test_register_via_spec():
    registry = ToolRegistry()

    async def _fn(x: int) -> int:
        return x * 2

    registry.register(ToolSpec(name="double", description="double a number", fn=_fn))
    assert registry.get("double").fn is _fn


def test_get_missing_raises():
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError, match="nope"):
        registry.get("nope")


def test_list_tools():
    registry = ToolRegistry()

    @registry.tool("a")
    async def a() -> None: ...

    @registry.tool("b")
    async def b() -> None: ...

    names = {s.name for s in registry.list_tools()}
    assert names == {"a", "b"}


def test_schema_format():
    registry = ToolRegistry()

    @registry.tool("do something", calls_per_min=30)
    async def do_it() -> None: ...

    schema = registry.schema()
    assert len(schema) == 1
    assert schema[0]["name"] == "do_it"
    assert schema[0]["calls_per_min"] == 30


def test_decorator_rejects_sync():
    registry = ToolRegistry()
    with pytest.raises(TypeError):
        @registry.tool("sync won't work")
        def sync_fn(): ...


def test_tool_result_ok_flag():
    r_ok = ToolResult(tool_name="t", output=42, latency_ms=5.0)
    r_err = ToolResult(tool_name="t", output=None, latency_ms=5.0, error="boom")
    assert r_ok.ok
    assert not r_err.ok
