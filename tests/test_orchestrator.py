"""Tests for AsyncOrchestrator."""

import asyncio
import time
import pytest

from async_toolformer import AsyncOrchestrator, ToolCall, ToolRegistry
from async_toolformer.exceptions import ToolNotFoundError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fast_registry():
    registry = ToolRegistry()

    @registry.tool("Return a string", timeout_s=2)
    async def echo(msg: str) -> str:
        return msg

    @registry.tool("Slow tool", timeout_s=2)
    async def slow(delay: float = 0.1) -> str:
        await asyncio.sleep(delay)
        return "done"

    @registry.tool("Raises an error")
    async def boom() -> None:
        raise ValueError("kaboom")

    @registry.tool("Very slow — used to test timeouts", timeout_s=0.1)
    async def very_slow() -> str:
        await asyncio.sleep(10)
        return "never"

    return registry


# ---------------------------------------------------------------------------
# Basic execution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_single_call(fast_registry):
    orch = AsyncOrchestrator(fast_registry)
    results = await orch.run([ToolCall("echo", {"msg": "hi"})])
    assert len(results) == 1
    r = results[0]
    assert r.ok
    assert r.output == "hi"
    assert r.tool_name == "echo"
    assert r.latency_ms > 0


@pytest.mark.asyncio
async def test_multiple_calls_parallel(fast_registry):
    """Running 4 slow tasks in parallel should be ~1x latency, not 4x."""
    orch = AsyncOrchestrator(fast_registry)
    calls = [ToolCall("slow", {"delay": 0.2}) for _ in range(4)]

    t0 = time.perf_counter()
    results = await orch.run(calls)
    elapsed = time.perf_counter() - t0

    assert all(r.ok for r in results)
    # Parallel: should finish in ~0.2s, not 0.8s
    assert elapsed < 0.5, f"Expected parallel execution, got {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_results_ordered(fast_registry):
    """Results must preserve submission order."""
    orch = AsyncOrchestrator(fast_registry)
    calls = [ToolCall("echo", {"msg": str(i)}) for i in range(5)]
    results = await orch.run(calls)
    for i, r in enumerate(results):
        assert r.output == str(i)


@pytest.mark.asyncio
async def test_tool_error_captured(fast_registry):
    """Tool errors should not propagate — they appear in ToolResult.error."""
    orch = AsyncOrchestrator(fast_registry)
    results = await orch.run([ToolCall("boom", {})])
    r = results[0]
    assert not r.ok
    assert "kaboom" in r.error


@pytest.mark.asyncio
async def test_missing_tool_raises(fast_registry):
    """Calling an unregistered tool should surface as an error result."""
    orch = AsyncOrchestrator(fast_registry)
    results = await orch.run([ToolCall("nonexistent", {})])
    r = results[0]
    assert not r.ok
    assert "nonexistent" in r.error


@pytest.mark.asyncio
async def test_timeout(fast_registry):
    """Tools that exceed their timeout return an error result."""
    orch = AsyncOrchestrator(fast_registry)
    results = await orch.run([ToolCall("very_slow", {})])
    r = results[0]
    assert not r.ok
    assert "timed out" in r.error.lower() or "timeout" in r.error.lower()


# ---------------------------------------------------------------------------
# Branch cancellation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel_branch_aborts_pending():
    """Cancelling a branch after rate-limit wait should abort the calls."""
    registry = ToolRegistry()

    @registry.tool("slow search", timeout_s=5)
    async def slow_search(q: str) -> str:
        await asyncio.sleep(1.0)
        return f"results for {q}"

    orch = AsyncOrchestrator(registry)
    calls = [
        ToolCall("slow_search", {"q": "a"}, branch_id="branch-1"),
        ToolCall("slow_search", {"q": "b"}, branch_id="branch-1"),
    ]

    async def cancel_soon():
        await asyncio.sleep(0.05)
        orch.cancel_branch("branch-1")

    t0 = time.perf_counter()
    _, results = await asyncio.gather(cancel_soon(), orch.run(calls))
    elapsed = time.perf_counter() - t0

    # Both should be cancelled (error), and wall-clock << 1s
    assert all(not r.ok for r in results)
    assert all("cancel" in (r.error or "").lower() for r in results)
    assert elapsed < 0.5, f"Cancellation didn't short-circuit: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_cancel_does_not_affect_other_branches():
    """Cancelling branch-A must not affect branch-B."""
    registry = ToolRegistry()

    @registry.tool("quick task", timeout_s=2)
    async def quick(val: str) -> str:
        await asyncio.sleep(0.05)
        return val

    orch = AsyncOrchestrator(registry)
    calls = [
        ToolCall("quick", {"val": "a"}, branch_id="branch-A"),
        ToolCall("quick", {"val": "b"}, branch_id="branch-B"),
    ]
    orch.cancel_branch("branch-A")  # pre-cancel A
    results = await orch.run(calls)

    result_map = {r.tool_name: r for r in results}
    # A should be cancelled (both calls are quick:val=a and quick:val=b)
    # Let's check by branch membership
    r_a, r_b = results
    assert not r_a.ok  # branch-A cancelled
    assert r_b.ok      # branch-B untouched
    assert r_b.output == "b"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_yields_results(fast_registry):
    """stream() should yield all results (possibly out of order)."""
    orch = AsyncOrchestrator(fast_registry)
    calls = [ToolCall("echo", {"msg": str(i)}) for i in range(3)]
    outputs = []
    async for r in orch.stream(calls):
        outputs.append(r.output)
    assert sorted(outputs) == ["0", "1", "2"]


@pytest.mark.asyncio
async def test_stream_first_result_arrives_early():
    """stream() should yield fast results before slow ones complete."""
    registry = ToolRegistry()

    @registry.tool("fast", timeout_s=2)
    async def fast_tool() -> str:
        await asyncio.sleep(0.01)
        return "fast"

    @registry.tool("slow", timeout_s=5)
    async def slow_tool() -> str:
        await asyncio.sleep(0.5)
        return "slow"

    orch = AsyncOrchestrator(registry)
    calls = [ToolCall("slow_tool", {}), ToolCall("fast_tool", {})]

    arrival_order = []
    t0 = time.perf_counter()
    async for r in orch.stream(calls):
        arrival_order.append((r.tool_name, time.perf_counter() - t0))

    # "fast_tool" should be yielded first (arrives ~0.01s, slow_tool arrives ~0.5s)
    assert arrival_order[0][0] == "fast_tool"
