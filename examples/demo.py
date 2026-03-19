#!/usr/bin/env python3
"""
Demo: parallel tool execution vs sequential, with rate limiting.

5 simulated tools, each with different latencies and rate limits.
Shows wall-clock time difference and per-tool latency.
"""

import asyncio
import time

from async_toolformer import AsyncOrchestrator, ToolCall, ToolRegistry

# ---------------------------------------------------------------------------
# Build a registry with 5 tools of varying speed / rate limits
# ---------------------------------------------------------------------------

registry = ToolRegistry()


@registry.tool("Fetch weather data", timeout_s=5, calls_per_min=60)
async def weather(city: str) -> dict:
    await asyncio.sleep(0.3)  # 300 ms simulated latency
    return {"city": city, "temp_c": 22, "condition": "sunny"}


@registry.tool("Look up stock price", timeout_s=5, calls_per_min=30)
async def stock_price(ticker: str) -> dict:
    await asyncio.sleep(0.5)  # 500 ms
    return {"ticker": ticker.upper(), "price": 182.34, "currency": "USD"}


@registry.tool("Translate text", timeout_s=3, calls_per_min=120)
async def translate(text: str, lang: str) -> str:
    await asyncio.sleep(0.2)  # 200 ms
    return f"[{lang.upper()}] {text}"


@registry.tool("Run web search", timeout_s=8, calls_per_min=20)
async def web_search(query: str) -> list[str]:
    await asyncio.sleep(0.8)  # 800 ms — slowest tool
    return [f"Result 1 for '{query}'", f"Result 2 for '{query}'"]


@registry.tool("Summarise document", timeout_s=10, calls_per_min=15)
async def summarise(doc_id: str) -> str:
    await asyncio.sleep(0.6)  # 600 ms
    return f"Summary of document {doc_id}: Lorem ipsum dolor sit amet."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CALLS = [
    ToolCall("weather", {"city": "New York"}),
    ToolCall("weather", {"city": "London"}),
    ToolCall("stock_price", {"ticker": "aapl"}),
    ToolCall("stock_price", {"ticker": "msft"}),
    ToolCall("translate", {"text": "Hello world", "lang": "es"}),
    ToolCall("web_search", {"query": "async LLM orchestration"}),
    ToolCall("summarise", {"doc_id": "doc-42"}),
]


def print_results(results, wall_ms: float, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}   wall-clock: {wall_ms:.0f} ms")
    print(f"{'='*60}")
    for r in results:
        status = "✓" if r.ok else "✗"
        print(f"  {status} {r.tool_name:<18} {r.latency_ms:>6.0f} ms", end="")
        if not r.ok:
            print(f"  ERROR: {r.error}", end="")
        print()


# ---------------------------------------------------------------------------
# Sequential baseline
# ---------------------------------------------------------------------------

async def run_sequential() -> tuple[list, float]:
    orch = AsyncOrchestrator(registry)
    t0 = time.perf_counter()
    results = []
    for call in CALLS:
        r = await orch.run([call])
        results.extend(r)
    wall_ms = (time.perf_counter() - t0) * 1000
    return results, wall_ms


# ---------------------------------------------------------------------------
# Parallel run
# ---------------------------------------------------------------------------

async def run_parallel() -> tuple[list, float]:
    orch = AsyncOrchestrator(registry)
    t0 = time.perf_counter()
    results = await orch.run(CALLS)
    wall_ms = (time.perf_counter() - t0) * 1000
    return results, wall_ms


# ---------------------------------------------------------------------------
# Branch cancellation demo
# ---------------------------------------------------------------------------

async def run_cancellation_demo() -> None:
    print(f"\n{'='*60}")
    print("  Branch cancellation demo")
    print(f"{'='*60}")

    orch = AsyncOrchestrator(registry)

    # Two parallel searches on "branch-A"; we cancel mid-flight
    calls = [
        ToolCall("web_search", {"query": "topic A first"}, branch_id="branch-A"),
        ToolCall("web_search", {"query": "topic A second"}, branch_id="branch-A"),
        ToolCall("summarise", {"doc_id": "doc-99"}, branch_id="branch-B"),
    ]

    async def cancel_after(delay: float, branch: str) -> None:
        await asyncio.sleep(delay)
        orch.cancel_branch(branch)
        print(f"  ⚡ cancelled branch {branch!r} at t+{delay*1000:.0f}ms")

    t0 = time.perf_counter()
    # Cancel branch-A after 100 ms (web_search takes 800 ms, so it will be cut)
    _, results = await asyncio.gather(
        cancel_after(0.1, "branch-A"),
        orch.run(calls),
    )
    wall_ms = (time.perf_counter() - t0) * 1000

    for r in results:
        status = "✓" if r.ok else f"✗ ({r.error})"
        print(f"  {r.tool_name:<18} {status}")
    print(f"\n  wall-clock: {wall_ms:.0f} ms")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    print("\nAsync Toolformer Orchestrator — parallel execution demo")
    print(f"  {len(CALLS)} tool calls across {len(registry)} tools\n")

    seq_results, seq_ms = await run_sequential()
    print_results(seq_results, seq_ms, "SEQUENTIAL")

    par_results, par_ms = await run_parallel()
    print_results(par_results, par_ms, "PARALLEL  ")

    speedup = seq_ms / par_ms if par_ms > 0 else float("inf")
    print(f"\n  Speedup: {speedup:.1f}x  ({seq_ms:.0f}ms → {par_ms:.0f}ms)\n")

    await run_cancellation_demo()


if __name__ == "__main__":
    asyncio.run(main())
