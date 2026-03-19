"""AsyncOrchestrator — parallel tool execution with rate limiting and cancellation."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from .exceptions import BranchCancelledError, ToolExecutionError, ToolNotFoundError, ToolTimeoutError
from .rate_limiter import RateLimiterRegistry
from .tools import ToolRegistry, ToolResult, ToolSpec


@dataclass
class ToolCall:
    """A single tool invocation request (tool name + kwargs)."""

    tool_name: str
    kwargs: dict[str, Any]
    branch_id: str | None = None
    """Optional branch identifier.  Calls sharing a branch_id are cancelled together."""


class AsyncOrchestrator:
    """
    Execute a batch of :class:`ToolCall` objects in parallel.

    Features
    --------
    - Parallel execution via :mod:`asyncio`
    - Per-tool token-bucket rate limiting
    - Per-tool hard timeouts
    - Branch cancellation: cancel all pending calls on a branch by name

    Example
    -------
    ::

        registry = ToolRegistry()

        @registry.tool("Fetch weather", timeout_s=3, calls_per_min=60)
        async def weather(city: str) -> str:
            ...

        orch = AsyncOrchestrator(registry)
        results = await orch.run([
            ToolCall("weather", {"city": "NYC"}),
            ToolCall("weather", {"city": "London"}),
        ])
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._rate_registry = RateLimiterRegistry()

        # Build rate-limiter buckets for every tool that has a limit
        for spec in registry.list_tools():
            if spec.calls_per_min is not None:
                self._rate_registry.register(spec.name, spec.calls_per_min)

        # branch_id -> asyncio.Event (set to signal cancellation)
        self._cancel_events: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        calls: list[ToolCall],
        *,
        default_timeout_s: float | None = None,
    ) -> list[ToolResult]:
        """
        Execute *calls* in parallel and return results in the same order.

        Parameters
        ----------
        calls:
            Tool calls to execute.
        default_timeout_s:
            Fallback timeout for tools that don't specify one.
        """
        tasks = [
            asyncio.create_task(
                self._run_one(call, default_timeout_s),
                name=f"tool:{call.tool_name}",
            )
            for call in calls
        ]
        return list(await asyncio.gather(*tasks, return_exceptions=False))

    async def stream(
        self,
        calls: list[ToolCall],
        *,
        default_timeout_s: float | None = None,
    ) -> AsyncIterator[ToolResult]:
        """
        Yield results as they complete (not in submission order).
        """
        tasks = {
            asyncio.create_task(
                self._run_one(call, default_timeout_s),
                name=f"tool:{call.tool_name}",
            ): call
            for call in calls
        }

        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                yield t.result()

    def cancel_branch(self, branch_id: str) -> None:
        """
        Signal all in-flight calls on *branch_id* to abort.

        Any future calls on the same branch_id will also be rejected
        immediately (useful for speculative execution where you already
        have the answer you need).
        """
        event = self._cancel_events.setdefault(branch_id, asyncio.Event())
        event.set()

    def reset_branch(self, branch_id: str) -> None:
        """Clear the cancellation signal for *branch_id*."""
        self._cancel_events.pop(branch_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_one(
        self,
        call: ToolCall,
        default_timeout_s: float | None,
    ) -> ToolResult:
        """Acquire rate-limit token, check cancellation, run tool, enforce timeout."""
        t0 = time.perf_counter()

        def _err(msg: str) -> ToolResult:
            return ToolResult(
                tool_name=call.tool_name,
                output=None,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=msg,
            )

        # Resolve tool spec first
        try:
            spec = self.registry.get(call.tool_name)
        except ToolNotFoundError as exc:
            return _err(str(exc))

        # Check branch cancellation (pre-flight)
        cancel_event: asyncio.Event | None = None
        if call.branch_id is not None:
            cancel_event = self._cancel_events.setdefault(call.branch_id, asyncio.Event())
            if cancel_event.is_set():
                return _err("Branch cancelled")

        # Acquire rate-limit token (may block)
        await self._rate_registry.acquire(call.tool_name)

        # Re-check cancellation after rate-limit wait
        if cancel_event is not None and cancel_event.is_set():
            return _err("Branch cancelled")

        # Determine effective timeout
        timeout_s = spec.timeout_s if spec.timeout_s is not None else default_timeout_s

        # Run the tool, racing against branch cancellation if applicable
        try:
            if cancel_event is not None:
                result = await self._invoke_with_cancel(spec, call.kwargs, timeout_s, cancel_event)
            else:
                result = await self._invoke_with_timeout(spec, call.kwargs, timeout_s)
        except ToolTimeoutError as exc:
            return _err(str(exc))
        except BranchCancelledError:
            return _err("Branch cancelled")
        except Exception as exc:  # noqa: BLE001
            return _err(f"Tool raised: {exc}")

        return ToolResult(
            tool_name=call.tool_name,
            output=result,
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    @staticmethod
    async def _invoke_with_timeout(
        spec: ToolSpec,
        kwargs: dict[str, Any],
        timeout_s: float | None,
    ) -> Any:
        coro = spec.fn(**kwargs)
        if timeout_s is None:
            return await coro
        try:
            return await asyncio.wait_for(coro, timeout=timeout_s)
        except asyncio.TimeoutError:
            raise ToolTimeoutError(spec.name, timeout_s)

    @staticmethod
    async def _invoke_with_cancel(
        spec: ToolSpec,
        kwargs: dict[str, Any],
        timeout_s: float | None,
        cancel_event: asyncio.Event,
    ) -> Any:
        """Run tool coroutine, aborting early if *cancel_event* is set."""
        tool_task = asyncio.ensure_future(spec.fn(**kwargs))
        cancel_task = asyncio.ensure_future(cancel_event.wait())

        if timeout_s is not None:
            # Wrap in wait_for: whichever finishes first (tool or cancel)
            try:
                done, pending = await asyncio.wait(
                    {tool_task, cancel_task},
                    timeout=timeout_s,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except Exception:
                tool_task.cancel()
                cancel_task.cancel()
                raise
        else:
            done, pending = await asyncio.wait(
                {tool_task, cancel_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

        # Clean up the loser
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        if not done:
            # Timeout: no tasks completed
            raise ToolTimeoutError(spec.name, timeout_s)

        first = next(iter(done))

        if first is cancel_task:
            raise BranchCancelledError("Branch cancelled while tool was running")

        if first is tool_task:
            if timeout_s is not None and not tool_task.done():
                raise ToolTimeoutError(spec.name, timeout_s)
            exc = tool_task.exception()
            if exc is not None:
                raise exc
            return tool_task.result()
