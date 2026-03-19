"""Tool registry — register, describe, and call async tools."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .exceptions import ToolNotFoundError

ToolFn = Callable[..., Awaitable[Any]]


@dataclass
class ToolResult:
    """Outcome of a single tool invocation."""

    tool_name: str
    output: Any
    latency_ms: float
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        status = "ok" if self.ok else f"err={self.error!r}"
        return f"ToolResult({self.tool_name!r}, {status}, {self.latency_ms:.1f}ms)"


@dataclass
class ToolSpec:
    """Metadata + callable for one tool."""

    name: str
    description: str
    fn: ToolFn
    timeout_s: float | None = None
    """Hard wall-clock timeout in seconds (None = no limit)."""
    calls_per_min: float | None = None
    """Token-bucket rate limit (None = unlimited)."""
    tags: list[str] = field(default_factory=list)


class ToolRegistry:
    """
    Flat registry of async tools.

    Usage::

        registry = ToolRegistry()

        @registry.tool("Look up weather", timeout_s=5, calls_per_min=30)
        async def get_weather(city: str) -> str:
            ...
    """

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def tool(
        self,
        description: str,
        *,
        name: str | None = None,
        timeout_s: float | None = None,
        calls_per_min: float | None = None,
        tags: list[str] | None = None,
    ) -> Callable[[ToolFn], ToolFn]:
        """Decorator that registers an async function as a tool."""

        def decorator(fn: ToolFn) -> ToolFn:
            if not asyncio.iscoroutinefunction(fn):
                raise TypeError(f"{fn.__name__} must be an async function")
            tool_name = name or fn.__name__
            self.register(
                ToolSpec(
                    name=tool_name,
                    description=description,
                    fn=fn,
                    timeout_s=timeout_s,
                    calls_per_min=calls_per_min,
                    tags=tags or [],
                )
            )
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise ToolNotFoundError(name)

    def list_tools(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    # ------------------------------------------------------------------
    # Schema (LLM-friendly summary)
    # ------------------------------------------------------------------

    def schema(self) -> list[dict[str, Any]]:
        """Return a list of tool descriptors suitable for LLM prompts."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "timeout_s": s.timeout_s,
                "calls_per_min": s.calls_per_min,
                "tags": s.tags,
            }
            for s in self._specs.values()
        ]
