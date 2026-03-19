"""Exceptions for the Async Toolformer Orchestrator."""

from __future__ import annotations


class OrchestratorError(Exception):
    """Base exception."""


class ToolNotFoundError(OrchestratorError):
    """Tool not registered."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Tool not registered: {name!r}")
        self.name = name


class ToolExecutionError(OrchestratorError):
    """Tool raised during execution."""

    def __init__(self, tool_name: str, cause: BaseException) -> None:
        super().__init__(f"Tool {tool_name!r} failed: {cause}")
        self.tool_name = tool_name
        self.cause = cause


class ToolTimeoutError(OrchestratorError):
    """Tool exceeded its timeout."""

    def __init__(self, tool_name: str, timeout_s: float) -> None:
        super().__init__(f"Tool {tool_name!r} timed out after {timeout_s}s")
        self.tool_name = tool_name
        self.timeout_s = timeout_s


class RateLimitError(OrchestratorError):
    """Rate limit exceeded and wait was rejected."""

    def __init__(self, tool_name: str, retry_after: float) -> None:
        super().__init__(
            f"Rate limit for {tool_name!r} exceeded; retry after {retry_after:.2f}s"
        )
        self.tool_name = tool_name
        self.retry_after = retry_after


class BranchCancelledError(OrchestratorError):
    """Branch was explicitly cancelled by the caller."""
