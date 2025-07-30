"""Custom exceptions for the Async Toolformer Orchestrator."""

from typing import Any, Optional


class OrchestratorError(Exception):
    """Base exception for all orchestrator-related errors."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ToolExecutionError(OrchestratorError):
    """Exception raised when a tool execution fails."""

    def __init__(
        self,
        tool_name: str,
        message: str,
        original_error: Optional[Exception] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details)
        self.tool_name = tool_name
        self.original_error = original_error


class RateLimitError(OrchestratorError):
    """Exception raised when rate limits are exceeded."""

    def __init__(
        self,
        service: str,
        limit_type: str,
        retry_after: Optional[float] = None,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        message = f"Rate limit exceeded for {service} ({limit_type})"
        if retry_after:
            message += f". Retry after {retry_after}s"
        super().__init__(message, details)
        self.service = service
        self.limit_type = limit_type
        self.retry_after = retry_after


class TimeoutError(OrchestratorError):
    """Exception raised when operations timeout."""

    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        message = f"Operation '{operation}' timed out after {timeout_seconds}s"
        super().__init__(message, details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class ConfigurationError(OrchestratorError):
    """Exception raised for configuration-related errors."""

    def __init__(
        self,
        parameter: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(f"Configuration error for '{parameter}': {message}", details)
        self.parameter = parameter


class SpeculationError(OrchestratorError):
    """Exception raised during speculative execution."""

    def __init__(
        self,
        message: str,
        speculation_id: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, details)
        self.speculation_id = speculation_id