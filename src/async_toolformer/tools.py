"""Tool system for the Async Toolformer Orchestrator."""

import asyncio
import functools
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

from .exceptions import ToolExecutionError

T = TypeVar('T')
ToolFunction = Callable[..., Awaitable[Any]]


@dataclass
class ToolResult:
    """Result from a tool execution."""

    tool_name: str
    success: bool
    data: Any
    execution_time_ms: float
    metadata: dict[str, Any]
    timestamp: datetime

    @classmethod
    def success_result(
        cls,
        tool_name: str,
        data: Any,
        execution_time_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        """Create a successful tool result."""
        return cls(
            tool_name=tool_name,
            success=True,
            data=data,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc),
        )

    @classmethod
    def error_result(
        cls,
        tool_name: str,
        error: Exception,
        execution_time_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        """Create an error tool result."""
        return cls(
            tool_name=tool_name,
            success=False,
            data=str(error),
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
            timestamp=datetime.now(timezone.utc),
        )


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""

    name: str
    description: str
    function: ToolFunction
    timeout_ms: int | None = None
    rate_limit_group: str | None = None
    retry_attempts: int | None = None
    tags: list[str] = None
    priority: int = 0

    def __post_init__(self) -> None:
        if self.tags is None:
            self.tags = []


class Tool:
    """Decorator for registering async tools."""

    def __init__(
        self,
        description: str,
        *,
        name: str | None = None,
        timeout_ms: int | None = None,
        rate_limit_group: str | None = None,
        retry_attempts: int | None = None,
        tags: list[str] | None = None,
        priority: int = 0,
    ):
        """
        Initialize tool decorator.

        Args:
            description: Description of what the tool does
            name: Custom name for the tool (defaults to function name)
            timeout_ms: Tool-specific timeout in milliseconds
            rate_limit_group: Rate limiting group for this tool
            retry_attempts: Number of retry attempts for this tool
            tags: Tags for categorizing the tool
            priority: Execution priority (higher = more important)
        """
        self.description = description
        self.name = name
        self.timeout_ms = timeout_ms
        self.rate_limit_group = rate_limit_group
        self.retry_attempts = retry_attempts
        self.tags = tags or []
        self.priority = priority

    def __call__(self, func: ToolFunction) -> ToolFunction:
        """Apply the decorator to a function."""
        if not asyncio.iscoroutinefunction(func):
            raise ValueError(f"Tool {func.__name__} must be an async function")

        tool_name = self.name or func.__name__

        # Store metadata on the function
        func._tool_metadata = ToolMetadata(
            name=tool_name,
            description=self.description,
            function=func,
            timeout_ms=self.timeout_ms,
            rate_limit_group=self.rate_limit_group,
            retry_attempts=self.retry_attempts,
            tags=self.tags,
            priority=self.priority,
        )

        # Add test-expected attributes for compatibility
        func._tool_description = self.description
        func._tool_timeout_ms = self.timeout_ms
        func._tool_schema = self._generate_schema(func)

        return func

    def _generate_schema(self, func: ToolFunction) -> dict[str, Any]:
        """Generate tool schema from function signature."""
        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            param_schema = {"type": "string"}  # Default type

            # Handle type annotations
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_schema["type"] = "integer"
                elif param.annotation == float:
                    param_schema["type"] = "number"
                elif param.annotation == bool:
                    param_schema["type"] = "boolean"

            # Handle default values
            if param.default != inspect.Parameter.empty:
                param_schema["default"] = param.default
            else:
                required.append(name)

            properties[name] = param_schema

        return {
            "type": "object",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


def ToolChain(
    func: ToolFunction | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parallel_sections: list[str] | None = None,
):
    """
    Decorator for creating complex tool workflows.

    Can be used with or without parameters:
    @ToolChain
    async def my_chain(): ...

    @ToolChain(name="custom_chain")
    async def my_chain(): ...
    """
    def decorator(f: ToolFunction) -> ToolFunction:
        if not asyncio.iscoroutinefunction(f):
            raise ValueError(f"ToolChain {f.__name__} must be an async function")

        chain_name = name or f.__name__
        chain_description = description or f"Tool chain: {chain_name}"

        # Store chain metadata
        f._chain_metadata = {
            "name": chain_name,
            "description": chain_description,
            "parallel_sections": parallel_sections or [],
            "function": f,
        }

        # Add test-expected attributes for compatibility
        f._is_tool_chain = True

        return f

    # Handle usage without parameters (@ToolChain)
    if func is not None:
        return decorator(func)

    # Handle usage with parameters (@ToolChain(...))
    return decorator


class ToolRegistry:
    """Registry for managing tools and tool chains."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolMetadata] = {}
        self._chains: dict[str, dict[str, Any]] = {}

    def register_tool(self, func: ToolFunction) -> None:
        """Register a tool function."""
        if not hasattr(func, '_tool_metadata'):
            raise ValueError(f"Function {func.__name__} is not decorated with @Tool")

        metadata = func._tool_metadata
        self._tools[metadata.name] = metadata

    def register_chain(self, func: ToolFunction) -> None:
        """Register a tool chain function."""
        if not hasattr(func, '_chain_metadata'):
            raise ValueError(f"Function {func.__name__} is not decorated with @ToolChain")

        metadata = func._chain_metadata
        self._chains[metadata['name']] = metadata

    def get_tool(self, name: str) -> ToolMetadata | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_chain(self, name: str) -> dict[str, Any] | None:
        """Get a tool chain by name."""
        return self._chains.get(name)

    def list_tools(self, tags: list[str] | None = None) -> list[ToolMetadata]:
        """List all registered tools, optionally filtered by tags."""
        tools = list(self._tools.values())

        if tags:
            tools = [
                tool for tool in tools
                if any(tag in tool.tags for tag in tags)
            ]

        return sorted(tools, key=lambda t: t.priority, reverse=True)

    def list_chains(self) -> list[dict[str, Any]]:
        """List all registered tool chains."""
        return list(self._chains.values())


# Utility functions for tool composition
async def parallel(*tasks: Awaitable[T]) -> list[T]:
    """Execute multiple async tasks in parallel."""
    return await asyncio.gather(*tasks)


async def sequential(*tasks: Awaitable[T]) -> list[T]:
    """Execute multiple async tasks sequentially."""
    results = []
    for task in tasks:
        result = await task
        results.append(result)
    return results


def timeout(seconds: float) -> Callable[[ToolFunction], ToolFunction]:
    """Decorator to add timeout to a tool function."""
    def decorator(func: ToolFunction) -> ToolFunction:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=seconds)
            except asyncio.TimeoutError:
                raise ToolExecutionError(
                    tool_name=func.__name__,
                    message=f"Tool timed out after {seconds}s",
                    details={"timeout_seconds": seconds}
                )
        return wrapper
    return decorator


def retry(attempts: int = 3, delay: float = 1.0, backoff: float = 1.0) -> Callable[[ToolFunction], ToolFunction]:
    """Decorator to add retry logic to a tool function."""
    def decorator(func: ToolFunction) -> ToolFunction:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < attempts - 1:  # Don't sleep on last attempt
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff

            # If we get here, all attempts failed
            raise ToolExecutionError(
                tool_name=func.__name__,
                message=f"Tool failed after {attempts} attempts",
                original_error=last_exception,
                details={"attempts": attempts, "delay": delay, "backoff": backoff}
            )
        return wrapper
    return decorator
