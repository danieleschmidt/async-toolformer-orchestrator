"""Tool system for the Async Toolformer Orchestrator."""

import asyncio
import functools
import inspect
from typing import Any, Awaitable, Callable, Dict, List, Optional, TypeVar, Union
from dataclasses import dataclass
from datetime import datetime

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
    metadata: Dict[str, Any]
    timestamp: datetime
    
    @classmethod
    def success_result(
        cls,
        tool_name: str,
        data: Any,
        execution_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """Create a successful tool result."""
        return cls(
            tool_name=tool_name,
            success=True,
            data=data,
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
            timestamp=datetime.utcnow(),
        )
    
    @classmethod
    def error_result(
        cls,
        tool_name: str,
        error: Exception,
        execution_time_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolResult":
        """Create an error tool result."""
        return cls(
            tool_name=tool_name,
            success=False,
            data=str(error),
            execution_time_ms=execution_time_ms,
            metadata=metadata or {},
            timestamp=datetime.utcnow(),
        )


@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    
    name: str
    description: str
    function: ToolFunction
    timeout_ms: Optional[int] = None
    rate_limit_group: Optional[str] = None
    retry_attempts: Optional[int] = None
    tags: List[str] = None
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
        name: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        rate_limit_group: Optional[str] = None,
        retry_attempts: Optional[int] = None,
        tags: Optional[List[str]] = None,
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
        
        return func


class ToolChain:
    """Decorator for creating complex tool workflows."""
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parallel_sections: Optional[List[str]] = None,
    ):
        """
        Initialize tool chain decorator.
        
        Args:
            name: Name of the tool chain
            description: Description of the tool chain
            parallel_sections: Sections that can run in parallel
        """
        self.name = name
        self.description = description
        self.parallel_sections = parallel_sections or []
    
    def __call__(self, func: ToolFunction) -> ToolFunction:
        """Apply the decorator to a function."""
        if not asyncio.iscoroutinefunction(func):
            raise ValueError(f"ToolChain {func.__name__} must be an async function")
        
        chain_name = self.name or func.__name__
        chain_description = self.description or f"Tool chain: {chain_name}"
        
        # Store chain metadata
        func._chain_metadata = {
            "name": chain_name,
            "description": chain_description,
            "parallel_sections": self.parallel_sections,
            "function": func,
        }
        
        return func


class ToolRegistry:
    """Registry for managing tools and tool chains."""
    
    def __init__(self) -> None:
        self._tools: Dict[str, ToolMetadata] = {}
        self._chains: Dict[str, Dict[str, Any]] = {}
    
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
    
    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_chain(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a tool chain by name."""
        return self._chains.get(name)
    
    def list_tools(self, tags: Optional[List[str]] = None) -> List[ToolMetadata]:
        """List all registered tools, optionally filtered by tags."""
        tools = list(self._tools.values())
        
        if tags:
            tools = [
                tool for tool in tools
                if any(tag in tool.tags for tag in tags)
            ]
        
        return sorted(tools, key=lambda t: t.priority, reverse=True)
    
    def list_chains(self) -> List[Dict[str, Any]]:
        """List all registered tool chains."""
        return list(self._chains.values())


# Utility functions for tool composition
async def parallel(*tasks: Awaitable[T]) -> List[T]:
    """Execute multiple async tasks in parallel."""
    return await asyncio.gather(*tasks)


async def sequential(*tasks: Awaitable[T]) -> List[T]:
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