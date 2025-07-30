"""Core AsyncOrchestrator implementation."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, AsyncIterator
import logging

from .config import OrchestratorConfig
from .tools import ToolFunction, ToolMetadata, ToolRegistry, ToolResult
from .exceptions import (
    OrchestratorError,
    ToolExecutionError,
    TimeoutError,
    ConfigurationError,
)

logger = logging.getLogger(__name__)


class AsyncOrchestrator:
    """
    Main orchestrator for parallel tool execution.
    
    Manages LLM tool calls with sophisticated parallelism, rate limiting,
    cancellation, and result streaming capabilities.
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        tools: Optional[List[ToolFunction]] = None,
        config: Optional[OrchestratorConfig] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the AsyncOrchestrator.
        
        Args:
            llm_client: LLM client (OpenAI, Anthropic, etc.)
            tools: List of tool functions to register
            config: Configuration object
            **kwargs: Additional configuration parameters
        """
        self.llm_client = llm_client
        self.config = config or OrchestratorConfig()
        self.registry = ToolRegistry()
        
        # Apply any kwargs to config
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Validate configuration
        try:
            self.config.validate()
        except ValueError as e:
            raise ConfigurationError("config", str(e))
        
        # Register provided tools
        if tools:
            for tool in tools:
                self.register_tool(tool)
        
        # Initialize internal state
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._results_cache: Dict[str, ToolResult] = {}
        self._speculation_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info(
            f"AsyncOrchestrator initialized with {len(self.registry._tools)} tools"
        )
    
    def register_tool(self, tool: ToolFunction) -> None:
        """Register a tool function."""
        self.registry.register_tool(tool)
        logger.debug(f"Registered tool: {tool.__name__}")
    
    def register_chain(self, chain: ToolFunction) -> None:
        """Register a tool chain function."""
        self.registry.register_chain(chain)
        logger.debug(f"Registered tool chain: {chain.__name__}")
    
    async def execute(
        self,
        prompt: str,
        tools: Optional[List[str]] = None,
        max_parallel: Optional[int] = None,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute tools based on LLM decision.
        
        Args:
            prompt: Input prompt for the LLM
            tools: Specific tools to use (defaults to all registered)
            max_parallel: Override max parallel execution
            timeout_ms: Override timeout
            
        Returns:
            Dictionary containing execution results
        """
        start_time = time.time()
        execution_id = f"exec_{int(start_time * 1000)}"
        
        try:
            # Get LLM decision on which tools to call
            tool_calls = await self._get_llm_tool_calls(prompt, tools)
            
            if not tool_calls:
                return {
                    "execution_id": execution_id,
                    "results": [],
                    "total_time_ms": (time.time() - start_time) * 1000,
                    "status": "no_tools_called",
                }
            
            # Execute tools in parallel
            results = await self._execute_tools_parallel(
                tool_calls,
                max_parallel or self.config.max_parallel_tools,
                timeout_ms or self.config.total_timeout_ms,
            )
            
            total_time_ms = (time.time() - start_time) * 1000
            
            return {
                "execution_id": execution_id,
                "results": results,
                "total_time_ms": total_time_ms,
                "status": "completed",
                "tools_executed": len(results),
                "successful_tools": sum(1 for r in results if r.success),
            }
            
        except Exception as e:
            logger.error(f"Execution {execution_id} failed: {e}")
            return {
                "execution_id": execution_id,
                "error": str(e),
                "total_time_ms": (time.time() - start_time) * 1000,
                "status": "failed",
            }
    
    async def stream_execute(
        self,
        prompt: str,
        tools: Optional[List[str]] = None,
        max_parallel: Optional[int] = None,
    ) -> AsyncIterator[ToolResult]:
        """
        Execute tools and stream results as they complete.
        
        Args:
            prompt: Input prompt for the LLM
            tools: Specific tools to use
            max_parallel: Override max parallel execution
            
        Yields:
            ToolResult objects as they complete
        """
        # Get LLM decision
        tool_calls = await self._get_llm_tool_calls(prompt, tools)
        
        if not tool_calls:
            return
        
        # Create tasks for all tool calls
        tasks = []
        for call in tool_calls:
            task = asyncio.create_task(
                self._execute_single_tool(call["name"], call.get("args", {}))
            )
            tasks.append(task)
        
        # Yield results as they complete
        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                yield result
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                # Could yield error result here
    
    async def _get_llm_tool_calls(
        self, prompt: str, tools: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get tool calls from LLM based on prompt."""
        if not self.llm_client:
            # For demo purposes, simulate LLM decision
            available_tools = tools or list(self.registry._tools.keys())
            return [
                {"name": tool_name, "args": {}}
                for tool_name in available_tools[:3]  # Simulate calling first 3 tools
            ]
        
        # TODO: Implement actual LLM integration
        # This would use the LLM client to get tool decisions
        return []
    
    async def _execute_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
        max_parallel: int,
        timeout_ms: int,
    ) -> List[ToolResult]:
        """Execute multiple tools in parallel with limits."""
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def execute_with_semaphore(call: Dict[str, Any]) -> ToolResult:
            async with semaphore:
                return await self._execute_single_tool(
                    call["name"], call.get("args", {})
                )
        
        # Create tasks
        tasks = [
            asyncio.create_task(execute_with_semaphore(call))
            for call in tool_calls
        ]
        
        try:
            # Wait for all tasks with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout_ms / 1000.0,
            )
            
            # Process results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Create error result
                    error_result = ToolResult.error_result(
                        tool_name=tool_calls[i]["name"],
                        error=result,
                        execution_time_ms=0,
                    )
                    processed_results.append(error_result)
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            raise TimeoutError(
                operation="parallel_tool_execution",
                timeout_seconds=timeout_ms / 1000.0,
            )
    
    async def _execute_single_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> ToolResult:
        """Execute a single tool with timing and error handling."""
        start_time = time.time()
        
        # Get tool metadata
        tool_metadata = self.registry.get_tool(tool_name)
        if not tool_metadata:
            raise ToolExecutionError(
                tool_name=tool_name,
                message=f"Tool '{tool_name}' not found in registry",
            )
        
        try:
            # Apply tool-specific timeout if configured
            timeout_seconds = None
            if tool_metadata.timeout_ms:
                timeout_seconds = tool_metadata.timeout_ms / 1000.0
            
            # Execute the tool
            if timeout_seconds:
                result = await asyncio.wait_for(
                    tool_metadata.function(**args),
                    timeout=timeout_seconds,
                )
            else:
                result = await tool_metadata.function(**args)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return ToolResult.success_result(
                tool_name=tool_name,
                data=result,
                execution_time_ms=execution_time_ms,
                metadata={
                    "args": args,
                    "priority": tool_metadata.priority,
                    "tags": tool_metadata.tags,
                },
            )
            
        except asyncio.TimeoutError:
            execution_time_ms = (time.time() - start_time) * 1000
            error = ToolExecutionError(
                tool_name=tool_name,
                message=f"Tool timed out after {timeout_seconds}s",
            )
            return ToolResult.error_result(
                tool_name=tool_name,
                error=error,
                execution_time_ms=execution_time_ms,
            )
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            error = ToolExecutionError(
                tool_name=tool_name,
                message=f"Tool execution failed: {str(e)}",
                original_error=e,
            )
            return ToolResult.error_result(
                tool_name=tool_name,
                error=error,
                execution_time_ms=execution_time_ms,
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current orchestrator metrics."""
        return {
            "registered_tools": len(self.registry._tools),
            "registered_chains": len(self.registry._chains),
            "active_tasks": len(self._active_tasks),
            "cached_results": len(self._results_cache),
            "speculation_tasks": len(self._speculation_tasks),
            "config": {
                "max_parallel_tools": self.config.max_parallel_tools,
                "tool_timeout_ms": self.config.tool_timeout_ms,
                "total_timeout_ms": self.config.total_timeout_ms,
            },
        }
    
    async def cleanup(self) -> None:
        """Clean up resources and cancel active tasks."""
        # Cancel active tasks
        for task in self._active_tasks.values():
            if not task.done():
                task.cancel()
        
        # Cancel speculation tasks
        for task in self._speculation_tasks.values():
            if not task.done():
                task.cancel()
        
        # Wait for cancellations
        if self._active_tasks:
            await asyncio.gather(
                *self._active_tasks.values(),
                return_exceptions=True,
            )
        
        if self._speculation_tasks:
            await asyncio.gather(
                *self._speculation_tasks.values(),
                return_exceptions=True,
            )
        
        # Clear state
        self._active_tasks.clear()
        self._results_cache.clear()
        self._speculation_tasks.clear()
        
        logger.info("AsyncOrchestrator cleanup completed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()