"""Core AsyncOrchestrator implementation."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union, AsyncIterator
import logging

from .config import OrchestratorConfig
from .tools import ToolFunction, ToolMetadata, ToolRegistry, ToolResult
from .llm_integration import LLMIntegration, create_llm_integration
from .simple_rate_limiter import SimpleRateLimiter
from .caching import ToolResultCache, create_memory_cache
from .connection_pool import ConnectionPoolManager, PoolConfig
from .exceptions import (
    OrchestratorError,
    ToolExecutionError,
    TimeoutError,
    ConfigurationError,
    RateLimitError,
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
        llm_integration: Optional[LLMIntegration] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the AsyncOrchestrator.
        
        Args:
            llm_client: LLM client (OpenAI, Anthropic, etc.) - for backward compatibility
            tools: List of tool functions to register
            config: Configuration object
            llm_integration: Pre-configured LLM integration instance
            **kwargs: Additional configuration parameters
        """
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
        
        # Set up LLM integration
        if llm_integration:
            self.llm_integration = llm_integration
        elif llm_client:
            # Auto-detect client type and create integration
            self.llm_integration = self._create_llm_integration_from_client(llm_client)
        else:
            # Create mock integration for testing
            self.llm_integration = create_llm_integration(use_mock=True)
        
        # Set up rate limiter
        self.rate_limiter = SimpleRateLimiter(self.config.rate_limit_config)
        
        # Set up caching
        cache_enabled = self.config.memory_config.compress_results
        cache_size = int(self.config.memory_config.max_memory_gb * 100)  # Rough approximation
        self.cache = create_memory_cache(
            max_size=cache_size,
            default_ttl=3600,  # 1 hour default
            enable_compression=cache_enabled
        )
        
        # Set up connection pooling
        pool_config = PoolConfig(
            max_connections=self.config.max_parallel_tools * 2,
            max_connections_per_host=self.config.max_parallel_per_type,
            timeout_seconds=self.config.tool_timeout_ms / 1000.0
        )
        self.connection_pool = ConnectionPoolManager(pool_config)
        
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
    
    def _create_llm_integration_from_client(self, client: Any) -> LLMIntegration:
        """Create LLM integration from a client instance."""
        client_type = client.__class__.__name__
        
        if "OpenAI" in client_type or "AsyncOpenAI" in client_type:
            return create_llm_integration(openai_client=client, default_provider="openai")
        elif "Anthropic" in client_type or "AsyncAnthropic" in client_type:
            return create_llm_integration(anthropic_client=client, default_provider="anthropic")
        else:
            logger.warning(f"Unknown client type: {client_type}, using mock provider")
            return create_llm_integration(use_mock=True)
    
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
            logger.info(f"Starting execution {execution_id}")
            
            # Get LLM decision on which tools to call
            tool_calls = await self._get_llm_tool_calls(prompt, tools)
            
            if not tool_calls:
                logger.info(f"No tools called for execution {execution_id}")
                return {
                    "execution_id": execution_id,
                    "results": [],
                    "total_time_ms": (time.time() - start_time) * 1000,
                    "status": "no_tools_called",
                }
            
            logger.info(f"Executing {len(tool_calls)} tools for {execution_id}")
            
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
        # Prepare tools for LLM
        available_tools = tools or list(self.registry._tools.keys())
        tool_definitions = []
        
        for tool_name in available_tools:
            tool_metadata = self.registry.get_tool(tool_name)
            if tool_metadata:
                # Create tool definition for LLM
                tool_def = {
                    "name": tool_name,
                    "description": tool_metadata.description,
                    "parameters": self._get_tool_parameters(tool_metadata.function)
                }
                tool_definitions.append(tool_def)
        
        if not tool_definitions:
            logger.warning("No tools available for LLM")
            return []
        
        # Get tool calls from LLM
        try:
            tool_calls = await self.llm_integration.get_tool_calls(
                prompt=prompt,
                tools=tool_definitions
            )
            
            # Convert to dict format
            result = []
            for tc in tool_calls:
                result.append({
                    "name": tc.name,
                    "args": tc.arguments,
                    "id": tc.id,
                    "metadata": tc.metadata
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get LLM tool calls: {e}")
            return []
    
    def _get_tool_parameters(self, func: ToolFunction) -> Dict[str, Any]:
        """Extract parameter schema from a tool function."""
        import inspect
        
        try:
            sig = inspect.signature(func)
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for param_name, param in sig.parameters.items():
                param_info = {"type": "string"}  # Default type
                
                # Try to infer type from annotation
                if param.annotation != inspect.Parameter.empty:
                    annotation = param.annotation
                    if annotation == int:
                        param_info["type"] = "integer"
                    elif annotation == float:
                        param_info["type"] = "number"
                    elif annotation == bool:
                        param_info["type"] = "boolean"
                    elif annotation == list:
                        param_info["type"] = "array"
                    elif annotation == dict:
                        param_info["type"] = "object"
                
                # Add description if available
                if hasattr(func, '__doc__') and func.__doc__:
                    # Try to extract parameter description from docstring
                    # This is a simple implementation - could be enhanced
                    param_info["description"] = f"Parameter {param_name}"
                
                parameters["properties"][param_name] = param_info
                
                # Mark as required if no default value
                if param.default == inspect.Parameter.empty:
                    parameters["required"].append(param_name)
            
            return parameters
            
        except Exception as e:
            logger.warning(f"Could not extract parameters for function {func.__name__}: {e}")
            return {
                "type": "object",
                "properties": {},
                "required": []
            }
    
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
        
        # Check cache first
        cached_result = await self.cache.get_cached_result(tool_name, args)
        if cached_result is not None:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.debug(f"Cache hit for tool {tool_name}")
            return ToolResult.success_result(
                tool_name=tool_name,
                data=cached_result,
                execution_time_ms=execution_time_ms,
                metadata={"cached": True, "args": args}
            )

        # Check rate limits
        try:
            rate_limit_group = tool_metadata.rate_limit_group or tool_name
            await self.rate_limiter.check_limit(rate_limit_group)
        except RateLimitError as e:
            execution_time_ms = (time.time() - start_time) * 1000
            return ToolResult.error_result(
                tool_name=tool_name,
                error=e,
                execution_time_ms=execution_time_ms,
            )
        
        try:
            # Apply tool-specific timeout if configured
            timeout_seconds = None
            if tool_metadata.timeout_ms:
                timeout_seconds = tool_metadata.timeout_ms / 1000.0
            
            # Execute the tool with retry logic
            retry_attempts = tool_metadata.retry_attempts or 1
            last_exception = None
            
            for attempt in range(retry_attempts):
                try:
                    if timeout_seconds:
                        result = await asyncio.wait_for(
                            tool_metadata.function(**args),
                            timeout=timeout_seconds,
                        )
                    else:
                        result = await tool_metadata.function(**args)
                    
                    # Success - break out of retry loop
                    break
                    
                except asyncio.TimeoutError:
                    # Don't retry timeout errors
                    raise
                    
                except Exception as e:
                    last_exception = e
                    if attempt < retry_attempts - 1:
                        logger.debug(f"Tool {tool_name} attempt {attempt + 1} failed: {e}, retrying...")
                        await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    else:
                        raise
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            # Cache successful result
            try:
                cache_ttl = 3600  # 1 hour default
                if tool_metadata.tags and 'no-cache' not in tool_metadata.tags:
                    await self.cache.cache_result(tool_name, args, result, cache_ttl)
            except Exception as e:
                logger.warning(f"Failed to cache result for {tool_name}: {e}")
            
            return ToolResult.success_result(
                tool_name=tool_name,
                data=result,
                execution_time_ms=execution_time_ms,
                metadata={
                    "args": args,
                    "priority": tool_metadata.priority,
                    "tags": tool_metadata.tags,
                    "cached": False,
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
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get current orchestrator metrics."""
        # Get cache stats
        cache_stats = await self.cache.get_stats()
        
        # Get connection pool stats
        pool_stats = await self.connection_pool.get_pool_stats()
        
        return {
            "registered_tools": len(self.registry._tools),
            "registered_chains": len(self.registry._chains),
            "active_tasks": len(self._active_tasks),
            "cached_results": len(self._results_cache),
            "speculation_tasks": len(self._speculation_tasks),
            "cache": cache_stats,
            "connection_pools": pool_stats,
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
        
        # Cleanup caching and connection pooling
        try:
            await self.cache.backend.clear()
            await self.connection_pool.close_all()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        
        logger.info("AsyncOrchestrator cleanup completed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()