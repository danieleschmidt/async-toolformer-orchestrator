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
from .simple_structured_logging import get_logger, CorrelationContext, log_execution_time
from .error_recovery import error_recovery, RecoveryPolicy, RecoveryStrategy
from .health_monitor import health_monitor
from .input_validation import tool_validator, ValidationLevel
from .exceptions import (
    OrchestratorError,
    ToolExecutionError,
    TimeoutError,
    ConfigurationError,
    RateLimitError,
)

logger = get_logger(__name__)


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
        
        # Set up error recovery policies
        self._setup_recovery_policies()
        
        # Register health checks
        self._register_health_checks()
        
        logger.info(
            "AsyncOrchestrator initialized successfully",
            registered_tools=len(self.registry._tools),
            max_parallel_tools=self.config.max_parallel_tools,
            tool_timeout_ms=self.config.tool_timeout_ms
        )
    
    def register_tool(self, tool: ToolFunction) -> None:
        """Register a tool function."""
        self.registry.register_tool(tool)
        logger.debug(f"Registered tool: {tool.__name__}")
    
    def register_chain(self, chain: ToolFunction) -> None:
        """Register a tool chain function."""
        self.registry.register_chain(chain)
        logger.debug(f"Registered tool chain: {chain.__name__}")
    
    def _setup_recovery_policies(self):
        """Set up error recovery policies for different components."""
        
        # LLM integration recovery - retry with exponential backoff
        error_recovery.register_policy(
            "llm_integration",
            RecoveryPolicy(
                strategy=RecoveryStrategy.RETRY,
                max_retries=3,
                backoff_factor=2.0,
                timeout_ms=self.config.llm_timeout_ms
            )
        )
        
        # Tool execution recovery - circuit breaker for failing tools
        error_recovery.register_policy(
            "tool_execution",
            RecoveryPolicy(
                strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
                max_retries=1,
                timeout_ms=self.config.total_timeout_ms
            )
        )
        
        # Individual tool recovery - retry with backoff
        error_recovery.register_policy(
            "single_tool",
            RecoveryPolicy(
                strategy=RecoveryStrategy.RETRY,
                max_retries=2,
                backoff_factor=1.5,
                timeout_ms=self.config.tool_timeout_ms
            )
        )
        
        logger.debug("Error recovery policies configured")
    
    def _register_health_checks(self):
        """Register orchestrator-specific health checks."""
        
        from .health_monitor import HealthCheck
        
        async def orchestrator_health():
            """Check orchestrator health."""
            try:
                metrics = await self.get_metrics()
                
                # Check if we have tools registered
                if metrics.get("registered_tools", 0) == 0:
                    return {
                        "status": "degraded",
                        "message": "No tools registered"
                    }
                
                # Check active tasks
                active_tasks = metrics.get("active_tasks", 0)
                if active_tasks > self.config.max_parallel_tools * 2:
                    return {
                        "status": "degraded",
                        "message": f"High number of active tasks: {active_tasks}"
                    }
                
                return {
                    "status": "healthy",
                    "message": "Orchestrator functioning normally",
                    "details": {
                        "registered_tools": metrics.get("registered_tools"),
                        "active_tasks": active_tasks
                    }
                }
                
            except Exception as e:
                return {
                    "status": "unhealthy",
                    "message": f"Health check failed: {str(e)}"
                }
        
        health_monitor.register_check(HealthCheck(
            name="orchestrator",
            check_function=orchestrator_health,
            interval_seconds=30
        ))
        
        logger.debug("Health checks registered")
    
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
    
    @log_execution_time("orchestrator_execute")
    async def execute(
        self,
        prompt: str,
        tools: Optional[List[str]] = None,
        max_parallel: Optional[int] = None,
        timeout_ms: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute tools based on LLM decision with enhanced robustness.
        
        Args:
            prompt: Input prompt for the LLM
            tools: Specific tools to use (defaults to all registered)
            max_parallel: Override max parallel execution
            timeout_ms: Override timeout
            user_id: User identifier for tracking
            
        Returns:
            Dictionary containing execution results
        """
        start_time = time.time()
        execution_id = f"exec_{int(start_time * 1000)}"
        
        # Set up correlation context for distributed tracing
        with CorrelationContext(
            execution_id_value=execution_id,
            user_id_value=user_id or ""
        ):
            try:
                logger.info(
                    "Starting orchestrator execution",
                    prompt_length=len(prompt),
                    requested_tools=tools,
                    max_parallel=max_parallel,
                    timeout_ms=timeout_ms
                )
                
                # Validate and sanitize prompt input
                validation_result = tool_validator.validate_and_sanitize(
                    {"prompt": prompt, "tools": tools or []},
                    "orchestrator.execute"
                )
                
                if not validation_result.is_valid:
                    logger.error(
                        "Input validation failed",
                        errors=validation_result.errors
                    )
                    return {
                        "execution_id": execution_id,
                        "error": "Input validation failed",
                        "validation_errors": validation_result.errors,
                        "total_time_ms": (time.time() - start_time) * 1000,
                        "status": "validation_failed",
                    }
                
                # Use sanitized inputs
                sanitized_data = validation_result.sanitized_data
                clean_prompt = sanitized_data["prompt"]
                clean_tools = sanitized_data["tools"]
                
                # Get LLM decision on which tools to call with error recovery
                tool_calls = await error_recovery.execute_with_recovery(
                    "llm_integration",
                    self._get_llm_tool_calls,
                    clean_prompt,
                    clean_tools
                )
                
                if not tool_calls:
                    logger.info("No tools selected by LLM")
                    return {
                        "execution_id": execution_id,
                        "results": [],
                        "total_time_ms": (time.time() - start_time) * 1000,
                        "status": "no_tools_called",
                    }
                
                logger.info(
                    "LLM selected tools for execution",
                    tool_count=len(tool_calls),
                    selected_tools=[tc["name"] for tc in tool_calls]
                )
                
                # Execute tools in parallel with recovery
                results = await error_recovery.execute_with_recovery(
                    "tool_execution",
                    self._execute_tools_parallel,
                    tool_calls,
                    max_parallel or self.config.max_parallel_tools,
                    timeout_ms or self.config.total_timeout_ms,
                )
                
                total_time_ms = (time.time() - start_time) * 1000
                successful_count = sum(1 for r in results if r.success)
                
                logger.info(
                    "Orchestrator execution completed",
                    total_time_ms=total_time_ms,
                    tools_executed=len(results),
                    successful_tools=successful_count,
                    success_rate=successful_count / len(results) if results else 0
                )
                
                return {
                    "execution_id": execution_id,
                    "results": results,
                    "total_time_ms": total_time_ms,
                    "status": "completed",
                    "tools_executed": len(results),
                    "successful_tools": successful_count,
                    "success_rate": successful_count / len(results) if results else 0,
                }
                
            except Exception as e:
                total_time_ms = (time.time() - start_time) * 1000
                
                logger.error(
                    "Orchestrator execution failed",
                    error=e,
                    total_time_ms=total_time_ms,
                    execution_stage="unknown"
                )
                
                return {
                    "execution_id": execution_id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "total_time_ms": total_time_ms,
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
    
    @log_execution_time("single_tool_execution")
    async def _execute_single_tool(
        self, tool_name: str, args: Dict[str, Any]
    ) -> ToolResult:
        """Execute a single tool with comprehensive error handling and recovery."""
        start_time = time.time()
        
        # Get tool metadata
        tool_metadata = self.registry.get_tool(tool_name)
        if not tool_metadata:
            error = ToolExecutionError(
                tool_name=tool_name,
                message=f"Tool '{tool_name}' not found in registry",
            )
            logger.error("Tool not found in registry", tool_name=tool_name)
            raise error
        
        # Validate tool inputs
        validation_result = tool_validator.validate_tool_input(tool_name, args)
        if not validation_result.is_valid:
            logger.error(
                "Tool input validation failed",
                tool_name=tool_name,
                errors=validation_result.errors
            )
            return ToolResult.error_result(
                tool_name=tool_name,
                error=ToolExecutionError(
                    tool_name=tool_name,
                    message=f"Input validation failed: {validation_result.errors}"
                ),
                execution_time_ms=(time.time() - start_time) * 1000
            )
        
        # Use validated arguments
        clean_args = validation_result.sanitized_data
        
        logger.debug(
            "Starting tool execution",
            tool_name=tool_name,
            arguments=clean_args,
            priority=tool_metadata.priority
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