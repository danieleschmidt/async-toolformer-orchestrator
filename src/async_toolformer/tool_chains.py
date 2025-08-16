"""Tool chains and composition for complex workflows."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from .exceptions import ToolChainError

logger = logging.getLogger(__name__)


@dataclass
class ChainStep:
    """Represents a step in a tool chain."""
    name: str
    function: Callable
    args: dict[str, Any]
    depends_on: list[str] = None
    parallel: bool = False
    retry_count: int = 0
    timeout_seconds: float | None = None


class ToolChain:
    """
    Represents a chain of tools that execute in sequence or parallel.
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize a tool chain.

        Args:
            name: Chain name
            description: Chain description
        """
        self.name = name
        self.description = description
        self.steps: list[ChainStep] = []
        self._results: dict[str, Any] = {}

    def add_step(
        self,
        function: Callable,
        args: dict[str, Any] | None = None,
        name: str | None = None,
        depends_on: list[str] | None = None,
        parallel: bool = False,
        retry_count: int = 0,
        timeout_seconds: float | None = None,
    ) -> str:
        """
        Add a step to the chain.

        Args:
            function: Function to execute
            args: Function arguments
            name: Step name (auto-generated if not provided)
            depends_on: List of step names this depends on
            parallel: Whether this can run in parallel with siblings
            retry_count: Number of retries on failure
            timeout_seconds: Timeout for this step

        Returns:
            Step name
        """
        if name is None:
            name = f"{function.__name__}_{len(self.steps)}"

        step = ChainStep(
            name=name,
            function=function,
            args=args or {},
            depends_on=depends_on or [],
            parallel=parallel,
            retry_count=retry_count,
            timeout_seconds=timeout_seconds,
        )

        self.steps.append(step)
        return name

    async def execute(self, initial_context: dict[str, Any] | None = None) -> Any:
        """
        Execute the tool chain.

        Args:
            initial_context: Initial context for the chain

        Returns:
            Final result of the chain
        """
        context = initial_context or {}
        self._results.clear()

        # Group steps by dependencies
        execution_groups = self._group_steps_by_dependencies()

        # Execute groups in order
        for group in execution_groups:
            if len(group) == 1 and not group[0].parallel:
                # Execute single step
                result = await self._execute_step(group[0], context)
                self._results[group[0].name] = result
                context[group[0].name] = result
            else:
                # Execute parallel steps
                tasks = [
                    self._execute_step(step, context)
                    for step in group
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for step, result in zip(group, results, strict=False):
                    if isinstance(result, Exception):
                        logger.error(f"Step {step.name} failed: {result}")
                        raise ToolChainError(
                            chain_name=self.name,
                            message=f"Step {step.name} failed: {result}",
                        )
                    self._results[step.name] = result
                    context[step.name] = result

        # Return the last result
        if self.steps:
            return self._results.get(self.steps[-1].name)
        return None

    async def _execute_step(
        self, step: ChainStep, context: dict[str, Any]
    ) -> Any:
        """Execute a single step with retries and timeout."""
        # Resolve arguments from context
        resolved_args = self._resolve_arguments(step.args, context)

        # Add retry logic
        last_error = None
        for attempt in range(step.retry_count + 1):
            try:
                # Apply timeout if specified
                if step.timeout_seconds:
                    result = await asyncio.wait_for(
                        step.function(**resolved_args),
                        timeout=step.timeout_seconds,
                    )
                else:
                    result = await step.function(**resolved_args)

                logger.debug(f"Step {step.name} completed successfully")
                return result

            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(
                    f"Step {step.name} timed out after {step.timeout_seconds}s "
                    f"(attempt {attempt + 1}/{step.retry_count + 1})"
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Step {step.name} failed: {e} "
                    f"(attempt {attempt + 1}/{step.retry_count + 1})"
                )

            if attempt < step.retry_count:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        raise ToolChainError(
            chain_name=self.name,
            message=f"Step {step.name} failed after {step.retry_count + 1} attempts",
            original_error=last_error,
        )

    def _resolve_arguments(
        self, args: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve arguments from context."""
        resolved = {}

        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$"):
                # Reference to context variable
                context_key = value[1:]
                resolved[key] = context.get(context_key, value)
            else:
                resolved[key] = value

        return resolved

    def _group_steps_by_dependencies(self) -> list[list[ChainStep]]:
        """Group steps by their dependencies for execution order."""
        groups = []
        executed = set()

        while len(executed) < len(self.steps):
            current_group = []

            for step in self.steps:
                if step.name in executed:
                    continue

                # Check if all dependencies are satisfied
                if all(dep in executed for dep in step.depends_on):
                    current_group.append(step)

            if not current_group:
                # Circular dependency or invalid configuration
                raise ToolChainError(
                    chain_name=self.name,
                    message="Circular dependency detected in chain steps",
                )

            groups.append(current_group)
            executed.update(step.name for step in current_group)

        return groups


def parallel(*coroutines) -> Callable:
    """
    Execute multiple coroutines in parallel.

    Args:
        *coroutines: Coroutines to execute

    Returns:
        Function that executes coroutines in parallel
    """
    async def execute():
        return await asyncio.gather(*coroutines)
    return execute


def sequential(*coroutines) -> Callable:
    """
    Execute multiple coroutines sequentially.

    Args:
        *coroutines: Coroutines to execute

    Returns:
        Function that executes coroutines sequentially
    """
    async def execute():
        results = []
        for coro in coroutines:
            result = await coro
            results.append(result)
        return results
    return execute


def chain(*functions) -> Callable:
    """
    Chain functions where output of one feeds into the next.

    Args:
        *functions: Functions to chain

    Returns:
        Chained function
    """
    async def execute(initial_input):
        result = initial_input
        for func in functions:
            if asyncio.iscoroutinefunction(func):
                result = await func(result)
            else:
                result = func(result)
        return result
    return execute


def retry(max_attempts: int = 3, delay_seconds: float = 1.0):
    """
    Decorator to add retry logic to a tool function.

    Args:
        max_attempts: Maximum retry attempts
        delay_seconds: Delay between retries
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        await asyncio.sleep(delay_seconds * (2 ** attempt))
            raise last_error
        return wrapper
    return decorator


def timeout(seconds: float):
    """
    Decorator to add timeout to a tool function.

    Args:
        seconds: Timeout in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=seconds
            )
        return wrapper
    return decorator


def conditional(condition: Callable[[Any], bool]):
    """
    Decorator to conditionally execute a tool.

    Args:
        condition: Function that returns True if tool should execute
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Check condition with first argument
            if args and condition(args[0]):
                return await func(*args, **kwargs)
            return None
        return wrapper
    return decorator


class ChainBuilder:
    """
    Builder for creating complex tool chains.
    """

    def __init__(self, name: str):
        """Initialize chain builder."""
        self.chain = ToolChain(name)
        self._current_group = []
        self._last_step_names = []

    def add(self, function: Callable, **kwargs) -> 'ChainBuilder':
        """Add a sequential step."""
        name = self.chain.add_step(
            function,
            depends_on=self._last_step_names.copy(),
            **kwargs
        )
        self._last_step_names = [name]
        return self

    def add_parallel(self, *functions, **common_kwargs) -> 'ChainBuilder':
        """Add parallel steps."""
        step_names = []
        for func in functions:
            name = self.chain.add_step(
                func,
                depends_on=self._last_step_names.copy(),
                parallel=True,
                **common_kwargs
            )
            step_names.append(name)
        self._last_step_names = step_names
        return self

    def add_conditional(
        self,
        condition: Callable,
        if_true: Callable,
        if_false: Callable | None = None,
        **kwargs
    ) -> 'ChainBuilder':
        """Add conditional branching."""
        # Wrap in conditional execution
        async def conditional_step(context):
            if condition(context):
                return await if_true(context)
            elif if_false:
                return await if_false(context)
            return None

        return self.add(conditional_step, **kwargs)

    def add_loop(
        self,
        function: Callable,
        condition: Callable[[Any], bool],
        max_iterations: int = 100,
        **kwargs
    ) -> 'ChainBuilder':
        """Add a loop step."""
        async def loop_step(context):
            results = []
            for _i in range(max_iterations):
                if not condition(context):
                    break
                result = await function(context)
                results.append(result)
                context = result  # Update context for next iteration
            return results

        return self.add(loop_step, **kwargs)

    def build(self) -> ToolChain:
        """Build and return the chain."""
        return self.chain


def create_map_reduce_chain(
    map_function: Callable,
    reduce_function: Callable,
    items: list[Any],
    name: str = "map_reduce",
    max_parallel: int = 10,
) -> ToolChain:
    """
    Create a map-reduce style tool chain.

    Args:
        map_function: Function to map over items
        reduce_function: Function to reduce results
        items: Items to process
        name: Chain name
        max_parallel: Maximum parallel map operations

    Returns:
        Configured tool chain
    """
    chain = ToolChain(name)

    # Add map steps
    semaphore = asyncio.Semaphore(max_parallel)

    async def limited_map(item):
        async with semaphore:
            return await map_function(item)

    # Create map tasks
    async def map_phase():
        tasks = [limited_map(item) for item in items]
        return await asyncio.gather(*tasks)

    # Add steps to chain
    chain.add_step(map_phase, name="map_phase")
    chain.add_step(
        reduce_function,
        args={"items": "$map_phase"},
        name="reduce_phase",
        depends_on=["map_phase"],
    )

    return chain
