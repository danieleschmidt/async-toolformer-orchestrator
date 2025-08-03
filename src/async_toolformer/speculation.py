"""Speculative execution engine for pre-fetching likely tool calls."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import time

from .tools import ToolResult
from .exceptions import SpeculationError

logger = logging.getLogger(__name__)


@dataclass
class SpeculationContext:
    """Context for speculative execution."""
    prompt: str
    confidence_threshold: float = 0.8
    max_speculations: int = 5
    speculation_model: str = "gpt-3.5-turbo"
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SpeculationResult:
    """Result of a speculative execution."""
    tool_name: str
    args: Dict[str, Any]
    confidence: float
    result: Optional[ToolResult] = None
    committed: bool = False
    cancelled: bool = False
    execution_time_ms: float = 0


class SpeculativeEngine:
    """
    Engine for speculative tool execution.
    
    Pre-fetches likely tool calls using a fast model before the main LLM confirms.
    """
    
    def __init__(
        self,
        orchestrator,
        speculation_model: str = "gpt-3.5-turbo",
        confidence_threshold: float = 0.8,
        max_speculations: int = 5,
        cache_ttl_seconds: int = 300,
    ):
        """
        Initialize the speculative engine.
        
        Args:
            orchestrator: Parent orchestrator instance
            speculation_model: Fast model for predictions
            confidence_threshold: Minimum confidence to speculate
            max_speculations: Maximum concurrent speculations
            cache_ttl_seconds: Cache TTL for speculation results
        """
        self.orchestrator = orchestrator
        self.speculation_model = speculation_model
        self.confidence_threshold = confidence_threshold
        self.max_speculations = max_speculations
        self.cache_ttl_seconds = cache_ttl_seconds
        
        # Track active speculations
        self._active_speculations: Dict[str, asyncio.Task] = {}
        self._speculation_results: Dict[str, SpeculationResult] = {}
        self._speculation_cache: Dict[str, Tuple[float, List[SpeculationResult]]] = {}
        
        # Metrics
        self._metrics = {
            "total_speculations": 0,
            "successful_commits": 0,
            "cancelled_speculations": 0,
            "cache_hits": 0,
            "average_confidence": 0.0,
        }
    
    async def predict_tools(
        self, context: SpeculationContext
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Predict likely tool calls based on context.
        
        Args:
            context: Speculation context with prompt and history
            
        Returns:
            List of (tool_name, args, confidence) tuples
        """
        # Check cache first
        cache_key = self._get_cache_key(context)
        if cache_key in self._speculation_cache:
            timestamp, cached_results = self._speculation_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl_seconds:
                self._metrics["cache_hits"] += 1
                return [
                    (r.tool_name, r.args, r.confidence)
                    for r in cached_results
                ]
        
        # Use fast model to predict tools
        predictions = await self._get_predictions_from_model(context)
        
        # Filter by confidence threshold
        filtered = [
            (name, args, conf)
            for name, args, conf in predictions
            if conf >= self.confidence_threshold
        ]
        
        # Limit to max speculations
        filtered = filtered[:self.max_speculations]
        
        # Update metrics
        if filtered:
            avg_conf = sum(conf for _, _, conf in filtered) / len(filtered)
            self._metrics["average_confidence"] = (
                self._metrics["average_confidence"] * 0.9 + avg_conf * 0.1
            )
        
        return filtered
    
    async def start_speculation(
        self, context: SpeculationContext
    ) -> List[str]:
        """
        Start speculative execution for predicted tools.
        
        Args:
            context: Speculation context
            
        Returns:
            List of speculation IDs
        """
        predictions = await self.predict_tools(context)
        speculation_ids = []
        
        for tool_name, args, confidence in predictions:
            speculation_id = self._generate_speculation_id(tool_name, args)
            
            # Skip if already speculating
            if speculation_id in self._active_speculations:
                continue
            
            # Create speculation result
            result = SpeculationResult(
                tool_name=tool_name,
                args=args,
                confidence=confidence,
            )
            self._speculation_results[speculation_id] = result
            
            # Start speculative execution
            task = asyncio.create_task(
                self._execute_speculation(speculation_id, result)
            )
            self._active_speculations[speculation_id] = task
            speculation_ids.append(speculation_id)
            
            self._metrics["total_speculations"] += 1
            
            logger.debug(
                f"Started speculation for {tool_name} with confidence {confidence:.2f}"
            )
        
        return speculation_ids
    
    async def commit_speculation(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Optional[ToolResult]:
        """
        Commit a speculation if it matches the actual tool call.
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
            
        Returns:
            Cached result if speculation was successful, None otherwise
        """
        speculation_id = self._generate_speculation_id(tool_name, args)
        
        if speculation_id not in self._speculation_results:
            return None
        
        result = self._speculation_results[speculation_id]
        
        # Wait for speculation to complete if still running
        if speculation_id in self._active_speculations:
            task = self._active_speculations[speculation_id]
            if not task.done():
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Speculation {speculation_id} timed out")
                    return None
        
        if result.result and result.result.success:
            result.committed = True
            self._metrics["successful_commits"] += 1
            logger.info(
                f"Committed speculation for {tool_name} "
                f"(saved {result.execution_time_ms:.0f}ms)"
            )
            return result.result
        
        return None
    
    async def cancel_speculation(self, speculation_id: str) -> None:
        """Cancel a speculation."""
        if speculation_id in self._active_speculations:
            task = self._active_speculations[speculation_id]
            if not task.done():
                task.cancel()
                self._metrics["cancelled_speculations"] += 1
        
        if speculation_id in self._speculation_results:
            self._speculation_results[speculation_id].cancelled = True
    
    async def cancel_uncommitted_speculations(self) -> None:
        """Cancel all uncommitted speculations."""
        for spec_id, result in self._speculation_results.items():
            if not result.committed and not result.cancelled:
                await self.cancel_speculation(spec_id)
    
    async def _execute_speculation(
        self, speculation_id: str, result: SpeculationResult
    ) -> None:
        """Execute a speculative tool call."""
        start_time = time.time()
        
        try:
            # Execute the tool
            tool_result = await self.orchestrator._execute_single_tool(
                result.tool_name, result.args
            )
            result.result = tool_result
            result.execution_time_ms = (time.time() - start_time) * 1000
            
            logger.debug(
                f"Speculation {speculation_id} completed in "
                f"{result.execution_time_ms:.0f}ms"
            )
            
        except Exception as e:
            logger.error(f"Speculation {speculation_id} failed: {e}")
            result.execution_time_ms = (time.time() - start_time) * 1000
        
        finally:
            # Remove from active speculations
            self._active_speculations.pop(speculation_id, None)
    
    async def _get_predictions_from_model(
        self, context: SpeculationContext
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Get tool predictions from the speculation model.
        
        This is a simplified implementation. In production, this would
        call the actual LLM API with the speculation model.
        """
        # Simulate fast model predictions based on keywords in prompt
        prompt_lower = context.prompt.lower()
        predictions = []
        
        # Pattern matching for common tool patterns
        patterns = {
            "search": ["search", "find", "look for", "query"],
            "analyze": ["analyze", "examine", "inspect", "review"],
            "fetch": ["get", "fetch", "retrieve", "download"],
            "calculate": ["calculate", "compute", "measure", "count"],
            "transform": ["convert", "transform", "translate", "format"],
        }
        
        for tool_type, keywords in patterns.items():
            if any(keyword in prompt_lower for keyword in keywords):
                # Generate predictions based on available tools
                for tool_name in self.orchestrator.registry._tools.keys():
                    if tool_type in tool_name.lower():
                        confidence = 0.85 + (0.1 if tool_type == "search" else 0.0)
                        predictions.append((tool_name, {}, confidence))
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[2], reverse=True)
        
        return predictions
    
    def _generate_speculation_id(
        self, tool_name: str, args: Dict[str, Any]
    ) -> str:
        """Generate a unique ID for a speculation."""
        import hashlib
        import json
        
        # Create stable hash from tool name and args
        content = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_cache_key(self, context: SpeculationContext) -> str:
        """Generate cache key for speculation context."""
        import hashlib
        
        # Use prompt and model as cache key
        content = f"{context.speculation_model}:{context.prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get speculation engine metrics."""
        hit_rate = 0.0
        if self._metrics["total_speculations"] > 0:
            hit_rate = (
                self._metrics["successful_commits"] /
                self._metrics["total_speculations"]
            )
        
        return {
            **self._metrics,
            "hit_rate": hit_rate,
            "active_speculations": len(self._active_speculations),
            "cached_contexts": len(self._speculation_cache),
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Cancel all active speculations
        for task in self._active_speculations.values():
            if not task.done():
                task.cancel()
        
        if self._active_speculations:
            await asyncio.gather(
                *self._active_speculations.values(),
                return_exceptions=True,
            )
        
        self._active_speculations.clear()
        self._speculation_results.clear()
        self._speculation_cache.clear()


class SpeculativeOrchestrator:
    """
    Orchestrator with speculative execution capabilities.
    
    Extends the base orchestrator with speculation engine.
    """
    
    def __init__(
        self,
        orchestrator,
        speculation_model: str = "gpt-3.5-turbo",
        confidence_threshold: float = 0.8,
        enable_speculation: bool = True,
    ):
        """
        Initialize speculative orchestrator.
        
        Args:
            orchestrator: Base orchestrator instance
            speculation_model: Model for speculation
            confidence_threshold: Minimum confidence
            enable_speculation: Whether to enable speculation
        """
        self.orchestrator = orchestrator
        self.enable_speculation = enable_speculation
        
        if enable_speculation:
            self.speculation_engine = SpeculativeEngine(
                orchestrator=orchestrator,
                speculation_model=speculation_model,
                confidence_threshold=confidence_threshold,
            )
        else:
            self.speculation_engine = None
    
    async def execute(
        self,
        prompt: str,
        tools: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute with speculative pre-fetching.
        
        Args:
            prompt: Input prompt
            tools: Available tools
            **kwargs: Additional arguments
            
        Returns:
            Execution results
        """
        if not self.enable_speculation or not self.speculation_engine:
            return await self.orchestrator.execute(prompt, tools, **kwargs)
        
        # Start speculation
        context = SpeculationContext(prompt=prompt)
        speculation_ids = await self.speculation_engine.start_speculation(context)
        
        # Get actual tool calls from LLM
        result = await self.orchestrator.execute(prompt, tools, **kwargs)
        
        # Check if any speculations were successful
        speculation_hits = 0
        if "results" in result:
            for tool_result in result["results"]:
                cached = await self.speculation_engine.commit_speculation(
                    tool_result.tool_name,
                    tool_result.metadata.get("args", {}),
                )
                if cached:
                    speculation_hits += 1
        
        # Cancel uncommitted speculations
        await self.speculation_engine.cancel_uncommitted_speculations()
        
        # Add speculation metrics to result
        result["speculation_metrics"] = {
            "enabled": True,
            "speculations_started": len(speculation_ids),
            "speculation_hits": speculation_hits,
            **self.speculation_engine.get_metrics(),
        }
        
        return result
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.speculation_engine:
            await self.speculation_engine.cleanup()
        await self.orchestrator.cleanup()