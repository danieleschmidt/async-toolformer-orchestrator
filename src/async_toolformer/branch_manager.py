"""Branch cancellation and management for parallel tool execution."""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum

from .tools import ToolResult
from .exceptions import BranchCancellationError

logger = logging.getLogger(__name__)


class BranchStatus(Enum):
    """Status of an execution branch."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class ExecutionBranch:
    """Represents an execution branch for a tool."""
    branch_id: str
    tool_name: str
    args: Dict[str, Any]
    task: Optional[asyncio.Task] = None
    status: BranchStatus = BranchStatus.PENDING
    result: Optional[ToolResult] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    priority: int = 0
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CancellationStrategy:
    """Strategy for branch cancellation."""
    timeout_ms: int = 5000
    cancel_on_better_result: bool = True
    keep_top_n_branches: int = 3
    cancel_slow_branches_after_ms: int = 2000
    score_threshold: float = 0.8
    custom_scorer: Optional[Callable[[ToolResult], float]] = None


class BranchManager:
    """
    Manages execution branches for parallel tool execution.
    
    Handles branch lifecycle, cancellation, and result aggregation.
    """
    
    def __init__(
        self,
        strategy: Optional[CancellationStrategy] = None,
        max_concurrent_branches: int = 50,
    ):
        """
        Initialize the branch manager.
        
        Args:
            strategy: Cancellation strategy
            max_concurrent_branches: Maximum concurrent branches
        """
        self.strategy = strategy or CancellationStrategy()
        self.max_concurrent_branches = max_concurrent_branches
        
        # Track branches
        self._branches: Dict[str, ExecutionBranch] = {}
        self._completed_branches: List[ExecutionBranch] = []
        self._cancelled_branches: List[ExecutionBranch] = []
        
        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent_branches)
        
        # Metrics
        self._metrics = {
            "total_branches": 0,
            "completed_branches": 0,
            "cancelled_branches": 0,
            "timeout_cancellations": 0,
            "score_cancellations": 0,
            "average_execution_time_ms": 0.0,
        }
    
    async def create_branch(
        self,
        tool_name: str,
        args: Dict[str, Any],
        executor: Callable,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new execution branch.
        
        Args:
            tool_name: Name of the tool
            args: Tool arguments
            executor: Async function to execute
            priority: Branch priority (higher = more important)
            metadata: Additional metadata
            
        Returns:
            Branch ID
        """
        branch_id = self._generate_branch_id(tool_name)
        
        branch = ExecutionBranch(
            branch_id=branch_id,
            tool_name=tool_name,
            args=args,
            priority=priority,
            metadata=metadata or {},
        )
        
        self._branches[branch_id] = branch
        self._metrics["total_branches"] += 1
        
        # Create execution task
        branch.task = asyncio.create_task(
            self._execute_branch(branch, executor)
        )
        
        logger.debug(f"Created branch {branch_id} for {tool_name}")
        
        return branch_id
    
    async def execute_branches(
        self,
        tools_and_args: List[Tuple[str, Dict[str, Any], Callable]],
        timeout_ms: Optional[int] = None,
    ) -> List[ToolResult]:
        """
        Execute multiple branches in parallel with management.
        
        Args:
            tools_and_args: List of (tool_name, args, executor) tuples
            timeout_ms: Overall timeout
            
        Returns:
            List of successful results
        """
        # Create branches
        branch_ids = []
        for tool_name, args, executor in tools_and_args:
            branch_id = await self.create_branch(tool_name, args, executor)
            branch_ids.append(branch_id)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_branches())
        
        try:
            # Wait for branches with timeout
            timeout = (timeout_ms or self.strategy.timeout_ms) / 1000.0
            await asyncio.wait_for(
                self._wait_for_branches(branch_ids),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Branch execution timed out after {timeout}s")
            self._metrics["timeout_cancellations"] += len(
                [b for b in self._branches.values() if b.status == BranchStatus.RUNNING]
            )
        finally:
            # Cancel monitoring
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            
            # Cancel remaining branches
            await self._cancel_remaining_branches()
        
        # Collect results
        results = []
        for branch in self._completed_branches:
            if branch.result and branch.result.success:
                results.append(branch.result)
        
        return results
    
    async def cancel_branch(self, branch_id: str, reason: str = "") -> None:
        """Cancel a specific branch."""
        if branch_id not in self._branches:
            return
        
        branch = self._branches[branch_id]
        
        if branch.status in [BranchStatus.COMPLETED, BranchStatus.CANCELLED]:
            return
        
        if branch.task and not branch.task.done():
            branch.task.cancel()
        
        branch.status = BranchStatus.CANCELLED
        branch.end_time = time.time()
        self._cancelled_branches.append(branch)
        self._metrics["cancelled_branches"] += 1
        
        logger.debug(f"Cancelled branch {branch_id}: {reason}")
    
    async def get_branch_status(self, branch_id: str) -> Optional[BranchStatus]:
        """Get the status of a branch."""
        if branch_id in self._branches:
            return self._branches[branch_id].status
        return None
    
    async def get_branch_result(self, branch_id: str) -> Optional[ToolResult]:
        """Get the result of a completed branch."""
        if branch_id in self._branches:
            branch = self._branches[branch_id]
            if branch.status == BranchStatus.COMPLETED:
                return branch.result
        return None
    
    async def _execute_branch(
        self, branch: ExecutionBranch, executor: Callable
    ) -> None:
        """Execute a single branch."""
        async with self._semaphore:
            branch.status = BranchStatus.RUNNING
            
            try:
                # Execute the tool
                result = await executor(branch.tool_name, branch.args)
                
                branch.result = result
                branch.status = BranchStatus.COMPLETED
                branch.end_time = time.time()
                
                # Calculate score
                if self.strategy.custom_scorer:
                    branch.score = self.strategy.custom_scorer(result)
                else:
                    branch.score = self._default_scorer(result)
                
                self._completed_branches.append(branch)
                self._metrics["completed_branches"] += 1
                
                # Update average execution time
                exec_time = (branch.end_time - branch.start_time) * 1000
                self._update_average_execution_time(exec_time)
                
                logger.debug(
                    f"Branch {branch.branch_id} completed with score {branch.score:.2f}"
                )
                
                # Check if we should cancel other branches
                if self.strategy.cancel_on_better_result:
                    await self._check_cancellation_on_result(branch)
                
            except asyncio.CancelledError:
                branch.status = BranchStatus.CANCELLED
                branch.end_time = time.time()
                raise
            except Exception as e:
                branch.status = BranchStatus.FAILED
                branch.end_time = time.time()
                logger.error(f"Branch {branch.branch_id} failed: {e}")
    
    async def _monitor_branches(self) -> None:
        """Monitor branches for timeout and performance-based cancellation."""
        while True:
            try:
                await asyncio.sleep(0.5)  # Check every 500ms
                
                current_time = time.time()
                running_branches = [
                    b for b in self._branches.values()
                    if b.status == BranchStatus.RUNNING
                ]
                
                # Check for slow branches
                if self.strategy.cancel_slow_branches_after_ms > 0:
                    for branch in running_branches:
                        elapsed_ms = (current_time - branch.start_time) * 1000
                        if elapsed_ms > self.strategy.cancel_slow_branches_after_ms:
                            # Check if we have better results
                            if self._has_better_results(branch):
                                await self.cancel_branch(
                                    branch.branch_id,
                                    f"Too slow ({elapsed_ms:.0f}ms)"
                                )
                
                # Keep only top N branches if configured
                if self.strategy.keep_top_n_branches > 0:
                    await self._keep_top_n_branches()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in branch monitor: {e}")
    
    async def _wait_for_branches(self, branch_ids: List[str]) -> None:
        """Wait for specified branches to complete."""
        tasks = []
        for branch_id in branch_ids:
            if branch_id in self._branches:
                branch = self._branches[branch_id]
                if branch.task:
                    tasks.append(branch.task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _cancel_remaining_branches(self) -> None:
        """Cancel all remaining running branches."""
        for branch in self._branches.values():
            if branch.status == BranchStatus.RUNNING:
                await self.cancel_branch(branch.branch_id, "Execution completed")
    
    async def _check_cancellation_on_result(self, completed_branch: ExecutionBranch) -> None:
        """Check if other branches should be cancelled based on a completed result."""
        if completed_branch.score >= self.strategy.score_threshold:
            # Cancel other branches if this result is good enough
            for branch in self._branches.values():
                if (
                    branch.branch_id != completed_branch.branch_id
                    and branch.status == BranchStatus.RUNNING
                    and branch.priority <= completed_branch.priority
                ):
                    await self.cancel_branch(
                        branch.branch_id,
                        f"Better result found (score: {completed_branch.score:.2f})"
                    )
                    self._metrics["score_cancellations"] += 1
    
    async def _keep_top_n_branches(self) -> None:
        """Keep only the top N branches by priority and score."""
        running_branches = [
            b for b in self._branches.values()
            if b.status == BranchStatus.RUNNING
        ]
        
        if len(running_branches) <= self.strategy.keep_top_n_branches:
            return
        
        # Sort by priority and estimated score
        running_branches.sort(
            key=lambda b: (b.priority, b.score),
            reverse=True
        )
        
        # Cancel branches beyond top N
        for branch in running_branches[self.strategy.keep_top_n_branches:]:
            await self.cancel_branch(
                branch.branch_id,
                "Not in top N branches"
            )
    
    def _has_better_results(self, branch: ExecutionBranch) -> bool:
        """Check if we have better completed results than this branch."""
        for completed in self._completed_branches:
            if (
                completed.score >= self.strategy.score_threshold
                and completed.priority >= branch.priority
            ):
                return True
        return False
    
    def _default_scorer(self, result: ToolResult) -> float:
        """Default scoring function for results."""
        if not result or not result.success:
            return 0.0
        
        # Base score on success and execution time
        base_score = 0.5
        
        # Bonus for fast execution
        if result.execution_time_ms < 100:
            base_score += 0.3
        elif result.execution_time_ms < 500:
            base_score += 0.2
        elif result.execution_time_ms < 1000:
            base_score += 0.1
        
        # Bonus for having data
        if result.data:
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def _generate_branch_id(self, tool_name: str) -> str:
        """Generate a unique branch ID."""
        import uuid
        return f"{tool_name}_{uuid.uuid4().hex[:8]}"
    
    def _update_average_execution_time(self, exec_time_ms: float) -> None:
        """Update rolling average execution time."""
        current_avg = self._metrics["average_execution_time_ms"]
        completed = self._metrics["completed_branches"]
        
        if completed == 1:
            self._metrics["average_execution_time_ms"] = exec_time_ms
        else:
            # Rolling average
            self._metrics["average_execution_time_ms"] = (
                current_avg * (completed - 1) + exec_time_ms
            ) / completed
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get branch manager metrics."""
        return {
            **self._metrics,
            "active_branches": len(
                [b for b in self._branches.values() if b.status == BranchStatus.RUNNING]
            ),
            "cancellation_rate": (
                self._metrics["cancelled_branches"] / max(1, self._metrics["total_branches"])
            ),
        }
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        # Cancel all running branches
        for branch in self._branches.values():
            if branch.status == BranchStatus.RUNNING and branch.task:
                branch.task.cancel()
        
        # Wait for cancellations
        tasks = [
            b.task for b in self._branches.values()
            if b.task and not b.task.done()
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self._branches.clear()
        self._completed_branches.clear()
        self._cancelled_branches.clear()