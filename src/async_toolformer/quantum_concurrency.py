"""
Quantum Concurrency Management Module for AsyncOrchestrator.

This module provides advanced concurrency control and coordination:
- Quantum-inspired task synchronization
- Advanced semaphore and lock management  
- Deadlock detection and prevention
- Resource contention resolution
- Coordinated parallel execution patterns
"""

import asyncio
import time
import math
import weakref
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import uuid

from .quantum_planner import QuantumTask, TaskState

logger = logging.getLogger(__name__)


class SynchronizationType(Enum):
    """Types of quantum synchronization."""
    ENTANGLEMENT = "entanglement"      # Tasks synchronized through quantum entanglement
    SUPERPOSITION = "superposition"    # Multiple states until measured
    COHERENCE = "coherence"           # Maintaining phase relationships
    INTERFERENCE = "interference"      # Constructive/destructive task interactions


class ResourceLockState(Enum):
    """States of quantum resource locks."""
    AVAILABLE = "available"
    SUPERPOSITION = "superposition"    # Lock exists in multiple states
    ENTANGLED = "entangled"           # Lock shared across multiple tasks
    EXCLUSIVE = "exclusive"           # Traditional exclusive lock
    COLLAPSED = "collapsed"           # Superposition has collapsed


@dataclass
class QuantumLock:
    """Quantum-inspired lock with superposition and entanglement."""
    lock_id: str
    resource_name: str
    state: ResourceLockState = ResourceLockState.AVAILABLE
    holders: Set[str] = field(default_factory=set)
    waiters: deque = field(default_factory=deque)
    entangled_locks: Set[str] = field(default_factory=set)
    coherence_phase: float = 0.0
    max_holders: int = 1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    def can_acquire(self, task_id: str) -> bool:
        """Check if task can acquire this lock."""
        if self.state == ResourceLockState.AVAILABLE:
            return True
        elif self.state == ResourceLockState.SUPERPOSITION:
            return len(self.holders) < self.max_holders
        elif self.state == ResourceLockState.ENTANGLED:
            return task_id in self.holders or len(self.holders) < self.max_holders
        else:
            return task_id in self.holders
    
    def acquire(self, task_id: str) -> bool:
        """Attempt to acquire the lock."""
        if not self.can_acquire(task_id):
            return False
        
        self.holders.add(task_id)
        self.last_accessed = time.time()
        
        # Update state based on holders
        if len(self.holders) == 1:
            if self.max_holders == 1:
                self.state = ResourceLockState.EXCLUSIVE
            else:
                self.state = ResourceLockState.SUPERPOSITION
        elif len(self.holders) > 1:
            self.state = ResourceLockState.ENTANGLED
        
        return True
    
    def release(self, task_id: str) -> bool:
        """Release the lock."""
        if task_id not in self.holders:
            return False
        
        self.holders.remove(task_id)
        self.last_accessed = time.time()
        
        # Update state
        if not self.holders:
            self.state = ResourceLockState.AVAILABLE
        elif len(self.holders) == 1 and self.max_holders == 1:
            self.state = ResourceLockState.EXCLUSIVE
        
        # Wake up waiters
        if self.waiters and self.can_acquire_next():
            return True
        
        return True
    
    def can_acquire_next(self) -> bool:
        """Check if next waiter can acquire."""
        return len(self.holders) < self.max_holders


@dataclass
class ConcurrencyMetrics:
    """Concurrency-related metrics."""
    timestamp: float = field(default_factory=time.time)
    
    # Lock metrics
    total_locks: int = 0
    active_locks: int = 0
    waiting_tasks: int = 0
    lock_contention_rate: float = 0.0
    average_wait_time_ms: float = 0.0
    
    # Deadlock metrics
    deadlock_detections: int = 0
    deadlock_resolutions: int = 0
    circular_wait_chains: int = 0
    
    # Quantum synchronization metrics
    entangled_task_pairs: int = 0
    superposition_collapses: int = 0
    coherence_violations: int = 0
    interference_patterns: int = 0
    
    # Performance metrics
    parallel_efficiency: float = 0.0
    coordination_overhead_ms: float = 0.0
    synchronization_success_rate: float = 0.0


class QuantumConcurrencyManager:
    """
    Quantum-inspired concurrency manager for task orchestration.
    
    Features:
    - Quantum locks with superposition and entanglement
    - Deadlock detection and prevention
    - Advanced synchronization primitives
    - Resource contention resolution
    - Performance-aware coordination
    """
    
    def __init__(
        self,
        enable_quantum_synchronization: bool = True,
        deadlock_detection_enabled: bool = True,
        max_wait_time_seconds: float = 30.0,
        cleanup_interval_seconds: float = 60.0,
        coherence_threshold: float = 0.1,
    ):
        """
        Initialize the quantum concurrency manager.
        
        Args:
            enable_quantum_synchronization: Enable quantum sync features
            deadlock_detection_enabled: Enable deadlock detection
            max_wait_time_seconds: Maximum time to wait for locks
            cleanup_interval_seconds: Interval for cleanup operations
            coherence_threshold: Minimum coherence to maintain
        """
        self.enable_quantum_sync = enable_quantum_synchronization
        self.deadlock_detection_enabled = deadlock_detection_enabled
        self.max_wait_time = max_wait_time_seconds
        self.cleanup_interval = cleanup_interval_seconds
        self.coherence_threshold = coherence_threshold
        
        # Lock management
        self._locks: Dict[str, QuantumLock] = {}
        self._task_locks: Dict[str, Set[str]] = defaultdict(set)  # task_id -> lock_ids
        self._lock_wait_times: Dict[str, float] = {}
        
        # Synchronization primitives
        self._barriers: Dict[str, asyncio.Barrier] = {}
        self._events: Dict[str, asyncio.Event] = {}
        self._conditions: Dict[str, asyncio.Condition] = {}
        
        # Deadlock detection
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._deadlock_history: List[Dict[str, Any]] = []
        
        # Quantum entanglement tracking
        self._entangled_tasks: Dict[str, Set[str]] = defaultdict(set)
        self._superposition_states: Dict[str, List[str]] = defaultdict(list)
        
        # Metrics and monitoring
        self._metrics = ConcurrencyMetrics()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        logger.info("QuantumConcurrencyManager initialized")
    
    async def start(self):
        """Start the concurrency manager."""
        if self._is_running:
            return
        
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("QuantumConcurrencyManager started")
    
    async def stop(self):
        """Stop the concurrency manager."""
        self._is_running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Release all locks
        await self._release_all_locks()
        
        logger.info("QuantumConcurrencyManager stopped")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired locks and states."""
        try:
            while self._is_running:
                await self._cleanup_expired_locks()
                await self._cleanup_orphaned_states()
                
                if self.deadlock_detection_enabled:
                    await self._detect_deadlocks()
                
                await self._update_metrics()
                await asyncio.sleep(self.cleanup_interval)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")
    
    async def acquire_quantum_lock(
        self,
        resource_name: str,
        task_id: str,
        max_holders: int = 1,
        timeout_seconds: Optional[float] = None,
        synchronization_type: SynchronizationType = SynchronizationType.ENTANGLEMENT,
    ) -> Optional[str]:
        """
        Acquire a quantum lock for a resource.
        
        Args:
            resource_name: Name of the resource to lock
            task_id: ID of the task requesting the lock
            max_holders: Maximum number of simultaneous holders
            timeout_seconds: Timeout for acquisition
            synchronization_type: Type of quantum synchronization
            
        Returns:
            Lock ID if successful, None if failed
        """
        timeout = timeout_seconds or self.max_wait_time
        start_time = time.time()
        
        # Find or create lock
        lock_id = f"{resource_name}_{uuid.uuid4().hex[:8]}"
        
        if resource_name in [lock.resource_name for lock in self._locks.values()]:
            # Find existing lock for this resource
            existing_lock = next(
                (lock for lock in self._locks.values() if lock.resource_name == resource_name),
                None
            )
            if existing_lock:
                lock_id = existing_lock.lock_id
        else:
            # Create new lock
            quantum_lock = QuantumLock(
                lock_id=lock_id,
                resource_name=resource_name,
                max_holders=max_holders,
            )
            self._locks[lock_id] = quantum_lock
        
        lock = self._locks[lock_id]
        
        # Try to acquire immediately
        if lock.acquire(task_id):
            self._task_locks[task_id].add(lock_id)
            await self._apply_quantum_synchronization(lock, task_id, synchronization_type)
            
            logger.debug(f"Task {task_id} acquired lock {lock_id} for {resource_name}")
            return lock_id
        
        # Add to wait queue
        wait_start = time.time()
        lock.waiters.append(task_id)
        self._lock_wait_times[task_id] = wait_start
        
        # Wait for lock with timeout
        try:
            while time.time() - start_time < timeout:
                if lock.acquire(task_id):
                    self._task_locks[task_id].add(lock_id)
                    await self._apply_quantum_synchronization(lock, task_id, synchronization_type)
                    
                    # Remove from wait queue
                    if task_id in lock.waiters:
                        lock.waiters.remove(task_id)
                    
                    wait_time = time.time() - wait_start
                    self._metrics.average_wait_time_ms = (
                        self._metrics.average_wait_time_ms * 0.9 + wait_time * 1000 * 0.1
                    )
                    
                    logger.debug(f"Task {task_id} acquired lock {lock_id} after {wait_time:.3f}s")
                    return lock_id
                
                await asyncio.sleep(0.01)  # Small delay before retry
            
            # Timeout - remove from wait queue
            if task_id in lock.waiters:
                lock.waiters.remove(task_id)
            
            logger.warning(f"Task {task_id} timed out waiting for lock {lock_id}")
            return None
            
        except Exception as e:
            # Error - cleanup wait state
            if task_id in lock.waiters:
                lock.waiters.remove(task_id)
            logger.error(f"Error acquiring lock {lock_id}: {e}")
            return None
    
    async def release_quantum_lock(self, lock_id: str, task_id: str) -> bool:
        """
        Release a quantum lock.
        
        Args:
            lock_id: ID of the lock to release
            task_id: ID of the task releasing the lock
            
        Returns:
            True if successfully released
        """
        if lock_id not in self._locks:
            logger.warning(f"Attempted to release non-existent lock: {lock_id}")
            return False
        
        lock = self._locks[lock_id]
        
        if lock.release(task_id):
            self._task_locks[task_id].discard(lock_id)
            await self._handle_quantum_decoherence(lock, task_id)
            
            # Wake up next waiter
            await self._wake_next_waiter(lock)
            
            logger.debug(f"Task {task_id} released lock {lock_id}")
            return True
        
        logger.warning(f"Task {task_id} attempted to release lock {lock_id} it doesn't hold")
        return False
    
    async def _apply_quantum_synchronization(
        self,
        lock: QuantumLock,
        task_id: str,
        sync_type: SynchronizationType,
    ):
        """Apply quantum synchronization effects."""
        if not self.enable_quantum_sync:
            return
        
        if sync_type == SynchronizationType.ENTANGLEMENT:
            # Create entanglement with other holders
            for holder_id in lock.holders:
                if holder_id != task_id:
                    self._entangled_tasks[task_id].add(holder_id)
                    self._entangled_tasks[holder_id].add(task_id)
                    self._metrics.entangled_task_pairs += 1
        
        elif sync_type == SynchronizationType.SUPERPOSITION:
            # Add task to superposition state
            self._superposition_states[lock.resource_name].append(task_id)
            lock.state = ResourceLockState.SUPERPOSITION
        
        elif sync_type == SynchronizationType.COHERENCE:
            # Maintain phase coherence
            lock.coherence_phase = (lock.coherence_phase + math.pi / 4) % (2 * math.pi)
            
            # Check coherence violations
            if abs(math.sin(lock.coherence_phase)) < self.coherence_threshold:
                self._metrics.coherence_violations += 1
                logger.warning(f"Coherence violation detected for lock {lock.lock_id}")
    
    async def _handle_quantum_decoherence(self, lock: QuantumLock, task_id: str):
        """Handle quantum decoherence when lock is released."""
        if not self.enable_quantum_sync:
            return
        
        # Remove entanglements
        if task_id in self._entangled_tasks:
            for entangled_id in self._entangled_tasks[task_id]:
                self._entangled_tasks[entangled_id].discard(task_id)
            self._entangled_tasks[task_id].clear()
        
        # Collapse superposition if task was in one
        for resource, tasks in self._superposition_states.items():
            if task_id in tasks:
                tasks.remove(task_id)
                if len(tasks) <= 1:
                    # Superposition collapsed
                    self._metrics.superposition_collapses += 1
                    if lock.resource_name == resource:
                        lock.state = ResourceLockState.COLLAPSED
    
    async def _wake_next_waiter(self, lock: QuantumLock):
        """Wake up the next task waiting for this lock."""
        while lock.waiters and lock.can_acquire_next():
            next_task = lock.waiters.popleft()
            
            # Check if task is still waiting (not timed out)
            if next_task in self._lock_wait_times:
                wait_time = time.time() - self._lock_wait_times[next_task]
                del self._lock_wait_times[next_task]
                
                if wait_time < self.max_wait_time:
                    # Try to acquire for this task
                    if lock.acquire(next_task):
                        self._task_locks[next_task].add(lock.lock_id)
                        logger.debug(f"Woke up task {next_task} for lock {lock.lock_id}")
                        break
    
    async def create_quantum_barrier(
        self,
        barrier_name: str,
        parties: int,
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """
        Create a quantum barrier for synchronization.
        
        Args:
            barrier_name: Name of the barrier
            parties: Number of tasks that must reach the barrier
            timeout_seconds: Timeout for barrier wait
            
        Returns:
            True if barrier created successfully
        """
        if barrier_name in self._barriers:
            logger.warning(f"Barrier {barrier_name} already exists")
            return False
        
        try:
            barrier = asyncio.Barrier(parties)
            self._barriers[barrier_name] = barrier
            logger.info(f"Created quantum barrier {barrier_name} for {parties} parties")
            return True
        except Exception as e:
            logger.error(f"Failed to create barrier {barrier_name}: {e}")
            return False
    
    async def wait_at_quantum_barrier(
        self,
        barrier_name: str,
        task_id: str,
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """
        Wait at a quantum barrier.
        
        Args:
            barrier_name: Name of the barrier
            task_id: ID of the waiting task
            timeout_seconds: Timeout for wait
            
        Returns:
            True if barrier was reached by all parties
        """
        if barrier_name not in self._barriers:
            logger.error(f"Barrier {barrier_name} does not exist")
            return False
        
        barrier = self._barriers[barrier_name]
        timeout = timeout_seconds or self.max_wait_time
        
        try:
            await asyncio.wait_for(barrier.wait(), timeout=timeout)
            logger.debug(f"Task {task_id} passed barrier {barrier_name}")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Task {task_id} timed out at barrier {barrier_name}")
            return False
        except Exception as e:
            logger.error(f"Error waiting at barrier {barrier_name}: {e}")
            return False
    
    async def signal_quantum_event(self, event_name: str, task_id: str) -> bool:
        """
        Signal a quantum event.
        
        Args:
            event_name: Name of the event
            task_id: ID of the signaling task
            
        Returns:
            True if event was signaled
        """
        if event_name not in self._events:
            self._events[event_name] = asyncio.Event()
        
        event = self._events[event_name]
        event.set()
        
        logger.debug(f"Task {task_id} signaled event {event_name}")
        return True
    
    async def wait_for_quantum_event(
        self,
        event_name: str,
        task_id: str,
        timeout_seconds: Optional[float] = None,
    ) -> bool:
        """
        Wait for a quantum event.
        
        Args:
            event_name: Name of the event
            task_id: ID of the waiting task
            timeout_seconds: Timeout for wait
            
        Returns:
            True if event was received
        """
        if event_name not in self._events:
            self._events[event_name] = asyncio.Event()
        
        event = self._events[event_name]
        timeout = timeout_seconds or self.max_wait_time
        
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            logger.debug(f"Task {task_id} received event {event_name}")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Task {task_id} timed out waiting for event {event_name}")
            return False
        except Exception as e:
            logger.error(f"Error waiting for event {event_name}: {e}")
            return False
    
    async def coordinate_parallel_execution(
        self,
        tasks: List[QuantumTask],
        coordination_strategy: str = "quantum_entanglement",
    ) -> AsyncIterator[Tuple[str, Any]]:
        """
        Coordinate parallel execution of quantum tasks.
        
        Args:
            tasks: List of tasks to coordinate
            coordination_strategy: Strategy for coordination
            
        Yields:
            Task results as they complete
        """
        if not tasks:
            return
        
        logger.info(f"Coordinating {len(tasks)} tasks with {coordination_strategy}")
        
        # Set up coordination based on strategy
        if coordination_strategy == "quantum_entanglement":
            await self._setup_entanglement_coordination(tasks)
        elif coordination_strategy == "superposition":
            await self._setup_superposition_coordination(tasks)
        elif coordination_strategy == "barrier_sync":
            await self._setup_barrier_coordination(tasks)
        
        # Execute tasks with coordination
        task_futures = {}
        
        for task in tasks:
            future = asyncio.create_task(self._execute_coordinated_task(task))
            task_futures[task.id] = future
        
        # Yield results as they complete
        for completed_future in asyncio.as_completed(task_futures.values()):
            try:
                result = await completed_future
                task_id = next(
                    tid for tid, future in task_futures.items() 
                    if future == completed_future
                )
                yield task_id, result
            except Exception as e:
                logger.error(f"Task execution failed: {e}")
                yield "error", e
    
    async def _setup_entanglement_coordination(self, tasks: List[QuantumTask]):
        """Set up quantum entanglement coordination."""
        # Create entanglement links between related tasks
        for i, task1 in enumerate(tasks):
            for j, task2 in enumerate(tasks[i+1:], i+1):
                # Create entanglement based on shared dependencies or resources
                shared_deps = task1.dependencies & task2.dependencies
                shared_resources = set(task1.resource_requirements.keys()) & set(task2.resource_requirements.keys())
                
                if shared_deps or shared_resources:
                    self._entangled_tasks[task1.id].add(task2.id)
                    self._entangled_tasks[task2.id].add(task1.id)
                    
                    # Link their entangled_with sets
                    task1.entangled_with.add(task2.id)
                    task2.entangled_with.add(task1.id)
    
    async def _setup_superposition_coordination(self, tasks: List[QuantumTask]):
        """Set up superposition coordination."""
        # Group tasks that can exist in superposition
        resource_groups = defaultdict(list)
        
        for task in tasks:
            for resource in task.resource_requirements.keys():
                resource_groups[resource].append(task)
        
        # Create superposition states for resource groups
        for resource, resource_tasks in resource_groups.items():
            if len(resource_tasks) > 1:
                task_ids = [task.id for task in resource_tasks]
                self._superposition_states[resource] = task_ids
                
                # Set task states to superposition
                for task in resource_tasks:
                    task.state = TaskState.SUPERPOSITION
    
    async def _setup_barrier_coordination(self, tasks: List[QuantumTask]):
        """Set up barrier-based coordination."""
        # Create barriers for tasks with dependencies
        phase_groups = defaultdict(list)
        
        # Group tasks by dependency depth
        for task in tasks:
            depth = len(task.dependencies)
            phase_groups[depth].append(task)
        
        # Create barriers between phases
        for phase, phase_tasks in phase_groups.items():
            if len(phase_tasks) > 1:
                barrier_name = f"phase_{phase}_barrier"
                await self.create_quantum_barrier(barrier_name, len(phase_tasks))
    
    async def _execute_coordinated_task(self, task: QuantumTask) -> Any:
        """Execute a task with coordination."""
        start_time = time.time()
        
        try:
            # Apply pre-execution coordination
            await self._pre_execution_coordination(task)
            
            # Execute task function
            if task.function:
                if asyncio.iscoroutinefunction(task.function):
                    result = await task.function(**task.args)
                else:
                    result = task.function(**task.args)
            else:
                # Simulate execution
                await asyncio.sleep(task.estimated_duration_ms / 1000.0)
                result = f"Coordinated result for {task.name}"
            
            # Apply post-execution coordination
            await self._post_execution_coordination(task)
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                "task_id": task.id,
                "result": result,
                "execution_time_ms": execution_time,
                "success": True,
            }
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Coordinated task {task.id} failed: {e}")
            
            return {
                "task_id": task.id,
                "error": str(e),
                "execution_time_ms": execution_time,
                "success": False,
            }
    
    async def _pre_execution_coordination(self, task: QuantumTask):
        """Apply pre-execution coordination."""
        # Wait for entangled tasks if necessary
        if task.id in self._entangled_tasks:
            for entangled_id in self._entangled_tasks[task.id]:
                # Check if entangled task needs to complete first
                if entangled_id in task.dependencies:
                    await self._wait_for_task_completion(entangled_id)
        
        # Handle superposition collapse
        if task.state == TaskState.SUPERPOSITION:
            await self._collapse_superposition(task)
    
    async def _post_execution_coordination(self, task: QuantumTask):
        """Apply post-execution coordination."""
        # Signal completion to entangled tasks
        completion_event = f"task_completed_{task.id}"
        await self.signal_quantum_event(completion_event, task.id)
        
        # Update task state
        task.state = TaskState.COMPLETED
    
    async def _wait_for_task_completion(self, task_id: str):
        """Wait for a specific task to complete."""
        completion_event = f"task_completed_{task_id}"
        await self.wait_for_quantum_event(completion_event, task_id, timeout_seconds=30.0)
    
    async def _collapse_superposition(self, task: QuantumTask):
        """Collapse quantum superposition for a task."""
        # Find superposition state containing this task
        for resource, task_ids in self._superposition_states.items():
            if task.id in task_ids:
                # Collapse superposition - this task becomes definite
                task.state = TaskState.COLLAPSED
                self._metrics.superposition_collapses += 1
                
                # Remove from superposition
                task_ids.remove(task.id)
                
                logger.debug(f"Collapsed superposition for task {task.id} on resource {resource}")
                break
    
    async def _detect_deadlocks(self):
        """Detect potential deadlocks."""
        if not self.deadlock_detection_enabled:
            return
        
        # Build dependency graph from current lock states
        dependency_graph = defaultdict(set)
        
        for task_id, lock_ids in self._task_locks.items():
            for lock_id in lock_ids:
                if lock_id in self._locks:
                    lock = self._locks[lock_id]
                    # Task depends on all waiters of locks it holds
                    for waiter in lock.waiters:
                        if waiter != task_id:
                            dependency_graph[task_id].add(waiter)
        
        # Detect cycles using DFS
        visited = set()
        rec_stack = set()
        deadlock_chains = []
        
        def dfs(node, path):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                deadlock_chains.append(cycle)
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph[node]:
                if dfs(neighbor, path + [node]):
                    return True
            
            rec_stack.remove(node)
            return False
        
        # Check all nodes for cycles
        for task_id in dependency_graph:
            if task_id not in visited:
                dfs(task_id, [])
        
        # Handle detected deadlocks
        if deadlock_chains:
            self._metrics.deadlock_detections += len(deadlock_chains)
            self._metrics.circular_wait_chains += len(deadlock_chains)
            
            logger.warning(f"Detected {len(deadlock_chains)} potential deadlocks")
            
            for chain in deadlock_chains:
                await self._resolve_deadlock(chain)
    
    async def _resolve_deadlock(self, deadlock_chain: List[str]):
        """Resolve a detected deadlock."""
        logger.warning(f"Resolving deadlock chain: {' -> '.join(deadlock_chain)}")
        
        # Simple resolution: release locks for the task with the most locks
        task_lock_counts = {
            task_id: len(self._task_locks[task_id])
            for task_id in deadlock_chain
        }
        
        victim_task = max(task_lock_counts, key=task_lock_counts.get)
        
        # Release all locks held by victim task
        victim_locks = list(self._task_locks[victim_task])
        for lock_id in victim_locks:
            await self.release_quantum_lock(lock_id, victim_task)
        
        self._metrics.deadlock_resolutions += 1
        
        # Record deadlock event
        self._deadlock_history.append({
            "timestamp": time.time(),
            "chain": deadlock_chain,
            "victim": victim_task,
            "resolution": "lock_release",
        })
        
        logger.info(f"Resolved deadlock by releasing locks for task {victim_task}")
    
    async def _cleanup_expired_locks(self):
        """Clean up expired and unused locks."""
        current_time = time.time()
        expired_locks = []
        
        for lock_id, lock in self._locks.items():
            # Remove locks that haven't been accessed recently and have no holders
            if (not lock.holders and 
                not lock.waiters and 
                current_time - lock.last_accessed > 300):  # 5 minutes
                expired_locks.append(lock_id)
        
        for lock_id in expired_locks:
            del self._locks[lock_id]
            logger.debug(f"Cleaned up expired lock: {lock_id}")
    
    async def _cleanup_orphaned_states(self):
        """Clean up orphaned quantum states."""
        # Clean up empty superposition states
        empty_superpositions = [
            resource for resource, tasks in self._superposition_states.items()
            if not tasks
        ]
        
        for resource in empty_superpositions:
            del self._superposition_states[resource]
        
        # Clean up broken entanglements
        for task_id in list(self._entangled_tasks.keys()):
            entangled_set = self._entangled_tasks[task_id]
            # Remove references to non-existent tasks
            valid_entanglements = {
                eid for eid in entangled_set
                if eid in self._entangled_tasks
            }
            
            if valid_entanglements != entangled_set:
                self._entangled_tasks[task_id] = valid_entanglements
            
            # Remove empty entanglement sets
            if not valid_entanglements:
                del self._entangled_tasks[task_id]
    
    async def _release_all_locks(self):
        """Release all locks (used during shutdown)."""
        for task_id in list(self._task_locks.keys()):
            lock_ids = list(self._task_locks[task_id])
            for lock_id in lock_ids:
                await self.release_quantum_lock(lock_id, task_id)
    
    async def _update_metrics(self):
        """Update concurrency metrics."""
        self._metrics.timestamp = time.time()
        self._metrics.total_locks = len(self._locks)
        self._metrics.active_locks = len([
            lock for lock in self._locks.values() if lock.holders
        ])
        self._metrics.waiting_tasks = sum(
            len(lock.waiters) for lock in self._locks.values()
        )
        
        # Calculate contention rate
        total_attempts = self._metrics.active_locks + self._metrics.waiting_tasks
        if total_attempts > 0:
            self._metrics.lock_contention_rate = self._metrics.waiting_tasks / total_attempts
    
    def get_concurrency_metrics(self) -> ConcurrencyMetrics:
        """Get current concurrency metrics."""
        return self._metrics
    
    def get_lock_status(self) -> Dict[str, Any]:
        """Get current lock status information."""
        return {
            "total_locks": len(self._locks),
            "locks": {
                lock_id: {
                    "resource": lock.resource_name,
                    "state": lock.state.value,
                    "holders": list(lock.holders),
                    "waiters": list(lock.waiters),
                    "max_holders": lock.max_holders,
                    "entangled_locks": list(lock.entangled_locks),
                    "created_at": lock.created_at,
                    "last_accessed": lock.last_accessed,
                }
                for lock_id, lock in self._locks.items()
            },
            "task_locks": {
                task_id: list(lock_ids)
                for task_id, lock_ids in self._task_locks.items()
            },
            "entangled_tasks": {
                task_id: list(entangled_ids)
                for task_id, entangled_ids in self._entangled_tasks.items()
            },
            "superposition_states": dict(self._superposition_states),
        }
    
    def get_deadlock_history(self) -> List[Dict[str, Any]]:
        """Get deadlock detection and resolution history."""
        return list(self._deadlock_history)


# Convenience functions
def create_concurrency_manager(
    enable_quantum_sync: bool = True,
    enable_deadlock_detection: bool = True,
    **kwargs
) -> QuantumConcurrencyManager:
    """Create a QuantumConcurrencyManager with sensible defaults."""
    return QuantumConcurrencyManager(
        enable_quantum_synchronization=enable_quantum_sync,
        deadlock_detection_enabled=enable_deadlock_detection,
        **kwargs
    )


async def coordinate_tasks(
    concurrency_manager: QuantumConcurrencyManager,
    tasks: List[QuantumTask],
    strategy: str = "quantum_entanglement",
) -> List[Any]:
    """Coordinate execution of multiple tasks."""
    results = []
    
    async for task_id, result in concurrency_manager.coordinate_parallel_execution(
        tasks, strategy
    ):
        results.append((task_id, result))
    
    return results