"""
Quantum-Inspired Task Planner for AsyncOrchestrator.

This module implements quantum-inspired algorithms for optimal task scheduling,
parallel execution planning, and dependency resolution based on quantum
superposition and entanglement principles.
"""

import asyncio
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class TaskState(Enum):
    """Quantum-inspired task states."""
    SUPERPOSITION = "superposition"  # Task exists in multiple potential states
    ENTANGLED = "entangled"         # Task depends on other tasks
    COLLAPSED = "collapsed"         # Task state is determined
    EXECUTING = "executing"         # Task is currently running
    COMPLETED = "completed"         # Task finished successfully
    FAILED = "failed"              # Task failed


@dataclass
class QuantumTask:
    """A quantum-inspired task representation."""
    id: str
    name: str
    function: Optional[callable] = None
    args: Dict[str, Any] = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)
    entangled_with: Set[str] = field(default_factory=set)
    priority: float = 1.0
    probability_amplitude: complex = complex(1.0, 0.0)
    state: TaskState = TaskState.SUPERPOSITION
    estimated_duration_ms: float = 1000.0
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    success_probability: float = 0.9
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Normalize probability amplitude."""
        self._normalize_amplitude()
    
    def _normalize_amplitude(self):
        """Normalize the probability amplitude."""
        magnitude = abs(self.probability_amplitude)
        if magnitude > 0:
            self.probability_amplitude = self.probability_amplitude / magnitude
    
    @property
    def probability(self) -> float:
        """Get the probability of this task being selected."""
        return abs(self.probability_amplitude) ** 2
    
    def collapse_to_state(self, state: TaskState) -> None:
        """Collapse the quantum superposition to a definite state."""
        self.state = state
        if state in [TaskState.EXECUTING, TaskState.COMPLETED, TaskState.FAILED]:
            self.probability_amplitude = complex(1.0, 0.0)


@dataclass
class ExecutionPlan:
    """Quantum-optimized execution plan."""
    phases: List[List[QuantumTask]] = field(default_factory=list)
    total_estimated_time_ms: float = 0.0
    parallelism_factor: float = 1.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    optimization_score: float = 0.0
    quantum_coherence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumInspiredPlanner:
    """
    Quantum-inspired task planner using superposition and entanglement principles.
    
    This planner uses quantum computing concepts to optimize task scheduling:
    - Superposition: Tasks exist in multiple potential execution states
    - Entanglement: Task dependencies create quantum correlations
    - Interference: Optimization through wave interference patterns
    - Measurement: Collapsing superposition to actual execution plan
    """
    
    def __init__(
        self,
        max_parallel_tasks: int = 10,
        resource_limits: Optional[Dict[str, float]] = None,
        optimization_iterations: int = 100,
        quantum_coherence_decay: float = 0.95,
        enable_entanglement: bool = True,
    ):
        """
        Initialize the quantum-inspired planner.
        
        Args:
            max_parallel_tasks: Maximum tasks to run in parallel
            resource_limits: Resource constraints (CPU, memory, etc.)
            optimization_iterations: Number of quantum optimization cycles
            quantum_coherence_decay: Rate of coherence decay over time
            enable_entanglement: Whether to use task entanglement
        """
        self.max_parallel_tasks = max_parallel_tasks
        self.resource_limits = resource_limits or {"cpu": 100.0, "memory": 8192.0}
        self.optimization_iterations = optimization_iterations
        self.quantum_coherence_decay = quantum_coherence_decay
        self.enable_entanglement = enable_entanglement
        
        # Quantum state tracking
        self._quantum_register: Dict[str, QuantumTask] = {}
        self._entanglement_matrix: Dict[Tuple[str, str], float] = {}
        self._coherence_time: float = time.time()
        
        logger.info(
            f"QuantumInspiredPlanner initialized with {max_parallel_tasks} parallel tasks"
        )
    
    def register_task(
        self,
        task_id: str,
        name: str,
        function: Optional[callable] = None,
        args: Optional[Dict[str, Any]] = None,
        dependencies: Optional[Set[str]] = None,
        priority: float = 1.0,
        estimated_duration_ms: float = 1000.0,
        resource_requirements: Optional[Dict[str, float]] = None,
        success_probability: float = 0.9,
        **metadata,
    ) -> QuantumTask:
        """
        Register a task in the quantum register.
        
        Args:
            task_id: Unique task identifier
            name: Human-readable task name
            function: Callable to execute
            args: Function arguments
            dependencies: Set of task IDs this task depends on
            priority: Task priority (higher = more important)
            estimated_duration_ms: Estimated execution time
            resource_requirements: Required resources
            success_probability: Probability of successful completion
            **metadata: Additional task metadata
            
        Returns:
            QuantumTask instance
        """
        task = QuantumTask(
            id=task_id,
            name=name,
            function=function,
            args=args or {},
            dependencies=dependencies or set(),
            priority=priority,
            estimated_duration_ms=estimated_duration_ms,
            resource_requirements=resource_requirements or {},
            success_probability=success_probability,
            metadata=metadata,
        )
        
        # Set initial probability amplitude based on priority
        amplitude = math.sqrt(priority / 10.0)  # Normalize priority
        phase = random.uniform(0, 2 * math.pi)  # Random quantum phase
        task.probability_amplitude = complex(
            amplitude * math.cos(phase),
            amplitude * math.sin(phase)
        )
        
        self._quantum_register[task_id] = task
        
        # Create entanglements with dependencies
        if self.enable_entanglement:
            self._create_entanglements(task)
        
        logger.debug(f"Registered quantum task: {task_id} with amplitude {task.probability_amplitude}")
        return task
    
    def _create_entanglements(self, task: QuantumTask) -> None:
        """Create quantum entanglements between dependent tasks."""
        for dep_id in task.dependencies:
            if dep_id in self._quantum_register:
                # Create entanglement with dependency
                entanglement_strength = 0.8  # Strong correlation
                self._entanglement_matrix[(task.id, dep_id)] = entanglement_strength
                self._entanglement_matrix[(dep_id, task.id)] = entanglement_strength
                
                # Add to entangled sets
                task.entangled_with.add(dep_id)
                self._quantum_register[dep_id].entangled_with.add(task.id)
    
    def create_execution_plan(
        self,
        task_ids: Optional[List[str]] = None,
        optimize: bool = True,
    ) -> ExecutionPlan:
        """
        Create an optimized execution plan using quantum algorithms.
        
        Args:
            task_ids: Specific tasks to plan (None = all registered tasks)
            optimize: Whether to run quantum optimization
            
        Returns:
            ExecutionPlan with optimal task scheduling
        """
        # Select tasks to plan
        if task_ids is None:
            tasks = list(self._quantum_register.values())
        else:
            tasks = [self._quantum_register[tid] for tid in task_ids if tid in self._quantum_register]
        
        if not tasks:
            return ExecutionPlan()
        
        logger.info(f"Creating execution plan for {len(tasks)} tasks")
        
        # Apply quantum optimization if enabled
        if optimize:
            tasks = self._quantum_optimize_tasks(tasks)
        
        # Create execution phases using topological sort with quantum insights
        phases = self._create_execution_phases(tasks)
        
        # Calculate plan metrics
        plan = ExecutionPlan(phases=phases)
        self._calculate_plan_metrics(plan)
        
        logger.info(
            f"Created execution plan with {len(phases)} phases, "
            f"estimated time: {plan.total_estimated_time_ms:.1f}ms, "
            f"optimization score: {plan.optimization_score:.3f}"
        )
        
        return plan
    
    def _quantum_optimize_tasks(self, tasks: List[QuantumTask]) -> List[QuantumTask]:
        """
        Optimize task scheduling using quantum-inspired algorithms.
        
        This uses quantum annealing-like optimization to find the best
        task scheduling configuration by minimizing an energy function.
        """
        logger.debug(f"Running quantum optimization for {len(tasks)} tasks")
        
        best_configuration = tasks.copy()
        best_energy = self._calculate_system_energy(tasks)
        
        current_temperature = 1.0
        cooling_rate = 0.95
        
        for iteration in range(self.optimization_iterations):
            # Create quantum superposition of possible configurations
            candidate_tasks = self._quantum_mutation(tasks)
            
            # Calculate energy of new configuration
            candidate_energy = self._calculate_system_energy(candidate_tasks)
            
            # Quantum tunneling: accept worse solutions with quantum probability
            energy_delta = candidate_energy - best_energy
            quantum_probability = math.exp(-energy_delta / (current_temperature + 1e-10))
            
            if energy_delta < 0 or random.random() < quantum_probability:
                best_configuration = candidate_tasks
                best_energy = candidate_energy
                tasks = candidate_tasks
            
            # Simulated quantum decoherence
            current_temperature *= cooling_rate
            
            # Update quantum coherence
            self._update_quantum_coherence()
        
        logger.debug(f"Quantum optimization completed with energy: {best_energy:.3f}")
        return best_configuration
    
    def _quantum_mutation(self, tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Apply quantum mutations to task configuration."""
        mutated_tasks = []
        
        for task in tasks:
            mutated_task = QuantumTask(
                id=task.id,
                name=task.name,
                function=task.function,
                args=task.args,
                dependencies=task.dependencies.copy(),
                entangled_with=task.entangled_with.copy(),
                priority=task.priority,
                probability_amplitude=task.probability_amplitude,
                state=task.state,
                estimated_duration_ms=task.estimated_duration_ms,
                resource_requirements=task.resource_requirements.copy(),
                success_probability=task.success_probability,
                metadata=task.metadata.copy(),
            )
            
            # Quantum mutation: slightly adjust probability amplitude
            mutation_strength = 0.1
            phase_mutation = random.uniform(-mutation_strength, mutation_strength)
            amplitude_mutation = random.uniform(-mutation_strength, mutation_strength)
            
            current_amplitude = mutated_task.probability_amplitude
            new_real = current_amplitude.real + amplitude_mutation
            new_imag = current_amplitude.imag + phase_mutation
            
            mutated_task.probability_amplitude = complex(new_real, new_imag)
            mutated_task._normalize_amplitude()
            
            mutated_tasks.append(mutated_task)
        
        return mutated_tasks
    
    def _calculate_system_energy(self, tasks: List[QuantumTask]) -> float:
        """
        Calculate the energy of the task system configuration.
        
        Lower energy = better configuration
        Energy components:
        - Task priority (higher priority = lower energy)
        - Resource utilization efficiency
        - Dependency satisfaction
        - Parallelization potential
        """
        total_energy = 0.0
        
        # Priority energy: higher priority tasks should have lower energy
        priority_energy = sum(1.0 / max(task.priority, 0.1) for task in tasks)
        
        # Resource efficiency energy
        resource_usage = self._calculate_resource_usage(tasks)
        resource_energy = sum(
            (usage / limit) ** 2 for usage, limit in zip(
                resource_usage.values(), self.resource_limits.values()
            )
        )
        
        # Dependency energy: penalize unsatisfied dependencies
        dependency_energy = 0.0
        task_ids = {task.id for task in tasks}
        for task in tasks:
            unsatisfied_deps = task.dependencies - task_ids
            dependency_energy += len(unsatisfied_deps) * 10.0
        
        # Entanglement energy: consider quantum correlations
        entanglement_energy = self._calculate_entanglement_energy(tasks)
        
        total_energy = (
            priority_energy * 0.3 +
            resource_energy * 0.3 +
            dependency_energy * 0.3 +
            entanglement_energy * 0.1
        )
        
        return total_energy
    
    def _calculate_entanglement_energy(self, tasks: List[QuantumTask]) -> float:
        """Calculate energy contribution from quantum entanglements."""
        entanglement_energy = 0.0
        task_ids = {task.id for task in tasks}
        
        for task in tasks:
            for entangled_id in task.entangled_with:
                if entangled_id in task_ids:
                    # Get entanglement strength
                    strength = self._entanglement_matrix.get((task.id, entangled_id), 0.0)
                    
                    # Calculate quantum correlation energy
                    # Higher correlation should lead to better scheduling
                    correlation = abs(task.probability_amplitude * 
                                    self._quantum_register[entangled_id].probability_amplitude.conjugate())
                    entanglement_energy -= strength * correlation  # Negative = good
        
        return entanglement_energy
    
    def _calculate_resource_usage(self, tasks: List[QuantumTask]) -> Dict[str, float]:
        """Calculate resource usage for a set of tasks."""
        usage = {resource: 0.0 for resource in self.resource_limits.keys()}
        
        for task in tasks:
            for resource, requirement in task.resource_requirements.items():
                if resource in usage:
                    usage[resource] += requirement
        
        return usage
    
    def _create_execution_phases(self, tasks: List[QuantumTask]) -> List[List[QuantumTask]]:
        """
        Create execution phases using quantum-enhanced topological sorting.
        
        Tasks are grouped into phases based on:
        1. Dependency constraints
        2. Quantum entanglement patterns
        3. Resource availability
        4. Parallelization opportunities
        """
        phases = []
        remaining_tasks = tasks.copy()
        completed_task_ids = set()
        
        while remaining_tasks:
            # Find tasks that can execute in parallel (quantum superposition)
            ready_tasks = []
            
            for task in remaining_tasks:
                # Check if all dependencies are satisfied
                if task.dependencies.issubset(completed_task_ids):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Break circular dependencies using quantum tunneling
                logger.warning("Circular dependencies detected, applying quantum tunneling")
                ready_tasks = self._resolve_circular_dependencies(remaining_tasks)
            
            # Select optimal subset for parallel execution
            phase_tasks = self._select_parallel_tasks(ready_tasks)
            
            if phase_tasks:
                phases.append(phase_tasks)
                
                # Mark tasks as completed
                for task in phase_tasks:
                    completed_task_ids.add(task.id)
                    remaining_tasks.remove(task)
                    task.collapse_to_state(TaskState.COLLAPSED)
            else:
                # Emergency fallback
                logger.error("Unable to make progress, adding first remaining task")
                phases.append([remaining_tasks[0]])
                completed_task_ids.add(remaining_tasks[0].id)
                remaining_tasks.pop(0)
        
        return phases
    
    def _resolve_circular_dependencies(self, tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Resolve circular dependencies using quantum tunneling."""
        # Find tasks with highest quantum probability
        tasks_by_probability = sorted(tasks, key=lambda t: t.probability, reverse=True)
        
        # Select top tasks that can "tunnel" through dependency barriers
        max_tunneling_tasks = min(3, len(tasks_by_probability))
        return tasks_by_probability[:max_tunneling_tasks]
    
    def _select_parallel_tasks(self, ready_tasks: List[QuantumTask]) -> List[QuantumTask]:
        """
        Select optimal subset of tasks for parallel execution.
        
        Uses quantum superposition principle to evaluate all possible
        combinations and select the best one.
        """
        if not ready_tasks:
            return []
        
        if len(ready_tasks) <= self.max_parallel_tasks:
            return ready_tasks
        
        # Quantum selection: use probability amplitudes to guide selection
        tasks_with_scores = []
        
        for task in ready_tasks:
            # Calculate quantum selection score
            score = (
                task.probability * 0.4 +              # Quantum probability
                task.priority / 10.0 * 0.3 +          # Priority
                task.success_probability * 0.2 +       # Success likelihood
                (1.0 / max(task.estimated_duration_ms / 1000.0, 0.1)) * 0.1  # Speed
            )
            tasks_with_scores.append((task, score))
        
        # Sort by quantum score and select top tasks
        tasks_with_scores.sort(key=lambda x: x[1], reverse=True)
        selected_tasks = [task for task, _ in tasks_with_scores[:self.max_parallel_tasks]]
        
        # Verify resource constraints
        if not self._check_resource_constraints(selected_tasks):
            # Reduce selection to fit resource constraints
            selected_tasks = self._reduce_for_resources(selected_tasks)
        
        return selected_tasks
    
    def _check_resource_constraints(self, tasks: List[QuantumTask]) -> bool:
        """Check if tasks fit within resource constraints."""
        usage = self._calculate_resource_usage(tasks)
        
        for resource, used in usage.items():
            if resource in self.resource_limits:
                if used > self.resource_limits[resource]:
                    return False
        
        return True
    
    def _reduce_for_resources(self, tasks: List[QuantumTask]) -> List[QuantumTask]:
        """Reduce task selection to fit resource constraints."""
        # Greedy selection based on resource efficiency
        selected = []
        current_usage = {resource: 0.0 for resource in self.resource_limits.keys()}
        
        # Sort by resource efficiency (quantum score / resource usage)
        efficiency_sorted = []
        for task in tasks:
            total_resources = sum(task.resource_requirements.values()) + 1e-6
            efficiency = task.probability / total_resources
            efficiency_sorted.append((task, efficiency))
        
        efficiency_sorted.sort(key=lambda x: x[1], reverse=True)
        
        for task, _ in efficiency_sorted:
            # Check if we can add this task
            can_add = True
            for resource, requirement in task.resource_requirements.items():
                if resource in current_usage:
                    if current_usage[resource] + requirement > self.resource_limits[resource]:
                        can_add = False
                        break
            
            if can_add:
                selected.append(task)
                for resource, requirement in task.resource_requirements.items():
                    if resource in current_usage:
                        current_usage[resource] += requirement
        
        return selected
    
    def _calculate_plan_metrics(self, plan: ExecutionPlan) -> None:
        """Calculate metrics for the execution plan."""
        if not plan.phases:
            return
        
        # Calculate total estimated time (phases run sequentially)
        plan.total_estimated_time_ms = sum(
            max(task.estimated_duration_ms for task in phase) if phase else 0
            for phase in plan.phases
        )
        
        # Calculate parallelism factor
        total_tasks = sum(len(phase) for phase in plan.phases)
        sequential_time = sum(
            sum(task.estimated_duration_ms for task in phase)
            for phase in plan.phases
        )
        
        if plan.total_estimated_time_ms > 0:
            plan.parallelism_factor = sequential_time / plan.total_estimated_time_ms
        
        # Calculate resource utilization
        all_tasks = [task for phase in plan.phases for task in phase]
        plan.resource_utilization = self._calculate_resource_usage(all_tasks)
        
        # Calculate optimization score (0-1, higher is better)
        plan.optimization_score = min(1.0, plan.parallelism_factor / len(plan.phases))
        
        # Calculate quantum coherence
        plan.quantum_coherence = self._get_current_coherence()
        
        # Add metadata
        plan.metadata = {
            "total_tasks": total_tasks,
            "avg_tasks_per_phase": total_tasks / len(plan.phases) if plan.phases else 0,
            "quantum_entanglements": len([
                task for task in all_tasks if task.entangled_with
            ]),
            "optimization_iterations": self.optimization_iterations,
        }
    
    def _update_quantum_coherence(self) -> None:
        """Update quantum coherence based on time decay."""
        current_time = time.time()
        time_elapsed = current_time - self._coherence_time
        
        # Apply coherence decay
        for task in self._quantum_register.values():
            decay_factor = math.exp(-time_elapsed * (1 - self.quantum_coherence_decay))
            task.probability_amplitude *= decay_factor
            task._normalize_amplitude()
        
        self._coherence_time = current_time
    
    def _get_current_coherence(self) -> float:
        """Get current system coherence level."""
        if not self._quantum_register:
            return 1.0
        
        # Calculate coherence as average amplitude magnitude
        total_amplitude = sum(
            abs(task.probability_amplitude) for task in self._quantum_register.values()
        )
        
        return total_amplitude / len(self._quantum_register)
    
    async def execute_plan(
        self,
        plan: ExecutionPlan,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Execute the quantum-optimized plan.
        
        Args:
            plan: ExecutionPlan to execute
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with execution results
        """
        logger.info(f"Executing quantum plan with {len(plan.phases)} phases")
        
        start_time = time.time()
        results = []
        phase_results = []
        
        for phase_idx, phase_tasks in enumerate(plan.phases):
            logger.info(f"Executing phase {phase_idx + 1}/{len(plan.phases)} with {len(phase_tasks)} tasks")
            
            # Execute phase tasks in parallel
            phase_start = time.time()
            
            # Collapse task states to executing
            for task in phase_tasks:
                task.collapse_to_state(TaskState.EXECUTING)
            
            # Create async tasks
            async_tasks = []
            for task in phase_tasks:
                if task.function:
                    async_task = asyncio.create_task(
                        self._execute_quantum_task(task)
                    )
                    async_tasks.append((task, async_task))
            
            # Wait for phase completion
            phase_task_results = []
            for task, async_task in async_tasks:
                try:
                    result = await async_task
                    task.collapse_to_state(TaskState.COMPLETED)
                    phase_task_results.append(result)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Task {task.id} failed: {e}")
                    task.collapse_to_state(TaskState.FAILED)
                    error_result = {
                        "task_id": task.id,
                        "task_name": task.name,
                        "success": False,
                        "error": str(e),
                        "execution_time_ms": (time.time() - phase_start) * 1000,
                    }
                    phase_task_results.append(error_result)
                    results.append(error_result)
            
            phase_time = (time.time() - phase_start) * 1000
            phase_results.append({
                "phase": phase_idx + 1,
                "tasks": len(phase_tasks),
                "results": phase_task_results,
                "execution_time_ms": phase_time,
            })
            
            # Progress callback
            if progress_callback:
                await progress_callback(phase_idx + 1, len(plan.phases), phase_results[-1])
        
        total_time = (time.time() - start_time) * 1000
        
        execution_result = {
            "success": True,
            "total_phases": len(plan.phases),
            "total_tasks": sum(len(phase) for phase in plan.phases),
            "successful_tasks": sum(1 for r in results if r.get("success", False)),
            "failed_tasks": sum(1 for r in results if not r.get("success", False)),
            "total_execution_time_ms": total_time,
            "estimated_time_ms": plan.total_estimated_time_ms,
            "time_efficiency": plan.total_estimated_time_ms / total_time if total_time > 0 else 0,
            "parallelism_achieved": plan.parallelism_factor,
            "quantum_coherence": plan.quantum_coherence,
            "phases": phase_results,
            "results": results,
        }
        
        logger.info(
            f"Quantum execution completed: {execution_result['successful_tasks']}"
            f"/{execution_result['total_tasks']} tasks successful in {total_time:.1f}ms"
        )
        
        return execution_result
    
    async def _execute_quantum_task(self, task: QuantumTask) -> Dict[str, Any]:
        """Execute a single quantum task."""
        start_time = time.time()
        
        try:
            if task.function:
                # Execute the actual function
                if asyncio.iscoroutinefunction(task.function):
                    result = await task.function(**task.args)
                else:
                    result = task.function(**task.args)
                
                execution_time = (time.time() - start_time) * 1000
                
                return {
                    "task_id": task.id,
                    "task_name": task.name,
                    "success": True,
                    "result": result,
                    "execution_time_ms": execution_time,
                    "quantum_probability": task.probability,
                    "metadata": task.metadata,
                }
            else:
                # Simulate task execution
                await asyncio.sleep(task.estimated_duration_ms / 1000.0)
                execution_time = (time.time() - start_time) * 1000
                
                return {
                    "task_id": task.id,
                    "task_name": task.name,
                    "success": True,
                    "result": f"Simulated result for {task.name}",
                    "execution_time_ms": execution_time,
                    "quantum_probability": task.probability,
                    "metadata": task.metadata,
                }
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Quantum task {task.id} failed: {e}")
            
            return {
                "task_id": task.id,
                "task_name": task.name,
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time,
                "quantum_probability": task.probability,
                "metadata": task.metadata,
            }
    
    def get_quantum_metrics(self) -> Dict[str, Any]:
        """Get current quantum system metrics."""
        return {
            "registered_tasks": len(self._quantum_register),
            "entangled_pairs": len(self._entanglement_matrix) // 2,
            "quantum_coherence": self._get_current_coherence(),
            "superposition_tasks": len([
                t for t in self._quantum_register.values() 
                if t.state == TaskState.SUPERPOSITION
            ]),
            "collapsed_tasks": len([
                t for t in self._quantum_register.values() 
                if t.state == TaskState.COLLAPSED
            ]),
            "executing_tasks": len([
                t for t in self._quantum_register.values() 
                if t.state == TaskState.EXECUTING
            ]),
            "completed_tasks": len([
                t for t in self._quantum_register.values() 
                if t.state == TaskState.COMPLETED
            ]),
            "resource_limits": self.resource_limits,
            "optimization_iterations": self.optimization_iterations,
        }
    
    def visualize_quantum_state(self) -> str:
        """Generate a text visualization of the quantum state."""
        lines = ["🌌 QUANTUM TASK PLANNER STATE", "=" * 40]
        
        # System overview
        metrics = self.get_quantum_metrics()
        lines.append(f"📊 Tasks: {metrics['registered_tasks']} total")
        lines.append(f"🔗 Entanglements: {metrics['entangled_pairs']} pairs")
        lines.append(f"🌊 Coherence: {metrics['quantum_coherence']:.3f}")
        lines.append("")
        
        # Task states
        states = {
            TaskState.SUPERPOSITION: "⚡",
            TaskState.ENTANGLED: "🔗",
            TaskState.COLLAPSED: "📍",
            TaskState.EXECUTING: "🚀",
            TaskState.COMPLETED: "✅",
            TaskState.FAILED: "❌",
        }
        
        for state, icon in states.items():
            count = metrics.get(f"{state.value}_tasks", 0)
            if count > 0:
                lines.append(f"{icon} {state.value.title()}: {count}")
        
        lines.append("")
        
        # Top tasks by quantum probability
        top_tasks = sorted(
            self._quantum_register.values(),
            key=lambda t: t.probability,
            reverse=True
        )[:5]
        
        lines.append("🎯 Top Tasks by Quantum Probability:")
        for i, task in enumerate(top_tasks, 1):
            prob_bar = "█" * int(task.probability * 20)
            lines.append(f"  {i}. {task.name} [{prob_bar}] {task.probability:.3f}")
        
        return "\n".join(lines)