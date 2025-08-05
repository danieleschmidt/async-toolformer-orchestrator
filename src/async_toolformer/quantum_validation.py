"""
Quantum Validation Module for AsyncOrchestrator.

This module provides comprehensive validation for quantum-inspired task execution:
- Input parameter validation with quantum constraints
- Task dependency validation
- Resource constraint validation  
- Output validation and verification
- Quantum state consistency checks
"""

import asyncio
import math
import re
from typing import Any, Dict, List, Optional, Set, Union, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import inspect
from collections import defaultdict

from .quantum_planner import QuantumTask, TaskState
from .exceptions import ConfigurationError, ToolExecutionError

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    QUANTUM_COHERENT = "quantum_coherent"


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, field: str, value: Any, message: str, validation_level: ValidationLevel):
        self.field = field
        self.value = value
        self.message = message
        self.validation_level = validation_level
        super().__init__(f"Validation error for {field}: {message}")


@dataclass
class ValidationRule:
    """A validation rule for parameters or tasks."""
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    applies_to: Set[str] = field(default_factory=set)  # Parameter names or task types
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, field: str, value: Any, message: str, level: ValidationLevel):
        """Add a validation error."""
        error = ValidationError(field, value, message, level)
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add a validation warning."""
        self.warnings.append(message)
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        if not other.is_valid:
            self.is_valid = False
        self.metadata.update(other.metadata)


class QuantumValidator:
    """
    Quantum-inspired validation system for task orchestration.
    
    Provides comprehensive validation including:
    - Parameter type and value validation
    - Quantum state consistency checks
    - Resource constraint validation
    - Dependency cycle detection
    - Output format validation
    """
    
    def __init__(
        self,
        validation_level: ValidationLevel = ValidationLevel.STANDARD,
        enable_quantum_coherence_checks: bool = True,
        max_dependency_depth: int = 10,
        strict_type_checking: bool = True,
    ):
        """
        Initialize the quantum validator.
        
        Args:
            validation_level: Default validation strictness level
            enable_quantum_coherence_checks: Whether to check quantum coherence
            max_dependency_depth: Maximum allowed dependency chain depth
            strict_type_checking: Whether to enforce strict type checking
        """
        self.validation_level = validation_level
        self.enable_quantum_coherence_checks = enable_quantum_coherence_checks
        self.max_dependency_depth = max_dependency_depth
        self.strict_type_checking = strict_type_checking
        
        # Validation rules registry
        self._validation_rules: Dict[str, List[ValidationRule]] = defaultdict(list)
        self._quantum_constraints: Dict[str, Dict[str, Any]] = {}
        
        # Initialize built-in validation rules
        self._initialize_builtin_rules()
        
        logger.info("QuantumValidator initialized")
    
    def _initialize_builtin_rules(self):
        """Initialize built-in validation rules."""
        # String validation rules
        self.add_validation_rule(ValidationRule(
            name="string_length",
            validator=lambda x: isinstance(x, str) and 0 < len(x) <= 10000,
            error_message="String must be between 1 and 10000 characters",
            validation_level=ValidationLevel.BASIC
        ))
        
        self.add_validation_rule(ValidationRule(
            name="no_null_bytes",
            validator=lambda x: isinstance(x, str) and '\x00' not in x,
            error_message="String must not contain null bytes",
            validation_level=ValidationLevel.STANDARD
        ))
        
        # Numeric validation rules
        self.add_validation_rule(ValidationRule(
            name="finite_number",
            validator=lambda x: isinstance(x, (int, float)) and math.isfinite(x),
            error_message="Number must be finite (not infinity or NaN)",
            validation_level=ValidationLevel.BASIC
        ))
        
        self.add_validation_rule(ValidationRule(
            name="reasonable_range",
            validator=lambda x: isinstance(x, (int, float)) and -10**12 <= x <= 10**12,
            error_message="Number must be within reasonable range (-10^12 to 10^12)",
            validation_level=ValidationLevel.STANDARD
        ))
        
        # Quantum-specific rules
        self.add_validation_rule(ValidationRule(
            name="quantum_probability",
            validator=lambda x: isinstance(x, (int, float)) and 0 <= x <= 1,
            error_message="Quantum probability must be between 0 and 1",
            validation_level=ValidationLevel.QUANTUM_COHERENT,
            applies_to={"probability", "success_probability", "quantum_probability"}
        ))
        
        self.add_validation_rule(ValidationRule(
            name="quantum_amplitude",
            validator=lambda x: isinstance(x, complex) and abs(x) <= 1,
            error_message="Quantum amplitude must have magnitude <= 1",
            validation_level=ValidationLevel.QUANTUM_COHERENT,
            applies_to={"amplitude", "probability_amplitude", "quantum_amplitude"}
        ))
        
        # Resource validation rules
        self.add_validation_rule(ValidationRule(
            name="positive_resource",
            validator=lambda x: isinstance(x, (int, float)) and x > 0,
            error_message="Resource requirement must be positive",
            validation_level=ValidationLevel.STANDARD,
            applies_to={"cpu", "memory", "network", "io", "duration_ms"}
        ))
        
        # Collection validation rules
        self.add_validation_rule(ValidationRule(
            name="reasonable_collection_size",
            validator=lambda x: isinstance(x, (list, dict, set)) and len(x) <= 1000,
            error_message="Collection size must not exceed 1000 items",
            validation_level=ValidationLevel.STANDARD
        ))
    
    def add_validation_rule(self, rule: ValidationRule):
        """Add a custom validation rule."""
        self._validation_rules[rule.name].append(rule)
        logger.debug(f"Added validation rule: {rule.name}")
    
    def add_quantum_constraint(self, parameter: str, constraint: Dict[str, Any]):
        """Add quantum-specific constraints for a parameter."""
        self._quantum_constraints[parameter] = constraint
        logger.debug(f"Added quantum constraint for parameter: {parameter}")
    
    def validate_task_parameters(
        self,
        task: QuantumTask,
        validation_level: Optional[ValidationLevel] = None
    ) -> ValidationResult:
        """
        Validate task parameters against all applicable rules.
        
        Args:
            task: QuantumTask to validate
            validation_level: Override validation level
            
        Returns:
            ValidationResult with validation outcome
        """
        level = validation_level or self.validation_level
        result = ValidationResult(is_valid=True)
        
        try:
            # Validate task metadata
            self._validate_task_metadata(task, result, level)
            
            # Validate function parameters
            if task.function:
                self._validate_function_parameters(task.function, task.args, result, level)
            
            # Validate task arguments
            self._validate_task_arguments(task.args, result, level)
            
            # Validate quantum properties
            if level == ValidationLevel.QUANTUM_COHERENT:
                self._validate_quantum_properties(task, result)
            
            # Validate resource requirements
            self._validate_resource_requirements(task.resource_requirements, result, level)
            
            result.metadata.update({
                "task_id": task.id,
                "validation_level": level.value,
                "rules_checked": len(self._get_applicable_rules(level)),
                "quantum_validation": level == ValidationLevel.QUANTUM_COHERENT,
            })
            
        except Exception as e:
            result.add_error("validation", task, f"Validation failed: {e}", level)
            logger.error(f"Error validating task {task.id}: {e}")
        
        return result
    
    def _validate_task_metadata(self, task: QuantumTask, result: ValidationResult, level: ValidationLevel):
        """Validate task metadata fields."""
        # Required fields
        if not task.id:
            result.add_error("id", task.id, "Task ID is required", level)
        elif not isinstance(task.id, str) or len(task.id) < 1:
            result.add_error("id", task.id, "Task ID must be a non-empty string", level)
        
        if not task.name:
            result.add_error("name", task.name, "Task name is required", level)
        elif not isinstance(task.name, str) or len(task.name) < 1:
            result.add_error("name", task.name, "Task name must be a non-empty string", level)
        
        # Numeric fields
        if not isinstance(task.priority, (int, float)) or task.priority < 0:
            result.add_error("priority", task.priority, "Priority must be a non-negative number", level)
        
        if not isinstance(task.estimated_duration_ms, (int, float)) or task.estimated_duration_ms < 0:
            result.add_error("estimated_duration_ms", task.estimated_duration_ms, 
                           "Estimated duration must be non-negative", level)
        
        if not isinstance(task.success_probability, (int, float)) or not 0 <= task.success_probability <= 1:
            result.add_error("success_probability", task.success_probability,
                           "Success probability must be between 0 and 1", level)
        
        # State validation
        if not isinstance(task.state, TaskState):
            result.add_error("state", task.state, "State must be a valid TaskState", level)
    
    def _validate_function_parameters(
        self,
        function: Callable,
        args: Dict[str, Any],
        result: ValidationResult,
        level: ValidationLevel
    ):
        """Validate function parameters against function signature."""
        try:
            sig = inspect.signature(function)
            
            # Check for required parameters
            for param_name, param in sig.parameters.items():
                if param.default == inspect.Parameter.empty and param_name not in args:
                    result.add_error(param_name, None, 
                                   f"Required parameter '{param_name}' is missing", level)
                
                # Type checking if enabled and annotation available
                if (self.strict_type_checking and param_name in args and 
                    param.annotation != inspect.Parameter.empty):
                    self._validate_parameter_type(param_name, args[param_name], 
                                                param.annotation, result, level)
            
            # Check for unexpected parameters
            expected_params = set(sig.parameters.keys())
            provided_params = set(args.keys())
            unexpected = provided_params - expected_params
            
            if unexpected:
                for param in unexpected:
                    result.add_warning(f"Unexpected parameter '{param}' provided")
                    
        except Exception as e:
            result.add_error("function_signature", function, 
                           f"Error validating function signature: {e}", level)
    
    def _validate_parameter_type(
        self,
        param_name: str,
        value: Any,
        expected_type: type,
        result: ValidationResult,
        level: ValidationLevel
    ):
        """Validate parameter type against annotation."""
        try:
            # Handle Union types (Optional, etc.)
            if hasattr(expected_type, '__origin__'):
                if expected_type.__origin__ is Union:
                    # Check if value matches any of the union types
                    union_types = expected_type.__args__
                    if not any(isinstance(value, t) for t in union_types if t != type(None)):
                        result.add_error(param_name, value,
                                       f"Parameter must be one of {union_types}", level)
                    return
                elif expected_type.__origin__ in (list, List):
                    if not isinstance(value, list):
                        result.add_error(param_name, value, "Parameter must be a list", level)
                    return
                elif expected_type.__origin__ in (dict, Dict):
                    if not isinstance(value, dict):
                        result.add_error(param_name, value, "Parameter must be a dict", level)
                    return
            
            # Basic type checking
            if not isinstance(value, expected_type):
                result.add_error(param_name, value,
                               f"Parameter must be of type {expected_type.__name__}", level)
                
        except Exception as e:
            result.add_warning(f"Could not validate type for parameter '{param_name}': {e}")
    
    def _validate_task_arguments(
        self,
        args: Dict[str, Any],
        result: ValidationResult,
        level: ValidationLevel
    ):
        """Validate task arguments against validation rules."""
        applicable_rules = self._get_applicable_rules(level)
        
        for arg_name, arg_value in args.items():
            for rule_name, rules in applicable_rules.items():
                for rule in rules:
                    # Check if rule applies to this argument
                    if rule.applies_to and arg_name not in rule.applies_to:
                        continue
                    
                    # Apply validation rule
                    try:
                        if not rule.validator(arg_value):
                            result.add_error(arg_name, arg_value, rule.error_message, level)
                    except Exception as e:
                        result.add_error(arg_name, arg_value,
                                       f"Validation rule '{rule_name}' failed: {e}", level)
            
            # Apply quantum constraints if they exist
            if arg_name in self._quantum_constraints:
                self._apply_quantum_constraint(arg_name, arg_value, result, level)
    
    def _validate_quantum_properties(self, task: QuantumTask, result: ValidationResult):
        """Validate quantum-specific properties of a task."""
        # Validate probability amplitude
        amplitude = task.probability_amplitude
        if not isinstance(amplitude, complex):
            result.add_error("probability_amplitude", amplitude,
                           "Probability amplitude must be a complex number",
                           ValidationLevel.QUANTUM_COHERENT)
        elif abs(amplitude) > 1.0001:  # Small tolerance for floating point errors
            result.add_error("probability_amplitude", amplitude,
                           "Probability amplitude magnitude must not exceed 1",
                           ValidationLevel.QUANTUM_COHERENT)
        
        # Validate quantum probability consistency
        calculated_prob = abs(amplitude) ** 2
        if abs(calculated_prob - task.probability) > 0.001:
            result.add_warning(
                f"Probability inconsistency detected: calculated={calculated_prob:.3f}, "
                f"stored={task.probability:.3f}"
            )
        
        # Validate entanglement relationships
        if task.entangled_with:
            for entangled_id in task.entangled_with:
                if not isinstance(entangled_id, str) or not entangled_id:
                    result.add_error("entangled_with", entangled_id,
                                   "Entangled task IDs must be non-empty strings",
                                   ValidationLevel.QUANTUM_COHERENT)
        
        # Validate dependencies don't create circular references
        if task.dependencies:
            for dep_id in task.dependencies:
                if dep_id == task.id:
                    result.add_error("dependencies", dep_id,
                                   "Task cannot depend on itself",
                                   ValidationLevel.QUANTUM_COHERENT)
    
    def _validate_resource_requirements(
        self,
        resources: Dict[str, float],
        result: ValidationResult,
        level: ValidationLevel
    ):
        """Validate resource requirements."""
        valid_resources = {"cpu", "memory", "network", "io", "disk", "gpu"}
        
        for resource, value in resources.items():
            if not isinstance(resource, str):
                result.add_error("resource_name", resource,
                               "Resource name must be a string", level)
                continue
            
            if level in [ValidationLevel.STRICT, ValidationLevel.QUANTUM_COHERENT]:
                if resource not in valid_resources:
                    result.add_warning(f"Unknown resource type: {resource}")
            
            if not isinstance(value, (int, float)) or value < 0:
                result.add_error(f"resource_{resource}", value,
                               "Resource requirement must be a non-negative number", level)
            
            # Check for reasonable bounds
            max_reasonable = {
                "cpu": 1000.0,      # CPU units
                "memory": 32768.0,  # MB
                "network": 10000.0, # MB/s
                "io": 10000.0,      # IOPS
                "disk": 1000000.0,  # MB
                "gpu": 8.0,         # GPU units
            }
            
            if resource in max_reasonable and value > max_reasonable[resource]:
                result.add_warning(
                    f"Resource requirement for {resource} ({value}) seems very high"
                )
    
    def _apply_quantum_constraint(
        self,
        param_name: str,
        value: Any,
        result: ValidationResult,
        level: ValidationLevel
    ):
        """Apply quantum-specific constraints to a parameter."""
        constraint = self._quantum_constraints[param_name]
        
        # Range constraints
        if "min" in constraint and value < constraint["min"]:
            result.add_error(param_name, value,
                           f"Value must be >= {constraint['min']}", level)
        
        if "max" in constraint and value > constraint["max"]:
            result.add_error(param_name, value,
                           f"Value must be <= {constraint['max']}", level)
        
        # Quantum coherence constraints
        if "coherence_required" in constraint and constraint["coherence_required"]:
            if isinstance(value, complex) and abs(value) < 0.1:
                result.add_warning(f"Low coherence detected for {param_name}")
        
        # Entanglement constraints
        if "entanglement_allowed" in constraint and not constraint["entanglement_allowed"]:
            if hasattr(value, "entangled_with") and value.entangled_with:
                result.add_error(param_name, value,
                               "Entanglement not allowed for this parameter", level)
    
    def _get_applicable_rules(self, level: ValidationLevel) -> Dict[str, List[ValidationRule]]:
        """Get validation rules applicable at the given level."""
        applicable = defaultdict(list)
        
        level_order = [ValidationLevel.BASIC, ValidationLevel.STANDARD, 
                      ValidationLevel.STRICT, ValidationLevel.QUANTUM_COHERENT]
        max_level_index = level_order.index(level)
        
        for rule_name, rules in self._validation_rules.items():
            for rule in rules:
                if level_order.index(rule.validation_level) <= max_level_index:
                    applicable[rule_name].append(rule)
        
        return applicable
    
    def validate_task_dependencies(
        self,
        tasks: List[QuantumTask],
        validation_level: Optional[ValidationLevel] = None
    ) -> ValidationResult:
        """
        Validate task dependencies for cycles and depth.
        
        Args:
            tasks: List of tasks to validate
            validation_level: Override validation level
            
        Returns:
            ValidationResult with dependency validation outcome
        """
        level = validation_level or self.validation_level
        result = ValidationResult(is_valid=True)
        
        try:
            task_map = {task.id: task for task in tasks}
            
            # Check for circular dependencies using DFS
            visited = set()
            rec_stack = set()
            
            def has_cycle(task_id: str, path: List[str]) -> bool:
                if task_id in rec_stack:
                    cycle_start = path.index(task_id)
                    cycle = path[cycle_start:] + [task_id]
                    result.add_error("circular_dependency", cycle,
                                   f"Circular dependency detected: {' -> '.join(cycle)}", level)
                    return True
                
                if task_id in visited:
                    return False
                
                visited.add(task_id)
                rec_stack.add(task_id)
                
                task = task_map.get(task_id)
                if task:
                    for dep_id in task.dependencies:
                        if has_cycle(dep_id, path + [task_id]):
                            return True
                
                rec_stack.remove(task_id)
                return False
            
            # Check each task for cycles
            for task in tasks:
                if task.id not in visited:
                    has_cycle(task.id, [])
            
            # Check dependency depth
            def calculate_depth(task_id: str, memo: Dict[str, int]) -> int:
                if task_id in memo:
                    return memo[task_id]
                
                task = task_map.get(task_id)
                if not task or not task.dependencies:
                    memo[task_id] = 0
                    return 0
                
                max_dep_depth = max(
                    calculate_depth(dep_id, memo) for dep_id in task.dependencies
                    if dep_id in task_map
                )
                
                depth = max_dep_depth + 1
                memo[task_id] = depth
                
                if depth > self.max_dependency_depth:
                    result.add_error("dependency_depth", task_id,
                                   f"Dependency chain too deep: {depth} > {self.max_dependency_depth}",
                                   level)
                
                return depth
            
            depth_memo = {}
            for task in tasks:
                calculate_depth(task.id, depth_memo)
            
            # Check for missing dependencies
            all_task_ids = {task.id for task in tasks}
            for task in tasks:
                missing_deps = task.dependencies - all_task_ids
                if missing_deps:
                    result.add_error("missing_dependencies", list(missing_deps),
                                   f"Task {task.id} has missing dependencies: {missing_deps}",
                                   level)
            
            result.metadata.update({
                "tasks_validated": len(tasks),
                "max_depth_found": max(depth_memo.values()) if depth_memo else 0,
                "total_dependencies": sum(len(task.dependencies) for task in tasks),
                "validation_level": level.value,
            })
            
        except Exception as e:
            result.add_error("dependency_validation", tasks, 
                           f"Dependency validation failed: {e}", level)
            logger.error(f"Error validating task dependencies: {e}")
        
        return result
    
    def validate_execution_plan(
        self,
        execution_plan: 'ExecutionPlan',
        validation_level: Optional[ValidationLevel] = None
    ) -> ValidationResult:
        """
        Validate an execution plan for correctness and feasibility.
        
        Args:
            execution_plan: ExecutionPlan to validate
            validation_level: Override validation level
            
        Returns:
            ValidationResult with plan validation outcome
        """
        level = validation_level or self.validation_level
        result = ValidationResult(is_valid=True)
        
        try:
            # Validate plan structure
            if not execution_plan.phases:
                result.add_error("phases", execution_plan.phases,
                               "Execution plan must have at least one phase", level)
                return result
            
            # Validate each phase
            all_tasks = []
            for phase_idx, phase_tasks in enumerate(execution_plan.phases):
                if not phase_tasks:
                    result.add_warning(f"Phase {phase_idx + 1} is empty")
                    continue
                
                # Validate tasks in phase
                for task in phase_tasks:
                    if not isinstance(task, QuantumTask):
                        result.add_error(f"phase_{phase_idx}_task", task,
                                       "Phase must contain QuantumTask objects", level)
                        continue
                    
                    all_tasks.append(task)
                    
                    # Validate task parameters
                    task_result = self.validate_task_parameters(task, level)
                    result.merge(task_result)
            
            # Validate dependencies across phases
            dep_result = self.validate_task_dependencies(all_tasks, level)
            result.merge(dep_result)
            
            # Validate quantum coherence if enabled
            if self.enable_quantum_coherence_checks and level == ValidationLevel.QUANTUM_COHERENT:
                coherence_result = self._validate_quantum_coherence(execution_plan)
                result.merge(coherence_result)
            
            # Validate resource constraints
            resource_result = self._validate_plan_resources(execution_plan, level)
            result.merge(resource_result)
            
            result.metadata.update({
                "total_phases": len(execution_plan.phases),
                "total_tasks": len(all_tasks),
                "estimated_time_ms": execution_plan.total_estimated_time_ms,
                "parallelism_factor": execution_plan.parallelism_factor,
                "optimization_score": execution_plan.optimization_score,
                "validation_level": level.value,
            })
            
        except Exception as e:
            result.add_error("plan_validation", execution_plan,
                           f"Plan validation failed: {e}", level)
            logger.error(f"Error validating execution plan: {e}")
        
        return result
    
    def _validate_quantum_coherence(self, execution_plan: 'ExecutionPlan') -> ValidationResult:
        """Validate quantum coherence properties of the execution plan."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check overall coherence
            if execution_plan.quantum_coherence < 0.1:
                result.add_warning("Very low quantum coherence detected")
            elif execution_plan.quantum_coherence > 1.0:
                result.add_error("quantum_coherence", execution_plan.quantum_coherence,
                               "Quantum coherence cannot exceed 1.0",
                               ValidationLevel.QUANTUM_COHERENT)
            
            # Check amplitude normalization across phases
            for phase_idx, phase_tasks in enumerate(execution_plan.phases):
                total_probability = sum(task.probability for task in phase_tasks)
                
                if abs(total_probability - len(phase_tasks)) > 0.1:
                    result.add_warning(
                        f"Phase {phase_idx + 1} probability normalization may be incorrect"
                    )
            
            # Check entanglement consistency
            all_tasks = [task for phase in execution_plan.phases for task in phase]
            task_ids = {task.id for task in all_tasks}
            
            for task in all_tasks:
                for entangled_id in task.entangled_with:
                    if entangled_id not in task_ids:
                        result.add_error("entanglement", entangled_id,
                                       f"Task {task.id} entangled with non-existent task {entangled_id}",
                                       ValidationLevel.QUANTUM_COHERENT)
            
        except Exception as e:
            result.add_error("quantum_coherence", execution_plan,
                           f"Quantum coherence validation failed: {e}",
                           ValidationLevel.QUANTUM_COHERENT)
        
        return result
    
    def _validate_plan_resources(
        self,
        execution_plan: 'ExecutionPlan',
        level: ValidationLevel
    ) -> ValidationResult:
        """Validate resource utilization in the execution plan."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Check resource utilization
            for resource, utilization in execution_plan.resource_utilization.items():
                if not isinstance(utilization, (int, float)) or utilization < 0:
                    result.add_error(f"resource_{resource}", utilization,
                                   "Resource utilization must be non-negative", level)
                elif utilization > 1000000:  # Reasonable upper bound
                    result.add_warning(f"Very high resource utilization for {resource}: {utilization}")
            
            # Check parallelism factor
            if execution_plan.parallelism_factor < 1.0:
                result.add_error("parallelism_factor", execution_plan.parallelism_factor,
                               "Parallelism factor must be >= 1.0", level)
            elif execution_plan.parallelism_factor > 1000:
                result.add_warning("Extremely high parallelism factor detected")
            
            # Check optimization score
            if not 0 <= execution_plan.optimization_score <= 1:
                result.add_error("optimization_score", execution_plan.optimization_score,
                               "Optimization score must be between 0 and 1", level)
                
        except Exception as e:
            result.add_error("resource_validation", execution_plan,
                           f"Resource validation failed: {e}", level)
        
        return result
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation system statistics."""
        return {
            "validation_level": self.validation_level.value,
            "total_rules": sum(len(rules) for rules in self._validation_rules.values()),
            "quantum_constraints": len(self._quantum_constraints),
            "quantum_coherence_checks": self.enable_quantum_coherence_checks,
            "max_dependency_depth": self.max_dependency_depth,
            "strict_type_checking": self.strict_type_checking,
            "rule_categories": list(self._validation_rules.keys()),
            "constraint_parameters": list(self._quantum_constraints.keys()),
        }


# Convenience functions
def validate_task(
    task: QuantumTask,
    validator: Optional[QuantumValidator] = None,
    level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationResult:
    """Validate a single task with default settings."""
    if validator is None:
        validator = QuantumValidator(validation_level=level)
    return validator.validate_task_parameters(task, level)


def validate_tasks(
    tasks: List[QuantumTask],
    validator: Optional[QuantumValidator] = None,
    level: ValidationLevel = ValidationLevel.STANDARD
) -> ValidationResult:
    """Validate a list of tasks including dependencies."""
    if validator is None:
        validator = QuantumValidator(validation_level=level)
    
    result = ValidationResult(is_valid=True)
    
    # Validate individual tasks
    for task in tasks:
        task_result = validator.validate_task_parameters(task, level)
        result.merge(task_result)
    
    # Validate dependencies
    dep_result = validator.validate_task_dependencies(tasks, level)
    result.merge(dep_result)
    
    return result