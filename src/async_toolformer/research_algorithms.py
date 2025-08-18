"""
Research Mode: Novel Algorithm Development and Benchmarking

Advanced research implementations including novel quantum-inspired algorithms,
comparative studies, and performance breakthroughs for academic publication.
"""

import asyncio
import time
import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict, deque
import statistics

from .simple_structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class ResearchMetric:
    """Research performance metric."""
    algorithm_name: str
    metric_name: str
    value: float
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    statistical_significance: Optional[float] = None


@dataclass
class ExperimentResult:
    """Result of a controlled experiment."""
    experiment_id: str
    algorithm_name: str
    baseline_algorithm: str
    performance_metrics: List[ResearchMetric]
    improvement_percentage: float
    statistical_significance: float
    reproducible: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumInspiredTaskScheduler:
    """
    Novel quantum-inspired task scheduling algorithm based on
    superposition principles and entanglement for optimal resource allocation.
    
    Research Hypothesis: Quantum superposition principles can improve
    task scheduling efficiency by 15-30% over traditional algorithms.
    """
    
    def __init__(self, coherence_factor: float = 0.85):
        self.coherence_factor = coherence_factor
        self.quantum_states = {}
        self.entangled_tasks = defaultdict(set)
        self.measurements = []
        
    def create_task_superposition(self, task_id: str, execution_strategies: List[Dict[str, Any]]) -> None:
        """Create quantum superposition of execution strategies for a task."""
        # Normalize probabilities
        total_weight = sum(strategy.get('weight', 1.0) for strategy in execution_strategies)
        
        superposition = []
        for strategy in execution_strategies:
            probability = strategy.get('weight', 1.0) / total_weight
            superposition.append({
                'strategy': strategy,
                'amplitude': math.sqrt(probability),
                'phase': random.random() * 2 * math.pi
            })
        
        self.quantum_states[task_id] = {
            'superposition': superposition,
            'coherence': self.coherence_factor,
            'creation_time': time.time()
        }
    
    def entangle_tasks(self, task_ids: List[str], entanglement_strength: float = 0.8) -> None:
        """Create quantum entanglement between related tasks."""
        for i, task_a in enumerate(task_ids):
            for task_b in task_ids[i+1:]:
                self.entangled_tasks[task_a].add((task_b, entanglement_strength))
                self.entangled_tasks[task_b].add((task_a, entanglement_strength))
    
    def measure_optimal_strategy(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collapse superposition to optimal strategy based on current context.
        
        Novel contribution: Uses context-aware quantum measurement to select
        optimal execution strategy with consideration of entangled task states.
        """
        if task_id not in self.quantum_states:
            return {"strategy": "default", "confidence": 0.0}
        
        state = self.quantum_states[task_id]
        superposition = state['superposition']
        
        # Apply decoherence based on time elapsed
        time_elapsed = time.time() - state['creation_time']
        current_coherence = state['coherence'] * math.exp(-time_elapsed / 100)
        
        # Calculate context-aware probabilities
        contextualized_probabilities = []
        for quantum_state in superposition:
            strategy = quantum_state['strategy']
            base_amplitude = quantum_state['amplitude']
            
            # Context influence on amplitude
            context_factor = self._calculate_context_influence(strategy, context)
            
            # Entanglement influence
            entanglement_factor = self._calculate_entanglement_influence(task_id, strategy)
            
            # Combined probability with quantum effects
            effective_amplitude = base_amplitude * context_factor * entanglement_factor * current_coherence
            probability = effective_amplitude ** 2
            
            contextualized_probabilities.append({
                'strategy': strategy,
                'probability': probability,
                'quantum_confidence': current_coherence
            })
        
        # Select strategy based on quantum measurement
        total_prob = sum(cp['probability'] for cp in contextualized_probabilities)
        if total_prob == 0:
            # Fallback to uniform distribution
            selected = random.choice(contextualized_probabilities)
        else:
            # Weighted random selection (quantum measurement)
            random_value = random.random() * total_prob
            cumulative = 0
            selected = contextualized_probabilities[0]
            
            for cp in contextualized_probabilities:
                cumulative += cp['probability']
                if random_value <= cumulative:
                    selected = cp
                    break
        
        # Record measurement
        measurement = {
            'task_id': task_id,
            'selected_strategy': selected['strategy'],
            'quantum_confidence': selected['quantum_confidence'],
            'context': context,
            'timestamp': time.time()
        }
        self.measurements.append(measurement)
        
        return {
            'strategy': selected['strategy'],
            'confidence': selected['quantum_confidence'],
            'measurement_id': len(self.measurements) - 1
        }
    
    def _calculate_context_influence(self, strategy: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate how current context influences strategy selection."""
        influence = 1.0
        
        # Resource availability influence
        if 'cpu_usage' in context and 'cpu_requirement' in strategy:
            cpu_factor = max(0.1, 1.0 - (context['cpu_usage'] / 100.0))
            if strategy['cpu_requirement'] == 'high':
                influence *= cpu_factor
            elif strategy['cpu_requirement'] == 'low':
                influence *= (2.0 - cpu_factor)
        
        # Workload influence
        if 'workload_size' in context and 'parallelism' in strategy:
            workload_factor = min(2.0, context['workload_size'] / 10.0)
            if strategy['parallelism'] == 'high':
                influence *= workload_factor
            elif strategy['parallelism'] == 'low':
                influence *= (2.0 - workload_factor)
        
        return max(0.1, min(2.0, influence))
    
    def _calculate_entanglement_influence(self, task_id: str, strategy: Dict[str, Any]) -> float:
        """Calculate influence of entangled tasks on strategy selection."""
        if task_id not in self.entangled_tasks:
            return 1.0
        
        influence = 1.0
        for entangled_task, strength in self.entangled_tasks[task_id]:
            if entangled_task in self.quantum_states:
                # Consider the state of entangled tasks
                entangled_state = self.quantum_states[entangled_task]
                coherence = entangled_state['coherence']
                
                # Entanglement influence based on quantum correlation
                correlation = strength * coherence
                influence *= (1.0 + correlation * 0.5)
        
        return max(0.5, min(1.5, influence))


class AdaptiveLoadBalancer:
    """
    Novel adaptive load balancing algorithm using reinforcement learning
    principles for dynamic resource allocation optimization.
    
    Research Hypothesis: RL-based load balancing can achieve 20-40%
    better resource utilization than static algorithms.
    """
    
    def __init__(self, learning_rate: float = 0.1, exploration_rate: float = 0.2):
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.action_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        
    def get_state_representation(self, context: Dict[str, Any]) -> str:
        """Convert context to state representation for Q-learning."""
        # Discretize continuous values for state representation
        cpu_bin = min(9, int(context.get('cpu_usage', 0) / 10))
        memory_bin = min(9, int(context.get('memory_usage', 0) / 10))
        queue_bin = min(9, int(context.get('queue_length', 0) / 5))
        load_bin = min(9, int(context.get('system_load', 0) / 2))
        
        return f"{cpu_bin}_{memory_bin}_{queue_bin}_{load_bin}"
    
    def select_action(self, state: str, available_actions: List[str]) -> str:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.exploration_rate:
            # Exploration: random action
            return random.choice(available_actions)
        else:
            # Exploitation: best known action
            q_values = {action: self.q_table[state][action] for action in available_actions}
            return max(q_values, key=q_values.get)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str) -> None:
        """Update Q-value using Q-learning algorithm."""
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + 0.9 * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def calculate_reward(self, performance_metrics: Dict[str, float]) -> float:
        """Calculate reward based on performance metrics."""
        # Reward function considering multiple objectives
        throughput_score = performance_metrics.get('throughput', 0) / 100.0
        latency_score = max(0, 1.0 - performance_metrics.get('avg_latency', 1000) / 1000.0)
        resource_efficiency = performance_metrics.get('resource_efficiency', 0.5)
        
        # Weighted combination
        reward = (throughput_score * 0.4 + latency_score * 0.4 + resource_efficiency * 0.2)
        return reward * 100  # Scale to reasonable range
    
    async def optimize_load_distribution(self, workload: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize load distribution using learned policy."""
        context = self._analyze_current_context(workload)
        state = self.get_state_representation(context)
        
        # Available load balancing actions
        actions = ['round_robin', 'least_connections', 'cpu_based', 'adaptive_weighted']
        selected_action = self.select_action(state, actions)
        
        # Apply load balancing strategy
        distribution = await self._apply_balancing_strategy(selected_action, workload, context)
        
        # Record action for learning
        self.action_history.append({
            'state': state,
            'action': selected_action,
            'context': context,
            'timestamp': time.time()
        })
        
        return {
            'distribution': distribution,
            'strategy': selected_action,
            'state': state,
            'confidence': self._calculate_confidence(state, selected_action)
        }
    
    def _analyze_current_context(self, workload: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze current system context."""
        return {
            'cpu_usage': random.uniform(20, 80),  # Simulated
            'memory_usage': random.uniform(30, 70),
            'queue_length': len(workload),
            'system_load': random.uniform(1, 8),
            'workload_size': len(workload)
        }
    
    async def _apply_balancing_strategy(self, strategy: str, workload: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply the selected load balancing strategy."""
        if strategy == 'round_robin':
            return self._round_robin_distribution(workload)
        elif strategy == 'least_connections':
            return self._least_connections_distribution(workload, context)
        elif strategy == 'cpu_based':
            return self._cpu_based_distribution(workload, context)
        else:  # adaptive_weighted
            return self._adaptive_weighted_distribution(workload, context)
    
    def _round_robin_distribution(self, workload: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple round-robin distribution."""
        num_workers = 4  # Simulated worker count
        distribution = []
        for i, task in enumerate(workload):
            worker_id = i % num_workers
            distribution.append({**task, 'assigned_worker': worker_id})
        return distribution
    
    def _least_connections_distribution(self, workload: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Least connections distribution."""
        # Simulate worker connection counts
        worker_connections = [random.randint(0, 10) for _ in range(4)]
        distribution = []
        
        for task in workload:
            # Assign to worker with least connections
            worker_id = worker_connections.index(min(worker_connections))
            worker_connections[worker_id] += 1
            distribution.append({**task, 'assigned_worker': worker_id})
        
        return distribution
    
    def _cpu_based_distribution(self, workload: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """CPU usage-based distribution."""
        # Simulate worker CPU usage
        worker_cpu = [random.uniform(20, 80) for _ in range(4)]
        distribution = []
        
        for task in workload:
            # Assign to worker with lowest CPU usage
            worker_id = worker_cpu.index(min(worker_cpu))
            task_cpu_requirement = task.get('cpu_requirement', 10)
            worker_cpu[worker_id] += task_cpu_requirement
            distribution.append({**task, 'assigned_worker': worker_id})
        
        return distribution
    
    def _adaptive_weighted_distribution(self, workload: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Adaptive weighted distribution based on multiple factors."""
        # Simulate worker capacities
        workers = [
            {'cpu': random.uniform(20, 80), 'memory': random.uniform(30, 70), 'load': random.uniform(0, 5)}
            for _ in range(4)
        ]
        
        distribution = []
        for task in workload:
            # Calculate scores for each worker
            scores = []
            for i, worker in enumerate(workers):
                # Multi-factor scoring
                cpu_score = max(0, 1.0 - worker['cpu'] / 100.0)
                memory_score = max(0, 1.0 - worker['memory'] / 100.0)
                load_score = max(0, 1.0 - worker['load'] / 10.0)
                
                # Weighted combination
                total_score = cpu_score * 0.4 + memory_score * 0.3 + load_score * 0.3
                scores.append(total_score)
            
            # Assign to best worker
            worker_id = scores.index(max(scores))
            
            # Update worker state
            workers[worker_id]['cpu'] += task.get('cpu_requirement', 10)
            workers[worker_id]['memory'] += task.get('memory_requirement', 5)
            workers[worker_id]['load'] += 0.5
            
            distribution.append({**task, 'assigned_worker': worker_id, 'assignment_score': max(scores)})
        
        return distribution
    
    def _calculate_confidence(self, state: str, action: str) -> float:
        """Calculate confidence in the selected action."""
        if state not in self.q_table or action not in self.q_table[state]:
            return 0.0
        
        q_value = self.q_table[state][action]
        max_q = max(self.q_table[state].values()) if self.q_table[state] else 0
        
        # Confidence based on relative Q-value
        if max_q > 0:
            return min(1.0, q_value / max_q)
        return 0.5


class ResearchBenchmarkFramework:
    """
    Comprehensive benchmarking framework for algorithmic research
    with statistical significance testing and reproducibility validation.
    """
    
    def __init__(self):
        self.experiments = []
        self.baselines = {}
        self.results_cache = {}
        
    def register_baseline(self, name: str, algorithm: Any) -> None:
        """Register a baseline algorithm for comparison."""
        self.baselines[name] = algorithm
        logger.info(f"Registered baseline algorithm: {name}")
    
    async def run_comparative_study(
        self,
        novel_algorithm: Any,
        novel_name: str,
        baseline_name: str,
        test_scenarios: List[Dict[str, Any]],
        runs_per_scenario: int = 10
    ) -> ExperimentResult:
        """
        Run comprehensive comparative study with statistical significance testing.
        """
        experiment_id = f"exp_{int(time.time())}_{novel_name}_vs_{baseline_name}"
        logger.info(f"Starting comparative study: {experiment_id}")
        
        if baseline_name not in self.baselines:
            raise ValueError(f"Baseline {baseline_name} not registered")
        
        baseline_algorithm = self.baselines[baseline_name]
        all_metrics = []
        
        for scenario_idx, scenario in enumerate(test_scenarios):
            logger.info(f"Running scenario {scenario_idx + 1}/{len(test_scenarios)}")
            
            # Run multiple trials for statistical significance
            novel_results = []
            baseline_results = []
            
            for run in range(runs_per_scenario):
                # Test novel algorithm
                novel_start = time.time()
                try:
                    if hasattr(novel_algorithm, 'optimize_load_distribution'):
                        novel_result = await novel_algorithm.optimize_load_distribution(scenario['workload'])
                    else:
                        novel_result = await self._run_algorithm(novel_algorithm, scenario)
                    novel_duration = (time.time() - novel_start) * 1000
                    novel_results.append({
                        'duration_ms': novel_duration,
                        'result': novel_result,
                        'success': True
                    })
                except Exception as e:
                    novel_results.append({
                        'duration_ms': 999999,
                        'result': None,
                        'success': False,
                        'error': str(e)
                    })
                
                # Test baseline algorithm
                baseline_start = time.time()
                try:
                    if hasattr(baseline_algorithm, 'optimize_load_distribution'):
                        baseline_result = await baseline_algorithm.optimize_load_distribution(scenario['workload'])
                    else:
                        baseline_result = await self._run_algorithm(baseline_algorithm, scenario)
                    baseline_duration = (time.time() - baseline_start) * 1000
                    baseline_results.append({
                        'duration_ms': baseline_duration,
                        'result': baseline_result,
                        'success': True
                    })
                except Exception as e:
                    baseline_results.append({
                        'duration_ms': 999999,
                        'result': None,
                        'success': False,
                        'error': str(e)
                    })
            
            # Calculate scenario metrics
            scenario_metrics = self._calculate_scenario_metrics(
                novel_results, baseline_results, scenario_idx, novel_name, baseline_name
            )
            all_metrics.extend(scenario_metrics)
        
        # Calculate overall performance improvement
        novel_avg_duration = statistics.mean([m.value for m in all_metrics if m.algorithm_name == novel_name and m.metric_name == 'duration_ms'])
        baseline_avg_duration = statistics.mean([m.value for m in all_metrics if m.algorithm_name == baseline_name and m.metric_name == 'duration_ms'])
        
        improvement_percentage = ((baseline_avg_duration - novel_avg_duration) / baseline_avg_duration) * 100
        
        # Statistical significance testing
        novel_durations = [m.value for m in all_metrics if m.algorithm_name == novel_name and m.metric_name == 'duration_ms']
        baseline_durations = [m.value for m in all_metrics if m.algorithm_name == baseline_name and m.metric_name == 'duration_ms']
        
        statistical_significance = self._calculate_statistical_significance(novel_durations, baseline_durations)
        
        # Test reproducibility
        reproducible = await self._test_reproducibility(novel_algorithm, test_scenarios[0])
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            algorithm_name=novel_name,
            baseline_algorithm=baseline_name,
            performance_metrics=all_metrics,
            improvement_percentage=improvement_percentage,
            statistical_significance=statistical_significance,
            reproducible=reproducible,
            metadata={
                'test_scenarios': len(test_scenarios),
                'runs_per_scenario': runs_per_scenario,
                'total_runs': len(test_scenarios) * runs_per_scenario * 2
            }
        )
        
        self.experiments.append(result)
        logger.info(f"Comparative study completed: {improvement_percentage:.2f}% improvement, p-value: {statistical_significance:.4f}")
        
        return result
    
    async def _run_algorithm(self, algorithm: Any, scenario: Dict[str, Any]) -> Any:
        """Run algorithm with given scenario."""
        # Generic algorithm runner - adapt based on algorithm interface
        if hasattr(algorithm, 'measure_optimal_strategy'):
            return algorithm.measure_optimal_strategy('test_task', scenario)
        else:
            # Fallback for unknown algorithms
            await asyncio.sleep(0.1)  # Simulate work
            return {'result': 'completed', 'strategy': 'default'}
    
    def _calculate_scenario_metrics(
        self,
        novel_results: List[Dict[str, Any]],
        baseline_results: List[Dict[str, Any]],
        scenario_idx: int,
        novel_name: str,
        baseline_name: str
    ) -> List[ResearchMetric]:
        """Calculate metrics for a single scenario."""
        metrics = []
        timestamp = datetime.now()
        
        # Duration metrics
        novel_durations = [r['duration_ms'] for r in novel_results if r['success']]
        baseline_durations = [r['duration_ms'] for r in baseline_results if r['success']]
        
        if novel_durations:
            metrics.append(ResearchMetric(
                algorithm_name=novel_name,
                metric_name='duration_ms',
                value=statistics.mean(novel_durations),
                timestamp=timestamp,
                parameters={'scenario': scenario_idx}
            ))
        
        if baseline_durations:
            metrics.append(ResearchMetric(
                algorithm_name=baseline_name,
                metric_name='duration_ms',
                value=statistics.mean(baseline_durations),
                timestamp=timestamp,
                parameters={'scenario': scenario_idx}
            ))
        
        # Success rate metrics
        novel_success_rate = sum(1 for r in novel_results if r['success']) / len(novel_results)
        baseline_success_rate = sum(1 for r in baseline_results if r['success']) / len(baseline_results)
        
        metrics.extend([
            ResearchMetric(novel_name, 'success_rate', novel_success_rate, timestamp, {'scenario': scenario_idx}),
            ResearchMetric(baseline_name, 'success_rate', baseline_success_rate, timestamp, {'scenario': scenario_idx})
        ])
        
        return metrics
    
    def _calculate_statistical_significance(self, group1: List[float], group2: List[float]) -> float:
        """Calculate statistical significance using Welch's t-test approximation."""
        if len(group1) < 2 or len(group2) < 2:
            return 1.0  # No significance with insufficient data
        
        mean1, mean2 = statistics.mean(group1), statistics.mean(group2)
        
        try:
            var1 = statistics.variance(group1)
            var2 = statistics.variance(group2)
        except statistics.StatisticsError:
            return 1.0  # No variance, no significance
        
        n1, n2 = len(group1), len(group2)
        
        # Welch's t-test statistic
        pooled_se = math.sqrt(var1/n1 + var2/n2)
        if pooled_se == 0:
            return 1.0
        
        t_stat = abs(mean1 - mean2) / pooled_se
        
        # Approximate degrees of freedom (Welch-Satterthwaite equation)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # Simplified p-value approximation (for demonstration)
        # In real research, use proper statistical libraries
        if t_stat > 2.5:  # Rough threshold for p < 0.05
            return 0.01
        elif t_stat > 2.0:
            return 0.05
        elif t_stat > 1.5:
            return 0.1
        else:
            return 0.2
    
    async def _test_reproducibility(self, algorithm: Any, scenario: Dict[str, Any]) -> bool:
        """Test algorithm reproducibility across multiple runs."""
        results = []
        
        for _ in range(3):  # Run 3 times for reproducibility test
            try:
                if hasattr(algorithm, 'optimize_load_distribution'):
                    result = await algorithm.optimize_load_distribution(scenario['workload'])
                else:
                    result = await self._run_algorithm(algorithm, scenario)
                results.append(str(result))
            except Exception:
                return False
        
        # Check if results are consistent (simplified check)
        # In real research, would need more sophisticated consistency measures
        return len(set(results)) <= 2  # Allow minor variations
    
    def generate_research_report(self, experiment: ExperimentResult) -> Dict[str, Any]:
        """Generate comprehensive research report."""
        report = {
            'experiment_id': experiment.experiment_id,
            'title': f'Comparative Study: {experiment.algorithm_name} vs {experiment.baseline_algorithm}',
            'abstract': f'Novel algorithm shows {experiment.improvement_percentage:.2f}% improvement over baseline',
            'methodology': {
                'test_scenarios': experiment.metadata.get('test_scenarios', 0),
                'runs_per_scenario': experiment.metadata.get('runs_per_scenario', 0),
                'total_measurements': len(experiment.performance_metrics),
                'statistical_testing': 'Welch\'s t-test approximation'
            },
            'results': {
                'performance_improvement': experiment.improvement_percentage,
                'statistical_significance': experiment.statistical_significance,
                'reproducible': experiment.reproducible,
                'confidence_level': '95%' if experiment.statistical_significance < 0.05 else '90%' if experiment.statistical_significance < 0.1 else 'Low'
            },
            'metrics_summary': self._summarize_metrics(experiment.performance_metrics),
            'conclusion': self._generate_conclusion(experiment),
            'recommendations': self._generate_recommendations(experiment),
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def _summarize_metrics(self, metrics: List[ResearchMetric]) -> Dict[str, Any]:
        """Summarize performance metrics."""
        summary = defaultdict(lambda: defaultdict(list))
        
        for metric in metrics:
            summary[metric.algorithm_name][metric.metric_name].append(metric.value)
        
        result = {}
        for algorithm, algorithm_metrics in summary.items():
            result[algorithm] = {}
            for metric_name, values in algorithm_metrics.items():
                result[algorithm][metric_name] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return dict(result)
    
    def _generate_conclusion(self, experiment: ExperimentResult) -> str:
        """Generate research conclusion."""
        if experiment.statistical_significance < 0.05:
            significance_text = "statistically significant"
        elif experiment.statistical_significance < 0.1:
            significance_text = "marginally significant"
        else:
            significance_text = "not statistically significant"
        
        reproducibility_text = "reproducible" if experiment.reproducible else "not consistently reproducible"
        
        return (f"The novel {experiment.algorithm_name} algorithm demonstrates a "
                f"{experiment.improvement_percentage:.2f}% performance improvement over "
                f"{experiment.baseline_algorithm}, which is {significance_text} "
                f"(p = {experiment.statistical_significance:.4f}). "
                f"The results are {reproducibility_text} across multiple runs.")
    
    def _generate_recommendations(self, experiment: ExperimentResult) -> List[str]:
        """Generate research recommendations."""
        recommendations = []
        
        if experiment.improvement_percentage > 15:
            recommendations.append("Significant performance improvement warrants production deployment consideration")
        
        if experiment.statistical_significance < 0.05:
            recommendations.append("Strong statistical evidence supports algorithm effectiveness")
        elif experiment.statistical_significance > 0.1:
            recommendations.append("Additional testing needed to establish statistical significance")
        
        if not experiment.reproducible:
            recommendations.append("Investigate sources of non-determinism to improve reproducibility")
        
        if experiment.improvement_percentage < 5:
            recommendations.append("Consider cost-benefit analysis before deployment")
        
        return recommendations


# Global research framework instance
research_framework = ResearchBenchmarkFramework()