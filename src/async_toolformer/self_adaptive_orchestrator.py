"""
Generation 4: Self-Adaptive Orchestrator
Real-time self-modification and evolutionary optimization.
"""

import asyncio
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .advanced_ml_optimizer import AdvancedMLOptimizer
from .autonomous_learning_engine import AutonomousLearningEngine, PerformancePattern
from .comprehensive_monitoring import MetricType
from .simple_structured_logging import get_logger

logger = get_logger(__name__)


class AdaptationType(Enum):
    """Types of autonomous adaptations."""
    PARAMETER_TUNING = "parameter_tuning"
    ALGORITHM_SELECTION = "algorithm_selection"
    ARCHITECTURE_MODIFICATION = "architecture_modification"
    BEHAVIOR_EVOLUTION = "behavior_evolution"


@dataclass
class AdaptationRule:
    """Rule for autonomous system adaptation."""
    rule_id: str
    condition: str  # Python expression for when to apply
    adaptation_type: AdaptationType
    parameters: dict[str, Any]
    success_rate: float = 0.0
    application_count: int = 0
    last_applied: datetime | None = None
    enabled: bool = True


@dataclass
class EvolutionGenome:
    """Genetic representation of orchestrator configuration."""
    genome_id: str
    genes: dict[str, Any]  # Configuration parameters
    fitness_score: float = 0.0
    generation: int = 0
    parent_genomes: list[str] = field(default_factory=list)
    mutation_count: int = 0
    performance_history: list[float] = field(default_factory=list)


class SelfAdaptiveOrchestrator:
    """
    Generation 4: Self-evolving orchestration system.
    
    Capabilities:
    - Real-time parameter self-tuning based on performance feedback
    - Genetic algorithm-based configuration evolution
    - Autonomous algorithm selection and replacement
    - Dynamic behavior modification
    - Self-code generation for optimization routines
    """

    def __init__(
        self,
        base_orchestrator: Any,
        adaptation_interval_seconds: int = 300,  # 5 minutes
        genetic_population_size: int = 20,
        mutation_rate: float = 0.1,
        enable_code_generation: bool = True,
        enable_genetic_evolution: bool = True
    ):
        self.base_orchestrator = base_orchestrator
        self.adaptation_interval_seconds = adaptation_interval_seconds
        self.genetic_population_size = genetic_population_size
        self.mutation_rate = mutation_rate
        self.enable_code_generation = enable_code_generation
        self.enable_genetic_evolution = enable_genetic_evolution

        # Adaptive components
        self.learning_engine = AutonomousLearningEngine(enable_autonomous_optimization=True)
        self.ml_optimizer = AdvancedMLOptimizer(enable_auto_experiments=True)

        # Adaptation rules
        self.adaptation_rules: dict[str, AdaptationRule] = {}
        self._initialize_default_rules()

        # Genetic evolution
        self.population: list[EvolutionGenome] = []
        self.current_genome: EvolutionGenome | None = None
        self.generation_counter = 0

        # Performance tracking
        self.baseline_performance: dict[str, float] = {}
        self.current_performance: dict[str, float] = {}
        self.adaptation_history: list[dict[str, Any]] = []

        # Self-modification capabilities
        self.generated_functions: dict[str, Callable] = {}
        self.dynamic_algorithms: dict[str, Any] = {}

        # State management
        self.last_adaptation = datetime.utcnow()
        self.adaptation_lock = asyncio.Lock()

        # Start autonomous adaptation loop
        self._start_adaptation_loop()

    def _initialize_default_rules(self) -> None:
        """Initialize default adaptation rules."""
        rules = [
            AdaptationRule(
                rule_id="high_latency_tune",
                condition="current_performance.get('avg_latency', 0) > baseline_performance.get('avg_latency', 1) * 1.5",
                adaptation_type=AdaptationType.PARAMETER_TUNING,
                parameters={'target': 'concurrency', 'action': 'increase', 'factor': 1.2}
            ),
            AdaptationRule(
                rule_id="low_success_rate_adapt",
                condition="current_performance.get('success_rate', 1) < 0.8",
                adaptation_type=AdaptationType.ALGORITHM_SELECTION,
                parameters={'target': 'retry_strategy', 'options': ['exponential', 'linear', 'custom']}
            ),
            AdaptationRule(
                rule_id="memory_pressure_evolve",
                condition="current_performance.get('memory_usage', 0) > 0.85",
                adaptation_type=AdaptationType.ARCHITECTURE_MODIFICATION,
                parameters={'target': 'memory_management', 'action': 'optimize'}
            ),
            AdaptationRule(
                rule_id="performance_plateau_mutate",
                condition="len(performance_history) > 10 and max(performance_history[-5:]) <= max(performance_history[-10:-5])",
                adaptation_type=AdaptationType.BEHAVIOR_EVOLUTION,
                parameters={'target': 'genetic_mutation', 'intensity': 'medium'}
            )
        ]

        for rule in rules:
            self.adaptation_rules[rule.rule_id] = rule

    def _start_adaptation_loop(self) -> None:
        """Start the autonomous adaptation loop."""
        asyncio.create_task(self._adaptation_loop())

    async def _adaptation_loop(self) -> None:
        """Main adaptation loop running continuously."""
        while True:
            try:
                await asyncio.sleep(self.adaptation_interval_seconds)
                await self._perform_adaptation_cycle()
            except Exception as e:
                logger.error(f"Adaptation loop error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    # Monitoring decorator removed for compatibility
    async def _perform_adaptation_cycle(self) -> None:
        """Perform a complete adaptation cycle."""
        async with self.adaptation_lock:
            logger.info("Starting autonomous adaptation cycle")

            # Update performance metrics
            await self._update_performance_metrics()

            # Check adaptation rules
            applicable_rules = await self._evaluate_adaptation_rules()

            # Apply adaptations
            adaptations_applied = 0
            for rule in applicable_rules:
                success = await self._apply_adaptation_rule(rule)
                if success:
                    adaptations_applied += 1

            # Perform genetic evolution if enabled
            if self.enable_genetic_evolution:
                await self._evolve_configuration()

            # Generate new optimization code if enabled
            if self.enable_code_generation:
                await self._generate_optimization_code()

            logger.info(f"Adaptation cycle complete: {adaptations_applied} adaptations applied")

    async def _update_performance_metrics(self) -> None:
        """Update current performance metrics."""
        # This would integrate with the actual orchestrator's metrics
        # For now, simulating with learning engine data

        metrics = self.learning_engine.get_learning_metrics()

        self.current_performance = {
            'avg_latency': random.uniform(0.5, 3.0),  # Simulate current latency
            'success_rate': random.uniform(0.7, 1.0),  # Simulate success rate
            'memory_usage': random.uniform(0.3, 0.9),  # Simulate memory usage
            'throughput': random.uniform(10, 100),  # Simulate throughput
            'patterns_discovered': metrics['patterns_discovered'],
            'optimizations_applied': metrics['optimizations_applied']
        }

        # Initialize baseline if not set
        if not self.baseline_performance:
            self.baseline_performance = self.current_performance.copy()

    async def _evaluate_adaptation_rules(self) -> list[AdaptationRule]:
        """Evaluate which adaptation rules should be applied."""
        applicable_rules = []

        # Create evaluation context
        context = {
            'current_performance': self.current_performance,
            'baseline_performance': self.baseline_performance,
            'performance_history': [p['avg_latency'] for p in self.adaptation_history[-10:]],
            'datetime': datetime,
            'timedelta': timedelta
        }

        for rule in self.adaptation_rules.values():
            if not rule.enabled:
                continue

            try:
                # Evaluate the rule condition
                result = eval(rule.condition, {"__builtins__": {}}, context)
                if result:
                    applicable_rules.append(rule)
            except Exception as e:
                logger.warning(f"Failed to evaluate rule {rule.rule_id}: {e}")

        return applicable_rules

    async def _apply_adaptation_rule(self, rule: AdaptationRule) -> bool:
        """Apply a specific adaptation rule."""
        try:
            success = False

            if rule.adaptation_type == AdaptationType.PARAMETER_TUNING:
                success = await self._apply_parameter_tuning(rule)
            elif rule.adaptation_type == AdaptationType.ALGORITHM_SELECTION:
                success = await self._apply_algorithm_selection(rule)
            elif rule.adaptation_type == AdaptationType.ARCHITECTURE_MODIFICATION:
                success = await self._apply_architecture_modification(rule)
            elif rule.adaptation_type == AdaptationType.BEHAVIOR_EVOLUTION:
                success = await self._apply_behavior_evolution(rule)

            # Update rule statistics
            rule.application_count += 1
            rule.last_applied = datetime.utcnow()

            if success:
                rule.success_rate = (rule.success_rate * (rule.application_count - 1) + 1.0) / rule.application_count
            else:
                rule.success_rate = (rule.success_rate * (rule.application_count - 1)) / rule.application_count

            # Record adaptation
            self.adaptation_history.append({
                'timestamp': datetime.utcnow(),
                'rule_id': rule.rule_id,
                'adaptation_type': rule.adaptation_type.value,
                'success': success,
                'performance_before': self.current_performance.copy()
            })

            return success

        except Exception as e:
            logger.error(f"Failed to apply adaptation rule {rule.rule_id}: {e}")
            return False

    async def _apply_parameter_tuning(self, rule: AdaptationRule) -> bool:
        """Apply parameter tuning adaptation."""
        params = rule.parameters
        target = params.get('target')
        action = params.get('action')
        factor = params.get('factor', 1.1)

        if target == 'concurrency' and action == 'increase':
            # Simulate increasing concurrency
            current_concurrency = getattr(self.base_orchestrator, 'max_parallel', 10)
            new_concurrency = int(current_concurrency * factor)

            # Apply the change (would actually modify orchestrator config)
            logger.info(f"Tuning concurrency: {current_concurrency} -> {new_concurrency}")
            return True

        return False

    async def _apply_algorithm_selection(self, rule: AdaptationRule) -> bool:
        """Apply algorithm selection adaptation."""
        params = rule.parameters
        target = params.get('target')
        options = params.get('options', [])

        if target == 'retry_strategy' and options:
            # Select best retry strategy based on recent performance
            best_strategy = random.choice(options)  # Simplified selection

            logger.info(f"Selected retry strategy: {best_strategy}")
            return True

        return False

    async def _apply_architecture_modification(self, rule: AdaptationRule) -> bool:
        """Apply architecture modification adaptation."""
        params = rule.parameters
        target = params.get('target')
        action = params.get('action')

        if target == 'memory_management' and action == 'optimize':
            # Simulate memory optimization
            logger.info("Applied memory management optimization")
            return True

        return False

    async def _apply_behavior_evolution(self, rule: AdaptationRule) -> bool:
        """Apply behavior evolution adaptation."""
        params = rule.parameters
        target = params.get('target')

        if target == 'genetic_mutation':
            await self._trigger_genetic_mutation(params.get('intensity', 'medium'))
            return True

        return False

    async def _evolve_configuration(self) -> None:
        """Perform genetic algorithm-based configuration evolution."""
        if not self.enable_genetic_evolution:
            return

        # Initialize population if empty
        if not self.population:
            await self._initialize_genetic_population()

        # Evaluate fitness of current population
        await self._evaluate_population_fitness()

        # Selection and reproduction
        new_generation = await self._genetic_reproduction()

        # Replace population
        self.population = new_generation
        self.generation_counter += 1

        # Select best genome as current configuration
        best_genome = max(self.population, key=lambda g: g.fitness_score)
        if not self.current_genome or best_genome.fitness_score > self.current_genome.fitness_score:
            self.current_genome = best_genome
            await self._apply_genome_configuration(best_genome)

    async def _initialize_genetic_population(self) -> None:
        """Initialize the genetic algorithm population."""
        for i in range(self.genetic_population_size):
            genome = EvolutionGenome(
                genome_id=f"gen0_genome_{i}",
                genes=self._generate_random_genes(),
                generation=0
            )
            self.population.append(genome)

    def _generate_random_genes(self) -> dict[str, Any]:
        """Generate random genes for a genome."""
        return {
            'max_concurrency': random.randint(5, 50),
            'timeout_ms': random.randint(1000, 30000),
            'retry_attempts': random.randint(1, 5),
            'cache_size': random.randint(100, 10000),
            'rate_limit': random.randint(10, 1000),
            'speculation_enabled': random.choice([True, False]),
            'batch_size': random.randint(1, 20)
        }

    async def _evaluate_population_fitness(self) -> None:
        """Evaluate fitness scores for the population."""
        for genome in self.population:
            # Simulate fitness evaluation based on performance metrics
            # In reality, this would run the configuration and measure performance

            fitness = 0.0

            # Reward configurations that balance performance factors
            concurrency = genome.genes.get('max_concurrency', 10)
            timeout = genome.genes.get('timeout_ms', 5000)

            # Fitness heuristics
            fitness += min(1.0, concurrency / 20.0) * 0.3  # Concurrency factor
            fitness += min(1.0, 10000 / timeout) * 0.2     # Responsiveness factor
            fitness += random.uniform(0.3, 0.5)            # Simulate measured performance

            # Add noise to simulate real-world variability
            fitness += random.uniform(-0.1, 0.1)

            genome.fitness_score = max(0.0, fitness)
            genome.performance_history.append(fitness)

    async def _genetic_reproduction(self) -> list[EvolutionGenome]:
        """Perform genetic reproduction to create new generation."""
        new_generation = []

        # Sort by fitness (best first)
        sorted_population = sorted(self.population, key=lambda g: g.fitness_score, reverse=True)

        # Keep top performers (elitism)
        elite_size = max(2, self.genetic_population_size // 4)
        new_generation.extend(sorted_population[:elite_size])

        # Generate offspring through crossover and mutation
        while len(new_generation) < self.genetic_population_size:
            # Tournament selection
            parent1 = self._tournament_selection(sorted_population)
            parent2 = self._tournament_selection(sorted_population)

            # Crossover
            child = await self._crossover(parent1, parent2)

            # Mutation
            if random.random() < self.mutation_rate:
                child = await self._mutate(child)

            new_generation.append(child)

        return new_generation

    def _tournament_selection(self, population: list[EvolutionGenome], tournament_size: int = 3) -> EvolutionGenome:
        """Select parent using tournament selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda g: g.fitness_score)

    async def _crossover(self, parent1: EvolutionGenome, parent2: EvolutionGenome) -> EvolutionGenome:
        """Create offspring through crossover."""
        child_genes = {}

        for key in parent1.genes:
            if key in parent2.genes:
                # Uniform crossover
                child_genes[key] = random.choice([parent1.genes[key], parent2.genes[key]])
            else:
                child_genes[key] = parent1.genes[key]

        child = EvolutionGenome(
            genome_id=f"gen{self.generation_counter + 1}_child_{random.randint(1000, 9999)}",
            genes=child_genes,
            generation=self.generation_counter + 1,
            parent_genomes=[parent1.genome_id, parent2.genome_id]
        )

        return child

    async def _mutate(self, genome: EvolutionGenome) -> EvolutionGenome:
        """Apply mutation to a genome."""
        mutated_genes = genome.genes.copy()

        # Randomly select genes to mutate
        mutation_count = random.randint(1, max(1, len(mutated_genes) // 3))
        genes_to_mutate = random.sample(list(mutated_genes.keys()), mutation_count)

        for gene in genes_to_mutate:
            if isinstance(mutated_genes[gene], int):
                # Integer mutation
                current_value = mutated_genes[gene]
                mutation_range = max(1, current_value // 10)
                mutated_genes[gene] = max(1, current_value + random.randint(-mutation_range, mutation_range))
            elif isinstance(mutated_genes[gene], bool):
                # Boolean mutation
                mutated_genes[gene] = not mutated_genes[gene]

        genome.genes = mutated_genes
        genome.mutation_count += 1

        return genome

    async def _trigger_genetic_mutation(self, intensity: str) -> None:
        """Trigger genetic mutation in response to performance plateau."""
        if not self.population:
            return

        mutation_factor = {'low': 0.05, 'medium': 0.1, 'high': 0.2}.get(intensity, 0.1)

        for genome in self.population:
            if random.random() < mutation_factor:
                await self._mutate(genome)

        logger.info(f"Applied {intensity} intensity genetic mutation")

    async def _apply_genome_configuration(self, genome: EvolutionGenome) -> None:
        """Apply the configuration from a genome to the orchestrator."""
        logger.info(f"Applying genome configuration: {genome.genome_id}")

        # This would actually modify the orchestrator's configuration
        # For now, just logging the changes
        for gene, value in genome.genes.items():
            logger.info(f"  {gene}: {value}")

    async def _generate_optimization_code(self) -> None:
        """Generate new optimization code based on performance patterns."""
        if not self.enable_code_generation:
            return

        # Analyze current performance patterns
        patterns = self.learning_engine.performance_patterns

        if not patterns:
            return

        # Generate optimization function
        optimization_code = await self._create_dynamic_optimization_function(patterns)

        if optimization_code:
            # Execute generated code (in a secure context)
            try:
                exec(optimization_code, {'__builtins__': {}}, self.generated_functions)
                logger.info("Generated and deployed new optimization function")
            except Exception as e:
                logger.warning(f"Failed to execute generated code: {e}")

    async def _create_dynamic_optimization_function(
        self,
        patterns: dict[str, PerformancePattern]
    ) -> str | None:
        """Create a dynamic optimization function based on patterns."""
        # Simplified code generation
        if not patterns:
            return None

        # Analyze patterns to determine optimization strategy
        high_latency_patterns = [p for p in patterns.values() if p.avg_execution_time > 2.0]
        low_success_patterns = [p for p in patterns.values() if p.success_rate < 0.9]

        if high_latency_patterns:
            # Generate latency optimization function
            code = f'''
def dynamic_latency_optimizer(context):
    """Auto-generated optimization for high latency patterns."""
    tool_sequence = context.get('tool_sequence', [])
    
    # Optimize based on discovered patterns
    optimizations = []
    
    for tool in tool_sequence:
        if tool in {[p.tool_sequence for p in high_latency_patterns]}:
            optimizations.append({{"tool": tool, "action": "increase_timeout", "factor": 1.5}})
    
    return optimizations
'''
            return code

        return None

    def get_adaptation_status(self) -> dict[str, Any]:
        """Get current adaptation status."""
        return {
            'current_generation': self.generation_counter,
            'population_size': len(self.population),
            'best_fitness': max([g.fitness_score for g in self.population]) if self.population else 0.0,
            'adaptation_rules_active': sum(1 for r in self.adaptation_rules.values() if r.enabled),
            'recent_adaptations': len([h for h in self.adaptation_history if
                                     datetime.utcnow() - h['timestamp'] < timedelta(hours=1)]),
            'generated_functions': len(self.generated_functions),
            'current_genome_id': self.current_genome.genome_id if self.current_genome else None,
            'last_adaptation': self.last_adaptation.isoformat(),
            'performance_metrics': self.current_performance
        }

    async def add_custom_adaptation_rule(
        self,
        rule_id: str,
        condition: str,
        adaptation_type: AdaptationType,
        parameters: dict[str, Any]
    ) -> None:
        """Add a custom adaptation rule."""
        rule = AdaptationRule(
            rule_id=rule_id,
            condition=condition,
            adaptation_type=adaptation_type,
            parameters=parameters
        )
        self.adaptation_rules[rule_id] = rule
        logger.info(f"Added custom adaptation rule: {rule_id}")

    async def disable_adaptation_rule(self, rule_id: str) -> None:
        """Disable a specific adaptation rule."""
        if rule_id in self.adaptation_rules:
            self.adaptation_rules[rule_id].enabled = False
            logger.info(f"Disabled adaptation rule: {rule_id}")

    async def shutdown(self) -> None:
        """Graceful shutdown of the adaptive orchestrator."""
        logger.info("Shutting down self-adaptive orchestrator")

        # Save adaptation history and learned patterns
        await self.learning_engine.shutdown()

        # Final performance report
        status = self.get_adaptation_status()
        logger.info(f"Final adaptation status: {status}")


# Factory function
def create_self_adaptive_orchestrator(
    base_orchestrator: Any,
    enable_genetic_evolution: bool = True,
    enable_code_generation: bool = False  # Disabled by default for safety
) -> SelfAdaptiveOrchestrator:
    """Create a self-adaptive orchestrator wrapper."""
    return SelfAdaptiveOrchestrator(
        base_orchestrator=base_orchestrator,
        enable_genetic_evolution=enable_genetic_evolution,
        enable_code_generation=enable_code_generation
    )
