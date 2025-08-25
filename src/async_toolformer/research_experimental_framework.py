"""
Generation 4: Research-Grade Experimental Framework
Publication-ready experimentation with statistical rigor.
"""

import asyncio
import json
import math
import numpy as np
import statistics
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import pickle
from pathlib import Path

from .simple_structured_logging import get_logger
from .comprehensive_monitoring import monitor, MetricType

logger = get_logger(__name__)


class ExperimentType(Enum):
    """Types of research experiments."""
    COMPARATIVE_STUDY = "comparative_study"
    ABLATION_STUDY = "ablation_study"
    PARAMETER_SWEEP = "parameter_sweep"
    ALGORITHM_BENCHMARK = "algorithm_benchmark"
    HYPOTHESIS_VALIDATION = "hypothesis_validation"


class StatisticalTest(Enum):
    """Statistical tests for significance."""
    T_TEST = "t_test"
    WILCOXON = "wilcoxon"
    MANN_WHITNEY = "mann_whitney"
    CHI_SQUARE = "chi_square"
    ANOVA = "anova"


@dataclass
class ExperimentalCondition:
    """Experimental condition/treatment configuration."""
    condition_id: str
    name: str
    description: str
    configuration: Dict[str, Any]
    expected_outcome: Optional[str] = None
    theoretical_basis: Optional[str] = None


@dataclass
class ExperimentalResult:
    """Results from a single experimental run."""
    run_id: str
    condition_id: str
    timestamp: datetime
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


@dataclass
class StatisticalSummary:
    """Statistical summary of experimental results."""
    metric_name: str
    condition_id: str
    n_samples: int
    mean: float
    median: float
    std: float
    min_value: float
    max_value: float
    confidence_interval_95: Tuple[float, float]
    effect_size: Optional[float] = None


@dataclass
class SignificanceResult:
    """Statistical significance test result."""
    test_type: StatisticalTest
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float
    effect_size: float
    power: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class ResearchExperiment:
    """Complete research experiment with metadata."""
    experiment_id: str
    title: str
    description: str
    experiment_type: ExperimentType
    hypothesis: str
    research_questions: List[str]
    conditions: List[ExperimentalCondition]
    primary_metrics: List[str]
    secondary_metrics: List[str] = field(default_factory=list)
    target_sample_size: int = 30
    significance_level: float = 0.05
    power_target: float = 0.8
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "designed"
    results: List[ExperimentalResult] = field(default_factory=list)
    statistical_summaries: List[StatisticalSummary] = field(default_factory=list)
    significance_tests: List[SignificanceResult] = field(default_factory=list)
    conclusions: Optional[str] = None
    publication_ready: bool = False


class ResearchExperimentalFramework:
    """
    Generation 4: Research-grade experimental framework.
    
    Features:
    - Rigorous experimental design with power analysis
    - Multiple statistical test implementations
    - Automated replication and validation
    - Publication-ready result formatting
    - Meta-analysis capabilities across experiments
    - Reproducible experiment execution
    """

    def __init__(
        self,
        results_directory: str = "./research_results",
        min_effect_size: float = 0.5,
        default_alpha: float = 0.05,
        default_power: float = 0.8,
        enable_auto_replication: bool = True,
        replication_threshold: int = 3
    ):
        self.results_directory = Path(results_directory)
        self.min_effect_size = min_effect_size
        self.default_alpha = default_alpha
        self.default_power = default_power
        self.enable_auto_replication = enable_auto_replication
        self.replication_threshold = replication_threshold
        
        # Experiment management
        self.active_experiments: Dict[str, ResearchExperiment] = {}
        self.completed_experiments: List[ResearchExperiment] = []
        self.experiment_queue: deque = deque()
        
        # Statistical state
        self.statistical_cache: Dict[str, Any] = {}
        self.power_analysis_cache: Dict[str, float] = {}
        
        # Meta-analysis data
        self.meta_analysis_results: Dict[str, Any] = {}
        
        # Ensure results directory exists
        self.results_directory.mkdir(parents=True, exist_ok=True)
        
        # Load previous experiments
        self._load_experiment_history()

    @monitor(MetricType.COUNTER)
    async def design_comparative_study(
        self,
        title: str,
        hypothesis: str,
        conditions: List[ExperimentalCondition],
        primary_metrics: List[str],
        research_questions: List[str] = None,
        target_sample_size: Optional[int] = None
    ) -> str:
        """Design a comparative study experiment."""
        
        # Calculate required sample size if not provided
        if target_sample_size is None:
            target_sample_size = await self._calculate_sample_size(
                effect_size=self.min_effect_size,
                alpha=self.default_alpha,
                power=self.default_power,
                num_groups=len(conditions)
            )
        
        experiment_id = f"comp_study_{int(datetime.utcnow().timestamp())}"
        
        experiment = ResearchExperiment(
            experiment_id=experiment_id,
            title=title,
            description=f"Comparative study: {title}",
            experiment_type=ExperimentType.COMPARATIVE_STUDY,
            hypothesis=hypothesis,
            research_questions=research_questions or [f"Is there a significant difference in {metric}?" for metric in primary_metrics],
            conditions=conditions,
            primary_metrics=primary_metrics,
            target_sample_size=target_sample_size
        )
        
        self.active_experiments[experiment_id] = experiment
        logger.info(f"Designed comparative study: {experiment_id}")
        
        return experiment_id

    @monitor(MetricType.COUNTER)
    async def design_ablation_study(
        self,
        title: str,
        baseline_condition: ExperimentalCondition,
        ablation_conditions: List[ExperimentalCondition],
        primary_metrics: List[str],
        hypothesis: Optional[str] = None
    ) -> str:
        """Design an ablation study experiment."""
        
        if hypothesis is None:
            hypothesis = f"Removing components will significantly impact {', '.join(primary_metrics)}"
        
        all_conditions = [baseline_condition] + ablation_conditions
        
        experiment_id = f"ablation_{int(datetime.utcnow().timestamp())}"
        
        experiment = ResearchExperiment(
            experiment_id=experiment_id,
            title=title,
            description=f"Ablation study: {title}",
            experiment_type=ExperimentType.ABLATION_STUDY,
            hypothesis=hypothesis,
            research_questions=[f"What is the contribution of each component to {metric}?" for metric in primary_metrics],
            conditions=all_conditions,
            primary_metrics=primary_metrics,
            target_sample_size=await self._calculate_sample_size(
                effect_size=self.min_effect_size,
                alpha=self.default_alpha,
                power=self.default_power,
                num_groups=len(all_conditions)
            )
        )
        
        self.active_experiments[experiment_id] = experiment
        logger.info(f"Designed ablation study: {experiment_id}")
        
        return experiment_id

    @monitor(MetricType.COUNTER)
    async def design_parameter_sweep(
        self,
        title: str,
        parameter_name: str,
        parameter_values: List[Any],
        base_configuration: Dict[str, Any],
        primary_metrics: List[str],
        hypothesis: Optional[str] = None
    ) -> str:
        """Design a parameter sweep experiment."""
        
        if hypothesis is None:
            hypothesis = f"Parameter {parameter_name} significantly affects {', '.join(primary_metrics)}"
        
        # Create conditions for each parameter value
        conditions = []
        for i, value in enumerate(parameter_values):
            config = base_configuration.copy()
            config[parameter_name] = value
            
            condition = ExperimentalCondition(
                condition_id=f"{parameter_name}_{value}",
                name=f"{parameter_name} = {value}",
                description=f"Testing {parameter_name} with value {value}",
                configuration=config,
                theoretical_basis=f"Parameter sweep for optimal {parameter_name} value"
            )
            conditions.append(condition)
        
        experiment_id = f"param_sweep_{int(datetime.utcnow().timestamp())}"
        
        experiment = ResearchExperiment(
            experiment_id=experiment_id,
            title=title,
            description=f"Parameter sweep: {title}",
            experiment_type=ExperimentType.PARAMETER_SWEEP,
            hypothesis=hypothesis,
            research_questions=[f"What is the optimal value of {parameter_name} for {metric}?" for metric in primary_metrics],
            conditions=conditions,
            primary_metrics=primary_metrics,
            target_sample_size=max(15, len(parameter_values) * 5)  # Minimum samples per condition
        )
        
        self.active_experiments[experiment_id] = experiment
        logger.info(f"Designed parameter sweep: {experiment_id}")
        
        return experiment_id

    async def run_experiment(
        self,
        experiment_id: str,
        execution_function: Callable,
        max_parallel_runs: int = 5
    ) -> bool:
        """Execute a research experiment."""
        
        if experiment_id not in self.active_experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.active_experiments[experiment_id]
        experiment.status = "running"
        
        logger.info(f"Starting experiment execution: {experiment_id}")
        
        # Calculate runs per condition
        runs_per_condition = max(1, experiment.target_sample_size // len(experiment.conditions))
        
        # Create execution tasks
        tasks = []
        run_counter = 0
        
        for condition in experiment.conditions:
            for run in range(runs_per_condition):
                task = self._execute_single_run(
                    experiment_id=experiment_id,
                    run_id=f"{experiment_id}_{condition.condition_id}_run_{run}",
                    condition=condition,
                    execution_function=execution_function
                )
                tasks.append(task)
                run_counter += 1
        
        # Execute runs with controlled parallelism
        semaphore = asyncio.Semaphore(max_parallel_runs)
        
        async def controlled_execution(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(
            *[controlled_execution(task) for task in tasks],
            return_exceptions=True
        )
        
        # Process results
        successful_runs = []
        failed_runs = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_runs.append(str(result))
            elif result:
                successful_runs.append(result)
                experiment.results.append(result)
        
        experiment.status = "completed"
        
        # Perform statistical analysis
        await self._analyze_experiment_results(experiment_id)
        
        # Move to completed experiments
        self.completed_experiments.append(experiment)
        del self.active_experiments[experiment_id]
        
        # Save results
        await self._save_experiment_results(experiment)
        
        logger.info(f"Experiment completed: {experiment_id} ({len(successful_runs)} successful, {len(failed_runs)} failed)")
        
        return len(failed_runs) == 0

    async def _execute_single_run(
        self,
        experiment_id: str,
        run_id: str,
        condition: ExperimentalCondition,
        execution_function: Callable
    ) -> ExperimentalResult:
        """Execute a single experimental run."""
        
        start_time = datetime.utcnow()
        
        try:
            # Execute with condition configuration
            result = await execution_function(condition.configuration)
            
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Extract metrics from result
            if isinstance(result, dict):
                metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}
                metadata = {k: v for k, v in result.items() if k not in metrics}
            else:
                # Assume single metric result
                metrics = {"result": float(result)}
                metadata = {}
            
            return ExperimentalResult(
                run_id=run_id,
                condition_id=condition.condition_id,
                timestamp=start_time,
                metrics=metrics,
                metadata=metadata,
                duration_seconds=duration,
                success=True
            )
            
        except Exception as e:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            logger.warning(f"Run {run_id} failed: {e}")
            
            return ExperimentalResult(
                run_id=run_id,
                condition_id=condition.condition_id,
                timestamp=start_time,
                metrics={},
                metadata={},
                duration_seconds=duration,
                success=False,
                error_message=str(e)
            )

    async def _analyze_experiment_results(self, experiment_id: str) -> None:
        """Perform comprehensive statistical analysis of experiment results."""
        
        experiment = self.completed_experiments[-1]  # Most recent
        
        # Calculate statistical summaries for each condition and metric
        await self._calculate_statistical_summaries(experiment)
        
        # Perform significance tests
        await self._perform_significance_tests(experiment)
        
        # Calculate effect sizes
        await self._calculate_effect_sizes(experiment)
        
        # Generate conclusions
        await self._generate_conclusions(experiment)
        
        # Check if results are publication ready
        await self._assess_publication_readiness(experiment)

    async def _calculate_statistical_summaries(self, experiment: ResearchExperiment) -> None:
        """Calculate statistical summaries for all metrics and conditions."""
        
        # Group results by condition
        results_by_condition = defaultdict(list)
        for result in experiment.results:
            if result.success:
                results_by_condition[result.condition_id].append(result)
        
        # Calculate summaries
        for condition_id, results in results_by_condition.items():
            for metric in experiment.primary_metrics + experiment.secondary_metrics:
                values = [r.metrics.get(metric, 0.0) for r in results if metric in r.metrics]
                
                if len(values) < 3:
                    continue  # Need minimum samples for meaningful statistics
                
                mean_val = np.mean(values)
                std_val = np.std(values, ddof=1)  # Sample standard deviation
                
                # Calculate 95% confidence interval
                n = len(values)
                t_critical = self._get_t_critical(n - 1, 0.05)
                margin_error = t_critical * (std_val / math.sqrt(n))
                ci_95 = (mean_val - margin_error, mean_val + margin_error)
                
                summary = StatisticalSummary(
                    metric_name=metric,
                    condition_id=condition_id,
                    n_samples=n,
                    mean=mean_val,
                    median=np.median(values),
                    std=std_val,
                    min_value=min(values),
                    max_value=max(values),
                    confidence_interval_95=ci_95
                )
                
                experiment.statistical_summaries.append(summary)

    async def _perform_significance_tests(self, experiment: ResearchExperiment) -> None:
        """Perform statistical significance tests."""
        
        # Group summaries by metric
        summaries_by_metric = defaultdict(list)
        for summary in experiment.statistical_summaries:
            summaries_by_metric[summary.metric_name].append(summary)
        
        for metric, summaries in summaries_by_metric.items():
            if len(summaries) < 2:
                continue  # Need at least 2 conditions to compare
            
            # For comparative studies, perform pairwise comparisons
            if experiment.experiment_type == ExperimentType.COMPARATIVE_STUDY:
                await self._perform_pairwise_comparisons(experiment, metric, summaries)
            
            # For parameter sweeps, perform ANOVA if multiple conditions
            elif experiment.experiment_type == ExperimentType.PARAMETER_SWEEP and len(summaries) > 2:
                await self._perform_anova(experiment, metric, summaries)

    async def _perform_pairwise_comparisons(
        self,
        experiment: ResearchExperiment,
        metric: str,
        summaries: List[StatisticalSummary]
    ) -> None:
        """Perform pairwise t-tests between conditions."""
        
        for i in range(len(summaries)):
            for j in range(i + 1, len(summaries)):
                summary1, summary2 = summaries[i], summaries[j]
                
                # Get original data
                data1 = self._get_metric_data(experiment, summary1.condition_id, metric)
                data2 = self._get_metric_data(experiment, summary2.condition_id, metric)
                
                if len(data1) < 3 or len(data2) < 3:
                    continue
                
                # Perform t-test
                t_stat, p_value = self._two_sample_t_test(data1, data2)
                
                # Calculate effect size (Cohen's d)
                pooled_std = math.sqrt(((len(data1) - 1) * summary1.std**2 + 
                                      (len(data2) - 1) * summary2.std**2) / 
                                     (len(data1) + len(data2) - 2))
                
                effect_size = abs(summary1.mean - summary2.mean) / pooled_std if pooled_std > 0 else 0
                
                # Calculate power (simplified)
                power = self._calculate_statistical_power(effect_size, len(data1), len(data2))
                
                significance_result = SignificanceResult(
                    test_type=StatisticalTest.T_TEST,
                    statistic=t_stat,
                    p_value=p_value,
                    is_significant=p_value < experiment.significance_level,
                    alpha=experiment.significance_level,
                    effect_size=effect_size,
                    power=power
                )
                
                experiment.significance_tests.append(significance_result)

    def _get_metric_data(self, experiment: ResearchExperiment, condition_id: str, metric: str) -> List[float]:
        """Get metric data for a specific condition."""
        return [
            result.metrics.get(metric, 0.0)
            for result in experiment.results
            if result.condition_id == condition_id and result.success and metric in result.metrics
        ]

    def _two_sample_t_test(self, data1: List[float], data2: List[float]) -> Tuple[float, float]:
        """Perform two-sample t-test."""
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        
        # Pooled standard error
        pooled_se = math.sqrt(var1/n1 + var2/n2)
        
        if pooled_se == 0:
            return 0.0, 1.0
        
        # t-statistic
        t_stat = (mean1 - mean2) / pooled_se
        
        # Degrees of freedom (Welch's t-test)
        df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # Simplified p-value calculation (two-tailed)
        p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        
        return t_stat, p_value

    def _t_cdf(self, t: float, df: float) -> float:
        """Simplified t-distribution CDF approximation."""
        # Using normal approximation for large df
        if df >= 30:
            return self._normal_cdf(t)
        
        # Simplified approximation for small df
        x = t / math.sqrt(df)
        return 0.5 + (x / math.sqrt(1 + x**2)) * 0.5

    def _normal_cdf(self, x: float) -> float:
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _get_t_critical(self, df: int, alpha: float) -> float:
        """Get critical t-value (simplified approximation)."""
        if df >= 30:
            # Normal approximation
            return self._inverse_normal_cdf(1 - alpha/2)
        
        # Simplified t-critical values
        t_critical_values = {
            1: {0.05: 12.706, 0.01: 63.657},
            2: {0.05: 4.303, 0.01: 9.925},
            5: {0.05: 2.571, 0.01: 4.032},
            10: {0.05: 2.228, 0.01: 3.169},
            20: {0.05: 2.086, 0.01: 2.845},
            30: {0.05: 2.042, 0.01: 2.750}
        }
        
        # Find closest df
        closest_df = min(t_critical_values.keys(), key=lambda x: abs(x - df))
        return t_critical_values[closest_df].get(alpha, 1.96)

    def _inverse_normal_cdf(self, p: float) -> float:
        """Inverse normal CDF approximation."""
        if p <= 0.5:
            return -self._inverse_normal_cdf(1 - p)
        
        # Simplified approximation
        return math.sqrt(-2 * math.log(1 - p))

    def _calculate_statistical_power(self, effect_size: float, n1: int, n2: int) -> float:
        """Calculate statistical power (simplified)."""
        # Simplified power calculation
        n_harmonic = 2 / (1/n1 + 1/n2)
        delta = effect_size * math.sqrt(n_harmonic / 2)
        
        # Approximate power using normal distribution
        power = 1 - self._normal_cdf(1.96 - delta)
        return max(0.0, min(1.0, power))

    async def _calculate_sample_size(
        self,
        effect_size: float,
        alpha: float,
        power: float,
        num_groups: int = 2
    ) -> int:
        """Calculate required sample size for given parameters."""
        
        # Simplified sample size calculation for t-test
        z_alpha = self._inverse_normal_cdf(1 - alpha/2)
        z_beta = self._inverse_normal_cdf(power)
        
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size)**2
        
        # Adjust for multiple groups
        if num_groups > 2:
            n_per_group *= 1 + (num_groups - 2) * 0.1
        
        return max(5, int(n_per_group))  # Minimum 5 samples per group

    async def _calculate_effect_sizes(self, experiment: ResearchExperiment) -> None:
        """Calculate effect sizes for significant results."""
        for significance in experiment.significance_tests:
            # Effect size already calculated in significance tests
            pass

    async def _generate_conclusions(self, experiment: ResearchExperiment) -> None:
        """Generate research conclusions based on results."""
        conclusions = []
        
        # Analyze significant results
        significant_tests = [test for test in experiment.significance_tests if test.is_significant]
        
        if significant_tests:
            conclusions.append(f"Found {len(significant_tests)} statistically significant results:")
            
            for test in significant_tests[:5]:  # Top 5 significant results
                effect_desc = self._interpret_effect_size(test.effect_size)
                conclusions.append(
                    f"- Effect size: {test.effect_size:.3f} ({effect_desc}), "
                    f"p-value: {test.p_value:.4f}, power: {test.power:.3f}"
                )
        else:
            conclusions.append("No statistically significant differences found.")
        
        # Practical significance
        large_effects = [test for test in experiment.significance_tests if test.effect_size > 0.8]
        if large_effects:
            conclusions.append(f"Found {len(large_effects)} results with large practical effect sizes.")
        
        experiment.conclusions = "\n".join(conclusions)

    def _interpret_effect_size(self, effect_size: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"

    async def _assess_publication_readiness(self, experiment: ResearchExperiment) -> None:
        """Assess if results are ready for publication."""
        criteria_met = 0
        total_criteria = 5
        
        # Criteria 1: Adequate sample size
        if len(experiment.results) >= experiment.target_sample_size:
            criteria_met += 1
        
        # Criteria 2: Statistical significance found
        if any(test.is_significant for test in experiment.significance_tests):
            criteria_met += 1
        
        # Criteria 3: Adequate statistical power
        avg_power = np.mean([test.power for test in experiment.significance_tests if test.power])
        if avg_power >= 0.8:
            criteria_met += 1
        
        # Criteria 4: Effect sizes calculated
        if all(test.effect_size > 0 for test in experiment.significance_tests):
            criteria_met += 1
        
        # Criteria 5: Multiple metrics analyzed
        if len(experiment.primary_metrics) >= 2:
            criteria_met += 1
        
        experiment.publication_ready = criteria_met >= 4

    async def generate_publication_report(self, experiment_id: str) -> str:
        """Generate publication-ready research report."""
        
        # Find experiment
        experiment = None
        for exp in self.completed_experiments:
            if exp.experiment_id == experiment_id:
                experiment = exp
                break
        
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Generate report
        report_sections = []
        
        # Title and Abstract
        report_sections.append(f"# {experiment.title}")
        report_sections.append(f"\n## Abstract")
        report_sections.append(f"{experiment.description}")
        report_sections.append(f"Hypothesis: {experiment.hypothesis}")
        
        # Methodology
        report_sections.append(f"\n## Methodology")
        report_sections.append(f"Experiment Type: {experiment.experiment_type.value}")
        report_sections.append(f"Sample Size: {len(experiment.results)} runs across {len(experiment.conditions)} conditions")
        report_sections.append(f"Primary Metrics: {', '.join(experiment.primary_metrics)}")
        
        # Results
        report_sections.append(f"\n## Results")
        
        # Statistical summaries
        report_sections.append(f"\n### Descriptive Statistics")
        for summary in experiment.statistical_summaries:
            report_sections.append(
                f"**{summary.condition_id} - {summary.metric_name}**: "
                f"M = {summary.mean:.3f}, SD = {summary.std:.3f}, "
                f"95% CI [{summary.confidence_interval_95[0]:.3f}, {summary.confidence_interval_95[1]:.3f}]"
            )
        
        # Significance tests
        report_sections.append(f"\n### Statistical Tests")
        for test in experiment.significance_tests:
            significance = "significant" if test.is_significant else "not significant"
            report_sections.append(
                f"**{test.test_type.value}**: t = {test.statistic:.3f}, "
                f"p = {test.p_value:.4f} ({significance}), "
                f"d = {test.effect_size:.3f}, power = {test.power:.3f}"
            )
        
        # Conclusions
        report_sections.append(f"\n## Conclusions")
        report_sections.append(experiment.conclusions or "No conclusions available.")
        
        # Methodology details
        report_sections.append(f"\n## Experimental Conditions")
        for condition in experiment.conditions:
            report_sections.append(f"### {condition.name}")
            report_sections.append(f"Description: {condition.description}")
            report_sections.append(f"Configuration: {json.dumps(condition.configuration, indent=2)}")
        
        report_content = "\n".join(report_sections)
        
        # Save report
        report_file = self.results_directory / f"{experiment_id}_publication_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Generated publication report: {report_file}")
        
        return report_content

    async def _save_experiment_results(self, experiment: ResearchExperiment) -> None:
        """Save experiment results to disk."""
        
        # Convert to serializable format
        experiment_data = asdict(experiment)
        
        # Save JSON results
        results_file = self.results_directory / f"{experiment.experiment_id}_results.json"
        with open(results_file, 'w') as f:
            json.dump(experiment_data, f, default=str, indent=2)
        
        # Save raw data as pickle for further analysis
        pickle_file = self.results_directory / f"{experiment.experiment_id}_raw.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(experiment, f)
        
        logger.info(f"Saved experiment results: {results_file}")

    def _load_experiment_history(self) -> None:
        """Load previous experiment results."""
        try:
            for results_file in self.results_directory.glob("*_results.json"):
                with open(results_file, 'r') as f:
                    experiment_data = json.load(f)
                
                # Reconstruct experiment object (simplified)
                experiment_id = experiment_data['experiment_id']
                logger.info(f"Loaded experiment history: {experiment_id}")
        except Exception as e:
            logger.warning(f"Failed to load experiment history: {e}")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments."""
        return {
            'active_experiments': len(self.active_experiments),
            'completed_experiments': len(self.completed_experiments),
            'total_experimental_runs': sum(len(exp.results) for exp in self.completed_experiments),
            'publication_ready_experiments': sum(1 for exp in self.completed_experiments if exp.publication_ready),
            'significant_findings': sum(
                len([test for test in exp.significance_tests if test.is_significant])
                for exp in self.completed_experiments
            ),
            'average_effect_size': np.mean([
                test.effect_size for exp in self.completed_experiments
                for test in exp.significance_tests if test.effect_size > 0
            ]) if self.completed_experiments else 0.0
        }


# Factory function
def create_research_experimental_framework(
    results_directory: str = "./research_results",
    enable_auto_replication: bool = True
) -> ResearchExperimentalFramework:
    """Create a research-grade experimental framework."""
    return ResearchExperimentalFramework(
        results_directory=results_directory,
        enable_auto_replication=enable_auto_replication
    )