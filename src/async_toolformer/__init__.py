"""Async Toolformer Orchestrator - Parallel tool execution for LLMs."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "async-tools@yourdomain.com"

from .advanced_ml_optimizer import (
    AdvancedMLOptimizer,
    MLPrediction,
    OptimizationExperiment,
    create_advanced_ml_optimizer,
)

# Generation 4: Autonomous AI Components
from .autonomous_learning_engine import (
    AutonomousLearningEngine,
    OptimizationRecommendation,
    PerformancePattern,
    create_autonomous_learning_engine,
)

# Generation 5: Quantum Leap Components
from .quantum_quality_gates import (
    QuantumQualityGateOrchestrator,
    QuantumQualityLevel,
    ValidationDimension,
    QuantumValidationResult,
    create_quantum_quality_gate_orchestrator,
)
from .autonomous_intelligence_engine import (
    AutonomousIntelligenceEngine,
    IntelligenceLevel,
    DecisionDomain,
    AutonomousDecision,
    create_autonomous_intelligence_engine,
)
from .research_innovation_framework import (
    ResearchInnovationFramework,
    ResearchDomain,
    InnovationLevel,
    ResearchExperiment,
    create_research_innovation_framework,
)
from .zero_trust_security_framework import (
    ZeroTrustSecurityFramework,
    ThreatLevel,
    SecurityEvent,
    SecurityContext,
    ThreatDetection,
    create_zero_trust_security_framework,
)
from .global_edge_orchestrator import (
    GlobalEdgeOrchestrator,
    EdgeRegion,
    ScalingStrategy,
    ResourceType,
    WorkloadRequest,
    DeploymentPlan,
    create_global_edge_orchestrator,
)
from .config import (
    BackpressureStrategy,
    CancellationStrategy,
    CancellationType,
    EventLoopConfig,
    MemoryConfig,
    ObservabilityConfig,
    OrchestratorConfig,
    RateLimitConfig,
    SpeculationConfig,
)
from .exceptions import (
    ConfigurationError,
    OrchestratorError,
    RateLimitError,
    SpeculationError,
    TimeoutError,
    ToolExecutionError,
)
from .orchestrator import AsyncOrchestrator
from .quantum_concurrency import QuantumConcurrencyManager, SynchronizationType
from .quantum_orchestrator import (
    QuantumAsyncOrchestrator,
    QuantumToolRegistry,
    create_quantum_orchestrator,
)
from .quantum_performance import (
    OptimizationStrategy,
    PerformanceMetrics,
    QuantumPerformanceOptimizer,
)
from .quantum_planner import (
    ExecutionPlan,
    QuantumInspiredPlanner,
    QuantumTask,
    TaskState,
)
from .quantum_security import (
    AccessLevel,
    QuantumSecurityManager,
    SecurityContext,
    SecurityLevel,
)
from .quantum_validation import QuantumValidator, ValidationLevel, ValidationResult
from .research_experimental_framework import (
    ExperimentalCondition,
    ExperimentType,
    ResearchExperiment,
    ResearchExperimentalFramework,
    create_research_experimental_framework,
)
from .self_adaptive_orchestrator import (
    AdaptationRule,
    AdaptationType,
    EvolutionGenome,
    SelfAdaptiveOrchestrator,
    create_self_adaptive_orchestrator,
)
from .tools import (
    Tool,
    ToolChain,
    ToolRegistry,
    ToolResult,
    parallel,
    retry,
    sequential,
    timeout,
)

__all__ = [
    "AsyncOrchestrator",
    "QuantumAsyncOrchestrator",
    "QuantumToolRegistry",
    "create_quantum_orchestrator",
    "QuantumInspiredPlanner",
    "QuantumTask",
    "ExecutionPlan",
    "TaskState",
    "QuantumSecurityManager",
    "SecurityContext",
    "SecurityLevel",
    "AccessLevel",
    "QuantumValidator",
    "ValidationLevel",
    "ValidationResult",
    "QuantumPerformanceOptimizer",
    "OptimizationStrategy",
    "PerformanceMetrics",
    "QuantumConcurrencyManager",
    "SynchronizationType",
    # Generation 4 Autonomous AI Components
    "AutonomousLearningEngine",
    "PerformancePattern",
    "OptimizationRecommendation",
    "create_autonomous_learning_engine",
    "AdvancedMLOptimizer",
    "MLPrediction",
    "OptimizationExperiment",
    "create_advanced_ml_optimizer",
    "SelfAdaptiveOrchestrator",
    "AdaptationType",
    "AdaptationRule",
    "EvolutionGenome",
    "create_self_adaptive_orchestrator",
    "ResearchExperimentalFramework",
    "ExperimentType",
    "ExperimentalCondition",
    "ResearchExperiment",
    "create_research_experimental_framework",
    # Generation 5 Quantum Leap Components
    "QuantumQualityGateOrchestrator",
    "QuantumQualityLevel",
    "ValidationDimension",
    "QuantumValidationResult",
    "create_quantum_quality_gate_orchestrator",
    "AutonomousIntelligenceEngine",
    "IntelligenceLevel",
    "DecisionDomain",
    "AutonomousDecision",
    "create_autonomous_intelligence_engine",
    "ResearchInnovationFramework",
    "ResearchDomain",
    "InnovationLevel",
    "ResearchExperiment",
    "create_research_innovation_framework",
    "ZeroTrustSecurityFramework",
    "ThreatLevel",
    "SecurityEvent",
    "SecurityContext",
    "ThreatDetection",
    "create_zero_trust_security_framework",
    "GlobalEdgeOrchestrator",
    "EdgeRegion",
    "ScalingStrategy",
    "ResourceType",
    "WorkloadRequest",
    "DeploymentPlan",
    "create_global_edge_orchestrator",
    # Core Tools
    "Tool",
    "ToolChain",
    "ToolResult",
    "ToolRegistry",
    "parallel",
    "sequential",
    "timeout",
    "retry",
    "OrchestratorConfig",
    "RateLimitConfig",
    "CancellationStrategy",
    "SpeculationConfig",
    "ObservabilityConfig",
    "MemoryConfig",
    "EventLoopConfig",
    "BackpressureStrategy",
    "CancellationType",
    "OrchestratorError",
    "ToolExecutionError",
    "RateLimitError",
    "TimeoutError",
    "ConfigurationError",
    "SpeculationError",
]

# Version information
__generation__ = "5"
__codename__ = "Quantum Leap"
__features__ = [
    "Quantum Quality Gates",
    "Autonomous Intelligence Engine",
    "Research Innovation Framework", 
    "Zero Trust Security",
    "Global Edge Orchestration",
    "ML-Driven Optimization",
    "Predictive Analytics",
    "Self-Healing Systems"
]
