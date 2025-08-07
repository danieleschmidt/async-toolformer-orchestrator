"""Async Toolformer Orchestrator - Parallel tool execution for LLMs."""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "async-tools@yourdomain.com"

from .orchestrator import AsyncOrchestrator
from .quantum_orchestrator import QuantumAsyncOrchestrator, QuantumToolRegistry, create_quantum_orchestrator
from .quantum_planner import QuantumInspiredPlanner, QuantumTask, ExecutionPlan, TaskState
from .quantum_security import QuantumSecurityManager, SecurityContext, SecurityLevel, AccessLevel
from .quantum_validation import QuantumValidator, ValidationLevel, ValidationResult
from .quantum_performance import QuantumPerformanceOptimizer, OptimizationStrategy, PerformanceMetrics
from .quantum_concurrency import QuantumConcurrencyManager, SynchronizationType
from .tools import Tool, ToolChain, ToolResult, ToolRegistry, parallel, sequential, timeout, retry
from .sentiment_analyzer import (
    analyze_text_sentiment,
    analyze_batch_sentiment, 
    analyze_social_media_sentiment,
    compare_sentiment_sources,
    SentimentResult,
    SentimentScore,
    SentimentPolarity,
    EmotionScore,
    EmotionType,
    BatchSentimentResult
)
from .quantum_sentiment import (
    QuantumSentimentAnalyzer,
    QuantumSentimentConfig,
    create_quantum_sentiment_analyzer
)
from .sentiment_intelligence import (
    QuantumSentimentIntelligence,
    create_quantum_sentiment_intelligence
)
from .sentiment_validation import (
    SentimentValidator,
    SentimentValidationConfig,
    SentimentSecurityManager,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity
)
from .sentiment_monitoring import (
    SentimentMonitor,
    get_sentiment_monitor,
    set_sentiment_monitor
)
from .sentiment_globalization import (
    global_sentiment_analysis,
    MultiLanguageSentimentAnalyzer,
    GlobalComplianceManager,
    SupportedLanguage,
    ComplianceRegion,
    get_supported_languages,
    get_supported_regions
)
from .config import (
    OrchestratorConfig,
    RateLimitConfig, 
    CancellationStrategy,
    SpeculationConfig,
    ObservabilityConfig,
    MemoryConfig,
    EventLoopConfig,
    BackpressureStrategy,
    CancellationType,
)
from .exceptions import (
    OrchestratorError,
    ToolExecutionError,
    RateLimitError,
    TimeoutError,
    ConfigurationError,
    SpeculationError,
)

__all__ = [
    # Core Orchestrator
    "AsyncOrchestrator",
    "QuantumAsyncOrchestrator",
    "QuantumToolRegistry", 
    "create_quantum_orchestrator",
    
    # Quantum Components
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
    
    # Tools
    "Tool",
    "ToolChain",
    "ToolResult",
    "ToolRegistry",
    "parallel",
    "sequential",
    "timeout",
    "retry",
    
    # Sentiment Analysis
    "analyze_text_sentiment",
    "analyze_batch_sentiment", 
    "analyze_social_media_sentiment",
    "compare_sentiment_sources",
    "global_sentiment_analysis",
    "SentimentResult",
    "SentimentScore",
    "SentimentPolarity",
    "EmotionScore",
    "EmotionType",
    "BatchSentimentResult",
    "QuantumSentimentAnalyzer",
    "QuantumSentimentConfig",
    "create_quantum_sentiment_analyzer",
    "QuantumSentimentIntelligence",
    "create_quantum_sentiment_intelligence",
    "SentimentValidator",
    "SentimentValidationConfig",
    "SentimentSecurityManager",
    "ValidationIssue",
    "ValidationSeverity",
    "SentimentMonitor",
    "get_sentiment_monitor",
    "set_sentiment_monitor",
    "MultiLanguageSentimentAnalyzer",
    "GlobalComplianceManager",
    "SupportedLanguage",
    "ComplianceRegion",
    "get_supported_languages",
    "get_supported_regions",
    
    # Configuration
    "OrchestratorConfig",
    "RateLimitConfig",
    "CancellationStrategy",
    "SpeculationConfig",
    "ObservabilityConfig",
    "MemoryConfig",
    "EventLoopConfig",
    "BackpressureStrategy",
    "CancellationType",
    
    # Exceptions
    "OrchestratorError",
    "ToolExecutionError", 
    "RateLimitError",
    "TimeoutError",
    "ConfigurationError",
    "SpeculationError",
]