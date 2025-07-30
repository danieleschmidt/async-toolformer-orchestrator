"""Configuration classes for the Async Toolformer Orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Union


class BackpressureStrategy(Enum):
    """Strategies for handling backpressure."""
    
    FAIL_FAST = "fail_fast"
    ADAPTIVE = "adaptive"
    QUEUE = "queue"
    DROP = "drop"


class CancellationType(Enum):
    """Types of cancellation strategies."""
    
    TIMEOUT = "timeout"
    BETTER_RESULT = "better_result"
    RESOURCE_LIMIT = "resource_limit"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    global_max: int = 100
    """Maximum requests per second globally."""
    
    service_limits: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=dict)
    """Per-service rate limits with format: {service: {limit_type: value}}."""
    
    use_redis: bool = False
    """Whether to use Redis for distributed rate limiting."""
    
    redis_url: Optional[str] = None
    """Redis connection URL for distributed rate limiting."""
    
    backpressure_strategy: BackpressureStrategy = BackpressureStrategy.ADAPTIVE
    """Strategy for handling rate limit backpressure."""
    
    retry_backoff_base: float = 1.0
    """Base backoff time for retries in seconds."""
    
    retry_backoff_max: float = 60.0
    """Maximum backoff time for retries in seconds."""
    
    def get_service_limit(self, service: str, limit_type: str) -> Optional[Union[int, float]]:
        """Get a specific limit for a service."""
        return self.service_limits.get(service, {}).get(limit_type)


@dataclass
class CancellationStrategy:
    """Configuration for task cancellation."""
    
    timeout_ms: int = 30000
    """Default timeout for tasks in milliseconds."""
    
    cancel_on_better_result: bool = True
    """Whether to cancel tasks when better results are found."""
    
    keep_top_n_branches: int = 5
    """Number of top-performing branches to keep active."""
    
    min_completion_threshold: float = 0.8
    """Minimum completion score to consider cancelling other branches."""
    
    cancellation_delay_ms: int = 100
    """Delay before applying cancellation to allow for completion."""


@dataclass
class SpeculationConfig:
    """Configuration for speculative execution."""
    
    enabled: bool = True
    """Whether speculative execution is enabled."""
    
    speculation_model: str = "gpt-3.5-turbo"
    """Model to use for speculation (should be faster/cheaper)."""
    
    confidence_threshold: float = 0.7
    """Minimum confidence to act on speculation."""
    
    max_speculative_calls: int = 10
    """Maximum number of speculative calls per request."""
    
    speculation_timeout_ms: int = 5000
    """Timeout for speculation requests."""


@dataclass
class ObservabilityConfig:
    """Configuration for observability and monitoring."""
    
    enable_tracing: bool = True
    """Whether to enable distributed tracing."""
    
    enable_metrics: bool = True
    """Whether to enable Prometheus metrics."""
    
    metrics_endpoint: Optional[str] = None
    """Prometheus metrics endpoint URL."""
    
    trace_sampling_rate: float = 0.1
    """Sampling rate for distributed tracing (0.0-1.0)."""
    
    jaeger_host: Optional[str] = None
    """Jaeger host for trace collection."""
    
    log_level: str = "INFO"
    """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""
    
    structured_logging: bool = True
    """Whether to use structured logging."""


@dataclass
class MemoryConfig:
    """Configuration for memory management."""
    
    max_memory_gb: float = 8.0
    """Maximum memory usage in GB."""
    
    gc_threshold_gb: float = 6.0
    """Garbage collection threshold in GB."""
    
    compress_results: bool = True
    """Whether to compress large results."""
    
    max_result_size_mb: int = 100
    """Maximum size for individual results in MB."""
    
    swap_to_disk: bool = False
    """Whether to swap large results to disk."""
    
    disk_path: str = "/tmp/orchestrator"
    """Path for disk swapping."""


@dataclass
class EventLoopConfig:
    """Configuration for asyncio event loop optimization."""
    
    max_tasks: int = 10000
    """Maximum number of concurrent tasks."""
    
    executor_workers: int = 50
    """Number of thread pool executor workers."""
    
    use_process_pool: bool = False
    """Whether to use process pool for CPU-bound tasks."""
    
    use_uvloop: bool = True
    """Whether to use uvloop for better performance."""


@dataclass
class OrchestratorConfig:
    """Main configuration for the AsyncOrchestrator."""
    
    # Core parallelism settings
    max_parallel_tools: int = 30
    """Maximum number of tools to execute in parallel."""
    
    max_parallel_per_type: int = 10
    """Maximum parallel executions per tool type."""
    
    # Timeout settings
    tool_timeout_ms: int = 10000
    """Default timeout for individual tools."""
    
    llm_timeout_ms: int = 30000
    """Timeout for LLM requests."""
    
    total_timeout_ms: int = 60000
    """Total timeout for entire orchestration."""
    
    # Retry configuration
    retry_attempts: int = 3
    """Number of retry attempts for failed operations."""
    
    retry_backoff: str = "exponential"
    """Backoff strategy: 'linear', 'exponential', or 'fixed'."""
    
    # Sub-configurations
    rate_limit_config: RateLimitConfig = field(default_factory=RateLimitConfig)
    """Rate limiting configuration."""
    
    cancellation_strategy: CancellationStrategy = field(default_factory=CancellationStrategy)
    """Task cancellation configuration."""
    
    speculation_config: SpeculationConfig = field(default_factory=SpeculationConfig)
    """Speculative execution configuration."""
    
    observability_config: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    """Observability and monitoring configuration."""
    
    memory_config: MemoryConfig = field(default_factory=MemoryConfig)
    """Memory management configuration."""
    
    event_loop_config: EventLoopConfig = field(default_factory=EventLoopConfig)
    """Event loop optimization configuration."""
    
    # Additional settings
    enable_result_streaming: bool = True
    """Whether to enable streaming of partial results."""
    
    result_aggregation_strategy: str = "intelligent"
    """Strategy for aggregating results: 'simple', 'intelligent', 'custom'."""
    
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    """Custom configuration settings for extensions."""
    
    def validate(self) -> None:
        """Validate the configuration settings."""
        if self.max_parallel_tools <= 0:
            raise ValueError("max_parallel_tools must be positive")
        
        if self.max_parallel_per_type > self.max_parallel_tools:
            raise ValueError("max_parallel_per_type cannot exceed max_parallel_tools")
        
        if self.tool_timeout_ms <= 0:
            raise ValueError("tool_timeout_ms must be positive")
        
        if self.total_timeout_ms < self.tool_timeout_ms:
            raise ValueError("total_timeout_ms must be >= tool_timeout_ms")
        
        if not 0.0 <= self.observability_config.trace_sampling_rate <= 1.0:
            raise ValueError("trace_sampling_rate must be between 0.0 and 1.0")