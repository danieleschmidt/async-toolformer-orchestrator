"""Database models for result persistence."""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict
import json


@dataclass
class ExecutionRecord:
    """Record of a tool execution."""
    
    execution_id: str
    prompt: str
    status: str  # pending, running, completed, failed
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    
    # Execution details
    tools_called: List[str] = field(default_factory=list)
    tools_completed: int = 0
    tools_failed: int = 0
    
    # Performance metrics
    total_duration_ms: float = 0
    llm_duration_ms: float = 0
    tool_duration_ms: float = 0
    
    # Results
    results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> 'ExecutionRecord':
        """Create a new execution record."""
        now = datetime.utcnow()
        return cls(
            execution_id=str(uuid.uuid4()),
            prompt=prompt,
            status="pending",
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert datetime objects to ISO format
        for key in ['created_at', 'updated_at', 'completed_at']:
            if data[key]:
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExecutionRecord':
        """Create from dictionary."""
        # Convert ISO strings back to datetime
        for key in ['created_at', 'updated_at', 'completed_at']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


@dataclass
class ToolExecutionRecord:
    """Record of a single tool execution."""
    
    tool_execution_id: str
    execution_id: str  # Parent execution
    tool_name: str
    status: str  # pending, running, completed, failed, cancelled
    
    # Timing
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: float = 0
    
    # Input/Output
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    
    # Speculation
    was_speculative: bool = False
    speculation_hit: bool = False
    
    # Branch info
    branch_id: Optional[str] = None
    branch_score: float = 0.0
    
    @classmethod
    def create(
        cls,
        execution_id: str,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> 'ToolExecutionRecord':
        """Create a new tool execution record."""
        return cls(
            tool_execution_id=str(uuid.uuid4()),
            execution_id=execution_id,
            tool_name=tool_name,
            status="pending",
            started_at=datetime.utcnow(),
            arguments=arguments,
        )
    
    def complete(self, result: Any, error: Optional[str] = None) -> None:
        """Mark as completed."""
        self.completed_at = datetime.utcnow()
        self.duration_ms = (self.completed_at - self.started_at).total_seconds() * 1000
        self.result = result
        self.error = error
        self.status = "failed" if error else "completed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert datetime objects
        for key in ['started_at', 'completed_at']:
            if data[key]:
                data[key] = data[key].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolExecutionRecord':
        """Create from dictionary."""
        for key in ['started_at', 'completed_at']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        return cls(**data)


@dataclass
class RateLimitRecord:
    """Record of rate limit events."""
    
    record_id: str
    service: str
    timestamp: datetime
    
    # Rate limit details
    limit_type: str  # calls, tokens, etc.
    limit_value: int
    current_usage: int
    exceeded: bool
    
    # Context
    execution_id: Optional[str] = None
    retry_after: Optional[float] = None
    
    @classmethod
    def create(
        cls,
        service: str,
        limit_type: str,
        limit_value: int,
        current_usage: int,
        exceeded: bool,
        execution_id: Optional[str] = None,
    ) -> 'RateLimitRecord':
        """Create a rate limit record."""
        return cls(
            record_id=str(uuid.uuid4()),
            service=service,
            timestamp=datetime.utcnow(),
            limit_type=limit_type,
            limit_value=limit_value,
            current_usage=current_usage,
            exceeded=exceeded,
            execution_id=execution_id,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = data['timestamp'].isoformat()
        return data


@dataclass
class SpeculationRecord:
    """Record of speculation attempts."""
    
    speculation_id: str
    execution_id: str
    tool_name: str
    confidence: float
    
    # Timing
    created_at: datetime
    execution_time_ms: float = 0
    
    # Outcome
    committed: bool = False
    cancelled: bool = False
    time_saved_ms: float = 0
    
    @classmethod
    def create(
        cls,
        execution_id: str,
        tool_name: str,
        confidence: float,
    ) -> 'SpeculationRecord':
        """Create a speculation record."""
        return cls(
            speculation_id=str(uuid.uuid4()),
            execution_id=execution_id,
            tool_name=tool_name,
            confidence=confidence,
            created_at=datetime.utcnow(),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = data['created_at'].isoformat()
        return data


@dataclass
class MetricsSnapshot:
    """Snapshot of system metrics."""
    
    snapshot_id: str
    timestamp: datetime
    
    # Performance metrics
    active_executions: int = 0
    completed_executions: int = 0
    failed_executions: int = 0
    
    # Tool metrics
    tools_executing: int = 0
    tools_completed: int = 0
    tools_failed: int = 0
    average_tool_duration_ms: float = 0
    
    # Speculation metrics
    speculation_hit_rate: float = 0
    speculations_active: int = 0
    
    # Rate limiting
    rate_limit_hits: int = 0
    rate_limit_remaining: Dict[str, int] = field(default_factory=dict)
    
    # Resource usage
    memory_usage_gb: float = 0
    cpu_percent: float = 0
    
    # Cache metrics
    cache_hit_rate: float = 0
    cache_size: int = 0
    
    @classmethod
    def create(cls, metrics: Dict[str, Any]) -> 'MetricsSnapshot':
        """Create from current metrics."""
        return cls(
            snapshot_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            **metrics,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['timestamp'] = data['timestamp'].isoformat()
        return data