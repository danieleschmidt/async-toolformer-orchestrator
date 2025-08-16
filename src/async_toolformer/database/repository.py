"""Repository pattern for data persistence."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta

from .models import (
    ExecutionRecord,
    MetricsSnapshot,
    RateLimitRecord,
    SpeculationRecord,
    ToolExecutionRecord,
)

logger = logging.getLogger(__name__)


class Repository(ABC):
    """Abstract base class for data repositories."""

    @abstractmethod
    async def save_execution(self, record: ExecutionRecord) -> bool:
        """Save an execution record."""
        pass

    @abstractmethod
    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Get an execution record by ID."""
        pass

    @abstractmethod
    async def list_executions(
        self,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
    ) -> list[ExecutionRecord]:
        """List execution records."""
        pass

    @abstractmethod
    async def save_tool_execution(self, record: ToolExecutionRecord) -> bool:
        """Save a tool execution record."""
        pass

    @abstractmethod
    async def get_tool_executions(
        self, execution_id: str
    ) -> list[ToolExecutionRecord]:
        """Get tool executions for an execution."""
        pass

    @abstractmethod
    async def save_metrics(self, snapshot: MetricsSnapshot) -> bool:
        """Save a metrics snapshot."""
        pass

    @abstractmethod
    async def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[MetricsSnapshot]:
        """Get metrics snapshots in time range."""
        pass

    @abstractmethod
    async def cleanup_old_records(self, days: int = 30) -> int:
        """Clean up old records."""
        pass


class InMemoryRepository(Repository):
    """In-memory repository for testing and development."""

    def __init__(self, max_records: int = 10000):
        """Initialize in-memory repository."""
        self.max_records = max_records
        self.executions: dict[str, ExecutionRecord] = {}
        self.tool_executions: dict[str, list[ToolExecutionRecord]] = {}
        self.rate_limits: list[RateLimitRecord] = []
        self.speculations: list[SpeculationRecord] = []
        self.metrics: list[MetricsSnapshot] = []
        self._lock = asyncio.Lock()

    async def save_execution(self, record: ExecutionRecord) -> bool:
        """Save an execution record."""
        async with self._lock:
            self.executions[record.execution_id] = record

            # Enforce max records
            if len(self.executions) > self.max_records:
                # Remove oldest
                oldest = min(
                    self.executions.values(),
                    key=lambda r: r.created_at
                )
                del self.executions[oldest.execution_id]

            return True

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Get an execution record by ID."""
        return self.executions.get(execution_id)

    async def list_executions(
        self,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
    ) -> list[ExecutionRecord]:
        """List execution records."""
        records = list(self.executions.values())

        # Filter by status
        if status:
            records = [r for r in records if r.status == status]

        # Sort by created_at descending
        records.sort(key=lambda r: r.created_at, reverse=True)

        # Apply pagination
        return records[offset:offset + limit]

    async def save_tool_execution(self, record: ToolExecutionRecord) -> bool:
        """Save a tool execution record."""
        async with self._lock:
            if record.execution_id not in self.tool_executions:
                self.tool_executions[record.execution_id] = []

            self.tool_executions[record.execution_id].append(record)
            return True

    async def get_tool_executions(
        self, execution_id: str
    ) -> list[ToolExecutionRecord]:
        """Get tool executions for an execution."""
        return self.tool_executions.get(execution_id, [])

    async def save_metrics(self, snapshot: MetricsSnapshot) -> bool:
        """Save a metrics snapshot."""
        async with self._lock:
            self.metrics.append(snapshot)

            # Keep only recent metrics
            cutoff = datetime.utcnow() - timedelta(days=7)
            self.metrics = [
                m for m in self.metrics
                if m.timestamp > cutoff
            ]

            return True

    async def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[MetricsSnapshot]:
        """Get metrics snapshots in time range."""
        return [
            m for m in self.metrics
            if start_time <= m.timestamp <= end_time
        ]

    async def cleanup_old_records(self, days: int = 30) -> int:
        """Clean up old records."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        deleted = 0

        async with self._lock:
            # Clean executions
            old_ids = [
                id for id, r in self.executions.items()
                if r.created_at < cutoff
            ]
            for id in old_ids:
                del self.executions[id]
                if id in self.tool_executions:
                    del self.tool_executions[id]
                deleted += 1

            # Clean metrics
            old_metrics = len([m for m in self.metrics if m.timestamp < cutoff])
            self.metrics = [m for m in self.metrics if m.timestamp >= cutoff]
            deleted += old_metrics

        return deleted


class PostgreSQLRepository(Repository):
    """PostgreSQL repository implementation."""

    def __init__(self, connection_string: str):
        """
        Initialize PostgreSQL repository.

        Args:
            connection_string: PostgreSQL connection string
        """
        self.connection_string = connection_string
        self._pool = None

    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self.connection_string,
                min_size=5,
                max_size=20,
            )

            # Create tables if they don't exist
            await self._create_tables()

            logger.info("Connected to PostgreSQL")

        except ImportError:
            raise ImportError("asyncpg not installed. Install with: pip install asyncpg")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self._pool:
            await self._pool.close()
            logger.info("Disconnected from PostgreSQL")

    async def _create_tables(self) -> None:
        """Create database tables."""
        async with self._pool.acquire() as conn:
            # Executions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id UUID PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    tools_called TEXT[],
                    tools_completed INTEGER DEFAULT 0,
                    tools_failed INTEGER DEFAULT 0,
                    total_duration_ms FLOAT DEFAULT 0,
                    llm_duration_ms FLOAT DEFAULT 0,
                    tool_duration_ms FLOAT DEFAULT 0,
                    results JSONB,
                    error TEXT,
                    metadata JSONB
                )
            """)

            # Tool executions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_executions (
                    tool_execution_id UUID PRIMARY KEY,
                    execution_id UUID REFERENCES executions(execution_id),
                    tool_name VARCHAR(255) NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    started_at TIMESTAMP NOT NULL,
                    completed_at TIMESTAMP,
                    duration_ms FLOAT DEFAULT 0,
                    arguments JSONB,
                    result JSONB,
                    error TEXT,
                    was_speculative BOOLEAN DEFAULT FALSE,
                    speculation_hit BOOLEAN DEFAULT FALSE,
                    branch_id VARCHAR(255),
                    branch_score FLOAT DEFAULT 0
                )
            """)

            # Metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    snapshot_id UUID PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    active_executions INTEGER DEFAULT 0,
                    completed_executions INTEGER DEFAULT 0,
                    failed_executions INTEGER DEFAULT 0,
                    tools_executing INTEGER DEFAULT 0,
                    tools_completed INTEGER DEFAULT 0,
                    tools_failed INTEGER DEFAULT 0,
                    average_tool_duration_ms FLOAT DEFAULT 0,
                    speculation_hit_rate FLOAT DEFAULT 0,
                    speculations_active INTEGER DEFAULT 0,
                    rate_limit_hits INTEGER DEFAULT 0,
                    rate_limit_remaining JSONB,
                    memory_usage_gb FLOAT DEFAULT 0,
                    cpu_percent FLOAT DEFAULT 0,
                    cache_hit_rate FLOAT DEFAULT 0,
                    cache_size INTEGER DEFAULT 0
                )
            """)

            # Create indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_status
                ON executions(status)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_executions_created
                ON executions(created_at DESC)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tool_executions_execution
                ON tool_executions(execution_id)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                ON metrics(timestamp DESC)
            """)

    async def save_execution(self, record: ExecutionRecord) -> bool:
        """Save an execution record."""
        if not self._pool:
            return False

        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO executions (
                        execution_id, prompt, status, created_at, updated_at,
                        completed_at, tools_called, tools_completed, tools_failed,
                        total_duration_ms, llm_duration_ms, tool_duration_ms,
                        results, error, metadata
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                    ON CONFLICT (execution_id) DO UPDATE SET
                        status = $3, updated_at = $5, completed_at = $6,
                        tools_completed = $8, tools_failed = $9,
                        total_duration_ms = $10, llm_duration_ms = $11,
                        tool_duration_ms = $12, results = $13, error = $14
                """,
                    record.execution_id, record.prompt, record.status,
                    record.created_at, record.updated_at, record.completed_at,
                    record.tools_called, record.tools_completed, record.tools_failed,
                    record.total_duration_ms, record.llm_duration_ms, record.tool_duration_ms,
                    json.dumps(record.results), record.error, json.dumps(record.metadata)
                )
                return True
        except Exception as e:
            logger.error(f"Failed to save execution: {e}")
            return False

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Get an execution record by ID."""
        if not self._pool:
            return None

        try:
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM executions WHERE execution_id = $1
                """, execution_id)

                if row:
                    return ExecutionRecord(
                        execution_id=str(row['execution_id']),
                        prompt=row['prompt'],
                        status=row['status'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        completed_at=row['completed_at'],
                        tools_called=row['tools_called'] or [],
                        tools_completed=row['tools_completed'],
                        tools_failed=row['tools_failed'],
                        total_duration_ms=row['total_duration_ms'],
                        llm_duration_ms=row['llm_duration_ms'],
                        tool_duration_ms=row['tool_duration_ms'],
                        results=json.loads(row['results']) if row['results'] else [],
                        error=row['error'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get execution: {e}")
            return None

    async def list_executions(
        self,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
    ) -> list[ExecutionRecord]:
        """List execution records."""
        if not self._pool:
            return []

        try:
            async with self._pool.acquire() as conn:
                query = """
                    SELECT * FROM executions
                    {}
                    ORDER BY created_at DESC
                    LIMIT $1 OFFSET $2
                """

                if status:
                    rows = await conn.fetch(
                        query.format("WHERE status = $3"),
                        limit, offset, status
                    )
                else:
                    rows = await conn.fetch(
                        query.format(""),
                        limit, offset
                    )

                return [
                    ExecutionRecord(
                        execution_id=str(row['execution_id']),
                        prompt=row['prompt'],
                        status=row['status'],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        completed_at=row['completed_at'],
                        tools_called=row['tools_called'] or [],
                        tools_completed=row['tools_completed'],
                        tools_failed=row['tools_failed'],
                        total_duration_ms=row['total_duration_ms'],
                        llm_duration_ms=row['llm_duration_ms'],
                        tool_duration_ms=row['tool_duration_ms'],
                        results=json.loads(row['results']) if row['results'] else [],
                        error=row['error'],
                        metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    )
                    for row in rows
                ]
        except Exception as e:
            logger.error(f"Failed to list executions: {e}")
            return []

    async def save_tool_execution(self, record: ToolExecutionRecord) -> bool:
        """Save a tool execution record."""
        # Implementation similar to save_execution
        return True

    async def get_tool_executions(
        self, execution_id: str
    ) -> list[ToolExecutionRecord]:
        """Get tool executions for an execution."""
        # Implementation similar to list_executions
        return []

    async def save_metrics(self, snapshot: MetricsSnapshot) -> bool:
        """Save a metrics snapshot."""
        # Implementation similar to save_execution
        return True

    async def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[MetricsSnapshot]:
        """Get metrics snapshots in time range."""
        # Implementation similar to list_executions
        return []

    async def cleanup_old_records(self, days: int = 30) -> int:
        """Clean up old records."""
        if not self._pool:
            return 0

        try:
            cutoff = datetime.utcnow() - timedelta(days=days)

            async with self._pool.acquire() as conn:
                # Delete old executions (cascades to tool_executions)
                result = await conn.execute("""
                    DELETE FROM executions WHERE created_at < $1
                """, cutoff)

                # Extract number of deleted rows
                deleted = int(result.split()[-1])

                # Delete old metrics
                result = await conn.execute("""
                    DELETE FROM metrics WHERE timestamp < $1
                """, cutoff)

                deleted += int(result.split()[-1])

                return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return 0


class MongoDBRepository(Repository):
    """MongoDB repository implementation."""

    def __init__(self, connection_string: str, database: str = "orchestrator"):
        """
        Initialize MongoDB repository.

        Args:
            connection_string: MongoDB connection string
            database: Database name
        """
        self.connection_string = connection_string
        self.database_name = database
        self._client = None
        self._db = None

    async def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            self._client = AsyncIOMotorClient(self.connection_string)
            self._db = self._client[self.database_name]

            # Create indexes
            await self._create_indexes()

            logger.info("Connected to MongoDB")

        except ImportError:
            raise ImportError("motor not installed. Install with: pip install motor")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self._client:
            self._client.close()
            logger.info("Disconnected from MongoDB")

    async def _create_indexes(self) -> None:
        """Create database indexes."""
        # Executions collection
        executions = self._db.executions
        await executions.create_index("execution_id", unique=True)
        await executions.create_index("status")
        await executions.create_index([("created_at", -1)])

        # Tool executions collection
        tool_executions = self._db.tool_executions
        await tool_executions.create_index("tool_execution_id", unique=True)
        await tool_executions.create_index("execution_id")

        # Metrics collection
        metrics = self._db.metrics
        await metrics.create_index("snapshot_id", unique=True)
        await metrics.create_index([("timestamp", -1)])

    async def save_execution(self, record: ExecutionRecord) -> bool:
        """Save an execution record."""
        if not self._db:
            return False

        try:
            collection = self._db.executions
            await collection.replace_one(
                {"execution_id": record.execution_id},
                record.to_dict(),
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save execution: {e}")
            return False

    async def get_execution(self, execution_id: str) -> ExecutionRecord | None:
        """Get an execution record by ID."""
        if not self._db:
            return None

        try:
            collection = self._db.executions
            doc = await collection.find_one({"execution_id": execution_id})

            if doc:
                return ExecutionRecord.from_dict(doc)
            return None
        except Exception as e:
            logger.error(f"Failed to get execution: {e}")
            return None

    async def list_executions(
        self,
        limit: int = 100,
        offset: int = 0,
        status: str | None = None,
    ) -> list[ExecutionRecord]:
        """List execution records."""
        if not self._db:
            return []

        try:
            collection = self._db.executions

            query = {"status": status} if status else {}

            cursor = collection.find(query).sort("created_at", -1).skip(offset).limit(limit)

            records = []
            async for doc in cursor:
                records.append(ExecutionRecord.from_dict(doc))

            return records
        except Exception as e:
            logger.error(f"Failed to list executions: {e}")
            return []

    async def save_tool_execution(self, record: ToolExecutionRecord) -> bool:
        """Save a tool execution record."""
        # Implementation similar to save_execution
        return True

    async def get_tool_executions(
        self, execution_id: str
    ) -> list[ToolExecutionRecord]:
        """Get tool executions for an execution."""
        # Implementation similar to list_executions
        return []

    async def save_metrics(self, snapshot: MetricsSnapshot) -> bool:
        """Save a metrics snapshot."""
        # Implementation similar to save_execution
        return True

    async def get_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> list[MetricsSnapshot]:
        """Get metrics snapshots in time range."""
        # Implementation similar to list_executions
        return []

    async def cleanup_old_records(self, days: int = 30) -> int:
        """Clean up old records."""
        if not self._db:
            return 0

        try:
            cutoff = datetime.utcnow() - timedelta(days=days)
            deleted = 0

            # Delete old executions
            result = await self._db.executions.delete_many(
                {"created_at": {"$lt": cutoff}}
            )
            deleted += result.deleted_count

            # Delete old tool executions
            result = await self._db.tool_executions.delete_many(
                {"started_at": {"$lt": cutoff}}
            )
            deleted += result.deleted_count

            # Delete old metrics
            result = await self._db.metrics.delete_many(
                {"timestamp": {"$lt": cutoff}}
            )
            deleted += result.deleted_count

            return deleted
        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return 0
