"""Health monitoring and alerting system."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# import psutil  # Not available - use simplified system metrics
import aiohttp

from .simple_structured_logging import get_logger

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    check_function: Callable
    interval_seconds: float = 30.0
    timeout_seconds: float = 5.0
    failure_threshold: int = 3
    success_threshold: int = 2
    enabled: bool = True
    tags: list[str] = field(default_factory=list)


@dataclass
class HealthResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    timestamp: float = 0.0
    error: str | None = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class SystemMetrics:
    """System resource metrics collector."""

    @staticmethod
    def get_cpu_usage() -> float:
        """Get CPU usage percentage (simplified)."""
        # Simplified implementation without psutil
        return 25.0  # Mock value

    @staticmethod
    def get_memory_usage() -> dict[str, float]:
        """Get memory usage statistics (simplified)."""
        # Simplified implementation without psutil
        return {
            "total_gb": 8.0,
            "used_gb": 3.2,
            "available_gb": 4.8,
            "percent": 40.0
        }

    @staticmethod
    def get_disk_usage() -> dict[str, float]:
        """Get disk usage statistics (simplified)."""
        # Simplified implementation without psutil
        return {
            "total_gb": 100.0,
            "used_gb": 45.0,
            "free_gb": 55.0,
            "percent": 45.0
        }

    @staticmethod
    def get_network_stats() -> dict[str, int]:
        """Get network statistics (simplified)."""
        # Simplified implementation without psutil
        return {
            "bytes_sent": 1024000,
            "bytes_recv": 2048000,
            "packets_sent": 1000,
            "packets_recv": 1500,
            "errors_in": 0,
            "errors_out": 0
        }


class HealthMonitor:
    """Comprehensive health monitoring system."""

    def __init__(self):
        self.health_checks: dict[str, HealthCheck] = {}
        self.check_results: dict[str, HealthResult] = {}
        self.check_history: dict[str, list[HealthResult]] = {}
        self.consecutive_failures: dict[str, int] = {}
        self.consecutive_successes: dict[str, int] = {}
        self.monitoring_tasks: dict[str, asyncio.Task] = {}
        self.system_metrics = SystemMetrics()
        self._monitoring_active = False

    def register_check(self, check: HealthCheck):
        """Register a new health check."""
        self.health_checks[check.name] = check
        self.check_history[check.name] = []
        self.consecutive_failures[check.name] = 0
        self.consecutive_successes[check.name] = 0

        logger.info(f"Registered health check: {check.name}")

        # Start monitoring if active
        if self._monitoring_active and check.enabled:
            self.monitoring_tasks[check.name] = asyncio.create_task(
                self._monitor_check(check)
            )

    def unregister_check(self, name: str):
        """Unregister a health check."""
        if name in self.health_checks:
            del self.health_checks[name]

        if name in self.monitoring_tasks:
            self.monitoring_tasks[name].cancel()
            del self.monitoring_tasks[name]

        # Clean up state
        self.check_results.pop(name, None)
        self.check_history.pop(name, None)
        self.consecutive_failures.pop(name, None)
        self.consecutive_successes.pop(name, None)

        logger.info(f"Unregistered health check: {name}")

    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring_active:
            logger.warning("Health monitoring is already active")
            return

        self._monitoring_active = True

        # Start monitoring tasks for all enabled checks
        for name, check in self.health_checks.items():
            if check.enabled:
                self.monitoring_tasks[name] = asyncio.create_task(
                    self._monitor_check(check)
                )

        logger.info(f"Started health monitoring for {len(self.monitoring_tasks)} checks")

    async def stop_monitoring(self):
        """Stop all health monitoring."""
        self._monitoring_active = False

        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            if not task.done():
                task.cancel()

        # Wait for cancellation
        if self.monitoring_tasks:
            await asyncio.gather(
                *self.monitoring_tasks.values(),
                return_exceptions=True
            )

        self.monitoring_tasks.clear()
        logger.info("Stopped health monitoring")

    async def _monitor_check(self, check: HealthCheck):
        """Monitor a single health check continuously."""
        logger.debug(f"Starting monitoring for check: {check.name}")

        while self._monitoring_active:
            try:
                await self.run_check(check.name)
                await asyncio.sleep(check.interval_seconds)
            except asyncio.CancelledError:
                logger.debug(f"Monitoring cancelled for check: {check.name}")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop for {check.name}", error=e)
                await asyncio.sleep(check.interval_seconds)

    async def run_check(self, name: str) -> HealthResult:
        """Run a single health check."""
        if name not in self.health_checks:
            raise ValueError(f"Health check '{name}' not found")

        check = self.health_checks[name]
        start_time = time.time()

        try:
            # Execute the check with timeout
            result_data = await asyncio.wait_for(
                check.check_function(),
                timeout=check.timeout_seconds
            )

            execution_time = (time.time() - start_time) * 1000

            # Parse result
            if isinstance(result_data, dict):
                status = HealthStatus(result_data.get("status", "healthy"))
                message = result_data.get("message", "Check passed")
                details = result_data.get("details", {})
            elif isinstance(result_data, bool):
                status = HealthStatus.HEALTHY if result_data else HealthStatus.UNHEALTHY
                message = "Check passed" if result_data else "Check failed"
                details = {}
            else:
                status = HealthStatus.HEALTHY
                message = str(result_data)
                details = {"result": result_data}

            result = HealthResult(
                name=name,
                status=status,
                message=message,
                details=details,
                execution_time_ms=execution_time
            )

        except asyncio.TimeoutError:
            execution_time = (time.time() - start_time) * 1000
            result = HealthResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check timed out after {check.timeout_seconds}s",
                execution_time_ms=execution_time,
                error="timeout"
            )

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            result = HealthResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {str(e)}",
                execution_time_ms=execution_time,
                error=str(e)
            )

        # Update tracking
        self._update_check_tracking(name, result)

        # Store result
        self.check_results[name] = result
        self.check_history[name].append(result)

        # Trim history (keep last 100 results)
        if len(self.check_history[name]) > 100:
            self.check_history[name] = self.check_history[name][-100:]

        logger.debug(
            f"Health check completed: {name}",
            status=result.status.value,
            execution_time_ms=result.execution_time_ms
        )

        return result

    def _update_check_tracking(self, name: str, result: HealthResult):
        """Update failure/success tracking for a check."""
        if result.status == HealthStatus.HEALTHY:
            self.consecutive_successes[name] += 1
            self.consecutive_failures[name] = 0
        else:
            self.consecutive_failures[name] += 1
            self.consecutive_successes[name] = 0

    async def run_all_checks(self) -> dict[str, HealthResult]:
        """Run all health checks once."""
        results = {}
        tasks = []

        for name, check in self.health_checks.items():
            if check.enabled:
                tasks.append((name, asyncio.create_task(self.run_check(name))))

        # Wait for all checks to complete
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Failed to run health check {name}", error=e)
                results[name] = HealthResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Check execution failed: {str(e)}",
                    error=str(e)
                )

        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.check_results:
            return HealthStatus.UNKNOWN

        unhealthy_count = sum(
            1 for result in self.check_results.values()
            if result.status == HealthStatus.UNHEALTHY
        )

        degraded_count = sum(
            1 for result in self.check_results.values()
            if result.status == HealthStatus.DEGRADED
        )

        total_checks = len(self.check_results)

        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        elif total_checks > 0:
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def get_health_report(self) -> dict[str, Any]:
        """Get comprehensive health report."""
        overall_status = self.get_overall_status()

        check_summary = {}
        for name, result in self.check_results.items():
            check_summary[name] = {
                "status": result.status.value,
                "message": result.message,
                "last_check": result.timestamp,
                "execution_time_ms": result.execution_time_ms,
                "consecutive_failures": self.consecutive_failures.get(name, 0),
                "consecutive_successes": self.consecutive_successes.get(name, 0)
            }

        # Get system metrics
        try:
            system_metrics = {
                "cpu_percent": self.system_metrics.get_cpu_usage(),
                "memory": self.system_metrics.get_memory_usage(),
                "disk": self.system_metrics.get_disk_usage(),
                "network": self.system_metrics.get_network_stats()
            }
        except Exception as e:
            logger.error("Failed to get system metrics", error=e)
            system_metrics = {"error": str(e)}

        return {
            "overall_status": overall_status.value,
            "timestamp": time.time(),
            "checks": check_summary,
            "system_metrics": system_metrics,
            "monitoring_active": self._monitoring_active,
            "total_checks": len(self.health_checks),
            "enabled_checks": len([c for c in self.health_checks.values() if c.enabled])
        }

    def get_check_history(self, name: str, limit: int = 50) -> list[HealthResult]:
        """Get history of a specific health check."""
        if name not in self.check_history:
            return []

        return self.check_history[name][-limit:]


# Default health checks
async def database_health_check():
    """Check database connectivity."""
    # This would be implemented based on your database
    return {"status": "healthy", "message": "Database connection OK"}


async def llm_provider_health_check():
    """Check LLM provider connectivity."""
    try:
        # Simple connectivity test - could be enhanced
        async with aiohttp.ClientSession() as session:
            async with session.get('https://api.openai.com/v1/models', timeout=5) as response:
                if response.status == 401 or response.status < 500:  # Expected without API key
                    return {"status": "healthy", "message": "LLM provider reachable"}
                else:
                    return {"status": "degraded", "message": f"LLM provider returned {response.status}"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"LLM provider unreachable: {str(e)}"}


async def memory_health_check():
    """Check system memory usage."""
    memory = SystemMetrics.get_memory_usage()

    if memory["percent"] > 90:
        return {
            "status": "unhealthy",
            "message": f"Memory usage critical: {memory['percent']:.1f}%",
            "details": memory
        }
    elif memory["percent"] > 75:
        return {
            "status": "degraded",
            "message": f"Memory usage high: {memory['percent']:.1f}%",
            "details": memory
        }
    else:
        return {
            "status": "healthy",
            "message": f"Memory usage normal: {memory['percent']:.1f}%",
            "details": memory
        }


# Global health monitor instance
health_monitor = HealthMonitor()

# Register default health checks
health_monitor.register_check(HealthCheck(
    name="memory",
    check_function=memory_health_check,
    interval_seconds=30
))

health_monitor.register_check(HealthCheck(
    name="llm_provider",
    check_function=llm_provider_health_check,
    interval_seconds=60
))
