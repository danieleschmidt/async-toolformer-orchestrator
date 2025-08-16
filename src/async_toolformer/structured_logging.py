"""Structured logging with correlation IDs and contextual information."""

import asyncio
import contextvars
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from enum import Enum

import structlog

# Correlation context variables
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='')
execution_id: contextvars.ContextVar[str] = contextvars.ContextVar('execution_id', default='')
user_id: contextvars.ContextVar[str] = contextvars.ContextVar('user_id', default='')


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogContext:
    """Structured log context."""
    correlation_id: str = ""
    execution_id: str = ""
    user_id: str = ""
    component: str = ""
    operation: str = ""
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.correlation_id:
            self.correlation_id = correlation_id.get('')


class StructuredLogger:
    """Enhanced structured logger with correlation tracking."""

    def __init__(self, name: str):
        self.name = name
        self.logger = structlog.get_logger(name)

    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, error: Exception | None = None, **kwargs):
        """Log error message with context."""
        if error:
            kwargs.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_details": getattr(error, 'details', {})
            })
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, error: Exception | None = None, **kwargs):
        """Log critical message with context."""
        if error:
            kwargs.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_details": getattr(error, 'details', {})
            })
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def _log(self, level: LogLevel, message: str, **kwargs):
        """Internal log method with context enrichment."""
        context = LogContext(
            correlation_id=correlation_id.get(''),
            execution_id=execution_id.get(''),
            user_id=user_id.get(''),
            component=self.name
        )

        # Merge context with additional kwargs
        log_data = asdict(context)
        log_data.update(kwargs)
        log_data["message"] = message

        # Route to appropriate structlog method
        log_method = getattr(self.logger, level.value)
        log_method(**log_data)


def configure_structured_logging(
    log_level: str = "INFO",
    enable_json: bool = True,
    enable_correlation: bool = True,
    include_timestamps: bool = True
):
    """Configure structured logging for the entire application."""

    processors = []

    # Add correlation ID processor
    if enable_correlation:
        processors.append(add_correlation_id)

    # Add timestamp processor
    if include_timestamps:
        processors.append(structlog.processors.TimeStamper(fmt="ISO"))

    # Add log level processor
    processors.append(structlog.stdlib.add_log_level)

    # Add logger name processor
    processors.append(structlog.stdlib.add_logger_name)

    if enable_json:
        # JSON output for production
        processors.extend([
            structlog.processors.JSONRenderer()
        ])
    else:
        # Human-readable output for development
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, log_level.upper())),
        logger_factory=structlog.WriteLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


def add_correlation_id(logger, name, event_dict):
    """Add correlation ID to log entries."""
    event_dict["correlation_id"] = correlation_id.get('')
    event_dict["execution_id"] = execution_id.get('')
    event_dict["user_id"] = user_id.get('')
    return event_dict


class CorrelationContext:
    """Context manager for correlation tracking."""

    def __init__(self,
                 correlation_id_value: str | None = None,
                 execution_id_value: str | None = None,
                 user_id_value: str | None = None):
        self.correlation_id_value = correlation_id_value or str(uuid.uuid4())
        self.execution_id_value = execution_id_value or f"exec_{int(time.time() * 1000)}"
        self.user_id_value = user_id_value or ""

        self.correlation_token = None
        self.execution_token = None
        self.user_token = None

    def __enter__(self):
        self.correlation_token = correlation_id.set(self.correlation_id_value)
        self.execution_token = execution_id.set(self.execution_id_value)
        self.user_token = user_id.set(self.user_id_value)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.correlation_token:
            correlation_id.reset(self.correlation_token)
        if self.execution_token:
            execution_id.reset(self.execution_token)
        if self.user_token:
            user_id.reset(self.user_token)


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger instance."""
    return StructuredLogger(name)


def get_correlation_id() -> str:
    """Get current correlation ID."""
    return correlation_id.get('')


def get_execution_id() -> str:
    """Get current execution ID."""
    return execution_id.get('')


def set_correlation_context(corr_id: str, exec_id: str = "", usr_id: str = ""):
    """Set correlation context variables."""
    correlation_id.set(corr_id)
    if exec_id:
        execution_id.set(exec_id)
    if usr_id:
        user_id.set(usr_id)


# Performance tracking decorator
def log_execution_time(operation: str):
    """Decorator to log execution time of functions."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000

                logger.info(
                    f"{operation} completed successfully",
                    operation=operation,
                    execution_time_ms=execution_time,
                    function=func.__name__
                )

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000

                logger.error(
                    f"{operation} failed",
                    operation=operation,
                    execution_time_ms=execution_time,
                    function=func.__name__,
                    error=e
                )

                raise

        def sync_wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000

                logger.info(
                    f"{operation} completed successfully",
                    operation=operation,
                    execution_time_ms=execution_time,
                    function=func.__name__
                )

                return result

            except Exception as e:
                execution_time = (time.time() - start_time) * 1000

                logger.error(
                    f"{operation} failed",
                    operation=operation,
                    execution_time_ms=execution_time,
                    function=func.__name__,
                    error=e
                )

                raise

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Initialize structured logging
configure_structured_logging()
