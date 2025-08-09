"""Simplified structured logging without external dependencies."""

import json
import logging
import time
import uuid
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict
import contextvars

# Correlation context variables
correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar('correlation_id', default='')
execution_id: contextvars.ContextVar[str] = contextvars.ContextVar('execution_id', default='')
user_id: contextvars.ContextVar[str] = contextvars.ContextVar('user_id', default='')


@dataclass
class LogContext:
    """Structured log context."""
    correlation_id: str = ""
    execution_id: str = ""
    user_id: str = ""
    component: str = ""
    timestamp: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.correlation_id:
            self.correlation_id = correlation_id.get('')


class StructuredLogger:
    """Simple structured logger."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Set up basic formatting
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with context."""
        if error:
            kwargs.update({
                "error_type": type(error).__name__,
                "error_message": str(error)
            })
        self._log(logging.ERROR, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal log method with context enrichment."""
        if kwargs:
            # Add structured data to message
            try:
                structured_data = json.dumps(kwargs, default=str)
                enhanced_message = f"{message} | {structured_data}"
            except Exception:
                enhanced_message = f"{message} | {kwargs}"
        else:
            enhanced_message = message
            
        self.logger.log(level, enhanced_message)


class CorrelationContext:
    """Context manager for correlation tracking."""
    
    def __init__(self,
                 correlation_id_value: Optional[str] = None,
                 execution_id_value: Optional[str] = None,
                 user_id_value: Optional[str] = None):
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
                    f"{operation} completed",
                    execution_time_ms=execution_time,
                    function=func.__name__
                )
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                logger.error(
                    f"{operation} failed",
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
                    f"{operation} completed",
                    execution_time_ms=execution_time,
                    function=func.__name__
                )
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                logger.error(
                    f"{operation} failed",
                    execution_time_ms=execution_time,
                    function=func.__name__,
                    error=e
                )
                
                raise
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator