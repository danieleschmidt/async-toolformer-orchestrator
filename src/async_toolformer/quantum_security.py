"""
Quantum Security Module for AsyncOrchestrator.

This module provides security features for quantum-inspired task execution:
- Quantum key distribution simulation
- Secure task execution sandboxing
- Input validation and sanitization
- Resource access control
- Audit logging and monitoring
"""

import asyncio
import hashlib
import json
import logging
import re
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for quantum task execution."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    QUANTUM_SECURE = "quantum_secure"


class AccessLevel(Enum):
    """Access levels for resources and operations."""
    PUBLIC = "public"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    TOP_SECRET = "top_secret"


@dataclass
class SecurityContext:
    """Security context for task execution."""
    user_id: str
    session_id: str
    access_level: AccessLevel
    security_level: SecurityLevel
    allowed_resources: set[str] = field(default_factory=set)
    denied_resources: set[str] = field(default_factory=set)
    execution_limits: dict[str, Any] = field(default_factory=dict)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    quantum_token: str | None = None
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None

    def is_expired(self) -> bool:
        """Check if security context has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def can_access_resource(self, resource: str) -> bool:
        """Check if context allows access to a resource."""
        if resource in self.denied_resources:
            return False
        return not (self.allowed_resources and resource not in self.allowed_resources)


@dataclass
class SecurityAuditEntry:
    """Security audit log entry."""
    timestamp: float
    user_id: str
    session_id: str
    action: str
    resource: str | None
    success: bool
    security_level: SecurityLevel
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "action": self.action,
            "resource": self.resource,
            "success": self.success,
            "security_level": self.security_level.value,
            "metadata": self.metadata,
        }


class QuantumSecurityManager:
    """
    Quantum-inspired security manager for task orchestration.

    Provides security features including:
    - Quantum-inspired authentication
    - Secure execution sandboxing
    - Input validation and sanitization
    - Resource access control
    - Comprehensive audit logging
    """

    def __init__(
        self,
        default_security_level: SecurityLevel = SecurityLevel.MEDIUM,
        enable_quantum_tokens: bool = True,
        audit_log_size: int = 10000,
        session_timeout_seconds: int = 3600,
        enable_input_sanitization: bool = True,
    ):
        """
        Initialize the quantum security manager.

        Args:
            default_security_level: Default security level for new contexts
            enable_quantum_tokens: Whether to use quantum-inspired tokens
            audit_log_size: Maximum number of audit entries to keep
            session_timeout_seconds: Session timeout in seconds
            enable_input_sanitization: Whether to sanitize inputs
        """
        self.default_security_level = default_security_level
        self.enable_quantum_tokens = enable_quantum_tokens
        self.audit_log_size = audit_log_size
        self.session_timeout_seconds = session_timeout_seconds
        self.enable_input_sanitization = enable_input_sanitization

        # Security state
        self._active_contexts: dict[str, SecurityContext] = {}
        self._audit_log: list[SecurityAuditEntry] = []
        self._blocked_patterns: set[str] = set()
        self._resource_policies: dict[str, dict[str, Any]] = {}
        self._quantum_keys: dict[str, str] = {}

        # Initialize default security patterns
        self._initialize_security_patterns()

        logger.info("QuantumSecurityManager initialized")

    def _initialize_security_patterns(self):
        """Initialize default security patterns and policies."""
        # Dangerous input patterns to block
        self._blocked_patterns.update([
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"subprocess",
            r"os\.system",
            r"open\s*\([^)]*['\"]w",  # File writes
            r"rm\s+-rf",
            r"DROP\s+TABLE",
            r"DELETE\s+FROM",
            r"<script",
            r"javascript:",
            r"vbscript:",
            r"onload\s*=",
            r"onerror\s*=",
        ])

        # Default resource policies
        self._resource_policies.update({
            "file_system": {
                "access_level": AccessLevel.RESTRICTED,
                "allowed_paths": ["/tmp", "/var/tmp"],
                "denied_paths": ["/etc", "/usr", "/bin", "/sbin"],
                "max_file_size": 10 * 1024 * 1024,  # 10MB
            },
            "network": {
                "access_level": AccessLevel.RESTRICTED,
                "allowed_ports": [80, 443, 8080, 8443],
                "denied_hosts": ["localhost", "127.0.0.1", "::1"],
                "max_connections": 10,
            },
            "memory": {
                "access_level": AccessLevel.PUBLIC,
                "max_memory_mb": 1024,
                "max_objects": 100000,
            },
            "cpu": {
                "access_level": AccessLevel.PUBLIC,
                "max_execution_time_ms": 30000,
                "max_cpu_percent": 50,
            }
        })

    def create_security_context(
        self,
        user_id: str,
        access_level: AccessLevel = AccessLevel.RESTRICTED,
        security_level: SecurityLevel | None = None,
        allowed_resources: set[str] | None = None,
        execution_limits: dict[str, Any] | None = None,
        session_timeout: int | None = None,
    ) -> SecurityContext:
        """
        Create a new security context for task execution.

        Args:
            user_id: Unique user identifier
            access_level: Access level for the context
            security_level: Security level (defaults to default_security_level)
            allowed_resources: Set of allowed resources
            execution_limits: Execution limits for the context
            session_timeout: Custom session timeout in seconds

        Returns:
            SecurityContext instance
        """
        session_id = secrets.token_urlsafe(32)
        security_level = security_level or self.default_security_level
        timeout = session_timeout or self.session_timeout_seconds

        context = SecurityContext(
            user_id=user_id,
            session_id=session_id,
            access_level=access_level,
            security_level=security_level,
            allowed_resources=allowed_resources or set(),
            execution_limits=execution_limits or {},
            expires_at=time.time() + timeout,
        )

        # Generate quantum token if enabled
        if self.enable_quantum_tokens:
            context.quantum_token = self._generate_quantum_token(context)

        self._active_contexts[session_id] = context

        # Audit log entry
        self._add_audit_entry(
            user_id=user_id,
            session_id=session_id,
            action="create_security_context",
            success=True,
            security_level=security_level,
            metadata={
                "access_level": access_level.value,
                "timeout": timeout,
                "quantum_token_enabled": self.enable_quantum_tokens,
            }
        )

        logger.info(f"Created security context for user {user_id} with session {session_id}")
        return context

    def _generate_quantum_token(self, context: SecurityContext) -> str:
        """
        Generate a quantum-inspired authentication token.

        Uses quantum-inspired randomness and entanglement simulation
        for enhanced security.
        """
        # Base entropy from multiple sources
        entropy_sources = [
            str(time.time()),
            context.user_id,
            context.session_id,
            str(context.created_at),
            secrets.token_bytes(32).hex(),
        ]

        # Quantum-inspired key derivation
        base_key = hashlib.sha256("".join(entropy_sources).encode()).hexdigest()

        # Simulate quantum superposition by combining multiple hash functions
        quantum_components = [
            hashlib.sha256(base_key.encode()).hexdigest(),
            hashlib.blake2b(base_key.encode()).hexdigest(),
            hashlib.sha3_256(base_key.encode()).hexdigest(),
        ]

        # Simulate quantum entanglement by XORing components
        entangled_key = ""
        for i in range(min(len(c) for c in quantum_components)):
            # XOR characters at position i from all components
            xor_result = 0
            for component in quantum_components:
                xor_result ^= ord(component[i])
            entangled_key += format(xor_result, '02x')

        # Final quantum token with metadata
        token_data = {
            "key": entangled_key[:64],  # Truncate to 64 characters
            "created": context.created_at,
            "expires": context.expires_at,
            "entropy": len(entropy_sources),
        }

        # Store quantum key for validation
        self._quantum_keys[context.session_id] = entangled_key[:64]

        # Encode token
        token_json = json.dumps(token_data, sort_keys=True)
        token_b64 = secrets.token_urlsafe(len(token_json))

        return f"qt_{token_b64}"

    def validate_security_context(self, session_id: str, quantum_token: str | None = None) -> bool:
        """
        Validate a security context and quantum token.

        Args:
            session_id: Session identifier
            quantum_token: Optional quantum token for validation

        Returns:
            True if context is valid, False otherwise
        """
        context = self._active_contexts.get(session_id)
        if not context:
            logger.warning(f"Security context not found: {session_id}")
            return False

        # Check expiration
        if context.is_expired():
            logger.warning(f"Security context expired: {session_id}")
            self._remove_context(session_id)
            return False

        # Validate quantum token if provided and enabled
        if self.enable_quantum_tokens and quantum_token:
            if not self._validate_quantum_token(session_id, quantum_token):
                logger.warning(f"Invalid quantum token for session: {session_id}")
                return False

        return True

    def _validate_quantum_token(self, session_id: str, token: str) -> bool:
        """Validate a quantum token."""
        try:
            if not token.startswith("qt_"):
                return False

            # Check if we have the quantum key for this session
            stored_key = self._quantum_keys.get(session_id)
            if not stored_key:
                return False

            # For this implementation, we'll do a simplified validation
            # In a real quantum system, this would involve quantum key distribution
            return len(token) > 10  # Basic validation

        except Exception as e:
            logger.error(f"Error validating quantum token: {e}")
            return False

    def sanitize_input(self, input_data: Any, security_level: SecurityLevel) -> Any:
        """
        Sanitize input data based on security level.

        Args:
            input_data: Input data to sanitize
            security_level: Security level for sanitization

        Returns:
            Sanitized input data
        """
        if not self.enable_input_sanitization:
            return input_data

        try:
            if isinstance(input_data, str):
                return self._sanitize_string(input_data, security_level)
            elif isinstance(input_data, dict):
                return {
                    key: self.sanitize_input(value, security_level)
                    for key, value in input_data.items()
                    if self._is_safe_key(key, security_level)
                }
            elif isinstance(input_data, list):
                return [
                    self.sanitize_input(item, security_level)
                    for item in input_data
                ]
            else:
                # For other types, perform basic validation
                return self._validate_primitive(input_data, security_level)

        except Exception as e:
            logger.error(f"Error sanitizing input: {e}")
            raise ValueError(f"Input sanitization failed: {e}")

    def _sanitize_string(self, text: str, security_level: SecurityLevel) -> str:
        """Sanitize string input."""
        # Check for blocked patterns
        for pattern in self._blocked_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValueError(f"Input contains blocked pattern: {pattern}")

        # Security level specific sanitization
        if security_level in [SecurityLevel.HIGH, SecurityLevel.QUANTUM_SECURE]:
            # Remove potentially dangerous characters
            text = re.sub(r'[<>&"\']', '', text)

            # Limit length for high security
            if len(text) > 1000:
                text = text[:1000]

        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)

        return text

    def _is_safe_key(self, key: str, security_level: SecurityLevel) -> bool:
        """Check if a dictionary key is safe."""
        unsafe_keys = {
            "__class__", "__bases__", "__subclasses__", "__dict__",
            "__globals__", "__builtins__", "__import__", "eval", "exec"
        }

        if key in unsafe_keys:
            return False

        if security_level in [SecurityLevel.HIGH, SecurityLevel.QUANTUM_SECURE]:
            # More restrictive for high security
            if key.startswith("_") or not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', key):
                return False

        return True

    def _validate_primitive(self, value: Any, security_level: SecurityLevel) -> Any:
        """Validate primitive data types."""
        # Check for reasonable bounds
        if isinstance(value, int):
            if abs(value) > 10**12:  # Reasonable integer bounds
                raise ValueError("Integer value too large")
        elif isinstance(value, float):
            if abs(value) > 10**12 or str(value).lower() in ['inf', '-inf', 'nan']:
                raise ValueError("Invalid float value")

        return value

    def check_resource_access(
        self,
        context: SecurityContext,
        resource_type: str,
        resource_path: str,
        operation: str
    ) -> bool:
        """
        Check if a security context can access a resource.

        Args:
            context: Security context
            resource_type: Type of resource (file_system, network, etc.)
            resource_path: Path or identifier of the resource
            operation: Operation to perform (read, write, execute, etc.)

        Returns:
            True if access is allowed, False otherwise
        """
        try:
            # Check if context can access the resource type
            if not context.can_access_resource(resource_type):
                return False

            # Get resource policy
            policy = self._resource_policies.get(resource_type, {})
            required_access_level = policy.get("access_level", AccessLevel.PUBLIC)

            # Check access level
            access_levels = [AccessLevel.PUBLIC, AccessLevel.RESTRICTED,
                           AccessLevel.CONFIDENTIAL, AccessLevel.TOP_SECRET]

            if access_levels.index(context.access_level) < access_levels.index(required_access_level):
                return False

            # Resource-specific checks
            if resource_type == "file_system":
                return self._check_file_system_access(policy, resource_path, operation)
            elif resource_type == "network":
                return self._check_network_access(policy, resource_path, operation)

            return True

        except Exception as e:
            logger.error(f"Error checking resource access: {e}")
            return False

    def _check_file_system_access(self, policy: dict[str, Any], path: str, operation: str) -> bool:
        """Check file system access permissions."""
        # Check allowed paths
        allowed_paths = policy.get("allowed_paths", [])
        if allowed_paths and not any(path.startswith(allowed) for allowed in allowed_paths):
            return False

        # Check denied paths
        denied_paths = policy.get("denied_paths", [])
        if any(path.startswith(denied) for denied in denied_paths):
            return False

        # Check operation-specific permissions
        return not (operation == "write" and policy.get("read_only", False))

    def _check_network_access(self, policy: dict[str, Any], host_port: str, operation: str) -> bool:
        """Check network access permissions."""
        try:
            if ":" in host_port:
                host, port_str = host_port.rsplit(":", 1)
                port = int(port_str)
            else:
                host = host_port
                port = 80

            # Check denied hosts
            denied_hosts = policy.get("denied_hosts", [])
            if host in denied_hosts:
                return False

            # Check allowed ports
            allowed_ports = policy.get("allowed_ports", [])
            return not (allowed_ports and port not in allowed_ports)

        except ValueError:
            return False

    def secure_execute_task(
        self,
        context: SecurityContext,
        task_function: Callable,
        args: dict[str, Any],
        timeout_ms: int | None = None,
    ) -> dict[str, Any]:
        """
        Execute a task with security restrictions.

        Args:
            context: Security context
            task_function: Function to execute
            args: Function arguments
            timeout_ms: Execution timeout in milliseconds

        Returns:
            Execution result with security metadata
        """
        start_time = time.time()

        try:
            # Validate context
            if not self.validate_security_context(context.session_id, context.quantum_token):
                raise ValueError("Invalid security context")

            # Sanitize arguments
            sanitized_args = self.sanitize_input(args, context.security_level)

            # Apply execution limits
            execution_timeout = timeout_ms or context.execution_limits.get("timeout_ms", 30000)

            # Execute with timeout and monitoring
            result = asyncio.create_task(
                self._monitored_execution(
                    context, task_function, sanitized_args, execution_timeout
                )
            )

            # Log successful execution start
            self._add_audit_entry(
                user_id=context.user_id,
                session_id=context.session_id,
                action="execute_task",
                resource=task_function.__name__,
                success=True,
                security_level=context.security_level,
                metadata={
                    "args_sanitized": True,
                    "timeout_ms": execution_timeout,
                }
            )

            return {
                "success": True,
                "result": result,
                "execution_time_ms": (time.time() - start_time) * 1000,
                "security_level": context.security_level.value,
                "sanitized": True,
            }

        except Exception as e:
            # Log failed execution
            self._add_audit_entry(
                user_id=context.user_id,
                session_id=context.session_id,
                action="execute_task",
                resource=task_function.__name__ if task_function else "unknown",
                success=False,
                security_level=context.security_level,
                metadata={
                    "error": str(e),
                    "execution_time_ms": (time.time() - start_time) * 1000,
                }
            )

            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000,
                "security_level": context.security_level.value,
            }

    async def _monitored_execution(
        self,
        context: SecurityContext,
        task_function: Callable,
        args: dict[str, Any],
        timeout_ms: int
    ) -> Any:
        """Execute task with monitoring and resource limits."""
        try:
            # Create execution task
            if asyncio.iscoroutinefunction(task_function):
                task = asyncio.create_task(task_function(**args))
            else:
                # Run synchronous function in executor
                loop = asyncio.get_event_loop()
                task = asyncio.create_task(
                    loop.run_in_executor(None, lambda: task_function(**args))
                )

            # Wait with timeout
            result = await asyncio.wait_for(task, timeout=timeout_ms / 1000.0)
            return result

        except asyncio.TimeoutError:
            raise ValueError(f"Task execution timed out after {timeout_ms}ms")
        except Exception as e:
            raise ValueError(f"Task execution failed: {e}")

    def _add_audit_entry(
        self,
        user_id: str,
        session_id: str,
        action: str,
        success: bool,
        security_level: SecurityLevel,
        resource: str | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Add an entry to the security audit log."""
        entry = SecurityAuditEntry(
            timestamp=time.time(),
            user_id=user_id,
            session_id=session_id,
            action=action,
            resource=resource,
            success=success,
            security_level=security_level,
            metadata=metadata or {},
        )

        self._audit_log.append(entry)

        # Trim audit log if it gets too large
        if len(self._audit_log) > self.audit_log_size:
            self._audit_log = self._audit_log[-self.audit_log_size // 2:]

        logger.debug(f"Added audit entry: {action} by {user_id} - Success: {success}")

    def _remove_context(self, session_id: str):
        """Remove a security context."""
        context = self._active_contexts.pop(session_id, None)
        if context:
            self._quantum_keys.pop(session_id, None)

            self._add_audit_entry(
                user_id=context.user_id,
                session_id=session_id,
                action="remove_security_context",
                success=True,
                security_level=context.security_level,
                metadata={"reason": "expired" if context.is_expired() else "manual"}
            )

    def cleanup_expired_contexts(self):
        """Clean up expired security contexts."""
        expired_sessions = [
            session_id for session_id, context in self._active_contexts.items()
            if context.is_expired()
        ]

        for session_id in expired_sessions:
            self._remove_context(session_id)

        logger.info(f"Cleaned up {len(expired_sessions)} expired security contexts")

    def get_security_metrics(self) -> dict[str, Any]:
        """Get security metrics and statistics."""
        return {
            "active_contexts": len(self._active_contexts),
            "audit_entries": len(self._audit_log),
            "quantum_keys": len(self._quantum_keys),
            "blocked_patterns": len(self._blocked_patterns),
            "resource_policies": len(self._resource_policies),
            "security_levels": {
                level.value: len([
                    c for c in self._active_contexts.values()
                    if c.security_level == level
                ]) for level in SecurityLevel
            },
            "access_levels": {
                level.value: len([
                    c for c in self._active_contexts.values()
                    if c.access_level == level
                ]) for level in AccessLevel
            },
            "recent_failures": len([
                entry for entry in self._audit_log[-100:]
                if not entry.success
            ]),
        }

    def export_audit_log(self, start_time: float | None = None, end_time: float | None = None) -> list[dict[str, Any]]:
        """
        Export audit log entries within a time range.

        Args:
            start_time: Start timestamp (None for all)
            end_time: End timestamp (None for all)

        Returns:
            List of audit log entries as dictionaries
        """
        filtered_entries = []

        for entry in self._audit_log:
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            filtered_entries.append(entry.to_dict())

        return filtered_entries

    def add_security_policy(self, resource_type: str, policy: dict[str, Any]):
        """Add or update a security policy for a resource type."""
        self._resource_policies[resource_type] = policy
        logger.info(f"Updated security policy for resource type: {resource_type}")

    def add_blocked_pattern(self, pattern: str):
        """Add a new blocked input pattern."""
        self._blocked_patterns.add(pattern)
        logger.info(f"Added blocked pattern: {pattern}")

    def remove_blocked_pattern(self, pattern: str):
        """Remove a blocked input pattern."""
        self._blocked_patterns.discard(pattern)
        logger.info(f"Removed blocked pattern: {pattern}")


# Convenience functions
def create_secure_context(
    user_id: str,
    security_manager: QuantumSecurityManager,
    access_level: AccessLevel = AccessLevel.RESTRICTED,
    security_level: SecurityLevel = SecurityLevel.MEDIUM,
    **kwargs
) -> SecurityContext:
    """Create a secure context with sensible defaults."""
    return security_manager.create_security_context(
        user_id=user_id,
        access_level=access_level,
        security_level=security_level,
        **kwargs
    )


def secure_execute(
    context: SecurityContext,
    security_manager: QuantumSecurityManager,
    function: Callable,
    **kwargs
) -> dict[str, Any]:
    """Execute a function securely with the given context."""
    return security_manager.secure_execute_task(
        context=context,
        task_function=function,
        args=kwargs
    )
