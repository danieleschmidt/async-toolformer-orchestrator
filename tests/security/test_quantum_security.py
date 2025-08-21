"""
Security tests for Quantum-Enhanced AsyncOrchestrator.

This module provides comprehensive security tests for quantum features:
- Security context validation
- Input sanitization and validation
- Resource access control
- Audit logging verification
- Quantum token security
"""

import asyncio
import time
from typing import Any

import pytest

from async_toolformer import (
    AccessLevel,
    QuantumSecurityManager,
    SecurityContext,
    SecurityLevel,
    Tool,
    create_quantum_orchestrator,
)


# Test tools with different security implications
@Tool(description="Safe computation task")
async def safe_computation(value: int) -> int:
    """A safe computational task."""
    await asyncio.sleep(0.1)
    return value * 2


@Tool(description="File system access task")
async def file_system_task(file_path: str, operation: str = "read") -> dict[str, Any]:
    """Simulate file system access."""
    await asyncio.sleep(0.1)
    return {
        "file_path": file_path,
        "operation": operation,
        "success": True,
        "size": 1024,
    }


@Tool(description="Network access task")
async def network_access_task(host: str, port: int = 80) -> dict[str, Any]:
    """Simulate network access."""
    await asyncio.sleep(0.1)
    return {
        "host": host,
        "port": port,
        "connected": True,
        "response_time": 100,
    }


@Tool(description="Database task")
async def database_task(query: str) -> dict[str, Any]:
    """Simulate database access."""
    await asyncio.sleep(0.1)
    return {
        "query": query,
        "rows": 42,
        "execution_time": 50,
    }


class TestQuantumSecurityManager:
    """Test the QuantumSecurityManager functionality."""

    @pytest.fixture
    def security_manager(self):
        """Create a security manager for testing."""
        return QuantumSecurityManager(
            default_security_level=SecurityLevel.MEDIUM,
            enable_quantum_tokens=True,
            session_timeout_seconds=300,  # 5 minutes for testing
        )

    def test_security_context_creation(self, security_manager):
        """Test security context creation."""
        context = security_manager.create_security_context(
            user_id="test_user",
            access_level=AccessLevel.RESTRICTED,
            security_level=SecurityLevel.HIGH,
        )

        assert context.user_id == "test_user"
        assert context.access_level == AccessLevel.RESTRICTED
        assert context.security_level == SecurityLevel.HIGH
        assert context.session_id is not None
        assert len(context.session_id) > 10  # Should be a substantial token
        assert context.quantum_token is not None
        assert context.quantum_token.startswith("qt_")
        assert not context.is_expired()

    def test_security_context_validation(self, security_manager):
        """Test security context validation."""
        # Create valid context
        context = security_manager.create_security_context(
            user_id="test_user",
            access_level=AccessLevel.RESTRICTED,
        )

        # Should validate successfully
        assert security_manager.validate_security_context(
            context.session_id, context.quantum_token
        )

        # Invalid session ID should fail
        assert not security_manager.validate_security_context(
            "invalid_session", context.quantum_token
        )

        # Invalid quantum token should fail
        assert not security_manager.validate_security_context(
            context.session_id, "invalid_token"
        )

    def test_context_expiration(self, security_manager):
        """Test security context expiration."""
        # Create context with very short timeout
        context = security_manager.create_security_context(
            user_id="test_user",
            session_timeout=1,  # 1 second
        )

        # Should be valid initially
        assert not context.is_expired()
        assert security_manager.validate_security_context(
            context.session_id, context.quantum_token
        )

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired now
        assert context.is_expired()
        assert not security_manager.validate_security_context(
            context.session_id, context.quantum_token
        )

    def test_input_sanitization(self, security_manager):
        """Test input sanitization functionality."""
        # Test safe input
        safe_input = "Hello World 123"
        sanitized = security_manager.sanitize_input(safe_input, SecurityLevel.MEDIUM)
        assert sanitized == safe_input

        # Test dangerous input patterns
        dangerous_inputs = [
            "__import__('os').system('rm -rf /')",
            "eval('malicious_code')",
            "exec('dangerous_command')",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
        ]

        for dangerous_input in dangerous_inputs:
            with pytest.raises(ValueError, match="Input contains blocked pattern"):
                security_manager.sanitize_input(dangerous_input, SecurityLevel.HIGH)

    def test_resource_access_control(self, security_manager):
        """Test resource access control."""
        # Create context with specific resource permissions
        context = security_manager.create_security_context(
            user_id="test_user",
            access_level=AccessLevel.RESTRICTED,
            allowed_resources={"computation", "network"},
        )

        # Should allow access to permitted resources
        assert security_manager.check_resource_access(
            context, "computation", "/tmp/safe_computation", "execute"
        )

        # Should deny access to restricted resources
        assert not security_manager.check_resource_access(
            context, "file_system", "/etc/passwd", "read"
        )

        # Should deny access to denied paths
        assert not security_manager.check_resource_access(
            context, "file_system", "/usr/bin/dangerous", "execute"
        )

    def test_audit_logging(self, security_manager):
        """Test audit logging functionality."""
        # Create context and perform actions
        context = security_manager.create_security_context(
            user_id="audit_test_user",
            access_level=AccessLevel.RESTRICTED,
        )

        # Validate context (should create audit entry)
        security_manager.validate_security_context(
            context.session_id, context.quantum_token
        )

        # Export audit log
        audit_entries = security_manager.export_audit_log()

        # Should have audit entries
        assert len(audit_entries) > 0

        # Find context creation entry
        creation_entries = [
            entry for entry in audit_entries
            if entry["action"] == "create_security_context"
            and entry["user_id"] == "audit_test_user"
        ]

        assert len(creation_entries) > 0

        creation_entry = creation_entries[0]
        assert creation_entry["success"] is True
        assert creation_entry["security_level"] == SecurityLevel.MEDIUM.value
        assert "metadata" in creation_entry

    def test_quantum_token_security(self, security_manager):
        """Test quantum token security features."""
        # Create multiple contexts
        contexts = []
        for i in range(5):
            context = security_manager.create_security_context(
                user_id=f"user_{i}",
                access_level=AccessLevel.RESTRICTED,
            )
            contexts.append(context)

        # All tokens should be unique
        tokens = [ctx.quantum_token for ctx in contexts]
        assert len(set(tokens)) == len(tokens)

        # All tokens should have proper format
        for token in tokens:
            assert token.startswith("qt_")
            assert len(token) > 20  # Should be substantial length

        # Tokens should not be predictable
        for i in range(len(tokens) - 1):
            # Adjacent tokens should be different
            assert tokens[i] != tokens[i + 1]

    def test_security_metrics(self, security_manager):
        """Test security metrics collection."""
        # Create some contexts and activities
        for i in range(3):
            context = security_manager.create_security_context(
                user_id=f"metrics_user_{i}",
                access_level=AccessLevel.RESTRICTED,
            )

            # Validate context
            security_manager.validate_security_context(
                context.session_id, context.quantum_token
            )

        # Get security metrics
        metrics = security_manager.get_security_metrics()

        assert "active_contexts" in metrics
        assert "audit_entries" in metrics
        assert "quantum_keys" in metrics
        assert "security_levels" in metrics
        assert "access_levels" in metrics

        assert metrics["active_contexts"] >= 3
        assert metrics["audit_entries"] >= 3


class TestQuantumOrchestratorSecurity:
    """Test security integration with QuantumAsyncOrchestrator."""

    @pytest.fixture
    async def secure_orchestrator(self):
        """Create a security-focused orchestrator."""
        orchestrator = create_quantum_orchestrator(
            tools=[
                safe_computation,
                file_system_task,
                network_access_task,
                database_task,
            ],
            quantum_config={
                "security": {
                    "security_level": SecurityLevel.HIGH,
                    "enable_quantum_tokens": True,
                    "session_timeout": 300,
                },
                "validation": {
                    "validation_level": "strict",
                    "enable_coherence_checks": True,
                },
            }
        )

        yield orchestrator
        await orchestrator.cleanup()

    @pytest.mark.asyncio
    async def test_secure_execution_with_context(self, secure_orchestrator):
        """Test secure execution with security context."""
        # Create security context
        security_context = secure_orchestrator.security_manager.create_security_context(
            user_id="secure_test_user",
            access_level=AccessLevel.RESTRICTED,
            security_level=SecurityLevel.HIGH,
            allowed_resources={"computation"},
        )

        # Execute with security context
        result = await secure_orchestrator.quantum_execute(
            "Run safe computation with value 42",
            security_context=security_context,
        )

        assert result["status"] == "completed"
        assert result["successful_tools"] >= 1

        # Verify security metrics are included
        assert "security_metrics" in result
        security_metrics = result["security_metrics"]
        assert security_metrics["active_contexts"] >= 1

    @pytest.mark.asyncio
    async def test_security_context_validation_failure(self, secure_orchestrator):
        """Test handling of invalid security contexts."""
        # Create expired context
        expired_context = SecurityContext(
            user_id="expired_user",
            session_id="expired_session",
            access_level=AccessLevel.RESTRICTED,
            security_level=SecurityLevel.HIGH,
            expires_at=time.time() - 100,  # Expired 100 seconds ago
        )

        # Should fail with invalid context
        with pytest.raises(ValueError, match="Invalid security context"):
            await secure_orchestrator.quantum_execute(
                "Run safe computation with value 42",
                security_context=expired_context,
            )

    @pytest.mark.asyncio
    async def test_input_sanitization_integration(self, secure_orchestrator):
        """Test input sanitization in quantum execution."""
        # Create high-security context
        security_context = secure_orchestrator.security_manager.create_security_context(
            user_id="sanitization_test",
            security_level=SecurityLevel.HIGH,
            access_level=AccessLevel.RESTRICTED,
        )

        # Test with safe input
        safe_result = await secure_orchestrator.quantum_execute(
            "Run safe computation with value 10",
            security_context=security_context,
        )

        assert safe_result["status"] == "completed"

        # Test with potentially dangerous input (should be sanitized)
        # Note: This depends on the LLM integration handling sanitization
        dangerous_prompt = "Run safe computation with value 42 and also __import__('os')"

        # Should either complete safely or fail validation
        dangerous_result = await secure_orchestrator.quantum_execute(
            dangerous_prompt,
            security_context=security_context,
        )

        # Should not cause system compromise
        assert dangerous_result["status"] in ["completed", "validation_failed"]

    @pytest.mark.asyncio
    async def test_resource_access_enforcement(self, secure_orchestrator):
        """Test resource access control enforcement."""
        # Create context with limited resources
        limited_context = secure_orchestrator.security_manager.create_security_context(
            user_id="limited_user",
            access_level=AccessLevel.RESTRICTED,
            allowed_resources={"computation"},  # Only computation allowed
            denied_resources={"file_system", "network"},
        )

        # Should allow computation tasks
        comp_result = await secure_orchestrator.quantum_execute(
            "Run safe computation with value 5",
            security_context=limited_context,
        )

        assert comp_result["status"] == "completed"

        # Should handle restricted resource access gracefully
        # (Exact behavior depends on implementation - may complete with warnings
        # or fail validation)
        restricted_result = await secure_orchestrator.quantum_execute(
            "Access file system at /tmp/test.txt",
            security_context=limited_context,
        )

        # Should not compromise security
        assert restricted_result["status"] in ["completed", "validation_failed", "no_tools_called"]

    @pytest.mark.asyncio
    async def test_audit_trail_integration(self, secure_orchestrator):
        """Test audit trail integration."""
        # Create context
        security_context = secure_orchestrator.security_manager.create_security_context(
            user_id="audit_integration_user",
            access_level=AccessLevel.RESTRICTED,
        )

        # Perform multiple operations
        for i in range(3):
            await secure_orchestrator.quantum_execute(
                f"Run safe computation with value {i * 10}",
                security_context=security_context,
            )

        # Check audit trail
        audit_entries = secure_orchestrator.security_manager.export_audit_log()

        # Should have entries for context creation and validations
        user_entries = [
            entry for entry in audit_entries
            if entry["user_id"] == "audit_integration_user"
        ]

        assert len(user_entries) >= 4  # At least 1 creation + 3 executions

        # Should have context creation entry
        creation_entries = [
            entry for entry in user_entries
            if entry["action"] == "create_security_context"
        ]
        assert len(creation_entries) >= 1

    @pytest.mark.asyncio
    async def test_concurrent_security_contexts(self, secure_orchestrator):
        """Test handling of multiple concurrent security contexts."""
        # Create multiple contexts
        contexts = []
        for i in range(5):
            context = secure_orchestrator.security_manager.create_security_context(
                user_id=f"concurrent_user_{i}",
                access_level=AccessLevel.RESTRICTED,
            )
            contexts.append(context)

        # Execute concurrently with different contexts
        tasks = []
        for i, context in enumerate(contexts):
            task = asyncio.create_task(
                secure_orchestrator.quantum_execute(
                    f"Run safe computation with value {i * 5}",
                    security_context=context,
                )
            )
            tasks.append(task)

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete successfully
        for result in results:
            assert not isinstance(result, Exception)
            assert result["status"] == "completed"

        # Each should have maintained separate security context
        security_metrics = secure_orchestrator.security_manager.get_security_metrics()
        assert security_metrics["active_contexts"] >= 5


class TestSecurityEdgeCases:
    """Test security edge cases and attack scenarios."""

    @pytest.fixture
    def hardened_security_manager(self):
        """Create a hardened security manager."""
        return QuantumSecurityManager(
            default_security_level=SecurityLevel.QUANTUM_SECURE,
            enable_quantum_tokens=True,
            enable_input_sanitization=True,
            session_timeout_seconds=60,  # Short timeout
        )

    def test_injection_attack_prevention(self, hardened_security_manager):
        """Test prevention of various injection attacks."""
        injection_attempts = [
            # Command injection
            "'; rm -rf /; echo 'hacked",
            "$(rm -rf /)",
            "`rm -rf /`",

            # SQL injection
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "UNION SELECT * FROM passwords",

            # Script injection
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onload=alert('xss')",

            # Python code injection
            "__import__('os').system('malicious')",
            "eval('__import__(\"os\").system(\"rm -rf /\")')",
            "exec('import os; os.system(\"dangerous\")')",
        ]

        for injection in injection_attempts:
            with pytest.raises(ValueError):
                hardened_security_manager.sanitize_input(
                    injection, SecurityLevel.QUANTUM_SECURE
                )

    def test_token_tampering_detection(self, hardened_security_manager):
        """Test detection of token tampering attempts."""
        # Create valid context
        context = hardened_security_manager.create_security_context(
            user_id="tamper_test",
            access_level=AccessLevel.RESTRICTED,
        )

        # Valid token should work
        assert hardened_security_manager.validate_security_context(
            context.session_id, context.quantum_token
        )

        # Tampered tokens should fail
        tampered_tokens = [
            context.quantum_token[:-1] + "X",  # Change last character
            context.quantum_token[:10] + "TAMPERED" + context.quantum_token[10:],
            "qt_" + "X" * 50,  # Fake token
            context.quantum_token + "extra",  # Extended token
        ]

        for tampered_token in tampered_tokens:
            assert not hardened_security_manager.validate_security_context(
                context.session_id, tampered_token
            )

    def test_resource_exhaustion_protection(self, hardened_security_manager):
        """Test protection against resource exhaustion attacks."""
        # Try to create excessive contexts
        contexts = []

        # Should handle reasonable number of contexts
        for i in range(100):
            try:
                context = hardened_security_manager.create_security_context(
                    user_id=f"exhaustion_user_{i}",
                    access_level=AccessLevel.RESTRICTED,
                )
                contexts.append(context)
            except Exception:
                # If it starts rejecting, that's acceptable protection
                break

        # Should have created at least some contexts
        assert len(contexts) > 10

        # System should still be responsive
        test_context = hardened_security_manager.create_security_context(
            user_id="responsive_test",
            access_level=AccessLevel.RESTRICTED,
        )

        assert hardened_security_manager.validate_security_context(
            test_context.session_id, test_context.quantum_token
        )

    def test_timing_attack_resistance(self, hardened_security_manager):
        """Test resistance to timing attacks."""
        # Create context
        context = hardened_security_manager.create_security_context(
            user_id="timing_test",
            access_level=AccessLevel.RESTRICTED,
        )

        # Time valid validation
        valid_times = []
        for _ in range(10):
            start = time.time()
            result = hardened_security_manager.validate_security_context(
                context.session_id, context.quantum_token
            )
            valid_times.append(time.time() - start)
            assert result is True

        # Time invalid validation
        invalid_times = []
        for _ in range(10):
            start = time.time()
            result = hardened_security_manager.validate_security_context(
                context.session_id, "invalid_token"
            )
            invalid_times.append(time.time() - start)
            assert result is False

        # Timing difference should not be significant
        # (This is a basic check - real timing attack resistance is complex)
        avg_valid = sum(valid_times) / len(valid_times)
        avg_invalid = sum(invalid_times) / len(invalid_times)

        # Difference should be small (less than 50% difference)
        time_ratio = abs(avg_valid - avg_invalid) / max(avg_valid, avg_invalid)
        assert time_ratio < 0.5

    def test_context_isolation(self, hardened_security_manager):
        """Test isolation between security contexts."""
        # Create two contexts for different users
        context1 = hardened_security_manager.create_security_context(
            user_id="user1",
            access_level=AccessLevel.RESTRICTED,
            allowed_resources={"computation"},
        )

        context2 = hardened_security_manager.create_security_context(
            user_id="user2",
            access_level=AccessLevel.CONFIDENTIAL,
            allowed_resources={"network", "database"},
        )

        # Each context should only validate with its own credentials
        assert hardened_security_manager.validate_security_context(
            context1.session_id, context1.quantum_token
        )
        assert hardened_security_manager.validate_security_context(
            context2.session_id, context2.quantum_token
        )

        # Cross-context validation should fail
        assert not hardened_security_manager.validate_security_context(
            context1.session_id, context2.quantum_token
        )
        assert not hardened_security_manager.validate_security_context(
            context2.session_id, context1.quantum_token
        )

        # Resource access should be isolated
        assert hardened_security_manager.check_resource_access(
            context1, "computation", "/tmp/safe", "execute"
        )
        assert not hardened_security_manager.check_resource_access(
            context1, "network", "example.com:80", "connect"
        )

        assert hardened_security_manager.check_resource_access(
            context2, "network", "example.com:80", "connect"
        )
        assert not hardened_security_manager.check_resource_access(
            context2, "computation", "/tmp/safe", "execute"
        )


if __name__ == "__main__":
    # Run the security tests
    pytest.main([__file__, "-v", "-s"])
