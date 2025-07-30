# Advanced Security Practices

Comprehensive security framework for the Async Toolformer Orchestrator, designed for production-grade security requirements in LLM orchestration systems.

## Security Architecture

### Threat Model
Primary security concerns for LLM orchestration:
- **API Key Exposure**: Unauthorized access to LLM provider APIs
- **Data Exfiltration**: Sensitive information leakage through tool calls
- **Injection Attacks**: Malicious prompts compromising system integrity  
- **Rate Limit Bypass**: Unauthorized resource consumption
- **Supply Chain**: Compromised dependencies or container images

### Security Layers
1. **Input Security**: Prompt sanitization and validation
2. **Runtime Security**: Secure execution environment and isolation
3. **Data Security**: Encryption at rest and in transit
4. **Access Control**: Authentication, authorization, and audit
5. **Infrastructure Security**: Container and deployment hardening

## Enhanced Security Controls

### 1. API Key Management

#### Secure Key Storage
Never store API keys in code or configuration files:

```python
# src/async_toolformer/security/key_manager.py
"""Secure API key management with rotation support."""

import os
from typing import Optional
from cryptography.fernet import Fernet
from azure.keyvault.secrets import SecretClient  # Example cloud provider

class SecureKeyManager:
    """Manages API keys with encryption and rotation."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.cipher_suite = Fernet(encryption_key or self._generate_key())
        self.vault_client = self._init_vault_client()
    
    def get_api_key(self, provider: str) -> str:
        """Retrieve decrypted API key for provider."""
        encrypted_key = os.environ.get(f"{provider.upper()}_API_KEY_ENCRYPTED")
        if not encrypted_key:
            raise SecurityError(f"No encrypted key found for {provider}")
        return self.cipher_suite.decrypt(encrypted_key.encode()).decode()
    
    def rotate_key(self, provider: str) -> None:
        """Rotate API key with zero-downtime."""
        # Implementation for key rotation
```

#### Key Rotation Strategy
Automated API key rotation:

```yaml
# config/security/key-rotation-schedule.yml
rotation_schedule:
  openai:
    frequency: "30d"
    overlap_period: "24h"
    notification_threshold: "7d"
  anthropic:
    frequency: "30d" 
    overlap_period: "24h"
    notification_threshold: "7d"
```

### 2. Input Validation and Sanitization

#### Prompt Injection Prevention
Multi-layered prompt security:

```python
# src/async_toolformer/security/prompt_security.py
"""Advanced prompt injection prevention."""

import re
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SecurityPolicy:
    """Security policy for prompt processing."""
    max_prompt_length: int = 10000
    forbidden_patterns: List[str] = None
    required_patterns: List[str] = None
    sanitization_rules: Dict[str, str] = None

class PromptSecurityValidator:
    """Validates and sanitizes prompts against injection attacks."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.injection_patterns = self._load_injection_patterns()
    
    def validate_prompt(self, prompt: str) -> bool:
        """Validate prompt against security policy."""
        if len(prompt) > self.policy.max_prompt_length:
            raise SecurityError("Prompt exceeds maximum length")
        
        for pattern in self.injection_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                raise SecurityError(f"Potential injection detected: {pattern}")
        
        return True
    
    def sanitize_prompt(self, prompt: str) -> str:
        """Sanitize prompt removing potentially dangerous content."""
        sanitized = prompt
        
        # Remove potential system prompts
        sanitized = re.sub(r'system\s*:', '', sanitized, flags=re.IGNORECASE)
        
        # Remove command injection attempts
        sanitized = re.sub(r'[`$]{1,2}[^`$]*[`$]{1,2}', '', sanitized)
        
        # Apply custom sanitization rules
        if self.policy.sanitization_rules:
            for pattern, replacement in self.policy.sanitization_rules.items():
                sanitized = re.sub(pattern, replacement, sanitized)
        
        return sanitized.strip()
    
    def _load_injection_patterns(self) -> List[str]:
        """Load known injection patterns."""
        return [
            r'ignore\s+previous\s+instructions',
            r'system\s*:\s*you\s+are\s+now',
            r'\\n\\n#\s*new\s+instructions',
            r'act\s+as\s+if\s+you\s+are',
            r'pretend\s+you\s+are\s+a',
            # Add more patterns based on threat intelligence
        ]
```

### 3. Runtime Security

#### Secure Execution Environment
Containerized isolation with security constraints:

```dockerfile
# Dockerfile.security
FROM python:3.11-slim

# Create non-root user
RUN groupadd -r asyncuser && useradd -r -g asyncuser asyncuser

# Install security tools
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set security-hardened environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Copy application with restricted permissions
COPY --chown=asyncuser:asyncuser . /app
WORKDIR /app

# Switch to non-root user
USER asyncuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run with restricted capabilities
CMD ["python", "-m", "async_toolformer", "--security-mode=strict"]
```

#### Resource Limits and Monitoring
Prevent resource exhaustion attacks:

```python
# src/async_toolformer/security/resource_monitor.py
"""Resource monitoring and enforcement."""

import psutil
import asyncio
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class ResourceLimits:
    """Resource consumption limits."""
    max_memory_mb: int = 512
    max_cpu_percent: int = 80
    max_open_files: int = 1000
    max_concurrent_tasks: int = 100

class ResourceMonitor:
    """Monitors and enforces resource consumption limits."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.current_tasks = 0
        self._monitoring = False
    
    async def start_monitoring(self):
        """Start continuous resource monitoring."""
        self._monitoring = True
        while self._monitoring:
            await self._check_resources()
            await asyncio.sleep(1)
    
    async def _check_resources(self):
        """Check current resource usage against limits."""
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        open_files = len(psutil.Process().open_files())
        
        if memory_usage > self.limits.max_memory_mb:
            raise SecurityError("Memory limit exceeded")
        
        if cpu_usage > self.limits.max_cpu_percent:
            raise SecurityError("CPU limit exceeded")
        
        if open_files > self.limits.max_open_files:
            raise SecurityError("File descriptor limit exceeded")
        
        if self.current_tasks > self.limits.max_concurrent_tasks:
            raise SecurityError("Task limit exceeded")
```

### 4. Data Protection

#### Encryption Implementation
Comprehensive encryption strategy:

```python
# src/async_toolformer/security/encryption.py
"""Data encryption utilities."""

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import secrets
import base64

class DataEncryption:
    """Handles data encryption/decryption operations."""
    
    def __init__(self, password: str):
        self.key = self._derive_key(password.encode())
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        iv = secrets.token_bytes(16)
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # Pad data to block size
        padded_data = self._pad_data(data.encode())
        encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
        
        # Combine IV and encrypted data
        combined = iv + encrypted_data
        return base64.b64encode(combined).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        combined = base64.b64decode(encrypted_data.encode())
        iv = combined[:16]
        ciphertext = combined[16:]
        
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        
        padded_data = decryptor.update(ciphertext) + decryptor.finalize()
        return self._unpad_data(padded_data).decode()
```

#### Secure Logging
Prevent sensitive data leakage in logs:

```python
# src/async_toolformer/security/secure_logging.py
"""Secure logging with sensitive data filtering."""

import re
import logging
from typing import List, Pattern

class SecureLogFilter(logging.Filter):
    """Filter sensitive information from log messages."""
    
    def __init__(self):
        super().__init__()
        self.sensitive_patterns = self._compile_patterns([
            r'sk-[a-zA-Z0-9]{48}',  # OpenAI API keys
            r'sk-ant-[a-zA-Z0-9-]{95}',  # Anthropic API keys
            r'"password"\s*:\s*"[^"]*"',  # Passwords in JSON
            r'Bearer\s+[a-zA-Z0-9\-._~+/]+=*',  # Bearer tokens
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
        ])
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive data from log record."""
        if hasattr(record, 'msg'):
            record.msg = self._sanitize_message(str(record.msg))
        
        if hasattr(record, 'args'):
            record.args = tuple(
                self._sanitize_message(str(arg)) if isinstance(arg, str) else arg
                for arg in record.args
            )
        
        return True
    
    def _sanitize_message(self, message: str) -> str:
        """Remove sensitive data from message."""
        sanitized = message
        for pattern in self.sensitive_patterns:
            sanitized = pattern.sub('[REDACTED]', sanitized)
        return sanitized
    
    def _compile_patterns(self, patterns: List[str]) -> List[Pattern]:
        """Compile regex patterns for efficiency."""
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
```

### 5. Access Control and Audit

#### Role-Based Access Control
Granular permission system:

```python
# src/async_toolformer/security/access_control.py
"""Role-based access control system."""

from enum import Enum
from typing import Set, Dict, Any
from dataclasses import dataclass

class Permission(Enum):
    """System permissions."""
    EXECUTE_TOOLS = "execute_tools"
    CONFIGURE_ORCHESTRATOR = "configure_orchestrator"
    VIEW_METRICS = "view_metrics"
    MANAGE_KEYS = "manage_keys"
    ADMIN_ACCESS = "admin_access"

@dataclass
class Role:
    """User role with permissions."""
    name: str
    permissions: Set[Permission]
    description: str

class AccessControlManager:
    """Manages user access and permissions."""
    
    def __init__(self):
        self.roles = self._initialize_roles()
        self.user_roles: Dict[str, Set[str]] = {}
    
    def check_permission(self, user_id: str, permission: Permission) -> bool:
        """Check if user has specific permission."""
        user_roles = self.user_roles.get(user_id, set())
        
        for role_name in user_roles:
            role = self.roles.get(role_name)
            if role and permission in role.permissions:
                return True
        
        return False
    
    def assign_role(self, user_id: str, role_name: str) -> None:
        """Assign role to user."""
        if role_name not in self.roles:
            raise SecurityError(f"Role {role_name} does not exist")
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role_name)
        self._audit_log(f"Assigned role {role_name} to user {user_id}")
    
    def _initialize_roles(self) -> Dict[str, Role]:
        """Initialize default roles."""
        return {
            "user": Role(
                name="user",
                permissions={Permission.EXECUTE_TOOLS, Permission.VIEW_METRICS},
                description="Standard user with basic access"
            ),
            "admin": Role(
                name="admin", 
                permissions=set(Permission),
                description="Administrator with full access"
            ),
            "service": Role(
                name="service",
                permissions={Permission.EXECUTE_TOOLS},
                description="Service account for automated systems"
            )
        }
```

### 6. Security Monitoring

#### Real-time Security Monitoring
Continuous security event monitoring:

```python
# src/async_toolformer/security/security_monitor.py
"""Real-time security monitoring and alerting."""

import asyncio
from typing import Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class SecurityEvent:
    """Security event data structure."""
    event_type: str
    severity: str
    user_id: str
    details: Dict[str, Any]
    timestamp: datetime

class SecurityMonitor:
    """Monitors security events and triggers alerts."""
    
    def __init__(self):
        self.event_handlers: Dict[str, Callable] = {}
        self.rate_limits: Dict[str, Dict] = {}
        self.failed_attempts: Dict[str, int] = {}
    
    async def log_security_event(self, event: SecurityEvent) -> None:
        """Log and process security event."""
        # Store event
        await self._store_event(event)
        
        # Check for suspicious patterns
        if await self._is_suspicious_activity(event):
            await self._trigger_alert(event)
        
        # Update rate limiting counters
        await self._update_rate_limits(event)
    
    async def _is_suspicious_activity(self, event: SecurityEvent) -> bool:
        """Detect suspicious activity patterns."""
        user_id = event.user_id
        
        # Check for multiple failed authentication attempts
        if event.event_type == "auth_failure":
            self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
            if self.failed_attempts[user_id] >= 5:
                return True
        
        # Check for unusual access patterns
        if event.event_type == "api_access":
            # Implement anomaly detection logic
            pass
        
        return False
    
    async def _trigger_alert(self, event: SecurityEvent) -> None:
        """Trigger security alert."""
        alert_data = {
            "event": event,
            "timestamp": datetime.utcnow(),
            "action_required": True
        }
        
        # Send to security team
        await self._send_security_alert(alert_data)
        
        # Implement automatic response if needed
        if event.severity == "critical":
            await self._automatic_response(event)
```

## Compliance Framework

### 1. Regulatory Compliance

#### GDPR Compliance
Data protection and privacy:

```python
# src/async_toolformer/compliance/gdpr.py
"""GDPR compliance utilities."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta

class GDPRCompliance:
    """Handles GDPR compliance requirements."""
    
    def __init__(self):
        self.data_retention_periods = {
            "logs": timedelta(days=365),
            "metrics": timedelta(days=730),
            "user_data": timedelta(days=2555)  # 7 years
        }
    
    def process_data_deletion_request(self, user_id: str) -> Dict[str, Any]:
        """Process right to erasure request."""
        deletion_report = {
            "user_id": user_id,
            "deletion_timestamp": datetime.utcnow(),
            "deleted_data_types": [],
            "retention_exceptions": []
        }
        
        # Delete user data across all systems
        # Implementation details...
        
        return deletion_report
    
    def generate_data_export(self, user_id: str) -> Dict[str, Any]:
        """Generate data export for data portability."""
        # Implementation for data export
        pass
```

### 2. SOC 2 Type II Readiness
Security controls documentation:

```yaml
# docs/compliance/soc2-controls.yml
controls:
  CC6.1:
    name: "Logical and Physical Access Controls"
    implementation: "Multi-factor authentication, role-based access"
    evidence: "Access logs, permission matrices"
    
  CC6.2:
    name: "System Access Controls"  
    implementation: "API key management, session controls"
    evidence: "Authentication logs, key rotation records"
    
  CC6.3:
    name: "Data Protection"
    implementation: "Encryption at rest and in transit"
    evidence: "Encryption certificates, key management logs"
```

## Security Testing Integration

### 1. Automated Security Testing
Continuous security validation:

```python
# tests/security/test_security_automation.py
"""Automated security testing suite."""

import pytest
from async_toolformer.security import SecurityValidator

class TestSecurityAutomation:
    """Comprehensive security testing."""
    
    @pytest.mark.security
    async def test_api_key_exposure_prevention(self):
        """Test that API keys are never exposed in logs or responses."""
        
    @pytest.mark.security  
    async def test_injection_attack_prevention(self):
        """Test prompt injection attack prevention."""
        
    @pytest.mark.security
    async def test_rate_limit_enforcement(self):
        """Test rate limiting cannot be bypassed."""
        
    @pytest.mark.security
    async def test_data_encryption_integrity(self):
        """Test data encryption and decryption integrity."""
```

### 2. Security Metrics
Track security posture:

```python
# src/async_toolformer/security/metrics.py
"""Security metrics collection."""

from prometheus_client import Counter, Histogram, Gauge

# Security metrics
security_events_total = Counter(
    'security_events_total',
    'Total security events',
    ['event_type', 'severity']
)

authentication_attempts_total = Counter(
    'authentication_attempts_total', 
    'Total authentication attempts',
    ['result']
)

api_key_rotations_total = Counter(
    'api_key_rotations_total',
    'Total API key rotations',
    ['provider']
)

security_scan_duration = Histogram(
    'security_scan_duration_seconds',
    'Security scan duration'
)
```

## Incident Response

### 1. Security Incident Playbook
Structured incident response:

```yaml
# docs/security/incident-response-playbook.yml
incidents:
  api_key_compromise:
    severity: "critical"
    response_time: "15 minutes"
    steps:
      - "Immediately rotate compromised keys"
      - "Audit usage logs for unauthorized access"
      - "Notify security team and stakeholders"
      - "Update security controls"
    
  data_breach:
    severity: "critical"
    response_time: "30 minutes"
    steps:
      - "Isolate affected systems"
      - "Assess scope of breach"
      - "Implement containment measures"
      - "Begin forensic analysis"
      - "Notify legal and compliance teams"
```

### 2. Automated Response
Immediate threat mitigation:

```python
# src/async_toolformer/security/incident_response.py
"""Automated incident response system."""

class IncidentResponseManager:
    """Manages automated responses to security incidents."""
    
    async def handle_security_incident(self, incident_type: str, details: Dict[str, Any]):
        """Handle security incident with automated response."""
        
        response_actions = {
            "api_key_compromise": self._handle_key_compromise,
            "rate_limit_violation": self._handle_rate_limit_violation,
            "injection_attempt": self._handle_injection_attempt,
            "data_breach": self._handle_data_breach
        }
        
        handler = response_actions.get(incident_type)
        if handler:
            await handler(details)
        else:
            await self._handle_unknown_incident(incident_type, details)
```

This advanced security framework provides production-grade security appropriate for a maturing LLM orchestration system handling sensitive data and API integrations.