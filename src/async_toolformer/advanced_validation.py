"""Advanced validation system for Generation 2 robustness."""

import re
import json
import html
import asyncio
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, Callable, Set, Tuple
import logging

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class ValidationCategory(Enum):
    """Categories of validation checks."""
    SECURITY = auto()
    PERFORMANCE = auto()
    CORRECTNESS = auto()
    COMPATIBILITY = auto()
    COMPLIANCE = auto()


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    field_path: str
    suggested_fix: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ValidationResult:
    """Result of validation process."""
    
    is_valid: bool
    issues: List[ValidationIssue]
    sanitized_data: Any
    warnings_count: int = 0
    errors_count: int = 0
    critical_count: int = 0
    
    def __post_init__(self):
        # Count issues by severity
        for issue in self.issues:
            if issue.severity == ValidationSeverity.WARNING:
                self.warnings_count += 1
            elif issue.severity == ValidationSeverity.ERROR:
                self.errors_count += 1
            elif issue.severity == ValidationSeverity.CRITICAL:
                self.critical_count += 1
                
        # Overall validity check
        self.is_valid = self.errors_count == 0 and self.critical_count == 0


class AdvancedValidator:
    """Advanced validation system with multiple validation strategies."""
    
    def __init__(self):
        self.validators: Dict[str, List[Callable]] = {}
        self.sanitizers: Dict[str, List[Callable]] = {}
        self.global_validators: List[Callable] = []
        
        # Security patterns
        self.security_patterns = {
            'sql_injection': [
                re.compile(r'(\bUNION\b.*\bSELECT\b)', re.IGNORECASE),
                re.compile(r'(\bOR\b.*\b1\s*=\s*1\b)', re.IGNORECASE),
                re.compile(r'(\bDROP\b.*\bTABLE\b)', re.IGNORECASE),
            ],
            'xss': [
                re.compile(r'<script\b[^>]*>(.*?)</script>', re.IGNORECASE | re.DOTALL),
                re.compile(r'javascript:', re.IGNORECASE),
                re.compile(r'on\w+\s*=', re.IGNORECASE),
            ],
            'code_injection': [
                re.compile(r'(__import__|exec|eval)\s*\(', re.IGNORECASE),
                re.compile(r'os\.system|subprocess\.(call|run|Popen)', re.IGNORECASE),
            ],
            'path_traversal': [
                re.compile(r'\.\.[\\/]'),
                re.compile(r'[\\/]\.\.[\\/]'),
                re.compile(r'^[\\/]\.\.'),
            ]
        }
        
        # Performance thresholds
        self.performance_limits = {
            'max_string_length': 1000000,  # 1MB
            'max_list_size': 10000,
            'max_dict_keys': 1000,
            'max_nesting_depth': 10,
        }
        
        # Compliance patterns
        self.compliance_patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'phone': re.compile(r'^\+?1?-?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$'),
            'ssn': re.compile(r'^\d{3}-?\d{2}-?\d{4}$'),
            'credit_card': re.compile(r'^[0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}$'),
        }
        
        self._setup_default_validators()
        
    def _setup_default_validators(self):
        """Setup default validation rules."""
        
        # Global security validator
        async def security_validator(data: Any, field_path: str = '') -> List[ValidationIssue]:
            issues = []
            
            if isinstance(data, str):
                # Check for security threats
                for threat_type, patterns in self.security_patterns.items():
                    for pattern in patterns:
                        if pattern.search(data):
                            issues.append(ValidationIssue(
                                category=ValidationCategory.SECURITY,
                                severity=ValidationSeverity.CRITICAL,
                                message=f"Potential {threat_type} detected",
                                field_path=field_path,
                                suggested_fix="Remove or escape potentially malicious content"
                            ))
                
                # Check for PII exposure
                pii_issues = self._check_pii_exposure(data, field_path)
                issues.extend(pii_issues)
                
            elif isinstance(data, dict):
                for key, value in data.items():
                    nested_issues = await security_validator(value, f"{field_path}.{key}")
                    issues.extend(nested_issues)
                    
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    nested_issues = await security_validator(item, f"{field_path}[{i}]")
                    issues.extend(nested_issues)
                    
            return issues
            
        # Performance validator
        async def performance_validator(data: Any, field_path: str = '') -> List[ValidationIssue]:
            issues = []
            
            # Check data size limits
            if isinstance(data, str) and len(data) > self.performance_limits['max_string_length']:
                issues.append(ValidationIssue(
                    category=ValidationCategory.PERFORMANCE,
                    severity=ValidationSeverity.WARNING,
                    message=f"String length ({len(data)}) exceeds recommended limit",
                    field_path=field_path,
                    suggested_fix="Consider truncating or paginating large text"
                ))
                
            elif isinstance(data, list) and len(data) > self.performance_limits['max_list_size']:
                issues.append(ValidationIssue(
                    category=ValidationCategory.PERFORMANCE,
                    severity=ValidationSeverity.WARNING,
                    message=f"List size ({len(data)}) exceeds recommended limit",
                    field_path=field_path,
                    suggested_fix="Consider paginating or filtering large lists"
                ))
                
            elif isinstance(data, dict) and len(data) > self.performance_limits['max_dict_keys']:
                issues.append(ValidationIssue(
                    category=ValidationCategory.PERFORMANCE,
                    severity=ValidationSeverity.WARNING,
                    message=f"Dictionary size ({len(data)}) exceeds recommended limit",
                    field_path=field_path,
                    suggested_fix="Consider restructuring large dictionaries"
                ))
            
            # Check nesting depth
            depth = self._calculate_nesting_depth(data)
            if depth > self.performance_limits['max_nesting_depth']:
                issues.append(ValidationIssue(
                    category=ValidationCategory.PERFORMANCE,
                    severity=ValidationSeverity.ERROR,
                    message=f"Nesting depth ({depth}) exceeds limit",
                    field_path=field_path,
                    suggested_fix="Flatten nested structures"
                ))
                
            return issues
            
        # Correctness validator
        async def correctness_validator(data: Any, field_path: str = '') -> List[ValidationIssue]:
            issues = []
            
            # Check for common correctness issues
            if isinstance(data, str):
                # Check for malformed JSON in strings
                if data.strip().startswith('{') or data.strip().startswith('['):
                    try:
                        json.loads(data)
                    except json.JSONDecodeError:
                        issues.append(ValidationIssue(
                            category=ValidationCategory.CORRECTNESS,
                            severity=ValidationSeverity.ERROR,
                            message="Malformed JSON detected in string",
                            field_path=field_path,
                            suggested_fix="Fix JSON syntax or use plain text"
                        ))
                        
                # Check for unescaped special characters
                if '<' in data and '>' in data:
                    if not all(c in html.entities.html5 for c in re.findall(r'&(\w+);', data)):
                        issues.append(ValidationIssue(
                            category=ValidationCategory.CORRECTNESS,
                            severity=ValidationSeverity.WARNING,
                            message="Unescaped HTML-like content detected",
                            field_path=field_path,
                            suggested_fix="Properly escape HTML entities"
                        ))
                        
            return issues
            
        self.global_validators.extend([
            security_validator,
            performance_validator,
            correctness_validator,
        ])
    
    def _check_pii_exposure(self, data: str, field_path: str) -> List[ValidationIssue]:
        """Check for personally identifiable information exposure."""
        issues = []
        
        # Check for common PII patterns
        pii_checks = [
            ('email', self.compliance_patterns['email'], 'Email address detected'),
            ('phone', self.compliance_patterns['phone'], 'Phone number detected'),
            ('ssn', self.compliance_patterns['ssn'], 'SSN detected'),
            ('credit_card', self.compliance_patterns['credit_card'], 'Credit card number detected'),
        ]
        
        for pii_type, pattern, message in pii_checks:
            if pattern.search(data):
                issues.append(ValidationIssue(
                    category=ValidationCategory.COMPLIANCE,
                    severity=ValidationSeverity.WARNING,
                    message=message,
                    field_path=field_path,
                    suggested_fix=f"Mask or redact {pii_type} information",
                    metadata={'pii_type': pii_type}
                ))
                
        return issues
    
    def _calculate_nesting_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of data structure."""
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._calculate_nesting_depth(v, current_depth + 1) for v in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._calculate_nesting_depth(item, current_depth + 1) for item in data)
        else:
            return current_depth
    
    def register_validator(self, field_pattern: str, validator_func: Callable) -> None:
        """Register a custom validator for specific fields."""
        if field_pattern not in self.validators:
            self.validators[field_pattern] = []
        self.validators[field_pattern].append(validator_func)
    
    def register_sanitizer(self, field_pattern: str, sanitizer_func: Callable) -> None:
        """Register a custom sanitizer for specific fields."""
        if field_pattern not in self.sanitizers:
            self.sanitizers[field_pattern] = []
        self.sanitizers[field_pattern].append(sanitizer_func)
    
    async def validate_and_sanitize(
        self, 
        data: Any, 
        context: str = 'unknown',
        strict_mode: bool = False
    ) -> ValidationResult:
        """Perform comprehensive validation and sanitization."""
        
        issues = []
        sanitized_data = data
        
        try:
            # Apply sanitizers first
            sanitized_data = await self._apply_sanitizers(sanitized_data)
            
            # Run global validators
            for validator in self.global_validators:
                try:
                    validator_issues = await validator(sanitized_data, context)
                    issues.extend(validator_issues)
                except Exception as e:
                    logger.warning(f"Validator failed: {e}")
                    issues.append(ValidationIssue(
                        category=ValidationCategory.CORRECTNESS,
                        severity=ValidationSeverity.ERROR,
                        message=f"Validation process failed: {str(e)}",
                        field_path=context
                    ))
            
            # Run field-specific validators
            field_issues = await self._run_field_validators(sanitized_data, context)
            issues.extend(field_issues)
            
            # Additional strict mode checks
            if strict_mode:
                strict_issues = await self._strict_mode_validation(sanitized_data, context)
                issues.extend(strict_issues)
            
            # Create result
            result = ValidationResult(
                is_valid=True,  # Will be updated in __post_init__
                issues=issues,
                sanitized_data=sanitized_data
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed for context {context}: {e}")
            return ValidationResult(
                is_valid=False,
                issues=[ValidationIssue(
                    category=ValidationCategory.CORRECTNESS,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Validation process failed: {str(e)}",
                    field_path=context
                )],
                sanitized_data=data
            )
    
    async def _apply_sanitizers(self, data: Any) -> Any:
        """Apply registered sanitizers to data."""
        
        if isinstance(data, str):
            # Basic HTML sanitization
            data = html.escape(data)
            
            # Remove null bytes
            data = data.replace('\x00', '')
            
            # Normalize whitespace
            data = ' '.join(data.split())
            
        elif isinstance(data, dict):
            sanitized_dict = {}
            for key, value in data.items():
                # Sanitize key
                clean_key = await self._apply_sanitizers(key) if isinstance(key, str) else key
                # Sanitize value
                clean_value = await self._apply_sanitizers(value)
                sanitized_dict[clean_key] = clean_value
            data = sanitized_dict
            
        elif isinstance(data, list):
            data = [await self._apply_sanitizers(item) for item in data]
        
        return data
    
    async def _run_field_validators(self, data: Any, field_path: str) -> List[ValidationIssue]:
        """Run field-specific validators."""
        issues = []
        
        for pattern, validators in self.validators.items():
            if re.match(pattern, field_path):
                for validator in validators:
                    try:
                        validator_issues = await validator(data, field_path)
                        if isinstance(validator_issues, list):
                            issues.extend(validator_issues)
                        else:
                            issues.append(validator_issues)
                    except Exception as e:
                        logger.warning(f"Field validator {pattern} failed: {e}")
        
        return issues
    
    async def _strict_mode_validation(self, data: Any, context: str) -> List[ValidationIssue]:
        """Additional validation checks for strict mode."""
        issues = []
        
        # Check for suspicious patterns in strict mode
        if isinstance(data, str):
            # Check for potential command injection
            command_patterns = [
                r';\s*(rm|del|format|sudo)',
                r'\|\s*(curl|wget|nc|netcat)',
                r'`[^`]*`',  # Backticks
                r'\$\([^)]*\)',  # Command substitution
            ]
            
            for pattern in command_patterns:
                if re.search(pattern, data, re.IGNORECASE):
                    issues.append(ValidationIssue(
                        category=ValidationCategory.SECURITY,
                        severity=ValidationSeverity.CRITICAL,
                        message="Potential command injection detected",
                        field_path=context,
                        suggested_fix="Remove or escape command sequences"
                    ))
        
        # Check for excessive resource usage patterns
        if isinstance(data, (list, dict)):
            # Check for recursive structures (simplified check)
            if self._has_circular_reference(data):
                issues.append(ValidationIssue(
                    category=ValidationCategory.PERFORMANCE,
                    severity=ValidationSeverity.ERROR,
                    message="Circular reference detected",
                    field_path=context,
                    suggested_fix="Remove circular references"
                ))
        
        return issues
    
    def _has_circular_reference(self, data: Any, seen: Optional[Set] = None) -> bool:
        """Check for circular references in data structure."""
        if seen is None:
            seen = set()
        
        if isinstance(data, (list, dict)):
            data_id = id(data)
            if data_id in seen:
                return True
            seen.add(data_id)
            
            if isinstance(data, dict):
                for value in data.values():
                    if self._has_circular_reference(value, seen.copy()):
                        return True
            elif isinstance(data, list):
                for item in data:
                    if self._has_circular_reference(item, seen.copy()):
                        return True
        
        return False
    
    async def validate_tool_input(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> ValidationResult:
        """Validate tool input arguments."""
        
        context = f"tool.{tool_name}"
        
        # Tool-specific validation rules
        tool_issues = []
        
        # Check for common tool input issues
        if not isinstance(arguments, dict):
            tool_issues.append(ValidationIssue(
                category=ValidationCategory.CORRECTNESS,
                severity=ValidationSeverity.ERROR,
                message="Tool arguments must be a dictionary",
                field_path=context
            ))
        
        # Validate each argument
        for arg_name, arg_value in arguments.items():
            arg_context = f"{context}.{arg_name}"
            
            # Basic argument validation
            if arg_value is None and arg_name.endswith('_required'):
                tool_issues.append(ValidationIssue(
                    category=ValidationCategory.CORRECTNESS,
                    severity=ValidationSeverity.ERROR,
                    message=f"Required argument '{arg_name}' is None",
                    field_path=arg_context
                ))
        
        # Run general validation
        result = await self.validate_and_sanitize(arguments, context, strict_mode=True)
        result.issues.extend(tool_issues)
        
        # Recalculate validity
        result.is_valid = not any(
            issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
            for issue in result.issues
        )
        
        return result
    
    async def get_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        
        total_issues = sum(len(result.issues) for result in results)
        total_warnings = sum(result.warnings_count for result in results)
        total_errors = sum(result.errors_count for result in results)
        total_critical = sum(result.critical_count for result in results)
        
        # Categorize issues
        category_counts = {}
        severity_counts = {}
        
        for result in results:
            for issue in result.issues:
                category = issue.category.name
                severity = issue.severity.name
                
                category_counts[category] = category_counts.get(category, 0) + 1
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Calculate validation rate
        valid_results = sum(1 for result in results if result.is_valid)
        validation_rate = valid_results / len(results) if results else 1.0
        
        return {
            "total_validations": len(results),
            "valid_results": valid_results,
            "validation_rate": validation_rate,
            "total_issues": total_issues,
            "issues_by_severity": {
                "warnings": total_warnings,
                "errors": total_errors,
                "critical": total_critical
            },
            "issues_by_category": category_counts,
            "severity_distribution": severity_counts,
            "recommendations": self._generate_recommendations(results)
        }
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Analyze common patterns
        all_issues = []
        for result in results:
            all_issues.extend(result.issues)
        
        # Security recommendations
        security_issues = [i for i in all_issues if i.category == ValidationCategory.SECURITY]
        if security_issues:
            recommendations.append("Implement additional input sanitization for security")
        
        # Performance recommendations  
        performance_issues = [i for i in all_issues if i.category == ValidationCategory.PERFORMANCE]
        if performance_issues:
            recommendations.append("Consider optimizing data structures for better performance")
        
        # Compliance recommendations
        compliance_issues = [i for i in all_issues if i.category == ValidationCategory.COMPLIANCE]
        if compliance_issues:
            recommendations.append("Review data handling practices for compliance requirements")
        
        return recommendations


# Global advanced validator instance
advanced_validator = AdvancedValidator()