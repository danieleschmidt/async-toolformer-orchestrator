"""Input validation and sanitization for security and reliability."""

import re
import html
import json
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass
from enum import Enum
# import bleach  # Not available in this environment - use simple HTML sanitization
from pydantic import BaseModel, ValidationError, validator

from .simple_structured_logging import get_logger
from .exceptions import ConfigurationError, ToolExecutionError

logger = get_logger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"  
    PERMISSIVE = "permissive"
    DISABLED = "disabled"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_data: Any = None
    errors: List[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class SecurityPolicy:
    """Security policy for input validation."""
    
    def __init__(self, 
                 validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.blocked_patterns = [
            # SQL injection patterns
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC)\b)",
            r"(UNION\s+SELECT)",
            r"(\-\-|\#|\/\*|\*\/)",
            
            # XSS patterns
            r"(<script[^>]*>.*?</script>)",
            r"(javascript:)",
            r"(on\w+\s*=)",
            
            # Command injection patterns
            r"(\$\(.*?\))",
            r"(`.*?`)",
            r"(;\s*(rm|cat|ls|wget|curl))",
            
            # Path traversal
            r"(\.\.\/|\.\.\\)",
            
            # Code execution
            r"(\beval\s*\()",
            r"(\bexec\s*\()",
            r"(__import__|getattr|setattr|delattr)",
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.DOTALL) 
            for pattern in self.blocked_patterns
        ]
        
        self.max_string_length = 10000
        self.max_list_items = 1000
        self.max_dict_keys = 100
        self.max_nesting_depth = 10
        
        # Allowed HTML tags for content sanitization
        self.allowed_html_tags = ['b', 'i', 'em', 'strong', 'p', 'br']
        self.allowed_html_attrs = {}


class InputValidator:
    """Comprehensive input validator and sanitizer."""
    
    def __init__(self, security_policy: Optional[SecurityPolicy] = None):
        self.policy = security_policy or SecurityPolicy()
        
    def validate_and_sanitize(self, 
                            data: Any,
                            field_name: str = "input") -> ValidationResult:
        """Validate and sanitize input data."""
        
        if self.policy.validation_level == ValidationLevel.DISABLED:
            return ValidationResult(is_valid=True, sanitized_data=data)
        
        try:
            sanitized = self._sanitize_recursive(data, field_name, depth=0)
            errors = self._validate_security(sanitized, field_name)
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                sanitized_data=sanitized,
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Validation failed for {field_name}", error=e)
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"]
            )
    
    def _sanitize_recursive(self, data: Any, field_name: str, depth: int) -> Any:
        """Recursively sanitize data structures."""
        
        # Check nesting depth
        if depth > self.policy.max_nesting_depth:
            raise ValueError(f"Maximum nesting depth exceeded for {field_name}")
        
        if isinstance(data, str):
            return self._sanitize_string(data, field_name)
            
        elif isinstance(data, dict):
            return self._sanitize_dict(data, field_name, depth)
            
        elif isinstance(data, (list, tuple)):
            return self._sanitize_list(data, field_name, depth)
            
        elif isinstance(data, (int, float, bool, type(None))):
            return data
            
        else:
            # Convert unknown types to string and sanitize
            logger.warning(f"Unknown data type for {field_name}, converting to string")
            return self._sanitize_string(str(data), field_name)
    
    def _sanitize_string(self, text: str, field_name: str) -> str:
        """Sanitize string input."""
        
        # Length check
        if len(text) > self.policy.max_string_length:
            logger.warning(f"String length exceeded for {field_name}, truncating")
            text = text[:self.policy.max_string_length]
        
        # Simple HTML sanitization for content fields
        if any(tag in field_name.lower() for tag in ['content', 'description', 'message']):
            # Simple tag stripping instead of bleach
            text = re.sub(r'<[^>]+>', '', text)
        
        # URL decode if it looks like encoded data
        if '%' in text and self.policy.validation_level != ValidationLevel.STRICT:
            try:
                decoded = urllib.parse.unquote(text)
                if decoded != text:
                    logger.debug(f"URL decoded {field_name}")
                    text = decoded
            except Exception:
                pass  # Keep original if decoding fails
        
        # HTML entity decode
        text = html.unescape(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _sanitize_dict(self, data: dict, field_name: str, depth: int) -> dict:
        """Sanitize dictionary data."""
        
        if len(data) > self.policy.max_dict_keys:
            logger.warning(f"Too many dictionary keys for {field_name}, truncating")
            data = dict(list(data.items())[:self.policy.max_dict_keys])
        
        sanitized = {}
        for key, value in data.items():
            # Sanitize key
            clean_key = self._sanitize_string(str(key), f"{field_name}.key")
            
            # Sanitize value
            clean_value = self._sanitize_recursive(
                value, 
                f"{field_name}.{clean_key}",
                depth + 1
            )
            
            sanitized[clean_key] = clean_value
        
        return sanitized
    
    def _sanitize_list(self, data: Union[list, tuple], field_name: str, depth: int) -> list:
        """Sanitize list/tuple data."""
        
        if len(data) > self.policy.max_list_items:
            logger.warning(f"Too many list items for {field_name}, truncating")
            data = data[:self.policy.max_list_items]
        
        sanitized = []
        for i, item in enumerate(data):
            clean_item = self._sanitize_recursive(
                item,
                f"{field_name}[{i}]",
                depth + 1
            )
            sanitized.append(clean_item)
        
        return sanitized
    
    def _validate_security(self, data: Any, field_name: str) -> List[str]:
        """Validate data against security patterns."""
        errors = []
        
        if self.policy.validation_level == ValidationLevel.PERMISSIVE:
            return errors
        
        # Convert data to string for pattern matching
        text_data = json.dumps(data, default=str) if not isinstance(data, str) else data
        
        # Check against blocked patterns
        for i, pattern in enumerate(self.policy.compiled_patterns):
            if pattern.search(text_data):
                error_msg = f"Security violation in {field_name}: blocked pattern detected"
                errors.append(error_msg)
                
                if self.policy.validation_level == ValidationLevel.STRICT:
                    # In strict mode, stop at first violation
                    break
        
        return errors


class ToolInputValidator(InputValidator):
    """Specialized validator for tool inputs."""
    
    def __init__(self, security_policy: Optional[SecurityPolicy] = None):
        super().__init__(security_policy)
        
        # Tool-specific validation rules
        self.tool_schemas: Dict[str, BaseModel] = {}
        self.custom_validators: Dict[str, Callable] = {}
    
    def register_tool_schema(self, tool_name: str, schema: BaseModel):
        """Register a Pydantic schema for tool validation."""
        self.tool_schemas[tool_name] = schema
        logger.info(f"Registered schema for tool: {tool_name}")
    
    def register_custom_validator(self, tool_name: str, validator: Callable):
        """Register a custom validator function for a tool."""
        self.custom_validators[tool_name] = validator
        logger.info(f"Registered custom validator for tool: {tool_name}")
    
    def validate_tool_input(self, 
                          tool_name: str,
                          arguments: Dict[str, Any]) -> ValidationResult:
        """Validate tool-specific input arguments."""
        
        # First apply general validation
        general_result = self.validate_and_sanitize(
            arguments, 
            f"tool.{tool_name}.arguments"
        )
        
        if not general_result.is_valid:
            return general_result
        
        # Apply tool-specific schema validation
        schema_errors = []
        if tool_name in self.tool_schemas:
            try:
                schema = self.tool_schemas[tool_name]
                validated_data = schema(**general_result.sanitized_data)
                general_result.sanitized_data = validated_data.dict()
            except ValidationError as e:
                schema_errors = [f"Schema validation error: {str(e)}"]
        
        # Apply custom validator
        custom_errors = []
        if tool_name in self.custom_validators:
            try:
                validator = self.custom_validators[tool_name]
                custom_result = validator(general_result.sanitized_data)
                if isinstance(custom_result, dict) and not custom_result.get('valid', True):
                    custom_errors = custom_result.get('errors', ['Custom validation failed'])
            except Exception as e:
                custom_errors = [f"Custom validation error: {str(e)}"]
        
        # Combine all errors
        all_errors = general_result.errors + schema_errors + custom_errors
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            sanitized_data=general_result.sanitized_data,
            errors=all_errors
        )


def create_file_path_validator(allowed_extensions: Set[str] = None,
                             allowed_directories: Set[str] = None) -> Callable:
    """Create a validator for file paths."""
    
    def validator(data: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        
        for key, value in data.items():
            if 'file' in key.lower() or 'path' in key.lower():
                if not isinstance(value, str):
                    continue
                
                # Check for path traversal
                if '..' in value:
                    errors.append(f"Path traversal detected in {key}")
                
                # Check allowed extensions
                if allowed_extensions:
                    extension = value.split('.')[-1].lower()
                    if extension not in allowed_extensions:
                        errors.append(f"File extension '{extension}' not allowed for {key}")
                
                # Check allowed directories
                if allowed_directories:
                    if not any(value.startswith(allowed_dir) for allowed_dir in allowed_directories):
                        errors.append(f"Directory not allowed for {key}")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    return validator


def create_url_validator(allowed_schemes: Set[str] = None,
                        allowed_domains: Set[str] = None) -> Callable:
    """Create a validator for URLs."""
    
    def validator(data: Dict[str, Any]) -> Dict[str, Any]:
        errors = []
        
        for key, value in data.items():
            if 'url' in key.lower() or 'link' in key.lower():
                if not isinstance(value, str):
                    continue
                
                try:
                    parsed = urllib.parse.urlparse(value)
                    
                    # Check allowed schemes
                    if allowed_schemes and parsed.scheme not in allowed_schemes:
                        errors.append(f"URL scheme '{parsed.scheme}' not allowed for {key}")
                    
                    # Check allowed domains
                    if allowed_domains and parsed.netloc not in allowed_domains:
                        errors.append(f"Domain '{parsed.netloc}' not allowed for {key}")
                        
                except Exception as e:
                    errors.append(f"Invalid URL format for {key}: {str(e)}")
        
        return {"valid": len(errors) == 0, "errors": errors}
    
    return validator


# Global validator instances
default_validator = InputValidator()
tool_validator = ToolInputValidator()