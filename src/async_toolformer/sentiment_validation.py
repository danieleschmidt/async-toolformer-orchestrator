"""Robust validation and error handling for sentiment analysis."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import structlog
from pydantic import BaseModel, ValidationError, validator

from .exceptions import ToolExecutionError, ConfigurationError
from .sentiment_analyzer import SentimentResult, BatchSentimentResult

logger = structlog.get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Individual validation issue."""
    severity: ValidationSeverity
    issue_type: str
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None


class ValidationResult(BaseModel):
    """Validation result with issues and recommendations."""
    is_valid: bool
    issues: List[ValidationIssue] = []
    warnings: List[str] = []
    confidence_score: float = 1.0
    processing_safe: bool = True
    
    class Config:
        arbitrary_types_allowed = True


class SentimentValidationConfig(BaseModel):
    """Configuration for sentiment validation."""
    min_text_length: int = 3
    max_text_length: int = 10000
    min_confidence_threshold: float = 0.3
    max_batch_size: int = 1000
    allowed_languages: List[str] = ["en", "es", "fr", "de", "it", "pt", "nl", "ru", "ja", "zh"]
    block_suspicious_patterns: bool = True
    require_ascii_printable: bool = False
    max_special_char_ratio: float = 0.3
    max_repeated_char_ratio: float = 0.5
    enable_toxicity_detection: bool = True
    enable_spam_detection: bool = True


class SentimentValidator:
    """Comprehensive validator for sentiment analysis inputs and outputs."""
    
    def __init__(self, config: Optional[SentimentValidationConfig] = None):
        """Initialize validator with configuration."""
        self.config = config or SentimentValidationConfig()
        self._setup_patterns()
    
    def _setup_patterns(self):
        """Setup regex patterns for validation."""
        # Suspicious patterns that might indicate malicious input
        self.suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # Javascript protocol
            r'data:text/html',  # Data URLs
            r'<iframe[^>]*>',  # Iframe tags
            r'eval\s*\(',  # Eval functions
            r'document\.',  # DOM manipulation
            r'window\.',  # Window object access
            r'alert\s*\(',  # Alert functions
        ]
        
        # Spam indicators
        self.spam_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
            r'(?:^|\s)(?:buy|sale|discount|offer|free|click|visit|download)\s',  # Spam keywords
            r'[A-Z]{3,}\s*[A-Z]{3,}',  # All caps words
            r'[$£€¥₹]{2,}',  # Multiple currency symbols
            r'[0-9]{3,}-[0-9]{3,}-[0-9]{4,}',  # Phone numbers
        ]
        
        # Toxicity patterns (basic)
        self.toxicity_patterns = [
            r'\b(?:hate|kill|die|murder|violence)\b',
            r'\b(?:stupid|idiot|moron|dumb)\b',
            r'\b(?:racism|sexism|discrimination)\b',
        ]
    
    def validate_input_text(self, text: str) -> ValidationResult:
        """Validate input text for sentiment analysis."""
        issues = []
        warnings = []
        confidence_score = 1.0
        
        # Basic validation
        if not isinstance(text, str):
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                issue_type="type_error",
                message="Input must be a string",
                suggestion="Convert input to string before analysis"
            ))
            return ValidationResult(is_valid=False, issues=issues, confidence_score=0.0)
        
        # Length validation
        if len(text.strip()) < self.config.min_text_length:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                issue_type="text_too_short",
                message=f"Text too short (min: {self.config.min_text_length} chars)",
                suggestion="Provide longer text for reliable analysis"
            ))
            confidence_score *= 0.5
        
        if len(text) > self.config.max_text_length:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                issue_type="text_too_long",
                message=f"Text too long (max: {self.config.max_text_length} chars)",
                suggestion="Truncate or split text into smaller chunks"
            ))
            confidence_score *= 0.7
        
        # Character composition analysis
        char_issues = self._validate_character_composition(text)
        issues.extend(char_issues)
        
        # Suspicious pattern detection
        if self.config.block_suspicious_patterns:
            suspicious_issues = self._detect_suspicious_patterns(text)
            issues.extend(suspicious_issues)
            if suspicious_issues:
                confidence_score *= 0.3
        
        # Spam detection
        if self.config.enable_spam_detection:
            spam_score = self._calculate_spam_score(text)
            if spam_score > 0.7:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    issue_type="potential_spam",
                    message=f"High spam probability: {spam_score:.2f}",
                    suggestion="Review text content for spam indicators"
                ))
                confidence_score *= 0.6
        
        # Toxicity detection
        if self.config.enable_toxicity_detection:
            toxicity_score = self._calculate_toxicity_score(text)
            if toxicity_score > 0.5:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    issue_type="potential_toxicity",
                    message=f"Potential toxic content detected: {toxicity_score:.2f}",
                    suggestion="Consider content moderation before processing"
                ))
        
        # Language detection hint
        detected_language = self._simple_language_detection(text)
        if detected_language and detected_language not in self.config.allowed_languages:
            warnings.append(f"Detected language '{detected_language}' not in allowed list")
            confidence_score *= 0.8
        
        # Overall safety assessment
        processing_safe = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)
        
        return ValidationResult(
            is_valid=processing_safe,
            issues=issues,
            warnings=warnings,
            confidence_score=max(0.1, confidence_score),
            processing_safe=processing_safe
        )
    
    def validate_batch_input(self, texts: List[str]) -> Tuple[ValidationResult, List[int]]:
        """Validate batch input and return invalid indices."""
        issues = []
        warnings = []
        invalid_indices = []
        
        # Batch size validation
        if len(texts) > self.config.max_batch_size:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.HIGH,
                issue_type="batch_too_large",
                message=f"Batch size {len(texts)} exceeds maximum {self.config.max_batch_size}",
                suggestion="Split batch into smaller chunks"
            ))
        
        # Validate each text
        invalid_count = 0
        for i, text in enumerate(texts):
            text_validation = self.validate_input_text(text)
            if not text_validation.processing_safe:
                invalid_indices.append(i)
                invalid_count += 1
        
        # Batch quality assessment
        if invalid_count > len(texts) * 0.3:  # More than 30% invalid
            issues.append(ValidationIssue(
                severity=ValidationSeverity.MEDIUM,
                issue_type="high_invalid_ratio",
                message=f"High invalid text ratio: {invalid_count}/{len(texts)}",
                suggestion="Review input quality before batch processing"
            ))
        
        confidence_score = 1.0 - (invalid_count / len(texts)) * 0.5
        
        return ValidationResult(
            is_valid=len(invalid_indices) == 0,
            issues=issues,
            warnings=warnings,
            confidence_score=max(0.1, confidence_score),
            processing_safe=invalid_count < len(texts)  # At least some texts are safe
        ), invalid_indices
    
    def validate_sentiment_result(self, result: SentimentResult) -> ValidationResult:
        """Validate sentiment analysis result."""
        issues = []
        warnings = []
        confidence_score = 1.0
        
        try:
            # Validate sentiment score range
            if not -1.0 <= result.sentiment.score <= 1.0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    issue_type="invalid_score_range",
                    message=f"Sentiment score {result.sentiment.score} outside valid range [-1, 1]",
                    field="sentiment.score"
                ))
            
            # Validate confidence range
            if not 0.0 <= result.sentiment.confidence <= 1.0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.CRITICAL,
                    issue_type="invalid_confidence_range",
                    message=f"Confidence {result.sentiment.confidence} outside valid range [0, 1]",
                    field="sentiment.confidence"
                ))
            
            # Check confidence threshold
            if result.sentiment.confidence < self.config.min_confidence_threshold:
                warnings.append(f"Low confidence result: {result.sentiment.confidence:.2f}")
                confidence_score *= 0.7
            
            # Validate emotions
            for emotion in result.emotions:
                if not 0.0 <= emotion.intensity <= 1.0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.HIGH,
                        issue_type="invalid_emotion_intensity",
                        message=f"Emotion intensity {emotion.intensity} outside valid range [0, 1]",
                        field="emotions"
                    ))
                
                if not 0.0 <= emotion.confidence <= 1.0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.HIGH,
                        issue_type="invalid_emotion_confidence",
                        message=f"Emotion confidence {emotion.confidence} outside valid range [0, 1]",
                        field="emotions"
                    ))
            
            # Validate processing time reasonableness
            if result.processing_time_ms < 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    issue_type="negative_processing_time",
                    message="Negative processing time",
                    field="processing_time_ms"
                ))
            
            if result.processing_time_ms > 30000:  # 30 seconds
                warnings.append(f"Very long processing time: {result.processing_time_ms:.2f}ms")
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                issue_type="validation_exception",
                message=f"Validation failed with exception: {e}",
                suggestion="Check result structure and data types"
            ))
        
        return ValidationResult(
            is_valid=not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues),
            issues=issues,
            warnings=warnings,
            confidence_score=max(0.1, confidence_score)
        )
    
    def _validate_character_composition(self, text: str) -> List[ValidationIssue]:
        """Validate character composition of text."""
        issues = []
        
        if not text:
            return issues
        
        # Count different character types
        total_chars = len(text)
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        repeated_chars = self._count_repeated_characters(text)
        
        # Special character ratio
        special_ratio = special_chars / total_chars
        if special_ratio > self.config.max_special_char_ratio:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.MEDIUM,
                issue_type="high_special_char_ratio",
                message=f"High special character ratio: {special_ratio:.2f}",
                suggestion="Review text for excessive special characters"
            ))
        
        # Repeated character ratio
        repeated_ratio = repeated_chars / total_chars
        if repeated_ratio > self.config.max_repeated_char_ratio:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.LOW,
                issue_type="high_repeated_char_ratio",
                message=f"High repeated character ratio: {repeated_ratio:.2f}",
                suggestion="Text may contain excessive repetition"
            ))
        
        # ASCII printable check
        if self.config.require_ascii_printable:
            non_ascii = sum(1 for c in text if ord(c) > 127)
            if non_ascii > 0:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.MEDIUM,
                    issue_type="non_ascii_characters",
                    message=f"Contains {non_ascii} non-ASCII characters",
                    suggestion="Consider text encoding handling"
                ))
        
        return issues
    
    def _count_repeated_characters(self, text: str) -> int:
        """Count characters that repeat consecutively."""
        repeated = 0
        for i in range(1, len(text)):
            if text[i] == text[i-1]:
                repeated += 1
        return repeated
    
    def _detect_suspicious_patterns(self, text: str) -> List[ValidationIssue]:
        """Detect suspicious patterns in text."""
        issues = []
        
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    issue_type="suspicious_pattern",
                    message=f"Suspicious pattern detected: {pattern}",
                    suggestion="Review text for potential security issues"
                ))
        
        return issues
    
    def _calculate_spam_score(self, text: str) -> float:
        """Calculate spam probability score."""
        spam_indicators = 0
        total_indicators = len(self.spam_patterns)
        
        for pattern in self.spam_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                spam_indicators += 1
        
        # Additional spam indicators
        if len(text.split()) < 5 and any(char in text for char in "!@#$%"):
            spam_indicators += 1
            total_indicators += 1
        
        if text.count('!') > 3:
            spam_indicators += 1
            total_indicators += 1
        
        return spam_indicators / total_indicators if total_indicators > 0 else 0.0
    
    def _calculate_toxicity_score(self, text: str) -> float:
        """Calculate toxicity probability score."""
        toxicity_indicators = 0
        
        for pattern in self.toxicity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            toxicity_indicators += len(matches)
        
        # Normalize by text length
        text_length = max(len(text.split()), 1)
        return min(1.0, toxicity_indicators / text_length * 5)  # Scale factor
    
    def _simple_language_detection(self, text: str) -> Optional[str]:
        """Simple language detection based on character patterns."""
        # Very basic language detection - can be enhanced with proper libraries
        text_lower = text.lower()
        
        # English indicators
        english_indicators = ["the", "and", "is", "in", "to", "of", "a", "that"]
        english_count = sum(1 for word in english_indicators if word in text_lower)
        
        # Spanish indicators  
        spanish_indicators = ["el", "la", "de", "que", "y", "en", "un", "es"]
        spanish_count = sum(1 for word in spanish_indicators if word in text_lower)
        
        # French indicators
        french_indicators = ["le", "de", "et", "à", "un", "il", "être", "et"]
        french_count = sum(1 for word in french_indicators if word in text_lower)
        
        # Simple heuristic
        if english_count >= 2:
            return "en"
        elif spanish_count >= 2:
            return "es"
        elif french_count >= 2:
            return "fr"
        
        return None


class SentimentSecurityManager:
    """Security manager for sentiment analysis operations."""
    
    def __init__(self):
        """Initialize security manager."""
        self.validator = SentimentValidator()
        self.security_log = []
    
    def secure_analyze(self, text: str, analyzer_func) -> Tuple[Any, ValidationResult]:
        """Securely analyze text with validation."""
        # Pre-analysis validation
        validation = self.validator.validate_input_text(text)
        
        if not validation.processing_safe:
            logger.warning("Text failed security validation", 
                         issues=[issue.message for issue in validation.issues])
            raise ToolExecutionError(f"Security validation failed: {validation.issues[0].message}")
        
        # Log security event
        self.security_log.append({
            "timestamp": logger.get_logger().bind().info,
            "action": "text_analysis",
            "validation_score": validation.confidence_score,
            "text_length": len(text)
        })
        
        try:
            # Execute analysis
            result = analyzer_func(text)
            
            # Post-analysis validation
            if hasattr(result, 'sentiment'):
                result_validation = self.validator.validate_sentiment_result(result)
                if not result_validation.is_valid:
                    logger.warning("Result failed validation", 
                                 issues=[issue.message for issue in result_validation.issues])
            
            return result, validation
            
        except Exception as e:
            logger.error("Secure analysis failed", error=str(e))
            raise
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security operation summary."""
        return {
            "total_operations": len(self.security_log),
            "recent_operations": self.security_log[-10:] if self.security_log else [],
            "validator_config": self.validator.config.dict()
        }


# Utility functions for error handling
def handle_sentiment_errors(func):
    """Decorator for robust error handling in sentiment analysis."""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ValidationError as e:
            logger.error("Validation error in sentiment analysis", error=str(e))
            raise ToolExecutionError(f"Input validation failed: {e}")
        except ValueError as e:
            logger.error("Value error in sentiment analysis", error=str(e))
            raise ToolExecutionError(f"Invalid value: {e}")
        except Exception as e:
            logger.error("Unexpected error in sentiment analysis", error=str(e))
            raise ToolExecutionError(f"Sentiment analysis failed: {e}")
    
    return wrapper


def sanitize_text_input(text: str) -> str:
    """Sanitize text input for safe processing."""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    # Limit length
    if len(text) > 50000:
        text = text[:50000] + "... [truncated]"
    
    # Remove potential script injections
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    
    return text.strip()