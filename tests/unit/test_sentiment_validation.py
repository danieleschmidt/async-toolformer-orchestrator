"""Unit tests for sentiment validation."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from async_toolformer.sentiment_validation import (
    SentimentValidator,
    SentimentValidationConfig,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    SentimentSecurityManager,
    handle_sentiment_errors,
    sanitize_text_input,
)
from async_toolformer.sentiment_analyzer import SentimentResult, SentimentScore, SentimentPolarity, EmotionScore, EmotionType
from async_toolformer.exceptions import ToolExecutionError


class TestSentimentValidationConfig:
    """Test SentimentValidationConfig model."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SentimentValidationConfig()
        
        assert config.min_text_length == 3
        assert config.max_text_length == 10000
        assert config.min_confidence_threshold == 0.3
        assert config.max_batch_size == 1000
        assert "en" in config.allowed_languages
        assert config.block_suspicious_patterns is True
        assert config.enable_toxicity_detection is True
        assert config.enable_spam_detection is True
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SentimentValidationConfig(
            min_text_length=5,
            max_text_length=5000,
            allowed_languages=["en", "es"],
            block_suspicious_patterns=False
        )
        
        assert config.min_text_length == 5
        assert config.max_text_length == 5000
        assert config.allowed_languages == ["en", "es"]
        assert config.block_suspicious_patterns is False


class TestValidationIssue:
    """Test ValidationIssue dataclass."""
    
    def test_validation_issue_creation(self):
        """Test validation issue creation."""
        issue = ValidationIssue(
            severity=ValidationSeverity.HIGH,
            issue_type="test_issue",
            message="Test message",
            field="test_field",
            suggestion="Test suggestion"
        )
        
        assert issue.severity == ValidationSeverity.HIGH
        assert issue.issue_type == "test_issue"
        assert issue.message == "Test message"
        assert issue.field == "test_field"
        assert issue.suggestion == "Test suggestion"
    
    def test_validation_issue_minimal(self):
        """Test validation issue with minimal fields."""
        issue = ValidationIssue(
            severity=ValidationSeverity.LOW,
            issue_type="minimal",
            message="Minimal issue"
        )
        
        assert issue.severity == ValidationSeverity.LOW
        assert issue.issue_type == "minimal"
        assert issue.message == "Minimal issue"
        assert issue.field is None
        assert issue.suggestion is None


class TestSentimentValidator:
    """Test SentimentValidator class."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = SentimentValidator()
        assert isinstance(validator.config, SentimentValidationConfig)
        assert len(validator.suspicious_patterns) > 0
        assert len(validator.spam_patterns) > 0
        assert len(validator.toxicity_patterns) > 0
    
    def test_validator_with_custom_config(self):
        """Test validator with custom config."""
        config = SentimentValidationConfig(min_text_length=10)
        validator = SentimentValidator(config)
        assert validator.config.min_text_length == 10
    
    def test_validate_input_text_valid(self):
        """Test validation with valid text."""
        validator = SentimentValidator()
        text = "This is a good product. I really like it!"
        
        result = validator.validate_input_text(text)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert result.processing_safe is True
        assert result.confidence_score > 0.5
        assert len(result.issues) == 0 or all(issue.severity != ValidationSeverity.CRITICAL for issue in result.issues)
    
    def test_validate_input_text_too_short(self):
        """Test validation with text that's too short."""
        validator = SentimentValidator()
        text = "Hi"  # Only 2 characters, default min is 3
        
        result = validator.validate_input_text(text)
        
        assert len(result.issues) > 0
        short_issues = [issue for issue in result.issues if issue.issue_type == "text_too_short"]
        assert len(short_issues) > 0
        assert short_issues[0].severity == ValidationSeverity.HIGH
        assert result.confidence_score < 1.0
    
    def test_validate_input_text_too_long(self):
        """Test validation with text that's too long."""
        validator = SentimentValidator()
        text = "word " * 3000  # Much longer than default max of 10000 chars
        
        result = validator.validate_input_text(text)
        
        long_issues = [issue for issue in result.issues if issue.issue_type == "text_too_long"]
        assert len(long_issues) > 0
        assert long_issues[0].severity == ValidationSeverity.HIGH
        assert result.confidence_score < 1.0
    
    def test_validate_input_text_non_string(self):
        """Test validation with non-string input."""
        validator = SentimentValidator()
        
        result = validator.validate_input_text(123)  # Not a string
        
        assert result.is_valid is False
        assert result.processing_safe is False
        assert result.confidence_score == 0.0
        type_issues = [issue for issue in result.issues if issue.issue_type == "type_error"]
        assert len(type_issues) > 0
        assert type_issues[0].severity == ValidationSeverity.CRITICAL
    
    def test_validate_input_suspicious_patterns(self):
        """Test validation with suspicious patterns."""
        validator = SentimentValidator()
        text = "This product is good. <script>alert('test')</script>"
        
        result = validator.validate_input_text(text)
        
        suspicious_issues = [issue for issue in result.issues if issue.issue_type == "suspicious_pattern"]
        assert len(suspicious_issues) > 0
        assert suspicious_issues[0].severity == ValidationSeverity.HIGH
        assert result.confidence_score < 0.5
    
    def test_validate_input_spam_detection(self):
        """Test spam detection."""
        validator = SentimentValidator()
        text = "BUY NOW!!! AMAZING DISCOUNT!!! CLICK HERE: https://spam-site.com"
        
        result = validator.validate_input_text(text)
        
        # Should detect spam patterns
        spam_issues = [issue for issue in result.issues if issue.issue_type == "potential_spam"]
        assert len(spam_issues) > 0 or result.confidence_score < 0.8
    
    def test_validate_input_toxicity_detection(self):
        """Test toxicity detection."""
        validator = SentimentValidator()
        text = "I hate this stupid product and want to kill the developer"
        
        result = validator.validate_input_text(text)
        
        # Should detect toxicity
        toxicity_issues = [issue for issue in result.issues if issue.issue_type == "potential_toxicity"]
        assert len(toxicity_issues) > 0
    
    def test_validate_input_disabled_features(self):
        """Test validation with disabled security features."""
        config = SentimentValidationConfig(
            block_suspicious_patterns=False,
            enable_spam_detection=False,
            enable_toxicity_detection=False
        )
        validator = SentimentValidator(config)
        text = "This product is good. <script>alert('test')</script> BUY NOW!!!"
        
        result = validator.validate_input_text(text)
        
        # Should have fewer issues with security features disabled
        suspicious_issues = [issue for issue in result.issues if issue.issue_type == "suspicious_pattern"]
        spam_issues = [issue for issue in result.issues if issue.issue_type == "potential_spam"]
        toxicity_issues = [issue for issue in result.issues if issue.issue_type == "potential_toxicity"]
        
        assert len(suspicious_issues) == 0
        assert len(spam_issues) == 0
        assert len(toxicity_issues) == 0
    
    def test_validate_batch_input_valid(self):
        """Test batch input validation with valid texts."""
        validator = SentimentValidator()
        texts = [
            "This product is great!",
            "I love this service.",
            "Amazing quality and fast delivery."
        ]
        
        result, invalid_indices = validator.validate_batch_input(texts)
        
        assert isinstance(result, ValidationResult)
        assert result.processing_safe is True
        assert len(invalid_indices) == 0
    
    def test_validate_batch_input_too_large(self):
        """Test batch validation with too many texts."""
        config = SentimentValidationConfig(max_batch_size=5)
        validator = SentimentValidator(config)
        texts = ["Good product!"] * 10  # Exceeds max_batch_size
        
        result, invalid_indices = validator.validate_batch_input(texts)
        
        large_batch_issues = [issue for issue in result.issues if issue.issue_type == "batch_too_large"]
        assert len(large_batch_issues) > 0
        assert large_batch_issues[0].severity == ValidationSeverity.HIGH
    
    def test_validate_batch_input_mixed_quality(self):
        """Test batch validation with mixed quality texts."""
        validator = SentimentValidator()
        texts = [
            "This is a good product.",  # Valid
            "",  # Too short
            "Bad quality.",  # Valid
            "<script>alert('hack')</script>",  # Suspicious
            "Great service!"  # Valid
        ]
        
        result, invalid_indices = validator.validate_batch_input(texts)
        
        assert len(invalid_indices) >= 1  # At least the suspicious one should be invalid
        assert result.processing_safe is True  # Some texts are still safe
    
    def test_validate_sentiment_result_valid(self):
        """Test sentiment result validation with valid result."""
        validator = SentimentValidator()
        
        sentiment_score = SentimentScore(
            polarity=SentimentPolarity.POSITIVE,
            confidence=0.85,
            score=0.7
        )
        
        emotion_score = EmotionScore(
            emotion=EmotionType.JOY,
            intensity=0.6,
            confidence=0.8
        )
        
        result = SentimentResult(
            text="Test text",
            sentiment=sentiment_score,
            emotions=[emotion_score],
            processing_time_ms=150.0
        )
        
        validation_result = validator.validate_sentiment_result(result)
        
        assert validation_result.is_valid is True
        assert len([issue for issue in validation_result.issues 
                   if issue.severity == ValidationSeverity.CRITICAL]) == 0
    
    def test_validate_sentiment_result_invalid_score_range(self):
        """Test sentiment result validation with invalid score range."""
        validator = SentimentValidator()
        
        # Create invalid sentiment score
        sentiment_score = SentimentScore(
            polarity=SentimentPolarity.POSITIVE,
            confidence=0.85,
            score=1.5  # Invalid range
        )
        
        result = SentimentResult(
            text="Test text",
            sentiment=sentiment_score,
            processing_time_ms=150.0
        )
        
        validation_result = validator.validate_sentiment_result(result)
        
        assert validation_result.is_valid is False
        score_issues = [issue for issue in validation_result.issues 
                       if issue.issue_type == "invalid_score_range"]
        assert len(score_issues) > 0
        assert score_issues[0].severity == ValidationSeverity.CRITICAL
    
    def test_validate_sentiment_result_low_confidence(self):
        """Test sentiment result validation with low confidence."""
        config = SentimentValidationConfig(min_confidence_threshold=0.5)
        validator = SentimentValidator(config)
        
        sentiment_score = SentimentScore(
            polarity=SentimentPolarity.POSITIVE,
            confidence=0.3,  # Below threshold
            score=0.7
        )
        
        result = SentimentResult(
            text="Test text",
            sentiment=sentiment_score,
            processing_time_ms=150.0
        )
        
        validation_result = validator.validate_sentiment_result(result)
        
        assert "Low confidence result" in str(validation_result.warnings)
        assert validation_result.confidence_score < 1.0
    
    def test_validate_sentiment_result_invalid_emotion_values(self):
        """Test sentiment result validation with invalid emotion values."""
        validator = SentimentValidator()
        
        sentiment_score = SentimentScore(
            polarity=SentimentPolarity.POSITIVE,
            confidence=0.85,
            score=0.7
        )
        
        # Invalid emotion score
        emotion_score = EmotionScore(
            emotion=EmotionType.JOY,
            intensity=1.5,  # Invalid range
            confidence=0.8
        )
        
        result = SentimentResult(
            text="Test text",
            sentiment=sentiment_score,
            emotions=[emotion_score],
            processing_time_ms=150.0
        )
        
        validation_result = validator.validate_sentiment_result(result)
        
        emotion_issues = [issue for issue in validation_result.issues 
                         if issue.issue_type == "invalid_emotion_intensity"]
        assert len(emotion_issues) > 0
        assert emotion_issues[0].severity == ValidationSeverity.HIGH


class TestSentimentSecurityManager:
    """Test SentimentSecurityManager class."""
    
    def test_security_manager_initialization(self):
        """Test security manager initialization."""
        security_manager = SentimentSecurityManager()
        
        assert isinstance(security_manager.validator, SentimentValidator)
        assert isinstance(security_manager.security_log, list)
        assert len(security_manager.security_log) == 0
    
    @pytest.mark.asyncio
    async def test_secure_analyze_valid_text(self):
        """Test secure analysis with valid text."""
        security_manager = SentimentSecurityManager()
        
        # Mock analyzer function
        async def mock_analyzer(text):
            sentiment_score = SentimentScore(
                polarity=SentimentPolarity.POSITIVE,
                confidence=0.8,
                score=0.6
            )
            return SentimentResult(
                text=text,
                sentiment=sentiment_score,
                processing_time_ms=100.0
            )
        
        result, validation = await security_manager.secure_analyze(
            "This is a good product!",
            mock_analyzer
        )
        
        assert isinstance(result, SentimentResult)
        assert isinstance(validation, ValidationResult)
        assert validation.processing_safe is True
        assert len(security_manager.security_log) == 1
    
    @pytest.mark.asyncio
    async def test_secure_analyze_invalid_text(self):
        """Test secure analysis with invalid text."""
        security_manager = SentimentSecurityManager()
        
        async def mock_analyzer(text):
            return "should not be called"
        
        with pytest.raises(ToolExecutionError):
            await security_manager.secure_analyze(
                "<script>alert('hack')</script>",
                mock_analyzer
            )
    
    def test_get_security_summary(self):
        """Test security summary generation."""
        security_manager = SentimentSecurityManager()
        
        # Add some log entries
        security_manager.security_log = [
            {"action": "test1", "timestamp": datetime.utcnow()},
            {"action": "test2", "timestamp": datetime.utcnow()}
        ]
        
        summary = security_manager.get_security_summary()
        
        assert "total_operations" in summary
        assert "recent_operations" in summary
        assert "validator_config" in summary
        assert summary["total_operations"] == 2


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_sanitize_text_input_basic(self):
        """Test basic text sanitization."""
        text = "This is a normal text message."
        result = sanitize_text_input(text)
        
        assert result == "This is a normal text message."
    
    def test_sanitize_text_input_script_removal(self):
        """Test script tag removal."""
        text = "Good product! <script>alert('hack')</script> Recommended."
        result = sanitize_text_input(text)
        
        assert "<script>" not in result
        assert "alert" not in result
        assert "Good product!" in result
        assert "Recommended." in result
    
    def test_sanitize_text_input_javascript_removal(self):
        """Test javascript protocol removal."""
        text = "Click here: javascript:alert('hack') for more info."
        result = sanitize_text_input(text)
        
        assert "javascript:" not in result
        assert "Click here:" in result
        assert "for more info." in result
    
    def test_sanitize_text_input_control_characters(self):
        """Test control character removal."""
        text = "Normal text\x00\x01\x02with control chars\x1f."
        result = sanitize_text_input(text)
        
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x02" not in result
        assert "\x1f" not in result
        assert "Normal text" in result
        assert "with control chars" in result
    
    def test_sanitize_text_input_length_limit(self):
        """Test length limiting."""
        text = "word " * 20000  # Very long text
        result = sanitize_text_input(text)
        
        assert len(result) <= 50020  # 50000 + "... [truncated]"
        assert result.endswith("... [truncated]")
    
    def test_sanitize_text_input_non_string(self):
        """Test sanitization of non-string input."""
        result = sanitize_text_input(123)
        assert result == "123"
        
        result = sanitize_text_input(None)
        assert result == "None"
        
        result = sanitize_text_input([1, 2, 3])
        assert result == "[1, 2, 3]"
    
    @pytest.mark.asyncio
    async def test_handle_sentiment_errors_decorator(self):
        """Test error handling decorator."""
        
        @handle_sentiment_errors
        async def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ToolExecutionError, match="Invalid value: Test error"):
            await failing_function()
        
        @handle_sentiment_errors
        async def working_function():
            return "success"
        
        result = await working_function()
        assert result == "success"


class TestValidationResult:
    """Test ValidationResult model."""
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation."""
        issues = [
            ValidationIssue(
                severity=ValidationSeverity.HIGH,
                issue_type="test_issue",
                message="Test message"
            )
        ]
        
        warnings = ["Test warning"]
        
        result = ValidationResult(
            is_valid=False,
            issues=issues,
            warnings=warnings,
            confidence_score=0.7,
            processing_safe=True
        )
        
        assert result.is_valid is False
        assert len(result.issues) == 1
        assert result.issues[0].issue_type == "test_issue"
        assert len(result.warnings) == 1
        assert result.warnings[0] == "Test warning"
        assert result.confidence_score == 0.7
        assert result.processing_safe is True
    
    def test_validation_result_defaults(self):
        """Test ValidationResult with default values."""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid is True
        assert len(result.issues) == 0
        assert len(result.warnings) == 0
        assert result.confidence_score == 1.0
        assert result.processing_safe is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])