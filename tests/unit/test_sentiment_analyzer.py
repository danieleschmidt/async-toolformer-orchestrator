"""Unit tests for sentiment analyzer."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from async_toolformer.sentiment_analyzer import (
    analyze_text_sentiment,
    analyze_batch_sentiment,
    analyze_social_media_sentiment,
    compare_sentiment_sources,
    SentimentPolarity,
    SentimentScore,
    EmotionType,
    EmotionScore,
    SentimentResult,
    BatchSentimentResult,
    _preprocess_text,
    _calculate_rule_based_sentiment,
    _analyze_emotions,
    _extract_sentiment_keywords,
    _extract_hashtags,
    _extract_mentions,
    _extract_urls,
)


class TestSentimentScore:
    """Test SentimentScore class."""
    
    def test_valid_sentiment_score(self):
        """Test valid sentiment score creation."""
        score = SentimentScore(
            polarity=SentimentPolarity.POSITIVE,
            confidence=0.8,
            score=0.6
        )
        
        assert score.polarity == SentimentPolarity.POSITIVE
        assert score.confidence == 0.8
        assert score.score == 0.6
    
    def test_invalid_score_range(self):
        """Test invalid score range validation."""
        with pytest.raises(ValueError, match="Score must be between -1.0 and 1.0"):
            SentimentScore(
                polarity=SentimentPolarity.POSITIVE,
                confidence=0.8,
                score=1.5  # Invalid
            )
    
    def test_invalid_confidence_range(self):
        """Test invalid confidence range validation."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            SentimentScore(
                polarity=SentimentPolarity.POSITIVE,
                confidence=1.2,  # Invalid
                score=0.6
            )


class TestEmotionScore:
    """Test EmotionScore class."""
    
    def test_emotion_score_creation(self):
        """Test emotion score creation."""
        emotion = EmotionScore(
            emotion=EmotionType.JOY,
            intensity=0.7,
            confidence=0.85
        )
        
        assert emotion.emotion == EmotionType.JOY
        assert emotion.intensity == 0.7
        assert emotion.confidence == 0.85


class TestPreprocessing:
    """Test text preprocessing functions."""
    
    def test_preprocess_text_basic(self):
        """Test basic text preprocessing."""
        text = "  Hello   world!  This   is  a   test.  "
        result = _preprocess_text(text)
        
        assert result == "Hello world! This is a test."
    
    def test_preprocess_text_contractions(self):
        """Test contraction expansion."""
        text = "I can't believe it won't work. They're not here."
        result = _preprocess_text(text)
        
        assert "cannot" in result
        assert "will not" in result
        assert " are" in result
    
    def test_preprocess_text_empty(self):
        """Test empty text preprocessing."""
        result = _preprocess_text("")
        assert result == ""
    
    def test_preprocess_text_none(self):
        """Test None input preprocessing."""
        result = _preprocess_text(None)
        assert result == ""


class TestRuleBasedSentiment:
    """Test rule-based sentiment analysis."""
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection."""
        text = "I love this product! It's amazing and excellent."
        result = _calculate_rule_based_sentiment(text)
        
        assert result.polarity == SentimentPolarity.POSITIVE
        assert result.score > 0.1
        assert result.confidence > 0.5
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection."""
        text = "This product is terrible and awful. I hate it."
        result = _calculate_rule_based_sentiment(text)
        
        assert result.polarity == SentimentPolarity.NEGATIVE
        assert result.score < -0.1
        assert result.confidence > 0.5
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection."""
        text = "This is a book. It has pages and words."
        result = _calculate_rule_based_sentiment(text)
        
        assert result.polarity == SentimentPolarity.NEUTRAL
        assert -0.1 <= result.score <= 0.1
    
    def test_negation_handling(self):
        """Test negation handling."""
        text = "This is not bad at all."
        result = _calculate_rule_based_sentiment(text)
        
        # Should be positive due to negation of "bad"
        assert result.score > 0
    
    def test_intensifier_handling(self):
        """Test intensifier handling."""
        text = "This is very good."
        result = _calculate_rule_based_sentiment(text)
        
        text_without_intensifier = "This is good."
        result_without = _calculate_rule_based_sentiment(text_without_intensifier)
        
        # Intensified version should have higher score
        assert result.score > result_without.score
    
    def test_empty_text_sentiment(self):
        """Test empty text sentiment."""
        result = _calculate_rule_based_sentiment("")
        
        assert result.polarity == SentimentPolarity.NEUTRAL
        assert result.score == 0.0
        assert result.confidence == 0.5


class TestEmotionAnalysis:
    """Test emotion analysis."""
    
    def test_joy_emotion_detection(self):
        """Test joy emotion detection."""
        text = "I'm so happy and excited! This is joyful!"
        emotions = _analyze_emotions(text)
        
        joy_emotions = [e for e in emotions if e.emotion == EmotionType.JOY]
        assert len(joy_emotions) > 0
        assert joy_emotions[0].intensity > 0
    
    def test_anger_emotion_detection(self):
        """Test anger emotion detection."""
        text = "I'm so angry and furious! This is irritating!"
        emotions = _analyze_emotions(text)
        
        anger_emotions = [e for e in emotions if e.emotion == EmotionType.ANGER]
        assert len(anger_emotions) > 0
        assert anger_emotions[0].intensity > 0
    
    def test_no_emotions(self):
        """Test text with no clear emotions."""
        text = "The weather is 72 degrees today."
        emotions = _analyze_emotions(text)
        
        # Should have very low intensity emotions or none
        for emotion in emotions:
            assert emotion.intensity < 0.3
    
    def test_emotion_limit(self):
        """Test emotion result limit."""
        text = "I'm happy, angry, sad, afraid, surprised, and disgusted all at once!"
        emotions = _analyze_emotions(text)
        
        # Should return at most 3 emotions (as per implementation)
        assert len(emotions) <= 3


class TestKeywordExtraction:
    """Test keyword extraction."""
    
    def test_sentiment_keywords(self):
        """Test sentiment keyword extraction."""
        text = "I love this amazing product! It's excellent and fantastic."
        keywords = _extract_sentiment_keywords(text)
        
        assert "love" in keywords
        assert "amazing" in keywords
        assert "excellent" in keywords
        assert "fantastic" in keywords
    
    def test_keyword_deduplication(self):
        """Test keyword deduplication."""
        text = "good good good bad bad excellent excellent"
        keywords = _extract_sentiment_keywords(text)
        
        # Should have unique keywords only
        assert keywords.count("good") == 1
        assert keywords.count("bad") == 1
        assert keywords.count("excellent") == 1
    
    def test_keyword_limit(self):
        """Test keyword count limit."""
        text = " ".join(["good", "bad", "great", "terrible", "amazing", "awful", 
                        "excellent", "horrible", "fantastic", "disgusting", 
                        "wonderful", "pathetic", "perfect", "worst"])
        keywords = _extract_sentiment_keywords(text)
        
        # Should return at most 10 keywords
        assert len(keywords) <= 10


class TestSocialMediaExtraction:
    """Test social media-specific extraction functions."""
    
    def test_hashtag_extraction(self):
        """Test hashtag extraction."""
        text = "I love this #product! #amazing #musttry"
        hashtags = _extract_hashtags(text)
        
        assert "product" in hashtags
        assert "amazing" in hashtags
        assert "musttry" in hashtags
        assert len(hashtags) == 3
    
    def test_mention_extraction(self):
        """Test mention extraction."""
        text = "Thanks @company for the great service! @support was helpful."
        mentions = _extract_mentions(text)
        
        assert "company" in mentions
        assert "support" in mentions
        assert len(mentions) == 2
    
    def test_url_extraction(self):
        """Test URL extraction."""
        text = "Check out https://example.com and http://test.org for more info."
        urls = _extract_urls(text)
        
        assert "https://example.com" in urls
        assert "http://test.org" in urls
        assert len(urls) == 2
    
    def test_no_social_elements(self):
        """Test text with no social media elements."""
        text = "This is just regular text without any special elements."
        
        assert len(_extract_hashtags(text)) == 0
        assert len(_extract_mentions(text)) == 0
        assert len(_extract_urls(text)) == 0


@pytest.mark.asyncio
class TestSentimentAnalysisTools:
    """Test sentiment analysis tool functions."""
    
    async def test_analyze_text_sentiment_basic(self):
        """Test basic text sentiment analysis."""
        text = "I really love this amazing product!"
        result = await analyze_text_sentiment(text)
        
        assert isinstance(result, SentimentResult)
        assert result.text == text
        assert result.sentiment.polarity == SentimentPolarity.POSITIVE
        assert result.sentiment.score > 0
        assert result.sentiment.confidence > 0
        assert result.processing_time_ms > 0
        assert len(result.emotions) > 0
        assert len(result.keywords) > 0
    
    async def test_analyze_text_sentiment_options(self):
        """Test text sentiment analysis with options."""
        text = "This product is okay, nothing special."
        result = await analyze_text_sentiment(
            text,
            include_emotions=False,
            include_keywords=False,
            language="en"
        )
        
        assert isinstance(result, SentimentResult)
        assert len(result.emotions) == 0
        assert len(result.keywords) == 0
        assert result.language == "en"
    
    async def test_analyze_batch_sentiment(self):
        """Test batch sentiment analysis."""
        texts = [
            "I love this product!",
            "This is terrible.",
            "It's okay, I guess.",
            "Amazing quality!",
            "Not worth the money."
        ]
        
        result = await analyze_batch_sentiment(texts, max_parallel=3)
        
        assert isinstance(result, BatchSentimentResult)
        assert len(result.results) == 5
        assert result.total_texts == 5
        assert result.processing_time_ms > 0
        assert "polarity_distribution" in result.summary
        assert "average_sentiment_score" in result.summary
    
    async def test_analyze_batch_sentiment_empty(self):
        """Test batch sentiment analysis with empty list."""
        result = await analyze_batch_sentiment([])
        
        assert isinstance(result, BatchSentimentResult)
        assert len(result.results) == 0
        assert result.total_texts == 0
    
    async def test_analyze_social_media_sentiment(self):
        """Test social media sentiment analysis."""
        text = "OMG! @company this is amazing! 😍 #love #musthave https://example.com"
        result = await analyze_social_media_sentiment(text, platform="twitter")
        
        assert isinstance(result, SentimentResult)
        assert result.sentiment.polarity == SentimentPolarity.POSITIVE
        assert "hashtags" in result.entities
        assert "mentions" in result.entities
        assert "urls" in result.entities
        assert "love" in result.entities["hashtags"]
        assert "company" in result.entities["mentions"]
    
    async def test_compare_sentiment_sources(self):
        """Test sentiment source comparison."""
        sources = {
            "review1": "This product is absolutely fantastic! Love it!",
            "review2": "Pretty good product, satisfied with purchase.",
            "social": "Meh, it's okay I guess. Nothing special.",
        }
        
        result = await compare_sentiment_sources(sources)
        
        assert "source_results" in result
        assert "weighted_sentiment" in result
        assert "consensus_analysis" in result
        assert len(result["source_results"]) == 3
        
        # Check weighted sentiment structure
        weighted = result["weighted_sentiment"]
        assert "score" in weighted
        assert "confidence" in weighted
        assert "polarity" in weighted
    
    async def test_compare_sentiment_sources_with_weights(self):
        """Test sentiment source comparison with custom weights."""
        sources = {
            "expert_review": "This is excellent quality.",
            "user_comment": "Not bad, but could be better."
        }
        weights = {
            "expert_review": 2.0,
            "user_comment": 1.0
        }
        
        result = await compare_sentiment_sources(
            sources, 
            weight_sources=True,
            source_weights=weights
        )
        
        assert "weighted_sentiment" in result
        assert "total_weight" in result["weighted_sentiment"]
        assert result["weighted_sentiment"]["total_weight"] == 3.0
    
    async def test_compare_sentiment_sources_empty(self):
        """Test sentiment source comparison with empty sources."""
        sources = {"empty": ""}
        
        with pytest.raises(ValueError, match="No valid sources to analyze"):
            await compare_sentiment_sources(sources)


@pytest.mark.asyncio
class TestErrorHandling:
    """Test error handling in sentiment analysis."""
    
    async def test_analyze_text_sentiment_with_invalid_input(self):
        """Test sentiment analysis with invalid input."""
        # Test with very long text
        long_text = "word " * 10000
        result = await analyze_text_sentiment(long_text)
        
        # Should still work but might have warnings
        assert isinstance(result, SentimentResult)
    
    async def test_batch_sentiment_with_mixed_input(self):
        """Test batch analysis with mixed valid/invalid input."""
        texts = [
            "Good product!",
            "",  # Empty text
            "Bad quality.",
            "   ",  # Whitespace only
            "Amazing!"
        ]
        
        result = await analyze_batch_sentiment(texts)
        
        # Should process valid texts
        assert isinstance(result, BatchSentimentResult)
        assert len(result.results) >= 3  # At least the clearly valid ones


class TestSentimentResult:
    """Test SentimentResult model."""
    
    def test_sentiment_result_creation(self):
        """Test SentimentResult creation."""
        sentiment_score = SentimentScore(
            polarity=SentimentPolarity.POSITIVE,
            confidence=0.8,
            score=0.6
        )
        
        emotion_score = EmotionScore(
            emotion=EmotionType.JOY,
            intensity=0.7,
            confidence=0.8
        )
        
        result = SentimentResult(
            text="Test text",
            sentiment=sentiment_score,
            emotions=[emotion_score],
            keywords=["good", "great"],
            language="en",
            processing_time_ms=150.0
        )
        
        assert result.text == "Test text"
        assert result.sentiment == sentiment_score
        assert len(result.emotions) == 1
        assert result.emotions[0] == emotion_score
        assert result.keywords == ["good", "great"]
        assert result.language == "en"
        assert result.processing_time_ms == 150.0
        assert isinstance(result.timestamp, datetime)


class TestBatchSentimentResult:
    """Test BatchSentimentResult model."""
    
    def test_batch_sentiment_result_creation(self):
        """Test BatchSentimentResult creation."""
        individual_results = []
        
        # Create some individual results
        for i in range(3):
            sentiment = SentimentScore(
                polarity=SentimentPolarity.POSITIVE,
                confidence=0.8,
                score=0.6
            )
            
            result = SentimentResult(
                text=f"Test text {i}",
                sentiment=sentiment,
                processing_time_ms=100.0
            )
            individual_results.append(result)
        
        summary = {
            "total_positive": 3,
            "average_score": 0.6
        }
        
        batch_result = BatchSentimentResult(
            results=individual_results,
            summary=summary,
            total_texts=3,
            processing_time_ms=300.0
        )
        
        assert len(batch_result.results) == 3
        assert batch_result.summary == summary
        assert batch_result.total_texts == 3
        assert batch_result.processing_time_ms == 300.0
        assert isinstance(batch_result.timestamp, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])