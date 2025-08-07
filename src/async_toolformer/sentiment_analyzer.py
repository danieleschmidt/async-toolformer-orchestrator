"""Quantum-Enhanced Sentiment Analysis Tools for Async Toolformer Orchestrator."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import aiohttp
import structlog
from pydantic import BaseModel, Field, validator

from .tools import Tool, ToolResult
from .quantum_orchestrator import QuantumAsyncOrchestrator
from .quantum_security import SecurityContext, SecurityLevel
from .exceptions import ToolExecutionError

logger = structlog.get_logger(__name__)


class SentimentPolarity(Enum):
    """Sentiment polarity classification."""
    POSITIVE = "positive"
    NEGATIVE = "negative" 
    NEUTRAL = "neutral"
    MIXED = "mixed"


class EmotionType(Enum):
    """Basic emotion classification."""
    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    ANTICIPATION = "anticipation"
    TRUST = "trust"


@dataclass
class SentimentScore:
    """Sentiment analysis score with confidence."""
    polarity: SentimentPolarity
    confidence: float
    score: float  # -1.0 (negative) to 1.0 (positive)
    
    def __post_init__(self):
        """Validate score ranges."""
        if not -1.0 <= self.score <= 1.0:
            raise ValueError("Score must be between -1.0 and 1.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass
class EmotionScore:
    """Emotion analysis result."""
    emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0


class SentimentResult(BaseModel):
    """Comprehensive sentiment analysis result."""
    text: str
    sentiment: SentimentScore
    emotions: List[EmotionScore] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    entities: Dict[str, Any] = Field(default_factory=dict)
    language: str = "en"
    processing_time_ms: float = 0.0
    analyzer_version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True


class BatchSentimentResult(BaseModel):
    """Batch sentiment analysis results."""
    results: List[SentimentResult]
    summary: Dict[str, Any]
    total_texts: int
    processing_time_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Sentiment Analysis Tools
@Tool(
    name="analyze_text_sentiment",
    description="Analyze sentiment of a single text using rule-based approach"
)
async def analyze_text_sentiment(
    text: str,
    include_emotions: bool = True,
    include_keywords: bool = True,
    language: str = "en"
) -> SentimentResult:
    """Analyze sentiment of text using rule-based approach."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Preprocess text
        clean_text = _preprocess_text(text)
        
        # Calculate sentiment score
        sentiment_score = _calculate_rule_based_sentiment(clean_text, language)
        
        # Analyze emotions if requested
        emotions = []
        if include_emotions:
            emotions = _analyze_emotions(clean_text)
            
        # Extract keywords if requested
        keywords = []
        if include_keywords:
            keywords = _extract_sentiment_keywords(clean_text)
            
        # Extract entities
        entities = _extract_entities(clean_text)
        
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return SentimentResult(
            text=text,
            sentiment=sentiment_score,
            emotions=emotions,
            keywords=keywords,
            entities=entities,
            language=language,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Error in sentiment analysis", error=str(e), text=text[:100])
        raise ToolExecutionError(f"Sentiment analysis failed: {e}")


@Tool(
    name="analyze_batch_sentiment", 
    description="Analyze sentiment of multiple texts in parallel"
)
async def analyze_batch_sentiment(
    texts: List[str],
    include_emotions: bool = True,
    include_keywords: bool = True,
    language: str = "en",
    max_parallel: int = 10
) -> BatchSentimentResult:
    """Analyze sentiment of multiple texts in parallel."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def analyze_single(text: str) -> SentimentResult:
            async with semaphore:
                return await analyze_text_sentiment(
                    text, include_emotions, include_keywords, language
                )
        
        # Process all texts in parallel
        results = await asyncio.gather(
            *[analyze_single(text) for text in texts],
            return_exceptions=True
        )
        
        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Failed to analyze text {i}", error=str(result))
            else:
                valid_results.append(result)
        
        # Calculate summary statistics
        summary = _calculate_batch_summary(valid_results)
        
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return BatchSentimentResult(
            results=valid_results,
            summary=summary,
            total_texts=len(texts),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Error in batch sentiment analysis", error=str(e))
        raise ToolExecutionError(f"Batch sentiment analysis failed: {e}")


@Tool(
    name="analyze_social_media_sentiment",
    description="Analyze sentiment from social media content with hashtag and mention extraction"
)
async def analyze_social_media_sentiment(
    text: str,
    platform: str = "twitter",
    extract_hashtags: bool = True,
    extract_mentions: bool = True
) -> SentimentResult:
    """Analyze sentiment from social media content."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Preprocess for social media
        clean_text = _preprocess_social_media_text(text, platform)
        
        # Calculate sentiment
        sentiment_score = _calculate_social_media_sentiment(clean_text, platform)
        
        # Extract social media specific entities
        entities = {}
        if extract_hashtags:
            entities["hashtags"] = _extract_hashtags(text)
        if extract_mentions:
            entities["mentions"] = _extract_mentions(text)
            
        # Add URL extraction
        entities["urls"] = _extract_urls(text)
        
        # Social media emotion analysis
        emotions = _analyze_social_media_emotions(clean_text)
        
        # Keywords specific to social media
        keywords = _extract_social_media_keywords(clean_text, platform)
        
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return SentimentResult(
            text=text,
            sentiment=sentiment_score,
            emotions=emotions,
            keywords=keywords,
            entities=entities,
            language="en",  # Could be enhanced with language detection
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error("Error in social media sentiment analysis", error=str(e))
        raise ToolExecutionError(f"Social media sentiment analysis failed: {e}")


@Tool(
    name="compare_sentiment_sources",
    description="Compare sentiment across multiple text sources and provide aggregated analysis"
)
async def compare_sentiment_sources(
    sources: Dict[str, str],  # source_name -> text
    weight_sources: bool = True,
    source_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Compare sentiment across multiple sources."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Analyze each source
        source_results = {}
        for source_name, text in sources.items():
            if text.strip():  # Skip empty texts
                result = await analyze_text_sentiment(text)
                source_results[source_name] = result
        
        if not source_results:
            raise ValueError("No valid sources to analyze")
            
        # Calculate weighted sentiment if requested
        if weight_sources and source_weights:
            weighted_sentiment = _calculate_weighted_sentiment(
                source_results, source_weights
            )
        else:
            weighted_sentiment = _calculate_average_sentiment(source_results)
            
        # Find consensus and disagreements
        consensus_analysis = _analyze_sentiment_consensus(source_results)
        
        # Aggregate emotions and keywords
        aggregated_emotions = _aggregate_emotions(source_results)
        aggregated_keywords = _aggregate_keywords(source_results)
        
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return {
            "source_results": {name: result.dict() for name, result in source_results.items()},
            "weighted_sentiment": weighted_sentiment,
            "consensus_analysis": consensus_analysis,
            "aggregated_emotions": [emotion.__dict__ for emotion in aggregated_emotions],
            "aggregated_keywords": aggregated_keywords,
            "processing_time_ms": processing_time,
            "total_sources": len(sources),
            "valid_sources": len(source_results)
        }
        
    except Exception as e:
        logger.error("Error in sentiment source comparison", error=str(e))
        raise ToolExecutionError(f"Sentiment source comparison failed: {e}")


# Helper Functions
def _preprocess_text(text: str) -> str:
    """Clean and preprocess text for sentiment analysis."""
    if not text:
        return ""
        
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Handle common abbreviations and contractions
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    
    return text


def _preprocess_social_media_text(text: str, platform: str) -> str:
    """Preprocess text specifically for social media platforms."""
    text = _preprocess_text(text)
    
    # Remove URLs but keep the context
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Handle platform-specific patterns
    if platform.lower() == "twitter":
        # Keep hashtags and mentions but normalize
        pass  # Hashtags and mentions are meaningful for sentiment
    elif platform.lower() == "reddit":
        # Handle Reddit-specific formatting
        text = re.sub(r'/u/\w+', '', text)  # Remove user mentions
        text = re.sub(r'/r/\w+', '', text)  # Remove subreddit mentions
        
    return text


def _calculate_rule_based_sentiment(text: str, language: str = "en") -> SentimentScore:
    """Calculate sentiment using rule-based approach."""
    # Simple lexicon-based approach (can be enhanced with VADER, TextBlob, etc.)
    positive_words = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic", 
        "love", "like", "happy", "joy", "pleased", "satisfied", "perfect",
        "awesome", "brilliant", "outstanding", "superb", "magnificent"
    }
    
    negative_words = {
        "bad", "terrible", "awful", "horrible", "hate", "dislike", "angry",
        "sad", "disappointed", "frustrated", "annoyed", "upset", "furious",
        "disgusting", "pathetic", "useless", "worthless", "stupid"
    }
    
    intensifiers = {"very", "extremely", "really", "quite", "so", "too"}
    negations = {"not", "no", "never", "nothing", "nobody", "nowhere", "neither", "nor"}
    
    words = text.lower().split()
    
    positive_score = 0
    negative_score = 0
    total_words = len(words)
    
    i = 0
    while i < len(words):
        word = words[i].strip('.,!?;:"()[]{}')
        
        # Check for negation
        negated = False
        if i > 0 and words[i-1] in negations:
            negated = True
            
        # Check for intensifier
        intensified = False
        if i > 0 and words[i-1] in intensifiers:
            intensified = True
            
        # Score the word
        base_score = 1
        if intensified:
            base_score = 1.5
            
        if word in positive_words:
            if negated:
                negative_score += base_score
            else:
                positive_score += base_score
        elif word in negative_words:
            if negated:
                positive_score += base_score
            else:
                negative_score += base_score
                
        i += 1
    
    # Calculate final score
    if total_words == 0:
        score = 0.0
        polarity = SentimentPolarity.NEUTRAL
        confidence = 0.5
    else:
        net_score = positive_score - negative_score
        score = max(-1.0, min(1.0, net_score / max(total_words * 0.1, 1)))
        
        # Determine polarity
        if score > 0.1:
            polarity = SentimentPolarity.POSITIVE
        elif score < -0.1:
            polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.NEUTRAL
            
        # Calculate confidence based on number of sentiment words
        sentiment_word_count = positive_score + negative_score
        confidence = min(1.0, sentiment_word_count / max(total_words * 0.3, 1))
        confidence = max(0.5, confidence)  # Minimum confidence
    
    return SentimentScore(polarity=polarity, confidence=confidence, score=score)


def _calculate_social_media_sentiment(text: str, platform: str) -> SentimentScore:
    """Calculate sentiment for social media content."""
    # Base sentiment calculation
    sentiment = _calculate_rule_based_sentiment(text)
    
    # Adjust for social media patterns
    emoji_score = _calculate_emoji_sentiment(text)
    caps_penalty = _calculate_caps_penalty(text)
    
    # Combine scores
    final_score = sentiment.score * 0.7 + emoji_score * 0.3
    final_score = max(-1.0, min(1.0, final_score - caps_penalty))
    
    # Adjust polarity based on final score
    if final_score > 0.1:
        polarity = SentimentPolarity.POSITIVE
    elif final_score < -0.1:
        polarity = SentimentPolarity.NEGATIVE
    else:
        polarity = SentimentPolarity.NEUTRAL
        
    return SentimentScore(
        polarity=polarity,
        confidence=sentiment.confidence,
        score=final_score
    )


def _calculate_emoji_sentiment(text: str) -> float:
    """Calculate sentiment contribution from emojis."""
    positive_emojis = "😊😀😁😂🤣😍🥰😘🤗🎉👍💕❤️🔥💯✨"
    negative_emojis = "😠😡🤬😢😭😤😒🙄💔👎😩😫🤮"
    
    positive_count = sum(1 for char in text if char in positive_emojis)
    negative_count = sum(1 for char in text if char in negative_emojis)
    
    if positive_count == 0 and negative_count == 0:
        return 0.0
        
    total = positive_count + negative_count
    return (positive_count - negative_count) / total * 0.5


def _calculate_caps_penalty(text: str) -> float:
    """Calculate penalty for excessive capitalization."""
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    return min(0.3, caps_ratio) if caps_ratio > 0.5 else 0.0


def _analyze_emotions(text: str) -> List[EmotionScore]:
    """Analyze emotions in text."""
    # Simple emotion word mapping
    emotion_lexicon = {
        EmotionType.JOY: {"happy", "joy", "excited", "thrilled", "elated", "cheerful"},
        EmotionType.ANGER: {"angry", "mad", "furious", "rage", "irritated", "annoyed"},
        EmotionType.SADNESS: {"sad", "depressed", "blue", "down", "miserable", "gloomy"},
        EmotionType.FEAR: {"afraid", "scared", "terrified", "anxious", "worried", "nervous"},
        EmotionType.SURPRISE: {"surprised", "amazed", "shocked", "astonished", "stunned"},
        EmotionType.DISGUST: {"disgusting", "revolting", "gross", "sick", "nasty"},
        EmotionType.ANTICIPATION: {"excited", "eager", "looking forward", "anticipating"},
        EmotionType.TRUST: {"trust", "confident", "reliable", "faithful", "loyal"}
    }
    
    words = text.lower().split()
    emotion_scores = []
    
    for emotion, emotion_words in emotion_lexicon.items():
        intensity = 0.0
        matches = 0
        
        for word in words:
            clean_word = word.strip('.,!?;:"()[]{}')
            if clean_word in emotion_words:
                intensity += 1.0
                matches += 1
                
        if matches > 0:
            # Normalize intensity by text length
            normalized_intensity = min(1.0, intensity / max(len(words) * 0.1, 1))
            confidence = min(1.0, matches / max(len(words) * 0.05, 1))
            
            emotion_scores.append(EmotionScore(
                emotion=emotion,
                intensity=normalized_intensity,
                confidence=max(0.3, confidence)
            ))
    
    return sorted(emotion_scores, key=lambda x: x.intensity, reverse=True)[:3]


def _analyze_social_media_emotions(text: str) -> List[EmotionScore]:
    """Analyze emotions specific to social media content."""
    # Use base emotion analysis but enhance with social media patterns
    emotions = _analyze_emotions(text)
    
    # Add social media specific emotion indicators
    if re.search(r'[!]{2,}', text):  # Multiple exclamations
        # Boost excitement/anger
        for emotion in emotions:
            if emotion.emotion in [EmotionType.JOY, EmotionType.ANGER]:
                emotion.intensity = min(1.0, emotion.intensity * 1.2)
                
    return emotions


def _extract_sentiment_keywords(text: str) -> List[str]:
    """Extract keywords relevant to sentiment."""
    # Simple keyword extraction (can be enhanced with TF-IDF, etc.)
    sentiment_indicators = {
        "love", "hate", "like", "dislike", "good", "bad", "great", "terrible",
        "amazing", "awful", "excellent", "horrible", "perfect", "worst",
        "best", "fantastic", "pathetic", "wonderful", "disgusting"
    }
    
    words = [word.strip('.,!?;:"()[]{}').lower() for word in text.split()]
    keywords = [word for word in words if word in sentiment_indicators]
    
    return list(set(keywords))[:10]  # Return unique keywords, max 10


def _extract_social_media_keywords(text: str, platform: str) -> List[str]:
    """Extract keywords specific to social media platforms."""
    keywords = _extract_sentiment_keywords(text)
    
    # Add hashtags as keywords
    hashtags = _extract_hashtags(text)
    keywords.extend([tag.lower() for tag in hashtags])
    
    return list(set(keywords))[:15]


def _extract_entities(text: str) -> Dict[str, Any]:
    """Extract basic entities from text."""
    # Simple entity extraction
    entities = {}
    
    # Extract potential names (capitalized words)
    names = re.findall(r'\b[A-Z][a-z]+\b', text)
    if names:
        entities["potential_names"] = list(set(names))
        
    # Extract numbers
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    if numbers:
        entities["numbers"] = numbers
        
    return entities


def _extract_hashtags(text: str) -> List[str]:
    """Extract hashtags from text."""
    return re.findall(r'#(\w+)', text)


def _extract_mentions(text: str) -> List[str]:
    """Extract mentions from text."""
    return re.findall(r'@(\w+)', text)


def _extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)


def _calculate_batch_summary(results: List[SentimentResult]) -> Dict[str, Any]:
    """Calculate summary statistics for batch results."""
    if not results:
        return {}
        
    sentiments = [r.sentiment for r in results]
    
    # Polarity distribution
    polarity_counts = {}
    for polarity in SentimentPolarity:
        polarity_counts[polarity.value] = sum(1 for s in sentiments if s.polarity == polarity)
    
    # Average scores
    avg_score = sum(s.score for s in sentiments) / len(sentiments)
    avg_confidence = sum(s.confidence for s in sentiments) / len(sentiments)
    
    # Processing time stats
    processing_times = [r.processing_time_ms for r in results]
    
    return {
        "polarity_distribution": polarity_counts,
        "average_sentiment_score": avg_score,
        "average_confidence": avg_confidence,
        "total_processing_time_ms": sum(processing_times),
        "average_processing_time_ms": sum(processing_times) / len(processing_times),
        "min_processing_time_ms": min(processing_times),
        "max_processing_time_ms": max(processing_times)
    }


def _calculate_weighted_sentiment(
    results: Dict[str, SentimentResult], 
    weights: Dict[str, float]
) -> Dict[str, Any]:
    """Calculate weighted sentiment across sources."""
    total_weight = 0
    weighted_score = 0
    weighted_confidence = 0
    
    for source, result in results.items():
        weight = weights.get(source, 1.0)
        total_weight += weight
        weighted_score += result.sentiment.score * weight
        weighted_confidence += result.sentiment.confidence * weight
    
    if total_weight == 0:
        return {"score": 0.0, "confidence": 0.5, "polarity": "neutral"}
        
    final_score = weighted_score / total_weight
    final_confidence = weighted_confidence / total_weight
    
    # Determine polarity
    if final_score > 0.1:
        polarity = "positive"
    elif final_score < -0.1:
        polarity = "negative"
    else:
        polarity = "neutral"
        
    return {
        "score": final_score,
        "confidence": final_confidence,
        "polarity": polarity,
        "total_weight": total_weight
    }


def _calculate_average_sentiment(results: Dict[str, SentimentResult]) -> Dict[str, Any]:
    """Calculate average sentiment across sources."""
    if not results:
        return {"score": 0.0, "confidence": 0.5, "polarity": "neutral"}
        
    scores = [result.sentiment.score for result in results.values()]
    confidences = [result.sentiment.confidence for result in results.values()]
    
    avg_score = sum(scores) / len(scores)
    avg_confidence = sum(confidences) / len(confidences)
    
    # Determine polarity
    if avg_score > 0.1:
        polarity = "positive"
    elif avg_score < -0.1:
        polarity = "negative"
    else:
        polarity = "neutral"
        
    return {
        "score": avg_score,
        "confidence": avg_confidence,
        "polarity": polarity
    }


def _analyze_sentiment_consensus(results: Dict[str, SentimentResult]) -> Dict[str, Any]:
    """Analyze consensus and disagreement in sentiment results."""
    if len(results) < 2:
        return {"consensus": "insufficient_data"}
        
    polarities = [result.sentiment.polarity for result in results.values()]
    scores = [result.sentiment.score for result in results.values()]
    
    # Check polarity agreement
    unique_polarities = set(polarities)
    polarity_agreement = len(unique_polarities) == 1
    
    # Check score variance
    if len(scores) > 1:
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
    else:
        std_dev = 0.0
    
    # Determine consensus level
    if polarity_agreement and std_dev < 0.3:
        consensus = "high"
    elif std_dev < 0.5:
        consensus = "moderate"
    else:
        consensus = "low"
        
    return {
        "consensus": consensus,
        "polarity_agreement": polarity_agreement,
        "score_variance": variance if len(scores) > 1 else 0.0,
        "score_std_dev": std_dev,
        "unique_polarities": list(unique_polarities)
    }


def _aggregate_emotions(results: Dict[str, SentimentResult]) -> List[EmotionScore]:
    """Aggregate emotions across all sources."""
    emotion_sums = {}
    emotion_counts = {}
    
    for result in results.values():
        for emotion_score in result.emotions:
            emotion = emotion_score.emotion
            if emotion not in emotion_sums:
                emotion_sums[emotion] = 0
                emotion_counts[emotion] = 0
            emotion_sums[emotion] += emotion_score.intensity
            emotion_counts[emotion] += 1
    
    aggregated = []
    for emotion, total_intensity in emotion_sums.items():
        avg_intensity = total_intensity / emotion_counts[emotion]
        confidence = min(1.0, emotion_counts[emotion] / len(results))
        
        aggregated.append(EmotionScore(
            emotion=emotion,
            intensity=avg_intensity,
            confidence=confidence
        ))
    
    return sorted(aggregated, key=lambda x: x.intensity, reverse=True)


def _aggregate_keywords(results: Dict[str, SentimentResult]) -> List[str]:
    """Aggregate keywords across all sources."""
    all_keywords = []
    for result in results.values():
        all_keywords.extend(result.keywords)
    
    # Count frequency and return most common
    keyword_counts = {}
    for keyword in all_keywords:
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
    
    # Sort by frequency
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    
    return [keyword for keyword, count in sorted_keywords[:20]]  # Top 20 keywords