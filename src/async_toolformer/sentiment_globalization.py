"""Global-first sentiment analysis with multi-language support and compliance."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog

from .sentiment_analyzer import SentimentResult, SentimentPolarity, EmotionType, EmotionScore
from .sentiment_validation import SentimentValidator, ValidationResult
from .tools import Tool
from .exceptions import ToolExecutionError, ConfigurationError

logger = structlog.get_logger(__name__)


class SupportedLanguage(Enum):
    """Supported languages with ISO codes."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-cn"
    CHINESE_TRADITIONAL = "zh-tw"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"


class ComplianceRegion(Enum):
    """Compliance regions with different data protection laws."""
    EU = "eu"           # GDPR
    US = "us"           # CCPA
    CANADA = "ca"       # PIPEDA
    UK = "uk"           # UK GDPR
    SINGAPORE = "sg"    # PDPA
    BRAZIL = "br"       # LGPD
    AUSTRALIA = "au"    # Privacy Act
    JAPAN = "jp"        # APPI
    SOUTH_KOREA = "kr"  # PIPA
    GLOBAL = "global"   # Strictest compliance


@dataclass
class LanguageConfig:
    """Configuration for language-specific sentiment analysis."""
    language: SupportedLanguage
    sentiment_lexicon: Dict[str, Dict[str, float]] = field(default_factory=dict)
    cultural_adjustments: Dict[str, float] = field(default_factory=dict)
    formality_indicators: List[str] = field(default_factory=list)
    negation_patterns: List[str] = field(default_factory=list)
    intensifier_patterns: List[str] = field(default_factory=list)
    emotion_expressions: Dict[EmotionType, List[str]] = field(default_factory=dict)


@dataclass
class ComplianceSettings:
    """Settings for regulatory compliance."""
    region: ComplianceRegion
    data_retention_days: int = 30
    anonymization_required: bool = True
    consent_required: bool = True
    audit_logging: bool = True
    cross_border_transfer_allowed: bool = False
    sensitive_data_categories: Set[str] = field(default_factory=set)
    data_subject_rights: Set[str] = field(default_factory=set)


class MultiLanguageSentimentAnalyzer:
    """Multi-language sentiment analysis with cultural awareness."""
    
    def __init__(self):
        """Initialize multi-language analyzer."""
        self.language_configs = self._initialize_language_configs()
        self.language_detector = SimpleLanguageDetector()
        
    def _initialize_language_configs(self) -> Dict[SupportedLanguage, LanguageConfig]:
        """Initialize language-specific configurations."""
        configs = {}
        
        # English configuration
        configs[SupportedLanguage.ENGLISH] = LanguageConfig(
            language=SupportedLanguage.ENGLISH,
            sentiment_lexicon={
                "positive": {
                    "excellent": 0.9, "amazing": 0.8, "great": 0.7, "good": 0.6,
                    "wonderful": 0.8, "fantastic": 0.9, "brilliant": 0.8, "perfect": 0.9,
                    "outstanding": 0.9, "superb": 0.8, "marvelous": 0.8, "terrific": 0.8
                },
                "negative": {
                    "terrible": -0.9, "awful": -0.8, "horrible": -0.9, "bad": -0.6,
                    "disgusting": -0.9, "pathetic": -0.8, "useless": -0.7, "worthless": -0.8,
                    "disappointing": -0.6, "frustrating": -0.6, "annoying": -0.5
                }
            },
            cultural_adjustments={"politeness_boost": 0.1, "directness_factor": 1.0},
            negation_patterns=["not", "no", "never", "nothing", "nowhere", "nobody", "none"],
            intensifier_patterns=["very", "extremely", "really", "quite", "so", "too", "absolutely"],
            emotion_expressions={
                EmotionType.JOY: ["happy", "joyful", "excited", "thrilled", "elated", "cheerful"],
                EmotionType.ANGER: ["angry", "furious", "mad", "irritated", "annoyed", "outraged"],
                EmotionType.SADNESS: ["sad", "depressed", "blue", "down", "miserable", "gloomy"],
                EmotionType.FEAR: ["afraid", "scared", "terrified", "anxious", "worried", "nervous"]
            }
        )
        
        # Spanish configuration
        configs[SupportedLanguage.SPANISH] = LanguageConfig(
            language=SupportedLanguage.SPANISH,
            sentiment_lexicon={
                "positive": {
                    "excelente": 0.9, "increíble": 0.8, "genial": 0.7, "bueno": 0.6,
                    "maravilloso": 0.8, "fantástico": 0.9, "brillante": 0.8, "perfecto": 0.9,
                    "estupendo": 0.8, "magnífico": 0.8, "espléndido": 0.8
                },
                "negative": {
                    "terrible": -0.9, "horrible": -0.9, "malo": -0.6, "pésimo": -0.8,
                    "disgustante": -0.9, "patético": -0.8, "inútil": -0.7, "decepcionante": -0.6,
                    "frustrante": -0.6, "molesto": -0.5
                }
            },
            cultural_adjustments={"expressiveness_boost": 0.2, "formality_respect": 0.15},
            negation_patterns=["no", "nunca", "nada", "nadie", "ningún", "jamás"],
            intensifier_patterns=["muy", "extremadamente", "realmente", "bastante", "tan", "súper"],
            emotion_expressions={
                EmotionType.JOY: ["feliz", "alegre", "emocionado", "contento", "gozoso"],
                EmotionType.ANGER: ["enojado", "furioso", "molesto", "irritado", "indignado"],
                EmotionType.SADNESS: ["triste", "deprimido", "melancólico", "abatido"],
                EmotionType.FEAR: ["asustado", "temeroso", "ansioso", "preocupado", "nervioso"]
            }
        )
        
        # French configuration
        configs[SupportedLanguage.FRENCH] = LanguageConfig(
            language=SupportedLanguage.FRENCH,
            sentiment_lexicon={
                "positive": {
                    "excellent": 0.9, "incroyable": 0.8, "génial": 0.7, "bon": 0.6,
                    "merveilleux": 0.8, "fantastique": 0.9, "brillant": 0.8, "parfait": 0.9,
                    "superbe": 0.8, "magnifique": 0.8, "splendide": 0.8
                },
                "negative": {
                    "terrible": -0.9, "horrible": -0.9, "mauvais": -0.6, "affreux": -0.8,
                    "dégoûtant": -0.9, "pathétique": -0.8, "inutile": -0.7, "décevant": -0.6,
                    "frustrant": -0.6, "ennuyeux": -0.5
                }
            },
            cultural_adjustments={"sophistication_factor": 0.1, "formality_preference": 0.2},
            negation_patterns=["ne", "pas", "non", "jamais", "rien", "personne", "aucun"],
            intensifier_patterns=["très", "extrêmement", "vraiment", "assez", "si", "tellement"],
            emotion_expressions={
                EmotionType.JOY: ["heureux", "joyeux", "excité", "ravi", "enchanté"],
                EmotionType.ANGER: ["en colère", "furieux", "irrité", "fâché", "indigné"],
                EmotionType.SADNESS: ["triste", "déprimé", "mélancolique", "abattu"],
                EmotionType.FEAR: ["effrayé", "craintif", "anxieux", "inquiet", "nerveux"]
            }
        )
        
        # German configuration
        configs[SupportedLanguage.GERMAN] = LanguageConfig(
            language=SupportedLanguage.GERMAN,
            sentiment_lexicon={
                "positive": {
                    "ausgezeichnet": 0.9, "unglaublich": 0.8, "großartig": 0.7, "gut": 0.6,
                    "wunderbar": 0.8, "fantastisch": 0.9, "brillant": 0.8, "perfekt": 0.9,
                    "hervorragend": 0.8, "prächtig": 0.8, "vorzüglich": 0.8
                },
                "negative": {
                    "schrecklich": -0.9, "furchtbar": -0.9, "schlecht": -0.6, "entsetzlich": -0.8,
                    "widerlich": -0.9, "erbärmlich": -0.8, "nutzlos": -0.7, "enttäuschend": -0.6,
                    "frustrierend": -0.6, "ärgerlich": -0.5
                }
            },
            cultural_adjustments={"precision_value": 0.15, "directness_acceptance": 0.1},
            negation_patterns=["nicht", "kein", "keine", "niemals", "nichts", "niemand"],
            intensifier_patterns=["sehr", "äußerst", "wirklich", "ziemlich", "so", "absolut"],
            emotion_expressions={
                EmotionType.JOY: ["glücklich", "freudig", "aufgeregt", "fröhlich", "heiter"],
                EmotionType.ANGER: ["wütend", "zornig", "verärgert", "irritiert", "empört"],
                EmotionType.SADNESS: ["traurig", "deprimiert", "niedergeschlagen", "bedrückt"],
                EmotionType.FEAR: ["ängstlich", "verängstigt", "besorgt", "nervös", "unruhig"]
            }
        )
        
        # Japanese configuration (basic)
        configs[SupportedLanguage.JAPANESE] = LanguageConfig(
            language=SupportedLanguage.JAPANESE,
            sentiment_lexicon={
                "positive": {
                    "素晴らしい": 0.9, "最高": 0.9, "良い": 0.6, "完璧": 0.9,
                    "優秀": 0.8, "美しい": 0.7, "楽しい": 0.7
                },
                "negative": {
                    "最悪": -0.9, "悪い": -0.6, "ひどい": -0.8, "だめ": -0.6,
                    "つまらない": -0.5, "がっかり": -0.6
                }
            },
            cultural_adjustments={"politeness_critical": 0.3, "context_dependency": 0.4},
            formality_indicators=["です", "ます", "である", "でございます"],
            emotion_expressions={
                EmotionType.JOY: ["嬉しい", "楽しい", "喜ぶ", "幸せ"],
                EmotionType.ANGER: ["怒り", "腹立つ", "イライラ", "むかつく"],
                EmotionType.SADNESS: ["悲しい", "寂しい", "憂鬱", "落ち込む"],
                EmotionType.FEAR: ["怖い", "不安", "心配", "恐れる"]
            }
        )
        
        return configs
    
    async def analyze_multilingual_sentiment(
        self, 
        text: str, 
        language: Optional[SupportedLanguage] = None,
        cultural_context: Optional[Dict[str, Any]] = None
    ) -> SentimentResult:
        """Analyze sentiment with language-specific processing."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Detect language if not provided
            if language is None:
                detected_lang = await self.language_detector.detect_language(text)
                language = detected_lang if detected_lang in self.language_configs else SupportedLanguage.ENGLISH
            
            # Get language configuration
            lang_config = self.language_configs.get(language, self.language_configs[SupportedLanguage.ENGLISH])
            
            # Perform language-specific sentiment analysis
            sentiment_score = await self._analyze_sentiment_with_language_config(text, lang_config, cultural_context)
            
            # Extract emotions with language-specific expressions
            emotions = self._extract_language_specific_emotions(text, lang_config)
            
            # Extract keywords
            keywords = self._extract_language_keywords(text, lang_config)
            
            processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return SentimentResult(
                text=text,
                sentiment=sentiment_score,
                emotions=emotions,
                keywords=keywords,
                language=language.value,
                processing_time_ms=processing_time,
                analyzer_version="multilingual-1.0.0"
            )
            
        except Exception as e:
            logger.error("Multilingual sentiment analysis failed", error=str(e), language=language)
            raise ToolExecutionError(f"Multilingual sentiment analysis failed: {e}")
    
    async def _analyze_sentiment_with_language_config(
        self, 
        text: str, 
        lang_config: LanguageConfig,
        cultural_context: Optional[Dict[str, Any]]
    ) -> 'SentimentScore':
        """Analyze sentiment using language-specific configuration."""
        from .sentiment_analyzer import SentimentScore
        
        words = text.lower().split()
        positive_score = 0.0
        negative_score = 0.0
        total_sentiment_words = 0
        
        # Process each word
        i = 0
        while i < len(words):
            word = words[i].strip('.,!?;:"()[]{}')
            
            # Check for negation
            negated = False
            if i > 0 and any(neg in words[i-1] for neg in lang_config.negation_patterns):
                negated = True
            
            # Check for intensifiers
            intensified = False
            intensity_factor = 1.0
            if i > 0 and any(intens in words[i-1] for intens in lang_config.intensifier_patterns):
                intensified = True
                intensity_factor = 1.5
            
            # Score positive words
            if word in lang_config.sentiment_lexicon.get("positive", {}):
                score = lang_config.sentiment_lexicon["positive"][word] * intensity_factor
                if negated:
                    negative_score += score
                else:
                    positive_score += score
                total_sentiment_words += 1
            
            # Score negative words
            elif word in lang_config.sentiment_lexicon.get("negative", {}):
                score = abs(lang_config.sentiment_lexicon["negative"][word]) * intensity_factor
                if negated:
                    positive_score += score
                else:
                    negative_score += score
                total_sentiment_words += 1
            
            i += 1
        
        # Calculate base sentiment score
        if total_sentiment_words == 0:
            base_score = 0.0
            confidence = 0.5
        else:
            net_score = positive_score - negative_score
            base_score = max(-1.0, min(1.0, net_score / max(len(words) * 0.1, 1)))
            confidence = min(1.0, total_sentiment_words / max(len(words) * 0.3, 1))
            confidence = max(0.3, confidence)
        
        # Apply cultural adjustments
        if cultural_context and lang_config.cultural_adjustments:
            for adjustment, factor in lang_config.cultural_adjustments.items():
                if adjustment == "politeness_boost" and self._is_polite_text(text, lang_config):
                    base_score *= (1 + factor * 0.1)
                elif adjustment == "expressiveness_boost":
                    exclamation_count = text.count('!')
                    if exclamation_count > 1:
                        base_score *= (1 + factor * min(0.3, exclamation_count * 0.1))
                elif adjustment == "formality_preference" and self._is_formal_text(text, lang_config):
                    confidence *= (1 + factor * 0.1)
        
        # Determine polarity
        if base_score > 0.1:
            polarity = SentimentPolarity.POSITIVE
        elif base_score < -0.1:
            polarity = SentimentPolarity.NEGATIVE
        else:
            polarity = SentimentPolarity.NEUTRAL
        
        return SentimentScore(
            polarity=polarity,
            confidence=min(1.0, confidence),
            score=max(-1.0, min(1.0, base_score))
        )
    
    def _extract_language_specific_emotions(self, text: str, lang_config: LanguageConfig) -> List[EmotionScore]:
        """Extract emotions using language-specific expressions."""
        emotions = []
        text_lower = text.lower()
        words = text_lower.split()
        
        for emotion_type, expressions in lang_config.emotion_expressions.items():
            intensity = 0.0
            matches = 0
            
            for expression in expressions:
                if expression in text_lower:
                    intensity += 1.0
                    matches += 1
            
            if matches > 0:
                normalized_intensity = min(1.0, intensity / max(len(words) * 0.1, 1))
                confidence = min(1.0, matches / max(len(words) * 0.05, 1))
                
                emotions.append(EmotionScore(
                    emotion=emotion_type,
                    intensity=normalized_intensity,
                    confidence=max(0.3, confidence)
                ))
        
        return sorted(emotions, key=lambda x: x.intensity, reverse=True)[:3]
    
    def _extract_language_keywords(self, text: str, lang_config: LanguageConfig) -> List[str]:
        """Extract sentiment keywords for the specific language."""
        keywords = []
        words = [word.strip('.,!?;:"()[]{}').lower() for word in text.split()]
        
        # Extract positive keywords
        for word in words:
            if word in lang_config.sentiment_lexicon.get("positive", {}):
                keywords.append(word)
        
        # Extract negative keywords
        for word in words:
            if word in lang_config.sentiment_lexicon.get("negative", {}):
                keywords.append(word)
        
        return list(set(keywords))[:10]
    
    def _is_polite_text(self, text: str, lang_config: LanguageConfig) -> bool:
        """Check if text contains polite language patterns."""
        polite_indicators = {
            SupportedLanguage.ENGLISH: ["please", "thank you", "thanks", "kindly", "appreciate"],
            SupportedLanguage.SPANISH: ["por favor", "gracias", "muchas gracias", "amablemente"],
            SupportedLanguage.FRENCH: ["s'il vous plaît", "merci", "merci beaucoup", "aimablement"],
            SupportedLanguage.GERMAN: ["bitte", "danke", "vielen dank", "freundlich"],
            SupportedLanguage.JAPANESE: ["お願い", "ありがとう", "すみません", "恐れ入ります"]
        }
        
        indicators = polite_indicators.get(lang_config.language, [])
        return any(indicator in text.lower() for indicator in indicators)
    
    def _is_formal_text(self, text: str, lang_config: LanguageConfig) -> bool:
        """Check if text uses formal language patterns."""
        if not lang_config.formality_indicators:
            return False
        
        return any(indicator in text for indicator in lang_config.formality_indicators)


class SimpleLanguageDetector:
    """Simple language detection based on character patterns and common words."""
    
    def __init__(self):
        """Initialize language detector."""
        self.language_indicators = {
            SupportedLanguage.ENGLISH: {
                "common_words": ["the", "and", "is", "in", "to", "of", "a", "that", "it", "for"],
                "char_patterns": []
            },
            SupportedLanguage.SPANISH: {
                "common_words": ["el", "la", "de", "que", "y", "en", "un", "es", "se", "no"],
                "char_patterns": ["ñ", "á", "é", "í", "ó", "ú"]
            },
            SupportedLanguage.FRENCH: {
                "common_words": ["le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir"],
                "char_patterns": ["ç", "à", "é", "è", "ê", "ë", "î", "ï", "ô", "û", "ù"]
            },
            SupportedLanguage.GERMAN: {
                "common_words": ["der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich"],
                "char_patterns": ["ä", "ö", "ü", "ß"]
            },
            SupportedLanguage.JAPANESE: {
                "common_words": ["です", "である", "の", "に", "を", "は", "が", "と", "で"],
                "char_patterns": ["ひ", "か", "た", "な", "ら", "わ", "が", "ざ", "だ", "ば"]
            }
        }
    
    async def detect_language(self, text: str) -> Optional[SupportedLanguage]:
        """Detect language of the given text."""
        if not text.strip():
            return None
        
        text_lower = text.lower()
        language_scores = {}
        
        for language, indicators in self.language_indicators.items():
            score = 0.0
            
            # Check common words
            common_words = indicators["common_words"]
            words = text_lower.split()
            common_word_count = sum(1 for word in words if word in common_words)
            if words:
                score += (common_word_count / len(words)) * 0.7
            
            # Check character patterns
            char_patterns = indicators["char_patterns"]
            if char_patterns:
                char_pattern_count = sum(1 for char in char_patterns if char in text_lower)
                score += min(0.3, char_pattern_count * 0.1)
            
            language_scores[language] = score
        
        # Return language with highest score
        if language_scores:
            best_language = max(language_scores.items(), key=lambda x: x[1])
            if best_language[1] > 0.1:  # Minimum confidence threshold
                return best_language[0]
        
        return None


class GlobalComplianceManager:
    """Manage global regulatory compliance for sentiment analysis."""
    
    def __init__(self):
        """Initialize compliance manager."""
        self.compliance_configs = self._initialize_compliance_configs()
    
    def _initialize_compliance_configs(self) -> Dict[ComplianceRegion, ComplianceSettings]:
        """Initialize compliance configurations for different regions."""
        configs = {}
        
        # EU GDPR
        configs[ComplianceRegion.EU] = ComplianceSettings(
            region=ComplianceRegion.EU,
            data_retention_days=90,
            anonymization_required=True,
            consent_required=True,
            audit_logging=True,
            cross_border_transfer_allowed=False,
            sensitive_data_categories={"personal", "biometric", "health", "racial", "political"},
            data_subject_rights={"access", "rectification", "erasure", "portability", "restriction", "objection"}
        )
        
        # US CCPA
        configs[ComplianceRegion.US] = ComplianceSettings(
            region=ComplianceRegion.US,
            data_retention_days=365,
            anonymization_required=False,
            consent_required=False,  # Opt-out model
            audit_logging=True,
            cross_border_transfer_allowed=True,
            sensitive_data_categories={"personal", "biometric", "geolocation", "financial"},
            data_subject_rights={"access", "deletion", "opt_out"}
        )
        
        # Singapore PDPA
        configs[ComplianceRegion.SINGAPORE] = ComplianceSettings(
            region=ComplianceRegion.SINGAPORE,
            data_retention_days=180,
            anonymization_required=True,
            consent_required=True,
            audit_logging=True,
            cross_border_transfer_allowed=False,  # Requires adequate protection
            sensitive_data_categories={"personal", "financial", "health"},
            data_subject_rights={"access", "correction", "withdrawal"}
        )
        
        # Global (strictest requirements)
        configs[ComplianceRegion.GLOBAL] = ComplianceSettings(
            region=ComplianceRegion.GLOBAL,
            data_retention_days=30,  # Shortest retention
            anonymization_required=True,
            consent_required=True,
            audit_logging=True,
            cross_border_transfer_allowed=False,
            sensitive_data_categories={"personal", "biometric", "health", "racial", "political", "financial", "geolocation"},
            data_subject_rights={"access", "rectification", "erasure", "portability", "restriction", "objection", "opt_out"}
        )
        
        return configs
    
    def get_compliance_requirements(self, region: ComplianceRegion) -> ComplianceSettings:
        """Get compliance requirements for a specific region."""
        return self.compliance_configs.get(region, self.compliance_configs[ComplianceRegion.GLOBAL])
    
    def validate_compliance(
        self, 
        text: str, 
        region: ComplianceRegion,
        user_consent: bool = False,
        data_categories: Optional[Set[str]] = None
    ) -> ValidationResult:
        """Validate compliance requirements."""
        from .sentiment_validation import ValidationResult, ValidationIssue, ValidationSeverity
        
        compliance_settings = self.get_compliance_requirements(region)
        issues = []
        warnings = []
        
        # Check consent requirements
        if compliance_settings.consent_required and not user_consent:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                issue_type="missing_consent",
                message=f"User consent required for {region.value} compliance",
                suggestion="Obtain explicit user consent before processing"
            ))
        
        # Check for sensitive data categories
        if data_categories:
            forbidden_categories = data_categories.intersection(compliance_settings.sensitive_data_categories)
            if forbidden_categories:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    issue_type="sensitive_data_detected",
                    message=f"Sensitive data categories detected: {forbidden_categories}",
                    suggestion="Remove or anonymize sensitive data before processing"
                ))
        
        # Check text for potential personal information
        if self._contains_personal_info(text):
            if compliance_settings.anonymization_required:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.HIGH,
                    issue_type="personal_info_detected",
                    message="Personal information detected in text",
                    suggestion="Anonymize or remove personal information"
                ))
            else:
                warnings.append("Personal information detected - consider anonymization")
        
        # Check data retention implications
        if compliance_settings.data_retention_days < 90:
            warnings.append(f"Short data retention period: {compliance_settings.data_retention_days} days")
        
        confidence_score = 1.0
        if issues:
            # Reduce confidence based on severity of issues
            critical_issues = sum(1 for issue in issues if issue.severity == ValidationSeverity.CRITICAL)
            high_issues = sum(1 for issue in issues if issue.severity == ValidationSeverity.HIGH)
            confidence_score = max(0.1, 1.0 - (critical_issues * 0.5) - (high_issues * 0.2))
        
        return ValidationResult(
            is_valid=not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues),
            issues=issues,
            warnings=warnings,
            confidence_score=confidence_score,
            processing_safe=len([issue for issue in issues if issue.severity == ValidationSeverity.CRITICAL]) == 0
        )
    
    def _contains_personal_info(self, text: str) -> bool:
        """Check if text contains potential personal information."""
        # Email pattern
        if re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
            return True
        
        # Phone number patterns
        if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text):
            return True
        
        # Credit card-like patterns
        if re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', text):
            return True
        
        # Social security number-like patterns
        if re.search(r'\b\d{3}-\d{2}-\d{4}\b', text):
            return True
        
        return False
    
    def anonymize_text(self, text: str, region: ComplianceRegion) -> str:
        """Anonymize text according to regional compliance requirements."""
        compliance_settings = self.get_compliance_requirements(region)
        
        if not compliance_settings.anonymization_required:
            return text
        
        anonymized_text = text
        
        # Replace email addresses
        anonymized_text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL_REDACTED]',
            anonymized_text
        )
        
        # Replace phone numbers
        anonymized_text = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE_REDACTED]',
            anonymized_text
        )
        
        # Replace credit card numbers
        anonymized_text = re.sub(
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
            '[CARD_REDACTED]',
            anonymized_text
        )
        
        # Replace SSN-like numbers
        anonymized_text = re.sub(
            r'\b\d{3}-\d{2}-\d{4}\b',
            '[SSN_REDACTED]',
            anonymized_text
        )
        
        # Replace potential names (simple heuristic)
        anonymized_text = re.sub(
            r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b',
            '[NAME_REDACTED]',
            anonymized_text
        )
        
        return anonymized_text


@Tool(
    name="global_sentiment_analysis",
    description="Perform sentiment analysis with multi-language support and global compliance"
)
async def global_sentiment_analysis(
    text: str,
    language: Optional[str] = None,
    compliance_region: str = "global",
    user_consent: bool = False,
    cultural_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Perform global sentiment analysis with compliance checks."""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Parse inputs
        lang = None
        if language:
            try:
                lang = SupportedLanguage(language.lower())
            except ValueError:
                logger.warning(f"Unsupported language: {language}, defaulting to English")
                lang = SupportedLanguage.ENGLISH
        
        try:
            region = ComplianceRegion(compliance_region.lower())
        except ValueError:
            logger.warning(f"Unsupported compliance region: {compliance_region}, defaulting to global")
            region = ComplianceRegion.GLOBAL
        
        # Initialize components
        multilang_analyzer = MultiLanguageSentimentAnalyzer()
        compliance_manager = GlobalComplianceManager()
        
        # Compliance validation
        compliance_validation = compliance_manager.validate_compliance(
            text=text,
            region=region,
            user_consent=user_consent
        )
        
        if not compliance_validation.processing_safe:
            raise ToolExecutionError(f"Compliance validation failed: {compliance_validation.issues[0].message}")
        
        # Anonymize text if required
        processed_text = compliance_manager.anonymize_text(text, region)
        
        # Perform multilingual sentiment analysis
        sentiment_result = await multilang_analyzer.analyze_multilingual_sentiment(
            text=processed_text,
            language=lang,
            cultural_context=cultural_context
        )
        
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return {
            "sentiment_analysis": sentiment_result.dict(),
            "language_detected": sentiment_result.language,
            "compliance_region": region.value,
            "compliance_validation": {
                "is_compliant": compliance_validation.is_valid,
                "issues": [issue.__dict__ for issue in compliance_validation.issues],
                "warnings": compliance_validation.warnings
            },
            "text_anonymized": processed_text != text,
            "cultural_adjustments_applied": bool(cultural_context),
            "global_processing_time_ms": processing_time,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error("Global sentiment analysis failed", error=str(e))
        raise ToolExecutionError(f"Global sentiment analysis failed: {e}")


# Utility functions for easy access
def get_supported_languages() -> List[str]:
    """Get list of supported language codes."""
    return [lang.value for lang in SupportedLanguage]


def get_supported_regions() -> List[str]:
    """Get list of supported compliance regions."""
    return [region.value for region in ComplianceRegion]


def create_language_config(language_code: str) -> Optional[LanguageConfig]:
    """Create language configuration for a specific language."""
    try:
        lang = SupportedLanguage(language_code.lower())
        analyzer = MultiLanguageSentimentAnalyzer()
        return analyzer.language_configs.get(lang)
    except ValueError:
        return None