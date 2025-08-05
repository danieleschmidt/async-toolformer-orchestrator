"""
Internationalization (i18n) Module for Quantum AsyncOrchestrator.

This module provides comprehensive internationalization support:
- Multi-language error messages and logs
- Localized quantum state descriptions
- Regional compliance features
- Cultural adaptation for different markets
"""

import logging
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, field
import json
import os

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ITALIAN = "it"
    DUTCH = "nl"
    ARABIC = "ar"
    HINDI = "hi"


class RegionalCompliance(Enum):
    """Regional compliance frameworks."""
    GDPR = "gdpr"          # European Union
    CCPA = "ccpa"          # California, USA
    PDPA = "pdpa"          # Singapore, Thailand
    LGPD = "lgpd"          # Brazil
    PIPEDA = "pipeda"      # Canada
    APPI = "appi"          # Japan
    PIPL = "pipl"          # China
    DPA = "dpa"            # UK


@dataclass
class LocalizationConfig:
    """Configuration for localization settings."""
    primary_language: SupportedLanguage = SupportedLanguage.ENGLISH
    fallback_language: SupportedLanguage = SupportedLanguage.ENGLISH
    regional_compliance: List[RegionalCompliance] = field(default_factory=list)
    timezone: str = "UTC"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    number_format: str = "en_US"
    currency: str = "USD"
    enable_rtl: bool = False  # Right-to-left languages
    custom_translations: Dict[str, Dict[str, str]] = field(default_factory=dict)


class QuantumInternationalization:
    """
    Internationalization manager for quantum orchestrator.
    
    Provides localized messages, compliance features, and cultural adaptations
    for global deployment of quantum-enhanced task orchestration.
    """
    
    def __init__(self, config: LocalizationConfig = None):
        """
        Initialize internationalization manager.
        
        Args:
            config: Localization configuration
        """
        self.config = config or LocalizationConfig()
        self._translations: Dict[str, Dict[str, str]] = {}
        self._compliance_rules: Dict[RegionalCompliance, Dict[str, Any]] = {}
        
        # Initialize built-in translations
        self._load_builtin_translations()
        
        # Initialize compliance rules
        self._load_compliance_rules()
        
        logger.info(f"Quantum i18n initialized for {self.config.primary_language.value}")
    
    def _load_builtin_translations(self):
        """Load built-in translations for supported languages."""
        
        # Core system messages
        self._translations = {
            SupportedLanguage.ENGLISH.value: {
                # Quantum planner messages
                "quantum.planner.initialized": "Quantum planner initialized with {max_tasks} parallel tasks",
                "quantum.planner.task_registered": "Quantum task '{task_name}' registered successfully",
                "quantum.planner.plan_created": "Execution plan created with {phases} phases",
                "quantum.planner.optimization_complete": "Quantum optimization completed with score {score}",
                "quantum.planner.coherence_violation": "Quantum coherence violation detected",
                "quantum.planner.entanglement_created": "Quantum entanglement created between tasks",
                
                # Security messages
                "security.context_created": "Security context created for user {user_id}",
                "security.context_invalid": "Invalid security context",
                "security.context_expired": "Security context has expired",
                "security.input_blocked": "Input blocked due to security policy",
                "security.access_denied": "Access denied to resource {resource}",
                "security.audit_entry": "Security audit entry recorded",
                
                # Performance messages
                "performance.optimization_applied": "Performance optimization applied",
                "performance.scaling_triggered": "Resource scaling triggered",
                "performance.bottleneck_detected": "Performance bottleneck detected",
                "performance.metrics_collected": "Performance metrics collected",
                
                # Execution messages
                "execution.started": "Quantum execution started",
                "execution.completed": "Execution completed successfully",
                "execution.failed": "Execution failed with error",
                "execution.timeout": "Execution timed out",
                "execution.cancelled": "Execution was cancelled",
                
                # Error messages
                "error.configuration": "Configuration error: {details}",
                "error.validation": "Validation error: {field} - {message}",
                "error.tool_execution": "Tool execution failed: {tool_name}",
                "error.rate_limit": "Rate limit exceeded",
                "error.resource_exhausted": "Resource exhausted: {resource}",
                
                # Compliance messages
                "compliance.gdpr_enabled": "GDPR compliance features enabled",
                "compliance.data_processed": "Personal data processed under compliance framework",
                "compliance.audit_required": "Audit trail required for compliance",
                "compliance.retention_policy": "Data retention policy applied",
            },
            
            SupportedLanguage.SPANISH.value: {
                "quantum.planner.initialized": "Planificador cuántico inicializado con {max_tasks} tareas paralelas",
                "quantum.planner.task_registered": "Tarea cuántica '{task_name}' registrada exitosamente",
                "quantum.planner.plan_created": "Plan de ejecución creado con {phases} fases",
                "quantum.planner.optimization_complete": "Optimización cuántica completada con puntuación {score}",
                "security.context_created": "Contexto de seguridad creado para usuario {user_id}",
                "security.context_invalid": "Contexto de seguridad inválido",
                "security.access_denied": "Acceso denegado al recurso {resource}",
                "execution.started": "Ejecución cuántica iniciada",
                "execution.completed": "Ejecución completada exitosamente",
                "execution.failed": "Ejecución falló con error",
                "error.configuration": "Error de configuración: {details}",
                "error.validation": "Error de validación: {field} - {message}",
                "compliance.gdpr_enabled": "Funciones de cumplimiento GDPR habilitadas",
            },
            
            SupportedLanguage.FRENCH.value: {
                "quantum.planner.initialized": "Planificateur quantique initialisé avec {max_tasks} tâches parallèles",
                "quantum.planner.task_registered": "Tâche quantique '{task_name}' enregistrée avec succès",
                "quantum.planner.plan_created": "Plan d'exécution créé avec {phases} phases",
                "quantum.planner.optimization_complete": "Optimisation quantique terminée avec le score {score}",
                "security.context_created": "Contexte de sécurité créé pour l'utilisateur {user_id}",
                "security.context_invalid": "Contexte de sécurité invalide",
                "security.access_denied": "Accès refusé à la ressource {resource}",
                "execution.started": "Exécution quantique démarrée",
                "execution.completed": "Exécution terminée avec succès",
                "execution.failed": "L'exécution a échoué avec une erreur",
                "error.configuration": "Erreur de configuration: {details}",
                "error.validation": "Erreur de validation: {field} - {message}",
                "compliance.gdpr_enabled": "Fonctionnalités de conformité GDPR activées",
            },
            
            SupportedLanguage.GERMAN.value: {
                "quantum.planner.initialized": "Quantenplaner mit {max_tasks} parallelen Aufgaben initialisiert",
                "quantum.planner.task_registered": "Quantenaufgabe '{task_name}' erfolgreich registriert",
                "quantum.planner.plan_created": "Ausführungsplan mit {phases} Phasen erstellt",
                "quantum.planner.optimization_complete": "Quantenoptimierung mit Bewertung {score} abgeschlossen",
                "security.context_created": "Sicherheitskontext für Benutzer {user_id} erstellt",
                "security.context_invalid": "Ungültiger Sicherheitskontext",
                "security.access_denied": "Zugriff auf Ressource {resource} verweigert",
                "execution.started": "Quantenausführung gestartet",
                "execution.completed": "Ausführung erfolgreich abgeschlossen",
                "execution.failed": "Ausführung mit Fehler fehlgeschlagen",
                "error.configuration": "Konfigurationsfehler: {details}",
                "error.validation": "Validierungsfehler: {field} - {message}",
                "compliance.gdpr_enabled": "DSGVO-Compliance-Funktionen aktiviert",
            },
            
            SupportedLanguage.JAPANESE.value: {
                "quantum.planner.initialized": "{max_tasks}個の並列タスクで量子プランナーを初期化しました",
                "quantum.planner.task_registered": "量子タスク '{task_name}' が正常に登録されました",
                "quantum.planner.plan_created": "{phases}個のフェーズで実行プランが作成されました",
                "quantum.planner.optimization_complete": "スコア{score}で量子最適化が完了しました",
                "security.context_created": "ユーザー {user_id} のセキュリティコンテキストが作成されました",
                "security.context_invalid": "無効なセキュリティコンテキスト",
                "security.access_denied": "リソース {resource} へのアクセスが拒否されました",
                "execution.started": "量子実行が開始されました",
                "execution.completed": "実行が正常に完了しました",
                "execution.failed": "実行がエラーで失敗しました",
                "error.configuration": "設定エラー: {details}",
                "error.validation": "検証エラー: {field} - {message}",
                "compliance.gdpr_enabled": "GDPR準拠機能が有効になりました",
            },
            
            SupportedLanguage.CHINESE_SIMPLIFIED.value: {
                "quantum.planner.initialized": "量子规划器已初始化，支持{max_tasks}个并行任务",
                "quantum.planner.task_registered": "量子任务 '{task_name}' 注册成功",
                "quantum.planner.plan_created": "执行计划已创建，包含{phases}个阶段",
                "quantum.planner.optimization_complete": "量子优化完成，得分为{score}",
                "security.context_created": "已为用户 {user_id} 创建安全上下文",
                "security.context_invalid": "无效的安全上下文",
                "security.access_denied": "对资源 {resource} 的访问被拒绝",
                "execution.started": "量子执行已开始",
                "execution.completed": "执行成功完成",
                "execution.failed": "执行失败并出现错误",
                "error.configuration": "配置错误：{details}",
                "error.validation": "验证错误：{field} - {message}",
                "compliance.gdpr_enabled": "GDPR合规功能已启用",
            },
        }
        
        # Add custom translations from config
        for lang, translations in self.config.custom_translations.items():
            if lang in self._translations:
                self._translations[lang].update(translations)
            else:
                self._translations[lang] = translations
    
    def _load_compliance_rules(self):
        """Load regional compliance rules."""
        self._compliance_rules = {
            RegionalCompliance.GDPR: {
                "data_retention_days": 365,
                "require_consent": True,
                "require_audit_trail": True,
                "data_portability": True,
                "right_to_erasure": True,
                "data_minimization": True,
                "privacy_by_design": True,
                "dpo_required": True,
                "breach_notification_hours": 72,
                "allowed_data_transfers": ["EU", "EEA", "Adequacy Decision Countries"],
            },
            
            RegionalCompliance.CCPA: {
                "data_retention_days": 730,  # 2 years
                "require_consent": False,    # Opt-out model
                "require_audit_trail": True,
                "data_portability": True,
                "right_to_erasure": True,
                "data_minimization": False,
                "sale_opt_out": True,
                "consumer_rights_notice": True,
                "revenue_threshold": 25000000,  # $25M
                "personal_info_threshold": 50000,
            },
            
            RegionalCompliance.PDPA: {
                "data_retention_days": 365,
                "require_consent": True,
                "require_audit_trail": True,
                "data_portability": True,
                "right_to_erasure": True,
                "data_breach_notification": True,
                "cross_border_transfer_restrictions": True,
                "dpo_appointment": "recommended",
            },
            
            RegionalCompliance.LGPD: {
                "data_retention_days": 365,
                "require_consent": True,
                "require_audit_trail": True,
                "data_portability": True,
                "right_to_erasure": True,
                "data_minimization": True,
                "privacy_impact_assessment": True,
                "dpo_required": True,
            },
        }
    
    def translate(self, key: str, **kwargs) -> str:
        """
        Translate a message key to the configured language.
        
        Args:
            key: Message key to translate
            **kwargs: Parameters for message formatting
            
        Returns:
            Translated and formatted message
        """
        # Try primary language
        primary_lang = self.config.primary_language.value
        if primary_lang in self._translations and key in self._translations[primary_lang]:
            message = self._translations[primary_lang][key]
        # Fallback to fallback language
        elif (self.config.fallback_language.value in self._translations and 
              key in self._translations[self.config.fallback_language.value]):
            message = self._translations[self.config.fallback_language.value][key]
        # Ultimate fallback to key itself
        else:
            message = key
            logger.warning(f"Translation not found for key: {key}")
        
        # Format message with parameters
        try:
            return message.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing parameter {e} for translation key: {key}")
            return message
    
    def t(self, key: str, **kwargs) -> str:
        """Shorthand for translate method."""
        return self.translate(key, **kwargs)
    
    def get_compliance_rule(self, compliance: RegionalCompliance, rule: str) -> Any:
        """
        Get a specific compliance rule value.
        
        Args:
            compliance: Regional compliance framework
            rule: Specific rule name
            
        Returns:
            Rule value or None if not found
        """
        return self._compliance_rules.get(compliance, {}).get(rule)
    
    def is_compliance_required(self, compliance: RegionalCompliance) -> bool:
        """Check if a compliance framework is required."""
        return compliance in self.config.regional_compliance
    
    def get_data_retention_period(self) -> int:
        """
        Get the maximum data retention period based on active compliance frameworks.
        
        Returns:
            Maximum retention period in days
        """
        max_retention = 365  # Default 1 year
        
        for compliance in self.config.regional_compliance:
            retention = self.get_compliance_rule(compliance, "data_retention_days")
            if retention:
                max_retention = max(max_retention, retention)
        
        return max_retention
    
    def requires_consent(self) -> bool:
        """Check if explicit consent is required based on compliance frameworks."""
        for compliance in self.config.regional_compliance:
            if self.get_compliance_rule(compliance, "require_consent"):
                return True
        return False
    
    def requires_audit_trail(self) -> bool:
        """Check if audit trail is required."""
        for compliance in self.config.regional_compliance:
            if self.get_compliance_rule(compliance, "require_audit_trail"):
                return True
        return False
    
    def format_datetime(self, timestamp: float) -> str:
        """
        Format datetime according to regional preferences.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            Formatted datetime string
        """
        import datetime
        dt = datetime.datetime.fromtimestamp(timestamp)
        return dt.strftime(self.config.date_format)
    
    def format_number(self, number: Union[int, float]) -> str:
        """
        Format number according to regional preferences.
        
        Args:
            number: Number to format
            
        Returns:
            Formatted number string
        """
        try:
            import locale
            # This is a simplified implementation
            # In production, you'd use proper locale formatting
            if isinstance(number, float):
                return f"{number:,.2f}"
            else:
                return f"{number:,}"
        except:
            return str(number)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return [lang.value for lang in SupportedLanguage]
    
    def get_language_name(self, lang_code: str) -> str:
        """
        Get human-readable language name.
        
        Args:
            lang_code: Language code
            
        Returns:
            Human-readable language name
        """
        language_names = {
            "en": "English",
            "es": "Español",
            "fr": "Français", 
            "de": "Deutsch",
            "ja": "日本語",
            "zh-CN": "简体中文",
            "zh-TW": "繁體中文",
            "ko": "한국어",
            "pt": "Português",
            "ru": "Русский",
            "it": "Italiano",
            "nl": "Nederlands",
            "ar": "العربية",
            "hi": "हिन्दी",
        }
        return language_names.get(lang_code, lang_code)
    
    def validate_compliance_configuration(self) -> Dict[str, Any]:
        """
        Validate current compliance configuration.
        
        Returns:
            Validation results with issues and recommendations
        """
        issues = []
        recommendations = []
        
        # Check for conflicting compliance requirements
        if (RegionalCompliance.GDPR in self.config.regional_compliance and
            RegionalCompliance.CCPA in self.config.regional_compliance):
            
            gdpr_consent = self.get_compliance_rule(RegionalCompliance.GDPR, "require_consent")
            ccpa_consent = self.get_compliance_rule(RegionalCompliance.CCPA, "require_consent")
            
            if gdpr_consent != ccpa_consent:
                issues.append("GDPR requires opt-in consent while CCPA uses opt-out model")
                recommendations.append("Implement opt-in consent to satisfy both frameworks")
        
        # Check data retention alignment
        retention_periods = []
        for compliance in self.config.regional_compliance:
            retention = self.get_compliance_rule(compliance, "data_retention_days")
            if retention:
                retention_periods.append((compliance.value, retention))
        
        if len(set(period for _, period in retention_periods)) > 1:
            recommendations.append(f"Consider using maximum retention period: {max(period for _, period in retention_periods)} days")
        
        # Check language support for compliance regions
        compliance_languages = {
            RegionalCompliance.GDPR: ["en", "de", "fr", "es", "it", "nl"],
            RegionalCompliance.CCPA: ["en", "es"],
            RegionalCompliance.PDPA: ["en"],
            RegionalCompliance.LGPD: ["pt"],
        }
        
        for compliance in self.config.regional_compliance:
            required_languages = compliance_languages.get(compliance, ["en"])
            if self.config.primary_language.value not in required_languages:
                recommendations.append(f"Consider adding {compliance.value} compliance language support")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations,
            "active_frameworks": [c.value for c in self.config.regional_compliance],
            "data_retention_days": self.get_data_retention_period(),
            "requires_consent": self.requires_consent(),
            "requires_audit_trail": self.requires_audit_trail(),
        }
    
    def generate_privacy_notice(self, service_name: str) -> str:
        """
        Generate a privacy notice based on active compliance frameworks.
        
        Args:
            service_name: Name of the service
            
        Returns:
            Generated privacy notice text
        """
        notice_parts = []
        
        # Header
        notice_parts.append(self.translate("privacy.notice.header", service_name=service_name))
        
        # Data collection notice
        if self.requires_consent():
            notice_parts.append(self.translate("privacy.notice.consent_required"))
        
        # Data retention
        retention_days = self.get_data_retention_period()
        notice_parts.append(self.translate("privacy.notice.retention", days=retention_days))
        
        # User rights based on compliance frameworks
        rights = set()
        for compliance in self.config.regional_compliance:
            if self.get_compliance_rule(compliance, "data_portability"):
                rights.add("data_portability")
            if self.get_compliance_rule(compliance, "right_to_erasure"):
                rights.add("right_to_erasure")
        
        if rights:
            notice_parts.append(self.translate("privacy.notice.user_rights", rights=", ".join(rights)))
        
        # Contact information
        notice_parts.append(self.translate("privacy.notice.contact"))
        
        return "\n\n".join(notice_parts)
    
    def log_with_translation(self, level: str, key: str, **kwargs):
        """
        Log a message with translation.
        
        Args:
            level: Log level (info, warning, error, etc.)
            key: Translation key
            **kwargs: Parameters for translation
        """
        translated_message = self.translate(key, **kwargs)
        
        if hasattr(logger, level):
            getattr(logger, level)(translated_message)
        else:
            logger.info(translated_message)


# Global i18n instance
_i18n_instance: Optional[QuantumInternationalization] = None

def get_i18n() -> QuantumInternationalization:
    """Get the global i18n instance."""
    global _i18n_instance
    if _i18n_instance is None:
        _i18n_instance = QuantumInternationalization()
    return _i18n_instance

def configure_i18n(config: LocalizationConfig):
    """Configure the global i18n instance."""
    global _i18n_instance
    _i18n_instance = QuantumInternationalization(config)

def t(key: str, **kwargs) -> str:
    """Global translation function."""
    return get_i18n().translate(key, **kwargs)

# Convenience functions for common operations
def translate_error(error_type: str, **kwargs) -> str:
    """Translate an error message."""
    return t(f"error.{error_type}", **kwargs)

def translate_security(action: str, **kwargs) -> str:
    """Translate a security message."""
    return t(f"security.{action}", **kwargs)

def translate_quantum(component: str, action: str, **kwargs) -> str:
    """Translate a quantum system message."""
    return t(f"quantum.{component}.{action}", **kwargs)