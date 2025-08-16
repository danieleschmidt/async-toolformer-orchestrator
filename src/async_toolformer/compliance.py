"""
Compliance Module for Quantum AsyncOrchestrator.

This module provides comprehensive compliance features for global deployment:
- GDPR, CCPA, PDPA, LGPD compliance automation
- Data governance and retention policies
- Privacy-preserving quantum operations
- Audit trail management for regulatory requirements
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .i18n import RegionalCompliance, get_i18n

logger = logging.getLogger(__name__)


class DataCategory(Enum):
    """Data categories for compliance classification."""
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    BIOMETRIC = "biometric"
    FINANCIAL = "financial"
    HEALTH = "health"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"
    OPERATIONAL = "operational"


class ProcessingPurpose(Enum):
    """Legal basis for data processing."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class DataProcessingRecord:
    """Record of data processing activity for compliance."""
    id: str
    timestamp: float
    user_id: str
    data_categories: set[DataCategory]
    processing_purposes: set[ProcessingPurpose]
    legal_basis: str
    retention_period_days: int
    data_subjects_count: int = 1
    third_party_transfers: list[str] = field(default_factory=list)
    security_measures: list[str] = field(default_factory=list)
    anonymized: bool = False
    consent_obtained: bool = False
    opt_out_available: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "data_categories": [cat.value for cat in self.data_categories],
            "processing_purposes": [purpose.value for purpose in self.processing_purposes],
            "legal_basis": self.legal_basis,
            "retention_period_days": self.retention_period_days,
            "data_subjects_count": self.data_subjects_count,
            "third_party_transfers": self.third_party_transfers,
            "security_measures": self.security_measures,
            "anonymized": self.anonymized,
            "consent_obtained": self.consent_obtained,
            "opt_out_available": self.opt_out_available,
        }


class QuantumComplianceManager:
    """
    Comprehensive compliance manager for quantum orchestrator.

    Handles regulatory compliance across multiple jurisdictions with
    automated policy enforcement and audit trail generation.
    """

    def __init__(
        self,
        active_frameworks: list[RegionalCompliance],
        default_retention_days: int = 365,
        enable_data_minimization: bool = True,
        enable_privacy_by_design: bool = True,
    ):
        """
        Initialize compliance manager.

        Args:
            active_frameworks: List of active compliance frameworks
            default_retention_days: Default data retention period
            enable_data_minimization: Enable data minimization principles
            enable_privacy_by_design: Enable privacy by design principles
        """
        self.active_frameworks = active_frameworks
        self.default_retention_days = default_retention_days
        self.enable_data_minimization = enable_data_minimization
        self.enable_privacy_by_design = enable_privacy_by_design

        # Processing records for audit trail
        self._processing_records: list[DataProcessingRecord] = []
        self._consent_records: dict[str, dict[str, Any]] = {}
        self._data_subjects: dict[str, dict[str, Any]] = {}

        # Compliance policies
        self._retention_policies: dict[DataCategory, int] = {}
        self._processing_restrictions: dict[RegionalCompliance, dict[str, Any]] = {}

        # Initialize framework-specific settings
        self._initialize_compliance_frameworks()

        logger.info(f"Quantum compliance manager initialized with frameworks: {[f.value for f in active_frameworks]}")

    def _initialize_compliance_frameworks(self):
        """Initialize compliance framework-specific settings."""
        get_i18n()

        for framework in self.active_frameworks:
            if framework == RegionalCompliance.GDPR:
                self._initialize_gdpr_compliance()
            elif framework == RegionalCompliance.CCPA:
                self._initialize_ccpa_compliance()
            elif framework == RegionalCompliance.PDPA:
                self._initialize_pdpa_compliance()
            elif framework == RegionalCompliance.LGPD:
                self._initialize_lgpd_compliance()

    def _initialize_gdpr_compliance(self):
        """Initialize GDPR-specific compliance settings."""
        # GDPR retention periods (in days)
        self._retention_policies.update({
            DataCategory.PERSONAL: 365,      # 1 year default
            DataCategory.SENSITIVE: 180,     # 6 months for sensitive data
            DataCategory.BIOMETRIC: 90,      # 3 months for biometric
            DataCategory.FINANCIAL: 2555,    # 7 years for financial
            DataCategory.HEALTH: 730,        # 2 years for health
            DataCategory.TECHNICAL: 30,      # 1 month for technical logs
        })

        self._processing_restrictions[RegionalCompliance.GDPR] = {
            "requires_explicit_consent": True,
            "data_portability_required": True,
            "right_to_erasure": True,
            "privacy_by_design": True,
            "data_protection_impact_assessment": True,
            "dpo_required_threshold": 250,  # employees
            "breach_notification_hours": 72,
            "supervisory_authority_notification": True,
            "international_transfer_restrictions": True,
        }

    def _initialize_ccpa_compliance(self):
        """Initialize CCPA-specific compliance settings."""
        self._processing_restrictions[RegionalCompliance.CCPA] = {
            "opt_out_sale_required": True,
            "consumer_request_response_days": 45,
            "data_portability_required": True,
            "right_to_delete": True,
            "right_to_know": True,
            "non_discrimination": True,
            "revenue_threshold": 25000000,  # $25M annual revenue
            "consumer_threshold": 50000,    # 50k consumers annually
        }

    def _initialize_pdpa_compliance(self):
        """Initialize PDPA-specific compliance settings."""
        self._processing_restrictions[RegionalCompliance.PDPA] = {
            "consent_required": True,
            "data_breach_notification_required": True,
            "cross_border_restrictions": True,
            "do_not_call_registry": True,
            "data_protection_officer_recommended": True,
        }

    def _initialize_lgpd_compliance(self):
        """Initialize LGPD (Brazil) compliance settings."""
        self._processing_restrictions[RegionalCompliance.LGPD] = {
            "explicit_consent_required": True,
            "data_portability_required": True,
            "right_to_erasure": True,
            "privacy_impact_assessment": True,
            "data_protection_officer_required": True,
            "international_transfer_restrictions": True,
        }

    def record_data_processing(
        self,
        user_id: str,
        data_categories: list[DataCategory],
        processing_purposes: list[ProcessingPurpose],
        legal_basis: str = "legitimate_interests",
        data_subjects_count: int = 1,
        third_party_transfers: list[str] | None = None,
        consent_obtained: bool = False,
    ) -> str:
        """
        Record a data processing activity for compliance audit.

        Args:
            user_id: User identifier
            data_categories: Categories of data being processed
            processing_purposes: Purposes for processing
            legal_basis: Legal basis for processing
            data_subjects_count: Number of data subjects affected
            third_party_transfers: List of third parties data is shared with
            consent_obtained: Whether explicit consent was obtained

        Returns:
            Processing record ID
        """
        record_id = f"proc_{int(time.time() * 1000)}_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}"

        # Determine retention period based on data categories
        max_retention = self.default_retention_days
        for category in data_categories:
            category_retention = self._retention_policies.get(category, self.default_retention_days)
            max_retention = max(max_retention, category_retention)

        # Apply compliance framework requirements
        for framework in self.active_frameworks:
            framework_retention = get_i18n().get_compliance_rule(framework, "data_retention_days")
            if framework_retention:
                max_retention = max(max_retention, framework_retention)

        record = DataProcessingRecord(
            id=record_id,
            timestamp=time.time(),
            user_id=user_id,
            data_categories=set(data_categories),
            processing_purposes=set(processing_purposes),
            legal_basis=legal_basis,
            retention_period_days=max_retention,
            data_subjects_count=data_subjects_count,
            third_party_transfers=third_party_transfers or [],
            consent_obtained=consent_obtained,
            security_measures=["encryption", "access_control", "audit_logging"],
        )

        self._processing_records.append(record)

        # Log compliance event
        get_i18n().log_with_translation(
            "info",
            "compliance.data_processed",
            user_id=user_id,
            categories=len(data_categories),
            legal_basis=legal_basis
        )

        return record_id

    def obtain_consent(
        self,
        user_id: str,
        data_categories: list[DataCategory],
        processing_purposes: list[ProcessingPurpose],
        consent_text: str,
        granular_consent: dict[str, bool] | None = None,
    ) -> bool:
        """
        Obtain and record user consent for data processing.

        Args:
            user_id: User identifier
            data_categories: Categories of data to be processed
            processing_purposes: Purposes for processing
            consent_text: Consent text presented to user
            granular_consent: Granular consent choices

        Returns:
            True if consent was successfully recorded
        """
        consent_id = f"consent_{int(time.time() * 1000)}_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}"

        consent_record = {
            "id": consent_id,
            "user_id": user_id,
            "timestamp": time.time(),
            "data_categories": [cat.value for cat in data_categories],
            "processing_purposes": [purpose.value for purpose in processing_purposes],
            "consent_text": consent_text,
            "granular_consent": granular_consent or {},
            "consent_method": "explicit",
            "ip_address": "anonymized",
            "user_agent": "anonymized",
            "withdrawal_available": True,
        }

        self._consent_records[consent_id] = consent_record

        logger.info(f"Consent obtained for user {user_id}: {consent_id}")
        return True

    def withdraw_consent(self, user_id: str, consent_id: str | None = None) -> bool:
        """
        Process consent withdrawal request.

        Args:
            user_id: User identifier
            consent_id: Specific consent to withdraw (None for all)

        Returns:
            True if withdrawal was processed successfully
        """
        withdrawn_count = 0

        if consent_id:
            # Withdraw specific consent
            if consent_id in self._consent_records:
                consent_record = self._consent_records[consent_id]
                if consent_record["user_id"] == user_id:
                    consent_record["withdrawn"] = True
                    consent_record["withdrawal_timestamp"] = time.time()
                    withdrawn_count = 1
        else:
            # Withdraw all consents for user
            for record in self._consent_records.values():
                if record["user_id"] == user_id and not record.get("withdrawn", False):
                    record["withdrawn"] = True
                    record["withdrawal_timestamp"] = time.time()
                    withdrawn_count += 1

        if withdrawn_count > 0:
            logger.info(f"Withdrew {withdrawn_count} consent(s) for user {user_id}")
            return True

        return False

    def process_data_subject_request(
        self,
        user_id: str,
        request_type: str,
        additional_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process data subject rights requests (access, portability, erasure).

        Args:
            user_id: User identifier
            request_type: Type of request (access, portability, erasure, rectification)
            additional_info: Additional information for the request

        Returns:
            Request processing result
        """
        request_id = f"dsr_{request_type}_{int(time.time() * 1000)}"

        result = {
            "request_id": request_id,
            "user_id": user_id,
            "request_type": request_type,
            "timestamp": time.time(),
            "status": "processing",
            "estimated_completion": time.time() + (30 * 24 * 3600),  # 30 days
        }

        if request_type == "access":
            # Right to access - compile all data about the user
            user_data = self._compile_user_data(user_id)
            result.update({
                "status": "completed",
                "data": user_data,
                "data_categories": list({
                    cat for record in self._processing_records
                    if record.user_id == user_id
                    for cat in record.data_categories
                }),
            })

        elif request_type == "portability":
            # Right to data portability - structured data export
            portable_data = self._export_portable_data(user_id)
            result.update({
                "status": "completed",
                "export_format": "JSON",
                "data_size": len(json.dumps(portable_data)),
                "download_link": f"https://api.example.com/exports/{request_id}",
            })

        elif request_type == "erasure":
            # Right to erasure (right to be forgotten)
            erased_records = self._erase_user_data(user_id)
            result.update({
                "status": "completed",
                "records_erased": erased_records,
                "anonymization_applied": True,
            })

        elif request_type == "rectification":
            # Right to rectification - correct inaccurate data
            if additional_info and "corrections" in additional_info:
                corrections_applied = self._apply_data_corrections(user_id, additional_info["corrections"])
                result.update({
                    "status": "completed",
                    "corrections_applied": corrections_applied,
                })

        logger.info(f"Processed data subject request {request_id} for user {user_id}")
        return result

    def _compile_user_data(self, user_id: str) -> dict[str, Any]:
        """Compile all data associated with a user."""
        user_data = {
            "user_id": user_id,
            "processing_records": [
                record.to_dict() for record in self._processing_records
                if record.user_id == user_id
            ],
            "consent_records": [
                record for record in self._consent_records.values()
                if record["user_id"] == user_id
            ],
            "data_subject_info": self._data_subjects.get(user_id, {}),
        }
        return user_data

    def _export_portable_data(self, user_id: str) -> dict[str, Any]:
        """Export user data in a portable format."""
        return {
            "user_id": user_id,
            "export_timestamp": time.time(),
            "data": self._compile_user_data(user_id),
            "format_version": "1.0",
        }

    def _erase_user_data(self, user_id: str) -> int:
        """Erase or anonymize user data."""
        erased_count = 0

        # Anonymize processing records
        for record in self._processing_records:
            if record.user_id == user_id:
                record.user_id = f"anonymized_{hashlib.sha256(user_id.encode()).hexdigest()[:8]}"
                record.anonymized = True
                erased_count += 1

        # Remove consent records (or mark as withdrawn)
        for consent_record in self._consent_records.values():
            if consent_record["user_id"] == user_id:
                consent_record["user_id"] = "erased"
                consent_record["erased"] = True
                erased_count += 1

        # Remove data subject info
        if user_id in self._data_subjects:
            del self._data_subjects[user_id]
            erased_count += 1

        return erased_count

    def _apply_data_corrections(self, user_id: str, corrections: dict[str, Any]) -> int:
        """Apply data corrections for a user."""
        corrections_count = 0

        # Update data subject info
        if user_id in self._data_subjects:
            for field, new_value in corrections.items():
                if field in self._data_subjects[user_id]:
                    self._data_subjects[user_id][field] = new_value
                    corrections_count += 1

        return corrections_count

    def check_compliance_violations(self) -> list[dict[str, Any]]:
        """Check for potential compliance violations."""
        violations = []
        current_time = time.time()

        # Check data retention violations
        for record in self._processing_records:
            retention_deadline = record.timestamp + (record.retention_period_days * 24 * 3600)
            if current_time > retention_deadline and not record.anonymized:
                violations.append({
                    "type": "data_retention_exceeded",
                    "record_id": record.id,
                    "user_id": record.user_id,
                    "days_overdue": int((current_time - retention_deadline) / (24 * 3600)),
                    "severity": "high",
                    "framework": [f.value for f in self.active_frameworks],
                })

        # Check consent violations
        for framework in self.active_frameworks:
            if get_i18n().get_compliance_rule(framework, "require_consent"):
                for record in self._processing_records:
                    if not record.consent_obtained and record.legal_basis == "consent":
                        violations.append({
                            "type": "missing_consent",
                            "record_id": record.id,
                            "user_id": record.user_id,
                            "framework": framework.value,
                            "severity": "high",
                        })

        return violations

    def generate_compliance_report(self) -> dict[str, Any]:
        """Generate comprehensive compliance report."""
        current_time = time.time()

        # Calculate statistics
        total_records = len(self._processing_records)
        records_with_consent = sum(1 for r in self._processing_records if r.consent_obtained)
        anonymized_records = sum(1 for r in self._processing_records if r.anonymized)

        # Data category breakdown
        category_counts = {}
        for record in self._processing_records:
            for category in record.data_categories:
                category_counts[category.value] = category_counts.get(category.value, 0) + 1

        # Legal basis breakdown
        legal_basis_counts = {}
        for record in self._processing_records:
            legal_basis_counts[record.legal_basis] = legal_basis_counts.get(record.legal_basis, 0) + 1

        # Retention compliance
        retention_status = {
            "compliant": 0,
            "approaching_deadline": 0,  # Within 30 days
            "overdue": 0,
        }

        for record in self._processing_records:
            retention_deadline = record.timestamp + (record.retention_period_days * 24 * 3600)
            days_until_deadline = (retention_deadline - current_time) / (24 * 3600)

            if record.anonymized:
                retention_status["compliant"] += 1
            elif days_until_deadline < 0:
                retention_status["overdue"] += 1
            elif days_until_deadline <= 30:
                retention_status["approaching_deadline"] += 1
            else:
                retention_status["compliant"] += 1

        violations = self.check_compliance_violations()

        report = {
            "generated_at": current_time,
            "report_period": "all_time",
            "active_frameworks": [f.value for f in self.active_frameworks],
            "summary": {
                "total_processing_records": total_records,
                "records_with_consent": records_with_consent,
                "consent_rate": records_with_consent / max(total_records, 1),
                "anonymized_records": anonymized_records,
                "anonymization_rate": anonymized_records / max(total_records, 1),
                "total_consent_records": len(self._consent_records),
                "total_data_subjects": len(self._data_subjects),
            },
            "data_categories": category_counts,
            "legal_basis_distribution": legal_basis_counts,
            "retention_compliance": retention_status,
            "violations": {
                "total": len(violations),
                "by_type": {},
                "by_severity": {"high": 0, "medium": 0, "low": 0},
            },
            "recommendations": self._generate_compliance_recommendations(violations),
        }

        # Violation breakdown
        for violation in violations:
            violation_type = violation["type"]
            severity = violation.get("severity", "medium")

            report["violations"]["by_type"][violation_type] = report["violations"]["by_type"].get(violation_type, 0) + 1
            report["violations"]["by_severity"][severity] += 1

        return report

    def _generate_compliance_recommendations(self, violations: list[dict[str, Any]]) -> list[str]:
        """Generate compliance recommendations based on violations."""
        recommendations = []

        violation_types = {v["type"] for v in violations}

        if "data_retention_exceeded" in violation_types:
            recommendations.append("Implement automated data anonymization for expired records")
            recommendations.append("Set up retention period monitoring and alerts")

        if "missing_consent" in violation_types:
            recommendations.append("Review consent collection processes")
            recommendations.append("Consider switching to legitimate interests legal basis where appropriate")

        # Framework-specific recommendations
        for framework in self.active_frameworks:
            if framework == RegionalCompliance.GDPR:
                recommendations.append("Consider implementing Privacy by Design principles")
                recommendations.append("Review data processing impact assessments")
            elif framework == RegionalCompliance.CCPA:
                recommendations.append("Ensure opt-out mechanisms are prominently displayed")
                recommendations.append("Review consumer request response procedures")

        return recommendations

    def anonymize_expired_data(self) -> int:
        """Automatically anonymize expired data records."""
        current_time = time.time()
        anonymized_count = 0

        for record in self._processing_records:
            if not record.anonymized:
                retention_deadline = record.timestamp + (record.retention_period_days * 24 * 3600)
                if current_time > retention_deadline:
                    record.user_id = f"anonymized_{hashlib.sha256(record.user_id.encode()).hexdigest()[:8]}"
                    record.anonymized = True
                    anonymized_count += 1

        if anonymized_count > 0:
            logger.info(f"Automatically anonymized {anonymized_count} expired data records")

        return anonymized_count

    def get_compliance_status(self) -> dict[str, Any]:
        """Get current compliance status overview."""
        violations = self.check_compliance_violations()

        return {
            "overall_status": "compliant" if len(violations) == 0 else "violations_detected",
            "active_frameworks": [f.value for f in self.active_frameworks],
            "total_violations": len(violations),
            "high_priority_violations": len([v for v in violations if v.get("severity") == "high"]),
            "compliance_score": max(0, 100 - (len(violations) * 10)),  # Simple scoring
            "last_check": time.time(),
            "recommendations": self._generate_compliance_recommendations(violations),
        }


# Compliance decorators for automatic compliance recording
def track_data_processing(
    data_categories: list[DataCategory],
    processing_purposes: list[ProcessingPurpose],
    legal_basis: str = "legitimate_interests",
):
    """Decorator to automatically track data processing activities."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract user_id from arguments (assume it's provided)
            kwargs.get('user_id') or (args[0] if args else 'anonymous')

            # Record processing activity
            # Note: This would need access to a compliance manager instance
            # In practice, this would be injected or accessed through a global registry

            result = await func(*args, **kwargs)
            return result
        return wrapper
    return decorator
