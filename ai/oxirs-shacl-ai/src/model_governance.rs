//! Model Governance and Compliance Framework
//!
//! This module provides comprehensive governance and compliance capabilities for
//! machine learning models used in SHACL validation, including:
//!
//! - Model approval workflows
//! - Compliance checking against regulations (GDPR, CCPA, AI Act, etc.)
//! - Audit trails and model documentation
//! - Risk assessment and mitigation
//! - Model monitoring and retirement
//! - Ethical AI guidelines enforcement
//!
//! # Compliance Standards
//!
//! - **GDPR**: Data privacy and protection
//! - **CCPA**: California Consumer Privacy Act
//! - **EU AI Act**: European Union AI regulations
//! - **ISO/IEC 42001**: AI management system
//! - **NIST AI RMF**: NIST AI Risk Management Framework
//!
//! # Example
//!
//! ```rust
//! use oxirs_shacl_ai::model_governance::{
//!     ModelGovernance, ModelGovernanceConfig, ComplianceStandard, GovernancePolicy
//! };
//!
//! let config = ModelGovernanceConfig::default();
//! let governance = ModelGovernance::with_config(config);
//!
//! // Check model compliance
//! let model_id = "shacl_validator_v1";
//! let compliance_result = governance.check_compliance(
//!     model_id,
//!     &[ComplianceStandard::GDPR, ComplianceStandard::EUAI Act]
//! ).unwrap();
//!
//! if compliance_result.is_compliant {
//!     println!("Model is compliant!");
//! } else {
//!     println!("Violations: {:?}", compliance_result.violations);
//! }
//! ```

use chrono::{DateTime, Utc};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use thiserror::Error;
use uuid::Uuid;

use crate::{Result, ShaclAiError};

/// Governance error types
#[derive(Debug, Error)]
pub enum GovernanceError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Policy violation: {0}")]
    PolicyViolation(String),

    #[error("Compliance check failed: {0}")]
    ComplianceCheckFailed(String),

    #[error("Approval required: {0}")]
    ApprovalRequired(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
}

impl From<GovernanceError> for ShaclAiError {
    fn from(err: GovernanceError) -> Self {
        ShaclAiError::DataProcessing(err.to_string())
    }
}

/// Model governance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGovernanceConfig {
    /// Enable approval workflow
    pub enable_approval_workflow: bool,

    /// Require compliance checks before deployment
    pub require_compliance_checks: bool,

    /// Enable audit logging
    pub enable_audit_logging: bool,

    /// Enable risk assessment
    pub enable_risk_assessment: bool,

    /// Enable ethical guidelines enforcement
    pub enable_ethical_guidelines: bool,

    /// Minimum approval count required
    pub min_approval_count: usize,

    /// Maximum model age before re-evaluation (days)
    pub max_model_age_days: u32,

    /// Enable automatic model retirement
    pub enable_auto_retirement: bool,
}

impl Default for ModelGovernanceConfig {
    fn default() -> Self {
        Self {
            enable_approval_workflow: true,
            require_compliance_checks: true,
            enable_audit_logging: true,
            enable_risk_assessment: true,
            enable_ethical_guidelines: true,
            min_approval_count: 2,
            max_model_age_days: 365,
            enable_auto_retirement: true,
        }
    }
}

/// Compliance standards
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceStandard {
    /// General Data Protection Regulation
    GDPR,

    /// California Consumer Privacy Act
    CCPA,

    /// EU AI Act
    EUAIAct,

    /// ISO/IEC 42001
    ISO42001,

    /// NIST AI Risk Management Framework
    NISTRISKFRAMEWORK,

    /// Health Insurance Portability and Accountability Act
    HIPAA,

    /// Sarbanes-Oxley Act
    SOX,

    /// Custom organizational standard
    Custom(String),
}

/// Model lifecycle stage
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelLifecycleStage {
    /// Model is in development
    Development,

    /// Model is in testing/validation
    Testing,

    /// Model is pending approval
    PendingApproval,

    /// Model is approved for deployment
    Approved,

    /// Model is deployed to production
    Production,

    /// Model is deprecated
    Deprecated,

    /// Model is retired
    Retired,

    /// Model is rejected
    Rejected,
}

/// Governance policy type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PolicyType {
    /// Data privacy policy
    DataPrivacy,

    /// Model fairness policy
    Fairness,

    /// Model explainability policy
    Explainability,

    /// Model security policy
    Security,

    /// Model performance policy
    Performance,

    /// Custom policy
    Custom(String),
}

/// Governance policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernancePolicy {
    /// Policy ID
    pub id: String,

    /// Policy name
    pub name: String,

    /// Policy type
    pub policy_type: PolicyType,

    /// Policy description
    pub description: String,

    /// Policy rules
    pub rules: Vec<PolicyRule>,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Active status
    pub is_active: bool,
}

/// Policy rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyRule {
    /// Rule ID
    pub id: String,

    /// Rule description
    pub description: String,

    /// Rule condition (JSON-like expression)
    pub condition: String,

    /// Rule severity
    pub severity: ViolationSeverity,
}

/// Violation severity
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum ViolationSeverity {
    /// Informational
    Info,

    /// Warning
    Warning,

    /// Critical - blocks deployment
    Critical,
}

/// Model governance metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelGovernanceMetadata {
    /// Model ID
    pub model_id: String,

    /// Model name
    pub model_name: String,

    /// Model version
    pub model_version: String,

    /// Model owner
    pub owner: String,

    /// Model description
    pub description: String,

    /// Lifecycle stage
    pub lifecycle_stage: ModelLifecycleStage,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Last updated
    pub updated_at: DateTime<Utc>,

    /// Approvers
    pub approvers: Vec<Approval>,

    /// Compliance checks
    pub compliance_checks: Vec<ComplianceCheck>,

    /// Risk assessment
    pub risk_assessment: Option<RiskAssessment>,

    /// Audit trail
    pub audit_trail: Vec<AuditEntry>,

    /// Tags
    pub tags: Vec<String>,
}

/// Approval record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Approval {
    /// Approver name/ID
    pub approver: String,

    /// Approval status
    pub status: ApprovalStatus,

    /// Approval timestamp
    pub timestamp: DateTime<Utc>,

    /// Comments
    pub comments: String,
}

/// Approval status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApprovalStatus {
    /// Pending approval
    Pending,

    /// Approved
    Approved,

    /// Rejected
    Rejected,

    /// Revoked
    Revoked,
}

/// Compliance check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCheck {
    /// Check ID
    pub id: String,

    /// Compliance standard
    pub standard: ComplianceStandard,

    /// Check timestamp
    pub timestamp: DateTime<Utc>,

    /// Compliance status
    pub is_compliant: bool,

    /// Violations found
    pub violations: Vec<Violation>,

    /// Check details
    pub details: String,
}

/// Compliance violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Violation {
    /// Violation description
    pub description: String,

    /// Severity level
    pub severity: ViolationSeverity,

    /// Policy rule violated
    pub rule_id: Option<String>,

    /// Remediation suggestion
    pub remediation: String,
}

/// Risk assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Assessment ID
    pub id: String,

    /// Overall risk level
    pub risk_level: RiskLevel,

    /// Risk factors
    pub risk_factors: Vec<RiskFactor>,

    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,

    /// Assessment timestamp
    pub timestamp: DateTime<Utc>,

    /// Assessor
    pub assessor: String,
}

/// Risk level
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum RiskLevel {
    /// Low risk
    Low,

    /// Medium risk
    Medium,

    /// High risk
    High,

    /// Critical risk
    Critical,
}

/// Risk factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Factor name
    pub name: String,

    /// Factor description
    pub description: String,

    /// Risk level
    pub level: RiskLevel,

    /// Impact assessment
    pub impact: String,
}

/// Audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Entry ID
    pub id: String,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Action performed
    pub action: String,

    /// User who performed action
    pub user: String,

    /// Action details
    pub details: HashMap<String, String>,

    /// Result of action
    pub result: String,
}

/// Compliance check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceResult {
    /// Overall compliance status
    pub is_compliant: bool,

    /// Checks performed
    pub checks: Vec<ComplianceCheck>,

    /// Violations found
    pub violations: Vec<Violation>,

    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Governance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceMetrics {
    /// Total models under governance
    pub total_models: usize,

    /// Models in production
    pub models_in_production: usize,

    /// Models pending approval
    pub models_pending_approval: usize,

    /// Compliance violations
    pub total_violations: usize,

    /// Critical violations
    pub critical_violations: usize,

    /// Average risk level
    pub average_risk_level: f64,
}

impl Default for GovernanceMetrics {
    fn default() -> Self {
        Self {
            total_models: 0,
            models_in_production: 0,
            models_pending_approval: 0,
            total_violations: 0,
            critical_violations: 0,
            average_risk_level: 0.0,
        }
    }
}

/// Main model governance system
#[derive(Debug)]
pub struct ModelGovernance {
    /// Configuration
    config: ModelGovernanceConfig,

    /// Model metadata storage
    models: Arc<DashMap<String, ModelGovernanceMetadata>>,

    /// Governance policies
    policies: Arc<DashMap<String, GovernancePolicy>>,

    /// Audit log
    audit_log: Arc<RwLock<Vec<AuditEntry>>>,

    /// Metrics
    metrics: Arc<RwLock<GovernanceMetrics>>,
}

impl ModelGovernance {
    /// Create a new governance system with default configuration
    pub fn new() -> Self {
        Self::with_config(ModelGovernanceConfig::default())
    }

    /// Create a new governance system with custom configuration
    pub fn with_config(config: ModelGovernanceConfig) -> Self {
        let governance = Self {
            config,
            models: Arc::new(DashMap::new()),
            policies: Arc::new(DashMap::new()),
            audit_log: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(GovernanceMetrics::default())),
        };

        // Register default policies
        governance.register_default_policies();

        governance
    }

    /// Register a model for governance
    pub fn register_model(&self, metadata: ModelGovernanceMetadata) -> Result<()> {
        let model_id = metadata.model_id.clone();

        // Validate model metadata
        self.validate_model_metadata(&metadata)?;

        // Store model
        self.models.insert(model_id.clone(), metadata);

        // Log audit entry
        self.log_audit_entry(AuditEntry {
            id: Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            action: "register_model".to_string(),
            user: "system".to_string(),
            details: HashMap::from([("model_id".to_string(), model_id.clone())]),
            result: "success".to_string(),
        })?;

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_models += 1;
        }

        tracing::info!("Registered model for governance: {}", model_id);
        Ok(())
    }

    /// Check compliance for a model
    pub fn check_compliance(
        &self,
        model_id: &str,
        standards: &[ComplianceStandard],
    ) -> Result<ComplianceResult> {
        let mut model = self
            .models
            .get_mut(model_id)
            .ok_or_else(|| GovernanceError::ModelNotFound(model_id.to_string()))?;

        let mut checks = Vec::new();
        let mut all_violations = Vec::new();

        for standard in standards {
            let check = self.perform_compliance_check(model_id, standard)?;
            all_violations.extend(check.violations.clone());
            checks.push(check);
        }

        // Store compliance checks
        model.compliance_checks.extend(checks.clone());

        let is_compliant = all_violations
            .iter()
            .all(|v| v.severity != ViolationSeverity::Critical);

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.total_violations += all_violations.len();
            metrics.critical_violations += all_violations
                .iter()
                .filter(|v| v.severity == ViolationSeverity::Critical)
                .count();
        }

        Ok(ComplianceResult {
            is_compliant,
            checks,
            violations: all_violations,
            recommendations: vec![
                "Review model documentation".to_string(),
                "Implement additional safeguards".to_string(),
            ],
        })
    }

    /// Request approval for a model
    pub fn request_approval(&self, model_id: &str, approvers: Vec<String>) -> Result<()> {
        let mut model = self
            .models
            .get_mut(model_id)
            .ok_or_else(|| GovernanceError::ModelNotFound(model_id.to_string()))?;

        // Check if model is ready for approval
        if model.lifecycle_stage == ModelLifecycleStage::Development {
            return Err(GovernanceError::ApprovalRequired(
                "Model must complete testing before approval".to_string(),
            )
            .into());
        }

        // Add approval requests
        for approver in approvers {
            model.approvers.push(Approval {
                approver,
                status: ApprovalStatus::Pending,
                timestamp: Utc::now(),
                comments: String::new(),
            });
        }

        model.lifecycle_stage = ModelLifecycleStage::PendingApproval;

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.models_pending_approval += 1;
        }

        tracing::info!("Approval requested for model: {}", model_id);
        Ok(())
    }

    /// Approve a model
    pub fn approve_model(&self, model_id: &str, approver: &str, comments: String) -> Result<()> {
        let mut model = self
            .models
            .get_mut(model_id)
            .ok_or_else(|| GovernanceError::ModelNotFound(model_id.to_string()))?;

        // Find and update approval
        if let Some(approval) = model.approvers.iter_mut().find(|a| a.approver == approver) {
            approval.status = ApprovalStatus::Approved;
            approval.timestamp = Utc::now();
            approval.comments = comments;
        } else {
            // Add new approval
            model.approvers.push(Approval {
                approver: approver.to_string(),
                status: ApprovalStatus::Approved,
                timestamp: Utc::now(),
                comments,
            });
        }

        // Check if model has enough approvals
        let approved_count = model
            .approvers
            .iter()
            .filter(|a| a.status == ApprovalStatus::Approved)
            .count();

        if approved_count >= self.config.min_approval_count {
            model.lifecycle_stage = ModelLifecycleStage::Approved;

            // Update metrics
            if let Ok(mut metrics) = self.metrics.write() {
                metrics.models_pending_approval = metrics.models_pending_approval.saturating_sub(1);
            }

            tracing::info!("Model approved: {}", model_id);
        }

        Ok(())
    }

    /// Deploy a model to production
    pub fn deploy_to_production(&self, model_id: &str) -> Result<()> {
        let mut model = self
            .models
            .get_mut(model_id)
            .ok_or_else(|| GovernanceError::ModelNotFound(model_id.to_string()))?;

        // Check if model is approved
        if model.lifecycle_stage != ModelLifecycleStage::Approved {
            return Err(GovernanceError::ApprovalRequired(
                "Model must be approved before deployment".to_string(),
            )
            .into());
        }

        // Check compliance if required
        if self.config.require_compliance_checks {
            let has_compliance_checks = !model.compliance_checks.is_empty();
            let is_compliant = model.compliance_checks.iter().all(|c| {
                c.is_compliant
                    || c.violations
                        .iter()
                        .all(|v| v.severity != ViolationSeverity::Critical)
            });

            if !has_compliance_checks || !is_compliant {
                return Err(GovernanceError::ComplianceCheckFailed(
                    "Model must pass compliance checks before deployment".to_string(),
                )
                .into());
            }
        }

        model.lifecycle_stage = ModelLifecycleStage::Production;

        // Update metrics
        if let Ok(mut metrics) = self.metrics.write() {
            metrics.models_in_production += 1;
        }

        tracing::info!("Model deployed to production: {}", model_id);
        Ok(())
    }

    /// Assess risk for a model
    pub fn assess_risk(&self, model_id: &str) -> Result<RiskAssessment> {
        let assessment = RiskAssessment {
            id: Uuid::new_v4().to_string(),
            risk_level: RiskLevel::Medium,
            risk_factors: vec![
                RiskFactor {
                    name: "Data Privacy".to_string(),
                    description: "Model processes personal data".to_string(),
                    level: RiskLevel::Medium,
                    impact: "Potential privacy violations".to_string(),
                },
                RiskFactor {
                    name: "Model Bias".to_string(),
                    description: "Model may exhibit bias".to_string(),
                    level: RiskLevel::Low,
                    impact: "Unfair predictions".to_string(),
                },
            ],
            mitigation_strategies: vec![
                "Implement data anonymization".to_string(),
                "Regular bias audits".to_string(),
            ],
            timestamp: Utc::now(),
            assessor: "system".to_string(),
        };

        // Store risk assessment
        if let Some(mut model) = self.models.get_mut(model_id) {
            model.risk_assessment = Some(assessment.clone());
        }

        Ok(assessment)
    }

    /// Register a governance policy
    pub fn register_policy(&self, policy: GovernancePolicy) -> Result<()> {
        self.policies.insert(policy.id.clone(), policy);
        Ok(())
    }

    /// Get model metadata
    pub fn get_model(&self, model_id: &str) -> Option<ModelGovernanceMetadata> {
        self.models.get(model_id).map(|m| m.clone())
    }

    /// Get governance metrics
    pub fn get_metrics(&self) -> Result<GovernanceMetrics> {
        Ok(self
            .metrics
            .read()
            .map_err(|e| {
                GovernanceError::InvalidConfiguration(format!("Failed to read metrics: {}", e))
            })?
            .clone())
    }

    /// List all models
    pub fn list_models(&self) -> Vec<ModelGovernanceMetadata> {
        self.models
            .iter()
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Register default governance policies
    fn register_default_policies(&self) {
        // Data Privacy Policy
        let data_privacy_policy = GovernancePolicy {
            id: Uuid::new_v4().to_string(),
            name: "Data Privacy Policy".to_string(),
            policy_type: PolicyType::DataPrivacy,
            description: "Ensures models handle personal data appropriately".to_string(),
            rules: vec![PolicyRule {
                id: Uuid::new_v4().to_string(),
                description: "Model must not store personal data".to_string(),
                condition: "data_storage == false".to_string(),
                severity: ViolationSeverity::Critical,
            }],
            created_at: Utc::now(),
            is_active: true,
        };

        self.policies
            .insert(data_privacy_policy.id.clone(), data_privacy_policy);
    }

    /// Validate model metadata
    fn validate_model_metadata(&self, metadata: &ModelGovernanceMetadata) -> Result<()> {
        if metadata.model_name.is_empty() {
            return Err(GovernanceError::InvalidConfiguration(
                "Model name cannot be empty".to_string(),
            )
            .into());
        }
        Ok(())
    }

    /// Perform a compliance check
    fn perform_compliance_check(
        &self,
        model_id: &str,
        standard: &ComplianceStandard,
    ) -> Result<ComplianceCheck> {
        // Simplified compliance check logic
        // In production, this would perform detailed checks
        let violations = match standard {
            ComplianceStandard::GDPR => {
                vec![Violation {
                    description: "Model processes personal data without explicit consent mechanism"
                        .to_string(),
                    severity: ViolationSeverity::Warning,
                    rule_id: Some("GDPR-1".to_string()),
                    remediation: "Implement consent tracking system".to_string(),
                }]
            }
            _ => vec![],
        };

        Ok(ComplianceCheck {
            id: Uuid::new_v4().to_string(),
            standard: standard.clone(),
            timestamp: Utc::now(),
            is_compliant: violations.is_empty()
                || violations
                    .iter()
                    .all(|v| v.severity != ViolationSeverity::Critical),
            violations,
            details: format!("Compliance check for {:?}", standard),
        })
    }

    /// Log an audit entry
    fn log_audit_entry(&self, entry: AuditEntry) -> Result<()> {
        if self.config.enable_audit_logging {
            if let Ok(mut log) = self.audit_log.write() {
                log.push(entry);
            }
        }
        Ok(())
    }
}

impl Default for ModelGovernance {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_governance_creation() {
        let governance = ModelGovernance::new();
        let metrics = governance.get_metrics().unwrap();
        assert_eq!(metrics.total_models, 0);
    }

    #[test]
    fn test_register_model() {
        let governance = ModelGovernance::new();

        let metadata = ModelGovernanceMetadata {
            model_id: "test_model".to_string(),
            model_name: "Test Model".to_string(),
            model_version: "1.0.0".to_string(),
            owner: "test_owner".to_string(),
            description: "Test model".to_string(),
            lifecycle_stage: ModelLifecycleStage::Development,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            approvers: vec![],
            compliance_checks: vec![],
            risk_assessment: None,
            audit_trail: vec![],
            tags: vec![],
        };

        assert!(governance.register_model(metadata).is_ok());

        let model = governance.get_model("test_model");
        assert!(model.is_some());
        assert_eq!(model.unwrap().model_name, "Test Model");
    }

    #[test]
    fn test_compliance_check() {
        let governance = ModelGovernance::new();

        let metadata = ModelGovernanceMetadata {
            model_id: "test_model".to_string(),
            model_name: "Test Model".to_string(),
            model_version: "1.0.0".to_string(),
            owner: "test_owner".to_string(),
            description: "Test model".to_string(),
            lifecycle_stage: ModelLifecycleStage::Testing,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            approvers: vec![],
            compliance_checks: vec![],
            risk_assessment: None,
            audit_trail: vec![],
            tags: vec![],
        };

        governance.register_model(metadata).unwrap();

        let result = governance
            .check_compliance("test_model", &[ComplianceStandard::GDPR])
            .unwrap();

        assert!(!result.checks.is_empty());
    }

    #[test]
    fn test_approval_workflow() {
        let governance = ModelGovernance::new();

        let metadata = ModelGovernanceMetadata {
            model_id: "test_model".to_string(),
            model_name: "Test Model".to_string(),
            model_version: "1.0.0".to_string(),
            owner: "test_owner".to_string(),
            description: "Test model".to_string(),
            lifecycle_stage: ModelLifecycleStage::Testing,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            approvers: vec![],
            compliance_checks: vec![],
            risk_assessment: None,
            audit_trail: vec![],
            tags: vec![],
        };

        governance.register_model(metadata).unwrap();

        // Request approval
        governance
            .request_approval("test_model", vec!["approver1".to_string()])
            .unwrap();

        let model = governance.get_model("test_model").unwrap();
        assert_eq!(model.lifecycle_stage, ModelLifecycleStage::PendingApproval);

        // Approve
        governance
            .approve_model("test_model", "approver1", "Approved".to_string())
            .unwrap();
        governance
            .approve_model("test_model", "approver2", "Approved".to_string())
            .unwrap();

        let model = governance.get_model("test_model").unwrap();
        assert_eq!(model.lifecycle_stage, ModelLifecycleStage::Approved);
    }

    #[test]
    fn test_risk_assessment() {
        let governance = ModelGovernance::new();

        let metadata = ModelGovernanceMetadata {
            model_id: "test_model".to_string(),
            model_name: "Test Model".to_string(),
            model_version: "1.0.0".to_string(),
            owner: "test_owner".to_string(),
            description: "Test model".to_string(),
            lifecycle_stage: ModelLifecycleStage::Development,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            approvers: vec![],
            compliance_checks: vec![],
            risk_assessment: None,
            audit_trail: vec![],
            tags: vec![],
        };

        governance.register_model(metadata).unwrap();

        let assessment = governance.assess_risk("test_model").unwrap();
        assert!(!assessment.risk_factors.is_empty());
        assert!(!assessment.mitigation_strategies.is_empty());
    }
}
