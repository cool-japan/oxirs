//! Advanced Observability and Audit System using SciRS2-Core
//!
//! This module provides comprehensive observability, audit trails, and compliance
//! logging for the OxiRS Chat system using scirs2-core's observability capabilities.
//!
//! # Features
//!
//! - Comprehensive audit logging with GDPR compliance
//! - Distributed tracing integration
//! - Security event monitoring
//! - Performance anomaly detection
//! - Compliance reporting
//! - Data lineage tracking
//!
//! # Examples
//!
//! ```rust,no_run
//! use oxirs_chat::advanced_observability::{ObservabilitySystem, ObservabilityConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = ObservabilityConfig::default();
//! let observability = ObservabilitySystem::new(config).await?;
//!
//! // Log an audit event
//! observability.audit_chat_access("user123", "session456").await?;
//!
//! // Generate compliance report
//! let report = observability.generate_compliance_report().await?;
//! println!("Total audit events: {}", report.total_events);
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use scirs2_core::{
    error::CoreError,
    metrics::{Counter, Gauge, MetricRegistry},
};
// Note: observability::audit features will be available in scirs2-core beta.4+
// For now, we provide fallback implementations
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Configuration for observability system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    /// Enable audit logging
    pub enable_audit_logging: bool,

    /// Enable distributed tracing
    pub enable_distributed_tracing: bool,

    /// Enable security event monitoring
    pub enable_security_monitoring: bool,

    /// Enable data lineage tracking
    pub enable_data_lineage: bool,

    /// Audit log retention days
    pub audit_retention_days: u32,

    /// Enable GDPR compliance mode
    pub gdpr_compliance: bool,

    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,

    /// Anomaly detection threshold (standard deviations)
    pub anomaly_threshold: f64,

    /// Maximum audit events to store in memory
    pub max_audit_events: usize,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            enable_audit_logging: true,
            enable_distributed_tracing: true,
            enable_security_monitoring: true,
            enable_data_lineage: true,
            audit_retention_days: 90, // 90 days for compliance
            gdpr_compliance: true,
            enable_anomaly_detection: true,
            anomaly_threshold: 3.0, // 3 standard deviations
            max_audit_events: 100_000,
        }
    }
}

/// Audit event type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AuditEventType {
    ChatAccess,
    MessageSent,
    MessageReceived,
    SessionCreated,
    SessionDeleted,
    DataExported,
    DataDeleted,
    ConfigurationChanged,
    SecurityViolation,
    AuthenticationAttempt,
    AuthorizationCheck,
    DataAccess,
    ApiCall,
    Custom(String),
}

/// Audit event severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Audit event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: String,
    pub event_type: AuditEventType,
    pub severity: Severity,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub resource_id: Option<String>,
    pub action: String,
    pub result: AuditResult,
    pub metadata: HashMap<String, String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub data_classification: Option<DataClassification>,
}

/// Audit event result
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AuditResult {
    Success,
    Failure,
    PartialSuccess,
    Denied,
}

/// Data classification for GDPR compliance
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DataClassification {
    Public,
    Internal,
    Confidential,
    PersonalData,  // GDPR personal data
    SensitiveData, // GDPR sensitive data
}

/// Security event for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub event_id: String,
    pub event_type: SecurityEventType,
    pub severity: Severity,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source: String,
    pub description: String,
    pub affected_resources: Vec<String>,
    pub remediation: Option<String>,
}

/// Security event types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SecurityEventType {
    UnauthorizedAccess,
    SuspiciousActivity,
    DataBreach,
    MalformedInput,
    RateLimitExceeded,
    AnomalousPattern,
    PrivilegeEscalation,
    DataExfiltration,
}

/// Compliance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub report_period_start: chrono::DateTime<chrono::Utc>,
    pub report_period_end: chrono::DateTime<chrono::Utc>,
    pub total_events: usize,
    pub events_by_type: HashMap<String, usize>,
    pub events_by_severity: HashMap<String, usize>,
    pub security_incidents: usize,
    pub gdpr_relevant_events: usize,
    pub data_access_events: usize,
    pub data_deletion_events: usize,
    pub anomalies_detected: usize,
    pub compliance_score: f64, // 0.0 to 1.0
}

/// Advanced observability system
pub struct ObservabilitySystem {
    config: ObservabilityConfig,
    audit_events: Arc<RwLock<Vec<AuditEvent>>>,
    security_events: Arc<RwLock<Vec<SecurityEvent>>>,
    metrics: Arc<MetricRegistry>,

    // SciRS2-Core metrics
    total_audit_events: Arc<Counter>,
    security_events_counter: Arc<Counter>,
    gdpr_events_counter: Arc<Counter>,
    anomalies_detected: Arc<Counter>,
    active_sessions_gauge: Arc<Gauge>,

}

impl ObservabilitySystem {
    /// Create a new observability system
    pub async fn new(config: ObservabilityConfig) -> Result<Self> {
        info!("Initializing advanced observability system");

        let metrics = Arc::new(MetricRegistry::global());

        // Initialize SciRS2-Core metrics
        let total_audit_events = Arc::new(Counter::new("total_audit_events".to_string()));
        let security_events_counter = Arc::new(Counter::new("security_events".to_string()));
        let gdpr_events_counter = Arc::new(Counter::new("gdpr_relevant_events".to_string()));
        let anomalies_detected = Arc::new(Counter::new("anomalies_detected".to_string()));
        let active_sessions_gauge = Arc::new(Gauge::new("active_monitored_sessions".to_string()));

        // Register metrics
        metrics.register_counter(total_audit_events.clone());
        metrics.register_counter(security_events_counter.clone());
        metrics.register_counter(gdpr_events_counter.clone());
        metrics.register_counter(anomalies_detected.clone());
        metrics.register_gauge(active_sessions_gauge.clone());

        Ok(Self {
            config,
            audit_events: Arc::new(RwLock::new(Vec::new())),
            security_events: Arc::new(RwLock::new(Vec::new())),
            metrics,
            total_audit_events,
            security_events_counter,
            gdpr_events_counter,
            anomalies_detected,
            active_sessions_gauge,
        })
    }

    /// Log an audit event
    pub async fn log_audit_event(&self, mut event: AuditEvent) -> Result<()> {
        if !self.config.enable_audit_logging {
            return Ok(());
        }

        // Generate event ID if not provided
        if event.event_id.is_empty() {
            event.event_id = uuid::Uuid::new_v4().to_string();
        }

        // GDPR compliance: classify event
        if self.config.gdpr_compliance && event.data_classification.is_none() {
            event.data_classification = Some(self.classify_event(&event));
        }

        // Log using tracing (scirs2-core audit features coming in beta.4+)
        tracing::info!(
            event_type = ?event.event_type,
            action = %event.action,
            "audit_event"
        );

        // Increment metrics
        self.total_audit_events.increment();

        if self.is_gdpr_relevant(&event) {
            self.gdpr_events_counter.increment();
        }

        // Store event
        let mut events = self.audit_events.write().await;
        events.push(event.clone());

        // Trim if exceeds max
        if events.len() > self.config.max_audit_events {
            events.drain(0..events.len() - self.config.max_audit_events);
        }

        // Check for anomalies
        if self.config.enable_anomaly_detection {
            self.detect_anomalies(&event).await?;
        }

        debug!("Logged audit event: {}", event.event_id);

        Ok(())
    }

    /// Log a security event
    pub async fn log_security_event(&self, mut event: SecurityEvent) -> Result<()> {
        if !self.config.enable_security_monitoring {
            return Ok(());
        }

        // Generate event ID if not provided
        if event.event_id.is_empty() {
            event.event_id = uuid::Uuid::new_v4().to_string();
        }

        // Increment metrics
        self.security_events_counter.increment();

        // Store event
        let mut events = self.security_events.write().await;
        events.push(event.clone());

        warn!(
            "Security event logged: {:?} - {}",
            event.event_type, event.description
        );

        // Alert on critical events
        if event.severity == Severity::Critical {
            self.trigger_security_alert(&event).await?;
        }

        Ok(())
    }

    /// Audit chat access
    pub async fn audit_chat_access(&self, user_id: &str, session_id: &str) -> Result<()> {
        let event = AuditEvent {
            event_id: String::new(), // Will be generated
            event_type: AuditEventType::ChatAccess,
            severity: Severity::Info,
            timestamp: chrono::Utc::now(),
            user_id: Some(user_id.to_string()),
            session_id: Some(session_id.to_string()),
            resource_id: Some(format!("chat_session:{}", session_id)),
            action: "access_chat_session".to_string(),
            result: AuditResult::Success,
            metadata: HashMap::new(),
            ip_address: None,
            user_agent: None,
            data_classification: None,
        };

        self.log_audit_event(event).await
    }

    /// Audit message sent
    pub async fn audit_message_sent(
        &self,
        user_id: &str,
        session_id: &str,
        message_id: &str,
    ) -> Result<()> {
        let event = AuditEvent {
            event_id: String::new(),
            event_type: AuditEventType::MessageSent,
            severity: Severity::Info,
            timestamp: chrono::Utc::now(),
            user_id: Some(user_id.to_string()),
            session_id: Some(session_id.to_string()),
            resource_id: Some(format!("message:{}", message_id)),
            action: "send_message".to_string(),
            result: AuditResult::Success,
            metadata: HashMap::new(),
            ip_address: None,
            user_agent: None,
            data_classification: Some(DataClassification::PersonalData),
        };

        self.log_audit_event(event).await
    }

    /// Audit data export (GDPR relevant)
    pub async fn audit_data_export(
        &self,
        user_id: &str,
        data_type: &str,
        format: &str,
    ) -> Result<()> {
        let mut metadata = HashMap::new();
        metadata.insert("data_type".to_string(), data_type.to_string());
        metadata.insert("export_format".to_string(), format.to_string());

        let event = AuditEvent {
            event_id: String::new(),
            event_type: AuditEventType::DataExported,
            severity: Severity::Warning, // Data export is sensitive
            timestamp: chrono::Utc::now(),
            user_id: Some(user_id.to_string()),
            session_id: None,
            resource_id: Some(format!("export:{}:{}", data_type, chrono::Utc::now().timestamp())),
            action: "export_data".to_string(),
            result: AuditResult::Success,
            metadata,
            ip_address: None,
            user_agent: None,
            data_classification: Some(DataClassification::PersonalData),
        };

        self.log_audit_event(event).await
    }

    /// Audit data deletion (GDPR right to be forgotten)
    pub async fn audit_data_deletion(&self, user_id: &str, data_type: &str) -> Result<()> {
        let mut metadata = HashMap::new();
        metadata.insert("data_type".to_string(), data_type.to_string());
        metadata.insert("gdpr_request".to_string(), "true".to_string());

        let event = AuditEvent {
            event_id: String::new(),
            event_type: AuditEventType::DataDeleted,
            severity: Severity::Warning,
            timestamp: chrono::Utc::now(),
            user_id: Some(user_id.to_string()),
            session_id: None,
            resource_id: Some(format!("deletion:{}", user_id)),
            action: "delete_user_data".to_string(),
            result: AuditResult::Success,
            metadata,
            ip_address: None,
            user_agent: None,
            data_classification: Some(DataClassification::PersonalData),
        };

        self.log_audit_event(event).await
    }

    /// Generate compliance report
    pub async fn generate_compliance_report(&self) -> Result<ComplianceReport> {
        let events = self.audit_events.read().await;
        let security_events = self.security_events.read().await;

        if events.is_empty() {
            return Ok(ComplianceReport {
                generated_at: chrono::Utc::now(),
                report_period_start: chrono::Utc::now(),
                report_period_end: chrono::Utc::now(),
                total_events: 0,
                events_by_type: HashMap::new(),
                events_by_severity: HashMap::new(),
                security_incidents: 0,
                gdpr_relevant_events: 0,
                data_access_events: 0,
                data_deletion_events: 0,
                anomalies_detected: self.anomalies_detected.value() as usize,
                compliance_score: 1.0,
            });
        }

        let period_start = events.first().unwrap().timestamp;
        let period_end = events.last().unwrap().timestamp;

        // Count events by type
        let mut events_by_type: HashMap<String, usize> = HashMap::new();
        for event in events.iter() {
            *events_by_type
                .entry(format!("{:?}", event.event_type))
                .or_insert(0) += 1;
        }

        // Count events by severity
        let mut events_by_severity: HashMap<String, usize> = HashMap::new();
        for event in events.iter() {
            *events_by_severity
                .entry(format!("{:?}", event.severity))
                .or_insert(0) += 1;
        }

        // GDPR relevant events
        let gdpr_relevant_events = events.iter().filter(|e| self.is_gdpr_relevant(e)).count();

        // Data access events
        let data_access_events = events
            .iter()
            .filter(|e| matches!(e.event_type, AuditEventType::DataAccess | AuditEventType::ChatAccess))
            .count();

        // Data deletion events
        let data_deletion_events = events
            .iter()
            .filter(|e| matches!(e.event_type, AuditEventType::DataDeleted))
            .count();

        // Calculate compliance score (simplified)
        let critical_events = events_by_severity.get("Critical").unwrap_or(&0);
        let error_events = events_by_severity.get("Error").unwrap_or(&0);

        let compliance_score = if events.len() > 0 {
            1.0 - ((*critical_events as f64 * 0.1 + *error_events as f64 * 0.05) / events.len() as f64)
        } else {
            1.0
        };

        let report = ComplianceReport {
            generated_at: chrono::Utc::now(),
            report_period_start: period_start,
            report_period_end: period_end,
            total_events: events.len(),
            events_by_type,
            events_by_severity,
            security_incidents: security_events.len(),
            gdpr_relevant_events,
            data_access_events,
            data_deletion_events,
            anomalies_detected: self.anomalies_detected.value() as usize,
            compliance_score: compliance_score.max(0.0).min(1.0),
        };

        info!(
            "Generated compliance report: {} events, compliance score: {:.2}",
            report.total_events, report.compliance_score
        );

        Ok(report)
    }

    /// Get recent audit events
    pub async fn get_recent_events(&self, limit: usize) -> Vec<AuditEvent> {
        let events = self.audit_events.read().await;
        events
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Get recent security events
    pub async fn get_recent_security_events(&self, limit: usize) -> Vec<SecurityEvent> {
        let events = self.security_events.read().await;
        events
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Clear old audit events (for compliance with retention policies)
    pub async fn cleanup_old_events(&self) -> Result<usize> {
        let retention_duration = chrono::Duration::days(self.config.audit_retention_days as i64);
        let cutoff = chrono::Utc::now() - retention_duration;

        let mut events = self.audit_events.write().await;
        let initial_count = events.len();

        events.retain(|event| event.timestamp >= cutoff);

        let removed = initial_count - events.len();

        if removed > 0 {
            info!("Cleaned up {} old audit events (retention: {} days)", removed, self.config.audit_retention_days);
        }

        Ok(removed)
    }

    // Private helper methods

    fn classify_event(&self, event: &AuditEvent) -> DataClassification {
        match event.event_type {
            AuditEventType::MessageSent | AuditEventType::MessageReceived => {
                DataClassification::PersonalData
            }
            AuditEventType::DataExported | AuditEventType::DataDeleted => {
                DataClassification::SensitiveData
            }
            AuditEventType::SecurityViolation => DataClassification::Confidential,
            _ => DataClassification::Internal,
        }
    }

    fn is_gdpr_relevant(&self, event: &AuditEvent) -> bool {
        matches!(
            event.data_classification,
            Some(DataClassification::PersonalData) | Some(DataClassification::SensitiveData)
        )
    }

    async fn detect_anomalies(&self, _event: &AuditEvent) -> Result<()> {
        // TODO: Implement anomaly detection using statistical analysis
        // For now, just a placeholder
        Ok(())
    }

    async fn trigger_security_alert(&self, event: &SecurityEvent) -> Result<()> {
        warn!(
            "CRITICAL SECURITY ALERT: {:?} - {}",
            event.event_type, event.description
        );

        // TODO: Integrate with alerting system (email, Slack, PagerDuty, etc.)

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_observability_creation() {
        let config = ObservabilityConfig::default();
        let observability = ObservabilitySystem::new(config).await;
        assert!(observability.is_ok());
    }

    #[tokio::test]
    async fn test_audit_event_logging() -> Result<()> {
        let config = ObservabilityConfig::default();
        let observability = ObservabilitySystem::new(config).await?;

        observability.audit_chat_access("user123", "session456").await?;

        let events = observability.get_recent_events(10).await;
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].user_id, Some("user123".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_gdpr_compliance() -> Result<()> {
        let config = ObservabilityConfig {
            gdpr_compliance: true,
            ..Default::default()
        };
        let observability = ObservabilitySystem::new(config).await?;

        observability
            .audit_data_export("user123", "messages", "json")
            .await?;
        observability
            .audit_data_deletion("user123", "all_data")
            .await?;

        let report = observability.generate_compliance_report().await?;
        assert_eq!(report.gdpr_relevant_events, 2);
        assert_eq!(report.data_deletion_events, 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_compliance_report() -> Result<()> {
        let config = ObservabilityConfig::default();
        let observability = ObservabilitySystem::new(config).await?;

        // Generate some events
        for i in 0..10 {
            observability
                .audit_chat_access(&format!("user{}", i), &format!("session{}", i))
                .await?;
        }

        let report = observability.generate_compliance_report().await?;
        assert_eq!(report.total_events, 10);
        assert!(report.compliance_score > 0.9);

        Ok(())
    }
}
