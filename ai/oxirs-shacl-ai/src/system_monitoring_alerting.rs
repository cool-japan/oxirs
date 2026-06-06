//! Alerting engine for the system monitoring subsystem.
//!
//! Threshold evaluation, alert deduplication, escalation chains, and
//! notification routing (webhook / email / PagerDuty stubs).  All alerting
//! types live in [`crate::system_monitoring_types`]; the [`AnomalyDetector`]
//! engine lives in [`crate::system_monitoring_anomaly`].

// Re-export all alerting types so callers can import from this module.
pub use crate::system_monitoring_anomaly::{
    AnomalyDetectionConfig, AnomalyDetector, AnomalyEvent, AnomalyModel, AnomalyModelType,
    AnomalySeverity, BaselineProfile, BusinessImpact, ImpactAssessment, SeasonalPatternData,
};
pub use crate::system_monitoring_types::{
    Alert, AlertCondition, AlertHistoryEntry, AlertManager, AlertNotification, AlertRule,
    AlertSeverity, AlertType, EscalationRule, NotificationChannel, NotificationEngine,
    NotificationFormat, NotificationSender, NotificationStatus, NotificationTemplate, RateLimit,
    ResolutionMethod, SuppressionCondition, SuppressionRule,
};
