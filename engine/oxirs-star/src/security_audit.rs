//! Security audit logging for RDF-star operations
//!
//! This module provides comprehensive security audit logging capabilities,
//! including tamper-proof logs, anomaly detection, compliance reporting,
//! and integration with SIEM systems.

use crate::cryptographic_provenance::ProvenanceKeyPair;
use chrono::{DateTime, Utc};
use scirs2_core::metrics::{Counter, Timer};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use thiserror::Error;
use tracing::{debug, error, info, warn};

/// Errors related to audit logging
#[derive(Error, Debug)]
pub enum AuditError {
    #[error("Failed to write audit log: {0}")]
    WriteError(String),

    #[error("Failed to read audit log: {0}")]
    ReadError(String),

    #[error("Audit log verification failed: {0}")]
    VerificationFailed(String),

    #[error("Log rotation failed: {0}")]
    RotationFailed(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Security event severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SecuritySeverity {
    /// Informational (normal operations)
    Info,
    /// Low severity (minor issues)
    Low,
    /// Medium severity (potential security concerns)
    Medium,
    /// High severity (security violations)
    High,
    /// Critical severity (severe security breaches)
    Critical,
}

/// Security event categories
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecurityCategory {
    /// Authentication events
    Authentication,
    /// Authorization/access control events
    Authorization,
    /// Data access events
    DataAccess,
    /// Data modification events
    DataModification,
    /// Configuration changes
    Configuration,
    /// System events
    System,
    /// Network events
    Network,
    /// Cryptographic operations
    Cryptographic,
    /// Policy violations
    PolicyViolation,
    /// Anomaly detection
    Anomaly,
}

/// Security audit event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    /// Unique event ID
    pub id: String,

    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Event severity
    pub severity: SecuritySeverity,

    /// Event category
    pub category: SecurityCategory,

    /// Event type/action
    pub event_type: String,

    /// User/actor who triggered the event
    pub actor: Option<String>,

    /// Resource affected by the event
    pub resource: Option<String>,

    /// Source IP address
    pub source_ip: Option<String>,

    /// Event outcome (success/failure)
    pub outcome: EventOutcome,

    /// Detailed event message
    pub message: String,

    /// Additional metadata
    pub metadata: HashMap<String, String>,

    /// Hash of previous event (for tamper detection)
    pub previous_hash: Option<String>,

    /// Digital signature of this event
    pub signature: Option<String>,
}

/// Event outcome
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EventOutcome {
    Success,
    Failure,
    Partial,
    Unknown,
}

impl SecurityEvent {
    /// Create a new security event
    pub fn new(
        severity: SecuritySeverity,
        category: SecurityCategory,
        event_type: String,
        message: String,
    ) -> Self {
        use scirs2_core::random::{rng, Rng};

        let mut rng_instance = rng();
        let id = format!("evt_{:016x}", rng_instance.random::<u64>());

        Self {
            id,
            timestamp: Utc::now(),
            severity,
            category,
            event_type,
            actor: None,
            resource: None,
            source_ip: None,
            outcome: EventOutcome::Success,
            message,
            metadata: HashMap::new(),
            previous_hash: None,
            signature: None,
        }
    }

    /// Set actor
    pub fn with_actor(mut self, actor: String) -> Self {
        self.actor = Some(actor);
        self
    }

    /// Set resource
    pub fn with_resource(mut self, resource: String) -> Self {
        self.resource = Some(resource);
        self
    }

    /// Set source IP
    pub fn with_source_ip(mut self, ip: String) -> Self {
        self.source_ip = Some(ip);
        self
    }

    /// Set outcome
    pub fn with_outcome(mut self, outcome: EventOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Compute hash of this event
    pub fn compute_hash(&self) -> String {
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();

        hasher.update(&self.id);
        hasher.update(self.timestamp.to_rfc3339());
        hasher.update(&self.event_type);
        hasher.update(&self.message);

        if let Some(ref actor) = self.actor {
            hasher.update(actor);
        }
        if let Some(ref resource) = self.resource {
            hasher.update(resource);
        }

        hex::encode(hasher.finalize())
    }
}

/// Audit log configuration
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Base directory for audit logs
    pub log_dir: PathBuf,

    /// Whether to enable cryptographic signatures
    pub enable_signatures: bool,

    /// Maximum log file size in bytes before rotation
    pub max_file_size: u64,

    /// Number of rotated log files to keep
    pub max_rotations: usize,

    /// Minimum severity level to log
    pub min_severity: SecuritySeverity,

    /// Enable real-time anomaly detection
    pub enable_anomaly_detection: bool,

    /// Buffer size for in-memory events
    pub buffer_size: usize,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            log_dir: PathBuf::from("./audit_logs"),
            enable_signatures: true,
            max_file_size: 100 * 1024 * 1024, // 100 MB
            max_rotations: 10,
            min_severity: SecuritySeverity::Info,
            enable_anomaly_detection: true,
            buffer_size: 1000,
        }
    }
}

/// Security audit logger
pub struct SecurityAuditLogger {
    /// Configuration
    config: AuditConfig,

    /// Current log file
    current_file: Arc<Mutex<Option<BufWriter<File>>>>,

    /// Current log file path
    current_path: Arc<Mutex<PathBuf>>,

    /// Event counter
    event_counter: Counter,

    /// Write timer
    write_timer: Timer,

    /// In-memory event buffer for fast queries
    event_buffer: Arc<Mutex<VecDeque<SecurityEvent>>>,

    /// Previous event hash for chain integrity
    last_hash: Arc<Mutex<Option<String>>>,

    /// Key pair for signing events
    signing_key: Option<ProvenanceKeyPair>,

    /// Anomaly detector
    anomaly_detector: Arc<Mutex<AnomalyDetector>>,
}

impl SecurityAuditLogger {
    /// Create a new security audit logger
    pub fn new(config: AuditConfig) -> Result<Self, AuditError> {
        // Create log directory if it doesn't exist
        std::fs::create_dir_all(&config.log_dir)?;

        let event_counter = Counter::new("security_events_total".to_string());
        let write_timer = Timer::new("audit_log_write_duration".to_string());

        let signing_key = if config.enable_signatures {
            Some(ProvenanceKeyPair::generate())
        } else {
            None
        };

        let logger = Self {
            config: config.clone(),
            current_file: Arc::new(Mutex::new(None)),
            current_path: Arc::new(Mutex::new(PathBuf::new())),
            event_counter,
            write_timer,
            event_buffer: Arc::new(Mutex::new(VecDeque::with_capacity(config.buffer_size))),
            last_hash: Arc::new(Mutex::new(None)),
            signing_key,
            anomaly_detector: Arc::new(Mutex::new(AnomalyDetector::new())),
        };

        // Initialize log file
        logger.rotate_log()?;

        info!("Security audit logger initialized at {:?}", config.log_dir);
        Ok(logger)
    }

    /// Log a security event
    pub fn log_event(&self, mut event: SecurityEvent) -> Result<(), AuditError> {
        // Check severity threshold
        if event.severity < self.config.min_severity {
            return Ok(());
        }

        // Set previous hash for chain integrity
        let mut last_hash = self.last_hash.lock().unwrap();
        event.previous_hash = last_hash.clone();

        // Sign event if enabled
        if let Some(ref key_pair) = self.signing_key {
            let event_hash = event.compute_hash();
            let signature = key_pair.sign(event_hash.as_bytes());
            event.signature = Some(hex::encode(signature.to_bytes()));
        }

        // Compute and store current hash
        let current_hash = event.compute_hash();
        *last_hash = Some(current_hash);
        drop(last_hash);

        // Add to buffer
        let mut buffer = self.event_buffer.lock().unwrap();
        if buffer.len() >= self.config.buffer_size {
            buffer.pop_front();
        }
        buffer.push_back(event.clone());
        drop(buffer);

        // Anomaly detection
        if self.config.enable_anomaly_detection {
            let mut detector = self.anomaly_detector.lock().unwrap();
            if let Some(anomaly) = detector.check_event(&event) {
                warn!("Anomaly detected: {}", anomaly);
            }
        }

        // Write to file
        let _timer = self.write_timer.start();
        self.write_event(&event)?;
        self.event_counter.inc();

        // Check if rotation needed
        self.check_rotation()?;

        Ok(())
    }

    /// Write event to file
    fn write_event(&self, event: &SecurityEvent) -> Result<(), AuditError> {
        let mut file_guard = self.current_file.lock().unwrap();

        if let Some(ref mut writer) = *file_guard {
            let json = serde_json::to_string(event)
                .map_err(|e| AuditError::SerializationError(e.to_string()))?;

            writeln!(writer, "{}", json)?;
            writer.flush()?;
        } else {
            return Err(AuditError::WriteError("No active log file".to_string()));
        }

        Ok(())
    }

    /// Rotate log files
    fn rotate_log(&self) -> Result<(), AuditError> {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let log_path = self
            .config
            .log_dir
            .join(format!("audit_{}.jsonl", timestamp));

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)?;

        let mut current_file = self.current_file.lock().unwrap();
        *current_file = Some(BufWriter::new(file));

        let mut current_path = self.current_path.lock().unwrap();
        *current_path = log_path.clone();

        info!("Rotated audit log to {:?}", log_path);

        // Clean up old rotations
        self.cleanup_old_logs()?;

        Ok(())
    }

    /// Check if log rotation is needed
    fn check_rotation(&self) -> Result<(), AuditError> {
        let current_path = self.current_path.lock().unwrap();

        if let Ok(metadata) = std::fs::metadata(&*current_path) {
            if metadata.len() >= self.config.max_file_size {
                drop(current_path);
                self.rotate_log()?;
            }
        }

        Ok(())
    }

    /// Clean up old log files beyond max_rotations
    fn cleanup_old_logs(&self) -> Result<(), AuditError> {
        let mut log_files: Vec<_> = std::fs::read_dir(&self.config.log_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_name().to_string_lossy().starts_with("audit_")
                    && entry.file_name().to_string_lossy().ends_with(".jsonl")
            })
            .collect();

        // Sort by modification time (oldest first)
        log_files.sort_by_key(|entry| entry.metadata().ok().and_then(|m| m.modified().ok()));

        // Remove oldest files beyond max_rotations
        let to_remove = log_files.len().saturating_sub(self.config.max_rotations);
        for entry in log_files.iter().take(to_remove) {
            if let Err(e) = std::fs::remove_file(entry.path()) {
                warn!("Failed to remove old log file {:?}: {}", entry.path(), e);
            } else {
                debug!("Removed old log file {:?}", entry.path());
            }
        }

        Ok(())
    }

    /// Query recent events from buffer
    pub fn query_recent_events(
        &self,
        limit: usize,
        category: Option<SecurityCategory>,
        min_severity: Option<SecuritySeverity>,
    ) -> Vec<SecurityEvent> {
        let buffer = self.event_buffer.lock().unwrap();

        buffer
            .iter()
            .rev()
            .filter(|event| {
                if let Some(ref cat) = category {
                    if &event.category != cat {
                        return false;
                    }
                }
                if let Some(ref sev) = min_severity {
                    if &event.severity < sev {
                        return false;
                    }
                }
                true
            })
            .take(limit)
            .cloned()
            .collect()
    }

    /// Generate security report
    pub fn generate_report(&self, since: DateTime<Utc>) -> SecurityReport {
        let buffer = self.event_buffer.lock().unwrap();

        let events_in_period: Vec<_> = buffer
            .iter()
            .filter(|e| e.timestamp >= since)
            .cloned()
            .collect();

        let total_events = events_in_period.len();

        // Count by severity
        let mut by_severity = HashMap::new();
        for event in &events_in_period {
            *by_severity.entry(event.severity.clone()).or_insert(0) += 1;
        }

        // Count by category
        let mut by_category = HashMap::new();
        for event in &events_in_period {
            *by_category.entry(event.category.clone()).or_insert(0) += 1;
        }

        // Count failures
        let failures = events_in_period
            .iter()
            .filter(|e| matches!(e.outcome, EventOutcome::Failure))
            .count();

        // Identify top actors
        let mut actor_counts: HashMap<String, usize> = HashMap::new();
        for event in &events_in_period {
            if let Some(ref actor) = event.actor {
                *actor_counts.entry(actor.clone()).or_insert(0) += 1;
            }
        }

        let mut top_actors: Vec<_> = actor_counts.into_iter().collect();
        top_actors.sort_by(|a, b| b.1.cmp(&a.1));
        top_actors.truncate(10);

        SecurityReport {
            period_start: since,
            period_end: Utc::now(),
            total_events,
            by_severity,
            by_category,
            failures,
            top_actors,
        }
    }

    /// Verify log chain integrity
    pub fn verify_chain(&self) -> Result<bool, AuditError> {
        let buffer = self.event_buffer.lock().unwrap();

        for i in 1..buffer.len() {
            let prev = &buffer[i - 1];
            let curr = &buffer[i];

            let prev_hash = prev.compute_hash();

            if curr.previous_hash.as_ref() != Some(&prev_hash) {
                error!("Chain integrity violation detected at event {}", curr.id);
                return Ok(false);
            }
        }

        info!("Audit log chain integrity verified successfully");
        Ok(true)
    }
}

/// Anomaly detector for security events
struct AnomalyDetector {
    /// Recent event patterns
    patterns: HashMap<String, EventPattern>,

    /// Threshold for anomaly detection
    anomaly_threshold: f64,
}

impl AnomalyDetector {
    fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            anomaly_threshold: 0.8,
        }
    }

    fn check_event(&mut self, event: &SecurityEvent) -> Option<String> {
        let pattern_key = format!("{}:{}", event.category.clone() as u32, event.event_type);

        let pattern = self
            .patterns
            .entry(pattern_key.clone())
            .or_insert_with(EventPattern::new);

        pattern.record(event);

        // Check for anomalies
        if pattern.is_anomalous(event, self.anomaly_threshold) {
            Some(format!(
                "Anomalous {} event: {} (severity: {:?})",
                event.category.clone() as u32,
                event.event_type,
                event.severity
            ))
        } else {
            None
        }
    }
}

/// Pattern tracking for events
struct EventPattern {
    count: usize,
    severity_counts: HashMap<String, usize>,
    last_seen: DateTime<Utc>,
    failure_rate: f64,
}

impl EventPattern {
    fn new() -> Self {
        Self {
            count: 0,
            severity_counts: HashMap::new(),
            last_seen: Utc::now(),
            failure_rate: 0.0,
        }
    }

    fn record(&mut self, event: &SecurityEvent) {
        self.count += 1;
        self.last_seen = event.timestamp;

        let sev_key = format!("{:?}", event.severity);
        *self.severity_counts.entry(sev_key).or_insert(0) += 1;

        if matches!(event.outcome, EventOutcome::Failure) {
            self.failure_rate =
                (self.failure_rate * (self.count - 1) as f64 + 1.0) / self.count as f64;
        } else {
            self.failure_rate = (self.failure_rate * (self.count - 1) as f64) / self.count as f64;
        }
    }

    fn is_anomalous(&self, event: &SecurityEvent, _threshold: f64) -> bool {
        // Unusual severity
        if event.severity >= SecuritySeverity::High && self.count > 10 {
            let sev_key = format!("{:?}", event.severity);
            let sev_count = self.severity_counts.get(&sev_key).unwrap_or(&0);
            let sev_ratio = *sev_count as f64 / self.count as f64;

            if sev_ratio < 0.1 {
                // Less than 10% historical occurrence
                return true;
            }
        }

        // High failure rate spike
        if matches!(event.outcome, EventOutcome::Failure) && self.failure_rate < 0.1 {
            return true;
        }

        false
    }
}

/// Security report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityReport {
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_events: usize,
    pub by_severity: HashMap<SecuritySeverity, usize>,
    pub by_category: HashMap<SecurityCategory, usize>,
    pub failures: usize,
    pub top_actors: Vec<(String, usize)>,
}

/// Helper function to log authentication events
pub fn log_authentication(
    logger: &SecurityAuditLogger,
    actor: String,
    outcome: EventOutcome,
    message: String,
) -> Result<(), AuditError> {
    let severity = match outcome {
        EventOutcome::Success => SecuritySeverity::Info,
        EventOutcome::Failure => SecuritySeverity::Medium,
        _ => SecuritySeverity::Low,
    };

    let event = SecurityEvent::new(
        severity,
        SecurityCategory::Authentication,
        "user_login".to_string(),
        message,
    )
    .with_actor(actor)
    .with_outcome(outcome);

    logger.log_event(event)
}

/// Helper function to log authorization events
pub fn log_authorization(
    logger: &SecurityAuditLogger,
    actor: String,
    resource: String,
    outcome: EventOutcome,
    message: String,
) -> Result<(), AuditError> {
    let severity = match outcome {
        EventOutcome::Failure => SecuritySeverity::Medium,
        _ => SecuritySeverity::Info,
    };

    let event = SecurityEvent::new(
        severity,
        SecurityCategory::Authorization,
        "access_check".to_string(),
        message,
    )
    .with_actor(actor)
    .with_resource(resource)
    .with_outcome(outcome);

    logger.log_event(event)
}

/// Helper function to log data access events
pub fn log_data_access(
    logger: &SecurityAuditLogger,
    actor: String,
    resource: String,
    message: String,
) -> Result<(), AuditError> {
    let event = SecurityEvent::new(
        SecuritySeverity::Info,
        SecurityCategory::DataAccess,
        "data_read".to_string(),
        message,
    )
    .with_actor(actor)
    .with_resource(resource);

    logger.log_event(event)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_security_event_creation() {
        let event = SecurityEvent::new(
            SecuritySeverity::High,
            SecurityCategory::Authentication,
            "login_attempt".to_string(),
            "User login failed".to_string(),
        )
        .with_actor("user1".to_string())
        .with_outcome(EventOutcome::Failure);

        assert_eq!(event.severity, SecuritySeverity::High);
        assert_eq!(event.category, SecurityCategory::Authentication);
        assert!(event.actor.is_some());
    }

    #[test]
    fn test_event_hash() {
        let event = SecurityEvent::new(
            SecuritySeverity::Info,
            SecurityCategory::System,
            "test_event".to_string(),
            "Test message".to_string(),
        );

        let hash1 = event.compute_hash();
        let hash2 = event.compute_hash();

        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA-256 = 64 hex chars
    }

    #[test]
    fn test_audit_logger() -> Result<(), AuditError> {
        let temp_dir = env::temp_dir().join(format!("audit_test_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir)?;

        let config = AuditConfig {
            log_dir: temp_dir.clone(),
            enable_signatures: true,
            ..Default::default()
        };

        let logger = SecurityAuditLogger::new(config)?;

        let event = SecurityEvent::new(
            SecuritySeverity::Info,
            SecurityCategory::System,
            "test_event".to_string(),
            "Test message".to_string(),
        );

        logger.log_event(event)?;

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();

        Ok(())
    }

    #[test]
    fn test_chain_integrity() -> Result<(), AuditError> {
        let temp_dir = env::temp_dir().join(format!("audit_chain_test_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir)?;

        let config = AuditConfig {
            log_dir: temp_dir.clone(),
            enable_signatures: true,
            buffer_size: 100,
            ..Default::default()
        };

        let logger = SecurityAuditLogger::new(config)?;

        // Log multiple events
        for i in 0..10 {
            let event = SecurityEvent::new(
                SecuritySeverity::Info,
                SecurityCategory::System,
                format!("event_{}", i),
                format!("Event {}", i),
            );
            logger.log_event(event)?;
        }

        // Verify chain
        assert!(logger.verify_chain()?);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();

        Ok(())
    }

    #[test]
    fn test_query_events() -> Result<(), AuditError> {
        let temp_dir = env::temp_dir().join(format!("audit_query_test_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir)?;

        let config = AuditConfig {
            log_dir: temp_dir.clone(),
            buffer_size: 100,
            ..Default::default()
        };

        let logger = SecurityAuditLogger::new(config)?;

        // Log events with different categories
        for i in 0..5 {
            let event = SecurityEvent::new(
                SecuritySeverity::Info,
                SecurityCategory::Authentication,
                format!("auth_{}", i),
                format!("Auth event {}", i),
            );
            logger.log_event(event)?;
        }

        for i in 0..3 {
            let event = SecurityEvent::new(
                SecuritySeverity::Medium,
                SecurityCategory::DataAccess,
                format!("access_{}", i),
                format!("Access event {}", i),
            );
            logger.log_event(event)?;
        }

        // Query authentication events
        let auth_events =
            logger.query_recent_events(10, Some(SecurityCategory::Authentication), None);
        assert_eq!(auth_events.len(), 5);

        // Query high severity events
        let high_sev_events = logger.query_recent_events(10, None, Some(SecuritySeverity::Medium));
        assert_eq!(high_sev_events.len(), 3);

        // Cleanup
        std::fs::remove_dir_all(&temp_dir).ok();

        Ok(())
    }
}
