//! Security audit logging

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Security event for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SecurityEvent {
    InputValidation {
        user_id: String,
        input_length: usize,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    RateLimitExceeded {
        user_id: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    CredentialStored {
        provider: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    CredentialAccessed {
        provider: String,
        user_id: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    IpBlocked {
        ip: String,
        reason: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    AuthenticationFailure {
        user_id: String,
        reason: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    SuspiciousActivity {
        user_id: String,
        activity_type: String,
        description: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    MemoryLimitExceeded {
        requested_bytes: usize,
        available_bytes: usize,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
}

/// Audit logger
pub struct AuditLogger {
    events: Arc<RwLock<Vec<SecurityEvent>>>,
    log_file: Option<Arc<RwLock<BufWriter<File>>>>,
    max_memory_events: usize,
}

impl AuditLogger {
    /// Create new audit logger
    pub fn new() -> Result<Self> {
        Ok(Self {
            events: Arc::new(RwLock::new(Vec::new())),
            log_file: None,
            max_memory_events: 10_000,
        })
    }

    /// Create audit logger with file output
    pub fn with_file(path: PathBuf) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        let writer = BufWriter::new(file);

        Ok(Self {
            events: Arc::new(RwLock::new(Vec::new())),
            log_file: Some(Arc::new(RwLock::new(writer))),
            max_memory_events: 10_000,
        })
    }

    /// Log a security event
    pub fn log_event(&self, event: SecurityEvent) -> Result<()> {
        // Write to file if configured
        if let Some(log_file) = &self.log_file {
            let json = serde_json::to_string(&event)?;
            let mut writer = log_file.write()
                .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;
            writeln!(writer, "{}", json)?;
            writer.flush()?;
        }

        // Store in memory
        let mut events = self.events.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        events.push(event);

        // Trim if exceeding max
        if events.len() > self.max_memory_events {
            let excess = events.len() - self.max_memory_events;
            events.drain(0..excess);
        }

        Ok(())
    }

    /// Get events since timestamp
    pub fn get_events_since(&self, since: chrono::DateTime<chrono::Utc>) -> Result<Vec<SecurityEvent>> {
        let events = self.events.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        Ok(events
            .iter()
            .filter(|event| Self::event_timestamp(event) >= &since)
            .cloned()
            .collect())
    }

    /// Get all events in memory
    pub fn get_all_events(&self) -> Result<Vec<SecurityEvent>> {
        let events = self.events.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        Ok(events.clone())
    }

    /// Get event count
    pub fn event_count(&self) -> Result<usize> {
        let events = self.events.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        Ok(events.len())
    }

    /// Clear all events from memory
    pub fn clear(&self) -> Result<()> {
        let mut events = self.events.write()
            .map_err(|e| anyhow!("Failed to acquire write lock: {}", e))?;

        events.clear();
        Ok(())
    }

    /// Get statistics
    pub fn get_statistics(&self) -> Result<AuditStatistics> {
        let events = self.events.read()
            .map_err(|e| anyhow!("Failed to acquire read lock: {}", e))?;

        let mut stats = AuditStatistics::default();

        for event in events.iter() {
            match event {
                SecurityEvent::InputValidation { .. } => stats.input_validations += 1,
                SecurityEvent::RateLimitExceeded { .. } => stats.rate_limit_violations += 1,
                SecurityEvent::CredentialStored { .. } => stats.credential_operations += 1,
                SecurityEvent::CredentialAccessed { .. } => stats.credential_operations += 1,
                SecurityEvent::IpBlocked { .. } => stats.ip_blocks += 1,
                SecurityEvent::AuthenticationFailure { .. } => stats.auth_failures += 1,
                SecurityEvent::SuspiciousActivity { .. } => stats.suspicious_activities += 1,
                SecurityEvent::MemoryLimitExceeded { .. } => stats.memory_limit_violations += 1,
            }
        }

        Ok(stats)
    }

    fn event_timestamp(event: &SecurityEvent) -> &chrono::DateTime<chrono::Utc> {
        match event {
            SecurityEvent::InputValidation { timestamp, .. } => timestamp,
            SecurityEvent::RateLimitExceeded { timestamp, .. } => timestamp,
            SecurityEvent::CredentialStored { timestamp, .. } => timestamp,
            SecurityEvent::CredentialAccessed { timestamp, .. } => timestamp,
            SecurityEvent::IpBlocked { timestamp, .. } => timestamp,
            SecurityEvent::AuthenticationFailure { timestamp, .. } => timestamp,
            SecurityEvent::SuspiciousActivity { timestamp, .. } => timestamp,
            SecurityEvent::MemoryLimitExceeded { timestamp, .. } => timestamp,
        }
    }
}

/// Audit statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AuditStatistics {
    pub input_validations: usize,
    pub rate_limit_violations: usize,
    pub credential_operations: usize,
    pub ip_blocks: usize,
    pub auth_failures: usize,
    pub suspicious_activities: usize,
    pub memory_limit_violations: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_event() {
        let logger = AuditLogger::new().unwrap();

        let event = SecurityEvent::InputValidation {
            user_id: "test_user".to_string(),
            input_length: 100,
            timestamp: chrono::Utc::now(),
        };

        logger.log_event(event).unwrap();
        assert_eq!(logger.event_count().unwrap(), 1);
    }

    #[test]
    fn test_get_events_since() {
        let logger = AuditLogger::new().unwrap();

        let now = chrono::Utc::now();
        let past = now - chrono::Duration::minutes(10);

        let event = SecurityEvent::RateLimitExceeded {
            user_id: "test_user".to_string(),
            timestamp: now,
        };

        logger.log_event(event).unwrap();

        let events = logger.get_events_since(past).unwrap();
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_statistics() {
        let logger = AuditLogger::new().unwrap();

        logger.log_event(SecurityEvent::InputValidation {
            user_id: "user1".to_string(),
            input_length: 100,
            timestamp: chrono::Utc::now(),
        }).unwrap();

        logger.log_event(SecurityEvent::RateLimitExceeded {
            user_id: "user2".to_string(),
            timestamp: chrono::Utc::now(),
        }).unwrap();

        let stats = logger.get_statistics().unwrap();
        assert_eq!(stats.input_validations, 1);
        assert_eq!(stats.rate_limit_violations, 1);
    }

    #[test]
    fn test_max_memory_events() {
        let logger = AuditLogger::new().unwrap();

        // Add more events than max_memory_events
        for i in 0..15_000 {
            logger.log_event(SecurityEvent::InputValidation {
                user_id: format!("user{}", i),
                input_length: 100,
                timestamp: chrono::Utc::now(),
            }).unwrap();
        }

        // Should be trimmed to max
        assert_eq!(logger.event_count().unwrap(), 10_000);
    }
}
