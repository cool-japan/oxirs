//! Backup policy definition.
//!
//! A `BackupPolicy` combines a schedule, retention tier, optional GFS
//! rotation, encryption configuration and destination into a single
//! serialisable struct that drives the backup executor.

use serde::{Deserialize, Serialize};

use super::{destination::DestinationConfig, gfs::GfsRotation, retention::RetentionTier};

/// A complete backup policy describing when, what, and where to back up.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupPolicy {
    /// Human-readable identifier for this policy.
    pub name: String,
    /// When this policy runs.
    pub schedule: CronSchedule,
    /// How long to keep backups across hot/warm/cold tiers.
    pub retention: RetentionTier,
    /// Optional Grandfather-Father-Son rotation.
    pub gfs: Option<GfsRotation>,
    /// Encryption settings applied at write time.
    pub encryption: EncryptionConfig,
    /// Where to write backup artefacts.
    pub destination: DestinationConfig,
}

/// Cron-compatible schedule expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronSchedule {
    /// Standard 5-field cron expression (`minute hour day month weekday`).
    pub expression: String,
    /// Human-readable description.
    pub description: String,
}

impl CronSchedule {
    /// Every day at 02:00 UTC.
    pub fn daily() -> Self {
        CronSchedule {
            expression: "0 2 * * *".into(),
            description: "Every day at 02:00 UTC".into(),
        }
    }

    /// Every Sunday at 02:00 UTC.
    pub fn weekly() -> Self {
        CronSchedule {
            expression: "0 2 * * 0".into(),
            description: "Every Sunday at 02:00 UTC".into(),
        }
    }

    /// First day of every month at 02:00 UTC.
    pub fn monthly() -> Self {
        CronSchedule {
            expression: "0 2 1 * *".into(),
            description: "First day of month at 02:00 UTC".into(),
        }
    }
}

/// Encryption settings for a backup policy.
///
/// When `enabled` is `true`, the executor tags the backup artefact with
/// `key_id` so the restore path knows which key to use.  Actual
/// encryption is delegated to `oxirs_cluster::encryption`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Whether encryption is required for this policy.
    pub enabled: bool,
    /// Key identifier looked up in the key-management store.
    pub key_id: Option<String>,
}

impl EncryptionConfig {
    /// Encryption disabled.
    pub fn none() -> Self {
        EncryptionConfig {
            enabled: false,
            key_id: None,
        }
    }

    /// Encryption enabled with the given key identifier.
    pub fn with_key(key_id: &str) -> Self {
        EncryptionConfig {
            enabled: true,
            key_id: Some(key_id.to_owned()),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backup::{
        destination::DestinationConfig, gfs::GfsRotation, retention::RetentionTier,
    };
    use std::env;

    fn make_policy(name: &str) -> BackupPolicy {
        BackupPolicy {
            name: name.into(),
            schedule: CronSchedule::daily(),
            retention: RetentionTier::standard(),
            gfs: Some(GfsRotation::default()),
            encryption: EncryptionConfig::none(),
            destination: DestinationConfig::Filesystem {
                path: env::temp_dir().join("oxirs_policy_test"),
            },
        }
    }

    #[test]
    fn policy_builds_and_serialises() {
        let p = make_policy("daily-test");
        let json = serde_json::to_string(&p).unwrap();
        let back: BackupPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, p.name);
        assert_eq!(back.schedule.expression, p.schedule.expression);
    }

    #[test]
    fn cron_schedules_have_correct_expressions() {
        assert_eq!(CronSchedule::daily().expression, "0 2 * * *");
        assert_eq!(CronSchedule::weekly().expression, "0 2 * * 0");
        assert_eq!(CronSchedule::monthly().expression, "0 2 1 * *");
    }

    #[test]
    fn encryption_config_none() {
        let cfg = EncryptionConfig::none();
        assert!(!cfg.enabled);
        assert!(cfg.key_id.is_none());
    }

    #[test]
    fn encryption_config_with_key() {
        let cfg = EncryptionConfig::with_key("key-abc");
        assert!(cfg.enabled);
        assert_eq!(cfg.key_id.as_deref(), Some("key-abc"));
    }

    #[test]
    fn encryption_config_serialises() {
        let cfg = EncryptionConfig::with_key("key-123");
        let json = serde_json::to_string(&cfg).unwrap();
        let back: EncryptionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.key_id.as_deref(), Some("key-123"));
        assert!(back.enabled);
    }
}
