//! Backup policy executor with structured audit log.
//!
//! `BackupExecutor` consumes a `BackupPolicy` and writes backup artefacts to
//! the configured `DestinationConfig`.  Every operation (start, complete,
//! fail, prune) is appended to an in-memory audit log that callers can query.

use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use thiserror::Error;

use super::policy::BackupPolicy;

// ---------------------------------------------------------------------------
// Audit types
// ---------------------------------------------------------------------------

/// A single entry in the backup audit log.
#[derive(Debug)]
pub struct BackupAuditEntry {
    /// When this entry was recorded.
    pub timestamp: SystemTime,
    /// Name of the policy that triggered this action.
    pub policy_name: String,
    /// What happened.
    pub action: AuditAction,
    /// Bytes written (only set for `BackupCompleted`).
    pub size_bytes: Option<u64>,
    /// Error message (only set for `BackupFailed`).
    pub error: Option<String>,
}

/// Discriminant for audit log entries.
#[derive(Debug)]
pub enum AuditAction {
    /// A backup has been initiated.
    BackupStarted,
    /// A backup completed successfully.
    BackupCompleted,
    /// A backup failed.
    BackupFailed,
    /// An old backup artefact was removed during retention enforcement.
    RetentionPruned {
        /// Identifier of the backup that was pruned.
        backup_id: u64,
    },
}

impl AuditAction {
    fn label(&self) -> &'static str {
        match self {
            AuditAction::BackupStarted => "Started",
            AuditAction::BackupCompleted => "Completed",
            AuditAction::BackupFailed => "Failed",
            AuditAction::RetentionPruned { .. } => "Pruned",
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors produced by `BackupExecutor`.
#[derive(Debug, Error)]
pub enum BackupError {
    /// A filesystem I/O error.
    #[error("IO error: {0}")]
    IoError(String),
    /// The backup was written but post-write verification failed.
    #[error("Verification failed: {0}")]
    VerificationFailed(String),
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Policy-driven backup executor with an append-only audit log.
pub struct BackupExecutor {
    /// Thread-safe append-only audit log.
    audit_log: Mutex<Vec<BackupAuditEntry>>,
}

impl BackupExecutor {
    /// Create a new executor with an empty audit log.
    pub fn new() -> Self {
        BackupExecutor {
            audit_log: Mutex::new(Vec::new()),
        }
    }

    /// Execute a backup according to `policy`.
    ///
    /// Writes `data` as a single flat file under `destination`.  The file
    /// name embeds the current Unix timestamp to prevent collisions.
    ///
    /// Returns the number of bytes written on success.
    pub fn execute_backup(
        &self,
        policy: &BackupPolicy,
        data: &[u8],
        destination: &Path,
    ) -> Result<u64, BackupError> {
        self.append_audit(policy.name.clone(), AuditAction::BackupStarted, None, None);

        // Build a unique file name based on current timestamp
        let ts_secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        let dest_file = destination.join(format!("backup_{}.bin", ts_secs));

        // Ensure the destination directory exists
        if let Err(e) = std::fs::create_dir_all(destination) {
            let msg = format!("create_dir_all failed: {e}");
            self.append_audit(
                policy.name.clone(),
                AuditAction::BackupFailed,
                None,
                Some(msg.clone()),
            );
            return Err(BackupError::IoError(msg));
        }

        // Write the backup artefact
        if let Err(e) = std::fs::write(&dest_file, data) {
            let msg = format!("write failed: {e}");
            self.append_audit(
                policy.name.clone(),
                AuditAction::BackupFailed,
                None,
                Some(msg.clone()),
            );
            return Err(BackupError::IoError(msg));
        }

        let size = data.len() as u64;
        self.append_audit(
            policy.name.clone(),
            AuditAction::BackupCompleted,
            Some(size),
            None,
        );

        Ok(size)
    }

    /// Record that a backup artefact was pruned by retention enforcement.
    pub fn record_prune(&self, policy_name: &str, backup_id: u64) {
        self.append_audit(
            policy_name.to_owned(),
            AuditAction::RetentionPruned { backup_id },
            None,
            None,
        );
    }

    /// Return a human-readable snapshot of the audit log.
    ///
    /// Format: `"<policy>: <unix_ts> — <action>"`
    pub fn audit_entries(&self) -> Vec<String> {
        self.lock_log()
            .iter()
            .map(|e| {
                let ts = e
                    .timestamp
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_secs();
                format!("{}: {} — {}", e.policy_name, ts, e.action.label())
            })
            .collect()
    }

    /// Raw access to the audit log entries (useful for tests).
    pub fn raw_audit_log(&self) -> Vec<String> {
        self.audit_entries()
    }

    /// Number of entries in the audit log.
    pub fn audit_len(&self) -> usize {
        self.lock_log().len()
    }

    // Internal helpers

    fn append_audit(
        &self,
        policy_name: String,
        action: AuditAction,
        size_bytes: Option<u64>,
        error: Option<String>,
    ) {
        let entry = BackupAuditEntry {
            timestamp: SystemTime::now(),
            policy_name,
            action,
            size_bytes,
            error,
        };
        self.lock_log().push(entry);
    }

    fn lock_log(&self) -> MutexGuard<'_, Vec<BackupAuditEntry>> {
        self.audit_log
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backup::{
        destination::DestinationConfig,
        gfs::GfsRotation,
        policy::{BackupPolicy, CronSchedule, EncryptionConfig},
        retention::RetentionTier,
    };
    use std::env;

    fn make_policy(dir: &Path) -> BackupPolicy {
        BackupPolicy {
            name: "test-policy".into(),
            schedule: CronSchedule::daily(),
            retention: RetentionTier::minimal(),
            gfs: Some(GfsRotation::default()),
            encryption: EncryptionConfig::none(),
            destination: DestinationConfig::Filesystem {
                path: dir.to_owned(),
            },
        }
    }

    #[test]
    fn execute_backup_writes_file_and_returns_size() {
        let dir = env::temp_dir().join(format!(
            "oxirs_executor_test_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos()
        ));

        let executor = BackupExecutor::new();
        let policy = make_policy(&dir);
        let data = b"backup payload data 1234567890";

        let size = executor
            .execute_backup(&policy, data, &dir)
            .expect("backup should succeed");

        assert_eq!(size, data.len() as u64);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn audit_log_contains_started_and_completed() {
        let dir = env::temp_dir().join(format!(
            "oxirs_audit_test_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos()
        ));

        let executor = BackupExecutor::new();
        let policy = make_policy(&dir);

        executor
            .execute_backup(&policy, b"data", &dir)
            .expect("backup should succeed");

        let entries = executor.audit_entries();
        assert!(
            entries.len() >= 2,
            "expected ≥2 audit entries; got {}",
            entries.len()
        );

        let has_started = entries.iter().any(|e| e.contains("Started"));
        let has_completed = entries.iter().any(|e| e.contains("Completed"));
        assert!(has_started, "missing 'Started' entry: {entries:?}");
        assert!(has_completed, "missing 'Completed' entry: {entries:?}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_prune_adds_pruned_entry() {
        let executor = BackupExecutor::new();
        executor.record_prune("my-policy", 42);
        let entries = executor.audit_entries();
        assert!(entries.iter().any(|e| e.contains("Pruned")));
    }

    #[test]
    fn audit_len_increases_with_operations() {
        let dir = env::temp_dir().join(format!(
            "oxirs_len_test_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos()
        ));

        let executor = BackupExecutor::new();
        assert_eq!(executor.audit_len(), 0);

        let policy = make_policy(&dir);
        executor.execute_backup(&policy, b"x", &dir).ok();

        assert!(executor.audit_len() >= 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn backup_to_invalid_path_returns_error_and_logs_failed() {
        let executor = BackupExecutor::new();

        // Create a temp file to use as destination (files can't be used as dirs)
        let temp_file = env::temp_dir().join(format!(
            "oxirs_invalid_{}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos()
        ));
        std::fs::write(&temp_file, b"block").ok();

        // Try to create a file *inside* another file — guaranteed to fail
        let bad_dest = temp_file.join("subdir");

        let policy = BackupPolicy {
            name: "fail-test".into(),
            schedule: CronSchedule::daily(),
            retention: RetentionTier::minimal(),
            gfs: None,
            encryption: EncryptionConfig::none(),
            destination: DestinationConfig::Filesystem {
                path: bad_dest.clone(),
            },
        };

        let result = executor.execute_backup(&policy, b"data", &bad_dest);
        assert!(result.is_err(), "Expected Err for invalid destination");

        let entries = executor.audit_entries();
        assert!(
            entries.iter().any(|e| e.contains("Failed")),
            "Expected 'Failed' audit entry"
        );

        let _ = std::fs::remove_file(&temp_file);
    }
}
