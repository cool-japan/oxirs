//! Backup policy executor with structured audit log.
//!
//! `BackupExecutor` consumes a `BackupPolicy` and writes backup artefacts to
//! the configured `DestinationConfig`.  Every operation (start, complete,
//! fail, prune) is appended to an in-memory audit log that callers can query.

use std::borrow::Cow;
use std::path::Path;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use thiserror::Error;

use super::policy::BackupPolicy;
use crate::at_rest::{self, AtRestKey, AtRestKeyring};

/// Context bound into every encrypted artefact written by [`BackupExecutor`].
const BACKUP_CONTEXT: &[u8] = b"oxirs-cluster/backup-policy";

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
    /// The policy requires encryption but it could not be applied (no key,
    /// unknown key id), or an artefact could not be decrypted.
    #[error("Encryption error: {0}")]
    Encryption(String),
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Policy-driven backup executor with an append-only audit log.
pub struct BackupExecutor {
    /// Thread-safe append-only audit log.
    audit_log: Mutex<Vec<BackupAuditEntry>>,
    /// Keys for policies with encryption enabled, looked up by the policy's
    /// `key_id`.
    keyring: AtRestKeyring,
}

impl BackupExecutor {
    /// Create a new executor with an empty audit log and no encryption keys.
    ///
    /// Policies that enable encryption fail with [`BackupError::Encryption`]
    /// on such an executor; use [`BackupExecutor::with_keyring`] for them.
    pub fn new() -> Self {
        Self::with_keyring(AtRestKeyring::new())
    }

    /// Create an executor that encrypts backups for every policy whose
    /// `encryption.key_id` names a key in `keyring`.
    pub fn with_keyring(keyring: AtRestKeyring) -> Self {
        BackupExecutor {
            audit_log: Mutex::new(Vec::new()),
            keyring,
        }
    }

    /// Execute a backup according to `policy`.
    ///
    /// Writes `data` as a single flat file under `destination`.  The file
    /// name embeds the current Unix timestamp to prevent collisions.  When
    /// `policy.encryption.enabled` is set, the file holds `data` sealed with
    /// XChaCha20-Poly1305 (see [`crate::at_rest`]) under the key named by
    /// `policy.encryption.key_id`; read it back with
    /// [`BackupExecutor::read_backup`].
    ///
    /// Returns the number of bytes written on success.
    ///
    /// # Errors
    ///
    /// Fails with [`BackupError::Encryption`] when encryption is enabled but
    /// the policy names no key or the key is not in the executor's keyring;
    /// nothing is written in that case.
    pub fn execute_backup(
        &self,
        policy: &BackupPolicy,
        data: &[u8],
        destination: &Path,
    ) -> Result<u64, BackupError> {
        self.append_audit(policy.name.clone(), AuditAction::BackupStarted, None, None);

        let artefact = match self.encode_artefact(policy, data) {
            Ok(artefact) => artefact,
            Err(e) => {
                self.append_audit(
                    policy.name.clone(),
                    AuditAction::BackupFailed,
                    None,
                    Some(e.to_string()),
                );
                return Err(e);
            }
        };

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
        if let Err(e) = std::fs::write(&dest_file, &artefact) {
            let msg = format!("write failed: {e}");
            self.append_audit(
                policy.name.clone(),
                AuditAction::BackupFailed,
                None,
                Some(msg.clone()),
            );
            return Err(BackupError::IoError(msg));
        }

        let size = artefact.len() as u64;
        self.append_audit(
            policy.name.clone(),
            AuditAction::BackupCompleted,
            Some(size),
            None,
        );

        Ok(size)
    }

    /// Read back a backup artefact written by [`BackupExecutor::execute_backup`]
    /// under `policy`, decrypting it when the policy requires encryption.
    ///
    /// # Errors
    ///
    /// Fails with [`BackupError::IoError`] when the file cannot be read, and
    /// with [`BackupError::Encryption`] when the policy requires encryption but
    /// the file is not sealed, is sealed under another key, or does not
    /// authenticate — or when the file is sealed but the policy has
    /// encryption disabled.
    pub fn read_backup(&self, policy: &BackupPolicy, path: &Path) -> Result<Vec<u8>, BackupError> {
        let bytes =
            std::fs::read(path).map_err(|e| BackupError::IoError(format!("read failed: {e}")))?;

        if policy.encryption.enabled {
            if !at_rest::is_sealed(&bytes) {
                return Err(BackupError::Encryption(format!(
                    "{} is not encrypted, but policy '{}' requires encryption",
                    path.display(),
                    policy.name
                )));
            }
            let key = self.policy_key(policy)?;
            at_rest::open(key, BACKUP_CONTEXT, &bytes)
                .map_err(|e| BackupError::Encryption(e.to_string()))
        } else if at_rest::is_sealed(&bytes) {
            Err(BackupError::Encryption(format!(
                "{} is encrypted, but policy '{}' has encryption disabled",
                path.display(),
                policy.name
            )))
        } else {
            Ok(bytes)
        }
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

    /// The bytes to store for `data` under `policy`: sealed when the policy
    /// requires encryption, unchanged otherwise.
    fn encode_artefact<'a>(
        &self,
        policy: &BackupPolicy,
        data: &'a [u8],
    ) -> Result<Cow<'a, [u8]>, BackupError> {
        if !policy.encryption.enabled {
            return Ok(Cow::Borrowed(data));
        }
        let key = self.policy_key(policy)?;
        at_rest::seal(key, BACKUP_CONTEXT, data)
            .map(Cow::Owned)
            .map_err(|e| BackupError::Encryption(e.to_string()))
    }

    /// The key named by an encrypting policy.
    fn policy_key(&self, policy: &BackupPolicy) -> Result<&AtRestKey, BackupError> {
        let key_id = policy.encryption.key_id.as_deref().ok_or_else(|| {
            BackupError::Encryption(format!(
                "policy '{}' enables encryption but names no key_id",
                policy.name
            ))
        })?;
        self.keyring.get(key_id).ok_or_else(|| {
            BackupError::Encryption(format!(
                "policy '{}' requires key '{key_id}', which is not in the executor's keyring",
                policy.name
            ))
        })
    }

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
    use crate::at_rest::{AtRestKey, AtRestKeyring};
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

    fn encrypted_policy(dir: &Path, key_id: &str) -> BackupPolicy {
        BackupPolicy {
            encryption: EncryptionConfig::with_key(key_id),
            ..make_policy(dir)
        }
    }

    fn keyring_with(id: &str, byte: u8) -> AtRestKeyring {
        let mut keyring = AtRestKeyring::new();
        keyring.insert(AtRestKey::new(id, [byte; 32]).expect("valid key id"));
        keyring
    }

    fn only_artefact(dir: &Path) -> std::path::PathBuf {
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .expect("failed to list backup dir")
            .map(|entry| entry.expect("bad dir entry").path())
            .collect();
        assert_eq!(files.len(), 1, "expected exactly one artefact: {files:?}");
        files.remove(0)
    }

    #[test]
    fn encrypted_backup_round_trips_through_read_backup() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let policy = encrypted_policy(dir.path(), "k1");
        let executor = BackupExecutor::with_keyring(keyring_with("k1", 1));

        let written = executor
            .execute_backup(&policy, b"secret payload", dir.path())
            .expect("encrypted backup should succeed");

        let artefact = only_artefact(dir.path());
        let bytes = std::fs::read(&artefact).expect("read artefact");
        assert_eq!(written, bytes.len() as u64);
        assert!(at_rest::is_sealed(&bytes));
        assert_eq!(
            executor
                .read_backup(&policy, &artefact)
                .expect("read_backup"),
            b"secret payload"
        );
    }

    #[test]
    fn encrypted_backup_without_key_fails_and_writes_nothing() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let policy = encrypted_policy(dir.path(), "missing");
        let executor = BackupExecutor::with_keyring(keyring_with("k1", 1));

        let result = executor.execute_backup(&policy, b"secret payload", dir.path());

        assert!(matches!(result, Err(BackupError::Encryption(_))));
        assert!(executor
            .audit_entries()
            .iter()
            .any(|e| e.contains("Failed")));
        assert_eq!(std::fs::read_dir(dir.path()).expect("list").count(), 0);

        let no_key_id = BackupPolicy {
            encryption: EncryptionConfig {
                enabled: true,
                key_id: None,
            },
            ..make_policy(dir.path())
        };
        assert!(matches!(
            executor.execute_backup(&no_key_id, b"x", dir.path()),
            Err(BackupError::Encryption(_))
        ));
    }

    #[test]
    fn read_backup_rejects_tampered_wrong_key_and_mixed_artefacts() {
        let dir = tempfile::TempDir::new().expect("temp dir");
        let policy = encrypted_policy(dir.path(), "k1");
        let executor = BackupExecutor::with_keyring(keyring_with("k1", 1));
        executor
            .execute_backup(&policy, b"secret payload", dir.path())
            .expect("encrypted backup should succeed");
        let artefact = only_artefact(dir.path());

        // Same key id, different key bytes.
        let impostor = BackupExecutor::with_keyring(keyring_with("k1", 2));
        assert!(impostor.read_backup(&policy, &artefact).is_err());

        // A sealed artefact read under a policy without encryption.
        let plain_policy = make_policy(dir.path());
        assert!(executor.read_backup(&plain_policy, &artefact).is_err());

        // Tampering with the ciphertext.
        let mut bytes = std::fs::read(&artefact).expect("read artefact");
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        std::fs::write(&artefact, &bytes).expect("write tampered artefact");
        assert!(executor.read_backup(&policy, &artefact).is_err());

        // A plaintext file where an encrypted artefact is required.
        std::fs::write(&artefact, b"secret payload").expect("write plaintext");
        assert!(executor.read_backup(&policy, &artefact).is_err());
    }
}
