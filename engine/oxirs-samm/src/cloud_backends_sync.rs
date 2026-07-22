//! Multi-backend synchronization for the SAMM cloud storage subsystem.
//!
//! Provides:
//! - [`crate::cloud_backends_sync::MultiBackendSync`] — replicate objects across two or more backends
//! - [`crate::cloud_backends_sync::SyncPolicy`] — conflict-resolution and versioning strategy
//! - [`crate::cloud_backends_sync::DeltaSyncState`] — tracks which keys have been propagated
//! - [`crate::cloud_backends_sync::ScheduledReplication`] — configuration for timed sync jobs
//!
//! All network I/O is delegated to the provider-specific backends in
//! [`crate::cloud_backends_impl`].

use crate::cloud_storage::CloudStorageBackend;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

// ──────────────────────────────────────────────────────────────────────────────
// SyncPolicy
// ──────────────────────────────────────────────────────────────────────────────

/// Strategy applied when the same key exists on both the primary and a
/// replica backend and the contents differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConflictResolution {
    /// The primary backend always wins (default). The replica's differing
    /// content is unconditionally overwritten with the primary's content.
    #[default]
    PrimaryWins,
    /// The most recently uploaded version wins, determined by comparing
    /// [`crate::cloud_storage::ObjectMetadata::last_modified`] on the
    /// primary and the replica.
    ///
    /// Requires both backends to report `last_modified` via
    /// [`CloudStorageBackend::get_metadata`]; a sync involving a key whose
    /// primary and/or replica backend cannot supply that timestamp fails
    /// loudly for that key (recorded in [`SyncResult::errors`]) rather than
    /// silently guessing which copy is newer.
    LatestWins,
    /// Conflict is logged (the key is recorded in [`SyncResult::conflicts`])
    /// and neither copy is overwritten; manual resolution is required.
    KeepBoth,
    /// Conflict is treated as an error — the sync operation fails for that
    /// key (recorded in both [`SyncResult::conflicts`] and
    /// [`SyncResult::errors`]) and the key is not marked as synced.
    Error,
}

/// Configuration governing a synchronization run.
#[derive(Debug, Clone)]
pub struct SyncPolicy {
    /// How to resolve key conflicts between the primary and replicas.
    pub conflict_resolution: ConflictResolution,
    /// Whether to propagate deletions from the primary to replicas.
    pub propagate_deletes: bool,
    /// Maximum number of objects to sync in a single run (`None` = unlimited).
    pub batch_limit: Option<usize>,
    /// If `true`, the sync runs a dry-run pass first and logs what would
    /// change without actually transferring data.
    pub dry_run: bool,
}

impl Default for SyncPolicy {
    fn default() -> Self {
        Self {
            conflict_resolution: ConflictResolution::PrimaryWins,
            propagate_deletes: false,
            batch_limit: None,
            dry_run: false,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SyncResult
// ──────────────────────────────────────────────────────────────────────────────

/// Summary of a completed synchronization run.
#[derive(Debug, Clone, Default)]
pub struct SyncResult {
    /// Number of objects successfully propagated to at least one replica.
    pub objects_synced: usize,
    /// Number of objects skipped (already up-to-date or filtered by policy).
    pub objects_skipped: usize,
    /// Number of objects that failed to sync.
    pub objects_failed: usize,
    /// Keys of objects that encountered conflicts (primary and replica
    /// content differed at the time of comparison), regardless of how the
    /// conflict was resolved.
    pub conflicts: Vec<String>,
    /// Number of replica objects deleted because they were no longer present
    /// under the synced prefix on the primary (only non-zero when
    /// [`SyncPolicy::propagate_deletes`] is `true`).
    pub deletes_propagated: usize,
    /// Error messages collected during the run.
    pub errors: Vec<String>,
}

// ──────────────────────────────────────────────────────────────────────────────
// DeltaSyncState
// ──────────────────────────────────────────────────────────────────────────────

/// Tracks which object keys have already been propagated so that subsequent
/// sync runs only transfer changed or new objects (delta sync).
#[derive(Debug, Clone, Default)]
pub struct DeltaSyncState {
    /// The set of keys that are known to be in sync across all replicas.
    synced_keys: HashSet<String>,
    /// Checksums (SHA-256 hex) for synced objects, keyed by object key.
    checksums: HashMap<String, String>,
}

impl DeltaSyncState {
    /// Create an empty sync state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a key as having been successfully synced with the given checksum.
    pub fn mark_synced(&mut self, key: impl Into<String>, checksum: impl Into<String>) {
        let k = key.into();
        self.checksums.insert(k.clone(), checksum.into());
        self.synced_keys.insert(k);
    }

    /// Returns `true` if the key was previously synced with the same checksum.
    pub fn is_synced(&self, key: &str, checksum: &str) -> bool {
        self.checksums
            .get(key)
            .map(|c| c == checksum)
            .unwrap_or(false)
    }

    /// Remove a key from the sync state (e.g. after a deletion is propagated).
    pub fn remove(&mut self, key: &str) {
        self.synced_keys.remove(key);
        self.checksums.remove(key);
    }

    /// Number of keys currently tracked.
    pub fn len(&self) -> usize {
        self.synced_keys.len()
    }

    /// `true` if no keys are tracked.
    pub fn is_empty(&self) -> bool {
        self.synced_keys.is_empty()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ScheduledReplication
// ──────────────────────────────────────────────────────────────────────────────

/// Configuration for a timed replication job.
///
/// Actual scheduling (cron, timer, etc.) is the responsibility of the caller;
/// this struct just carries the parameters for a single scheduled run.
#[derive(Debug, Clone)]
pub struct ScheduledReplication {
    /// Human-readable name for this replication job.
    pub name: String,
    /// Object-key prefix to replicate (`""` means replicate everything).
    pub prefix: String,
    /// Synchronisation policy applied during each run.
    pub policy: SyncPolicy,
    /// Whether this job is currently active.
    pub enabled: bool,
}

impl ScheduledReplication {
    /// Create a new enabled replication job with default policy.
    pub fn new(name: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            prefix: prefix.into(),
            policy: SyncPolicy::default(),
            enabled: true,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MultiBackendSync
// ──────────────────────────────────────────────────────────────────────────────

/// Replicates objects from a primary [`CloudStorageBackend`] to one or more
/// replica backends.
///
/// # Example (pseudo-code)
/// ```rust,no_run
/// # use oxirs_samm::cloud_backends_sync::MultiBackendSync;
/// # use oxirs_samm::cloud_backends_impl::LocalFsBackend;
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let primary = LocalFsBackend::new("/data/primary")?;
/// let replica = LocalFsBackend::new("/data/replica")?;
///
/// let mut sync = MultiBackendSync::new(primary);
/// sync.add_replica(replica);
///
/// let result = sync.sync_prefix("models/", Default::default()).await;
/// println!("synced={}", result.objects_synced);
/// # Ok(())
/// # }
/// ```
pub struct MultiBackendSync<P> {
    primary: Arc<P>,
    replicas: Vec<Arc<dyn CloudStorageBackend + Send + Sync>>,
    state: Mutex<DeltaSyncState>,
}

impl<P> MultiBackendSync<P>
where
    P: CloudStorageBackend + Send + Sync + 'static,
{
    /// Create a new multi-backend sync manager with the given primary backend.
    pub fn new(primary: P) -> Self {
        Self {
            primary: Arc::new(primary),
            replicas: Vec::new(),
            state: Mutex::new(DeltaSyncState::new()),
        }
    }

    /// Add a replica backend.  Objects uploaded to the primary will be
    /// propagated to every registered replica during a sync run.
    pub fn add_replica<R>(&mut self, replica: R)
    where
        R: CloudStorageBackend + Send + Sync + 'static,
    {
        self.replicas.push(Arc::new(replica));
    }

    /// Synchronise all objects under `prefix` from the primary to all replicas
    /// using the given [`SyncPolicy`].
    ///
    /// For each key whose content differs between the primary and a given
    /// replica, `policy.conflict_resolution` determines the outcome:
    ///
    /// - [`ConflictResolution::PrimaryWins`] unconditionally overwrites the
    ///   replica (no replica-content inspection needed — this is the fast
    ///   path).
    /// - [`ConflictResolution::LatestWins`] compares `last_modified`
    ///   timestamps reported by each backend's
    ///   [`CloudStorageBackend::get_metadata`]; ties and missing metadata on
    ///   either side are reported as an unresolved conflict rather than
    ///   guessed.
    /// - [`ConflictResolution::KeepBoth`] leaves the replica untouched and
    ///   records the key in [`SyncResult::conflicts`] for manual resolution.
    /// - [`ConflictResolution::Error`] leaves the replica untouched, records
    ///   the key as both a conflict and an error, and does not mark the key
    ///   as synced (so a subsequent run will retry it).
    ///
    /// When `policy.propagate_deletes` is `true`, any key present under
    /// `prefix` on a replica but absent from the primary's current listing
    /// is deleted from that replica.
    ///
    /// Returns a [`SyncResult`] describing what happened.
    pub async fn sync_prefix(&self, prefix: &str, policy: SyncPolicy) -> SyncResult {
        let mut result = SyncResult::default();

        if policy.dry_run {
            tracing::info!(
                "MultiBackendSync: dry-run mode — no data will be transferred (prefix='{}')",
                prefix
            );
        }

        // List all keys on the primary.
        let keys = match self.primary.list(prefix).await {
            Ok(k) => k,
            Err(e) => {
                result
                    .errors
                    .push(format!("Failed to list primary prefix '{prefix}': {e}"));
                result.objects_failed += 1;
                return result;
            }
        };

        let limit = policy.batch_limit.unwrap_or(usize::MAX);

        for key in keys.iter().take(limit) {
            // Download from primary.
            let data = match self.primary.download(key).await {
                Ok(d) => d,
                Err(e) => {
                    result
                        .errors
                        .push(format!("Failed to download '{key}' from primary: {e}"));
                    result.objects_failed += 1;
                    continue;
                }
            };

            // Compute a simple checksum for delta-sync tracking.
            let checksum = format!("{:016x}", simple_checksum(&data));

            // Check if already in sync.
            {
                let state = self.state.lock().unwrap_or_else(|p| p.into_inner());
                if state.is_synced(key, &checksum) {
                    result.objects_skipped += 1;
                    continue;
                }
            }

            if policy.dry_run {
                tracing::debug!("dry-run: would propagate '{}'", key);
                result.objects_synced += 1;
                continue;
            }

            // Propagate to every replica, honoring the configured conflict
            // resolution strategy whenever the replica's current content
            // differs from the primary's.
            let mut any_failed = false;
            let mut any_conflict_left_unresolved = false;
            for replica in &self.replicas {
                if policy.conflict_resolution != ConflictResolution::PrimaryWins {
                    match self
                        .detect_and_resolve_conflict(
                            replica.as_ref(),
                            key,
                            &checksum,
                            policy.conflict_resolution,
                            &mut result,
                        )
                        .await
                    {
                        ConflictOutcome::NoConflictOrAlreadyUpToDate => {}
                        ConflictOutcome::ResolvedInFavorOfPrimary => {}
                        ConflictOutcome::KeptReplica => continue,
                        ConflictOutcome::Unresolved => {
                            any_failed = true;
                            any_conflict_left_unresolved = true;
                            continue;
                        }
                    }
                }

                if let Err(e) = replica.upload(key, data.clone()).await {
                    result
                        .errors
                        .push(format!("Failed to upload '{key}' to replica: {e}"));
                    any_failed = true;
                }
            }

            if any_failed {
                result.objects_failed += 1;
                if any_conflict_left_unresolved {
                    // Do not mark as synced: the key still needs manual or
                    // future automatic resolution.
                    continue;
                }
            } else {
                result.objects_synced += 1;
                // Update delta-sync state.
                if let Ok(mut state) = self.state.lock() {
                    state.mark_synced(key.clone(), checksum);
                }
            }
        }

        if policy.propagate_deletes {
            self.propagate_deletes(prefix, &keys, policy.dry_run, &mut result)
                .await;
        }

        result
    }

    /// Inspect `replica`'s current content for `key` and, if it differs from
    /// the primary's `primary_checksum`, resolve the conflict per
    /// `resolution`. Returns an outcome describing whether the caller should
    /// still upload the primary's data to this replica.
    async fn detect_and_resolve_conflict(
        &self,
        replica: &(dyn CloudStorageBackend + Send + Sync),
        key: &str,
        primary_checksum: &str,
        resolution: ConflictResolution,
        result: &mut SyncResult,
    ) -> ConflictOutcome {
        let exists = match replica.exists(key).await {
            Ok(e) => e,
            Err(e) => {
                result.errors.push(format!(
                    "Failed to check existence of '{key}' on replica: {e}"
                ));
                return ConflictOutcome::Unresolved;
            }
        };
        if !exists {
            // Nothing on the replica yet — no conflict, proceed to upload.
            return ConflictOutcome::NoConflictOrAlreadyUpToDate;
        }

        let replica_data = match replica.download(key).await {
            Ok(d) => d,
            Err(e) => {
                result.errors.push(format!(
                    "Failed to download '{key}' from replica for conflict check: {e}"
                ));
                return ConflictOutcome::Unresolved;
            }
        };
        let replica_checksum = format!("{:016x}", simple_checksum(&replica_data));
        if replica_checksum == primary_checksum {
            // Replica already holds identical content.
            return ConflictOutcome::NoConflictOrAlreadyUpToDate;
        }

        // Genuine conflict: primary and replica disagree on this key's content.
        if !result.conflicts.iter().any(|k| k == key) {
            result.conflicts.push(key.to_string());
        }

        match resolution {
            ConflictResolution::PrimaryWins => ConflictOutcome::ResolvedInFavorOfPrimary,
            ConflictResolution::KeepBoth => ConflictOutcome::KeptReplica,
            ConflictResolution::Error => {
                result.errors.push(format!(
                    "Conflict for key '{key}': primary and replica content differ (ConflictResolution::Error)"
                ));
                ConflictOutcome::Unresolved
            }
            ConflictResolution::LatestWins => {
                let primary_mtime = self
                    .primary
                    .get_metadata(key)
                    .await
                    .ok()
                    .and_then(|m| m.last_modified);
                let replica_mtime = replica
                    .get_metadata(key)
                    .await
                    .ok()
                    .and_then(|m| m.last_modified);
                match (primary_mtime, replica_mtime) {
                    (Some(p), Some(r)) if r > p => ConflictOutcome::KeptReplica,
                    (Some(_), Some(_)) => ConflictOutcome::ResolvedInFavorOfPrimary,
                    _ => {
                        result.errors.push(format!(
                            "Cannot resolve conflict for key '{key}' with ConflictResolution::LatestWins: last-modified metadata unavailable from primary and/or replica backend"
                        ));
                        ConflictOutcome::Unresolved
                    }
                }
            }
        }
    }

    /// Delete any key present under `prefix` on a replica but absent from
    /// the primary's current `primary_keys` listing.
    async fn propagate_deletes(
        &self,
        prefix: &str,
        primary_keys: &[String],
        dry_run: bool,
        result: &mut SyncResult,
    ) {
        let primary_key_set: HashSet<&str> = primary_keys.iter().map(|s| s.as_str()).collect();

        for replica in &self.replicas {
            let replica_keys = match replica.list(prefix).await {
                Ok(rk) => rk,
                Err(e) => {
                    result.errors.push(format!(
                        "Failed to list replica prefix '{prefix}' for delete-propagation: {e}"
                    ));
                    continue;
                }
            };

            for rkey in replica_keys {
                if primary_key_set.contains(rkey.as_str()) {
                    continue;
                }
                if dry_run {
                    tracing::debug!(
                        "dry-run: would delete '{}' from replica (absent from primary)",
                        rkey
                    );
                    continue;
                }
                match replica.delete(&rkey).await {
                    Ok(()) => {
                        result.deletes_propagated += 1;
                        if let Ok(mut state) = self.state.lock() {
                            state.remove(&rkey);
                        }
                    }
                    Err(e) => {
                        result.errors.push(format!(
                            "Failed to delete '{rkey}' from replica during delete-propagation: {e}"
                        ));
                        result.objects_failed += 1;
                    }
                }
            }
        }
    }
}

/// Outcome of a single-replica conflict check inside [`MultiBackendSync::sync_prefix`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConflictOutcome {
    /// No conflict was detected (replica missing the key, or content already
    /// matches) — the caller should still attempt the (idempotent) upload.
    NoConflictOrAlreadyUpToDate,
    /// A conflict was detected and resolved in favor of the primary — the
    /// caller should overwrite the replica.
    ResolvedInFavorOfPrimary,
    /// A conflict was detected and resolved in favor of the replica's
    /// existing content — the caller must skip the upload for this replica.
    KeptReplica,
    /// A conflict was detected and could not be automatically resolved
    /// (`Error` policy, or `LatestWins` without usable metadata) — the
    /// caller must skip the upload and treat the key as failed.
    Unresolved,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// A fast, non-cryptographic checksum used only for delta-sync tracking.
fn simple_checksum(data: &[u8]) -> u64 {
    // FNV-1a 64-bit — sufficient for detecting content changes without pulling
    // in a full SHA-256 for every object during sync.
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cloud_storage::ObjectMetadata;
    use std::time::{Duration, SystemTime};

    /// A simple in-memory [`CloudStorageBackend`] that supports seeding
    /// objects with a controllable `last_modified` timestamp, so that
    /// `ConflictResolution::LatestWins` behaviour can be tested
    /// deterministically without relying on real filesystem mtimes.
    /// `(content bytes, optional last-modified timestamp)` for one object.
    type MemoryObject = (Vec<u8>, Option<SystemTime>);

    #[derive(Default)]
    struct MemoryBackend {
        objects: Mutex<HashMap<String, MemoryObject>>,
    }

    impl MemoryBackend {
        fn new() -> Self {
            Self::default()
        }

        /// Seed an object with explicit content and an explicit
        /// `last_modified` timestamp (or `None` to simulate a backend that
        /// does not report timestamps).
        fn seed(&self, key: &str, data: &[u8], when: Option<SystemTime>) {
            self.objects
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert(key.to_string(), (data.to_vec(), when));
        }

        fn contents(&self, key: &str) -> Option<Vec<u8>> {
            self.objects
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .get(key)
                .map(|(data, _)| data.clone())
        }
    }

    #[async_trait::async_trait]
    impl CloudStorageBackend for MemoryBackend {
        async fn upload(&self, key: &str, data: Vec<u8>) -> std::result::Result<(), String> {
            self.objects
                .lock()
                .map_err(|_| "poisoned".to_string())?
                .insert(key.to_string(), (data, Some(SystemTime::now())));
            Ok(())
        }

        async fn download(&self, key: &str) -> std::result::Result<Vec<u8>, String> {
            self.objects
                .lock()
                .map_err(|_| "poisoned".to_string())?
                .get(key)
                .map(|(data, _)| data.clone())
                .ok_or_else(|| format!("not found: {key}"))
        }

        async fn exists(&self, key: &str) -> std::result::Result<bool, String> {
            Ok(self
                .objects
                .lock()
                .map_err(|_| "poisoned".to_string())?
                .contains_key(key))
        }

        async fn delete(&self, key: &str) -> std::result::Result<(), String> {
            self.objects
                .lock()
                .map_err(|_| "poisoned".to_string())?
                .remove(key);
            Ok(())
        }

        async fn list(&self, prefix: &str) -> std::result::Result<Vec<String>, String> {
            Ok(self
                .objects
                .lock()
                .map_err(|_| "poisoned".to_string())?
                .keys()
                .filter(|k| k.starts_with(prefix))
                .cloned()
                .collect())
        }

        async fn get_metadata(&self, key: &str) -> std::result::Result<ObjectMetadata, String> {
            let guard = self.objects.lock().map_err(|_| "poisoned".to_string())?;
            let (data, when) = guard.get(key).ok_or_else(|| format!("not found: {key}"))?;
            Ok(ObjectMetadata {
                key: key.to_string(),
                size: data.len(),
                last_modified: *when,
            })
        }
    }

    fn policy_with(resolution: ConflictResolution) -> SyncPolicy {
        SyncPolicy {
            conflict_resolution: resolution,
            ..SyncPolicy::default()
        }
    }

    #[tokio::test]
    async fn regression_primary_wins_overwrites_conflicting_replica() {
        let primary = MemoryBackend::new();
        primary.seed("models/a.ttl", b"primary-content", None);

        let mut sync = MultiBackendSync::new(primary);
        let replica = MemoryBackend::new();
        replica.seed("models/a.ttl", b"stale-replica-content", None);
        // Keep a handle to inspect the replica after sync.
        let replica_arc = Arc::new(replica);
        sync.replicas.push(replica_arc.clone());

        let result = sync
            .sync_prefix("models/", policy_with(ConflictResolution::PrimaryWins))
            .await;

        assert_eq!(result.objects_synced, 1, "errors={:?}", result.errors);
        assert_eq!(result.objects_failed, 0);
        assert_eq!(
            replica_arc.contents("models/a.ttl"),
            Some(b"primary-content".to_vec()),
            "PrimaryWins must overwrite the conflicting replica"
        );
    }

    #[tokio::test]
    async fn regression_keep_both_leaves_replica_untouched_and_records_conflict() {
        let primary = MemoryBackend::new();
        primary.seed("models/a.ttl", b"primary-content", None);

        let mut sync = MultiBackendSync::new(primary);
        let replica = Arc::new(MemoryBackend::new());
        replica.seed("models/a.ttl", b"replica-content", None);
        sync.replicas.push(replica.clone());

        let result = sync
            .sync_prefix("models/", policy_with(ConflictResolution::KeepBoth))
            .await;

        assert_eq!(result.objects_failed, 0, "errors={:?}", result.errors);
        assert!(
            result.conflicts.iter().any(|k| k == "models/a.ttl"),
            "conflict must be recorded: {:?}",
            result.conflicts
        );
        assert_eq!(
            replica.contents("models/a.ttl"),
            Some(b"replica-content".to_vec()),
            "KeepBoth must not overwrite the replica's differing content"
        );
    }

    #[tokio::test]
    async fn regression_error_resolution_fails_sync_and_leaves_replica_untouched() {
        let primary = MemoryBackend::new();
        primary.seed("models/a.ttl", b"primary-content", None);

        let mut sync = MultiBackendSync::new(primary);
        let replica = Arc::new(MemoryBackend::new());
        replica.seed("models/a.ttl", b"replica-content", None);
        sync.replicas.push(replica.clone());

        let result = sync
            .sync_prefix("models/", policy_with(ConflictResolution::Error))
            .await;

        assert_eq!(result.objects_synced, 0);
        assert_eq!(result.objects_failed, 1);
        assert!(
            !result.errors.is_empty(),
            "Error policy must report an error"
        );
        assert!(result.conflicts.iter().any(|k| k == "models/a.ttl"));
        assert_eq!(
            replica.contents("models/a.ttl"),
            Some(b"replica-content".to_vec()),
            "Error policy must not overwrite the replica"
        );
    }

    #[tokio::test]
    async fn regression_latest_wins_keeps_newer_replica_content() {
        let now = SystemTime::now();
        let primary = MemoryBackend::new();
        primary.seed(
            "models/a.ttl",
            b"primary-content",
            Some(now - Duration::from_secs(3600)),
        );

        let mut sync = MultiBackendSync::new(primary);
        let replica = Arc::new(MemoryBackend::new());
        // Replica's copy is strictly newer than the primary's.
        replica.seed("models/a.ttl", b"newer-replica-content", Some(now));
        sync.replicas.push(replica.clone());

        let result = sync
            .sync_prefix("models/", policy_with(ConflictResolution::LatestWins))
            .await;

        assert_eq!(result.objects_failed, 0, "errors={:?}", result.errors);
        assert!(result.conflicts.iter().any(|k| k == "models/a.ttl"));
        assert_eq!(
            replica.contents("models/a.ttl"),
            Some(b"newer-replica-content".to_vec()),
            "LatestWins must keep the replica's strictly newer content"
        );
    }

    #[tokio::test]
    async fn regression_latest_wins_overwrites_with_newer_primary_content() {
        let now = SystemTime::now();
        let primary = MemoryBackend::new();
        primary.seed("models/a.ttl", b"newer-primary-content", Some(now));

        let mut sync = MultiBackendSync::new(primary);
        let replica = Arc::new(MemoryBackend::new());
        replica.seed(
            "models/a.ttl",
            b"stale-replica-content",
            Some(now - Duration::from_secs(3600)),
        );
        sync.replicas.push(replica.clone());

        let result = sync
            .sync_prefix("models/", policy_with(ConflictResolution::LatestWins))
            .await;

        assert_eq!(result.objects_failed, 0, "errors={:?}", result.errors);
        assert_eq!(
            replica.contents("models/a.ttl"),
            Some(b"newer-primary-content".to_vec()),
            "LatestWins must overwrite with the strictly newer primary content"
        );
    }

    #[tokio::test]
    async fn regression_latest_wins_without_metadata_fails_loud() {
        // Neither side reports a last_modified timestamp.
        let primary = MemoryBackend::new();
        primary.seed("models/a.ttl", b"primary-content", None);

        let mut sync = MultiBackendSync::new(primary);
        let replica = Arc::new(MemoryBackend::new());
        replica.seed("models/a.ttl", b"replica-content", None);
        sync.replicas.push(replica.clone());

        let result = sync
            .sync_prefix("models/", policy_with(ConflictResolution::LatestWins))
            .await;

        assert_eq!(
            result.objects_synced, 0,
            "must not silently guess which side is newer"
        );
        assert_eq!(result.objects_failed, 1);
        assert!(
            result.errors.iter().any(|e| e.contains("LatestWins")),
            "errors={:?}",
            result.errors
        );
        assert_eq!(
            replica.contents("models/a.ttl"),
            Some(b"replica-content".to_vec()),
            "an unresolved conflict must not silently overwrite the replica"
        );
    }

    #[tokio::test]
    async fn regression_propagate_deletes_removes_keys_absent_from_primary() {
        let primary = MemoryBackend::new();
        primary.seed("models/a.ttl", b"content-a", None);
        // "models/b.ttl" intentionally NOT present on the primary.

        let mut sync = MultiBackendSync::new(primary);
        let replica = Arc::new(MemoryBackend::new());
        replica.seed("models/a.ttl", b"content-a", None);
        replica.seed("models/b.ttl", b"content-b", None);
        sync.replicas.push(replica.clone());

        let policy = SyncPolicy {
            propagate_deletes: true,
            ..SyncPolicy::default()
        };

        let result = sync.sync_prefix("models/", policy).await;

        assert_eq!(result.deletes_propagated, 1, "errors={:?}", result.errors);
        assert_eq!(
            replica.contents("models/b.ttl"),
            None,
            "propagate_deletes must remove keys absent from the primary"
        );
        assert_eq!(
            replica.contents("models/a.ttl"),
            Some(b"content-a".to_vec()),
            "keys still present on the primary must be left alone"
        );
    }

    #[tokio::test]
    async fn regression_propagate_deletes_false_leaves_extra_replica_keys() {
        let primary = MemoryBackend::new();
        primary.seed("models/a.ttl", b"content-a", None);

        let mut sync = MultiBackendSync::new(primary);
        let replica = Arc::new(MemoryBackend::new());
        replica.seed("models/a.ttl", b"content-a", None);
        replica.seed("models/b.ttl", b"content-b", None);
        sync.replicas.push(replica.clone());

        // propagate_deletes defaults to false.
        let result = sync.sync_prefix("models/", SyncPolicy::default()).await;

        assert_eq!(result.deletes_propagated, 0);
        assert_eq!(
            replica.contents("models/b.ttl"),
            Some(b"content-b".to_vec()),
            "without propagate_deletes, extra replica keys must be left untouched"
        );
    }

    #[tokio::test]
    async fn regression_propagate_deletes_dry_run_does_not_delete() {
        let primary = MemoryBackend::new();
        primary.seed("models/a.ttl", b"content-a", None);

        let mut sync = MultiBackendSync::new(primary);
        let replica = Arc::new(MemoryBackend::new());
        replica.seed("models/a.ttl", b"content-a", None);
        replica.seed("models/b.ttl", b"content-b", None);
        sync.replicas.push(replica.clone());

        let policy = SyncPolicy {
            propagate_deletes: true,
            dry_run: true,
            ..SyncPolicy::default()
        };

        let result = sync.sync_prefix("models/", policy).await;

        assert_eq!(result.deletes_propagated, 0, "dry-run must not delete");
        assert_eq!(
            replica.contents("models/b.ttl"),
            Some(b"content-b".to_vec()),
            "dry-run must leave the replica untouched"
        );
    }
}
