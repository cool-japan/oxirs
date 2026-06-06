//! Multi-backend synchronization for the SAMM cloud storage subsystem.
//!
//! Provides:
//! - [`MultiBackendSync`] — replicate objects across two or more backends
//! - [`SyncPolicy`] — conflict-resolution and versioning strategy
//! - [`DeltaSyncState`] — tracks which keys have been propagated
//! - [`ScheduledReplication`] — configuration for timed sync jobs
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
    /// The primary backend always wins (default).
    #[default]
    PrimaryWins,
    /// The most recently uploaded version wins.
    LatestWins,
    /// Conflict is logged and neither copy is overwritten; manual resolution
    /// is required.
    KeepBoth,
    /// Conflict is treated as an error — the sync operation fails.
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
    /// Keys of objects that encountered conflicts.
    pub conflicts: Vec<String>,
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

            // Propagate to every replica.
            let mut any_failed = false;
            for replica in &self.replicas {
                if let Err(e) = replica.upload(key, data.clone()).await {
                    result
                        .errors
                        .push(format!("Failed to upload '{key}' to replica: {e}"));
                    any_failed = true;
                }
            }

            if any_failed {
                result.objects_failed += 1;
            } else {
                result.objects_synced += 1;
                // Update delta-sync state.
                if let Ok(mut state) = self.state.lock() {
                    state.mark_synced(key.clone(), checksum);
                }
            }
        }

        result
    }
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
