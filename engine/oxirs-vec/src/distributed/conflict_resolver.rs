//! Conflict resolution for cross-DC vector index synchronisation.
//!
//! When two datacenters each accept writes during a network partition they can
//! accumulate divergent versions of the same index entry.  This module provides
//! strategies for reconciling those versions when connectivity is restored.
//!
//! # Design
//!
//! - `IndexVersion` carries a version number, timestamp and the actual vector
//!   data so the resolver can make a purely logical decision without touching
//!   the storage layer.
//! - `ConflictResolver` is stateless; call `resolve` any number of times.
//! - `Resolution` describes what the caller should do with the two versions.
//!
//! # Pure Rust Policy
//!
//! No CUDA runtime calls or FFI.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================
// IndexVersion
// ============================================================

/// A versioned snapshot of a single vector entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexVersion {
    /// The unique identifier of the vector
    pub vector_id: String,
    /// Monotonically increasing version number (e.g. Raft log index or DC seq)
    pub version: u64,
    /// Wall-clock timestamp when this version was written (Unix milliseconds)
    pub timestamp_ms: u64,
    /// The vector data
    pub vector: Vec<f32>,
    /// Associated metadata key-value pairs
    pub metadata: HashMap<String, String>,
    /// Originating datacenter identifier
    pub source_dc: String,
}

impl IndexVersion {
    /// Create a new version entry.
    pub fn new(
        vector_id: impl Into<String>,
        version: u64,
        timestamp_ms: u64,
        vector: Vec<f32>,
        source_dc: impl Into<String>,
    ) -> Self {
        Self {
            vector_id: vector_id.into(),
            version,
            timestamp_ms,
            vector,
            metadata: HashMap::new(),
            source_dc: source_dc.into(),
        }
    }

    /// Attach metadata to this version (builder-style).
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

// ============================================================
// MergedIndex
// ============================================================

/// A merged index entry produced when `ConflictPolicy::MergeUnion` is applied.
///
/// The vector data is taken from the higher-version side; metadata is a union
/// of both sides (local keys win on collision).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MergedIndex {
    /// The vector entry that will be written to the index
    pub version: IndexVersion,
    /// Which DC's vector data was chosen
    pub vector_source: String,
}

// ============================================================
// ConflictPolicy
// ============================================================

/// Strategy for resolving a write conflict between two index versions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictPolicy {
    /// The version with the later wall-clock timestamp wins.
    ///
    /// Ties are broken in favour of the remote (incoming) version.
    LastWriteWins,
    /// The version with the higher monotonic version number wins.
    ///
    /// Ties are broken in favour of the local version.
    HighestVersionWins,
    /// Take the vector from the higher-version side and merge all metadata.
    ///
    /// Local metadata keys take precedence on collision.
    MergeUnion,
    /// Surface the conflict to the caller for application-level handling.
    Manual,
}

// ============================================================
// Resolution
// ============================================================

/// The outcome of a conflict resolution operation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Resolution {
    /// Keep the local version unchanged.
    UseLocal,
    /// Replace local with the remote version.
    UseRemote,
    /// Replace local with the merged version.
    Merge(MergedIndex),
    /// The policy is `Manual`; the caller must decide.
    RequiresManual,
}

// ============================================================
// ConflictResolver
// ============================================================

/// Stateless conflict resolver.
///
/// # Example
/// ```
/// use oxirs_vec::distributed::{ConflictResolver, ConflictPolicy, IndexVersion, Resolution};
///
/// let resolver = ConflictResolver;
/// let local = IndexVersion::new("v1", 10, 1_000, vec![1.0, 0.0], "dc-a");
/// let remote = IndexVersion::new("v1", 20, 2_000, vec![0.0, 1.0], "dc-b");
///
/// let outcome = resolver.resolve(&local, &remote, &ConflictPolicy::HighestVersionWins);
/// assert_eq!(outcome, Resolution::UseRemote);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ConflictResolver;

impl ConflictResolver {
    /// Resolve a conflict between a `local` and `remote` version of the same
    /// vector entry according to `policy`.
    pub fn resolve(
        &self,
        local: &IndexVersion,
        remote: &IndexVersion,
        policy: &ConflictPolicy,
    ) -> Resolution {
        match policy {
            ConflictPolicy::LastWriteWins => {
                if remote.timestamp_ms > local.timestamp_ms {
                    Resolution::UseRemote
                } else if remote.timestamp_ms < local.timestamp_ms {
                    Resolution::UseLocal
                } else {
                    // Tie: favour remote (incoming wins)
                    Resolution::UseRemote
                }
            }
            ConflictPolicy::HighestVersionWins => {
                if remote.version > local.version {
                    Resolution::UseRemote
                } else if remote.version < local.version {
                    Resolution::UseLocal
                } else {
                    // Tie: favour local (stable)
                    Resolution::UseLocal
                }
            }
            ConflictPolicy::MergeUnion => {
                // Higher version provides the vector; metadata is unioned
                let (winner, loser) = if remote.version >= local.version {
                    (remote, local)
                } else {
                    (local, remote)
                };

                let mut merged_meta = loser.metadata.clone();
                // Local (winner here) keys win on collision
                for (k, v) in &winner.metadata {
                    merged_meta.insert(k.clone(), v.clone());
                }

                let merged_version = IndexVersion {
                    vector_id: local.vector_id.clone(),
                    version: winner.version,
                    timestamp_ms: winner.timestamp_ms,
                    vector: winner.vector.clone(),
                    metadata: merged_meta,
                    source_dc: winner.source_dc.clone(),
                };

                Resolution::Merge(MergedIndex {
                    version: merged_version,
                    vector_source: winner.source_dc.clone(),
                })
            }
            ConflictPolicy::Manual => Resolution::RequiresManual,
        }
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_version(version: u64, timestamp_ms: u64, dc: &str) -> IndexVersion {
        IndexVersion::new(
            "vec-1",
            version,
            timestamp_ms,
            vec![version as f32, 0.0],
            dc,
        )
    }

    // ---- LastWriteWins ----

    #[test]
    fn test_lww_remote_newer() {
        let r = ConflictResolver;
        let local = make_version(1, 1000, "dc-a");
        let remote = make_version(2, 2000, "dc-b");
        assert_eq!(
            r.resolve(&local, &remote, &ConflictPolicy::LastWriteWins),
            Resolution::UseRemote
        );
    }

    #[test]
    fn test_lww_local_newer() {
        let r = ConflictResolver;
        let local = make_version(2, 2000, "dc-a");
        let remote = make_version(1, 1000, "dc-b");
        assert_eq!(
            r.resolve(&local, &remote, &ConflictPolicy::LastWriteWins),
            Resolution::UseLocal
        );
    }

    #[test]
    fn test_lww_tie_prefers_remote() {
        let r = ConflictResolver;
        let local = make_version(1, 1000, "dc-a");
        let remote = make_version(2, 1000, "dc-b"); // same timestamp
        assert_eq!(
            r.resolve(&local, &remote, &ConflictPolicy::LastWriteWins),
            Resolution::UseRemote
        );
    }

    // ---- HighestVersionWins ----

    #[test]
    fn test_hvw_remote_higher() {
        let r = ConflictResolver;
        let local = make_version(5, 1000, "dc-a");
        let remote = make_version(10, 500, "dc-b"); // older timestamp but higher version
        assert_eq!(
            r.resolve(&local, &remote, &ConflictPolicy::HighestVersionWins),
            Resolution::UseRemote
        );
    }

    #[test]
    fn test_hvw_local_higher() {
        let r = ConflictResolver;
        let local = make_version(10, 1000, "dc-a");
        let remote = make_version(5, 2000, "dc-b");
        assert_eq!(
            r.resolve(&local, &remote, &ConflictPolicy::HighestVersionWins),
            Resolution::UseLocal
        );
    }

    #[test]
    fn test_hvw_tie_prefers_local() {
        let r = ConflictResolver;
        let local = make_version(7, 1000, "dc-a");
        let remote = make_version(7, 1000, "dc-b"); // exact tie
        assert_eq!(
            r.resolve(&local, &remote, &ConflictPolicy::HighestVersionWins),
            Resolution::UseLocal
        );
    }

    // ---- MergeUnion ----

    #[test]
    fn test_merge_union_higher_version_provides_vector() {
        let r = ConflictResolver;
        let local = IndexVersion::new("vec-1", 5, 1000, vec![1.0, 2.0], "dc-a");
        let remote = IndexVersion::new("vec-1", 10, 2000, vec![3.0, 4.0], "dc-b");

        if let Resolution::Merge(merged) = r.resolve(&local, &remote, &ConflictPolicy::MergeUnion) {
            assert_eq!(merged.version.vector, vec![3.0, 4.0]);
            assert_eq!(merged.vector_source, "dc-b");
        } else {
            panic!("Expected Merge resolution");
        }
    }

    #[test]
    fn test_merge_union_metadata_union() {
        let r = ConflictResolver;
        let mut local = IndexVersion::new("vec-1", 5, 1000, vec![1.0], "dc-a");
        local.metadata.insert("key_a".into(), "val_a".into());
        local.metadata.insert("shared".into(), "local_val".into());

        let mut remote = IndexVersion::new("vec-1", 10, 2000, vec![2.0], "dc-b");
        remote.metadata.insert("key_b".into(), "val_b".into());
        remote.metadata.insert("shared".into(), "remote_val".into());

        if let Resolution::Merge(merged) = r.resolve(&local, &remote, &ConflictPolicy::MergeUnion) {
            // Both keys present
            assert!(merged.version.metadata.contains_key("key_a"));
            assert!(merged.version.metadata.contains_key("key_b"));
            // Winner (remote, higher version) overwrites shared key
            assert_eq!(merged.version.metadata["shared"], "remote_val");
        } else {
            panic!("Expected Merge resolution");
        }
    }

    #[test]
    fn test_merge_union_equal_versions_picks_remote() {
        let r = ConflictResolver;
        let local = IndexVersion::new("vec-1", 5, 1000, vec![1.0], "dc-a");
        let remote = IndexVersion::new("vec-1", 5, 2000, vec![2.0], "dc-b");

        if let Resolution::Merge(merged) = r.resolve(&local, &remote, &ConflictPolicy::MergeUnion) {
            // Equal versions: remote >= local so remote vector is chosen
            assert_eq!(merged.version.vector, vec![2.0]);
        } else {
            panic!("Expected Merge resolution");
        }
    }

    // ---- Manual ----

    #[test]
    fn test_manual_policy_requires_manual() {
        let r = ConflictResolver;
        let local = make_version(1, 1000, "dc-a");
        let remote = make_version(2, 2000, "dc-b");
        assert_eq!(
            r.resolve(&local, &remote, &ConflictPolicy::Manual),
            Resolution::RequiresManual
        );
    }

    // ---- IndexVersion helpers ----

    #[test]
    fn test_index_version_with_metadata() {
        let v =
            IndexVersion::new("v1", 1, 1000, vec![0.0], "dc-a").with_metadata("tag", "important");
        assert_eq!(v.metadata["tag"], "important");
    }

    #[test]
    fn test_resolution_is_clone() {
        let r = Resolution::UseLocal;
        let _r2 = r.clone();
    }

    #[test]
    fn test_merged_index_carries_correct_version_number() {
        let r = ConflictResolver;
        let local = IndexVersion::new("v1", 3, 100, vec![1.0], "dc-a");
        let remote = IndexVersion::new("v1", 8, 200, vec![8.0], "dc-b");

        if let Resolution::Merge(merged) = r.resolve(&local, &remote, &ConflictPolicy::MergeUnion) {
            assert_eq!(merged.version.version, 8);
        } else {
            panic!("Expected merge");
        }
    }
}
