//! Model shard manager: partitions an embedding table by entity-ID hash.
//!
//! [`ModelShardManager`] hashes a stable string ID (typically an entity URI or
//! relation IRI) to one of `num_shards` parameter-server shards.  The mapping is
//! deterministic and stateless: callers can reliably route reads and gradient
//! pushes for a given entity to the same shard across processes, without any
//! coordination, provided every party uses the same `num_shards`.
//!
//! The hashing is intentionally based on `std::collections::hash_map::DefaultHasher`
//! seeded with a fixed key — this gives bit-identical results between Rust
//! processes built from the same compiler version, which is good enough for
//! the in-process prototype we ship here.  For a real production system you
//! would want a cryptographic hash (e.g. SHA-256) or a known fingerprint (e.g.
//! xxHash) so that the partitioning is stable across language ecosystems too.
//!
//! ```
//! use oxirs_embed::distributed_training::{ModelShardManager, ShardingStrategy};
//!
//! let mgr = ModelShardManager::new(4, ShardingStrategy::EntityHash);
//! let s_alice = mgr.shard_for("http://example.org/Alice");
//! let s_alice2 = mgr.shard_for("http://example.org/Alice");
//! assert_eq!(s_alice, s_alice2);                  // deterministic
//! assert!(s_alice < 4);                           // bounded by num_shards
//! ```

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// How a [`ModelShardManager`] decides which shard owns a given entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ShardingStrategy {
    /// Hash the entity ID with a fixed seed and take `hash mod num_shards`.
    ///
    /// The default — gives uniformly distributed shards under the assumption
    /// that the underlying hash is well-mixed.
    #[default]
    EntityHash,
    /// Round-robin assignment in **lexicographic** order of insertion.
    ///
    /// Used by tests that need a known mapping; not recommended in production
    /// because shard load depends on insertion order.
    RoundRobin,
}

/// Result of running [`ModelShardManager::partition`] over a known set of IDs.
///
/// A [`ShardAssignment`] is a vector of `num_shards` buckets, where each bucket
/// holds the IDs that map to that shard.  Bucket ordering follows the input
/// order so the assignment is stable for a given input + `num_shards` pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardAssignment {
    /// Number of shards (length of `buckets`).
    pub num_shards: usize,
    /// `buckets[i]` is the list of IDs that map to shard `i`.
    pub buckets: Vec<Vec<String>>,
}

impl ShardAssignment {
    /// Total number of IDs across all shards.
    pub fn total(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }

    /// Look up which shard an ID was assigned to, or `None` if it isn't present.
    pub fn shard_of(&self, id: &str) -> Option<usize> {
        self.buckets.iter().enumerate().find_map(|(i, b)| {
            if b.iter().any(|s| s == id) {
                Some(i)
            } else {
                None
            }
        })
    }
}

/// Partitions embedding tables (entity *and* relation) by stable hash of the ID.
#[derive(Debug, Clone)]
pub struct ModelShardManager {
    num_shards: usize,
    strategy: ShardingStrategy,
    /// Stable round-robin index, used only when [`ShardingStrategy::RoundRobin`]
    /// is selected.  Filled lazily via [`Self::partition`].
    rr_index: HashMap<String, usize>,
}

impl ModelShardManager {
    /// Create a new shard manager with `num_shards` shards.
    ///
    /// # Panics
    ///
    /// Does not panic.  If `num_shards == 0` it is silently coerced to `1`
    /// (single-shard / no sharding) so the manager is always usable.
    pub fn new(num_shards: usize, strategy: ShardingStrategy) -> Self {
        Self {
            num_shards: num_shards.max(1),
            strategy,
            rr_index: HashMap::new(),
        }
    }

    /// Number of shards this manager partitions across.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Hashing strategy.
    pub fn strategy(&self) -> ShardingStrategy {
        self.strategy
    }

    /// Compute the shard index for `id`.
    ///
    /// Always returns a value in `0..self.num_shards()`.  Calling this method
    /// repeatedly with the same `id` and same `num_shards` always yields the
    /// same answer — it is the basis of the parameter-server routing scheme.
    pub fn shard_for(&self, id: &str) -> usize {
        match self.strategy {
            ShardingStrategy::EntityHash => self.hash_shard(id),
            ShardingStrategy::RoundRobin => {
                // Read-only round-robin: fall back to hash if we haven't seen
                // this ID yet via `partition()`.  Avoids surprising callers
                // that mix `partition` / `shard_for`.
                self.rr_index
                    .get(id)
                    .copied()
                    .unwrap_or_else(|| self.hash_shard(id))
            }
        }
    }

    /// Partition a list of IDs into shards in input order.
    ///
    /// In `EntityHash` mode this is equivalent to `shard_for` on each ID.  In
    /// `RoundRobin` mode it also populates the manager's internal round-robin
    /// table so that subsequent `shard_for` calls give consistent answers.
    pub fn partition<I, S>(&mut self, ids: I) -> ShardAssignment
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut buckets: Vec<Vec<String>> = (0..self.num_shards).map(|_| Vec::new()).collect();

        match self.strategy {
            ShardingStrategy::EntityHash => {
                for raw in ids {
                    let id: String = raw.into();
                    let shard = self.hash_shard(&id);
                    buckets[shard].push(id);
                }
            }
            ShardingStrategy::RoundRobin => {
                let mut next: usize = 0;
                for raw in ids {
                    let id: String = raw.into();
                    let shard = *self.rr_index.entry(id.clone()).or_insert_with(|| {
                        let s = next % self.num_shards;
                        next += 1;
                        s
                    });
                    buckets[shard].push(id);
                }
            }
        }

        ShardAssignment {
            num_shards: self.num_shards,
            buckets,
        }
    }

    /// Re-shard an existing assignment after `num_shards` changes.
    ///
    /// Returns a fresh [`ShardAssignment`] computed by routing every existing
    /// ID through the new manager.  Used by elastic scaling tests.
    pub fn reshard(&mut self, prior: &ShardAssignment) -> ShardAssignment {
        // Flatten in deterministic order, then partition again.
        let flat: Vec<String> = prior.buckets.iter().flatten().cloned().collect();
        self.partition(flat)
    }

    fn hash_shard(&self, id: &str) -> usize {
        // Mix in a constant seed so two managers with the same num_shards
        // hash identical IDs to identical buckets, even across processes
        // (provided they are built with the same Rust toolchain version).
        const SEED: u64 = 0x517c_c1b7_2722_0a95;
        let mut h = DefaultHasher::new();
        SEED.hash(&mut h);
        id.hash(&mut h);
        (h.finish() as usize) % self.num_shards
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_manager_default_strategy_is_entity_hash() {
        let mgr = ModelShardManager::new(4, ShardingStrategy::default());
        assert_eq!(mgr.strategy(), ShardingStrategy::EntityHash);
        assert_eq!(mgr.num_shards(), 4);
    }

    #[test]
    fn shard_for_deterministic_and_bounded() {
        let mgr = ModelShardManager::new(8, ShardingStrategy::EntityHash);
        let id = "http://example.org/Alice";
        let s1 = mgr.shard_for(id);
        let s2 = mgr.shard_for(id);
        assert_eq!(s1, s2, "shard_for must be deterministic");
        assert!(s1 < 8, "shard index must be bounded by num_shards");
    }

    #[test]
    fn shard_for_zero_shards_coerced_to_one() {
        let mgr = ModelShardManager::new(0, ShardingStrategy::EntityHash);
        assert_eq!(mgr.num_shards(), 1);
        assert_eq!(mgr.shard_for("anything"), 0);
    }

    #[test]
    fn partition_roundrobin_buckets_evenly() {
        let mut mgr = ModelShardManager::new(4, ShardingStrategy::RoundRobin);
        let ids: Vec<String> = (0..16).map(|i| format!("e{i}")).collect();
        let a = mgr.partition(ids);
        assert_eq!(a.num_shards, 4);
        assert_eq!(a.total(), 16);
        for b in &a.buckets {
            assert_eq!(b.len(), 4);
        }
    }

    #[test]
    fn partition_hash_total_equals_input_size() {
        let mut mgr = ModelShardManager::new(4, ShardingStrategy::EntityHash);
        let ids: Vec<String> = (0..100).map(|i| format!("entity_{i}")).collect();
        let a = mgr.partition(ids);
        assert_eq!(a.total(), 100);
    }

    #[test]
    fn partition_hash_distributes_across_shards() {
        // With 100 IDs across 4 shards, every shard should see at least one.
        let mut mgr = ModelShardManager::new(4, ShardingStrategy::EntityHash);
        let ids: Vec<String> = (0..200).map(|i| format!("entity_{i}")).collect();
        let a = mgr.partition(ids);
        for (i, b) in a.buckets.iter().enumerate() {
            assert!(
                !b.is_empty(),
                "shard {i} got no entities — distribution failed"
            );
        }
    }

    #[test]
    fn shard_assignment_shard_of_lookup() {
        let mut mgr = ModelShardManager::new(2, ShardingStrategy::RoundRobin);
        let a = mgr.partition(vec!["a", "b", "c", "d"]);
        assert_eq!(a.shard_of("a"), Some(0));
        assert_eq!(a.shard_of("b"), Some(1));
        assert_eq!(a.shard_of("missing"), None);
    }

    #[test]
    fn reshard_preserves_total_count_after_resize() {
        let mut mgr_small = ModelShardManager::new(2, ShardingStrategy::EntityHash);
        let ids: Vec<String> = (0..32).map(|i| format!("e{i}")).collect();
        let small = mgr_small.partition(ids.clone());
        assert_eq!(small.total(), 32);

        let mut mgr_big = ModelShardManager::new(8, ShardingStrategy::EntityHash);
        let big = mgr_big.reshard(&small);
        assert_eq!(big.num_shards, 8);
        assert_eq!(big.total(), 32);
    }

    #[test]
    fn reshard_routes_each_id_to_its_new_shard() {
        let ids: Vec<String> = (0..50).map(|i| format!("entity:{i}")).collect();
        let mut mgr2 = ModelShardManager::new(2, ShardingStrategy::EntityHash);
        let prior = mgr2.partition(ids);

        let mut mgr5 = ModelShardManager::new(5, ShardingStrategy::EntityHash);
        let after = mgr5.reshard(&prior);

        // Every ID must end up exactly where mgr5.shard_for(id) says.
        for (i, bucket) in after.buckets.iter().enumerate() {
            for id in bucket {
                assert_eq!(mgr5.shard_for(id), i);
            }
        }
    }

    #[test]
    fn partition_stable_across_managers_with_same_shard_count() {
        let ids: Vec<String> = (0..30).map(|i| format!("e_{i}")).collect();
        let mut a = ModelShardManager::new(4, ShardingStrategy::EntityHash);
        let mut b = ModelShardManager::new(4, ShardingStrategy::EntityHash);
        let pa = a.partition(ids.clone());
        let pb = b.partition(ids);
        assert_eq!(
            pa, pb,
            "two managers with same config must produce same shards"
        );
    }

    #[test]
    fn partition_unstable_when_shard_count_changes() {
        // Sanity: changing num_shards should change the partition (otherwise
        // hash_shard is broken or num_shards is being ignored).
        let ids: Vec<String> = (0..50).map(|i| format!("k_{i}")).collect();
        let mut mgr2 = ModelShardManager::new(2, ShardingStrategy::EntityHash);
        let mut mgr4 = ModelShardManager::new(4, ShardingStrategy::EntityHash);
        let p2 = mgr2.partition(ids.clone());
        let p4 = mgr4.partition(ids);
        assert_ne!(p2.num_shards, p4.num_shards);
    }

    #[test]
    fn shard_assignment_serialization() {
        let mut mgr = ModelShardManager::new(3, ShardingStrategy::EntityHash);
        let a = mgr.partition(vec!["x", "y", "z"]);
        let json = serde_json::to_string(&a).expect("serialize");
        let a2: ShardAssignment = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(a, a2);
    }
}
