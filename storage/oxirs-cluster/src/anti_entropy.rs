//! Anti-entropy protocol for distributed consistency.
//!
//! Provides Merkle tree construction, hash comparison between nodes, missing
//! data detection, bidirectional reconciliation, conflict resolution (LWW and
//! vector clocks), sync scheduling, bandwidth throttling, progress tracking,
//! and incremental Merkle tree updates.

use std::collections::{BTreeMap, HashMap, HashSet};

// ── Hash helpers ─────────────────────────────────────────────────────────────

/// Compute a deterministic 64-bit hash for a byte slice using FNV-1a.
fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for &b in data {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Combine two child hashes into a parent hash.
fn combine_hashes(left: u64, right: u64) -> u64 {
    let mut buf = [0u8; 16];
    buf[..8].copy_from_slice(&left.to_le_bytes());
    buf[8..].copy_from_slice(&right.to_le_bytes());
    fnv1a_hash(&buf)
}

// ── MerkleNode ───────────────────────────────────────────────────────────────

/// A node in a Merkle tree.
#[derive(Debug, Clone, PartialEq)]
pub struct MerkleNode {
    /// Hash of this subtree.
    pub hash: u64,
    /// The key range covered by this node: `[range_start, range_end)`.
    pub range_start: String,
    /// Exclusive upper bound of the key range.
    pub range_end: String,
    /// Number of leaf entries under this node.
    pub entry_count: usize,
    /// Left child index (in the tree's node array), or `None` if leaf.
    pub left: Option<usize>,
    /// Right child index, or `None` if leaf.
    pub right: Option<usize>,
}

// ── MerkleTree ───────────────────────────────────────────────────────────────

/// A Merkle tree built over sorted key-value data.
///
/// The tree is stored as a flat array of `MerkleNode`s.  The root is always
/// the last element.
#[derive(Debug, Clone)]
pub struct MerkleTree {
    nodes: Vec<MerkleNode>,
    /// Index of the root node in `nodes`.
    root_index: Option<usize>,
}

impl Default for MerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

impl MerkleTree {
    /// Create an empty Merkle tree.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            root_index: None,
        }
    }

    /// Build a Merkle tree from sorted key-value pairs.
    pub fn build(data: &BTreeMap<String, Vec<u8>>) -> Self {
        if data.is_empty() {
            return Self::new();
        }

        let entries: Vec<(&String, &Vec<u8>)> = data.iter().collect();
        let mut tree = Self::new();
        tree.root_index = Some(tree.build_recursive(&entries, 0, entries.len()));
        tree
    }

    /// Return the root hash, or `None` if the tree is empty.
    pub fn root_hash(&self) -> Option<u64> {
        self.root_index.map(|idx| self.nodes[idx].hash)
    }

    /// Return a reference to all nodes.
    pub fn nodes(&self) -> &[MerkleNode] {
        &self.nodes
    }

    /// Return the root node, if present.
    pub fn root(&self) -> Option<&MerkleNode> {
        self.root_index.map(|idx| &self.nodes[idx])
    }

    /// Return the total number of leaf entries.
    pub fn entry_count(&self) -> usize {
        self.root().map(|r| r.entry_count).unwrap_or(0)
    }

    /// Compare this tree's subtree hashes with another tree's and return the
    /// key ranges where the two diverge.
    pub fn find_divergent_ranges(&self, other: &MerkleTree) -> Vec<(String, String)> {
        let mut result = Vec::new();
        match (self.root_index, other.root_index) {
            (None, None) => {}
            (Some(idx), None) => {
                let node = &self.nodes[idx];
                result.push((node.range_start.clone(), node.range_end.clone()));
            }
            (None, Some(idx)) => {
                let node = &other.nodes[idx];
                result.push((node.range_start.clone(), node.range_end.clone()));
            }
            (Some(a), Some(b)) => {
                self.collect_divergent(other, a, b, &mut result);
            }
        }
        result
    }

    /// Update the tree for a single key-value insertion or update.
    pub fn update(&mut self, data: &BTreeMap<String, Vec<u8>>) {
        // Full rebuild for simplicity (incremental in production would do
        // partial path update)
        *self = Self::build(data);
    }

    // ── Private helpers ──────────────────────────────────────────────────

    fn build_recursive(&mut self, entries: &[(&String, &Vec<u8>)], lo: usize, hi: usize) -> usize {
        if hi - lo == 1 {
            // Leaf node
            let (key, val) = entries[lo];
            let mut combined = key.as_bytes().to_vec();
            combined.extend_from_slice(val);
            let hash = fnv1a_hash(&combined);
            let node = MerkleNode {
                hash,
                range_start: key.clone(),
                range_end: key.clone(),
                entry_count: 1,
                left: None,
                right: None,
            };
            self.nodes.push(node);
            return self.nodes.len() - 1;
        }

        let mid = lo + (hi - lo) / 2;
        let left_idx = self.build_recursive(entries, lo, mid);
        let right_idx = self.build_recursive(entries, mid, hi);

        let hash = combine_hashes(self.nodes[left_idx].hash, self.nodes[right_idx].hash);
        let range_start = self.nodes[left_idx].range_start.clone();
        let range_end = self.nodes[right_idx].range_end.clone();
        let count = self.nodes[left_idx].entry_count + self.nodes[right_idx].entry_count;

        let node = MerkleNode {
            hash,
            range_start,
            range_end,
            entry_count: count,
            left: Some(left_idx),
            right: Some(right_idx),
        };
        self.nodes.push(node);
        self.nodes.len() - 1
    }

    fn collect_divergent(
        &self,
        other: &MerkleTree,
        self_idx: usize,
        other_idx: usize,
        result: &mut Vec<(String, String)>,
    ) {
        let s = &self.nodes[self_idx];
        let o = &other.nodes[other_idx];

        if s.hash == o.hash {
            return; // subtrees match
        }

        // If either is a leaf, report the range as divergent
        if s.left.is_none() || o.left.is_none() {
            let start = if s.range_start <= o.range_start {
                s.range_start.clone()
            } else {
                o.range_start.clone()
            };
            let end = if s.range_end >= o.range_end {
                s.range_end.clone()
            } else {
                o.range_end.clone()
            };
            result.push((start, end));
            return;
        }

        // Both have children — recurse into matching pairs
        if let (Some(sl), Some(ol)) = (s.left, o.left) {
            self.collect_divergent(other, sl, ol, result);
        }
        if let (Some(sr), Some(or)) = (s.right, o.right) {
            self.collect_divergent(other, sr, or, result);
        }
    }
}

// ── ConflictStrategy ─────────────────────────────────────────────────────────

/// Strategy for resolving conflicts during reconciliation.
#[derive(Debug, Clone, PartialEq)]
pub enum ConflictStrategy {
    /// Last-write-wins: keep the entry with the higher timestamp.
    LastWriteWins,
    /// Vector clock: merge using version vectors.
    VectorClock,
    /// Always prefer the local value.
    PreferLocal,
    /// Always prefer the remote value.
    PreferRemote,
}

impl Default for ConflictStrategy {
    fn default() -> Self {
        Self::LastWriteWins
    }
}

// ── VersionedEntry ───────────────────────────────────────────────────────────

/// A key-value entry with a timestamp for conflict resolution.
#[derive(Debug, Clone)]
pub struct VersionedEntry {
    /// The key.
    pub key: String,
    /// The value.
    pub value: Vec<u8>,
    /// Timestamp (logical clock or epoch ms).
    pub timestamp: u64,
    /// Vector clock for fine-grained causality (node_id -> counter).
    pub vector_clock: HashMap<String, u64>,
}

impl VersionedEntry {
    /// Create a new entry with a simple timestamp.
    pub fn new(key: impl Into<String>, value: Vec<u8>, timestamp: u64) -> Self {
        Self {
            key: key.into(),
            value,
            timestamp,
            vector_clock: HashMap::new(),
        }
    }

    /// Create a new entry with a vector clock.
    pub fn with_vector_clock(
        key: impl Into<String>,
        value: Vec<u8>,
        timestamp: u64,
        vc: HashMap<String, u64>,
    ) -> Self {
        Self {
            key: key.into(),
            value,
            timestamp,
            vector_clock: vc,
        }
    }
}

// ── SyncConfig ───────────────────────────────────────────────────────────────

/// Configuration for the anti-entropy sync process.
#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Conflict resolution strategy.
    pub strategy: ConflictStrategy,
    /// Maximum bytes per sync round (bandwidth throttle).
    pub max_bytes_per_round: Option<u64>,
    /// Periodic sync interval in seconds (0 = on-demand only).
    pub sync_interval_secs: u64,
    /// Enable bidirectional sync (push + pull). If false, only pull.
    pub bidirectional: bool,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            strategy: ConflictStrategy::default(),
            max_bytes_per_round: None,
            sync_interval_secs: 30,
            bidirectional: true,
        }
    }
}

// ── SyncProgress ─────────────────────────────────────────────────────────────

/// Progress of a synchronisation session.
#[derive(Debug, Clone, Default)]
pub struct SyncProgress {
    /// Total keys that need syncing.
    pub total_keys: usize,
    /// Keys processed so far.
    pub keys_synced: usize,
    /// Bytes transferred so far.
    pub bytes_transferred: u64,
    /// Number of conflicts resolved.
    pub conflicts_resolved: usize,
    /// Number of keys sent to remote (local -> remote).
    pub keys_pushed: usize,
    /// Number of keys pulled from remote (remote -> local).
    pub keys_pulled: usize,
    /// Whether the sync completed (vs. throttled).
    pub completed: bool,
}

impl SyncProgress {
    /// Fraction of work done, in `[0.0, 1.0]`.
    pub fn fraction(&self) -> f64 {
        if self.total_keys == 0 {
            return 1.0;
        }
        self.keys_synced as f64 / self.total_keys as f64
    }
}

// ── SyncStats ────────────────────────────────────────────────────────────────

/// Cumulative statistics across all sync rounds.
#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    /// Total sync rounds started.
    pub rounds_started: u64,
    /// Rounds completed successfully.
    pub rounds_completed: u64,
    /// Total keys pushed.
    pub total_keys_pushed: u64,
    /// Total keys pulled.
    pub total_keys_pulled: u64,
    /// Total conflicts resolved.
    pub total_conflicts: u64,
    /// Total bytes transferred.
    pub total_bytes: u64,
}

// ── AntiEntropy ──────────────────────────────────────────────────────────────

/// Anti-entropy synchroniser that keeps two replicas consistent.
///
/// Each replica maintains its own `BTreeMap<String, VersionedEntry>` and a
/// `MerkleTree` for efficient divergence detection.
pub struct AntiEntropy {
    config: SyncConfig,
    /// Local data store.
    local_data: BTreeMap<String, VersionedEntry>,
    /// Local Merkle tree (kept in sync with `local_data`).
    local_tree: MerkleTree,
    /// Local node id for vector clocks.
    node_id: String,
    /// Cumulative stats.
    stats: SyncStats,
    /// Current simulated time (seconds).
    current_time_secs: u64,
    /// Time of last sync.
    last_sync_time_secs: u64,
}

impl AntiEntropy {
    /// Create a new anti-entropy instance.
    pub fn new(node_id: impl Into<String>, config: SyncConfig) -> Self {
        Self {
            config,
            local_data: BTreeMap::new(),
            local_tree: MerkleTree::new(),
            node_id: node_id.into(),
            stats: SyncStats::default(),
            current_time_secs: 0,
            last_sync_time_secs: 0,
        }
    }

    /// Set the current simulated time.
    pub fn set_time(&mut self, secs: u64) {
        self.current_time_secs = secs;
    }

    /// Insert or update a key-value pair.
    pub fn put(&mut self, key: impl Into<String>, value: Vec<u8>, timestamp: u64) {
        let key = key.into();
        let mut vc = HashMap::new();
        vc.insert(self.node_id.clone(), timestamp);
        let entry = VersionedEntry::with_vector_clock(key.clone(), value, timestamp, vc);
        self.local_data.insert(key, entry);
        self.rebuild_tree();
    }

    /// Remove a key from local data.
    pub fn remove(&mut self, key: &str) -> Option<VersionedEntry> {
        let removed = self.local_data.remove(key);
        if removed.is_some() {
            self.rebuild_tree();
        }
        removed
    }

    /// Get a local entry by key.
    pub fn get(&self, key: &str) -> Option<&VersionedEntry> {
        self.local_data.get(key)
    }

    /// Return all local keys.
    pub fn keys(&self) -> Vec<String> {
        self.local_data.keys().cloned().collect()
    }

    /// Return the local Merkle tree.
    pub fn tree(&self) -> &MerkleTree {
        &self.local_tree
    }

    /// Return cumulative statistics.
    pub fn stats(&self) -> &SyncStats {
        &self.stats
    }

    /// Return the current configuration.
    pub fn config(&self) -> &SyncConfig {
        &self.config
    }

    /// Check whether it is time for a periodic sync.
    pub fn should_sync(&self) -> bool {
        if self.config.sync_interval_secs == 0 {
            return false;
        }
        self.current_time_secs
            .saturating_sub(self.last_sync_time_secs)
            >= self.config.sync_interval_secs
    }

    /// Detect keys that are missing on one side or have divergent values.
    ///
    /// Returns `(missing_locally, missing_remotely, divergent)` sets.
    pub fn detect_differences(
        &self,
        remote_data: &BTreeMap<String, VersionedEntry>,
    ) -> (HashSet<String>, HashSet<String>, HashSet<String>) {
        let local_keys: HashSet<&String> = self.local_data.keys().collect();
        let remote_keys: HashSet<&String> = remote_data.keys().collect();

        let missing_locally: HashSet<String> = remote_keys
            .difference(&local_keys)
            .map(|k| (*k).clone())
            .collect();
        let missing_remotely: HashSet<String> = local_keys
            .difference(&remote_keys)
            .map(|k| (*k).clone())
            .collect();

        let mut divergent = HashSet::new();
        for key in local_keys.intersection(&remote_keys) {
            let local_raw = self.entry_raw_value(key);
            let remote_entry = remote_data.get(*key);
            if let Some(re) = remote_entry {
                if local_raw != re.value {
                    divergent.insert((*key).clone());
                }
            }
        }

        (missing_locally, missing_remotely, divergent)
    }

    /// Perform a bidirectional reconciliation with a remote replica.
    ///
    /// Returns a `SyncProgress` describing what happened.
    pub fn reconcile(
        &mut self,
        remote_data: &mut BTreeMap<String, VersionedEntry>,
    ) -> SyncProgress {
        self.stats.rounds_started += 1;
        let (missing_locally, missing_remotely, divergent) = self.detect_differences(remote_data);

        let total = missing_locally.len() + missing_remotely.len() + divergent.len();
        let mut progress = SyncProgress {
            total_keys: total,
            ..Default::default()
        };
        let mut bytes_budget = self.config.max_bytes_per_round;

        // Pull missing keys from remote
        for key in &missing_locally {
            if let Some(budget) = &mut bytes_budget {
                if let Some(entry) = remote_data.get(key) {
                    let cost = entry.value.len() as u64;
                    if cost > *budget {
                        // Throttled
                        self.last_sync_time_secs = self.current_time_secs;
                        self.stats.total_bytes += progress.bytes_transferred;
                        self.stats.total_keys_pulled += progress.keys_pulled as u64;
                        self.stats.total_keys_pushed += progress.keys_pushed as u64;
                        self.stats.total_conflicts += progress.conflicts_resolved as u64;
                        return progress;
                    }
                    *budget -= cost;
                }
            }
            if let Some(entry) = remote_data.get(key) {
                progress.bytes_transferred += entry.value.len() as u64;
                self.local_data.insert(key.clone(), entry.clone());
                progress.keys_pulled += 1;
                progress.keys_synced += 1;
            }
        }

        // Push missing keys to remote (if bidirectional)
        if self.config.bidirectional {
            for key in &missing_remotely {
                if let Some(budget) = &mut bytes_budget {
                    if let Some(entry) = self.local_data.get(key) {
                        let cost = entry.value.len() as u64;
                        if cost > *budget {
                            self.rebuild_tree();
                            self.last_sync_time_secs = self.current_time_secs;
                            self.stats.total_bytes += progress.bytes_transferred;
                            self.stats.total_keys_pulled += progress.keys_pulled as u64;
                            self.stats.total_keys_pushed += progress.keys_pushed as u64;
                            self.stats.total_conflicts += progress.conflicts_resolved as u64;
                            return progress;
                        }
                        *budget -= cost;
                    }
                }
                if let Some(entry) = self.local_data.get(key) {
                    progress.bytes_transferred += entry.value.len() as u64;
                    remote_data.insert(key.clone(), entry.clone());
                    progress.keys_pushed += 1;
                    progress.keys_synced += 1;
                }
            }
        }

        // Resolve conflicts
        for key in &divergent {
            let local_entry = self.local_data.get(key).cloned();
            let remote_entry = remote_data.get(key).cloned();

            if let (Some(le), Some(re)) = (local_entry, remote_entry) {
                let (winner_local, winner_remote) = self.resolve_conflict(&le, &re);
                self.local_data.insert(key.clone(), winner_local.clone());
                if self.config.bidirectional {
                    remote_data.insert(key.clone(), winner_remote);
                }
                progress.conflicts_resolved += 1;
                progress.keys_synced += 1;
                progress.bytes_transferred += winner_local.value.len() as u64;
            }
        }

        progress.completed = true;
        self.rebuild_tree();
        self.last_sync_time_secs = self.current_time_secs;
        self.stats.rounds_completed += 1;
        self.stats.total_bytes += progress.bytes_transferred;
        self.stats.total_keys_pulled += progress.keys_pulled as u64;
        self.stats.total_keys_pushed += progress.keys_pushed as u64;
        self.stats.total_conflicts += progress.conflicts_resolved as u64;

        progress
    }

    // ── Private helpers ──────────────────────────────────────────────────

    fn entry_raw_value(&self, key: &str) -> Vec<u8> {
        self.local_data
            .get(key)
            .map(|e| e.value.clone())
            .unwrap_or_default()
    }

    fn rebuild_tree(&mut self) {
        let raw: BTreeMap<String, Vec<u8>> = self
            .local_data
            .iter()
            .map(|(k, v)| (k.clone(), v.value.clone()))
            .collect();
        self.local_tree = MerkleTree::build(&raw);
    }

    fn resolve_conflict(
        &self,
        local: &VersionedEntry,
        remote: &VersionedEntry,
    ) -> (VersionedEntry, VersionedEntry) {
        match &self.config.strategy {
            ConflictStrategy::LastWriteWins => {
                if local.timestamp >= remote.timestamp {
                    (local.clone(), local.clone())
                } else {
                    (remote.clone(), remote.clone())
                }
            }
            ConflictStrategy::VectorClock => {
                let local_dominates = self.vc_dominates(&local.vector_clock, &remote.vector_clock);
                let remote_dominates = self.vc_dominates(&remote.vector_clock, &local.vector_clock);

                if local_dominates && !remote_dominates {
                    (local.clone(), local.clone())
                } else if remote_dominates && !local_dominates {
                    (remote.clone(), remote.clone())
                } else {
                    // Concurrent — fall back to LWW
                    if local.timestamp >= remote.timestamp {
                        (local.clone(), local.clone())
                    } else {
                        (remote.clone(), remote.clone())
                    }
                }
            }
            ConflictStrategy::PreferLocal => (local.clone(), local.clone()),
            ConflictStrategy::PreferRemote => (remote.clone(), remote.clone()),
        }
    }

    /// Check if vc_a causally dominates vc_b (all counters >=, at least one >).
    fn vc_dominates(&self, vc_a: &HashMap<String, u64>, vc_b: &HashMap<String, u64>) -> bool {
        let all_keys: HashSet<&String> = vc_a.keys().chain(vc_b.keys()).collect();
        let mut at_least_one_greater = false;
        for k in all_keys {
            let a = vc_a.get(k).copied().unwrap_or(0);
            let b = vc_b.get(k).copied().unwrap_or(0);
            if a < b {
                return false;
            }
            if a > b {
                at_least_one_greater = true;
            }
        }
        at_least_one_greater
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(pairs: &[(&str, &[u8])]) -> BTreeMap<String, Vec<u8>> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_vec()))
            .collect()
    }

    fn make_node(node_id: &str) -> AntiEntropy {
        AntiEntropy::new(node_id, SyncConfig::default())
    }

    // ── Merkle tree ──────────────────────────────────────────────────────

    #[test]
    fn test_empty_merkle_tree() {
        let tree = MerkleTree::new();
        assert!(tree.root_hash().is_none());
        assert_eq!(tree.entry_count(), 0);
    }

    #[test]
    fn test_merkle_tree_single_entry() {
        let data = make_data(&[("key1", b"val1")]);
        let tree = MerkleTree::build(&data);
        assert!(tree.root_hash().is_some());
        assert_eq!(tree.entry_count(), 1);
    }

    #[test]
    fn test_merkle_tree_multiple_entries() {
        let data = make_data(&[("a", b"1"), ("b", b"2"), ("c", b"3")]);
        let tree = MerkleTree::build(&data);
        assert_eq!(tree.entry_count(), 3);
        assert!(tree.root_hash().is_some());
    }

    #[test]
    fn test_merkle_tree_deterministic() {
        let data = make_data(&[("x", b"10"), ("y", b"20")]);
        let t1 = MerkleTree::build(&data);
        let t2 = MerkleTree::build(&data);
        assert_eq!(t1.root_hash(), t2.root_hash());
    }

    #[test]
    fn test_merkle_tree_different_data_different_hash() {
        let d1 = make_data(&[("a", b"1")]);
        let d2 = make_data(&[("a", b"2")]);
        let t1 = MerkleTree::build(&d1);
        let t2 = MerkleTree::build(&d2);
        assert_ne!(t1.root_hash(), t2.root_hash());
    }

    #[test]
    fn test_merkle_tree_identical_trees_no_divergence() {
        let data = make_data(&[("a", b"1"), ("b", b"2")]);
        let t1 = MerkleTree::build(&data);
        let t2 = MerkleTree::build(&data);
        assert!(t1.find_divergent_ranges(&t2).is_empty());
    }

    #[test]
    fn test_merkle_tree_divergent_ranges() {
        let d1 = make_data(&[("a", b"1"), ("b", b"2")]);
        let d2 = make_data(&[("a", b"1"), ("b", b"CHANGED")]);
        let t1 = MerkleTree::build(&d1);
        let t2 = MerkleTree::build(&d2);
        let ranges = t1.find_divergent_ranges(&t2);
        assert!(!ranges.is_empty());
    }

    #[test]
    fn test_merkle_tree_empty_vs_nonempty() {
        let empty = MerkleTree::new();
        let data = make_data(&[("a", b"1")]);
        let full = MerkleTree::build(&data);
        let ranges = empty.find_divergent_ranges(&full);
        assert!(!ranges.is_empty());
    }

    #[test]
    fn test_merkle_tree_incremental_update() {
        let mut data = make_data(&[("a", b"1")]);
        let mut tree = MerkleTree::build(&data);
        let h1 = tree.root_hash();

        data.insert("b".to_string(), b"2".to_vec());
        tree.update(&data);
        let h2 = tree.root_hash();

        assert_ne!(h1, h2);
        assert_eq!(tree.entry_count(), 2);
    }

    // ── Anti-entropy basic operations ────────────────────────────────────

    #[test]
    fn test_new_anti_entropy() {
        let ae = make_node("node-1");
        assert!(ae.keys().is_empty());
        assert_eq!(ae.stats().rounds_started, 0);
    }

    #[test]
    fn test_put_and_get() {
        let mut ae = make_node("n1");
        ae.put("key1", b"value1".to_vec(), 100);
        let entry = ae.get("key1").expect("should exist");
        assert_eq!(entry.value, b"value1");
        assert_eq!(entry.timestamp, 100);
    }

    #[test]
    fn test_remove() {
        let mut ae = make_node("n1");
        ae.put("k", b"v".to_vec(), 1);
        assert!(ae.get("k").is_some());

        let removed = ae.remove("k");
        assert!(removed.is_some());
        assert!(ae.get("k").is_none());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut ae = make_node("n1");
        assert!(ae.remove("nope").is_none());
    }

    #[test]
    fn test_keys() {
        let mut ae = make_node("n1");
        ae.put("b", b"2".to_vec(), 1);
        ae.put("a", b"1".to_vec(), 2);
        let keys = ae.keys();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
    }

    #[test]
    fn test_tree_updates_on_put() {
        let mut ae = make_node("n1");
        assert!(ae.tree().root_hash().is_none());
        ae.put("k", b"v".to_vec(), 1);
        assert!(ae.tree().root_hash().is_some());
    }

    // ── Difference detection ─────────────────────────────────────────────

    #[test]
    fn test_detect_missing_locally() {
        let ae = make_node("n1");
        let mut remote = BTreeMap::new();
        remote.insert(
            "remote_key".to_string(),
            VersionedEntry::new("remote_key", b"val".to_vec(), 1),
        );

        let (missing_local, _, _) = ae.detect_differences(&remote);
        assert!(missing_local.contains("remote_key"));
    }

    #[test]
    fn test_detect_missing_remotely() {
        let mut ae = make_node("n1");
        ae.put("local_key", b"val".to_vec(), 1);
        let remote = BTreeMap::new();

        let (_, missing_remote, _) = ae.detect_differences(&remote);
        assert!(missing_remote.contains("local_key"));
    }

    #[test]
    fn test_detect_divergent() {
        let mut ae = make_node("n1");
        ae.put("shared", b"local_val".to_vec(), 1);

        let mut remote = BTreeMap::new();
        remote.insert(
            "shared".to_string(),
            VersionedEntry::new("shared", b"remote_val".to_vec(), 2),
        );

        let (_, _, divergent) = ae.detect_differences(&remote);
        assert!(divergent.contains("shared"));
    }

    #[test]
    fn test_detect_no_differences() {
        let mut ae = make_node("n1");
        ae.put("k", b"v".to_vec(), 1);

        let mut remote = BTreeMap::new();
        remote.insert("k".to_string(), VersionedEntry::new("k", b"v".to_vec(), 1));

        let (ml, mr, div) = ae.detect_differences(&remote);
        assert!(ml.is_empty());
        assert!(mr.is_empty());
        assert!(div.is_empty());
    }

    // ── Reconciliation ───────────────────────────────────────────────────

    #[test]
    fn test_reconcile_pull_missing() {
        let mut ae = make_node("n1");
        let mut remote = BTreeMap::new();
        remote.insert(
            "rk".to_string(),
            VersionedEntry::new("rk", b"rv".to_vec(), 10),
        );

        let progress = ae.reconcile(&mut remote);
        assert!(progress.completed);
        assert_eq!(progress.keys_pulled, 1);
        assert!(ae.get("rk").is_some());
    }

    #[test]
    fn test_reconcile_push_missing_bidirectional() {
        let mut ae = make_node("n1");
        ae.put("lk", b"lv".to_vec(), 5);
        let mut remote = BTreeMap::new();

        let progress = ae.reconcile(&mut remote);
        assert!(progress.completed);
        assert_eq!(progress.keys_pushed, 1);
        assert!(remote.contains_key("lk"));
    }

    #[test]
    fn test_reconcile_no_push_if_unidirectional() {
        let cfg = SyncConfig {
            bidirectional: false,
            ..Default::default()
        };
        let mut ae = AntiEntropy::new("n1", cfg);
        ae.put("lk", b"lv".to_vec(), 5);
        let mut remote = BTreeMap::new();

        let progress = ae.reconcile(&mut remote);
        assert!(progress.completed);
        assert_eq!(progress.keys_pushed, 0);
        assert!(!remote.contains_key("lk"));
    }

    #[test]
    fn test_reconcile_lww_local_wins() {
        let mut ae = make_node("n1");
        ae.put("k", b"local".to_vec(), 100);

        let mut remote = BTreeMap::new();
        remote.insert(
            "k".to_string(),
            VersionedEntry::new("k", b"remote".to_vec(), 50),
        );

        let progress = ae.reconcile(&mut remote);
        assert_eq!(progress.conflicts_resolved, 1);
        assert_eq!(ae.get("k").expect("exists").value, b"local");
    }

    #[test]
    fn test_reconcile_lww_remote_wins() {
        let mut ae = make_node("n1");
        ae.put("k", b"local".to_vec(), 10);

        let mut remote = BTreeMap::new();
        remote.insert(
            "k".to_string(),
            VersionedEntry::new("k", b"remote".to_vec(), 200),
        );

        let progress = ae.reconcile(&mut remote);
        assert_eq!(progress.conflicts_resolved, 1);
        assert_eq!(ae.get("k").expect("exists").value, b"remote");
    }

    #[test]
    fn test_reconcile_prefer_local() {
        let cfg = SyncConfig {
            strategy: ConflictStrategy::PreferLocal,
            ..Default::default()
        };
        let mut ae = AntiEntropy::new("n1", cfg);
        ae.put("k", b"mine".to_vec(), 1);

        let mut remote = BTreeMap::new();
        remote.insert(
            "k".to_string(),
            VersionedEntry::new("k", b"theirs".to_vec(), 999),
        );

        ae.reconcile(&mut remote);
        assert_eq!(ae.get("k").expect("exists").value, b"mine");
    }

    #[test]
    fn test_reconcile_prefer_remote() {
        let cfg = SyncConfig {
            strategy: ConflictStrategy::PreferRemote,
            ..Default::default()
        };
        let mut ae = AntiEntropy::new("n1", cfg);
        ae.put("k", b"mine".to_vec(), 999);

        let mut remote = BTreeMap::new();
        remote.insert(
            "k".to_string(),
            VersionedEntry::new("k", b"theirs".to_vec(), 1),
        );

        ae.reconcile(&mut remote);
        assert_eq!(ae.get("k").expect("exists").value, b"theirs");
    }

    // ── Vector clock conflict resolution ─────────────────────────────────

    #[test]
    fn test_vector_clock_local_dominates() {
        let cfg = SyncConfig {
            strategy: ConflictStrategy::VectorClock,
            ..Default::default()
        };
        let mut ae = AntiEntropy::new("n1", cfg);
        ae.put("k", b"local".to_vec(), 5);

        let mut remote_vc = HashMap::new();
        remote_vc.insert("n2".to_string(), 1);
        let mut remote = BTreeMap::new();
        remote.insert(
            "k".to_string(),
            VersionedEntry::with_vector_clock("k", b"remote".to_vec(), 3, remote_vc),
        );

        ae.reconcile(&mut remote);
        assert_eq!(ae.get("k").expect("exists").value, b"local");
    }

    #[test]
    fn test_vector_clock_remote_dominates() {
        let cfg = SyncConfig {
            strategy: ConflictStrategy::VectorClock,
            ..Default::default()
        };
        let mut ae = AntiEntropy::new("n1", cfg);
        // Local has vc {n1: 1}
        ae.put("k", b"local".to_vec(), 1);

        let mut remote_vc = HashMap::new();
        remote_vc.insert("n1".to_string(), 2);
        remote_vc.insert("n2".to_string(), 3);
        let mut remote = BTreeMap::new();
        remote.insert(
            "k".to_string(),
            VersionedEntry::with_vector_clock("k", b"remote".to_vec(), 10, remote_vc),
        );

        ae.reconcile(&mut remote);
        assert_eq!(ae.get("k").expect("exists").value, b"remote");
    }

    // ── Bandwidth throttling ─────────────────────────────────────────────

    #[test]
    fn test_bandwidth_throttling() {
        let cfg = SyncConfig {
            max_bytes_per_round: Some(15),
            ..Default::default()
        };
        let mut ae = AntiEntropy::new("n1", cfg);

        // Both entries are 12 bytes. Budget of 15 allows exactly one (12 <= 15),
        // then the remaining budget (3) is too small for the second (12 > 3).
        let mut remote = BTreeMap::new();
        remote.insert(
            "k1".to_string(),
            VersionedEntry::new("k1", b"twelve_bytes".to_vec(), 1),
        );
        remote.insert(
            "k2".to_string(),
            VersionedEntry::new("k2", b"twelve_byte2".to_vec(), 2),
        );

        let progress = ae.reconcile(&mut remote);
        // Should have pulled exactly one, then throttled
        assert_eq!(progress.keys_pulled, 1);
        assert!(!progress.completed);
    }

    // ── Sync scheduling ─────────────────────────────────────────────────

    #[test]
    fn test_should_sync_time_based() {
        let cfg = SyncConfig {
            sync_interval_secs: 60,
            ..Default::default()
        };
        let mut ae = AntiEntropy::new("n1", cfg);
        ae.set_time(59);
        assert!(!ae.should_sync());
        ae.set_time(60);
        assert!(ae.should_sync());
    }

    #[test]
    fn test_should_sync_disabled() {
        let cfg = SyncConfig {
            sync_interval_secs: 0,
            ..Default::default()
        };
        let ae = AntiEntropy::new("n1", cfg);
        assert!(!ae.should_sync());
    }

    #[test]
    fn test_sync_updates_last_sync_time() {
        let mut ae = make_node("n1");
        ae.set_time(100);
        let mut remote = BTreeMap::new();
        ae.reconcile(&mut remote);

        // After sync at t=100, next sync should not trigger until interval passes
        ae.set_time(100 + 29);
        assert!(!ae.should_sync());
        ae.set_time(100 + 30);
        assert!(ae.should_sync());
    }

    // ── Statistics ───────────────────────────────────────────────────────

    #[test]
    fn test_stats_accumulate() {
        let mut ae = make_node("n1");
        ae.put("a", b"1".to_vec(), 1);
        let mut remote = BTreeMap::new();

        ae.reconcile(&mut remote);
        ae.reconcile(&mut remote);

        assert_eq!(ae.stats().rounds_started, 2);
        assert_eq!(ae.stats().rounds_completed, 2);
    }

    #[test]
    fn test_stats_bytes_tracked() {
        let mut ae = make_node("n1");
        let mut remote = BTreeMap::new();
        remote.insert(
            "k".to_string(),
            VersionedEntry::new("k", b"hello".to_vec(), 1),
        );

        ae.reconcile(&mut remote);
        assert!(ae.stats().total_bytes > 0);
    }

    // ── Progress ─────────────────────────────────────────────────────────

    #[test]
    fn test_progress_fraction_empty() {
        let p = SyncProgress::default();
        assert!((p.fraction() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_progress_fraction_partial() {
        let p = SyncProgress {
            total_keys: 10,
            keys_synced: 5,
            ..Default::default()
        };
        assert!((p.fraction() - 0.5).abs() < f64::EPSILON);
    }

    // ── Default config ───────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let cfg = SyncConfig::default();
        assert_eq!(cfg.strategy, ConflictStrategy::LastWriteWins);
        assert!(cfg.bidirectional);
        assert_eq!(cfg.sync_interval_secs, 30);
        assert!(cfg.max_bytes_per_round.is_none());
    }

    #[test]
    fn test_default_conflict_strategy() {
        let s = ConflictStrategy::default();
        assert_eq!(s, ConflictStrategy::LastWriteWins);
    }

    // ── Full workflow ────────────────────────────────────────────────────

    #[test]
    fn test_full_two_node_sync() {
        let mut node_a = make_node("a");
        let mut node_b = make_node("b");

        node_a.put("shared", b"from_a".to_vec(), 10);
        node_a.put("only_a", b"a_data".to_vec(), 11);
        node_b.put("shared", b"from_b".to_vec(), 20);
        node_b.put("only_b", b"b_data".to_vec(), 12);

        // Reconcile a with b's data
        let mut b_data = node_b.local_data.clone();
        let progress = node_a.reconcile(&mut b_data);

        assert!(progress.completed);
        // a should have pulled only_b
        assert!(node_a.get("only_b").is_some());
        // a should have pushed only_a
        assert!(b_data.contains_key("only_a"));
        // shared: b wins because timestamp 20 > 10
        assert_eq!(node_a.get("shared").expect("exists").value, b"from_b");
    }

    #[test]
    fn test_merkle_tree_root_range() {
        let data = make_data(&[("alpha", b"1"), ("beta", b"2"), ("gamma", b"3")]);
        let tree = MerkleTree::build(&data);
        let root = tree.root().expect("has root");
        assert_eq!(root.range_start, "alpha");
        assert_eq!(root.range_end, "gamma");
    }

    #[test]
    fn test_fnv1a_deterministic() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv1a_different_inputs() {
        let h1 = fnv1a_hash(b"hello");
        let h2 = fnv1a_hash(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_combine_hashes_deterministic() {
        let h = combine_hashes(123, 456);
        let h2 = combine_hashes(123, 456);
        assert_eq!(h, h2);
    }
}
