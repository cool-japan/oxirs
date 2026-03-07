//! Stream checkpoint/offset tracking for at-least-once delivery.
//!
//! Provides `CheckpointStore` which commits consumer offsets per (stream_id, partition)
//! pair and keeps a bounded history for replay and auditing.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A committed offset for one stream partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkpoint {
    /// Logical stream name.
    pub stream_id: String,
    /// Partition index within the stream.
    pub partition: u32,
    /// Consumed-up-to offset (inclusive).
    pub offset: i64,
    /// Wall-clock timestamp when the checkpoint was committed (epoch millis).
    pub timestamp: i64,
    /// Arbitrary application metadata (e.g. consumer group, host, version).
    pub metadata: HashMap<String, String>,
}

impl Checkpoint {
    /// Create a minimal checkpoint with no metadata.
    pub fn new(stream_id: impl Into<String>, partition: u32, offset: i64, timestamp: i64) -> Self {
        Self {
            stream_id: stream_id.into(),
            partition,
            offset,
            timestamp,
            metadata: HashMap::new(),
        }
    }

    /// Builder helper: attach a single metadata entry.
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Persistent checkpoint store with bounded history per (stream, partition) key.
///
/// The store always keeps the *latest* checkpoint as the authoritative offset
/// and additionally retains up to `max_history` older entries for auditing or
/// replay purposes.
#[derive(Debug)]
pub struct CheckpointStore {
    /// Latest checkpoint per (stream_id, partition).
    checkpoints: HashMap<(String, u32), Checkpoint>,
    /// Ordered commit history (newest last).
    history: Vec<Checkpoint>,
    /// Maximum entries kept in `history`.
    max_history: usize,
}

impl CheckpointStore {
    /// Create a new store.
    ///
    /// # Parameters
    /// - `max_history`: maximum number of historical checkpoints retained.
    pub fn new(max_history: usize) -> Self {
        Self {
            checkpoints: HashMap::new(),
            history: Vec::new(),
            max_history,
        }
    }

    /// Commit a checkpoint.
    ///
    /// Updates the latest offset for the (stream, partition) key and appends
    /// the entry to the rolling history buffer.
    pub fn commit(&mut self, checkpoint: Checkpoint) {
        let key = (checkpoint.stream_id.clone(), checkpoint.partition);
        self.checkpoints.insert(key, checkpoint.clone());

        self.history.push(checkpoint);
        // Trim history to the configured limit.
        if self.history.len() > self.max_history {
            let excess = self.history.len() - self.max_history;
            self.history.drain(0..excess);
        }
    }

    /// Return a reference to the latest checkpoint for a (stream, partition) pair.
    pub fn get(&self, stream_id: &str, partition: u32) -> Option<&Checkpoint> {
        self.checkpoints.get(&(stream_id.to_owned(), partition))
    }

    /// Return the latest committed offset, if any.
    pub fn latest_offset(&self, stream_id: &str, partition: u32) -> Option<i64> {
        self.get(stream_id, partition).map(|c| c.offset)
    }

    /// Remove the checkpoint for the given (stream, partition) key.
    ///
    /// Historical entries for this key are *not* removed so that auditing
    /// still works.
    pub fn reset(&mut self, stream_id: &str, partition: u32) {
        self.checkpoints.remove(&(stream_id.to_owned(), partition));
    }

    /// Roll back the checkpoint to a specific offset.
    ///
    /// Returns `true` if a checkpoint existed and was updated, `false` otherwise.
    /// Historical entries are preserved unchanged.
    pub fn reset_to(&mut self, stream_id: &str, partition: u32, offset: i64) -> bool {
        let key = (stream_id.to_owned(), partition);
        if let Some(cp) = self.checkpoints.get_mut(&key) {
            cp.offset = offset;
            true
        } else {
            false
        }
    }

    /// Return all stream IDs that have at least one active checkpoint.
    pub fn all_streams(&self) -> Vec<&str> {
        let mut seen: Vec<&str> = Vec::new();
        for (stream_id, _partition) in self.checkpoints.keys() {
            let s: &str = stream_id.as_str();
            if !seen.contains(&s) {
                seen.push(s);
            }
        }
        seen
    }

    /// Return all active partitions for a given stream.
    pub fn partitions(&self, stream_id: &str) -> Vec<u32> {
        self.checkpoints
            .keys()
            .filter(|(s, _)| s == stream_id)
            .map(|(_, p)| *p)
            .collect()
    }

    /// Return historical checkpoints for a (stream, partition) pair, oldest first.
    pub fn history(&self, stream_id: &str, partition: u32) -> Vec<&Checkpoint> {
        self.history
            .iter()
            .filter(|c| c.stream_id == stream_id && c.partition == partition)
            .collect()
    }

    /// Return the total number of commits across all partitions and streams.
    pub fn total_committed(&self) -> usize {
        self.history.len()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cp(stream: &str, partition: u32, offset: i64, ts: i64) -> Checkpoint {
        Checkpoint::new(stream, partition, offset, ts)
    }

    // ── Basic construction ────────────────────────────────────────────────────

    #[test]
    fn test_new_store_is_empty() {
        let store = CheckpointStore::new(100);
        assert_eq!(store.total_committed(), 0);
        assert!(store.all_streams().is_empty());
    }

    #[test]
    fn test_commit_single() {
        let mut store = CheckpointStore::new(10);
        store.commit(make_cp("topic-a", 0, 42, 1000));
        assert_eq!(store.latest_offset("topic-a", 0), Some(42));
        assert_eq!(store.total_committed(), 1);
    }

    #[test]
    fn test_commit_updates_latest() {
        let mut store = CheckpointStore::new(10);
        store.commit(make_cp("topic-a", 0, 10, 1000));
        store.commit(make_cp("topic-a", 0, 20, 2000));
        assert_eq!(store.latest_offset("topic-a", 0), Some(20));
    }

    #[test]
    fn test_commit_multiple_partitions() {
        let mut store = CheckpointStore::new(10);
        store.commit(make_cp("t", 0, 5, 100));
        store.commit(make_cp("t", 1, 15, 200));
        store.commit(make_cp("t", 2, 25, 300));

        assert_eq!(store.latest_offset("t", 0), Some(5));
        assert_eq!(store.latest_offset("t", 1), Some(15));
        assert_eq!(store.latest_offset("t", 2), Some(25));
    }

    #[test]
    fn test_get_none_for_unknown() {
        let store = CheckpointStore::new(10);
        assert!(store.get("missing", 0).is_none());
        assert!(store.latest_offset("missing", 0).is_none());
    }

    // ── History ───────────────────────────────────────────────────────────────

    #[test]
    fn test_history_is_ordered() {
        let mut store = CheckpointStore::new(50);
        for i in 0..5_i64 {
            store.commit(make_cp("events", 0, i, i * 100));
        }
        let hist = store.history("events", 0);
        assert_eq!(hist.len(), 5);
        // Oldest first
        assert_eq!(hist[0].offset, 0);
        assert_eq!(hist[4].offset, 4);
    }

    #[test]
    fn test_history_bounded_by_max() {
        let max = 5_usize;
        let mut store = CheckpointStore::new(max);
        for i in 0..10_i64 {
            store.commit(make_cp("stream", 0, i, i));
        }
        // total_committed returns history len, bounded by max_history
        assert_eq!(store.total_committed(), max);
    }

    #[test]
    fn test_history_only_for_matching_partition() {
        let mut store = CheckpointStore::new(50);
        store.commit(make_cp("s", 0, 1, 1));
        store.commit(make_cp("s", 1, 2, 2));
        store.commit(make_cp("s", 0, 3, 3));

        let h0 = store.history("s", 0);
        let h1 = store.history("s", 1);
        assert_eq!(h0.len(), 2);
        assert_eq!(h1.len(), 1);
    }

    #[test]
    fn test_history_empty_for_unknown() {
        let store = CheckpointStore::new(10);
        assert!(store.history("none", 0).is_empty());
    }

    // ── Reset ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_removes_checkpoint() {
        let mut store = CheckpointStore::new(10);
        store.commit(make_cp("r", 0, 99, 999));
        store.reset("r", 0);
        assert!(store.get("r", 0).is_none());
    }

    #[test]
    fn test_reset_preserves_history() {
        let mut store = CheckpointStore::new(10);
        store.commit(make_cp("r", 0, 10, 1));
        store.reset("r", 0);
        // History still contains the committed entry
        assert_eq!(store.history("r", 0).len(), 1);
    }

    #[test]
    fn test_reset_nonexistent_is_noop() {
        let mut store = CheckpointStore::new(10);
        // Should not panic
        store.reset("phantom", 99);
        assert_eq!(store.total_committed(), 0);
    }

    // ── Reset-to ──────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_to_existing() {
        let mut store = CheckpointStore::new(10);
        store.commit(make_cp("x", 0, 50, 100));
        let ok = store.reset_to("x", 0, 30);
        assert!(ok);
        assert_eq!(store.latest_offset("x", 0), Some(30));
    }

    #[test]
    fn test_reset_to_nonexistent_returns_false() {
        let mut store = CheckpointStore::new(10);
        let ok = store.reset_to("none", 0, 10);
        assert!(!ok);
    }

    #[test]
    fn test_reset_to_negative_offset() {
        let mut store = CheckpointStore::new(10);
        store.commit(make_cp("neg", 0, 100, 1));
        let ok = store.reset_to("neg", 0, -1);
        assert!(ok);
        assert_eq!(store.latest_offset("neg", 0), Some(-1));
    }

    // ── All streams / partitions ───────────────────────────────────────────────

    #[test]
    fn test_all_streams_unique() {
        let mut store = CheckpointStore::new(20);
        store.commit(make_cp("alpha", 0, 1, 1));
        store.commit(make_cp("alpha", 1, 2, 2));
        store.commit(make_cp("beta", 0, 3, 3));

        let mut streams = store.all_streams();
        streams.sort_unstable();
        assert_eq!(streams, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_partitions_for_stream() {
        let mut store = CheckpointStore::new(20);
        store.commit(make_cp("p", 0, 1, 1));
        store.commit(make_cp("p", 2, 2, 2));
        store.commit(make_cp("p", 5, 3, 3));

        let mut parts = store.partitions("p");
        parts.sort_unstable();
        assert_eq!(parts, vec![0, 2, 5]);
    }

    #[test]
    fn test_partitions_empty_for_unknown_stream() {
        let store = CheckpointStore::new(10);
        assert!(store.partitions("unknown").is_empty());
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    #[test]
    fn test_checkpoint_metadata() {
        let cp = Checkpoint::new("stream", 0, 42, 1000)
            .with_meta("consumer_group", "grp1")
            .with_meta("host", "worker-01");
        assert_eq!(cp.metadata["consumer_group"], "grp1");
        assert_eq!(cp.metadata["host"], "worker-01");
    }

    #[test]
    fn test_metadata_stored_in_checkpoint() {
        let mut store = CheckpointStore::new(10);
        let cp = Checkpoint::new("s", 0, 1, 100).with_meta("key", "val");
        store.commit(cp);
        let stored = store.get("s", 0).expect("checkpoint should exist");
        assert_eq!(stored.metadata["key"], "val");
    }

    // ── Multi-stream isolation ────────────────────────────────────────────────

    #[test]
    fn test_multiple_streams_independent() {
        let mut store = CheckpointStore::new(50);
        store.commit(make_cp("stream-1", 0, 100, 1000));
        store.commit(make_cp("stream-2", 0, 200, 2000));
        store.commit(make_cp("stream-1", 0, 150, 3000));

        assert_eq!(store.latest_offset("stream-1", 0), Some(150));
        assert_eq!(store.latest_offset("stream-2", 0), Some(200));
    }

    #[test]
    fn test_streams_do_not_share_history() {
        let mut store = CheckpointStore::new(50);
        store.commit(make_cp("a", 0, 1, 1));
        store.commit(make_cp("b", 0, 2, 2));

        let ha = store.history("a", 0);
        let hb = store.history("b", 0);
        assert_eq!(ha.len(), 1);
        assert_eq!(hb.len(), 1);
        assert_eq!(ha[0].stream_id, "a");
        assert_eq!(hb[0].stream_id, "b");
    }

    // ── Edge cases ────────────────────────────────────────────────────────────

    #[test]
    fn test_zero_max_history() {
        let mut store = CheckpointStore::new(0);
        store.commit(make_cp("z", 0, 1, 1));
        // No history is kept but checkpoint is still committed
        assert_eq!(store.latest_offset("z", 0), Some(1));
        assert_eq!(store.total_committed(), 0);
    }

    #[test]
    fn test_large_offset() {
        let mut store = CheckpointStore::new(10);
        store.commit(make_cp("huge", 0, i64::MAX, i64::MAX));
        assert_eq!(store.latest_offset("huge", 0), Some(i64::MAX));
    }

    #[test]
    fn test_checkpoint_equality() {
        let c1 = make_cp("s", 0, 10, 100);
        let c2 = make_cp("s", 0, 10, 100);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_history_across_partitions_in_same_store() {
        let mut store = CheckpointStore::new(50);
        for p in 0..3_u32 {
            for o in 0..3_i64 {
                store.commit(make_cp("multi", p, o, (p as i64) * 10 + o));
            }
        }
        // 3 partitions × 3 offsets = 9 total
        assert_eq!(store.total_committed(), 9);
        for p in 0..3_u32 {
            assert_eq!(store.history("multi", p).len(), 3);
        }
    }

    #[test]
    fn test_many_commits_bounded_history() {
        let mut store = CheckpointStore::new(20);
        for i in 0..100_i64 {
            store.commit(make_cp("bounded", 0, i, i));
        }
        assert!(store.total_committed() <= 20);
        // Latest should still be correct
        assert_eq!(store.latest_offset("bounded", 0), Some(99));
    }

    #[test]
    fn test_all_streams_after_reset() {
        let mut store = CheckpointStore::new(20);
        store.commit(make_cp("a", 0, 1, 1));
        store.commit(make_cp("b", 0, 2, 2));
        store.reset("a", 0);

        let streams = store.all_streams();
        assert!(!streams.contains(&"a"));
        assert!(streams.contains(&"b"));
    }

    #[test]
    fn test_partitions_after_reset() {
        let mut store = CheckpointStore::new(20);
        store.commit(make_cp("s", 0, 1, 1));
        store.commit(make_cp("s", 1, 2, 2));
        store.reset("s", 0);

        let parts = store.partitions("s");
        assert!(!parts.contains(&0));
        assert!(parts.contains(&1));
    }

    #[test]
    fn test_checkpoint_new_and_clone() {
        let cp = Checkpoint::new("s", 3, 77, 999);
        let cp2 = cp.clone();
        assert_eq!(cp, cp2);
    }

    #[test]
    fn test_commit_same_offset_twice() {
        let mut store = CheckpointStore::new(10);
        store.commit(make_cp("dup", 0, 5, 1));
        store.commit(make_cp("dup", 0, 5, 2));
        // Should still work; latest offset unchanged
        assert_eq!(store.latest_offset("dup", 0), Some(5));
        assert_eq!(store.total_committed(), 2);
    }

    #[test]
    fn test_reset_to_zero() {
        let mut store = CheckpointStore::new(10);
        store.commit(make_cp("zero", 0, 50, 1));
        assert!(store.reset_to("zero", 0, 0));
        assert_eq!(store.latest_offset("zero", 0), Some(0));
    }
}
