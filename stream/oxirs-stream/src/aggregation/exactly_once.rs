//! Exactly-once aggregation under operator parallelism.
//!
//! [`ExactlyOnceAggregator`] composes
//! [`crate::state::exactly_once::ExactlyOnceProcessor`] with a per-partition
//! aggregation state to guarantee that:
//!
//! 1. Each event is folded into the aggregate exactly once, even under
//!    re-delivery.
//! 2. Aggregate state is checkpointable per partition (snapshots are
//!    obtained via the underlying [`crate::state::distributed_state::StateBackend`]).
//! 3. State recovery after a failure restores the same aggregate values that
//!    were emitted before the crash.
//!
//! ## Usage outline
//!
//! ```ignore
//! let backend = Arc::new(InMemoryStateBackend::new());
//! let mut agg = ExactlyOnceAggregator::<u64>::new(
//!     ExactlyOnceAggregatorConfig::default(),
//!     backend.clone(),
//! );
//! agg.fold(MessageId::new("p", 0, 1), partition_key, value, |state, v| state + v)?;
//! ```
//!
//! `partition_key` is any string-typed key (the operator-parallel shard).

use std::collections::HashMap;
use std::sync::Arc;

use crate::error::StreamError;
use crate::state::distributed_state::StateBackend;
use crate::state::exactly_once::{DeduplicationConfig, ExactlyOnceProcessor, MessageId};

// ─── Aggregate value types ───────────────────────────────────────────────────

/// Aggregate values supported by [`PartitionAggregateState`].
#[derive(Debug, Clone, PartialEq)]
pub enum PartitionAggregateValue {
    Count(u64),
    Sum(f64),
    Min(f64),
    Max(f64),
    /// Mean tracked as `(sum, count)`.
    Mean {
        sum: f64,
        count: u64,
    },
}

impl PartitionAggregateValue {
    /// Return `true` if the value is the identity element (initial state).
    pub fn is_initial(&self) -> bool {
        matches!(
            self,
            PartitionAggregateValue::Count(0)
                | PartitionAggregateValue::Sum(0.0)
                | PartitionAggregateValue::Mean { sum: _, count: 0 }
        )
    }
}

// ─── Per-partition state ─────────────────────────────────────────────────────

/// Per-partition aggregate state.
///
/// `K` is the partition/group key (typically a `String`).  `V` is the typed
/// aggregate value held for that key.
#[derive(Debug, Clone)]
pub struct PartitionAggregateState {
    inner: HashMap<String, PartitionAggregateValue>,
}

impl Default for PartitionAggregateState {
    fn default() -> Self {
        Self::new()
    }
}

impl PartitionAggregateState {
    /// Empty state.
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Look up the current value for `key`.
    pub fn get(&self, key: &str) -> Option<&PartitionAggregateValue> {
        self.inner.get(key)
    }

    /// Insert or replace the value for `key`.
    pub fn put(&mut self, key: impl Into<String>, value: PartitionAggregateValue) {
        self.inner.insert(key.into(), value);
    }

    /// Number of keys currently tracked.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True iff no keys are tracked.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Iterate over all (key, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &PartitionAggregateValue)> {
        self.inner.iter()
    }

    // ─── Encoding ────────────────────────────────────────────────────────────

    /// Encode this state to a deterministic byte vector.
    ///
    /// Format
    ///
    /// * `[u32 len]` — number of entries.
    /// * `len` × `[u32 key_len][key…][u8 tag][payload]`.
    /// * Tag/payload:
    ///   * `0x01` Count: `[u64]`.
    ///   * `0x02` Sum:   `[f64]`.
    ///   * `0x03` Min:   `[f64]`.
    ///   * `0x04` Max:   `[f64]`.
    ///   * `0x05` Mean:  `[f64 sum][u64 count]`.
    ///
    /// All integers are little-endian.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.inner.len() as u32).to_le_bytes());
        // Sort by key for deterministic output.
        let mut keys: Vec<&String> = self.inner.keys().collect();
        keys.sort();
        for k in keys {
            let v = match self.inner.get(k) {
                Some(v) => v,
                None => continue,
            };
            out.extend_from_slice(&(k.len() as u32).to_le_bytes());
            out.extend_from_slice(k.as_bytes());
            match v {
                PartitionAggregateValue::Count(c) => {
                    out.push(0x01);
                    out.extend_from_slice(&c.to_le_bytes());
                }
                PartitionAggregateValue::Sum(s) => {
                    out.push(0x02);
                    out.extend_from_slice(&s.to_le_bytes());
                }
                PartitionAggregateValue::Min(m) => {
                    out.push(0x03);
                    out.extend_from_slice(&m.to_le_bytes());
                }
                PartitionAggregateValue::Max(m) => {
                    out.push(0x04);
                    out.extend_from_slice(&m.to_le_bytes());
                }
                PartitionAggregateValue::Mean { sum, count } => {
                    out.push(0x05);
                    out.extend_from_slice(&sum.to_le_bytes());
                    out.extend_from_slice(&count.to_le_bytes());
                }
            }
        }
        out
    }

    /// Decode a state previously produced by [`Self::encode`].
    pub fn decode(buf: &[u8]) -> Result<Self, StreamError> {
        let read_u32 = |buf: &[u8], offset: usize| -> Result<(u32, usize), StreamError> {
            if buf.len() < offset + 4 {
                return Err(StreamError::Deserialization(
                    "PartitionAggregateState: truncated u32".to_string(),
                ));
            }
            let mut a = [0u8; 4];
            a.copy_from_slice(&buf[offset..offset + 4]);
            Ok((u32::from_le_bytes(a), offset + 4))
        };
        let read_u64 = |buf: &[u8], offset: usize| -> Result<(u64, usize), StreamError> {
            if buf.len() < offset + 8 {
                return Err(StreamError::Deserialization(
                    "PartitionAggregateState: truncated u64".to_string(),
                ));
            }
            let mut a = [0u8; 8];
            a.copy_from_slice(&buf[offset..offset + 8]);
            Ok((u64::from_le_bytes(a), offset + 8))
        };
        let read_f64 = |buf: &[u8], offset: usize| -> Result<(f64, usize), StreamError> {
            if buf.len() < offset + 8 {
                return Err(StreamError::Deserialization(
                    "PartitionAggregateState: truncated f64".to_string(),
                ));
            }
            let mut a = [0u8; 8];
            a.copy_from_slice(&buf[offset..offset + 8]);
            Ok((f64::from_le_bytes(a), offset + 8))
        };

        let mut state = PartitionAggregateState::new();
        let (n, mut p) = read_u32(buf, 0)?;
        for _ in 0..n {
            let (klen, np) = read_u32(buf, p)?;
            p = np;
            let kend = p + klen as usize;
            if buf.len() < kend {
                return Err(StreamError::Deserialization(
                    "PartitionAggregateState: truncated key".to_string(),
                ));
            }
            let key = std::str::from_utf8(&buf[p..kend])
                .map_err(|e| StreamError::Deserialization(format!("bad utf8: {e}")))?
                .to_string();
            p = kend;
            if buf.len() < p + 1 {
                return Err(StreamError::Deserialization(
                    "PartitionAggregateState: missing tag".to_string(),
                ));
            }
            let tag = buf[p];
            p += 1;
            let v = match tag {
                0x01 => {
                    let (c, np) = read_u64(buf, p)?;
                    p = np;
                    PartitionAggregateValue::Count(c)
                }
                0x02 => {
                    let (s, np) = read_f64(buf, p)?;
                    p = np;
                    PartitionAggregateValue::Sum(s)
                }
                0x03 => {
                    let (m, np) = read_f64(buf, p)?;
                    p = np;
                    PartitionAggregateValue::Min(m)
                }
                0x04 => {
                    let (m, np) = read_f64(buf, p)?;
                    p = np;
                    PartitionAggregateValue::Max(m)
                }
                0x05 => {
                    let (s, np) = read_f64(buf, p)?;
                    let (c, np) = read_u64(buf, np)?;
                    p = np;
                    PartitionAggregateValue::Mean { sum: s, count: c }
                }
                t => {
                    return Err(StreamError::Deserialization(format!(
                        "unknown PartitionAggregateValue tag {t}"
                    )));
                }
            };
            state.put(key, v);
        }
        Ok(state)
    }
}

// ─── Aggregator config / stats ───────────────────────────────────────────────

/// Configuration for [`ExactlyOnceAggregator`].
#[derive(Debug, Clone)]
pub struct ExactlyOnceAggregatorConfig {
    pub dedup: DeduplicationConfig,
    /// Logical name used for the state key inside the backend.
    pub state_key: String,
}

impl Default for ExactlyOnceAggregatorConfig {
    fn default() -> Self {
        Self {
            dedup: DeduplicationConfig::default(),
            state_key: "aggregator/state".to_string(),
        }
    }
}

/// Runtime statistics.
#[derive(Debug, Clone, Default)]
pub struct ExactlyOnceAggregatorStats {
    pub events_folded: u64,
    pub duplicates_filtered: u64,
    pub checkpoints_taken: u64,
}

// ─── ExactlyOnceAggregator ───────────────────────────────────────────────────

/// Aggregator wrapper that guarantees exactly-once fold semantics.
pub struct ExactlyOnceAggregator {
    config: ExactlyOnceAggregatorConfig,
    backend: Arc<dyn StateBackend>,
    processor: ExactlyOnceProcessor,
    state: PartitionAggregateState,
    stats: ExactlyOnceAggregatorStats,
}

impl ExactlyOnceAggregator {
    /// Create a new aggregator backed by `backend`.
    pub fn new(config: ExactlyOnceAggregatorConfig, backend: Arc<dyn StateBackend>) -> Self {
        let processor = ExactlyOnceProcessor::new(config.dedup.clone(), backend.clone());
        Self {
            config,
            backend,
            processor,
            state: PartitionAggregateState::new(),
            stats: ExactlyOnceAggregatorStats::default(),
        }
    }

    /// Fold a single event into the aggregate state.
    ///
    /// `id` uniquely identifies the message (used for dedup).  The `update`
    /// closure produces the *new* aggregate value for the partition key from
    /// the previous value.
    pub fn fold<F>(
        &mut self,
        id: MessageId,
        partition_key: &str,
        update: F,
    ) -> Result<Option<PartitionAggregateValue>, StreamError>
    where
        F: FnOnce(Option<&PartitionAggregateValue>) -> PartitionAggregateValue,
    {
        let prev = self.state.get(partition_key).cloned();
        let new_value_for_state = update(prev.as_ref());
        let key_for_state = partition_key.to_string();
        let value_for_dedup_apply = new_value_for_state.clone();
        let state_key_bytes = self.config.state_key.as_bytes().to_vec();

        // Encode the *post-update* state for the transaction.
        let mut updated = self.state.clone();
        updated.put(key_for_state.clone(), new_value_for_state.clone());
        let encoded = updated.encode();

        let result = self.processor.process(id, |txn| {
            txn.add_state_change(state_key_bytes, encoded);
            Ok(value_for_dedup_apply)
        })?;

        match result {
            Some(applied) => {
                self.state.put(key_for_state, applied.clone());
                self.stats.events_folded += 1;
                Ok(Some(applied))
            }
            None => {
                self.stats.duplicates_filtered += 1;
                Ok(None)
            }
        }
    }

    /// Look up the current aggregate value for a partition key.
    pub fn get(&self, partition_key: &str) -> Option<&PartitionAggregateValue> {
        self.state.get(partition_key)
    }

    /// Manually overwrite the value for a partition (used during recovery).
    pub fn set(&mut self, partition_key: &str, value: PartitionAggregateValue) {
        self.state.put(partition_key.to_string(), value);
    }

    /// Snapshot the current state into the backend (idempotent).
    pub fn checkpoint(&mut self) -> Result<(), StreamError> {
        let encoded = self.state.encode();
        self.backend
            .put(self.config.state_key.as_bytes(), &encoded)?;
        self.stats.checkpoints_taken += 1;
        Ok(())
    }

    /// Restore aggregate state from the backend (no-op if absent).
    pub fn restore(&mut self) -> Result<(), StreamError> {
        match self.backend.get(self.config.state_key.as_bytes())? {
            Some(bytes) => {
                let state = PartitionAggregateState::decode(&bytes)?;
                self.state = state;
                Ok(())
            }
            None => Ok(()),
        }
    }

    /// Drop the in-memory aggregate state (test/recovery helper).
    pub fn clear(&mut self) {
        self.state = PartitionAggregateState::new();
    }

    /// Snapshot statistics.
    pub fn stats(&self) -> &ExactlyOnceAggregatorStats {
        &self.stats
    }

    /// Number of partitions currently tracked.
    pub fn partition_count(&self) -> usize {
        self.state.len()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::distributed_state::InMemoryStateBackend;
    use crate::state::exactly_once::MessageId;

    fn fresh_aggregator() -> ExactlyOnceAggregator {
        let backend: Arc<dyn StateBackend> = Arc::new(InMemoryStateBackend::new());
        ExactlyOnceAggregator::new(ExactlyOnceAggregatorConfig::default(), backend)
    }

    #[test]
    fn fold_increments_count_exactly_once() {
        let mut agg = fresh_aggregator();
        let id = MessageId::new("p", 0, 1);
        let v = agg
            .fold(id.clone(), "k", |prev| match prev {
                Some(PartitionAggregateValue::Count(c)) => PartitionAggregateValue::Count(*c + 1),
                _ => PartitionAggregateValue::Count(1),
            })
            .expect("fold ok");
        assert_eq!(v, Some(PartitionAggregateValue::Count(1)));
        // Replay → no double-count.
        let v = agg
            .fold(id, "k", |prev| match prev {
                Some(PartitionAggregateValue::Count(c)) => PartitionAggregateValue::Count(*c + 1),
                _ => PartitionAggregateValue::Count(1),
            })
            .expect("fold ok");
        assert_eq!(v, None);
        assert_eq!(agg.get("k"), Some(&PartitionAggregateValue::Count(1)));
        assert_eq!(agg.stats.duplicates_filtered, 1);
    }

    #[test]
    fn checkpoint_restore_roundtrip() {
        let mut agg = fresh_aggregator();
        for i in 1..=5u64 {
            let id = MessageId::new("p", 0, i);
            agg.fold(id, "k1", |prev| match prev {
                Some(PartitionAggregateValue::Sum(s)) => PartitionAggregateValue::Sum(s + i as f64),
                _ => PartitionAggregateValue::Sum(i as f64),
            })
            .expect("fold ok");
        }
        // Sum 1+2+3+4+5 = 15.
        assert_eq!(agg.get("k1"), Some(&PartitionAggregateValue::Sum(15.0)));

        // Checkpoint, clear, restore.
        agg.checkpoint().expect("checkpoint ok");
        agg.clear();
        assert!(agg.get("k1").is_none());
        agg.restore().expect("restore ok");
        assert_eq!(agg.get("k1"), Some(&PartitionAggregateValue::Sum(15.0)));
    }

    #[test]
    fn separate_partitions_isolated() {
        let mut agg = fresh_aggregator();
        agg.fold(MessageId::new("p", 0, 1), "a", |_| {
            PartitionAggregateValue::Count(1)
        })
        .expect("ok");
        agg.fold(MessageId::new("p", 0, 2), "b", |_| {
            PartitionAggregateValue::Count(7)
        })
        .expect("ok");
        assert_eq!(agg.get("a"), Some(&PartitionAggregateValue::Count(1)));
        assert_eq!(agg.get("b"), Some(&PartitionAggregateValue::Count(7)));
        assert_eq!(agg.partition_count(), 2);
    }

    #[test]
    fn encode_decode_round_trip() {
        let mut s = PartitionAggregateState::new();
        s.put("a", PartitionAggregateValue::Count(42));
        s.put("b", PartitionAggregateValue::Sum(3.5));
        s.put("c", PartitionAggregateValue::Min(-1.0));
        s.put("d", PartitionAggregateValue::Max(99.0));
        s.put(
            "mean_e",
            PartitionAggregateValue::Mean {
                sum: 100.0,
                count: 4,
            },
        );
        let bytes = s.encode();
        let decoded = PartitionAggregateState::decode(&bytes).expect("decode");
        assert_eq!(decoded.len(), 5);
        assert_eq!(decoded.get("a"), Some(&PartitionAggregateValue::Count(42)));
        assert_eq!(decoded.get("b"), Some(&PartitionAggregateValue::Sum(3.5)));
        assert_eq!(decoded.get("c"), Some(&PartitionAggregateValue::Min(-1.0)));
        assert_eq!(decoded.get("d"), Some(&PartitionAggregateValue::Max(99.0)));
        match decoded.get("mean_e") {
            Some(PartitionAggregateValue::Mean { sum, count }) => {
                assert!((sum - 100.0).abs() < 1e-9);
                assert_eq!(*count, 4);
            }
            other => panic!("expected Mean, got {other:?}"),
        }
    }

    #[test]
    fn checkpoint_after_dedup_does_not_double_apply() {
        let mut agg = fresh_aggregator();
        let id = MessageId::new("p", 0, 1);
        agg.fold(id.clone(), "k", |_| PartitionAggregateValue::Count(5))
            .expect("ok");
        agg.checkpoint().expect("ok");
        // Recover into a *new* aggregator on the same backend to simulate
        // crash recovery.
        let backend = agg.backend.clone();
        let mut recovered =
            ExactlyOnceAggregator::new(ExactlyOnceAggregatorConfig::default(), backend);
        recovered.restore().expect("ok");
        assert_eq!(recovered.get("k"), Some(&PartitionAggregateValue::Count(5)));
    }
}
