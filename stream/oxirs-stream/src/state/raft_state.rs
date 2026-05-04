//! # Raft-backed operator state
//!
//! Bridges stream operator state (windowed aggregates, joins) onto the cluster
//! Raft log. Each *update* to the state goes out as an `RdfCommand::Insert`
//! through the W3-S9 [`oxirs_cluster::streaming::cluster_sink::ClusterSink`]
//! and a *snapshot* of the latest committed value is held in memory for fast
//! local reads. Linearizable reads route through [`super::linearizable_reader`].
//!
//! ## Encoding of operator state into RDF triples
//!
//! Operator state is encoded into a synthetic RDF triple namespace shared with
//! the cluster's existing `RdfApp`:
//!
//! - **Subject**: `oxirs://stream-state/{operator_id}/{key}`
//! - **Predicate**: `http://oxirs.dev/stream-state#value`
//! - **Object**: a JSON literal carrying the serialized [`StateValue`] payload.
//!
//! This is intentionally additive: existing cluster query paths see the
//! triples without any state-machine extension, and operators can query their
//! own state through the same Raft-replicated store.
//!
//! ## Concurrency
//!
//! [`RaftBackedOperatorState`] is `Send + Sync` and uses fine-grained per-key
//! locking (`DashMap` shards) for the local cache. Writes go through the
//! cluster sink (Raft) and update the local cache only after the proposal
//! succeeds, so cached values reflect committed state.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn};

use oxirs_cluster::stream_integration::{StreamMessage, StreamTriple};
use oxirs_cluster::streaming::cluster_sink::{SinkError, StreamSink};

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors raised by [`RaftBackedOperatorState`].
#[derive(Debug, Error)]
pub enum RaftStateError {
    /// Underlying [`StreamSink`] returned an error.
    #[error("sink error: {0}")]
    Sink(String),
    /// Failed to serialise/deserialise a [`StateValue`].
    #[error("encoding error: {0}")]
    Encoding(String),
    /// Caller asked for a key that does not exist.
    #[error("unknown key: {0}")]
    UnknownKey(String),
}

impl From<SinkError> for RaftStateError {
    fn from(err: SinkError) -> Self {
        RaftStateError::Sink(err.to_string())
    }
}

/// Convenience alias.
pub type RaftStateResult<T> = std::result::Result<T, RaftStateError>;

// ─── StateValue ─────────────────────────────────────────────────────────────

/// Operator-state value type. Held both in the local cache and as the JSON
/// payload of the Raft log entries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StateValue {
    /// 64-bit signed counter.
    Counter(i64),
    /// Floating-point gauge.
    Gauge(f64),
    /// Arbitrary string blob.
    Text(String),
    /// Byte blob (base64 in the JSON payload).
    Bytes(Vec<u8>),
    /// Structured JSON value (passed through verbatim).
    Json(serde_json::Value),
}

impl StateValue {
    fn encode(&self) -> RaftStateResult<String> {
        serde_json::to_string(self).map_err(|e| RaftStateError::Encoding(e.to_string()))
    }

    fn decode(s: &str) -> RaftStateResult<Self> {
        serde_json::from_str(s).map_err(|e| RaftStateError::Encoding(e.to_string()))
    }
}

// ─── Stats ─────────────────────────────────────────────────────────────────

/// Runtime statistics for [`RaftBackedOperatorState`].
#[derive(Debug, Default)]
pub struct RaftStateStats {
    /// Total `put` calls received.
    pub puts_received: AtomicU64,
    /// `put` calls that successfully committed through the sink.
    pub puts_committed: AtomicU64,
    /// `put` calls that failed at the sink layer.
    pub puts_failed: AtomicU64,
    /// `get` calls served from local cache.
    pub local_gets: AtomicU64,
    /// `delete` calls received.
    pub deletes_received: AtomicU64,
    /// `delete` calls that committed.
    pub deletes_committed: AtomicU64,
}

impl RaftStateStats {
    /// Snapshot the current counters into a serializable shape.
    pub fn snapshot(&self) -> RaftStateStatsSnapshot {
        RaftStateStatsSnapshot {
            puts_received: self.puts_received.load(Ordering::Relaxed),
            puts_committed: self.puts_committed.load(Ordering::Relaxed),
            puts_failed: self.puts_failed.load(Ordering::Relaxed),
            local_gets: self.local_gets.load(Ordering::Relaxed),
            deletes_received: self.deletes_received.load(Ordering::Relaxed),
            deletes_committed: self.deletes_committed.load(Ordering::Relaxed),
        }
    }
}

/// Plain serializable snapshot of [`RaftStateStats`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RaftStateStatsSnapshot {
    pub puts_received: u64,
    pub puts_committed: u64,
    pub puts_failed: u64,
    pub local_gets: u64,
    pub deletes_received: u64,
    pub deletes_committed: u64,
}

// ─── RaftBackedOperatorState ───────────────────────────────────────────────

/// Configuration for [`RaftBackedOperatorState`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaftStateConfig {
    /// Logical operator identifier; included in the encoded RDF subject so
    /// multiple operators can share the same Raft log without collisions.
    pub operator_id: String,
    /// Optional stream identifier exposed to the sink (otherwise
    /// `"stream-state-{operator_id}"`).
    pub stream_id: Option<String>,
}

impl RaftStateConfig {
    fn stream_id(&self) -> String {
        self.stream_id
            .clone()
            .unwrap_or_else(|| format!("stream-state-{}", self.operator_id))
    }

    fn subject_for(&self, key: &str) -> String {
        format!("oxirs://stream-state/{}/{key}", self.operator_id)
    }
}

/// Raft-backed operator state.
pub struct RaftBackedOperatorState {
    config: RaftStateConfig,
    sink: Arc<dyn StreamSink>,
    local: DashMap<String, StateValue>,
    offset: AtomicU64,
    stats: Arc<RaftStateStats>,
}

impl RaftBackedOperatorState {
    /// Create a new state instance.
    pub fn new(config: RaftStateConfig, sink: Arc<dyn StreamSink>) -> Self {
        Self {
            config,
            sink,
            local: DashMap::new(),
            offset: AtomicU64::new(0),
            stats: Arc::new(RaftStateStats::default()),
        }
    }

    /// Configured operator identifier.
    pub fn operator_id(&self) -> &str {
        &self.config.operator_id
    }

    /// Reference to the stats snapshot.
    pub fn stats(&self) -> &Arc<RaftStateStats> {
        &self.stats
    }

    /// Get the current locally-cached value for `key`. The cache reflects the
    /// last *committed* value; for linearizable reads, see
    /// [`super::linearizable_reader::LinearizableReader`].
    pub fn get_local(&self, key: &str) -> Option<StateValue> {
        self.stats.local_gets.fetch_add(1, Ordering::Relaxed);
        self.local.get(key).map(|v| v.value().clone())
    }

    /// Put the latest value for `key`, routing it through Raft.
    pub async fn put(&self, key: &str, value: StateValue) -> RaftStateResult<()> {
        self.stats.puts_received.fetch_add(1, Ordering::Relaxed);
        let object = format!("\"{}\"", escape_quotes(&value.encode()?));
        let triple = StreamTriple::new(
            self.config.subject_for(key),
            "http://oxirs.dev/stream-state#value",
            object,
        );
        let off = self.offset.fetch_add(1, Ordering::Relaxed) + 1;
        let msg = StreamMessage::insert(self.config.stream_id(), off, vec![triple]);
        match self.sink.write_batch(vec![msg]).await {
            Ok(()) => {
                self.local.insert(key.to_string(), value);
                self.stats.puts_committed.fetch_add(1, Ordering::Relaxed);
                debug!(
                    operator = %self.config.operator_id,
                    key,
                    offset = off,
                    "RaftBackedOperatorState: put committed"
                );
                Ok(())
            }
            Err(err) => {
                self.stats.puts_failed.fetch_add(1, Ordering::Relaxed);
                warn!(error = %err, "put failed");
                Err(err.into())
            }
        }
    }

    /// Delete a key from operator state.
    pub async fn delete(&self, key: &str) -> RaftStateResult<()> {
        self.stats.deletes_received.fetch_add(1, Ordering::Relaxed);
        let object = "\"\"".to_string();
        let triple = StreamTriple::new(
            self.config.subject_for(key),
            "http://oxirs.dev/stream-state#value",
            object,
        );
        let off = self.offset.fetch_add(1, Ordering::Relaxed) + 1;
        let msg = StreamMessage::delete(self.config.stream_id(), off, vec![triple]);
        match self.sink.write_batch(vec![msg]).await {
            Ok(()) => {
                self.local.remove(key);
                self.stats.deletes_committed.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(err) => Err(err.into()),
        }
    }

    /// Iterate the keys currently cached locally.
    pub fn keys(&self) -> Vec<String> {
        self.local.iter().map(|e| e.key().clone()).collect()
    }

    /// Number of keys held locally.
    pub fn len(&self) -> usize {
        self.local.len()
    }

    /// True if the local cache is empty.
    pub fn is_empty(&self) -> bool {
        self.local.is_empty()
    }

    /// Replay an externally-committed encoded value (e.g. from a Raft snapshot
    /// during recovery) into the local cache. Used by
    /// [`super::linearizable_reader::LinearizableReader`].
    pub fn restore_from_encoded(&self, key: &str, payload: &str) -> RaftStateResult<()> {
        let value = StateValue::decode(payload)?;
        self.local.insert(key.to_string(), value);
        Ok(())
    }
}

fn escape_quotes(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use parking_lot::Mutex;
    use std::collections::VecDeque;

    #[derive(Default)]
    struct MockSink {
        batches: Mutex<Vec<Vec<StreamMessage>>>,
        next_errors: Mutex<VecDeque<SinkError>>,
    }

    #[async_trait]
    impl StreamSink for MockSink {
        async fn write_batch(&self, events: Vec<StreamMessage>) -> Result<(), SinkError> {
            if let Some(err) = self.next_errors.lock().pop_front() {
                return Err(err);
            }
            self.batches.lock().push(events);
            Ok(())
        }
    }

    fn make_state() -> (Arc<RaftBackedOperatorState>, Arc<MockSink>) {
        let sink = Arc::new(MockSink::default());
        let cfg = RaftStateConfig {
            operator_id: "agg".into(),
            stream_id: Some("agg-state".into()),
        };
        let state = Arc::new(RaftBackedOperatorState::new(cfg, sink.clone()));
        (state, sink)
    }

    #[tokio::test]
    async fn put_and_get_round_trip() {
        let (state, sink) = make_state();
        state
            .put("count", StateValue::Counter(42))
            .await
            .expect("put");
        let snap = state.get_local("count").expect("hit");
        assert_eq!(snap, StateValue::Counter(42));
        assert_eq!(sink.batches.lock().len(), 1);
        let stats = state.stats().snapshot();
        assert_eq!(stats.puts_received, 1);
        assert_eq!(stats.puts_committed, 1);
    }

    #[tokio::test]
    async fn put_failure_does_not_update_cache() {
        let (state, sink) = make_state();
        sink.next_errors.lock().push_back(SinkError::NotLeader);
        let err = state
            .put("count", StateValue::Counter(99))
            .await
            .expect_err("should fail");
        assert!(matches!(err, RaftStateError::Sink(_)));
        assert!(state.get_local("count").is_none());
        let stats = state.stats().snapshot();
        assert_eq!(stats.puts_committed, 0);
        assert_eq!(stats.puts_failed, 1);
    }

    #[tokio::test]
    async fn delete_removes_key_from_cache() {
        let (state, _sink) = make_state();
        state
            .put("k1", StateValue::Text("hello".into()))
            .await
            .expect("put");
        state.delete("k1").await.expect("delete");
        assert!(state.get_local("k1").is_none());
        let stats = state.stats().snapshot();
        assert_eq!(stats.deletes_committed, 1);
    }

    #[tokio::test]
    async fn restore_from_encoded_populates_cache() {
        let (state, _sink) = make_state();
        state
            .restore_from_encoded("counter", "{\"Counter\":7}")
            .expect("restore");
        let v = state.get_local("counter").expect("hit");
        assert_eq!(v, StateValue::Counter(7));
    }

    #[test]
    fn encode_decode_state_value() {
        let value = StateValue::Json(serde_json::json!({"a": 1, "b": [true, null]}));
        let s = value.encode().expect("encode");
        let back = StateValue::decode(&s).expect("decode");
        assert_eq!(value, back);
    }
}
