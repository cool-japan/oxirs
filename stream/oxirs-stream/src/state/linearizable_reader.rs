//! # Linearizable reads on Raft-backed operator state
//!
//! Reads served from the local cache reflect *committed* state (the cache is
//! only updated after the underlying Raft proposal succeeds). For workloads
//! that need a true linearizable read — i.e. "see every value committed before
//! the read started" — we route the read through the cluster's
//! [`oxirs_cluster::consensus::ConsensusManager`].
//!
//! ## Two read paths
//!
//! [`LinearizableReader`] exposes two flavours of read:
//!
//! * [`LinearizableReader::get`] — the **leader-stickiness** path. It checks
//!   that the local node is the leader, snapshots the current term, then
//!   serves from the local cache (which reflects every committed put issued
//!   on this node). Cheap; guaranteed monotonic when reading from the leader.
//! * [`LinearizableReader::get_with_barrier`] — the **strict barrier** path.
//!   It issues a no-op `BeginTransaction` / `RollbackTransaction` pair through
//!   [`ConsensusManager::propose_command`] before reading. The pair is a
//!   round-trip through Raft, so by the time it returns every commit issued
//!   before this call has been applied. Strictly stronger than the
//!   leader-stickiness path; pays one Raft round-trip.
//!
//! ## Concurrency window
//!
//! A pure leader-stickiness check (`get`) has a concurrency window between
//! `is_leader()` returning `true` and the cache read in which the local node
//! could lose leadership. Callers that cannot tolerate that window must use
//! `get_with_barrier`.
//!
//! ## Why we don't add a dedicated `RdfCommand::Barrier`
//!
//! The cluster's `RdfCommand` enum is shared with the rest of the cluster
//! crate and adding a fresh variant is a larger ABI change than this slice
//! owns. Begin/Rollback is an existing, idempotent no-op pair that the
//! state machine already understands.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::debug;

use oxirs_cluster::consensus::ConsensusManager;

use super::raft_state::{RaftBackedOperatorState, StateValue};

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors raised by [`LinearizableReader`].
#[derive(Debug, Error)]
pub enum LinearizableReadError {
    /// Local node is not the cluster leader and `require_leader` is set.
    #[error("local node is not the leader")]
    NotLeader,
    /// Underlying state error.
    #[error("state error: {0}")]
    State(String),
}

/// Convenience alias.
pub type LinearizableReadResult<T> = std::result::Result<T, LinearizableReadError>;

// ─── Reader ────────────────────────────────────────────────────────────────

/// Configuration for [`LinearizableReader`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearizableReadConfig {
    /// If `true`, [`LinearizableReader::get`] errors with [`LinearizableReadError::NotLeader`]
    /// when the local node is not the leader. Defaults to `true` so reads are
    /// served exclusively from the leader's view.
    pub require_leader: bool,
}

impl Default for LinearizableReadConfig {
    fn default() -> Self {
        Self {
            require_leader: true,
        }
    }
}

/// A read accessor that goes through Raft consensus before returning a value.
pub struct LinearizableReader {
    consensus: Arc<ConsensusManager>,
    state: Arc<RaftBackedOperatorState>,
    config: LinearizableReadConfig,
}

impl LinearizableReader {
    /// Build a reader.
    pub fn new(
        consensus: Arc<ConsensusManager>,
        state: Arc<RaftBackedOperatorState>,
        config: LinearizableReadConfig,
    ) -> Self {
        Self {
            consensus,
            state,
            config,
        }
    }

    /// Returns `true` if the local node currently believes itself the leader.
    pub async fn is_leader(&self) -> bool {
        self.consensus.is_leader().await
    }

    /// Returns the latest known consensus term (for diagnostics).
    pub async fn term(&self) -> u64 {
        self.consensus.current_term().await
    }

    /// Leader-stickiness read of a key.
    ///
    /// Cheap read path: checks that the local node is the leader, snapshots
    /// the current term, then serves from the local cache. The cache only
    /// reflects committed state, so this is monotonic when read from the
    /// leader. For strictly linearizable reads under contention, use
    /// [`Self::get_with_barrier`].
    pub async fn get(&self, key: &str) -> LinearizableReadResult<Option<StateValue>> {
        let term = self.consensus.current_term().await;
        if self.config.require_leader && !self.consensus.is_leader().await {
            return Err(LinearizableReadError::NotLeader);
        }
        debug!(term, key, "linearizable read served (sticky)");
        Ok(self.state.get_local(key))
    }

    /// Strict linearizable read of a key.
    ///
    /// Issues a `BeginTransaction` / `RollbackTransaction` pair through
    /// `ConsensusManager::propose_command` before reading the local cache.
    /// The pair is a Raft round-trip, so on return every commit issued
    /// before this call has been applied to the local replica.
    ///
    /// Costs one Raft round-trip per call; use sparingly.
    pub async fn get_with_barrier(&self, key: &str) -> LinearizableReadResult<Option<StateValue>> {
        if self.config.require_leader && !self.consensus.is_leader().await {
            return Err(LinearizableReadError::NotLeader);
        }
        let tx_id = format!("oxirs-stream-barrier-{}", uuid::Uuid::new_v4());
        // Begin → rollback is an idempotent no-op as far as the state machine
        // is concerned, but it forces the cluster sink to commit the round
        // through Raft, giving us our barrier.
        self.consensus
            .begin_transaction(tx_id.clone())
            .await
            .map_err(|e| LinearizableReadError::State(e.to_string()))?;
        self.consensus
            .rollback_transaction(tx_id)
            .await
            .map_err(|e| LinearizableReadError::State(e.to_string()))?;
        Ok(self.state.get_local(key))
    }

    /// Underlying operator state handle.
    pub fn state(&self) -> &Arc<RaftBackedOperatorState> {
        &self.state
    }

    /// Underlying consensus manager handle.
    pub fn consensus(&self) -> &Arc<ConsensusManager> {
        &self.consensus
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::raft_state::{RaftBackedOperatorState, RaftStateConfig, StateValue};
    use async_trait::async_trait;
    use oxirs_cluster::stream_integration::StreamMessage;
    use oxirs_cluster::streaming::cluster_sink::{SinkError, StreamSink};
    use parking_lot::Mutex;

    #[derive(Default)]
    struct PassThroughSink {
        committed: Mutex<u64>,
    }

    #[async_trait]
    impl StreamSink for PassThroughSink {
        async fn write_batch(&self, _events: Vec<StreamMessage>) -> Result<(), SinkError> {
            *self.committed.lock() += 1;
            Ok(())
        }
    }

    #[tokio::test]
    async fn read_returns_locally_cached_value_when_not_requiring_leader() {
        let consensus = Arc::new(ConsensusManager::new(1, vec![]));
        let sink = Arc::new(PassThroughSink::default());
        let state = Arc::new(RaftBackedOperatorState::new(
            RaftStateConfig {
                operator_id: "lin-test".into(),
                stream_id: None,
            },
            sink,
        ));
        state.put("k", StateValue::Counter(11)).await.expect("put");

        let reader = LinearizableReader::new(
            consensus,
            state,
            LinearizableReadConfig {
                require_leader: false,
            },
        );
        let v = reader.get("k").await.expect("ok");
        assert_eq!(v, Some(StateValue::Counter(11)));
    }

    #[tokio::test]
    async fn missing_key_returns_none() {
        let consensus = Arc::new(ConsensusManager::new(2, vec![]));
        let sink = Arc::new(PassThroughSink::default());
        let state = Arc::new(RaftBackedOperatorState::new(
            RaftStateConfig {
                operator_id: "lin-test".into(),
                stream_id: None,
            },
            sink,
        ));
        let reader = LinearizableReader::new(
            consensus,
            state,
            LinearizableReadConfig {
                require_leader: false,
            },
        );
        let v = reader.get("ghost").await.expect("ok");
        assert!(v.is_none());
    }

    #[tokio::test]
    async fn get_with_barrier_succeeds_on_leader_after_put() {
        use oxirs_cluster::raft::{init_global_shared_storage, reset_global_shared_storage};
        static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());
        let _g = TEST_LOCK.lock().await;
        init_global_shared_storage();
        reset_global_shared_storage().await;

        let consensus = Arc::new(ConsensusManager::new(11, vec![]));
        let sink = Arc::new(PassThroughSink::default());
        let state = Arc::new(RaftBackedOperatorState::new(
            RaftStateConfig {
                operator_id: "lin-barrier".into(),
                stream_id: None,
            },
            sink,
        ));
        state
            .put("counter", StateValue::Counter(7))
            .await
            .expect("put");

        let reader = LinearizableReader::new(
            consensus,
            state,
            LinearizableReadConfig {
                require_leader: true,
            },
        );
        let v = reader.get_with_barrier("counter").await.expect("ok");
        assert_eq!(v, Some(StateValue::Counter(7)));
    }

    #[tokio::test]
    async fn require_leader_true_serves_when_local_is_leader() {
        // The non-raft fallback in `RaftNode::is_leader` returns true, so
        // single-node test setups exercise the require_leader=true happy path.
        let consensus = Arc::new(ConsensusManager::new(10, vec![]));
        let sink = Arc::new(PassThroughSink::default());
        let state = Arc::new(RaftBackedOperatorState::new(
            RaftStateConfig {
                operator_id: "lin-test".into(),
                stream_id: None,
            },
            sink,
        ));
        state.put("k", StateValue::Counter(99)).await.expect("put");
        let reader = LinearizableReader::new(
            consensus,
            state,
            LinearizableReadConfig {
                require_leader: true,
            },
        );
        assert!(reader.is_leader().await);
        let v = reader.get("k").await.expect("ok");
        assert_eq!(v, Some(StateValue::Counter(99)));
    }

    #[tokio::test]
    async fn term_query_returns_finite_value() {
        let consensus = Arc::new(ConsensusManager::new(3, vec![]));
        let sink = Arc::new(PassThroughSink::default());
        let state = Arc::new(RaftBackedOperatorState::new(
            RaftStateConfig {
                operator_id: "lin-test".into(),
                stream_id: None,
            },
            sink,
        ));
        let reader = LinearizableReader::new(consensus, state, LinearizableReadConfig::default());
        // Smoke test: term should always be representable as u64.
        let _ = reader.term().await;
    }
}
