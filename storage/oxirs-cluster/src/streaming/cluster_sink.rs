//! [`StreamSink`] trait + [`ClusterSink`] production implementation.
//!
//! A [`ClusterSink`] is the bridge between an external streaming source and
//! the cluster's Raft log. Each batch of [`StreamMessage`] events is
//! translated through an [`EventMapper`] into [`crate::raft::RdfCommand`] entries which
//! are then proposed to the leader through the [`ConsensusManager`]. Once
//! committed, read replicas serve queries against the resulting `RdfApp`
//! state.
//!
//! ## Design notes
//!
//! * The sink trait is intentionally **local** to `oxirs-cluster`. It avoids
//!   pulling `oxirs-stream` into the cluster's dependency closure (which
//!   would create a cycle once W3-S11 wires the producer side from the
//!   `oxirs-stream` direction).
//! * Backpressure is exposed through a [`BackpressureBridge`]. The sink
//!   adjusts the bridge's queue depth as it accepts and completes batches;
//!   upstream operators read [`BackpressureBridge::signal`] and respond.
//! * If the local node is **not the leader**, the sink returns
//!   [`SinkError::NotLeader`] without losing the batch — the caller is
//!   expected to redirect or retry (matching `ClusterNode::insert_triple`
//!   semantics).
//! * [`ClusterSinkConfig::max_batch_commands`] caps the number of `RdfCommand`
//!   entries proposed per `write_batch` call to bound the time the sink
//!   spends inside a single proposal cycle.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn};

use crate::consensus::ConsensusManager;
use crate::stream_integration::StreamMessage;
use crate::streaming::backpressure_bridge::{BackpressureBridge, BackpressureSignal};
use crate::streaming::event_mapper::{EventMapper, MapperError};

/// Errors returned by [`StreamSink::write_batch`].
#[derive(Debug, Error)]
pub enum SinkError {
    /// This node is not the cluster leader; the batch was not proposed.
    #[error("local node is not the leader; cannot accept stream batch")]
    NotLeader,
    /// The cluster is in `Stop` backpressure state.
    #[error("cluster backpressure is Stop; refusing batch")]
    BackpressureStopped,
    /// Mapping a stream event into Raft commands failed.
    #[error(transparent)]
    Mapping(#[from] MapperError),
    /// Proposing a command through Raft failed.
    #[error("consensus error: {0}")]
    Consensus(String),
}

/// Convenience alias for fallible sink operations.
pub type SinkResult<T> = std::result::Result<T, SinkError>;

/// Local trait describing a write-only streaming sink.
///
/// Producers (such as the upstream `oxirs-stream` ingest pipeline) call
/// [`StreamSink::write_batch`] with a batch of events. The implementation
/// is responsible for delivering the batch durably (in the cluster case,
/// through Raft).
#[async_trait]
pub trait StreamSink: Send + Sync {
    /// Writes a batch of [`StreamMessage`] events.
    async fn write_batch(&self, events: Vec<StreamMessage>) -> SinkResult<()>;

    /// Returns the latest [`BackpressureSignal`] the sink would surface.
    /// Default implementation returns [`BackpressureSignal::Continue`].
    fn backpressure_signal(&self) -> BackpressureSignal {
        BackpressureSignal::Continue
    }
}

/// Configuration for [`ClusterSink`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSinkConfig {
    /// If `true`, the sink will refuse a `write_batch` call when the
    /// backpressure bridge reports [`BackpressureSignal::Stop`].
    pub honor_backpressure_stop: bool,
    /// Maximum number of `RdfCommand` entries proposed per `write_batch`
    /// call. If `0`, no cap is applied.
    pub max_batch_commands: usize,
    /// If `true`, [`ClusterSink::write_batch`] will require leadership
    /// before proposing. If `false`, the sink will propose unconditionally
    /// (used in single-node and test setups).
    pub require_leader: bool,
}

impl Default for ClusterSinkConfig {
    fn default() -> Self {
        Self {
            honor_backpressure_stop: true,
            max_batch_commands: 4_096,
            require_leader: true,
        }
    }
}

/// Runtime statistics for [`ClusterSink`].
#[derive(Debug, Default)]
pub struct ClusterSinkStats {
    /// Total batches accepted by `write_batch` (regardless of outcome).
    pub batches_received: AtomicU64,
    /// Total batches successfully committed.
    pub batches_committed: AtomicU64,
    /// Total batches rejected with [`SinkError::NotLeader`].
    pub batches_rejected_not_leader: AtomicU64,
    /// Total batches rejected with [`SinkError::BackpressureStopped`].
    pub batches_rejected_backpressure: AtomicU64,
    /// Total individual `RdfCommand` entries successfully committed.
    pub commands_committed: AtomicU64,
    /// Total individual `RdfCommand` entries that returned a `RdfResponse::Error`.
    pub commands_failed: AtomicU64,
}

/// Production [`StreamSink`] that forwards events through Raft.
pub struct ClusterSink {
    consensus: Arc<ConsensusManager>,
    mapper: Arc<dyn EventMapper>,
    bridge: BackpressureBridge,
    config: ClusterSinkConfig,
    stats: Arc<ClusterSinkStats>,
}

impl ClusterSink {
    /// Creates a new sink wired to the given consensus manager and mapper.
    pub fn new(
        consensus: Arc<ConsensusManager>,
        mapper: Arc<dyn EventMapper>,
        config: ClusterSinkConfig,
    ) -> Self {
        Self {
            consensus,
            mapper,
            bridge: BackpressureBridge::default(),
            config,
            stats: Arc::new(ClusterSinkStats::default()),
        }
    }

    /// Creates a new sink with a caller-provided [`BackpressureBridge`].
    pub fn with_bridge(
        consensus: Arc<ConsensusManager>,
        mapper: Arc<dyn EventMapper>,
        bridge: BackpressureBridge,
        config: ClusterSinkConfig,
    ) -> Self {
        Self {
            consensus,
            mapper,
            bridge,
            config,
            stats: Arc::new(ClusterSinkStats::default()),
        }
    }

    /// Returns a clone of the underlying backpressure bridge so upstream
    /// operators can poll it without holding a reference to the sink.
    pub fn bridge(&self) -> BackpressureBridge {
        self.bridge.clone()
    }

    /// Returns the runtime statistics.
    pub fn stats(&self) -> &Arc<ClusterSinkStats> {
        &self.stats
    }

    /// Returns the active configuration.
    pub fn config(&self) -> &ClusterSinkConfig {
        &self.config
    }

    /// Accessor: `consensus` for tests / advanced callers.
    pub fn consensus(&self) -> &Arc<ConsensusManager> {
        &self.consensus
    }
}

#[async_trait]
impl StreamSink for ClusterSink {
    async fn write_batch(&self, events: Vec<StreamMessage>) -> SinkResult<()> {
        self.stats.batches_received.fetch_add(1, Ordering::Relaxed);

        // Backpressure gate.
        if self.config.honor_backpressure_stop
            && matches!(self.bridge.signal(), BackpressureSignal::Stop)
        {
            self.stats
                .batches_rejected_backpressure
                .fetch_add(1, Ordering::Relaxed);
            return Err(SinkError::BackpressureStopped);
        }

        // Leader check.
        if self.config.require_leader && !self.consensus.is_leader().await {
            self.stats
                .batches_rejected_not_leader
                .fetch_add(1, Ordering::Relaxed);
            return Err(SinkError::NotLeader);
        }

        // Map streaming events to Raft commands.
        let mut commands = self.mapper.map_batch(&events)?;

        // Apply per-batch cap.
        if self.config.max_batch_commands > 0 && commands.len() > self.config.max_batch_commands {
            warn!(
                command_count = commands.len(),
                cap = self.config.max_batch_commands,
                "ClusterSink batch exceeds max_batch_commands; truncating"
            );
            commands.truncate(self.config.max_batch_commands);
        }

        let cmd_count = commands.len() as u64;
        debug!(
            event_count = events.len(),
            command_count = cmd_count,
            "ClusterSink proposing batch"
        );

        // Track the in-flight queue depth via the bridge.
        let _ = self.bridge.add(cmd_count);

        // Dispatch each command through consensus.
        let mut errors = 0u64;
        for cmd in commands.into_iter() {
            match self.consensus.propose_command(cmd).await {
                Ok(resp) => {
                    if let crate::raft::RdfResponse::Error(msg) = resp {
                        warn!(?msg, "ClusterSink: command apply returned error");
                        errors += 1;
                    } else {
                        self.stats
                            .commands_committed
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
                Err(e) => {
                    let _ = self.bridge.sub(1);
                    self.stats.commands_failed.fetch_add(1, Ordering::Relaxed);
                    return Err(SinkError::Consensus(e.to_string()));
                }
            }
        }

        // All proposals returned (errors counted separately).
        let _ = self.bridge.sub(cmd_count);
        self.stats
            .commands_failed
            .fetch_add(errors, Ordering::Relaxed);

        if errors == 0 {
            self.stats.batches_committed.fetch_add(1, Ordering::Relaxed);
        }

        Ok(())
    }

    fn backpressure_signal(&self) -> BackpressureSignal {
        self.bridge.signal()
    }
}

/// Helper used by callers (and tests) that want to short-circuit when the
/// signal would refuse new data anyway.
pub fn should_pause(signal: BackpressureSignal) -> bool {
    !matches!(signal, BackpressureSignal::Continue)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raft::{init_global_shared_storage, reset_global_shared_storage};
    use crate::stream_integration::{StreamMessage, StreamTriple};
    use crate::streaming::event_mapper::DefaultEventMapper;

    static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    fn make_sink(node_id: u64) -> ClusterSink {
        let consensus = Arc::new(ConsensusManager::new(node_id, vec![]));
        let mapper = Arc::new(DefaultEventMapper::default());
        ClusterSink::new(consensus, mapper, ClusterSinkConfig::default())
    }

    fn triple(i: usize) -> StreamTriple {
        StreamTriple::new(
            format!("http://example.org/s/{i}"),
            "http://example.org/p/has",
            format!("\"value-{i}\""),
        )
    }

    #[tokio::test]
    async fn write_batch_proposes_through_consensus() {
        let _g = TEST_LOCK.lock().await;
        init_global_shared_storage();
        reset_global_shared_storage().await;

        let sink = make_sink(1);
        let msg = StreamMessage::insert("rdf-stream", 1, vec![triple(0), triple(1)]);
        sink.write_batch(vec![msg]).await.expect("commit");

        assert_eq!(sink.stats().batches_committed.load(Ordering::Relaxed), 1);
        assert_eq!(sink.stats().commands_committed.load(Ordering::Relaxed), 2);

        // Triples are now visible through the regular cluster query path.
        let len = sink.consensus().len().await;
        assert_eq!(len, 2);
    }

    #[tokio::test]
    async fn write_batch_respects_backpressure_stop() {
        let _g = TEST_LOCK.lock().await;
        init_global_shared_storage();
        reset_global_shared_storage().await;

        let sink = make_sink(2);
        // Force the bridge into Stop.
        let _ = sink.bridge().observe(10_000_000);
        assert_eq!(sink.bridge().signal(), BackpressureSignal::Stop);

        let msg = StreamMessage::insert("rdf-stream", 1, vec![triple(0)]);
        let err = sink
            .write_batch(vec![msg])
            .await
            .expect_err("should refuse");
        assert!(matches!(err, SinkError::BackpressureStopped));
        assert_eq!(
            sink.stats()
                .batches_rejected_backpressure
                .load(Ordering::Relaxed),
            1
        );
    }

    #[tokio::test]
    async fn write_batch_truncates_oversized() {
        let _g = TEST_LOCK.lock().await;
        init_global_shared_storage();
        reset_global_shared_storage().await;

        let consensus = Arc::new(ConsensusManager::new(3, vec![]));
        let mapper = Arc::new(DefaultEventMapper::default());
        let cfg = ClusterSinkConfig {
            max_batch_commands: 2,
            ..Default::default()
        };
        let sink = ClusterSink::new(consensus, mapper, cfg);

        let triples: Vec<_> = (0..5).map(triple).collect();
        let msg = StreamMessage::insert("rdf-stream", 1, triples);
        sink.write_batch(vec![msg]).await.expect("commit");

        // Only 2 commands should have been committed because of the cap.
        assert_eq!(sink.stats().commands_committed.load(Ordering::Relaxed), 2);
    }

    #[tokio::test]
    async fn write_batch_handles_empty_input() {
        let _g = TEST_LOCK.lock().await;
        init_global_shared_storage();
        reset_global_shared_storage().await;

        let sink = make_sink(4);
        sink.write_batch(vec![]).await.expect("ok");
        assert_eq!(sink.stats().batches_committed.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn backpressure_signal_default_is_continue() {
        let _g = TEST_LOCK.lock().await;
        init_global_shared_storage();
        reset_global_shared_storage().await;

        let sink = make_sink(5);
        assert_eq!(sink.backpressure_signal(), BackpressureSignal::Continue);
    }

    #[test]
    fn should_pause_flags_non_continue() {
        assert!(!should_pause(BackpressureSignal::Continue));
        assert!(should_pause(BackpressureSignal::Slow));
        assert!(should_pause(BackpressureSignal::Stop));
    }

    #[tokio::test]
    async fn require_leader_false_proposes_anyway() {
        let _g = TEST_LOCK.lock().await;
        init_global_shared_storage();
        reset_global_shared_storage().await;

        let consensus = Arc::new(ConsensusManager::new(6, vec![]));
        let mapper = Arc::new(DefaultEventMapper::default());
        let cfg = ClusterSinkConfig {
            require_leader: false,
            ..Default::default()
        };
        let sink = ClusterSink::new(consensus, mapper, cfg);

        let msg = StreamMessage::insert("rdf-stream", 1, vec![triple(7)]);
        sink.write_batch(vec![msg]).await.expect("ok");
        assert_eq!(sink.stats().commands_committed.load(Ordering::Relaxed), 1);
    }
}
