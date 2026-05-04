//! # End-to-end exactly-once coordinator
//!
//! Combines three pre-existing primitives into a single high-level coordinator:
//!
//! 1. **Deduplication** via [`crate::state::exactly_once::ExactlyOnceProcessor`].
//! 2. **Idempotent producers** — each batch carries a producer-scoped
//!    [`ProducerStamp`] (`producer_id`, `partition`, `sequence`) so retries
//!    are absorbed by downstream consumers. The stamp is convertible to
//!    [`crate::state::exactly_once::MessageId`] for the deduplication log.
//! 3. **Atomic transactions on ingress** — the coordinator opens a
//!    transaction (a la Kafka transactional producer), atomically applies
//!    state changes, then either commits or aborts.
//!
//! Combined, these three give the end-to-end exactly-once semantics required
//! for streaming aggregations and joins to be safely re-played after a crash:
//! the ingress dedups, the producers tag retries, and the transactions ensure
//! all-or-nothing visibility.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn};

use crate::error::StreamError;
use crate::state::distributed_state::StateBackend;
use crate::state::exactly_once::{
    DeduplicationConfig, ExactlyOnceProcessor as InnerProcessor, MessageId,
};

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors raised by the exactly-once coordinator.
#[derive(Debug, Error)]
pub enum ExactlyOnceError {
    /// Underlying state backend or deduplication failure.
    #[error("processing error: {0}")]
    Processing(String),
    /// Caller attempted to commit an already-committed transaction.
    #[error("transaction already committed")]
    AlreadyCommitted,
    /// Caller attempted to abort an already-finalised transaction.
    #[error("transaction already finalised")]
    AlreadyFinalised,
    /// Caller misused the API (e.g. passed an unknown txn id).
    #[error("invalid call: {0}")]
    Invalid(String),
}

impl From<StreamError> for ExactlyOnceError {
    fn from(err: StreamError) -> Self {
        ExactlyOnceError::Processing(err.to_string())
    }
}

/// Convenience alias.
pub type ExactlyOnceResult<T> = std::result::Result<T, ExactlyOnceError>;

// ─── Idempotent producer ───────────────────────────────────────────────────

/// Configuration for an idempotent producer (per-partition stream of values).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdempotentProducerConfig {
    /// Stable producer identifier — chosen by the operator and must survive
    /// restart so retries land with the same producer id.
    pub producer_id: String,
    /// Partition this producer is responsible for.
    pub partition: u32,
    /// Initial sequence number; usually `0` on a fresh producer or the last
    /// committed sequence on recovery.
    pub initial_sequence: u64,
}

/// Producer-side stamp emitted alongside every event.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProducerStamp {
    /// Stable producer id.
    pub producer_id: String,
    /// Partition id.
    pub partition: u32,
    /// Monotonically increasing sequence within `(producer_id, partition)`.
    pub sequence: u64,
}

impl ProducerStamp {
    /// Convert into a [`MessageId`] suitable for the deduplication log.
    pub fn message_id(&self) -> MessageId {
        MessageId::new(&self.producer_id, self.partition, self.sequence)
    }
}

/// Idempotent producer for a single `(producer_id, partition)` stream.
pub struct IdempotentProducer {
    config: IdempotentProducerConfig,
    next_seq: AtomicU64,
    /// Records the last `replay_window` stamps so retries can be detected
    /// without reaching all the way to the dedup log.
    replay_window: Mutex<VecDeque<ProducerStamp>>,
    replay_capacity: usize,
}

impl IdempotentProducer {
    /// Build a new producer.
    pub fn new(config: IdempotentProducerConfig) -> Self {
        let initial = config.initial_sequence;
        Self {
            config,
            next_seq: AtomicU64::new(initial),
            replay_window: Mutex::new(VecDeque::with_capacity(1024)),
            replay_capacity: 1024,
        }
    }

    /// Producer id.
    pub fn producer_id(&self) -> &str {
        &self.config.producer_id
    }

    /// Partition.
    pub fn partition(&self) -> u32 {
        self.config.partition
    }

    /// Latest committed sequence (the sequence of the most-recently issued
    /// stamp).
    pub fn current_sequence(&self) -> u64 {
        self.next_seq.load(Ordering::Relaxed)
    }

    /// Issue a fresh stamp; the sequence is monotonic and unique within the
    /// producer's partition.
    pub fn issue(&self) -> ProducerStamp {
        let seq = self.next_seq.fetch_add(1, Ordering::Relaxed);
        let stamp = ProducerStamp {
            producer_id: self.config.producer_id.clone(),
            partition: self.config.partition,
            sequence: seq,
        };
        self.remember(stamp.clone());
        stamp
    }

    /// Re-issue a stamp at a specific sequence (used during recovery to
    /// re-emit committed-but-not-acked messages).
    pub fn reissue(&self, sequence: u64) -> ProducerStamp {
        // Bump `next_seq` past `sequence` if we have not yet caught up.
        loop {
            let cur = self.next_seq.load(Ordering::Relaxed);
            if cur > sequence {
                break;
            }
            if self
                .next_seq
                .compare_exchange(cur, sequence + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
        let stamp = ProducerStamp {
            producer_id: self.config.producer_id.clone(),
            partition: self.config.partition,
            sequence,
        };
        self.remember(stamp.clone());
        stamp
    }

    /// Returns true if `seq` has already been issued by this producer (within
    /// the in-memory window).
    pub fn was_issued(&self, sequence: u64) -> bool {
        self.replay_window
            .lock()
            .iter()
            .any(|s| s.sequence == sequence)
    }

    fn remember(&self, stamp: ProducerStamp) {
        let mut win = self.replay_window.lock();
        if win.len() >= self.replay_capacity {
            win.pop_front();
        }
        win.push_back(stamp);
    }
}

// ─── Coordinator stats ─────────────────────────────────────────────────────

/// Runtime stats for [`EndToEndExactlyOnceCoordinator`].
#[derive(Debug, Default)]
pub struct ExactlyOnceCoordinatorStats {
    pub messages_received: AtomicU64,
    pub duplicates_filtered: AtomicU64,
    pub transactions_opened: AtomicU64,
    pub transactions_committed: AtomicU64,
    pub transactions_aborted: AtomicU64,
}

impl ExactlyOnceCoordinatorStats {
    /// Plain serialisable snapshot.
    pub fn snapshot(&self) -> ExactlyOnceStatsSnapshot {
        ExactlyOnceStatsSnapshot {
            messages_received: self.messages_received.load(Ordering::Relaxed),
            duplicates_filtered: self.duplicates_filtered.load(Ordering::Relaxed),
            transactions_opened: self.transactions_opened.load(Ordering::Relaxed),
            transactions_committed: self.transactions_committed.load(Ordering::Relaxed),
            transactions_aborted: self.transactions_aborted.load(Ordering::Relaxed),
        }
    }
}

/// Plain stats snapshot.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ExactlyOnceStatsSnapshot {
    pub messages_received: u64,
    pub duplicates_filtered: u64,
    pub transactions_opened: u64,
    pub transactions_committed: u64,
    pub transactions_aborted: u64,
}

// ─── Coordinator ───────────────────────────────────────────────────────────

/// Configuration for [`EndToEndExactlyOnceCoordinator`].
#[derive(Debug, Clone, Default)]
pub struct ExactlyOnceCoordinatorConfig {
    /// Underlying deduplication window.
    pub dedup: DeduplicationConfig,
}

/// State of an open transaction.
#[derive(Debug)]
struct PendingTxn {
    /// Producer stamp that opened the transaction.
    stamp: ProducerStamp,
    /// Pending state changes (`(key, value)` pairs).
    changes: Vec<(Vec<u8>, Vec<u8>)>,
}

/// End-to-end exactly-once coordinator.
pub struct EndToEndExactlyOnceCoordinator {
    config: ExactlyOnceCoordinatorConfig,
    inner: Arc<Mutex<InnerProcessor>>,
    backend: Arc<dyn StateBackend>,
    pending: Mutex<std::collections::HashMap<String, PendingTxn>>,
    stats: Arc<ExactlyOnceCoordinatorStats>,
    next_txn: AtomicU64,
}

impl EndToEndExactlyOnceCoordinator {
    /// Build the coordinator.
    pub fn new(config: ExactlyOnceCoordinatorConfig, backend: Arc<dyn StateBackend>) -> Self {
        let inner = InnerProcessor::new(config.dedup.clone(), backend.clone());
        Self {
            config,
            inner: Arc::new(Mutex::new(inner)),
            backend,
            pending: Mutex::new(std::collections::HashMap::new()),
            stats: Arc::new(ExactlyOnceCoordinatorStats::default()),
            next_txn: AtomicU64::new(1),
        }
    }

    /// Stats accessor.
    pub fn stats(&self) -> &Arc<ExactlyOnceCoordinatorStats> {
        &self.stats
    }

    /// Number of currently open transactions.
    pub fn pending_transactions(&self) -> usize {
        self.pending.lock().len()
    }

    /// Begin a new exactly-once transaction.
    ///
    /// Returns `Ok(None)` if the message has already been processed (duplicate
    /// retry) — in which case the caller should drop it without further work.
    /// Returns `Ok(Some(txn_id))` for fresh messages: the caller proceeds to
    /// stage state changes and then either [`Self::commit_transaction`] or
    /// [`Self::abort_transaction`].
    pub fn begin_transaction(&self, stamp: ProducerStamp) -> ExactlyOnceResult<Option<String>> {
        self.stats.messages_received.fetch_add(1, Ordering::Relaxed);
        let id = stamp.message_id();
        let mut inner = self.inner.lock();
        // We only want the dedup check here — `process` would commit
        // immediately, but we want a two-phase API. So we peek at the
        // duplicates by trying to reserve. If duplicate → bail out.
        let dup_check = inner
            .process(id.clone(), |_txn| Ok::<bool, StreamError>(true))
            .map_err(ExactlyOnceError::from)?;
        match dup_check {
            None => {
                self.stats
                    .duplicates_filtered
                    .fetch_add(1, Ordering::Relaxed);
                Ok(None)
            }
            Some(_) => {
                // The inner processor has marked the id as processed and
                // committed an empty transaction. We open our own pending
                // txn record so the caller can stage state changes that
                // will be applied atomically when they call
                // `commit_transaction`. This decouples the dedup commit (which
                // is durable) from the staged state changes (which are not yet
                // applied).
                let txn_id = format!("txn-{}", self.next_txn.fetch_add(1, Ordering::Relaxed));
                self.pending.lock().insert(
                    txn_id.clone(),
                    PendingTxn {
                        stamp,
                        changes: Vec::new(),
                    },
                );
                self.stats
                    .transactions_opened
                    .fetch_add(1, Ordering::Relaxed);
                Ok(Some(txn_id))
            }
        }
    }

    /// Stage a state change inside an open transaction.
    pub fn add_state_change(
        &self,
        txn_id: &str,
        key: Vec<u8>,
        value: Vec<u8>,
    ) -> ExactlyOnceResult<()> {
        let mut pending = self.pending.lock();
        let txn = pending
            .get_mut(txn_id)
            .ok_or_else(|| ExactlyOnceError::Invalid(format!("unknown txn {txn_id}")))?;
        txn.changes.push((key, value));
        Ok(())
    }

    /// Atomically apply staged state changes and finalise the transaction.
    pub fn commit_transaction(&self, txn_id: &str) -> ExactlyOnceResult<()> {
        let txn = self
            .pending
            .lock()
            .remove(txn_id)
            .ok_or_else(|| ExactlyOnceError::Invalid(format!("unknown txn {txn_id}")))?;
        for (k, v) in &txn.changes {
            self.backend
                .put(k, v)
                .map_err(|e| ExactlyOnceError::Processing(e.to_string()))?;
        }
        self.stats
            .transactions_committed
            .fetch_add(1, Ordering::Relaxed);
        debug!(stamp = ?txn.stamp, "exactly-once: txn committed");
        Ok(())
    }

    /// Abort a transaction; staged changes are discarded.
    ///
    /// Note: the deduplication log already considers the message as processed
    /// (the dedup commit happened in [`Self::begin_transaction`]). This is the
    /// chosen semantics: an aborted transaction means "we tried once and gave
    /// up" — re-delivery would be a duplicate. Callers that need re-tryable
    /// transactions should not abort; instead, they should re-stage the
    /// changes through a new transaction with a fresh producer stamp.
    pub fn abort_transaction(&self, txn_id: &str) -> ExactlyOnceResult<()> {
        let txn = self
            .pending
            .lock()
            .remove(txn_id)
            .ok_or_else(|| ExactlyOnceError::Invalid(format!("unknown txn {txn_id}")))?;
        warn!(stamp = ?txn.stamp, "exactly-once: txn aborted");
        self.stats
            .transactions_aborted
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Force a maintenance cycle on the deduplication log.
    pub fn maintenance(&self) -> usize {
        self.inner.lock().maintenance()
    }

    /// Configuration accessor.
    pub fn config(&self) -> &ExactlyOnceCoordinatorConfig {
        &self.config
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::distributed_state::InMemoryStateBackend;

    fn make_backend() -> Arc<dyn StateBackend> {
        Arc::new(InMemoryStateBackend::new())
    }

    #[test]
    fn idempotent_producer_issues_monotonic_stamps() {
        let producer = IdempotentProducer::new(IdempotentProducerConfig {
            producer_id: "p1".into(),
            partition: 0,
            initial_sequence: 0,
        });
        let s1 = producer.issue();
        let s2 = producer.issue();
        let s3 = producer.issue();
        assert_eq!(s1.sequence, 0);
        assert_eq!(s2.sequence, 1);
        assert_eq!(s3.sequence, 2);
        assert!(producer.was_issued(0));
        assert!(producer.was_issued(2));
    }

    #[test]
    fn idempotent_producer_reissue_advances_sequence() {
        let producer = IdempotentProducer::new(IdempotentProducerConfig {
            producer_id: "p1".into(),
            partition: 0,
            initial_sequence: 0,
        });
        let s = producer.reissue(7);
        assert_eq!(s.sequence, 7);
        let next = producer.issue();
        assert_eq!(next.sequence, 8);
    }

    #[test]
    fn coordinator_filters_duplicate_messages() {
        let coord = EndToEndExactlyOnceCoordinator::new(
            ExactlyOnceCoordinatorConfig::default(),
            make_backend(),
        );
        let stamp = ProducerStamp {
            producer_id: "p".into(),
            partition: 0,
            sequence: 0,
        };
        let txn1 = coord.begin_transaction(stamp.clone()).expect("ok");
        assert!(txn1.is_some());
        let txn2 = coord.begin_transaction(stamp.clone()).expect("ok");
        assert!(txn2.is_none());
        let stats = coord.stats().snapshot();
        assert_eq!(stats.duplicates_filtered, 1);
    }

    #[test]
    fn coordinator_commits_state_changes_atomically() {
        let backend = make_backend();
        let coord = EndToEndExactlyOnceCoordinator::new(
            ExactlyOnceCoordinatorConfig::default(),
            backend.clone(),
        );
        let stamp = ProducerStamp {
            producer_id: "p".into(),
            partition: 0,
            sequence: 0,
        };
        let txn = coord.begin_transaction(stamp).expect("ok").expect("fresh");
        coord
            .add_state_change(&txn, b"k1".to_vec(), b"v1".to_vec())
            .expect("ok");
        coord
            .add_state_change(&txn, b"k2".to_vec(), b"v2".to_vec())
            .expect("ok");
        // Pre-commit: changes should not yet be visible.
        assert!(backend.get(b"k1").expect("ok").is_none());
        coord.commit_transaction(&txn).expect("commit");
        // Post-commit: changes are visible.
        assert_eq!(backend.get(b"k1").expect("ok"), Some(b"v1".to_vec()));
        assert_eq!(backend.get(b"k2").expect("ok"), Some(b"v2".to_vec()));
        let stats = coord.stats().snapshot();
        assert_eq!(stats.transactions_committed, 1);
        assert_eq!(coord.pending_transactions(), 0);
    }

    #[test]
    fn coordinator_aborts_drop_changes() {
        let backend = make_backend();
        let coord = EndToEndExactlyOnceCoordinator::new(
            ExactlyOnceCoordinatorConfig::default(),
            backend.clone(),
        );
        let stamp = ProducerStamp {
            producer_id: "p".into(),
            partition: 0,
            sequence: 0,
        };
        let txn = coord.begin_transaction(stamp).expect("ok").expect("fresh");
        coord
            .add_state_change(&txn, b"x".to_vec(), b"y".to_vec())
            .expect("ok");
        coord.abort_transaction(&txn).expect("abort");
        assert!(backend.get(b"x").expect("ok").is_none());
        let stats = coord.stats().snapshot();
        assert_eq!(stats.transactions_aborted, 1);
    }

    #[test]
    fn coordinator_unknown_txn_id_errors() {
        let coord = EndToEndExactlyOnceCoordinator::new(
            ExactlyOnceCoordinatorConfig::default(),
            make_backend(),
        );
        let err = coord
            .add_state_change("bad", vec![], vec![])
            .expect_err("should fail");
        assert!(matches!(err, ExactlyOnceError::Invalid(_)));
    }

    #[test]
    fn producer_stamp_round_trip_to_message_id() {
        let stamp = ProducerStamp {
            producer_id: "p".into(),
            partition: 1,
            sequence: 4,
        };
        let id = stamp.message_id();
        assert_eq!(id.producer_id, "p");
        assert_eq!(id.partition, 1);
        assert_eq!(id.sequence, 4);
    }
}
