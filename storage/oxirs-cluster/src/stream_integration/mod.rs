//! Real-time streaming integration for OxiRS cluster.
//!
//! This module bridges external streaming sources (e.g., Kafka, NATS, or any
//! byte-oriented channel) with the cluster's RDF triple store, providing:
//!
//! - **StreamingTripleConsumer**: consumes RDF triples arriving as streaming messages.
//! - **StreamingMutationLog**: ordered log of mutations with monotonic sequence numbers.
//! - **StreamSyncCoordinator**: coordinates stream consumption across cluster nodes
//!   so that each triple is processed exactly once.
//! - **StreamingCheckpointer**: persists stream consumption offsets for crash recovery.

use crate::error::{ClusterError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info};

// ─────────────────────────────────────────────
//  RDF Triple representation for streaming
// ─────────────────────────────────────────────

/// A simplified RDF triple for streaming ingestion.
///
/// Each component is an N-Triples-style string (IRI, blank node, or literal).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StreamTriple {
    /// Subject (IRI or blank node).
    pub subject: String,
    /// Predicate (IRI).
    pub predicate: String,
    /// Object (IRI, blank node, or literal).
    pub object: String,
    /// Optional named graph IRI.
    pub graph: Option<String>,
}

impl StreamTriple {
    /// Creates a triple without a named graph.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            graph: None,
        }
    }

    /// Creates a triple within a named graph.
    pub fn in_graph(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        graph: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            graph: Some(graph.into()),
        }
    }

    /// Validates the triple (non-empty fields).
    pub fn validate(&self) -> Result<()> {
        if self.subject.is_empty() {
            return Err(ClusterError::Config(
                "Triple subject cannot be empty".into(),
            ));
        }
        if self.predicate.is_empty() {
            return Err(ClusterError::Config(
                "Triple predicate cannot be empty".into(),
            ));
        }
        if self.object.is_empty() {
            return Err(ClusterError::Config("Triple object cannot be empty".into()));
        }
        Ok(())
    }
}

impl std::fmt::Display for StreamTriple {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "<{}> <{}> {} .",
            self.subject, self.predicate, self.object
        )
    }
}

// ─────────────────────────────────────────────
//  Streaming message
// ─────────────────────────────────────────────

/// The mutation operation type carried in a streaming message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MutationOp {
    /// Insert one or more triples.
    Insert,
    /// Delete one or more triples.
    Delete,
    /// Truncate (clear) a named graph or default graph.
    Truncate {
        /// Named graph to clear; `None` for the default graph.
        graph: Option<String>,
    },
}

/// A message arriving from the streaming source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMessage {
    /// Source stream/topic identifier.
    pub stream_id: String,
    /// Monotonically increasing offset within the source stream.
    pub offset: u64,
    /// The mutation to apply.
    pub op: MutationOp,
    /// Triples affected by this mutation.
    pub triples: Vec<StreamTriple>,
    /// Wall-clock timestamp of the message (Unix ms).
    pub timestamp_ms: u64,
}

impl StreamMessage {
    /// Creates a new insert message.
    pub fn insert(stream_id: impl Into<String>, offset: u64, triples: Vec<StreamTriple>) -> Self {
        Self {
            stream_id: stream_id.into(),
            offset,
            op: MutationOp::Insert,
            triples,
            timestamp_ms: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_millis() as u64,
        }
    }

    /// Creates a new delete message.
    pub fn delete(stream_id: impl Into<String>, offset: u64, triples: Vec<StreamTriple>) -> Self {
        Self {
            stream_id: stream_id.into(),
            offset,
            op: MutationOp::Delete,
            triples,
            timestamp_ms: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_millis() as u64,
        }
    }
}

// ─────────────────────────────────────────────
//  StreamingTripleConsumer
// ─────────────────────────────────────────────

/// Configuration for the streaming triple consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerConfig {
    /// Maximum number of messages to buffer in the internal queue.
    pub max_buffer_size: usize,
    /// How long to wait for a batch before flushing (in milliseconds).
    pub flush_interval_ms: u64,
    /// Maximum number of triples per flush batch.
    pub batch_size: usize,
    /// Whether to validate each triple before inserting.
    pub validate_triples: bool,
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10_000,
            flush_interval_ms: 100,
            batch_size: 500,
            validate_triples: true,
        }
    }
}

/// Consumer statistics.
#[derive(Debug, Default)]
pub struct ConsumerStats {
    /// Total messages consumed.
    pub messages_consumed: AtomicU64,
    /// Total triples inserted.
    pub triples_inserted: AtomicU64,
    /// Total triples deleted.
    pub triples_deleted: AtomicU64,
    /// Total validation errors.
    pub validation_errors: AtomicU64,
    /// Total bytes processed.
    pub bytes_processed: AtomicU64,
}

/// Consumes RDF triples from a streaming source and buffers them for batch
/// application to the cluster store.
pub struct StreamingTripleConsumer {
    config: ConsumerConfig,
    buffer: Arc<Mutex<VecDeque<StreamMessage>>>,
    stats: Arc<ConsumerStats>,
    running: Arc<AtomicBool>,
}

impl StreamingTripleConsumer {
    /// Creates a new consumer with the given configuration.
    pub fn new(config: ConsumerConfig) -> Result<Self> {
        if config.max_buffer_size == 0 {
            return Err(ClusterError::Config("max_buffer_size must be > 0".into()));
        }
        if config.batch_size == 0 {
            return Err(ClusterError::Config("batch_size must be > 0".into()));
        }
        Ok(Self {
            config,
            buffer: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(ConsumerStats::default()),
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Submits a streaming message to the consumer's buffer.
    ///
    /// Returns `Err` if the buffer is full.
    pub async fn submit(&self, msg: StreamMessage) -> Result<()> {
        if self.config.validate_triples {
            for triple in &msg.triples {
                triple.validate().map_err(|e| {
                    self.stats.validation_errors.fetch_add(1, Ordering::Relaxed);
                    e
                })?;
            }
        }

        let mut buf = self.buffer.lock().await;
        if buf.len() >= self.config.max_buffer_size {
            return Err(ClusterError::ResourceLimit(
                "streaming consumer buffer is full".into(),
            ));
        }

        let triple_count = msg.triples.len() as u64;
        self.stats.messages_consumed.fetch_add(1, Ordering::Relaxed);
        match &msg.op {
            MutationOp::Insert => {
                self.stats
                    .triples_inserted
                    .fetch_add(triple_count, Ordering::Relaxed);
            }
            MutationOp::Delete => {
                self.stats
                    .triples_deleted
                    .fetch_add(triple_count, Ordering::Relaxed);
            }
            MutationOp::Truncate { .. } => {}
        }

        buf.push_back(msg);
        Ok(())
    }

    /// Drains up to `batch_size` messages from the buffer.
    pub async fn drain_batch(&self) -> Vec<StreamMessage> {
        let mut buf = self.buffer.lock().await;
        let n = self.config.batch_size.min(buf.len());
        buf.drain(..n).collect()
    }

    /// Returns the current buffer occupancy.
    pub async fn buffer_len(&self) -> usize {
        self.buffer.lock().await.len()
    }

    /// Returns consumer statistics.
    pub fn stats(&self) -> &ConsumerStats {
        &self.stats
    }

    /// Marks the consumer as running.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        info!("StreamingTripleConsumer started");
    }

    /// Marks the consumer as stopped.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        info!("StreamingTripleConsumer stopped");
    }

    /// Returns whether the consumer is currently running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

// ─────────────────────────────────────────────
//  StreamingMutationLog
// ─────────────────────────────────────────────

/// A mutation log entry with a globally monotonic sequence number.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationEntry {
    /// Globally assigned sequence number (monotonically increasing).
    pub seq: u64,
    /// The originating stream message.
    pub message: StreamMessage,
    /// Node that assigned this sequence number.
    pub assigned_by: u64,
}

/// An ordered log of streaming mutations with strict sequence-number ordering.
///
/// Producers append to the log; consumers read entries in order.  Entries
/// before `committed_seq` are considered durably applied.
pub struct StreamingMutationLog {
    /// Entries indexed by sequence number.
    log: Arc<RwLock<BTreeMap<u64, MutationEntry>>>,
    /// Monotonic sequence counter.
    next_seq: AtomicU64,
    /// The highest sequence number that has been applied/committed.
    committed_seq: AtomicU64,
    /// Maximum log capacity before oldest entries are evicted.
    max_capacity: usize,
    /// Node that owns this log instance.
    node_id: u64,
}

impl StreamingMutationLog {
    /// Creates a new mutation log owned by the given node.
    pub fn new(node_id: u64, max_capacity: usize) -> Result<Self> {
        if max_capacity == 0 {
            return Err(ClusterError::Config("max_capacity must be > 0".into()));
        }
        Ok(Self {
            log: Arc::new(RwLock::new(BTreeMap::new())),
            next_seq: AtomicU64::new(1),
            committed_seq: AtomicU64::new(0),
            max_capacity,
            node_id,
        })
    }

    /// Appends a stream message, assigning the next monotonic sequence number.
    ///
    /// Returns the assigned sequence number.
    pub async fn append(&self, message: StreamMessage) -> Result<u64> {
        let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
        let entry = MutationEntry {
            seq,
            message,
            assigned_by: self.node_id,
        };

        let mut log = self.log.write().await;

        // Evict oldest entries if at capacity.
        while log.len() >= self.max_capacity {
            if let Some((&oldest_seq, _)) = log.iter().next() {
                log.remove(&oldest_seq);
                debug!(oldest_seq, "Evicted oldest mutation log entry");
            } else {
                break;
            }
        }

        log.insert(seq, entry);
        Ok(seq)
    }

    /// Marks entries up to and including `seq` as committed.
    pub fn commit_up_to(&self, seq: u64) {
        let current = self.committed_seq.load(Ordering::SeqCst);
        if seq > current {
            self.committed_seq.store(seq, Ordering::SeqCst);
        }
    }

    /// Reads all entries in the log with sequence numbers in `[from, to]`.
    pub async fn read_range(&self, from: u64, to: u64) -> Vec<MutationEntry> {
        let log = self.log.read().await;
        log.range(from..=to).map(|(_, e)| e.clone()).collect()
    }

    /// Returns the highest committed sequence number.
    pub fn committed_seq(&self) -> u64 {
        self.committed_seq.load(Ordering::SeqCst)
    }

    /// Returns the next sequence number that will be assigned.
    pub fn next_seq(&self) -> u64 {
        self.next_seq.load(Ordering::SeqCst)
    }

    /// Returns the number of entries currently in the log.
    pub async fn len(&self) -> usize {
        self.log.read().await.len()
    }

    /// Returns `true` if the log is empty.
    pub async fn is_empty(&self) -> bool {
        self.log.read().await.is_empty()
    }
}

// ─────────────────────────────────────────────
//  StreamSyncCoordinator
// ─────────────────────────────────────────────

/// Tracks which offset each node has processed for a given stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeOffset {
    /// Node identifier.
    pub node_id: u64,
    /// Last committed offset for this stream.
    pub offset: u64,
    /// Wall-clock timestamp when this offset was last updated (Unix ms).
    pub updated_at_ms: u64,
}

/// Coordinates stream consumption across multiple cluster nodes.
///
/// Nodes periodically report their current stream offsets.  The coordinator
/// derives the cluster-wide minimum committed offset (the "safe offset") that
/// can be safely evicted from the mutation log.
pub struct StreamSyncCoordinator {
    /// Per-node offsets, keyed by (stream_id, node_id).
    offsets: Arc<RwLock<HashMap<(String, u64), NodeOffset>>>,
    /// Known stream IDs.
    streams: Arc<RwLock<HashSet<String>>>,
}

impl StreamSyncCoordinator {
    /// Creates a new empty coordinator.
    pub fn new() -> Self {
        Self {
            offsets: Arc::new(RwLock::new(HashMap::new())),
            streams: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Reports a node's current stream offset.
    pub async fn report_offset(&self, stream_id: impl Into<String>, node_id: u64, offset: u64) {
        let stream_id = stream_id.into();
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        {
            let mut streams = self.streams.write().await;
            streams.insert(stream_id.clone());
        }

        let mut offsets = self.offsets.write().await;
        offsets.insert(
            (stream_id, node_id),
            NodeOffset {
                node_id,
                offset,
                updated_at_ms: now_ms,
            },
        );
    }

    /// Returns the minimum offset seen across all nodes for a given stream.
    ///
    /// This represents the offset that every node has confirmed processing.
    /// Returns `None` if no nodes have reported for this stream.
    pub async fn min_offset(&self, stream_id: &str) -> Option<u64> {
        let offsets = self.offsets.read().await;
        let relevant: Vec<u64> = offsets
            .iter()
            .filter(|((s, _), _)| s == stream_id)
            .map(|(_, no)| no.offset)
            .collect();
        if relevant.is_empty() {
            None
        } else {
            relevant.into_iter().min()
        }
    }

    /// Returns the maximum offset seen across all nodes for a given stream.
    pub async fn max_offset(&self, stream_id: &str) -> Option<u64> {
        let offsets = self.offsets.read().await;
        let relevant: Vec<u64> = offsets
            .iter()
            .filter(|((s, _), _)| s == stream_id)
            .map(|(_, no)| no.offset)
            .collect();
        if relevant.is_empty() {
            None
        } else {
            relevant.into_iter().max()
        }
    }

    /// Returns the lag (max - min) for a stream.  A lag of 0 means all nodes
    /// are in perfect sync.
    pub async fn lag(&self, stream_id: &str) -> Option<u64> {
        let min = self.min_offset(stream_id).await?;
        let max = self.max_offset(stream_id).await?;
        Some(max.saturating_sub(min))
    }

    /// Lists all known stream IDs.
    pub async fn streams(&self) -> Vec<String> {
        self.streams.read().await.iter().cloned().collect()
    }

    /// Returns node offsets for a specific stream.
    pub async fn node_offsets_for(&self, stream_id: &str) -> Vec<NodeOffset> {
        let offsets = self.offsets.read().await;
        offsets
            .iter()
            .filter(|((s, _), _)| s == stream_id)
            .map(|(_, no)| no.clone())
            .collect()
    }
}

impl Default for StreamSyncCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
//  StreamingCheckpointer
// ─────────────────────────────────────────────

/// A checkpoint recording the last processed offset for a stream on a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Stream identifier.
    pub stream_id: String,
    /// Cluster node identifier.
    pub node_id: u64,
    /// Last durably processed offset.
    pub offset: u64,
    /// Timestamp of the checkpoint (Unix ms).
    pub checkpoint_ms: u64,
    /// Version counter for optimistic concurrency.
    pub version: u64,
}

/// Persists stream consumption checkpoints in memory (production code would
/// additionally flush to durable storage such as oxirs-tdb).
///
/// Supports:
/// - Atomic checkpoint updates with version validation.
/// - Listing all checkpoints for a node or stream.
/// - Bulk restore of checkpoints (for crash recovery).
pub struct StreamingCheckpointer {
    /// Checkpoints indexed by (stream_id, node_id).
    checkpoints: Arc<RwLock<HashMap<(String, u64), Checkpoint>>>,
    /// Monotonic version counter.
    version_counter: AtomicU64,
}

impl StreamingCheckpointer {
    /// Creates a new empty checkpointer.
    pub fn new() -> Self {
        Self {
            checkpoints: Arc::new(RwLock::new(HashMap::new())),
            version_counter: AtomicU64::new(1),
        }
    }

    /// Saves (or updates) a checkpoint.
    ///
    /// If a checkpoint already exists with a higher version, the update is
    /// rejected to prevent stale writes from overwriting newer state.
    pub async fn save(
        &self,
        stream_id: impl Into<String>,
        node_id: u64,
        offset: u64,
    ) -> Result<Checkpoint> {
        let stream_id = stream_id.into();
        let version = self.version_counter.fetch_add(1, Ordering::SeqCst);
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        let cp = Checkpoint {
            stream_id: stream_id.clone(),
            node_id,
            offset,
            checkpoint_ms: now_ms,
            version,
        };

        let mut cps = self.checkpoints.write().await;
        let key = (stream_id.clone(), node_id);

        // Reject if existing version is newer.
        if let Some(existing) = cps.get(&key) {
            if existing.version > version {
                return Err(ClusterError::Config(format!(
                    "Checkpoint version conflict: existing {} > new {}",
                    existing.version, version
                )));
            }
        }

        cps.insert(key, cp.clone());
        debug!(stream_id = %stream_id, node_id, offset, version, "Checkpoint saved");
        Ok(cp)
    }

    /// Loads the checkpoint for a given stream and node.
    pub async fn load(&self, stream_id: &str, node_id: u64) -> Option<Checkpoint> {
        let cps = self.checkpoints.read().await;
        cps.get(&(stream_id.to_string(), node_id)).cloned()
    }

    /// Lists all checkpoints for a specific node.
    pub async fn checkpoints_for_node(&self, node_id: u64) -> Vec<Checkpoint> {
        let cps = self.checkpoints.read().await;
        cps.values()
            .filter(|c| c.node_id == node_id)
            .cloned()
            .collect()
    }

    /// Lists all checkpoints for a specific stream.
    pub async fn checkpoints_for_stream(&self, stream_id: &str) -> Vec<Checkpoint> {
        let cps = self.checkpoints.read().await;
        cps.values()
            .filter(|c| c.stream_id == stream_id)
            .cloned()
            .collect()
    }

    /// Bulk-restores checkpoints (e.g., from durable storage after a restart).
    pub async fn restore(&self, checkpoints: Vec<Checkpoint>) {
        let mut cps = self.checkpoints.write().await;
        for cp in checkpoints {
            let key = (cp.stream_id.clone(), cp.node_id);
            cps.insert(key, cp);
        }
        info!("Checkpoint store restored");
    }

    /// Returns the total number of checkpoints stored.
    pub async fn count(&self) -> usize {
        self.checkpoints.read().await.len()
    }
}

impl Default for StreamingCheckpointer {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_triple() -> StreamTriple {
        StreamTriple::new(
            "http://example.org/subject",
            "http://example.org/predicate",
            "\"hello\"",
        )
    }

    fn insert_msg(offset: u64) -> StreamMessage {
        StreamMessage::insert("stream-rdf", offset, vec![sample_triple()])
    }

    // ── StreamingTripleConsumer ──────────────

    #[tokio::test]
    async fn test_consumer_submit_and_drain() {
        let consumer = StreamingTripleConsumer::new(ConsumerConfig::default()).expect("new");
        consumer.start();
        consumer.submit(insert_msg(1)).await.expect("submit 1");
        consumer.submit(insert_msg(2)).await.expect("submit 2");
        assert_eq!(consumer.buffer_len().await, 2);
        let batch = consumer.drain_batch().await;
        assert_eq!(batch.len(), 2);
        assert_eq!(consumer.buffer_len().await, 0);
    }

    #[tokio::test]
    async fn test_consumer_buffer_overflow() {
        let cfg = ConsumerConfig {
            max_buffer_size: 2,
            batch_size: 10,
            ..Default::default()
        };
        let consumer = StreamingTripleConsumer::new(cfg).expect("new");
        consumer.submit(insert_msg(1)).await.expect("1");
        consumer.submit(insert_msg(2)).await.expect("2");
        // Third submit should fail.
        assert!(consumer.submit(insert_msg(3)).await.is_err());
    }

    #[tokio::test]
    async fn test_consumer_stats_tracking() {
        let consumer = StreamingTripleConsumer::new(ConsumerConfig::default()).expect("new");
        consumer.submit(insert_msg(1)).await.expect("submit");
        assert_eq!(
            consumer.stats().messages_consumed.load(Ordering::Relaxed),
            1
        );
        assert_eq!(consumer.stats().triples_inserted.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_consumer_validation_rejects_empty_triple() {
        let consumer = StreamingTripleConsumer::new(ConsumerConfig {
            validate_triples: true,
            ..Default::default()
        })
        .expect("new");
        let bad_triple = StreamTriple::new("", "http://pred", "obj");
        let msg = StreamMessage::insert("s", 1, vec![bad_triple]);
        assert!(consumer.submit(msg).await.is_err());
    }

    // ── StreamingMutationLog ─────────────────

    #[tokio::test]
    async fn test_mutation_log_append_and_read() {
        let log = StreamingMutationLog::new(1, 1000).expect("new");
        let seq1 = log.append(insert_msg(1)).await.expect("append 1");
        let seq2 = log.append(insert_msg(2)).await.expect("append 2");
        assert_eq!(seq2, seq1 + 1);

        let entries = log.read_range(seq1, seq2).await;
        assert_eq!(entries.len(), 2);
    }

    #[tokio::test]
    async fn test_mutation_log_commit() {
        let log = StreamingMutationLog::new(1, 1000).expect("new");
        let seq = log.append(insert_msg(10)).await.expect("append");
        assert_eq!(log.committed_seq(), 0);
        log.commit_up_to(seq);
        assert_eq!(log.committed_seq(), seq);
    }

    #[tokio::test]
    async fn test_mutation_log_capacity_eviction() {
        let log = StreamingMutationLog::new(1, 3).expect("new");
        for i in 0..5 {
            log.append(insert_msg(i)).await.expect("append");
        }
        // Capacity is 3; older entries should be evicted.
        assert!(log.len().await <= 3);
    }

    // ── StreamSyncCoordinator ────────────────

    #[tokio::test]
    async fn test_sync_coordinator_offsets() {
        let coord = StreamSyncCoordinator::new();
        coord.report_offset("rdf-stream", 1, 100).await;
        coord.report_offset("rdf-stream", 2, 200).await;
        coord.report_offset("rdf-stream", 3, 150).await;

        assert_eq!(coord.min_offset("rdf-stream").await, Some(100));
        assert_eq!(coord.max_offset("rdf-stream").await, Some(200));
        assert_eq!(coord.lag("rdf-stream").await, Some(100));
    }

    #[tokio::test]
    async fn test_sync_coordinator_unknown_stream() {
        let coord = StreamSyncCoordinator::new();
        assert!(coord.min_offset("nonexistent").await.is_none());
        assert!(coord.lag("nonexistent").await.is_none());
    }

    // ── StreamingCheckpointer ────────────────

    #[tokio::test]
    async fn test_checkpointer_save_load() {
        let cp = StreamingCheckpointer::new();
        let saved = cp.save("my-stream", 42, 999).await.expect("save");
        assert_eq!(saved.offset, 999);
        assert_eq!(saved.node_id, 42);

        let loaded = cp.load("my-stream", 42).await.expect("load");
        assert_eq!(loaded.offset, 999);
    }

    #[tokio::test]
    async fn test_checkpointer_restore() {
        let cp = StreamingCheckpointer::new();
        let checkpoints = vec![
            Checkpoint {
                stream_id: "s1".into(),
                node_id: 1,
                offset: 50,
                checkpoint_ms: 0,
                version: 1,
            },
            Checkpoint {
                stream_id: "s2".into(),
                node_id: 1,
                offset: 75,
                checkpoint_ms: 0,
                version: 2,
            },
        ];
        cp.restore(checkpoints).await;
        assert_eq!(cp.count().await, 2);
        let for_node = cp.checkpoints_for_node(1).await;
        assert_eq!(for_node.len(), 2);
    }
}
