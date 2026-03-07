//! Efficient change detection for RDF datasets.
//!
//! `ChangeTracker` records every mutation applied to an RDF dataset and emits
//! structured `ChangeEvent` values that downstream components (subscription
//! manager, broadcaster) can act on.
//!
//! Design goals:
//! - Zero-copy hot path: individual change events are allocated once and cloned
//!   only when multiple subscribers are interested.
//! - Atomic sequence numbering so consumers can detect gaps and request replays.
//! - Optional named-graph context so per-graph subscriptions can be filtered
//!   cheaply before any serialisation occurs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use tokio::sync::broadcast;
use tracing::{debug, warn};

/// The kind of mutation that produced a `ChangeEvent`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeType {
    /// One or more triples were added to the dataset.
    Insert,
    /// One or more triples were removed from the dataset.
    Delete,
    /// An existing triple's object was replaced (delete + insert in one logical operation).
    Update,
}

impl std::fmt::Display for ChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChangeType::Insert => f.write_str("Insert"),
            ChangeType::Delete => f.write_str("Delete"),
            ChangeType::Update => f.write_str("Update"),
        }
    }
}

/// A single mutation event recorded by the `ChangeTracker`.
///
/// All three of `subject`, `predicate`, and `object` carry the canonical
/// string representation of the RDF term (IRI in angle brackets, literals
/// with datatype/language tag, blank node labels).
#[derive(Debug, Clone)]
pub struct ChangeEvent {
    /// Monotonically increasing sequence number, unique within this tracker instance.
    pub sequence: u64,
    /// The nature of the change.
    pub event_type: ChangeType,
    /// The subject of the affected triple.
    pub subject: String,
    /// The predicate of the affected triple.
    pub predicate: String,
    /// The object of the affected triple (new value for `Update`).
    pub object: String,
    /// Wall-clock time at which the change was recorded.
    pub timestamp: SystemTime,
    /// Optional IRI of the named graph that was modified; `None` for the default graph.
    pub graph: Option<String>,
}

impl ChangeEvent {
    /// Construct a new event, assigning the given sequence number.
    pub fn new(
        sequence: u64,
        event_type: ChangeType,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        graph: Option<String>,
    ) -> Self {
        Self {
            sequence,
            event_type,
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            timestamp: SystemTime::now(),
            graph,
        }
    }

    /// Returns `true` when this event modified the default (unnamed) graph.
    pub fn is_default_graph(&self) -> bool {
        self.graph.is_none()
    }
}

/// Statistics snapshot for a `ChangeTracker` instance.
#[derive(Debug, Clone)]
pub struct ChangeTrackerStats {
    /// Total number of change events recorded since creation.
    pub total_recorded: u64,
    /// Sequence number of the most recently recorded event, or `None` if empty.
    pub last_sequence: Option<u64>,
    /// Number of active broadcast receivers currently listening.
    pub active_listeners: usize,
}

/// Efficient, thread-safe recorder of RDF dataset mutations.
///
/// `ChangeTracker` wraps a `tokio::sync::broadcast` channel.  Every call to
/// `record` atomically allocates the next sequence number, constructs a
/// `ChangeEvent`, and fans it out to all current receivers.
///
/// Callers obtain a `broadcast::Receiver<ChangeEvent>` by calling
/// `subscribe`.  Receivers are independent – a slow consumer only affects
/// itself (events are dropped from its personal ring buffer, not lost
/// globally).
pub struct ChangeTracker {
    sender: broadcast::Sender<Arc<ChangeEvent>>,
    next_sequence: Arc<AtomicU64>,
    total_recorded: Arc<AtomicU64>,
}

impl std::fmt::Debug for ChangeTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChangeTracker")
            .field("next_sequence", &self.next_sequence.load(Ordering::Relaxed))
            .field(
                "total_recorded",
                &self.total_recorded.load(Ordering::Relaxed),
            )
            .field("active_listeners", &self.sender.receiver_count())
            .finish()
    }
}

impl ChangeTracker {
    /// Create a new tracker with the given broadcast channel capacity.
    ///
    /// `capacity` controls how many events can be buffered per receiver before
    /// older ones are overwritten (receivers that fall behind receive a
    /// `broadcast::error::RecvError::Lagged` error).
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity.max(1));
        Self {
            sender,
            next_sequence: Arc::new(AtomicU64::new(1)),
            total_recorded: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Record an RDF mutation and broadcast it to all listeners.
    ///
    /// Returns the sequence number assigned to the new event, or `None` if
    /// there were no active receivers (the event is still recorded internally
    /// but nobody was listening).
    pub fn record(
        &self,
        event_type: ChangeType,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        graph: Option<String>,
    ) -> u64 {
        let sequence = self.next_sequence.fetch_add(1, Ordering::SeqCst);
        self.total_recorded.fetch_add(1, Ordering::Relaxed);

        let event = Arc::new(ChangeEvent::new(
            sequence, event_type, subject, predicate, object, graph,
        ));

        match self.sender.send(event) {
            Ok(n) => {
                debug!(sequence, receivers = n, "ChangeEvent broadcast");
            }
            Err(_) => {
                debug!(sequence, "ChangeEvent recorded; no active receivers");
            }
        }

        sequence
    }

    /// Convenience wrapper – record an `Insert` event.
    pub fn record_insert(
        &self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        graph: Option<String>,
    ) -> u64 {
        self.record(ChangeType::Insert, subject, predicate, object, graph)
    }

    /// Convenience wrapper – record a `Delete` event.
    pub fn record_delete(
        &self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        graph: Option<String>,
    ) -> u64 {
        self.record(ChangeType::Delete, subject, predicate, object, graph)
    }

    /// Convenience wrapper – record an `Update` event (old value replaced by new).
    pub fn record_update(
        &self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        new_object: impl Into<String>,
        graph: Option<String>,
    ) -> u64 {
        self.record(ChangeType::Update, subject, predicate, new_object, graph)
    }

    /// Subscribe to change events.
    ///
    /// Each subscriber receives its own independent view of the stream.
    /// Events published before this call was made are **not** replayed (use
    /// the `SubscriptionMultiplexer` with a `ResumeToken` for that).
    pub fn subscribe(&self) -> broadcast::Receiver<Arc<ChangeEvent>> {
        self.sender.subscribe()
    }

    /// Return a statistics snapshot.
    pub fn stats(&self) -> ChangeTrackerStats {
        let total = self.total_recorded.load(Ordering::Relaxed);
        let next = self.next_sequence.load(Ordering::Relaxed);
        ChangeTrackerStats {
            total_recorded: total,
            last_sequence: if total == 0 { None } else { Some(next - 1) },
            active_listeners: self.sender.receiver_count(),
        }
    }

    /// Number of broadcast receivers currently subscribed.
    pub fn listener_count(&self) -> usize {
        self.sender.receiver_count()
    }
}

/// A `ChangeTracker` that batches mutations recorded within a logical
/// transaction and publishes them as a single atomic notification on commit.
///
/// Uncommitted events are buffered in memory; if the batch is dropped without
/// calling `commit` the changes are silently discarded (rolled back).
pub struct BatchChangeTracker {
    tracker: Arc<ChangeTracker>,
    pending: Vec<(ChangeType, String, String, String, Option<String>)>,
}

impl BatchChangeTracker {
    /// Start a new batch rooted in the given `ChangeTracker`.
    pub fn begin(tracker: Arc<ChangeTracker>) -> Self {
        Self {
            tracker,
            pending: Vec::new(),
        }
    }

    /// Stage an insert mutation in this batch.
    pub fn stage_insert(
        &mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        graph: Option<String>,
    ) {
        self.pending.push((
            ChangeType::Insert,
            subject.into(),
            predicate.into(),
            object.into(),
            graph,
        ));
    }

    /// Stage a delete mutation in this batch.
    pub fn stage_delete(
        &mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        graph: Option<String>,
    ) {
        self.pending.push((
            ChangeType::Delete,
            subject.into(),
            predicate.into(),
            object.into(),
            graph,
        ));
    }

    /// Number of staged (uncommitted) mutations.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Commit all staged mutations to the underlying tracker.
    ///
    /// Publishes each staged event in order.  Returns the sequence number of
    /// the last committed event, or `None` if the batch was empty.
    pub fn commit(self) -> Option<u64> {
        if self.pending.is_empty() {
            warn!("BatchChangeTracker committed with no staged events");
            return None;
        }

        let mut last_seq = 0u64;
        for (event_type, subject, predicate, object, graph) in self.pending {
            last_seq = self
                .tracker
                .record(event_type, subject, predicate, object, graph);
        }
        Some(last_seq)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::timeout;

    fn tracker() -> ChangeTracker {
        ChangeTracker::new(64)
    }

    #[test]
    fn test_change_type_display() {
        assert_eq!(ChangeType::Insert.to_string(), "Insert");
        assert_eq!(ChangeType::Delete.to_string(), "Delete");
        assert_eq!(ChangeType::Update.to_string(), "Update");
    }

    #[test]
    fn test_change_type_equality() {
        assert_eq!(ChangeType::Insert, ChangeType::Insert);
        assert_ne!(ChangeType::Insert, ChangeType::Delete);
    }

    #[test]
    fn test_change_event_new() {
        let ev = ChangeEvent::new(
            1,
            ChangeType::Insert,
            "http://ex.org/s",
            "http://ex.org/p",
            "http://ex.org/o",
            None,
        );
        assert_eq!(ev.sequence, 1);
        assert_eq!(ev.event_type, ChangeType::Insert);
        assert!(ev.is_default_graph());
    }

    #[test]
    fn test_change_event_named_graph() {
        let ev = ChangeEvent::new(
            2,
            ChangeType::Delete,
            "s",
            "p",
            "o",
            Some("http://ex.org/g".to_string()),
        );
        assert!(!ev.is_default_graph());
        assert_eq!(ev.graph.as_deref(), Some("http://ex.org/g"));
    }

    #[test]
    fn test_tracker_stats_empty() {
        let t = tracker();
        let stats = t.stats();
        assert_eq!(stats.total_recorded, 0);
        assert!(stats.last_sequence.is_none());
        assert_eq!(stats.active_listeners, 0);
    }

    #[test]
    fn test_tracker_listener_count() {
        let t = tracker();
        assert_eq!(t.listener_count(), 0);
        let _r1 = t.subscribe();
        assert_eq!(t.listener_count(), 1);
        let _r2 = t.subscribe();
        assert_eq!(t.listener_count(), 2);
    }

    #[tokio::test]
    async fn test_record_insert_broadcasts_event() {
        let t = tracker();
        let mut rx = t.subscribe();

        let seq = t.record_insert("s", "p", "o", None);
        assert_eq!(seq, 1);

        let ev = timeout(Duration::from_millis(100), rx.recv())
            .await
            .expect("no timeout")
            .expect("received");

        assert_eq!(ev.sequence, 1);
        assert_eq!(ev.event_type, ChangeType::Insert);
    }

    #[tokio::test]
    async fn test_record_delete_broadcasts_event() {
        let t = tracker();
        let mut rx = t.subscribe();

        t.record_delete("s", "p", "o", None);

        let ev = timeout(Duration::from_millis(100), rx.recv())
            .await
            .expect("no timeout")
            .expect("received");

        assert_eq!(ev.event_type, ChangeType::Delete);
    }

    #[tokio::test]
    async fn test_record_update_broadcasts_event() {
        let t = tracker();
        let mut rx = t.subscribe();

        t.record_update("s", "p", "new_val", Some("http://ex.org/g".to_string()));

        let ev = timeout(Duration::from_millis(100), rx.recv())
            .await
            .expect("no timeout")
            .expect("received");

        assert_eq!(ev.event_type, ChangeType::Update);
        assert_eq!(ev.graph.as_deref(), Some("http://ex.org/g"));
    }

    #[test]
    fn test_sequence_numbers_are_monotonic() {
        let t = tracker();
        let s1 = t.record_insert("s1", "p", "o", None);
        let s2 = t.record_insert("s2", "p", "o", None);
        let s3 = t.record_delete("s3", "p", "o", None);
        assert!(s1 < s2);
        assert!(s2 < s3);
    }

    #[test]
    fn test_stats_after_records() {
        let t = tracker();
        t.record_insert("s", "p", "o", None);
        t.record_insert("s", "p", "o2", None);
        let stats = t.stats();
        assert_eq!(stats.total_recorded, 2);
        assert_eq!(stats.last_sequence, Some(2));
    }

    #[test]
    fn test_batch_tracker_stage_and_commit() {
        let t = Arc::new(tracker());
        let mut batch = BatchChangeTracker::begin(Arc::clone(&t));
        batch.stage_insert("s", "p", "o", None);
        batch.stage_delete("s2", "p2", "o2", None);
        assert_eq!(batch.pending_count(), 2);

        let last_seq = batch.commit();
        assert!(last_seq.is_some());
        assert_eq!(t.stats().total_recorded, 2);
    }

    #[test]
    fn test_batch_tracker_empty_commit_returns_none() {
        let t = Arc::new(tracker());
        let batch = BatchChangeTracker::begin(Arc::clone(&t));
        let last_seq = batch.commit();
        assert!(last_seq.is_none());
        assert_eq!(t.stats().total_recorded, 0);
    }

    #[tokio::test]
    async fn test_multiple_subscribers_all_receive_event() {
        let t = tracker();
        let mut rx1 = t.subscribe();
        let mut rx2 = t.subscribe();

        t.record_insert("s", "p", "o", None);

        let e1 = timeout(Duration::from_millis(100), rx1.recv()).await;
        let e2 = timeout(Duration::from_millis(100), rx2.recv()).await;
        assert!(e1.is_ok());
        assert!(e2.is_ok());
    }
}
