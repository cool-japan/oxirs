//! Dead Letter Queue (DLQ) for failed stream messages.
//!
//! Provides a bounded, time-aware queue for messages that have exceeded retry
//! limits or failed validation.  Supports FIFO pop, topic-based lookup, replay,
//! and reason grouping.

use std::collections::{HashMap, VecDeque};

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Why a message ended up in the dead letter queue.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FailureReason {
    MaxRetriesExceeded,
    Timeout,
    ProcessingError(String),
    SchemaValidationFailed,
    PoisonMessage,
}

impl FailureReason {
    /// A short string label used for grouping.
    fn label(&self) -> String {
        match self {
            FailureReason::MaxRetriesExceeded => "MaxRetriesExceeded".to_string(),
            FailureReason::Timeout => "Timeout".to_string(),
            FailureReason::ProcessingError(msg) => format!("ProcessingError({})", msg),
            FailureReason::SchemaValidationFailed => "SchemaValidationFailed".to_string(),
            FailureReason::PoisonMessage => "PoisonMessage".to_string(),
        }
    }
}

/// A single dead-lettered message.
#[derive(Debug, Clone)]
pub struct DeadLetter {
    pub message_id: String,
    pub payload: Vec<u8>,
    pub original_topic: String,
    pub failure_reason: FailureReason,
    pub retry_count: usize,
    pub first_failed_at: u64,
    pub last_failed_at: u64,
    pub metadata: HashMap<String, String>,
}

/// Configuration for the [`DeadLetterQueue`].
#[derive(Debug, Clone)]
pub struct DlqConfig {
    /// Maximum number of entries the queue may hold.
    pub max_size: usize,
    /// Maximum age of an entry in milliseconds; older entries are purged.
    pub max_age_ms: u64,
    /// Whether the queue allows replaying (removing + returning) entries.
    pub enable_replay: bool,
}

impl Default for DlqConfig {
    fn default() -> Self {
        Self {
            max_size: 10_000,
            max_age_ms: 7 * 24 * 60 * 60 * 1_000, // 7 days
            enable_replay: true,
        }
    }
}

/// Errors returned by [`DeadLetterQueue`] operations.
#[derive(Debug)]
pub enum DlqError {
    QueueFull,
    ReplayDisabled,
}

impl std::fmt::Display for DlqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DlqError::QueueFull => write!(f, "dead letter queue is full"),
            DlqError::ReplayDisabled => write!(f, "replay is disabled for this queue"),
        }
    }
}

impl std::error::Error for DlqError {}

/// A bounded dead letter queue.
pub struct DeadLetterQueue {
    config: DlqConfig,
    letters: VecDeque<DeadLetter>,
    total_received: u64,
}

impl DeadLetterQueue {
    /// Create a new queue with the given configuration.
    pub fn new(config: DlqConfig) -> Self {
        let capacity = config.max_size;
        Self {
            config,
            letters: VecDeque::with_capacity(capacity),
            total_received: 0,
        }
    }

    /// Push a dead letter into the queue.
    ///
    /// Returns [`DlqError::QueueFull`] if the queue has reached `max_size`.
    pub fn push(&mut self, letter: DeadLetter) -> Result<(), DlqError> {
        if self.letters.len() >= self.config.max_size {
            return Err(DlqError::QueueFull);
        }
        self.letters.push_back(letter);
        self.total_received += 1;
        Ok(())
    }

    /// Remove and return the oldest dead letter (FIFO).
    pub fn pop(&mut self) -> Option<DeadLetter> {
        self.letters.pop_front()
    }

    /// Peek at the oldest dead letter without removing it.
    pub fn peek(&self) -> Option<&DeadLetter> {
        self.letters.front()
    }

    /// Number of entries currently in the queue.
    pub fn len(&self) -> usize {
        self.letters.len()
    }

    /// Returns `true` when the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.letters.is_empty()
    }

    /// Total number of dead letters ever pushed into this queue (including
    /// ones that were later popped or purged).
    pub fn total_received(&self) -> u64 {
        self.total_received
    }

    /// Remove entries whose `last_failed_at` is older than `max_age_ms`
    /// relative to `current_time_ms`.
    ///
    /// Returns the number of entries removed.
    pub fn purge_expired(&mut self, current_time_ms: u64) -> usize {
        let cutoff = current_time_ms.saturating_sub(self.config.max_age_ms);
        let before = self.letters.len();
        self.letters.retain(|l| l.last_failed_at >= cutoff);
        before - self.letters.len()
    }

    /// Return references to all entries whose `original_topic` equals `topic`.
    pub fn find_by_topic(&self, topic: &str) -> Vec<&DeadLetter> {
        self.letters
            .iter()
            .filter(|l| l.original_topic == topic)
            .collect()
    }

    /// If `enable_replay` is true, remove the entry with the given
    /// `message_id` and return it for re-processing.
    ///
    /// Returns `None` if no matching entry exists.
    /// Returns `Err(DlqError::ReplayDisabled)` if replay is turned off.
    pub fn replay(&mut self, message_id: &str) -> Result<Option<DeadLetter>, DlqError> {
        if !self.config.enable_replay {
            return Err(DlqError::ReplayDisabled);
        }
        if let Some(pos) = self.letters.iter().position(|l| l.message_id == message_id) {
            Ok(self.letters.remove(pos))
        } else {
            Ok(None)
        }
    }

    /// Group entries by their failure reason label and return a map of
    /// `label → count`.
    pub fn group_by_reason(&self) -> HashMap<String, usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        for letter in &self.letters {
            *map.entry(letter.failure_reason.label()).or_insert(0) += 1;
        }
        map
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_letter(id: &str, topic: &str, reason: FailureReason, last_ms: u64) -> DeadLetter {
        DeadLetter {
            message_id: id.to_string(),
            payload: id.as_bytes().to_vec(),
            original_topic: topic.to_string(),
            failure_reason: reason,
            retry_count: 1,
            first_failed_at: last_ms.saturating_sub(100),
            last_failed_at: last_ms,
            metadata: HashMap::new(),
        }
    }

    fn default_queue() -> DeadLetterQueue {
        DeadLetterQueue::new(DlqConfig {
            max_size: 5,
            max_age_ms: 1_000,
            enable_replay: true,
        })
    }

    // ── push / pop ───────────────────────────────────────────────────────────

    #[test]
    fn test_push_and_pop_single() {
        let mut q = default_queue();
        let letter = make_letter("m1", "t1", FailureReason::Timeout, 1000);
        q.push(letter).expect("push should succeed");
        assert_eq!(q.len(), 1);
        let popped = q.pop().expect("should have an element");
        assert_eq!(popped.message_id, "m1");
        assert!(q.is_empty());
    }

    #[test]
    fn test_push_pop_fifo_order() {
        let mut q = default_queue();
        for i in 0..3u8 {
            q.push(make_letter(
                &format!("m{i}"),
                "t",
                FailureReason::MaxRetriesExceeded,
                1000,
            ))
            .unwrap();
        }
        for i in 0..3u8 {
            let popped = q.pop().unwrap();
            assert_eq!(popped.message_id, format!("m{i}"));
        }
    }

    #[test]
    fn test_pop_empty_returns_none() {
        let mut q = default_queue();
        assert!(q.pop().is_none());
    }

    // ── max_size enforcement ─────────────────────────────────────────────────

    #[test]
    fn test_max_size_enforced() {
        let mut q = default_queue(); // max_size = 5
        for i in 0..5u8 {
            q.push(make_letter(
                &format!("m{i}"),
                "t",
                FailureReason::Timeout,
                1000,
            ))
            .unwrap();
        }
        let result = q.push(make_letter("overflow", "t", FailureReason::Timeout, 1000));
        assert!(matches!(result, Err(DlqError::QueueFull)));
    }

    #[test]
    fn test_queue_full_error_display() {
        let err = DlqError::QueueFull;
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn test_queue_full_after_fill() {
        let mut q = DeadLetterQueue::new(DlqConfig {
            max_size: 2,
            ..DlqConfig::default()
        });
        q.push(make_letter("a", "t", FailureReason::Timeout, 0))
            .unwrap();
        q.push(make_letter("b", "t", FailureReason::Timeout, 0))
            .unwrap();
        assert!(matches!(
            q.push(make_letter("c", "t", FailureReason::Timeout, 0)),
            Err(DlqError::QueueFull)
        ));
    }

    // ── purge_expired ────────────────────────────────────────────────────────

    #[test]
    fn test_purge_expired_removes_old_entries() {
        let mut q = default_queue(); // max_age_ms = 1_000
                                     // old entry: last_failed_at = 0  (age = 2000 > 1000)
        q.push(make_letter("old", "t", FailureReason::Timeout, 0))
            .unwrap();
        // fresh entry: last_failed_at = 1500 (age = 500 < 1000)
        q.push(make_letter("new", "t", FailureReason::Timeout, 1500))
            .unwrap();

        let removed = q.purge_expired(2000);
        assert_eq!(removed, 1);
        assert_eq!(q.len(), 1);
        assert_eq!(q.peek().unwrap().message_id, "new");
    }

    #[test]
    fn test_purge_expired_keeps_all_if_none_old() {
        let mut q = default_queue();
        q.push(make_letter("m1", "t", FailureReason::Timeout, 900))
            .unwrap();
        q.push(make_letter("m2", "t", FailureReason::Timeout, 950))
            .unwrap();
        // current_time = 1000, max_age = 1000 → cutoff = 0, nothing too old
        let removed = q.purge_expired(1000);
        assert_eq!(removed, 0);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_purge_expired_removes_all() {
        let mut q = default_queue();
        for i in 0..3u64 {
            q.push(make_letter(
                &format!("m{i}"),
                "t",
                FailureReason::Timeout,
                i * 10,
            ))
            .unwrap();
        }
        // very large current time → everything is old
        let removed = q.purge_expired(u64::MAX);
        assert_eq!(removed, 3);
        assert!(q.is_empty());
    }

    // ── peek ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_peek_does_not_remove() {
        let mut q = default_queue();
        q.push(make_letter("m1", "t", FailureReason::Timeout, 0))
            .unwrap();
        let peeked = q.peek().unwrap();
        assert_eq!(peeked.message_id, "m1");
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_peek_empty_returns_none() {
        let q = default_queue();
        assert!(q.peek().is_none());
    }

    // ── find_by_topic ────────────────────────────────────────────────────────

    #[test]
    fn test_find_by_topic_returns_matching() {
        let mut q = default_queue();
        q.push(make_letter("a", "topic-A", FailureReason::Timeout, 0))
            .unwrap();
        q.push(make_letter("b", "topic-B", FailureReason::Timeout, 0))
            .unwrap();
        q.push(make_letter("c", "topic-A", FailureReason::Timeout, 0))
            .unwrap();

        let found = q.find_by_topic("topic-A");
        assert_eq!(found.len(), 2);
        assert!(found.iter().all(|l| l.original_topic == "topic-A"));
    }

    #[test]
    fn test_find_by_topic_none_matching() {
        let mut q = default_queue();
        q.push(make_letter("a", "topic-A", FailureReason::Timeout, 0))
            .unwrap();
        let found = q.find_by_topic("topic-Z");
        assert!(found.is_empty());
    }

    #[test]
    fn test_find_by_topic_empty_queue() {
        let q = default_queue();
        assert!(q.find_by_topic("anything").is_empty());
    }

    // ── replay ───────────────────────────────────────────────────────────────

    #[test]
    fn test_replay_removes_and_returns_entry() {
        let mut q = default_queue();
        q.push(make_letter("m1", "t", FailureReason::Timeout, 0))
            .unwrap();
        q.push(make_letter("m2", "t", FailureReason::Timeout, 0))
            .unwrap();

        let replayed = q.replay("m1").unwrap().unwrap();
        assert_eq!(replayed.message_id, "m1");
        assert_eq!(q.len(), 1);
        assert_eq!(q.pop().unwrap().message_id, "m2");
    }

    #[test]
    fn test_replay_missing_id_returns_none() {
        let mut q = default_queue();
        q.push(make_letter("m1", "t", FailureReason::Timeout, 0))
            .unwrap();
        let result = q.replay("missing").unwrap();
        assert!(result.is_none());
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_replay_disabled_returns_error() {
        let mut q = DeadLetterQueue::new(DlqConfig {
            max_size: 10,
            max_age_ms: 1_000,
            enable_replay: false,
        });
        q.push(make_letter("m1", "t", FailureReason::Timeout, 0))
            .unwrap();
        assert!(matches!(q.replay("m1"), Err(DlqError::ReplayDisabled)));
    }

    #[test]
    fn test_replay_disabled_error_display() {
        let err = DlqError::ReplayDisabled;
        assert!(!err.to_string().is_empty());
    }

    // ── group_by_reason ──────────────────────────────────────────────────────

    #[test]
    fn test_group_by_reason_counts() {
        let mut q = default_queue();
        q.push(make_letter("a", "t", FailureReason::Timeout, 0))
            .unwrap();
        q.push(make_letter("b", "t", FailureReason::Timeout, 0))
            .unwrap();
        q.push(make_letter("c", "t", FailureReason::MaxRetriesExceeded, 0))
            .unwrap();

        let groups = q.group_by_reason();
        assert_eq!(groups["Timeout"], 2);
        assert_eq!(groups["MaxRetriesExceeded"], 1);
    }

    #[test]
    fn test_group_by_reason_empty_queue() {
        let q = default_queue();
        assert!(q.group_by_reason().is_empty());
    }

    #[test]
    fn test_group_by_reason_processing_error() {
        let mut q = default_queue();
        q.push(make_letter(
            "e",
            "t",
            FailureReason::ProcessingError("oops".to_string()),
            0,
        ))
        .unwrap();
        let groups = q.group_by_reason();
        assert_eq!(groups.len(), 1);
        assert!(groups.keys().any(|k| k.contains("ProcessingError")));
    }

    // ── total_received ───────────────────────────────────────────────────────

    #[test]
    fn test_total_received_increments_on_push() {
        let mut q = default_queue();
        assert_eq!(q.total_received(), 0);
        q.push(make_letter("m1", "t", FailureReason::Timeout, 0))
            .unwrap();
        assert_eq!(q.total_received(), 1);
        q.push(make_letter("m2", "t", FailureReason::Timeout, 0))
            .unwrap();
        assert_eq!(q.total_received(), 2);
    }

    #[test]
    fn test_total_received_does_not_decrement_on_pop() {
        let mut q = default_queue();
        q.push(make_letter("m1", "t", FailureReason::Timeout, 0))
            .unwrap();
        q.pop();
        assert_eq!(q.total_received(), 1);
    }

    #[test]
    fn test_total_received_not_incremented_on_full_push() {
        let mut q = DeadLetterQueue::new(DlqConfig {
            max_size: 1,
            ..DlqConfig::default()
        });
        q.push(make_letter("m1", "t", FailureReason::Timeout, 0))
            .unwrap();
        // This should fail — total_received must not increase
        let _ = q.push(make_letter("m2", "t", FailureReason::Timeout, 0));
        assert_eq!(q.total_received(), 1);
    }

    // ── FailureReason variants ───────────────────────────────────────────────

    #[test]
    fn test_failure_reason_max_retries() {
        let reason = FailureReason::MaxRetriesExceeded;
        assert_eq!(reason.label(), "MaxRetriesExceeded");
    }

    #[test]
    fn test_failure_reason_timeout() {
        let reason = FailureReason::Timeout;
        assert_eq!(reason.label(), "Timeout");
    }

    #[test]
    fn test_failure_reason_processing_error() {
        let reason = FailureReason::ProcessingError("bad data".to_string());
        assert!(reason.label().contains("ProcessingError"));
        assert!(reason.label().contains("bad data"));
    }

    #[test]
    fn test_failure_reason_schema_validation_failed() {
        let reason = FailureReason::SchemaValidationFailed;
        assert_eq!(reason.label(), "SchemaValidationFailed");
    }

    #[test]
    fn test_failure_reason_poison_message() {
        let reason = FailureReason::PoisonMessage;
        assert_eq!(reason.label(), "PoisonMessage");
    }

    // ── misc / edge cases ────────────────────────────────────────────────────

    #[test]
    fn test_is_empty_initial() {
        let q = default_queue();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_len_after_push_pop() {
        let mut q = default_queue();
        q.push(make_letter("m1", "t", FailureReason::Timeout, 0))
            .unwrap();
        assert_eq!(q.len(), 1);
        q.pop();
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn test_metadata_preserved() {
        let mut q = default_queue();
        let mut meta = HashMap::new();
        meta.insert("env".to_string(), "prod".to_string());
        let letter = DeadLetter {
            message_id: "m".to_string(),
            payload: b"data".to_vec(),
            original_topic: "topic".to_string(),
            failure_reason: FailureReason::Timeout,
            retry_count: 3,
            first_failed_at: 100,
            last_failed_at: 200,
            metadata: meta.clone(),
        };
        q.push(letter).unwrap();
        let popped = q.pop().unwrap();
        assert_eq!(popped.metadata["env"], "prod");
    }

    #[test]
    fn test_payload_preserved() {
        let mut q = default_queue();
        let payload = b"hello world".to_vec();
        let letter = DeadLetter {
            message_id: "p".to_string(),
            payload: payload.clone(),
            original_topic: "t".to_string(),
            failure_reason: FailureReason::PoisonMessage,
            retry_count: 0,
            first_failed_at: 0,
            last_failed_at: 0,
            metadata: HashMap::new(),
        };
        q.push(letter).unwrap();
        assert_eq!(q.pop().unwrap().payload, payload);
    }

    #[test]
    fn test_retry_count_preserved() {
        let mut q = default_queue();
        let mut letter = make_letter("m", "t", FailureReason::Timeout, 0);
        letter.retry_count = 42;
        q.push(letter).unwrap();
        assert_eq!(q.pop().unwrap().retry_count, 42);
    }

    #[test]
    fn test_purge_then_push_succeeds() {
        let mut q = DeadLetterQueue::new(DlqConfig {
            max_size: 2,
            max_age_ms: 100,
            enable_replay: true,
        });
        q.push(make_letter("old", "t", FailureReason::Timeout, 0))
            .unwrap();
        q.push(make_letter("old2", "t", FailureReason::Timeout, 0))
            .unwrap();
        // Queue full — purge old entries
        q.purge_expired(200);
        // Now there is room
        q.push(make_letter("new", "t", FailureReason::Timeout, 150))
            .unwrap();
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_replay_middle_element() {
        let mut q = default_queue();
        q.push(make_letter("a", "t", FailureReason::Timeout, 0))
            .unwrap();
        q.push(make_letter("b", "t", FailureReason::Timeout, 0))
            .unwrap();
        q.push(make_letter("c", "t", FailureReason::Timeout, 0))
            .unwrap();

        let replayed = q.replay("b").unwrap().unwrap();
        assert_eq!(replayed.message_id, "b");
        assert_eq!(q.len(), 2);
        // Remaining should be a, c in order
        assert_eq!(q.pop().unwrap().message_id, "a");
        assert_eq!(q.pop().unwrap().message_id, "c");
    }

    #[test]
    fn test_group_by_reason_all_variants() {
        let mut q = DeadLetterQueue::new(DlqConfig {
            max_size: 10,
            ..DlqConfig::default()
        });
        q.push(make_letter("a", "t", FailureReason::MaxRetriesExceeded, 0))
            .unwrap();
        q.push(make_letter("b", "t", FailureReason::Timeout, 0))
            .unwrap();
        q.push(make_letter(
            "c",
            "t",
            FailureReason::ProcessingError("e".into()),
            0,
        ))
        .unwrap();
        q.push(make_letter(
            "d",
            "t",
            FailureReason::SchemaValidationFailed,
            0,
        ))
        .unwrap();
        q.push(make_letter("e", "t", FailureReason::PoisonMessage, 0))
            .unwrap();

        let groups = q.group_by_reason();
        assert_eq!(groups.len(), 5);
        assert_eq!(groups["MaxRetriesExceeded"], 1);
        assert_eq!(groups["Timeout"], 1);
        assert_eq!(groups["SchemaValidationFailed"], 1);
        assert_eq!(groups["PoisonMessage"], 1);
    }

    #[test]
    fn test_find_by_topic_all_same_topic() {
        let mut q = default_queue();
        for i in 0..4u8 {
            q.push(make_letter(
                &format!("m{i}"),
                "same-topic",
                FailureReason::Timeout,
                0,
            ))
            .unwrap();
        }
        assert_eq!(q.find_by_topic("same-topic").len(), 4);
    }

    #[test]
    fn test_failure_reason_equality() {
        assert_eq!(FailureReason::Timeout, FailureReason::Timeout);
        assert_ne!(FailureReason::Timeout, FailureReason::PoisonMessage);
        assert_eq!(
            FailureReason::ProcessingError("x".into()),
            FailureReason::ProcessingError("x".into())
        );
        assert_ne!(
            FailureReason::ProcessingError("x".into()),
            FailureReason::ProcessingError("y".into())
        );
    }

    #[test]
    fn test_dlq_default_config() {
        let cfg = DlqConfig::default();
        assert!(cfg.max_size > 0);
        assert!(cfg.max_age_ms > 0);
        assert!(cfg.enable_replay);
    }

    #[test]
    fn test_push_after_pop_succeeds_on_full_queue() {
        let mut q = DeadLetterQueue::new(DlqConfig {
            max_size: 1,
            ..DlqConfig::default()
        });
        q.push(make_letter("m1", "t", FailureReason::Timeout, 0))
            .unwrap();
        // Queue is full - pop to make room
        q.pop();
        q.push(make_letter("m2", "t", FailureReason::Timeout, 0))
            .unwrap();
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_total_received_after_replay() {
        let mut q = default_queue();
        q.push(make_letter("m1", "t", FailureReason::Timeout, 0))
            .unwrap();
        let _ = q.replay("m1").unwrap();
        // total_received should not decrement on replay
        assert_eq!(q.total_received(), 1);
    }

    #[test]
    fn test_group_by_reason_two_of_same() {
        let mut q = default_queue();
        q.push(make_letter("a", "t", FailureReason::PoisonMessage, 0))
            .unwrap();
        q.push(make_letter("b", "t", FailureReason::PoisonMessage, 0))
            .unwrap();
        let groups = q.group_by_reason();
        assert_eq!(groups.get("PoisonMessage").copied().unwrap_or(0), 2);
    }

    #[test]
    fn test_find_by_topic_does_not_consume() {
        let mut q = default_queue();
        q.push(make_letter("m1", "topic-X", FailureReason::Timeout, 0))
            .unwrap();
        let found = q.find_by_topic("topic-X");
        assert_eq!(found.len(), 1);
        // Queue still has the item
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn test_first_failed_at_preserved() {
        let mut q = default_queue();
        let mut letter = make_letter("m", "t", FailureReason::Timeout, 500);
        letter.first_failed_at = 100;
        q.push(letter).unwrap();
        let popped = q.pop().unwrap();
        assert_eq!(popped.first_failed_at, 100);
        assert_eq!(popped.last_failed_at, 500);
    }

    #[test]
    fn test_replay_on_empty_queue_returns_none() {
        let mut q = default_queue();
        let result = q.replay("nonexistent").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_purge_returns_correct_count() {
        let mut q = DeadLetterQueue::new(DlqConfig {
            max_size: 10,
            max_age_ms: 100,
            enable_replay: true,
        });
        for i in 0..5u64 {
            // All very old
            q.push(make_letter(
                &format!("old{i}"),
                "t",
                FailureReason::Timeout,
                i,
            ))
            .unwrap();
        }
        for i in 0..3u64 {
            // These are fresh (within max_age_ms=100)
            q.push(make_letter(
                &format!("new{i}"),
                "t",
                FailureReason::Timeout,
                950 + i,
            ))
            .unwrap();
        }
        let removed = q.purge_expired(1000);
        assert_eq!(removed, 5);
        assert_eq!(q.len(), 3);
    }
}
