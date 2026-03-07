//! Event replay buffer with seek and position tracking.
//!
//! Provides an in-memory ring buffer for stream events that supports
//! random access via sequence IDs, timestamps, or byte offsets. Useful
//! for replaying historical events to late-joining consumers or during
//! failure recovery.

use std::collections::VecDeque;
use std::fmt;

// ── Error ────────────────────────────────────────────────────────────────────

/// Errors that can occur during replay buffer operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayError {
    /// Buffer is at maximum capacity and `append` was called without eviction enabled.
    BufferFull,
    /// Seek position is structurally invalid (e.g. offset beyond end).
    InvalidPosition(String),
    /// No event with the requested sequence ID exists in the buffer.
    SequenceNotFound(u64),
    /// No event with a timestamp >= the requested value exists in the buffer.
    TimestampNotFound(u64),
}

impl fmt::Display for ReplayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BufferFull => write!(f, "replay buffer is full"),
            Self::InvalidPosition(msg) => write!(f, "invalid seek position: {msg}"),
            Self::SequenceNotFound(id) => write!(f, "sequence id {id} not found in buffer"),
            Self::TimestampNotFound(ts) => write!(f, "no event at or after timestamp {ts}"),
        }
    }
}

impl std::error::Error for ReplayError {}

// ── Core types ────────────────────────────────────────────────────────────────

/// A single event stored in the replay buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamEvent {
    /// Monotonically increasing sequence identifier.
    pub sequence_id: u64,
    /// Unix epoch milliseconds at event creation.
    pub timestamp: u64,
    /// Partition this event belongs to.
    pub partition: u32,
    /// Optional routing / deduplication key.
    pub key: Option<String>,
    /// Raw event payload.
    pub payload: Vec<u8>,
}

impl StreamEvent {
    /// Create a new `StreamEvent`.
    pub fn new(
        sequence_id: u64,
        timestamp: u64,
        partition: u32,
        key: Option<String>,
        payload: Vec<u8>,
    ) -> Self {
        Self {
            sequence_id,
            timestamp,
            partition,
            key,
            payload,
        }
    }

    /// Total bytes occupied by the payload.
    pub fn payload_bytes(&self) -> usize {
        self.payload.len()
    }
}

/// Describes where a seek operation should move the cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeekPosition {
    /// Move to the very first event in the buffer.
    Beginning,
    /// Move to one past the last event (EOF).
    End,
    /// Move to the event with this exact sequence ID.
    SequenceId(u64),
    /// Move to the first event whose `timestamp >=` the supplied value.
    Timestamp(u64),
    /// Move to an absolute buffer offset (0-based index).
    Offset(usize),
}

/// Snapshot of replay buffer statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayStats {
    /// Total events currently held in the buffer.
    pub total_events: usize,
    /// Number of events that have been delivered by `read_next` / `read_batch`.
    pub replayed_events: usize,
    /// Current cursor position (0-based index into the buffer).
    pub current_position: usize,
    /// Sum of all payload bytes delivered so far.
    pub bytes_replayed: usize,
}

// ── ReplayBuffer ─────────────────────────────────────────────────────────────

/// An in-memory event replay buffer.
///
/// Events are appended sequentially. When the buffer exceeds `max_capacity`
/// the oldest event is evicted automatically. A cursor tracks the current
/// read position; seek / read operations advance it.
pub struct ReplayBuffer {
    events: VecDeque<StreamEvent>,
    max_capacity: usize,
    /// Index of the next event to read (logical position inside `events`).
    cursor: usize,
    replayed_events: usize,
    bytes_replayed: usize,
}

impl ReplayBuffer {
    /// Create a new `ReplayBuffer` with the given maximum capacity.
    ///
    /// When `max_capacity` is 0 the buffer has no effective limit (usize::MAX).
    pub fn new(max_capacity: usize) -> Self {
        let effective_capacity = if max_capacity == 0 {
            usize::MAX
        } else {
            max_capacity
        };
        Self {
            events: VecDeque::new(),
            max_capacity: effective_capacity,
            cursor: 0,
            replayed_events: 0,
            bytes_replayed: 0,
        }
    }

    /// Append a new event to the buffer.
    ///
    /// If the buffer is at `max_capacity` the oldest event is evicted and the
    /// cursor is adjusted so it never points past the start.
    pub fn append(&mut self, event: StreamEvent) -> Result<(), ReplayError> {
        if self.events.len() >= self.max_capacity {
            // Evict the oldest event; adjust cursor if it was pointing at it.
            self.events.pop_front();
            if self.cursor > 0 {
                self.cursor -= 1;
            }
        }
        self.events.push_back(event);
        Ok(())
    }

    /// Move the read cursor to `pos`.
    ///
    /// Returns the new cursor position on success.
    pub fn seek(&mut self, pos: SeekPosition) -> Result<usize, ReplayError> {
        let new_cursor = match pos {
            SeekPosition::Beginning => 0,
            SeekPosition::End => self.events.len(),
            SeekPosition::SequenceId(id) => self
                .events
                .iter()
                .position(|e| e.sequence_id == id)
                .ok_or(ReplayError::SequenceNotFound(id))?,
            SeekPosition::Timestamp(ts) => self
                .events
                .iter()
                .position(|e| e.timestamp >= ts)
                .ok_or(ReplayError::TimestampNotFound(ts))?,
            SeekPosition::Offset(off) => {
                if off > self.events.len() {
                    return Err(ReplayError::InvalidPosition(format!(
                        "offset {off} exceeds buffer length {}",
                        self.events.len()
                    )));
                }
                off
            }
        };
        self.cursor = new_cursor;
        Ok(self.cursor)
    }

    /// Read the event at the current cursor position and advance the cursor.
    ///
    /// Returns `None` when the cursor is past the last event.
    pub fn read_next(&mut self) -> Option<&StreamEvent> {
        if self.cursor >= self.events.len() {
            return None;
        }
        // Capture payload length before any mutation.
        let payload_len = self.events[self.cursor].payload.len();
        let idx = self.cursor;
        self.cursor += 1;
        self.replayed_events += 1;
        self.bytes_replayed += payload_len;
        self.events.get(idx)
    }

    /// Read up to `count` events starting at the current cursor, advancing
    /// the cursor past each returned event.
    pub fn read_batch(&mut self, count: usize) -> Vec<&StreamEvent> {
        let available = self.events.len().saturating_sub(self.cursor);
        let to_read = count.min(available);
        let start = self.cursor;
        self.cursor += to_read;

        // Accumulate payload stats.
        let mut bytes = 0usize;
        for i in start..self.cursor {
            if let Some(e) = self.events.get(i) {
                bytes += e.payload.len();
            }
        }
        self.replayed_events += to_read;
        self.bytes_replayed += bytes;

        // Collect references from the range.
        (start..self.cursor)
            .filter_map(|i| self.events.get(i))
            .collect()
    }

    /// Reset the cursor to the beginning of the buffer.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Number of events remaining from the cursor to the end of the buffer.
    pub fn events_remaining(&self) -> usize {
        self.events.len().saturating_sub(self.cursor)
    }

    /// Snapshot of current buffer statistics.
    pub fn stats(&self) -> ReplayStats {
        ReplayStats {
            total_events: self.events.len(),
            replayed_events: self.replayed_events,
            current_position: self.cursor,
            bytes_replayed: self.bytes_replayed,
        }
    }

    /// Total number of events currently stored in the buffer.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Returns `true` if the buffer contains no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(seq: u64, ts: u64, partition: u32, payload: &[u8]) -> StreamEvent {
        StreamEvent::new(seq, ts, partition, None, payload.to_vec())
    }

    fn make_keyed_event(seq: u64, ts: u64, key: &str, payload: &[u8]) -> StreamEvent {
        StreamEvent::new(seq, ts, 0, Some(key.to_string()), payload.to_vec())
    }

    // ── StreamEvent basics ──────────────────────────────────────────────────

    #[test]
    fn test_stream_event_new() {
        let e = make_event(1, 1000, 0, b"hello");
        assert_eq!(e.sequence_id, 1);
        assert_eq!(e.timestamp, 1000);
        assert_eq!(e.partition, 0);
        assert_eq!(e.payload, b"hello");
        assert!(e.key.is_none());
    }

    #[test]
    fn test_stream_event_with_key() {
        let e = make_keyed_event(2, 2000, "my-key", b"data");
        assert_eq!(e.key, Some("my-key".to_string()));
    }

    #[test]
    fn test_stream_event_payload_bytes() {
        let e = make_event(1, 0, 0, b"hello world");
        assert_eq!(e.payload_bytes(), 11);
    }

    #[test]
    fn test_stream_event_empty_payload() {
        let e = make_event(1, 0, 0, b"");
        assert_eq!(e.payload_bytes(), 0);
    }

    // ── ReplayBuffer construction ───────────────────────────────────────────

    #[test]
    fn test_new_buffer_is_empty() {
        let buf = ReplayBuffer::new(100);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_zero_capacity_means_unlimited() {
        let buf = ReplayBuffer::new(0);
        assert_eq!(buf.max_capacity, usize::MAX);
    }

    // ── append ─────────────────────────────────────────────────────────────

    #[test]
    fn test_append_single_event() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 100, 0, b"a")).expect("append ok");
        assert_eq!(buf.len(), 1);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_append_multiple_events() {
        let mut buf = ReplayBuffer::new(10);
        for i in 0..5u64 {
            buf.append(make_event(i, i * 100, 0, b"x"))
                .expect("append ok");
        }
        assert_eq!(buf.len(), 5);
    }

    #[test]
    fn test_append_evicts_oldest_on_full() {
        let mut buf = ReplayBuffer::new(3);
        buf.append(make_event(1, 100, 0, b"a")).expect("ok");
        buf.append(make_event(2, 200, 0, b"b")).expect("ok");
        buf.append(make_event(3, 300, 0, b"c")).expect("ok");
        // Buffer full — appending seq 4 should evict seq 1.
        buf.append(make_event(4, 400, 0, b"d")).expect("ok");
        assert_eq!(buf.len(), 3);
        // seq 1 should no longer be present.
        let ids: Vec<u64> = buf.events.iter().map(|e| e.sequence_id).collect();
        assert_eq!(ids, vec![2, 3, 4]);
    }

    #[test]
    fn test_append_eviction_adjusts_cursor() {
        let mut buf = ReplayBuffer::new(3);
        buf.append(make_event(1, 100, 0, b"a")).expect("ok");
        buf.append(make_event(2, 200, 0, b"b")).expect("ok");
        buf.append(make_event(3, 300, 0, b"c")).expect("ok");
        // Advance cursor to position 2.
        buf.seek(SeekPosition::Offset(2)).expect("seek ok");
        assert_eq!(buf.cursor, 2);
        // Eviction should move cursor back by 1.
        buf.append(make_event(4, 400, 0, b"d")).expect("ok");
        assert_eq!(buf.cursor, 1);
    }

    // ── read_next ──────────────────────────────────────────────────────────

    #[test]
    fn test_read_next_returns_events_in_order() {
        let mut buf = ReplayBuffer::new(10);
        for i in 1u64..=3 {
            buf.append(make_event(i, i * 10, 0, b"x")).expect("ok");
        }
        let e1 = buf.read_next().expect("has event");
        assert_eq!(e1.sequence_id, 1);
        let e2 = buf.read_next().expect("has event");
        assert_eq!(e2.sequence_id, 2);
        let e3 = buf.read_next().expect("has event");
        assert_eq!(e3.sequence_id, 3);
        assert!(buf.read_next().is_none());
    }

    #[test]
    fn test_read_next_advances_cursor() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 0, 0, b"a")).expect("ok");
        assert_eq!(buf.cursor, 0);
        buf.read_next();
        assert_eq!(buf.cursor, 1);
    }

    #[test]
    fn test_read_next_updates_stats() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 0, 0, b"abc")).expect("ok"); // 3 bytes
        buf.read_next();
        let stats = buf.stats();
        assert_eq!(stats.replayed_events, 1);
        assert_eq!(stats.bytes_replayed, 3);
    }

    #[test]
    fn test_read_next_empty_buffer_returns_none() {
        let mut buf = ReplayBuffer::new(10);
        assert!(buf.read_next().is_none());
    }

    // ── read_batch ─────────────────────────────────────────────────────────

    #[test]
    fn test_read_batch_reads_requested_count() {
        let mut buf = ReplayBuffer::new(20);
        for i in 1u64..=10 {
            buf.append(make_event(i, i, 0, b"y")).expect("ok");
        }
        let batch = buf.read_batch(5);
        assert_eq!(batch.len(), 5);
        assert_eq!(batch[0].sequence_id, 1);
        assert_eq!(batch[4].sequence_id, 5);
    }

    #[test]
    fn test_read_batch_clamps_to_available() {
        let mut buf = ReplayBuffer::new(10);
        for i in 1u64..=3 {
            buf.append(make_event(i, i, 0, b"z")).expect("ok");
        }
        let batch = buf.read_batch(100);
        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_read_batch_advances_cursor() {
        let mut buf = ReplayBuffer::new(10);
        for i in 0u64..5 {
            buf.append(make_event(i, i, 0, b"q")).expect("ok");
        }
        buf.read_batch(3);
        assert_eq!(buf.cursor, 3);
    }

    #[test]
    fn test_read_batch_updates_stats() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 0, 0, b"ab")).expect("ok"); // 2 bytes
        buf.append(make_event(2, 1, 0, b"cde")).expect("ok"); // 3 bytes
        buf.read_batch(2);
        let s = buf.stats();
        assert_eq!(s.replayed_events, 2);
        assert_eq!(s.bytes_replayed, 5);
    }

    #[test]
    fn test_read_batch_empty_buffer() {
        let mut buf = ReplayBuffer::new(10);
        let batch = buf.read_batch(5);
        assert!(batch.is_empty());
    }

    // ── seek ───────────────────────────────────────────────────────────────

    #[test]
    fn test_seek_beginning() {
        let mut buf = ReplayBuffer::new(10);
        for i in 1u64..=5 {
            buf.append(make_event(i, i * 10, 0, b"x")).expect("ok");
        }
        buf.read_batch(3);
        assert_eq!(buf.cursor, 3);
        let pos = buf.seek(SeekPosition::Beginning).expect("seek ok");
        assert_eq!(pos, 0);
        assert_eq!(buf.cursor, 0);
    }

    #[test]
    fn test_seek_end() {
        let mut buf = ReplayBuffer::new(10);
        for i in 1u64..=5 {
            buf.append(make_event(i, i * 10, 0, b"x")).expect("ok");
        }
        let pos = buf.seek(SeekPosition::End).expect("seek ok");
        assert_eq!(pos, 5);
        assert!(buf.read_next().is_none());
    }

    #[test]
    fn test_seek_sequence_id_found() {
        let mut buf = ReplayBuffer::new(10);
        for i in 10u64..=14 {
            buf.append(make_event(i, i, 0, b"x")).expect("ok");
        }
        let pos = buf.seek(SeekPosition::SequenceId(12)).expect("seek ok");
        // seq 12 is at index 2 (10→0, 11→1, 12→2).
        assert_eq!(pos, 2);
        let e = buf.read_next().expect("event");
        assert_eq!(e.sequence_id, 12);
    }

    #[test]
    fn test_seek_sequence_id_not_found() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 0, 0, b"x")).expect("ok");
        let err = buf.seek(SeekPosition::SequenceId(99)).unwrap_err();
        assert_eq!(err, ReplayError::SequenceNotFound(99));
    }

    #[test]
    fn test_seek_timestamp_found() {
        let mut buf = ReplayBuffer::new(10);
        for i in 0u64..5 {
            buf.append(make_event(i, i * 100, 0, b"x")).expect("ok");
        }
        // Timestamps: 0, 100, 200, 300, 400
        let pos = buf.seek(SeekPosition::Timestamp(200)).expect("seek ok");
        assert_eq!(pos, 2); // event with ts=200 is at index 2.
        let e = buf.read_next().expect("event");
        assert_eq!(e.timestamp, 200);
    }

    #[test]
    fn test_seek_timestamp_not_found() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 50, 0, b"x")).expect("ok");
        let err = buf.seek(SeekPosition::Timestamp(9999)).unwrap_err();
        assert_eq!(err, ReplayError::TimestampNotFound(9999));
    }

    #[test]
    fn test_seek_offset_valid() {
        let mut buf = ReplayBuffer::new(10);
        for i in 0u64..5 {
            buf.append(make_event(i, i, 0, b"x")).expect("ok");
        }
        let pos = buf.seek(SeekPosition::Offset(3)).expect("seek ok");
        assert_eq!(pos, 3);
        let e = buf.read_next().expect("event");
        assert_eq!(e.sequence_id, 3);
    }

    #[test]
    fn test_seek_offset_exactly_at_end() {
        let mut buf = ReplayBuffer::new(10);
        for i in 0u64..3 {
            buf.append(make_event(i, i, 0, b"x")).expect("ok");
        }
        let pos = buf.seek(SeekPosition::Offset(3)).expect("seek ok");
        assert_eq!(pos, 3);
    }

    #[test]
    fn test_seek_offset_beyond_end_errors() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 0, 0, b"x")).expect("ok");
        let err = buf.seek(SeekPosition::Offset(100)).unwrap_err();
        matches!(err, ReplayError::InvalidPosition(_));
    }

    // ── reset ──────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_moves_cursor_to_zero() {
        let mut buf = ReplayBuffer::new(10);
        for i in 1u64..=5 {
            buf.append(make_event(i, i, 0, b"x")).expect("ok");
        }
        buf.read_batch(4);
        assert_eq!(buf.cursor, 4);
        buf.reset();
        assert_eq!(buf.cursor, 0);
    }

    #[test]
    fn test_reset_allows_rereading() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 0, 0, b"hello")).expect("ok");
        let e1 = buf.read_next().expect("first read");
        assert_eq!(e1.sequence_id, 1);
        buf.reset();
        let e2 = buf.read_next().expect("second read after reset");
        assert_eq!(e2.sequence_id, 1);
    }

    // ── events_remaining ───────────────────────────────────────────────────

    #[test]
    fn test_events_remaining_full_buffer() {
        let mut buf = ReplayBuffer::new(10);
        for i in 0u64..5 {
            buf.append(make_event(i, i, 0, b"x")).expect("ok");
        }
        assert_eq!(buf.events_remaining(), 5);
    }

    #[test]
    fn test_events_remaining_after_reads() {
        let mut buf = ReplayBuffer::new(10);
        for i in 0u64..5 {
            buf.append(make_event(i, i, 0, b"x")).expect("ok");
        }
        buf.read_batch(3);
        assert_eq!(buf.events_remaining(), 2);
    }

    #[test]
    fn test_events_remaining_at_end() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 0, 0, b"x")).expect("ok");
        buf.seek(SeekPosition::End).expect("seek ok");
        assert_eq!(buf.events_remaining(), 0);
    }

    // ── stats ──────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_initial_state() {
        let buf = ReplayBuffer::new(10);
        let s = buf.stats();
        assert_eq!(s.total_events, 0);
        assert_eq!(s.replayed_events, 0);
        assert_eq!(s.current_position, 0);
        assert_eq!(s.bytes_replayed, 0);
    }

    #[test]
    fn test_stats_after_appends_and_reads() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 0, 0, b"ab")).expect("ok"); // 2 bytes
        buf.append(make_event(2, 1, 0, b"cde")).expect("ok"); // 3 bytes
        buf.read_next(); // reads seq 1
        let s = buf.stats();
        assert_eq!(s.total_events, 2);
        assert_eq!(s.replayed_events, 1);
        assert_eq!(s.current_position, 1);
        assert_eq!(s.bytes_replayed, 2);
    }

    #[test]
    fn test_stats_total_events_decreases_on_eviction() {
        let mut buf = ReplayBuffer::new(2);
        buf.append(make_event(1, 0, 0, b"x")).expect("ok");
        buf.append(make_event(2, 1, 0, b"x")).expect("ok");
        // Evict seq 1 by adding seq 3.
        buf.append(make_event(3, 2, 0, b"x")).expect("ok");
        let s = buf.stats();
        assert_eq!(s.total_events, 2); // 2 and 3
    }

    // ── capacity eviction ──────────────────────────────────────────────────

    #[test]
    fn test_capacity_one_always_keeps_latest() {
        let mut buf = ReplayBuffer::new(1);
        for i in 1u64..=5 {
            buf.append(make_event(i, i, 0, b"x")).expect("ok");
        }
        assert_eq!(buf.len(), 1);
        let e = buf.read_next().expect("event");
        assert_eq!(e.sequence_id, 5);
    }

    #[test]
    fn test_capacity_eviction_fifo_order() {
        let mut buf = ReplayBuffer::new(3);
        for i in 1u64..=6 {
            buf.append(make_event(i, i, 0, b"x")).expect("ok");
        }
        // After 6 appends with capacity 3, should have seq 4, 5, 6.
        let ids: Vec<u64> = buf.events.iter().map(|e| e.sequence_id).collect();
        assert_eq!(ids, vec![4, 5, 6]);
    }

    // ── ReplayError Display ────────────────────────────────────────────────

    #[test]
    fn test_replay_error_display_buffer_full() {
        let e = ReplayError::BufferFull;
        assert!(e.to_string().contains("full"));
    }

    #[test]
    fn test_replay_error_display_invalid_position() {
        let e = ReplayError::InvalidPosition("offset 100".to_string());
        assert!(e.to_string().contains("offset 100"));
    }

    #[test]
    fn test_replay_error_display_sequence_not_found() {
        let e = ReplayError::SequenceNotFound(42);
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn test_replay_error_display_timestamp_not_found() {
        let e = ReplayError::TimestampNotFound(9999);
        assert!(e.to_string().contains("9999"));
    }

    // ── seek_timestamp boundary ────────────────────────────────────────────

    #[test]
    fn test_seek_timestamp_exact_match() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 100, 0, b"a")).expect("ok");
        buf.append(make_event(2, 200, 0, b"b")).expect("ok");
        buf.append(make_event(3, 300, 0, b"c")).expect("ok");
        let pos = buf.seek(SeekPosition::Timestamp(100)).expect("ok");
        assert_eq!(pos, 0);
    }

    #[test]
    fn test_seek_timestamp_between_events() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 100, 0, b"a")).expect("ok");
        buf.append(make_event(2, 300, 0, b"b")).expect("ok");
        // Seek ts=200 should land at event with ts=300 (index 1).
        let pos = buf.seek(SeekPosition::Timestamp(200)).expect("ok");
        assert_eq!(pos, 1);
    }

    // ── partition filtering ────────────────────────────────────────────────

    #[test]
    fn test_events_on_different_partitions() {
        let mut buf = ReplayBuffer::new(10);
        buf.append(make_event(1, 0, 0, b"part0")).expect("ok");
        buf.append(make_event(2, 1, 1, b"part1")).expect("ok");
        buf.append(make_event(3, 2, 0, b"part0_again")).expect("ok");
        assert_eq!(buf.len(), 3);
        let e = buf.read_next().expect("event");
        assert_eq!(e.partition, 0);
        let e = buf.read_next().expect("event");
        assert_eq!(e.partition, 1);
    }

    // ── concurrent seek-and-read patterns ─────────────────────────────────

    #[test]
    fn test_seek_read_seek_read_pattern() {
        let mut buf = ReplayBuffer::new(20);
        for i in 1u64..=10 {
            let label = format!("event-{i}");
            buf.append(make_event(i, i * 10, 0, label.as_bytes()))
                .expect("ok");
        }
        // Read first 3.
        let batch1 = buf.read_batch(3);
        assert_eq!(batch1.len(), 3);
        // Seek back to seq 5.
        buf.seek(SeekPosition::SequenceId(5)).expect("ok");
        let e = buf.read_next().expect("event");
        assert_eq!(e.sequence_id, 5);
    }
}
