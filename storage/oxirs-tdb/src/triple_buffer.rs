/// Write buffer for batching triple insertions into the TDB store.
///
/// Accumulates triples in memory and triggers a flush when the buffer
/// reaches capacity. Provides bulk insert utilities and statistics.
use std::time::Instant;

/// A triple (or quad) pending insertion into the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferedTriple {
    /// Subject IRI or blank node identifier.
    pub subject: String,
    /// Predicate IRI.
    pub predicate: String,
    /// Object value (IRI, blank node, or literal).
    pub object: String,
    /// Optional named graph IRI.
    pub graph: Option<String>,
}

impl BufferedTriple {
    /// Create a new buffered triple.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        graph: Option<impl Into<String>>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            graph: graph.map(|g| g.into()),
        }
    }

    /// Return true if this triple is in a named graph.
    pub fn is_quad(&self) -> bool {
        self.graph.is_some()
    }
}

/// Result of a buffer flush operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlushResult {
    /// Number of triples that were flushed.
    pub triples_flushed: usize,
    /// Duration of the flush in microseconds.
    pub duration_us: u64,
}

impl FlushResult {
    /// Return true if no triples were flushed.
    pub fn is_empty(&self) -> bool {
        self.triples_flushed == 0
    }
}

/// Write buffer for batching triple insertions.
///
/// Triples are accumulated in an internal buffer. When the buffer
/// reaches its capacity, an automatic flush is triggered (the caller
/// is responsible for persisting the drained data).
pub struct TripleBuffer {
    buffer: Vec<BufferedTriple>,
    capacity: usize,
    total_flushed: usize,
    flush_count: usize,
}

impl TripleBuffer {
    /// Create a new buffer with the given capacity.
    ///
    /// # Panics
    /// Panics if `capacity` is zero.
    pub fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "TripleBuffer capacity must be greater than zero"
        );
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            total_flushed: 0,
            flush_count: 0,
        }
    }

    /// Insert a single triple into the buffer.
    ///
    /// Returns `true` if the insertion triggered an automatic flush
    /// (buffer was full before insertion). The caller should handle
    /// the flushed data via [`drain`](Self::drain).
    pub fn insert(&mut self, s: &str, p: &str, o: &str, graph: Option<&str>) -> bool {
        let triple = BufferedTriple {
            subject: s.to_string(),
            predicate: p.to_string(),
            object: o.to_string(),
            graph: graph.map(|g| g.to_string()),
        };
        self.buffer.push(triple);

        if self.buffer.len() >= self.capacity {
            let _ = self.flush();
            true
        } else {
            false
        }
    }

    /// Insert a batch of triples, automatically flushing as needed.
    ///
    /// Returns the number of automatic flushes that occurred.
    pub fn insert_batch(&mut self, triples: &[(&str, &str, &str, Option<&str>)]) -> usize {
        let mut flush_count = 0;
        for (s, p, o, g) in triples {
            if self.insert(s, p, o, *g) {
                flush_count += 1;
            }
        }
        flush_count
    }

    /// Flush the buffer, recording statistics.
    ///
    /// After flushing, the buffer is empty. The flushed triples are
    /// discarded (caller should use [`drain`](Self::drain) to retrieve them
    /// before calling flush if needed).
    pub fn flush(&mut self) -> FlushResult {
        let start = Instant::now();
        let count = self.buffer.len();
        self.buffer.clear();
        let duration_us = start.elapsed().as_micros() as u64;

        self.total_flushed += count;
        self.flush_count += 1;

        FlushResult {
            triples_flushed: count,
            duration_us,
        }
    }

    /// Return the number of triples currently in the buffer.
    pub fn pending(&self) -> usize {
        self.buffer.len()
    }

    /// Return true if the buffer is at or above capacity.
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Return the maximum buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Return the total number of triples flushed since creation.
    pub fn total_flushed(&self) -> usize {
        self.total_flushed
    }

    /// Return the number of flush operations performed since creation.
    pub fn flush_count(&self) -> usize {
        self.flush_count
    }

    /// Drain all pending triples from the buffer without recording flush stats.
    ///
    /// This is useful when the caller wants to process triples before
    /// a full flush, e.g., for a final partial flush.
    pub fn drain(&mut self) -> Vec<BufferedTriple> {
        self.buffer.drain(..).collect()
    }

    /// Return a slice of the pending triples for inspection.
    pub fn peek(&self) -> &[BufferedTriple] {
        &self.buffer
    }

    /// Return true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buffer(cap: usize) -> TripleBuffer {
        TripleBuffer::new(cap)
    }

    // --- BufferedTriple ---

    #[test]
    fn test_buffered_triple_new() {
        let t = BufferedTriple::new("ex:s", "ex:p", "ex:o", None::<String>);
        assert_eq!(t.subject, "ex:s");
        assert_eq!(t.predicate, "ex:p");
        assert_eq!(t.object, "ex:o");
        assert!(t.graph.is_none());
    }

    #[test]
    fn test_buffered_triple_with_graph() {
        let t = BufferedTriple::new("ex:s", "ex:p", "ex:o", Some("ex:g"));
        assert_eq!(t.graph, Some("ex:g".to_string()));
        assert!(t.is_quad());
    }

    #[test]
    fn test_buffered_triple_is_not_quad_without_graph() {
        let t = BufferedTriple::new("ex:s", "ex:p", "ex:o", None::<String>);
        assert!(!t.is_quad());
    }

    #[test]
    fn test_buffered_triple_clone() {
        let t = BufferedTriple::new("s", "p", "o", Some("g"));
        let t2 = t.clone();
        assert_eq!(t, t2);
    }

    #[test]
    fn test_buffered_triple_debug() {
        let t = BufferedTriple::new("s", "p", "o", None::<String>);
        let dbg = format!("{t:?}");
        assert!(dbg.contains('s'));
    }

    // --- FlushResult ---

    #[test]
    fn test_flush_result_is_empty_true() {
        let fr = FlushResult {
            triples_flushed: 0,
            duration_us: 5,
        };
        assert!(fr.is_empty());
    }

    #[test]
    fn test_flush_result_is_empty_false() {
        let fr = FlushResult {
            triples_flushed: 3,
            duration_us: 10,
        };
        assert!(!fr.is_empty());
    }

    #[test]
    fn test_flush_result_clone() {
        let fr = FlushResult {
            triples_flushed: 7,
            duration_us: 100,
        };
        let fr2 = fr.clone();
        assert_eq!(fr, fr2);
    }

    // --- TripleBuffer construction ---

    #[test]
    fn test_new_buffer() {
        let buf = make_buffer(10);
        assert_eq!(buf.capacity(), 10);
        assert_eq!(buf.pending(), 0);
        assert_eq!(buf.total_flushed(), 0);
        assert_eq!(buf.flush_count(), 0);
    }

    #[test]
    fn test_new_buffer_capacity_one() {
        let buf = make_buffer(1);
        assert_eq!(buf.capacity(), 1);
    }

    #[test]
    #[should_panic(expected = "capacity must be greater than zero")]
    fn test_new_buffer_zero_capacity_panics() {
        let _ = TripleBuffer::new(0);
    }

    // --- is_empty / pending ---

    #[test]
    fn test_is_empty_initially_true() {
        let buf = make_buffer(5);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_is_empty_after_insert_false() {
        let mut buf = make_buffer(5);
        buf.insert("s", "p", "o", None);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_pending_increases_on_insert() {
        let mut buf = make_buffer(10);
        buf.insert("s1", "p", "o1", None);
        buf.insert("s2", "p", "o2", None);
        assert_eq!(buf.pending(), 2);
    }

    // --- insert ---

    #[test]
    fn test_insert_below_capacity_returns_false() {
        let mut buf = make_buffer(5);
        let flushed = buf.insert("s", "p", "o", None);
        assert!(!flushed);
        // Insert below capacity does not flush, so 1 element is pending.
        assert_eq!(buf.pending(), 1);
    }

    #[test]
    fn test_insert_below_capacity_does_not_flush() {
        let mut buf = make_buffer(5);
        for i in 0..4 {
            let flushed = buf.insert(&format!("s{i}"), "p", "o", None);
            assert!(!flushed);
        }
        assert_eq!(buf.flush_count(), 0);
    }

    #[test]
    fn test_insert_at_capacity_triggers_flush() {
        let mut buf = make_buffer(3);
        buf.insert("s1", "p", "o1", None);
        buf.insert("s2", "p", "o2", None);
        let flushed = buf.insert("s3", "p", "o3", None);
        assert!(flushed);
        assert_eq!(buf.flush_count(), 1);
        assert_eq!(buf.total_flushed(), 3);
    }

    #[test]
    fn test_insert_with_graph() {
        let mut buf = make_buffer(10);
        buf.insert("s", "p", "o", Some("g"));
        let t = &buf.peek()[0];
        assert_eq!(t.graph.as_deref(), Some("g"));
    }

    #[test]
    fn test_insert_without_graph() {
        let mut buf = make_buffer(10);
        buf.insert("s", "p", "o", None);
        let t = &buf.peek()[0];
        assert!(t.graph.is_none());
    }

    // --- insert_batch ---

    #[test]
    fn test_insert_batch_no_flush() {
        let mut buf = make_buffer(10);
        let triples = vec![("s1", "p", "o1", None), ("s2", "p", "o2", None)];
        let flushes = buf.insert_batch(&triples);
        assert_eq!(flushes, 0);
        assert_eq!(buf.pending(), 2);
    }

    #[test]
    fn test_insert_batch_triggers_flush() {
        let mut buf = make_buffer(3);
        let triples = vec![
            ("s1", "p", "o1", None),
            ("s2", "p", "o2", None),
            ("s3", "p", "o3", None),
        ];
        let flushes = buf.insert_batch(&triples);
        assert_eq!(flushes, 1);
    }

    #[test]
    fn test_insert_batch_multiple_flushes() {
        let mut buf = make_buffer(2);
        let triples = vec![
            ("a", "p", "1", None),
            ("b", "p", "2", None),
            ("c", "p", "3", None),
            ("d", "p", "4", None),
        ];
        let flushes = buf.insert_batch(&triples);
        assert_eq!(flushes, 2);
        assert_eq!(buf.total_flushed(), 4);
    }

    #[test]
    fn test_insert_batch_empty() {
        let mut buf = make_buffer(5);
        let flushes = buf.insert_batch(&[]);
        assert_eq!(flushes, 0);
        assert!(buf.is_empty());
    }

    // --- flush ---

    #[test]
    fn test_flush_returns_count() {
        let mut buf = make_buffer(10);
        buf.insert("s1", "p", "o1", None);
        buf.insert("s2", "p", "o2", None);
        let result = buf.flush();
        assert_eq!(result.triples_flushed, 2);
    }

    #[test]
    fn test_flush_clears_buffer() {
        let mut buf = make_buffer(10);
        buf.insert("s", "p", "o", None);
        buf.flush();
        assert!(buf.is_empty());
        assert_eq!(buf.pending(), 0);
    }

    #[test]
    fn test_flush_empty_buffer() {
        let mut buf = make_buffer(10);
        let result = buf.flush();
        assert_eq!(result.triples_flushed, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_flush_increments_flush_count() {
        let mut buf = make_buffer(10);
        buf.flush();
        buf.flush();
        assert_eq!(buf.flush_count(), 2);
    }

    #[test]
    fn test_flush_accumulates_total_flushed() {
        let mut buf = make_buffer(10);
        buf.insert("s1", "p", "o1", None);
        buf.insert("s2", "p", "o2", None);
        buf.flush();
        buf.insert("s3", "p", "o3", None);
        buf.flush();
        assert_eq!(buf.total_flushed(), 3);
    }

    #[test]
    fn test_flush_duration_us_is_non_negative() {
        let mut buf = make_buffer(5);
        buf.insert("s", "p", "o", None);
        let result = buf.flush();
        // Duration should be a valid u64 (may be 0 on fast systems).
        let _ = result.duration_us;
    }

    // --- drain ---

    #[test]
    fn test_drain_returns_all_pending() {
        let mut buf = make_buffer(10);
        buf.insert("s1", "p", "o1", None);
        buf.insert("s2", "p", "o2", None);
        let drained = buf.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].subject, "s1");
    }

    #[test]
    fn test_drain_leaves_buffer_empty() {
        let mut buf = make_buffer(10);
        buf.insert("s", "p", "o", None);
        let _ = buf.drain();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_drain_does_not_update_flush_stats() {
        let mut buf = make_buffer(10);
        buf.insert("s", "p", "o", None);
        let _ = buf.drain();
        // drain does NOT increment flush_count or total_flushed.
        assert_eq!(buf.flush_count(), 0);
        assert_eq!(buf.total_flushed(), 0);
    }

    #[test]
    fn test_drain_empty_buffer() {
        let mut buf = make_buffer(10);
        let drained = buf.drain();
        assert!(drained.is_empty());
    }

    // --- is_full ---

    #[test]
    fn test_is_full_false_when_below_capacity() {
        let mut buf = make_buffer(5);
        buf.insert("s", "p", "o", None);
        assert!(!buf.is_full());
    }

    #[test]
    fn test_is_full_after_flush_is_false() {
        let mut buf = make_buffer(2);
        buf.insert("s1", "p", "o1", None);
        buf.insert("s2", "p", "o2", None);
        // Buffer was auto-flushed, now empty.
        assert!(!buf.is_full());
    }

    // --- peek ---

    #[test]
    fn test_peek_returns_all_triples() {
        let mut buf = make_buffer(5);
        buf.insert("a", "p", "1", None);
        buf.insert("b", "p", "2", None);
        let peek = buf.peek();
        assert_eq!(peek.len(), 2);
    }

    #[test]
    fn test_peek_does_not_remove() {
        let mut buf = make_buffer(5);
        buf.insert("a", "p", "1", None);
        let _ = buf.peek();
        assert_eq!(buf.pending(), 1);
    }

    // --- Integration ---

    #[test]
    fn test_high_volume_insert() {
        let mut buf = make_buffer(100);
        for i in 0..500 {
            buf.insert(&format!("s{i}"), "p", &format!("o{i}"), None);
        }
        // 5 auto flushes of 100 each = 500 total flushed, 0 pending.
        assert_eq!(buf.total_flushed(), 500);
        assert_eq!(buf.flush_count(), 5);
        assert_eq!(buf.pending(), 0);
    }

    #[test]
    fn test_drain_then_insert_workflow() {
        let mut buf = make_buffer(10);
        for i in 0..5 {
            buf.insert(&format!("s{i}"), "p", "o", None);
        }
        let drained = buf.drain();
        assert_eq!(drained.len(), 5);
        buf.insert("new_s", "p", "new_o", None);
        assert_eq!(buf.pending(), 1);
    }
}
