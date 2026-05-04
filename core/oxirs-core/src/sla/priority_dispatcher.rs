//! Priority-based query dispatcher for SLA-aware scheduling.
//!
//! [`PriorityDispatcher`] is a max-heap where each entry carries the
//! [`SlaClass::dispatch_priority`] of the originating tenant.  Dequeuing
//! always returns the highest-priority pending query first:
//!
//! ```text
//! Platinum (4) > Gold (3) > Silver (2) > Bronze (1)
//! ```

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::class::SlaClass;

// ─────────────────────────────────────────────────────────────────────────────
// PrioritizedQuery
// ─────────────────────────────────────────────────────────────────────────────

/// A query payload annotated with its dispatch priority and originating tenant.
pub struct PrioritizedQuery<T> {
    /// Numeric priority — higher value means earlier dequeue.
    pub priority: u8,
    /// Identifier of the tenant that submitted the query.
    pub tenant_id: String,
    /// Monotonic insertion sequence used to break priority ties (lower comes first).
    pub sequence: u64,
    /// The query payload (type-erased by the caller).
    pub payload: T,
}

// Manual trait impls so we can use PrioritizedQuery<T> in a BinaryHeap
// without requiring T: Ord.

impl<T> PartialEq for PrioritizedQuery<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.sequence == other.sequence
    }
}

impl<T> Eq for PrioritizedQuery<T> {}

impl<T> PartialOrd for PrioritizedQuery<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for PrioritizedQuery<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier sequence first.
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PriorityDispatcher
// ─────────────────────────────────────────────────────────────────────────────

/// Priority dispatcher backed by a max-heap keyed on [`SlaClass::dispatch_priority`].
///
/// Enqueue from any tier; dequeue always returns the highest-priority item.
/// Within the same priority, items are returned in FIFO order based on the
/// monotonic insertion sequence.
///
/// ```rust
/// use oxirs_core::sla::{SlaClass, PriorityDispatcher};
///
/// let mut dispatcher: PriorityDispatcher<&str> = PriorityDispatcher::new();
/// dispatcher.enqueue("bronze_tenant".into(), SlaClass::Bronze, "low-pri query");
/// dispatcher.enqueue("platinum_tenant".into(), SlaClass::Platinum, "high-pri query");
///
/// // Platinum is dequeued first
/// let first = dispatcher.dequeue().expect("dispatcher has at least one entry");
/// assert_eq!(first.payload, "high-pri query");
/// ```
pub struct PriorityDispatcher<T> {
    heap: BinaryHeap<PrioritizedQuery<T>>,
    next_sequence: u64,
}

impl<T> Default for PriorityDispatcher<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> PriorityDispatcher<T> {
    /// Create an empty dispatcher.
    pub fn new() -> Self {
        PriorityDispatcher {
            heap: BinaryHeap::new(),
            next_sequence: 0,
        }
    }

    /// Enqueue a query for `tenant_id` at the priority of `sla`.
    pub fn enqueue(&mut self, tenant_id: String, sla: SlaClass, payload: T) {
        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        self.heap.push(PrioritizedQuery {
            priority: sla.dispatch_priority(),
            tenant_id,
            sequence,
            payload,
        });
    }

    /// Dequeue the highest-priority query, or `None` if the queue is empty.
    pub fn dequeue(&mut self) -> Option<PrioritizedQuery<T>> {
        self.heap.pop()
    }

    /// Peek at the highest-priority query without removing it.
    pub fn peek(&self) -> Option<&PrioritizedQuery<T>> {
        self.heap.peek()
    }

    /// Return the number of queued items.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Return `true` when the queue has no items.
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Drain all items in priority order (highest first).
    pub fn drain_ordered(&mut self) -> Vec<PrioritizedQuery<T>> {
        let mut result = Vec::with_capacity(self.heap.len());
        while let Some(item) = self.heap.pop() {
            result.push(item);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platinum_dequeued_first() {
        let mut d: PriorityDispatcher<&str> = PriorityDispatcher::new();
        d.enqueue("t_bronze".into(), SlaClass::Bronze, "b");
        d.enqueue("t_gold".into(), SlaClass::Gold, "g");
        d.enqueue("t_platinum".into(), SlaClass::Platinum, "p");
        d.enqueue("t_silver".into(), SlaClass::Silver, "s");

        let first = d.dequeue().expect("non-empty");
        assert_eq!(first.payload, "p", "Platinum must be first");
        let second = d.dequeue().expect("non-empty");
        assert_eq!(second.payload, "g", "Gold must be second");
        let third = d.dequeue().expect("non-empty");
        assert_eq!(third.payload, "s", "Silver must be third");
        let fourth = d.dequeue().expect("non-empty");
        assert_eq!(fourth.payload, "b", "Bronze must be last");
        assert!(d.is_empty());
    }

    #[test]
    fn test_dequeue_empty_returns_none() {
        let mut d: PriorityDispatcher<u32> = PriorityDispatcher::new();
        assert!(d.dequeue().is_none());
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut d: PriorityDispatcher<i32> = PriorityDispatcher::new();
        assert!(d.is_empty());
        d.enqueue("t".into(), SlaClass::Silver, 42);
        assert_eq!(d.len(), 1);
        assert!(!d.is_empty());
        d.dequeue();
        assert!(d.is_empty());
    }

    #[test]
    fn test_multiple_same_class_fifo_within_priority() {
        let mut d: PriorityDispatcher<u32> = PriorityDispatcher::new();
        for i in 0..5u32 {
            d.enqueue("gold".into(), SlaClass::Gold, i);
        }
        assert_eq!(d.len(), 5);
        // FIFO within same priority — should drain 0, 1, 2, 3, 4 in order.
        let drained = d.drain_ordered();
        let payloads: Vec<u32> = drained.iter().map(|q| q.payload).collect();
        assert_eq!(payloads, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_peek_does_not_remove() {
        let mut d: PriorityDispatcher<&str> = PriorityDispatcher::new();
        d.enqueue("t".into(), SlaClass::Platinum, "hello");
        assert!(d.peek().is_some());
        assert_eq!(d.len(), 1); // peek did not remove
        d.dequeue();
        assert!(d.peek().is_none());
    }

    #[test]
    fn test_drain_ordered_highest_first() {
        let mut d: PriorityDispatcher<u8> = PriorityDispatcher::new();
        d.enqueue("a".into(), SlaClass::Silver, 2);
        d.enqueue("b".into(), SlaClass::Platinum, 4);
        d.enqueue("c".into(), SlaClass::Bronze, 1);
        d.enqueue("d".into(), SlaClass::Gold, 3);

        let drained = d.drain_ordered();
        let payloads: Vec<u8> = drained.iter().map(|q| q.payload).collect();
        assert_eq!(payloads, vec![4, 3, 2, 1]);
    }
}
