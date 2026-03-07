//! Change tracking and audit log for the temporal RDF store.
//!
//! The `ChangeLog` maintains an ordered list of `ChangeEntry` records,
//! each describing a single insert or delete operation on the dataset.
//! Entries can be replayed, filtered by time range, or summarised into
//! aggregated statistics.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::version_store::VersionedTriple;

/// The kind of operation recorded in a change entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChangeOperation {
    /// A triple was inserted (made valid).
    Insert,
    /// A triple was deleted (marked invalid).
    Delete,
}

impl std::fmt::Display for ChangeOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChangeOperation::Insert => write!(f, "INSERT"),
            ChangeOperation::Delete => write!(f, "DELETE"),
        }
    }
}

/// A single record in the change log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeEntry {
    /// Monotonically increasing entry sequence number (1-based).
    pub sequence: u64,
    /// The type of change.
    pub operation: ChangeOperation,
    /// The triple that was affected.
    pub triple: VersionedTriple,
    /// The transaction that performed this change.
    pub transaction_id: u64,
    /// Wall-clock time at which the change was recorded.
    pub timestamp: DateTime<Utc>,
}

impl ChangeEntry {
    /// Describe the change as a human-readable string.
    pub fn describe(&self) -> String {
        format!(
            "[{}] seq={} txn={} ({}, {}, {})",
            self.operation,
            self.sequence,
            self.transaction_id,
            self.triple.subject,
            self.triple.predicate,
            self.triple.object,
        )
    }
}

/// Summary statistics over a range of change log entries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChangeLogStats {
    /// Total number of entries examined.
    pub total_entries: usize,
    /// Number of insert operations.
    pub inserts: usize,
    /// Number of delete operations.
    pub deletes: usize,
    /// Number of distinct transactions referenced.
    pub distinct_transactions: usize,
    /// Earliest entry timestamp in the range (None if empty).
    pub earliest: Option<DateTime<Utc>>,
    /// Latest entry timestamp in the range (None if empty).
    pub latest: Option<DateTime<Utc>>,
}

/// An append-only ordered log of all changes to the temporal store.
///
/// Designed for auditing, replication, and debugging.  The log grows
/// monotonically; entries are never removed or updated.
#[derive(Debug, Default)]
pub struct ChangeLog {
    entries: Vec<ChangeEntry>,
    next_sequence: u64,
}

impl ChangeLog {
    /// Create an empty change log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_sequence: 1,
        }
    }

    /// Append an insert entry for the given triple.
    pub fn record_insert(&mut self, triple: VersionedTriple, transaction_id: u64) {
        let seq = self.next_sequence;
        self.next_sequence += 1;
        self.entries.push(ChangeEntry {
            sequence: seq,
            operation: ChangeOperation::Insert,
            triple,
            transaction_id,
            timestamp: Utc::now(),
        });
    }

    /// Append a delete entry for the given triple.
    pub fn record_delete(&mut self, triple: VersionedTriple, transaction_id: u64) {
        let seq = self.next_sequence;
        self.next_sequence += 1;
        self.entries.push(ChangeEntry {
            sequence: seq,
            operation: ChangeOperation::Delete,
            triple,
            transaction_id,
            timestamp: Utc::now(),
        });
    }

    /// Return all entries in the log (in insertion order).
    pub fn all_entries(&self) -> &[ChangeEntry] {
        &self.entries
    }

    /// Return entries whose wall-clock timestamp falls in `[from, to)`.
    pub fn entries_in_range(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> Vec<&ChangeEntry> {
        self.entries
            .iter()
            .filter(|e| e.timestamp >= from && e.timestamp < to)
            .collect()
    }

    /// Return entries for a specific transaction ID.
    pub fn entries_for_transaction(&self, transaction_id: u64) -> Vec<&ChangeEntry> {
        self.entries
            .iter()
            .filter(|e| e.transaction_id == transaction_id)
            .collect()
    }

    /// Return only insert entries.
    pub fn inserts(&self) -> Vec<&ChangeEntry> {
        self.entries
            .iter()
            .filter(|e| e.operation == ChangeOperation::Insert)
            .collect()
    }

    /// Return only delete entries.
    pub fn deletes(&self) -> Vec<&ChangeEntry> {
        self.entries
            .iter()
            .filter(|e| e.operation == ChangeOperation::Delete)
            .collect()
    }

    /// Compute statistics over all entries.
    pub fn stats(&self) -> ChangeLogStats {
        self.stats_for(&self.entries)
    }

    /// Compute statistics for a range of entries (by timestamp).
    pub fn stats_in_range(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> ChangeLogStats {
        let range: Vec<&ChangeEntry> = self.entries_in_range(from, to);
        let cloned: Vec<ChangeEntry> = range.into_iter().cloned().collect();
        self.stats_for(&cloned)
    }

    /// Total number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true when the log contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    // --- helpers ---

    fn stats_for(&self, entries: &[ChangeEntry]) -> ChangeLogStats {
        let mut stats = ChangeLogStats {
            total_entries: entries.len(),
            ..Default::default()
        };
        let mut txn_ids = std::collections::HashSet::new();

        for e in entries {
            match e.operation {
                ChangeOperation::Insert => stats.inserts += 1,
                ChangeOperation::Delete => stats.deletes += 1,
            }
            txn_ids.insert(e.transaction_id);
            stats.earliest = Some(match stats.earliest {
                None => e.timestamp,
                Some(prev) => prev.min(e.timestamp),
            });
            stats.latest = Some(match stats.latest {
                None => e.timestamp,
                Some(prev) => prev.max(e.timestamp),
            });
        }

        stats.distinct_transactions = txn_ids.len();
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    fn make_triple(s: &str, p: &str, o: &str) -> VersionedTriple {
        VersionedTriple::new(s, p, o, Utc::now(), 1)
    }

    #[test]
    fn test_record_and_retrieve() {
        let mut log = ChangeLog::new();
        log.record_insert(make_triple("s", "p", "o"), 1);
        log.record_delete(make_triple("s", "p", "o"), 2);

        assert_eq!(log.len(), 2);
        assert_eq!(log.inserts().len(), 1);
        assert_eq!(log.deletes().len(), 1);
    }

    #[test]
    fn test_sequence_numbers() {
        let mut log = ChangeLog::new();
        log.record_insert(make_triple("a", "b", "c"), 10);
        log.record_insert(make_triple("x", "y", "z"), 11);

        let entries = log.all_entries();
        assert_eq!(entries[0].sequence, 1);
        assert_eq!(entries[1].sequence, 2);
    }

    #[test]
    fn test_entries_for_transaction() {
        let mut log = ChangeLog::new();
        log.record_insert(make_triple("s1", "p", "o"), 42);
        log.record_insert(make_triple("s2", "p", "o"), 43);
        log.record_delete(make_triple("s1", "p", "o"), 42);

        let txn42 = log.entries_for_transaction(42);
        assert_eq!(txn42.len(), 2);
        let txn43 = log.entries_for_transaction(43);
        assert_eq!(txn43.len(), 1);
    }

    #[test]
    fn test_stats() {
        let mut log = ChangeLog::new();
        log.record_insert(make_triple("s", "p", "o1"), 1);
        log.record_insert(make_triple("s", "p", "o2"), 2);
        log.record_delete(make_triple("s", "p", "o1"), 3);

        let stats = log.stats();
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.inserts, 2);
        assert_eq!(stats.deletes, 1);
        assert_eq!(stats.distinct_transactions, 3);
    }

    #[test]
    fn test_describe() {
        let mut log = ChangeLog::new();
        log.record_insert(make_triple("Alice", "knows", "Bob"), 7);
        let desc = log.all_entries()[0].describe();
        assert!(desc.contains("INSERT"));
        assert!(desc.contains("Alice"));
        assert!(desc.contains("Bob"));
    }

    #[test]
    fn test_empty_log() {
        let log = ChangeLog::new();
        assert!(log.is_empty());
        let stats = log.stats();
        assert_eq!(stats.total_entries, 0);
        assert!(stats.earliest.is_none());
    }
}
