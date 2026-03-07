//! Versioned triple storage with append-only semantics and timestamps.
//!
//! This module provides temporal storage for RDF triples, tracking when
//! each triple was valid (valid_from / valid_to) and which transaction
//! made the change.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{Result, TdbError};

/// A single RDF triple with temporal validity information.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionedTriple {
    /// Subject of the triple
    pub subject: String,
    /// Predicate of the triple
    pub predicate: String,
    /// Object of the triple
    pub object: String,
    /// Timestamp when this triple became valid
    pub valid_from: DateTime<Utc>,
    /// Timestamp when this triple was retracted (None = still valid)
    pub valid_to: Option<DateTime<Utc>>,
    /// Transaction that created this version
    pub transaction_id: u64,
}

impl VersionedTriple {
    /// Create a new currently-valid versioned triple.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        valid_from: DateTime<Utc>,
        transaction_id: u64,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
            valid_from,
            valid_to: None,
            transaction_id,
        }
    }

    /// Returns true if the triple is active at the given timestamp.
    pub fn is_active_at(&self, ts: DateTime<Utc>) -> bool {
        self.valid_from <= ts && self.valid_to.map_or(true, |end| ts < end)
    }

    /// Build a canonical lookup key for this triple (s, p, o).
    pub fn key(&self) -> TripleKey {
        TripleKey {
            subject: self.subject.clone(),
            predicate: self.predicate.clone(),
            object: self.object.clone(),
        }
    }
}

/// A newtype key used to look up triples regardless of temporal state.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TripleKey {
    /// Subject component
    pub subject: String,
    /// Predicate component
    pub predicate: String,
    /// Object component
    pub object: String,
}

impl TripleKey {
    /// Create a new triple key.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

/// Represents the set of additions and removals between two points in time.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalDiff {
    /// Triples that were added in the interval [from, to)
    pub added: Vec<VersionedTriple>,
    /// Triples that were removed in the interval [from, to)
    pub removed: Vec<VersionedTriple>,
}

impl TemporalDiff {
    /// Returns true when there are no changes.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty()
    }
}

/// Append-only versioned triple store.
///
/// Every insert creates a new `VersionedTriple` with `valid_from` set to
/// the provided timestamp.  Every delete marks the matching live version's
/// `valid_to` field.  Historical records are never discarded, enabling
/// full time-travel queries.
#[derive(Debug, Default)]
pub struct TemporalVersionStore {
    /// All versions ever stored, grouped by (s, p, o) key.
    versions: HashMap<TripleKey, Vec<VersionedTriple>>,
    /// Monotonically increasing transaction counter.
    next_txn_id: u64,
}

impl TemporalVersionStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
            next_txn_id: 1,
        }
    }

    /// Allocate and return the next transaction ID.
    fn alloc_txn_id(&mut self) -> u64 {
        let id = self.next_txn_id;
        self.next_txn_id += 1;
        id
    }

    /// Insert a new triple into the store.
    ///
    /// If an active version of the triple already exists at `valid_from`,
    /// this is a no-op (idempotent insert).
    pub fn insert(&mut self, triple: VersionedTriple) -> Result<()> {
        let key = triple.key();

        // Check for duplicate live triple
        if let Some(versions) = self.versions.get(&key) {
            for v in versions {
                if v.valid_to.is_none() {
                    // Already live – treat as idempotent
                    return Ok(());
                }
            }
        }

        self.versions.entry(key).or_default().push(triple);
        Ok(())
    }

    /// Insert a triple using auto-generated timestamp and transaction ID.
    pub fn insert_now(
        &mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Result<u64> {
        let txn_id = self.alloc_txn_id();
        let triple = VersionedTriple::new(subject, predicate, object, Utc::now(), txn_id);
        self.insert(triple)?;
        Ok(txn_id)
    }

    /// Insert a triple with an explicit timestamp (useful for tests / ingestion).
    pub fn insert_at(
        &mut self,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
        ts: DateTime<Utc>,
    ) -> Result<u64> {
        let txn_id = self.alloc_txn_id();
        let triple = VersionedTriple::new(subject, predicate, object, ts, txn_id);
        self.insert(triple)?;
        Ok(txn_id)
    }

    /// Mark the live version of a triple as deleted at the given timestamp.
    ///
    /// Returns `TdbError::InvalidInput` if no live version is found.
    pub fn delete_at(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
        ts: DateTime<Utc>,
    ) -> Result<()> {
        let key = TripleKey::new(subject, predicate, object);

        match self.versions.get_mut(&key) {
            None => Err(TdbError::InvalidInput(format!(
                "Triple not found: ({subject}, {predicate}, {object})"
            ))),
            Some(versions) => {
                let live = versions.iter_mut().find(|v| v.valid_to.is_none());
                match live {
                    None => Err(TdbError::InvalidInput(format!(
                        "No live version for triple: ({subject}, {predicate}, {object})"
                    ))),
                    Some(v) => {
                        if ts < v.valid_from {
                            return Err(TdbError::InvalidInput(format!(
                                "Delete timestamp {ts} is before valid_from {}",
                                v.valid_from
                            )));
                        }
                        v.valid_to = Some(ts);
                        Ok(())
                    }
                }
            }
        }
    }

    /// Delete a triple using the current wall-clock time.
    pub fn delete(&mut self, subject: &str, predicate: &str, object: &str) -> Result<()> {
        self.delete_at(subject, predicate, object, Utc::now())
    }

    /// Return all triples that were active at the given point in time.
    pub fn query_at(&self, timestamp: DateTime<Utc>) -> Vec<VersionedTriple> {
        self.versions
            .values()
            .flatten()
            .filter(|v| v.is_active_at(timestamp))
            .cloned()
            .collect()
    }

    /// Return the full history of a specific triple (all versions).
    pub fn history(&self, subject: &str, predicate: &str, object: &str) -> Vec<VersionedTriple> {
        let key = TripleKey::new(subject, predicate, object);
        self.versions.get(&key).cloned().unwrap_or_default()
    }

    /// Return all currently active (live) triples.
    pub fn current(&self) -> Vec<VersionedTriple> {
        self.versions
            .values()
            .flatten()
            .filter(|v| v.valid_to.is_none())
            .cloned()
            .collect()
    }

    /// Compute the diff between two timestamps.
    ///
    /// `added`   = triples whose `valid_from` is in `[from, to)`
    /// `removed` = triples whose `valid_to`   is in `[from, to)`
    pub fn diff(&self, from: DateTime<Utc>, to: DateTime<Utc>) -> TemporalDiff {
        let mut diff = TemporalDiff::default();

        for v in self.versions.values().flatten() {
            if v.valid_from >= from && v.valid_from < to {
                diff.added.push(v.clone());
            }
            if let Some(end) = v.valid_to {
                if end >= from && end < to {
                    diff.removed.push(v.clone());
                }
            }
        }

        diff
    }

    /// Return the total number of stored version records (including history).
    pub fn version_count(&self) -> usize {
        self.versions.values().map(|v| v.len()).sum()
    }

    /// Return the number of distinct triple keys tracked.
    pub fn triple_key_count(&self) -> usize {
        self.versions.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    #[test]
    fn test_insert_and_query_at() {
        let mut store = TemporalVersionStore::new();
        store
            .insert_at("Alice", "knows", "Bob", ts(2024, 1, 1))
            .unwrap();
        store
            .insert_at("Alice", "knows", "Carol", ts(2024, 6, 1))
            .unwrap();

        let snap = store.query_at(ts(2024, 3, 1));
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].object, "Bob");
    }

    #[test]
    fn test_delete_marks_valid_to() {
        let mut store = TemporalVersionStore::new();
        store
            .insert_at("Alice", "knows", "Bob", ts(2024, 1, 1))
            .unwrap();
        store
            .delete_at("Alice", "knows", "Bob", ts(2024, 6, 1))
            .unwrap();

        // Should not appear after deletion
        let snap = store.query_at(ts(2024, 7, 1));
        assert!(snap.is_empty());

        // Should appear before deletion
        let snap_before = store.query_at(ts(2024, 3, 1));
        assert_eq!(snap_before.len(), 1);
    }

    #[test]
    fn test_idempotent_insert() {
        let mut store = TemporalVersionStore::new();
        store.insert_at("s", "p", "o", ts(2024, 1, 1)).unwrap();
        // Second insert is idempotent
        store.insert_at("s", "p", "o", ts(2024, 2, 1)).unwrap();

        assert_eq!(store.version_count(), 1);
    }

    #[test]
    fn test_delete_nonexistent_returns_error() {
        let mut store = TemporalVersionStore::new();
        let result = store.delete_at("s", "p", "o", ts(2024, 1, 1));
        assert!(result.is_err());
    }

    #[test]
    fn test_diff() {
        let mut store = TemporalVersionStore::new();
        store.insert_at("s", "p", "o1", ts(2024, 1, 1)).unwrap();
        store.insert_at("s", "p", "o2", ts(2024, 4, 1)).unwrap();
        store.delete_at("s", "p", "o1", ts(2024, 5, 1)).unwrap();

        let diff = store.diff(ts(2024, 3, 1), ts(2024, 6, 1));
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0].object, "o2");
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0].object, "o1");
    }
}
