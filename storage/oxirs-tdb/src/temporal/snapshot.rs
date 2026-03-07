//! Point-in-time snapshots of an RDF dataset.
//!
//! A `TemporalSnapshot` captures the set of triples that were active at a
//! specific moment, enabling read-only access without modifying the live store.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::version_store::VersionedTriple;

/// An immutable view of the RDF dataset at a specific point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSnapshot {
    /// The timestamp this snapshot was taken at.
    pub as_of: DateTime<Utc>,
    /// The triples that were active at `as_of`.
    triples: Vec<VersionedTriple>,
}

impl TemporalSnapshot {
    /// Create a snapshot from a pre-filtered list of active triples.
    pub fn new(as_of: DateTime<Utc>, triples: Vec<VersionedTriple>) -> Self {
        Self { as_of, triples }
    }

    /// Return all triples in this snapshot.
    pub fn triples(&self) -> &[VersionedTriple] {
        &self.triples
    }

    /// Return the number of triples in this snapshot.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Returns true when the snapshot contains no triples.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }

    /// Find triples matching an optional subject filter.
    pub fn find_by_subject(&self, subject: &str) -> Vec<&VersionedTriple> {
        self.triples
            .iter()
            .filter(|t| t.subject == subject)
            .collect()
    }

    /// Find triples matching an optional predicate filter.
    pub fn find_by_predicate(&self, predicate: &str) -> Vec<&VersionedTriple> {
        self.triples
            .iter()
            .filter(|t| t.predicate == predicate)
            .collect()
    }

    /// Find triples matching an optional object filter.
    pub fn find_by_object(&self, object: &str) -> Vec<&VersionedTriple> {
        self.triples.iter().filter(|t| t.object == object).collect()
    }

    /// Find triples matching a full (subject, predicate, object) pattern.
    /// Each component is optional (`None` acts as a wildcard).
    pub fn find(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<&VersionedTriple> {
        self.triples
            .iter()
            .filter(|t| {
                subject.map_or(true, |s| t.subject == s)
                    && predicate.map_or(true, |p| t.predicate == p)
                    && object.map_or(true, |o| t.object == o)
            })
            .collect()
    }

    /// Diff this snapshot against another (newer) snapshot.
    ///
    /// Returns `(added, removed)` where:
    /// - `added`   = triples present in `other` but not in `self`
    /// - `removed` = triples present in `self` but not in `other`
    pub fn diff_against<'a>(
        &'a self,
        other: &'a TemporalSnapshot,
    ) -> (Vec<&'a VersionedTriple>, Vec<&'a VersionedTriple>) {
        let self_keys: std::collections::HashSet<_> = self
            .triples
            .iter()
            .map(|t| (&t.subject, &t.predicate, &t.object))
            .collect();
        let other_keys: std::collections::HashSet<_> = other
            .triples
            .iter()
            .map(|t| (&t.subject, &t.predicate, &t.object))
            .collect();

        let added = other
            .triples
            .iter()
            .filter(|t| !self_keys.contains(&(&t.subject, &t.predicate, &t.object)))
            .collect();

        let removed = self
            .triples
            .iter()
            .filter(|t| !other_keys.contains(&(&t.subject, &t.predicate, &t.object)))
            .collect();

        (added, removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ts(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    fn make_triple(s: &str, p: &str, o: &str, from: DateTime<Utc>) -> VersionedTriple {
        VersionedTriple::new(s, p, o, from, 1)
    }

    #[test]
    fn test_snapshot_find_by_subject() {
        let triples = vec![
            make_triple("Alice", "knows", "Bob", ts(2024, 1, 1)),
            make_triple("Alice", "likes", "Rust", ts(2024, 1, 1)),
            make_triple("Bob", "knows", "Carol", ts(2024, 1, 1)),
        ];
        let snap = TemporalSnapshot::new(ts(2024, 6, 1), triples);
        let found = snap.find_by_subject("Alice");
        assert_eq!(found.len(), 2);
    }

    #[test]
    fn test_snapshot_find_wildcard() {
        let triples = vec![
            make_triple("Alice", "knows", "Bob", ts(2024, 1, 1)),
            make_triple("Alice", "likes", "Rust", ts(2024, 1, 1)),
        ];
        let snap = TemporalSnapshot::new(ts(2024, 6, 1), triples);
        let found = snap.find(Some("Alice"), None, None);
        assert_eq!(found.len(), 2);
        let found2 = snap.find(None, Some("knows"), None);
        assert_eq!(found2.len(), 1);
    }

    #[test]
    fn test_diff_against() {
        let t1 = ts(2024, 1, 1);
        let t2 = ts(2024, 6, 1);

        let snap1 = TemporalSnapshot::new(
            t1,
            vec![
                make_triple("Alice", "knows", "Bob", t1),
                make_triple("Alice", "likes", "Rust", t1),
            ],
        );
        let snap2 = TemporalSnapshot::new(
            t2,
            vec![
                make_triple("Alice", "knows", "Bob", t1), // kept
                make_triple("Alice", "knows", "Carol", t2), // added
                                                          // "Alice likes Rust" was removed
            ],
        );

        let (added, removed) = snap1.diff_against(&snap2);
        assert_eq!(added.len(), 1);
        assert_eq!(added[0].object, "Carol");
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].object, "Rust");
    }

    #[test]
    fn test_empty_snapshot() {
        let snap = TemporalSnapshot::new(ts(2024, 1, 1), vec![]);
        assert!(snap.is_empty());
        assert_eq!(snap.len(), 0);
    }
}
