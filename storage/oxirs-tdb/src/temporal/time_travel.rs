//! Time-travel query support.
//!
//! Allows querying the RDF dataset as it existed at any past (or present)
//! point in time.  A `TimeTravelQuery` pairs a target timestamp with a
//! triple pattern (each component is optional) and evaluates against a
//! `TemporalVersionStore` to produce a `TemporalSnapshot`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TdbError};

use super::snapshot::TemporalSnapshot;
use super::version_store::{TemporalVersionStore, VersionedTriple};

/// Optional component of a triple pattern (None = wildcard).
pub type PatternComponent = Option<String>;

/// A triple pattern with optional subject, predicate, and object components.
///
/// `None` in any position acts as a wildcard that matches any value.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TriplePattern {
    /// Optional subject constraint
    pub subject: PatternComponent,
    /// Optional predicate constraint
    pub predicate: PatternComponent,
    /// Optional object constraint
    pub object: PatternComponent,
}

impl TriplePattern {
    /// Create a pattern with all wildcards (matches everything).
    pub fn wildcard() -> Self {
        Self::default()
    }

    /// Create a pattern with a fixed subject.
    pub fn with_subject(subject: impl Into<String>) -> Self {
        Self {
            subject: Some(subject.into()),
            ..Default::default()
        }
    }

    /// Create a pattern with a fixed predicate.
    pub fn with_predicate(predicate: impl Into<String>) -> Self {
        Self {
            predicate: Some(predicate.into()),
            ..Default::default()
        }
    }

    /// Create a pattern with a fixed object.
    pub fn with_object(object: impl Into<String>) -> Self {
        Self {
            object: Some(object.into()),
            ..Default::default()
        }
    }

    /// Create a fully-specified pattern (no wildcards).
    pub fn exact(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: Some(subject.into()),
            predicate: Some(predicate.into()),
            object: Some(object.into()),
        }
    }

    /// Returns true if the given triple matches this pattern.
    pub fn matches(&self, triple: &VersionedTriple) -> bool {
        self.subject
            .as_deref()
            .map_or(true, |s| s == triple.subject)
            && self
                .predicate
                .as_deref()
                .map_or(true, |p| p == triple.predicate)
            && self.object.as_deref().map_or(true, |o| o == triple.object)
    }
}

/// A time-travel query that retrieves triples matching a pattern as of a
/// specific historical timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeTravelQuery {
    /// Point in time to evaluate the query against.
    pub as_of: DateTime<Utc>,
    /// Triple pattern to match.
    pub pattern: TriplePattern,
}

impl TimeTravelQuery {
    /// Create a new time-travel query.
    pub fn new(as_of: DateTime<Utc>, pattern: TriplePattern) -> Self {
        Self { as_of, pattern }
    }

    /// Convenience constructor for a full wildcard query at a given time.
    pub fn at(as_of: DateTime<Utc>) -> Self {
        Self {
            as_of,
            pattern: TriplePattern::wildcard(),
        }
    }

    /// Execute this query against the given store and return a snapshot.
    pub fn execute(&self, store: &TemporalVersionStore) -> Result<TemporalSnapshot> {
        if self.as_of > Utc::now() + chrono::Duration::days(365 * 10) {
            return Err(TdbError::InvalidInput(
                "as_of timestamp is implausibly far in the future".into(),
            ));
        }

        let matching: Vec<VersionedTriple> = store
            .query_at(self.as_of)
            .into_iter()
            .filter(|t| self.pattern.matches(t))
            .collect();

        Ok(TemporalSnapshot::new(self.as_of, matching))
    }
}

/// Builder for constructing `TimeTravelQuery` instances fluently.
#[derive(Debug, Default)]
pub struct TimeTravelQueryBuilder {
    as_of: Option<DateTime<Utc>>,
    pattern: TriplePattern,
}

impl TimeTravelQueryBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the target timestamp.
    pub fn as_of(mut self, ts: DateTime<Utc>) -> Self {
        self.as_of = Some(ts);
        self
    }

    /// Set the triple pattern.
    pub fn pattern(mut self, pattern: TriplePattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Set a subject filter.
    pub fn subject(mut self, subject: impl Into<String>) -> Self {
        self.pattern.subject = Some(subject.into());
        self
    }

    /// Set a predicate filter.
    pub fn predicate(mut self, predicate: impl Into<String>) -> Self {
        self.pattern.predicate = Some(predicate.into());
        self
    }

    /// Set an object filter.
    pub fn object(mut self, object: impl Into<String>) -> Self {
        self.pattern.object = Some(object.into());
        self
    }

    /// Build the query.
    ///
    /// Returns `TdbError::InvalidInput` if `as_of` was not set.
    pub fn build(self) -> Result<TimeTravelQuery> {
        let as_of = self.as_of.ok_or_else(|| {
            TdbError::InvalidInput("TimeTravelQueryBuilder: as_of timestamp is required".into())
        })?;
        Ok(TimeTravelQuery {
            as_of,
            pattern: self.pattern,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal::version_store::TemporalVersionStore;
    use chrono::TimeZone;

    fn ts(y: i32, m: u32, d: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, m, d, 0, 0, 0).unwrap()
    }

    fn populated_store() -> TemporalVersionStore {
        let mut store = TemporalVersionStore::new();
        store
            .insert_at("Alice", "knows", "Bob", ts(2024, 1, 1))
            .unwrap();
        store
            .insert_at("Alice", "knows", "Carol", ts(2024, 6, 1))
            .unwrap();
        store
            .insert_at("Bob", "likes", "Rust", ts(2024, 3, 1))
            .unwrap();
        store
            .delete_at("Alice", "knows", "Bob", ts(2024, 9, 1))
            .unwrap();
        store
    }

    #[test]
    fn test_time_travel_wildcard() {
        let store = populated_store();
        let q = TimeTravelQuery::at(ts(2024, 2, 1));
        let snap = q.execute(&store).unwrap();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap.triples()[0].object, "Bob");
    }

    #[test]
    fn test_time_travel_with_pattern() {
        let store = populated_store();
        let pattern = TriplePattern::with_subject("Alice");
        let q = TimeTravelQuery::new(ts(2024, 7, 1), pattern);
        let snap = q.execute(&store).unwrap();
        // Alice knows Bob (still active before Sept), Alice knows Carol (from June)
        assert_eq!(snap.len(), 2);
    }

    #[test]
    fn test_time_travel_after_delete() {
        let store = populated_store();
        let pattern = TriplePattern::exact("Alice", "knows", "Bob");
        let q = TimeTravelQuery::new(ts(2024, 10, 1), pattern);
        let snap = q.execute(&store).unwrap();
        assert!(snap.is_empty(), "Bob should be deleted after Sept 2024");
    }

    #[test]
    fn test_builder() {
        let store = populated_store();
        let q = TimeTravelQueryBuilder::new()
            .as_of(ts(2024, 4, 1))
            .subject("Bob")
            .build()
            .unwrap();
        let snap = q.execute(&store).unwrap();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap.triples()[0].predicate, "likes");
    }

    #[test]
    fn test_builder_missing_as_of() {
        let result = TimeTravelQueryBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_pattern_matches() {
        let t = VersionedTriple::new("s", "p", "o", Utc::now(), 1);
        assert!(TriplePattern::wildcard().matches(&t));
        assert!(TriplePattern::with_subject("s").matches(&t));
        assert!(!TriplePattern::with_subject("x").matches(&t));
        assert!(TriplePattern::exact("s", "p", "o").matches(&t));
        assert!(!TriplePattern::exact("s", "p", "z").matches(&t));
    }
}
