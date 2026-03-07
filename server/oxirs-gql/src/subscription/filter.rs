//! Subscription filter: match `ChangeEvent`s against SPARQL-style patterns.
//!
//! A `SubscriptionFilter` combines optional constraints on subject, predicate,
//! graph, and event type.  All provided constraints must match simultaneously
//! (logical AND).  When a constraint field is `None` / empty it is treated as a
//! wildcard (matches everything).
//!
//! Matching is intentionally kept allocation-free on the hot path: constraints
//! are stored as owned `String`s and compared with `==` / `starts_with`.  For
//! prefix-based matching (e.g. "all triples in the `foaf:` namespace"), the
//! `MatchStrategy` enum allows selecting between exact and prefix modes.

use crate::subscription::change_tracker::{ChangeEvent, ChangeType};

/// How a string constraint is evaluated against an RDF term.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum MatchStrategy {
    /// The term must be exactly equal to the constraint value.
    #[default]
    Exact,
    /// The term must start with the constraint value (useful for IRI prefixes).
    Prefix,
}

/// A single string constraint paired with a matching strategy.
#[derive(Debug, Clone)]
pub struct StringConstraint {
    pub value: String,
    pub strategy: MatchStrategy,
}

impl StringConstraint {
    /// Create an exact-match constraint.
    pub fn exact(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            strategy: MatchStrategy::Exact,
        }
    }

    /// Create a prefix-match constraint.
    pub fn prefix(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            strategy: MatchStrategy::Prefix,
        }
    }

    /// Test whether `term` satisfies this constraint.
    pub fn matches_term(&self, term: &str) -> bool {
        match self.strategy {
            MatchStrategy::Exact => term == self.value,
            MatchStrategy::Prefix => term.starts_with(&self.value),
        }
    }
}

/// Filter that selects which `ChangeEvent`s a subscription is interested in.
///
/// All non-`None` constraints are ANDed:
/// - `subject_pattern`  – `None` = accept any subject.
/// - `predicate_pattern` – `None` = accept any predicate.
/// - `graph`            – `None` = accept default graph and all named graphs.
/// - `event_types`      – empty `Vec` = accept all change types.
///
/// Build via `SubscriptionFilter::builder()` or construct directly.
#[derive(Debug, Clone)]
pub struct SubscriptionFilter {
    /// Optional constraint on the triple's subject.
    pub subject_pattern: Option<StringConstraint>,
    /// Optional constraint on the triple's predicate.
    pub predicate_pattern: Option<StringConstraint>,
    /// Optional constraint on the named graph IRI (exact match).
    /// - `Some(None)`  – only the default graph.
    /// - `Some(Some(iri))` – only the named graph with `iri`.
    /// - `None`         – any graph (default or named).
    pub graph: Option<Option<String>>,
    /// Accepted event types.  Empty = accept all.
    pub event_types: Vec<ChangeType>,
}

impl SubscriptionFilter {
    /// A filter that accepts every `ChangeEvent`.
    pub fn all() -> Self {
        Self {
            subject_pattern: None,
            predicate_pattern: None,
            graph: None,
            event_types: vec![],
        }
    }

    /// Convenience: match only `Insert` events.
    pub fn inserts_only() -> Self {
        Self {
            subject_pattern: None,
            predicate_pattern: None,
            graph: None,
            event_types: vec![ChangeType::Insert],
        }
    }

    /// Convenience: match only `Delete` events.
    pub fn deletes_only() -> Self {
        Self {
            subject_pattern: None,
            predicate_pattern: None,
            graph: None,
            event_types: vec![ChangeType::Delete],
        }
    }

    /// Convenience: match only events in the default graph.
    pub fn default_graph_only() -> Self {
        Self {
            subject_pattern: None,
            predicate_pattern: None,
            graph: Some(None),
            event_types: vec![],
        }
    }

    /// Convenience: match only events in a specific named graph.
    pub fn named_graph(iri: impl Into<String>) -> Self {
        Self {
            subject_pattern: None,
            predicate_pattern: None,
            graph: Some(Some(iri.into())),
            event_types: vec![],
        }
    }

    /// Start building a filter with a fluent API.
    pub fn builder() -> FilterBuilder {
        FilterBuilder::new()
    }

    /// Test whether a `ChangeEvent` satisfies this filter.
    ///
    /// Returns `true` iff every non-`None` / non-empty constraint is satisfied.
    pub fn matches(&self, event: &ChangeEvent) -> bool {
        // --- event type constraint ---
        if !self.event_types.is_empty() && !self.event_types.contains(&event.event_type) {
            return false;
        }

        // --- graph constraint ---
        if let Some(ref required_graph) = self.graph {
            if event.graph != *required_graph {
                return false;
            }
        }

        // --- subject constraint ---
        if let Some(ref sc) = self.subject_pattern {
            if !sc.matches_term(&event.subject) {
                return false;
            }
        }

        // --- predicate constraint ---
        if let Some(ref pc) = self.predicate_pattern {
            if !pc.matches_term(&event.predicate) {
                return false;
            }
        }

        true
    }
}

/// Fluent builder for `SubscriptionFilter`.
#[derive(Debug, Default)]
pub struct FilterBuilder {
    subject_pattern: Option<StringConstraint>,
    predicate_pattern: Option<StringConstraint>,
    graph: Option<Option<String>>,
    event_types: Vec<ChangeType>,
}

impl FilterBuilder {
    /// Create a new builder (all wildcards).
    pub fn new() -> Self {
        Self::default()
    }

    /// Constrain to an exact subject IRI.
    pub fn subject(mut self, iri: impl Into<String>) -> Self {
        self.subject_pattern = Some(StringConstraint::exact(iri));
        self
    }

    /// Constrain to any subject whose IRI starts with the given prefix.
    pub fn subject_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.subject_pattern = Some(StringConstraint::prefix(prefix));
        self
    }

    /// Constrain to an exact predicate IRI.
    pub fn predicate(mut self, iri: impl Into<String>) -> Self {
        self.predicate_pattern = Some(StringConstraint::exact(iri));
        self
    }

    /// Constrain to any predicate whose IRI starts with the given prefix.
    pub fn predicate_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.predicate_pattern = Some(StringConstraint::prefix(prefix));
        self
    }

    /// Constrain to the default (unnamed) graph.
    pub fn default_graph(mut self) -> Self {
        self.graph = Some(None);
        self
    }

    /// Constrain to a specific named graph IRI.
    pub fn graph(mut self, iri: impl Into<String>) -> Self {
        self.graph = Some(Some(iri.into()));
        self
    }

    /// Add an accepted event type.  May be called multiple times.
    pub fn event_type(mut self, et: ChangeType) -> Self {
        self.event_types.push(et);
        self
    }

    /// Produce the `SubscriptionFilter`.
    pub fn build(self) -> SubscriptionFilter {
        SubscriptionFilter {
            subject_pattern: self.subject_pattern,
            predicate_pattern: self.predicate_pattern,
            graph: self.graph,
            event_types: self.event_types,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subscription::change_tracker::{ChangeEvent, ChangeType};

    fn insert_event(subject: &str, predicate: &str, graph: Option<&str>) -> ChangeEvent {
        ChangeEvent::new(
            1,
            ChangeType::Insert,
            subject,
            predicate,
            "http://ex.org/o",
            graph.map(|g| g.to_string()),
        )
    }

    fn delete_event(subject: &str, predicate: &str) -> ChangeEvent {
        ChangeEvent::new(
            2,
            ChangeType::Delete,
            subject,
            predicate,
            "http://ex.org/o",
            None,
        )
    }

    #[test]
    fn test_filter_all_matches_any_event() {
        let f = SubscriptionFilter::all();
        assert!(f.matches(&insert_event("s", "p", None)));
        assert!(f.matches(&delete_event("s2", "p2")));
    }

    #[test]
    fn test_filter_inserts_only() {
        let f = SubscriptionFilter::inserts_only();
        assert!(f.matches(&insert_event("s", "p", None)));
        assert!(!f.matches(&delete_event("s", "p")));
    }

    #[test]
    fn test_filter_deletes_only() {
        let f = SubscriptionFilter::deletes_only();
        assert!(!f.matches(&insert_event("s", "p", None)));
        assert!(f.matches(&delete_event("s", "p")));
    }

    #[test]
    fn test_filter_default_graph_only() {
        let f = SubscriptionFilter::default_graph_only();
        assert!(f.matches(&insert_event("s", "p", None)));
        assert!(!f.matches(&insert_event("s", "p", Some("http://ex.org/g"))));
    }

    #[test]
    fn test_filter_named_graph() {
        let f = SubscriptionFilter::named_graph("http://ex.org/g");
        assert!(f.matches(&insert_event("s", "p", Some("http://ex.org/g"))));
        assert!(!f.matches(&insert_event("s", "p", None)));
        assert!(!f.matches(&insert_event("s", "p", Some("http://ex.org/other"))));
    }

    #[test]
    fn test_filter_builder_exact_subject() {
        let f = SubscriptionFilter::builder()
            .subject("http://ex.org/person1")
            .build();
        assert!(f.matches(&insert_event("http://ex.org/person1", "p", None)));
        assert!(!f.matches(&insert_event("http://ex.org/person2", "p", None)));
    }

    #[test]
    fn test_filter_builder_subject_prefix() {
        let f = SubscriptionFilter::builder()
            .subject_prefix("http://ex.org/person")
            .build();
        assert!(f.matches(&insert_event("http://ex.org/person1", "p", None)));
        assert!(f.matches(&insert_event("http://ex.org/personX", "p", None)));
        assert!(!f.matches(&insert_event("http://ex.org/thing1", "p", None)));
    }

    #[test]
    fn test_filter_builder_exact_predicate() {
        let f = SubscriptionFilter::builder()
            .predicate("http://xmlns.com/foaf/0.1/name")
            .build();
        assert!(f.matches(&insert_event("s", "http://xmlns.com/foaf/0.1/name", None)));
        assert!(!f.matches(&insert_event("s", "http://xmlns.com/foaf/0.1/age", None)));
    }

    #[test]
    fn test_filter_builder_predicate_prefix() {
        let f = SubscriptionFilter::builder()
            .predicate_prefix("http://xmlns.com/foaf/")
            .build();
        assert!(f.matches(&insert_event("s", "http://xmlns.com/foaf/name", None)));
        assert!(!f.matches(&insert_event(
            "s",
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            None
        )));
    }

    #[test]
    fn test_filter_builder_multiple_event_types() {
        let f = SubscriptionFilter::builder()
            .event_type(ChangeType::Insert)
            .event_type(ChangeType::Update)
            .build();

        let update_ev = ChangeEvent::new(3, ChangeType::Update, "s", "p", "o", None);
        assert!(f.matches(&insert_event("s", "p", None)));
        assert!(f.matches(&update_ev));
        assert!(!f.matches(&delete_event("s", "p")));
    }

    #[test]
    fn test_filter_builder_combined_constraints() {
        let f = SubscriptionFilter::builder()
            .subject("http://ex.org/alice")
            .predicate_prefix("http://xmlns.com/foaf/")
            .graph("http://ex.org/people")
            .event_type(ChangeType::Insert)
            .build();

        let matching = ChangeEvent::new(
            1,
            ChangeType::Insert,
            "http://ex.org/alice",
            "http://xmlns.com/foaf/name",
            "Alice",
            Some("http://ex.org/people".to_string()),
        );
        assert!(f.matches(&matching));

        // Wrong subject
        let wrong_subject = ChangeEvent::new(
            2,
            ChangeType::Insert,
            "http://ex.org/bob",
            "http://xmlns.com/foaf/name",
            "Bob",
            Some("http://ex.org/people".to_string()),
        );
        assert!(!f.matches(&wrong_subject));
    }

    #[test]
    fn test_string_constraint_exact() {
        let c = StringConstraint::exact("foo");
        assert!(c.matches_term("foo"));
        assert!(!c.matches_term("foobar"));
        assert!(!c.matches_term("bar"));
    }

    #[test]
    fn test_string_constraint_prefix() {
        let c = StringConstraint::prefix("http://ex.org/");
        assert!(c.matches_term("http://ex.org/thing"));
        assert!(!c.matches_term("http://other.org/thing"));
    }
}
