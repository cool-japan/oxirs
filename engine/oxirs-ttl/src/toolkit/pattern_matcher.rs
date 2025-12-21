//! Pattern Matching and Query Utilities
//!
//! This module provides simple SPARQL-like pattern matching for in-memory RDF graphs.
//! Useful for quick queries without setting up a full SPARQL engine.
//!
//! # Examples
//!
//! ## Basic Pattern Matching
//!
//! ```rust
//! use oxirs_ttl::toolkit::pattern_matcher::{PatternMatcher, TriplePattern};
//! use oxirs_core::model::{Triple, NamedNode};
//!
//! let triples = vec![
//!     Triple::new(
//!         NamedNode::new("http://example.org/alice")?,
//!         NamedNode::new("http://xmlns.com/foaf/0.1/knows")?,
//!         NamedNode::new("http://example.org/bob")?,
//!     ),
//!     Triple::new(
//!         NamedNode::new("http://example.org/bob")?,
//!         NamedNode::new("http://xmlns.com/foaf/0.1/knows")?,
//!         NamedNode::new("http://example.org/charlie")?,
//!     ),
//! ];
//!
//! let matcher = PatternMatcher::new(&triples);
//!
//! // Find all triples with "knows" predicate
//! let pattern = TriplePattern::new()
//!     .with_predicate("http://xmlns.com/foaf/0.1/knows");
//!
//! let results = matcher.find_matches(&pattern);
//! assert_eq!(results.len(), 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Variable Binding
//!
//! ```rust
//! use oxirs_ttl::toolkit::pattern_matcher::{PatternMatcher, TriplePattern};
//! use oxirs_core::model::{Triple, NamedNode};
//!
//! let triples = vec![
//!     Triple::new(
//!         NamedNode::new("http://example.org/alice")?,
//!         NamedNode::new("http://example.org/age")?,
//!         NamedNode::new("http://example.org/30")?,
//!     ),
//! ];
//!
//! let matcher = PatternMatcher::new(&triples);
//! let pattern = TriplePattern::new()
//!     .with_subject("http://example.org/alice");
//!
//! let results = matcher.find_matches(&pattern);
//! assert_eq!(results.len(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use oxirs_core::model::Triple;
use oxirs_core::RdfTerm;
use std::collections::HashMap;

/// A pattern for matching RDF triples
///
/// Patterns can specify exact values or leave components unbound (wildcard).
#[derive(Debug, Clone)]
pub struct TriplePattern {
    subject: Option<String>,
    predicate: Option<String>,
    object: Option<String>,
}

impl TriplePattern {
    /// Create a new empty pattern (matches all triples)
    pub fn new() -> Self {
        Self {
            subject: None,
            predicate: None,
            object: None,
        }
    }

    /// Set the subject pattern
    pub fn with_subject(mut self, subject: &str) -> Self {
        self.subject = Some(subject.to_string());
        self
    }

    /// Set the predicate pattern
    pub fn with_predicate(mut self, predicate: &str) -> Self {
        self.predicate = Some(predicate.to_string());
        self
    }

    /// Set the object pattern
    pub fn with_object(mut self, object: &str) -> Self {
        self.object = Some(object.to_string());
        self
    }

    /// Check if a triple matches this pattern
    pub fn matches(&self, triple: &Triple) -> bool {
        if let Some(ref subject) = self.subject {
            if !triple.subject().as_str().contains(subject) {
                return false;
            }
        }

        if let Some(ref predicate) = self.predicate {
            if !triple.predicate().as_str().contains(predicate) {
                return false;
            }
        }

        if let Some(ref object) = self.object {
            if !triple.object().as_str().contains(object) {
                return false;
            }
        }

        true
    }

    /// Check if pattern is empty (matches all)
    pub fn is_empty(&self) -> bool {
        self.subject.is_none() && self.predicate.is_none() && self.object.is_none()
    }
}

impl Default for TriplePattern {
    fn default() -> Self {
        Self::new()
    }
}

/// Pattern matcher for RDF triples
///
/// Provides efficient pattern matching over in-memory triple collections.
#[derive(Debug)]
pub struct PatternMatcher<'a> {
    triples: &'a [Triple],
    // Index for faster lookups
    subject_index: HashMap<String, Vec<usize>>,
    predicate_index: HashMap<String, Vec<usize>>,
    object_index: HashMap<String, Vec<usize>>,
}

impl<'a> PatternMatcher<'a> {
    /// Create a new pattern matcher for a collection of triples
    ///
    /// This builds indices for efficient pattern matching.
    pub fn new(triples: &'a [Triple]) -> Self {
        let mut subject_index: HashMap<String, Vec<usize>> = HashMap::new();
        let mut predicate_index: HashMap<String, Vec<usize>> = HashMap::new();
        let mut object_index: HashMap<String, Vec<usize>> = HashMap::new();

        for (idx, triple) in triples.iter().enumerate() {
            subject_index
                .entry(triple.subject().to_string())
                .or_default()
                .push(idx);

            predicate_index
                .entry(triple.predicate().to_string())
                .or_default()
                .push(idx);

            object_index
                .entry(triple.object().to_string())
                .or_default()
                .push(idx);
        }

        Self {
            triples,
            subject_index,
            predicate_index,
            object_index,
        }
    }

    /// Find all triples matching a pattern
    ///
    /// Returns references to matching triples.
    pub fn find_matches(&self, pattern: &TriplePattern) -> Vec<&Triple> {
        if pattern.is_empty() {
            return self.triples.iter().collect();
        }

        // Use indices for more efficient lookup
        let candidates = if let Some(ref subject) = pattern.subject {
            self.subject_index
                .iter()
                .filter(|(k, _)| k.contains(subject))
                .flat_map(|(_, indices)| indices.iter().copied())
                .collect::<Vec<_>>()
        } else if let Some(ref predicate) = pattern.predicate {
            self.predicate_index
                .iter()
                .filter(|(k, _)| k.contains(predicate))
                .flat_map(|(_, indices)| indices.iter().copied())
                .collect::<Vec<_>>()
        } else if let Some(ref object) = pattern.object {
            self.object_index
                .iter()
                .filter(|(k, _)| k.contains(object))
                .flat_map(|(_, indices)| indices.iter().copied())
                .collect::<Vec<_>>()
        } else {
            (0..self.triples.len()).collect()
        };

        candidates
            .into_iter()
            .filter_map(|idx| {
                let triple = &self.triples[idx];
                if pattern.matches(triple) {
                    Some(triple)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Count matches for a pattern without collecting results
    pub fn count_matches(&self, pattern: &TriplePattern) -> usize {
        if pattern.is_empty() {
            return self.triples.len();
        }

        self.triples.iter().filter(|t| pattern.matches(t)).count()
    }

    /// Check if any triple matches the pattern
    pub fn has_match(&self, pattern: &TriplePattern) -> bool {
        if pattern.is_empty() {
            return !self.triples.is_empty();
        }

        self.triples.iter().any(|t| pattern.matches(t))
    }

    /// Get all unique subjects
    pub fn subjects(&self) -> Vec<String> {
        self.subject_index.keys().cloned().collect()
    }

    /// Get all unique predicates
    pub fn predicates(&self) -> Vec<String> {
        self.predicate_index.keys().cloned().collect()
    }

    /// Get all unique objects
    pub fn objects(&self) -> Vec<String> {
        self.object_index.keys().cloned().collect()
    }
}

/// Query builder for more complex patterns
#[derive(Debug)]
pub struct QueryBuilder {
    patterns: Vec<TriplePattern>,
    limit: Option<usize>,
    offset: usize,
}

impl QueryBuilder {
    /// Create a new query builder
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
            limit: None,
            offset: 0,
        }
    }

    /// Add a pattern to the query
    pub fn pattern(mut self, pattern: TriplePattern) -> Self {
        self.patterns.push(pattern);
        self
    }

    /// Set a limit on results
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set an offset for results
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Execute the query against a matcher
    pub fn execute<'a>(&self, matcher: &'a PatternMatcher) -> Vec<&'a Triple> {
        if self.patterns.is_empty() {
            return Vec::new();
        }

        // Start with first pattern
        let mut results = matcher.find_matches(&self.patterns[0]);

        // Intersect with remaining patterns
        for pattern in &self.patterns[1..] {
            let pattern_results = matcher.find_matches(pattern);
            results.retain(|t| pattern_results.contains(t));
        }

        // Apply offset and limit
        let results: Vec<_> = results.into_iter().skip(self.offset).collect();

        if let Some(limit) = self.limit {
            results.into_iter().take(limit).collect()
        } else {
            results
        }
    }

    /// Count results without collecting
    pub fn count(&self, matcher: &PatternMatcher) -> usize {
        if self.patterns.is_empty() {
            return 0;
        }

        let mut results = matcher.find_matches(&self.patterns[0]);

        for pattern in &self.patterns[1..] {
            let pattern_results = matcher.find_matches(pattern);
            results.retain(|t| pattern_results.contains(t));
        }

        if self.offset >= results.len() {
            return 0;
        }

        let remaining = results.len() - self.offset;
        self.limit.map_or(remaining, |limit| remaining.min(limit))
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxirs_core::model::NamedNode;

    fn create_test_triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(
            NamedNode::new(s).unwrap(),
            NamedNode::new(p).unwrap(),
            NamedNode::new(o).unwrap(),
        )
    }

    #[test]
    fn test_pattern_matching() {
        let triples = vec![
            create_test_triple(
                "http://example.org/alice",
                "http://xmlns.com/foaf/0.1/knows",
                "http://example.org/bob",
            ),
            create_test_triple(
                "http://example.org/bob",
                "http://xmlns.com/foaf/0.1/knows",
                "http://example.org/charlie",
            ),
            create_test_triple(
                "http://example.org/alice",
                "http://example.org/age",
                "http://example.org/30",
            ),
        ];

        let matcher = PatternMatcher::new(&triples);

        // Match by predicate
        let pattern = TriplePattern::new().with_predicate("foaf/0.1/knows");
        let results = matcher.find_matches(&pattern);
        assert_eq!(results.len(), 2);

        // Match by subject
        let pattern = TriplePattern::new().with_subject("alice");
        let results = matcher.find_matches(&pattern);
        assert_eq!(results.len(), 2);

        // Match by multiple criteria
        let pattern = TriplePattern::new()
            .with_subject("alice")
            .with_predicate("knows");
        let results = matcher.find_matches(&pattern);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_count_and_has_match() {
        let triples = vec![create_test_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        )];

        let matcher = PatternMatcher::new(&triples);
        let pattern = TriplePattern::new().with_subject("example.org/s");

        assert_eq!(matcher.count_matches(&pattern), 1);
        assert!(matcher.has_match(&pattern));

        let no_match = TriplePattern::new().with_subject("notfound");
        assert!(!matcher.has_match(&no_match));
    }

    #[test]
    fn test_query_builder() {
        let triples = vec![
            create_test_triple(
                "http://example.org/alice",
                "http://xmlns.com/foaf/0.1/knows",
                "http://example.org/bob",
            ),
            create_test_triple(
                "http://example.org/bob",
                "http://xmlns.com/foaf/0.1/knows",
                "http://example.org/charlie",
            ),
            create_test_triple(
                "http://example.org/alice",
                "http://example.org/age",
                "http://example.org/30",
            ),
        ];

        let matcher = PatternMatcher::new(&triples);

        let query = QueryBuilder::new()
            .pattern(TriplePattern::new().with_predicate("knows"))
            .limit(1);

        let results = query.execute(&matcher);
        assert_eq!(results.len(), 1);

        let count = query.count(&matcher);
        assert_eq!(count, 1);
    }

    #[test]
    fn test_empty_pattern() {
        let triples = vec![create_test_triple(
            "http://example.org/s",
            "http://example.org/p",
            "http://example.org/o",
        )];

        let matcher = PatternMatcher::new(&triples);
        let pattern = TriplePattern::new();

        assert!(pattern.is_empty());
        assert_eq!(matcher.find_matches(&pattern).len(), 1);
    }

    #[test]
    fn test_index_queries() {
        let triples = vec![
            create_test_triple(
                "http://example.org/s1",
                "http://example.org/p",
                "http://example.org/o1",
            ),
            create_test_triple(
                "http://example.org/s2",
                "http://example.org/p",
                "http://example.org/o2",
            ),
        ];

        let matcher = PatternMatcher::new(&triples);

        let subjects = matcher.subjects();
        assert_eq!(subjects.len(), 2);

        let predicates = matcher.predicates();
        assert_eq!(predicates.len(), 1);

        let objects = matcher.objects();
        assert_eq!(objects.len(), 2);
    }

    #[test]
    fn test_query_with_offset() {
        let triples = vec![
            create_test_triple(
                "http://example.org/s1",
                "http://example.org/p",
                "http://example.org/o1",
            ),
            create_test_triple(
                "http://example.org/s2",
                "http://example.org/p",
                "http://example.org/o2",
            ),
            create_test_triple(
                "http://example.org/s3",
                "http://example.org/p",
                "http://example.org/o3",
            ),
        ];

        let matcher = PatternMatcher::new(&triples);

        let query = QueryBuilder::new()
            .pattern(TriplePattern::new().with_predicate("example.org/p"))
            .offset(1)
            .limit(1);

        let results = query.execute(&matcher);
        assert_eq!(results.len(), 1);
    }
}
