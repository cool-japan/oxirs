//! SPARQL-star query pattern matching.
//!
//! This module provides pattern types and an evaluator for SPARQL-star query
//! patterns over RDF-star graphs:
//!
//! - [`EmbeddedTriplePattern`]: represents `<< s p o >>` in a WHERE clause.
//! - [`AnnotationPattern`]: represents the `?s |p| ?o` annotation syntax.
//! - [`SparqlStarEvaluator`]: evaluates patterns over a graph slice.
//!
//! # Example
//!
//! ```
//! use oxirs_star::sparql::{EmbeddedTriplePattern, SparqlStarEvaluator};
//! use oxirs_star::model::StarTerm;
//!
//! let pattern = EmbeddedTriplePattern::new(
//!     StarTerm::variable("s").unwrap(),
//!     StarTerm::iri("http://example.org/age").unwrap(),
//!     StarTerm::variable("o").unwrap(),
//! );
//!
//! let evaluator = SparqlStarEvaluator::new();
//! // (graph would contain RDF-star triples here)
//! let bindings = evaluator.evaluate_embedded_pattern(&pattern, &[]);
//! assert!(bindings.is_empty());
//! ```

use std::collections::HashMap;

use crate::model::{StarTerm, StarTriple};

/// A variable binding map: maps variable names to [`StarTerm`] values.
pub type Binding = HashMap<String, StarTerm>;

// ============================================================================
// EmbeddedTriplePattern
// ============================================================================

/// A pattern for matching embedded (quoted) triples in SPARQL-star queries.
///
/// Each position (subject, predicate, object) can be either a concrete
/// [`StarTerm`] or a variable (`StarTerm::Variable`). Variables bind to any
/// matching term; concrete terms must match exactly.
///
/// Corresponds to the `<< s p o >>` construct in SPARQL-star WHERE clauses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddedTriplePattern {
    /// Subject pattern (variable or concrete term).
    pub subject: StarTerm,
    /// Predicate pattern (variable or concrete term).
    pub predicate: StarTerm,
    /// Object pattern (variable or concrete term).
    pub object: StarTerm,
}

impl EmbeddedTriplePattern {
    /// Create a new embedded triple pattern.
    pub fn new(subject: StarTerm, predicate: StarTerm, object: StarTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Returns `true` if all positions are variables (completely unbound pattern).
    pub fn is_fully_unbound(&self) -> bool {
        self.subject.is_variable() && self.predicate.is_variable() && self.object.is_variable()
    }

    /// Returns `true` if all positions are concrete terms (no variables).
    pub fn is_ground(&self) -> bool {
        !self.subject.is_variable() && !self.predicate.is_variable() && !self.object.is_variable()
    }

    /// Attempt to unify this pattern against `triple`, returning a
    /// [`Binding`] on success or `None` on failure.
    ///
    /// Unification rules:
    /// - A variable in the pattern binds to the corresponding triple term.
    /// - A concrete term in the pattern must equal the corresponding triple term.
    /// - If the same variable appears in multiple positions, all bound values
    ///   must be equal (occurs check).
    pub fn unify(&self, triple: &StarTriple) -> Option<Binding> {
        let mut bindings = Binding::new();
        unify_term(&self.subject, &triple.subject, &mut bindings)?;
        unify_term(&self.predicate, &triple.predicate, &mut bindings)?;
        unify_term(&self.object, &triple.object, &mut bindings)?;
        Some(bindings)
    }

    /// Match this pattern against a quoted triple embedded inside another triple.
    ///
    /// Given a term that is expected to be a [`StarTerm::QuotedTriple`], attempt
    /// to unify the inner triple with this pattern.
    pub fn unify_embedded(&self, term: &StarTerm) -> Option<Binding> {
        if let StarTerm::QuotedTriple(inner) = term {
            self.unify(inner)
        } else {
            None
        }
    }
}

// ============================================================================
// AnnotationPattern
// ============================================================================

/// An annotation pattern matching the SPARQL-star annotation syntax:
///
/// ```sparql
/// ?s |ex:certainty| ?confidence
/// ```
///
/// This matches triples where the subject is a quoted triple `<< ?s_inner ?p_inner
/// ?o_inner >>` that is annotated with the annotation predicate and object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnotationPattern {
    /// Pattern for the subject position of the inner (embedded) triple.
    pub embedded_subject: StarTerm,
    /// Pattern for the predicate position of the inner (embedded) triple.
    pub embedded_predicate: StarTerm,
    /// Pattern for the object position of the inner (embedded) triple.
    pub embedded_object: StarTerm,
    /// The annotation predicate (may be a variable).
    pub annotation_predicate: StarTerm,
    /// The annotation object (may be a variable).
    pub annotation_object: StarTerm,
}

impl AnnotationPattern {
    /// Create a new annotation pattern.
    pub fn new(
        embedded_subject: StarTerm,
        embedded_predicate: StarTerm,
        embedded_object: StarTerm,
        annotation_predicate: StarTerm,
        annotation_object: StarTerm,
    ) -> Self {
        Self {
            embedded_subject,
            embedded_predicate,
            embedded_object,
            annotation_predicate,
            annotation_object,
        }
    }

    /// Attempt to match a graph triple against this annotation pattern.
    ///
    /// The outer triple must have a [`StarTerm::QuotedTriple`] as its subject.
    /// The embedded triple pattern is unified against the inner triple, and the
    /// annotation predicate and object are unified against the outer triple's
    /// predicate and object.
    pub fn match_triple(&self, outer: &StarTriple) -> Option<Binding> {
        // The outer triple's subject must be a quoted triple
        let inner = match &outer.subject {
            StarTerm::QuotedTriple(inner) => inner.as_ref(),
            _ => return None,
        };

        let mut bindings = Binding::new();

        // Unify embedded triple pattern
        unify_term(&self.embedded_subject, &inner.subject, &mut bindings)?;
        unify_term(&self.embedded_predicate, &inner.predicate, &mut bindings)?;
        unify_term(&self.embedded_object, &inner.object, &mut bindings)?;

        // Unify annotation predicate and object
        unify_term(&self.annotation_predicate, &outer.predicate, &mut bindings)?;
        unify_term(&self.annotation_object, &outer.object, &mut bindings)?;

        Some(bindings)
    }
}

// ============================================================================
// Unification helper
// ============================================================================

/// Unify `pattern_term` against `data_term`, merging results into `bindings`.
///
/// Returns `None` if unification fails (type mismatch or occurs check failure).
fn unify_term(pattern_term: &StarTerm, data_term: &StarTerm, bindings: &mut Binding) -> Option<()> {
    match pattern_term {
        StarTerm::Variable(var) => {
            // Check occurs: if already bound, the bound value must match
            if let Some(existing) = bindings.get(&var.name) {
                if existing != data_term {
                    return None;
                }
            } else {
                bindings.insert(var.name.clone(), data_term.clone());
            }
            Some(())
        }
        StarTerm::QuotedTriple(pattern_inner) => {
            // Match a nested quoted triple pattern
            if let StarTerm::QuotedTriple(data_inner) = data_term {
                unify_term(&pattern_inner.subject, &data_inner.subject, bindings)?;
                unify_term(&pattern_inner.predicate, &data_inner.predicate, bindings)?;
                unify_term(&pattern_inner.object, &data_inner.object, bindings)?;
                Some(())
            } else {
                None
            }
        }
        concrete => {
            // Concrete term: must equal the data term exactly
            if concrete == data_term {
                Some(())
            } else {
                None
            }
        }
    }
}

// ============================================================================
// SparqlStarEvaluator
// ============================================================================

/// Evaluates SPARQL-star patterns over RDF-star graphs.
///
/// Provides high-level evaluation functions that return all matching
/// [`Binding`]s.
pub struct SparqlStarEvaluator {
    /// Whether to also recurse into nested quoted triples during evaluation.
    pub recurse_nested: bool,
}

impl Default for SparqlStarEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl SparqlStarEvaluator {
    /// Create a new evaluator (recursion disabled by default).
    pub fn new() -> Self {
        Self {
            recurse_nested: false,
        }
    }

    /// Create an evaluator that also searches inside nested quoted triples.
    pub fn with_recursion() -> Self {
        Self {
            recurse_nested: true,
        }
    }

    // ------------------------------------------------------------------
    // evaluate_embedded_pattern
    // ------------------------------------------------------------------

    /// Find all triples in `graph` whose *subject* is a quoted triple matching
    /// `pattern`, and return the variable bindings for each match.
    ///
    /// For every outer triple `<< s p o >> q_pred q_obj`, attempt to unify the
    /// embedded `<< s p o >>` against `pattern`.  If the outer triple's subject
    /// is not a quoted triple it is skipped.
    pub fn evaluate_embedded_pattern(
        &self,
        pattern: &EmbeddedTriplePattern,
        graph: &[StarTriple],
    ) -> Vec<Binding> {
        let mut results = Vec::new();

        for triple in graph {
            // Try to match quoted subject
            if let Some(bindings) = pattern.unify_embedded(&triple.subject) {
                results.push(bindings);
            }

            // Try to match quoted object
            if let Some(bindings) = pattern.unify_embedded(&triple.object) {
                results.push(bindings);
            }

            // Optionally recurse into nested quoted triples
            if self.recurse_nested {
                self.recurse_embedded(&triple.subject, pattern, &mut results);
                self.recurse_embedded(&triple.object, pattern, &mut results);
            }
        }

        results
    }

    fn recurse_embedded(
        &self,
        term: &StarTerm,
        pattern: &EmbeddedTriplePattern,
        results: &mut Vec<Binding>,
    ) {
        if let StarTerm::QuotedTriple(inner) = term {
            if let Some(bindings) = pattern.unify(inner) {
                results.push(bindings);
            }
            // Go deeper
            self.recurse_embedded(&inner.subject, pattern, results);
            self.recurse_embedded(&inner.predicate, pattern, results);
            self.recurse_embedded(&inner.object, pattern, results);
        }
    }

    // ------------------------------------------------------------------
    // evaluate_annotation_pattern
    // ------------------------------------------------------------------

    /// Find all triples in `graph` that match `pattern` (annotation syntax).
    ///
    /// An annotation triple has a quoted-triple subject and carries annotation
    /// metadata in its predicate and object positions.
    pub fn evaluate_annotation_pattern(
        &self,
        pattern: &AnnotationPattern,
        graph: &[StarTriple],
    ) -> Vec<Binding> {
        let mut results = Vec::new();

        for triple in graph {
            if let Some(bindings) = pattern.match_triple(triple) {
                results.push(bindings);
            }
        }

        results
    }

    // ------------------------------------------------------------------
    // evaluate_direct_pattern (bonus: match non-embedded triples)
    // ------------------------------------------------------------------

    /// Find all triples in `graph` that directly match `pattern` (treating each
    /// graph triple as a potential match, not requiring embedded subjects).
    ///
    /// This is analogous to a basic SPARQL triple pattern.
    pub fn evaluate_direct_pattern(
        &self,
        pattern: &EmbeddedTriplePattern,
        graph: &[StarTriple],
    ) -> Vec<Binding> {
        let mut results = Vec::new();

        for triple in graph {
            if let Some(bindings) = pattern.unify(triple) {
                results.push(bindings);
            }
        }

        results
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::StarTriple;

    fn iri(s: &str) -> StarTerm {
        StarTerm::iri(s).expect("iri")
    }
    fn lit(s: &str) -> StarTerm {
        StarTerm::literal(s).expect("lit")
    }
    fn var(name: &str) -> StarTerm {
        StarTerm::variable(name).expect("var")
    }

    fn simple_triple(s: &str, p: &str, o: &str) -> StarTriple {
        StarTriple::new(iri(s), iri(p), iri(o))
    }

    fn quoted_triple(s: &str, p: &str, o: &str) -> StarTerm {
        StarTerm::quoted_triple(simple_triple(s, p, o))
    }

    // -------------------------------------------------------------------
    // EmbeddedTriplePattern unification
    // -------------------------------------------------------------------

    #[test]
    fn test_unify_fully_variable_pattern() {
        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let triple = simple_triple("http://a", "http://b", "http://c");
        let bindings = pattern.unify(&triple).expect("should match");
        assert_eq!(bindings.get("s"), Some(&iri("http://a")));
        assert_eq!(bindings.get("p"), Some(&iri("http://b")));
        assert_eq!(bindings.get("o"), Some(&iri("http://c")));
    }

    #[test]
    fn test_unify_concrete_predicate_match() {
        let pattern = EmbeddedTriplePattern::new(var("s"), iri("http://p"), var("o"));
        let triple = simple_triple("http://a", "http://p", "http://c");
        assert!(pattern.unify(&triple).is_some());
    }

    #[test]
    fn test_unify_concrete_predicate_mismatch() {
        let pattern = EmbeddedTriplePattern::new(var("s"), iri("http://x"), var("o"));
        let triple = simple_triple("http://a", "http://p", "http://c");
        assert!(pattern.unify(&triple).is_none());
    }

    #[test]
    fn test_unify_all_concrete_match() {
        let pattern = EmbeddedTriplePattern::new(iri("http://a"), iri("http://p"), iri("http://c"));
        let triple = simple_triple("http://a", "http://p", "http://c");
        assert!(pattern.unify(&triple).is_some());
    }

    #[test]
    fn test_unify_all_concrete_no_match() {
        let pattern =
            EmbeddedTriplePattern::new(iri("http://a"), iri("http://p"), iri("http://DIFFERENT"));
        let triple = simple_triple("http://a", "http://p", "http://c");
        assert!(pattern.unify(&triple).is_none());
    }

    #[test]
    fn test_unify_literal_object() {
        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), lit("hello"));
        let triple = StarTriple::new(iri("http://a"), iri("http://p"), lit("hello"));
        assert!(pattern.unify(&triple).is_some());
    }

    #[test]
    fn test_unify_literal_mismatch() {
        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), lit("hello"));
        let triple = StarTriple::new(iri("http://a"), iri("http://p"), lit("world"));
        assert!(pattern.unify(&triple).is_none());
    }

    #[test]
    fn test_unify_same_variable_in_subject_and_object_consistent() {
        // ?s p ?s — subject and object must be the same value
        let pattern = EmbeddedTriplePattern::new(var("x"), iri("http://p"), var("x"));
        let same = StarTriple::new(iri("http://a"), iri("http://p"), iri("http://a"));
        assert!(pattern.unify(&same).is_some());

        let different = StarTriple::new(iri("http://a"), iri("http://p"), iri("http://b"));
        assert!(pattern.unify(&different).is_none());
    }

    #[test]
    fn test_embedded_triple_pattern_is_ground() {
        let ground = EmbeddedTriplePattern::new(iri("http://a"), iri("http://p"), iri("http://o"));
        assert!(ground.is_ground());
        assert!(!ground.is_fully_unbound());
    }

    #[test]
    fn test_embedded_triple_pattern_is_fully_unbound() {
        let unbound = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        assert!(unbound.is_fully_unbound());
        assert!(!unbound.is_ground());
    }

    // -------------------------------------------------------------------
    // EmbeddedTriplePattern::unify_embedded
    // -------------------------------------------------------------------

    #[test]
    fn test_unify_embedded_matches_quoted_subject() {
        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let embedded = quoted_triple("http://a", "http://p", "http://c");
        let bindings = pattern.unify_embedded(&embedded).expect("should match");
        assert!(bindings.contains_key("s"));
    }

    #[test]
    fn test_unify_embedded_non_quoted_term_returns_none() {
        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let plain = iri("http://example.org/plain");
        assert!(pattern.unify_embedded(&plain).is_none());
    }

    // -------------------------------------------------------------------
    // SparqlStarEvaluator::evaluate_embedded_pattern
    // -------------------------------------------------------------------

    #[test]
    fn test_evaluate_embedded_pattern_empty_graph() {
        let evaluator = SparqlStarEvaluator::new();
        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let results = evaluator.evaluate_embedded_pattern(&pattern, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_evaluate_embedded_pattern_finds_quoted_subject() {
        let evaluator = SparqlStarEvaluator::new();
        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));

        let inner = simple_triple("http://alice", "http://age", "http://v");
        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://certainty"),
            lit("high"),
        );
        let results = evaluator.evaluate_embedded_pattern(&pattern, &[outer]);
        assert_eq!(results.len(), 1);
        assert!(results[0].contains_key("s"));
    }

    #[test]
    fn test_evaluate_embedded_pattern_with_predicate_filter() {
        let evaluator = SparqlStarEvaluator::new();
        let pattern = EmbeddedTriplePattern::new(var("s"), iri("http://age"), var("o"));

        let matching_inner = simple_triple("http://alice", "http://age", "http://v");
        let non_matching_inner = simple_triple("http://alice", "http://name", "http://v");

        let t1 = StarTriple::new(
            StarTerm::quoted_triple(matching_inner),
            iri("http://cert"),
            lit("hi"),
        );
        let t2 = StarTriple::new(
            StarTerm::quoted_triple(non_matching_inner),
            iri("http://cert"),
            lit("hi"),
        );

        let results = evaluator.evaluate_embedded_pattern(&pattern, &[t1, t2]);
        assert_eq!(results.len(), 1, "only the age triple should match");
    }

    #[test]
    fn test_evaluate_embedded_pattern_multiple_matches() {
        let evaluator = SparqlStarEvaluator::new();
        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));

        let t1 = StarTriple::new(
            quoted_triple("http://a", "http://p", "http://b"),
            iri("http://meta"),
            lit("x"),
        );
        let t2 = StarTriple::new(
            quoted_triple("http://c", "http://p", "http://d"),
            iri("http://meta"),
            lit("y"),
        );
        let results = evaluator.evaluate_embedded_pattern(&pattern, &[t1, t2]);
        assert_eq!(results.len(), 2);
    }

    // -------------------------------------------------------------------
    // AnnotationPattern
    // -------------------------------------------------------------------

    #[test]
    fn test_annotation_pattern_matches_valid_annotation() {
        let pattern = AnnotationPattern::new(
            var("s"),
            iri("http://age"),
            var("age_val"),
            iri("http://certainty"),
            var("cert"),
        );

        let inner = StarTriple::new(iri("http://alice"), iri("http://age"), lit("30"));
        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://certainty"),
            lit("0.9"),
        );

        let bindings = pattern.match_triple(&outer).expect("should match");
        assert_eq!(bindings.get("s"), Some(&iri("http://alice")));
        assert_eq!(bindings.get("age_val"), Some(&lit("30")));
        assert_eq!(bindings.get("cert"), Some(&lit("0.9")));
    }

    #[test]
    fn test_annotation_pattern_no_match_plain_subject() {
        let pattern =
            AnnotationPattern::new(var("s"), var("p"), var("o"), var("ann_p"), var("ann_o"));

        // Outer triple has a plain IRI as subject, not a quoted triple
        let plain_outer = StarTriple::new(iri("http://alice"), iri("http://certainty"), lit("0.9"));

        assert!(pattern.match_triple(&plain_outer).is_none());
    }

    #[test]
    fn test_annotation_pattern_predicate_filter() {
        let pattern = AnnotationPattern::new(
            var("s"),
            var("p"),
            var("o"),
            iri("http://source"), // must match annotation predicate
            var("ann_o"),
        );

        let inner = simple_triple("http://a", "http://p", "http://b");

        // Matching annotation predicate
        let outer_match = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            iri("http://source"),
            lit("db"),
        );
        assert!(pattern.match_triple(&outer_match).is_some());

        // Non-matching annotation predicate
        let outer_no_match = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://certainty"), // different predicate
            lit("db"),
        );
        assert!(pattern.match_triple(&outer_no_match).is_none());
    }

    // -------------------------------------------------------------------
    // SparqlStarEvaluator::evaluate_annotation_pattern
    // -------------------------------------------------------------------

    #[test]
    fn test_evaluate_annotation_pattern_empty_graph() {
        let evaluator = SparqlStarEvaluator::new();
        let pattern = AnnotationPattern::new(var("s"), var("p"), var("o"), var("ap"), var("ao"));
        let results = evaluator.evaluate_annotation_pattern(&pattern, &[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_evaluate_annotation_pattern_finds_annotations() {
        let evaluator = SparqlStarEvaluator::new();
        let pattern = AnnotationPattern::new(
            var("s"),
            var("p"),
            var("o"),
            iri("http://certainty"),
            var("cert"),
        );

        let inner = simple_triple("http://alice", "http://age", "http://v");
        let annotated = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://certainty"),
            lit("high"),
        );

        let results = evaluator.evaluate_annotation_pattern(&pattern, &[annotated]);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].get("cert"), Some(&lit("high")));
    }

    // -------------------------------------------------------------------
    // SparqlStarEvaluator::evaluate_direct_pattern
    // -------------------------------------------------------------------

    #[test]
    fn test_evaluate_direct_pattern_matches_plain_triples() {
        let evaluator = SparqlStarEvaluator::new();
        let pattern = EmbeddedTriplePattern::new(var("s"), iri("http://age"), var("o"));

        let t1 = StarTriple::new(iri("http://alice"), iri("http://age"), lit("30"));
        let t2 = StarTriple::new(iri("http://bob"), iri("http://name"), lit("Bob"));

        let results = evaluator.evaluate_direct_pattern(&pattern, &[t1, t2]);
        assert_eq!(results.len(), 1, "only the age triple matches");
        assert_eq!(results[0].get("s"), Some(&iri("http://alice")));
    }

    #[test]
    fn test_evaluate_direct_pattern_empty_graph() {
        let evaluator = SparqlStarEvaluator::new();
        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let results = evaluator.evaluate_direct_pattern(&pattern, &[]);
        assert!(results.is_empty());
    }
}
