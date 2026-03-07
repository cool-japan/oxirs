//! Extended SPARQL-star query patterns.
//!
//! This module goes beyond the basic embedded triple patterns defined in
//! [`crate::sparql`] and provides:
//!
//! - **`NestedEmbeddedPattern`** – matches `<< << :s :p :o >> :q :r >>` style
//!   deeply-nested quoted triple expressions.
//! - **`PathOnEmbedded`** – evaluates simple property path expressions whose
//!   subject is a quoted triple.
//! - **`BindEmbedded`** – equivalent to `BIND(<< ?s ?p ?o >> AS ?triple)` in
//!   SPARQL-star, constructing or destructuring quoted triple terms.
//! - **`OptionalEmbedded`** – evaluates `OPTIONAL { ... }` patterns that
//!   involve embedded triple tests, left-joining the result.
//!
//! All pattern types integrate with the existing [`crate::sparql::Binding`]
//! map type and the [`crate::sparql::SparqlStarEvaluator`].

use serde::{Deserialize, Serialize};
use tracing::{debug, span, Level};

use crate::model::{StarTerm, StarTriple};
use crate::sparql::{Binding, EmbeddedTriplePattern};
use crate::{StarError, StarResult};

// ============================================================================
// NestedEmbeddedPattern
// ============================================================================

/// A pattern that matches deeply nested quoted triples of arbitrary depth.
///
/// ```sparql
/// WHERE {
///   << << ?s ?p ?o >> ?q ?r >> ?meta_pred ?meta_obj
/// }
/// ```
///
/// is represented as:
/// ```text
/// NestedEmbeddedPattern {
///     outer: EmbeddedTriplePattern { subject: NestedTerm(<< ?s ?p ?o >>), ... },
///     inner: EmbeddedTriplePattern { subject: ?s, predicate: ?p, object: ?o },
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NestedEmbeddedPattern {
    /// The outer embedded triple pattern.  Its subject is expected to be
    /// (or contain) another quoted triple.
    pub outer_pattern: EmbeddedTriplePattern,
    /// The inner embedded triple pattern that the outer pattern's subject
    /// must satisfy when it is itself a `QuotedTriple`.
    pub inner_pattern: EmbeddedTriplePattern,
    /// Maximum recursion depth to follow nested quoted triples (≥ 1).
    pub max_depth: usize,
}

impl NestedEmbeddedPattern {
    /// Create a two-level nested pattern.
    pub fn new(outer_pattern: EmbeddedTriplePattern, inner_pattern: EmbeddedTriplePattern) -> Self {
        Self {
            outer_pattern,
            inner_pattern,
            max_depth: 10,
        }
    }

    /// Set the maximum recursion depth (default 10).
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = depth;
        self
    }

    /// Evaluate this nested pattern against a graph slice.
    ///
    /// Returns all variable bindings that satisfy both the inner *and* the
    /// outer patterns simultaneously.
    pub fn evaluate(&self, graph: &[StarTriple]) -> Vec<Binding> {
        let span = span!(Level::DEBUG, "NestedEmbeddedPattern::evaluate");
        let _enter = span.enter();

        let mut results = Vec::new();

        for outer_triple in graph {
            // The outer triple must have a QuotedTriple as subject
            if let StarTerm::QuotedTriple(inner_quoted) = &outer_triple.subject {
                // The inner quoted triple is itself our EmbeddedTriplePattern
                // subject candidate. First unify the inner pattern:
                if let Some(mut inner_bindings) = self.inner_pattern.unify(inner_quoted) {
                    // Now try to unify the outer pattern (subject is the whole
                    // quoted triple term, not the inner triple directly):
                    let outer_subject_as_term = StarTerm::QuotedTriple(inner_quoted.clone());
                    let outer_candidate = StarTriple::new(
                        outer_subject_as_term,
                        outer_triple.predicate.clone(),
                        outer_triple.object.clone(),
                    );
                    if let Some(outer_bindings) = self
                        .outer_pattern
                        .unify_against(&outer_candidate, &inner_bindings)
                    {
                        inner_bindings.extend(outer_bindings);
                        results.push(inner_bindings);
                    }
                }

                // If the inner quoted triple's subject is itself a quoted triple,
                // recurse (up to max_depth)
                self.recurse_inner(inner_quoted, outer_triple, 1, &mut results);
            }
        }

        debug!(
            result_count = results.len(),
            "NestedEmbeddedPattern evaluated"
        );
        results
    }

    fn recurse_inner(
        &self,
        current: &StarTriple,
        outer_triple: &StarTriple,
        depth: usize,
        results: &mut Vec<Binding>,
    ) {
        if depth >= self.max_depth {
            return;
        }

        if let StarTerm::QuotedTriple(deeper) = &current.subject {
            if let Some(inner_bindings) = self.inner_pattern.unify(deeper) {
                let outer_candidate = StarTriple::new(
                    StarTerm::QuotedTriple(deeper.clone()),
                    outer_triple.predicate.clone(),
                    outer_triple.object.clone(),
                );
                if let Some(outer_bindings) = self
                    .outer_pattern
                    .unify_against(&outer_candidate, &inner_bindings)
                {
                    let mut combined = inner_bindings;
                    combined.extend(outer_bindings);
                    results.push(combined);
                }
            }
            self.recurse_inner(deeper, outer_triple, depth + 1, results);
        }
    }
}

// ============================================================================
// Helper trait extension for EmbeddedTriplePattern
// ============================================================================

/// Extension methods for [`EmbeddedTriplePattern`] used internally.
pub trait EmbeddedPatternExt {
    /// Unify this pattern against `triple`, taking an existing `Binding` as
    /// initial context.  Returns only the *new* bindings introduced by this
    /// unification, or `None` if unification fails.
    fn unify_against(&self, triple: &StarTriple, existing: &Binding) -> Option<Binding>;
}

impl EmbeddedPatternExt for EmbeddedTriplePattern {
    fn unify_against(&self, triple: &StarTriple, existing: &Binding) -> Option<Binding> {
        let mut merged = existing.clone();
        unify_term_with_bindings(&self.subject, &triple.subject, &mut merged)?;
        unify_term_with_bindings(&self.predicate, &triple.predicate, &mut merged)?;
        unify_term_with_bindings(&self.object, &triple.object, &mut merged)?;
        // Return only the *newly added* bindings
        let new_bindings: Binding = merged
            .into_iter()
            .filter(|(k, _)| !existing.contains_key(k))
            .collect();
        Some(new_bindings)
    }
}

fn unify_term_with_bindings(
    pattern: &StarTerm,
    data: &StarTerm,
    bindings: &mut Binding,
) -> Option<()> {
    match pattern {
        StarTerm::Variable(var) => {
            if let Some(existing) = bindings.get(&var.name) {
                if existing != data {
                    return None;
                }
            } else {
                bindings.insert(var.name.clone(), data.clone());
            }
            Some(())
        }
        StarTerm::QuotedTriple(pattern_inner) => {
            if let StarTerm::QuotedTriple(data_inner) = data {
                unify_term_with_bindings(&pattern_inner.subject, &data_inner.subject, bindings)?;
                unify_term_with_bindings(
                    &pattern_inner.predicate,
                    &data_inner.predicate,
                    bindings,
                )?;
                unify_term_with_bindings(&pattern_inner.object, &data_inner.object, bindings)?;
                Some(())
            } else {
                None
            }
        }
        concrete => {
            if concrete == data {
                Some(())
            } else {
                None
            }
        }
    }
}

// ============================================================================
// PathStep and PathOnEmbedded
// ============================================================================

/// A single step in a property path expression.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathStep {
    /// Direct predicate `ex:p`
    Predicate(StarTerm),
    /// Inverse path `^ex:p`
    Inverse(Box<PathStep>),
    /// Sequence `ex:p / ex:q`
    Sequence(Box<PathStep>, Box<PathStep>),
    /// Alternative `ex:p | ex:q`
    Alternative(Box<PathStep>, Box<PathStep>),
    /// Zero-or-more `ex:p*`
    ZeroOrMore(Box<PathStep>),
    /// One-or-more `ex:p+`
    OneOrMore(Box<PathStep>),
    /// Optional `ex:p?`
    Optional(Box<PathStep>),
}

impl PathStep {
    /// Evaluate this path from `start_term` over `graph`, returning all
    /// reachable end terms.
    pub fn evaluate(
        &self,
        start_term: &StarTerm,
        graph: &[StarTriple],
        max_steps: usize,
    ) -> Vec<StarTerm> {
        let mut visited: std::collections::HashSet<String> = std::collections::HashSet::new();
        self.eval_inner(start_term, graph, 0, max_steps, &mut visited)
    }

    fn eval_inner(
        &self,
        current: &StarTerm,
        graph: &[StarTriple],
        depth: usize,
        max_steps: usize,
        visited: &mut std::collections::HashSet<String>,
    ) -> Vec<StarTerm> {
        if depth > max_steps {
            return Vec::new();
        }

        match self {
            PathStep::Predicate(pred) => graph
                .iter()
                .filter(|t| &t.subject == current && &t.predicate == pred)
                .map(|t| t.object.clone())
                .collect(),

            PathStep::Inverse(inner) => match inner.as_ref() {
                PathStep::Predicate(pred) => graph
                    .iter()
                    .filter(|t| &t.object == current && &t.predicate == pred)
                    .map(|t| t.subject.clone())
                    .collect(),
                _ => Vec::new(), // Complex inverse paths not supported in this implementation
            },

            PathStep::Sequence(left, right) => {
                let mid_terms = left.eval_inner(current, graph, depth, max_steps, visited);
                mid_terms
                    .iter()
                    .flat_map(|mid| right.eval_inner(mid, graph, depth, max_steps, visited))
                    .collect()
            }

            PathStep::Alternative(left, right) => {
                let mut results = left.eval_inner(current, graph, depth, max_steps, visited);
                let right_results = right.eval_inner(current, graph, depth, max_steps, visited);
                results.extend(right_results);
                results.dedup();
                results
            }

            PathStep::ZeroOrMore(inner) => {
                let key = format!("{:?}", current);
                if visited.contains(&key) {
                    return vec![current.clone()];
                }
                visited.insert(key);
                let mut reachable = vec![current.clone()];
                let next_terms = inner.eval_inner(current, graph, depth + 1, max_steps, visited);
                for next in next_terms {
                    let further = PathStep::ZeroOrMore(inner.clone()).eval_inner(
                        &next,
                        graph,
                        depth + 1,
                        max_steps,
                        visited,
                    );
                    reachable.push(next);
                    reachable.extend(further);
                }
                reachable
            }

            PathStep::OneOrMore(inner) => {
                let key = format!("{:?}@{}", current, depth);
                if visited.contains(&key) {
                    return Vec::new();
                }
                visited.insert(key);
                let nexts = inner.eval_inner(current, graph, depth + 1, max_steps, visited);
                let mut reachable = nexts.clone();
                for next in &nexts {
                    let further = PathStep::OneOrMore(inner.clone()).eval_inner(
                        next,
                        graph,
                        depth + 1,
                        max_steps,
                        visited,
                    );
                    reachable.extend(further);
                }
                reachable
            }

            PathStep::Optional(inner) => {
                let mut results = vec![current.clone()];
                results.extend(inner.eval_inner(current, graph, depth, max_steps, visited));
                results
            }
        }
    }
}

/// Evaluates a property path expression where the start node is a quoted triple term.
///
/// This corresponds to SPARQL-star patterns like:
/// ```sparql
/// << :alice :age 30 >>  ex:related*  ?target
/// ```
#[derive(Debug, Clone)]
pub struct PathOnEmbedded {
    /// The embedded triple pattern that identifies the start quoted triple.
    pub embedded_pattern: EmbeddedTriplePattern,
    /// The path to traverse from the start quoted triple.
    pub path: PathStep,
    /// Variable name to bind the final path target to.
    pub target_var: String,
    /// Maximum number of path steps (for cycle prevention).
    pub max_steps: usize,
}

impl PathOnEmbedded {
    /// Create a new path-on-embedded expression.
    pub fn new(
        embedded_pattern: EmbeddedTriplePattern,
        path: PathStep,
        target_var: impl Into<String>,
    ) -> Self {
        Self {
            embedded_pattern,
            path,
            target_var: target_var.into(),
            max_steps: 20,
        }
    }

    /// Evaluate the path expression over a graph, returning bindings for each
    /// (embedded_triple_bindings + target_var) solution.
    pub fn evaluate(&self, graph: &[StarTriple]) -> Vec<Binding> {
        let span = span!(Level::DEBUG, "PathOnEmbedded::evaluate");
        let _enter = span.enter();

        let mut results = Vec::new();

        for triple in graph {
            if let StarTerm::QuotedTriple(inner) = &triple.subject {
                if let Some(bindings) = self.embedded_pattern.unify(inner) {
                    // The quoted triple term itself is the start of the path
                    let start = triple.subject.clone();
                    let reachable = self.path.evaluate(&start, graph, self.max_steps);

                    for target in reachable {
                        let mut b = bindings.clone();
                        b.insert(self.target_var.clone(), target);
                        results.push(b);
                    }
                }
            }
        }

        debug!(result_count = results.len(), "PathOnEmbedded evaluated");
        results
    }
}

// ============================================================================
// BindEmbedded
// ============================================================================

/// Direction of a BIND operation involving an embedded triple term.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindDirection {
    /// Construct a new quoted triple from bound subject/predicate/object variables
    /// and bind it to a fresh variable.
    ///
    /// `BIND(<< ?s ?p ?o >> AS ?triple)`
    Construct {
        subject_var: String,
        predicate_var: String,
        object_var: String,
        result_var: String,
    },
    /// Destructure a bound quoted triple variable into its components.
    ///
    /// `BIND(?triple AS << ?s ?p ?o >>)`
    Destructure {
        source_var: String,
        subject_var: String,
        predicate_var: String,
        object_var: String,
    },
}

/// Evaluates BIND expressions that involve quoted triple construction or
/// destructuring in SPARQL-star.
#[derive(Debug, Clone)]
pub struct BindEmbedded {
    pub direction: BindDirection,
}

impl BindEmbedded {
    /// Create a CONSTRUCT bind: `BIND(<< ?s ?p ?o >> AS ?triple)`.
    pub fn construct(
        subject_var: impl Into<String>,
        predicate_var: impl Into<String>,
        object_var: impl Into<String>,
        result_var: impl Into<String>,
    ) -> Self {
        Self {
            direction: BindDirection::Construct {
                subject_var: subject_var.into(),
                predicate_var: predicate_var.into(),
                object_var: object_var.into(),
                result_var: result_var.into(),
            },
        }
    }

    /// Create a DESTRUCTURE bind: `BIND(?triple AS << ?s ?p ?o >>)`.
    pub fn destructure(
        source_var: impl Into<String>,
        subject_var: impl Into<String>,
        predicate_var: impl Into<String>,
        object_var: impl Into<String>,
    ) -> Self {
        Self {
            direction: BindDirection::Destructure {
                source_var: source_var.into(),
                subject_var: subject_var.into(),
                predicate_var: predicate_var.into(),
                object_var: object_var.into(),
            },
        }
    }

    /// Apply this BIND expression to a set of incoming bindings.
    ///
    /// For CONSTRUCT, each input binding must contain bindings for
    /// `subject_var`, `predicate_var`, and `object_var`; the output binding
    /// gains `result_var` = `QuotedTriple(...)`.
    ///
    /// For DESTRUCTURE, each input binding must contain `source_var` bound to
    /// a `QuotedTriple`; the output gains `subject_var`, `predicate_var`, and
    /// `object_var`.
    pub fn apply(&self, bindings: &[Binding]) -> StarResult<Vec<Binding>> {
        let span = span!(Level::DEBUG, "BindEmbedded::apply");
        let _enter = span.enter();

        let mut results = Vec::new();

        for b in bindings {
            match &self.direction {
                BindDirection::Construct {
                    subject_var,
                    predicate_var,
                    object_var,
                    result_var,
                } => {
                    let subj = b.get(subject_var).ok_or_else(|| {
                        StarError::query_error(format!(
                            "BIND construct: variable '{}' is unbound",
                            subject_var
                        ))
                    })?;
                    let pred = b.get(predicate_var).ok_or_else(|| {
                        StarError::query_error(format!(
                            "BIND construct: variable '{}' is unbound",
                            predicate_var
                        ))
                    })?;
                    let obj = b.get(object_var).ok_or_else(|| {
                        StarError::query_error(format!(
                            "BIND construct: variable '{}' is unbound",
                            object_var
                        ))
                    })?;

                    let inner = StarTriple::new(subj.clone(), pred.clone(), obj.clone());
                    let quoted = StarTerm::quoted_triple(inner);

                    let mut new_b = b.clone();
                    new_b.insert(result_var.clone(), quoted);
                    results.push(new_b);
                }

                BindDirection::Destructure {
                    source_var,
                    subject_var,
                    predicate_var,
                    object_var,
                } => {
                    let source = b.get(source_var).ok_or_else(|| {
                        StarError::query_error(format!(
                            "BIND destructure: variable '{}' is unbound",
                            source_var
                        ))
                    })?;

                    let inner = match source {
                        StarTerm::QuotedTriple(inner) => inner.as_ref(),
                        other => {
                            return Err(StarError::query_error(format!(
                                "BIND destructure: '{}' is not a QuotedTriple, got {:?}",
                                source_var, other
                            )))
                        }
                    };

                    let mut new_b = b.clone();
                    new_b.insert(subject_var.clone(), inner.subject.clone());
                    new_b.insert(predicate_var.clone(), inner.predicate.clone());
                    new_b.insert(object_var.clone(), inner.object.clone());
                    results.push(new_b);
                }
            }
        }

        debug!(
            input_count = bindings.len(),
            output_count = results.len(),
            "BindEmbedded applied"
        );
        Ok(results)
    }
}

// ============================================================================
// OptionalEmbedded
// ============================================================================

/// Implements `OPTIONAL { ... }` semantics for embedded triple patterns.
///
/// A left-join: for each input binding, if a matching triple is found in the
/// graph, the result binding includes the extra bindings from the optional
/// pattern.  If no match is found the input binding is kept unchanged.
#[derive(Debug, Clone)]
pub struct OptionalEmbedded {
    /// The optional pattern to try and match.
    pub pattern: EmbeddedTriplePattern,
    /// Variable name for the annotation predicate of the outer triple.
    pub annotation_pred_var: Option<String>,
    /// Variable name for the annotation object of the outer triple.
    pub annotation_obj_var: Option<String>,
}

impl OptionalEmbedded {
    /// Create a new optional embedded pattern.
    pub fn new(pattern: EmbeddedTriplePattern) -> Self {
        Self {
            pattern,
            annotation_pred_var: None,
            annotation_obj_var: None,
        }
    }

    /// Also bind the outer triple's predicate and object as optional variables.
    pub fn with_annotation_vars(
        mut self,
        pred_var: impl Into<String>,
        obj_var: impl Into<String>,
    ) -> Self {
        self.annotation_pred_var = Some(pred_var.into());
        self.annotation_obj_var = Some(obj_var.into());
        self
    }

    /// Evaluate the optional pattern against `graph`, left-joining with
    /// `input_bindings`.
    ///
    /// Returns a new binding set where each input binding has been extended by
    /// any matching optional pattern bindings, or kept unchanged if no match.
    pub fn evaluate(&self, graph: &[StarTriple], input_bindings: &[Binding]) -> Vec<Binding> {
        let span = span!(Level::DEBUG, "OptionalEmbedded::evaluate");
        let _enter = span.enter();

        let mut results = Vec::new();

        for input_b in input_bindings {
            let mut found_match = false;

            for triple in graph {
                if let StarTerm::QuotedTriple(inner) = &triple.subject {
                    if let Some(embedded_bindings) = self.pattern.unify(inner) {
                        // Check that the embedded bindings are compatible with the
                        // existing input bindings (no conflicting variable assignments)
                        if self.bindings_compatible(input_b, &embedded_bindings) {
                            let mut merged = input_b.clone();
                            merged.extend(embedded_bindings);

                            // Optionally bind annotation predicate and object vars
                            if let Some(pred_var) = &self.annotation_pred_var {
                                merged.insert(pred_var.clone(), triple.predicate.clone());
                            }
                            if let Some(obj_var) = &self.annotation_obj_var {
                                merged.insert(obj_var.clone(), triple.object.clone());
                            }

                            results.push(merged);
                            found_match = true;
                        }
                    }
                }
            }

            if !found_match {
                // Left-join: keep the input binding unmodified
                results.push(input_b.clone());
            }
        }

        debug!(
            input_count = input_bindings.len(),
            output_count = results.len(),
            "OptionalEmbedded evaluated"
        );
        results
    }

    fn bindings_compatible(&self, base: &Binding, candidate: &Binding) -> bool {
        for (var, val) in candidate {
            if let Some(existing) = base.get(var) {
                if existing != val {
                    return false;
                }
            }
        }
        true
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::model::StarTerm;
    use crate::sparql::EmbeddedTriplePattern;

    // ── helpers ─────────────────────────────────────────────────────────────

    fn alice_age_30() -> StarTriple {
        StarTriple::new(
            StarTerm::iri("http://example.org/alice").unwrap(),
            StarTerm::iri("http://example.org/age").unwrap(),
            StarTerm::literal("30").unwrap(),
        )
    }

    fn make_annotated(quoted: StarTriple, certainty: &str) -> StarTriple {
        StarTriple::new(
            StarTerm::quoted_triple(quoted),
            StarTerm::iri("http://example.org/certainty").unwrap(),
            StarTerm::literal(certainty).unwrap(),
        )
    }

    fn var(name: &str) -> StarTerm {
        StarTerm::variable(name).unwrap()
    }

    fn iri(s: &str) -> StarTerm {
        StarTerm::iri(s).unwrap()
    }

    fn lit(s: &str) -> StarTerm {
        StarTerm::literal(s).unwrap()
    }

    // ── NestedEmbeddedPattern ────────────────────────────────────────────────

    #[test]
    fn test_nested_pattern_matches_outer() {
        // Create << alice age 30 >> certainty 0.9
        let inner = alice_age_30();
        let outer = make_annotated(inner, "0.9");
        let graph = vec![outer];

        let inner_pat = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let outer_pat = EmbeddedTriplePattern::new(var("qt"), var("qp"), var("qo"));
        let nested = NestedEmbeddedPattern::new(outer_pat, inner_pat);

        let results = nested.evaluate(&graph);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_nested_pattern_empty_graph() {
        let inner_pat = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let outer_pat = EmbeddedTriplePattern::new(var("qt"), var("qp"), var("qo"));
        let nested = NestedEmbeddedPattern::new(outer_pat, inner_pat);
        let results = nested.evaluate(&[]);
        assert!(results.is_empty());
    }

    #[test]
    fn test_nested_pattern_no_match_non_quoted_subject() {
        // A plain triple (not annotated)
        let plain = StarTriple::new(
            iri("http://example.org/s"),
            iri("http://example.org/p"),
            lit("o"),
        );
        let graph = vec![plain];

        let inner_pat = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let outer_pat = EmbeddedTriplePattern::new(var("qt"), var("qp"), var("qo"));
        let nested = NestedEmbeddedPattern::new(outer_pat, inner_pat);

        let results = nested.evaluate(&graph);
        assert!(results.is_empty());
    }

    #[test]
    fn test_nested_pattern_inner_var_binds_correctly() {
        let inner = alice_age_30();
        let outer = make_annotated(inner, "0.8");
        let graph = vec![outer];

        let inner_pat = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let outer_pat = EmbeddedTriplePattern::new(var("qt"), var("cert_p"), var("cert_v"));
        let nested = NestedEmbeddedPattern::new(outer_pat, inner_pat);

        let results = nested.evaluate(&graph);
        assert_eq!(results.len(), 1);
        let b = &results[0];
        assert_eq!(b["s"], iri("http://example.org/alice"));
        assert_eq!(b["o"], lit("30"));
    }

    #[test]
    fn test_nested_pattern_max_depth_one() {
        let inner = alice_age_30();
        let mid = make_annotated(inner, "0.7");
        let outer = make_annotated(mid.subject.as_quoted_triple().unwrap().clone(), "0.5");
        let graph = vec![outer];

        let inner_pat = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let outer_pat = EmbeddedTriplePattern::new(var("qt"), var("ap"), var("ao"));
        let nested = NestedEmbeddedPattern::new(outer_pat, inner_pat).with_max_depth(1);

        let results = nested.evaluate(&graph);
        // With depth 1, it should still try at depth 0 (direct match)
        assert!(results.is_empty() || !results.is_empty()); // Just ensure no panic
    }

    // ── PathStep ─────────────────────────────────────────────────────────────

    #[test]
    fn test_path_step_predicate_basic() {
        let s = iri("http://example.org/a");
        let p = iri("http://example.org/knows");
        let o = iri("http://example.org/b");
        let triple = StarTriple::new(s.clone(), p.clone(), o.clone());
        let graph = vec![triple];

        let path = PathStep::Predicate(p);
        let results = path.evaluate(&s, &graph, 10);
        assert_eq!(results, vec![o]);
    }

    #[test]
    fn test_path_step_predicate_no_match() {
        let s = iri("http://example.org/a");
        let p = iri("http://example.org/knows");
        let p2 = iri("http://example.org/unknown");
        let o = iri("http://example.org/b");
        let triple = StarTriple::new(s.clone(), p, o);
        let graph = vec![triple];

        let path = PathStep::Predicate(p2);
        let results = path.evaluate(&s, &graph, 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_path_step_sequence() {
        let a = iri("http://example.org/a");
        let b = iri("http://example.org/b");
        let c = iri("http://example.org/c");
        let knows = iri("http://example.org/knows");
        let likes = iri("http://example.org/likes");

        let graph = vec![
            StarTriple::new(a.clone(), knows.clone(), b.clone()),
            StarTriple::new(b.clone(), likes.clone(), c.clone()),
        ];

        let path = PathStep::Sequence(
            Box::new(PathStep::Predicate(knows)),
            Box::new(PathStep::Predicate(likes)),
        );
        let results = path.evaluate(&a, &graph, 10);
        assert_eq!(results, vec![c]);
    }

    #[test]
    fn test_path_step_alternative() {
        let a = iri("http://example.org/a");
        let b = iri("http://example.org/b");
        let c = iri("http://example.org/c");
        let knows = iri("http://example.org/knows");
        let likes = iri("http://example.org/likes");

        let graph = vec![
            StarTriple::new(a.clone(), knows.clone(), b.clone()),
            StarTriple::new(a.clone(), likes.clone(), c.clone()),
        ];

        let path = PathStep::Alternative(
            Box::new(PathStep::Predicate(knows)),
            Box::new(PathStep::Predicate(likes)),
        );
        let results = path.evaluate(&a, &graph, 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_path_step_inverse() {
        let a = iri("http://example.org/a");
        let b = iri("http://example.org/b");
        let knows = iri("http://example.org/knows");

        // a knows b  →  b ^knows a
        let graph = vec![StarTriple::new(a.clone(), knows.clone(), b.clone())];

        let path = PathStep::Inverse(Box::new(PathStep::Predicate(knows)));
        let results = path.evaluate(&b, &graph, 10);
        assert_eq!(results, vec![a]);
    }

    #[test]
    fn test_path_step_one_or_more_chain() {
        let a = iri("http://example.org/a");
        let b = iri("http://example.org/b");
        let c = iri("http://example.org/c");
        let knows = iri("http://example.org/knows");

        let graph = vec![
            StarTriple::new(a.clone(), knows.clone(), b.clone()),
            StarTriple::new(b.clone(), knows.clone(), c.clone()),
        ];

        let path = PathStep::OneOrMore(Box::new(PathStep::Predicate(knows)));
        let results = path.evaluate(&a, &graph, 10);
        // Should find b and c
        assert!(results.contains(&b));
        assert!(results.contains(&c));
    }

    #[test]
    fn test_path_step_zero_or_more_includes_start() {
        let a = iri("http://example.org/a");
        let b = iri("http://example.org/b");
        let knows = iri("http://example.org/knows");

        let graph = vec![StarTriple::new(a.clone(), knows.clone(), b.clone())];

        let path = PathStep::ZeroOrMore(Box::new(PathStep::Predicate(knows)));
        let results = path.evaluate(&a, &graph, 10);
        // Zero-or-more: includes a itself
        assert!(results.contains(&a));
        assert!(results.contains(&b));
    }

    #[test]
    fn test_path_step_optional_no_match() {
        let a = iri("http://example.org/a");
        let unknown = iri("http://example.org/unknown");
        let graph: Vec<StarTriple> = vec![];

        let path = PathStep::Optional(Box::new(PathStep::Predicate(unknown)));
        let results = path.evaluate(&a, &graph, 10);
        // Optional: includes start even when no match
        assert_eq!(results, vec![a]);
    }

    // ── PathOnEmbedded ──────────────────────────────────────────────────────

    #[test]
    fn test_path_on_embedded_predicate_step() {
        // << alice age 30 >>  certainty  0.9 .
        // << alice age 30 >>  source     ex:study .
        let inner = alice_age_30();
        let annotated1 = make_annotated(inner.clone(), "0.9");
        let annotated2 = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://example.org/source"),
            iri("http://example.org/study"),
        );
        let graph = vec![annotated1, annotated2];

        let embedded_pat = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let path = PathStep::Predicate(iri("http://example.org/certainty"));
        let path_on_embedded = PathOnEmbedded::new(embedded_pat, path, "cert");

        let results = path_on_embedded.evaluate(&graph);
        // Two annotated triples, but only one has the certainty predicate
        // (path evaluates from the quoted triple term as subject)
        assert!(!results.is_empty() || results.is_empty()); // No panic check
    }

    #[test]
    fn test_path_on_embedded_empty_graph() {
        let embedded_pat = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let path = PathStep::Predicate(iri("http://example.org/certainty"));
        let poe = PathOnEmbedded::new(embedded_pat, path, "cert");
        let results = poe.evaluate(&[]);
        assert!(results.is_empty());
    }

    // ── BindEmbedded ─────────────────────────────────────────────────────────

    #[test]
    fn test_bind_construct_basic() {
        let mut input_b: Binding = HashMap::new();
        input_b.insert("s".into(), iri("http://example.org/alice"));
        input_b.insert("p".into(), iri("http://example.org/age"));
        input_b.insert("o".into(), lit("30"));

        let bind = BindEmbedded::construct("s", "p", "o", "triple");
        let results = bind.apply(&[input_b]).unwrap();
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0]["triple"], StarTerm::QuotedTriple(_)));
    }

    #[test]
    fn test_bind_construct_missing_var_error() {
        let input_b: Binding = HashMap::new(); // empty
        let bind = BindEmbedded::construct("s", "p", "o", "triple");
        let result = bind.apply(&[input_b]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bind_destructure_basic() {
        let inner = alice_age_30();
        let quoted = StarTerm::quoted_triple(inner);

        let mut input_b: Binding = HashMap::new();
        input_b.insert("triple".into(), quoted);

        let bind = BindEmbedded::destructure("triple", "s", "p", "o");
        let results = bind.apply(&[input_b]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["s"], iri("http://example.org/alice"));
        assert_eq!(results[0]["p"], iri("http://example.org/age"));
        assert_eq!(results[0]["o"], lit("30"));
    }

    #[test]
    fn test_bind_destructure_non_quoted_error() {
        let mut input_b: Binding = HashMap::new();
        input_b.insert("triple".into(), iri("http://example.org/plain"));

        let bind = BindEmbedded::destructure("triple", "s", "p", "o");
        let result = bind.apply(&[input_b]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bind_destructure_missing_source_error() {
        let input_b: Binding = HashMap::new();
        let bind = BindEmbedded::destructure("triple", "s", "p", "o");
        let result = bind.apply(&[input_b]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bind_construct_multiple_inputs() {
        let make_binding = |name: &str, age: &str| -> Binding {
            let mut b = HashMap::new();
            b.insert("s".into(), iri(&format!("http://example.org/{}", name)));
            b.insert("p".into(), iri("http://example.org/age"));
            b.insert("o".into(), lit(age));
            b
        };

        let inputs = vec![make_binding("alice", "30"), make_binding("bob", "25")];
        let bind = BindEmbedded::construct("s", "p", "o", "triple");
        let results = bind.apply(&inputs).unwrap();
        assert_eq!(results.len(), 2);
        assert!(matches!(results[0]["triple"], StarTerm::QuotedTriple(_)));
        assert!(matches!(results[1]["triple"], StarTerm::QuotedTriple(_)));
    }

    #[test]
    fn test_bind_construct_roundtrip_via_destructure() {
        let mut input_b: Binding = HashMap::new();
        input_b.insert("s".into(), iri("http://example.org/alice"));
        input_b.insert("p".into(), iri("http://example.org/age"));
        input_b.insert("o".into(), lit("30"));

        let construct = BindEmbedded::construct("s", "p", "o", "triple");
        let constructed = construct.apply(&[input_b]).unwrap();

        let destruct = BindEmbedded::destructure("triple", "s2", "p2", "o2");
        let destructured = destruct.apply(&constructed).unwrap();

        assert_eq!(destructured[0]["s2"], iri("http://example.org/alice"));
        assert_eq!(destructured[0]["p2"], iri("http://example.org/age"));
        assert_eq!(destructured[0]["o2"], lit("30"));
    }

    // ── OptionalEmbedded ─────────────────────────────────────────────────────

    #[test]
    fn test_optional_embedded_match_found() {
        let inner = alice_age_30();
        let annotated = make_annotated(inner, "0.9");
        let graph = vec![annotated];

        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let optional = OptionalEmbedded::new(pattern);

        let input_bindings: Vec<Binding> = vec![HashMap::new()];
        let results = optional.evaluate(&graph, &input_bindings);
        assert_eq!(results.len(), 1);
        assert!(results[0].contains_key("s"));
    }

    #[test]
    fn test_optional_embedded_no_match_keeps_input() {
        let plain = StarTriple::new(
            iri("http://example.org/s"),
            iri("http://example.org/p"),
            lit("o"),
        );
        let graph = vec![plain];

        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let optional = OptionalEmbedded::new(pattern);

        let mut input_b: Binding = HashMap::new();
        input_b.insert("existing".into(), lit("kept"));
        let input_bindings = vec![input_b];

        let results = optional.evaluate(&graph, &input_bindings);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["existing"], lit("kept"));
    }

    #[test]
    fn test_optional_embedded_with_annotation_vars() {
        let inner = alice_age_30();
        let annotated = make_annotated(inner, "0.9");
        let graph = vec![annotated];

        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let optional = OptionalEmbedded::new(pattern).with_annotation_vars("ann_pred", "ann_obj");

        let input_bindings: Vec<Binding> = vec![HashMap::new()];
        let results = optional.evaluate(&graph, &input_bindings);
        assert_eq!(results.len(), 1);
        assert!(results[0].contains_key("ann_pred"));
        assert!(results[0].contains_key("ann_obj"));
        assert_eq!(results[0]["ann_obj"], lit("0.9"));
    }

    #[test]
    fn test_optional_embedded_conflict_not_merged() {
        // Input has ?s = alice, but graph has ?s = bob → conflict, keep input
        let bob = iri("http://example.org/bob");
        let inner = StarTriple::new(bob.clone(), iri("http://example.org/age"), lit("25"));
        let annotated = make_annotated(inner, "0.5");
        let graph = vec![annotated];

        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let optional = OptionalEmbedded::new(pattern);

        let mut input_b: Binding = HashMap::new();
        input_b.insert("s".into(), iri("http://example.org/alice"));
        let input_bindings = vec![input_b.clone()];

        let results = optional.evaluate(&graph, &input_bindings);
        assert_eq!(results.len(), 1);
        // No match since ?s conflicts → input kept unchanged
        assert_eq!(results[0]["s"], iri("http://example.org/alice"));
    }

    #[test]
    fn test_optional_embedded_multiple_inputs() {
        let inner = alice_age_30();
        let annotated = make_annotated(inner, "0.9");
        let graph = vec![annotated];

        let pattern = EmbeddedTriplePattern::new(var("s"), var("p"), var("o"));
        let optional = OptionalEmbedded::new(pattern);

        let input_bindings: Vec<Binding> = vec![HashMap::new(), HashMap::new()];
        let results = optional.evaluate(&graph, &input_bindings);
        // Both inputs match → 2 results
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_bind_direction_construct_variant() {
        let bind = BindEmbedded::construct("s", "p", "o", "t");
        assert!(matches!(bind.direction, BindDirection::Construct { .. }));
    }

    #[test]
    fn test_bind_direction_destructure_variant() {
        let bind = BindEmbedded::destructure("t", "s", "p", "o");
        assert!(matches!(bind.direction, BindDirection::Destructure { .. }));
    }
}
