//! Parallel BGP (Basic Graph Pattern) evaluator for SPARQL-star queries.
//!
//! This module provides high-performance, parallel evaluation of RDF-star triple
//! patterns. It leverages Rayon for data-parallel matching and implements
//! hash-join and nested-loop join strategies based on result-set cardinality.
//!
//! # Design
//!
//! - Independent patterns are matched in parallel across the store.
//! - Patterns sharing variables are joined after individual matching.
//! - Small result sets use nested-loop join; larger sets use hash join.
//!
//! # Example
//!
//! ```rust,ignore
//! use oxirs_star::query::parallel::{ParallelStarBgpEvaluator, StarBgpPattern, StarBgpTerm, StarBgpTriple};
//! use oxirs_star::query::parallel::InMemoryStarStore;
//!
//! let store = InMemoryStarStore::new(vec![]);
//! let evaluator = ParallelStarBgpEvaluator::new(4);
//! let results = evaluator.evaluate(&[], &store).unwrap();
//! assert!(results.is_empty());
//! ```

use rayon::prelude::*;
use std::collections::HashMap;

use crate::{StarResult, StarTerm, StarTriple};

// ---- Public types -----------------------------------------------------------

/// A binding row mapping variable names to concrete RDF-star terms.
pub type StarBinding = HashMap<String, StarTerm>;

/// A triple pattern element: either a concrete term or a variable placeholder.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StarBgpTerm {
    /// Concrete RDF-star term (IRI, literal, blank node, quoted triple).
    Concrete(StarTerm),
    /// SPARQL variable (name without leading `?`).
    Variable(String),
}

impl StarBgpTerm {
    /// Convenience constructor for a variable.
    pub fn var(name: impl Into<String>) -> Self {
        Self::Variable(name.into())
    }

    /// Convenience constructor for a concrete term.
    pub fn term(t: StarTerm) -> Self {
        Self::Concrete(t)
    }

    /// Return the variable name if this is a variable, otherwise `None`.
    pub fn variable_name(&self) -> Option<&str> {
        match self {
            Self::Variable(name) => Some(name.as_str()),
            Self::Concrete(_) => None,
        }
    }
}

/// A triple pattern where each position is either concrete or a variable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StarBgpTriple {
    pub subject: StarBgpTerm,
    pub predicate: StarBgpTerm,
    pub object: StarBgpTerm,
}

impl StarBgpTriple {
    /// Construct a new triple pattern.
    pub fn new(subject: StarBgpTerm, predicate: StarBgpTerm, object: StarBgpTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }

    /// Collect all variable names referenced by this pattern.
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::with_capacity(3);
        if let Some(v) = self.subject.variable_name() {
            vars.push(v.to_owned());
        }
        if let Some(v) = self.predicate.variable_name() {
            vars.push(v.to_owned());
        }
        if let Some(v) = self.object.variable_name() {
            vars.push(v.to_owned());
        }
        vars
    }
}

// ---- Store trait ------------------------------------------------------------

/// Trait that a triple store must implement for parallel BGP evaluation.
///
/// Implementations are expected to be `Send + Sync` so they can be shared
/// across Rayon worker threads.
pub trait StarTripleStore: Send + Sync {
    /// Return all bindings that satisfy `pattern` given the optional seed
    /// bindings.  Implementations may use the seed to narrow the search space.
    fn match_pattern(
        &self,
        pattern: &StarBgpTriple,
        seed: Option<&StarBinding>,
    ) -> StarResult<Vec<StarBinding>>;

    /// Return the total number of triples in the store (used for cost
    /// estimation).
    fn triple_count(&self) -> usize;
}

// ---- Evaluator --------------------------------------------------------------

/// Parallel evaluator for SPARQL-star Basic Graph Patterns (BGPs).
///
/// Evaluates a set of triple patterns against a store, applying hash-join or
/// nested-loop join depending on result cardinality.
pub struct ParallelStarBgpEvaluator {
    /// Degree of parallelism to apply (number of Rayon threads).
    parallelism: usize,
    /// Threshold (in bindings) below which nested-loop join is preferred.
    nested_loop_threshold: usize,
}

impl ParallelStarBgpEvaluator {
    /// Create an evaluator with the given parallelism degree.
    ///
    /// `parallelism` controls the Rayon thread pool size.  Use
    /// `num_cpus::get()` for the full machine parallelism.
    pub fn new(parallelism: usize) -> Self {
        Self {
            parallelism: parallelism.max(1),
            nested_loop_threshold: 1024,
        }
    }

    /// Set the threshold for nested-loop vs hash join selection.
    pub fn with_nested_loop_threshold(mut self, threshold: usize) -> Self {
        self.nested_loop_threshold = threshold;
        self
    }

    /// Evaluate a list of star triple patterns against the store and return
    /// all satisfying variable bindings.
    ///
    /// Independent patterns (those that share no variables with already-
    /// processed results) are matched in parallel, then joined lazily.
    pub fn evaluate(
        &self,
        patterns: &[StarBgpTriple],
        store: &dyn StarTripleStore,
    ) -> StarResult<Vec<StarBinding>> {
        if patterns.is_empty() {
            // A BGP with zero patterns has exactly one empty solution.
            return Ok(vec![StarBinding::new()]);
        }

        // Order patterns by estimated selectivity (fewest variables first).
        let mut ordered: Vec<&StarBgpTriple> = patterns.iter().collect();
        ordered.sort_by_key(|p| p.variables().len());

        // Bootstrap: evaluate the first (most selective) pattern.
        let first = ordered[0];
        let mut result = store.match_pattern(first, None)?;

        // Incrementally join the remaining patterns.
        for pattern in ordered.iter().skip(1) {
            let join_vars: Vec<String> = shared_variables_with_bindings(pattern, &result);

            if join_vars.is_empty() {
                // No shared variables – cross product (uncommon in real queries).
                let right = store.match_pattern(pattern, None)?;
                result = cross_product(result, right);
            } else {
                // Use hash join when the result set is large enough.
                let right = self.evaluate_with_seeds(pattern, &result, store)?;
                result = if result.len() <= self.nested_loop_threshold {
                    Self::nested_loop_join(result, right, &join_vars)
                } else {
                    Self::hash_join(result, right, &join_vars)
                };
            }
        }

        Ok(result)
    }

    /// Evaluate a pattern in parallel, using the existing bindings as seeds
    /// to restrict lookups where the store supports it.
    fn evaluate_with_seeds(
        &self,
        pattern: &StarBgpTriple,
        seeds: &[StarBinding],
        store: &dyn StarTripleStore,
    ) -> StarResult<Vec<StarBinding>> {
        let chunk_size = ((seeds.len() / self.parallelism) + 1).max(16);

        // Parallel evaluation across seed chunks.
        let chunks: Vec<&[StarBinding]> = seeds.chunks(chunk_size).collect();
        let results: Vec<StarResult<Vec<StarBinding>>> = chunks
            .par_iter()
            .map(|chunk| {
                let mut local: Vec<StarBinding> = Vec::new();
                for seed in *chunk {
                    let rows = store.match_pattern(pattern, Some(seed))?;
                    local.extend(rows);
                }
                Ok(local)
            })
            .collect();

        let mut flat: Vec<StarBinding> = Vec::new();
        for r in results {
            flat.extend(r?);
        }
        Ok(flat)
    }

    // ---- Join algorithms ----------------------------------------------------

    /// Nested-loop join – O(|left| × |right|) – best for small inputs.
    fn nested_loop_join(
        left: Vec<StarBinding>,
        right: Vec<StarBinding>,
        join_vars: &[String],
    ) -> Vec<StarBinding> {
        let mut out = Vec::new();
        for lb in &left {
            for rb in &right {
                if let Some(merged) = merge_bindings_compatible(lb, rb, join_vars) {
                    out.push(merged);
                }
            }
        }
        out
    }

    /// Hash join – O(|left| + |right|) – preferred for larger inputs.
    fn hash_join(
        left: Vec<StarBinding>,
        right: Vec<StarBinding>,
        join_vars: &[String],
    ) -> Vec<StarBinding> {
        // Build a hash table keyed on join-variable values from the right side.
        let mut table: HashMap<Vec<Option<StarTerm>>, Vec<StarBinding>> = HashMap::new();
        for rb in right {
            let key: Vec<Option<StarTerm>> = join_vars.iter().map(|v| rb.get(v).cloned()).collect();
            table.entry(key).or_default().push(rb);
        }

        // Probe with each left binding.
        let mut out = Vec::new();
        for lb in &left {
            let key: Vec<Option<StarTerm>> = join_vars.iter().map(|v| lb.get(v).cloned()).collect();
            if let Some(right_matches) = table.get(&key) {
                for rb in right_matches {
                    let mut merged = lb.clone();
                    for (k, v) in rb {
                        if !merged.contains_key(k.as_str()) {
                            merged.insert(k.clone(), v.clone());
                        }
                    }
                    out.push(merged);
                }
            }
        }
        out
    }
}

// ---- Helper functions -------------------------------------------------------

/// Return variable names that appear in `pattern` AND in at least one existing
/// binding row.
fn shared_variables_with_bindings(
    pattern: &StarBgpTriple,
    bindings: &[StarBinding],
) -> Vec<String> {
    let pattern_vars = pattern.variables();
    if bindings.is_empty() || pattern_vars.is_empty() {
        return Vec::new();
    }
    // Collect variable names present in the existing bindings.
    let bound_vars: std::collections::HashSet<&str> = bindings
        .iter()
        .flat_map(|b| b.keys().map(|k| k.as_str()))
        .collect();
    pattern_vars
        .into_iter()
        .filter(|v| bound_vars.contains(v.as_str()))
        .collect()
}

/// Return variable names shared between two patterns.
pub fn shared_variables(a: &StarBgpTriple, b: &StarBgpTriple) -> Vec<String> {
    let a_vars: std::collections::HashSet<String> = a.variables().into_iter().collect();
    b.variables()
        .into_iter()
        .filter(|v| a_vars.contains(v))
        .collect()
}

/// Check whether a BGP term matches a concrete candidate term given the
/// current variable bindings.
///
/// - `Concrete(t)` matches `candidate` if `t == candidate`.
/// - `Variable(v)` matches if `v` is unbound, or if `bindings[v] == candidate`.
pub fn term_matches(pattern: &StarBgpTerm, candidate: &StarTerm, bindings: &StarBinding) -> bool {
    match pattern {
        StarBgpTerm::Concrete(t) => t == candidate,
        StarBgpTerm::Variable(v) => match bindings.get(v.as_str()) {
            Some(bound) => bound == candidate,
            None => true,
        },
    }
}

/// Attempt to merge two binding rows that share `join_vars`.
///
/// Returns `None` if the rows are incompatible (the same variable is bound to
/// different values in the two rows).
pub fn merge_bindings_compatible(
    a: &StarBinding,
    b: &StarBinding,
    join_vars: &[String],
) -> Option<StarBinding> {
    // Check compatibility on join variables.
    for var in join_vars {
        match (a.get(var.as_str()), b.get(var.as_str())) {
            (Some(va), Some(vb)) if va != vb => return None,
            _ => {}
        }
    }
    // Merge all bindings from both rows.
    let mut merged = a.clone();
    for (k, v) in b {
        merged.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Some(merged)
}

/// Compute the Cartesian product of two binding sets (used when patterns share
/// no variables).
fn cross_product(left: Vec<StarBinding>, right: Vec<StarBinding>) -> Vec<StarBinding> {
    let mut out = Vec::with_capacity(left.len() * right.len());
    for lb in &left {
        for rb in &right {
            let mut merged = lb.clone();
            for (k, v) in rb {
                merged.insert(k.clone(), v.clone());
            }
            out.push(merged);
        }
    }
    out
}

// ---- Reference in-memory store ----------------------------------------------

/// A simple in-memory store backed by a `Vec<StarTriple>`, suitable for
/// testing and small datasets.
pub struct InMemoryStarStore {
    triples: Vec<StarTriple>,
}

impl InMemoryStarStore {
    /// Create a new store pre-populated with the given triples.
    pub fn new(triples: Vec<StarTriple>) -> Self {
        Self { triples }
    }

    /// Insert a triple into the store.
    pub fn insert(&mut self, triple: StarTriple) {
        self.triples.push(triple);
    }
}

impl StarTripleStore for InMemoryStarStore {
    fn match_pattern(
        &self,
        pattern: &StarBgpTriple,
        seed: Option<&StarBinding>,
    ) -> StarResult<Vec<StarBinding>> {
        let empty_binding = StarBinding::new();
        let seed_ref = seed.unwrap_or(&empty_binding);

        let bindings: Vec<StarBinding> = self
            .triples
            .par_iter()
            .filter_map(|triple| {
                if !term_matches(&pattern.subject, &triple.subject, seed_ref) {
                    return None;
                }
                if !term_matches(&pattern.predicate, &triple.predicate, seed_ref) {
                    return None;
                }
                if !term_matches(&pattern.object, &triple.object, seed_ref) {
                    return None;
                }
                // Build new bindings for variables not yet bound by the seed.
                let mut new_bindings = seed_ref.clone();
                bind_if_var(&pattern.subject, &triple.subject, &mut new_bindings);
                bind_if_var(&pattern.predicate, &triple.predicate, &mut new_bindings);
                bind_if_var(&pattern.object, &triple.object, &mut new_bindings);
                Some(new_bindings)
            })
            .collect();

        Ok(bindings)
    }

    fn triple_count(&self) -> usize {
        self.triples.len()
    }
}

/// Bind a variable to a concrete term in `bindings` if the pattern position is
/// a variable.
fn bind_if_var(pattern: &StarBgpTerm, concrete: &StarTerm, bindings: &mut StarBinding) {
    if let StarBgpTerm::Variable(v) = pattern {
        bindings
            .entry(v.clone())
            .or_insert_with(|| concrete.clone());
    }
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triple(s: &str, p: &str, o: &str) -> StarTriple {
        StarTriple::new(
            StarTerm::iri(s).expect("valid IRI"),
            StarTerm::iri(p).expect("valid IRI"),
            StarTerm::iri(o).expect("valid IRI"),
        )
    }

    fn make_store() -> InMemoryStarStore {
        InMemoryStarStore::new(vec![
            make_triple(
                "http://ex.org/alice",
                "http://ex.org/knows",
                "http://ex.org/bob",
            ),
            make_triple(
                "http://ex.org/alice",
                "http://ex.org/age",
                "http://ex.org/30",
            ),
            make_triple(
                "http://ex.org/bob",
                "http://ex.org/knows",
                "http://ex.org/carol",
            ),
            make_triple(
                "http://ex.org/carol",
                "http://ex.org/age",
                "http://ex.org/25",
            ),
        ])
    }

    #[test]
    fn test_empty_bgp_yields_one_empty_binding() {
        let store = make_store();
        let evaluator = ParallelStarBgpEvaluator::new(2);
        let results = evaluator.evaluate(&[], &store).expect("evaluate ok");
        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
    }

    #[test]
    fn test_single_pattern_with_concrete_subject() {
        let store = make_store();
        let evaluator = ParallelStarBgpEvaluator::new(2);

        let alice = StarTerm::iri("http://ex.org/alice").expect("valid IRI");
        let pattern = StarBgpTriple::new(
            StarBgpTerm::term(alice),
            StarBgpTerm::var("p"),
            StarBgpTerm::var("o"),
        );

        let results = evaluator.evaluate(&[pattern], &store).expect("evaluate ok");
        // Alice has two triples (knows, age).
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_two_pattern_join_via_shared_variable() {
        let store = make_store();
        let evaluator = ParallelStarBgpEvaluator::new(2);

        let knows = StarTerm::iri("http://ex.org/knows").expect("valid IRI");
        let age = StarTerm::iri("http://ex.org/age").expect("valid IRI");

        // ?person knows ?friend .  ?friend age ?a .
        let p1 = StarBgpTriple::new(
            StarBgpTerm::var("person"),
            StarBgpTerm::term(knows),
            StarBgpTerm::var("friend"),
        );
        let p2 = StarBgpTriple::new(
            StarBgpTerm::var("friend"),
            StarBgpTerm::term(age),
            StarBgpTerm::var("age"),
        );

        let results = evaluator.evaluate(&[p1, p2], &store).expect("evaluate ok");
        // alice→bob (bob has no age), alice→carol nope... carol has age=25, bob→carol (carol age=25).
        // Wait: alice knows bob, bob knows carol.
        // alice knows bob → bob age? no age triple for bob. So no join.
        // bob knows carol → carol age 25. That gives one result.
        assert!(!results.is_empty(), "should have at least one result");
        for row in &results {
            assert!(row.contains_key("person"), "row should have ?person");
            assert!(row.contains_key("friend"), "row should have ?friend");
            assert!(row.contains_key("age"), "row should have ?age");
        }
    }

    #[test]
    fn test_quoted_triple_pattern_matching() {
        // Build a store containing a quoted triple as subject.
        let inner = StarTriple::new(
            StarTerm::iri("http://ex.org/alice").expect("valid IRI"),
            StarTerm::iri("http://ex.org/age").expect("valid IRI"),
            StarTerm::literal("30").expect("valid literal"),
        );
        let outer = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            StarTerm::iri("http://ex.org/certainty").expect("valid IRI"),
            StarTerm::literal("0.9").expect("valid literal"),
        );
        let store = InMemoryStarStore::new(vec![outer]);

        let evaluator = ParallelStarBgpEvaluator::new(2);

        // Query: << ex:alice ex:age 30 >> ex:certainty ?c .
        let certainty = StarTerm::iri("http://ex.org/certainty").expect("valid IRI");
        let pattern = StarBgpTriple::new(
            StarBgpTerm::term(StarTerm::quoted_triple(inner)),
            StarBgpTerm::term(certainty),
            StarBgpTerm::var("c"),
        );

        let results = evaluator.evaluate(&[pattern], &store).expect("evaluate ok");
        assert_eq!(results.len(), 1);
        let c = results[0].get("c").expect("?c should be bound");
        assert_eq!(*c, StarTerm::literal("0.9").expect("valid literal"));
    }

    #[test]
    fn test_hash_join_used_for_large_sets() {
        // Build a store with many triples to trigger hash join.
        let mut triples = Vec::new();
        for i in 0..2000_usize {
            triples.push(StarTriple::new(
                StarTerm::iri(&format!("http://ex.org/s{i}")).expect("valid IRI"),
                StarTerm::iri("http://ex.org/knows").expect("valid IRI"),
                StarTerm::iri(&format!("http://ex.org/o{i}")).expect("valid IRI"),
            ));
            triples.push(StarTriple::new(
                StarTerm::iri(&format!("http://ex.org/o{i}")).expect("valid IRI"),
                StarTerm::iri("http://ex.org/age").expect("valid IRI"),
                StarTerm::literal(&format!("{i}")).expect("valid literal"),
            ));
        }
        let store = InMemoryStarStore::new(triples);

        let evaluator = ParallelStarBgpEvaluator::new(4);
        let knows = StarTerm::iri("http://ex.org/knows").expect("valid IRI");
        let age = StarTerm::iri("http://ex.org/age").expect("valid IRI");

        let p1 = StarBgpTriple::new(
            StarBgpTerm::var("s"),
            StarBgpTerm::term(knows),
            StarBgpTerm::var("o"),
        );
        let p2 = StarBgpTriple::new(
            StarBgpTerm::var("o"),
            StarBgpTerm::term(age),
            StarBgpTerm::var("a"),
        );

        let results = evaluator.evaluate(&[p1, p2], &store).expect("evaluate ok");
        assert_eq!(results.len(), 2000);
    }

    #[test]
    fn test_shared_variables_helper() {
        let a = StarBgpTriple::new(
            StarBgpTerm::var("x"),
            StarBgpTerm::var("p"),
            StarBgpTerm::var("y"),
        );
        let b = StarBgpTriple::new(
            StarBgpTerm::var("y"),
            StarBgpTerm::var("q"),
            StarBgpTerm::var("z"),
        );
        let shared = shared_variables(&a, &b);
        assert_eq!(shared, vec!["y".to_owned()]);
    }

    #[test]
    fn test_term_matches_variable() {
        let alice = StarTerm::iri("http://ex.org/alice").expect("valid IRI");
        let mut bindings = StarBinding::new();
        // Unbound variable matches anything.
        assert!(term_matches(&StarBgpTerm::var("x"), &alice, &bindings));
        // Bound variable matches only its bound value.
        bindings.insert("x".to_owned(), alice.clone());
        assert!(term_matches(&StarBgpTerm::var("x"), &alice, &bindings));
        let bob = StarTerm::iri("http://ex.org/bob").expect("valid IRI");
        assert!(!term_matches(&StarBgpTerm::var("x"), &bob, &bindings));
    }
}
