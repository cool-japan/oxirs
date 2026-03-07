//! Distributed RDF-star query processing.
//!
//! Provides `DistributedStarQuery`, `ShardRouter`, and `ResultMerger` for
//! splitting and executing SPARQL-star quoted-triple queries across shards.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::model::{StarTerm, StarTriple};
use crate::{StarError, StarResult};

// ============================================================================
// Shard identifier
// ============================================================================

/// An opaque identifier for a query shard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ShardId(pub u32);

impl fmt::Display for ShardId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shard:{}", self.0)
    }
}

// ============================================================================
// Query binding (single variable → term mapping)
// ============================================================================

/// A single solution binding produced by a query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Binding {
    /// Variable name → bound term.
    pub vars: HashMap<String, StarTerm>,
}

impl Binding {
    /// Create an empty binding.
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    /// Bind a variable to a term.
    pub fn bind(&mut self, var: impl Into<String>, term: StarTerm) {
        self.vars.insert(var.into(), term);
    }

    /// Look up a bound variable.
    pub fn get(&self, var: &str) -> Option<&StarTerm> {
        self.vars.get(var)
    }

    /// Merge another binding into `self`.  Returns `Err` on conflicting bindings.
    pub fn merge(&self, other: &Binding) -> StarResult<Binding> {
        let mut merged = self.clone();
        for (var, term) in &other.vars {
            match merged.vars.get(var) {
                Some(existing) if existing != term => {
                    return Err(StarError::QueryError {
                        message: format!(
                            "Conflicting binding for ?{var}: {existing:?} vs {term:?}"
                        ),
                        query_fragment: None,
                        position: None,
                        suggestion: None,
                    });
                }
                _ => {
                    merged.vars.insert(var.clone(), term.clone());
                }
            }
        }
        Ok(merged)
    }
}

impl Default for Binding {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Query result set
// ============================================================================

/// A set of `Binding`s returned from a query or sub-query.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryResult {
    /// All solution mappings.
    pub bindings: Vec<Binding>,
    /// The shard this result came from (`None` if already merged).
    pub source_shard: Option<ShardId>,
}

impl QueryResult {
    /// Create an empty result set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a result attributed to a specific shard.
    pub fn from_shard(shard: ShardId) -> Self {
        Self {
            bindings: Vec::new(),
            source_shard: Some(shard),
        }
    }

    /// Add a binding to the result.
    pub fn push(&mut self, binding: Binding) {
        self.bindings.push(binding);
    }

    /// Number of solution bindings.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// `true` if no bindings.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }
}

// ============================================================================
// ShardRouter
// ============================================================================

/// Routes quoted-triple query patterns to the correct shard.
///
/// Uses a consistent-hashing strategy on the **subject** component of the
/// outermost triple (or its hash if it is itself a quoted triple).
#[derive(Debug, Clone)]
pub struct ShardRouter {
    /// Total number of shards.
    num_shards: u32,
}

impl ShardRouter {
    /// Create a `ShardRouter` for the given number of shards.
    pub fn new(num_shards: u32) -> StarResult<Self> {
        if num_shards == 0 {
            return Err(StarError::QueryError {
                message: "shard count must be > 0".into(),
                query_fragment: None,
                position: None,
                suggestion: Some("use at least 1 shard".into()),
            });
        }
        Ok(Self { num_shards })
    }

    /// Determine which shard should own the given `StarTriple`.
    ///
    /// Routes on the hash of the outermost subject term.
    pub fn route_triple(&self, triple: &StarTriple) -> ShardId {
        ShardId(term_hash(&triple.subject) % self.num_shards)
    }

    /// Determine which shards a query *pattern* must visit.
    ///
    /// If the subject is bound (not a variable), only one shard is visited.
    /// Otherwise, all shards must be queried.
    pub fn shards_for_pattern(&self, pattern: &StarTriple) -> Vec<ShardId> {
        match &pattern.subject {
            StarTerm::Variable(_) => (0..self.num_shards).map(ShardId).collect(),
            term => vec![ShardId(term_hash(term) % self.num_shards)],
        }
    }

    /// Number of shards.
    pub fn num_shards(&self) -> u32 {
        self.num_shards
    }
}

/// Compute a stable hash for a `StarTerm` (for routing).
fn term_hash(term: &StarTerm) -> u32 {
    use std::collections::hash_map::DefaultHasher;
    let mut h = DefaultHasher::new();
    term.hash(&mut h);
    (h.finish() & 0xFFFF_FFFF) as u32
}

// ============================================================================
// DistributedStarQuery
// ============================================================================

/// A SPARQL-star query partitioned across shards.
///
/// Splits a set of triple patterns into per-shard sub-queries, executes them
/// (simulated via an in-memory shard store), and merges the results.
#[derive(Debug)]
pub struct DistributedStarQuery {
    /// The patterns that make up the query.
    patterns: Vec<StarTriple>,
    /// Router used to determine shard assignments.
    router: ShardRouter,
    /// Per-shard data store (simulated in-process).
    shards: HashMap<ShardId, Vec<StarTriple>>,
}

impl DistributedStarQuery {
    /// Create a new distributed query with `num_shards` shards.
    pub fn new(num_shards: u32) -> StarResult<Self> {
        Ok(Self {
            patterns: Vec::new(),
            router: ShardRouter::new(num_shards)?,
            shards: HashMap::new(),
        })
    }

    /// Add a triple pattern to the query.
    pub fn add_pattern(&mut self, pattern: StarTriple) {
        self.patterns.push(pattern);
    }

    /// Insert a triple into the appropriate shard store.
    pub fn insert_triple(&mut self, triple: StarTriple) {
        let shard = self.router.route_triple(&triple);
        self.shards.entry(shard).or_default().push(triple);
    }

    /// Execute the query by dispatching sub-queries to shards and merging results.
    pub fn execute(&self) -> StarResult<QueryResult> {
        let mut partial_results: Vec<QueryResult> = Vec::new();

        for pattern in &self.patterns {
            let target_shards = self.router.shards_for_pattern(pattern);
            for shard_id in &target_shards {
                let triples = self.shards.get(shard_id).map(Vec::as_slice).unwrap_or(&[]);
                let sub = self.match_pattern_in_shard(pattern, triples, *shard_id)?;
                partial_results.push(sub);
            }
        }

        ResultMerger::merge_all(partial_results)
    }

    /// Match a single pattern against a shard's triples.
    fn match_pattern_in_shard(
        &self,
        pattern: &StarTriple,
        triples: &[StarTriple],
        shard_id: ShardId,
    ) -> StarResult<QueryResult> {
        let mut result = QueryResult::from_shard(shard_id);

        for triple in triples {
            if let Some(binding) = try_match_pattern(pattern, triple) {
                result.push(binding);
            }
        }

        Ok(result)
    }
}

/// Try to match `pattern` against `triple`, returning bindings if successful.
fn try_match_pattern(pattern: &StarTriple, triple: &StarTriple) -> Option<Binding> {
    let mut binding = Binding::new();

    if !bind_term(&pattern.subject, &triple.subject, &mut binding) {
        return None;
    }
    if !bind_term(&pattern.predicate, &triple.predicate, &mut binding) {
        return None;
    }
    if !bind_term(&pattern.object, &triple.object, &mut binding) {
        return None;
    }

    Some(binding)
}

/// Attempt to unify `pattern_term` with `data_term`, updating `binding`.
fn bind_term(pattern_term: &StarTerm, data_term: &StarTerm, binding: &mut Binding) -> bool {
    match pattern_term {
        StarTerm::Variable(v) => match binding.get(&v.name) {
            Some(existing) if existing != data_term => false,
            _ => {
                binding.bind(v.name.clone(), data_term.clone());
                true
            }
        },
        StarTerm::QuotedTriple(pq) => {
            if let StarTerm::QuotedTriple(dq) = data_term {
                // Recursively match inner quoted triples.
                bind_term(&pq.subject, &dq.subject, binding)
                    && bind_term(&pq.predicate, &dq.predicate, binding)
                    && bind_term(&pq.object, &dq.object, binding)
            } else {
                false
            }
        }
        _ => pattern_term == data_term,
    }
}

// ============================================================================
// ResultMerger
// ============================================================================

/// Merges `QueryResult`s from multiple shards preserving RDF-star semantics.
///
/// Uses a union strategy: all bindings from all shards are combined.
/// Duplicate bindings (same variable → value mapping) are removed.
pub struct ResultMerger;

impl ResultMerger {
    /// Merge a list of per-shard results into a single `QueryResult`.
    pub fn merge_all(results: Vec<QueryResult>) -> StarResult<QueryResult> {
        let mut merged = QueryResult::new();
        let mut seen: HashSet<String> = HashSet::new();

        for result in results {
            for binding in result.bindings {
                let key = binding_key(&binding);
                if seen.insert(key) {
                    merged.push(binding);
                }
            }
        }

        Ok(merged)
    }

    /// Join two result sets (natural join on shared variables).
    pub fn natural_join(left: QueryResult, right: QueryResult) -> StarResult<QueryResult> {
        let mut joined = QueryResult::new();

        for l_bind in &left.bindings {
            for r_bind in &right.bindings {
                if let Ok(merged_binding) = l_bind.merge(r_bind) {
                    joined.push(merged_binding);
                }
            }
        }

        Ok(joined)
    }
}

/// Produce a stable string key for a `Binding` (for deduplication).
fn binding_key(binding: &Binding) -> String {
    let mut pairs: Vec<_> = binding.vars.iter().collect();
    pairs.sort_by_key(|(k, _)| k.as_str());
    pairs
        .into_iter()
        .map(|(k, v)| format!("{k}={v:?}"))
        .collect::<Vec<_>>()
        .join(";")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{StarTerm, StarTriple, Variable};

    fn iri(s: &str) -> StarTerm {
        StarTerm::iri(s).expect("valid IRI")
    }

    fn var(name: &str) -> StarTerm {
        StarTerm::Variable(Variable { name: name.into() })
    }

    fn triple(s: &str, p: &str, o: &str) -> StarTriple {
        StarTriple::new(iri(s), iri(p), iri(o))
    }

    // ------------------------------------------------------------------
    // ShardRouter tests
    // ------------------------------------------------------------------

    #[test]
    fn test_shard_router_single_shard() {
        let router = ShardRouter::new(1).expect("ok");
        let t = triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        assert_eq!(router.route_triple(&t), ShardId(0));
    }

    #[test]
    fn test_shard_router_zero_shards_error() {
        assert!(ShardRouter::new(0).is_err());
    }

    #[test]
    fn test_shard_router_bound_subject_one_shard() {
        let router = ShardRouter::new(4).expect("ok");
        let pattern = triple(
            "http://ex.org/alice",
            "http://ex.org/knows",
            "http://ex.org/bob",
        );
        let shards = router.shards_for_pattern(&pattern);
        assert_eq!(shards.len(), 1);
    }

    #[test]
    fn test_shard_router_variable_subject_all_shards() {
        let router = ShardRouter::new(4).expect("ok");
        let pattern = StarTriple::new(var("s"), iri("http://ex.org/p"), iri("http://ex.org/o"));
        let shards = router.shards_for_pattern(&pattern);
        assert_eq!(shards.len(), 4);
    }

    #[test]
    fn test_shard_router_deterministic() {
        let router = ShardRouter::new(8).expect("ok");
        let t = triple(
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
        );
        let s1 = router.route_triple(&t);
        let s2 = router.route_triple(&t);
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_shard_router_quoted_triple_subject() {
        let router = ShardRouter::new(4).expect("ok");
        let inner = triple("http://ex.org/a", "http://ex.org/b", "http://ex.org/c");
        let t = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://ex.org/cert"),
            StarTerm::literal("0.9").expect("ok"),
        );
        let shard = router.route_triple(&t);
        assert!(shard.0 < 4);
    }

    // ------------------------------------------------------------------
    // Binding tests
    // ------------------------------------------------------------------

    #[test]
    fn test_binding_merge_compatible() {
        let mut b1 = Binding::new();
        b1.bind("x", iri("http://ex.org/alice"));

        let mut b2 = Binding::new();
        b2.bind("y", iri("http://ex.org/bob"));

        let merged = b1.merge(&b2).expect("ok");
        assert_eq!(merged.get("x"), Some(&iri("http://ex.org/alice")));
        assert_eq!(merged.get("y"), Some(&iri("http://ex.org/bob")));
    }

    #[test]
    fn test_binding_merge_conflict() {
        let mut b1 = Binding::new();
        b1.bind("x", iri("http://ex.org/alice"));

        let mut b2 = Binding::new();
        b2.bind("x", iri("http://ex.org/bob"));

        assert!(b1.merge(&b2).is_err());
    }

    #[test]
    fn test_binding_merge_same_value() {
        let mut b1 = Binding::new();
        b1.bind("x", iri("http://ex.org/alice"));

        let mut b2 = Binding::new();
        b2.bind("x", iri("http://ex.org/alice"));

        let merged = b1.merge(&b2).expect("same-value merge ok");
        assert_eq!(merged.get("x"), Some(&iri("http://ex.org/alice")));
    }

    // ------------------------------------------------------------------
    // DistributedStarQuery tests
    // ------------------------------------------------------------------

    #[test]
    fn test_distributed_query_basic() {
        let mut q = DistributedStarQuery::new(2).expect("ok");
        q.insert_triple(triple(
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
        ));
        q.insert_triple(triple(
            "http://ex.org/bob",
            "http://ex.org/age",
            "http://ex.org/25",
        ));

        let pattern = StarTriple::new(var("s"), iri("http://ex.org/age"), var("o"));
        q.add_pattern(pattern);

        let result = q.execute().expect("execute ok");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_distributed_query_empty_result() {
        let mut q = DistributedStarQuery::new(2).expect("ok");
        q.insert_triple(triple(
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
        ));

        let pattern = triple(
            "http://ex.org/nobody",
            "http://ex.org/age",
            "http://ex.org/0",
        );
        q.add_pattern(pattern);

        let result = q.execute().expect("execute ok");
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_distributed_query_quoted_triple_pattern() {
        let mut q = DistributedStarQuery::new(2).expect("ok");

        let inner = triple(
            "http://ex.org/alice",
            "http://ex.org/age",
            "http://ex.org/30",
        );
        let meta = StarTriple::new(
            StarTerm::quoted_triple(inner.clone()),
            iri("http://ex.org/certainty"),
            StarTerm::literal("0.9").expect("ok"),
        );
        q.insert_triple(meta.clone());

        let pattern = StarTriple::new(
            StarTerm::quoted_triple(inner),
            iri("http://ex.org/certainty"),
            var("cert"),
        );
        q.add_pattern(pattern);

        let result = q.execute().expect("execute ok");
        assert_eq!(result.len(), 1);
        let cert = result.bindings[0].get("cert");
        assert!(cert.is_some());
    }

    #[test]
    fn test_distributed_query_no_patterns() {
        let q = DistributedStarQuery::new(3).expect("ok");
        let result = q.execute().expect("execute ok");
        assert!(result.is_empty());
    }

    // ------------------------------------------------------------------
    // ResultMerger tests
    // ------------------------------------------------------------------

    #[test]
    fn test_result_merger_deduplication() {
        let mut r1 = QueryResult::from_shard(ShardId(0));
        let mut b = Binding::new();
        b.bind("x", iri("http://ex.org/alice"));
        r1.push(b.clone());

        let mut r2 = QueryResult::from_shard(ShardId(1));
        r2.push(b.clone()); // duplicate

        let merged = ResultMerger::merge_all(vec![r1, r2]).expect("ok");
        assert_eq!(merged.len(), 1);
    }

    #[test]
    fn test_result_merger_distinct_bindings() {
        let mut r1 = QueryResult::from_shard(ShardId(0));
        let mut b1 = Binding::new();
        b1.bind("x", iri("http://ex.org/alice"));
        r1.push(b1);

        let mut r2 = QueryResult::from_shard(ShardId(1));
        let mut b2 = Binding::new();
        b2.bind("x", iri("http://ex.org/bob"));
        r2.push(b2);

        let merged = ResultMerger::merge_all(vec![r1, r2]).expect("ok");
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_natural_join_basic() {
        let mut left = QueryResult::new();
        let mut bl = Binding::new();
        bl.bind("x", iri("http://ex.org/alice"));
        left.push(bl);

        let mut right = QueryResult::new();
        let mut br = Binding::new();
        br.bind("y", iri("http://ex.org/knows"));
        right.push(br);

        let joined = ResultMerger::natural_join(left, right).expect("ok");
        assert_eq!(joined.len(), 1);
        let b = &joined.bindings[0];
        assert_eq!(b.get("x"), Some(&iri("http://ex.org/alice")));
        assert_eq!(b.get("y"), Some(&iri("http://ex.org/knows")));
    }

    #[test]
    fn test_natural_join_conflict_excluded() {
        let mut left = QueryResult::new();
        let mut bl = Binding::new();
        bl.bind("x", iri("http://ex.org/alice"));
        left.push(bl);

        let mut right = QueryResult::new();
        let mut br = Binding::new();
        br.bind("x", iri("http://ex.org/bob")); // conflict!
        right.push(br);

        let joined = ResultMerger::natural_join(left, right).expect("ok");
        assert_eq!(joined.len(), 0); // No compatible bindings.
    }

    #[test]
    fn test_shard_id_display() {
        assert_eq!(format!("{}", ShardId(3)), "shard:3");
    }

    #[test]
    fn test_query_result_len_and_empty() {
        let mut r = QueryResult::new();
        assert!(r.is_empty());
        r.push(Binding::new());
        assert_eq!(r.len(), 1);
        assert!(!r.is_empty());
    }
}
