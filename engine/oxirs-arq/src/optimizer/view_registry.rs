//! View Registry for Materialized Query Views
//!
//! This module provides a focused view registry for caching and reusing
//! materialized query results. It implements:
//!
//! - [`ViewDefinition`] — stored binding sets for a given algebra pattern
//! - [`ViewRegistry`] — LRU-based registry with structural algebra matching
//!   and graph-update-based invalidation
//!
//! # Design
//!
//! Views are identified by a structural hash of their defining [`Algebra`]
//! expression. When a graph is updated, all views whose algebra references
//! that graph name are evicted. Eviction follows an LRU-like policy based
//! on `hit_count` and `size_bytes`.

use crate::algebra::Algebra;
use std::collections::{hash_map::DefaultHasher, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::Instant;

// ---------------------------------------------------------------------------
// BindingSet
// ---------------------------------------------------------------------------

/// A single row of query result bindings: variable name → string value.
pub type BindingSet = HashMap<String, String>;

// ---------------------------------------------------------------------------
// ViewDefinition
// ---------------------------------------------------------------------------

/// A materialized view: cached result bindings for a given algebra pattern.
#[derive(Debug, Clone)]
pub struct ViewDefinition {
    /// The algebra pattern that defines this view.
    pub pattern: Algebra,
    /// Cached result bindings.
    pub bindings: Vec<BindingSet>,
    /// When this view was created.
    pub created_at: Instant,
    /// How many times this view has been accessed.
    pub hit_count: u64,
    /// Approximate memory usage of the stored bindings (bytes).
    pub size_bytes: usize,
    /// Set of graph names referenced by the pattern (for invalidation).
    pub referenced_graphs: Vec<String>,
}

impl ViewDefinition {
    /// Create a new view from an algebra pattern and pre-computed bindings.
    pub fn new(pattern: Algebra, bindings: Vec<BindingSet>) -> Self {
        let size_bytes = Self::estimate_size(&bindings);
        let referenced_graphs = extract_graph_names(&pattern);
        Self {
            pattern,
            bindings,
            created_at: Instant::now(),
            hit_count: 0,
            size_bytes,
            referenced_graphs,
        }
    }

    /// Increment the hit counter and return the updated count.
    pub fn record_hit(&mut self) -> u64 {
        self.hit_count += 1;
        self.hit_count
    }

    /// Age of this view in seconds.
    pub fn age_secs(&self) -> u64 {
        self.created_at.elapsed().as_secs()
    }

    /// Estimate byte size of a set of binding rows.
    fn estimate_size(bindings: &[BindingSet]) -> usize {
        bindings.iter().fold(0usize, |acc, row| {
            acc + row
                .iter()
                .map(|(k, v)| k.len() + v.len() + 16)
                .sum::<usize>()
        })
    }
}

// ---------------------------------------------------------------------------
// Graph name extraction helper
// ---------------------------------------------------------------------------

/// Recursively collect all graph names referenced in a `Graph` algebra node.
fn extract_graph_names(algebra: &Algebra) -> Vec<String> {
    let mut names = Vec::new();
    collect_graph_names(algebra, &mut names);
    names.sort();
    names.dedup();
    names
}

fn collect_graph_names(algebra: &Algebra, out: &mut Vec<String>) {
    match algebra {
        Algebra::Graph { graph, pattern } => {
            use crate::algebra::Term;
            if let Term::Iri(iri) = graph {
                out.push(iri.as_str().to_owned());
            }
            collect_graph_names(pattern, out);
        }
        Algebra::Join { left, right } => {
            collect_graph_names(left, out);
            collect_graph_names(right, out);
        }
        Algebra::LeftJoin { left, right, .. } => {
            collect_graph_names(left, out);
            collect_graph_names(right, out);
        }
        Algebra::Union { left, right } => {
            collect_graph_names(left, out);
            collect_graph_names(right, out);
        }
        Algebra::Filter { pattern, .. } => collect_graph_names(pattern, out),
        Algebra::Extend { pattern, .. } => collect_graph_names(pattern, out),
        Algebra::Minus { left, right } => {
            collect_graph_names(left, out);
            collect_graph_names(right, out);
        }
        Algebra::Project { pattern, .. } => collect_graph_names(pattern, out),
        Algebra::Distinct { pattern } => collect_graph_names(pattern, out),
        Algebra::Slice { pattern, .. } => collect_graph_names(pattern, out),
        Algebra::Service { pattern, .. } => collect_graph_names(pattern, out),
        Algebra::Bgp(_)
        | Algebra::OrderBy { .. }
        | Algebra::Group { .. }
        | Algebra::Reduced { .. }
        | Algebra::Values { .. }
        | Algebra::PropertyPath { .. }
        | Algebra::Having { .. }
        | Algebra::Table
        | Algebra::Zero
        | Algebra::Empty => {}
    }
}

// ---------------------------------------------------------------------------
// Algebra hashing for structural matching
// ---------------------------------------------------------------------------

/// Compute a structural hash of an `Algebra` tree for view cache lookup.
///
/// Uses `DefaultHasher` applied to the `Debug` representation; sufficient
/// for caching purposes since exact query repetitions are the target.
pub fn algebra_hash(algebra: &Algebra) -> u64 {
    let mut hasher = DefaultHasher::new();
    // Use Debug output as a stable structural representation.
    format!("{algebra:?}").hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// ViewRegistry
// ---------------------------------------------------------------------------

/// LRU-based registry for materialized query views.
///
/// Views are stored by a structural hash of their algebra expression.
/// When `max_size` is exceeded, the least-recently-used view (lowest
/// `hit_count`, breaking ties by largest `size_bytes`) is evicted.
pub struct ViewRegistry {
    /// Views indexed by algebra hash.
    views: HashMap<u64, ViewDefinition>,
    /// Insertion-order tracking for LRU eviction.
    insertion_order: VecDeque<u64>,
    /// Maximum number of views allowed.
    pub max_size: usize,
}

impl ViewRegistry {
    /// Create a new registry with the given maximum number of views.
    pub fn new(max_size: usize) -> Self {
        Self {
            views: HashMap::new(),
            insertion_order: VecDeque::new(),
            max_size: max_size.max(1),
        }
    }

    // ------------------------------------------------------------------
    // Insertion
    // ------------------------------------------------------------------

    /// Insert a new view into the registry.
    ///
    /// If the registry is full, the LRU view is evicted first.
    pub fn insert(&mut self, id: String, view: ViewDefinition) {
        let key = Self::id_hash(&id);
        if self.views.len() >= self.max_size && !self.views.contains_key(&key) {
            self.evict_lru();
        }
        if !self.insertion_order.contains(&key) {
            self.insertion_order.push_back(key);
        }
        self.views.insert(key, view);
    }

    /// Insert a view using the algebra hash as the key (alternative to string ID).
    pub fn insert_by_algebra(&mut self, view: ViewDefinition) {
        let key = algebra_hash(&view.pattern);
        if self.views.len() >= self.max_size && !self.views.contains_key(&key) {
            self.evict_lru();
        }
        if !self.insertion_order.contains(&key) {
            self.insertion_order.push_back(key);
        }
        self.views.insert(key, view);
    }

    // ------------------------------------------------------------------
    // Lookup
    // ------------------------------------------------------------------

    /// Find a view whose defining pattern structurally matches `query`.
    ///
    /// Returns an immutable reference and increments the view's hit counter
    /// via interior mutability would require `RefCell`; instead we return
    /// `Option<&ViewDefinition>` without incrementing.  Use
    /// [`record_hit`](Self::record_hit) to track usage separately.
    pub fn find_matching_view(&self, query: &Algebra) -> Option<&ViewDefinition> {
        let key = algebra_hash(query);
        self.views.get(&key)
    }

    /// Record a cache hit for the view identified by `query`.
    pub fn record_hit(&mut self, query: &Algebra) {
        let key = algebra_hash(query);
        if let Some(view) = self.views.get_mut(&key) {
            view.record_hit();
        }
    }

    /// Retrieve a view by string ID.
    pub fn get_by_id(&self, id: &str) -> Option<&ViewDefinition> {
        let key = Self::id_hash(id);
        self.views.get(&key)
    }

    // ------------------------------------------------------------------
    // Invalidation
    // ------------------------------------------------------------------

    /// Invalidate all views that reference `updated_graph`.
    ///
    /// Called when triples in `updated_graph` are inserted or deleted.
    pub fn invalidate_on_update(&mut self, updated_graph: &str) {
        let to_remove: Vec<u64> = self
            .views
            .iter()
            .filter(|(_, v)| v.referenced_graphs.iter().any(|g| g == updated_graph))
            .map(|(k, _)| *k)
            .collect();
        for key in &to_remove {
            self.views.remove(key);
            self.insertion_order.retain(|k| k != key);
        }
    }

    /// Remove a specific view by string ID.
    pub fn remove(&mut self, id: &str) -> Option<ViewDefinition> {
        let key = Self::id_hash(id);
        let view = self.views.remove(&key);
        if view.is_some() {
            self.insertion_order.retain(|k| *k != key);
        }
        view
    }

    /// Clear all views.
    pub fn clear(&mut self) {
        self.views.clear();
        self.insertion_order.clear();
    }

    // ------------------------------------------------------------------
    // Eviction
    // ------------------------------------------------------------------

    /// Evict the least-recently-used view.
    ///
    /// LRU policy: lowest `hit_count` wins; ties broken by largest `size_bytes`
    /// (evict the largest unused view first).
    pub fn evict_lru(&mut self) {
        if self.views.is_empty() {
            return;
        }
        // Find the key with the lowest hit_count (ties: largest size_bytes).
        let evict_key = self
            .views
            .iter()
            .min_by(|(_, a), (_, b)| {
                a.hit_count
                    .cmp(&b.hit_count)
                    .then(b.size_bytes.cmp(&a.size_bytes))
            })
            .map(|(k, _)| *k);

        if let Some(key) = evict_key {
            self.views.remove(&key);
            self.insertion_order.retain(|k| *k != key);
        }
    }

    // ------------------------------------------------------------------
    // Metrics
    // ------------------------------------------------------------------

    /// Number of views in the registry.
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// Total number of hits across all views.
    pub fn total_hits(&self) -> u64 {
        self.views.values().map(|v| v.hit_count).sum()
    }

    /// Total estimated memory usage across all views (bytes).
    pub fn total_size_bytes(&self) -> usize {
        self.views.values().map(|v| v.size_bytes).sum()
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    fn id_hash(id: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for ViewRegistry {
    fn default() -> Self {
        Self::new(64)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Algebra, Term, TriplePattern};
    use oxirs_core::model::NamedNode;

    fn var(name: &str) -> Term {
        use oxirs_core::model::Variable;
        Term::Variable(Variable::new(name).expect("valid variable"))
    }

    fn iri(s: &str) -> Term {
        Term::Iri(NamedNode::new(s).expect("valid IRI"))
    }

    fn make_bgp(s: Term, p: Term, o: Term) -> Algebra {
        Algebra::Bgp(vec![TriplePattern {
            subject: s,
            predicate: p,
            object: o,
        }])
    }

    fn make_bindings(rows: usize) -> Vec<BindingSet> {
        (0..rows)
            .map(|i| {
                let mut row = BindingSet::new();
                row.insert("s".to_string(), format!("http://ex.org/s{i}"));
                row
            })
            .collect()
    }

    // ------------------------------------------------------------------
    // ViewDefinition tests
    // ------------------------------------------------------------------

    #[test]
    fn test_view_definition_new() {
        let pattern = make_bgp(var("s"), var("p"), var("o"));
        let bindings = make_bindings(10);
        let view = ViewDefinition::new(pattern, bindings);
        assert_eq!(view.hit_count, 0);
        assert!(view.size_bytes > 0);
    }

    #[test]
    fn test_view_definition_record_hit() {
        let pattern = make_bgp(var("s"), var("p"), var("o"));
        let mut view = ViewDefinition::new(pattern, vec![]);
        assert_eq!(view.record_hit(), 1);
        assert_eq!(view.record_hit(), 2);
        assert_eq!(view.hit_count, 2);
    }

    #[test]
    fn test_view_definition_age() {
        let pattern = make_bgp(var("s"), var("p"), var("o"));
        let view = ViewDefinition::new(pattern, vec![]);
        // Should be 0 seconds old (just created).
        assert!(view.age_secs() < 2);
    }

    #[test]
    fn test_view_definition_graph_extraction() {
        let inner = make_bgp(var("s"), var("p"), var("o"));
        let graph_term = iri("http://ex.org/graph1");
        let pattern = Algebra::Graph {
            graph: graph_term,
            pattern: Box::new(inner),
        };
        let view = ViewDefinition::new(pattern, vec![]);
        assert!(
            view.referenced_graphs
                .contains(&"http://ex.org/graph1".to_string()),
            "graph name should be extracted"
        );
    }

    #[test]
    fn test_view_definition_no_graphs() {
        let pattern = make_bgp(var("s"), var("p"), var("o"));
        let view = ViewDefinition::new(pattern, vec![]);
        assert!(view.referenced_graphs.is_empty());
    }

    // ------------------------------------------------------------------
    // algebra_hash tests
    // ------------------------------------------------------------------

    #[test]
    fn test_algebra_hash_same_algebra_same_hash() {
        let a = make_bgp(var("s"), var("p"), var("o"));
        let b = make_bgp(var("s"), var("p"), var("o"));
        assert_eq!(algebra_hash(&a), algebra_hash(&b));
    }

    #[test]
    fn test_algebra_hash_different_algebra_different_hash() {
        let a = make_bgp(var("s"), var("p"), var("o"));
        let b = make_bgp(var("x"), var("y"), var("z"));
        assert_ne!(algebra_hash(&a), algebra_hash(&b));
    }

    // ------------------------------------------------------------------
    // ViewRegistry tests
    // ------------------------------------------------------------------

    #[test]
    fn test_registry_new_empty() {
        let reg = ViewRegistry::new(10);
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_insert_and_len() {
        let mut reg = ViewRegistry::new(10);
        let pattern = make_bgp(var("s"), var("p"), var("o"));
        let view = ViewDefinition::new(pattern, make_bindings(5));
        reg.insert("view1".to_string(), view);
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_registry_find_matching_view() {
        let mut reg = ViewRegistry::new(10);
        let pattern = make_bgp(var("s"), var("p"), var("o"));
        let view = ViewDefinition::new(pattern.clone(), make_bindings(3));
        reg.insert_by_algebra(view);

        let result = reg.find_matching_view(&pattern);
        assert!(result.is_some(), "should find matching view");
    }

    #[test]
    fn test_registry_find_no_match() {
        let reg = ViewRegistry::new(10);
        let query = make_bgp(var("s"), var("p"), var("o"));
        assert!(reg.find_matching_view(&query).is_none());
    }

    #[test]
    fn test_registry_record_hit() {
        let mut reg = ViewRegistry::new(10);
        let pattern = make_bgp(var("s"), var("p"), var("o"));
        let view = ViewDefinition::new(pattern.clone(), vec![]);
        reg.insert_by_algebra(view);

        reg.record_hit(&pattern);
        reg.record_hit(&pattern);

        let found = reg.find_matching_view(&pattern).expect("should find view");
        assert_eq!(found.hit_count, 2);
    }

    #[test]
    fn test_registry_invalidate_on_update_removes_view() {
        let mut reg = ViewRegistry::new(10);
        let inner = make_bgp(var("s"), var("p"), var("o"));
        let graph_pattern = Algebra::Graph {
            graph: iri("http://ex.org/graph1"),
            pattern: Box::new(inner),
        };
        let view = ViewDefinition::new(graph_pattern, vec![]);
        reg.insert_by_algebra(view);
        assert_eq!(reg.len(), 1);

        reg.invalidate_on_update("http://ex.org/graph1");
        assert_eq!(reg.len(), 0, "view should be invalidated");
    }

    #[test]
    fn test_registry_invalidate_on_update_keeps_unrelated_view() {
        let mut reg = ViewRegistry::new(10);
        let inner = make_bgp(var("s"), var("p"), var("o"));
        let graph_pattern = Algebra::Graph {
            graph: iri("http://ex.org/graph1"),
            pattern: Box::new(inner),
        };
        let view = ViewDefinition::new(graph_pattern, vec![]);
        reg.insert_by_algebra(view);

        // Update a different graph — view should remain.
        reg.invalidate_on_update("http://ex.org/graph2");
        assert_eq!(reg.len(), 1, "unrelated view should be kept");
    }

    #[test]
    fn test_registry_evict_lru_lowest_hit_count() {
        let mut reg = ViewRegistry::new(2);
        // Insert two views.
        let p1 = make_bgp(var("s"), var("p"), var("o"));
        let p2 = make_bgp(var("x"), var("y"), var("z"));
        let mut view1 = ViewDefinition::new(p1.clone(), make_bindings(1));
        view1.hit_count = 10;
        let view2 = ViewDefinition::new(p2.clone(), make_bindings(1));
        // view2 has hit_count = 0.
        reg.insert_by_algebra(view1);
        reg.insert_by_algebra(view2);
        assert_eq!(reg.len(), 2);

        // Evict LRU — should remove view2 (hit_count=0).
        reg.evict_lru();
        assert_eq!(reg.len(), 1);
        // view1 (hit_count=10) should remain.
        assert!(reg.find_matching_view(&p1).is_some());
    }

    #[test]
    fn test_registry_max_size_triggers_eviction() {
        let mut reg = ViewRegistry::new(2);
        for i in 0..5 {
            let p = make_bgp(iri(&format!("http://ex.org/s{i}")), var("p"), var("o"));
            let view = ViewDefinition::new(p, make_bindings(1));
            reg.insert_by_algebra(view);
        }
        // Registry should never exceed max_size.
        assert!(reg.len() <= 2, "registry should respect max_size");
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = ViewRegistry::new(10);
        let pattern = make_bgp(var("s"), var("p"), var("o"));
        let view = ViewDefinition::new(pattern, make_bindings(2));
        reg.insert("my_view".to_string(), view);
        assert_eq!(reg.len(), 1);

        let removed = reg.remove("my_view");
        assert!(removed.is_some());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_remove_nonexistent() {
        let mut reg = ViewRegistry::new(10);
        let result = reg.remove("does_not_exist");
        assert!(result.is_none());
    }

    #[test]
    fn test_registry_clear() {
        let mut reg = ViewRegistry::new(10);
        for i in 0..3 {
            let p = make_bgp(var(&format!("s{i}")), var("p"), var("o"));
            reg.insert_by_algebra(ViewDefinition::new(p, vec![]));
        }
        reg.clear();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_total_hits() {
        let mut reg = ViewRegistry::new(10);
        let p1 = make_bgp(var("s"), var("p"), var("o"));
        let p2 = make_bgp(var("x"), var("y"), var("z"));
        reg.insert_by_algebra(ViewDefinition::new(p1.clone(), vec![]));
        reg.insert_by_algebra(ViewDefinition::new(p2.clone(), vec![]));

        reg.record_hit(&p1);
        reg.record_hit(&p1);
        reg.record_hit(&p2);

        assert_eq!(reg.total_hits(), 3);
    }

    #[test]
    fn test_registry_total_size_bytes() {
        let mut reg = ViewRegistry::new(10);
        let p = make_bgp(var("s"), var("p"), var("o"));
        let view = ViewDefinition::new(p, make_bindings(100));
        let expected_size = view.size_bytes;
        reg.insert_by_algebra(view);
        assert_eq!(reg.total_size_bytes(), expected_size);
    }

    #[test]
    fn test_registry_get_by_id() {
        let mut reg = ViewRegistry::new(10);
        let pattern = make_bgp(var("s"), var("p"), var("o"));
        let view = ViewDefinition::new(pattern, make_bindings(2));
        reg.insert("test_view".to_string(), view);
        let found = reg.get_by_id("test_view");
        assert!(found.is_some());
    }

    #[test]
    fn test_registry_evict_empty() {
        let mut reg = ViewRegistry::new(10);
        // Should not panic on empty registry.
        reg.evict_lru();
        assert!(reg.is_empty());
    }
}
