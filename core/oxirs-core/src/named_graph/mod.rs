//! Named Graph Management API (Jena Dataset-compatible)
//!
//! Provides a comprehensive API for managing named RDF graphs within a dataset.
//! This module extends the basic `Dataset` with Jena-compatible operations,
//! iterators, transactions, and a [`GraphStore`] trait for pluggable backends.
//!
//! # Overview
//!
//! An RDF Dataset consists of:
//! - A **default graph** (unnamed, always present)
//! - Zero or more **named graphs**, each identified by an IRI or blank node
//!
//! This module provides:
//! - [`NamedGraphManager`] – high-level dataset manager with Jena-compatible API
//! - [`GraphStore`] – trait for pluggable triple-store backends
//! - [`NamedGraphIterator`] – typed iterator over `(name, graph)` pairs
//! - [`GraphStatistics`] – statistics about a dataset's contents
//! - [`GraphName`] re-export and helper constructors

use crate::model::{Graph, GraphName, NamedNode, Quad, Triple};
use crate::{OxirsError, Result};
use std::collections::HashMap;

// ── Re-exports ────────────────────────────────────────────────────────────────

/// A named-graph key: either a [`NamedNode`] IRI or blank node name (string)
pub use crate::model::GraphName as GraphNameKey;

// ── NamedGraphManager ────────────────────────────────────────────────────────

/// High-level manager for named RDF graphs within a single dataset
///
/// Provides a Jena-compatible API for creating, querying, and managing
/// named graphs. Each graph is identified by a [`GraphName`] key.
///
/// # Example
///
/// ```rust
/// use oxirs_core::named_graph::NamedGraphManager;
/// use oxirs_core::model::{GraphName, NamedNode, Literal, Triple};
///
/// let mut mgr = NamedGraphManager::new();
/// let graph_iri = NamedNode::new("http://example.org/graph1").unwrap();
/// let key = GraphName::NamedNode(graph_iri.clone());
///
/// // Create a named graph (auto-created on first insert)
/// mgr.add_named_graph(key.clone());
///
/// // Insert a triple into that graph
/// let subj = NamedNode::new("http://example.org/s").unwrap();
/// let pred = NamedNode::new("http://example.org/p").unwrap();
/// let obj  = Literal::new("hello");
/// let triple = Triple::new(subj, pred, obj);
/// mgr.insert_triple(&key, triple).unwrap();
///
/// assert_eq!(mgr.triple_count_in(&key), 1);
/// ```
#[derive(Debug, Clone)]
pub struct NamedGraphManager {
    default_graph: Graph,
    named_graphs: HashMap<GraphName, Graph>,
}

impl NamedGraphManager {
    /// Create a new empty dataset manager
    pub fn new() -> Self {
        NamedGraphManager {
            default_graph: Graph::new(),
            named_graphs: HashMap::new(),
        }
    }

    /// Create a dataset manager with an initial capacity hint for named graphs
    pub fn with_capacity(named_graph_capacity: usize) -> Self {
        NamedGraphManager {
            default_graph: Graph::new(),
            named_graphs: HashMap::with_capacity(named_graph_capacity),
        }
    }

    // ── Default graph ─────────────────────────────────────────────────────────

    /// Returns a shared reference to the default (unnamed) graph
    pub fn default_graph(&self) -> &Graph {
        &self.default_graph
    }

    /// Returns an exclusive reference to the default (unnamed) graph
    pub fn default_graph_mut(&mut self) -> &mut Graph {
        &mut self.default_graph
    }

    // ── Named graph lifecycle ─────────────────────────────────────────────────

    /// Create (or ensure the existence of) a named graph
    ///
    /// Returns `true` if the graph was newly created, `false` if it already existed.
    /// Has no effect on the default graph key.
    pub fn add_named_graph(&mut self, name: GraphName) -> bool {
        if name.is_default_graph() {
            return false;
        }
        if self.named_graphs.contains_key(&name) {
            return false;
        }
        self.named_graphs.insert(name, Graph::new());
        true
    }

    /// Remove a named graph from the dataset
    ///
    /// Returns the removed [`Graph`] if it existed, `None` otherwise.
    /// Calling this with the default graph key clears the default graph and
    /// returns the old contents.
    pub fn remove_named_graph(&mut self, name: &GraphName) -> Option<Graph> {
        if name.is_default_graph() {
            let mut old = Graph::new();
            std::mem::swap(&mut old, &mut self.default_graph);
            Some(old)
        } else {
            self.named_graphs.remove(name)
        }
    }

    /// Returns a shared reference to the named graph with the given key
    ///
    /// Returns `None` if no such graph exists. Use the default-graph key to
    /// access the default graph.
    pub fn get_named_graph(&self, name: &GraphName) -> Option<&Graph> {
        if name.is_default_graph() {
            Some(&self.default_graph)
        } else {
            self.named_graphs.get(name)
        }
    }

    /// Returns an exclusive reference to the named graph with the given key,
    /// creating an empty graph entry if it does not exist yet.
    pub fn get_or_create_named_graph_mut(&mut self, name: &GraphName) -> &mut Graph {
        if name.is_default_graph() {
            &mut self.default_graph
        } else {
            self.named_graphs.entry(name.clone()).or_default()
        }
    }

    /// Returns an exclusive reference to an existing named graph
    ///
    /// Returns `None` if the graph does not exist (unlike
    /// `get_or_create_named_graph_mut`).
    pub fn get_named_graph_mut(&mut self, name: &GraphName) -> Option<&mut Graph> {
        if name.is_default_graph() {
            Some(&mut self.default_graph)
        } else {
            self.named_graphs.get_mut(name)
        }
    }

    /// Returns `true` if a named graph with the given key exists
    pub fn contains_named_graph(&self, name: &GraphName) -> bool {
        if name.is_default_graph() {
            true // default graph always exists
        } else {
            self.named_graphs.contains_key(name)
        }
    }

    /// Returns an iterator over the names of all named graphs (excluding default)
    pub fn list_named_graphs(&self) -> impl Iterator<Item = &GraphName> {
        self.named_graphs.keys()
    }

    /// Returns the number of named graphs (excluding the default graph)
    pub fn graph_count(&self) -> usize {
        self.named_graphs.len()
    }

    /// Returns the total number of graphs including the default graph
    pub fn total_graph_count(&self) -> usize {
        self.named_graphs.len() + 1
    }

    // ── Triple operations ─────────────────────────────────────────────────────

    /// Insert a triple into the specified graph
    ///
    /// If the graph does not exist it is created automatically.
    /// Returns `true` if the triple was newly inserted.
    pub fn insert_triple(&mut self, graph: &GraphName, triple: Triple) -> Result<bool> {
        let target = self.get_or_create_named_graph_mut(graph);
        Ok(target.insert(triple))
    }

    /// Remove a triple from the specified graph
    ///
    /// Returns `true` if the triple was present and removed.
    /// Returns an error if the graph does not exist.
    pub fn remove_triple(&mut self, graph: &GraphName, triple: &Triple) -> Result<bool> {
        let target = self
            .get_named_graph_mut(graph)
            .ok_or_else(|| OxirsError::Store(format!("Graph does not exist: {graph}")))?;
        Ok(target.remove(triple))
    }

    /// Returns `true` if the specified graph contains the given triple
    pub fn contains_triple(&self, graph: &GraphName, triple: &Triple) -> bool {
        self.get_named_graph(graph)
            .is_some_and(|g| g.contains(triple))
    }

    /// Returns an owned `Vec` of all triples in the specified graph
    ///
    /// Returns an empty vector if the graph does not exist.
    pub fn triples_in_graph(&self, graph: &GraphName) -> Vec<Triple> {
        match self.get_named_graph(graph) {
            Some(g) => g.iter().cloned().collect(),
            None => vec![],
        }
    }

    /// Returns the number of triples in the specified graph
    pub fn triple_count_in(&self, graph: &GraphName) -> usize {
        self.get_named_graph(graph).map_or(0, Graph::len)
    }

    /// Returns the total number of triples across all graphs (named + default)
    pub fn triple_count(&self) -> usize {
        let named_count: usize = self.named_graphs.values().map(Graph::len).sum();
        named_count + self.default_graph.len()
    }

    /// Remove all triples from the specified graph (graph itself remains)
    ///
    /// Returns the number of triples removed.
    pub fn clear_graph(&mut self, graph: &GraphName) -> Result<usize> {
        let target = self
            .get_named_graph_mut(graph)
            .ok_or_else(|| OxirsError::Store(format!("Graph does not exist: {graph}")))?;
        let count = target.len();
        target.clear();
        Ok(count)
    }

    /// Remove all triples from the specified graph and delete the graph entry
    ///
    /// Returns `true` if the graph existed. For the default graph this clears
    /// its contents but does not destroy the graph slot.
    pub fn drop_graph(&mut self, graph: &GraphName) -> bool {
        if graph.is_default_graph() {
            let was_empty = self.default_graph.is_empty();
            self.default_graph.clear();
            !was_empty
        } else {
            self.named_graphs.remove(graph).is_some()
        }
    }

    // ── Union / aggregation operations ────────────────────────────────────────

    /// Build the union graph: all triples from all named graphs merged together
    ///
    /// The returned [`Graph`] is a fresh copy; modifications do not affect
    /// the original graphs.
    pub fn union_graph(&self) -> Graph {
        let mut union = self.default_graph.clone();
        for graph in self.named_graphs.values() {
            for triple in graph.iter() {
                union.insert(triple.clone());
            }
        }
        union
    }

    /// Merge all named graphs (but NOT the default graph) into a union graph
    ///
    /// Useful when implementing the SPARQL `UNION DEFAULT GRAPH` feature.
    pub fn union_of_named_graphs(&self) -> Graph {
        let mut union = Graph::new();
        for graph in self.named_graphs.values() {
            for triple in graph.iter() {
                union.insert(triple.clone());
            }
        }
        union
    }

    // ── Quad operations ───────────────────────────────────────────────────────

    /// Insert a [`Quad`] (routing it to the appropriate named graph)
    pub fn insert_quad(&mut self, quad: Quad) -> Result<bool> {
        let triple = quad.to_triple();
        let graph_name = quad.graph_name().clone();
        self.insert_triple(&graph_name, triple)
    }

    /// Returns an iterator over all quads in the dataset
    pub fn iter_quads(&self) -> impl Iterator<Item = Quad> + '_ {
        let default_quads = self
            .default_graph
            .iter()
            .map(|t| Quad::from_triple(t.clone()));
        let named_quads = self.named_graphs.iter().flat_map(|(name, graph)| {
            let name = name.clone();
            graph
                .iter()
                .map(move |t| Quad::from_triple_in_graph(t.clone(), name.clone()))
        });
        default_quads.chain(named_quads)
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Compute statistics for this dataset
    pub fn statistics(&self) -> GraphStatistics {
        let named_triple_counts: HashMap<GraphName, usize> = self
            .named_graphs
            .iter()
            .map(|(name, g)| (name.clone(), g.len()))
            .collect();

        let largest_graph = self
            .named_graphs
            .iter()
            .max_by_key(|(_, g)| g.len())
            .map(|(n, _)| n.clone());

        GraphStatistics {
            named_graph_count: self.named_graphs.len(),
            default_graph_triple_count: self.default_graph.len(),
            named_triple_counts,
            total_triple_count: self.triple_count(),
            largest_named_graph: largest_graph,
        }
    }

    // ── Bulk operations ───────────────────────────────────────────────────────

    /// Load all triples from an iterator into the given named graph
    ///
    /// Returns the number of triples inserted (deduplicated).
    pub fn bulk_insert<I>(&mut self, graph: &GraphName, triples: I) -> usize
    where
        I: IntoIterator<Item = Triple>,
    {
        let target = self.get_or_create_named_graph_mut(graph);
        let before = target.len();
        target.extend(triples);
        target.len() - before
    }

    /// Copy all triples from `source` graph into `destination` graph
    ///
    /// Returns the number of triples copied (may be fewer if destination
    /// already contains some of them).
    pub fn copy_graph(&mut self, source: &GraphName, destination: &GraphName) -> Result<usize> {
        // Collect source triples first to avoid borrow conflict
        let source_triples: Vec<Triple> = match self.get_named_graph(source) {
            Some(g) => g.iter().cloned().collect(),
            None => {
                return Err(OxirsError::Store(format!(
                    "Source graph does not exist: {source}"
                )))
            }
        };
        Ok(self.bulk_insert(destination, source_triples))
    }

    /// Move all triples from `source` graph into `destination` graph
    ///
    /// The source graph is emptied (but kept) after the operation.
    /// Returns the number of triples moved.
    pub fn move_graph(&mut self, source: &GraphName, destination: &GraphName) -> Result<usize> {
        // Drain source
        let source_triples: Vec<Triple> = {
            let src = self.get_named_graph_mut(source).ok_or_else(|| {
                OxirsError::Store(format!("Source graph does not exist: {source}"))
            })?;
            let triples: Vec<_> = src.iter().cloned().collect();
            src.clear();
            triples
        };
        Ok(self.bulk_insert(destination, source_triples))
    }
}

impl Default for NamedGraphManager {
    fn default() -> Self {
        Self::new()
    }
}

// ── GraphStatistics ───────────────────────────────────────────────────────────

/// Statistics collected from a [`NamedGraphManager`]
#[derive(Debug, Clone)]
pub struct GraphStatistics {
    /// Number of named graphs (excluding default)
    pub named_graph_count: usize,
    /// Number of triples in the default graph
    pub default_graph_triple_count: usize,
    /// Per-graph triple counts
    pub named_triple_counts: HashMap<GraphName, usize>,
    /// Total triples across all graphs
    pub total_triple_count: usize,
    /// The name of the named graph with the most triples
    pub largest_named_graph: Option<GraphName>,
}

impl GraphStatistics {
    /// Returns the average triple count across named graphs (0.0 if none)
    pub fn average_named_graph_size(&self) -> f64 {
        if self.named_graph_count == 0 {
            return 0.0;
        }
        let total: usize = self.named_triple_counts.values().sum();
        total as f64 / self.named_graph_count as f64
    }

    /// Returns `true` if the dataset is completely empty
    pub fn is_empty(&self) -> bool {
        self.total_triple_count == 0
    }
}

// ── NamedGraphIterator ────────────────────────────────────────────────────────

/// An owned iterator over `(GraphName, Graph)` pairs from a [`NamedGraphManager`]
pub struct NamedGraphIterator {
    inner: std::vec::IntoIter<(GraphName, Graph)>,
}

impl NamedGraphIterator {
    /// Create an iterator from a [`NamedGraphManager`], consuming the named graphs
    pub fn from_manager(manager: NamedGraphManager) -> Self {
        let pairs: Vec<(GraphName, Graph)> = manager.named_graphs.into_iter().collect();
        NamedGraphIterator {
            inner: pairs.into_iter(),
        }
    }
}

impl Iterator for NamedGraphIterator {
    type Item = (GraphName, Graph);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for NamedGraphIterator {}

// ── GraphStore trait ──────────────────────────────────────────────────────────

/// Trait for pluggable named-graph triple-store backends
///
/// Implementors provide the six core operations needed to manage triples
/// within named graphs. See [`NamedGraphManager`] for an in-memory implementation.
pub trait GraphStore {
    /// Add a triple to the given graph
    ///
    /// Returns `Ok(true)` if the triple was newly inserted.
    fn add_triple(&mut self, graph: &GraphName, triple: Triple) -> Result<bool>;

    /// Remove a triple from the given graph
    ///
    /// Returns `Ok(true)` if the triple was present.
    fn remove_triple(&mut self, graph: &GraphName, triple: &Triple) -> Result<bool>;

    /// Returns `true` if the graph contains the triple
    fn contains_triple(&self, graph: &GraphName, triple: &Triple) -> Result<bool>;

    /// Returns an owned list of all triples in the graph
    fn triples_in_graph(&self, graph: &GraphName) -> Result<Vec<Triple>>;

    /// Remove all triples from the graph (graph entry is kept)
    ///
    /// Returns the number of triples removed.
    fn clear_graph(&mut self, graph: &GraphName) -> Result<usize>;

    /// Delete the graph and all its triples
    ///
    /// Returns `Ok(true)` if the graph existed.
    fn drop_graph(&mut self, graph: &GraphName) -> Result<bool>;
}

/// In-memory implementation of [`GraphStore`] backed by [`NamedGraphManager`]
impl GraphStore for NamedGraphManager {
    fn add_triple(&mut self, graph: &GraphName, triple: Triple) -> Result<bool> {
        self.insert_triple(graph, triple)
    }

    fn remove_triple(&mut self, graph: &GraphName, triple: &Triple) -> Result<bool> {
        self.remove_triple(graph, triple)
    }

    fn contains_triple(&self, graph: &GraphName, triple: &Triple) -> Result<bool> {
        Ok(self.contains_triple(graph, triple))
    }

    fn triples_in_graph(&self, graph: &GraphName) -> Result<Vec<Triple>> {
        Ok(self.triples_in_graph(graph))
    }

    fn clear_graph(&mut self, graph: &GraphName) -> Result<usize> {
        self.clear_graph(graph)
    }

    fn drop_graph(&mut self, graph: &GraphName) -> Result<bool> {
        Ok(self.drop_graph(graph))
    }
}

// ── Helper constructors ───────────────────────────────────────────────────────

/// Convenience function: create a [`GraphName`] key from an IRI string
///
/// Returns `Err` if the IRI is invalid.
pub fn graph_name_from_iri(iri: &str) -> Result<GraphName> {
    let node = NamedNode::new(iri).map_err(|e| OxirsError::Parse(e.to_string()))?;
    Ok(GraphName::NamedNode(node))
}

/// Convenience function: create a default-graph [`GraphName`] key
pub fn default_graph_name() -> GraphName {
    GraphName::DefaultGraph
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode, Triple};

    // ── Helper constructors ───────────────────────────────────────────────────

    fn iri(s: &str) -> NamedNode {
        NamedNode::new(s).expect("valid IRI")
    }

    fn graph_key(s: &str) -> GraphName {
        GraphName::NamedNode(iri(s))
    }

    fn triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(iri(s), iri(p), Literal::new(o))
    }

    // ── NamedGraphManager::new / default ─────────────────────────────────────

    #[test]
    fn test_new_manager_is_empty() {
        let mgr = NamedGraphManager::new();
        assert_eq!(mgr.graph_count(), 0);
        assert_eq!(mgr.triple_count(), 0);
    }

    #[test]
    fn test_default_manager_is_empty() {
        let mgr = NamedGraphManager::default();
        assert_eq!(mgr.graph_count(), 0);
    }

    #[test]
    fn test_with_capacity_starts_empty() {
        let mgr = NamedGraphManager::with_capacity(32);
        assert_eq!(mgr.graph_count(), 0);
        assert!(mgr.triple_count() == 0);
    }

    #[test]
    fn test_total_graph_count_includes_default() {
        let mgr = NamedGraphManager::new();
        assert_eq!(mgr.total_graph_count(), 1); // only default
    }

    // ── add_named_graph ───────────────────────────────────────────────────────

    #[test]
    fn test_add_named_graph_returns_true_on_creation() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        assert!(mgr.add_named_graph(key));
    }

    #[test]
    fn test_add_named_graph_returns_false_if_already_exists() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        mgr.add_named_graph(key.clone());
        assert!(!mgr.add_named_graph(key));
    }

    #[test]
    fn test_add_named_graph_increments_count() {
        let mut mgr = NamedGraphManager::new();
        mgr.add_named_graph(graph_key("http://example.org/g1"));
        assert_eq!(mgr.graph_count(), 1);
    }

    #[test]
    fn test_add_default_graph_key_is_noop() {
        let mut mgr = NamedGraphManager::new();
        let result = mgr.add_named_graph(GraphName::DefaultGraph);
        assert!(!result);
        assert_eq!(mgr.graph_count(), 0);
    }

    #[test]
    fn test_add_multiple_named_graphs() {
        let mut mgr = NamedGraphManager::new();
        for i in 0..5 {
            mgr.add_named_graph(graph_key(&format!("http://example.org/g{i}")));
        }
        assert_eq!(mgr.graph_count(), 5);
        assert_eq!(mgr.total_graph_count(), 6);
    }

    // ── contains_named_graph ──────────────────────────────────────────────────

    #[test]
    fn test_contains_named_graph_true_after_add() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        mgr.add_named_graph(key.clone());
        assert!(mgr.contains_named_graph(&key));
    }

    #[test]
    fn test_contains_named_graph_false_before_add() {
        let mgr = NamedGraphManager::new();
        assert!(!mgr.contains_named_graph(&graph_key("http://example.org/missing")));
    }

    #[test]
    fn test_contains_default_graph_always_true() {
        let mgr = NamedGraphManager::new();
        assert!(mgr.contains_named_graph(&GraphName::DefaultGraph));
    }

    // ── get_named_graph ───────────────────────────────────────────────────────

    #[test]
    fn test_get_named_graph_returns_some_after_add() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        mgr.add_named_graph(key.clone());
        assert!(mgr.get_named_graph(&key).is_some());
    }

    #[test]
    fn test_get_named_graph_returns_none_if_absent() {
        let mgr = NamedGraphManager::new();
        assert!(mgr
            .get_named_graph(&graph_key("http://example.org/absent"))
            .is_none());
    }

    #[test]
    fn test_get_default_graph_always_some() {
        let mgr = NamedGraphManager::new();
        assert!(mgr.get_named_graph(&GraphName::DefaultGraph).is_some());
    }

    // ── remove_named_graph ────────────────────────────────────────────────────

    #[test]
    fn test_remove_named_graph_returns_some_if_existed() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        mgr.add_named_graph(key.clone());
        assert!(mgr.remove_named_graph(&key).is_some());
    }

    #[test]
    fn test_remove_named_graph_returns_none_if_absent() {
        let mut mgr = NamedGraphManager::new();
        assert!(mgr
            .remove_named_graph(&graph_key("http://example.org/absent"))
            .is_none());
    }

    #[test]
    fn test_remove_named_graph_decrements_count() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        mgr.add_named_graph(key.clone());
        mgr.remove_named_graph(&key);
        assert_eq!(mgr.graph_count(), 0);
    }

    #[test]
    fn test_remove_default_graph_clears_contents() {
        let mut mgr = NamedGraphManager::new();
        let t = triple("http://example.org/s", "http://example.org/p", "hello");
        mgr.insert_triple(&GraphName::DefaultGraph, t).unwrap();
        assert_eq!(mgr.default_graph().len(), 1);
        let removed = mgr.remove_named_graph(&GraphName::DefaultGraph);
        assert!(removed.is_some());
        assert_eq!(mgr.default_graph().len(), 0);
    }

    // ── insert_triple / contains_triple ──────────────────────────────────────

    #[test]
    fn test_insert_triple_returns_true_on_new() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        assert!(mgr.insert_triple(&key, t).unwrap());
    }

    #[test]
    fn test_insert_triple_returns_false_on_duplicate() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        mgr.insert_triple(&key, t.clone()).unwrap();
        assert!(!mgr.insert_triple(&key, t).unwrap());
    }

    #[test]
    fn test_insert_triple_auto_creates_graph() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        assert!(!mgr.contains_named_graph(&key));
        mgr.insert_triple(&key, t).unwrap();
        assert!(mgr.contains_named_graph(&key));
    }

    #[test]
    fn test_contains_triple_true_after_insert() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        mgr.insert_triple(&key, t.clone()).unwrap();
        assert!(mgr.contains_triple(&key, &t));
    }

    #[test]
    fn test_contains_triple_false_in_wrong_graph() {
        let mut mgr = NamedGraphManager::new();
        let key1 = graph_key("http://example.org/g1");
        let key2 = graph_key("http://example.org/g2");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        mgr.insert_triple(&key1, t.clone()).unwrap();
        assert!(!mgr.contains_triple(&key2, &t));
    }

    #[test]
    fn test_triple_count_in_named_graph() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        for i in 0..7 {
            let t = triple(&format!("http://s.org/s{i}"), "http://p.org/p", "o");
            mgr.insert_triple(&key, t).unwrap();
        }
        assert_eq!(mgr.triple_count_in(&key), 7);
    }

    #[test]
    fn test_triple_count_total_across_graphs() {
        let mut mgr = NamedGraphManager::new();
        for g in 0..3 {
            let key = graph_key(&format!("http://example.org/g{g}"));
            for i in 0..4 {
                let t = triple(&format!("http://s.org/s{i}"), "http://p.org/p", "o");
                mgr.insert_triple(&key, t).unwrap();
            }
        }
        assert_eq!(mgr.triple_count(), 12);
    }

    // ── remove_triple ─────────────────────────────────────────────────────────

    #[test]
    fn test_remove_triple_returns_true_if_present() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        mgr.insert_triple(&key, t.clone()).unwrap();
        assert!(mgr.remove_triple(&key, &t).unwrap());
    }

    #[test]
    fn test_remove_triple_returns_false_if_absent() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        mgr.add_named_graph(key.clone());
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        assert!(!mgr.remove_triple(&key, &t).unwrap());
    }

    #[test]
    fn test_remove_triple_from_nonexistent_graph_is_error() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/absent");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        assert!(mgr.remove_triple(&key, &t).is_err());
    }

    #[test]
    fn test_remove_triple_decrements_count() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        mgr.insert_triple(&key, t.clone()).unwrap();
        mgr.remove_triple(&key, &t).unwrap();
        assert_eq!(mgr.triple_count_in(&key), 0);
    }

    // ── triples_in_graph ──────────────────────────────────────────────────────

    #[test]
    fn test_triples_in_graph_returns_all() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t1 = triple("http://s.org/s1", "http://p.org/p", "o1");
        let t2 = triple("http://s.org/s2", "http://p.org/p", "o2");
        mgr.insert_triple(&key, t1.clone()).unwrap();
        mgr.insert_triple(&key, t2.clone()).unwrap();
        let triples = mgr.triples_in_graph(&key);
        assert_eq!(triples.len(), 2);
        assert!(triples.contains(&t1));
        assert!(triples.contains(&t2));
    }

    #[test]
    fn test_triples_in_nonexistent_graph_is_empty() {
        let mgr = NamedGraphManager::new();
        assert!(mgr
            .triples_in_graph(&graph_key("http://example.org/absent"))
            .is_empty());
    }

    // ── clear_graph ───────────────────────────────────────────────────────────

    #[test]
    fn test_clear_graph_removes_all_triples() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        mgr.insert_triple(&key, t).unwrap();
        let cleared = mgr.clear_graph(&key).unwrap();
        assert_eq!(cleared, 1);
        assert_eq!(mgr.triple_count_in(&key), 0);
    }

    #[test]
    fn test_clear_graph_keeps_graph_entry() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        mgr.add_named_graph(key.clone());
        mgr.clear_graph(&key).unwrap();
        assert!(mgr.contains_named_graph(&key));
    }

    #[test]
    fn test_clear_nonexistent_graph_is_error() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/absent");
        assert!(mgr.clear_graph(&key).is_err());
    }

    // ── drop_graph ────────────────────────────────────────────────────────────

    #[test]
    fn test_drop_graph_removes_graph_entry() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        mgr.add_named_graph(key.clone());
        assert!(mgr.drop_graph(&key));
        assert!(!mgr.contains_named_graph(&key));
    }

    #[test]
    fn test_drop_graph_returns_false_if_absent() {
        let mut mgr = NamedGraphManager::new();
        assert!(!mgr.drop_graph(&graph_key("http://example.org/absent")));
    }

    #[test]
    fn test_drop_default_graph_clears_triples() {
        let mut mgr = NamedGraphManager::new();
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        mgr.insert_triple(&GraphName::DefaultGraph, t).unwrap();
        let result = mgr.drop_graph(&GraphName::DefaultGraph);
        assert!(result);
        assert_eq!(mgr.default_graph().len(), 0);
    }

    // ── union_graph ───────────────────────────────────────────────────────────

    #[test]
    fn test_union_graph_merges_all_triples() {
        let mut mgr = NamedGraphManager::new();
        let k1 = graph_key("http://example.org/g1");
        let k2 = graph_key("http://example.org/g2");
        let t1 = triple("http://s.org/s1", "http://p.org/p", "o1");
        let t2 = triple("http://s.org/s2", "http://p.org/p", "o2");
        mgr.insert_triple(&k1, t1.clone()).unwrap();
        mgr.insert_triple(&k2, t2.clone()).unwrap();
        let union = mgr.union_graph();
        assert!(union.contains(&t1));
        assert!(union.contains(&t2));
        assert_eq!(union.len(), 2);
    }

    #[test]
    fn test_union_graph_includes_default_graph() {
        let mut mgr = NamedGraphManager::new();
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        mgr.insert_triple(&GraphName::DefaultGraph, t.clone())
            .unwrap();
        let union = mgr.union_graph();
        assert!(union.contains(&t));
    }

    #[test]
    fn test_union_graph_deduplicates() {
        let mut mgr = NamedGraphManager::new();
        let k1 = graph_key("http://example.org/g1");
        let k2 = graph_key("http://example.org/g2");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        mgr.insert_triple(&k1, t.clone()).unwrap();
        mgr.insert_triple(&k2, t.clone()).unwrap();
        let union = mgr.union_graph();
        assert_eq!(union.len(), 1);
    }

    #[test]
    fn test_union_of_named_graphs_excludes_default() {
        let mut mgr = NamedGraphManager::new();
        let k1 = graph_key("http://example.org/g1");
        let t_named = triple("http://s.org/named", "http://p.org/p", "o");
        let t_default = triple("http://s.org/default", "http://p.org/p", "o");
        mgr.insert_triple(&k1, t_named.clone()).unwrap();
        mgr.insert_triple(&GraphName::DefaultGraph, t_default.clone())
            .unwrap();
        let union = mgr.union_of_named_graphs();
        assert!(union.contains(&t_named));
        assert!(!union.contains(&t_default));
    }

    // ── list_named_graphs ─────────────────────────────────────────────────────

    #[test]
    fn test_list_named_graphs_empty_when_none() {
        let mgr = NamedGraphManager::new();
        assert_eq!(mgr.list_named_graphs().count(), 0);
    }

    #[test]
    fn test_list_named_graphs_returns_correct_count() {
        let mut mgr = NamedGraphManager::new();
        for i in 0..4 {
            mgr.add_named_graph(graph_key(&format!("http://example.org/g{i}")));
        }
        assert_eq!(mgr.list_named_graphs().count(), 4);
    }

    #[test]
    fn test_list_named_graphs_contains_added_keys() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/myGraph");
        mgr.add_named_graph(key.clone());
        let names: Vec<_> = mgr.list_named_graphs().collect();
        assert!(names.contains(&&key));
    }

    // ── bulk operations ───────────────────────────────────────────────────────

    #[test]
    fn test_bulk_insert_returns_inserted_count() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let triples: Vec<Triple> = (0..5)
            .map(|i| triple(&format!("http://s.org/s{i}"), "http://p.org/p", "o"))
            .collect();
        let count = mgr.bulk_insert(&key, triples);
        assert_eq!(count, 5);
    }

    #[test]
    fn test_bulk_insert_deduplicates() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        mgr.bulk_insert(&key, vec![t.clone(), t]);
        assert_eq!(mgr.triple_count_in(&key), 1);
    }

    #[test]
    fn test_copy_graph_copies_all_triples() {
        let mut mgr = NamedGraphManager::new();
        let src = graph_key("http://example.org/src");
        let dst = graph_key("http://example.org/dst");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        mgr.insert_triple(&src, t.clone()).unwrap();
        mgr.copy_graph(&src, &dst).unwrap();
        assert!(mgr.contains_triple(&dst, &t));
        assert!(mgr.contains_triple(&src, &t)); // source not cleared
    }

    #[test]
    fn test_copy_nonexistent_graph_is_error() {
        let mut mgr = NamedGraphManager::new();
        let src = graph_key("http://example.org/absent");
        let dst = graph_key("http://example.org/dst");
        assert!(mgr.copy_graph(&src, &dst).is_err());
    }

    #[test]
    fn test_move_graph_empties_source() {
        let mut mgr = NamedGraphManager::new();
        let src = graph_key("http://example.org/src");
        let dst = graph_key("http://example.org/dst");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        mgr.insert_triple(&src, t.clone()).unwrap();
        mgr.move_graph(&src, &dst).unwrap();
        assert_eq!(mgr.triple_count_in(&src), 0);
        assert!(mgr.contains_triple(&dst, &t));
    }

    #[test]
    fn test_move_nonexistent_graph_is_error() {
        let mut mgr = NamedGraphManager::new();
        let src = graph_key("http://example.org/absent");
        let dst = graph_key("http://example.org/dst");
        assert!(mgr.move_graph(&src, &dst).is_err());
    }

    // ── insert_quad ───────────────────────────────────────────────────────────

    #[test]
    fn test_insert_quad_into_named_graph() {
        let mut mgr = NamedGraphManager::new();
        let graph_node = iri("http://example.org/g1");
        let quad = Quad::new(
            iri("http://s.org/s"),
            iri("http://p.org/p"),
            Literal::new("o"),
            graph_node,
        );
        mgr.insert_quad(quad).unwrap();
        assert_eq!(mgr.triple_count(), 1);
    }

    #[test]
    fn test_insert_quad_into_default_graph() {
        let mut mgr = NamedGraphManager::new();
        let quad = Quad::new_default_graph(
            iri("http://s.org/s"),
            iri("http://p.org/p"),
            Literal::new("o"),
        );
        mgr.insert_quad(quad).unwrap();
        assert_eq!(mgr.default_graph().len(), 1);
    }

    // ── iter_quads ────────────────────────────────────────────────────────────

    #[test]
    fn test_iter_quads_covers_all_graphs() {
        let mut mgr = NamedGraphManager::new();
        let k1 = graph_key("http://example.org/g1");
        let k2 = graph_key("http://example.org/g2");
        mgr.insert_triple(&k1, triple("http://s.org/s1", "http://p.org/p", "o"))
            .unwrap();
        mgr.insert_triple(&k2, triple("http://s.org/s2", "http://p.org/p", "o"))
            .unwrap();
        mgr.insert_triple(
            &GraphName::DefaultGraph,
            triple("http://s.org/s3", "http://p.org/p", "o"),
        )
        .unwrap();
        let count = mgr.iter_quads().count();
        assert_eq!(count, 3);
    }

    // ── GraphStatistics ───────────────────────────────────────────────────────

    #[test]
    fn test_statistics_empty_dataset() {
        let mgr = NamedGraphManager::new();
        let stats = mgr.statistics();
        assert_eq!(stats.named_graph_count, 0);
        assert_eq!(stats.total_triple_count, 0);
        assert!(stats.is_empty());
    }

    #[test]
    fn test_statistics_named_graph_count() {
        let mut mgr = NamedGraphManager::new();
        for i in 0..3 {
            mgr.add_named_graph(graph_key(&format!("http://example.org/g{i}")));
        }
        assert_eq!(mgr.statistics().named_graph_count, 3);
    }

    #[test]
    fn test_statistics_total_triple_count() {
        let mut mgr = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        for i in 0..6 {
            mgr.insert_triple(
                &key,
                triple(&format!("http://s.org/s{i}"), "http://p.org/p", "o"),
            )
            .unwrap();
        }
        assert_eq!(mgr.statistics().total_triple_count, 6);
    }

    #[test]
    fn test_statistics_largest_named_graph() {
        let mut mgr = NamedGraphManager::new();
        let big = graph_key("http://example.org/big");
        let small = graph_key("http://example.org/small");
        for i in 0..10 {
            mgr.insert_triple(
                &big,
                triple(&format!("http://s.org/s{i}"), "http://p.org/p", "o"),
            )
            .unwrap();
        }
        mgr.insert_triple(&small, triple("http://s.org/s", "http://p.org/p", "o"))
            .unwrap();
        let stats = mgr.statistics();
        assert_eq!(stats.largest_named_graph, Some(big));
    }

    #[test]
    fn test_statistics_average_size_zero_when_no_named_graphs() {
        let mgr = NamedGraphManager::new();
        assert_eq!(mgr.statistics().average_named_graph_size(), 0.0);
    }

    #[test]
    fn test_statistics_average_named_graph_size() {
        let mut mgr = NamedGraphManager::new();
        let k1 = graph_key("http://example.org/g1");
        let k2 = graph_key("http://example.org/g2");
        for i in 0..4 {
            mgr.insert_triple(
                &k1,
                triple(&format!("http://s.org/s{i}"), "http://p.org/p", "o"),
            )
            .unwrap();
        }
        for i in 0..2 {
            mgr.insert_triple(
                &k2,
                triple(&format!("http://s.org/s{i}"), "http://p.org/p", "x"),
            )
            .unwrap();
        }
        // avg = (4 + 2) / 2 = 3.0
        assert!((mgr.statistics().average_named_graph_size() - 3.0).abs() < 1e-10);
    }

    // ── NamedGraphIterator ────────────────────────────────────────────────────

    #[test]
    fn test_named_graph_iterator_yields_all_graphs() {
        let mut mgr = NamedGraphManager::new();
        for i in 0..4 {
            mgr.add_named_graph(graph_key(&format!("http://example.org/g{i}")));
        }
        let iter = NamedGraphIterator::from_manager(mgr);
        assert_eq!(iter.count(), 4);
    }

    #[test]
    fn test_named_graph_iterator_size_hint() {
        let mut mgr = NamedGraphManager::new();
        for i in 0..3 {
            mgr.add_named_graph(graph_key(&format!("http://example.org/g{i}")));
        }
        let iter = NamedGraphIterator::from_manager(mgr);
        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 3);
        assert_eq!(upper, Some(3));
    }

    // ── GraphStore trait ──────────────────────────────────────────────────────

    #[test]
    fn test_graph_store_add_triple() {
        let mut store: Box<dyn GraphStore> = Box::new(NamedGraphManager::new());
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        assert!(store.add_triple(&key, t).unwrap());
    }

    #[test]
    fn test_graph_store_contains_triple() {
        let mut store = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        GraphStore::add_triple(&mut store, &key, t.clone()).unwrap();
        assert!(GraphStore::contains_triple(&store, &key, &t).unwrap());
    }

    #[test]
    fn test_graph_store_remove_triple() {
        let mut store = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        GraphStore::add_triple(&mut store, &key, t.clone()).unwrap();
        assert!(GraphStore::remove_triple(&mut store, &key, &t).unwrap());
    }

    #[test]
    fn test_graph_store_triples_in_graph() {
        let mut store = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        GraphStore::add_triple(&mut store, &key, t.clone()).unwrap();
        let triples = GraphStore::triples_in_graph(&store, &key).unwrap();
        assert_eq!(triples.len(), 1);
        assert!(triples.contains(&t));
    }

    #[test]
    fn test_graph_store_clear_graph() {
        let mut store = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        let t = triple("http://s.org/s", "http://p.org/p", "o");
        GraphStore::add_triple(&mut store, &key, t).unwrap();
        let cleared = GraphStore::clear_graph(&mut store, &key).unwrap();
        assert_eq!(cleared, 1);
    }

    #[test]
    fn test_graph_store_drop_graph() {
        let mut store = NamedGraphManager::new();
        let key = graph_key("http://example.org/g1");
        store.add_named_graph(key.clone());
        assert!(GraphStore::drop_graph(&mut store, &key).unwrap());
    }

    // ── Helper functions ──────────────────────────────────────────────────────

    #[test]
    fn test_graph_name_from_iri_valid() {
        let key = graph_name_from_iri("http://example.org/g1");
        assert!(key.is_ok());
        assert!(matches!(key.unwrap(), GraphName::NamedNode(_)));
    }

    #[test]
    fn test_graph_name_from_iri_invalid() {
        let key = graph_name_from_iri("not a valid IRI!!!");
        assert!(key.is_err());
    }

    #[test]
    fn test_default_graph_name_is_default() {
        let key = default_graph_name();
        assert!(key.is_default_graph());
    }
}
