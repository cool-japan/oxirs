//! In-memory named graphs RDF dataset.
//!
//! Provides [`RdfDataset`] and [`NamedGraph`] for storing and querying RDF triples
//! across multiple named graphs, including a default (unnamed) graph.

use std::collections::HashMap;

/// A single named (or default) graph holding RDF triples.
///
/// Each triple is represented as `(subject, predicate, object)` strings.
#[derive(Debug, Clone)]
pub struct NamedGraph {
    /// The name of this graph, or `None` for the default graph.
    pub name: Option<String>,
    triples: Vec<(String, String, String)>,
}

impl NamedGraph {
    /// Create a new, empty named graph.
    pub fn new(name: Option<String>) -> Self {
        Self {
            name,
            triples: Vec::new(),
        }
    }

    /// Add a triple to this graph.
    pub fn add(&mut self, s: String, p: String, o: String) {
        self.triples.push((s, p, o));
    }

    /// Remove the first matching triple from this graph.
    ///
    /// Returns `true` if a triple was removed.
    pub fn remove(&mut self, s: &str, p: &str, o: &str) -> bool {
        if let Some(pos) = self
            .triples
            .iter()
            .position(|(ts, tp, to)| ts == s && tp == p && to == o)
        {
            self.triples.remove(pos);
            true
        } else {
            false
        }
    }

    /// Returns `true` if this graph contains a triple matching all three components.
    pub fn contains(&self, s: &str, p: &str, o: &str) -> bool {
        self.triples
            .iter()
            .any(|(ts, tp, to)| ts == s && tp == p && to == o)
    }

    /// Iterate over all triples in this graph.
    pub fn iter(&self) -> impl Iterator<Item = &(String, String, String)> {
        self.triples.iter()
    }

    /// Number of triples in this graph.
    pub fn len(&self) -> usize {
        self.triples.len()
    }

    /// Returns `true` if this graph has no triples.
    pub fn is_empty(&self) -> bool {
        self.triples.is_empty()
    }
}

/// An in-memory RDF dataset consisting of a default graph and zero or more named graphs.
///
/// Graph identity is determined by the graph name (`None` = default graph).
#[derive(Debug, Clone, Default)]
pub struct RdfDataset {
    /// Map from graph name to [`NamedGraph`].
    graphs: HashMap<Option<String>, NamedGraph>,
}

impl RdfDataset {
    /// Create a new, empty dataset (no graphs, not even a default graph).
    pub fn new() -> Self {
        Self {
            graphs: HashMap::new(),
        }
    }

    /// Add (or ensure the existence of) a graph with the given name.
    ///
    /// If the graph already exists this is a no-op.
    pub fn add_graph(&mut self, name: Option<String>) {
        self.graphs
            .entry(name.clone())
            .or_insert_with(|| NamedGraph::new(name));
    }

    /// Remove the graph identified by `name`.
    ///
    /// Returns `true` if the graph existed and was removed.
    pub fn remove_graph(&mut self, name: Option<&str>) -> bool {
        let key = name.map(|s| s.to_owned());
        self.graphs.remove(&key).is_some()
    }

    /// Get a shared reference to the graph identified by `name`.
    pub fn get_graph(&self, name: Option<&str>) -> Option<&NamedGraph> {
        let key = name.map(|s| s.to_owned());
        self.graphs.get(&key)
    }

    /// Get a mutable reference to the graph identified by `name`.
    pub fn get_graph_mut(&mut self, name: Option<&str>) -> Option<&mut NamedGraph> {
        let key = name.map(|s| s.to_owned());
        self.graphs.get_mut(&key)
    }

    /// Add a triple to the specified graph, creating the graph if it does not exist.
    pub fn add_triple(&mut self, graph: Option<&str>, s: &str, p: &str, o: &str) {
        let key = graph.map(|g| g.to_owned());
        let g = self
            .graphs
            .entry(key.clone())
            .or_insert_with(|| NamedGraph::new(key));
        g.add(s.to_owned(), p.to_owned(), o.to_owned());
    }

    /// Remove the first matching triple from the specified graph.
    ///
    /// Returns `true` if a triple was removed, `false` if the graph does not
    /// exist or the triple is not present.
    pub fn remove_triple(&mut self, graph: Option<&str>, s: &str, p: &str, o: &str) -> bool {
        let key = graph.map(|g| g.to_owned());
        match self.graphs.get_mut(&key) {
            Some(g) => g.remove(s, p, o),
            None => false,
        }
    }

    /// Returns `true` if the specified graph contains the given triple.
    pub fn contains(&self, graph: Option<&str>, s: &str, p: &str, o: &str) -> bool {
        let key = graph.map(|g| g.to_owned());
        match self.graphs.get(&key) {
            Some(g) => g.contains(s, p, o),
            None => false,
        }
    }

    /// Iterate over all triples in the specified graph.
    ///
    /// Returns an empty iterator if the graph does not exist.
    pub fn triples_in(
        &self,
        graph: Option<&str>,
    ) -> impl Iterator<Item = &(String, String, String)> {
        let key = graph.map(|g| g.to_owned());
        // Collect into a Vec so we can return a concrete iterator regardless
        // of whether the graph exists.
        let triples: Vec<&(String, String, String)> = match self.graphs.get(&key) {
            Some(g) => g.iter().collect(),
            None => Vec::new(),
        };
        triples.into_iter()
    }

    /// Iterate over every triple in the dataset, yielding `(graph_name, triple)`.
    pub fn all_triples(
        &self,
    ) -> impl Iterator<Item = (&Option<String>, &(String, String, String))> {
        self.graphs
            .iter()
            .flat_map(|(name, g)| g.iter().map(move |t| (name, t)))
    }

    /// Return a list of all graph names (including `None` for the default graph).
    pub fn graph_names(&self) -> Vec<Option<&str>> {
        self.graphs
            .keys()
            .map(|k| k.as_deref())
            .collect()
    }

    /// Number of graphs in the dataset (including the default graph if it exists).
    pub fn graph_count(&self) -> usize {
        self.graphs.len()
    }

    /// Total number of triples across all graphs.
    pub fn triple_count(&self) -> usize {
        self.graphs.values().map(|g| g.len()).sum()
    }

    /// Number of triples in the specified graph.
    ///
    /// Returns 0 if the graph does not exist.
    pub fn triple_count_in(&self, graph: Option<&str>) -> usize {
        let key = graph.map(|g| g.to_owned());
        self.graphs.get(&key).map_or(0, |g| g.len())
    }

    /// Copy all triples from every named graph into the default graph.
    ///
    /// Named graphs are left intact.  The default graph is created if it does
    /// not yet exist.
    pub fn merge_into_default(&mut self) {
        // Collect triples from every named graph.
        let to_merge: Vec<(String, String, String)> = self
            .graphs
            .iter()
            .filter(|(k, _)| k.is_some())
            .flat_map(|(_, g)| {
                g.iter()
                    .cloned()
                    .collect::<Vec<(String, String, String)>>()
            })
            .collect();

        // Ensure the default graph exists.
        let default = self
            .graphs
            .entry(None)
            .or_insert_with(|| NamedGraph::new(None));

        for (s, p, o) in to_merge {
            default.add(s, p, o);
        }
    }

    /// Remove all graphs and all triples from the dataset.
    pub fn clear(&mut self) {
        self.graphs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── NamedGraph tests ────────────────────────────────────────────────────

    #[test]
    fn test_named_graph_new_empty() {
        let g = NamedGraph::new(Some("http://example.org/g1".to_owned()));
        assert!(g.is_empty());
        assert_eq!(g.len(), 0);
        assert_eq!(g.name, Some("http://example.org/g1".to_owned()));
    }

    #[test]
    fn test_named_graph_default_name() {
        let g = NamedGraph::new(None);
        assert!(g.name.is_none());
    }

    #[test]
    fn test_named_graph_add_triple() {
        let mut g = NamedGraph::new(None);
        g.add("s".into(), "p".into(), "o".into());
        assert_eq!(g.len(), 1);
        assert!(!g.is_empty());
    }

    #[test]
    fn test_named_graph_add_multiple() {
        let mut g = NamedGraph::new(None);
        g.add("s1".into(), "p1".into(), "o1".into());
        g.add("s2".into(), "p2".into(), "o2".into());
        g.add("s3".into(), "p3".into(), "o3".into());
        assert_eq!(g.len(), 3);
    }

    #[test]
    fn test_named_graph_contains_true() {
        let mut g = NamedGraph::new(None);
        g.add("s".into(), "p".into(), "o".into());
        assert!(g.contains("s", "p", "o"));
    }

    #[test]
    fn test_named_graph_contains_false() {
        let mut g = NamedGraph::new(None);
        g.add("s".into(), "p".into(), "o".into());
        assert!(!g.contains("x", "p", "o"));
        assert!(!g.contains("s", "x", "o"));
        assert!(!g.contains("s", "p", "x"));
    }

    #[test]
    fn test_named_graph_remove_existing() {
        let mut g = NamedGraph::new(None);
        g.add("s".into(), "p".into(), "o".into());
        let removed = g.remove("s", "p", "o");
        assert!(removed);
        assert!(g.is_empty());
    }

    #[test]
    fn test_named_graph_remove_missing() {
        let mut g = NamedGraph::new(None);
        g.add("s".into(), "p".into(), "o".into());
        let removed = g.remove("x", "p", "o");
        assert!(!removed);
        assert_eq!(g.len(), 1);
    }

    #[test]
    fn test_named_graph_remove_first_only() {
        let mut g = NamedGraph::new(None);
        g.add("s".into(), "p".into(), "o".into());
        g.add("s".into(), "p".into(), "o".into());
        g.remove("s", "p", "o");
        assert_eq!(g.len(), 1); // One duplicate remains.
    }

    #[test]
    fn test_named_graph_iter() {
        let mut g = NamedGraph::new(None);
        g.add("s1".into(), "p1".into(), "o1".into());
        g.add("s2".into(), "p2".into(), "o2".into());
        let collected: Vec<_> = g.iter().collect();
        assert_eq!(collected.len(), 2);
    }

    // ── RdfDataset construction ─────────────────────────────────────────────

    #[test]
    fn test_dataset_new_empty() {
        let ds = RdfDataset::new();
        assert_eq!(ds.graph_count(), 0);
        assert_eq!(ds.triple_count(), 0);
    }

    #[test]
    fn test_dataset_default() {
        let ds = RdfDataset::default();
        assert_eq!(ds.graph_count(), 0);
    }

    // ── add_graph / remove_graph ────────────────────────────────────────────

    #[test]
    fn test_add_named_graph() {
        let mut ds = RdfDataset::new();
        ds.add_graph(Some("http://example.org/g1".into()));
        assert_eq!(ds.graph_count(), 1);
    }

    #[test]
    fn test_add_default_graph() {
        let mut ds = RdfDataset::new();
        ds.add_graph(None);
        assert_eq!(ds.graph_count(), 1);
    }

    #[test]
    fn test_add_graph_idempotent() {
        let mut ds = RdfDataset::new();
        ds.add_graph(Some("http://example.org/g1".into()));
        ds.add_graph(Some("http://example.org/g1".into()));
        assert_eq!(ds.graph_count(), 1);
    }

    #[test]
    fn test_add_multiple_graphs() {
        let mut ds = RdfDataset::new();
        ds.add_graph(Some("http://example.org/g1".into()));
        ds.add_graph(Some("http://example.org/g2".into()));
        ds.add_graph(None);
        assert_eq!(ds.graph_count(), 3);
    }

    #[test]
    fn test_remove_named_graph() {
        let mut ds = RdfDataset::new();
        ds.add_graph(Some("http://example.org/g1".into()));
        let removed = ds.remove_graph(Some("http://example.org/g1"));
        assert!(removed);
        assert_eq!(ds.graph_count(), 0);
    }

    #[test]
    fn test_remove_default_graph() {
        let mut ds = RdfDataset::new();
        ds.add_graph(None);
        let removed = ds.remove_graph(None);
        assert!(removed);
        assert_eq!(ds.graph_count(), 0);
    }

    #[test]
    fn test_remove_nonexistent_graph() {
        let mut ds = RdfDataset::new();
        let removed = ds.remove_graph(Some("http://example.org/missing"));
        assert!(!removed);
    }

    // ── get_graph / get_graph_mut ───────────────────────────────────────────

    #[test]
    fn test_get_graph_exists() {
        let mut ds = RdfDataset::new();
        ds.add_graph(Some("http://example.org/g1".into()));
        assert!(ds.get_graph(Some("http://example.org/g1")).is_some());
    }

    #[test]
    fn test_get_graph_missing() {
        let ds = RdfDataset::new();
        assert!(ds.get_graph(Some("http://example.org/missing")).is_none());
    }

    #[test]
    fn test_get_graph_mut_add_triples() {
        let mut ds = RdfDataset::new();
        ds.add_graph(Some("http://example.org/g1".into()));
        {
            let g = ds.get_graph_mut(Some("http://example.org/g1")).expect("graph should exist");
            g.add("s".into(), "p".into(), "o".into());
        }
        assert_eq!(ds.triple_count_in(Some("http://example.org/g1")), 1);
    }

    // ── add_triple / remove_triple / contains ──────────────────────────────

    #[test]
    fn test_add_triple_creates_graph() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("http://example.org/g1"), "s", "p", "o");
        assert_eq!(ds.graph_count(), 1);
        assert_eq!(ds.triple_count(), 1);
    }

    #[test]
    fn test_add_triple_default_graph() {
        let mut ds = RdfDataset::new();
        ds.add_triple(None, "s", "p", "o");
        assert!(ds.contains(None, "s", "p", "o"));
    }

    #[test]
    fn test_add_triple_named_graph() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "s", "p", "o");
        assert!(ds.contains(Some("g1"), "s", "p", "o"));
    }

    #[test]
    fn test_contains_false_wrong_graph() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "s", "p", "o");
        // Triple is in g1, not in default graph.
        assert!(!ds.contains(None, "s", "p", "o"));
    }

    #[test]
    fn test_remove_triple_success() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "s", "p", "o");
        let removed = ds.remove_triple(Some("g1"), "s", "p", "o");
        assert!(removed);
        assert_eq!(ds.triple_count(), 0);
    }

    #[test]
    fn test_remove_triple_missing() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "s", "p", "o");
        let removed = ds.remove_triple(Some("g1"), "x", "p", "o");
        assert!(!removed);
        assert_eq!(ds.triple_count(), 1);
    }

    #[test]
    fn test_remove_triple_graph_missing() {
        let mut ds = RdfDataset::new();
        let removed = ds.remove_triple(Some("missing"), "s", "p", "o");
        assert!(!removed);
    }

    // ── triple_count / triple_count_in ─────────────────────────────────────

    #[test]
    fn test_triple_count_multiple_graphs() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "s1", "p1", "o1");
        ds.add_triple(Some("g1"), "s2", "p2", "o2");
        ds.add_triple(Some("g2"), "s3", "p3", "o3");
        assert_eq!(ds.triple_count(), 3);
    }

    #[test]
    fn test_triple_count_in_specific() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "s1", "p1", "o1");
        ds.add_triple(Some("g1"), "s2", "p2", "o2");
        ds.add_triple(Some("g2"), "s3", "p3", "o3");
        assert_eq!(ds.triple_count_in(Some("g1")), 2);
        assert_eq!(ds.triple_count_in(Some("g2")), 1);
    }

    #[test]
    fn test_triple_count_in_missing_graph() {
        let ds = RdfDataset::new();
        assert_eq!(ds.triple_count_in(Some("missing")), 0);
    }

    // ── triples_in ──────────────────────────────────────────────────────────

    #[test]
    fn test_triples_in_returns_correct_triples() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "s1", "p1", "o1");
        ds.add_triple(Some("g1"), "s2", "p2", "o2");
        ds.add_triple(Some("g2"), "s3", "p3", "o3");
        let g1_triples: Vec<_> = ds.triples_in(Some("g1")).collect();
        assert_eq!(g1_triples.len(), 2);
    }

    #[test]
    fn test_triples_in_missing_graph_empty_iter() {
        let ds = RdfDataset::new();
        let triples: Vec<_> = ds.triples_in(Some("missing")).collect();
        assert!(triples.is_empty());
    }

    // ── all_triples ─────────────────────────────────────────────────────────

    #[test]
    fn test_all_triples() {
        let mut ds = RdfDataset::new();
        ds.add_triple(None, "s1", "p1", "o1");
        ds.add_triple(Some("g1"), "s2", "p2", "o2");
        let all: Vec<_> = ds.all_triples().collect();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_all_triples_empty() {
        let ds = RdfDataset::new();
        let all: Vec<_> = ds.all_triples().collect();
        assert!(all.is_empty());
    }

    // ── graph_names ─────────────────────────────────────────────────────────

    #[test]
    fn test_graph_names_includes_default() {
        let mut ds = RdfDataset::new();
        ds.add_graph(None);
        ds.add_graph(Some("http://example.org/g1".into()));
        let names = ds.graph_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&None));
        assert!(names.contains(&Some("http://example.org/g1")));
    }

    #[test]
    fn test_graph_names_empty_dataset() {
        let ds = RdfDataset::new();
        assert!(ds.graph_names().is_empty());
    }

    // ── merge_into_default ─────────────────────────────────────────────────

    #[test]
    fn test_merge_into_default_creates_default() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "s1", "p1", "o1");
        ds.add_triple(Some("g2"), "s2", "p2", "o2");
        ds.merge_into_default();
        assert!(ds.contains(None, "s1", "p1", "o1"));
        assert!(ds.contains(None, "s2", "p2", "o2"));
    }

    #[test]
    fn test_merge_into_default_preserves_named_graphs() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "s1", "p1", "o1");
        ds.merge_into_default();
        // Named graph still exists.
        assert!(ds.contains(Some("g1"), "s1", "p1", "o1"));
    }

    #[test]
    fn test_merge_into_default_appends_to_existing() {
        let mut ds = RdfDataset::new();
        ds.add_triple(None, "existing_s", "existing_p", "existing_o");
        ds.add_triple(Some("g1"), "new_s", "new_p", "new_o");
        ds.merge_into_default();
        assert!(ds.contains(None, "existing_s", "existing_p", "existing_o"));
        assert!(ds.contains(None, "new_s", "new_p", "new_o"));
    }

    #[test]
    fn test_merge_into_default_empty_named_graphs() {
        let mut ds = RdfDataset::new();
        ds.add_graph(Some("g1".into()));
        ds.merge_into_default();
        // Default graph is created but empty.
        assert_eq!(ds.triple_count_in(None), 0);
    }

    #[test]
    fn test_merge_into_default_only_default_existing() {
        let mut ds = RdfDataset::new();
        ds.add_triple(None, "s", "p", "o");
        let before = ds.triple_count_in(None);
        ds.merge_into_default();
        // Nothing added — no named graphs.
        assert_eq!(ds.triple_count_in(None), before);
    }

    // ── clear ───────────────────────────────────────────────────────────────

    #[test]
    fn test_clear() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "s", "p", "o");
        ds.add_triple(None, "s2", "p2", "o2");
        ds.clear();
        assert_eq!(ds.graph_count(), 0);
        assert_eq!(ds.triple_count(), 0);
    }

    #[test]
    fn test_clear_empty_dataset() {
        let mut ds = RdfDataset::new();
        ds.clear(); // Should not panic.
        assert_eq!(ds.graph_count(), 0);
    }

    // ── cross-graph isolation ───────────────────────────────────────────────

    #[test]
    fn test_cross_graph_isolation() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "alice", "knows", "bob");
        ds.add_triple(Some("g2"), "carol", "knows", "dave");
        assert!(!ds.contains(Some("g1"), "carol", "knows", "dave"));
        assert!(!ds.contains(Some("g2"), "alice", "knows", "bob"));
    }

    #[test]
    fn test_same_triple_different_graphs() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "s", "p", "o");
        ds.add_triple(Some("g2"), "s", "p", "o");
        assert_eq!(ds.triple_count(), 2);
        assert_eq!(ds.triple_count_in(Some("g1")), 1);
        assert_eq!(ds.triple_count_in(Some("g2")), 1);
    }

    // ── graph_count corner cases ────────────────────────────────────────────

    #[test]
    fn test_graph_count_after_add_and_remove() {
        let mut ds = RdfDataset::new();
        ds.add_graph(Some("g1".into()));
        ds.add_graph(Some("g2".into()));
        ds.remove_graph(Some("g1"));
        assert_eq!(ds.graph_count(), 1);
    }

    #[test]
    fn test_triple_count_after_remove_triple() {
        let mut ds = RdfDataset::new();
        ds.add_triple(None, "s", "p", "o");
        ds.add_triple(None, "s2", "p2", "o2");
        ds.remove_triple(None, "s", "p", "o");
        assert_eq!(ds.triple_count(), 1);
    }

    #[test]
    fn test_large_dataset() {
        let mut ds = RdfDataset::new();
        for i in 0..100usize {
            ds.add_triple(
                Some(&format!("g{}", i % 10)),
                &format!("s{}", i),
                "p",
                &format!("o{}", i),
            );
        }
        assert_eq!(ds.graph_count(), 10);
        assert_eq!(ds.triple_count(), 100);
    }

    #[test]
    fn test_triples_in_default_graph() {
        let mut ds = RdfDataset::new();
        ds.add_triple(None, "s1", "p1", "o1");
        ds.add_triple(None, "s2", "p2", "o2");
        let triples: Vec<_> = ds.triples_in(None).collect();
        assert_eq!(triples.len(), 2);
    }

    #[test]
    fn test_all_triples_graph_name_included() {
        let mut ds = RdfDataset::new();
        ds.add_triple(Some("g1"), "s", "p", "o");
        let all: Vec<_> = ds.all_triples().collect();
        assert_eq!(all.len(), 1);
        assert_eq!(*all[0].0, Some("g1".to_owned()));
    }

    #[test]
    fn test_contains_after_remove() {
        let mut ds = RdfDataset::new();
        ds.add_triple(None, "s", "p", "o");
        ds.remove_triple(None, "s", "p", "o");
        assert!(!ds.contains(None, "s", "p", "o"));
    }

    #[test]
    fn test_add_triple_to_existing_graph() {
        let mut ds = RdfDataset::new();
        ds.add_graph(Some("g1".into()));
        ds.add_triple(Some("g1"), "s", "p", "o");
        assert_eq!(ds.graph_count(), 1); // No extra graph created.
        assert_eq!(ds.triple_count(), 1);
    }

    #[test]
    fn test_merge_into_default_multiple_named_graphs() {
        let mut ds = RdfDataset::new();
        for i in 0..5usize {
            ds.add_triple(Some(&format!("g{}", i)), "s", "p", &format!("o{}", i));
        }
        ds.merge_into_default();
        assert_eq!(ds.triple_count_in(None), 5);
    }
}
