//! Named graph support for OxiRS WASM
//!
//! Provides multi-graph (quad) storage and SPARQL GRAPH pattern queries.
//!
//! A *named graph* is an IRI that identifies a set of triples.  Triples stored
//! without an explicit graph are placed in the *default graph* (IRI = "").
//!
//! ## Example
//! ```no_run
//! use oxirs_wasm::named_graph::{NamedGraphStore, GraphPattern};
//!
//! let mut store = NamedGraphStore::new();
//!
//! store.add_named_graph("http://g1");
//! store.insert_into("http://g1", "http://s", "http://p", "http://o");
//!
//! let results = store.query_graph("http://g1", None, None, None);
//! assert_eq!(results.len(), 1);
//! ```

use crate::store::OxiRSStore;
use std::collections::{HashMap, HashSet};

/// The default graph identifier
pub const DEFAULT_GRAPH: &str = "";

// -----------------------------------------------------------------------
// Quad (triple + named graph)
// -----------------------------------------------------------------------

/// An RDF quad: a triple associated with a named graph
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Quad {
    pub graph: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Quad {
    /// Create a new quad
    pub fn new(
        graph: impl Into<String>,
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            graph: graph.into(),
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

// -----------------------------------------------------------------------
// SPARQL GRAPH pattern
// -----------------------------------------------------------------------

/// A SPARQL GRAPH pattern clause used for query dispatch
#[derive(Debug, Clone)]
pub enum GraphPattern {
    /// `GRAPH <iri> { triple_pattern }`
    Named {
        graph_iri: String,
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
    },
    /// GRAPH ?var { triple_pattern } – variable graph name
    Variable {
        var_name: String,
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
    },
    /// Match across all graphs
    AllGraphs {
        subject: Option<String>,
        predicate: Option<String>,
        object: Option<String>,
    },
}

// -----------------------------------------------------------------------
// NamedGraphStore
// -----------------------------------------------------------------------

/// A multi-graph RDF store built on top of [`OxiRSStore`].
///
/// Each named graph is stored as a separate [`OxiRSStore`] keyed by graph IRI.
/// The default graph (no explicit graph) is stored under the key `""`.
pub struct NamedGraphStore {
    /// One sub-store per named graph (including the default graph)
    graphs: HashMap<String, OxiRSStore>,
    /// Set of registered graph names
    graph_names: HashSet<String>,
}

impl NamedGraphStore {
    /// Create a new, empty named-graph store.
    ///
    /// The default graph is created automatically.
    pub fn new() -> Self {
        let mut graphs = HashMap::new();
        graphs.insert(DEFAULT_GRAPH.to_string(), OxiRSStore::new());
        let mut graph_names = HashSet::new();
        graph_names.insert(DEFAULT_GRAPH.to_string());
        Self {
            graphs,
            graph_names,
        }
    }

    // -----------------------------------------------------------------------
    // Graph management
    // -----------------------------------------------------------------------

    /// Register a new named graph.
    ///
    /// If the graph already exists, this is a no-op.
    pub fn add_named_graph(&mut self, graph_iri: &str) {
        if !self.graph_names.contains(graph_iri) {
            self.graphs.insert(graph_iri.to_string(), OxiRSStore::new());
            self.graph_names.insert(graph_iri.to_string());
        }
    }

    /// Remove a named graph and all its triples.
    ///
    /// Returns `true` if the graph existed and was removed.
    /// The default graph cannot be removed.
    pub fn remove_named_graph(&mut self, graph_iri: &str) -> bool {
        if graph_iri == DEFAULT_GRAPH {
            return false;
        }
        if self.graph_names.remove(graph_iri) {
            self.graphs.remove(graph_iri);
            true
        } else {
            false
        }
    }

    /// Check whether a named graph is registered
    pub fn has_graph(&self, graph_iri: &str) -> bool {
        self.graph_names.contains(graph_iri)
    }

    /// Return the names of all registered graphs
    pub fn graph_names(&self) -> Vec<String> {
        self.graph_names.iter().cloned().collect()
    }

    /// Return the number of registered graphs (including the default graph)
    pub fn graph_count(&self) -> usize {
        self.graph_names.len()
    }

    // -----------------------------------------------------------------------
    // Triple management
    // -----------------------------------------------------------------------

    /// Insert a triple into the default graph.
    pub fn insert(&mut self, subject: &str, predicate: &str, object: &str) -> bool {
        self.insert_into(DEFAULT_GRAPH, subject, predicate, object)
    }

    /// Insert a triple into a specific named graph.
    ///
    /// The graph is auto-created if it does not exist.
    pub fn insert_into(
        &mut self,
        graph_iri: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> bool {
        self.add_named_graph(graph_iri);
        self.graphs
            .get_mut(graph_iri)
            .map(|g| g.insert(subject, predicate, object))
            .unwrap_or(false)
    }

    /// Delete a triple from a specific named graph.
    ///
    /// Returns `true` if the triple was found and deleted.
    pub fn delete_from(
        &mut self,
        graph_iri: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> bool {
        self.graphs
            .get_mut(graph_iri)
            .map(|g| g.delete(subject, predicate, object))
            .unwrap_or(false)
    }

    /// Check whether a triple exists in a specific named graph
    pub fn contains_in(
        &self,
        graph_iri: &str,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> bool {
        self.graphs
            .get(graph_iri)
            .map(|g| g.contains(subject, predicate, object))
            .unwrap_or(false)
    }

    /// Total number of triples across ALL graphs
    pub fn total_size(&self) -> usize {
        self.graphs.values().map(|g| g.size()).sum()
    }

    /// Number of triples in a specific named graph
    pub fn size_of(&self, graph_iri: &str) -> usize {
        self.graphs.get(graph_iri).map(|g| g.size()).unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // Querying
    // -----------------------------------------------------------------------

    /// Query triples in a specific named graph with optional subject/predicate/object filters.
    ///
    /// `None` means wildcard (match anything).
    ///
    /// Returns a list of quads `[graph, subject, predicate, object]`.
    pub fn query_graph(
        &self,
        graph_iri: &str,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<[String; 4]> {
        let Some(g) = self.graphs.get(graph_iri) else {
            return Vec::new();
        };
        g.all_triples()
            .filter(|t| {
                subject.map_or(true, |s| t.subject == s)
                    && predicate.map_or(true, |p| t.predicate == p)
                    && object.map_or(true, |o| t.object == o)
            })
            .map(|t| {
                [
                    graph_iri.to_string(),
                    t.subject.clone(),
                    t.predicate.clone(),
                    t.object.clone(),
                ]
            })
            .collect()
    }

    /// Evaluate a SPARQL GRAPH pattern, returning matching quads.
    pub fn evaluate_graph_pattern(&self, pattern: &GraphPattern) -> Vec<Quad> {
        match pattern {
            GraphPattern::Named {
                graph_iri,
                subject,
                predicate,
                object,
            } => self
                .query_graph(
                    graph_iri,
                    subject.as_deref(),
                    predicate.as_deref(),
                    object.as_deref(),
                )
                .into_iter()
                .map(|arr| Quad::new(&arr[0], &arr[1], &arr[2], &arr[3]))
                .collect(),

            GraphPattern::Variable {
                var_name: _,
                subject,
                predicate,
                object,
            } => {
                // Bind var to each graph name, collect all results
                self.query_all_graphs(subject.as_deref(), predicate.as_deref(), object.as_deref())
            }

            GraphPattern::AllGraphs {
                subject,
                predicate,
                object,
            } => self.query_all_graphs(subject.as_deref(), predicate.as_deref(), object.as_deref()),
        }
    }

    /// Query across ALL named graphs with optional filters.
    ///
    /// Returns a flat list of [`Quad`] values.
    pub fn query_all_graphs(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> Vec<Quad> {
        let mut results = Vec::new();
        for (graph_iri, g) in &self.graphs {
            for t in g.all_triples() {
                if subject.map_or(true, |s| t.subject == s)
                    && predicate.map_or(true, |p| t.predicate == p)
                    && object.map_or(true, |o| t.object == o)
                {
                    results.push(Quad::new(
                        graph_iri.clone(),
                        t.subject.clone(),
                        t.predicate.clone(),
                        t.object.clone(),
                    ));
                }
            }
        }
        results
    }

    // -----------------------------------------------------------------------
    // Serialisation
    // -----------------------------------------------------------------------

    /// Export all quads as N-Quads text
    pub fn to_nquads(&self) -> String {
        let mut output = String::new();
        for (graph_iri, g) in &self.graphs {
            for t in g.all_triples() {
                let graph_token = if graph_iri.is_empty() {
                    String::new()
                } else {
                    format!(" <{}>", graph_iri)
                };
                output.push_str(&format!(
                    "<{}> <{}> {} {}.\n",
                    t.subject,
                    t.predicate,
                    format_object(&t.object),
                    graph_token,
                ));
            }
        }
        output
    }
}

impl Default for NamedGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Format an object term for N-Quads output
fn format_object(object: &str) -> String {
    if object.starts_with("http://") || object.starts_with("https://") || object.starts_with("urn:")
    {
        format!("<{}>", object)
    } else if object.starts_with('"') {
        object.to_string()
    } else {
        format!("\"{}\"", object)
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> NamedGraphStore {
        NamedGraphStore::new()
    }

    #[test]
    fn test_add_and_remove_named_graph() {
        let mut store = make_store();
        store.add_named_graph("http://g1");
        assert!(store.has_graph("http://g1"));

        let removed = store.remove_named_graph("http://g1");
        assert!(removed);
        assert!(!store.has_graph("http://g1"));
    }

    #[test]
    fn test_default_graph_cannot_be_removed() {
        let mut store = make_store();
        let removed = store.remove_named_graph(DEFAULT_GRAPH);
        assert!(!removed);
        assert!(store.has_graph(DEFAULT_GRAPH));
    }

    #[test]
    fn test_insert_and_query_named_graph() {
        let mut store = make_store();
        store.add_named_graph("http://g1");
        store.insert_into("http://g1", "http://s", "http://p", "http://o");

        let results = store.query_graph("http://g1", None, None, None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0][0], "http://g1");
        assert_eq!(results[0][1], "http://s");
    }

    #[test]
    fn test_isolation_between_graphs() {
        let mut store = make_store();
        store.insert_into("http://g1", "http://a", "http://p", "http://x");
        store.insert_into("http://g2", "http://b", "http://p", "http://y");

        // g1 should not see g2's triples
        let g1_results = store.query_graph("http://g1", None, None, None);
        assert_eq!(g1_results.len(), 1);
        assert_eq!(g1_results[0][1], "http://a");

        let g2_results = store.query_graph("http://g2", None, None, None);
        assert_eq!(g2_results.len(), 1);
        assert_eq!(g2_results[0][1], "http://b");
    }

    #[test]
    fn test_query_all_graphs() {
        let mut store = make_store();
        store.insert_into("http://g1", "http://s1", "http://p", "http://o");
        store.insert_into("http://g2", "http://s2", "http://p", "http://o");

        let all = store.query_all_graphs(None, None, None);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_graph_pattern_named() {
        let mut store = make_store();
        store.insert_into("http://g1", "http://alice", "http://knows", "http://bob");
        store.insert_into("http://g2", "http://carol", "http://knows", "http://dave");

        let pattern = GraphPattern::Named {
            graph_iri: "http://g1".to_string(),
            subject: None,
            predicate: Some("http://knows".to_string()),
            object: None,
        };
        let quads = store.evaluate_graph_pattern(&pattern);
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].graph, "http://g1");
        assert_eq!(quads[0].subject, "http://alice");
    }

    #[test]
    fn test_graph_pattern_all_graphs() {
        let mut store = make_store();
        store.insert_into("http://g1", "http://alice", "http://name", "\"Alice\"");
        store.insert_into("http://g2", "http://bob", "http://name", "\"Bob\"");

        let pattern = GraphPattern::AllGraphs {
            subject: None,
            predicate: Some("http://name".to_string()),
            object: None,
        };
        let quads = store.evaluate_graph_pattern(&pattern);
        assert_eq!(quads.len(), 2);
    }

    #[test]
    fn test_total_size() {
        let mut store = make_store();
        store.insert_into("http://g1", "http://a", "http://b", "http://c");
        store.insert_into("http://g2", "http://x", "http://y", "http://z");
        store.insert("http://d", "http://e", "http://f"); // default graph
        assert_eq!(store.total_size(), 3);
    }

    #[test]
    fn test_to_nquads() {
        let mut store = make_store();
        store.insert_into("http://g1", "http://s", "http://p", "http://o");
        let nq = store.to_nquads();
        assert!(nq.contains("<http://g1>"));
        assert!(nq.contains("<http://s>"));
    }
}
