//! SPARQL 1.1 UPDATE Graph Management — Types and Data Structures
//!
//! Contains all the core structs, enums, and data-model types used by the
//! graph management operations defined in `update_graph_management_ops`.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// A single RDF triple (subject, predicate, object all as plain strings / IRIs).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

impl Triple {
    /// Create a new triple.
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        object: impl Into<String>,
    ) -> Self {
        Self {
            subject: subject.into(),
            predicate: predicate.into(),
            object: object.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Graph target
// ---------------------------------------------------------------------------

/// The target for graph management operations (mirrors the SPARQL 1.1 grammar).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphManagementTarget {
    /// The unnamed default graph.
    Default,
    /// A specific named graph identified by its IRI.
    Named(String),
    /// All named graphs (does **not** include the default graph unless combined
    /// with `All`).
    AllNamed,
    /// All graphs: default graph plus every named graph.
    All,
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/// A SPARQL 1.1 UPDATE graph management operation.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphManagementOp {
    /// `LOAD <iri> [SILENT] [INTO GRAPH <g>]`
    Load {
        /// The IRI of the remote document to load.
        iri: String,
        /// Destination graph; `None` means the default graph.
        into_graph: Option<String>,
        /// If `true`, errors are suppressed.
        silent: bool,
    },

    /// `CLEAR [SILENT] (DEFAULT | NAMED | ALL | GRAPH <g>)`
    Clear {
        /// Which graph(s) to empty.
        target: GraphManagementTarget,
        /// If `true`, errors are suppressed.
        silent: bool,
    },

    /// `DROP [SILENT] (DEFAULT | NAMED | ALL | GRAPH <g>)`
    Drop {
        /// Which graph(s) to drop entirely.
        target: GraphManagementTarget,
        /// If `true`, errors (e.g. non-existent graph) are suppressed.
        silent: bool,
    },

    /// `CREATE [SILENT] GRAPH <g>`
    Create {
        /// IRI of the graph to create.
        graph: String,
        /// If `true`, errors (e.g. graph already exists) are suppressed.
        silent: bool,
    },

    /// `COPY [SILENT] <source> TO <dest>`
    ///
    /// The destination graph is first cleared, then all triples from the source
    /// are copied into it.  The source graph is left unchanged.
    Copy {
        /// Source graph.
        source: GraphManagementTarget,
        /// Destination graph.
        destination: GraphManagementTarget,
        /// If `true`, errors are suppressed.
        silent: bool,
    },

    /// `MOVE [SILENT] <source> TO <dest>`
    ///
    /// Like `COPY`, but the source graph is dropped after the copy.
    Move {
        /// Source graph.
        source: GraphManagementTarget,
        /// Destination graph.
        destination: GraphManagementTarget,
        /// If `true`, errors are suppressed.
        silent: bool,
    },

    /// `ADD [SILENT] <source> TO <dest>`
    ///
    /// Adds (merges) all triples from the source graph into the destination
    /// graph without clearing the destination first.
    Add {
        /// Source graph.
        source: GraphManagementTarget,
        /// Destination graph.
        destination: GraphManagementTarget,
        /// If `true`, errors are suppressed.
        silent: bool,
    },
}

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result returned by [`crate::update_graph_management_ops::GraphManagementExecutor::execute`].
#[derive(Debug, Clone, Default)]
pub struct GraphManagementResult {
    /// Number of triples that were inserted or copied.
    pub triples_affected: usize,
    /// IRIs (or `"DEFAULT"`) of graphs that were created, cleared, dropped or
    /// otherwise affected by the operation.
    pub graphs_affected: Vec<String>,
}

// ---------------------------------------------------------------------------
// In-memory dataset
// ---------------------------------------------------------------------------

/// A lightweight in-memory RDF dataset used for graph management operations.
///
/// It maintains a *default graph* (unnamed) and any number of *named graphs*.
/// All graphs are identified by their IRI strings.
#[derive(Debug, Clone, Default)]
pub struct GraphManagementDataset {
    /// Triples in the unnamed default graph.
    pub default_graph: Vec<Triple>,
    /// Named graphs, keyed by their IRI string.
    pub named_graphs: HashMap<String, Vec<Triple>>,
}

impl GraphManagementDataset {
    /// Create an empty dataset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a triple to a graph.
    ///
    /// `graph` is `None` for the default graph, or `Some(iri)` for a named
    /// graph.  If the named graph does not yet exist it is created implicitly.
    pub fn add_triple(&mut self, graph: Option<&str>, triple: Triple) {
        match graph {
            None => self.default_graph.push(triple),
            Some(iri) => {
                self.named_graphs
                    .entry(iri.to_owned())
                    .or_default()
                    .push(triple);
            }
        }
    }

    /// Return a slice of the triples in the given graph.
    ///
    /// Returns an empty slice for graphs that do not exist.
    pub fn get_graph(&self, graph: Option<&str>) -> &[Triple] {
        match graph {
            None => &self.default_graph,
            Some(iri) => self.named_graphs.get(iri).map(Vec::as_slice).unwrap_or(&[]),
        }
    }

    /// Return the IRI strings of every named graph in the dataset.
    pub fn graph_names(&self) -> Vec<String> {
        self.named_graphs.keys().cloned().collect()
    }

    /// Return the triple count for the given graph.
    pub fn triple_count(&self, graph: Option<&str>) -> usize {
        self.get_graph(graph).len()
    }

    /// Check whether a named graph with the given IRI exists.
    pub fn named_graph_exists(&self, iri: &str) -> bool {
        self.named_graphs.contains_key(iri)
    }
}
