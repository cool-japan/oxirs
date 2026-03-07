//! SPARQL 1.1 UPDATE Graph Management Operations
//!
//! This module implements the full set of SPARQL 1.1 UPDATE graph management
//! operations as specified at:
//! <https://www.w3.org/TR/sparql11-update/#graphManagement>
//!
//! Supported operations:
//! - `LOAD <iri> [SILENT] [INTO GRAPH <g>]`
//! - `CLEAR [SILENT] (DEFAULT | NAMED | ALL | GRAPH <g>)`
//! - `DROP [SILENT] (DEFAULT | NAMED | ALL | GRAPH <g>)`
//! - `CREATE [SILENT] GRAPH <g>`
//! - `COPY [SILENT] <source> TO <dest>`
//! - `MOVE [SILENT] <source> TO <dest>`
//! - `ADD [SILENT] <source> TO <dest>`

use anyhow::{anyhow, Result};
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

/// Result returned by [`GraphManagementExecutor::execute`].
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

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Executor for SPARQL 1.1 UPDATE graph management operations.
///
/// All operations are applied to a [`GraphManagementDataset`] in memory.
/// HTTP-based `LOAD` is intentionally not implemented (see
/// [`Self::execute`] for details).
pub struct GraphManagementExecutor;

impl GraphManagementExecutor {
    /// Execute a graph management operation against the supplied dataset.
    ///
    /// ### Notes on `LOAD`
    /// `LOAD` requires an HTTP (or file-system) client and is therefore **not**
    /// implemented here.  Calling `LOAD` with `silent = false` returns an
    /// error; with `silent = true` it returns a no-op success.  Real
    /// implementations should use `reqwest` or `ureq` to fetch the document.
    pub fn execute(
        op: &GraphManagementOp,
        dataset: &mut GraphManagementDataset,
    ) -> Result<GraphManagementResult> {
        match op {
            GraphManagementOp::Load {
                iri,
                into_graph,
                silent,
            } => Self::execute_load(iri, into_graph.as_deref(), *silent),
            GraphManagementOp::Clear { target, silent } => {
                Self::execute_clear(target, *silent, dataset)
            }
            GraphManagementOp::Drop { target, silent } => {
                Self::execute_drop(target, *silent, dataset)
            }
            GraphManagementOp::Create { graph, silent } => {
                Self::execute_create(graph, *silent, dataset)
            }
            GraphManagementOp::Copy {
                source,
                destination,
                silent,
            } => Self::execute_copy(source, destination, *silent, dataset),
            GraphManagementOp::Move {
                source,
                destination,
                silent,
            } => Self::execute_move(source, destination, *silent, dataset),
            GraphManagementOp::Add {
                source,
                destination,
                silent,
            } => Self::execute_add(source, destination, *silent, dataset),
        }
    }

    // -----------------------------------------------------------------------
    // LOAD
    // -----------------------------------------------------------------------

    fn execute_load(
        iri: &str,
        _into_graph: Option<&str>,
        silent: bool,
    ) -> Result<GraphManagementResult> {
        // HTTP loading is not implemented in this in-memory executor.
        if silent {
            Ok(GraphManagementResult::default())
        } else {
            Err(anyhow!(
                "LOAD <{iri}> is not supported by the in-memory graph management executor. \
                 Use a network-capable executor or specify SILENT to suppress this error."
            ))
        }
    }

    // -----------------------------------------------------------------------
    // CLEAR
    // -----------------------------------------------------------------------

    fn execute_clear(
        target: &GraphManagementTarget,
        silent: bool,
        dataset: &mut GraphManagementDataset,
    ) -> Result<GraphManagementResult> {
        let mut result = GraphManagementResult::default();

        match target {
            GraphManagementTarget::Default => {
                result.triples_affected = dataset.default_graph.len();
                dataset.default_graph.clear();
                result.graphs_affected.push("DEFAULT".to_owned());
            }
            GraphManagementTarget::Named(iri) => match dataset.named_graphs.get_mut(iri) {
                Some(triples) => {
                    result.triples_affected = triples.len();
                    triples.clear();
                    result.graphs_affected.push(iri.clone());
                }
                None => {
                    if !silent {
                        return Err(anyhow!("CLEAR GRAPH <{iri}>: named graph does not exist"));
                    }
                }
            },
            GraphManagementTarget::AllNamed => {
                for (iri, triples) in &mut dataset.named_graphs {
                    result.triples_affected += triples.len();
                    triples.clear();
                    result.graphs_affected.push(iri.clone());
                }
            }
            GraphManagementTarget::All => {
                result.triples_affected += dataset.default_graph.len();
                dataset.default_graph.clear();
                result.graphs_affected.push("DEFAULT".to_owned());

                for (iri, triples) in &mut dataset.named_graphs {
                    result.triples_affected += triples.len();
                    triples.clear();
                    result.graphs_affected.push(iri.clone());
                }
            }
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // DROP
    // -----------------------------------------------------------------------

    fn execute_drop(
        target: &GraphManagementTarget,
        silent: bool,
        dataset: &mut GraphManagementDataset,
    ) -> Result<GraphManagementResult> {
        let mut result = GraphManagementResult::default();

        match target {
            GraphManagementTarget::Default => {
                result.triples_affected = dataset.default_graph.len();
                dataset.default_graph.clear();
                result.graphs_affected.push("DEFAULT".to_owned());
            }
            GraphManagementTarget::Named(iri) => match dataset.named_graphs.remove(iri) {
                Some(triples) => {
                    result.triples_affected = triples.len();
                    result.graphs_affected.push(iri.clone());
                }
                None => {
                    if !silent {
                        return Err(anyhow!("DROP GRAPH <{iri}>: named graph does not exist"));
                    }
                }
            },
            GraphManagementTarget::AllNamed => {
                for (iri, triples) in dataset.named_graphs.drain() {
                    result.triples_affected += triples.len();
                    result.graphs_affected.push(iri);
                }
            }
            GraphManagementTarget::All => {
                result.triples_affected += dataset.default_graph.len();
                dataset.default_graph.clear();
                result.graphs_affected.push("DEFAULT".to_owned());

                for (iri, triples) in dataset.named_graphs.drain() {
                    result.triples_affected += triples.len();
                    result.graphs_affected.push(iri);
                }
            }
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // CREATE
    // -----------------------------------------------------------------------

    fn execute_create(
        graph: &str,
        silent: bool,
        dataset: &mut GraphManagementDataset,
    ) -> Result<GraphManagementResult> {
        if dataset.named_graphs.contains_key(graph) {
            if !silent {
                return Err(anyhow!("CREATE GRAPH <{graph}>: graph already exists"));
            }
            return Ok(GraphManagementResult::default());
        }

        dataset.named_graphs.insert(graph.to_owned(), Vec::new());

        Ok(GraphManagementResult {
            triples_affected: 0,
            graphs_affected: vec![graph.to_owned()],
        })
    }

    // -----------------------------------------------------------------------
    // Helpers for resolving graph contents
    // -----------------------------------------------------------------------

    /// Retrieve a *clone* of all triples in the given target, validating that
    /// it exists when `silent` is false.
    fn get_triples_for_target(
        target: &GraphManagementTarget,
        silent: bool,
        dataset: &GraphManagementDataset,
    ) -> Result<Vec<Triple>> {
        match target {
            GraphManagementTarget::Default => Ok(dataset.default_graph.clone()),
            GraphManagementTarget::Named(iri) => match dataset.named_graphs.get(iri) {
                Some(triples) => Ok(triples.clone()),
                None => {
                    if silent {
                        Ok(vec![])
                    } else {
                        Err(anyhow!("Graph <{iri}> does not exist"))
                    }
                }
            },
            // COPY/MOVE/ADD with AllNamed or All as source is a SPARQL error
            // (the spec requires a single graph as source).
            GraphManagementTarget::AllNamed | GraphManagementTarget::All => {
                if silent {
                    Ok(vec![])
                } else {
                    Err(anyhow!(
                        "COPY/MOVE/ADD source must be a single graph (DEFAULT or a named graph IRI), \
                         not ALL or NAMED"
                    ))
                }
            }
        }
    }

    /// Clear the destination graph storage and return the mutable reference.
    fn clear_destination(
        target: &GraphManagementTarget,
        dataset: &mut GraphManagementDataset,
    ) -> Result<()> {
        match target {
            GraphManagementTarget::Default => {
                dataset.default_graph.clear();
            }
            GraphManagementTarget::Named(iri) => {
                dataset.named_graphs.entry(iri.clone()).or_default().clear();
            }
            GraphManagementTarget::AllNamed | GraphManagementTarget::All => {
                return Err(anyhow!(
                    "Destination must be a single graph (DEFAULT or a named graph IRI)"
                ));
            }
        }
        Ok(())
    }

    /// Write triples into the destination graph (appending).
    fn write_triples_to_destination(
        target: &GraphManagementTarget,
        triples: Vec<Triple>,
        dataset: &mut GraphManagementDataset,
    ) -> Result<usize> {
        let count = triples.len();
        match target {
            GraphManagementTarget::Default => {
                dataset.default_graph.extend(triples);
            }
            GraphManagementTarget::Named(iri) => {
                dataset
                    .named_graphs
                    .entry(iri.clone())
                    .or_default()
                    .extend(triples);
            }
            GraphManagementTarget::AllNamed | GraphManagementTarget::All => {
                return Err(anyhow!(
                    "Destination must be a single graph (DEFAULT or a named graph IRI)"
                ));
            }
        }
        Ok(count)
    }

    /// Remove the source graph from the dataset (used by MOVE).
    fn drop_source(target: &GraphManagementTarget, dataset: &mut GraphManagementDataset) {
        match target {
            GraphManagementTarget::Default => {
                dataset.default_graph.clear();
            }
            GraphManagementTarget::Named(iri) => {
                dataset.named_graphs.remove(iri);
            }
            // These variants are caught earlier; safe to no-op here.
            GraphManagementTarget::AllNamed | GraphManagementTarget::All => {}
        }
    }

    /// Return the canonical string label for a target (for reporting).
    fn target_label(target: &GraphManagementTarget) -> String {
        match target {
            GraphManagementTarget::Default => "DEFAULT".to_owned(),
            GraphManagementTarget::Named(iri) => iri.clone(),
            GraphManagementTarget::AllNamed => "NAMED".to_owned(),
            GraphManagementTarget::All => "ALL".to_owned(),
        }
    }

    // -----------------------------------------------------------------------
    // COPY
    // -----------------------------------------------------------------------

    fn execute_copy(
        source: &GraphManagementTarget,
        dest: &GraphManagementTarget,
        silent: bool,
        dataset: &mut GraphManagementDataset,
    ) -> Result<GraphManagementResult> {
        // If source == destination this is a no-op per the W3C spec.
        if source == dest {
            return Ok(GraphManagementResult {
                triples_affected: 0,
                graphs_affected: vec![Self::target_label(dest)],
            });
        }

        let source_triples = Self::get_triples_for_target(source, silent, dataset)?;
        let count = source_triples.len();

        Self::clear_destination(dest, dataset)?;
        Self::write_triples_to_destination(dest, source_triples, dataset)?;

        Ok(GraphManagementResult {
            triples_affected: count,
            graphs_affected: vec![Self::target_label(source), Self::target_label(dest)],
        })
    }

    // -----------------------------------------------------------------------
    // MOVE
    // -----------------------------------------------------------------------

    fn execute_move(
        source: &GraphManagementTarget,
        dest: &GraphManagementTarget,
        silent: bool,
        dataset: &mut GraphManagementDataset,
    ) -> Result<GraphManagementResult> {
        // MOVE to self is a no-op.
        if source == dest {
            return Ok(GraphManagementResult {
                triples_affected: 0,
                graphs_affected: vec![Self::target_label(dest)],
            });
        }

        let source_triples = Self::get_triples_for_target(source, silent, dataset)?;
        let count = source_triples.len();

        Self::clear_destination(dest, dataset)?;
        Self::write_triples_to_destination(dest, source_triples, dataset)?;
        Self::drop_source(source, dataset);

        Ok(GraphManagementResult {
            triples_affected: count,
            graphs_affected: vec![Self::target_label(source), Self::target_label(dest)],
        })
    }

    // -----------------------------------------------------------------------
    // ADD
    // -----------------------------------------------------------------------

    fn execute_add(
        source: &GraphManagementTarget,
        dest: &GraphManagementTarget,
        silent: bool,
        dataset: &mut GraphManagementDataset,
    ) -> Result<GraphManagementResult> {
        // ADD to self is a no-op.
        if source == dest {
            return Ok(GraphManagementResult {
                triples_affected: 0,
                graphs_affected: vec![Self::target_label(dest)],
            });
        }

        let source_triples = Self::get_triples_for_target(source, silent, dataset)?;
        let count = source_triples.len();

        Self::write_triples_to_destination(dest, source_triples, dataset)?;

        Ok(GraphManagementResult {
            triples_affected: count,
            graphs_affected: vec![Self::target_label(source), Self::target_label(dest)],
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper builders
    // -----------------------------------------------------------------------

    fn triple(s: &str, p: &str, o: &str) -> Triple {
        Triple::new(s, p, o)
    }

    fn ex(local: &str) -> String {
        format!("http://example.org/{local}")
    }

    fn g1() -> String {
        ex("g1")
    }
    fn g2() -> String {
        ex("g2")
    }
    fn g3() -> String {
        ex("g3")
    }

    fn t1() -> Triple {
        triple(&ex("s1"), &ex("p1"), &ex("o1"))
    }
    fn t2() -> Triple {
        triple(&ex("s2"), &ex("p2"), &ex("o2"))
    }
    fn t3() -> Triple {
        triple(&ex("s3"), &ex("p3"), &ex("o3"))
    }

    fn dataset_with_default_triples(triples: &[Triple]) -> GraphManagementDataset {
        let mut ds = GraphManagementDataset::new();
        for t in triples {
            ds.add_triple(None, t.clone());
        }
        ds
    }

    fn dataset_with_named_triples(iri: &str, triples: &[Triple]) -> GraphManagementDataset {
        let mut ds = GraphManagementDataset::new();
        for t in triples {
            ds.add_triple(Some(iri), t.clone());
        }
        ds
    }

    // -----------------------------------------------------------------------
    // Triple / dataset basics
    // -----------------------------------------------------------------------

    #[test]
    fn test_triple_new() {
        let t = triple("s", "p", "o");
        assert_eq!(t.subject, "s");
        assert_eq!(t.predicate, "p");
        assert_eq!(t.object, "o");
    }

    #[test]
    fn test_triple_equality() {
        let a = triple("s", "p", "o");
        let b = triple("s", "p", "o");
        let c = triple("x", "p", "o");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_dataset_new_is_empty() {
        let ds = GraphManagementDataset::new();
        assert_eq!(ds.triple_count(None), 0);
        assert!(ds.graph_names().is_empty());
    }

    #[test]
    fn test_dataset_add_triple_default_graph() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(None, t1());
        assert_eq!(ds.triple_count(None), 1);
    }

    #[test]
    fn test_dataset_add_triple_named_graph() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        assert_eq!(ds.triple_count(Some(&g1())), 1);
        assert!(ds.named_graph_exists(&g1()));
    }

    #[test]
    fn test_dataset_add_triple_creates_named_graph() {
        let mut ds = GraphManagementDataset::new();
        assert!(!ds.named_graph_exists(&g1()));
        ds.add_triple(Some(&g1()), t1());
        assert!(ds.named_graph_exists(&g1()));
    }

    #[test]
    fn test_dataset_get_graph_default_empty() {
        let ds = GraphManagementDataset::new();
        assert_eq!(ds.get_graph(None).len(), 0);
    }

    #[test]
    fn test_dataset_get_graph_nonexistent_named_returns_empty() {
        let ds = GraphManagementDataset::new();
        assert_eq!(ds.get_graph(Some("http://no-such-graph")).len(), 0);
    }

    #[test]
    fn test_dataset_graph_names() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2());
        let mut names = ds.graph_names();
        names.sort();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_dataset_triple_count_multiple() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(None, t1());
        ds.add_triple(None, t2());
        ds.add_triple(Some(&g1()), t3());
        assert_eq!(ds.triple_count(None), 2);
        assert_eq!(ds.triple_count(Some(&g1())), 1);
    }

    // -----------------------------------------------------------------------
    // CLEAR tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_default_removes_all_triples() {
        let mut ds = dataset_with_default_triples(&[t1(), t2(), t3()]);
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::Default,
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(None), 0);
        assert_eq!(result.triples_affected, 3);
        assert!(result.graphs_affected.contains(&"DEFAULT".to_owned()));
    }

    #[test]
    fn test_clear_named_graph_empties_it() {
        let mut ds = dataset_with_named_triples(&g1(), &[t1(), t2()]);
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 0);
        // The graph itself still exists (CLEAR does not drop)
        assert!(ds.named_graph_exists(&g1()));
    }

    #[test]
    fn test_clear_named_keeps_other_graphs_intact() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2());
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 0);
        assert_eq!(ds.triple_count(Some(&g2())), 1);
    }

    #[test]
    fn test_clear_named_nonexistent_without_silent_errors() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_clear_named_nonexistent_with_silent_succeeds() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::Named(g1()),
            silent: true,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_ok());
    }

    #[test]
    fn test_clear_all_named_removes_all_named_graphs_content() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2());
        ds.add_triple(None, t3()); // default should remain untouched
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::AllNamed,
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 2);
        assert_eq!(ds.triple_count(Some(&g1())), 0);
        assert_eq!(ds.triple_count(Some(&g2())), 0);
        assert_eq!(ds.triple_count(None), 1); // default untouched
    }

    #[test]
    fn test_clear_all_removes_everything() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(None, t1());
        ds.add_triple(Some(&g1()), t2());
        ds.add_triple(Some(&g2()), t3());
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::All,
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 3);
        assert_eq!(ds.triple_count(None), 0);
        assert_eq!(ds.triple_count(Some(&g1())), 0);
        assert_eq!(ds.triple_count(Some(&g2())), 0);
    }

    #[test]
    fn test_clear_all_on_empty_dataset_is_ok() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::All,
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 0);
    }

    #[test]
    fn test_clear_returns_graphs_affected_list() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Clear {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(result.graphs_affected.contains(&g1()));
    }

    // -----------------------------------------------------------------------
    // DROP tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_drop_named_graph_removes_it() {
        let mut ds = dataset_with_named_triples(&g1(), &[t1(), t2()]);
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(!ds.named_graph_exists(&g1()));
    }

    #[test]
    fn test_drop_named_graph_reports_triples_affected() {
        let mut ds = dataset_with_named_triples(&g1(), &[t1(), t2(), t3()]);
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 3);
    }

    #[test]
    fn test_drop_silent_on_nonexistent_graph_succeeds() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::Named(g1()),
            silent: true,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_ok());
    }

    #[test]
    fn test_drop_nonsilent_on_nonexistent_graph_errors() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_drop_default_clears_default_graph() {
        let mut ds = dataset_with_default_triples(&[t1(), t2()]);
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::Default,
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(None), 0);
    }

    #[test]
    fn test_drop_all_named_removes_all_named_graphs() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2());
        ds.add_triple(None, t3());
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::AllNamed,
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(ds.graph_names().is_empty());
        assert_eq!(ds.triple_count(None), 1); // default untouched
    }

    #[test]
    fn test_drop_all_removes_default_and_named() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(None, t1());
        ds.add_triple(Some(&g1()), t2());
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::All,
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 2);
        assert_eq!(ds.triple_count(None), 0);
        assert!(ds.graph_names().is_empty());
    }

    #[test]
    fn test_drop_named_does_not_affect_other_named_graphs() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2());
        let op = GraphManagementOp::Drop {
            target: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(!ds.named_graph_exists(&g1()));
        assert!(ds.named_graph_exists(&g2()));
    }

    // -----------------------------------------------------------------------
    // CREATE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_creates_empty_named_graph() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Create {
            graph: g1(),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(ds.named_graph_exists(&g1()));
        assert_eq!(ds.triple_count(Some(&g1())), 0);
    }

    #[test]
    fn test_create_reports_affected_graph() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Create {
            graph: g1(),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(result.graphs_affected.contains(&g1()));
    }

    #[test]
    fn test_create_existing_graph_without_silent_errors() {
        let mut ds = GraphManagementDataset::new();
        ds.named_graphs.insert(g1(), vec![]);
        let op = GraphManagementOp::Create {
            graph: g1(),
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_create_silent_on_existing_graph_succeeds() {
        let mut ds = GraphManagementDataset::new();
        ds.named_graphs.insert(g1(), vec![t1()]);
        let op = GraphManagementOp::Create {
            graph: g1(),
            silent: true,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        // Content must be preserved
        assert_eq!(ds.triple_count(Some(&g1())), 1);
        assert_eq!(result.triples_affected, 0);
    }

    #[test]
    fn test_create_multiple_graphs() {
        let mut ds = GraphManagementDataset::new();
        for g in [&g1(), &g2(), &g3()] {
            GraphManagementExecutor::execute(
                &GraphManagementOp::Create {
                    graph: g.clone(),
                    silent: false,
                },
                &mut ds,
            )
            .unwrap();
        }
        assert_eq!(ds.graph_names().len(), 3);
    }

    // -----------------------------------------------------------------------
    // COPY tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_copy_named_to_named_copies_triples() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g2())), 2);
    }

    #[test]
    fn test_copy_clears_destination_first() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t3()); // pre-existing in dest
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        // Destination should contain only what was in source
        assert_eq!(ds.triple_count(Some(&g2())), 1);
        assert_eq!(ds.get_graph(Some(&g2()))[0], t1());
    }

    #[test]
    fn test_copy_leaves_source_unchanged() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 2);
    }

    #[test]
    fn test_copy_from_default_to_named() {
        let mut ds = dataset_with_default_triples(&[t1(), t2()]);
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Default,
            destination: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 2);
        assert_eq!(ds.triple_count(None), 2); // source unchanged
    }

    #[test]
    fn test_copy_from_named_to_default() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(None, t3()); // pre-existing default should be cleared
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Default,
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(None), 1);
        assert_eq!(ds.get_graph(None)[0], t1());
    }

    #[test]
    fn test_copy_self_to_self_is_noop() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 0);
        assert_eq!(ds.triple_count(Some(&g1())), 1); // content preserved
    }

    #[test]
    fn test_copy_nonexistent_source_without_silent_errors() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_copy_nonexistent_source_with_silent_succeeds() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: true,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_ok());
    }

    #[test]
    fn test_copy_reports_triples_affected() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 2);
    }

    // -----------------------------------------------------------------------
    // MOVE tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_move_named_to_named_copies_then_drops_source() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g2())), 2);
        assert!(!ds.named_graph_exists(&g1())); // source dropped
    }

    #[test]
    fn test_move_clears_destination_before_writing() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t3()); // pre-existing
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g2())), 1);
        assert_eq!(ds.get_graph(Some(&g2()))[0], t1());
    }

    #[test]
    fn test_move_from_default_to_named() {
        let mut ds = dataset_with_default_triples(&[t1(), t2()]);
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Default,
            destination: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 2);
        assert_eq!(ds.triple_count(None), 0); // default cleared
    }

    #[test]
    fn test_move_from_named_to_default() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Default,
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(None), 1);
        assert!(!ds.named_graph_exists(&g1()));
    }

    #[test]
    fn test_move_self_to_self_is_noop() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 0);
        // Content preserved
        assert_eq!(ds.triple_count(Some(&g1())), 1);
    }

    #[test]
    fn test_move_nonexistent_source_without_silent_errors() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_move_nonexistent_source_with_silent_succeeds() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: true,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_ok());
    }

    #[test]
    fn test_move_reports_triples_affected() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());
        ds.add_triple(Some(&g1()), t3());
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 3);
    }

    // -----------------------------------------------------------------------
    // ADD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_merges_triples_into_destination() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2());
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g2())), 2); // t2 + t1
    }

    #[test]
    fn test_add_does_not_clear_destination() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g2()), t2()); // pre-existing
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        // Both original and new triples must be present
        let g2_triples = ds.get_graph(Some(&g2()));
        assert!(g2_triples.contains(&t1()));
        assert!(g2_triples.contains(&t2()));
    }

    #[test]
    fn test_add_leaves_source_unchanged() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 1);
    }

    #[test]
    fn test_add_from_default_to_named() {
        let mut ds = dataset_with_default_triples(&[t1(), t2()]);
        ds.add_triple(Some(&g1()), t3());
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Default,
            destination: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 3);
        assert_eq!(ds.triple_count(None), 2); // default unchanged
    }

    #[test]
    fn test_add_from_named_to_default() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(None, t2());
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Default,
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(None), 2);
        assert_eq!(ds.triple_count(Some(&g1())), 1); // unchanged
    }

    #[test]
    fn test_add_self_to_self_is_noop() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g1()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 0);
        assert_eq!(ds.triple_count(Some(&g1())), 1);
    }

    #[test]
    fn test_add_nonexistent_source_without_silent_errors() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_add_nonexistent_source_with_silent_succeeds() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: true,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_ok());
    }

    #[test]
    fn test_add_reports_triples_affected() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(result.triples_affected, 2);
    }

    // -----------------------------------------------------------------------
    // LOAD tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_without_silent_returns_error() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Load {
            iri: "http://example.org/data.ttl".to_owned(),
            into_graph: None,
            silent: false,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_err());
    }

    #[test]
    fn test_load_with_silent_returns_ok() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Load {
            iri: "http://example.org/data.ttl".to_owned(),
            into_graph: None,
            silent: true,
        };
        assert!(GraphManagementExecutor::execute(&op, &mut ds).is_ok());
    }

    #[test]
    fn test_load_silent_into_named_graph_returns_ok() {
        let mut ds = GraphManagementDataset::new();
        let op = GraphManagementOp::Load {
            iri: "http://example.org/data.ttl".to_owned(),
            into_graph: Some(g1()),
            silent: true,
        };
        let result = GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        // Silent load is a no-op; dataset unchanged
        assert_eq!(result.triples_affected, 0);
    }

    // -----------------------------------------------------------------------
    // Composed / interaction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_create_then_add_then_clear_lifecycle() {
        let mut ds = GraphManagementDataset::new();

        // 1. CREATE
        GraphManagementExecutor::execute(
            &GraphManagementOp::Create {
                graph: g1(),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();
        assert!(ds.named_graph_exists(&g1()));

        // 2. ADD from default (empty) to g1 — no-op effectively
        ds.add_triple(None, t1());
        GraphManagementExecutor::execute(
            &GraphManagementOp::Add {
                source: GraphManagementTarget::Default,
                destination: GraphManagementTarget::Named(g1()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 1);

        // 3. CLEAR g1
        GraphManagementExecutor::execute(
            &GraphManagementOp::Clear {
                target: GraphManagementTarget::Named(g1()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 0);
        assert!(ds.named_graph_exists(&g1())); // graph still exists after CLEAR
    }

    #[test]
    fn test_move_then_drop_leaves_clean_dataset() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        ds.add_triple(Some(&g1()), t2());

        // MOVE g1 -> g2
        GraphManagementExecutor::execute(
            &GraphManagementOp::Move {
                source: GraphManagementTarget::Named(g1()),
                destination: GraphManagementTarget::Named(g2()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();

        // DROP g2
        GraphManagementExecutor::execute(
            &GraphManagementOp::Drop {
                target: GraphManagementTarget::Named(g2()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();

        assert!(ds.graph_names().is_empty());
    }

    #[test]
    fn test_copy_then_add_accumulates_duplicates() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());

        // COPY g1 -> g2
        GraphManagementExecutor::execute(
            &GraphManagementOp::Copy {
                source: GraphManagementTarget::Named(g1()),
                destination: GraphManagementTarget::Named(g2()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();

        // ADD g1 -> g2 (g2 already has the triple; duplicates are allowed in a bag model)
        GraphManagementExecutor::execute(
            &GraphManagementOp::Add {
                source: GraphManagementTarget::Named(g1()),
                destination: GraphManagementTarget::Named(g2()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();

        assert_eq!(ds.triple_count(Some(&g2())), 2);
    }

    #[test]
    fn test_graph_management_result_default() {
        let r = GraphManagementResult::default();
        assert_eq!(r.triples_affected, 0);
        assert!(r.graphs_affected.is_empty());
    }

    #[test]
    fn test_target_label_default() {
        let label = GraphManagementExecutor::target_label(&GraphManagementTarget::Default);
        assert_eq!(label, "DEFAULT");
    }

    #[test]
    fn test_target_label_named() {
        let label = GraphManagementExecutor::target_label(&GraphManagementTarget::Named(g1()));
        assert_eq!(label, g1());
    }

    #[test]
    fn test_target_label_all() {
        let label = GraphManagementExecutor::target_label(&GraphManagementTarget::All);
        assert_eq!(label, "ALL");
    }

    #[test]
    fn test_target_label_all_named() {
        let label = GraphManagementExecutor::target_label(&GraphManagementTarget::AllNamed);
        assert_eq!(label, "NAMED");
    }

    #[test]
    fn test_graph_management_target_equality() {
        assert_eq!(
            GraphManagementTarget::Default,
            GraphManagementTarget::Default
        );
        assert_eq!(
            GraphManagementTarget::Named(g1()),
            GraphManagementTarget::Named(g1())
        );
        assert_ne!(
            GraphManagementTarget::Named(g1()),
            GraphManagementTarget::Named(g2())
        );
        assert_ne!(GraphManagementTarget::All, GraphManagementTarget::AllNamed);
    }

    #[test]
    fn test_add_to_nonexistent_destination_creates_it() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        // g2 does not exist yet
        let op = GraphManagementOp::Add {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert_eq!(ds.triple_count(Some(&g2())), 1);
    }

    #[test]
    fn test_copy_to_nonexistent_destination_creates_it() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Copy {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(ds.named_graph_exists(&g2()));
        assert_eq!(ds.triple_count(Some(&g2())), 1);
    }

    #[test]
    fn test_move_to_nonexistent_destination_creates_it() {
        let mut ds = GraphManagementDataset::new();
        ds.add_triple(Some(&g1()), t1());
        let op = GraphManagementOp::Move {
            source: GraphManagementTarget::Named(g1()),
            destination: GraphManagementTarget::Named(g2()),
            silent: false,
        };
        GraphManagementExecutor::execute(&op, &mut ds).unwrap();
        assert!(ds.named_graph_exists(&g2()));
        assert!(!ds.named_graph_exists(&g1()));
    }

    #[test]
    fn test_clear_named_graph_after_multiple_adds() {
        let mut ds = GraphManagementDataset::new();
        for t in [t1(), t2(), t3()] {
            ds.add_triple(Some(&g1()), t);
        }
        GraphManagementExecutor::execute(
            &GraphManagementOp::Clear {
                target: GraphManagementTarget::Named(g1()),
                silent: false,
            },
            &mut ds,
        )
        .unwrap();
        assert_eq!(ds.triple_count(Some(&g1())), 0);
    }

    #[test]
    fn test_drop_all_named_on_empty_dataset_ok() {
        let mut ds = GraphManagementDataset::new();
        let result = GraphManagementExecutor::execute(
            &GraphManagementOp::Drop {
                target: GraphManagementTarget::AllNamed,
                silent: false,
            },
            &mut ds,
        )
        .unwrap();
        assert_eq!(result.triples_affected, 0);
    }
}
