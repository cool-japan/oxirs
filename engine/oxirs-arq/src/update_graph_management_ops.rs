//! SPARQL 1.1 UPDATE Graph Management — Operation Execution
//!
//! Implements the [`GraphManagementExecutor`] which applies LOAD, CLEAR, DROP,
//! CREATE, COPY, MOVE, and ADD operations to a [`GraphManagementDataset`].

use anyhow::{anyhow, Result};

use crate::update_graph_management_types::{
    GraphManagementDataset, GraphManagementOp, GraphManagementResult, GraphManagementTarget, Triple,
};

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

    /// Clear the destination graph storage.
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
    pub fn target_label(target: &GraphManagementTarget) -> String {
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
