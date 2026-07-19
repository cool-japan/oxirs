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
            } => Self::execute_load(iri, into_graph.as_deref(), *silent, dataset),
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
        into_graph: Option<&str>,
        silent: bool,
        dataset: &mut GraphManagementDataset,
    ) -> Result<GraphManagementResult> {
        // A real HTTP fetch + RDF parse. On failure, `SILENT` suppresses the
        // error (returning a no-op result) while a non-`SILENT` LOAD surfaces
        // it — the two outcomes are therefore distinguishable in the error path
        // (an empty document loads zero triples with Ok, a failed fetch errors).
        match Self::load_document(iri, into_graph, dataset) {
            Ok(result) => Ok(result),
            Err(e) if silent => {
                tracing::warn!("SILENT LOAD <{iri}> failed, treating as no-op: {e}");
                Ok(GraphManagementResult::default())
            }
            Err(e) => Err(e),
        }
    }

    /// Fetch the document at `iri` over HTTP, parse it as RDF, and insert the
    /// resulting triples into `into_graph` (default graph when `None`).
    fn load_document(
        iri: &str,
        into_graph: Option<&str>,
        dataset: &mut GraphManagementDataset,
    ) -> Result<GraphManagementResult> {
        use oxirs_core::parser::Parser;

        let (body, content_type) =
            crate::service_federation::http_get_document(iri, std::time::Duration::from_secs(60))?;

        let format = detect_rdf_format(iri, content_type.as_deref());
        let triples = Parser::new(format)
            .parse_str_to_triples(&body)
            .map_err(|e| anyhow!("failed to parse LOAD <{iri}> as {format:?}: {e}"))?;

        let mut result = GraphManagementResult::default();
        for triple in &triples {
            dataset.add_triple(
                into_graph,
                Triple::new(
                    subject_to_string(triple.subject()),
                    predicate_to_string(triple.predicate()),
                    object_to_string(triple.object()),
                ),
            );
        }
        result.triples_affected = triples.len();
        result
            .graphs_affected
            .push(into_graph.map_or_else(|| "DEFAULT".to_owned(), str::to_owned));
        Ok(result)
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

// ---------------------------------------------------------------------------
// LOAD helpers
// ---------------------------------------------------------------------------

/// Choose an [`oxirs_core::parser::RdfFormat`] from the reported `Content-Type`
/// header (preferred) or, failing that, the document IRI's file extension.
/// Defaults to Turtle, the most common RDF interchange format.
fn detect_rdf_format(iri: &str, content_type: Option<&str>) -> oxirs_core::parser::RdfFormat {
    use oxirs_core::parser::RdfFormat;

    if let Some(ct) = content_type {
        let ct = ct
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        match ct.as_str() {
            "text/turtle" | "application/x-turtle" => return RdfFormat::Turtle,
            "application/n-triples" | "text/plain" => return RdfFormat::NTriples,
            "application/n-quads" => return RdfFormat::NQuads,
            "application/trig" => return RdfFormat::TriG,
            "application/rdf+xml" | "text/xml" | "application/xml" => return RdfFormat::RdfXml,
            "application/ld+json" | "application/json" => return RdfFormat::JsonLd,
            _ => {}
        }
    }

    // Fall back to the IRI's file extension.
    if let Some(ext) = iri.rsplit('.').next() {
        if let Some(fmt) = RdfFormat::from_extension(ext) {
            return fmt;
        }
    }

    RdfFormat::Turtle
}

/// Render a triple subject as the plain string used by this in-memory model
/// (bare IRIs, `_:` blank nodes).
fn subject_to_string(subject: &oxirs_core::model::Subject) -> String {
    use oxirs_core::model::Subject;
    match subject {
        Subject::NamedNode(n) => n.as_str().to_owned(),
        Subject::BlankNode(b) => format!("_:{}", b.as_str()),
        Subject::Variable(v) => format!("?{}", v.as_str()),
        Subject::QuotedTriple(qt) => format!("{qt}"),
    }
}

/// Render a triple predicate as a plain IRI string.
fn predicate_to_string(predicate: &oxirs_core::model::Predicate) -> String {
    use oxirs_core::model::Predicate;
    match predicate {
        Predicate::NamedNode(n) => n.as_str().to_owned(),
        Predicate::Variable(v) => format!("?{}", v.as_str()),
    }
}

/// Render a triple object as a plain string, keeping literals unambiguous by
/// preserving quotes, language tags and datatype IRIs.
fn object_to_string(object: &oxirs_core::model::Object) -> String {
    use oxirs_core::model::Object;
    match object {
        Object::NamedNode(n) => n.as_str().to_owned(),
        Object::BlankNode(b) => format!("_:{}", b.as_str()),
        Object::Variable(v) => format!("?{}", v.as_str()),
        Object::QuotedTriple(qt) => format!("{qt}"),
        Object::Literal(l) => {
            if let Some(lang) = l.language() {
                format!("\"{}\"@{}", l.value(), lang)
            } else {
                let dt = l.datatype().into_owned();
                if dt.as_str() == "http://www.w3.org/2001/XMLSchema#string" {
                    format!("\"{}\"", l.value())
                } else {
                    format!("\"{}\"^^<{}>", l.value(), dt.as_str())
                }
            }
        }
    }
}
