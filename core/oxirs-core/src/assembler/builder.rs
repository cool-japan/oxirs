//! Jena Assembler document parser.
//!
//! Converts a Turtle-format Jena Assembler document (or a pre-expanded set of
//! string triples) into an [`AssemblerConfig`] value.
//!
//! ## Design
//!
//! The implementation has two layers:
//!
//! 1. **[`AssemblerBuilder::from_triples`]** — the core graph-walking logic.
//!    Accepts `(subject, predicate, object)` string tuples (IRIs and literals
//!    in N-Triples-like form, already expanded), and produces the config.
//!    This is the primary testing seam: it works without any external parser.
//!
//! 2. **[`AssemblerBuilder::from_turtle`]** — uses `oxttl` (already a workspace
//!    dep) to parse Turtle into `oxrdf::Triple` values, then converts each
//!    triple into a string tuple and delegates to `from_triples`.
//!
//! ## Triple representation for `from_triples`
//!
//! Subjects and objects that are IRIs are stored as bare IRI strings
//! (e.g. `"http://example.org/ds"`).  Blank nodes are stored with a leading
//! `"_:"` sigil (e.g. `"_:b0"`).  Literal objects are stored with their
//! N-Triples quotation (e.g. `"\"/data/db\""` or `"\"hello\"^^<xsd:string>"`).
//! The builder strips the surrounding double-quotes when it extracts literal
//! values.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::PathBuf;

use super::config::{AssemblerConfig, DatasetConfig, GraphConfig, StoreBackend};
use super::vocab::{
    JA_CONTENT_URL, JA_DEFAULT_GRAPH, JA_GRAPH, JA_GRAPH_NAME, JA_MEMORY_DATASET, JA_MEMORY_MODEL,
    JA_NAMED_GRAPH, JA_RDF_DATASET, RDF_TYPE, TDB2_DATASET, TDB2_LOCATION,
};

// ---------------------------------------------------------------------------
// AssemblerError
// ---------------------------------------------------------------------------

/// Errors produced by the Jena Assembler parser.
#[derive(Debug)]
pub enum AssemblerError {
    /// The Turtle source could not be parsed.
    ParseError(String),

    /// A required triple was absent from the graph.
    MissingRequired { resource: String, property: String },

    /// The backend type IRI is recognised but could not be instantiated
    /// (e.g. `tdb2:DatasetTDB2` without a `tdb2:location`).
    InvalidLocation(String),
}

impl std::fmt::Display for AssemblerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssemblerError::ParseError(msg) => write!(f, "Assembler parse error: {msg}"),
            AssemblerError::MissingRequired { resource, property } => {
                write!(f, "Missing required property <{property}> on <{resource}>")
            }
            AssemblerError::InvalidLocation(msg) => {
                write!(f, "Invalid tdb2:location: {msg}")
            }
        }
    }
}

impl std::error::Error for AssemblerError {}

// ---------------------------------------------------------------------------
// Internal adjacency map
// ---------------------------------------------------------------------------

/// A minimal in-memory graph: subject → list of (predicate, object) pairs.
///
/// Keys may be bare IRIs or `_:`-prefixed blank-node IDs.
type AdjMap = HashMap<String, Vec<(String, String)>>;

fn adjacency_map(triples: &[(String, String, String)]) -> AdjMap {
    let mut map: AdjMap = HashMap::new();
    for (s, p, o) in triples {
        map.entry(s.clone())
            .or_default()
            .push((p.clone(), o.clone()));
    }
    map
}

/// Return all objects for the given subject + predicate pair.
fn objects_of<'a>(map: &'a AdjMap, subject: &str, predicate: &str) -> Vec<&'a str> {
    match map.get(subject) {
        None => vec![],
        Some(pairs) => pairs
            .iter()
            .filter(|(p, _)| p == predicate)
            .map(|(_, o)| o.as_str())
            .collect(),
    }
}

/// Return the first object for the given subject + predicate pair.
fn first_object<'a>(map: &'a AdjMap, subject: &str, predicate: &str) -> Option<&'a str> {
    objects_of(map, subject, predicate).into_iter().next()
}

// ---------------------------------------------------------------------------
// Literal stripping
// ---------------------------------------------------------------------------

/// Extract the lexical value from an N-Triples-like literal string.
///
/// Literals may arrive as:
/// - `"some text"` → `some text`
/// - `"some text"@en` → `some text`
/// - `"some text"^^<...>` → `some text`
///
/// Non-quoted strings are returned as-is (they are IRIs or blank nodes).
fn strip_literal(raw: &str) -> &str {
    if let Some(inner) = raw.strip_prefix('"') {
        // The closing quote may be followed by @lang or ^^type
        let end = inner.rfind('"').unwrap_or(inner.len());
        &inner[..end]
    } else {
        raw
    }
}

// ---------------------------------------------------------------------------
// Backend resolution
// ---------------------------------------------------------------------------

fn resolve_backend(
    map: &AdjMap,
    resource: &str,
    type_iri: &str,
) -> Result<StoreBackend, AssemblerError> {
    if type_iri == JA_MEMORY_MODEL || type_iri == JA_MEMORY_DATASET || type_iri == JA_RDF_DATASET {
        Ok(StoreBackend::InMemory)
    } else if type_iri == TDB2_DATASET {
        let loc_raw = first_object(map, resource, TDB2_LOCATION).ok_or_else(|| {
            AssemblerError::MissingRequired {
                resource: resource.to_owned(),
                property: TDB2_LOCATION.to_owned(),
            }
        })?;
        let loc_str = strip_literal(loc_raw);
        if loc_str.is_empty() {
            return Err(AssemblerError::InvalidLocation(
                "tdb2:location value is empty".to_owned(),
            ));
        }
        Ok(StoreBackend::Tdb2 {
            location: PathBuf::from(loc_str),
        })
    } else {
        Ok(StoreBackend::Unknown(type_iri.to_owned()))
    }
}

// ---------------------------------------------------------------------------
// GraphConfig resolution
// ---------------------------------------------------------------------------

/// Build a [`GraphConfig`] for a blank-node or IRI `graph_resource` subject.
///
/// `graph_name` is the named-graph IRI (or `None` for the default graph).
fn build_graph_config(
    map: &AdjMap,
    graph_resource: &str,
    graph_name: Option<String>,
) -> GraphConfig {
    // Determine backend from rdf:type on the model resource (best-effort; default InMemory)
    let backend = objects_of(map, graph_resource, RDF_TYPE)
        .into_iter()
        .find_map(|type_iri| resolve_backend(map, graph_resource, type_iri).ok())
        .unwrap_or(StoreBackend::InMemory);

    // Collect ja:contentURL values from the model resource itself and any
    // linked ja:content blank node.
    let mut content_urls: Vec<String> = Vec::new();

    // Direct contentURL on model resource
    for url_raw in objects_of(map, graph_resource, JA_CONTENT_URL) {
        content_urls.push(strip_literal(url_raw).to_owned());
    }

    // Indirect via ja:content → blank node → ja:contentURL
    for content_bnode in objects_of(map, graph_resource, super::vocab::JA_CONTENT) {
        for url_raw in objects_of(map, content_bnode, JA_CONTENT_URL) {
            content_urls.push(strip_literal(url_raw).to_owned());
        }
    }

    GraphConfig {
        graph_name,
        backend,
        content_urls,
    }
}

// ---------------------------------------------------------------------------
// DatasetConfig resolution
// ---------------------------------------------------------------------------

fn build_dataset_config(
    map: &AdjMap,
    resource: &str,
    type_iri: &str,
) -> Result<DatasetConfig, AssemblerError> {
    let backend = resolve_backend(map, resource, type_iri)?;

    // Collect named-graph descriptions: ja:namedGraph → blank node with
    //   ja:graphName <iri> and ja:graph <model>
    let mut named_graphs: Vec<GraphConfig> = Vec::new();

    for ng_bnode in objects_of(map, resource, JA_NAMED_GRAPH) {
        // ja:graphName gives the named-graph IRI
        let graph_name = first_object(map, ng_bnode, JA_GRAPH_NAME).map(|s| s.to_owned());

        // ja:graph gives the model resource
        if let Some(model_resource) = first_object(map, ng_bnode, JA_GRAPH) {
            named_graphs.push(build_graph_config(map, model_resource, graph_name));
        } else {
            // No model resource — treat the blank node itself as the model
            named_graphs.push(build_graph_config(map, ng_bnode, graph_name));
        }
    }

    // Default graph: ja:defaultGraph → model resource
    let default_graph = first_object(map, resource, JA_DEFAULT_GRAPH)
        .map(|model_resource| build_graph_config(map, model_resource, None));

    Ok(DatasetConfig {
        resource_iri: resource.to_owned(),
        backend,
        named_graphs,
        default_graph,
    })
}

// ---------------------------------------------------------------------------
// AssemblerBuilder
// ---------------------------------------------------------------------------

/// Parses Jena Assembler documents into [`AssemblerConfig`] values.
pub struct AssemblerBuilder;

impl AssemblerBuilder {
    /// Parse an `(subject, predicate, object)` triple set into an
    /// [`AssemblerConfig`].
    ///
    /// Each element of the slice is a `(String, String, String)` tuple where:
    /// - Subjects are bare IRI strings or `"_:id"` for blank nodes.
    /// - Predicates are bare IRI strings.
    /// - Objects are bare IRI strings, `"_:id"` blank nodes, or N-Triples
    ///   quoted literals (e.g. `"\"/data/db\""`).
    ///
    /// This function is the primary testing seam; it does not require a Turtle
    /// parser and works entirely from pre-expanded triples.
    pub fn from_triples(
        triples: &[(String, String, String)],
    ) -> Result<AssemblerConfig, AssemblerError> {
        let map = adjacency_map(triples);

        // Collect all subjects that are typed as dataset resources.
        //
        // Strategy: find every subject that has an `rdf:type` triple whose
        // object is one of the recognised Jena/TDB2 dataset class IRIs.  When
        // the type IRI is unrecognised, a `DatasetConfig` with
        // `backend: Unknown(type_iri)` is still produced — this preserves
        // information for callers that handle proprietary or future extensions.
        //
        // Blank-node subjects are skipped: they are intermediate nodes (e.g.
        // named-graph descriptions), not top-level dataset resources.
        let mut datasets: Vec<DatasetConfig> = Vec::new();

        // Collect (subject, type_iri) pairs for all typed, non-blank subjects
        let typed_subjects: Vec<(String, String)> = map
            .iter()
            .filter(|(subject, _)| !subject.starts_with("_:"))
            .flat_map(|(subject, pairs)| {
                pairs
                    .iter()
                    .filter(|(pred, _)| pred == RDF_TYPE)
                    .map(|(_, obj)| (subject.clone(), obj.clone()))
                    .collect::<Vec<_>>()
            })
            .collect();

        for (subject, type_iri) in typed_subjects {
            match build_dataset_config(&map, &subject, &type_iri) {
                Ok(cfg) => datasets.push(cfg),
                Err(e) => return Err(e),
            }
        }

        // Stable ordering by resource IRI for deterministic output
        datasets.sort_by(|a, b| a.resource_iri.cmp(&b.resource_iri));

        Ok(AssemblerConfig { datasets })
    }

    /// Parse a Turtle-format Jena Assembler document into an
    /// [`AssemblerConfig`].
    ///
    /// Uses `oxttl` (a workspace dependency) to parse the Turtle, then
    /// delegates to [`Self::from_triples`].
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_core::assembler::AssemblerBuilder;
    ///
    /// let ttl = r#"
    ///     @prefix ja: <http://jena.hpl.hp.com/2005/11/Assembler#> .
    ///     <http://example.org/ds> a ja:MemoryDataset .
    /// "#;
    /// let config = AssemblerBuilder::from_turtle(ttl).unwrap();
    /// assert_eq!(config.len(), 1);
    /// ```
    pub fn from_turtle(input: &str) -> Result<AssemblerConfig, AssemblerError> {
        let reader = Cursor::new(input.as_bytes());
        let parser = oxttl::TurtleParser::new().lenient();

        let mut triples: Vec<(String, String, String)> = Vec::new();

        for result in parser.for_reader(reader) {
            match result {
                Ok(triple) => {
                    let subject = subject_to_key(&triple.subject);
                    let predicate = triple.predicate.as_str().to_owned();
                    let object = term_to_value(&triple.object);
                    triples.push((subject, predicate, object));
                }
                Err(e) => {
                    return Err(AssemblerError::ParseError(e.to_string()));
                }
            }
        }

        Self::from_triples(&triples)
    }
}

// ---------------------------------------------------------------------------
// oxrdf term → string helpers
// ---------------------------------------------------------------------------

fn subject_to_key(subject: &oxrdf::NamedOrBlankNode) -> String {
    match subject {
        oxrdf::NamedOrBlankNode::NamedNode(n) => n.as_str().to_owned(),
        oxrdf::NamedOrBlankNode::BlankNode(b) => format!("_:{}", b.as_str()),
    }
}

fn term_to_value(term: &oxrdf::Term) -> String {
    match term {
        oxrdf::Term::NamedNode(n) => n.as_str().to_owned(),
        oxrdf::Term::BlankNode(b) => format!("_:{}", b.as_str()),
        oxrdf::Term::Literal(lit) => {
            // Store as a quoted string so strip_literal can unwrap it later
            format!("\"{}\"", lit.value())
        }
        // RDF-star triple terms — use N-Triples form
        #[allow(unreachable_patterns)]
        _ => term.to_string(),
    }
}
