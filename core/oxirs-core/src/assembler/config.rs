//! Configuration types produced by the Jena Assembler parser.
//!
//! These types are the output of parsing a Jena `.ttl` assembler file. They
//! are intentionally simple (no Rc, no Arc, no RDF terms) so they can be
//! cheaply cloned and serialised for logging or diffing.

use std::path::PathBuf;

// ---------------------------------------------------------------------------
// StoreBackend
// ---------------------------------------------------------------------------

/// The storage backend for a dataset or graph resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreBackend {
    /// In-memory storage (`ja:MemoryModel` or `ja:MemoryDataset`).
    InMemory,

    /// TDB2 disk-based storage (`tdb2:DatasetTDB2`).
    ///
    /// The `location` is the path supplied by the `tdb2:location` literal.
    Tdb2 { location: PathBuf },

    /// An unrecognised backend type — the full class IRI is preserved so that
    /// callers can handle proprietary or future extensions without losing
    /// information.
    Unknown(String),
}

// ---------------------------------------------------------------------------
// GraphConfig
// ---------------------------------------------------------------------------

/// Configuration for one named graph (or the default graph) within a dataset.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// The IRI of the named graph, or `None` for the default graph.
    pub graph_name: Option<String>,

    /// The storage backend for this graph.
    pub backend: StoreBackend,

    /// URLs from which to load initial RDF content at startup.
    ///
    /// Populated from `ja:contentURL` triples on any model resource referenced
    /// by `ja:graph` or `ja:defaultGraph`.
    pub content_urls: Vec<String>,
}

// ---------------------------------------------------------------------------
// DatasetConfig
// ---------------------------------------------------------------------------

/// Top-level dataset configuration parsed from one assembler resource.
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// The IRI of the assembled dataset resource (the subject of `rdf:type …Dataset`).
    pub resource_iri: String,

    /// The primary storage backend for the dataset itself.
    pub backend: StoreBackend,

    /// Ordered list of named-graph configurations.
    pub named_graphs: Vec<GraphConfig>,

    /// Configuration for the default graph, if explicitly described.
    pub default_graph: Option<GraphConfig>,
}

// ---------------------------------------------------------------------------
// AssemblerConfig
// ---------------------------------------------------------------------------

/// The complete result of parsing one Jena Assembler document.
///
/// A single assembler file may define multiple datasets; each one becomes a
/// separate [`DatasetConfig`] entry.
///
/// # Example
/// ```rust
/// use oxirs_core::assembler::{AssemblerBuilder, AssemblerConfig};
///
/// let config = AssemblerBuilder::from_triples(&[]).unwrap();
/// assert!(config.is_empty());
/// ```
#[derive(Debug, Clone)]
pub struct AssemblerConfig {
    /// All dataset resources found in the assembler document.
    pub datasets: Vec<DatasetConfig>,
}

impl AssemblerConfig {
    /// Returns `true` when no datasets were found.
    pub fn is_empty(&self) -> bool {
        self.datasets.is_empty()
    }

    /// Returns the number of datasets.
    pub fn len(&self) -> usize {
        self.datasets.len()
    }

    /// Returns the first dataset whose `resource_iri` matches `iri`, or `None`.
    pub fn find_dataset(&self, iri: &str) -> Option<&DatasetConfig> {
        self.datasets.iter().find(|d| d.resource_iri == iri)
    }
}
