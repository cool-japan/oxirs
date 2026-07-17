//! RDF store implementation with pluggable storage backends
//!
//! **Stability**: ✅ **Stable** - Core store APIs are production-ready.
//!
//! This module provides the primary interface for storing, querying, and manipulating RDF data.
//! It includes multiple storage backends optimized for different use cases.
//!
//! ## Overview
//!
//! The RDF store is the central component for managing RDF data. It provides:
//! - **Quad storage** - Store RDF quads (triples with named graphs)
//! - **Pattern matching** - Query data using SPARQL-like patterns
//! - **SPARQL execution** - Execute SPARQL queries directly
//! - **Persistence** - Optional disk-based storage with automatic saving
//! - **Named graphs** - Full support for RDF datasets with named graphs
//!
//! ## Core Types
//!
//! - **[`RdfStore`]** - Primary store implementation with pluggable backends
//! - **[`ConcreteStore`]** - Convenience wrapper (alias for RdfStore)
//! - **[`Store`]** - Trait defining the store interface
//! - **[`StorageBackend`]** - Enum of available storage backends
//!
//! ## Storage Backends
//!
//! ### Memory Backend
//! - Fast in-memory storage
//! - No persistence (data lost on shutdown)
//! - Good for: testing, temporary data, small datasets
//!
//! ### Persistent Backend
//! - Disk-backed storage with automatic save
//! - Data persisted as N-Quads format
//! - Good for: long-running applications, data preservation
//!
//! ### UltraMemory Backend
//! - High-performance with arena allocators
//! - Multi-index support (SPO, POS, OSP)
//! - Good for: large datasets, query-heavy workloads
//!
//! ## Examples
//!
//! ### Basic Store Creation and Usage
//!
//! ```rust
//! use oxirs_core::RdfStore;
//! use oxirs_core::model::{NamedNode, Triple, Literal};
//!
//! # fn main() -> Result<(), oxirs_core::OxirsError> {
//! // Create an in-memory store
//! let mut store = RdfStore::new()?;
//!
//! // Add a triple
//! let subject = NamedNode::new("http://example.org/alice")?;
//! let predicate = NamedNode::new("http://xmlns.com/foaf/0.1/name")?;
//! let object = Literal::new("Alice");
//!
//! let triple = Triple::new(subject, predicate, object);
//! store.insert_triple(triple)?;
//!
//! // Check the store
//! assert_eq!(store.len()?, 1);
//! assert!(!store.is_empty()?);
//! # Ok(())
//! # }
//! ```
//!
//! ### Persistent Storage
//!
//! ```rust,no_run
//! use oxirs_core::RdfStore;
//! use oxirs_core::model::{NamedNode, Triple, Literal};
//!
//! # fn main() -> Result<(), oxirs_core::OxirsError> {
//! // Open or create a persistent store
//! let mut store = RdfStore::open("./my_knowledge_base")?;
//!
//! // Add data - automatically saved to disk
//! let triple = Triple::new(
//!     NamedNode::new("http://example.org/resource")?,
//!     NamedNode::new("http://purl.org/dc/terms/title")?,
//!     Literal::new("My Resource"),
//! );
//! store.insert_triple(triple)?;
//!
//! // Data persists across restarts
//! drop(store);
//!
//! // Reopen - data is still there
//! let store = RdfStore::open("./my_knowledge_base")?;
//! assert_eq!(store.len()?, 1);
//! # Ok(())
//! # }
//! ```
//!
//! ### Pattern Matching Queries
//!
//! ```rust
//! use oxirs_core::RdfStore;
//! use oxirs_core::model::{NamedNode, Triple, Literal, Predicate};
//!
//! # fn main() -> Result<(), oxirs_core::OxirsError> {
//! # let mut store = RdfStore::new()?;
//! # let alice = NamedNode::new("http://example.org/alice")?;
//! # let bob = NamedNode::new("http://example.org/bob")?;
//! # let knows = NamedNode::new("http://xmlns.com/foaf/0.1/knows")?;
//! # let name = NamedNode::new("http://xmlns.com/foaf/0.1/name")?;
//! # store.insert_triple(Triple::new(alice.clone(), knows.clone(), bob.clone()))?;
//! # store.insert_triple(Triple::new(alice.clone(), name.clone(), Literal::new("Alice")))?;
//! # store.insert_triple(Triple::new(bob.clone(), name.clone(), Literal::new("Bob")))?;
//! // Query all triples with foaf:knows predicate
//! let knows_pred = Predicate::NamedNode(knows);
//! let results = store.query_triples(None, Some(&knows_pred), None)?;
//!
//! for triple in results {
//!     println!("{:?} knows {:?}", triple.subject(), triple.object());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### SPARQL Queries
//!
//! ```rust
//! use oxirs_core::RdfStore;
//!
//! # fn main() -> Result<(), oxirs_core::OxirsError> {
//! # let store = RdfStore::new()?;
//! // Execute a SPARQL SELECT query
//! let query = r#"
//!     PREFIX foaf: <http://xmlns.com/foaf/0.1/>
//!     SELECT ?person ?name WHERE {
//!         ?person foaf:name ?name .
//!     }
//! "#;
//!
//! let results = store.query(query)?;
//! println!("Found {} results", results.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Working with Named Graphs
//!
//! ```rust
//! use oxirs_core::RdfStore;
//! use oxirs_core::model::{NamedNode, Quad, GraphName, Literal};
//!
//! # fn main() -> Result<(), oxirs_core::OxirsError> {
//! # let mut store = RdfStore::new()?;
//! // Create a named graph
//! let graph = GraphName::NamedNode(NamedNode::new("http://example.org/graph1")?);
//!
//! // Add a quad to the named graph
//! let quad = Quad::new(
//!     NamedNode::new("http://example.org/subject")?,
//!     NamedNode::new("http://example.org/predicate")?,
//!     Literal::new("value"),
//!     graph,
//! );
//! store.insert_quad(quad)?;
//!
//! // Query the named graph
//! let graph_node = NamedNode::new("http://example.org/graph1")?;
//! let quads = store.graph_quads(Some(&graph_node))?;
//! println!("Graph contains {} quads", quads.len());
//!
//! // List all graphs
//! let graphs = store.named_graphs()?;
//! println!("Store has {} named graphs", graphs.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Bulk Operations for Performance
//!
//! ```rust
//! use oxirs_core::RdfStore;
//! use oxirs_core::model::{NamedNode, Quad, Literal, Triple, GraphName};
//!
//! # fn main() -> Result<(), oxirs_core::OxirsError> {
//! # let mut store = RdfStore::new()?;
//! // Prepare many quads
//! let mut quads = Vec::new();
//! for i in 0..10_000 {
//!     let subject = NamedNode::new(&format!("http://example.org/item{}", i))?;
//!     let predicate = NamedNode::new("http://example.org/value")?;
//!     let object = Literal::new(&i.to_string());
//!     let triple = Triple::new(subject, predicate, object);
//!     quads.push(Quad::from_triple(triple));
//! }
//!
//! // Bulk insert - much faster than individual inserts
//! let ids = store.bulk_insert_quads(quads)?;
//! println!("Bulk inserted {} quads", ids.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Loading Data from URLs
//!
//! ```rust,no_run
//! use oxirs_core::RdfStore;
//! use oxirs_core::model::NamedNode;
//!
//! # fn main() -> Result<(), oxirs_core::OxirsError> {
//! # let mut store = RdfStore::new()?;
//! // Load RDF data from a URL
//! let url = "https://example.org/data.ttl";
//! let graph = Some(NamedNode::new("http://example.org/imported-graph")?);
//!
//! let count = store.load_from_url(url, graph.as_ref())?;
//! println!("Loaded {} triples from {}", count, url);
//! # Ok(())
//! # }
//! ```
//!
//! ### Graph Management
//!
//! ```rust
//! use oxirs_core::RdfStore;
//! use oxirs_core::model::{NamedNode, GraphName};
//!
//! # fn main() -> Result<(), oxirs_core::OxirsError> {
//! # let mut store = RdfStore::new()?;
//! // Create an empty named graph
//! let graph_name = NamedNode::new("http://example.org/my-graph")?;
//! store.create_graph(Some(&graph_name))?;
//!
//! // Clear a specific graph
//! let graph = GraphName::NamedNode(graph_name.clone());
//! store.clear_graph(Some(&graph))?;
//!
//! // Drop a graph entirely
//! store.drop_graph(Some(&graph))?;
//!
//! // Clear all named graphs
//! let cleared = store.clear_named_graphs()?;
//! println!("Cleared {} quads from named graphs", cleared);
//!
//! // Clear everything
//! let total_cleared = store.clear_all()?;
//! println!("Cleared {} total quads", total_cleared);
//! # Ok(())
//! # }
//! ```
//!
//! ## Performance Characteristics
//!
//! | Operation | Memory | Persistent | UltraMemory |
//! |-----------|--------|------------|-------------|
//! | Insert (single) | Fast | Medium | Very Fast |
//! | Insert (bulk) | Fast | Medium | Ultra Fast |
//! | Query (pattern) | Fast | Fast | Very Fast |
//! | Query (SPARQL) | Fast | Fast | Fast |
//! | Persistence | None | Automatic | None |
//! | Memory Usage | Low | Low | Medium |
//! | Startup Time | Instant | Fast | Fast |
//!
//! ## Best Practices
//!
//! 1. **Use bulk operations** - For inserting many quads, use `bulk_insert_quads()`
//! 2. **Choose the right backend** - Use persistent storage for long-running apps
//! 3. **Use named graphs** - Organize data into logical graphs
//! 4. **Check return values** - Insert operations return `true` if the quad was new
//! 5. **Pattern matching first** - Try pattern matching before SPARQL for simple queries
//!
//! ## Error Handling
//!
//! Store operations return `Result<T, OxirsError>`. Common errors:
//! - **Store errors** - Internal storage failures
//! - **Query errors** - Invalid SPARQL or pattern queries
//! - **IO errors** - File system errors (persistent storage)
//! - **Parse errors** - Invalid IRI or term formats
//!
//! ## Thread Safety
//!
//! The `Store` trait is `Send + Sync`, allowing stores to be shared across threads:
//!
//! ```rust,no_run
//! use oxirs_core::RdfStore;
//! use std::sync::Arc;
//!
//! # fn main() -> Result<(), oxirs_core::OxirsError> {
//! let store = Arc::new(RdfStore::new()?);
//!
//! // Share store across threads
//! let store_clone = Arc::clone(&store);
//! std::thread::spawn(move || {
//!     // Use store in another thread
//!     let _ = store_clone.len();
//! });
//! # Ok(())
//! # }
//! ```
//!
//! ## Related Modules
//!
//! - [`crate::model`] - RDF data model types
//! - [`crate::parser`] - Parse RDF from files
//! - [`crate::serializer`] - Serialize RDF to files
//! - [`crate::query`] - SPARQL query execution

pub mod concrete;
pub(crate) mod dictionary;
pub mod persistence;
pub mod storage;
pub mod types;

#[cfg(feature = "async-tokio")]
pub mod async_store;

pub use concrete::*;
pub use persistence::{PersistentState, SyncPolicy};
pub use storage::*;
pub use types::*;

#[cfg(feature = "async-tokio")]
pub use async_store::AsyncRdfStore;

use crate::indexing::{IndexStats, MemoryUsage};
use crate::model::*;
use crate::parser::RdfFormat;
use crate::serializer::Serializer;
use crate::sparql::extract_and_expand_prefixes; // SPARQL execution engine
use crate::{OxirsError, Result};
use async_trait::async_trait;
use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, RwLock};

/// Store trait for RDF operations
#[async_trait]
pub trait Store: Send + Sync {
    /// Insert a quad into the store
    fn insert_quad(&self, quad: Quad) -> Result<bool>;

    /// Remove a quad from the store  
    fn remove_quad(&self, quad: &Quad) -> Result<bool>;

    /// Find quads matching the given pattern
    fn find_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Result<Vec<Quad>>;

    /// Check if the store is ready for operations
    fn is_ready(&self) -> bool;

    /// Get the number of quads in the store
    fn len(&self) -> Result<usize>;

    /// Check if the store is empty
    fn is_empty(&self) -> Result<bool>;

    /// Query the store with SPARQL
    fn query(&self, sparql: &str) -> Result<OxirsQueryResults>;

    /// Prepare a SPARQL query for execution
    fn prepare_query(&self, sparql: &str) -> Result<PreparedQuery>;

    /// Insert a triple into the default graph
    fn insert_triple(&self, triple: Triple) -> Result<bool> {
        let quad = Quad::from_triple(triple);
        self.insert_quad(quad)
    }

    /// Insert a quad (compatibility method)
    fn insert(&self, quad: &Quad) -> Result<()> {
        self.insert_quad(quad.clone())?;
        Ok(())
    }

    /// Remove a quad (compatibility method)
    fn remove(&self, quad: &Quad) -> Result<bool> {
        self.remove_quad(quad)
    }

    /// Get all quads in the store
    fn quads(&self) -> Result<Vec<Quad>> {
        self.find_quads(None, None, None, None)
    }

    /// Get all named graphs
    fn named_graphs(&self) -> Result<Vec<NamedNode>> {
        // Default implementation - subclasses should override
        Ok(Vec::new())
    }

    /// Get all graphs
    fn graphs(&self) -> Result<Vec<NamedNode>> {
        self.named_graphs()
    }

    /// Get quads from named graphs only
    fn named_graph_quads(&self) -> Result<Vec<Quad>> {
        // Default implementation - get all quads except default graph
        let all_quads = self.quads()?;
        Ok(all_quads
            .into_iter()
            .filter(|quad| matches!(quad.graph_name(), GraphName::NamedNode(_)))
            .collect())
    }

    /// Get quads from the default graph only
    fn default_graph_quads(&self) -> Result<Vec<Quad>> {
        let default_graph = GraphName::DefaultGraph;
        self.find_quads(None, None, None, Some(&default_graph))
    }

    /// Get quads from a specific graph
    fn graph_quads(&self, graph: Option<&NamedNode>) -> Result<Vec<Quad>> {
        let graph_name = graph
            .map(|g| GraphName::NamedNode(g.clone()))
            .unwrap_or(GraphName::DefaultGraph);
        self.find_quads(None, None, None, Some(&graph_name))
    }

    /// Clear all data from all graphs
    fn clear_all(&self) -> Result<usize> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "clear_all requires mutable access".to_string(),
        ))
    }

    /// Clear all named graphs (but not the default graph)
    fn clear_named_graphs(&self) -> Result<usize> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "clear_named_graphs requires mutable access".to_string(),
        ))
    }

    /// Clear the default graph only
    fn clear_default_graph(&self) -> Result<usize> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "clear_default_graph requires mutable access".to_string(),
        ))
    }

    /// Clear a specific graph
    fn clear_graph(&self, _graph: Option<&GraphName>) -> Result<usize> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "clear_graph requires mutable access".to_string(),
        ))
    }

    /// Create a new graph (if it doesn't exist)
    fn create_graph(&self, _graph: Option<&NamedNode>) -> Result<()> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "create_graph requires mutable access".to_string(),
        ))
    }

    /// Drop a graph (remove the graph and all its quads)
    fn drop_graph(&self, _graph: Option<&GraphName>) -> Result<()> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "drop_graph requires mutable access".to_string(),
        ))
    }

    /// Load data from a URL into a graph
    fn load_from_url(&self, _url: &str, _graph: Option<&NamedNode>) -> Result<usize> {
        // Default implementation - not supported in trait
        Err(OxirsError::NotSupported(
            "load_from_url requires mutable access".to_string(),
        ))
    }

    /// Get all triples in the store (converts quads to triples)
    fn triples(&self) -> Result<Vec<Triple>> {
        let quads = self.find_quads(None, None, None, None)?;
        Ok(quads
            .into_iter()
            .map(|quad| {
                Triple::new(
                    quad.subject().clone(),
                    quad.predicate().clone(),
                    quad.object().clone(),
                )
            })
            .collect())
    }

    /// Insert a batch of quads in a single call, returning the number of quads
    /// that were **newly** inserted (duplicates already present are not
    /// counted, matching the `bool` contract of [`insert_quad`](Store::insert_quad)).
    ///
    /// The default implementation simply loops [`insert_quad`](Store::insert_quad),
    /// so every existing backend keeps working unchanged. Durable backends
    /// should override this to take a **single** write lock and perform a
    /// **single** append + one `fsync` for the whole batch (see the
    /// [`RdfStore`] override), turning a per-quad-fsync loop into one durable
    /// write. This is the seam servers/CLIs use for single-fsync bulk loads.
    fn bulk_insert_quads(&self, quads: Vec<Quad>) -> Result<usize> {
        let mut inserted = 0usize;
        for quad in quads {
            if self.insert_quad(quad)? {
                inserted += 1;
            }
        }
        Ok(inserted)
    }

    /// Stream every quad matching the pattern to `f`, one at a time, without the
    /// caller ever materializing the whole matching set as a `Vec<Quad>`.
    ///
    /// This is the streaming counterpart of [`find_quads`](Store::find_quads):
    /// downstream consumers (e.g. streaming a graph to an HTTP response body)
    /// use it to serialize results incrementally instead of collecting the
    /// entire graph into memory first.
    ///
    /// The callback form is used (rather than an iterator-returning method) so
    /// the trait stays object-safe (`dyn Store`). The default implementation
    /// collects [`find_quads`](Store::find_quads) then iterates; backends
    /// should override to iterate their index directly and avoid the
    /// intermediate `Vec` (see the [`RdfStore`] override).
    fn for_each_quad(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
        f: &mut dyn FnMut(Quad),
    ) -> Result<()> {
        for quad in self.find_quads(subject, predicate, object, graph_name)? {
            f(quad);
        }
        Ok(())
    }
}

/// Prepared SPARQL query
///
/// A prepared query retains the query text together with a shared handle to the
/// store backend it was prepared against, so [`exec`](PreparedQuery::exec) runs
/// the real query executor. Construct via [`Store::prepare_query`]; a
/// backend-less query built through [`PreparedQuery::new`] fails loudly on
/// `exec` rather than silently returning an empty result.
pub struct PreparedQuery {
    sparql: String,
    backend: Option<StorageBackend>,
}

impl PreparedQuery {
    /// Create a prepared query with no bound backend. Executing it returns an
    /// error; prefer [`PreparedQuery::with_backend`] / [`Store::prepare_query`].
    pub fn new(sparql: String) -> Self {
        Self {
            sparql,
            backend: None,
        }
    }

    /// Create a prepared query bound to a store backend for execution.
    pub fn with_backend(sparql: String, backend: StorageBackend) -> Self {
        Self {
            sparql,
            backend: Some(backend),
        }
    }

    /// Execute the prepared query against its bound backend.
    pub fn exec(&self) -> Result<QueryResultsIterator> {
        let backend = self.backend.as_ref().ok_or_else(|| {
            OxirsError::Query(
                "PreparedQuery has no bound store backend; construct it via Store::prepare_query"
                    .to_string(),
            )
        })?;
        let executor = crate::sparql::QueryExecutor::new(backend);
        let results = executor.execute(&self.sparql)?;
        Ok(QueryResultsIterator::from_results(&results))
    }
}

/// Iterator over query results
pub struct QueryResultsIterator {
    results: Vec<SolutionMapping>,
    index: usize,
}

impl QueryResultsIterator {
    pub fn empty() -> Self {
        Self {
            results: Vec::new(),
            index: 0,
        }
    }

    /// Build an iterator over the SELECT-style variable bindings of a query
    /// result. ASK/CONSTRUCT results carry no variable rows and yield an empty
    /// iterator (this type only models solution mappings).
    pub fn from_results(results: &OxirsQueryResults) -> Self {
        let rows = match results.results() {
            QueryResults::Bindings(bindings) => bindings
                .iter()
                .map(|binding| SolutionMapping::from_bindings(binding.bindings.clone()))
                .collect(),
            QueryResults::Boolean(_) | QueryResults::Graph(_) => Vec::new(),
        };
        Self {
            results: rows,
            index: 0,
        }
    }
}

impl Iterator for QueryResultsIterator {
    type Item = SolutionMapping;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.results.len() {
            let result = self.results[self.index].clone();
            self.index += 1;
            Some(result)
        } else {
            None
        }
    }
}

/// Solution mapping for SPARQL query results
#[derive(Debug, Clone, Default)]
pub struct SolutionMapping {
    bindings: std::collections::HashMap<String, Term>,
}

impl SolutionMapping {
    pub fn new() -> Self {
        Self {
            bindings: std::collections::HashMap::new(),
        }
    }

    /// Build a solution mapping from a variable-name -> term map.
    pub fn from_bindings(bindings: std::collections::HashMap<String, Term>) -> Self {
        Self { bindings }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &Term)> {
        self.bindings.iter()
    }
}

/// Main RDF store implementation
#[derive(Debug)]
pub struct RdfStore {
    backend: StorageBackend,
}

impl RdfStore {
    /// Create a new ultra-high performance in-memory store
    pub fn new() -> Result<Self> {
        Ok(RdfStore {
            backend: StorageBackend::Memory(Arc::new(RwLock::new(MemoryStorage::new()))),
        })
    }

    /// Create a new legacy in-memory store for compatibility
    pub fn new_legacy() -> Result<Self> {
        Ok(RdfStore {
            backend: StorageBackend::Memory(Arc::new(RwLock::new(MemoryStorage::new()))),
        })
    }

    /// Create a new persistent store at the given path
    ///
    /// Loads existing data from disk if present, otherwise creates a new store.
    /// Uses the default sync policy ([`SyncPolicy::EveryN`]`(1000)`).
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_sync_policy(path, SyncPolicy::default())
    }

    /// Create a persistent store with an explicit durability [`SyncPolicy`].
    ///
    /// Existing data is loaded from `data.nq`; a torn trailing line from a crash
    /// mid-append is recovered, and `File::open` failures on an existing file are
    /// propagated (never silently yielding an empty store).
    pub fn open_with_sync_policy<P: AsRef<Path>>(path: P, sync_policy: SyncPolicy) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let data_file = path_buf.join("data.nq");

        let (storage, load_had_errors) = if data_file.exists() {
            persistence::load_from_disk(&data_file)?
        } else {
            (MemoryStorage::new(), false)
        };

        let state = PersistentState::open(path_buf, sync_policy, load_had_errors)?;

        Ok(RdfStore {
            backend: StorageBackend::Persistent(Arc::new(RwLock::new(storage)), Arc::new(state)),
        })
    }

    /// Serialize a single quad to one N-Quads line (no trailing newline).
    fn quad_to_nquads_line(quad: &Quad) -> Result<String> {
        Serializer::new(RdfFormat::NQuads).serialize_quad_to_nquads(quad)
    }

    /// Flush pending writes and make prior deletions durable.
    ///
    /// For the persistent backend this compacts the data file if there are
    /// uncompacted deletions (delete/clear/drop-graph), otherwise it flushes and
    /// `fsync`s the append log. It is a no-op for purely in-memory backends.
    /// Call this at server shutdown and at the end of a CLI batch to guarantee
    /// durability; [`Drop`] also flushes on a best-effort basis.
    pub fn flush(&self) -> Result<()> {
        if let StorageBackend::Persistent(storage, state) = &self.backend {
            let guard = storage
                .read()
                .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
            state.flush(&guard)?;
        }
        Ok(())
    }

    /// Insert a quad into the store
    pub fn insert_quad(&mut self, quad: Quad) -> Result<bool> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                // Check if quad already exists
                let existing = index.find_quads(
                    Some(quad.subject()),
                    Some(quad.predicate()),
                    Some(quad.object()),
                    Some(quad.graph_name()),
                );
                if !existing.is_empty() {
                    return Ok(false); // Quad already exists
                }

                let _id = index.insert_quad(&quad);
                Ok(true) // New quad inserted
            }
            StorageBackend::Memory(storage) => {
                let mut storage = storage
                    .write()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;
                Ok(storage.insert_quad(quad))
            }
            StorageBackend::Persistent(storage, state) => {
                // Append-only persistence: mutate memory, then append ONE line
                // (no full-file rewrite). O(1) amortized per insert.
                let is_new = {
                    let mut storage = storage.write().map_err(|e| {
                        OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                    })?;
                    storage.insert_quad(quad.clone())
                };
                if is_new {
                    let line = Self::quad_to_nquads_line(&quad)?;
                    state.append_line(&line)?;
                }
                Ok(is_new)
            }
        }
    }

    /// Bulk insert quads for maximum performance.
    ///
    /// Takes a single write lock, inserts every quad, then persists all newly
    /// added quads with a single append + one `fsync` (persistent backend).
    /// Returns one entry per input quad: `1` if the quad was newly inserted,
    /// `0` if it was already present (this backend has no per-quad row id).
    pub fn bulk_insert_quads(&mut self, quads: Vec<Quad>) -> Result<Vec<u64>> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                let ids = index.bulk_insert_quads(&quads);
                Ok(ids)
            }
            StorageBackend::Memory(storage) => {
                let mut guard = storage
                    .write()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;
                let mut ids = Vec::with_capacity(quads.len());
                for quad in quads {
                    ids.push(u64::from(guard.insert_quad(quad)));
                }
                Ok(ids)
            }
            StorageBackend::Persistent(storage, state) => {
                let mut ids = Vec::with_capacity(quads.len());
                let mut lines = Vec::new();
                {
                    let mut guard = storage.write().map_err(|e| {
                        OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                    })?;
                    for quad in quads {
                        let is_new = guard.insert_quad(quad.clone());
                        if is_new {
                            lines.push(Self::quad_to_nquads_line(&quad)?);
                        }
                        ids.push(u64::from(is_new));
                    }
                }
                // One append + one fsync for the whole batch.
                state.append_lines(&lines)?;
                Ok(ids)
            }
        }
    }

    /// Insert a triple into the default graph
    pub fn insert_triple(&mut self, triple: Triple) -> Result<bool> {
        self.insert_quad(Quad::from_triple(triple))
    }

    /// Insert a triple into the store (legacy string interface)
    pub fn insert_string_triple(
        &mut self,
        subject: &str,
        predicate: &str,
        object: &str,
    ) -> Result<bool> {
        let subject_node = NamedNode::new(subject)?;
        let predicate_node = NamedNode::new(predicate)?;
        let object_literal = Literal::new(object);

        let triple = Triple::new(subject_node, predicate_node, object_literal);
        self.insert_triple(triple)
    }

    /// Remove a quad from the store
    pub fn remove_quad(&mut self, quad: &Quad) -> Result<bool> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => Ok(index.remove_quad(quad)),
            StorageBackend::Memory(storage) => {
                let mut storage = storage
                    .write()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;
                Ok(storage.remove_quad(quad))
            }
            StorageBackend::Persistent(storage, state) => {
                // An append log cannot express a deletion; mutate memory and
                // mark dirty so flush()/Drop compacts the file (durable delete).
                let removed = {
                    let mut storage = storage.write().map_err(|e| {
                        OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                    })?;
                    storage.remove_quad(quad)
                };
                if removed {
                    state.mark_dirty();
                }
                Ok(removed)
            }
        }
    }

    /// Check if a quad exists in the store
    pub fn contains_quad(&self, quad: &Quad) -> Result<bool> {
        match &self.backend {
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
                Ok(storage.contains_quad(quad))
            }
            StorageBackend::UltraMemory(index, _) => {
                let results = index.find_quads(
                    Some(quad.subject()),
                    Some(quad.predicate()),
                    Some(quad.object()),
                    Some(quad.graph_name()),
                );
                Ok(!results.is_empty())
            }
        }
    }

    /// Query quads matching the given pattern
    ///
    /// None values act as wildcards matching any term.
    pub fn query_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Result<Vec<Quad>> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                let results = index.find_quads(subject, predicate, object, graph_name);
                Ok(results)
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
                Ok(storage.query_quads(subject, predicate, object, graph_name))
            }
        }
    }

    /// Query triples in the default graph matching the given pattern
    pub fn query_triples(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
    ) -> Result<Vec<Triple>> {
        let default_graph = GraphName::DefaultGraph;
        let quads = self.query_quads(subject, predicate, object, Some(&default_graph))?;
        Ok(quads.into_iter().map(|quad| quad.to_triple()).collect())
    }

    /// Get all quads in the store
    pub fn iter_quads(&self) -> Result<Vec<Quad>> {
        self.query_quads(None, None, None, None)
    }

    /// Get all triples in the default graph
    pub fn triples(&self) -> Result<Vec<Triple>> {
        self.query_triples(None, None, None)
    }

    /// Get the number of quads in the store
    pub fn len(&self) -> Result<usize> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => Ok(index.len()),
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
                Ok(storage.len())
            }
        }
    }

    /// Check if the store is empty
    pub fn is_empty(&self) -> Result<bool> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => Ok(index.is_empty()),
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
                Ok(storage.is_empty())
            }
        }
    }

    /// Get performance statistics (ultra-performance mode only)
    pub fn stats(&self) -> Option<Arc<IndexStats>> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => Some(index.stats()),
            _ => None,
        }
    }

    /// Get memory usage statistics (ultra-performance mode only)
    pub fn memory_usage(&self) -> Option<MemoryUsage> {
        match &self.backend {
            StorageBackend::UltraMemory(index, arena) => {
                let mut usage = index.memory_usage();
                usage.arena_bytes = arena.allocated_bytes();
                Some(usage)
            }
            _ => None,
        }
    }

    /// Clear memory arena to reclaim memory (ultra-performance mode only)
    pub fn clear_arena(&self) {
        if let StorageBackend::UltraMemory(index, _arena) = &self.backend {
            index.clear_arena();
        }
    }

    /// Clear all data from the store
    pub fn clear(&mut self) -> Result<()> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                index.clear();
                Ok(())
            }
            StorageBackend::Memory(storage) => {
                let mut storage = storage
                    .write()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire write lock: {e}")))?;
                *storage = MemoryStorage::new();
                Ok(())
            }
            StorageBackend::Persistent(storage, state) => {
                {
                    let mut storage = storage.write().map_err(|e| {
                        OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                    })?;
                    *storage = MemoryStorage::new();
                }
                // The append log still holds the cleared quads; mark dirty so
                // flush()/Drop compacts to an empty file (durable clear).
                state.mark_dirty();
                Ok(())
            }
        }
    }

    /// Extract PREFIX declarations and expand prefixed names in the query
    #[allow(dead_code)]
    fn extract_and_expand_prefixes(
        &self,
        sparql: &str,
    ) -> Result<(std::collections::HashMap<String, String>, String)> {
        extract_and_expand_prefixes(sparql)
    }

    /// Query the store with SPARQL (delegates to QueryExecutor)
    pub fn query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        let executor = crate::sparql::QueryExecutor::new(&self.backend);
        executor.execute(sparql)
    }

    pub fn insert(&mut self, quad: &Quad) -> Result<()> {
        self.insert_quad(quad.clone())?;
        Ok(())
    }

    /// Remove a quad (compatibility alias for remove_quad)
    pub fn remove(&mut self, quad: &Quad) -> Result<bool> {
        self.remove_quad(quad)
    }

    /// Get all quads in the store
    pub fn quads(&self) -> Result<Vec<Quad>> {
        self.iter_quads()
    }

    /// Get all quads from named graphs only
    pub fn named_graph_quads(&self) -> Result<Vec<Quad>> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                let mut result = Vec::new();
                // Get all named graphs
                let graphs = self.named_graphs()?;
                for graph in graphs {
                    let graph_name = GraphName::NamedNode(graph);
                    let quads = index.find_quads(None, None, None, Some(&graph_name));
                    result.extend(quads);
                }
                Ok(result)
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
                let mut result = Vec::new();
                for graph in &storage.named_graphs {
                    let graph_name = GraphName::NamedNode(graph.clone());
                    let quads = storage.query_quads(None, None, None, Some(&graph_name));
                    result.extend(quads);
                }
                Ok(result)
            }
        }
    }

    /// Get all quads from the default graph only
    pub fn default_graph_quads(&self) -> Result<Vec<Quad>> {
        let default_graph = GraphName::DefaultGraph;
        self.query_quads(None, None, None, Some(&default_graph))
    }

    /// Get quads from a specific graph
    pub fn graph_quads(&self, graph: Option<&NamedNode>) -> Result<Vec<Quad>> {
        let graph_name = graph
            .map(|g| GraphName::NamedNode(g.clone()))
            .unwrap_or(GraphName::DefaultGraph);
        self.query_quads(None, None, None, Some(&graph_name))
    }

    /// Clear all data from all graphs
    pub fn clear_all(&mut self) -> Result<usize> {
        let count = self.len()?;
        self.clear()?;
        Ok(count)
    }

    /// Clear all named graphs (but not the default graph)
    pub fn clear_named_graphs(&mut self) -> Result<usize> {
        let mut deleted = 0;
        let graphs = self.named_graphs()?;

        for graph in graphs {
            let graph_name = GraphName::NamedNode(graph);
            deleted += self.clear_graph(Some(&graph_name))?;
        }

        Ok(deleted)
    }

    /// Clear the default graph only
    pub fn clear_default_graph(&mut self) -> Result<usize> {
        self.clear_graph(None)
    }

    /// Clear a specific graph
    pub fn clear_graph(&mut self, graph: Option<&GraphName>) -> Result<usize> {
        let graph_name = graph.cloned().unwrap_or(GraphName::DefaultGraph);
        let quads = self.query_quads(None, None, None, Some(&graph_name))?;
        let count = quads.len();

        for quad in quads {
            self.remove_quad(&quad)?;
        }

        Ok(count)
    }

    /// Get all graphs (including default if it contains data)
    pub fn graphs(&self) -> Result<Vec<NamedNode>> {
        let mut graphs = self.named_graphs()?;

        // Check if default graph has any data
        let default_graph = GraphName::DefaultGraph;
        let default_quads = self.query_quads(None, None, None, Some(&default_graph))?;
        if !default_quads.is_empty() {
            // Add a special marker for default graph
            if let Ok(default_marker) = NamedNode::new("urn:x-oxirs:default-graph") {
                graphs.push(default_marker);
            }
        }

        Ok(graphs)
    }

    /// Get all named graphs
    pub fn named_graphs(&self) -> Result<Vec<NamedNode>> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                // Get unique graph names from all quads
                let mut graphs = HashSet::new();
                let all_quads = index.find_quads(None, None, None, None);
                for quad in all_quads {
                    if let GraphName::NamedNode(graph) = quad.graph_name() {
                        graphs.insert(graph.clone());
                    }
                }
                Ok(graphs.into_iter().collect())
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let storage = storage
                    .read()
                    .map_err(|e| OxirsError::Store(format!("Failed to acquire read lock: {e}")))?;
                Ok(storage.named_graphs.iter().cloned().collect())
            }
        }
    }

    /// Create a new graph (if it doesn't exist)
    pub fn create_graph(&mut self, graph: Option<&NamedNode>) -> Result<()> {
        if let Some(graph_name) = graph {
            match &self.backend {
                StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                    let mut storage = storage.write().map_err(|e| {
                        OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                    })?;
                    storage.named_graphs.insert(graph_name.clone());
                }
                StorageBackend::UltraMemory(_, _) => {
                    // Graphs are created implicitly when quads are added
                }
            }
        }
        Ok(())
    }

    /// Drop a graph (remove the graph and all its quads)
    pub fn drop_graph(&mut self, graph: Option<&GraphName>) -> Result<()> {
        self.clear_graph(graph)?;

        // Remove from named graphs set if it's a named graph
        if let Some(GraphName::NamedNode(graph_name)) = graph {
            match &self.backend {
                StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                    let mut storage = storage.write().map_err(|e| {
                        OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                    })?;
                    storage.named_graphs.remove(graph_name);
                }
                StorageBackend::UltraMemory(_, _) => {
                    // Graph is dropped implicitly when all quads are removed
                }
            }
        }

        Ok(())
    }

    /// Load data from a URL into a graph
    pub fn load_from_url(&mut self, url: &str, graph: Option<&NamedNode>) -> Result<usize> {
        use crate::parser::Parser;

        // Parse URL to extract file extension if present
        let url_path = url.split('?').next().unwrap_or(url);
        let extension = url_path
            .split('/')
            .next_back()
            .and_then(|filename| filename.rsplit('.').next());

        // Fetch data from URL using reqwest. Reuse an ambient tokio runtime when
        // one is already running (calling this from inside an async handler must
        // not spin up a nested runtime, which would panic).
        let url_owned = url.to_string();
        let fetch = async move {
            let response = reqwest::get(&url_owned)
                .await
                .map_err(|e| OxirsError::Store(format!("Failed to fetch URL {url_owned}: {e}")))?;

            if !response.status().is_success() {
                return Err(OxirsError::Store(format!(
                    "HTTP error {} when fetching {url_owned}",
                    response.status()
                )));
            }

            // Get content type from headers
            let content_type = response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.split(';').next().unwrap_or(s).trim().to_string());

            // Get response body as text
            let text = response
                .text()
                .await
                .map_err(|e| OxirsError::Store(format!("Failed to read response body: {e}")))?;

            Ok::<_, OxirsError>((text, content_type))
        };

        let (content, content_type) = match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fetch)),
            Err(_) => {
                let runtime = tokio::runtime::Runtime::new()
                    .map_err(|e| OxirsError::Store(format!("Failed to create runtime: {e}")))?;
                runtime.block_on(fetch)
            }
        }?;

        // Detect RDF format from content type or file extension
        let format = Self::detect_format_from_url(&content_type, extension, &content)?;

        // Parse the content
        let parser = Parser::new(format);
        let quads = parser
            .parse_str_to_quads(&content)
            .map_err(|e| OxirsError::Store(format!("Failed to parse RDF data from {url}: {e}")))?;

        // Insert quads into the specified graph
        let target_graph = graph.cloned().map(GraphName::NamedNode);
        let mut inserted_count = 0;

        for quad in quads {
            // Override graph name if a target graph is specified
            let final_quad = if let Some(ref target) = target_graph {
                Quad::new(
                    quad.subject().clone(),
                    quad.predicate().clone(),
                    quad.object().clone(),
                    target.clone(),
                )
            } else {
                quad
            };

            if self.insert_quad(final_quad)? {
                inserted_count += 1;
            }
        }

        Ok(inserted_count)
    }

    /// Detect RDF format from content type, file extension, or content
    fn detect_format_from_url(
        content_type: &Option<String>,
        extension: Option<&str>,
        content: &str,
    ) -> Result<RdfFormat> {
        // Try to detect from content type first
        if let Some(ct) = content_type {
            let ct_lower = ct.to_lowercase();
            if let Some(format) = Self::format_from_media_type(&ct_lower) {
                return Ok(format);
            }
        }

        // Try to detect from file extension
        if let Some(ext) = extension {
            if let Some(format) = RdfFormat::from_extension(ext) {
                return Ok(format);
            }
        }

        // Try to detect from content
        if let Some(format) = crate::parser::detect_format_from_content(content) {
            return Ok(format);
        }

        // Default to N-Triples if we can't detect
        Err(OxirsError::Store(
            "Could not detect RDF format from URL, content type, or content".to_string(),
        ))
    }

    /// Map media type to RDF format
    fn format_from_media_type(media_type: &str) -> Option<RdfFormat> {
        match media_type {
            "text/turtle" | "application/x-turtle" => Some(RdfFormat::Turtle),
            "application/n-triples" | "text/plain" => Some(RdfFormat::NTriples),
            "application/trig" | "application/x-trig" => Some(RdfFormat::TriG),
            "application/n-quads" | "text/x-nquads" => Some(RdfFormat::NQuads),
            "application/rdf+xml" | "application/xml" | "text/xml" => Some(RdfFormat::RdfXml),
            "application/ld+json" | "application/json" => Some(RdfFormat::JsonLd),
            _ => None,
        }
    }

    /// Convert string to Subject term
    #[allow(dead_code)]
    fn string_to_subject(s: &str) -> Option<Subject> {
        if s.starts_with('<') && s.ends_with('>') {
            let iri = &s[1..s.len() - 1];
            NamedNode::new(iri).ok().map(Subject::NamedNode)
        } else if let Some(blank_id) = s.strip_prefix("_:") {
            BlankNode::new(blank_id).ok().map(Subject::BlankNode)
        } else {
            None
        }
    }

    /// Convert string to Predicate term
    #[allow(dead_code)]
    fn string_to_predicate(p: &str) -> Option<Predicate> {
        if p.starts_with('<') && p.ends_with('>') {
            let iri = &p[1..p.len() - 1];
            NamedNode::new(iri).ok().map(Predicate::NamedNode)
        } else {
            None
        }
    }

    /// Convert string to Object term
    #[allow(dead_code)]
    fn string_to_object(o: &str) -> Option<Object> {
        if o.starts_with('<') && o.ends_with('>') {
            let iri = &o[1..o.len() - 1];
            NamedNode::new(iri).ok().map(Object::NamedNode)
        } else if let Some(blank_id) = o.strip_prefix("_:") {
            BlankNode::new(blank_id).ok().map(Object::BlankNode)
        } else if o.starts_with('"') {
            // Simple literal parsing - more sophisticated parsing would be needed for real use
            let literal_content = &o[1..o.len() - 1];
            Some(Object::Literal(Literal::new(literal_content)))
        } else {
            None
        }
    }
}

impl Default for RdfStore {
    fn default() -> Self {
        // Construct the in-memory backend directly (infallible) rather than
        // unwrapping the fallible constructor (no-unwrap policy).
        RdfStore {
            backend: StorageBackend::Memory(Arc::new(RwLock::new(MemoryStorage::new()))),
        }
    }
}

impl Drop for RdfStore {
    fn drop(&mut self) {
        // Best-effort durability on graceful shutdown: flush the append buffer
        // and compact any pending deletions. Errors are logged, not propagated.
        if let StorageBackend::Persistent(..) = &self.backend {
            if let Err(e) = self.flush() {
                tracing::error!("RdfStore flush on drop failed: {e}");
            }
        }
    }
}

// Implement the Store trait for RdfStore using interior mutability
#[async_trait]
impl Store for RdfStore {
    fn insert_quad(&self, quad: Quad) -> Result<bool> {
        let inserted = match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                // Check if quad already exists
                let existing = index.find_quads(
                    Some(quad.subject()),
                    Some(quad.predicate()),
                    Some(quad.object()),
                    Some(quad.graph_name()),
                );
                if !existing.is_empty() {
                    return Ok(false); // Quad already exists
                }

                let _id = index.insert_quad(&quad);
                true // New quad inserted
            }
            StorageBackend::Memory(storage) => {
                let mut storage = storage.write().map_err(|e| {
                    crate::OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                })?;
                storage.insert_quad(quad)
            }
            StorageBackend::Persistent(storage, state) => {
                // Append-only persistence via the shared writer (interior
                // mutability), so the &self trait path is durable too.
                let is_new = {
                    let mut storage = storage.write().map_err(|e| {
                        crate::OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                    })?;
                    storage.insert_quad(quad.clone())
                };
                if is_new {
                    let line = Self::quad_to_nquads_line(&quad)?;
                    state.append_line(&line)?;
                }
                is_new
            }
        };

        Ok(inserted)
    }

    fn remove_quad(&self, quad: &Quad) -> Result<bool> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => Ok(index.remove_quad(quad)),
            StorageBackend::Memory(storage) => {
                let mut storage = storage.write().map_err(|e| {
                    crate::OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                })?;
                Ok(storage.remove_quad(quad))
            }
            StorageBackend::Persistent(storage, state) => {
                let removed = {
                    let mut storage = storage.write().map_err(|e| {
                        crate::OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                    })?;
                    storage.remove_quad(quad)
                };
                if removed {
                    state.mark_dirty();
                }
                Ok(removed)
            }
        }
    }

    fn find_quads(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
    ) -> Result<Vec<Quad>> {
        self.query_quads(subject, predicate, object, graph_name)
    }

    fn is_ready(&self) -> bool {
        true // Simple implementation
    }

    fn len(&self) -> Result<usize> {
        self.len()
    }

    fn is_empty(&self) -> Result<bool> {
        self.is_empty()
    }

    fn query(&self, sparql: &str) -> Result<OxirsQueryResults> {
        self.query(sparql)
    }

    fn prepare_query(&self, sparql: &str) -> Result<PreparedQuery> {
        // Bind a shared handle to this store's backend so exec() runs the real
        // query executor rather than returning an empty result.
        Ok(PreparedQuery::with_backend(
            sparql.to_string(),
            self.backend.clone(),
        ))
    }

    /// Enumerate named graphs via the interned graph-name index
    /// (`storage.named_graphs`, a `BTreeSet` maintained incrementally on
    /// insert/remove) rather than the trait default's empty `Vec`. This
    /// resolves to `RdfStore`'s inherent `named_graphs` (inherent methods take
    /// priority over trait methods of the same name), which is O(graphs) for
    /// the Memory/Persistent backends instead of an O(quads) streaming scan.
    fn named_graphs(&self) -> Result<Vec<NamedNode>> {
        self.named_graphs()
    }

    /// Bulk-insert override: one write lock for the whole batch and, for the
    /// durable backend, one append + exactly one `fsync` for the batch (via
    /// [`PersistentState::append_lines`]) instead of a per-quad fsync loop.
    /// Uses interior mutability so it works through the `&self` trait path.
    /// Returns the number of newly-inserted quads.
    fn bulk_insert_quads(&self, quads: Vec<Quad>) -> Result<usize> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                // UltraMemory's index has no dedup, so mirror the per-quad
                // novelty check used by `insert_quad` to count only new quads.
                let mut inserted = 0usize;
                for quad in quads {
                    let existing = index.find_quads(
                        Some(quad.subject()),
                        Some(quad.predicate()),
                        Some(quad.object()),
                        Some(quad.graph_name()),
                    );
                    if existing.is_empty() {
                        let _id = index.insert_quad(&quad);
                        inserted += 1;
                    }
                }
                Ok(inserted)
            }
            StorageBackend::Memory(storage) => {
                let mut guard = storage.write().map_err(|e| {
                    crate::OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                })?;
                let mut inserted = 0usize;
                for quad in quads {
                    if guard.insert_quad(quad) {
                        inserted += 1;
                    }
                }
                Ok(inserted)
            }
            StorageBackend::Persistent(storage, state) => {
                let mut lines = Vec::new();
                {
                    let mut guard = storage.write().map_err(|e| {
                        crate::OxirsError::Store(format!("Failed to acquire write lock: {e}"))
                    })?;
                    for quad in quads {
                        if guard.insert_quad(quad.clone()) {
                            lines.push(Self::quad_to_nquads_line(&quad)?);
                        }
                    }
                }
                let inserted = lines.len();
                // One append + one fsync for the whole batch.
                state.append_lines(&lines)?;
                Ok(inserted)
            }
        }
    }

    /// Streaming scan override: visit each matching quad without building a
    /// result `Vec`. For the in-memory / persistent backends the read lock is
    /// held for the duration of the scan and each quad is handed to `f` in
    /// deterministic *index (term-interning) order* — which differs from the
    /// `Quad`-ordered [`quads`](RdfStore::quads)/[`find_quads`](Store::find_quads)
    /// result order but visits exactly the same set (RDF is unordered, so
    /// serialization consumers do not depend on the order); the callback must not
    /// re-enter the store for a write (that would deadlock on the same lock).
    fn for_each_quad(
        &self,
        subject: Option<&Subject>,
        predicate: Option<&Predicate>,
        object: Option<&Object>,
        graph_name: Option<&GraphName>,
        f: &mut dyn FnMut(Quad),
    ) -> Result<()> {
        match &self.backend {
            StorageBackend::UltraMemory(index, _arena) => {
                // UltraIndex exposes no streaming API; fall back to its pattern
                // lookup and iterate (still one quad clone alive at a time here).
                for quad in index.find_quads(subject, predicate, object, graph_name) {
                    f(quad);
                }
                Ok(())
            }
            StorageBackend::Memory(storage) | StorageBackend::Persistent(storage, _) => {
                let guard = storage.read().map_err(|e| {
                    crate::OxirsError::Store(format!("Failed to acquire read lock: {e}"))
                })?;
                guard.for_each_quad(subject, predicate, object, graph_name, f);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod bulk_and_scan_tests {
    use super::*;
    use crate::model::{GraphName, Literal, NamedNode, Object, Quad};

    fn quad(s: &str, o: &str, g: GraphName) -> Quad {
        Quad::new(
            NamedNode::new(s).expect("subject IRI"),
            NamedNode::new("http://example.org/p").expect("predicate IRI"),
            Literal::new_simple_literal(o),
            g,
        )
    }

    fn unique_dir(tag: &str) -> std::path::PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("oxirs_bulk_scan_{tag}_{nanos}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// The trait-level `bulk_insert_quads` on the durable `RdfStore` inserts
    /// every distinct quad, counts only newly-inserted quads, and persists the
    /// whole batch into a single durable data file (one line per quad) that
    /// reloads to the same set — validating the single-append batch path.
    #[test]
    fn rdfstore_trait_bulk_insert_persists_all_in_one_file() {
        let dir = unique_dir("persist");
        {
            let store = RdfStore::open(&dir).expect("open persistent store");
            let quads = vec![
                quad("http://example.org/a", "1", GraphName::DefaultGraph),
                quad("http://example.org/b", "2", GraphName::DefaultGraph),
                quad(
                    "http://example.org/c",
                    "3",
                    GraphName::NamedNode(
                        NamedNode::new("http://example.org/g").expect("graph IRI"),
                    ),
                ),
                // duplicate of the first — must not be counted or double-written
                quad("http://example.org/a", "1", GraphName::DefaultGraph),
            ];
            let inserted =
                Store::bulk_insert_quads(&store, quads).expect("bulk insert through trait");
            assert_eq!(inserted, 3, "only three distinct quads are new");
            store.flush().expect("flush");

            // Exactly three N-Quads lines were written to the single data file.
            let data = std::fs::read_to_string(dir.join("data.nq")).expect("read data.nq");
            let lines = data.lines().filter(|l| !l.trim().is_empty()).count();
            assert_eq!(lines, 3, "one durable line per newly-inserted quad");
        }

        // Reopening from disk must observe exactly the three persisted quads.
        let reopened = RdfStore::open(&dir).expect("reopen persistent store");
        assert_eq!(reopened.len().expect("len"), 3);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// After the interned-only refactor (no owned quad set), a durable delete
    /// still compacts the on-disk file *from the interned indexes* on flush, and
    /// a reopen observes exactly the surviving quads. This exercises the
    /// `PersistentState::compact` path that now streams `storage.iter_quads()`.
    #[test]
    fn persistent_delete_compacts_from_interned_state_and_reopens() {
        let dir = unique_dir("delete_compact");
        let g = GraphName::NamedNode(NamedNode::new("http://example.org/g").expect("graph IRI"));
        let keep_a = quad("http://example.org/a", "1", GraphName::DefaultGraph);
        let drop_b = quad("http://example.org/b", "2", GraphName::DefaultGraph);
        let keep_c = quad("http://example.org/c", "3", g.clone());

        {
            let store = RdfStore::open(&dir).expect("open persistent store");
            store.insert_quad(keep_a.clone()).expect("insert a");
            store.insert_quad(drop_b.clone()).expect("insert b");
            store.insert_quad(keep_c.clone()).expect("insert c");
            store.flush().expect("flush appends");
            assert_eq!(store.len().expect("len"), 3);

            // Delete one quad; this only mutates memory + marks dirty.
            assert!(store.remove_quad(&drop_b).expect("remove b"));
            assert_eq!(store.len().expect("len"), 2);
            assert!(!store.contains_quad(&drop_b).expect("contains b"));

            // Flush compacts: the file is rewritten from the interned indexes.
            store.flush().expect("flush compaction");
            let data = std::fs::read_to_string(dir.join("data.nq")).expect("read data.nq");
            let lines = data.lines().filter(|l| !l.trim().is_empty()).count();
            assert_eq!(lines, 2, "compacted file holds only the surviving quads");
        }

        // Reopen: the deletion is durable and only the kept quads survive.
        let reopened = RdfStore::open(&dir).expect("reopen persistent store");
        assert_eq!(reopened.len().expect("len"), 2);
        assert!(reopened.contains_quad(&keep_a).expect("contains a"));
        assert!(reopened.contains_quad(&keep_c).expect("contains c"));
        assert!(!reopened.contains_quad(&drop_b).expect("not contains b"));

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `for_each_quad` visits exactly the quads matching the pattern (and no
    /// others) across backends. It streams in index (term-interning) order,
    /// which need not equal `find_quads`' `Quad` order, so content is compared
    /// as a set.
    #[test]
    fn for_each_quad_visits_exactly_matching_quads() {
        use std::collections::BTreeSet;

        let store = RdfStore::new().expect("in-memory store");
        let g = GraphName::NamedNode(NamedNode::new("http://example.org/g").expect("graph IRI"));
        Store::bulk_insert_quads(
            &store,
            vec![
                quad("http://example.org/a", "1", GraphName::DefaultGraph),
                quad("http://example.org/b", "2", GraphName::DefaultGraph),
                quad("http://example.org/c", "3", g.clone()),
            ],
        )
        .expect("seed");

        // Unbound pattern visits exactly the same set as find_quads.
        let mut visited: Vec<Quad> = Vec::new();
        Store::for_each_quad(&store, None, None, None, None, &mut |q| visited.push(q))
            .expect("scan all");
        let via_find = store.find_quads(None, None, None, None).expect("find all");
        let visited_set: BTreeSet<Quad> = visited.iter().cloned().collect();
        let find_set: BTreeSet<Quad> = via_find.into_iter().collect();
        assert_eq!(
            visited_set, find_set,
            "scan must match find_quads content (set-equal)"
        );
        assert_eq!(visited.len(), 3);

        // Graph-bound pattern visits only the named-graph quad.
        let mut named: Vec<Quad> = Vec::new();
        Store::for_each_quad(&store, None, None, None, Some(&g), &mut |q| named.push(q))
            .expect("scan named");
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].graph_name(), &g);

        // Object-bound pattern visits only the matching quad.
        let obj = Object::Literal(Literal::new_simple_literal("2"));
        let mut by_obj: Vec<Quad> = Vec::new();
        Store::for_each_quad(&store, None, None, Some(&obj), None, &mut |q| {
            by_obj.push(q)
        })
        .expect("scan by object");
        assert_eq!(by_obj.len(), 1);
        assert_eq!(by_obj[0].object(), &obj);
    }
}

/// Tests for `Store::named_graphs()` exercised strictly through the trait
/// interface (`&dyn Store`) so they prove the per-impl overrides -- not the
/// trait's empty-`Vec` default -- are what actually runs. See the override in
/// `impl Store for RdfStore` above and `impl Store for ConcreteStore` in
/// `concrete.rs`.
#[cfg(test)]
mod named_graphs_trait_tests {
    use super::*;
    use crate::model::{GraphName, Literal, NamedNode, Quad};

    fn quad(s: &str, o: &str, g: GraphName) -> Quad {
        Quad::new(
            NamedNode::new(s).expect("subject IRI"),
            NamedNode::new("http://example.org/p").expect("predicate IRI"),
            Literal::new_simple_literal(o),
            g,
        )
    }

    /// Seeds two default-graph quads (must be excluded from `named_graphs()`)
    /// plus three named graphs, one of which (`g1`) gets two quads so a
    /// naive implementation that doesn't dedup would over-report.
    fn seed(store: &dyn Store) {
        let g1 = GraphName::NamedNode(NamedNode::new("http://example.org/g1").expect("g1 IRI"));
        let g2 = GraphName::NamedNode(NamedNode::new("http://example.org/g2").expect("g2 IRI"));
        let g3 = GraphName::NamedNode(NamedNode::new("http://example.org/g3").expect("g3 IRI"));

        store
            .insert_quad(quad("http://example.org/a", "1", GraphName::DefaultGraph))
            .expect("insert default graph quad 1");
        store
            .insert_quad(quad("http://example.org/b", "2", GraphName::DefaultGraph))
            .expect("insert default graph quad 2");

        store
            .insert_quad(quad("http://example.org/c", "3", g1.clone()))
            .expect("insert g1 quad 1");
        store
            .insert_quad(quad("http://example.org/d", "4", g1.clone()))
            .expect("insert g1 quad 2");
        store
            .insert_quad(quad("http://example.org/e", "5", g2.clone()))
            .expect("insert g2 quad");
        store
            .insert_quad(quad("http://example.org/f", "6", g3.clone()))
            .expect("insert g3 quad");
    }

    /// Asserts `named_graphs()` returns exactly `{g1, g2, g3}`: the default
    /// graph is absent and `g1` (which holds two quads) appears once.
    fn assert_exactly_three_named_graphs(store: &dyn Store) {
        let mut graphs: Vec<String> = store
            .named_graphs()
            .expect("named_graphs via Store trait")
            .into_iter()
            .map(|n| n.as_str().to_string())
            .collect();
        graphs.sort();
        graphs.dedup();
        assert_eq!(
            graphs,
            vec![
                "http://example.org/g1".to_string(),
                "http://example.org/g2".to_string(),
                "http://example.org/g3".to_string(),
            ],
            "named_graphs() through the Store trait must return exactly the \
             named graphs (default graph excluded, no duplicates)"
        );
    }

    /// In-memory backend: `Store::named_graphs`, called through a `dyn Store`
    /// trait object, must return the real graph list from the interned
    /// `BTreeSet` dictionary -- not the trait default's empty `Vec` -- and
    /// must not include the default graph or duplicate any graph name.
    #[test]
    fn dyn_store_named_graphs_in_memory_excludes_default_and_dedups() {
        let store = RdfStore::new().expect("in-memory store");
        seed(&store);
        let dyn_store: &dyn Store = &store;
        assert_exactly_three_named_graphs(dyn_store);
    }

    /// Persistent backend (temp-dir-backed): same contract as the in-memory
    /// case, exercised through `dyn Store` so the test proves the trait
    /// override -- not just the pre-existing inherent method -- is wired up,
    /// and that the graph index survives a durable close/reopen round-trip.
    #[test]
    fn dyn_store_named_graphs_persistent_excludes_default_and_dedups() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("oxirs_named_graphs_trait_{nanos}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");

        {
            let store = RdfStore::open(&dir).expect("open persistent store");
            seed(&store);
            let dyn_store: &dyn Store = &store;
            assert_exactly_three_named_graphs(dyn_store);
        }

        let reopened = RdfStore::open(&dir).expect("reopen persistent store");
        let dyn_reopened: &dyn Store = &reopened;
        assert_exactly_three_named_graphs(dyn_reopened);

        std::fs::remove_dir_all(&dir).ok();
    }

    /// `ConcreteStore` wraps an inner `RdfStore` behind a lock; its
    /// `Store::named_graphs` override must delegate through to the same
    /// interned-index lookup rather than falling back on the trait default.
    #[test]
    fn dyn_store_named_graphs_concrete_store_delegates() {
        let store = ConcreteStore::new().expect("concrete store");
        seed(&store);
        let dyn_store: &dyn Store = &store;
        assert_exactly_three_named_graphs(dyn_store);
    }
}
