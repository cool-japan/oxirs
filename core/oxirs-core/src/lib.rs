//! # OxiRS Core - RDF and SPARQL Foundation
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--rc.1-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-core/badge.svg)](https://docs.rs/oxirs-core)
//!
//! **Status**: Beta Release (v0.1.0-rc.1)
//! **Stability**: Public APIs are stabilizing. Production-ready for RDF/SPARQL core operations.
//!
//! ## Overview
//!
//! `oxirs-core` is a high-performance, zero-dependency Rust-native RDF data model and SPARQL foundation
//! for the OxiRS semantic web platform. It provides the fundamental types, traits, and operations that
//! all other OxiRS crates build upon.
//!
//! This crate is designed for:
//! - Building semantic web applications in Rust
//! - Processing large-scale knowledge graphs with minimal memory overhead
//! - Developing SPARQL query engines and triple stores
//! - Creating RDF-based data pipelines and transformations
//!
//! ## Key Features
//!
//! ### Core RDF Support
//! - **RDF 1.2 Compliance** - Full implementation of the RDF 1.2 data model
//! - **Multiple Serialization Formats** - Support for Turtle, N-Triples, TriG, N-Quads, RDF/XML, JSON-LD
//! - **Named Graphs** - Full quad support with named graph management
//! - **RDF-star** - Support for quoted triples (statement reification)
//!
//! ### Performance & Scalability
//! - **Memory Efficiency** - Arena-based allocators and optimized data structures
//! - **Concurrent Operations** - Lock-free data structures for high-throughput workloads
//! - **Parallel Processing** - SIMD and multi-threaded query execution
//! - **Zero-Copy Parsing** - Streaming parsers with minimal allocations
//!
//! ### SPARQL Foundation
//! - **SPARQL 1.1 Query** - SELECT, CONSTRUCT, ASK, DESCRIBE queries
//! - **SPARQL 1.1 Update** - INSERT, DELETE, LOAD, CLEAR operations
//! - **SPARQL 1.2 Features** - Enhanced property paths, aggregations, and functions
//! - **Federation** - SERVICE clause support for distributed queries
//!
//! ### Storage & Persistence
//! - **Pluggable Backends** - In-memory and disk-based storage options
//! - **Persistent Storage** - Automatic N-Quads serialization for durability
//! - **Indexed Access** - Multi-index support (SPO, POS, OSP) for fast pattern matching
//! - **Transaction Support** - ACID-compliant transactions for data integrity
//!
//! ### Production Features
//! - **Monitoring & Metrics** - Built-in performance instrumentation via SciRS2
//! - **Error Handling** - Rich error types with context and suggestions
//! - **Health Checks** - Circuit breakers and resource quotas
//! - **Benchmarking** - Comprehensive benchmark suite for performance validation
//!
//! ## Quick Start
//!
//! ### Creating a Store and Adding Data
//!
//! ```rust,no_run
//! use oxirs_core::RdfStore;
//! use oxirs_core::model::{NamedNode, Triple, Literal};
//!
//! # fn example() -> Result<(), oxirs_core::OxirsError> {
//! // Create an in-memory RDF store
//! let mut store = RdfStore::new()?;
//!
//! // Create RDF terms
//! let alice = NamedNode::new("http://example.org/alice")?;
//! let knows = NamedNode::new("http://xmlns.com/foaf/0.1/knows")?;
//! let bob = NamedNode::new("http://example.org/bob")?;
//! let name = NamedNode::new("http://xmlns.com/foaf/0.1/name")?;
//!
//! // Insert triples
//! store.insert_triple(Triple::new(alice.clone(), knows, bob))?;
//! store.insert_triple(Triple::new(alice, name, Literal::new("Alice")))?;
//!
//! println!("Store contains {} triples", store.len()?);
//! # Ok(())
//! # }
//! ```
//!
//! ### Querying with Pattern Matching
//!
//! ```rust,no_run
//! use oxirs_core::RdfStore;
//! use oxirs_core::model::{NamedNode, Subject, Predicate};
//!
//! # fn query_example() -> Result<(), oxirs_core::OxirsError> {
//! # let mut store = RdfStore::new()?;
//! // Query all triples with a specific predicate
//! let knows = NamedNode::new("http://xmlns.com/foaf/0.1/knows")?;
//! let predicate = Predicate::NamedNode(knows);
//!
//! let triples = store.query_triples(None, Some(&predicate), None)?;
//! for triple in triples {
//!     println!("{:?} knows {:?}", triple.subject(), triple.object());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Executing SPARQL Queries
//!
//! ```rust,no_run
//! use oxirs_core::RdfStore;
//!
//! # fn sparql_example() -> Result<(), oxirs_core::OxirsError> {
//! # let store = RdfStore::new()?;
//! // Execute a SPARQL SELECT query
//! let query = r#"
//!     PREFIX foaf: <http://xmlns.com/foaf/0.1/>
//!     SELECT ?name WHERE {
//!         ?person foaf:name ?name .
//!     }
//! "#;
//!
//! let results = store.query(query)?;
//! println!("Query returned {} results", results.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Loading Data from Files
//!
//! ```rust,no_run
//! use oxirs_core::RdfStore;
//! use oxirs_core::parser::{Parser, RdfFormat};
//! use std::fs;
//!
//! # fn load_example() -> Result<(), oxirs_core::OxirsError> {
//! # let mut store = RdfStore::new()?;
//! // Load RDF data from a Turtle file
//! let content = fs::read_to_string("data.ttl")
//!     .map_err(|e| oxirs_core::OxirsError::Io(e.to_string()))?;
//!
//! let parser = Parser::new(RdfFormat::Turtle);
//! let quads = parser.parse_str_to_quads(&content)?;
//!
//! for quad in quads {
//!     store.insert_quad(quad)?;
//! }
//!
//! println!("Loaded {} quads from file", store.len()?);
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
//! # fn persistent_example() -> Result<(), oxirs_core::OxirsError> {
//! // Create a persistent store (data saved to disk)
//! let mut store = RdfStore::open("./my_rdf_store")?;
//!
//! // Add data - automatically persisted
//! let subject = NamedNode::new("http://example.org/resource")?;
//! let predicate = NamedNode::new("http://purl.org/dc/terms/title")?;
//! let object = Literal::new("My Resource");
//!
//! store.insert_triple(Triple::new(subject, predicate, object))?;
//!
//! // Data is automatically saved to disk on modifications
//! println!("Data persisted to disk");
//! # Ok(())
//! # }
//! ```
//!
//! ## Advanced Features
//!
//! ### Named Graphs (Quads)
//!
//! ```rust,no_run
//! use oxirs_core::RdfStore;
//! use oxirs_core::model::{NamedNode, Quad, GraphName, Literal};
//!
//! # fn quads_example() -> Result<(), oxirs_core::OxirsError> {
//! # let mut store = RdfStore::new()?;
//! // Create a quad with a named graph
//! let graph = GraphName::NamedNode(NamedNode::new("http://example.org/graph1")?);
//! let subject = NamedNode::new("http://example.org/subject")?;
//! let predicate = NamedNode::new("http://example.org/predicate")?;
//! let object = Literal::new("value");
//!
//! let quad = Quad::new(subject, predicate, object, graph);
//! store.insert_quad(quad)?;
//!
//! // Query specific graph
//! let graph_node = NamedNode::new("http://example.org/graph1")?;
//! let quads = store.graph_quads(Some(&graph_node))?;
//! println!("Graph contains {} quads", quads.len());
//! # Ok(())
//! # }
//! ```
//!
//! ### Bulk Operations for Performance
//!
//! ```rust,ignore
//! use oxirs_core::RdfStore;
//! use oxirs_core::model::{NamedNode, Quad, Literal};
//!
//! # fn bulk_example() -> Result<(), oxirs_core::OxirsError> {
//! # let mut store = RdfStore::new()?;
//! // Prepare many quads for bulk insert
//! let mut quads = Vec::new();
//! for i in 0..1000 {
//!     let subject = NamedNode::new(&format!("http://example.org/item{}", i))?;
//!     let predicate = NamedNode::new("http://example.org/value")?;
//!     let object = Literal::new(&i.to_string());
//!     quads.push(Quad::from_triple(Triple::new(subject, predicate, object)));
//! }
//!
//! // Bulk insert for better performance
//! let ids = store.bulk_insert_quads(quads)?;
//! println!("Inserted {} quads with IDs: {:?}", ids.len(), &ids[..5]);
//! # Ok(())
//! # }
//! ```
//!
//! ## Module Organization
//!
//! - [`model`] - Core RDF data model types (IRI, Literal, Blank Node, Triple, Quad)
//! - [`rdf_store`] - RDF store implementations with pluggable storage backends
//! - [`parser`] - RDF parsers for multiple serialization formats
//! - [`serializer`] - RDF serializers for exporting data
//! - [`query`] - SPARQL query algebra and execution engine
//! - [`sparql`] - SPARQL query processing and federation
//! - [`storage`] - Storage backends (memory, disk, columnar)
//! - [`indexing`] - Multi-index structures for fast pattern matching
//! - [`concurrent`] - Lock-free data structures for concurrent access
//! - [`optimization`] - Query optimization and execution planning
//! - [`federation`] - SPARQL federation for distributed queries
//! - [`production`] - Production features (monitoring, health checks, circuit breakers)
//!
//! ## Feature Flags
//!
//! - `serde` - Enable serialization support (default)
//! - `parallel` - Enable parallel processing with rayon (default)
//! - `simd` - Enable SIMD optimizations for x86_64
//! - `async` - Enable async I/O support
//! - `rocksdb` - Enable RocksDB storage backend
//! - `rdf-star` - Enable RDF-star (quoted triples) support
//! - `sparql-12` - Enable SPARQL 1.2 features
//!
//! ## Performance Tips
//!
//! 1. **Use bulk operations** - `bulk_insert_quads()` is faster than individual inserts
//! 2. **Choose the right backend** - UltraMemory backend for maximum performance
//! 3. **Enable parallel feature** - Leverage multi-core CPUs for query execution
//! 4. **Use indexed queries** - Pattern matching benefits from multi-index structures
//! 5. **Profile with built-in metrics** - Use SciRS2 metrics for bottleneck identification
//!
//! ## API Stability
//!
//! As of Beta 1 (v0.1.0-rc.1):
//! - **Stable APIs**: Core RDF model types (`NamedNode`, `Literal`, `Triple`, `Quad`)
//! - **Stable APIs**: Store operations (`RdfStore`, `insert`, `query`, `remove`)
//! - **Stable APIs**: Parser and serializer interfaces
//! - **Unstable APIs**: Advanced optimization features (may change)
//! - **Experimental**: Consciousness, molecular, and quantum modules
//!
//! See [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) for migration from alpha versions.
//!
//! ## Architecture
//!
//! For detailed architecture documentation, see [ARCHITECTURE.md](../ARCHITECTURE.md).
//!
//! ## Examples
//!
//! More examples are available in the `examples/` directory:
//! - `basic_store.rs` - Basic store operations
//! - `sparql_query.rs` - SPARQL query examples
//! - `persistent_store.rs` - Persistent storage usage
//! - `bulk_loading.rs` - High-performance bulk data loading
//! - `federation.rs` - Federated SPARQL queries

pub mod ai;
pub mod concurrent;
pub mod consciousness; // Consciousness-inspired computing for intuitive query optimization
pub mod distributed;
pub mod federation; // SPARQL federation support for distributed query execution
pub mod format; // Phase 3: Complete RDF format support with zero dependencies
pub mod graph;
pub mod indexing;
pub mod interning;
pub mod io;
pub mod model;
pub mod molecular; // Molecular-level memory management inspired by biological systems
pub mod optimization;
pub mod parser;
pub mod production; // Production hardening features (Beta.1)
pub mod quantum; // Quantum-inspired computing for next-generation RDF processing
pub mod query;
pub mod rdf_store;
pub mod serializer;
pub mod sparql; // SPARQL query processing modules
pub mod storage;
pub mod store;
pub mod transaction;
pub mod vocab;
// pub mod config;
pub mod jsonld; // Re-enabled after fixing StringInterner method calls
pub mod oxigraph_compat;
pub mod rdfxml; // Oxigraph compatibility layer

// Core abstractions for OxiRS ecosystem
pub mod error;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod platform;
#[cfg(feature = "simd")]
pub mod simd;
pub mod simd_triple_matching; // SIMD-optimized triple pattern matching using SciRS2
pub mod zero_copy_rdf; // Zero-copy RDF operations using SciRS2-core memory management

// Re-export core types for convenience
pub use model::*;
pub use rdf_store::{ConcreteStore, RdfStore, Store};
pub use transaction::{
    AcidTransaction, IsolationLevel, TransactionId, TransactionManager, TransactionState,
};

/// Core error type for OxiRS operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum OxirsError {
    #[error("Store error: {0}")]
    Store(String),
    #[error("Query error: {0}")]
    Query(String),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Serialization error: {0}")]
    Serialize(String),
    #[error("IO error: {0}")]
    Io(String),
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),
    #[error("Quantum computing error: {0}")]
    QuantumError(String),
    #[error("Molecular optimization error: {0}")]
    MolecularError(String),
    #[error("Neural-symbolic fusion error: {0}")]
    NeuralSymbolicError(String),
    #[error("Operation not supported: {0}")]
    NotSupported(String),
    #[error("Update error: {0}")]
    Update(String),
    #[error("Federation error: {0}")]
    Federation(String),
}

impl From<std::io::Error> for OxirsError {
    fn from(err: std::io::Error) -> Self {
        OxirsError::Io(err.to_string())
    }
}

// Additional error conversions
impl From<bincode::error::EncodeError> for OxirsError {
    fn from(err: bincode::error::EncodeError) -> Self {
        OxirsError::Serialize(err.to_string())
    }
}

impl From<bincode::error::DecodeError> for OxirsError {
    fn from(err: bincode::error::DecodeError) -> Self {
        OxirsError::Serialize(err.to_string())
    }
}

impl From<serde_json::Error> for OxirsError {
    fn from(err: serde_json::Error) -> Self {
        OxirsError::Serialize(err.to_string())
    }
}

#[cfg(feature = "rocksdb")]
impl From<rocksdb::Error> for OxirsError {
    fn from(err: rocksdb::Error) -> Self {
        OxirsError::Store(err.to_string())
    }
}

// Note: DataFusion, Arrow, and Parquet error conversions removed
// Add these features to Cargo.toml if these integrations are needed

impl From<std::time::SystemTimeError> for OxirsError {
    fn from(err: std::time::SystemTimeError) -> Self {
        OxirsError::Io(format!("System time error: {err}"))
    }
}

/// Result type alias for OxiRS operations
pub type Result<T> = std::result::Result<T, OxirsError>;

/// Version information for OxiRS Core
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize OxiRS Core with default configuration
pub fn init() -> Result<()> {
    tracing::info!("Initializing OxiRS Core v{}", VERSION);
    Ok(())
}
