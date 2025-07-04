//! # OxiRS Core
//!
//! Core RDF and SPARQL functionality for OxiRS - a thin, safe re-export of oxigraph
//! with additional abstractions for the OxiRS ecosystem.
//!
//! This crate provides the foundational RDF and SPARQL operations that all other
//! OxiRS crates depend on, including:
//!
//! - RDF 1.2 parsing and serialization
//! - Triple store operations
//! - SPARQL query execution
//! - Memory and disk storage backends
//!
//! ## Examples
//!
//! ```rust
//! use oxirs_core::store::Store;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let store = Store::new()?;
//! // Use the store for RDF operations
//! # Ok(())
//! # }
//! ```

pub mod ai;
pub mod concurrent;
pub mod consciousness; // Consciousness-inspired computing for intuitive query optimization
pub mod distributed;
pub mod format; // Phase 3: Complete RDF format support with zero dependencies
pub mod graph;
pub mod indexing;
pub mod interning;
pub mod io;
pub mod model;
pub mod molecular; // Molecular-level memory management inspired by biological systems
pub mod optimization;
pub mod parser;
pub mod quantum; // Quantum-inspired computing for next-generation RDF processing
pub mod query;
pub mod rdf_store;
pub mod serializer;
pub mod storage;
pub mod store;
pub mod vocab;
// pub mod config;
pub mod jsonld; // Re-enabled after fixing StringInterner method calls
                // pub mod rdfxml; // TODO: Fix lifetime issues in serializer
pub mod oxigraph_compat; // Oxigraph compatibility layer

// Core abstractions for OxiRS ecosystem
pub mod error;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod platform;
#[cfg(feature = "simd")]
pub mod simd;

// Re-export core types for convenience
pub use model::*;
pub use rdf_store::{ConcreteStore, RdfStore, Store};

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
}

impl From<std::io::Error> for OxirsError {
    fn from(err: std::io::Error) -> Self {
        OxirsError::Io(err.to_string())
    }
}

// Additional error conversions
impl From<bincode::Error> for OxirsError {
    fn from(err: bincode::Error) -> Self {
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

#[cfg(feature = "datafusion")]
impl From<datafusion::error::DataFusionError> for OxirsError {
    fn from(err: datafusion::error::DataFusionError) -> Self {
        OxirsError::Query(err.to_string())
    }
}

#[cfg(feature = "arrow")]
impl From<arrow::error::ArrowError> for OxirsError {
    fn from(err: arrow::error::ArrowError) -> Self {
        OxirsError::Store(err.to_string())
    }
}

#[cfg(feature = "parquet")]
impl From<parquet::errors::ParquetError> for OxirsError {
    fn from(err: parquet::errors::ParquetError) -> Self {
        OxirsError::Store(err.to_string())
    }
}

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
