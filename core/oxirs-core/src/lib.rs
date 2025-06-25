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

pub mod distributed;
pub mod graph;
pub mod indexing;
pub mod interning;
pub mod model;
pub mod optimization;
pub mod parser;
pub mod query;
pub mod serializer;
pub mod storage;
pub mod store;
pub mod vocab;
// pub mod config;
// pub mod jsonld;
// pub mod rdfxml; // Disabled due to oxiri/oxilangtag dependencies - needs native implementation

// Re-export core types for convenience
pub use model::*;

/// Core error type for OxiRS operations
#[derive(Debug, thiserror::Error)]
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
    Io(#[from] std::io::Error),
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

impl From<rocksdb::Error> for OxirsError {
    fn from(err: rocksdb::Error) -> Self {
        OxirsError::Store(err.to_string())
    }
}

impl From<datafusion::error::DataFusionError> for OxirsError {
    fn from(err: datafusion::error::DataFusionError) -> Self {
        OxirsError::Query(err.to_string())
    }
}

impl From<arrow::error::ArrowError> for OxirsError {
    fn from(err: arrow::error::ArrowError) -> Self {
        OxirsError::Store(err.to_string())
    }
}

impl From<parquet::errors::ParquetError> for OxirsError {
    fn from(err: parquet::errors::ParquetError) -> Self {
        OxirsError::Store(err.to_string())
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
