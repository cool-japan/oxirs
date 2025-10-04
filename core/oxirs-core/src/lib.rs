//! # OxiRS Core - RDF and SPARQL Foundation
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--alpha.2-orange)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-core/badge.svg)](https://docs.rs/oxirs-core)
//!
//! **Status**: Alpha Release (v0.1.0-alpha.2)
//! ⚠️ APIs may change. Not recommended for production use.
//!
//! Zero-dependency, Rust-native RDF data model and SPARQL foundation for the OxiRS semantic web platform.
//! This crate provides the core types, traits, and operations that all other OxiRS crates build upon.
//!
//! ## Features
//!
//! - **RDF 1.2 Support** - Complete RDF data model implementation
//! - **SPARQL Foundation** - Core types for SPARQL query processing
//! - **Zero Dependencies** - Minimal external dependencies for maximum portability
//! - **Memory Efficiency** - Optimized data structures for large knowledge graphs
//! - **Concurrent Operations** - Thread-safe operations for multi-threaded environments
//! - **Storage Backends** - Pluggable storage with memory and disk options
//!
//! ## Quick Start Examples
//!
//! ### Basic Store Operations
//!
//! ```rust,no_run
//! use oxirs_core::store::ConcreteStore;
//! use oxirs_core::model::{NamedNode, Triple};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a new RDF store
//! let store = ConcreteStore::new();
//!
//! // Add RDF data
//! let subject = NamedNode::new("http://example.org/alice")?;
//! let predicate = NamedNode::new("http://example.org/knows")?;
//! let object = NamedNode::new("http://example.org/bob")?;
//! let triple = Triple::new(subject, predicate, object);
//!
//! store.insert_triple(triple)?;
//! println!("Triple added successfully");
//! # Ok(())
//! # }
//! ```
//!
//! ### Consciousness-Inspired Processing
//!
//! ```rust,no_run
//! use oxirs_core::consciousness::{ConsciousnessCoordinator, ConsciousnessLevel};
//!
//! # async fn consciousness_example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut coordinator = ConsciousnessCoordinator::new();
//!
//! // Configure consciousness-inspired optimization
//! coordinator.set_consciousness_level(ConsciousnessLevel::SelfAware);
//! coordinator.enable_dream_processing(true);
//! coordinator.enable_emotional_learning(true);
//!
//! println!("Consciousness coordinator initialized");
//! # Ok(())
//! # }
//! ```

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

// Re-export core types for convenience
pub use model::*;
pub use rdf_store::{ConcreteStore, RdfStore, Store};
pub use transaction::Transaction;

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
