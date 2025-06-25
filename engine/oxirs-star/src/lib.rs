//! # OxiRS RDF-Star
//!
//! RDF-star and SPARQL-star implementation providing comprehensive support for quoted triples.
//!
//! This crate extends the standard RDF model with RDF-star capabilities, allowing triples
//! to be used as subjects or objects in other triples (quoted triples). It provides:
//!
//! - Complete RDF-star data model with proper type safety
//! - Parsing support for Turtle-star, N-Triples-star, TriG-star, and N-Quads-star
//! - SPARQL-star query execution with quoted triple patterns
//! - Serialization to all major RDF-star formats
//! - Storage backend integration with oxirs-core
//! - Performance-optimized handling of nested quoted triples
//!
//! ## Examples
//!
//! ```rust
//! use oxirs_star::{StarStore, StarTriple, StarTerm};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut store = StarStore::new();
//!
//! // Create a quoted triple
//! let quoted = StarTriple::new(
//!     StarTerm::iri("http://example.org/person1")?,
//!     StarTerm::iri("http://example.org/age")?,
//!     StarTerm::literal("25")?,
//! );
//!
//! // Use the quoted triple as a subject
//! let meta_triple = StarTriple::new(
//!     StarTerm::quoted_triple(quoted),
//!     StarTerm::iri("http://example.org/certainty")?,
//!     StarTerm::literal("0.9")?,
//! );
//!
//! store.insert(&meta_triple)?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::str::FromStr;

use anyhow::{Context, Result};
use oxirs_core::{OxirsError, Result as CoreResult};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, error, info, span, Level};

pub mod model;
pub mod parser;
pub mod query;
pub mod reification;
pub mod serializer;
pub mod store;

// Re-export main types
pub use model::*;
pub use store::StarStore;

/// RDF-star specific error types
#[derive(Debug, Error)]
pub enum StarError {
    #[error("Invalid quoted triple: {0}")]
    InvalidQuotedTriple(String),
    #[error("Parse error in RDF-star format: {0}")]
    ParseError(String),
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("SPARQL-star query error: {0}")]
    QueryError(String),
    #[error("Core RDF error: {0}")]
    CoreError(#[from] OxirsError),
    #[error("Reification error: {0}")]
    ReificationError(String),
    #[error("Invalid term type for RDF-star context: {0}")]
    InvalidTermType(String),
}

/// Result type for RDF-star operations
pub type StarResult<T> = std::result::Result<T, StarError>;

/// Configuration for RDF-star processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarConfig {
    /// Maximum nesting depth for quoted triples (default: 10)
    pub max_nesting_depth: usize,
    /// Enable automatic reification fallback
    pub enable_reification_fallback: bool,
    /// Strict mode for parsing (reject invalid constructs)
    pub strict_mode: bool,
    /// Enable SPARQL-star extensions
    pub enable_sparql_star: bool,
    /// Buffer size for streaming operations
    pub buffer_size: usize,
}

impl Default for StarConfig {
    fn default() -> Self {
        Self {
            max_nesting_depth: 10,
            enable_reification_fallback: true,
            strict_mode: false,
            enable_sparql_star: true,
            buffer_size: 8192,
        }
    }
}

/// Statistics for RDF-star processing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StarStatistics {
    /// Total number of quoted triples processed
    pub quoted_triples_count: usize,
    /// Maximum nesting depth encountered
    pub max_nesting_encountered: usize,
    /// Number of reified triples
    pub reified_triples_count: usize,
    /// Number of SPARQL-star queries executed
    pub sparql_star_queries_count: usize,
    /// Processing time statistics (in microseconds)
    pub processing_time_us: u64,
}

/// Initialize the RDF-star system with configuration
pub fn init_star_system(config: StarConfig) -> StarResult<()> {
    let span = span!(Level::INFO, "init_star_system");
    let _enter = span.enter();

    info!("Initializing OxiRS RDF-star system");
    debug!("Configuration: {:?}", config);

    // Validate configuration
    if config.max_nesting_depth == 0 {
        return Err(StarError::InvalidQuotedTriple(
            "Max nesting depth must be greater than 0".to_string(),
        ));
    }

    if config.buffer_size == 0 {
        return Err(StarError::InvalidQuotedTriple(
            "Buffer size must be greater than 0".to_string(),
        ));
    }

    info!("RDF-star system initialized successfully");
    Ok(())
}

/// Utility function to validate quoted triple nesting depth
pub fn validate_nesting_depth(term: &StarTerm, max_depth: usize) -> StarResult<()> {
    fn check_depth(term: &StarTerm, current_depth: usize, max_depth: usize) -> StarResult<usize> {
        match term {
            StarTerm::QuotedTriple(triple) => {
                if current_depth >= max_depth {
                    return Err(StarError::InvalidQuotedTriple(format!(
                        "Nesting depth {} exceeds maximum {}",
                        current_depth, max_depth
                    )));
                }

                let subj_depth = check_depth(&triple.subject, current_depth + 1, max_depth)?;
                let pred_depth = check_depth(&triple.predicate, current_depth + 1, max_depth)?;
                let obj_depth = check_depth(&triple.object, current_depth + 1, max_depth)?;

                Ok(subj_depth.max(pred_depth).max(obj_depth))
            }
            _ => Ok(current_depth),
        }
    }

    check_depth(term, 0, max_depth)?;
    Ok(())
}

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = StarConfig::default();
        assert_eq!(config.max_nesting_depth, 10);
        assert!(config.enable_reification_fallback);
        assert!(!config.strict_mode);
        assert!(config.enable_sparql_star);
    }

    #[test]
    fn test_nesting_depth_validation() {
        let simple_term = StarTerm::iri("http://example.org/test").unwrap();
        assert!(validate_nesting_depth(&simple_term, 5).is_ok());

        // Test nested quoted triple
        let inner_triple = StarTriple::new(
            StarTerm::iri("http://example.org/s").unwrap(),
            StarTerm::iri("http://example.org/p").unwrap(),
            StarTerm::iri("http://example.org/o").unwrap(),
        );
        let nested_term = StarTerm::quoted_triple(inner_triple);
        assert!(validate_nesting_depth(&nested_term, 5).is_ok());
        assert!(validate_nesting_depth(&nested_term, 0).is_err());
    }
}
