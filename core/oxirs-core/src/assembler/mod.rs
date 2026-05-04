//! Jena Assembler vocabulary support.
//!
//! This module allows users migrating from Apache Jena to use their existing
//! `fuseki-config.ttl` assembler files with OxiRS.  A Jena assembler document
//! uses the `ja:` namespace (`http://jena.hpl.hp.com/2005/11/Assembler#`) to
//! describe datasets, named graphs, and storage backends in RDF/Turtle format.
//!
//! # Quick start
//!
//! ```rust
//! use oxirs_core::assembler;
//!
//! let ttl = r#"
//!     @prefix ja: <http://jena.hpl.hp.com/2005/11/Assembler#> .
//!     <http://example.org/ds> a ja:RDFDataset .
//! "#;
//! let config = assembler::from_turtle(ttl).unwrap();
//! assert_eq!(config.len(), 1);
//! ```
//!
//! # Supported vocabulary
//!
//! | Term | Description |
//! |------|-------------|
//! | `ja:RDFDataset` | Generic RDF dataset |
//! | `ja:MemoryDataset` | In-memory dataset |
//! | `ja:MemoryModel` | In-memory RDF model |
//! | `tdb2:DatasetTDB2` | TDB2 disk-backed dataset |
//! | `ja:namedGraph` | Named-graph description |
//! | `ja:graphName` | IRI of a named graph |
//! | `ja:graph` | Model resource for a named graph |
//! | `ja:defaultGraph` | Default-graph model resource |
//! | `ja:contentURL` | URL to load initial RDF content |
//! | `tdb2:location` | Filesystem path for TDB2 storage |

pub mod builder;
pub mod config;
pub mod vocab;

pub use builder::{AssemblerBuilder, AssemblerError};
pub use config::{AssemblerConfig, DatasetConfig, GraphConfig, StoreBackend};

/// Parse a Turtle-format Jena Assembler document into an [`AssemblerConfig`].
///
/// This is a convenience wrapper around [`AssemblerBuilder::from_turtle`].
///
/// # Example
///
/// ```rust
/// let ttl = r#"
///     @prefix ja: <http://jena.hpl.hp.com/2005/11/Assembler#> .
///     <http://example.org/ds> a ja:RDFDataset .
/// "#;
/// let config = oxirs_core::assembler::from_turtle(ttl).unwrap();
/// assert_eq!(config.len(), 1);
/// ```
pub fn from_turtle(ttl: &str) -> Result<AssemblerConfig, AssemblerError> {
    AssemblerBuilder::from_turtle(ttl)
}
