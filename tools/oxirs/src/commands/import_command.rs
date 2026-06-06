//! # Import Command — Multi-format RDF Importer
//!
//! Provides `ImportCommand` for parsing RDF data from Turtle, N-Triples,
//! N-Quads, JSON-LD, RDF/XML, TriG, and CSV formats.
//!
//! This module is a thin facade. The implementation is split across sibling
//! modules:
//! - [`import_command_types`](super::import_command_types): core data types
//!   ([`ImportFormat`], [`Triple`], [`ImportResult`], [`ImportError`]).
//! - [`import_command_runner`](super::import_command_runner): the
//!   [`ImportCommand`] entry point, format detection, and shared helpers.
//! - [`import_command_formats`](super::import_command_formats): the
//!   format-specific parsers.
//!
//! # Example
//!
//! ```rust
//! use oxirs::commands::import_command::{ImportCommand, ImportFormat};
//!
//! let nt = "<http://a.org/s> <http://a.org/p> <http://a.org/o> .\n";
//! let result = ImportCommand::import(nt, ImportFormat::NTriples).expect("ok");
//! assert_eq!(result.triple_count(), 1);
//! ```

pub use super::import_command_runner::*;
pub use super::import_command_types::*;
