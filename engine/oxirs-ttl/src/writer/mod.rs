//! Writer module: high-level Turtle serializer with prefix abbreviation
//!
//! Provides [`TurtleWriter`] together with the lightweight [`RdfTerm`] /
//! [`TermType`] types that are used across the standalone parser and diff
//! modules in this crate.

pub mod turtle_writer;

pub use turtle_writer::{RdfTerm, TermType, TurtleWriter, TurtleWriterConfig};
