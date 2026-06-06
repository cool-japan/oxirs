//! Vector compression — thin facade module.
//!
//! The implementation lives in sibling modules:
//! - [`crate::compression_types`]: codec/config enums, traits, metrics, analysis.
//! - [`crate::compression_codecs`]: concrete codec implementations.
//! - [`crate::compression_io`]: factory, method equivalence, adaptive compressor.

pub use crate::compression_codecs::*;
pub use crate::compression_io::*;
pub use crate::compression_types::*;
