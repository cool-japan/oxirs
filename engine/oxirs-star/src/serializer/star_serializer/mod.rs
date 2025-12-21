//! RDF-star serializer module

use crate::StarConfig;

pub mod core;
pub mod utils;

// Re-export ChunkedIterator for public use
pub use utils::ChunkedIterator;

/// RDF-star serializer with support for multiple formats
pub struct StarSerializer {
    pub(super) config: StarConfig,
    pub(super) simd_escaper: super::simd_escape::SimdEscaper,
}

impl Default for StarSerializer {
    fn default() -> Self {
        Self::new()
    }
}
