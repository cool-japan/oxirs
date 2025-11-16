//! RDF-star serializer module

use crate::StarConfig;

pub mod core;
pub mod utils;

// Re-export ChunkedIterator for public use
pub use utils::ChunkedIterator;

/// RDF-star serializer with support for multiple formats
pub struct StarSerializer {
    config: StarConfig,
}

impl Default for StarSerializer {
    fn default() -> Self {
        Self::new()
    }
}
