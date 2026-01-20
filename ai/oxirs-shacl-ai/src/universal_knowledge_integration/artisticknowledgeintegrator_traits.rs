//! # ArtisticKnowledgeIntegrator - Trait Implementations
//!
//! This module contains trait implementations for `ArtisticKnowledgeIntegrator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ArtisticKnowledgeIntegrator, UniversalKnowledgeConfig};

impl Default for ArtisticKnowledgeIntegrator {
    fn default() -> Self {
        Self::new(&UniversalKnowledgeConfig::default())
    }
}
