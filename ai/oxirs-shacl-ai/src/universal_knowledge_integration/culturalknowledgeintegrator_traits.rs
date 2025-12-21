//! # CulturalKnowledgeIntegrator - Trait Implementations
//!
//! This module contains trait implementations for `CulturalKnowledgeIntegrator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CulturalKnowledgeIntegrator, UniversalKnowledgeConfig};

impl Default for CulturalKnowledgeIntegrator {
    fn default() -> Self {
        Self::new(&UniversalKnowledgeConfig::default())
    }
}
