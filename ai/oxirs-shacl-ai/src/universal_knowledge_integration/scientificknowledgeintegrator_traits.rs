//! # ScientificKnowledgeIntegrator - Trait Implementations
//!
//! This module contains trait implementations for `ScientificKnowledgeIntegrator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ScientificKnowledgeIntegrator, UniversalKnowledgeConfig};

impl Default for ScientificKnowledgeIntegrator {
    fn default() -> Self {
        Self::new(&UniversalKnowledgeConfig::default())
    }
}
