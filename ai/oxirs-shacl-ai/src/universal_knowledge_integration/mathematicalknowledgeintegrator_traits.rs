//! # MathematicalKnowledgeIntegrator - Trait Implementations
//!
//! This module contains trait implementations for `MathematicalKnowledgeIntegrator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{MathematicalKnowledgeIntegrator, UniversalKnowledgeConfig};

impl Default for MathematicalKnowledgeIntegrator {
    fn default() -> Self {
        Self::new(&UniversalKnowledgeConfig::default())
    }
}
