//! # TechnicalKnowledgeIntegrator - Trait Implementations
//!
//! This module contains trait implementations for `TechnicalKnowledgeIntegrator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{TechnicalKnowledgeIntegrator, UniversalKnowledgeConfig};

impl Default for TechnicalKnowledgeIntegrator {
    fn default() -> Self {
        Self::new(&UniversalKnowledgeConfig::default())
    }
}
