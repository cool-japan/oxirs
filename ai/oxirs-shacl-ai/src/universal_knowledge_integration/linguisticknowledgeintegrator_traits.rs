//! # LinguisticKnowledgeIntegrator - Trait Implementations
//!
//! This module contains trait implementations for `LinguisticKnowledgeIntegrator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{LinguisticKnowledgeIntegrator, UniversalKnowledgeConfig};

impl Default for LinguisticKnowledgeIntegrator {
    fn default() -> Self {
        Self::new(&UniversalKnowledgeConfig::default())
    }
}
