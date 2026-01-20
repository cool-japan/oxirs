//! # HistoricalKnowledgeIntegrator - Trait Implementations
//!
//! This module contains trait implementations for `HistoricalKnowledgeIntegrator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{HistoricalKnowledgeIntegrator, UniversalKnowledgeConfig};

impl Default for HistoricalKnowledgeIntegrator {
    fn default() -> Self {
        Self::new(&UniversalKnowledgeConfig::default())
    }
}
