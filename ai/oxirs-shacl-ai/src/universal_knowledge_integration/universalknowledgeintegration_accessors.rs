//! # UniversalKnowledgeIntegration - accessors Methods
//!
//! This module contains method implementations for `UniversalKnowledgeIntegration`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::Result;

use super::types::UniversalKnowledgeMetrics;
use super::universalknowledgeintegration_type::UniversalKnowledgeIntegration;
use std::collections::{HashMap, HashSet};

impl UniversalKnowledgeIntegration {
    /// Get universal knowledge integration metrics
    pub async fn get_universal_knowledge_metrics(&self) -> Result<UniversalKnowledgeMetrics> {
        Ok(self.integration_metrics.read().await.clone())
    }
}
