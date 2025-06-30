//! Core traits for explainable AI components
//!
//! This module defines the fundamental traits that enable different components
//! of the explainable AI system to work together.

use crate::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use super::types::*;

/// Trait for components that can generate explanations
#[async_trait]
pub trait ExplanationGenerator: Send + Sync + Debug {
    /// Generate an explanation for the given data
    async fn generate_explanation(&self, data: &ExplanationData) -> Result<RawExplanation>;
    /// Clone this generator into a boxed trait object
    fn clone_box(&self) -> Box<dyn ExplanationGenerator>;
}

/// Trait for components that can analyze interpretability
#[async_trait]
pub trait InterpretabilityAnalyzer: Send + Sync + Debug {
    /// Analyze the interpretability of the given data
    async fn analyze(&self, data: &ExplanationData) -> Result<FeatureImportanceAnalysis>;
    /// Clone this analyzer into a boxed trait object
    fn clone_box(&self) -> Box<dyn InterpretabilityAnalyzer>;
}

/// Trait for decision trackers
pub trait DecisionTracker: Send + Sync + Debug {
    /// Track a decision
    fn track_decision(&mut self, decision: DecisionContext);
    /// Get decision history
    fn get_history(&self) -> Vec<DecisionContext>;
    /// Clear decision history
    fn clear_history(&mut self);
}