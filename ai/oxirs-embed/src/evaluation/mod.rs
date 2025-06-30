//! Evaluation module for embeddings and knowledge graphs
//!
//! This module provides comprehensive evaluation tools and metrics for assessing
//! the quality of embeddings and knowledge graph reasoning systems, including
//! advanced ML techniques for uncertainty quantification, adversarial robustness,
//! fairness assessment, and explanation quality evaluation.

pub mod advanced_evaluation;
pub mod query_evaluation;
pub mod reasoning_evaluation;

// Re-export commonly used types and functions, avoiding conflicts
pub use advanced_evaluation::{
    AdvancedEvaluationConfig, AdvancedEvaluationResults, AdvancedEvaluator,
    AdversarialAttackGenerator, AdversarialResults, BasicMetrics, CrossDomainResults,
    ExplanationQualityEvaluator, ExplanationResults, FairnessAssessment, FairnessResults,
    TemporalResults, UncertaintyQuantifier, UncertaintyResults,
};
pub use query_evaluation::{
    QueryAnsweringEvaluator, QueryEvaluationConfig, QueryEvaluationResults, QueryMetric,
    QueryResult, QueryTemplate, QueryType, ReasoningStep as QueryReasoningStep,
    TypeSpecificResults,
};
pub use reasoning_evaluation::{
    ReasoningChain, ReasoningEvaluationConfig, ReasoningEvaluationResults, ReasoningRule,
    ReasoningStep, ReasoningTaskEvaluator, ReasoningType,
};
