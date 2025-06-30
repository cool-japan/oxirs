//! Evaluation module for embeddings and knowledge graphs
//! 
//! This module provides comprehensive evaluation tools and metrics for assessing
//! the quality of embeddings and knowledge graph reasoning systems, including
//! advanced ML techniques for uncertainty quantification, adversarial robustness,
//! fairness assessment, and explanation quality evaluation.

pub mod query_evaluation;
pub mod reasoning_evaluation;
pub mod advanced_evaluation;

// Re-export commonly used types and functions, avoiding conflicts
pub use query_evaluation::{
    QueryAnsweringEvaluator, QueryEvaluationConfig, QueryEvaluationResults,
    QueryType, QueryMetric, QueryTemplate, QueryResult, TypeSpecificResults,
    ReasoningStep as QueryReasoningStep
};
pub use reasoning_evaluation::{
    ReasoningTaskEvaluator, ReasoningEvaluationConfig, ReasoningEvaluationResults,
    ReasoningType, ReasoningRule, ReasoningChain, ReasoningStep
};
pub use advanced_evaluation::{
    AdvancedEvaluator, AdvancedEvaluationConfig, AdvancedEvaluationResults,
    BasicMetrics, UncertaintyResults, AdversarialResults, FairnessResults,
    ExplanationResults, TemporalResults, CrossDomainResults,
    UncertaintyQuantifier, AdversarialAttackGenerator, FairnessAssessment,
    ExplanationQualityEvaluator
};