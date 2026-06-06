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

// A/B testing framework
pub mod ab_test;

// Re-exports for A/B testing
pub use ab_test::{
    evaluate_hits_at_k, evaluate_link_prediction, evaluate_silhouette, AbTestResult, AbTestRunner,
    EmbedMetric, ModelEvalResult, StatTest,
};

// Embedding quality metrics: link prediction, analogy, clustering
pub mod embedding_metrics;

pub use embedding_metrics::{
    AnalogicalReasoningBenchmark, AnalogyQuad, EmbeddingClusteringMetrics, EmbeddingEvaluator,
};

// ── KGC evaluation framework (FB15k-237 / WN18RR style) ──────────────────

/// Standard KGC evaluation metrics (MRR, Hits@K, Mean Rank, filtered variants).
pub mod kgc_metrics;

/// KGC dataset with train/valid/test splits and TSV loading.
pub mod kgc_dataset;

/// KGC evaluator and high-level evaluation suite.
pub mod kgc_evaluator;

// Flat re-exports for convenience.
pub use kgc_dataset::{EvaluationTriple, KgcDataset};
pub use kgc_evaluator::{EvalSplit, KgcEvaluationSuite, KgcEvaluator, KgcEvaluatorConfig};
pub use kgc_metrics::{
    compute_filtered_rank, hits_at_k, mean_rank, mean_reciprocal_rank, EvaluationMetrics,
};
