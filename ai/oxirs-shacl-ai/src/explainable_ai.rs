//! Explainable AI for SHACL-AI Interpretability
//!
//! This module provides comprehensive explainability and interpretability capabilities
//! for the SHACL-AI system, enabling users to understand how AI decisions are made,
//! why certain patterns are recognized, and how validation outcomes are determined.
//!
//! The module has been refactored into sub-modules for better organization:
//! - `explainable::traits` - Core traits for explainable AI components
//! - `explainable::types` - Data structures and configuration types
//! - `explainable::explainers` - Explanation generator implementations
//! - `explainable::analyzers` - Interpretability analyzer implementations
//! - `explainable::processors` - Natural language and processing utilities

mod explainable;

// Re-export everything from the explainable module for backward compatibility
pub use explainable::*;

// Additional convenience re-exports for commonly used items
pub use explainable::{
    AdaptationExplanation, ExplainableAI, ExplainableAIConfig, ExplanationData,
    InterpretabilityReport, PatternExplanation, PatternRecognitionContext, ProcessedExplanation,
    QuantumExplanation, QuantumPatternContext, RawExplanation, SimpleDecisionTracker,
    ValidationContext, ValidationExplanation,
};

pub use explainable::{DecisionTracker, ExplanationGenerator, InterpretabilityAnalyzer};

pub use explainable::{
    AdaptationLogicExplainer, NeuralDecisionExplainer, PatternRecognitionExplainer,
    QuantumPatternExplainer, ValidationReasoningExplainer,
};

pub use explainable::{
    AttentionAnalyzer, CounterfactualAnalyzer, DecisionPathAnalyzer, FeatureImportanceAnalyzer,
    ModelBehaviorAnalyzer,
};

pub use explainable::NaturalLanguageProcessor;
