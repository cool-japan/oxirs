//! SHACL Advanced Features (SHACL-AF)
//!
//! This module implements advanced SHACL features including:
//! - SHACL Rules for data transformation
//! - SHACL Functions for custom operations
//! - Advanced target definitions
//! - Conditional constraints
//! - Qualified value shapes
//! - Recursive shape definitions
//!
//! Based on the W3C SHACL Advanced Features specification.
//!
//! Note: This is an alpha implementation. Many features are placeholders
//! that will be fully implemented in future releases.

#![allow(dead_code, unused_variables)]

pub mod advanced_targets;
pub mod conditional;
pub mod functions;
pub mod parameterized_constraints;
pub mod qualified_shapes;
pub mod reasoning;
pub mod recursive_shapes;
pub mod rules;
pub mod shape_evolution;
pub mod shape_inference;
pub mod shape_operations;

// Re-export key types
pub use rules::{
    RuleEngine, RuleEngineStats, RuleExecutionResult, RuleMetadata, RuleType, ShaclRule,
};

pub use functions::{
    BuiltInFunctionExecutor, FunctionContext, FunctionExecutor, FunctionInvocation,
    FunctionMetadata, FunctionParameter, FunctionRegistry, FunctionResult, ParameterType,
    ReturnType, ShaclFunction,
};

pub use advanced_targets::{
    AdvancedTarget, AdvancedTargetSelector, CacheStats as TargetCacheStats, TargetCacheConfig,
    TargetSelectionStats,
};

pub use conditional::{
    ConditionalConstraint, ConditionalEvaluator, ConditionalResult, ShapeRegistry,
};

pub use shape_evolution::{
    compare_shapes, ShapeDifference, ShapeEvolutionEvent, ShapeEvolutionRegistry,
    ShapeEvolutionTracker, ShapeVersion,
};
pub use shape_inference::{
    Anomaly, AnomalyDetectionConfig, AnomalyDetectionMethod, AnomalyDetectionResult,
    AnomalyDetectionStats, AnomalyDetector, AnomalyType, InferenceMetadata, InferenceStats,
    InferenceStrategy, InferredShape, RefinementType, ShapeInferenceConfig, ShapeInferenceEngine,
    ShapeRefinement,
};
pub use shape_operations::{
    GeneralizationStrategy, MergeStrategy, RefactoringConfig, ShapeGeneralizer, ShapeMerger,
    ShapeRefactorer, ShapeSpecializer, SpecializationStrategy,
};

pub use qualified_shapes::{
    ComplexQualifiedConstraint, ComplexValidationResult, QualifiedShape, QualifiedShapesValidator,
    QualifiedValidationResult, QualifiedValueShapeConstraint,
    ShapeRegistry as QualifiedShapeRegistry,
};

pub use recursive_shapes::{
    RecursionStats, RecursionStrategy, RecursiveShapeValidator, RecursiveValidationConfig,
    RecursiveValidationResult, ShapeDependencyAnalyzer, ShapeResolver,
};

pub use parameterized_constraints::{
    ConstraintExecutionResult, ConstraintImplementation, ConstraintInstance, ConstraintParameter,
    ParameterTypeConstraint, ParameterValue, ParameterizedConstraintComponent,
    ParameterizedConstraintRegistry, ScriptLanguage,
};

pub use reasoning::{
    ClosedWorldValidator, CustomReasoning, EntailmentRegime, EvidenceData, InferredTriple, NafGoal,
    NegationAsFailure, ProbabilisticConfig, ProbabilisticStats, ProbabilisticValidationResult,
    ProbabilisticValidator, ReasoningConfig, ReasoningStats, ReasoningValidationResult,
    ReasoningValidator,
};

/// Version of SHACL-AF implementation
pub const SHACL_AF_VERSION: &str = "1.0.0-alpha";

/// Check if a feature is supported
pub fn is_feature_supported(feature: &str) -> bool {
    matches!(
        feature,
        "rules" | "functions" | "advanced-targets" | "triple-rules" | "construct-rules"
    )
}

/// Get all supported SHACL-AF features
pub fn supported_features() -> Vec<&'static str> {
    vec![
        "rules",
        "functions",
        "advanced-targets",
        "triple-rules",
        "construct-rules",
        "sparql-rules",
        "sparql-targets",
        "target-objects-of",
        "target-subjects-of",
        "implicit-targets",
        "path-targets",
        "function-targets",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_support() {
        assert!(is_feature_supported("rules"));
        assert!(is_feature_supported("functions"));
        assert!(is_feature_supported("advanced-targets"));
        assert!(!is_feature_supported("nonexistent-feature"));
    }

    #[test]
    fn test_supported_features() {
        let features = supported_features();
        assert!(!features.is_empty());
        assert!(features.contains(&"rules"));
        assert!(features.contains(&"functions"));
    }
}
