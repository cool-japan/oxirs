//! Automatic Constraint Generation Module
//!
//! This module provides AI-powered automatic constraint generation from RDF data,
//! including cardinality, datatype, value range, and relationship constraints.

pub mod cardinality;
pub mod config;
pub mod datatype;
pub mod generator;
pub mod pattern_based;
pub mod ranker;
pub mod suggestions;
pub mod transformer_based;
pub mod types;
pub mod validator;
pub mod value_range;

// Re-export key types
pub use cardinality::{CardinalityAnalyzer, CardinalityConstraint};
pub use config::{ConstraintGenerationConfig, ConstraintType as ConfigConstraintType};
pub use datatype::{DatatypeAnalyzer, DatatypeConstraint};
pub use generator::{ConstraintGenerator, GenerationResult};
pub use pattern_based::{PatternBasedGenerator, PatternConstraint};
pub use ranker::{ConstraintRanker, RankedConstraint, RankingCriteria};
pub use suggestions::{
    ConstraintSuggestion, SuggestionConfidence, SuggestionEngine, SuggestionReason,
};
pub use transformer_based::{
    ConstraintTrainingExample, FineTuningResult, PatternType, RdfPattern,
    TransformerConstraintConfig, TransformerConstraintGenerator, TransformerConstraintStats,
};
pub use types::{
    ConflictResolution, Constraint, ConstraintConflict, ConstraintMetadata, ConstraintQuality,
    GeneratedConstraint,
};
pub use validator::{ConstraintValidator, ValidationResult};
pub use value_range::{ValueRangeAnalyzer, ValueRangeConstraint};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_generator_creation() {
        let config = ConstraintGenerationConfig::default();
        let generator = ConstraintGenerator::new(config);
        assert!(generator.config().enabled);
    }

    #[test]
    fn test_constraint_types() {
        let types = [
            ConfigConstraintType::Cardinality,
            ConfigConstraintType::Datatype,
            ConfigConstraintType::ValueRange,
        ];
        assert_eq!(types.len(), 3);
    }
}
