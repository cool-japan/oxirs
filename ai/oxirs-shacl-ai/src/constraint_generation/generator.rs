//! Main constraint generator integrating all analyzers

use oxirs_core::{model::NamedNode, Store};
use serde::{Deserialize, Serialize};

use super::cardinality::CardinalityAnalyzer;
use super::config::ConstraintGenerationConfig;
use super::datatype::DatatypeAnalyzer;
use super::pattern_based::PatternBasedGenerator;
use super::ranker::{ConstraintRanker, RankingCriteria};
use super::suggestions::{ConstraintSuggestion, SuggestionEngine};
use super::types::GeneratedConstraint;
use super::validator::{ConstraintValidator, ValidationResult};
use super::value_range::ValueRangeAnalyzer;
use crate::{Result, ShaclAiError};

/// Complete generation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    /// Generated constraints
    pub constraints: Vec<GeneratedConstraint>,
    /// Suggestions
    pub suggestions: Vec<ConstraintSuggestion>,
    /// Validation results
    pub validation_results: Vec<ValidationResult>,
    /// Statistics
    pub statistics: GenerationStatistics,
}

/// Generation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationStatistics {
    /// Total constraints generated
    pub total_generated: usize,
    /// Valid constraints
    pub valid_constraints: usize,
    /// Invalid constraints
    pub invalid_constraints: usize,
    /// Conflicts detected
    pub conflicts_detected: usize,
    /// Properties analyzed
    pub properties_analyzed: usize,
    /// Generation time (ms)
    pub generation_time_ms: f64,
}

/// Main constraint generator
pub struct ConstraintGenerator {
    config: ConstraintGenerationConfig,
    cardinality_analyzer: CardinalityAnalyzer,
    datatype_analyzer: DatatypeAnalyzer,
    value_range_analyzer: ValueRangeAnalyzer,
    pattern_generator: PatternBasedGenerator,
    ranker: ConstraintRanker,
    validator: ConstraintValidator,
    suggestion_engine: SuggestionEngine,
}

impl ConstraintGenerator {
    pub fn new(config: ConstraintGenerationConfig) -> Self {
        let cardinality_analyzer = CardinalityAnalyzer::new()
            .with_min_sample_size(config.min_sample_size)
            .with_min_confidence(config.min_confidence);

        let datatype_analyzer = DatatypeAnalyzer::new()
            .with_min_sample_size(config.min_sample_size)
            .with_min_confidence(config.min_confidence);

        let value_range_analyzer =
            ValueRangeAnalyzer::new().with_min_sample_size(config.min_sample_size);

        let pattern_generator =
            PatternBasedGenerator::new().with_min_sample_size(config.min_sample_size);

        let ranker = ConstraintRanker::with_default_criteria();
        let validator = ConstraintValidator::non_strict();
        let suggestion_engine = SuggestionEngine::new().with_min_confidence(config.min_confidence);

        Self {
            config,
            cardinality_analyzer,
            datatype_analyzer,
            value_range_analyzer,
            pattern_generator,
            ranker,
            validator,
            suggestion_engine,
        }
    }

    pub fn config(&self) -> &ConstraintGenerationConfig {
        &self.config
    }

    /// Generate constraints for a property
    pub fn generate_for_property(
        &self,
        store: &dyn Store,
        property: &NamedNode,
        class: Option<&NamedNode>,
    ) -> Result<GenerationResult> {
        if !self.config.enabled {
            return Ok(GenerationResult {
                constraints: Vec::new(),
                suggestions: Vec::new(),
                validation_results: Vec::new(),
                statistics: GenerationStatistics {
                    total_generated: 0,
                    valid_constraints: 0,
                    invalid_constraints: 0,
                    conflicts_detected: 0,
                    properties_analyzed: 0,
                    generation_time_ms: 0.0,
                },
            });
        }

        let start_time = std::time::Instant::now();
        let mut all_constraints = Vec::new();

        // Generate cardinality constraints
        if self
            .config
            .enabled_types
            .contains(&super::config::ConstraintType::Cardinality)
        {
            let constraints = self
                .cardinality_analyzer
                .analyze_property(store, property, class)?;
            all_constraints.extend(constraints);
        }

        // Generate datatype constraints
        if self
            .config
            .enabled_types
            .contains(&super::config::ConstraintType::Datatype)
        {
            let constraints = self
                .datatype_analyzer
                .analyze_property(store, property, class)?;
            all_constraints.extend(constraints);
        }

        // Generate value range constraints
        if self
            .config
            .enabled_types
            .contains(&super::config::ConstraintType::ValueRange)
        {
            let constraints = self
                .value_range_analyzer
                .analyze_property(store, property, class)?;
            all_constraints.extend(constraints);
        }

        // Generate pattern constraints
        if self
            .config
            .enabled_types
            .contains(&super::config::ConstraintType::Pattern)
        {
            let constraints = self
                .pattern_generator
                .analyze_property(store, property, class)?;
            all_constraints.extend(constraints);
        }

        // Rank constraints
        let ranked = if self.config.enable_ranking {
            self.ranker.rank_constraints(all_constraints.clone());
            all_constraints
        } else {
            all_constraints
        };

        // Validate constraints
        let mut validation_results = Vec::new();
        let mut valid_constraints = Vec::new();
        let mut invalid_count = 0;

        if self.config.enable_validation {
            for constraint in ranked {
                let validation = self.validator.validate_constraint(&constraint);
                if validation.is_valid {
                    valid_constraints.push(constraint);
                } else {
                    invalid_count += 1;
                }
                validation_results.push(validation);
            }
        } else {
            valid_constraints = ranked;
        }

        // Detect conflicts
        let mut conflicts = Vec::new();
        if self.config.enable_conflict_detection {
            conflicts = self.validator.detect_conflicts(&valid_constraints);
            if !conflicts.is_empty() {
                valid_constraints = self
                    .validator
                    .resolve_conflicts(valid_constraints, conflicts.clone());
            }
        }

        // Generate suggestions
        let suggestions = self
            .suggestion_engine
            .generate_suggestions(valid_constraints.clone());

        // Limit to max constraints per property
        valid_constraints.truncate(self.config.max_constraints_per_property);

        let generation_time = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(GenerationResult {
            constraints: valid_constraints.clone(),
            suggestions,
            validation_results,
            statistics: GenerationStatistics {
                total_generated: valid_constraints.len() + invalid_count,
                valid_constraints: valid_constraints.len(),
                invalid_constraints: invalid_count,
                conflicts_detected: conflicts.len(),
                properties_analyzed: 1,
                generation_time_ms: generation_time,
            },
        })
    }

    /// Generate constraints for multiple properties
    pub fn generate_for_properties(
        &self,
        store: &dyn Store,
        properties: &[NamedNode],
        class: Option<&NamedNode>,
    ) -> Result<GenerationResult> {
        let start_time = std::time::Instant::now();

        let mut all_results = Vec::new();

        for property in properties {
            let result = self.generate_for_property(store, property, class)?;
            all_results.push(result);
        }

        // Aggregate results
        let mut all_constraints = Vec::new();
        let mut all_suggestions = Vec::new();
        let mut all_validations = Vec::new();
        let mut total_stats = GenerationStatistics {
            total_generated: 0,
            valid_constraints: 0,
            invalid_constraints: 0,
            conflicts_detected: 0,
            properties_analyzed: properties.len(),
            generation_time_ms: 0.0,
        };

        for result in all_results {
            all_constraints.extend(result.constraints);
            all_suggestions.extend(result.suggestions);
            all_validations.extend(result.validation_results);

            total_stats.total_generated += result.statistics.total_generated;
            total_stats.valid_constraints += result.statistics.valid_constraints;
            total_stats.invalid_constraints += result.statistics.invalid_constraints;
            total_stats.conflicts_detected += result.statistics.conflicts_detected;
        }

        total_stats.generation_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        Ok(GenerationResult {
            constraints: all_constraints,
            suggestions: all_suggestions,
            validation_results: all_validations,
            statistics: total_stats,
        })
    }

    /// Generate constraints for an entire class
    pub fn generate_for_class(
        &self,
        store: &dyn Store,
        class: &NamedNode,
    ) -> Result<GenerationResult> {
        // In a real implementation, this would discover properties of the class
        // For now, we return a stub
        self.generate_for_properties(store, &[], Some(class))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let config = ConstraintGenerationConfig::default();
        let generator = ConstraintGenerator::new(config);
        assert!(generator.config().enabled);
    }

    #[test]
    fn test_generation_statistics() {
        let stats = GenerationStatistics {
            total_generated: 10,
            valid_constraints: 8,
            invalid_constraints: 2,
            conflicts_detected: 1,
            properties_analyzed: 5,
            generation_time_ms: 150.0,
        };

        assert_eq!(stats.total_generated, 10);
        assert_eq!(stats.valid_constraints, 8);
    }

    #[test]
    fn test_disabled_generation() {
        let config = ConstraintGenerationConfig {
            enabled: false,
            ..Default::default()
        };

        let generator = ConstraintGenerator::new(config);
        let property = NamedNode::new_unchecked("http://example.org/prop");

        // Should compile but we can't test without a Store implementation
        // In real tests, this would use a mock store
    }
}
