//! Constraint suggestion engine with confidence scoring

use oxirs_core::model::NamedNode;
use serde::{Deserialize, Serialize};

use super::types::{ConstraintType, GeneratedConstraint};

/// Constraint suggestion with confidence and reasoning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSuggestion {
    /// Suggestion ID
    pub id: String,
    /// Suggested constraint
    pub constraint: GeneratedConstraint,
    /// Confidence level
    pub confidence: SuggestionConfidence,
    /// Reason for suggestion
    pub reason: SuggestionReason,
    /// Priority (0-10, higher = more important)
    pub priority: u8,
    /// Expected impact
    pub expected_impact: ImpactAssessment,
    /// Recommendation
    pub recommendation: String,
}

/// Confidence levels for suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionConfidence {
    /// Very High confidence (>95%)
    VeryHigh,
    /// High confidence (85-95%)
    High,
    /// Medium confidence (70-85%)
    Medium,
    /// Low confidence (50-70%)
    Low,
    /// Very low confidence (<50%)
    VeryLow,
}

impl SuggestionConfidence {
    pub fn from_score(score: f64) -> Self {
        if score >= 0.95 {
            Self::VeryHigh
        } else if score >= 0.85 {
            Self::High
        } else if score >= 0.70 {
            Self::Medium
        } else if score >= 0.50 {
            Self::Low
        } else {
            Self::VeryLow
        }
    }

    pub fn score(&self) -> f64 {
        match self {
            Self::VeryHigh => 0.975,
            Self::High => 0.90,
            Self::Medium => 0.775,
            Self::Low => 0.60,
            Self::VeryLow => 0.40,
        }
    }
}

/// Reasons for constraint suggestions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionReason {
    /// High data consistency
    DataConsistency { consistency_rate: f64 },
    /// Pattern detected in data
    PatternDetected { pattern_match_rate: f64 },
    /// Best practice recommendation
    BestPractice { practice_name: String },
    /// Prevent common errors
    ErrorPrevention {
        error_type: String,
        prevention_rate: f64,
    },
    /// Improve data quality
    QualityImprovement {
        current_quality: f64,
        expected_quality: f64,
    },
    /// Domain-specific constraint
    DomainSpecific { domain: String },
}

/// Impact assessment for applying constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    /// Positive impacts
    pub positive_impacts: Vec<String>,
    /// Potential negative impacts
    pub negative_impacts: Vec<String>,
    /// Affected entities count (estimate)
    pub affected_entities: usize,
    /// Risk level (0.0 = no risk, 1.0 = high risk)
    pub risk_level: f64,
}

/// Suggestion engine
pub struct SuggestionEngine {
    min_confidence: f64,
}

impl SuggestionEngine {
    pub fn new() -> Self {
        Self {
            min_confidence: 0.7,
        }
    }

    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Generate suggestions from constraints
    pub fn generate_suggestions(
        &self,
        constraints: Vec<GeneratedConstraint>,
    ) -> Vec<ConstraintSuggestion> {
        constraints
            .into_iter()
            .filter(|c| c.metadata.confidence >= self.min_confidence)
            .map(|c| self.create_suggestion(c))
            .collect()
    }

    fn create_suggestion(&self, constraint: GeneratedConstraint) -> ConstraintSuggestion {
        let confidence = SuggestionConfidence::from_score(constraint.metadata.confidence);
        let priority = self.calculate_priority(&constraint);
        let reason = self.determine_reason(&constraint);
        let impact = self.assess_impact(&constraint);
        let recommendation = self.generate_recommendation(&constraint);

        ConstraintSuggestion {
            id: format!("suggestion_{}", constraint.id),
            constraint,
            confidence,
            reason,
            priority,
            expected_impact: impact,
            recommendation,
        }
    }

    fn calculate_priority(&self, constraint: &GeneratedConstraint) -> u8 {
        let base_priority = constraint.constraint_type.priority();
        let confidence_boost = (constraint.metadata.confidence * 2.0) as u8;
        let quality_boost = (constraint.quality.overall_score * 2.0) as u8;

        (base_priority + confidence_boost + quality_boost).min(10)
    }

    fn determine_reason(&self, constraint: &GeneratedConstraint) -> SuggestionReason {
        match constraint.constraint_type {
            ConstraintType::Datatype | ConstraintType::NodeKind => {
                SuggestionReason::DataConsistency {
                    consistency_rate: constraint.metadata.support,
                }
            }
            ConstraintType::Pattern => SuggestionReason::PatternDetected {
                pattern_match_rate: constraint.metadata.support,
            },
            ConstraintType::Cardinality => SuggestionReason::BestPractice {
                practice_name: "Define clear cardinality constraints".to_string(),
            },
            ConstraintType::ValueRange => SuggestionReason::ErrorPrevention {
                error_type: "Out-of-range values".to_string(),
                prevention_rate: constraint.metadata.support,
            },
            _ => SuggestionReason::QualityImprovement {
                current_quality: 0.7,
                expected_quality: constraint.quality.overall_score,
            },
        }
    }

    fn assess_impact(&self, constraint: &GeneratedConstraint) -> ImpactAssessment {
        let mut positive_impacts = vec![
            "Improved data quality".to_string(),
            "Better validation coverage".to_string(),
        ];

        let mut negative_impacts = Vec::new();

        if constraint.metadata.counter_examples > 0 {
            negative_impacts.push(format!(
                "{} existing values will fail validation",
                constraint.metadata.counter_examples
            ));
        }

        let risk_level = if constraint.metadata.counter_examples as f64
            / constraint.metadata.sample_count.max(1) as f64
            > 0.1
        {
            0.7 // High risk if >10% violations
        } else if constraint.metadata.confidence < 0.8 {
            0.5 // Medium risk if low confidence
        } else {
            0.2 // Low risk
        };

        // Add type-specific impacts
        match constraint.constraint_type {
            ConstraintType::Cardinality => {
                positive_impacts.push("Prevent missing required values".to_string());
            }
            ConstraintType::ValueRange => {
                positive_impacts.push("Prevent invalid numeric values".to_string());
            }
            ConstraintType::Pattern => {
                positive_impacts.push("Ensure consistent format".to_string());
            }
            _ => {}
        }

        ImpactAssessment {
            positive_impacts,
            negative_impacts,
            affected_entities: constraint.metadata.sample_count,
            risk_level,
        }
    }

    fn generate_recommendation(&self, constraint: &GeneratedConstraint) -> String {
        if constraint.quality.is_high_quality() {
            format!(
                "✓ Strongly recommended: Apply this {} constraint (confidence: {:.1}%)",
                constraint.constraint_type.name(),
                constraint.metadata.confidence * 100.0
            )
        } else if constraint.metadata.confidence >= 0.7 {
            format!(
                "→ Recommended: Consider applying this {} constraint after review",
                constraint.constraint_type.name()
            )
        } else {
            format!(
                "? Review needed: Low confidence {} constraint requires validation",
                constraint.constraint_type.name()
            )
        }
    }

    /// Rank suggestions by priority and confidence
    pub fn rank_suggestions(
        &self,
        mut suggestions: Vec<ConstraintSuggestion>,
    ) -> Vec<ConstraintSuggestion> {
        suggestions.sort_by(|a, b| {
            // First by priority (descending)
            let priority_cmp = b.priority.cmp(&a.priority);
            if priority_cmp != std::cmp::Ordering::Equal {
                return priority_cmp;
            }

            // Then by confidence (descending)
            b.confidence
                .score()
                .partial_cmp(&a.confidence.score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        suggestions
    }
}

impl Default for SuggestionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint_generation::types::*;

    fn create_test_constraint() -> GeneratedConstraint {
        GeneratedConstraint {
            id: "test_1".to_string(),
            constraint_type: ConstraintType::Cardinality,
            target: NamedNode::new_unchecked("http://example.org/prop"),
            constraint: Constraint::Cardinality {
                min: Some(1),
                max: Some(1),
            },
            metadata: ConstraintMetadata {
                confidence: 0.9,
                support: 0.85,
                sample_count: 100,
                generation_method: "test".to_string(),
                generated_at: chrono::Utc::now(),
                evidence: vec![],
                counter_examples: 5,
            },
            quality: ConstraintQuality::calculate(0.9, 0.85),
        }
    }

    #[test]
    fn test_suggestion_engine_creation() {
        let engine = SuggestionEngine::new();
        assert_eq!(engine.min_confidence, 0.7);
    }

    #[test]
    fn test_generate_suggestions() {
        let engine = SuggestionEngine::new();
        let constraints = vec![create_test_constraint()];

        let suggestions = engine.generate_suggestions(constraints);
        assert_eq!(suggestions.len(), 1);
    }

    #[test]
    fn test_confidence_levels() {
        assert_eq!(
            SuggestionConfidence::from_score(0.96),
            SuggestionConfidence::VeryHigh
        );
        assert_eq!(
            SuggestionConfidence::from_score(0.90),
            SuggestionConfidence::High
        );
        assert_eq!(
            SuggestionConfidence::from_score(0.75),
            SuggestionConfidence::Medium
        );
        assert_eq!(
            SuggestionConfidence::from_score(0.60),
            SuggestionConfidence::Low
        );
        assert_eq!(
            SuggestionConfidence::from_score(0.40),
            SuggestionConfidence::VeryLow
        );
    }

    #[test]
    fn test_rank_suggestions() {
        let engine = SuggestionEngine::new();

        let constraints = vec![
            {
                let mut c = create_test_constraint();
                c.id = "c1".to_string();
                c.metadata.confidence = 0.7;
                c
            },
            {
                let mut c = create_test_constraint();
                c.id = "c2".to_string();
                c.metadata.confidence = 0.95;
                c
            },
        ];

        let suggestions = engine.generate_suggestions(constraints);
        let ranked = engine.rank_suggestions(suggestions);

        // Higher confidence should be ranked first
        assert_eq!(ranked[0].constraint.id, "c2");
    }
}
