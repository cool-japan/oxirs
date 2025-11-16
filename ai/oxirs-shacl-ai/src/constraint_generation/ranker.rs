//! Constraint ranking and prioritization

use serde::{Deserialize, Serialize};

use super::types::{ConstraintType, GeneratedConstraint};

/// Ranked constraint with score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedConstraint {
    /// Original constraint
    pub constraint: GeneratedConstraint,
    /// Ranking score (0.0 - 1.0)
    pub rank_score: f64,
    /// Ranking reason
    pub ranking_reason: String,
}

/// Ranking criteria
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RankingCriteria {
    /// Weight for confidence
    pub confidence_weight: f64,
    /// Weight for support
    pub support_weight: f64,
    /// Weight for quality
    pub quality_weight: f64,
    /// Weight for constraint type priority
    pub type_priority_weight: f64,
}

impl Default for RankingCriteria {
    fn default() -> Self {
        Self {
            confidence_weight: 0.35,
            support_weight: 0.25,
            quality_weight: 0.30,
            type_priority_weight: 0.10,
        }
    }
}

/// Constraint ranker
pub struct ConstraintRanker {
    criteria: RankingCriteria,
}

impl ConstraintRanker {
    pub fn new(criteria: RankingCriteria) -> Self {
        Self { criteria }
    }

    pub fn with_default_criteria() -> Self {
        Self {
            criteria: RankingCriteria::default(),
        }
    }

    /// Rank a single constraint
    pub fn rank_constraint(&self, constraint: GeneratedConstraint) -> RankedConstraint {
        let rank_score = self.calculate_rank_score(&constraint);
        let ranking_reason = self.generate_ranking_reason(&constraint, rank_score);

        RankedConstraint {
            constraint,
            rank_score,
            ranking_reason,
        }
    }

    /// Rank multiple constraints
    pub fn rank_constraints(&self, constraints: Vec<GeneratedConstraint>) -> Vec<RankedConstraint> {
        let mut ranked: Vec<_> = constraints
            .into_iter()
            .map(|c| self.rank_constraint(c))
            .collect();

        // Sort by rank score (descending)
        ranked.sort_by(|a, b| {
            b.rank_score
                .partial_cmp(&a.rank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ranked
    }

    fn calculate_rank_score(&self, constraint: &GeneratedConstraint) -> f64 {
        let confidence_score = constraint.metadata.confidence * self.criteria.confidence_weight;
        let support_score = constraint.metadata.support * self.criteria.support_weight;
        let quality_score = constraint.quality.overall_score * self.criteria.quality_weight;

        // Normalize type priority (0-10) to 0-1
        let type_priority_score = (constraint.constraint_type.priority() as f64 / 10.0)
            * self.criteria.type_priority_weight;

        confidence_score + support_score + quality_score + type_priority_score
    }

    fn generate_ranking_reason(&self, constraint: &GeneratedConstraint, score: f64) -> String {
        if score >= 0.9 {
            format!(
                "Excellent constraint: High confidence ({:.1}%), strong support ({:.1}%), and high quality",
                constraint.metadata.confidence * 100.0,
                constraint.metadata.support * 100.0
            )
        } else if score >= 0.75 {
            format!(
                "Good constraint: Confidence {:.1}%, Support {:.1}%",
                constraint.metadata.confidence * 100.0,
                constraint.metadata.support * 100.0
            )
        } else if score >= 0.6 {
            format!(
                "Acceptable constraint: May need review (confidence {:.1}%)",
                constraint.metadata.confidence * 100.0
            )
        } else {
            "Low-priority constraint: Requires validation before use".to_string()
        }
    }

    /// Filter constraints by minimum rank score
    pub fn filter_by_rank(
        &self,
        ranked: Vec<RankedConstraint>,
        min_score: f64,
    ) -> Vec<RankedConstraint> {
        ranked
            .into_iter()
            .filter(|r| r.rank_score >= min_score)
            .collect()
    }

    /// Group constraints by type and rank within groups
    pub fn rank_by_type(
        &self,
        constraints: Vec<GeneratedConstraint>,
    ) -> Vec<Vec<RankedConstraint>> {
        use std::collections::HashMap;

        let mut groups: HashMap<ConstraintType, Vec<GeneratedConstraint>> = HashMap::new();

        for constraint in constraints {
            groups
                .entry(constraint.constraint_type)
                .or_default()
                .push(constraint);
        }

        groups
            .into_values()
            .map(|group| self.rank_constraints(group))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint_generation::types::*;
    use oxirs_core::model::NamedNode;

    fn create_test_constraint(confidence: f64, support: f64) -> GeneratedConstraint {
        GeneratedConstraint {
            id: format!("test_{}", uuid::Uuid::new_v4()),
            constraint_type: ConstraintType::Cardinality,
            target: NamedNode::new_unchecked("http://example.org/prop"),
            constraint: Constraint::Cardinality {
                min: Some(1),
                max: Some(1),
            },
            metadata: ConstraintMetadata {
                confidence,
                support,
                sample_count: 100,
                generation_method: "test".to_string(),
                generated_at: chrono::Utc::now(),
                evidence: vec![],
                counter_examples: 0,
            },
            quality: ConstraintQuality::calculate(confidence, support),
        }
    }

    #[test]
    fn test_ranker_creation() {
        let ranker = ConstraintRanker::with_default_criteria();
        assert_eq!(ranker.criteria.confidence_weight, 0.35);
    }

    #[test]
    fn test_rank_constraint() {
        let ranker = ConstraintRanker::with_default_criteria();
        let constraint = create_test_constraint(0.9, 0.85);

        let ranked = ranker.rank_constraint(constraint);
        assert!(ranked.rank_score > 0.7);
        assert!(!ranked.ranking_reason.is_empty());
    }

    #[test]
    fn test_rank_multiple_constraints() {
        let ranker = ConstraintRanker::with_default_criteria();
        let constraints = vec![
            create_test_constraint(0.7, 0.6),
            create_test_constraint(0.95, 0.9),
            create_test_constraint(0.8, 0.75),
        ];

        let ranked = ranker.rank_constraints(constraints);
        assert_eq!(ranked.len(), 3);

        // Should be sorted by rank score
        assert!(ranked[0].rank_score >= ranked[1].rank_score);
        assert!(ranked[1].rank_score >= ranked[2].rank_score);
    }

    #[test]
    fn test_filter_by_rank() {
        let ranker = ConstraintRanker::with_default_criteria();
        let constraints = vec![
            create_test_constraint(0.95, 0.9),
            create_test_constraint(0.6, 0.5),
        ];

        let ranked = ranker.rank_constraints(constraints);
        let filtered = ranker.filter_by_rank(ranked, 0.8);

        // Should only keep high-scoring constraint
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_custom_criteria() {
        let criteria = RankingCriteria {
            confidence_weight: 0.5,
            support_weight: 0.3,
            quality_weight: 0.15,
            type_priority_weight: 0.05,
        };

        let ranker = ConstraintRanker::new(criteria);
        let constraint = create_test_constraint(0.9, 0.8);

        let ranked = ranker.rank_constraint(constraint);
        assert!(ranked.rank_score > 0.0);
    }
}
