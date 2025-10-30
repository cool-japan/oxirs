//! Core types for constraint generation

use oxirs_core::model::NamedNode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Generated constraint with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedConstraint {
    /// Constraint identifier
    pub id: String,
    /// Constraint type
    pub constraint_type: ConstraintType,
    /// Target property or class
    pub target: NamedNode,
    /// Constraint details
    pub constraint: Constraint,
    /// Metadata about generation
    pub metadata: ConstraintMetadata,
    /// Quality metrics
    pub quality: ConstraintQuality,
}

/// Types of constraints that can be generated
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum ConstraintType {
    /// Cardinality constraints (min/max count)
    Cardinality,
    /// Datatype constraints
    Datatype,
    /// Value range constraints
    ValueRange,
    /// Pattern constraints (regex)
    Pattern,
    /// Node kind constraints
    NodeKind,
    /// Class constraints
    Class,
    /// Relationship constraints
    Relationship,
    /// Unique value constraints
    Unique,
    /// Closed shape constraints
    Closed,
}

impl ConstraintType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cardinality => "Cardinality",
            Self::Datatype => "Datatype",
            Self::ValueRange => "ValueRange",
            Self::Pattern => "Pattern",
            Self::NodeKind => "NodeKind",
            Self::Class => "Class",
            Self::Relationship => "Relationship",
            Self::Unique => "Unique",
            Self::Closed => "Closed",
        }
    }

    pub fn priority(&self) -> u8 {
        match self {
            Self::Datatype => 10,
            Self::NodeKind => 9,
            Self::Class => 8,
            Self::Cardinality => 7,
            Self::ValueRange => 6,
            Self::Pattern => 5,
            Self::Unique => 4,
            Self::Relationship => 3,
            Self::Closed => 2,
        }
    }
}

/// Constraint specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Minimum and maximum cardinality
    Cardinality { min: Option<u32>, max: Option<u32> },
    /// Required datatype
    Datatype { datatype: String },
    /// Value range (for numeric types)
    ValueRange {
        min_inclusive: Option<f64>,
        max_inclusive: Option<f64>,
        min_exclusive: Option<f64>,
        max_exclusive: Option<f64>,
    },
    /// String pattern (regex)
    Pattern {
        pattern: String,
        flags: Option<String>,
    },
    /// Node kind (IRI, Literal, BlankNode)
    NodeKind { kind: NodeKindType },
    /// Required class
    Class { class: NamedNode },
    /// Relationship to another property
    Relationship {
        property: NamedNode,
        relationship_type: RelationshipType,
    },
    /// Unique values required
    Unique,
    /// Closed shape (no additional properties)
    Closed { ignored_properties: Vec<NamedNode> },
}

/// Node kind types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKindType {
    IRI,
    Literal,
    BlankNode,
    IRIOrLiteral,
    IRIOrBlankNode,
    BlankNodeOrLiteral,
}

/// Relationship types between properties
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipType {
    Equals,
    LessThan,
    LessThanOrEquals,
    Disjoint,
}

/// Metadata about constraint generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintMetadata {
    /// Confidence in this constraint (0.0 - 1.0)
    pub confidence: f64,
    /// Support (fraction of data supporting this constraint)
    pub support: f64,
    /// Number of samples analyzed
    pub sample_count: usize,
    /// Generation method used
    pub generation_method: String,
    /// Timestamp of generation
    pub generated_at: chrono::DateTime<chrono::Utc>,
    /// Evidence supporting this constraint
    pub evidence: Vec<String>,
    /// Counter-examples (violations found)
    pub counter_examples: usize,
}

/// Quality metrics for a generated constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintQuality {
    /// Precision (true positives / (true positives + false positives))
    pub precision: f64,
    /// Recall (true positives / (true positives + false negatives))
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// Specificity (true negatives / (true negatives + false positives))
    pub specificity: f64,
    /// Overall quality score (0.0 - 1.0)
    pub overall_score: f64,
}

impl ConstraintQuality {
    pub fn calculate(precision: f64, recall: f64) -> Self {
        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        let overall_score = (precision + recall + f1_score) / 3.0;

        Self {
            precision,
            recall,
            f1_score,
            specificity: 0.0, // To be calculated if negative samples available
            overall_score,
        }
    }

    pub fn is_high_quality(&self) -> bool {
        self.overall_score >= 0.8
    }
}

/// Constraint conflict detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintConflict {
    /// First conflicting constraint
    pub constraint1_id: String,
    /// Second conflicting constraint
    pub constraint2_id: String,
    /// Type of conflict
    pub conflict_type: ConflictType,
    /// Description of conflict
    pub description: String,
    /// Severity (0.0 = minor, 1.0 = severe)
    pub severity: f64,
    /// Suggested resolution
    pub resolution: ConflictResolution,
}

/// Types of constraint conflicts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictType {
    /// Constraints are mutually exclusive
    Contradiction,
    /// One constraint subsumes the other
    Subsumption,
    /// Constraints overlap but differ
    Overlap,
    /// Constraints are redundant
    Redundancy,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Keep the constraint with higher confidence
    KeepHigherConfidence,
    /// Merge both constraints
    Merge { merged_constraint: String },
    /// Keep more specific constraint
    KeepMoreSpecific,
    /// Manual review required
    ManualReview { reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_type_priority() {
        assert!(ConstraintType::Datatype.priority() > ConstraintType::Cardinality.priority());
        assert!(ConstraintType::NodeKind.priority() > ConstraintType::Pattern.priority());
    }

    #[test]
    fn test_constraint_quality_calculation() {
        let quality = ConstraintQuality::calculate(0.9, 0.8);
        assert!(quality.f1_score > 0.0);
        assert!(quality.overall_score > 0.0);
        assert!(quality.is_high_quality());
    }

    #[test]
    fn test_node_kind_types() {
        assert_eq!(
            std::mem::discriminant(&NodeKindType::IRI),
            std::mem::discriminant(&NodeKindType::IRI)
        );
    }
}
