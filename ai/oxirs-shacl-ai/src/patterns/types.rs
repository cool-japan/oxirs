//! Type definitions for pattern analysis

use oxirs_core::model::{NamedNode, Triple};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pattern recognition and analysis result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pattern {
    /// Class usage pattern
    ClassUsage {
        id: String,
        class: NamedNode,
        instance_count: u32,
        support: f64,
        confidence: f64,
        pattern_type: PatternType,
    },

    /// Property usage pattern
    PropertyUsage {
        id: String,
        property: NamedNode,
        usage_count: u32,
        support: f64,
        confidence: f64,
        pattern_type: PatternType,
    },

    /// Datatype usage pattern
    Datatype {
        id: String,
        property: NamedNode,
        datatype: NamedNode,
        usage_count: u32,
        support: f64,
        confidence: f64,
        pattern_type: PatternType,
    },

    /// Cardinality pattern
    Cardinality {
        id: String,
        property: NamedNode,
        cardinality_type: CardinalityType,
        min_count: Option<u32>,
        max_count: Option<u32>,
        avg_count: f64,
        support: f64,
        confidence: f64,
        pattern_type: PatternType,
    },

    /// Hierarchy pattern
    Hierarchy {
        id: String,
        subclass: NamedNode,
        superclass: NamedNode,
        relationship_type: HierarchyType,
        depth: u32,
        support: f64,
        confidence: f64,
        pattern_type: PatternType,
    },

    /// Constraint usage pattern in shapes
    ConstraintUsage {
        id: String,
        constraint_type: String,
        usage_count: u32,
        support: f64,
        confidence: f64,
        pattern_type: PatternType,
    },

    /// Target usage pattern in shapes
    TargetUsage {
        id: String,
        target_type: String,
        usage_count: u32,
        support: f64,
        confidence: f64,
        pattern_type: PatternType,
    },

    /// Path complexity pattern
    PathComplexity {
        id: String,
        complexity: usize,
        usage_count: u32,
        support: f64,
        confidence: f64,
        pattern_type: PatternType,
    },

    /// Shape complexity pattern
    ShapeComplexity {
        id: String,
        constraint_count: usize,
        shape_count: u32,
        support: f64,
        confidence: f64,
        pattern_type: PatternType,
    },

    /// Association rule pattern
    AssociationRule {
        id: String,
        antecedent: String,
        consequent: String,
        support: f64,
        confidence: f64,
        lift: f64,
        pattern_type: PatternType,
    },

    /// Cardinality rule pattern
    CardinalityRule {
        id: String,
        property: NamedNode,
        rule_type: String,
        min_count: Option<u32>,
        max_count: Option<u32>,
        support: f64,
        confidence: f64,
        pattern_type: PatternType,
    },
}

impl Pattern {
    /// Get the support value for this pattern
    pub fn support(&self) -> f64 {
        match self {
            Pattern::ClassUsage { support, .. } => *support,
            Pattern::PropertyUsage { support, .. } => *support,
            Pattern::Datatype { support, .. } => *support,
            Pattern::Cardinality { support, .. } => *support,
            Pattern::Hierarchy { support, .. } => *support,
            Pattern::ConstraintUsage { support, .. } => *support,
            Pattern::TargetUsage { support, .. } => *support,
            Pattern::PathComplexity { support, .. } => *support,
            Pattern::ShapeComplexity { support, .. } => *support,
            Pattern::AssociationRule { support, .. } => *support,
            Pattern::CardinalityRule { support, .. } => *support,
        }
    }

    /// Get the confidence value for this pattern
    pub fn confidence(&self) -> f64 {
        match self {
            Pattern::ClassUsage { confidence, .. } => *confidence,
            Pattern::PropertyUsage { confidence, .. } => *confidence,
            Pattern::Datatype { confidence, .. } => *confidence,
            Pattern::Cardinality { confidence, .. } => *confidence,
            Pattern::Hierarchy { confidence, .. } => *confidence,
            Pattern::ConstraintUsage { confidence, .. } => *confidence,
            Pattern::TargetUsage { confidence, .. } => *confidence,
            Pattern::PathComplexity { confidence, .. } => *confidence,
            Pattern::ShapeComplexity { confidence, .. } => *confidence,
            Pattern::AssociationRule { confidence, .. } => *confidence,
            Pattern::CardinalityRule { confidence, .. } => *confidence,
        }
    }

    /// Get the pattern type
    pub fn pattern_type(&self) -> &PatternType {
        match self {
            Pattern::ClassUsage { pattern_type, .. } => pattern_type,
            Pattern::PropertyUsage { pattern_type, .. } => pattern_type,
            Pattern::Datatype { pattern_type, .. } => pattern_type,
            Pattern::Cardinality { pattern_type, .. } => pattern_type,
            Pattern::Hierarchy { pattern_type, .. } => pattern_type,
            Pattern::ConstraintUsage { pattern_type, .. } => pattern_type,
            Pattern::TargetUsage { pattern_type, .. } => pattern_type,
            Pattern::PathComplexity { pattern_type, .. } => pattern_type,
            Pattern::ShapeComplexity { pattern_type, .. } => pattern_type,
            Pattern::AssociationRule { pattern_type, .. } => pattern_type,
            Pattern::CardinalityRule { pattern_type, .. } => pattern_type,
        }
    }

    /// Get the id value for this pattern
    pub fn id(&self) -> &str {
        match self {
            Pattern::ClassUsage { id, .. } => id,
            Pattern::PropertyUsage { id, .. } => id,
            Pattern::Datatype { id, .. } => id,
            Pattern::Cardinality { id, .. } => id,
            Pattern::Hierarchy { id, .. } => id,
            Pattern::ConstraintUsage { id, .. } => id,
            Pattern::TargetUsage { id, .. } => id,
            Pattern::PathComplexity { id, .. } => id,
            Pattern::ShapeComplexity { id, .. } => id,
            Pattern::AssociationRule { id, .. } => id,
            Pattern::CardinalityRule { id, .. } => id,
        }
    }

    /// Create a new pattern with a different id
    pub fn with_id(self, new_id: String) -> Self {
        match self {
            Pattern::ClassUsage { class, instance_count, support, confidence, pattern_type, .. } => {
                Pattern::ClassUsage { id: new_id, class, instance_count, support, confidence, pattern_type }
            }
            Pattern::PropertyUsage { property, usage_count, support, confidence, pattern_type, .. } => {
                Pattern::PropertyUsage { id: new_id, property, usage_count, support, confidence, pattern_type }
            }
            Pattern::Datatype { property, datatype, usage_count, support, confidence, pattern_type, .. } => {
                Pattern::Datatype { id: new_id, property, datatype, usage_count, support, confidence, pattern_type }
            }
            Pattern::Cardinality { property, cardinality_type, min_count, max_count, avg_count, support, confidence, pattern_type, .. } => {
                Pattern::Cardinality { id: new_id, property, cardinality_type, min_count, max_count, avg_count, support, confidence, pattern_type }
            }
            Pattern::Hierarchy { subclass, superclass, relationship_type, depth, support, confidence, pattern_type, .. } => {
                Pattern::Hierarchy { id: new_id, subclass, superclass, relationship_type, depth, support, confidence, pattern_type }
            }
            Pattern::ConstraintUsage { constraint_type, usage_count, support, confidence, pattern_type, .. } => {
                Pattern::ConstraintUsage { id: new_id, constraint_type, usage_count, support, confidence, pattern_type }
            }
            Pattern::TargetUsage { target_type, usage_count, support, confidence, pattern_type, .. } => {
                Pattern::TargetUsage { id: new_id, target_type, usage_count, support, confidence, pattern_type }
            }
            Pattern::PathComplexity { complexity, usage_count, support, confidence, pattern_type, .. } => {
                Pattern::PathComplexity { id: new_id, complexity, usage_count, support, confidence, pattern_type }
            }
            Pattern::ShapeComplexity { constraint_count, shape_count, support, confidence, pattern_type, .. } => {
                Pattern::ShapeComplexity { id: new_id, constraint_count, shape_count, support, confidence, pattern_type }
            }
            Pattern::AssociationRule { antecedent, consequent, support, confidence, lift, pattern_type, .. } => {
                Pattern::AssociationRule { id: new_id, antecedent, consequent, support, confidence, lift, pattern_type }
            }
            Pattern::CardinalityRule { property, rule_type, min_count, max_count, support, confidence, pattern_type, .. } => {
                Pattern::CardinalityRule { id: new_id, property, rule_type, min_count, max_count, support, confidence, pattern_type }
            }
        }
    }
}

/// Pattern category type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    Structural,
    Usage,
    ShapeComposition,
    Temporal,
    Anomalous,
    Association,
    Constraint,
    Datatype,
    Cardinality,
    Range,
}

/// Cardinality pattern type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CardinalityType {
    Required,
    Optional,
    Functional,
    InverseFunctional,
}

/// Hierarchy relationship type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HierarchyType {
    SubClassOf,
    SubPropertyOf,
    InstanceOf,
}

/// Pattern similarity measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternSimilarity {
    pub pattern1: Pattern,
    pub pattern2: Pattern,
    pub similarity_score: f64,
    pub similarity_type: SimilarityType,
}

/// Type of similarity measurement
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimilarityType {
    Structural,
    Semantic,
    Statistical,
}

/// Statistics for pattern analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternStatistics {
    pub total_analyses: usize,
    pub shape_analyses: usize,
    pub total_analysis_time: std::time::Duration,
    pub patterns_discovered: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub model_trained: bool,
}

/// Pattern model state for training
#[derive(Debug)]
pub struct PatternModelState {
    pub version: String,
    pub accuracy: f64,
    pub loss: f64,
    pub training_epochs: usize,
    pub last_training: Option<chrono::DateTime<chrono::Utc>>,
}

impl PatternModelState {
    pub fn new() -> Self {
        Self {
            version: "1.0.0".to_string(),
            accuracy: 0.7,
            loss: 0.3,
            training_epochs: 0,
            last_training: None,
        }
    }
}

/// Cached pattern analysis result
#[derive(Debug, Clone)]
pub struct CachedPatternResult {
    pub patterns: Vec<Pattern>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub ttl: std::time::Duration,
}

impl CachedPatternResult {
    pub fn is_expired(&self) -> bool {
        let now = chrono::Utc::now();
        let expiry = self.timestamp + chrono::Duration::from_std(self.ttl).unwrap_or_default();
        now > expiry
    }
}

/// Training data for pattern recognition
#[derive(Debug, Clone)]
pub struct PatternTrainingData {
    pub examples: Vec<PatternExample>,
    pub validation_examples: Vec<PatternExample>,
}

/// Example for pattern training
#[derive(Debug, Clone)]
pub struct PatternExample {
    pub graph_data: Vec<Triple>,
    pub expected_patterns: Vec<Pattern>,
    pub pattern_labels: Vec<String>,
}
