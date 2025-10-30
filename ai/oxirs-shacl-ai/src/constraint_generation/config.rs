//! Configuration for constraint generation

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Configuration for constraint generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintGenerationConfig {
    /// Enable constraint generation
    pub enabled: bool,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Minimum support threshold
    pub min_support: f64,
    /// Maximum constraints per property
    pub max_constraints_per_property: usize,
    /// Enabled constraint types
    pub enabled_types: HashSet<ConstraintType>,
    /// Enable conflict detection
    pub enable_conflict_detection: bool,
    /// Enable constraint validation
    pub enable_validation: bool,
    /// Enable ranking
    pub enable_ranking: bool,
    /// Minimum sample size for analysis
    pub min_sample_size: usize,
    /// Maximum violations allowed (as percentage)
    pub max_violation_rate: f64,
}

impl Default for ConstraintGenerationConfig {
    fn default() -> Self {
        let mut enabled_types = HashSet::new();
        enabled_types.insert(ConstraintType::Cardinality);
        enabled_types.insert(ConstraintType::Datatype);
        enabled_types.insert(ConstraintType::ValueRange);
        enabled_types.insert(ConstraintType::NodeKind);

        Self {
            enabled: true,
            min_confidence: 0.8,
            min_support: 0.7,
            max_constraints_per_property: 10,
            enabled_types,
            enable_conflict_detection: true,
            enable_validation: true,
            enable_ranking: true,
            min_sample_size: 10,
            max_violation_rate: 0.05, // 5% violations allowed
        }
    }
}

/// Constraint types for configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintType {
    Cardinality,
    Datatype,
    ValueRange,
    Pattern,
    NodeKind,
    Class,
    Relationship,
    Unique,
    Closed,
}

impl ConstraintType {
    pub fn all() -> Vec<Self> {
        vec![
            Self::Cardinality,
            Self::Datatype,
            Self::ValueRange,
            Self::Pattern,
            Self::NodeKind,
            Self::Class,
            Self::Relationship,
            Self::Unique,
            Self::Closed,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ConstraintGenerationConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_confidence, 0.8);
        assert!(config.enable_conflict_detection);
    }

    #[test]
    fn test_constraint_type_all() {
        let types = ConstraintType::all();
        assert_eq!(types.len(), 9);
    }
}
