//! Pattern structures and pattern-related data types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::*;

/// A discovered pattern with advanced metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedPattern {
    /// Pattern items (properties, classes, values)
    pub items: Vec<PatternItem>,

    /// Support count (absolute frequency)
    pub support_count: usize,

    /// Support ratio (relative frequency)
    pub support_ratio: f64,

    /// Confidence score
    pub confidence: f64,

    /// Lift measure (interest factor)
    pub lift: f64,

    /// Conviction measure
    pub conviction: f64,

    /// Quality score
    pub quality_score: f64,

    /// Pattern type classification
    pub pattern_type: PatternType,

    /// Temporal characteristics
    pub temporal_info: Option<TemporalPatternInfo>,

    /// Hierarchical level
    pub hierarchy_level: usize,

    /// Associated SHACL constraints
    pub suggested_constraints: Vec<SuggestedConstraint>,
}

/// Pattern item in a discovered pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternItem {
    /// Item type (property, class, value pattern)
    pub item_type: PatternItemType,

    /// URI or identifier
    pub identifier: String,

    /// Item role in pattern
    pub role: ItemRole,

    /// Frequency in pattern occurrences
    pub frequency: f64,
}

/// Temporal pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPatternInfo {
    /// Time series frequency
    pub frequency: f64,

    /// Seasonality indicators
    pub seasonality: Vec<SeasonalityComponent>,

    /// Trend direction
    pub trend: TrendDirection,

    /// Pattern stability over time
    pub stability_score: f64,
}

/// Seasonality component in temporal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityComponent {
    /// Period length (in time units)
    pub period: usize,

    /// Amplitude of seasonal effect
    pub amplitude: f64,

    /// Phase offset
    pub phase: f64,
}

/// Suggested SHACL constraint from pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedConstraint {
    /// Constraint type
    pub constraint_type: ConstraintType,

    /// Target path
    pub path: String,

    /// Constraint parameters
    pub parameters: HashMap<String, String>,

    /// Confidence in suggestion
    pub confidence: f64,

    /// Expected validation coverage
    pub coverage: f64,

    /// Severity recommendation
    pub severity: oxirs_shacl::Severity,
}