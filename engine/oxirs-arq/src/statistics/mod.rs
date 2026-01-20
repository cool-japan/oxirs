//! Statistics and cardinality estimation for query optimization
//!
//! This module provides advanced statistical techniques for accurate query optimization.

pub mod cardinality_estimator;

pub use cardinality_estimator::{
    CardinalityEstimator, EstimationError, EstimationMethod, HistogramBucket, HyperLogLogSketch,
    PredicateHistogram, ReservoirSample,
};

/// Pattern statistics for query optimization
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PatternStatistics {
    /// Total number of triples matching this pattern
    pub count: u64,
    /// Number of distinct subjects
    pub distinct_subjects: u64,
    /// Number of distinct objects
    pub distinct_objects: u64,
    /// Selectivity factor (0.0 to 1.0)
    pub selectivity: f64,
}
