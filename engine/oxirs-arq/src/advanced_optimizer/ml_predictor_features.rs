//! Feature extraction and histogram-based statistics for the ML predictor.
//!
//! This module contains the value histogram infrastructure used for cardinality
//! estimation, the cardinality estimator that builds histograms per feature, and
//! the supporting data types (`QueryCharacteristics`, `FeatureExtractor`,
//! `NormalizationParams`).
//!
//! These types are split from `ml_predictor.rs` to keep that file (and its
//! sibling units) below the workspace 2000-line refactor threshold while
//! preserving the public API surface re-exported through
//! `super::ml_predictor`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::ml_predictor_training::TrainingExample;

// ---------------------------------------------------------------------------
// Value histogram
// ---------------------------------------------------------------------------

/// Value histogram for cardinality estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueHistogram {
    /// Histogram buckets (value ranges)
    pub buckets: Vec<HistogramBucket>,
    /// Total number of values
    pub total_count: usize,
    /// Number of distinct values
    pub distinct_count: usize,
    /// Min value seen
    pub min_value: f64,
    /// Max value seen
    pub max_value: f64,
}

/// A single histogram bucket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    /// Lower bound of this bucket (inclusive)
    pub lower_bound: f64,
    /// Upper bound of this bucket (exclusive)
    pub upper_bound: f64,
    /// Number of values in this bucket
    pub count: usize,
    /// Number of distinct values in this bucket
    pub distinct_count: usize,
}

impl ValueHistogram {
    /// Create a new histogram from data with specified number of buckets
    pub fn from_data(data: &[f64], num_buckets: usize) -> Self {
        if data.is_empty() {
            return Self::empty();
        }

        let min_value = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max_value = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        if (max_value - min_value).abs() < 1e-10 {
            // All values are the same - add epsilon to upper_bound to include the value
            return Self {
                buckets: vec![HistogramBucket {
                    lower_bound: min_value,
                    upper_bound: max_value + 1e-10,
                    count: data.len(),
                    distinct_count: 1,
                }],
                total_count: data.len(),
                distinct_count: 1,
                min_value,
                max_value,
            };
        }

        let bucket_width = (max_value - min_value) / num_buckets as f64;
        let mut buckets = Vec::with_capacity(num_buckets);

        for i in 0..num_buckets {
            let lower = min_value + i as f64 * bucket_width;
            let upper = if i == num_buckets - 1 {
                max_value + 1e-10 // Include max value in last bucket
            } else {
                min_value + (i + 1) as f64 * bucket_width
            };

            buckets.push(HistogramBucket {
                lower_bound: lower,
                upper_bound: upper,
                count: 0,
                distinct_count: 0,
            });
        }

        // Fill buckets
        use std::collections::HashSet;
        let mut bucket_distinct: Vec<HashSet<u64>> = vec![HashSet::new(); num_buckets];

        for &value in data {
            let bucket_idx = if value >= max_value {
                num_buckets - 1
            } else {
                ((value - min_value) / bucket_width).floor() as usize
            };

            if bucket_idx < num_buckets {
                buckets[bucket_idx].count += 1;
                // Use integer representation for distinct counting
                let value_bits = value.to_bits();
                bucket_distinct[bucket_idx].insert(value_bits);
            }
        }

        // Update distinct counts
        for (i, bucket) in buckets.iter_mut().enumerate() {
            bucket.distinct_count = bucket_distinct[i].len();
        }

        let distinct_count = bucket_distinct.iter().map(|s| s.len()).sum();

        Self {
            buckets,
            total_count: data.len(),
            distinct_count,
            min_value,
            max_value,
        }
    }

    /// Create an empty histogram
    pub fn empty() -> Self {
        Self {
            buckets: Vec::new(),
            total_count: 0,
            distinct_count: 0,
            min_value: 0.0,
            max_value: 0.0,
        }
    }

    /// Estimate selectivity for equality predicate (value = x)
    pub fn estimate_equality_selectivity(&self, value: f64) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        // Find the bucket containing this value
        for bucket in &self.buckets {
            if value >= bucket.lower_bound && value < bucket.upper_bound {
                if bucket.distinct_count == 0 {
                    return 0.0;
                }
                // Assume uniform distribution within bucket
                let selectivity =
                    bucket.count as f64 / bucket.distinct_count as f64 / self.total_count as f64;
                return selectivity.min(1.0);
            }
        }

        // Value not in histogram range
        0.0
    }

    /// Estimate selectivity for range predicate (lower <= value < upper)
    pub fn estimate_range_selectivity(&self, lower: f64, upper: f64) -> f64 {
        if self.total_count == 0 || upper <= lower {
            return 0.0;
        }

        let mut selected_count = 0.0;

        for bucket in &self.buckets {
            // Check overlap between bucket and query range
            let overlap_lower = bucket.lower_bound.max(lower);
            let overlap_upper = bucket.upper_bound.min(upper);

            if overlap_upper > overlap_lower {
                let bucket_width = bucket.upper_bound - bucket.lower_bound;
                let overlap_width = overlap_upper - overlap_lower;

                if bucket_width > 1e-10 {
                    // Fraction of bucket that overlaps with query range
                    let fraction = overlap_width / bucket_width;
                    selected_count += bucket.count as f64 * fraction;
                }
            }
        }

        (selected_count / self.total_count as f64).min(1.0)
    }

    /// Estimate cardinality for a predicate
    pub fn estimate_cardinality(&self, selectivity: f64) -> usize {
        (self.total_count as f64 * selectivity).ceil() as usize
    }

    /// Get average bucket size
    pub fn avg_bucket_size(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }
        self.total_count as f64 / self.buckets.len() as f64
    }
}

// ---------------------------------------------------------------------------
// Histogram-based cardinality estimator
// ---------------------------------------------------------------------------

/// Histogram-based cardinality estimator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramCardinalityEstimator {
    /// Histograms for different features (indexed by feature name)
    pub histograms: HashMap<String, ValueHistogram>,
    /// Configuration
    pub config: HistogramConfig,
}

/// Configuration for histogram-based estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramConfig {
    /// Number of buckets per histogram
    pub num_buckets: usize,
    /// Enable histogram-based estimation
    pub enabled: bool,
}

impl Default for HistogramConfig {
    fn default() -> Self {
        Self {
            num_buckets: 100,
            enabled: true,
        }
    }
}

impl HistogramCardinalityEstimator {
    /// Create a new histogram estimator
    pub fn new(config: HistogramConfig) -> Self {
        Self {
            histograms: HashMap::new(),
            config,
        }
    }

    /// Build histograms from training data
    pub fn build_from_training_data(&mut self, training_data: &[TrainingExample]) {
        if !self.config.enabled || training_data.is_empty() {
            return;
        }

        let n_features = training_data[0].features.len();

        // Build histogram for each feature
        for feature_idx in 0..n_features {
            let feature_values: Vec<f64> = training_data
                .iter()
                .map(|example| example.features.get(feature_idx).copied().unwrap_or(0.0))
                .collect();

            let histogram = ValueHistogram::from_data(&feature_values, self.config.num_buckets);
            let feature_name = format!("feature_{}", feature_idx);
            self.histograms.insert(feature_name, histogram);
        }
    }

    /// Estimate cardinality using histogram-based selectivity
    pub fn estimate_cardinality_with_histogram(&self, features: &[f64]) -> Option<f64> {
        if !self.config.enabled || self.histograms.is_empty() {
            return None;
        }

        // Use geometric mean of selectivities (assumes independence)
        let mut product = 1.0;
        let mut count = 0;

        for (i, &feature_value) in features.iter().enumerate() {
            let feature_name = format!("feature_{}", i);
            if let Some(histogram) = self.histograms.get(&feature_name) {
                let selectivity = histogram.estimate_equality_selectivity(feature_value);
                if selectivity > 0.0 {
                    product *= selectivity;
                    count += 1;
                }
            }
        }

        if count > 0 {
            let geometric_mean = product.powf(1.0 / count as f64);
            // Estimate based on average total count across histograms
            let avg_total: f64 = self
                .histograms
                .values()
                .map(|h| h.total_count as f64)
                .sum::<f64>()
                / self.histograms.len() as f64;
            Some(avg_total * geometric_mean)
        } else {
            None
        }
    }

    /// Get histogram statistics
    pub fn get_statistics(&self) -> HistogramStatistics {
        let total_buckets: usize = self.histograms.values().map(|h| h.buckets.len()).sum();
        let avg_distinct: f64 = if !self.histograms.is_empty() {
            self.histograms
                .values()
                .map(|h| h.distinct_count as f64)
                .sum::<f64>()
                / self.histograms.len() as f64
        } else {
            0.0
        };

        HistogramStatistics {
            num_histograms: self.histograms.len(),
            total_buckets,
            avg_distinct_values: avg_distinct,
        }
    }
}

/// Statistics about histogram estimator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramStatistics {
    pub num_histograms: usize,
    pub total_buckets: usize,
    pub avg_distinct_values: f64,
}

// ---------------------------------------------------------------------------
// Query characteristics & feature extractor state
// ---------------------------------------------------------------------------

/// Query characteristics for feature extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCharacteristics {
    pub triple_pattern_count: usize,
    pub join_count: usize,
    pub filter_count: usize,
    pub optional_count: usize,
    pub has_aggregation: bool,
    pub has_sorting: bool,
    pub estimated_cardinality: usize,
    pub complexity_score: f64,
    pub query_graph_diameter: usize,
    pub avg_degree: f64,
    pub max_degree: usize,
}

/// Feature extractor for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractor {
    pub(super) feature_weights: HashMap<String, f64>,
    pub(super) normalization_params: Option<NormalizationParams>,
}

impl FeatureExtractor {
    /// Create a new feature extractor with empty weights and no normalization.
    pub fn new() -> Self {
        Self {
            feature_weights: HashMap::new(),
            normalization_params: None,
        }
    }
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

/// Normalization parameters for features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    pub(super) mean: Vec<f64>,
    pub(super) std_dev: Vec<f64>,
    pub(super) min_values: Vec<f64>,
    pub(super) max_values: Vec<f64>,
}

impl NormalizationParams {
    /// Create a new set of normalization parameters from per-feature statistics.
    pub fn new(
        mean: Vec<f64>,
        std_dev: Vec<f64>,
        min_values: Vec<f64>,
        max_values: Vec<f64>,
    ) -> Self {
        Self {
            mean,
            std_dev,
            min_values,
            max_values,
        }
    }

    /// Number of features described.
    pub fn len(&self) -> usize {
        self.mean.len()
    }

    /// Whether the parameter set is empty.
    pub fn is_empty(&self) -> bool {
        self.mean.is_empty()
    }

    /// Per-feature means.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Per-feature standard deviations.
    pub fn std_dev(&self) -> &[f64] {
        &self.std_dev
    }

    /// Per-feature minimums (kept for diagnostics).
    pub fn min_values(&self) -> &[f64] {
        &self.min_values
    }

    /// Per-feature maximums (kept for diagnostics).
    pub fn max_values(&self) -> &[f64] {
        &self.max_values
    }
}
