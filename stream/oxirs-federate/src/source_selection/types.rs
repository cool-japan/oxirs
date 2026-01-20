//! Source selection type definitions
//!
//! This module contains all type definitions for source selection including:
//! - Triple patterns and constraints
//! - Service statistics and indexes
//! - Configuration and results

#[cfg(feature = "caching")]
use bloom::BloomFilter;

#[cfg(not(feature = "caching"))]
mod cache_stubs {
    #[derive(Debug, Clone)]
    pub struct BloomFilter;

    impl BloomFilter {
        pub fn with_rate(_rate: f64, _capacity: u32) -> Self {
            Self
        }

        pub fn insert<T>(&mut self, _item: &T) {}
        pub fn contains<T>(&self, _item: &T) -> bool {
            false
        }
    }

    #[derive(Debug, Clone)]
    pub struct ASMS;
}

#[cfg(not(feature = "caching"))]
use cache_stubs::{BloomFilter, ASMS};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Triple pattern for SPARQL queries
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TriplePattern {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub graph: Option<String>,
}

/// Range constraint for numeric or temporal values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeConstraint {
    pub field: String,
    pub min_value: Option<String>,
    pub max_value: Option<String>,
    pub data_type: RangeDataType,
}

/// Supported data types for range constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RangeDataType {
    Integer,
    Float,
    DateTime,
    String,
    Uri,
}

/// Comprehensive source selection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSelectionConfig {
    pub enable_pattern_coverage: bool,
    pub enable_predicate_filtering: bool,
    pub enable_range_selection: bool,
    pub enable_ml_prediction: bool,
    pub coverage_threshold: f64,
    pub bloom_filter_capacity: usize,
    pub bloom_filter_fp_rate: f64,
    pub ml_confidence_threshold: f64,
    pub max_sources_per_pattern: usize,
}

impl Default for SourceSelectionConfig {
    fn default() -> Self {
        Self {
            enable_pattern_coverage: true,
            enable_predicate_filtering: true,
            enable_range_selection: true,
            enable_ml_prediction: true,
            coverage_threshold: 0.8,
            bloom_filter_capacity: 100000,
            bloom_filter_fp_rate: 0.01,
            ml_confidence_threshold: 0.7,
            max_sources_per_pattern: 10,
        }
    }
}

/// Advanced source selector with multiple algorithms
pub struct AdvancedSourceSelector {
    pub(crate) config: SourceSelectionConfig,
    pub(crate) pattern_analyzer: PatternCoverageAnalyzer,
    pub(crate) predicate_filter: PredicateBasedFilter,
    pub(crate) range_selector: RangeBasedSelector,
    pub(crate) ml_predictor: Option<MLSourcePredictor>,
    pub(crate) statistics: Arc<RwLock<SelectionStatistics>>,
}

/// Pattern coverage analyzer for triple pattern analysis
pub struct PatternCoverageAnalyzer {
    pub(crate) coverage_cache: Arc<RwLock<HashMap<String, PatternCoverageResult>>>,
    pub(crate) service_statistics: Arc<RwLock<HashMap<String, ServiceStatistics>>>,
}

/// Predicate-based filter using Bloom filters
pub struct PredicateBasedFilter {
    pub(crate) service_filters: Arc<RwLock<HashMap<String, ServiceBloomFilters>>>,
    pub(crate) last_update: Arc<RwLock<DateTime<Utc>>>,
}

/// Range-based selector for numeric and temporal constraints
pub struct RangeBasedSelector {
    pub(crate) range_indices: Arc<RwLock<HashMap<String, ServiceRangeIndex>>>,
    pub(crate) temporal_indices: Arc<RwLock<HashMap<String, ServiceTemporalIndex>>>,
}

/// ML-based source predictor
pub struct MLSourcePredictor {
    pub(crate) training_data: Vec<SourcePredictionSample>,
    pub(crate) feature_weights: HashMap<String, f64>,
    pub(crate) prediction_cache: Arc<RwLock<HashMap<String, PredictionResult>>>,
    #[allow(dead_code)]
    pub(crate) model_accuracy: f64,
}

/// Service statistics for coverage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStatistics {
    pub total_triples: u64,
    pub unique_predicates: u64,
    pub unique_subjects: u64,
    pub unique_objects: u64,
    pub predicate_frequency: HashMap<String, u64>,
    pub subject_frequency: HashMap<String, u64>,
    pub object_frequency: HashMap<String, u64>,
    pub last_updated: DateTime<Utc>,
    pub data_quality_score: f64,
}

/// Bloom filters for service predicate filtering
pub struct ServiceBloomFilters {
    pub predicate_filter: BloomFilter,
    pub subject_filter: BloomFilter,
    pub object_filter: BloomFilter,
    pub type_filter: BloomFilter,
    pub last_updated: DateTime<Utc>,
    pub estimated_elements: usize,
}

impl std::fmt::Debug for ServiceBloomFilters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServiceBloomFilters")
            .field("last_updated", &self.last_updated)
            .field("estimated_elements", &self.estimated_elements)
            .finish()
    }
}

impl Clone for ServiceBloomFilters {
    fn clone(&self) -> Self {
        let capacity = self.estimated_elements.max(1000) as u32;
        Self {
            predicate_filter: BloomFilter::with_rate(0.01, capacity),
            subject_filter: BloomFilter::with_rate(0.01, capacity),
            object_filter: BloomFilter::with_rate(0.01, capacity),
            type_filter: BloomFilter::with_rate(0.01, capacity),
            last_updated: self.last_updated,
            estimated_elements: self.estimated_elements,
        }
    }
}

/// Range index for a service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceRangeIndex {
    pub numeric_ranges: HashMap<String, NumericRange>,
    pub string_ranges: HashMap<String, StringRange>,
    pub uri_patterns: HashMap<String, UriPattern>,
    pub last_updated: DateTime<Utc>,
}

/// Temporal index for a service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceTemporalIndex {
    pub datetime_ranges: HashMap<String, DateTimeRange>,
    pub year_ranges: HashMap<String, YearRange>,
    pub temporal_patterns: HashMap<String, TemporalPattern>,
    pub last_updated: DateTime<Utc>,
}

/// Numeric range information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericRange {
    pub min_value: f64,
    pub max_value: f64,
    pub count: u64,
    pub sample_values: Vec<f64>,
}

/// String range information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StringRange {
    pub min_length: usize,
    pub max_length: usize,
    pub common_prefixes: Vec<String>,
    pub sample_values: Vec<String>,
}

/// URI pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UriPattern {
    pub base_uris: Vec<String>,
    pub path_patterns: Vec<String>,
    pub namespace_prefixes: Vec<String>,
}

/// DateTime range information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateTimeRange {
    pub earliest: DateTime<Utc>,
    pub latest: DateTime<Utc>,
    pub count: u64,
    pub granularity: TemporalGranularity,
}

/// Year range information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YearRange {
    pub earliest_year: i32,
    pub latest_year: i32,
    pub year_distribution: HashMap<i32, u64>,
}

/// Temporal pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    pub pattern_type: TemporalPatternType,
    pub frequency: u64,
    pub confidence: f64,
}

/// Temporal granularity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalGranularity {
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
}

/// Types of temporal patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalPatternType {
    Sequential,
    Periodic,
    Clustered,
    Random,
}

/// ML training sample for source prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourcePredictionSample {
    pub query_features: QueryFeatures,
    pub selected_sources: Vec<String>,
    pub actual_performance: PerformanceMetrics,
    pub timestamp: DateTime<Utc>,
}

/// Query features for ML prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    pub pattern_count: usize,
    pub variable_count: usize,
    pub predicate_types: Vec<String>,
    pub has_ranges: bool,
    pub has_joins: bool,
    pub complexity_score: f64,
    pub selectivity_estimate: f64,
}

/// Performance metrics for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub execution_time_ms: u64,
    pub result_count: u64,
    pub data_transfer_bytes: u64,
    pub success_rate: f64,
}

/// Pattern coverage analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternCoverageResult {
    pub pattern: TriplePattern,
    pub total_sources: usize,
    pub covering_sources: Vec<SourceCoverage>,
    pub coverage_score: f64,
    pub confidence: f64,
    pub estimated_result_size: u64,
}

/// Coverage information for a specific source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceCoverage {
    pub service_endpoint: String,
    pub coverage_score: f64,
    pub selectivity: f64,
    pub estimated_results: u64,
    pub data_quality: f64,
    pub response_time_estimate: u64,
}

/// Selection statistics for monitoring
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SelectionStatistics {
    pub total_selections: u64,
    pub pattern_coverage_hits: u64,
    pub predicate_filter_hits: u64,
    pub range_selection_hits: u64,
    pub ml_prediction_hits: u64,
    pub average_sources_per_query: f64,
    pub selection_accuracy: f64,
    pub last_updated: Option<DateTime<Utc>>,
}

/// ML prediction result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub recommended_sources: Vec<String>,
    pub confidence_scores: HashMap<String, f64>,
    pub predicted_performance: HashMap<String, f64>,
    pub feature_importance: HashMap<String, f64>,
}

/// Comprehensive source selection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSelectionResult {
    pub selected_sources: Vec<String>,
    pub selection_reasons: HashMap<String, Vec<String>>,
    pub confidence_scores: HashMap<String, f64>,
    pub estimated_performance: HashMap<String, PerformanceMetrics>,
    pub selection_method: SelectionMethod,
    pub fallback_sources: Vec<String>,
}

/// Method used for source selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionMethod {
    PatternCoverage,
    PredicateFiltering,
    RangeBased,
    MLPrediction,
    Hybrid,
    Fallback,
}
