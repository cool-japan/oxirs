//! Core types for the adaptive intelligent caching system

use crate::{similarity::SimilarityMetric, Vector, VectorId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant, SystemTime};

#[derive(Debug, Clone, PartialEq)]
pub struct CacheKey {
    pub query_vector: Vec<u8>, // Hashed query vector
    pub similarity_metric: SimilarityMetric,
    pub parameters: HashMap<String, String>,
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.query_vector.hash(state);
        // Hash the similarity metric by its discriminant
        std::mem::discriminant(&self.similarity_metric).hash(state);
        // For parameters, we'll sort them to ensure consistent hashing
        let mut params: Vec<_> = self.parameters.iter().collect();
        params.sort_by_key(|(k, _)| *k);
        params.hash(state);
    }
}

impl Eq for CacheKey {}

#[derive(Debug, Clone)]
pub struct CacheValue {
    pub results: Vec<(VectorId, f32)>, // Search results with scores
    pub metadata: CacheMetadata,
    pub created_at: SystemTime,
    pub last_accessed: SystemTime,
    pub access_count: u32,
}

#[derive(Debug, Clone)]
pub struct CacheMetadata {
    pub size_bytes: u64,
    pub computation_cost: f64,
    pub quality_score: f64,
    pub staleness_factor: f64,
}

#[derive(Debug, Clone)]
pub struct AccessEvent {
    pub timestamp: SystemTime,
    pub key: CacheKey,
    pub hit: bool,
    pub latency_ns: u64,
    pub user_context: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PrefetchItem {
    pub key: CacheKey,
    pub priority: f64,
    pub predicted_access_time: SystemTime,
    pub confidence: f64,
    pub strategy: PrefetchStrategy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefetchStrategy {
    SequentialPattern,
    SeasonalPattern,
    SimilarityBased,
    UserBehavior,
    MachineLearning,
}

#[derive(Debug, Clone)]
pub struct CacheItem {
    pub key: CacheKey,
    pub value: CacheValue,
    pub tier_id: u32,
    pub last_access: Instant,
    pub access_frequency: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CacheSizeInfo {
    pub used_bytes: u64,
    pub available_bytes: u64,
    pub total_capacity_bytes: u64,
    pub item_count: u64,
}

#[derive(Debug, Clone, Default)]
pub struct StorageStatistics {
    pub read_operations: u64,
    pub write_operations: u64,
    pub delete_operations: u64,
    pub avg_operation_latency_ns: u64,
    pub fragmentation_ratio: f64,
}

#[derive(Debug, Clone, Default)]
pub struct EvictionStatistics {
    pub total_evictions: u64,
    pub false_evictions: u64, // Items evicted but accessed shortly after
    pub eviction_accuracy: f64,
    pub avg_item_lifetime: Duration,
}

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub improvement_score: f64,
    pub changes_applied: Vec<OptimizationChange>,
    pub estimated_impact: EstimatedImpact,
}

#[derive(Debug, Clone)]
pub struct OptimizationChange {
    pub change_type: OptimizationChangeType,
    pub description: String,
    pub tier_id: Option<u32>,
    pub old_value: String,
    pub new_value: String,
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizationChangeType {
    TierSizeAdjustment,
    EvictionPolicyChange,
    TTLAdjustment,
    PrefetchingStrategy,
    TierRebalancing,
}

#[derive(Debug, Clone, Default)]
pub struct EstimatedImpact {
    pub hit_rate_improvement: f64,
    pub latency_reduction: f64,
    pub memory_efficiency_gain: f64,
    pub cost_reduction: f64,
}

#[derive(Debug, Clone)]
pub struct TierConfiguration {
    pub max_size_bytes: u64,
    pub default_ttl: Duration,
    pub compression_enabled: bool,
    pub persistence_enabled: bool,
    pub replication_factor: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierStatistics {
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
    pub size_bytes: u64,
    pub item_count: u64,
    pub avg_access_time: Duration,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierMetrics {
    pub hit_rate: f64,
    pub utilization: f64,
    pub avg_item_size: u64,
    pub hotness_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalMetric {
    pub timestamp: SystemTime,
    pub hit_rate: f64,
    pub latency_p99: u64,
    pub throughput: f64,
    pub memory_usage: u64,
}

#[derive(Debug, Clone)]
pub struct AccessTracker {
    pub(crate) access_counts: HashMap<CacheKey, u64>,
    pub(crate) access_times: HashMap<CacheKey, VecDeque<SystemTime>>,
    pub(crate) hot_keys: BTreeMap<u64, CacheKey>, // Frequency -> Key
}

#[derive(Debug, Clone)]
pub struct SeasonalPatternDetector {
    pub(crate) hourly_patterns: [f64; 24],
    pub(crate) daily_patterns: [f64; 7],
    pub(crate) monthly_patterns: [f64; 31],
    pub(crate) pattern_confidence: f64,
}

#[derive(Debug, Clone)]
pub struct QueryClusteringEngine {
    pub(crate) clusters: Vec<QueryCluster>,
    pub(crate) cluster_assignments: HashMap<CacheKey, u32>,
    pub(crate) cluster_centroids: Vec<Vector>,
}

#[derive(Debug, Clone)]
pub struct QueryCluster {
    pub cluster_id: u32,
    pub centroid: Vector,
    pub members: Vec<CacheKey>,
    pub access_pattern: AccessPattern,
}

#[derive(Debug, Clone)]
pub struct AccessPattern {
    pub frequency: f64,
    pub seasonality: SeasonalityInfo,
    pub correlation_score: f64,
}

#[derive(Debug, Clone)]
pub struct SeasonalityInfo {
    pub has_daily_pattern: bool,
    pub has_weekly_pattern: bool,
    pub peak_hours: Vec<u8>,
    pub peak_days: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct TemporalAccessPredictor {
    pub(crate) time_series_models: HashMap<CacheKey, TimeSeriesModel>,
    pub(crate) global_trend_model: GlobalTrendModel,
    pub(crate) prediction_horizon: Duration,
}

#[derive(Debug, Clone)]
pub struct TimeSeriesModel {
    pub coefficients: Vec<f64>,
    pub seasonal_components: Vec<f64>,
    pub trend_component: f64,
    pub accuracy_score: f64,
}

#[derive(Debug, Clone)]
pub struct GlobalTrendModel {
    pub hourly_multipliers: [f64; 24],
    pub daily_multipliers: [f64; 7],
    pub base_rate: f64,
}

#[derive(Debug, Clone)]
pub struct PrefetchModels {
    pub similarity_model: SimilarityPrefetchModel,
    pub sequence_model: SequencePrefetchModel,
    pub user_behavior_model: UserBehaviorModel,
}

#[derive(Debug, Clone)]
pub struct SimilarityPrefetchModel {
    pub similarity_threshold: f64,
    pub prefetch_depth: u32,
    pub confidence_weights: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct SequencePrefetchModel {
    pub sequence_patterns: HashMap<Vec<CacheKey>, f64>, // Pattern -> Probability
    pub max_sequence_length: u32,
    pub min_confidence: f64,
}

#[derive(Debug, Clone)]
pub struct UserBehaviorModel {
    pub user_profiles: HashMap<String, UserProfile>,
    pub default_profile: UserProfile,
}

#[derive(Debug, Clone)]
pub struct UserProfile {
    pub typical_query_patterns: Vec<QueryPattern>,
    pub access_frequency: f64,
    pub preference_weights: HashMap<SimilarityMetric, f64>,
}

#[derive(Debug, Clone)]
pub struct QueryPattern {
    pub pattern_vector: Vector,
    pub frequency: f64,
    pub time_distribution: Vec<f64>, // Hourly distribution
}

#[derive(Debug, Clone)]
pub struct PrefetchPerformance {
    pub successful_prefetches: u64,
    pub failed_prefetches: u64,
    pub cache_space_saved: u64,
    pub avg_prediction_accuracy: f64,
}

#[derive(Debug, Clone)]
pub struct OptimizationEvent {
    pub timestamp: SystemTime,
    pub algorithm: String,
    pub changes: Vec<OptimizationChange>,
    pub before_metrics: super::metrics::CachePerformanceMetrics,
    pub after_metrics: Option<super::metrics::CachePerformanceMetrics>,
}

#[derive(Debug, Clone)]
pub struct OptimizationState {
    pub last_optimization: SystemTime,
    pub optimization_frequency: Duration,
    pub pending_optimizations: Vec<String>,
    pub optimization_backlog: u32,
}

#[derive(Debug, Clone)]
pub struct ImprovementTracker {
    pub baseline_metrics: super::metrics::CachePerformanceMetrics,
    pub current_improvement: f64,
    pub improvement_history: VecDeque<ImprovementPoint>,
    pub regression_detection: RegressionDetector,
}

#[derive(Debug, Clone)]
pub struct ImprovementPoint {
    pub timestamp: SystemTime,
    pub improvement_score: f64,
    pub optimization_applied: String,
}

#[derive(Debug, Clone)]
pub struct RegressionDetector {
    pub regression_threshold: f64,
    pub detection_window: Duration,
    pub recent_scores: VecDeque<f64>,
}

#[derive(Debug, Clone)]
pub struct AccessPredictionModel {
    pub model_weights: Vec<f64>,
    pub feature_extractors: Vec<FeatureExtractor>,
    pub prediction_accuracy: f64,
}

#[derive(Debug, Clone)]
pub struct HitProbabilityModel {
    pub probability_matrix: HashMap<(CacheKey, u32), f64>, // (Key, Tier) -> Probability
    pub model_confidence: f64,
    pub last_update: SystemTime,
}

#[derive(Debug, Clone)]
pub struct TierPlacementModel {
    pub placement_scores: HashMap<CacheKey, Vec<f64>>, // Key -> Tier scores
    pub optimization_objective: OptimizationObjective,
}

#[derive(Debug, Clone)]
pub struct EvictionTimingModel {
    pub survival_functions: HashMap<CacheKey, SurvivalFunction>,
    pub hazard_rates: HashMap<CacheKey, f64>,
}

#[derive(Debug, Clone)]
pub struct SurvivalFunction {
    pub time_points: Vec<Duration>,
    pub survival_probabilities: Vec<f64>,
}

#[derive(Debug, Clone)]
pub enum FeatureExtractor {
    AccessFrequency,
    RecencyScore,
    SizeMetric,
    ComputationCost,
    UserContext,
    TemporalFeatures,
}

#[derive(Debug, Clone, Copy)]
pub enum OptimizationObjective {
    MaximizeHitRate,
    MinimizeLatency,
    MaximizeThroughput,
    MinimizeCost,
    BalancedPerformance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub total_requests: u64,
    pub avg_hit_latency_ns: u64,
    pub avg_miss_latency_ns: u64,
    pub cache_efficiency: f64,
    pub memory_utilization: f64,
    pub tier_statistics: Vec<TierStatistics>,
    pub prefetch_statistics: PrefetchStatistics,
    pub optimization_statistics: OptimizationStatistics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefetchStatistics {
    pub successful_prefetches: u64,
    pub failed_prefetches: u64,
    pub prefetch_hit_rate: f64,
    pub avg_prediction_accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStatistics {
    pub total_optimizations: u64,
    pub successful_optimizations: u64,
    pub avg_improvement_score: f64,
    pub last_optimization: Option<SystemTime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePerformanceData {
    pub metrics: super::metrics::CachePerformanceMetrics,
    pub statistics: CacheStatistics,
    pub configuration: super::config::CacheConfiguration,
    pub access_patterns: String,      // JSON encoded patterns
    pub optimization_history: String, // JSON encoded history
}

#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    Json,
    Prometheus,
    Csv,
}
