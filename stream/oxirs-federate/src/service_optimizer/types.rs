//! Type definitions for service optimization
//!
//! This module contains all the core data structures used in the service optimization system.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Duration;

use crate::{
    planner::{FilterExpression, TriplePattern},
    ServiceCapability,
};

/// Optimized SERVICE clause with patterns and execution strategy
#[derive(Debug, Clone)]
pub struct OptimizedServiceClause {
    pub service_id: String,
    pub endpoint: String,
    pub patterns: Vec<TriplePattern>,
    pub filters: Vec<FilterExpression>,
    pub pushed_filters: Vec<FilterExpression>,
    pub strategy: ServiceExecutionStrategy,
    pub estimated_cost: f64,
    pub capabilities: HashSet<ServiceCapability>,
}

/// Service execution strategy configuration
#[derive(Debug, Clone)]
pub struct ServiceExecutionStrategy {
    pub use_values_binding: bool,
    pub stream_results: bool,
    pub use_subqueries: bool,
    pub batch_size: usize,
    pub timeout_ms: u64,
}

/// Execution strategy for multiple services
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionStrategy {
    Sequential,
    Parallel,
    Pipeline,
    Adaptive,
}

/// SERVICE clause representation
#[derive(Debug, Clone)]
pub struct ServiceClause {
    pub endpoint: Option<String>,
    pub patterns: Vec<TriplePattern>,
    pub filters: Vec<FilterExpression>,
    pub silent: bool,
}

/// Cross-service join information
#[derive(Debug, Clone)]
pub struct CrossServiceJoin {
    pub left_service: String,
    pub right_service: String,
    pub join_variables: Vec<String>,
    pub join_type: JoinType,
}

/// Type of join between services
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    LeftOuter,
    Optional,
}

/// Complete optimized query representation
#[derive(Debug, Clone)]
pub struct OptimizedQuery {
    pub services: Vec<OptimizedServiceClause>,
    pub global_filters: Vec<FilterExpression>,
    pub cross_service_joins: Vec<CrossServiceJoin>,
    pub execution_strategy: ExecutionStrategy,
    pub estimated_cost: f64,
}

/// Service optimizer configuration
#[derive(Debug, Clone)]
pub struct ServiceOptimizerConfig {
    pub enable_pattern_grouping: bool,
    pub enable_filter_pushdown: bool,
    pub enable_cost_estimation: bool,
    pub max_join_complexity: usize,
}

impl Default for ServiceOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_pattern_grouping: true,
            enable_filter_pushdown: true,
            enable_cost_estimation: true,
            max_join_complexity: 5,
        }
    }
}

/// Statistics cache for optimization
#[derive(Debug)]
pub struct StatisticsCache {
    service_stats: HashMap<String, ServiceStatistics>,
}

impl StatisticsCache {
    pub fn new() -> Self {
        Self {
            service_stats: HashMap::new(),
        }
    }
}

/// Statistics for a service
#[derive(Debug, Clone)]
pub struct ServiceStatistics {
    pub avg_response_time: Duration,
    pub total_requests: u64,
    pub error_rate: f64,
    pub data_freshness: DateTime<Utc>,
}

/// Pattern complexity levels for cost calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternComplexity {
    Simple,
    Medium,
    Complex,
}

/// Service capacity analysis result
#[derive(Debug, Clone)]
pub struct ServiceCapacityAnalysis {
    pub max_concurrent_queries: u32,
    pub current_utilization: f64,
    pub recommended_max_load: u32,
    pub bottleneck_factors: Vec<String>,
    pub scaling_suggestions: Vec<String>,
}

/// Cost optimization objectives
#[derive(Debug, Clone)]
pub struct CostObjectives {
    pub time_weight: f64,
    pub latency_weight: f64,
    pub resource_weight: f64,
    pub reliability_weight: f64,
    pub quality_weight: f64,
}

impl Default for CostObjectives {
    fn default() -> Self {
        Self {
            time_weight: 1.0,
            latency_weight: 0.8,
            resource_weight: 0.6,
            reliability_weight: 1.2,
            quality_weight: 0.5,
        }
    }
}

/// Multi-objective cost score
#[derive(Debug, Clone)]
pub struct CostScore {
    pub total_cost: f64,
    pub execution_time_cost: f64,
    pub network_latency_cost: f64,
    pub resource_usage_cost: f64,
    pub reliability_cost: f64,
    pub quality_score: f64,
}

/// Pattern features for ML-based estimation
#[derive(Debug, Clone)]
pub struct PatternFeatures {
    pub predicate_frequency: f64,
    pub subject_specificity: f64,
    pub object_specificity: f64,
    pub service_data_size_factor: f64,
}

/// Optimization weights for multi-objective selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationWeights {
    pub cost_weight: f64,
    pub quality_weight: f64,
    pub latency_weight: f64,
    pub reliability_weight: f64,
}

impl Default for OptimizationWeights {
    fn default() -> Self {
        Self {
            cost_weight: 0.4,
            quality_weight: 0.3,
            latency_weight: 0.2,
            reliability_weight: 0.1,
        }
    }
}

/// Service objective scores for multi-objective optimization
#[derive(Debug, Clone)]
pub struct ServiceObjectiveScore {
    pub service_id: String,
    pub cost_score: f64,
    pub quality_score: f64,
    pub latency_score: f64,
    pub reliability_score: f64,
    pub combined_score: f64,
}

/// Optimized service selection result
#[derive(Debug, Clone)]
pub struct OptimizedServiceSelection {
    pub pattern: TriplePattern,
    pub ranked_services: Vec<ServiceObjectiveScore>,
    pub optimization_metadata: OptimizationMetadata,
}

/// Metadata about the optimization process
#[derive(Debug, Clone)]
pub struct OptimizationMetadata {
    pub total_candidates: usize,
    pub algorithm_used: String,
    pub computation_time_ms: u64,
}

/// Service performance update for dynamic ranking
#[derive(Debug, Clone)]
pub struct ServicePerformanceUpdate {
    pub service_id: String,
    pub metrics: ServicePerformanceMetrics,
    pub previous_ranking: f64,
    pub timestamp: DateTime<Utc>,
}

/// Performance metrics for service updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePerformanceMetrics {
    pub response_time_ms: u64,
    pub success_rate: f64,
    pub error_rate: f64,
    pub throughput_qps: f64,
    pub cpu_utilization: Option<f64>,
    pub memory_utilization: Option<f64>,
}

/// Statistics cache for service optimization
#[derive(Debug, Clone)]
pub struct StatisticsCache {
    predicate_stats: HashMap<String, PredicateStatistics>,
    service_rankings: HashMap<String, f64>,
    performance_history: HashMap<String, Vec<ServicePerformanceMetrics>>,
    last_update: DateTime<Utc>,
}

/// Statistics for specific predicates
#[derive(Debug, Clone)]
pub struct PredicateStatistics {
    pub frequency: u64,
    pub avg_result_size: u64,
    pub selectivity: f64,
    pub last_updated: DateTime<Utc>,
}

impl StatisticsCache {
    pub fn new() -> Self {
        Self {
            predicate_stats: HashMap::new(),
            service_rankings: HashMap::new(),
            performance_history: HashMap::new(),
            last_update: Utc::now(),
        }
    }

    pub fn get_predicate_stats(&self, predicate: &str) -> Option<&PredicateStatistics> {
        self.predicate_stats.get(predicate)
    }

    pub fn update_service_performance(&mut self, service_id: &str, metrics: &ServicePerformanceMetrics) {
        self.performance_history
            .entry(service_id.to_string())
            .or_insert_with(Vec::new)
            .push(metrics.clone());
        
        // Keep only last 100 entries per service
        if let Some(history) = self.performance_history.get_mut(service_id) {
            if history.len() > 100 {
                history.remove(0);
            }
        }
        
        self.last_update = Utc::now();
    }

    pub fn update_service_ranking(&mut self, service_id: &str, ranking: f64) {
        self.service_rankings.insert(service_id.to_string(), ranking);
        self.last_update = Utc::now();
    }

    pub fn get_service_ranking(&self, service_id: &str) -> Option<f64> {
        self.service_rankings.get(service_id).copied()
    }
}

/// Configuration for SERVICE optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceOptimizerConfig {
    /// Enable pattern grouping
    pub enable_pattern_grouping: bool,

    /// Enable SERVICE clause merging
    pub enable_service_merging: bool,

    /// Maximum patterns for VALUES binding
    pub max_patterns_for_values: usize,

    /// Minimum patterns for subquery
    pub min_patterns_for_subquery: usize,

    /// Result size threshold for streaming
    pub streaming_threshold: usize,

    /// Default batch size
    pub default_batch_size: usize,

    /// Service timeout in milliseconds
    pub service_timeout_ms: u64,

    /// Enable statistics-based optimization
    pub enable_statistics: bool,
}

impl Default for ServiceOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_pattern_grouping: true,
            enable_service_merging: true,
            max_patterns_for_values: 10,
            min_patterns_for_subquery: 5,
            streaming_threshold: 10000,
            default_batch_size: 1000,
            service_timeout_ms: 30000,
            enable_statistics: true,
        }
    }
}

/// SERVICE clause representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceClause {
    /// Service endpoint or identifier
    pub endpoint: Option<String>,

    /// Patterns in the SERVICE clause
    pub patterns: Vec<TriplePattern>,

    /// Filters in the SERVICE clause
    pub filters: Vec<FilterExpression>,

    /// Whether SERVICE is SILENT
    pub silent: bool,
}

/// Optimized SERVICE clause
#[derive(Debug, Clone)]
pub struct OptimizedServiceClause {
    /// Resolved service ID
    pub service_id: String,

    /// Service endpoint
    pub endpoint: String,

    /// Optimized patterns
    pub patterns: Vec<TriplePattern>,

    /// Local filters
    pub filters: Vec<FilterExpression>,

    /// Filters pushed down from global scope
    pub pushed_filters: Vec<FilterExpression>,

    /// Execution strategy
    pub strategy: ServiceExecutionStrategy,

    /// Estimated execution cost
    pub estimated_cost: f64,

    /// Service capabilities
    pub capabilities: HashSet<ServiceCapability>,
}

/// Execution strategy for a SERVICE clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceExecutionStrategy {
    /// Use VALUES clause for bindings
    pub use_values_binding: bool,

    /// Stream results
    pub stream_results: bool,

    /// Use subqueries
    pub use_subqueries: bool,

    /// Batch size for processing
    pub batch_size: usize,

    /// Timeout in milliseconds
    pub timeout_ms: u64,
}

/// Cross-service join information
#[derive(Debug, Clone)]
pub struct CrossServiceJoin {
    /// Left service ID
    pub left_service: String,

    /// Right service ID
    pub right_service: String,

    /// Join variables
    pub join_variables: Vec<String>,

    /// Join type
    pub join_type: JoinType,

    /// Estimated selectivity
    pub estimated_selectivity: f64,
}

/// Join types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    Inner,
    LeftOuter,
    RightOuter,
    Full,
}

/// Overall execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStrategy {
    Sequential,
    Parallel,
    ParallelWithJoin,
    Adaptive,
}

/// Optimized query result
#[derive(Debug)]
pub struct OptimizedQuery {
    /// Optimized SERVICE clauses
    pub services: Vec<OptimizedServiceClause>,

    /// Remaining global filters
    pub global_filters: Vec<FilterExpression>,

    /// Cross-service joins
    pub cross_service_joins: Vec<CrossServiceJoin>,

    /// Overall execution strategy
    pub execution_strategy: ExecutionStrategy,

    /// Total estimated cost
    pub estimated_cost: f64,
}

/// Statistics cache for optimization decisions
#[derive(Debug)]
pub struct StatisticsCache {
    pub predicate_stats: parking_lot::RwLock<HashMap<String, PredicateStatistics>>,
}

impl StatisticsCache {
    pub fn new() -> Self {
        Self {
            predicate_stats: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    pub fn get_predicate_stats(&self, predicate: &str) -> Option<PredicateStatistics> {
        self.predicate_stats.read().get(predicate).cloned()
    }

    pub fn update_predicate_stats(&self, predicate: String, stats: PredicateStatistics) {
        self.predicate_stats.write().insert(predicate, stats);
    }

    pub fn update_service_stats(
        &self,
        _service_endpoint: &str,
        _stats: ServiceStatistics,
    ) -> Result<()> {
        // Placeholder implementation for service statistics
        // In a full implementation, this would update service-specific stats
        Ok(())
    }

    pub fn add_service_stats(&self, _service_endpoint: &str, _stats: ServiceStatistics) -> Result<()> {
        // Placeholder implementation for adding new service statistics
        // In a full implementation, this would add new service-specific stats
        Ok(())
    }

    pub fn get_all_service_stats(&self) -> HashMap<String, ServiceStatistics> {
        // Placeholder implementation for getting all service statistics
        // In a full implementation, this would return all service-specific stats
        HashMap::new()
    }

    pub fn update_global_rankings(&self, _rankings: Vec<ServiceRanking>) -> Result<()> {
        // Placeholder implementation for updating global rankings
        // In a full implementation, this would update the global ranking cache
        Ok(())
    }
}

/// Statistics for a predicate
#[derive(Debug, Clone)]
pub struct PredicateStatistics {
    /// Frequency of the predicate
    pub frequency: u64,

    /// Average number of objects per subject
    pub avg_objects_per_subject: f64,

    /// Average result size for queries with this predicate
    pub avg_result_size: u64,

    /// Selectivity estimate
    pub selectivity: f64,
}

/// Pattern coverage analysis result
#[derive(Debug, Clone)]
pub struct PatternCoverageAnalysis {
    pub total_patterns: usize,
    pub covered_patterns: usize,
    pub partially_covered_patterns: usize,
    pub uncovered_patterns: usize,
    pub overall_coverage_score: f64,
    pub coverage_quality: CoverageQuality,
    pub pattern_scores: Vec<f64>,
}

/// Coverage quality assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoverageQuality {
    Excellent,
    Good,
    Fair,
    Poor,
}

/// Service predicate scoring result
#[derive(Debug, Clone)]
pub struct ServicePredicateScore {
    pub service_id: String,
    pub predicate: String,
    pub affinity_score: f64,
    pub estimated_result_count: u64,
    pub confidence_level: ConfidenceLevel,
}

/// Confidence level for estimations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceLevel {
    Low,
    Medium,
    High,
}

/// Value range for range-based selection
#[derive(Debug, Clone)]
pub enum ValueRange {
    Numeric(f64, f64),
    Temporal(DateTime<Utc>, DateTime<Utc>),
    Geospatial(f64, f64), // Simplified lat/lon bounds
}

/// Range service match result
#[derive(Debug, Clone)]
pub struct RangeServiceMatch {
    pub service_id: String,
    pub predicate: String,
    pub range: ValueRange,
    pub coverage_score: f64,
    pub estimated_result_count: u64,
    pub overlap_type: RangeOverlapType,
}

/// Range overlap classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeOverlapType {
    Complete, // Query range fully contained in service range
    Partial,  // Query range partially overlaps service range
    None,     // No overlap between ranges
    Unknown,  // Cannot determine overlap
}

/// Service Bloom filter for membership testing
#[cfg(feature = "caching")]
pub struct ServiceBloomFilter {
    pub predicate_filter: bloom::BloomFilter,
    pub resource_filter: bloom::BloomFilter,
    pub last_updated: DateTime<Utc>,
    pub false_positive_rate: f64,
    pub estimated_elements: usize,
}

/// Bloom filter test result
#[derive(Debug, Clone)]
pub struct BloomFilterResult {
    pub membership_probability: f64,
    pub likely_matches: Vec<String>,
    pub false_positive_rate: f64,
    pub confidence_score: f64,
}

/// ML-based source prediction
#[derive(Debug, Clone)]
pub struct MLSourcePrediction {
    pub service_endpoint: String,
    pub confidence_score: f64,
    pub predicted_latency_ms: f64,
    pub predicted_success_rate: f64,
    pub feature_importance: HashMap<String, f64>,
    pub model_version: String,
}

/// Query context for ML predictions
#[derive(Debug, Clone)]
pub struct QueryContext {
    pub query_type: String,
    pub user_context: Option<String>,
    pub time_constraints: Option<Duration>,
    pub quality_requirements: Option<QualityRequirements>,
}

/// Quality requirements for query execution
#[derive(Debug, Clone)]
pub struct QualityRequirements {
    pub max_latency_ms: f64,
    pub min_success_rate: f64,
    pub completeness_threshold: f64,
}

/// Historical query data for ML training
#[derive(Debug, Clone)]
pub struct HistoricalQueryData {
    pub queries: Vec<HistoricalQuery>,
    pub last_updated: DateTime<Utc>,
}

/// Individual historical query record
#[derive(Debug, Clone)]
pub struct HistoricalQuery {
    pub id: String,
    pub features: QueryFeatures,
    pub service_performance: HashMap<String, OptimizerServicePerformance>,
    pub timestamp: DateTime<Utc>,
}

/// Service performance data for optimization
#[derive(Debug, Clone)]
pub struct OptimizerServicePerformance {
    pub avg_latency_ms: f64,
    pub success_rate: f64,
    pub result_quality: f64,
}

/// Query features for ML
#[derive(Debug, Clone)]
pub struct QueryFeatures {
    pub pattern_count: usize,
    pub predicate_distribution: HashMap<String, usize>,
    pub namespace_distribution: HashMap<String, usize>,
    pub pattern_type_distribution: HashMap<String, usize>,
    pub complexity_score: f64,
    pub selectivity_estimate: f64,
    pub has_joins: bool,
    pub query_type: String,
    pub timestamp: DateTime<Utc>,
}

/// Similar query for pattern matching
#[derive(Debug, Clone)]
pub struct SimilarQuery {
    pub query_id: String,
    pub similarity_score: f64,
    pub service_performance: HashMap<String, OptimizerServicePerformance>,
    pub execution_timestamp: DateTime<Utc>,
}

/// Query execution result for performance tracking
#[derive(Debug, Clone)]
pub struct QueryExecutionResult {
    pub execution_time_ms: f64,
    pub success: bool,
    pub result_count: usize,
    pub error_type: Option<String>,
    pub query_info: QueryInfo,
}

/// Query information for complexity calculation
#[derive(Debug, Clone)]
pub struct QueryInfo {
    pub pattern_count: usize,
    pub has_joins: bool,
    pub filter_count: usize,
}

/// Performance update for dynamic ranking
#[derive(Debug, Clone)]
pub struct PerformanceUpdate {
    pub timestamp: DateTime<Utc>,
    pub latency_ms: f64,
    pub success: bool,
    pub result_count: usize,
    pub error_type: Option<String>,
    pub query_complexity: f64,
}

/// Service statistics for ranking
#[derive(Debug, Clone)]
pub struct ServiceStatistics {
    pub endpoint: String,
    pub avg_latency_ms: f64,
    pub success_rate: f64,
    pub total_queries: usize,
    pub last_updated: DateTime<Utc>,
    pub quality_score: f64,
    pub reliability_trend: ReliabilityTrend,
}

/// Reliability trend enumeration
#[derive(Debug, Clone)]
pub enum ReliabilityTrend {
    Improving,
    Stable,
    Degrading,
}

/// Service ranking information
#[derive(Debug, Clone)]
pub struct ServiceRanking {
    pub endpoint: String,
    pub ranking_score: f64,
    pub ranking_factors: RankingFactors,
    pub last_updated: DateTime<Utc>,
}

/// Factors contributing to service ranking
#[derive(Debug, Clone)]
pub struct RankingFactors {
    pub latency_score: f64,
    pub reliability_score: f64,
    pub availability_score: f64,
    pub quality_score: f64,
    pub trend_score: f64,
}

/// Statistics cache extension trait
pub trait StatisticsCacheExt {
    fn get_service_stats(&self, endpoint: &str) -> Option<ServiceStatistics>;
    fn update_service_stats(&mut self, endpoint: &str, stats: ServiceStatistics) -> Result<()>;
    fn add_service_stats(&mut self, endpoint: &str, stats: ServiceStatistics) -> Result<()>;
    fn get_all_service_stats(&self) -> HashMap<String, ServiceStatistics>;
    fn update_global_rankings(&mut self, rankings: Vec<ServiceRanking>) -> Result<()>;
}

impl StatisticsCacheExt for StatisticsCache {
    fn get_service_stats(&self, _endpoint: &str) -> Option<ServiceStatistics> {
        // This would be implemented based on the actual StatisticsCache structure
        None // Placeholder
    }

    fn update_service_stats(&mut self, _endpoint: &str, _stats: ServiceStatistics) -> Result<()> {
        // This would be implemented based on the actual StatisticsCache structure
        Ok(()) // Placeholder
    }

    fn add_service_stats(&mut self, _endpoint: &str, _stats: ServiceStatistics) -> Result<()> {
        // This would be implemented based on the actual StatisticsCache structure
        Ok(()) // Placeholder
    }

    fn get_all_service_stats(&self) -> HashMap<String, ServiceStatistics> {
        // This would be implemented based on the actual StatisticsCache structure
        HashMap::new() // Placeholder
    }

    fn update_global_rankings(&mut self, _rankings: Vec<ServiceRanking>) -> Result<()> {
        // This would be implemented based on the actual StatisticsCache structure
        Ok(()) // Placeholder
    }
}