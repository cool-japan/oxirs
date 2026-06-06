//! ML Optimizer — Type definitions
//!
//! Structs, enums, optimizer config, ML model types, and feature types
//! used throughout the ML-driven query optimization subsystem.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

/// ML model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    /// Enable performance prediction
    pub enable_performance_prediction: bool,
    /// Enable source selection learning
    pub enable_source_selection_learning: bool,
    /// Enable join order optimization
    pub enable_join_order_optimization: bool,
    /// Enable caching strategy learning
    pub enable_caching_strategy_learning: bool,
    /// Enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Model training interval
    pub training_interval: Duration,
    /// Feature history size
    pub feature_history_size: usize,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Regularization parameter
    pub regularization: f64,
    /// Confidence threshold
    pub confidence_threshold: f64,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            enable_performance_prediction: true,
            enable_source_selection_learning: true,
            enable_join_order_optimization: true,
            enable_caching_strategy_learning: true,
            enable_anomaly_detection: true,
            training_interval: Duration::from_secs(3600), // 1 hour
            feature_history_size: 10000,
            learning_rate: 0.01,
            regularization: 0.001,
            confidence_threshold: 0.7,
        }
    }
}

/// Query features for ML training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    /// Number of triple patterns
    pub pattern_count: usize,
    /// Number of joins
    pub join_count: usize,
    /// Number of filters
    pub filter_count: usize,
    /// Query complexity score
    pub complexity_score: f64,
    /// Estimated selectivity
    pub selectivity: f64,
    /// Number of services involved
    pub service_count: usize,
    /// Average service latency
    pub avg_service_latency: f64,
    /// Data size estimate
    pub data_size_estimate: u64,
    /// Query depth (nested patterns)
    pub query_depth: usize,
    /// Has optional patterns
    pub has_optional: bool,
    /// Has union patterns
    pub has_union: bool,
    /// Has aggregation
    pub has_aggregation: bool,
    /// Variable count
    pub variable_count: usize,
}

/// Performance outcome for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOutcome {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Network I/O time
    pub network_io_ms: f64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Success rate
    pub success_rate: f64,
    /// Error count
    pub error_count: u32,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Timestamp
    pub timestamp: SystemTime,
}

impl Default for PerformanceOutcome {
    fn default() -> Self {
        Self {
            execution_time_ms: 0.0,
            memory_usage_bytes: 0,
            network_io_ms: 0.0,
            cpu_usage_percent: 0.0,
            success_rate: 1.0,
            error_count: 0,
            cache_hit_rate: 0.0,
            timestamp: SystemTime::now(),
        }
    }
}

/// Training sample for ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Input features
    pub features: QueryFeatures,
    /// Target outcome
    pub outcome: PerformanceOutcome,
    /// Service selection decisions
    pub service_selections: Vec<String>,
    /// Join order used
    pub join_order: Vec<String>,
    /// Caching decisions
    pub caching_decisions: HashMap<String, bool>,
    /// Query identifier
    pub query_id: String,
}

/// Source selection prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSelectionPrediction {
    /// Recommended services
    pub recommended_services: Vec<String>,
    /// Confidence scores for each service
    pub confidence_scores: HashMap<String, f64>,
    /// Expected performance
    pub expected_performance: PerformanceOutcome,
    /// Alternative options
    pub alternatives: Vec<SourceAlternative>,
}

/// Alternative source selection option
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceAlternative {
    /// Service IDs
    pub services: Vec<String>,
    /// Expected performance
    pub expected_performance: PerformanceOutcome,
    /// Confidence score
    pub confidence: f64,
    /// Risk assessment
    pub risk_score: f64,
}

/// Join order optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinOrderOptimization {
    /// Recommended join order
    pub recommended_order: Vec<String>,
    /// Expected cost
    pub expected_cost: f64,
    /// Alternative orders
    pub alternatives: Vec<JoinOrderAlternative>,
    /// Optimization confidence
    pub confidence: f64,
}

/// Alternative join order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinOrderAlternative {
    /// Join order
    pub order: Vec<String>,
    /// Expected cost
    pub cost: f64,
    /// Risk score
    pub risk: f64,
}

/// Caching strategy recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingStrategy {
    /// Items to cache
    pub cache_items: HashMap<String, CacheRecommendation>,
    /// Cache eviction order
    pub eviction_order: Vec<String>,
    /// Expected cache hit rate
    pub expected_hit_rate: f64,
    /// Memory requirements
    pub memory_requirements: u64,
}

/// Cache recommendation for specific item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheRecommendation {
    /// Should cache this item
    pub should_cache: bool,
    /// Priority score
    pub priority: f64,
    /// Expected benefit
    pub expected_benefit: f64,
    /// TTL recommendation
    pub ttl_seconds: u64,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    /// Is anomalous
    pub is_anomalous: bool,
    /// Anomaly score (0.0 to 1.0)
    pub anomaly_score: f64,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Confidence in detection
    pub confidence: f64,
    /// Recommended actions
    pub recommendations: Vec<String>,
}

/// Types of anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Performance degradation
    PerformanceDegradation,
    /// Unusual resource usage
    ResourceAnomaly,
    /// Service behavior anomaly
    ServiceAnomaly,
    /// Pattern anomaly
    PatternAnomaly,
    /// Data quality issue
    DataQualityIssue,
    /// Security concern
    SecurityAnomaly,
}

/// Linear regression model for performance prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegressionModel {
    /// Feature weights
    pub weights: Vec<f64>,
    /// Bias term
    pub bias: f64,
    /// Training iterations
    pub iterations: u32,
    /// Model accuracy
    pub accuracy: f64,
    /// Last training time
    pub last_trained: SystemTime,
}

/// Neural network model for advanced performance prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkModel {
    /// Hidden layer weights (input to hidden)
    pub weights_input_hidden: Vec<Vec<f64>>,
    /// Hidden layer biases
    pub bias_hidden: Vec<f64>,
    /// Output layer weights (hidden to output)
    pub weights_hidden_output: Vec<f64>,
    /// Output bias
    pub bias_output: f64,
    /// Training iterations
    pub iterations: u32,
    /// Model accuracy
    pub accuracy: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Last training time
    pub last_trained: SystemTime,
}

/// ML optimizer statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct MLStatistics {
    /// Total predictions made
    pub total_predictions: u64,
    /// Successful predictions
    pub successful_predictions: u64,
    /// Training samples collected
    pub training_samples_count: u64,
    /// Model accuracy
    pub model_accuracy: f64,
    /// Last training time
    pub last_training: Option<SystemTime>,
    /// Anomalies detected
    pub anomalies_detected: u64,
    /// Cache hit improvement
    pub cache_hit_improvement: f64,
}
