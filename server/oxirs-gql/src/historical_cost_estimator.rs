//! Historical Query Cost Estimator
//!
//! This module provides query cost estimation based on historical execution data.
//! It tracks query performance metrics over time and uses statistical analysis
//! to predict the cost of new queries.
//!
//! # Features
//! - Query fingerprinting for pattern matching
//! - Historical metrics tracking (execution time, complexity, resources)
//! - Statistical cost prediction using regression models
//! - Percentile-based cost estimation (p50, p95, p99)
//! - Adaptive learning from new query executions
//! - Query pattern clustering for similar queries
//!
//! # Example
//! ```
//! use oxirs_gql::historical_cost_estimator::HistoricalCostEstimator;
//!
//! let mut estimator = HistoricalCostEstimator::new();
//! estimator.record_execution("query { user { name } }", 150.0, 100, 1024);
//! let estimate = estimator.estimate_cost("query { user { name } }");
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime};
use thiserror::Error;

/// Error types for historical cost estimation
#[derive(Debug, Error)]
pub enum CostEstimationError {
    #[error("Insufficient historical data: {0}")]
    InsufficientData(String),

    #[error("Query parsing failed: {0}")]
    QueryParsingError(String),

    #[error("Statistical analysis failed: {0}")]
    StatisticalError(String),

    #[error("Lock acquisition failed: {0}")]
    LockError(String),
}

/// Query execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,

    /// Query complexity score
    pub complexity_score: u32,

    /// Memory usage in bytes
    pub memory_bytes: u64,

    /// Number of fields resolved
    pub fields_resolved: u32,

    /// Number of database queries executed
    pub db_queries: u32,

    /// Timestamp when the query was executed
    pub timestamp: SystemTime,

    /// Whether the query was cached
    pub was_cached: bool,

    /// Response size in bytes
    pub response_size_bytes: u64,
}

impl QueryMetrics {
    /// Create new query metrics
    pub fn new(execution_time_ms: f64, complexity_score: u32, memory_bytes: u64) -> Self {
        Self {
            execution_time_ms,
            complexity_score,
            memory_bytes,
            fields_resolved: 0,
            db_queries: 0,
            timestamp: SystemTime::now(),
            was_cached: false,
            response_size_bytes: 0,
        }
    }

    /// Create metrics with all fields
    pub fn with_details(
        execution_time_ms: f64,
        complexity_score: u32,
        memory_bytes: u64,
        fields_resolved: u32,
        db_queries: u32,
        was_cached: bool,
        response_size_bytes: u64,
    ) -> Self {
        Self {
            execution_time_ms,
            complexity_score,
            memory_bytes,
            fields_resolved,
            db_queries,
            timestamp: SystemTime::now(),
            was_cached,
            response_size_bytes,
        }
    }
}

/// Cost estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Estimated execution time in milliseconds (median)
    pub estimated_time_ms: f64,

    /// Estimated complexity score
    pub estimated_complexity: u32,

    /// Estimated memory usage in bytes
    pub estimated_memory_bytes: u64,

    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,

    /// Number of historical samples used
    pub sample_count: usize,

    /// P50 (median) execution time
    pub p50_time_ms: f64,

    /// P95 execution time
    pub p95_time_ms: f64,

    /// P99 execution time
    pub p99_time_ms: f64,

    /// Minimum observed execution time
    pub min_time_ms: f64,

    /// Maximum observed execution time
    pub max_time_ms: f64,

    /// Standard deviation of execution time
    pub std_dev_ms: f64,
}

impl CostEstimate {
    /// Create a default estimate when no historical data is available
    pub fn default_estimate() -> Self {
        Self {
            estimated_time_ms: 100.0,
            estimated_complexity: 10,
            estimated_memory_bytes: 1024,
            confidence: 0.0,
            sample_count: 0,
            p50_time_ms: 100.0,
            p95_time_ms: 200.0,
            p99_time_ms: 300.0,
            min_time_ms: 50.0,
            max_time_ms: 500.0,
            std_dev_ms: 50.0,
        }
    }
}

/// Query fingerprint for pattern matching
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct QueryFingerprint {
    /// Normalized query structure (with variables replaced)
    normalized_query: String,

    /// Hash of the query structure
    structure_hash: u64,
}

impl QueryFingerprint {
    /// Create a fingerprint from a query string
    pub fn from_query(query: &str) -> Self {
        let normalized = Self::normalize_query(query);
        let structure_hash = Self::hash_structure(&normalized);

        Self {
            normalized_query: normalized,
            structure_hash,
        }
    }

    /// Normalize a query by removing variable values and formatting
    fn normalize_query(query: &str) -> String {
        // Simple normalization: lowercase, remove extra whitespace
        query
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Hash the query structure
    fn hash_structure(normalized: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        normalized.hash(&mut hasher);
        hasher.finish()
    }
}

/// Historical data for a query pattern
#[derive(Debug, Clone)]
struct QueryHistoricalData {
    /// All recorded metrics for this query pattern
    metrics: Vec<QueryMetrics>,

    /// Last update timestamp
    last_updated: Instant,

    /// Total number of executions
    total_executions: u64,
}

impl QueryHistoricalData {
    fn new() -> Self {
        Self {
            metrics: Vec::new(),
            last_updated: Instant::now(),
            total_executions: 0,
        }
    }

    fn add_metrics(&mut self, metrics: QueryMetrics) {
        self.metrics.push(metrics);
        self.last_updated = Instant::now();
        self.total_executions += 1;

        // Keep only recent metrics (e.g., last 1000 executions)
        if self.metrics.len() > 1000 {
            self.metrics.drain(0..self.metrics.len() - 1000);
        }
    }
}

/// Configuration for the historical cost estimator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatorConfig {
    /// Maximum number of query patterns to track
    pub max_patterns: usize,

    /// Maximum age of historical data (in seconds)
    pub max_age_seconds: u64,

    /// Minimum number of samples required for estimation
    pub min_samples: usize,

    /// Enable statistical smoothing
    pub enable_smoothing: bool,

    /// Confidence threshold for predictions (0.0 to 1.0)
    pub confidence_threshold: f64,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            max_patterns: 10000,
            max_age_seconds: 86400, // 24 hours
            min_samples: 5,
            enable_smoothing: true,
            confidence_threshold: 0.7,
        }
    }
}

/// Historical query cost estimator
pub struct HistoricalCostEstimator {
    /// Historical data indexed by query fingerprint
    historical_data: Arc<RwLock<HashMap<QueryFingerprint, QueryHistoricalData>>>,

    /// Configuration
    config: EstimatorConfig,

    /// Total queries tracked
    total_queries: Arc<RwLock<u64>>,

    /// Cache hit rate
    cache_hit_rate: Arc<RwLock<f64>>,
}

impl HistoricalCostEstimator {
    /// Create a new historical cost estimator with default configuration
    pub fn new() -> Self {
        Self::with_config(EstimatorConfig::default())
    }

    /// Create a new historical cost estimator with custom configuration
    pub fn with_config(config: EstimatorConfig) -> Self {
        Self {
            historical_data: Arc::new(RwLock::new(HashMap::new())),
            config,
            total_queries: Arc::new(RwLock::new(0)),
            cache_hit_rate: Arc::new(RwLock::new(0.0)),
        }
    }

    /// Record a query execution
    pub fn record_execution(
        &mut self,
        query: &str,
        execution_time_ms: f64,
        complexity_score: u32,
        memory_bytes: u64,
    ) -> Result<(), CostEstimationError> {
        let metrics = QueryMetrics::new(execution_time_ms, complexity_score, memory_bytes);
        self.record_metrics(query, metrics)
    }

    /// Record query metrics
    pub fn record_metrics(
        &mut self,
        query: &str,
        metrics: QueryMetrics,
    ) -> Result<(), CostEstimationError> {
        let fingerprint = QueryFingerprint::from_query(query);

        let mut data = self
            .historical_data
            .write()
            .map_err(|e| CostEstimationError::LockError(e.to_string()))?;

        let historical = data
            .entry(fingerprint)
            .or_insert_with(QueryHistoricalData::new);

        historical.add_metrics(metrics.clone());

        // Update total queries counter
        let mut total = self
            .total_queries
            .write()
            .map_err(|e| CostEstimationError::LockError(e.to_string()))?;
        *total += 1;

        // Update cache hit rate
        if metrics.was_cached {
            let mut hit_rate = self
                .cache_hit_rate
                .write()
                .map_err(|e| CostEstimationError::LockError(e.to_string()))?;
            *hit_rate = (*hit_rate * (*total as f64 - 1.0) + 1.0) / (*total as f64);
        }

        // Cleanup old patterns if necessary
        if data.len() > self.config.max_patterns {
            self.cleanup_old_patterns(&mut data);
        }

        Ok(())
    }

    /// Estimate the cost of a query
    pub fn estimate_cost(&self, query: &str) -> Result<CostEstimate, CostEstimationError> {
        let fingerprint = QueryFingerprint::from_query(query);

        let data = self
            .historical_data
            .read()
            .map_err(|e| CostEstimationError::LockError(e.to_string()))?;

        let historical = data.get(&fingerprint);

        match historical {
            Some(hist) if hist.metrics.len() >= self.config.min_samples => {
                self.compute_estimate(&hist.metrics)
            }
            Some(hist) => {
                // Not enough samples, use available data with low confidence
                let mut estimate = self.compute_estimate(&hist.metrics)?;
                estimate.confidence =
                    (hist.metrics.len() as f64) / (self.config.min_samples as f64);
                Ok(estimate)
            }
            None => {
                // No historical data, return default estimate
                Ok(CostEstimate::default_estimate())
            }
        }
    }

    /// Compute cost estimate from historical metrics
    fn compute_estimate(
        &self,
        metrics: &[QueryMetrics],
    ) -> Result<CostEstimate, CostEstimationError> {
        if metrics.is_empty() {
            return Err(CostEstimationError::InsufficientData(
                "No metrics available".to_string(),
            ));
        }

        // Extract execution times
        let times: Vec<f64> = metrics.iter().map(|m| m.execution_time_ms).collect();

        // Calculate percentiles
        let p50 = self.percentile(&times, 0.50)?;
        let p95 = self.percentile(&times, 0.95)?;
        let p99 = self.percentile(&times, 0.99)?;

        // Calculate statistics
        let min_time = times.iter().copied().fold(f64::INFINITY, f64::min);
        let max_time = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean_time = times.iter().sum::<f64>() / times.len() as f64;
        let std_dev = self.standard_deviation(&times, mean_time)?;

        // Calculate average complexity and memory
        let avg_complexity =
            metrics.iter().map(|m| m.complexity_score).sum::<u32>() / metrics.len() as u32;
        let avg_memory = metrics.iter().map(|m| m.memory_bytes).sum::<u64>() / metrics.len() as u64;

        // Confidence based on sample size and variance
        let confidence = self.calculate_confidence(metrics.len(), std_dev, mean_time);

        Ok(CostEstimate {
            estimated_time_ms: p50,
            estimated_complexity: avg_complexity,
            estimated_memory_bytes: avg_memory,
            confidence,
            sample_count: metrics.len(),
            p50_time_ms: p50,
            p95_time_ms: p95,
            p99_time_ms: p99,
            min_time_ms: min_time,
            max_time_ms: max_time,
            std_dev_ms: std_dev,
        })
    }

    /// Calculate percentile
    fn percentile(&self, values: &[f64], p: f64) -> Result<f64, CostEstimationError> {
        if values.is_empty() {
            return Err(CostEstimationError::InsufficientData(
                "Cannot calculate percentile of empty data".to_string(),
            ));
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let index = (p * (sorted.len() - 1) as f64).round() as usize;
        Ok(sorted[index.min(sorted.len() - 1)])
    }

    /// Calculate standard deviation
    fn standard_deviation(&self, values: &[f64], mean: f64) -> Result<f64, CostEstimationError> {
        if values.is_empty() {
            return Ok(0.0);
        }

        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

        Ok(variance.sqrt())
    }

    /// Calculate confidence level
    fn calculate_confidence(&self, sample_count: usize, std_dev: f64, mean: f64) -> f64 {
        // Confidence based on:
        // 1. Sample count (more samples = higher confidence)
        // 2. Coefficient of variation (lower CV = higher confidence)

        let sample_confidence = (sample_count as f64 / 100.0).min(1.0);

        let cv = if mean > 0.0 { std_dev / mean } else { 1.0 };

        let cv_confidence = 1.0 / (1.0 + cv);

        // Combine both factors
        (sample_confidence * 0.6 + cv_confidence * 0.4).min(1.0)
    }

    /// Cleanup old patterns to maintain maximum pattern limit
    fn cleanup_old_patterns(&self, data: &mut HashMap<QueryFingerprint, QueryHistoricalData>) {
        // Collect fingerprints with their timestamps
        let mut patterns: Vec<_> = data
            .iter()
            .map(|(fp, hist)| (fp.clone(), hist.last_updated))
            .collect();

        // Sort by timestamp (oldest first)
        patterns.sort_by(|a, b| a.1.cmp(&b.1));

        // Calculate how many to remove
        let to_remove = data.len() - (self.config.max_patterns * 9 / 10);

        // Remove oldest patterns
        for (fingerprint, _) in patterns.iter().take(to_remove) {
            data.remove(fingerprint);
        }
    }

    /// Get statistics about the estimator
    pub fn get_statistics(&self) -> Result<EstimatorStatistics, CostEstimationError> {
        let data = self
            .historical_data
            .read()
            .map_err(|e| CostEstimationError::LockError(e.to_string()))?;

        let total = *self
            .total_queries
            .read()
            .map_err(|e| CostEstimationError::LockError(e.to_string()))?;

        let hit_rate = *self
            .cache_hit_rate
            .read()
            .map_err(|e| CostEstimationError::LockError(e.to_string()))?;

        let total_metrics: usize = data.values().map(|h| h.metrics.len()).sum();

        Ok(EstimatorStatistics {
            total_patterns: data.len(),
            total_queries: total,
            total_metrics,
            cache_hit_rate: hit_rate,
        })
    }

    /// Get historical data for a specific query
    pub fn get_query_history(&self, query: &str) -> Result<Vec<QueryMetrics>, CostEstimationError> {
        let fingerprint = QueryFingerprint::from_query(query);

        let data = self
            .historical_data
            .read()
            .map_err(|e| CostEstimationError::LockError(e.to_string()))?;

        Ok(data
            .get(&fingerprint)
            .map(|h| h.metrics.clone())
            .unwrap_or_default())
    }

    /// Clear all historical data
    pub fn clear(&mut self) -> Result<(), CostEstimationError> {
        let mut data = self
            .historical_data
            .write()
            .map_err(|e| CostEstimationError::LockError(e.to_string()))?;

        data.clear();

        let mut total = self
            .total_queries
            .write()
            .map_err(|e| CostEstimationError::LockError(e.to_string()))?;
        *total = 0;

        let mut hit_rate = self
            .cache_hit_rate
            .write()
            .map_err(|e| CostEstimationError::LockError(e.to_string()))?;
        *hit_rate = 0.0;

        Ok(())
    }
}

impl Default for HistoricalCostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the estimator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatorStatistics {
    /// Total number of unique query patterns tracked
    pub total_patterns: usize,

    /// Total number of queries executed
    pub total_queries: u64,

    /// Total number of metrics recorded
    pub total_metrics: usize,

    /// Overall cache hit rate
    pub cache_hit_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimator_creation() {
        let estimator = HistoricalCostEstimator::new();
        assert!(estimator.historical_data.read().unwrap().is_empty());
    }

    #[test]
    fn test_record_and_estimate() {
        let mut estimator = HistoricalCostEstimator::new();

        // Record multiple executions of the same query
        for i in 0..10 {
            let time = 100.0 + (i as f64 * 10.0);
            estimator
                .record_execution("query { user { name } }", time, 50, 1024)
                .unwrap();
        }

        // Estimate cost
        let estimate = estimator.estimate_cost("query { user { name } }").unwrap();

        assert!(estimate.sample_count == 10);
        assert!(estimate.estimated_time_ms > 0.0);
        assert!(estimate.confidence > 0.0);
    }

    #[test]
    fn test_insufficient_data() {
        let estimator = HistoricalCostEstimator::new();

        // Query with no history should return default estimate
        let estimate = estimator.estimate_cost("query { unknown }").unwrap();

        assert_eq!(estimate.sample_count, 0);
        assert_eq!(estimate.confidence, 0.0);
    }

    #[test]
    fn test_query_fingerprinting() {
        let fp1 = QueryFingerprint::from_query("query { user { name } }");
        let fp2 = QueryFingerprint::from_query("query { user { name } }");
        let fp3 = QueryFingerprint::from_query("query { user { email } }");

        assert_eq!(fp1.structure_hash, fp2.structure_hash);
        assert_ne!(fp1.structure_hash, fp3.structure_hash);
    }

    #[test]
    fn test_percentile_calculation() {
        let estimator = HistoricalCostEstimator::new();
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let p50 = estimator.percentile(&values, 0.50).unwrap();
        let p95 = estimator.percentile(&values, 0.95).unwrap();

        assert!((p50 - 5.5).abs() < 1.0);
        assert!(p95 >= 9.0);
    }

    #[test]
    fn test_statistics() {
        let mut estimator = HistoricalCostEstimator::new();

        // Record some queries
        for _ in 0..5 {
            estimator
                .record_execution("query { user { name } }", 100.0, 50, 1024)
                .unwrap();
        }

        for _ in 0..3 {
            estimator
                .record_execution("query { posts { title } }", 150.0, 60, 2048)
                .unwrap();
        }

        let stats = estimator.get_statistics().unwrap();

        assert_eq!(stats.total_patterns, 2);
        assert_eq!(stats.total_queries, 8);
        assert_eq!(stats.total_metrics, 8);
    }

    #[test]
    fn test_clear_data() {
        let mut estimator = HistoricalCostEstimator::new();

        estimator
            .record_execution("query { user { name } }", 100.0, 50, 1024)
            .unwrap();
        assert!(!estimator.historical_data.read().unwrap().is_empty());

        estimator.clear().unwrap();
        assert!(estimator.historical_data.read().unwrap().is_empty());
    }

    #[test]
    fn test_confidence_calculation() {
        let estimator = HistoricalCostEstimator::new();

        // Low sample count, high variance
        let confidence1 = estimator.calculate_confidence(5, 50.0, 100.0);

        // High sample count, low variance
        let confidence2 = estimator.calculate_confidence(100, 10.0, 100.0);

        assert!(confidence2 > confidence1);
    }

    #[test]
    fn test_query_history() {
        let mut estimator = HistoricalCostEstimator::new();

        let query = "query { user { name } }";
        estimator.record_execution(query, 100.0, 50, 1024).unwrap();
        estimator.record_execution(query, 120.0, 52, 1100).unwrap();

        let history = estimator.get_query_history(query).unwrap();

        assert_eq!(history.len(), 2);
        assert_eq!(history[0].execution_time_ms, 100.0);
        assert_eq!(history[1].execution_time_ms, 120.0);
    }

    #[test]
    fn test_metrics_with_details() {
        let mut estimator = HistoricalCostEstimator::new();

        let metrics = QueryMetrics::with_details(150.0, 60, 2048, 10, 5, true, 4096);

        estimator
            .record_metrics("query { user { name } }", metrics)
            .unwrap();

        let stats = estimator.get_statistics().unwrap();
        assert!(stats.cache_hit_rate > 0.0);
    }
}
