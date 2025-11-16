//! Performance Profiling Tools
//!
//! Comprehensive performance profiling and analysis tools for identifying
//! bottlenecks, optimizing queries, and monitoring system performance.

use anyhow::Result;
use scirs2_core::profiling::Profiler as SciRSProfiler;
use scirs2_core::random::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Performance profiler for SPARQL queries and system operations
pub struct PerformanceProfiler {
    /// Query profiles
    query_profiles: Arc<RwLock<HashMap<String, QueryProfile>>>,
    /// Operation profiles
    operation_profiles: Arc<RwLock<HashMap<String, OperationProfile>>>,
    /// System metrics history
    metrics_history: Arc<RwLock<Vec<SystemMetricsSnapshot>>>,
    /// SciRS2 profiler integration
    scirs_profiler: Arc<Mutex<SciRSProfiler>>,
    /// Configuration
    config: ProfilerConfig,
}

/// Profiler configuration
#[derive(Debug, Clone)]
pub struct ProfilerConfig {
    /// Enable profiling
    pub enabled: bool,
    /// Maximum profiles to retain
    pub max_profiles: usize,
    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,
    /// Enable detailed tracing
    pub detailed_tracing: bool,
    /// Metrics retention period
    pub metrics_retention_duration: Duration,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_profiles: 10000,
            sampling_rate: 1.0,
            detailed_tracing: false,
            metrics_retention_duration: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Query profile information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryProfile {
    /// Query ID
    pub id: String,
    /// Query string
    pub query: String,
    /// Dataset name
    pub dataset: String,
    /// Execution time
    pub execution_time_ms: u64,
    /// Parse time
    pub parse_time_ms: u64,
    /// Planning time
    pub planning_time_ms: u64,
    /// Execution phases
    pub phases: Vec<ExecutionPhase>,
    /// Result count
    pub result_count: usize,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Performance score (0-100)
    pub performance_score: f64,
    /// Optimization suggestions
    pub suggestions: Vec<OptimizationSuggestion>,
}

/// Execution phase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPhase {
    /// Phase name
    pub name: String,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// CPU time in milliseconds
    pub cpu_time_ms: u64,
    /// Memory allocated in bytes
    pub memory_bytes: u64,
    /// Details
    pub details: HashMap<String, serde_json::Value>,
}

/// Optimization suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Severity (high, medium, low)
    pub severity: String,
    /// Description
    pub description: String,
    /// Expected improvement
    pub expected_improvement: String,
}

/// Suggestion type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SuggestionType {
    AddIndex,
    OptimizeJoin,
    LimitResults,
    UseFilter,
    SimplifyPattern,
    CacheResult,
    ReduceComplexity,
}

/// Operation profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationProfile {
    /// Operation name
    pub operation: String,
    /// Execution count
    pub execution_count: u64,
    /// Total time in milliseconds
    pub total_time_ms: u64,
    /// Average time in milliseconds
    pub avg_time_ms: f64,
    /// Min time in milliseconds
    pub min_time_ms: u64,
    /// Max time in milliseconds
    pub max_time_ms: u64,
    /// Standard deviation
    pub std_dev_ms: f64,
    /// Percentiles (p50, p95, p99)
    pub percentiles: HashMap<String, u64>,
}

/// System metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetricsSnapshot {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// CPU usage percent
    pub cpu_usage_percent: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Active queries
    pub active_queries: usize,
    /// Queries per second
    pub queries_per_second: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Report timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Time period
    pub period_seconds: u64,
    /// Top slow queries
    pub slow_queries: Vec<QueryProfile>,
    /// Top operations by time
    pub top_operations: Vec<OperationProfile>,
    /// System metrics summary
    pub metrics_summary: MetricsSummary,
    /// Performance trends
    pub trends: PerformanceTrends,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    /// Average CPU usage
    pub avg_cpu_percent: f64,
    /// Peak CPU usage
    pub peak_cpu_percent: f64,
    /// Average memory usage in MB
    pub avg_memory_mb: f64,
    /// Peak memory usage in MB
    pub peak_memory_mb: f64,
    /// Total queries executed
    pub total_queries: u64,
    /// Average QPS
    pub avg_qps: f64,
    /// Peak QPS
    pub peak_qps: f64,
}

/// Performance trends
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    /// CPU trend (increasing, stable, decreasing)
    pub cpu_trend: String,
    /// Memory trend
    pub memory_trend: String,
    /// Latency trend
    pub latency_trend: String,
    /// Throughput trend
    pub throughput_trend: String,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new(config: ProfilerConfig) -> Self {
        Self {
            query_profiles: Arc::new(RwLock::new(HashMap::new())),
            operation_profiles: Arc::new(RwLock::new(HashMap::new())),
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            scirs_profiler: Arc::new(Mutex::new(SciRSProfiler::new())),
            config,
        }
    }

    /// Start profiling a query
    pub async fn start_query_profile(
        &self,
        query_id: String,
        query: String,
        dataset: String,
    ) -> QueryProfiler {
        QueryProfiler {
            profiler: self.scirs_profiler.clone(),
            query_id,
            query,
            dataset,
            start_time: Instant::now(),
            phases: Vec::new(),
        }
    }

    /// Complete query profile
    pub async fn complete_query_profile(&self, profile: QueryProfile) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        // Sample based on sampling rate
        let mut rng = scirs2_core::random::rng();
        if rng.random::<f64>() > self.config.sampling_rate {
            return Ok(());
        }

        let mut profiles = self.query_profiles.write().await;

        // Add profile
        profiles.insert(profile.id.clone(), profile.clone());

        // Limit size
        if profiles.len() > self.config.max_profiles {
            // Remove oldest entries
            let oldest_key = profiles
                .iter()
                .min_by_key(|(_, p)| p.timestamp)
                .map(|(k, _)| k.clone());

            if let Some(key) = oldest_key {
                profiles.remove(&key);
            }
        }

        debug!("Completed query profile: {}", profile.id);

        Ok(())
    }

    /// Profile an operation
    pub async fn profile_operation<F, T>(&self, operation: &str, f: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        let start = Instant::now();

        if let Ok(mut profiler) = self.scirs_profiler.lock() {
            profiler.start();
        }

        let result = f();

        if let Ok(mut profiler) = self.scirs_profiler.lock() {
            profiler.stop();
        }

        let duration = start.elapsed();

        // Update operation profile
        let mut profiles = self.operation_profiles.write().await;
        let profile = profiles
            .entry(operation.to_string())
            .or_insert(OperationProfile {
                operation: operation.to_string(),
                execution_count: 0,
                total_time_ms: 0,
                avg_time_ms: 0.0,
                min_time_ms: u64::MAX,
                max_time_ms: 0,
                std_dev_ms: 0.0,
                percentiles: HashMap::new(),
            });

        profile.execution_count += 1;
        profile.total_time_ms += duration.as_millis() as u64;
        profile.avg_time_ms = profile.total_time_ms as f64 / profile.execution_count as f64;
        profile.min_time_ms = profile.min_time_ms.min(duration.as_millis() as u64);
        profile.max_time_ms = profile.max_time_ms.max(duration.as_millis() as u64);

        result
    }

    /// Record system metrics
    pub async fn record_metrics(&self, snapshot: SystemMetricsSnapshot) -> Result<()> {
        let mut history = self.metrics_history.write().await;

        history.push(snapshot);

        // Remove old metrics
        let cutoff = chrono::Utc::now()
            - chrono::Duration::seconds(self.config.metrics_retention_duration.as_secs() as i64);
        history.retain(|s| s.timestamp > cutoff);

        Ok(())
    }

    /// Generate performance report
    pub async fn generate_report(&self, period_seconds: u64) -> Result<PerformanceReport> {
        let cutoff = chrono::Utc::now() - chrono::Duration::seconds(period_seconds as i64);

        // Get slow queries
        let profiles = self.query_profiles.read().await;
        let mut slow_queries: Vec<QueryProfile> = profiles
            .values()
            .filter(|p| p.timestamp > cutoff)
            .cloned()
            .collect();
        slow_queries.sort_by_key(|p| std::cmp::Reverse(p.execution_time_ms));
        slow_queries.truncate(10);

        // Get top operations
        let op_profiles = self.operation_profiles.read().await;
        let mut top_operations: Vec<OperationProfile> = op_profiles.values().cloned().collect();
        top_operations.sort_by_key(|p| std::cmp::Reverse(p.total_time_ms));
        top_operations.truncate(10);

        // Calculate metrics summary
        let metrics = self.metrics_history.read().await;
        let recent_metrics: Vec<&SystemMetricsSnapshot> =
            metrics.iter().filter(|m| m.timestamp > cutoff).collect();

        let metrics_summary = if recent_metrics.is_empty() {
            MetricsSummary {
                avg_cpu_percent: 0.0,
                peak_cpu_percent: 0.0,
                avg_memory_mb: 0.0,
                peak_memory_mb: 0.0,
                total_queries: 0,
                avg_qps: 0.0,
                peak_qps: 0.0,
            }
        } else {
            MetricsSummary {
                avg_cpu_percent: recent_metrics
                    .iter()
                    .map(|m| m.cpu_usage_percent)
                    .sum::<f64>()
                    / recent_metrics.len() as f64,
                peak_cpu_percent: recent_metrics
                    .iter()
                    .map(|m| m.cpu_usage_percent)
                    .fold(0.0, f64::max),
                avg_memory_mb: recent_metrics
                    .iter()
                    .map(|m| m.memory_usage_bytes as f64 / 1024.0 / 1024.0)
                    .sum::<f64>()
                    / recent_metrics.len() as f64,
                peak_memory_mb: recent_metrics
                    .iter()
                    .map(|m| m.memory_usage_bytes as f64 / 1024.0 / 1024.0)
                    .fold(0.0, f64::max),
                total_queries: slow_queries.len() as u64,
                avg_qps: recent_metrics
                    .iter()
                    .map(|m| m.queries_per_second)
                    .sum::<f64>()
                    / recent_metrics.len() as f64,
                peak_qps: recent_metrics
                    .iter()
                    .map(|m| m.queries_per_second)
                    .fold(0.0, f64::max),
            }
        };

        // Analyze trends
        let trends = self.analyze_trends(&recent_metrics);

        // Generate recommendations
        let recommendations = self.generate_recommendations(&slow_queries, &metrics_summary);

        Ok(PerformanceReport {
            timestamp: chrono::Utc::now(),
            period_seconds,
            slow_queries,
            top_operations,
            metrics_summary,
            trends,
            recommendations,
        })
    }

    /// Analyze performance trends
    fn analyze_trends(&self, metrics: &[&SystemMetricsSnapshot]) -> PerformanceTrends {
        if metrics.len() < 2 {
            return PerformanceTrends {
                cpu_trend: "stable".to_string(),
                memory_trend: "stable".to_string(),
                latency_trend: "stable".to_string(),
                throughput_trend: "stable".to_string(),
            };
        }

        // Simple trend analysis
        let mid = metrics.len() / 2;
        let first_half = &metrics[..mid];
        let second_half = &metrics[mid..];

        let avg_cpu_first =
            first_half.iter().map(|m| m.cpu_usage_percent).sum::<f64>() / first_half.len() as f64;
        let avg_cpu_second =
            second_half.iter().map(|m| m.cpu_usage_percent).sum::<f64>() / second_half.len() as f64;

        let avg_latency_first =
            first_half.iter().map(|m| m.avg_latency_ms).sum::<f64>() / first_half.len() as f64;
        let avg_latency_second =
            second_half.iter().map(|m| m.avg_latency_ms).sum::<f64>() / second_half.len() as f64;

        PerformanceTrends {
            cpu_trend: if avg_cpu_second > avg_cpu_first * 1.1 {
                "increasing".to_string()
            } else if avg_cpu_second < avg_cpu_first * 0.9 {
                "decreasing".to_string()
            } else {
                "stable".to_string()
            },
            memory_trend: "stable".to_string(), // Simplified
            latency_trend: if avg_latency_second > avg_latency_first * 1.1 {
                "increasing".to_string()
            } else if avg_latency_second < avg_latency_first * 0.9 {
                "decreasing".to_string()
            } else {
                "stable".to_string()
            },
            throughput_trend: "stable".to_string(), // Simplified
        }
    }

    /// Generate optimization recommendations
    fn generate_recommendations(
        &self,
        slow_queries: &[QueryProfile],
        metrics: &MetricsSummary,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if metrics.avg_cpu_percent > 80.0 {
            recommendations.push("High CPU usage detected. Consider adding more compute resources or optimizing queries.".to_string());
        }

        if metrics.peak_memory_mb > 4096.0 {
            recommendations.push("High memory usage detected. Enable memory-efficient streaming for large result sets.".to_string());
        }

        if !slow_queries.is_empty() {
            recommendations.push(format!(
                "Found {} slow queries. Review query patterns and consider adding indexes.",
                slow_queries.len()
            ));
        }

        if metrics.avg_qps > 1000.0 {
            recommendations.push(
                "High query volume detected. Consider enabling query result caching.".to_string(),
            );
        }

        recommendations
    }

    /// Get query statistics
    pub async fn get_query_statistics(&self) -> HashMap<String, u64> {
        let profiles = self.query_profiles.read().await;

        let mut stats = HashMap::new();
        stats.insert("total_queries".to_string(), profiles.len() as u64);
        stats.insert(
            "avg_execution_time_ms".to_string(),
            profiles.values().map(|p| p.execution_time_ms).sum::<u64>()
                / profiles.len().max(1) as u64,
        );

        stats
    }
}

/// Query profiler helper
pub struct QueryProfiler {
    profiler: Arc<Mutex<SciRSProfiler>>,
    query_id: String,
    query: String,
    dataset: String,
    start_time: Instant,
    phases: Vec<ExecutionPhase>,
}

impl QueryProfiler {
    /// Start a new phase
    pub fn start_phase(&mut self, name: &str) {
        if let Ok(mut profiler) = self.profiler.lock() {
            profiler.start();
        }
    }

    /// End current phase
    pub fn end_phase(&mut self, name: &str, details: HashMap<String, serde_json::Value>) {
        if let Ok(mut profiler) = self.profiler.lock() {
            profiler.stop();
        }

        self.phases.push(ExecutionPhase {
            name: name.to_string(),
            duration_ms: 0, // Would get from profiler
            cpu_time_ms: 0,
            memory_bytes: 0,
            details,
        });
    }

    /// Complete profiling
    pub fn complete(self, result_count: usize) -> QueryProfile {
        let execution_time_ms = self.start_time.elapsed().as_millis() as u64;

        QueryProfile {
            id: self.query_id,
            query: self.query,
            dataset: self.dataset,
            execution_time_ms,
            parse_time_ms: 0,
            planning_time_ms: 0,
            phases: self.phases,
            result_count,
            timestamp: chrono::Utc::now(),
            performance_score: calculate_performance_score(execution_time_ms, result_count),
            suggestions: vec![],
        }
    }
}

/// Calculate performance score
fn calculate_performance_score(execution_time_ms: u64, result_count: usize) -> f64 {
    // Simple scoring: faster queries with reasonable result sizes score higher
    let time_score = (1000.0 / (execution_time_ms as f64 + 1.0)).min(100.0);
    let size_score = if result_count > 10000 {
        50.0
    } else if result_count > 1000 {
        75.0
    } else {
        100.0
    };

    (time_score + size_score) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_profiler_creation() {
        let profiler = PerformanceProfiler::new(ProfilerConfig::default());
        assert!(profiler.config.enabled);
    }

    #[tokio::test]
    async fn test_query_profile() {
        let profiler = PerformanceProfiler::new(ProfilerConfig::default());
        let query_profiler = profiler
            .start_query_profile(
                "q1".to_string(),
                "SELECT * WHERE { ?s ?p ?o }".to_string(),
                "test".to_string(),
            )
            .await;

        let profile = query_profiler.complete(100);
        assert_eq!(profile.id, "q1");
        assert_eq!(profile.result_count, 100);
    }

    #[test]
    fn test_performance_score() {
        let score = calculate_performance_score(100, 100);
        assert!(score > 0.0 && score <= 100.0);
    }
}
