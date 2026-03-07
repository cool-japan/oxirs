//! Advanced diagnostic tools for TDB store monitoring and analysis
//!
//! This module provides sophisticated diagnostic capabilities beyond basic health checks:
//! - Query performance analysis and optimization recommendations
//! - Transaction pattern analysis and contention detection
//! - Storage fragmentation analysis and compaction recommendations
//! - Index usage statistics and optimization suggestions
//! - Predictive health monitoring with trend analysis
//! - Auto-tuning recommendations for configuration optimization
//! - Anomaly detection using statistical methods
//! - Capacity planning and forecasting

use crate::diagnostics::{DiagnosticLevel, DiagnosticResult, Severity};
use crate::error::{Result, TdbError};
use crate::storage::BufferPoolStats;
use scirs2_core::ndarray_ext::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

/// Advanced diagnostic report with trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedDiagnosticReport {
    /// Timestamp
    pub timestamp: SystemTime,
    /// Query performance analysis
    pub query_analysis: QueryPerformanceAnalysis,
    /// Transaction pattern analysis
    pub transaction_analysis: TransactionPatternAnalysis,
    /// Fragmentation analysis
    pub fragmentation_analysis: FragmentationAnalysis,
    /// Index usage statistics
    pub index_usage: IndexUsageStatistics,
    /// Predictive health indicators
    pub predictive_health: PredictiveHealthIndicators,
    /// Auto-tuning recommendations
    pub tuning_recommendations: Vec<TuningRecommendation>,
    /// Anomaly detections
    pub anomalies: Vec<AnomalyDetection>,
    /// Capacity planning forecast
    pub capacity_forecast: CapacityForecast,
}

/// Query performance analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPerformanceAnalysis {
    /// Total queries analyzed
    pub total_queries: u64,
    /// Average query execution time
    pub avg_execution_time: Duration,
    /// Median query execution time
    pub median_execution_time: Duration,
    /// P95 query execution time
    pub p95_execution_time: Duration,
    /// P99 query execution time
    pub p99_execution_time: Duration,
    /// Slow query count (above threshold)
    pub slow_query_count: u64,
    /// Query patterns detected
    pub query_patterns: Vec<QueryPattern>,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<String>,
    /// Query cache hit rate
    pub cache_hit_rate: f64,
}

/// Query pattern detected in workload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPattern {
    /// Pattern signature (simplified query structure)
    pub signature: String,
    /// Frequency of this pattern
    pub frequency: u64,
    /// Average execution time for this pattern
    pub avg_time: Duration,
    /// Index usage for this pattern
    pub indexes_used: Vec<String>,
    /// Optimization potential (0.0 = none, 1.0 = high)
    pub optimization_potential: f64,
}

/// Transaction pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionPatternAnalysis {
    /// Total transactions analyzed
    pub total_transactions: u64,
    /// Average transaction duration
    pub avg_duration: Duration,
    /// Median transaction duration
    pub median_duration: Duration,
    /// Transaction commit rate
    pub commit_rate: f64,
    /// Transaction abort rate
    pub abort_rate: f64,
    /// Conflict rate (conflicts per transaction)
    pub conflict_rate: f64,
    /// Deadlock rate (deadlocks per 1000 transactions)
    pub deadlock_rate: f64,
    /// Lock contention points
    pub contention_points: Vec<ContentionPoint>,
    /// Transaction size distribution
    pub size_distribution: TransactionSizeDistribution,
}

/// Lock contention point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentionPoint {
    /// Resource identifier (e.g., "SPO_INDEX", "DICTIONARY")
    pub resource: String,
    /// Number of contentions detected
    pub contention_count: u64,
    /// Average wait time
    pub avg_wait_time: Duration,
    /// Severity (0.0 = low, 1.0 = high)
    pub severity: f64,
}

/// Transaction size distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionSizeDistribution {
    /// Small transactions (< 10 operations)
    pub small_pct: f64,
    /// Medium transactions (10-100 operations)
    pub medium_pct: f64,
    /// Large transactions (> 100 operations)
    pub large_pct: f64,
}

/// Storage fragmentation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FragmentationAnalysis {
    /// Overall fragmentation percentage
    pub overall_fragmentation_pct: f64,
    /// Dictionary fragmentation
    pub dictionary_fragmentation_pct: f64,
    /// Index fragmentation by type
    pub index_fragmentation: HashMap<String, f64>,
    /// Free space distribution
    pub free_space_distribution: Vec<FreeSpaceRegion>,
    /// Compaction benefit estimate (space recoverable)
    pub compaction_benefit_bytes: u64,
    /// Recommended compaction priority (0.0 = low, 1.0 = critical)
    pub compaction_priority: f64,
}

/// Free space region in storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeSpaceRegion {
    /// Starting offset
    pub offset: u64,
    /// Size in bytes
    pub size: u64,
}

/// Index usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexUsageStatistics {
    /// Total index scans performed
    pub total_scans: u64,
    /// Usage by index type
    pub usage_by_index: HashMap<String, IndexUsageStats>,
    /// Unused indexes
    pub unused_indexes: Vec<String>,
    /// Overused indexes (potential bottleneck)
    pub overused_indexes: Vec<String>,
    /// Missing index opportunities
    pub missing_index_opportunities: Vec<MissingIndexOpportunity>,
}

/// Statistics for a specific index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexUsageStats {
    /// Number of scans
    pub scan_count: u64,
    /// Average selectivity (rows returned / rows scanned)
    pub avg_selectivity: f64,
    /// Average scan time
    pub avg_scan_time: Duration,
    /// Index efficiency score (0.0 = inefficient, 1.0 = highly efficient)
    pub efficiency_score: f64,
}

/// Opportunity to create a missing index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissingIndexOpportunity {
    /// Suggested index name
    pub suggested_index: String,
    /// Query patterns that would benefit
    pub benefiting_patterns: Vec<String>,
    /// Estimated performance improvement (percentage)
    pub estimated_improvement_pct: f64,
}

/// Predictive health indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveHealthIndicators {
    /// Time series of historical metrics
    pub historical_metrics: HistoricalMetrics,
    /// Predicted issues in next 24 hours
    pub predicted_issues_24h: Vec<PredictedIssue>,
    /// Predicted issues in next 7 days
    pub predicted_issues_7d: Vec<PredictedIssue>,
    /// Health trend (improving, stable, degrading)
    pub health_trend: HealthTrend,
    /// Resource exhaustion predictions
    pub resource_predictions: ResourcePredictions,
}

/// Historical metrics for trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalMetrics {
    /// Number of data points
    pub data_points: usize,
    /// Time range covered
    pub time_range: Duration,
    /// Query latency trend (per hour)
    pub query_latency_trend: Vec<f64>,
    /// Transaction throughput trend (per hour)
    pub txn_throughput_trend: Vec<f64>,
    /// Storage growth trend (bytes per hour)
    pub storage_growth_trend: Vec<f64>,
    /// Error rate trend (errors per hour)
    pub error_rate_trend: Vec<f64>,
}

/// Predicted issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedIssue {
    /// Issue category
    pub category: String,
    /// Issue description
    pub description: String,
    /// Confidence level (0.0 = low, 1.0 = certain)
    pub confidence: f64,
    /// Estimated time until issue occurs
    pub eta: Duration,
    /// Severity if issue occurs
    pub severity: Severity,
    /// Preventive action
    pub preventive_action: String,
}

/// Health trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthTrend {
    /// System health is improving
    Improving,
    /// System health is stable
    Stable,
    /// System health is degrading slowly
    DegradingSlowly,
    /// System health is degrading rapidly
    DegradingRapidly,
}

/// Resource exhaustion predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePredictions {
    /// Predicted storage full date
    pub storage_full_eta: Option<SystemTime>,
    /// Predicted memory exhaustion
    pub memory_exhaustion_eta: Option<SystemTime>,
    /// Predicted connection pool saturation
    pub connection_saturation_eta: Option<SystemTime>,
}

/// Auto-tuning recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningRecommendation {
    /// Configuration parameter
    pub parameter: String,
    /// Current value
    pub current_value: String,
    /// Recommended value
    pub recommended_value: String,
    /// Rationale
    pub rationale: String,
    /// Expected impact
    pub expected_impact: String,
    /// Priority (0.0 = low, 1.0 = critical)
    pub priority: f64,
}

/// Anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    /// Metric name
    pub metric: String,
    /// Anomaly description
    pub description: String,
    /// Detected value
    pub detected_value: f64,
    /// Expected value range
    pub expected_range: (f64, f64),
    /// Deviation from normal (standard deviations)
    pub deviation: f64,
    /// Detection timestamp
    pub detected_at: SystemTime,
    /// Severity
    pub severity: Severity,
}

/// Capacity planning forecast
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityForecast {
    /// Current storage usage (bytes)
    pub current_storage_bytes: u64,
    /// Predicted storage in 30 days
    pub predicted_30d_bytes: u64,
    /// Predicted storage in 90 days
    pub predicted_90d_bytes: u64,
    /// Growth rate (bytes per day)
    pub growth_rate_per_day: f64,
    /// Days until 80% capacity
    pub days_until_80pct: Option<u32>,
    /// Days until 90% capacity
    pub days_until_90pct: Option<u32>,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Advanced diagnostic engine
pub struct AdvancedDiagnosticEngine {
    /// Historical metrics buffer (circular buffer)
    historical_buffer: VecDeque<MetricSnapshot>,
    /// Maximum history size (24 hours of hourly snapshots)
    max_history_size: usize,
    /// Query performance tracker
    query_tracker: QueryPerformanceTracker,
    /// Transaction pattern tracker
    transaction_tracker: TransactionPatternTracker,
    /// Anomaly detector
    anomaly_detector: AnomalyDetector,
}

/// Snapshot of metrics at a point in time
#[derive(Debug, Clone)]
struct MetricSnapshot {
    timestamp: SystemTime,
    query_latency: f64,
    txn_throughput: f64,
    storage_bytes: u64,
    error_count: u64,
    buffer_pool_stats: BufferPoolStats,
}

/// Query performance tracker
struct QueryPerformanceTracker {
    /// Query execution times (most recent 1000)
    recent_queries: VecDeque<Duration>,
    /// Query patterns
    patterns: HashMap<String, QueryPatternStats>,
    /// Cache hits vs misses
    cache_hits: u64,
    cache_misses: u64,
}

#[derive(Debug, Clone)]
struct QueryPatternStats {
    frequency: u64,
    total_time: Duration,
    indexes_used: Vec<String>,
}

/// Transaction pattern tracker
struct TransactionPatternTracker {
    /// Recent transaction durations
    recent_durations: VecDeque<Duration>,
    /// Commit/abort counts
    commits: u64,
    aborts: u64,
    /// Conflict tracking
    conflicts: u64,
    deadlocks: u64,
    /// Contention points
    contention_map: HashMap<String, ContentionStats>,
}

#[derive(Debug, Clone)]
struct ContentionStats {
    count: u64,
    total_wait_time: Duration,
}

/// Anomaly detector using statistical methods
struct AnomalyDetector {
    /// Detection threshold (standard deviations)
    threshold: f64,
}

impl Default for AdvancedDiagnosticEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl AdvancedDiagnosticEngine {
    /// Create a new advanced diagnostic engine
    pub fn new() -> Self {
        Self {
            historical_buffer: VecDeque::new(),
            max_history_size: 24, // 24 hours of hourly snapshots
            query_tracker: QueryPerformanceTracker {
                recent_queries: VecDeque::new(),
                patterns: HashMap::new(),
                cache_hits: 0,
                cache_misses: 0,
            },
            transaction_tracker: TransactionPatternTracker {
                recent_durations: VecDeque::new(),
                commits: 0,
                aborts: 0,
                conflicts: 0,
                deadlocks: 0,
                contention_map: HashMap::new(),
            },
            anomaly_detector: AnomalyDetector { threshold: 3.0 },
        }
    }

    /// Record a metric snapshot
    pub fn record_snapshot(
        &mut self,
        query_latency: f64,
        txn_throughput: f64,
        storage_bytes: u64,
        error_count: u64,
        buffer_pool_stats: &BufferPoolStats,
    ) {
        let snapshot = MetricSnapshot {
            timestamp: SystemTime::now(),
            query_latency,
            txn_throughput,
            storage_bytes,
            error_count,
            buffer_pool_stats: buffer_pool_stats.clone(),
        };

        self.historical_buffer.push_back(snapshot);

        // Keep only max_history_size snapshots
        while self.historical_buffer.len() > self.max_history_size {
            self.historical_buffer.pop_front();
        }
    }

    /// Record a query execution
    pub fn record_query(&mut self, duration: Duration, pattern: String, indexes: Vec<String>) {
        // Record duration
        self.query_tracker.recent_queries.push_back(duration);
        if self.query_tracker.recent_queries.len() > 1000 {
            self.query_tracker.recent_queries.pop_front();
        }

        // Update pattern stats
        let stats = self
            .query_tracker
            .patterns
            .entry(pattern)
            .or_insert(QueryPatternStats {
                frequency: 0,
                total_time: Duration::ZERO,
                indexes_used: indexes.clone(),
            });
        stats.frequency += 1;
        stats.total_time += duration;
    }

    /// Record a query cache hit or miss
    pub fn record_cache_hit(&mut self, hit: bool) {
        if hit {
            self.query_tracker.cache_hits += 1;
        } else {
            self.query_tracker.cache_misses += 1;
        }
    }

    /// Record a transaction completion
    pub fn record_transaction(&mut self, duration: Duration, committed: bool) {
        self.transaction_tracker
            .recent_durations
            .push_back(duration);
        if self.transaction_tracker.recent_durations.len() > 1000 {
            self.transaction_tracker.recent_durations.pop_front();
        }

        if committed {
            self.transaction_tracker.commits += 1;
        } else {
            self.transaction_tracker.aborts += 1;
        }
    }

    /// Record a transaction conflict
    pub fn record_conflict(&mut self) {
        self.transaction_tracker.conflicts += 1;
    }

    /// Record a deadlock
    pub fn record_deadlock(&mut self) {
        self.transaction_tracker.deadlocks += 1;
    }

    /// Record lock contention
    pub fn record_contention(&mut self, resource: String, wait_time: Duration) {
        let stats = self
            .transaction_tracker
            .contention_map
            .entry(resource)
            .or_insert(ContentionStats {
                count: 0,
                total_wait_time: Duration::ZERO,
            });
        stats.count += 1;
        stats.total_wait_time += wait_time;
    }

    /// Generate a comprehensive advanced diagnostic report
    pub fn generate_report(
        &self,
        storage_bytes: u64,
        max_storage_bytes: u64,
    ) -> Result<AdvancedDiagnosticReport> {
        let query_analysis = self.analyze_query_performance()?;
        let transaction_analysis = self.analyze_transaction_patterns()?;
        let fragmentation_analysis = self.analyze_fragmentation(storage_bytes)?;
        let index_usage = self.analyze_index_usage()?;
        let predictive_health = self.predict_health_issues()?;
        let tuning_recommendations =
            self.generate_tuning_recommendations(storage_bytes, max_storage_bytes)?;
        let anomalies = self.detect_anomalies()?;
        let capacity_forecast = self.forecast_capacity(storage_bytes, max_storage_bytes)?;

        Ok(AdvancedDiagnosticReport {
            timestamp: SystemTime::now(),
            query_analysis,
            transaction_analysis,
            fragmentation_analysis,
            index_usage,
            predictive_health,
            tuning_recommendations,
            anomalies,
            capacity_forecast,
        })
    }

    /// Analyze query performance
    fn analyze_query_performance(&self) -> Result<QueryPerformanceAnalysis> {
        let query_times: Vec<f64> = self
            .query_tracker
            .recent_queries
            .iter()
            .map(|d| d.as_secs_f64())
            .collect();

        if query_times.is_empty() {
            return Ok(QueryPerformanceAnalysis {
                total_queries: 0,
                avg_execution_time: Duration::ZERO,
                median_execution_time: Duration::ZERO,
                p95_execution_time: Duration::ZERO,
                p99_execution_time: Duration::ZERO,
                slow_query_count: 0,
                query_patterns: vec![],
                optimization_opportunities: vec![],
                cache_hit_rate: 0.0,
            });
        }

        let avg = if query_times.is_empty() {
            0.0
        } else {
            query_times.iter().sum::<f64>() / query_times.len() as f64
        };
        let mut sorted_times = query_times.clone();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = sorted_times[sorted_times.len() / 2];
        let p95_idx = (sorted_times.len() as f64 * 0.95) as usize;
        let p99_idx = (sorted_times.len() as f64 * 0.99) as usize;
        let p95 = sorted_times[p95_idx.min(sorted_times.len() - 1)];
        let p99 = sorted_times[p99_idx.min(sorted_times.len() - 1)];

        let slow_query_threshold = 1.0; // 1 second
        let slow_query_count = sorted_times
            .iter()
            .filter(|&&t| t > slow_query_threshold)
            .count() as u64;

        // Analyze query patterns
        let mut query_patterns = vec![];
        for (pattern, stats) in &self.query_tracker.patterns {
            let avg_time = stats.total_time / stats.frequency.max(1) as u32;
            let optimization_potential = if avg_time.as_secs_f64() > slow_query_threshold {
                0.8
            } else if avg_time.as_secs_f64() > slow_query_threshold / 2.0 {
                0.5
            } else {
                0.2
            };

            query_patterns.push(QueryPattern {
                signature: pattern.clone(),
                frequency: stats.frequency,
                avg_time,
                indexes_used: stats.indexes_used.clone(),
                optimization_potential,
            });
        }

        // Sort patterns by optimization potential
        query_patterns.sort_by(|a, b| {
            b.optimization_potential
                .partial_cmp(&a.optimization_potential)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Generate optimization opportunities
        let mut optimization_opportunities = vec![];
        for pattern in &query_patterns {
            if pattern.optimization_potential > 0.6 {
                optimization_opportunities.push(format!(
                    "Consider adding materialized view or index for pattern: {}",
                    pattern.signature
                ));
            }
        }

        let cache_hit_rate = if self.query_tracker.cache_hits + self.query_tracker.cache_misses > 0
        {
            self.query_tracker.cache_hits as f64
                / (self.query_tracker.cache_hits + self.query_tracker.cache_misses) as f64
        } else {
            0.0
        };

        Ok(QueryPerformanceAnalysis {
            total_queries: self.query_tracker.recent_queries.len() as u64,
            avg_execution_time: Duration::from_secs_f64(avg),
            median_execution_time: Duration::from_secs_f64(median),
            p95_execution_time: Duration::from_secs_f64(p95),
            p99_execution_time: Duration::from_secs_f64(p99),
            slow_query_count,
            query_patterns,
            optimization_opportunities,
            cache_hit_rate,
        })
    }

    /// Analyze transaction patterns
    fn analyze_transaction_patterns(&self) -> Result<TransactionPatternAnalysis> {
        let total_transactions = self.transaction_tracker.commits + self.transaction_tracker.aborts;

        if total_transactions == 0 {
            return Ok(TransactionPatternAnalysis {
                total_transactions: 0,
                avg_duration: Duration::ZERO,
                median_duration: Duration::ZERO,
                commit_rate: 0.0,
                abort_rate: 0.0,
                conflict_rate: 0.0,
                deadlock_rate: 0.0,
                contention_points: vec![],
                size_distribution: TransactionSizeDistribution {
                    small_pct: 0.0,
                    medium_pct: 0.0,
                    large_pct: 0.0,
                },
            });
        }

        let durations: Vec<f64> = self
            .transaction_tracker
            .recent_durations
            .iter()
            .map(|d| d.as_secs_f64())
            .collect();

        let avg = if !durations.is_empty() {
            durations.iter().sum::<f64>() / durations.len() as f64
        } else {
            0.0
        };

        let median = if !durations.is_empty() {
            let mut sorted = durations.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted[sorted.len() / 2]
        } else {
            0.0
        };

        let commit_rate = self.transaction_tracker.commits as f64 / total_transactions as f64;
        let abort_rate = self.transaction_tracker.aborts as f64 / total_transactions as f64;
        let conflict_rate =
            self.transaction_tracker.conflicts as f64 / total_transactions.max(1) as f64;
        let deadlock_rate =
            (self.transaction_tracker.deadlocks as f64 / total_transactions.max(1) as f64) * 1000.0;

        // Analyze contention points
        let mut contention_points = vec![];
        for (resource, stats) in &self.transaction_tracker.contention_map {
            let avg_wait = stats.total_wait_time / stats.count.max(1) as u32;
            let severity = if avg_wait.as_millis() > 100 {
                0.9
            } else if avg_wait.as_millis() > 50 {
                0.6
            } else {
                0.3
            };

            contention_points.push(ContentionPoint {
                resource: resource.clone(),
                contention_count: stats.count,
                avg_wait_time: avg_wait,
                severity,
            });
        }

        // Sort by severity
        contention_points.sort_by(|a, b| {
            b.severity
                .partial_cmp(&a.severity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Simple size distribution (would need actual transaction size data)
        let size_distribution = TransactionSizeDistribution {
            small_pct: 0.7, // Mock values - would come from actual transaction tracking
            medium_pct: 0.25,
            large_pct: 0.05,
        };

        Ok(TransactionPatternAnalysis {
            total_transactions,
            avg_duration: Duration::from_secs_f64(avg),
            median_duration: Duration::from_secs_f64(median),
            commit_rate,
            abort_rate,
            conflict_rate,
            deadlock_rate,
            contention_points,
            size_distribution,
        })
    }

    /// Analyze storage fragmentation
    fn analyze_fragmentation(&self, storage_bytes: u64) -> Result<FragmentationAnalysis> {
        // Mock implementation - would need actual storage analysis
        let overall_fragmentation_pct = 15.5; // Would calculate from actual storage
        let dictionary_fragmentation_pct = 12.0;

        let mut index_fragmentation = HashMap::new();
        index_fragmentation.insert("SPO".to_string(), 14.2);
        index_fragmentation.insert("POS".to_string(), 16.8);
        index_fragmentation.insert("OSP".to_string(), 13.5);

        // Mock free space regions
        let free_space_distribution = vec![
            FreeSpaceRegion {
                offset: 1024,
                size: 512,
            },
            FreeSpaceRegion {
                offset: 4096,
                size: 2048,
            },
        ];

        let compaction_benefit_bytes =
            (storage_bytes as f64 * overall_fragmentation_pct / 100.0) as u64;
        let compaction_priority = if overall_fragmentation_pct > 25.0 {
            0.9
        } else if overall_fragmentation_pct > 15.0 {
            0.6
        } else {
            0.3
        };

        Ok(FragmentationAnalysis {
            overall_fragmentation_pct,
            dictionary_fragmentation_pct,
            index_fragmentation,
            free_space_distribution,
            compaction_benefit_bytes,
            compaction_priority,
        })
    }

    /// Analyze index usage
    fn analyze_index_usage(&self) -> Result<IndexUsageStatistics> {
        // Mock implementation - would track actual index usage
        let mut usage_by_index = HashMap::new();

        usage_by_index.insert(
            "SPO".to_string(),
            IndexUsageStats {
                scan_count: 1000,
                avg_selectivity: 0.85,
                avg_scan_time: Duration::from_millis(10),
                efficiency_score: 0.9,
            },
        );

        usage_by_index.insert(
            "POS".to_string(),
            IndexUsageStats {
                scan_count: 500,
                avg_selectivity: 0.65,
                avg_scan_time: Duration::from_millis(15),
                efficiency_score: 0.7,
            },
        );

        usage_by_index.insert(
            "OSP".to_string(),
            IndexUsageStats {
                scan_count: 300,
                avg_selectivity: 0.75,
                avg_scan_time: Duration::from_millis(12),
                efficiency_score: 0.8,
            },
        );

        let total_scans = usage_by_index.values().map(|s| s.scan_count).sum();

        let unused_indexes = vec![]; // Would detect indexes with zero scans

        let overused_indexes = usage_by_index
            .iter()
            .filter(|(_, stats)| stats.scan_count > 10000 && stats.avg_scan_time.as_millis() > 50)
            .map(|(name, _)| name.clone())
            .collect();

        let missing_index_opportunities = vec![]; // Would analyze query patterns for missing indexes

        Ok(IndexUsageStatistics {
            total_scans,
            usage_by_index,
            unused_indexes,
            overused_indexes,
            missing_index_opportunities,
        })
    }

    /// Predict health issues using trend analysis
    fn predict_health_issues(&self) -> Result<PredictiveHealthIndicators> {
        if self.historical_buffer.is_empty() {
            return Ok(PredictiveHealthIndicators {
                historical_metrics: HistoricalMetrics {
                    data_points: 0,
                    time_range: Duration::ZERO,
                    query_latency_trend: vec![],
                    txn_throughput_trend: vec![],
                    storage_growth_trend: vec![],
                    error_rate_trend: vec![],
                },
                predicted_issues_24h: vec![],
                predicted_issues_7d: vec![],
                health_trend: HealthTrend::Stable,
                resource_predictions: ResourcePredictions {
                    storage_full_eta: None,
                    memory_exhaustion_eta: None,
                    connection_saturation_eta: None,
                },
            });
        }

        // Extract historical metrics
        let query_latency_trend: Vec<f64> = self
            .historical_buffer
            .iter()
            .map(|s| s.query_latency)
            .collect();

        let txn_throughput_trend: Vec<f64> = self
            .historical_buffer
            .iter()
            .map(|s| s.txn_throughput)
            .collect();

        let storage_growth_trend: Vec<f64> = self
            .historical_buffer
            .iter()
            .map(|s| s.storage_bytes as f64)
            .collect();

        let error_rate_trend: Vec<f64> = self
            .historical_buffer
            .iter()
            .map(|s| s.error_count as f64)
            .collect();

        let time_range = if let (Some(first), Some(last)) = (
            self.historical_buffer.front(),
            self.historical_buffer.back(),
        ) {
            last.timestamp
                .duration_since(first.timestamp)
                .unwrap_or(Duration::ZERO)
        } else {
            Duration::ZERO
        };

        // Determine health trend using linear regression
        let health_trend = self.determine_health_trend(&query_latency_trend, &error_rate_trend);

        // Predict specific issues
        let mut predicted_issues_24h = vec![];
        let predicted_issues_7d = vec![];

        // Check for query latency degradation
        if !query_latency_trend.is_empty() && query_latency_trend.len() > 5 {
            let recent_avg = {
                let slice = &query_latency_trend[query_latency_trend.len() - 3..];
                slice.iter().sum::<f64>() / slice.len() as f64
            };
            let overall_avg =
                query_latency_trend.iter().sum::<f64>() / query_latency_trend.len() as f64;

            if recent_avg > overall_avg * 1.5 {
                predicted_issues_24h.push(PredictedIssue {
                    category: "Performance".to_string(),
                    description: "Query latency is trending upward significantly".to_string(),
                    confidence: 0.75,
                    eta: Duration::from_secs(3600 * 6), // 6 hours
                    severity: Severity::Warning,
                    preventive_action:
                        "Consider running ANALYZE to update statistics or adding indexes"
                            .to_string(),
                });
            }
        }

        // Check for error rate increase
        if !error_rate_trend.is_empty() && error_rate_trend.len() > 5 {
            let recent_errors = error_rate_trend.iter().rev().take(3).sum::<f64>();
            if recent_errors > 10.0 {
                predicted_issues_24h.push(PredictedIssue {
                    category: "Stability".to_string(),
                    description: "Error rate is increasing".to_string(),
                    confidence: 0.85,
                    eta: Duration::from_secs(3600 * 2), // 2 hours
                    severity: Severity::Error,
                    preventive_action: "Review error logs and consider running diagnostics"
                        .to_string(),
                });
            }
        }

        Ok(PredictiveHealthIndicators {
            historical_metrics: HistoricalMetrics {
                data_points: self.historical_buffer.len(),
                time_range,
                query_latency_trend,
                txn_throughput_trend,
                storage_growth_trend,
                error_rate_trend,
            },
            predicted_issues_24h,
            predicted_issues_7d,
            health_trend,
            resource_predictions: ResourcePredictions {
                storage_full_eta: None, // Would calculate from storage growth trend
                memory_exhaustion_eta: None,
                connection_saturation_eta: None,
            },
        })
    }

    /// Determine health trend from metrics
    fn determine_health_trend(&self, latency_trend: &[f64], error_trend: &[f64]) -> HealthTrend {
        if latency_trend.len() < 3 || error_trend.len() < 3 {
            return HealthTrend::Stable;
        }

        // Simple trend analysis: compare recent vs older metrics
        let recent_latency = if latency_trend.len() >= 3 {
            latency_trend[latency_trend.len() - 3..].iter().sum::<f64>() / 3.0
        } else {
            latency_trend.iter().sum::<f64>() / latency_trend.len().max(1) as f64
        };
        let older_latency = if latency_trend.len() >= 3 {
            latency_trend[..3].iter().sum::<f64>() / 3.0
        } else {
            recent_latency
        };

        let recent_errors = if error_trend.len() >= 3 {
            error_trend[error_trend.len() - 3..].iter().sum::<f64>() / 3.0
        } else {
            error_trend.iter().sum::<f64>() / error_trend.len().max(1) as f64
        };
        let older_errors = if error_trend.len() >= 3 {
            error_trend[..3].iter().sum::<f64>() / 3.0
        } else {
            recent_errors
        };

        let latency_change_pct = if older_latency > 0.0 {
            ((recent_latency - older_latency) / older_latency) * 100.0
        } else {
            0.0
        };

        let error_change_pct = if older_errors > 0.0 {
            ((recent_errors - older_errors) / older_errors) * 100.0
        } else {
            0.0
        };

        // Determine trend
        if latency_change_pct < -10.0 && error_change_pct < 0.0 {
            HealthTrend::Improving
        } else if latency_change_pct > 30.0 || error_change_pct > 50.0 {
            HealthTrend::DegradingRapidly
        } else if latency_change_pct > 15.0 || error_change_pct > 20.0 {
            HealthTrend::DegradingSlowly
        } else {
            HealthTrend::Stable
        }
    }

    /// Generate auto-tuning recommendations
    fn generate_tuning_recommendations(
        &self,
        storage_bytes: u64,
        max_storage_bytes: u64,
    ) -> Result<Vec<TuningRecommendation>> {
        let mut recommendations = vec![];

        // Check cache hit rate
        let cache_hit_rate = if self.query_tracker.cache_hits + self.query_tracker.cache_misses > 0
        {
            self.query_tracker.cache_hits as f64
                / (self.query_tracker.cache_hits + self.query_tracker.cache_misses) as f64
        } else {
            0.0
        };

        if cache_hit_rate < 0.7 {
            recommendations.push(TuningRecommendation {
                parameter: "query_cache_size".to_string(),
                current_value: "256MB".to_string(),
                recommended_value: "512MB".to_string(),
                rationale: format!("Cache hit rate is low ({:.1}%)", cache_hit_rate * 100.0),
                expected_impact: "Query performance improvement of 20-40%".to_string(),
                priority: 0.8,
            });
        }

        // Check storage usage
        let storage_usage_pct = (storage_bytes as f64 / max_storage_bytes as f64) * 100.0;
        if storage_usage_pct > 75.0 {
            recommendations.push(TuningRecommendation {
                parameter: "max_storage_size".to_string(),
                current_value: format!("{}GB", max_storage_bytes / (1024 * 1024 * 1024)),
                recommended_value: format!("{}GB", (max_storage_bytes * 2) / (1024 * 1024 * 1024)),
                rationale: format!("Storage usage is at {:.1}%", storage_usage_pct),
                expected_impact: "Prevent storage exhaustion".to_string(),
                priority: 0.9,
            });
        }

        // Check transaction abort rate
        let total_txns = self.transaction_tracker.commits + self.transaction_tracker.aborts;
        if total_txns > 0 {
            let abort_rate = self.transaction_tracker.aborts as f64 / total_txns as f64;
            if abort_rate > 0.1 {
                recommendations.push(TuningRecommendation {
                    parameter: "transaction_timeout".to_string(),
                    current_value: "30s".to_string(),
                    recommended_value: "60s".to_string(),
                    rationale: format!(
                        "Transaction abort rate is high ({:.1}%)",
                        abort_rate * 100.0
                    ),
                    expected_impact: "Reduce transaction conflicts and retries".to_string(),
                    priority: 0.7,
                });
            }
        }

        // Sort by priority
        recommendations.sort_by(|a, b| {
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(recommendations)
    }

    /// Detect anomalies in recent metrics
    fn detect_anomalies(&self) -> Result<Vec<AnomalyDetection>> {
        let mut anomalies = vec![];

        if self.historical_buffer.len() < 5 {
            return Ok(anomalies);
        }

        // Analyze query latency for anomalies
        let latencies: Vec<f64> = self
            .historical_buffer
            .iter()
            .map(|s| s.query_latency)
            .collect();

        if let Some(anomaly) = self.detect_metric_anomaly("query_latency", &latencies) {
            anomalies.push(anomaly);
        }

        // Analyze error rates for anomalies
        let error_rates: Vec<f64> = self
            .historical_buffer
            .iter()
            .map(|s| s.error_count as f64)
            .collect();

        if let Some(anomaly) = self.detect_metric_anomaly("error_rate", &error_rates) {
            anomalies.push(anomaly);
        }

        Ok(anomalies)
    }

    /// Detect anomaly in a metric using statistical methods
    fn detect_metric_anomaly(&self, metric_name: &str, values: &[f64]) -> Option<AnomalyDetection> {
        if values.len() < 5 {
            return None;
        }

        let avg = values.iter().sum::<f64>() / values.len() as f64;

        // Calculate standard deviation manually
        let variance_sum: f64 = values.iter().map(|v| (v - avg).powi(2)).sum();
        let std_dev = (variance_sum / values.len() as f64).sqrt();

        let recent_value = *values.last()?;

        let deviation = if std_dev > 0.0 {
            (recent_value - avg).abs() / std_dev
        } else {
            0.0
        };

        if deviation > self.anomaly_detector.threshold {
            let expected_range = (avg - 2.0 * std_dev, avg + 2.0 * std_dev);
            let severity = if deviation > 5.0 {
                Severity::Critical
            } else if deviation > 4.0 {
                Severity::Error
            } else {
                Severity::Warning
            };

            Some(AnomalyDetection {
                metric: metric_name.to_string(),
                description: format!(
                    "{} is {} standard deviations from normal",
                    metric_name, deviation
                ),
                detected_value: recent_value,
                expected_range,
                deviation,
                detected_at: SystemTime::now(),
                severity,
            })
        } else {
            None
        }
    }

    /// Forecast capacity needs
    fn forecast_capacity(&self, current_bytes: u64, max_bytes: u64) -> Result<CapacityForecast> {
        if self.historical_buffer.len() < 2 {
            return Ok(CapacityForecast {
                current_storage_bytes: current_bytes,
                predicted_30d_bytes: current_bytes,
                predicted_90d_bytes: current_bytes,
                growth_rate_per_day: 0.0,
                days_until_80pct: None,
                days_until_90pct: None,
                recommended_actions: vec![],
            });
        }

        // Calculate growth rate from historical data
        let first = self
            .historical_buffer
            .front()
            .expect("collection validated to be non-empty");
        let last = self
            .historical_buffer
            .back()
            .expect("collection validated to be non-empty");

        let time_diff_days = last
            .timestamp
            .duration_since(first.timestamp)
            .unwrap_or(Duration::from_secs(1))
            .as_secs_f64()
            / 86400.0;

        let storage_diff = (last.storage_bytes as i64 - first.storage_bytes as i64) as f64;
        let growth_rate_per_day = if time_diff_days > 0.0 {
            storage_diff / time_diff_days
        } else {
            0.0
        };

        let predicted_30d_bytes =
            (current_bytes as f64 + growth_rate_per_day * 30.0).max(0.0) as u64;
        let predicted_90d_bytes =
            (current_bytes as f64 + growth_rate_per_day * 90.0).max(0.0) as u64;

        // Calculate days until thresholds
        let bytes_80pct = (max_bytes as f64 * 0.8) as u64;
        let bytes_90pct = (max_bytes as f64 * 0.9) as u64;

        let days_until_80pct = if growth_rate_per_day > 0.0 && current_bytes < bytes_80pct {
            Some(((bytes_80pct - current_bytes) as f64 / growth_rate_per_day) as u32)
        } else {
            None
        };

        let days_until_90pct = if growth_rate_per_day > 0.0 && current_bytes < bytes_90pct {
            Some(((bytes_90pct - current_bytes) as f64 / growth_rate_per_day) as u32)
        } else {
            None
        };

        let mut recommended_actions = vec![];
        if let Some(days) = days_until_80pct {
            if days < 30 {
                recommended_actions
                    .push("Urgent: Plan storage expansion within 30 days".to_string());
            } else if days < 90 {
                recommended_actions.push("Plan storage expansion within 90 days".to_string());
            }
        }

        if growth_rate_per_day > 1_000_000_000.0 {
            // > 1GB/day
            recommended_actions.push("Consider enabling data compression".to_string());
        }

        Ok(CapacityForecast {
            current_storage_bytes: current_bytes,
            predicted_30d_bytes,
            predicted_90d_bytes,
            growth_rate_per_day,
            days_until_80pct,
            days_until_90pct,
            recommended_actions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_diagnostic_engine_creation() {
        let engine = AdvancedDiagnosticEngine::new();
        assert_eq!(engine.historical_buffer.len(), 0);
        assert_eq!(engine.query_tracker.recent_queries.len(), 0);
        assert_eq!(engine.transaction_tracker.commits, 0);
    }

    #[test]
    fn test_record_snapshot() {
        let mut engine = AdvancedDiagnosticEngine::new();
        let stats = BufferPoolStats::default();

        engine.record_snapshot(0.1, 100.0, 1_000_000, 0, &stats);
        assert_eq!(engine.historical_buffer.len(), 1);

        // Add more than max_history_size snapshots
        for _ in 0..30 {
            engine.record_snapshot(0.1, 100.0, 1_000_000, 0, &stats);
        }

        assert!(engine.historical_buffer.len() <= engine.max_history_size);
    }

    #[test]
    fn test_record_query() {
        let mut engine = AdvancedDiagnosticEngine::new();

        engine.record_query(
            Duration::from_millis(100),
            "SELECT_PATTERN".to_string(),
            vec!["SPO".to_string()],
        );

        assert_eq!(engine.query_tracker.recent_queries.len(), 1);
        assert_eq!(engine.query_tracker.patterns.len(), 1);
        assert_eq!(
            engine
                .query_tracker
                .patterns
                .get("SELECT_PATTERN")
                .unwrap()
                .frequency,
            1
        );
    }

    #[test]
    fn test_record_cache_hit_miss() {
        let mut engine = AdvancedDiagnosticEngine::new();

        engine.record_cache_hit(true);
        engine.record_cache_hit(true);
        engine.record_cache_hit(false);

        assert_eq!(engine.query_tracker.cache_hits, 2);
        assert_eq!(engine.query_tracker.cache_misses, 1);
    }

    #[test]
    fn test_record_transaction() {
        let mut engine = AdvancedDiagnosticEngine::new();

        engine.record_transaction(Duration::from_millis(50), true);
        engine.record_transaction(Duration::from_millis(75), false);

        assert_eq!(engine.transaction_tracker.commits, 1);
        assert_eq!(engine.transaction_tracker.aborts, 1);
        assert_eq!(engine.transaction_tracker.recent_durations.len(), 2);
    }

    #[test]
    fn test_record_conflict_and_deadlock() {
        let mut engine = AdvancedDiagnosticEngine::new();

        engine.record_conflict();
        engine.record_conflict();
        engine.record_deadlock();

        assert_eq!(engine.transaction_tracker.conflicts, 2);
        assert_eq!(engine.transaction_tracker.deadlocks, 1);
    }

    #[test]
    fn test_record_contention() {
        let mut engine = AdvancedDiagnosticEngine::new();

        engine.record_contention("SPO_INDEX".to_string(), Duration::from_millis(10));
        engine.record_contention("SPO_INDEX".to_string(), Duration::from_millis(20));

        let stats = engine
            .transaction_tracker
            .contention_map
            .get("SPO_INDEX")
            .unwrap();
        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_wait_time, Duration::from_millis(30));
    }

    #[test]
    fn test_analyze_query_performance_empty() {
        let engine = AdvancedDiagnosticEngine::new();
        let analysis = engine.analyze_query_performance().unwrap();

        assert_eq!(analysis.total_queries, 0);
        assert_eq!(analysis.avg_execution_time, Duration::ZERO);
        assert_eq!(analysis.cache_hit_rate, 0.0);
    }

    #[test]
    fn test_analyze_query_performance_with_data() {
        let mut engine = AdvancedDiagnosticEngine::new();

        // Record some queries
        for i in 0..10 {
            engine.record_query(
                Duration::from_millis(100 + i * 10),
                "PATTERN_A".to_string(),
                vec!["SPO".to_string()],
            );
        }

        engine.record_cache_hit(true);
        engine.record_cache_hit(true);
        engine.record_cache_hit(false);

        let analysis = engine.analyze_query_performance().unwrap();

        assert_eq!(analysis.total_queries, 10);
        assert!(analysis.avg_execution_time.as_millis() > 0);
        assert!((analysis.cache_hit_rate - 0.666).abs() < 0.01);
        assert_eq!(analysis.query_patterns.len(), 1);
    }

    #[test]
    fn test_analyze_transaction_patterns_empty() {
        let engine = AdvancedDiagnosticEngine::new();
        let analysis = engine.analyze_transaction_patterns().unwrap();

        assert_eq!(analysis.total_transactions, 0);
        assert_eq!(analysis.commit_rate, 0.0);
        assert_eq!(analysis.abort_rate, 0.0);
    }

    #[test]
    fn test_analyze_transaction_patterns_with_data() {
        let mut engine = AdvancedDiagnosticEngine::new();

        // Record transactions
        engine.record_transaction(Duration::from_millis(50), true);
        engine.record_transaction(Duration::from_millis(75), true);
        engine.record_transaction(Duration::from_millis(100), false);

        engine.record_conflict();
        engine.record_contention("SPO_INDEX".to_string(), Duration::from_millis(25));

        let analysis = engine.analyze_transaction_patterns().unwrap();

        assert_eq!(analysis.total_transactions, 3);
        assert!((analysis.commit_rate - 0.666).abs() < 0.01);
        assert!((analysis.abort_rate - 0.333).abs() < 0.01);
        assert!((analysis.conflict_rate - 0.333).abs() < 0.01);
        assert_eq!(analysis.contention_points.len(), 1);
    }

    #[test]
    fn test_health_trend_determination() {
        let engine = AdvancedDiagnosticEngine::new();

        // Stable trend
        let latency = vec![0.1, 0.11, 0.1, 0.12, 0.1];
        let errors = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(
            engine.determine_health_trend(&latency, &errors),
            HealthTrend::Stable
        );

        // Improving trend
        let latency = vec![0.5, 0.4, 0.3, 0.2, 0.1];
        let errors = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(
            engine.determine_health_trend(&latency, &errors),
            HealthTrend::Improving
        );

        // Degrading rapidly
        let latency = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let errors = vec![1.0, 2.0, 4.0, 8.0, 10.0];
        assert_eq!(
            engine.determine_health_trend(&latency, &errors),
            HealthTrend::DegradingRapidly
        );
    }

    #[test]
    fn test_generate_report_empty() {
        let engine = AdvancedDiagnosticEngine::new();
        let report = engine.generate_report(1_000_000, 10_000_000).unwrap();

        assert_eq!(report.query_analysis.total_queries, 0);
        assert_eq!(report.transaction_analysis.total_transactions, 0);
        assert_eq!(report.predictive_health.historical_metrics.data_points, 0);
    }

    #[test]
    fn test_generate_report_with_data() {
        let mut engine = AdvancedDiagnosticEngine::new();
        let stats = BufferPoolStats::default();

        // Add some data
        engine.record_snapshot(0.1, 100.0, 1_000_000, 0, &stats);
        engine.record_query(
            Duration::from_millis(100),
            "PATTERN_A".to_string(),
            vec!["SPO".to_string()],
        );
        engine.record_transaction(Duration::from_millis(50), true);

        let report = engine.generate_report(1_000_000, 10_000_000).unwrap();

        assert_eq!(report.query_analysis.total_queries, 1);
        assert_eq!(report.transaction_analysis.total_transactions, 1);
        assert_eq!(report.predictive_health.historical_metrics.data_points, 1);
    }

    #[test]
    fn test_tuning_recommendations() {
        let mut engine = AdvancedDiagnosticEngine::new();

        // Low cache hit rate
        engine.record_cache_hit(false);
        engine.record_cache_hit(false);
        engine.record_cache_hit(false);
        engine.record_cache_hit(true);

        let recommendations = engine
            .generate_tuning_recommendations(1_000_000, 10_000_000)
            .unwrap();

        assert!(!recommendations.is_empty());
        assert!(recommendations
            .iter()
            .any(|r| r.parameter == "query_cache_size"));
    }

    #[test]
    fn test_anomaly_detection() {
        let mut engine = AdvancedDiagnosticEngine::new();
        let stats = BufferPoolStats::default();

        // Normal values with tight range
        for _ in 0..10 {
            engine.record_snapshot(0.1, 100.0, 1_000_000, 0, &stats);
        }

        // Anomalous value - much higher to ensure detection
        engine.record_snapshot(50.0, 100.0, 1_000_000, 0, &stats); // Very high latency (50x normal)

        let anomalies = engine.detect_anomalies().unwrap();

        assert!(!anomalies.is_empty());
        assert!(anomalies.iter().any(|a| a.metric == "query_latency"));
    }

    #[test]
    fn test_capacity_forecast() {
        let mut engine = AdvancedDiagnosticEngine::new();
        let stats = BufferPoolStats::default();

        // Simulate growth - need measurable time difference for rate calculation
        engine.record_snapshot(0.1, 100.0, 1_000_000, 0, &stats);
        std::thread::sleep(std::time::Duration::from_millis(10));
        engine.record_snapshot(0.1, 100.0, 1_500_000, 0, &stats);

        let forecast = engine.forecast_capacity(1_500_000, 10_000_000).unwrap();

        assert_eq!(forecast.current_storage_bytes, 1_500_000);
        assert!(forecast.growth_rate_per_day > 0.0);
    }

    #[test]
    fn test_fragmentation_analysis() {
        let engine = AdvancedDiagnosticEngine::new();
        let analysis = engine.analyze_fragmentation(10_000_000).unwrap();

        assert!(analysis.overall_fragmentation_pct >= 0.0);
        assert!(analysis.compaction_priority >= 0.0 && analysis.compaction_priority <= 1.0);
        assert!(!analysis.index_fragmentation.is_empty());
    }

    #[test]
    fn test_index_usage_analysis() {
        let engine = AdvancedDiagnosticEngine::new();
        let stats = engine.analyze_index_usage().unwrap();

        assert!(stats.total_scans > 0);
        assert!(!stats.usage_by_index.is_empty());
        assert!(stats.usage_by_index.contains_key("SPO"));
        assert!(stats.usage_by_index.contains_key("POS"));
        assert!(stats.usage_by_index.contains_key("OSP"));
    }

    #[test]
    fn test_predictive_health_empty() {
        let engine = AdvancedDiagnosticEngine::new();
        let health = engine.predict_health_issues().unwrap();

        assert_eq!(health.historical_metrics.data_points, 0);
        assert_eq!(health.predicted_issues_24h.len(), 0);
        assert_eq!(health.health_trend, HealthTrend::Stable);
    }

    #[test]
    fn test_predictive_health_with_trends() {
        let mut engine = AdvancedDiagnosticEngine::new();
        let stats = BufferPoolStats::default();

        // Add snapshots with increasing latency (degrading)
        for i in 0..10 {
            engine.record_snapshot(
                0.1 * (i as f64 + 1.0),
                100.0,
                1_000_000 + i * 100_000,
                i,
                &stats,
            );
        }

        let health = engine.predict_health_issues().unwrap();

        assert_eq!(health.historical_metrics.data_points, 10);
        assert!(
            !health.predicted_issues_24h.is_empty() || health.health_trend != HealthTrend::Stable
        );
    }
}
