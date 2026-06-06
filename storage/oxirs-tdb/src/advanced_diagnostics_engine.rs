//! Core engine for advanced TDB diagnostics.
//!
//! This module implements the [`AdvancedDiagnosticEngine`]: metric recording,
//! comprehensive report generation, query/transaction analysis, fragmentation
//! and index-usage analysis, predictive health monitoring, auto-tuning
//! recommendations, anomaly detection, and capacity forecasting.

use crate::advanced_diagnostics_types::{
    AdvancedDiagnosticEngine, AdvancedDiagnosticReport, AnomalyDetection, AnomalyDetector,
    CapacityForecast, ContentionPoint, ContentionStats, FragmentationAnalysis, FreeSpaceRegion,
    HealthTrend, HistoricalMetrics, IndexUsageStatistics, IndexUsageStats, MetricSnapshot,
    MissingIndexOpportunity, PredictedIssue, PredictiveHealthIndicators, QueryPattern,
    QueryPatternStats, QueryPerformanceAnalysis, QueryPerformanceTracker, ResourcePredictions,
    TransactionPatternAnalysis, TransactionPatternTracker, TransactionSizeDistribution,
    TuningRecommendation,
};
use crate::diagnostics::Severity;
use crate::error::Result;
use crate::storage::BufferPoolStats;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

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
    pub(crate) fn analyze_query_performance(&self) -> Result<QueryPerformanceAnalysis> {
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
    pub(crate) fn analyze_transaction_patterns(&self) -> Result<TransactionPatternAnalysis> {
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
    pub(crate) fn analyze_fragmentation(
        &self,
        storage_bytes: u64,
    ) -> Result<FragmentationAnalysis> {
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
    pub(crate) fn analyze_index_usage(&self) -> Result<IndexUsageStatistics> {
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
    pub(crate) fn predict_health_issues(&self) -> Result<PredictiveHealthIndicators> {
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
    pub(crate) fn determine_health_trend(
        &self,
        latency_trend: &[f64],
        error_trend: &[f64],
    ) -> HealthTrend {
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
    pub(crate) fn generate_tuning_recommendations(
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
    pub(crate) fn detect_anomalies(&self) -> Result<Vec<AnomalyDetection>> {
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
    pub(crate) fn detect_metric_anomaly(
        &self,
        metric_name: &str,
        values: &[f64],
    ) -> Option<AnomalyDetection> {
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
    pub(crate) fn forecast_capacity(
        &self,
        current_bytes: u64,
        max_bytes: u64,
    ) -> Result<CapacityForecast> {
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
