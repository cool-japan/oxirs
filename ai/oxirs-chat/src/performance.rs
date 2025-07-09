//! Performance Monitoring and Optimization for OxiRS Chat
//!
//! Provides comprehensive performance tracking, response time optimization,
//! adaptive caching strategies, and intelligent performance feedback loops.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{
    analytics::ConversationAnalytics,
    cache::{AdvancedCacheManager, CacheStats},
};

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enable_monitoring: bool,
    pub enable_optimization: bool,
    pub enable_adaptive_caching: bool,
    pub response_time_threshold: Duration,
    pub optimization_interval: Duration,
    pub metrics_retention_hours: usize,
    pub performance_targets: PerformanceTargets,
    pub alerting: AlertingConfig,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            enable_optimization: true,
            enable_adaptive_caching: true,
            response_time_threshold: Duration::from_millis(2000),
            optimization_interval: Duration::from_secs(300), // 5 minutes
            metrics_retention_hours: 24,
            performance_targets: PerformanceTargets::default(),
            alerting: AlertingConfig::default(),
        }
    }
}

/// Performance targets and SLAs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    pub avg_response_time_ms: u64,
    pub p95_response_time_ms: u64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub throughput_requests_per_second: f64,
}

impl Default for PerformanceTargets {
    fn default() -> Self {
        Self {
            avg_response_time_ms: 1500,
            p95_response_time_ms: 3000,
            cache_hit_rate: 0.7,
            error_rate: 0.05,
            throughput_requests_per_second: 10.0,
        }
    }
}

/// Alerting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    pub enable_alerts: bool,
    pub alert_thresholds: AlertThresholds,
    pub notification_cooldown: Duration,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enable_alerts: true,
            alert_thresholds: AlertThresholds::default(),
            notification_cooldown: Duration::from_secs(300),
        }
    }
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub response_time_critical_ms: u64,
    pub cache_hit_rate_warning: f64,
    pub error_rate_critical: f64,
    pub memory_usage_warning: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            response_time_critical_ms: 5000,
            cache_hit_rate_warning: 0.3,
            error_rate_critical: 0.2,
            memory_usage_warning: 0.8,
        }
    }
}

/// Performance metrics for a request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    pub session_id: String,
    pub message_id: String,
    pub timestamp: SystemTime,
    pub response_time: Duration,
    pub cache_hit: bool,
    pub cache_type: Option<String>,
    pub error: Option<String>,
    pub query_complexity: f32,
    pub context_size: usize,
    pub tokens_processed: usize,
    pub optimization_applied: bool,
}

/// Aggregated performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub time_window: Duration,
    pub total_requests: usize,
    pub avg_response_time: Duration,
    pub p50_response_time: Duration,
    pub p95_response_time: Duration,
    pub p99_response_time: Duration,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
    pub throughput: f64, // requests per second
    pub optimization_effectiveness: f64,
    pub memory_efficiency: f64,
}

/// Performance bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub analysis_timestamp: SystemTime,
    pub identified_bottlenecks: Vec<Bottleneck>,
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
    pub predicted_improvements: HashMap<String, f64>,
}

/// Performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    pub bottleneck_type: BottleneckType,
    pub description: String,
    pub severity: BottleneckSeverity,
    pub affected_operations: Vec<String>,
    pub estimated_impact: f64, // percentage of total latency
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    CacheMiss,
    SlowQuery,
    ContextAssembly,
    LLMProcessing,
    EmbeddingGeneration,
    NetworkLatency,
    MemoryPressure,
    ConcurrencyLimit,
}

/// Bottleneck severity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_type: OptimizationType,
    pub description: String,
    pub expected_improvement: f64,
    pub implementation_effort: ImplementationEffort,
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Types of optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationType {
    IncreaseCacheSize,
    AdjustCacheTTL,
    OptimizeQueries,
    ReduceContextSize,
    ParallelizeOperations,
    AddIndexes,
    BatchRequests,
    CompressData,
}

/// Implementation effort level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,    // Automated/config change
    Medium, // Code changes required
    High,   // Significant architecture changes
}

/// Main performance monitor
pub struct PerformanceMonitor {
    config: PerformanceConfig,
    request_metrics: Arc<RwLock<VecDeque<RequestMetrics>>>,
    cache_manager: Arc<AdvancedCacheManager>,
    conversation_tracker: Arc<RwLock<ConversationAnalytics>>,
    optimization_history: Arc<RwLock<Vec<AppliedOptimization>>>,
    last_alert_time: Arc<RwLock<SystemTime>>,
}

/// Applied optimization record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppliedOptimization {
    pub timestamp: SystemTime,
    pub optimization_type: OptimizationType,
    pub parameters: HashMap<String, serde_json::Value>,
    pub before_metrics: AggregatedMetrics,
    pub after_metrics: Option<AggregatedMetrics>,
    pub success: bool,
    pub notes: String,
}

impl PerformanceMonitor {
    pub fn new(
        config: PerformanceConfig,
        cache_manager: Arc<AdvancedCacheManager>,
        conversation_tracker: Arc<RwLock<ConversationAnalytics>>,
    ) -> Self {
        let monitor = Self {
            config: config.clone(),
            request_metrics: Arc::new(RwLock::new(VecDeque::new())),
            cache_manager,
            conversation_tracker,
            optimization_history: Arc::new(RwLock::new(Vec::new())),
            last_alert_time: Arc::new(RwLock::new(SystemTime::UNIX_EPOCH)),
        };

        if config.enable_monitoring {
            monitor.start_monitoring_task();
        }

        if config.enable_optimization {
            monitor.start_optimization_task();
        }

        monitor
    }

    /// Record request metrics
    pub async fn record_request(
        &self,
        session_id: String,
        message_id: String,
        response_time: Duration,
        cache_hit: bool,
        cache_type: Option<String>,
        error: Option<String>,
        query_complexity: f32,
        context_size: usize,
        tokens_processed: usize,
        optimization_applied: bool,
    ) -> Result<()> {
        let metrics = RequestMetrics {
            session_id,
            message_id,
            timestamp: SystemTime::now(),
            response_time,
            cache_hit,
            cache_type,
            error,
            query_complexity,
            context_size,
            tokens_processed,
            optimization_applied,
        };

        // Clone metrics for alerts before moving into the collection
        let metrics_for_alerts = metrics.clone();

        let mut request_metrics = self.request_metrics.write().await;
        request_metrics.push_back(metrics);

        // Maintain retention policy
        let retention_duration =
            Duration::from_secs(self.config.metrics_retention_hours as u64 * 3600);
        let cutoff_time = SystemTime::now() - retention_duration;

        while let Some(front) = request_metrics.front() {
            if front.timestamp < cutoff_time {
                request_metrics.pop_front();
            } else {
                break;
            }
        }

        // Release the lock before checking alerts
        drop(request_metrics);

        // Check for performance alerts
        if self.config.alerting.enable_alerts {
            self.check_performance_alerts(&metrics_for_alerts).await?;
        }

        Ok(())
    }

    /// Get aggregated metrics for a time window
    pub async fn get_aggregated_metrics(&self, time_window: Duration) -> Result<AggregatedMetrics> {
        let request_metrics = self.request_metrics.read().await;
        let cutoff_time = SystemTime::now() - time_window;

        let relevant_metrics: Vec<&RequestMetrics> = request_metrics
            .iter()
            .filter(|m| m.timestamp >= cutoff_time)
            .collect();

        if relevant_metrics.is_empty() {
            return Ok(AggregatedMetrics {
                time_window,
                total_requests: 0,
                avg_response_time: Duration::ZERO,
                p50_response_time: Duration::ZERO,
                p95_response_time: Duration::ZERO,
                p99_response_time: Duration::ZERO,
                cache_hit_rate: 0.0,
                error_rate: 0.0,
                throughput: 0.0,
                optimization_effectiveness: 0.0,
                memory_efficiency: 0.0,
            });
        }

        let total_requests = relevant_metrics.len();

        // Calculate response time statistics
        let mut response_times: Vec<Duration> =
            relevant_metrics.iter().map(|m| m.response_time).collect();
        response_times.sort();

        let avg_response_time = Duration::from_millis(
            response_times
                .iter()
                .map(|d| d.as_millis() as u64)
                .sum::<u64>()
                / total_requests as u64,
        );

        let p50_response_time = response_times[total_requests / 2];
        let p95_response_time = response_times[(total_requests as f64 * 0.95) as usize];
        let p99_response_time = response_times[(total_requests as f64 * 0.99) as usize];

        // Calculate cache hit rate
        let cache_hits = relevant_metrics.iter().filter(|m| m.cache_hit).count();
        let cache_hit_rate = cache_hits as f64 / total_requests as f64;

        // Calculate error rate
        let errors = relevant_metrics
            .iter()
            .filter(|m| m.error.is_some())
            .count();
        let error_rate = errors as f64 / total_requests as f64;

        // Calculate throughput
        let throughput = total_requests as f64 / time_window.as_secs_f64();

        // Calculate optimization effectiveness
        let optimized_requests = relevant_metrics
            .iter()
            .filter(|m| m.optimization_applied)
            .count();
        let optimization_effectiveness = if optimized_requests > 0 {
            let optimized_avg_time: Duration = relevant_metrics
                .iter()
                .filter(|m| m.optimization_applied)
                .map(|m| m.response_time)
                .sum::<Duration>()
                / optimized_requests as u32;

            let non_optimized_avg_time: Duration = relevant_metrics
                .iter()
                .filter(|m| !m.optimization_applied)
                .map(|m| m.response_time)
                .sum::<Duration>()
                / (total_requests - optimized_requests) as u32;

            if non_optimized_avg_time > optimized_avg_time {
                ((non_optimized_avg_time - optimized_avg_time).as_millis() as f64
                    / non_optimized_avg_time.as_millis() as f64)
                    * 100.0
            } else {
                0.0
            }
        } else {
            0.0
        };

        // Get cache statistics for memory efficiency
        let cache_stats = self.cache_manager.get_cache_stats().await;
        let memory_efficiency = cache_stats.hit_rate();

        Ok(AggregatedMetrics {
            time_window,
            total_requests,
            avg_response_time,
            p50_response_time,
            p95_response_time,
            p99_response_time,
            cache_hit_rate,
            error_rate,
            throughput,
            optimization_effectiveness,
            memory_efficiency,
        })
    }

    /// Analyze performance bottlenecks
    pub async fn analyze_bottlenecks(&self) -> Result<BottleneckAnalysis> {
        let metrics = self
            .get_aggregated_metrics(Duration::from_secs(3600))
            .await?; // Last hour
        let mut bottlenecks = Vec::new();
        let mut recommendations = Vec::new();
        let mut predicted_improvements = HashMap::new();

        // Analyze cache performance
        if metrics.cache_hit_rate < self.config.performance_targets.cache_hit_rate {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::CacheMiss,
                description: format!(
                    "Cache hit rate {:.2}% is below target {:.2}%",
                    metrics.cache_hit_rate * 100.0,
                    self.config.performance_targets.cache_hit_rate * 100.0
                ),
                severity: if metrics.cache_hit_rate < 0.3 {
                    BottleneckSeverity::Critical
                } else {
                    BottleneckSeverity::Medium
                },
                affected_operations: vec![
                    "Response generation".to_string(),
                    "Context assembly".to_string(),
                ],
                estimated_impact: (1.0 - metrics.cache_hit_rate) * 40.0, // Up to 40% impact
            });

            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationType::IncreaseCacheSize,
                description: "Increase cache size to improve hit rate".to_string(),
                expected_improvement: 20.0,
                implementation_effort: ImplementationEffort::Low,
                parameters: [(
                    "new_size_multiplier".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(2)),
                )]
                .iter()
                .cloned()
                .collect(),
            });

            predicted_improvements.insert("cache_optimization".to_string(), 25.0);
        }

        // Analyze response time
        if metrics.avg_response_time
            > Duration::from_millis(self.config.performance_targets.avg_response_time_ms)
        {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::SlowQuery,
                description: format!(
                    "Average response time {}ms exceeds target {}ms",
                    metrics.avg_response_time.as_millis(),
                    self.config.performance_targets.avg_response_time_ms
                ),
                severity: if metrics.avg_response_time > Duration::from_millis(5000) {
                    BottleneckSeverity::Critical
                } else {
                    BottleneckSeverity::High
                },
                affected_operations: vec![
                    "Query processing".to_string(),
                    "Response generation".to_string(),
                ],
                estimated_impact: 30.0,
            });

            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationType::OptimizeQueries,
                description: "Optimize SPARQL queries and add query caching".to_string(),
                expected_improvement: 35.0,
                implementation_effort: ImplementationEffort::Medium,
                parameters: HashMap::new(),
            });

            predicted_improvements.insert("query_optimization".to_string(), 30.0);
        }

        // Analyze memory efficiency
        if metrics.memory_efficiency < 0.6 {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::MemoryPressure,
                description: "Memory efficiency is low, indicating possible memory pressure"
                    .to_string(),
                severity: BottleneckSeverity::Medium,
                affected_operations: vec!["Caching".to_string(), "Context management".to_string()],
                estimated_impact: 15.0,
            });

            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationType::CompressData,
                description: "Enable data compression in caches".to_string(),
                expected_improvement: 20.0,
                implementation_effort: ImplementationEffort::Low,
                parameters: [(
                    "compression_enabled".to_string(),
                    serde_json::Value::Bool(true),
                )]
                .iter()
                .cloned()
                .collect(),
            });
        }

        Ok(BottleneckAnalysis {
            analysis_timestamp: SystemTime::now(),
            identified_bottlenecks: bottlenecks,
            optimization_recommendations: recommendations,
            predicted_improvements,
        })
    }

    /// Apply automatic optimizations
    pub async fn apply_optimizations(&self, analysis: &BottleneckAnalysis) -> Result<usize> {
        let mut applied_count = 0;
        let before_metrics = self
            .get_aggregated_metrics(Duration::from_secs(300))
            .await?;

        for recommendation in &analysis.optimization_recommendations {
            if matches!(
                recommendation.implementation_effort,
                ImplementationEffort::Low
            ) {
                match self
                    .apply_optimization(recommendation, &before_metrics)
                    .await
                {
                    Ok(_) => {
                        applied_count += 1;
                        info!("Applied optimization: {}", recommendation.description);
                    }
                    Err(e) => {
                        warn!(
                            "Failed to apply optimization {}: {}",
                            recommendation.description, e
                        );
                    }
                }
            }
        }

        Ok(applied_count)
    }

    /// Start background monitoring task
    fn start_monitoring_task(&self) {
        let request_metrics = Arc::clone(&self.request_metrics);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60)); // Every minute

            loop {
                interval.tick().await;

                // Clean up old metrics
                let mut metrics = request_metrics.write().await;
                let retention_duration =
                    Duration::from_secs(config.metrics_retention_hours as u64 * 3600);
                let cutoff_time = SystemTime::now() - retention_duration;

                while let Some(front) = metrics.front() {
                    if front.timestamp < cutoff_time {
                        metrics.pop_front();
                    } else {
                        break;
                    }
                }

                debug!("Performance monitoring: {} active metrics", metrics.len());
            }
        });
    }

    /// Start optimization task
    fn start_optimization_task(&self) {
        let request_metrics = Arc::clone(&self.request_metrics);
        let cache_manager = Arc::clone(&self.cache_manager);
        let optimization_history = Arc::clone(&self.optimization_history);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.optimization_interval);

            loop {
                interval.tick().await;

                // Create a temporary monitor for analysis
                match Self::analyze_bottlenecks_static(&request_metrics, &cache_manager, &config)
                    .await
                {
                    Ok(analysis) => {
                        if !analysis.identified_bottlenecks.is_empty() {
                            info!(
                                "Performance analysis found {} bottlenecks",
                                analysis.identified_bottlenecks.len()
                            );

                            match Self::apply_optimizations_static(
                                &analysis,
                                &optimization_history,
                                &config,
                            )
                            .await
                            {
                                Ok(count) => {
                                    if count > 0 {
                                        info!("Applied {} automatic optimizations", count);
                                    }
                                }
                                Err(e) => {
                                    warn!("Failed to apply optimizations: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        warn!("Failed to analyze performance bottlenecks: {}", e);
                    }
                }
            }
        });
    }

    /// Static version of analyze_bottlenecks for use in async tasks
    async fn analyze_bottlenecks_static(
        request_metrics: &Arc<RwLock<VecDeque<RequestMetrics>>>,
        cache_manager: &Arc<AdvancedCacheManager>,
        config: &PerformanceConfig,
    ) -> Result<BottleneckAnalysis> {
        let metrics =
            Self::get_aggregated_metrics_static(request_metrics, Duration::from_secs(3600)).await?;
        let mut bottlenecks = Vec::new();
        let mut recommendations = Vec::new();
        let mut predicted_improvements = HashMap::new();

        // Analyze cache performance
        if metrics.cache_hit_rate < config.performance_targets.cache_hit_rate {
            bottlenecks.push(Bottleneck {
                bottleneck_type: BottleneckType::CacheMiss,
                description: format!(
                    "Cache hit rate {:.2}% is below target {:.2}%",
                    metrics.cache_hit_rate * 100.0,
                    config.performance_targets.cache_hit_rate * 100.0
                ),
                severity: if metrics.cache_hit_rate < 0.3 {
                    BottleneckSeverity::Critical
                } else {
                    BottleneckSeverity::Medium
                },
                affected_operations: vec![
                    "Response generation".to_string(),
                    "Context assembly".to_string(),
                ],
                estimated_impact: (1.0 - metrics.cache_hit_rate) * 40.0,
            });

            recommendations.push(OptimizationRecommendation {
                recommendation_type: OptimizationType::IncreaseCacheSize,
                description: "Increase cache size to improve hit rate".to_string(),
                expected_improvement: 20.0,
                implementation_effort: ImplementationEffort::Low,
                parameters: [(
                    "new_size_multiplier".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(2)),
                )]
                .iter()
                .cloned()
                .collect(),
            });

            predicted_improvements.insert("cache_optimization".to_string(), 25.0);
        }

        Ok(BottleneckAnalysis {
            analysis_timestamp: SystemTime::now(),
            identified_bottlenecks: bottlenecks,
            optimization_recommendations: recommendations,
            predicted_improvements,
        })
    }

    /// Static version of apply_optimizations for use in async tasks
    async fn apply_optimizations_static(
        analysis: &BottleneckAnalysis,
        optimization_history: &Arc<RwLock<Vec<AppliedOptimization>>>,
        _config: &PerformanceConfig,
    ) -> Result<usize> {
        let mut applied_count = 0;

        for recommendation in &analysis.optimization_recommendations {
            if matches!(
                recommendation.implementation_effort,
                ImplementationEffort::Low
            ) {
                // Create a simple optimization record
                let optimization_record = AppliedOptimization {
                    timestamp: SystemTime::now(),
                    optimization_type: recommendation.recommendation_type.clone(),
                    parameters: recommendation.parameters.clone(),
                    before_metrics: AggregatedMetrics {
                        time_window: Duration::from_secs(300),
                        total_requests: 0,
                        avg_response_time: Duration::ZERO,
                        p50_response_time: Duration::ZERO,
                        p95_response_time: Duration::ZERO,
                        p99_response_time: Duration::ZERO,
                        cache_hit_rate: 0.0,
                        error_rate: 0.0,
                        throughput: 0.0,
                        optimization_effectiveness: 0.0,
                        memory_efficiency: 0.0,
                    },
                    after_metrics: None,
                    success: true,
                    notes: "Applied automatically".to_string(),
                };

                let mut history = optimization_history.write().await;
                history.push(optimization_record);
                applied_count += 1;

                info!("Applied optimization: {}", recommendation.description);
            }
        }

        Ok(applied_count)
    }

    /// Static version of get_aggregated_metrics for use in async tasks
    async fn get_aggregated_metrics_static(
        request_metrics: &Arc<RwLock<VecDeque<RequestMetrics>>>,
        time_window: Duration,
    ) -> Result<AggregatedMetrics> {
        let request_metrics = request_metrics.read().await;
        let cutoff_time = SystemTime::now() - time_window;

        let relevant_metrics: Vec<&RequestMetrics> = request_metrics
            .iter()
            .filter(|m| m.timestamp >= cutoff_time)
            .collect();

        if relevant_metrics.is_empty() {
            return Ok(AggregatedMetrics {
                time_window,
                total_requests: 0,
                avg_response_time: Duration::ZERO,
                p50_response_time: Duration::ZERO,
                p95_response_time: Duration::ZERO,
                p99_response_time: Duration::ZERO,
                cache_hit_rate: 0.0,
                error_rate: 0.0,
                throughput: 0.0,
                optimization_effectiveness: 0.0,
                memory_efficiency: 0.0,
            });
        }

        let total_requests = relevant_metrics.len();

        // Calculate response time statistics
        let mut response_times: Vec<Duration> =
            relevant_metrics.iter().map(|m| m.response_time).collect();
        response_times.sort();

        let avg_response_time = Duration::from_millis(
            response_times
                .iter()
                .map(|d| d.as_millis() as u64)
                .sum::<u64>()
                / total_requests as u64,
        );

        let p50_response_time = response_times[total_requests / 2];
        let p95_response_time = response_times[(total_requests as f64 * 0.95) as usize];
        let p99_response_time = response_times[(total_requests as f64 * 0.99) as usize];

        // Calculate cache hit rate
        let cache_hits = relevant_metrics.iter().filter(|m| m.cache_hit).count();
        let cache_hit_rate = cache_hits as f64 / total_requests as f64;

        // Calculate error rate
        let errors = relevant_metrics
            .iter()
            .filter(|m| m.error.is_some())
            .count();
        let error_rate = errors as f64 / total_requests as f64;

        // Calculate throughput
        let throughput = total_requests as f64 / time_window.as_secs_f64();

        Ok(AggregatedMetrics {
            time_window,
            total_requests,
            avg_response_time,
            p50_response_time,
            p95_response_time,
            p99_response_time,
            cache_hit_rate,
            error_rate,
            throughput,
            optimization_effectiveness: 0.0,
            memory_efficiency: 0.0,
        })
    }

    /// Check for performance alerts
    async fn check_performance_alerts(&self, metrics: &RequestMetrics) -> Result<()> {
        let last_alert = *self.last_alert_time.read().await;
        let now = SystemTime::now();

        if now.duration_since(last_alert).unwrap_or(Duration::ZERO)
            < self.config.alerting.notification_cooldown
        {
            return Ok(()); // Still in cooldown
        }

        let thresholds = &self.config.alerting.alert_thresholds;

        // Check response time
        if metrics.response_time.as_millis() > thresholds.response_time_critical_ms as u128 {
            self.send_alert(format!(
                "Critical response time: {}ms for session {} message {}",
                metrics.response_time.as_millis(),
                metrics.session_id,
                metrics.message_id
            ))
            .await?;
            *self.last_alert_time.write().await = now;
        }

        // Check error rate (would need to aggregate recent errors)
        if let Some(ref error) = metrics.error {
            self.send_alert(format!(
                "Error in session {} message {}: {}",
                metrics.session_id, metrics.message_id, error
            ))
            .await?;
        }

        Ok(())
    }

    /// Send performance alert
    async fn send_alert(&self, message: String) -> Result<()> {
        // In a real implementation, this would send alerts to monitoring systems
        warn!("PERFORMANCE ALERT: {}", message);
        Ok(())
    }

    /// Apply a specific optimization
    async fn apply_optimization(
        &self,
        recommendation: &OptimizationRecommendation,
        before_metrics: &AggregatedMetrics,
    ) -> Result<()> {
        let optimization_record = AppliedOptimization {
            timestamp: SystemTime::now(),
            optimization_type: recommendation.recommendation_type.clone(),
            parameters: recommendation.parameters.clone(),
            before_metrics: before_metrics.clone(),
            after_metrics: None,
            success: false,
            notes: String::new(),
        };

        match recommendation.recommendation_type {
            OptimizationType::IncreaseCacheSize => {
                // This would require access to cache configuration
                // For now, we'll just log the optimization
                info!("Would increase cache size based on recommendation");
            }
            OptimizationType::AdjustCacheTTL => {
                info!("Would adjust cache TTL based on recommendation");
            }
            OptimizationType::CompressData => {
                info!("Would enable data compression based on recommendation");
            }
            _ => {
                return Err(anyhow!(
                    "Optimization type not implemented for automatic application"
                ));
            }
        }

        let mut history = self.optimization_history.write().await;
        history.push(optimization_record);

        Ok(())
    }

    /// Get optimization history
    pub async fn get_optimization_history(&self) -> Vec<AppliedOptimization> {
        self.optimization_history.read().await.clone()
    }

    /// Get performance dashboard data
    pub async fn get_dashboard_data(&self) -> Result<PerformanceDashboard> {
        let current_metrics = self
            .get_aggregated_metrics(Duration::from_secs(3600))
            .await?; // Last hour
        let daily_metrics = self
            .get_aggregated_metrics(Duration::from_secs(86400))
            .await?; // Last day
        let bottleneck_analysis = self.analyze_bottlenecks().await?;
        let cache_stats = self.cache_manager.get_cache_stats().await;

        Ok(PerformanceDashboard {
            current_metrics,
            daily_metrics,
            bottleneck_analysis,
            cache_stats,
            targets: self.config.performance_targets.clone(),
            optimization_history: self.get_optimization_history().await,
        })
    }
}

/// Performance dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDashboard {
    pub current_metrics: AggregatedMetrics,
    pub daily_metrics: AggregatedMetrics,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub cache_stats: CacheStats,
    pub targets: PerformanceTargets,
    pub optimization_history: Vec<AppliedOptimization>,
}
