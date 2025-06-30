//! Federation Monitoring and Metrics
//!
//! This module provides comprehensive monitoring and metrics collection for federated
//! query processing, including performance tracking, error monitoring, and observability.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Federation performance monitor
#[derive(Debug)]
pub struct FederationMonitor {
    metrics: Arc<RwLock<FederationMetrics>>,
    config: FederationMonitorConfig,
    start_time: Instant,
}

impl FederationMonitor {
    /// Create a new federation monitor
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(FederationMetrics::new())),
            config: FederationMonitorConfig::default(),
            start_time: Instant::now(),
        }
    }

    /// Create a new federation monitor with custom configuration
    pub fn with_config(config: FederationMonitorConfig) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(FederationMetrics::new())),
            config,
            start_time: Instant::now(),
        }
    }

    /// Record a query execution
    pub async fn record_query_execution(
        &self,
        query_type: &str,
        duration: Duration,
        success: bool,
    ) {
        let mut metrics = self.metrics.write().await;

        metrics.total_queries += 1;
        if success {
            metrics.successful_queries += 1;
        } else {
            metrics.failed_queries += 1;
        }

        // Update type-specific metrics
        let type_metrics = metrics
            .query_type_metrics
            .entry(query_type.to_string())
            .or_insert_with(QueryTypeMetrics::new);
        type_metrics.total_count += 1;
        type_metrics.total_duration += duration;
        type_metrics.avg_duration = type_metrics.total_duration / type_metrics.total_count as u32;

        if success {
            type_metrics.success_count += 1;
        } else {
            type_metrics.error_count += 1;
        }

        // Update response time histogram
        self.update_response_time_histogram(&mut metrics, duration)
            .await;

        // Track recent queries for trend analysis
        let query_record = QueryRecord {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            query_type: query_type.to_string(),
            duration,
            success,
        };

        metrics.recent_queries.push(query_record);

        // Keep only recent queries (sliding window)
        if metrics.recent_queries.len() > self.config.max_recent_queries {
            metrics.recent_queries.remove(0);
        }

        debug!(
            "Recorded {} query execution: {}ms, success: {}",
            query_type,
            duration.as_millis(),
            success
        );
    }

    /// Record service interaction metrics
    pub async fn record_service_interaction(
        &self,
        service_id: &str,
        duration: Duration,
        success: bool,
        response_size: Option<usize>,
    ) {
        let mut metrics = self.metrics.write().await;

        let service_metrics = metrics
            .service_metrics
            .entry(service_id.to_string())
            .or_insert_with(ServiceMetrics::new);

        service_metrics.total_requests += 1;
        service_metrics.total_duration += duration;
        service_metrics.avg_duration =
            service_metrics.total_duration / service_metrics.total_requests as u32;

        if success {
            service_metrics.successful_requests += 1;
        } else {
            service_metrics.failed_requests += 1;
        }

        if let Some(size) = response_size {
            service_metrics.total_response_size += size as u64;
            service_metrics.avg_response_size = service_metrics.total_response_size
                / service_metrics.successful_requests.max(1) as u64;
        }

        // Update last seen timestamp
        service_metrics.last_seen = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        debug!(
            "Recorded service interaction for {}: {}ms, success: {}",
            service_id,
            duration.as_millis(),
            success
        );
    }

    /// Record federation-specific events
    pub async fn record_federation_event(&self, event_type: FederationEventType, details: &str) {
        let mut metrics = self.metrics.write().await;

        let event = FederationEvent {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            event_type,
            details: details.to_string(),
        };

        metrics.federation_events.push(event);

        // Keep only recent events
        if metrics.federation_events.len() > self.config.max_recent_events {
            metrics.federation_events.remove(0);
        }

        // Update event type counters
        *metrics.event_type_counts.entry(event_type).or_insert(0) += 1;

        info!("Federation event: {:?} - {}", event_type, details);
    }

    /// Record cache statistics
    pub async fn record_cache_hit(&self, cache_type: &str, hit: bool) {
        let mut metrics = self.metrics.write().await;

        let cache_metrics = metrics
            .cache_metrics
            .entry(cache_type.to_string())
            .or_insert_with(CacheMetrics::new);

        cache_metrics.total_requests += 1;
        if hit {
            cache_metrics.hits += 1;
        } else {
            cache_metrics.misses += 1;
        }

        cache_metrics.hit_rate = if cache_metrics.total_requests > 0 {
            cache_metrics.hits as f64 / cache_metrics.total_requests as f64
        } else {
            0.0
        };

        debug!(
            "Cache {} for {}: hit rate {:.2}%",
            if hit { "hit" } else { "miss" },
            cache_type,
            cache_metrics.hit_rate * 100.0
        );
    }

    /// Get comprehensive monitoring statistics
    pub async fn get_stats(&self) -> MonitorStats {
        let metrics = self.metrics.read().await;
        let uptime = self.start_time.elapsed();

        MonitorStats {
            uptime,
            total_queries: metrics.total_queries,
            successful_queries: metrics.successful_queries,
            failed_queries: metrics.failed_queries,
            success_rate: if metrics.total_queries > 0 {
                metrics.successful_queries as f64 / metrics.total_queries as f64
            } else {
                0.0
            },
            query_type_metrics: metrics.query_type_metrics.clone(),
            service_metrics: metrics.service_metrics.clone(),
            cache_metrics: metrics.cache_metrics.clone(),
            response_time_histogram: metrics.response_time_histogram.clone(),
            recent_events_count: metrics.federation_events.len(),
            event_type_counts: metrics.event_type_counts.clone(),
            avg_queries_per_second: if uptime.as_secs() > 0 {
                metrics.total_queries as f64 / uptime.as_secs() as f64
            } else {
                0.0
            },
        }
    }

    /// Get health metrics for monitoring systems
    pub async fn get_health_metrics(&self) -> HealthMetrics {
        let metrics = self.metrics.read().await;
        let stats = self.calculate_health_indicators(&metrics).await;

        HealthMetrics {
            overall_health: stats.overall_health,
            service_health: stats.service_health,
            error_rate: stats.error_rate,
            avg_response_time: stats.avg_response_time,
            active_services: metrics.service_metrics.len(),
            recent_error_count: stats.recent_error_count,
            cache_hit_rate: stats.cache_hit_rate,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Export metrics in Prometheus format
    pub async fn export_prometheus_metrics(&self) -> String {
        let metrics = self.metrics.read().await;
        let mut output = String::new();

        // Total queries metric
        output
            .push_str("# HELP oxirs_federation_queries_total Total number of federated queries\n");
        output.push_str("# TYPE oxirs_federation_queries_total counter\n");
        output.push_str(&format!(
            "oxirs_federation_queries_total {}\n",
            metrics.total_queries
        ));

        // Success rate metric
        let success_rate = if metrics.total_queries > 0 {
            metrics.successful_queries as f64 / metrics.total_queries as f64
        } else {
            0.0
        };
        output.push_str("# HELP oxirs_federation_success_rate Query success rate\n");
        output.push_str("# TYPE oxirs_federation_success_rate gauge\n");
        output.push_str(&format!(
            "oxirs_federation_success_rate {:.4}\n",
            success_rate
        ));

        // Query type metrics
        output.push_str("# HELP oxirs_federation_queries_by_type_total Queries by type\n");
        output.push_str("# TYPE oxirs_federation_queries_by_type_total counter\n");
        for (query_type, type_metrics) in &metrics.query_type_metrics {
            output.push_str(&format!(
                "oxirs_federation_queries_by_type_total{{type=\"{}\"}} {}\n",
                query_type, type_metrics.total_count
            ));
        }

        // Average response time by type
        output.push_str("# HELP oxirs_federation_avg_response_time_seconds Average response time by query type\n");
        output.push_str("# TYPE oxirs_federation_avg_response_time_seconds gauge\n");
        for (query_type, type_metrics) in &metrics.query_type_metrics {
            output.push_str(&format!(
                "oxirs_federation_avg_response_time_seconds{{type=\"{}\"}} {:.6}\n",
                query_type,
                type_metrics.avg_duration.as_secs_f64()
            ));
        }

        // Service metrics
        output.push_str("# HELP oxirs_federation_service_requests_total Requests per service\n");
        output.push_str("# TYPE oxirs_federation_service_requests_total counter\n");
        for (service_id, service_metrics) in &metrics.service_metrics {
            output.push_str(&format!(
                "oxirs_federation_service_requests_total{{service=\"{}\"}} {}\n",
                service_id, service_metrics.total_requests
            ));
        }

        // Cache hit rates
        output.push_str("# HELP oxirs_federation_cache_hit_rate Cache hit rate by type\n");
        output.push_str("# TYPE oxirs_federation_cache_hit_rate gauge\n");
        for (cache_type, cache_metrics) in &metrics.cache_metrics {
            output.push_str(&format!(
                "oxirs_federation_cache_hit_rate{{cache=\"{}\"}} {:.4}\n",
                cache_type, cache_metrics.hit_rate
            ));
        }

        output
    }

    /// Update response time histogram
    async fn update_response_time_histogram(
        &self,
        metrics: &mut FederationMetrics,
        duration: Duration,
    ) {
        let millis = duration.as_millis() as u64;

        let bucket = if millis < 10 {
            "0-10ms"
        } else if millis < 50 {
            "10-50ms"
        } else if millis < 100 {
            "50-100ms"
        } else if millis < 500 {
            "100-500ms"
        } else if millis < 1000 {
            "500ms-1s"
        } else if millis < 5000 {
            "1s-5s"
        } else {
            "5s+"
        };

        *metrics
            .response_time_histogram
            .entry(bucket.to_string())
            .or_insert(0) += 1;
    }

    /// Calculate health indicators
    async fn calculate_health_indicators(&self, metrics: &FederationMetrics) -> HealthIndicators {
        let error_rate = if metrics.total_queries > 0 {
            metrics.failed_queries as f64 / metrics.total_queries as f64
        } else {
            0.0
        };

        let avg_response_time = if !metrics.query_type_metrics.is_empty() {
            let total_duration: Duration = metrics
                .query_type_metrics
                .values()
                .map(|m| m.total_duration)
                .sum();
            let total_count: u64 = metrics
                .query_type_metrics
                .values()
                .map(|m| m.total_count)
                .sum();

            if total_count > 0 {
                total_duration / total_count as u32
            } else {
                Duration::from_secs(0)
            }
        } else {
            Duration::from_secs(0)
        };

        let recent_error_count = metrics.recent_queries.iter().filter(|q| !q.success).count();

        let cache_hit_rate = if !metrics.cache_metrics.is_empty() {
            let total_hit_rate: f64 = metrics.cache_metrics.values().map(|c| c.hit_rate).sum();
            total_hit_rate / metrics.cache_metrics.len() as f64
        } else {
            0.0
        };

        let overall_health = if error_rate > 0.1 {
            HealthStatus::Unhealthy
        } else if error_rate > 0.05 || avg_response_time > Duration::from_secs(5) {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };

        // Calculate per-service health
        let mut service_health = HashMap::new();
        for (service_id, service_metrics) in &metrics.service_metrics {
            let service_error_rate = if service_metrics.total_requests > 0 {
                service_metrics.failed_requests as f64 / service_metrics.total_requests as f64
            } else {
                0.0
            };

            let health = if service_error_rate > 0.2 {
                HealthStatus::Unhealthy
            } else if service_error_rate > 0.1
                || service_metrics.avg_duration > Duration::from_secs(3)
            {
                HealthStatus::Degraded
            } else {
                HealthStatus::Healthy
            };

            service_health.insert(service_id.clone(), health);
        }

        HealthIndicators {
            overall_health,
            service_health,
            error_rate,
            avg_response_time,
            recent_error_count,
            cache_hit_rate,
        }
    }

    /// Generate performance report
    pub async fn generate_performance_report(&self) -> PerformanceReport {
        let metrics = self.metrics.read().await;
        let uptime = self.start_time.elapsed();

        // Calculate trends
        let mut query_trends = HashMap::new();
        for (query_type, type_metrics) in &metrics.query_type_metrics {
            let trend = QueryTrend {
                query_type: query_type.clone(),
                total_queries: type_metrics.total_count,
                avg_response_time: type_metrics.avg_duration,
                error_rate: if type_metrics.total_count > 0 {
                    type_metrics.error_count as f64 / type_metrics.total_count as f64
                } else {
                    0.0
                },
                queries_per_second: if uptime.as_secs() > 0 {
                    type_metrics.total_count as f64 / uptime.as_secs() as f64
                } else {
                    0.0
                },
            };
            query_trends.insert(query_type.clone(), trend);
        }

        PerformanceReport {
            report_timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            uptime,
            total_queries: metrics.total_queries,
            overall_success_rate: if metrics.total_queries > 0 {
                metrics.successful_queries as f64 / metrics.total_queries as f64
            } else {
                0.0
            },
            query_trends,
            top_errors: self.get_top_errors(&metrics).await,
            performance_summary: self.get_performance_summary(&metrics).await,
            bottlenecks: self.identify_bottlenecks(&metrics).await,
            performance_regressions: self.detect_performance_regressions(&metrics).await,
            optimization_recommendations: self
                .generate_optimization_recommendations(&metrics)
                .await,
        }
    }

    /// Identify performance bottlenecks
    pub async fn identify_bottlenecks(&self, metrics: &FederationMetrics) -> Vec<BottleneckReport> {
        let mut bottlenecks = Vec::new();

        // Identify slow services
        for (service_id, service_metrics) in &metrics.service_metrics {
            if service_metrics.avg_duration > Duration::from_secs(2) {
                bottlenecks.push(BottleneckReport {
                    bottleneck_type: BottleneckType::SlowService,
                    component: service_id.clone(),
                    severity: if service_metrics.avg_duration > Duration::from_secs(5) {
                        BottleneckSeverity::Critical
                    } else if service_metrics.avg_duration > Duration::from_secs(3) {
                        BottleneckSeverity::High
                    } else {
                        BottleneckSeverity::Medium
                    },
                    description: format!(
                        "Service {} has high average response time: {:?}",
                        service_id, service_metrics.avg_duration
                    ),
                    metric_value: service_metrics.avg_duration.as_millis() as f64,
                    threshold: 2000.0, // 2 seconds in milliseconds
                    impact_score: self.calculate_service_impact_score(service_metrics),
                });
            }
        }

        // Identify high error rate services
        for (service_id, service_metrics) in &metrics.service_metrics {
            let error_rate = if service_metrics.total_requests > 0 {
                service_metrics.failed_requests as f64 / service_metrics.total_requests as f64
            } else {
                0.0
            };

            if error_rate > 0.05 {
                bottlenecks.push(BottleneckReport {
                    bottleneck_type: BottleneckType::HighErrorRate,
                    component: service_id.clone(),
                    severity: if error_rate > 0.2 {
                        BottleneckSeverity::Critical
                    } else if error_rate > 0.1 {
                        BottleneckSeverity::High
                    } else {
                        BottleneckSeverity::Medium
                    },
                    description: format!(
                        "Service {} has high error rate: {:.2}%",
                        service_id,
                        error_rate * 100.0
                    ),
                    metric_value: error_rate * 100.0,
                    threshold: 5.0, // 5% error rate threshold
                    impact_score: error_rate * service_metrics.total_requests as f64,
                });
            }
        }

        // Identify poor cache performance
        for (cache_name, cache_metrics) in &metrics.cache_metrics {
            if cache_metrics.hit_rate < 0.5 && cache_metrics.total_requests > 100 {
                bottlenecks.push(BottleneckReport {
                    bottleneck_type: BottleneckType::PoorCachePerformance,
                    component: format!("Cache: {}", cache_name),
                    severity: if cache_metrics.hit_rate < 0.2 {
                        BottleneckSeverity::High
                    } else {
                        BottleneckSeverity::Medium
                    },
                    description: format!(
                        "Cache {} has low hit rate: {:.2}%",
                        cache_name,
                        cache_metrics.hit_rate * 100.0
                    ),
                    metric_value: cache_metrics.hit_rate * 100.0,
                    threshold: 50.0, // 50% hit rate threshold
                    impact_score: (1.0 - cache_metrics.hit_rate)
                        * cache_metrics.total_requests as f64,
                });
            }
        }

        // Sort by impact score (highest first)
        bottlenecks.sort_by(|a, b| b.impact_score.partial_cmp(&a.impact_score).unwrap());

        bottlenecks
    }

    /// Detect performance regressions
    pub async fn detect_performance_regressions(
        &self,
        metrics: &FederationMetrics,
    ) -> Vec<RegressionReport> {
        let mut regressions = Vec::new();

        // Analyze recent query performance compared to historical averages
        let recent_window = Duration::from_secs(300); // Last 5 minutes
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        for (query_type, type_metrics) in &metrics.query_type_metrics {
            // Filter recent queries
            let recent_queries: Vec<&QueryRecord> = metrics
                .recent_queries
                .iter()
                .filter(|q| {
                    q.query_type == *query_type
                        && (current_time - q.timestamp) < recent_window.as_secs()
                })
                .collect();

            if recent_queries.len() >= 10 {
                let recent_avg_duration: Duration =
                    recent_queries.iter().map(|q| q.duration).sum::<Duration>()
                        / recent_queries.len() as u32;

                let historical_avg = type_metrics.avg_duration;

                // Check for significant performance degradation (>50% increase)
                if recent_avg_duration > historical_avg + Duration::from_millis(500)
                    && recent_avg_duration.as_millis() as f64
                        > historical_avg.as_millis() as f64 * 1.5
                {
                    let degradation_factor =
                        recent_avg_duration.as_millis() as f64 / historical_avg.as_millis() as f64;

                    regressions.push(RegressionReport {
                        component: format!("Query Type: {}", query_type),
                        regression_type: RegressionType::ResponseTimeIncrease,
                        severity: if degradation_factor > 3.0 {
                            RegressionSeverity::Critical
                        } else if degradation_factor > 2.0 {
                            RegressionSeverity::High
                        } else {
                            RegressionSeverity::Medium
                        },
                        description: format!(
                            "Query type {} response time increased from {:?} to {:?} ({:.1}x degradation)",
                            query_type, historical_avg, recent_avg_duration, degradation_factor
                        ),
                        historical_value: historical_avg.as_millis() as f64,
                        current_value: recent_avg_duration.as_millis() as f64,
                        detected_at: current_time,
                        confidence: self.calculate_regression_confidence(&recent_queries, type_metrics),
                    });
                }

                // Check for error rate increases
                let recent_error_rate = recent_queries.iter().filter(|q| !q.success).count() as f64
                    / recent_queries.len() as f64;

                let historical_error_rate = if type_metrics.total_count > 0 {
                    type_metrics.error_count as f64 / type_metrics.total_count as f64
                } else {
                    0.0
                };

                if recent_error_rate > historical_error_rate + 0.1
                    && recent_error_rate > historical_error_rate * 2.0
                {
                    regressions.push(RegressionReport {
                        component: format!("Query Type: {}", query_type),
                        regression_type: RegressionType::ErrorRateIncrease,
                        severity: if recent_error_rate > 0.2 {
                            RegressionSeverity::Critical
                        } else if recent_error_rate > 0.1 {
                            RegressionSeverity::High
                        } else {
                            RegressionSeverity::Medium
                        },
                        description: format!(
                            "Query type {} error rate increased from {:.2}% to {:.2}%",
                            query_type,
                            historical_error_rate * 100.0,
                            recent_error_rate * 100.0
                        ),
                        historical_value: historical_error_rate * 100.0,
                        current_value: recent_error_rate * 100.0,
                        detected_at: current_time,
                        confidence: self
                            .calculate_regression_confidence(&recent_queries, type_metrics),
                    });
                }
            }
        }

        regressions
    }

    /// Generate optimization recommendations
    pub async fn generate_optimization_recommendations(
        &self,
        metrics: &FederationMetrics,
    ) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze cache performance
        for (cache_name, cache_metrics) in &metrics.cache_metrics {
            if cache_metrics.hit_rate < 0.7 && cache_metrics.total_requests > 50 {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::Caching,
                    priority: if cache_metrics.hit_rate < 0.3 {
                        OptimizationPriority::High
                    } else {
                        OptimizationPriority::Medium
                    },
                    title: format!("Improve {} Cache Hit Rate", cache_name),
                    description: format!(
                        "Cache {} has a {:.1}% hit rate. Consider increasing cache size, \
                        improving cache key strategies, or extending TTL values.",
                        cache_name,
                        cache_metrics.hit_rate * 100.0
                    ),
                    estimated_impact: self.calculate_cache_improvement_impact(cache_metrics),
                    implementation_effort: ImplementationEffort::Low,
                    metrics_to_monitor: vec![
                        "cache_hit_rate".to_string(),
                        "cache_miss_rate".to_string(),
                        "response_time".to_string(),
                    ],
                });
            }
        }

        // Analyze service performance
        for (service_id, service_metrics) in &metrics.service_metrics {
            if service_metrics.avg_duration > Duration::from_secs(1) {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::Performance,
                    priority: if service_metrics.avg_duration > Duration::from_secs(3) {
                        OptimizationPriority::High
                    } else {
                        OptimizationPriority::Medium
                    },
                    title: format!("Optimize {} Service Performance", service_id),
                    description: format!(
                        "Service {} has an average response time of {:?}. Consider \
                        implementing connection pooling, query optimization, or scaling the service.",
                        service_id, service_metrics.avg_duration
                    ),
                    estimated_impact: self.calculate_service_optimization_impact(service_metrics),
                    implementation_effort: ImplementationEffort::High,
                    metrics_to_monitor: vec![
                        "service_response_time".to_string(),
                        "service_throughput".to_string(),
                        "error_rate".to_string(),
                    ],
                });
            }

            // Check for load balancing opportunities
            if service_metrics.total_requests > 1000
                && service_metrics.avg_duration > Duration::from_millis(500)
            {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::Scaling,
                    priority: OptimizationPriority::Medium,
                    title: format!("Consider Load Balancing for {}", service_id),
                    description: format!(
                        "Service {} handles {} requests with {}ms average response time. \
                        Load balancing could improve performance and reliability.",
                        service_id,
                        service_metrics.total_requests,
                        service_metrics.avg_duration.as_millis()
                    ),
                    estimated_impact: "20-40% performance improvement".to_string(),
                    implementation_effort: ImplementationEffort::High,
                    metrics_to_monitor: vec![
                        "request_distribution".to_string(),
                        "service_utilization".to_string(),
                        "response_time_variance".to_string(),
                    ],
                });
            }
        }

        // Analyze query patterns
        for (query_type, type_metrics) in &metrics.query_type_metrics {
            if type_metrics.avg_duration > Duration::from_millis(200)
                && type_metrics.total_count > 100
            {
                recommendations.push(OptimizationRecommendation {
                    category: OptimizationCategory::QueryOptimization,
                    priority: OptimizationPriority::Medium,
                    title: format!("Optimize {} Queries", query_type),
                    description: format!(
                        "{} queries have an average execution time of {:?}. Consider \
                        query optimization, indexing, or result caching.",
                        query_type, type_metrics.avg_duration
                    ),
                    estimated_impact: self.calculate_query_optimization_impact(type_metrics),
                    implementation_effort: ImplementationEffort::Medium,
                    metrics_to_monitor: vec![
                        "query_execution_time".to_string(),
                        "query_complexity".to_string(),
                        "cache_utilization".to_string(),
                    ],
                });
            }
        }

        // Sort by priority and estimated impact
        recommendations.sort_by(|a, b| {
            let priority_cmp = b.priority.cmp(&a.priority);
            if priority_cmp == std::cmp::Ordering::Equal {
                // If same priority, sort by estimated impact (higher first)
                b.estimated_impact.len().cmp(&a.estimated_impact.len())
            } else {
                priority_cmp
            }
        });

        recommendations
    }

    // Helper methods for calculations

    fn calculate_service_impact_score(&self, service_metrics: &ServiceMetrics) -> f64 {
        let duration_score = service_metrics.avg_duration.as_millis() as f64 / 1000.0; // Convert to seconds
        let volume_score = service_metrics.total_requests as f64 / 100.0; // Normalize request volume
        duration_score * volume_score
    }

    fn calculate_regression_confidence(
        &self,
        recent_queries: &[&QueryRecord],
        type_metrics: &QueryTypeMetrics,
    ) -> f64 {
        let sample_size_score = (recent_queries.len() as f64 / 50.0).min(1.0); // Max confidence at 50 samples
        let historical_data_score = (type_metrics.total_count as f64 / 100.0).min(1.0); // Max confidence at 100 historical queries
        (sample_size_score + historical_data_score) / 2.0
    }

    fn calculate_cache_improvement_impact(&self, cache_metrics: &CacheMetrics) -> String {
        let potential_improvement = (0.8 - cache_metrics.hit_rate) * 100.0;
        if potential_improvement > 30.0 {
            format!(
                "High impact: Up to {:.0}% improvement in cache performance",
                potential_improvement
            )
        } else if potential_improvement > 15.0 {
            format!(
                "Medium impact: Up to {:.0}% improvement in cache performance",
                potential_improvement
            )
        } else {
            format!(
                "Low impact: Up to {:.0}% improvement in cache performance",
                potential_improvement
            )
        }
    }

    fn calculate_service_optimization_impact(&self, service_metrics: &ServiceMetrics) -> String {
        let current_duration = service_metrics.avg_duration.as_millis();
        if current_duration > 3000 {
            "High impact: 40-60% response time improvement".to_string()
        } else if current_duration > 1000 {
            "Medium impact: 20-40% response time improvement".to_string()
        } else {
            "Low impact: 10-20% response time improvement".to_string()
        }
    }

    fn calculate_query_optimization_impact(&self, type_metrics: &QueryTypeMetrics) -> String {
        let avg_duration = type_metrics.avg_duration.as_millis();
        let query_volume = type_metrics.total_count;

        if avg_duration > 1000 && query_volume > 1000 {
            "High impact: 30-50% performance improvement".to_string()
        } else if avg_duration > 500 || query_volume > 500 {
            "Medium impact: 15-30% performance improvement".to_string()
        } else {
            "Low impact: 5-15% performance improvement".to_string()
        }
    }

    /// Advanced distributed tracing correlation
    pub async fn record_trace_span(
        &self,
        trace_id: &str,
        span_id: &str,
        operation_name: &str,
        duration: Duration,
        tags: Option<HashMap<String, String>>,
    ) {
        let mut metrics = self.metrics.write().await;

        let trace_span = TraceSpan {
            trace_id: trace_id.to_string(),
            span_id: span_id.to_string(),
            parent_span_id: None,
            operation_name: operation_name.to_string(),
            start_time: SystemTime::now(),
            duration,
            tags: tags.unwrap_or_default(),
            service_id: None,
        };

        metrics.trace_spans.push(trace_span);

        // Keep only recent spans (sliding window)
        if metrics.trace_spans.len() > self.config.max_trace_spans {
            metrics.trace_spans.remove(0);
        }

        // Update trace metrics
        self.update_trace_metrics(&mut metrics, duration).await;

        debug!(
            "Recorded trace span: {} in trace {} with duration {:?}",
            operation_name, trace_id, duration
        );
    }

    /// Enhanced anomaly detection
    pub async fn check_for_anomalies(&self) -> Vec<AnomalyReport> {
        let metrics = self.metrics.read().await;
        let mut anomalies = Vec::new();

        // Check for error spikes
        let recent_errors = metrics
            .federation_events
            .iter()
            .rev()
            .take(100)
            .filter(|e| matches!(e.event_type, FederationEventType::Error | FederationEventType::ServiceFailure))
            .count();

        if recent_errors > 10 {
            anomalies.push(AnomalyReport {
                anomaly_type: AnomalyType::ErrorSpike,
                detected_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                details: format!("High error frequency detected: {} errors in last 100 events", recent_errors),
                severity: AnomalySeverity::High,
                confidence: 0.9,
            });
        }

        // Check for performance anomalies
        for (query_type, type_metrics) in &metrics.query_type_metrics {
            let recent_queries: Vec<&QueryRecord> = metrics
                .recent_queries
                .iter()
                .filter(|q| q.query_type == *query_type)
                .rev()
                .take(20)
                .collect();

            if recent_queries.len() >= 10 {
                let recent_avg_duration: Duration = recent_queries
                    .iter()
                    .map(|q| q.duration)
                    .sum::<Duration>() / recent_queries.len() as u32;

                // Check for significant performance degradation
                if recent_avg_duration > type_metrics.avg_duration + Duration::from_millis(500) {
                    let degradation_factor = recent_avg_duration.as_millis() as f64 
                        / type_metrics.avg_duration.as_millis() as f64;

                    if degradation_factor > 2.0 {
                        anomalies.push(AnomalyReport {
                            anomaly_type: AnomalyType::PerformanceDegradation,
                            detected_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                            details: format!(
                                "Performance degradation in {}: {:.1}x slower than baseline",
                                query_type, degradation_factor
                            ),
                            severity: if degradation_factor > 5.0 {
                                AnomalySeverity::Critical
                            } else if degradation_factor > 3.0 {
                                AnomalySeverity::High
                            } else {
                                AnomalySeverity::Medium
                            },
                            confidence: self.calculate_prediction_confidence(recent_queries.len(), type_metrics.total_count),
                        });
                    }
                }
            }
        }

        anomalies
    }

    /// Analyze cross-service latency patterns
    pub async fn analyze_cross_service_latency(&self) -> CrossServiceLatencyAnalysis {
        let metrics = self.metrics.read().await;
        
        let mut service_interactions: HashMap<(String, String), Vec<Duration>> = HashMap::new();
        
        // Simulate service interaction analysis from trace spans
        for trace_span in &metrics.trace_spans {
            if let Some(ref service_id) = trace_span.service_id {
                // Group spans by service interactions (simplified)
                let interaction_key = ("federation_gateway".to_string(), service_id.clone());
                service_interactions
                    .entry(interaction_key)
                    .or_insert_with(Vec::new)
                    .push(trace_span.duration);
            }
        }

        // Calculate statistics for each service interaction
        let mut interactions = Vec::new();
        for ((from_service, to_service), durations) in service_interactions {
            if !durations.is_empty() {
                let avg_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
                let max_duration = *durations.iter().max().unwrap();
                let min_duration = *durations.iter().min().unwrap();
                
                interactions.push(ServiceInteractionLatency {
                    from_service,
                    to_service,
                    avg_latency: avg_duration,
                    min_latency: min_duration,
                    max_latency: max_duration,
                    sample_count: durations.len(),
                });
            }
        }

        // Sort by average latency (highest first)
        interactions.sort_by(|a, b| b.avg_latency.cmp(&a.avg_latency));

        CrossServiceLatencyAnalysis {
            interactions,
            total_traces_analyzed: metrics.trace_spans.len(),
            analysis_timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }

    /// Predict performance issues using historical patterns
    pub async fn predict_performance_issues(&self) -> Vec<PerformancePrediction> {
        let metrics = self.metrics.read().await;
        let mut predictions = Vec::new();

        // Analyze query performance trends
        for (query_type, type_metrics) in &metrics.query_type_metrics {
            if type_metrics.total_count > 50 {
                let recent_queries: Vec<&QueryRecord> = metrics
                    .recent_queries
                    .iter()
                    .filter(|q| q.query_type == *query_type)
                    .rev()
                    .take(20)
                    .collect();

                if recent_queries.len() >= 10 {
                    let recent_avg_duration: Duration = recent_queries
                        .iter()
                        .map(|q| q.duration)
                        .sum::<Duration>() / recent_queries.len() as u32;

                    // Predict performance degradation trend
                    if recent_avg_duration > type_metrics.avg_duration + Duration::from_millis(100) {
                        let degradation_trend = recent_avg_duration.as_millis() as f64 
                            / type_metrics.avg_duration.as_millis() as f64;

                        predictions.push(PerformancePrediction {
                            prediction_type: PredictionType::PerformanceDegradation,
                            component: format!("Query Type: {}", query_type),
                            predicted_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                            confidence: self.calculate_prediction_confidence(recent_queries.len(), type_metrics.total_count),
                            description: format!(
                                "Query type {} showing {:.1}x performance degradation trend",
                                query_type, degradation_trend
                            ),
                            recommended_actions: vec![
                                "Monitor query complexity".to_string(),
                                "Check service health".to_string(),
                                "Consider query optimization".to_string(),
                            ],
                        });
                    }
                }
            }
        }

        // Predict service capacity issues
        for (service_id, service_metrics) in &metrics.service_metrics {
            if service_metrics.total_requests > 100 {
                let current_load = service_metrics.total_requests as f64 / 3600.0; // Requests per hour
                
                // Simple linear extrapolation
                if current_load > 1000.0 { // High load threshold
                    predictions.push(PerformancePrediction {
                        prediction_type: PredictionType::CapacityIssue,
                        component: format!("Service: {}", service_id),
                        predicted_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                        confidence: 0.7,
                        description: format!(
                            "Service {} approaching capacity limits at {:.0} requests/hour",
                            service_id, current_load
                        ),
                        recommended_actions: vec![
                            "Scale service horizontally".to_string(),
                            "Implement load balancing".to_string(),
                            "Monitor resource utilization".to_string(),
                        ],
                    });
                }
            }
        }

        predictions
    }

    /// Update trace metrics for distributed tracing analysis
    async fn update_trace_metrics(&self, metrics: &mut FederationMetrics, duration: Duration) {
        metrics.trace_statistics.total_spans += 1;
        metrics.trace_statistics.total_duration += duration;
        if metrics.trace_statistics.total_spans > 0 {
            metrics.trace_statistics.avg_span_duration = 
                metrics.trace_statistics.total_duration / metrics.trace_statistics.total_spans as u32;
        }

        // Update span duration histogram
        let millis = duration.as_millis() as u64;
        let bucket = if millis < 1 {
            "0-1ms"
        } else if millis < 5 {
            "1-5ms"
        } else if millis < 10 {
            "5-10ms"
        } else if millis < 50 {
            "10-50ms"
        } else if millis < 100 {
            "50-100ms"
        } else if millis < 500 {
            "100-500ms"
        } else {
            "500ms+"
        };

        *metrics.trace_statistics.span_duration_histogram
            .entry(bucket.to_string())
            .or_insert(0) += 1;
    }

    /// Calculate prediction confidence based on sample size and history
    fn calculate_prediction_confidence(&self, recent_samples: usize, total_samples: u64) -> f64 {
        let sample_confidence = (recent_samples as f64 / 50.0).min(1.0);
        let history_confidence = (total_samples as f64 / 1000.0).min(1.0);
        (sample_confidence + history_confidence) / 2.0
    }

    async fn get_top_errors(&self, metrics: &FederationMetrics) -> Vec<ErrorSummary> {
        // Group recent errors by type/message
        let mut error_counts = HashMap::new();

        for event in &metrics.federation_events {
            if matches!(
                event.event_type,
                FederationEventType::Error | FederationEventType::ServiceFailure
            ) {
                *error_counts.entry(event.details.clone()).or_insert(0) += 1;
            }
        }

        let mut errors: Vec<ErrorSummary> = error_counts
            .into_iter()
            .map(|(message, count)| ErrorSummary { message, count })
            .collect();

        errors.sort_by(|a, b| b.count.cmp(&a.count));
        errors.truncate(10); // Top 10 errors

        errors
    }

    async fn get_performance_summary(&self, metrics: &FederationMetrics) -> PerformanceSummary {
        let total_services = metrics.service_metrics.len();
        let healthy_services = metrics
            .service_metrics
            .values()
            .filter(|s| {
                let error_rate = if s.total_requests > 0 {
                    s.failed_requests as f64 / s.total_requests as f64
                } else {
                    0.0
                };
                error_rate < 0.1
            })
            .count();

        PerformanceSummary {
            total_services,
            healthy_services,
            avg_query_time: if !metrics.query_type_metrics.is_empty() {
                let total_duration: Duration = metrics
                    .query_type_metrics
                    .values()
                    .map(|m| m.total_duration)
                    .sum();
                let total_count: u64 = metrics
                    .query_type_metrics
                    .values()
                    .map(|m| m.total_count)
                    .sum();

                if total_count > 0 {
                    total_duration / total_count as u32
                } else {
                    Duration::from_secs(0)
                }
            } else {
                Duration::from_secs(0)
            },
            cache_efficiency: if !metrics.cache_metrics.is_empty() {
                let avg_hit_rate: f64 = metrics
                    .cache_metrics
                    .values()
                    .map(|c| c.hit_rate)
                    .sum::<f64>()
                    / metrics.cache_metrics.len() as f64;
                avg_hit_rate
            } else {
                0.0
            },
        }
    }
}

impl Default for FederationMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for federation monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationMonitorConfig {
    pub enable_detailed_metrics: bool,
    pub metrics_retention_hours: u64,
    pub max_recent_queries: usize,
    pub max_recent_events: usize,
    pub enable_prometheus_export: bool,
    pub health_check_interval: Duration,
    pub max_trace_spans: usize,
}

impl Default for FederationMonitorConfig {
    fn default() -> Self {
        Self {
            enable_detailed_metrics: true,
            metrics_retention_hours: 24,
            max_recent_queries: 1000,
            max_recent_events: 500,
            enable_prometheus_export: true,
            health_check_interval: Duration::from_secs(30),
            max_trace_spans: 10000,
        }
    }
}

/// Internal metrics storage with advanced observability features
#[derive(Debug)]
struct FederationMetrics {
    total_queries: u64,
    successful_queries: u64,
    failed_queries: u64,
    query_type_metrics: HashMap<String, QueryTypeMetrics>,
    service_metrics: HashMap<String, ServiceMetrics>,
    cache_metrics: HashMap<String, CacheMetrics>,
    response_time_histogram: HashMap<String, u64>,
    federation_events: Vec<FederationEvent>,
    event_type_counts: HashMap<FederationEventType, u64>,
    recent_queries: Vec<QueryRecord>,
    /// Advanced distributed tracing spans
    trace_spans: Vec<TraceSpan>,
    /// Trace statistics for analysis
    trace_statistics: TraceStatistics,
    /// Anomaly reports for intelligent monitoring
    anomalies: Vec<AnomalyReport>,
}

impl FederationMetrics {
    fn new() -> Self {
        Self {
            total_queries: 0,
            successful_queries: 0,
            failed_queries: 0,
            query_type_metrics: HashMap::new(),
            service_metrics: HashMap::new(),
            cache_metrics: HashMap::new(),
            response_time_histogram: HashMap::new(),
            federation_events: Vec::new(),
            event_type_counts: HashMap::new(),
            recent_queries: Vec::new(),
            trace_spans: Vec::new(),
            trace_statistics: TraceStatistics::new(),
            anomalies: Vec::new(),
        }
    }
}

/// Metrics for specific query types
#[derive(Debug, Clone, Serialize)]
pub struct QueryTypeMetrics {
    pub total_count: u64,
    pub success_count: u64,
    pub error_count: u64,
    pub total_duration: Duration,
    pub avg_duration: Duration,
}

impl QueryTypeMetrics {
    fn new() -> Self {
        Self {
            total_count: 0,
            success_count: 0,
            error_count: 0,
            total_duration: Duration::from_secs(0),
            avg_duration: Duration::from_secs(0),
        }
    }
}

/// Metrics for individual services
#[derive(Debug, Clone, Serialize)]
pub struct ServiceMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub total_duration: Duration,
    pub avg_duration: Duration,
    pub total_response_size: u64,
    pub avg_response_size: u64,
    pub last_seen: u64,
}

impl ServiceMetrics {
    fn new() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            total_duration: Duration::from_secs(0),
            avg_duration: Duration::from_secs(0),
            total_response_size: 0,
            avg_response_size: 0,
            last_seen: 0,
        }
    }
}

/// Cache performance metrics
#[derive(Debug, Clone, Serialize)]
pub struct CacheMetrics {
    pub total_requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

impl CacheMetrics {
    fn new() -> Self {
        Self {
            total_requests: 0,
            hits: 0,
            misses: 0,
            hit_rate: 0.0,
        }
    }
}

/// Types of federation events
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum FederationEventType {
    QueryStart,
    QueryComplete,
    ServiceRegistered,
    ServiceUnregistered,
    ServiceFailure,
    SchemaUpdate,
    CacheInvalidation,
    Error,
    Warning,
    EntityUpdate,
    SchemaChange,
    ServiceAvailability,
}

/// Federation event record
#[derive(Debug, Clone)]
struct FederationEvent {
    timestamp: u64,
    event_type: FederationEventType,
    details: String,
}

/// Query execution record
#[derive(Debug, Clone)]
struct QueryRecord {
    timestamp: u64,
    query_type: String,
    duration: Duration,
    success: bool,
}

/// Health status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

/// Monitoring statistics
#[derive(Debug, Clone, Serialize)]
pub struct MonitorStats {
    pub uptime: Duration,
    pub total_queries: u64,
    pub successful_queries: u64,
    pub failed_queries: u64,
    pub success_rate: f64,
    pub query_type_metrics: HashMap<String, QueryTypeMetrics>,
    pub service_metrics: HashMap<String, ServiceMetrics>,
    pub cache_metrics: HashMap<String, CacheMetrics>,
    pub response_time_histogram: HashMap<String, u64>,
    pub recent_events_count: usize,
    pub event_type_counts: HashMap<FederationEventType, u64>,
    pub avg_queries_per_second: f64,
}

/// Health monitoring metrics
#[derive(Debug, Clone, Serialize)]
pub struct HealthMetrics {
    pub overall_health: HealthStatus,
    pub service_health: HashMap<String, HealthStatus>,
    pub error_rate: f64,
    pub avg_response_time: Duration,
    pub active_services: usize,
    pub recent_error_count: usize,
    pub cache_hit_rate: f64,
    pub timestamp: u64,
}

/// Internal health indicators
struct HealthIndicators {
    overall_health: HealthStatus,
    service_health: HashMap<String, HealthStatus>,
    error_rate: f64,
    avg_response_time: Duration,
    recent_error_count: usize,
    cache_hit_rate: f64,
}

/// Performance report
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceReport {
    pub report_timestamp: u64,
    pub uptime: Duration,
    pub total_queries: u64,
    pub overall_success_rate: f64,
    pub query_trends: HashMap<String, QueryTrend>,
    pub top_errors: Vec<ErrorSummary>,
    pub performance_summary: PerformanceSummary,
    pub bottlenecks: Vec<BottleneckReport>,
    pub performance_regressions: Vec<RegressionReport>,
    pub optimization_recommendations: Vec<OptimizationRecommendation>,
}

/// Query performance trend
#[derive(Debug, Clone, Serialize)]
pub struct QueryTrend {
    pub query_type: String,
    pub total_queries: u64,
    pub avg_response_time: Duration,
    pub error_rate: f64,
    pub queries_per_second: f64,
}

/// Error summary for reporting
#[derive(Debug, Clone, Serialize)]
pub struct ErrorSummary {
    pub message: String,
    pub count: u64,
}

/// Performance summary
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceSummary {
    pub total_services: usize,
    pub healthy_services: usize,
    pub avg_query_time: Duration,
    pub cache_efficiency: f64,
}

/// Bottleneck analysis report
#[derive(Debug, Clone, Serialize)]
pub struct BottleneckReport {
    pub bottleneck_type: BottleneckType,
    pub component: String,
    pub severity: BottleneckSeverity,
    pub description: String,
    pub metric_value: f64,
    pub threshold: f64,
    pub impact_score: f64,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, Serialize)]
pub enum BottleneckType {
    SlowService,
    HighErrorRate,
    PoorCachePerformance,
    NetworkLatency,
    ResourceContention,
    QueryComplexity,
}

/// Severity levels for bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum BottleneckSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Performance regression report
#[derive(Debug, Clone, Serialize)]
pub struct RegressionReport {
    pub component: String,
    pub regression_type: RegressionType,
    pub severity: RegressionSeverity,
    pub description: String,
    pub historical_value: f64,
    pub current_value: f64,
    pub detected_at: u64,
    pub confidence: f64,
}

/// Types of performance regressions
#[derive(Debug, Clone, Copy, Serialize)]
pub enum RegressionType {
    ResponseTimeIncrease,
    ErrorRateIncrease,
    ThroughputDecrease,
    CacheHitRateDecrease,
}

/// Severity levels for regressions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum RegressionSeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Optimization recommendation
#[derive(Debug, Clone, Serialize)]
pub struct OptimizationRecommendation {
    pub category: OptimizationCategory,
    pub priority: OptimizationPriority,
    pub title: String,
    pub description: String,
    pub estimated_impact: String,
    pub implementation_effort: ImplementationEffort,
    pub metrics_to_monitor: Vec<String>,
}

/// Categories of optimizations
#[derive(Debug, Clone, Copy, Serialize)]
pub enum OptimizationCategory {
    Caching,
    Performance,
    Scaling,
    QueryOptimization,
    NetworkOptimization,
    ResourceUtilization,
}

/// Priority levels for optimization recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum OptimizationPriority {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Implementation effort estimation
#[derive(Debug, Clone, Copy, Serialize)]
pub enum ImplementationEffort {
    Low,    // Hours to 1 day
    Medium, // 1-3 days
    High,   // 1+ weeks
}

/// Distributed tracing span
#[derive(Debug, Clone)]
pub struct TraceSpan {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub start_time: SystemTime,
    pub duration: Duration,
    pub tags: HashMap<String, String>,
    pub service_id: Option<String>,
}

/// Trace statistics for distributed tracing analysis
#[derive(Debug, Clone)]
pub struct TraceStatistics {
    pub total_spans: u64,
    pub total_duration: Duration,
    pub avg_span_duration: Duration,
    pub span_duration_histogram: HashMap<String, u64>,
}

impl TraceStatistics {
    fn new() -> Self {
        Self {
            total_spans: 0,
            total_duration: Duration::from_secs(0),
            avg_span_duration: Duration::from_secs(0),
            span_duration_histogram: HashMap::new(),
        }
    }
}

/// Anomaly detection report
#[derive(Debug, Clone, Serialize)]
pub struct AnomalyReport {
    pub anomaly_type: AnomalyType,
    pub detected_at: u64,
    pub details: String,
    pub severity: AnomalySeverity,
    pub confidence: f64,
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, Copy, Serialize)]
pub enum AnomalyType {
    ErrorSpike,
    PerformanceDegradation,
    UnusualTrafficPattern,
    ServiceUnavailability,
    MemoryLeak,
    ResourceExhaustion,
}

/// Severity levels for anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum AnomalySeverity {
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Performance prediction
#[derive(Debug, Clone, Serialize)]
pub struct PerformancePrediction {
    pub prediction_type: PredictionType,
    pub component: String,
    pub predicted_at: u64,
    pub confidence: f64,
    pub description: String,
    pub recommended_actions: Vec<String>,
}

/// Types of performance predictions
#[derive(Debug, Clone, Copy, Serialize)]
pub enum PredictionType {
    PerformanceDegradation,
    CapacityIssue,
    ServiceFailure,
    ResourceBottleneck,
}

/// Cross-service latency analysis
#[derive(Debug, Clone, Serialize)]
pub struct CrossServiceLatencyAnalysis {
    pub interactions: Vec<ServiceInteractionLatency>,
    pub total_traces_analyzed: usize,
    pub analysis_timestamp: u64,
}

/// Latency metrics for service interactions
#[derive(Debug, Clone, Serialize)]
pub struct ServiceInteractionLatency {
    pub from_service: String,
    pub to_service: String,
    pub avg_latency: Duration,
    pub min_latency: Duration,
    pub max_latency: Duration,
    pub sample_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_monitor_creation() {
        let monitor = FederationMonitor::new();
        let stats = monitor.get_stats().await;

        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.success_rate, 0.0);
    }

    #[tokio::test]
    async fn test_query_recording() {
        let monitor = FederationMonitor::new();

        monitor
            .record_query_execution("sparql", Duration::from_millis(100), true)
            .await;
        monitor
            .record_query_execution("graphql", Duration::from_millis(200), false)
            .await;

        let stats = monitor.get_stats().await;
        assert_eq!(stats.total_queries, 2);
        assert_eq!(stats.successful_queries, 1);
        assert_eq!(stats.failed_queries, 1);
        assert_eq!(stats.success_rate, 0.5);
    }

    #[tokio::test]
    async fn test_service_metrics() {
        let monitor = FederationMonitor::new();

        monitor
            .record_service_interaction("service1", Duration::from_millis(150), true, Some(1024))
            .await;
        monitor
            .record_service_interaction("service1", Duration::from_millis(250), false, None)
            .await;

        let stats = monitor.get_stats().await;
        assert!(stats.service_metrics.contains_key("service1"));

        let service_metrics = &stats.service_metrics["service1"];
        assert_eq!(service_metrics.total_requests, 2);
        assert_eq!(service_metrics.successful_requests, 1);
        assert_eq!(service_metrics.failed_requests, 1);
    }

    #[tokio::test]
    async fn test_cache_metrics() {
        let monitor = FederationMonitor::new();

        monitor.record_cache_hit("query_cache", true).await;
        monitor.record_cache_hit("query_cache", true).await;
        monitor.record_cache_hit("query_cache", false).await;

        let stats = monitor.get_stats().await;
        assert!(stats.cache_metrics.contains_key("query_cache"));

        let cache_metrics = &stats.cache_metrics["query_cache"];
        assert_eq!(cache_metrics.total_requests, 3);
        assert_eq!(cache_metrics.hits, 2);
        assert_eq!(cache_metrics.misses, 1);
        assert!((cache_metrics.hit_rate - 0.666).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_event_recording() {
        let monitor = FederationMonitor::new();

        monitor
            .record_federation_event(FederationEventType::ServiceRegistered, "New service added")
            .await;
        monitor
            .record_federation_event(FederationEventType::Error, "Query failed")
            .await;

        let stats = monitor.get_stats().await;
        assert_eq!(stats.recent_events_count, 2);
        assert_eq!(
            stats.event_type_counts[&FederationEventType::ServiceRegistered],
            1
        );
        assert_eq!(stats.event_type_counts[&FederationEventType::Error], 1);
    }

    #[tokio::test]
    async fn test_prometheus_export() {
        let monitor = FederationMonitor::new();

        monitor
            .record_query_execution("sparql", Duration::from_millis(100), true)
            .await;

        let prometheus_output = monitor.export_prometheus_metrics().await;
        assert!(prometheus_output.contains("oxirs_federation_queries_total"));
        assert!(prometheus_output.contains("oxirs_federation_success_rate"));
    }

    #[tokio::test]
    async fn test_health_metrics() {
        let monitor = FederationMonitor::new();

        // Record some successful queries
        for _ in 0..9 {
            monitor
                .record_query_execution("sparql", Duration::from_millis(100), true)
                .await;
        }
        // Record one failed query
        monitor
            .record_query_execution("sparql", Duration::from_millis(100), false)
            .await;

        let health = monitor.get_health_metrics().await;
        assert_eq!(health.overall_health, HealthStatus::Healthy); // 10% error rate is still healthy
    }
}
