//! Performance analyzer metrics collector and analysis engine.
//!
//! Houses the [`PerformanceAnalyzer`] struct, which collects system, service,
//! and query metrics, identifies bottlenecks, computes trends, and raises
//! alerts based on configurable thresholds.

use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::performance_analyzer_types::*;

/// Performance analysis engine for the federation system
pub struct PerformanceAnalyzer {
    pub(crate) config: AnalyzerConfig,
    pub(crate) metrics_history: Arc<RwLock<MetricsHistory>>,
    pub(crate) bottleneck_detector: BottleneckDetector,
    pub(crate) recommendation_engine: RecommendationEngine,
    pub(crate) alert_thresholds: AlertThresholds,
}

impl PerformanceAnalyzer {
    /// Create a new performance analyzer
    pub fn new() -> Self {
        Self::with_config(AnalyzerConfig::default())
    }

    /// Create a new performance analyzer with configuration
    pub fn with_config(config: AnalyzerConfig) -> Self {
        let metrics_history = Arc::new(RwLock::new(MetricsHistory {
            system_metrics: VecDeque::new(),
            service_metrics: HashMap::new(),
            query_metrics: VecDeque::new(),
            max_entries: (config.history_retention_hours * 60) as usize, // Assume 1 entry per minute
        }));

        let bottleneck_detector = BottleneckDetector {
            baseline_metrics: None,
            detection_thresholds: DetectionThresholds {
                latency_degradation_threshold: 50.0,    // 50% increase
                throughput_degradation_threshold: 20.0, // 20% decrease
                error_rate_threshold: 0.05,             // 5%
                memory_usage_threshold: 0.85,           // 85%
                cpu_usage_threshold: 0.90,              // 90%
                cache_hit_rate_threshold: 0.80,         // 80%
            },
        };

        let recommendation_engine = RecommendationEngine {
            rule_base: Self::initialize_rule_base(),
        };

        let alert_thresholds = AlertThresholds {
            critical_latency_ms: 5000,   // 5 seconds
            critical_error_rate: 0.10,   // 10%
            critical_memory_usage: 0.95, // 95%
            critical_cpu_usage: 0.95,    // 95%
        };

        Self {
            config,
            metrics_history,
            bottleneck_detector,
            recommendation_engine,
            alert_thresholds,
        }
    }

    /// Record system performance metrics
    pub async fn record_system_metrics(&self, metrics: SystemPerformanceMetrics) -> Result<()> {
        let mut history = self.metrics_history.write().await;

        // Add new metrics
        history.system_metrics.push_back(metrics.clone());

        // Maintain size limit
        while history.system_metrics.len() > history.max_entries {
            history.system_metrics.pop_front();
        }

        debug!(
            "Recorded system metrics - Latency P95: {}ms, Throughput: {:.1} QPS, Error Rate: {:.3}%",
            metrics.overall_latency_p95.as_millis(),
            metrics.throughput_qps,
            metrics.error_rate * 100.0
        );

        Ok(())
    }

    /// Record service performance metrics
    pub async fn record_service_metrics(&self, metrics: ServicePerformanceMetrics) -> Result<()> {
        let mut history = self.metrics_history.write().await;

        let max_entries = history.max_entries; // Cache the value before mutable borrow
        let service_history = history
            .service_metrics
            .entry(metrics.service_id.clone())
            .or_default();

        service_history.push_back(metrics.clone());

        // Maintain size limit
        while service_history.len() > max_entries {
            service_history.pop_front();
        }

        debug!(
            "Recorded service metrics for {} - Response Time P95: {}ms, RPS: {:.1}",
            metrics.service_id,
            metrics.response_time_p95.as_millis(),
            metrics.requests_per_second
        );

        Ok(())
    }

    /// Record query execution metrics
    pub async fn record_query_metrics(&self, metrics: QueryExecutionMetrics) -> Result<()> {
        let mut history = self.metrics_history.write().await;

        history.query_metrics.push_back(metrics.clone());

        // Maintain size limit
        while history.query_metrics.len() > history.max_entries {
            history.query_metrics.pop_front();
        }

        debug!(
            "Recorded query metrics - Query ID: {}, Total Time: {}ms, Services: {}",
            metrics.query_id,
            metrics.total_execution_time.as_millis(),
            metrics.services_involved.len()
        );

        Ok(())
    }

    /// Analyze performance and identify bottlenecks
    pub async fn analyze_performance(&self) -> Result<BottleneckAnalysis> {
        let history = self.metrics_history.read().await;

        if history.system_metrics.len() < self.config.min_data_points {
            return Err(anyhow!(
                "Insufficient data points for analysis: {} < {}",
                history.system_metrics.len(),
                self.config.min_data_points
            ));
        }

        let recent_metrics = history
            .system_metrics
            .back()
            .expect("operation should succeed");
        let baseline = self
            .bottleneck_detector
            .baseline_metrics
            .as_ref()
            .unwrap_or(
                history
                    .system_metrics
                    .front()
                    .expect("reference should be available"),
            );

        let mut analysis = BottleneckAnalysis {
            primary_bottleneck: BottleneckType::Unknown,
            contributing_factors: Vec::new(),
            severity_score: 0.0,
            confidence_level: 0.0,
            impact_on_throughput: 0.0,
            recommended_actions: Vec::new(),
        };

        // Analyze different bottleneck types
        self.analyze_network_bottlenecks(&mut analysis, recent_metrics, baseline);
        self.analyze_resource_bottlenecks(&mut analysis, recent_metrics, baseline);
        self.analyze_service_bottlenecks(&mut analysis, &history, recent_metrics);
        self.analyze_cache_performance(&mut analysis, recent_metrics, baseline);

        // Determine primary bottleneck
        if !analysis.contributing_factors.is_empty() {
            let primary_factor = analysis
                .contributing_factors
                .iter()
                .max_by(|a, b| {
                    a.weight
                        .partial_cmp(&b.weight)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("collection should not be empty");

            analysis.primary_bottleneck = match primary_factor.factor_type {
                FactorType::Latency => BottleneckType::NetworkLatency,
                FactorType::ResourceUtilization => BottleneckType::CPUUtilization,
                FactorType::CachePerformance => BottleneckType::CacheInefficiency,
                _ => BottleneckType::Unknown,
            };

            analysis.severity_score = primary_factor.weight;
            analysis.confidence_level = self.calculate_confidence_level(&analysis);
        }

        // Generate recommendations
        analysis.recommended_actions = self.generate_bottleneck_recommendations(&analysis).await;

        info!(
            "Performance analysis completed - Primary bottleneck: {:?}, Severity: {:.2}",
            analysis.primary_bottleneck, analysis.severity_score
        );

        Ok(analysis)
    }

    /// Analyze performance trends
    pub async fn analyze_trends(&self) -> Result<Vec<PerformanceTrend>> {
        let history = self.metrics_history.read().await;
        let mut trends = Vec::new();

        if history.system_metrics.len() < self.config.min_data_points {
            return Ok(trends);
        }

        // Analyze latency trend
        let latency_values: Vec<f64> = history
            .system_metrics
            .iter()
            .map(|m| m.overall_latency_p95.as_millis() as f64)
            .collect();

        trends.push(self.calculate_trend("latency_p95", &latency_values));

        // Analyze throughput trend
        let throughput_values: Vec<f64> = history
            .system_metrics
            .iter()
            .map(|m| m.throughput_qps)
            .collect();

        trends.push(self.calculate_trend("throughput_qps", &throughput_values));

        // Analyze error rate trend
        let error_rate_values: Vec<f64> = history
            .system_metrics
            .iter()
            .map(|m| m.error_rate)
            .collect();

        trends.push(self.calculate_trend("error_rate", &error_rate_values));

        Ok(trends)
    }

    /// Check for performance alerts
    pub async fn check_alerts(&self) -> Result<Vec<PerformanceAlert>> {
        let history = self.metrics_history.read().await;
        let mut alerts = Vec::new();

        if let Some(recent_metrics) = history.system_metrics.back() {
            // Check critical latency
            if recent_metrics.overall_latency_p95.as_millis()
                > self.alert_thresholds.critical_latency_ms
            {
                alerts.push(PerformanceAlert {
                    severity: AlertSeverity::Critical,
                    title: "High Latency Detected".to_string(),
                    description: format!(
                        "P95 latency is {}ms, exceeding threshold of {}ms",
                        recent_metrics.overall_latency_p95.as_millis(),
                        self.alert_thresholds.critical_latency_ms
                    ),
                    timestamp: recent_metrics.timestamp,
                });
            }

            // Check critical error rate
            if recent_metrics.error_rate > self.alert_thresholds.critical_error_rate {
                alerts.push(PerformanceAlert {
                    severity: AlertSeverity::Critical,
                    title: "High Error Rate Detected".to_string(),
                    description: format!(
                        "Error rate is {:.1}%, exceeding threshold of {:.1}%",
                        recent_metrics.error_rate * 100.0,
                        self.alert_thresholds.critical_error_rate * 100.0
                    ),
                    timestamp: recent_metrics.timestamp,
                });
            }

            // Check memory usage
            if recent_metrics.memory_usage_mb > self.alert_thresholds.critical_memory_usage {
                alerts.push(PerformanceAlert {
                    severity: AlertSeverity::Warning,
                    title: "High Memory Usage".to_string(),
                    description: format!(
                        "Memory usage is {:.1}MB, approaching limits",
                        recent_metrics.memory_usage_mb
                    ),
                    timestamp: recent_metrics.timestamp,
                });
            }
        }

        if !alerts.is_empty() {
            warn!("Performance alerts detected: {} alerts", alerts.len());
        }

        Ok(alerts)
    }

    // Private helper methods used internally and by the reporter sibling.

    pub(crate) fn analyze_network_bottlenecks(
        &self,
        analysis: &mut BottleneckAnalysis,
        current: &SystemPerformanceMetrics,
        baseline: &SystemPerformanceMetrics,
    ) {
        let latency_increase = (current.overall_latency_p95.as_millis() as f64
            - baseline.overall_latency_p95.as_millis() as f64)
            / baseline.overall_latency_p95.as_millis() as f64;

        if latency_increase
            > self
                .bottleneck_detector
                .detection_thresholds
                .latency_degradation_threshold
                / 100.0
        {
            analysis.contributing_factors.push(BottleneckFactor {
                factor_type: FactorType::Latency,
                description: "Network latency has increased significantly".to_string(),
                weight: (latency_increase * 100.0).min(100.0),
                metric_value: current.overall_latency_p95.as_millis() as f64,
                threshold: baseline.overall_latency_p95.as_millis() as f64 * 1.5,
            });
        }
    }

    pub(crate) fn analyze_resource_bottlenecks(
        &self,
        analysis: &mut BottleneckAnalysis,
        current: &SystemPerformanceMetrics,
        _baseline: &SystemPerformanceMetrics,
    ) {
        // Check CPU utilization
        if current.cpu_usage_percent
            > self
                .bottleneck_detector
                .detection_thresholds
                .cpu_usage_threshold
        {
            analysis.contributing_factors.push(BottleneckFactor {
                factor_type: FactorType::ResourceUtilization,
                description: "High CPU utilization detected".to_string(),
                weight: current.cpu_usage_percent,
                metric_value: current.cpu_usage_percent,
                threshold: self
                    .bottleneck_detector
                    .detection_thresholds
                    .cpu_usage_threshold,
            });
        }

        // Check memory usage
        if current.memory_usage_mb
            > self
                .bottleneck_detector
                .detection_thresholds
                .memory_usage_threshold
        {
            analysis.contributing_factors.push(BottleneckFactor {
                factor_type: FactorType::ResourceUtilization,
                description: "High memory utilization detected".to_string(),
                weight: (current.memory_usage_mb / 1024.0) * 100.0, // Convert to percentage
                metric_value: current.memory_usage_mb,
                threshold: self
                    .bottleneck_detector
                    .detection_thresholds
                    .memory_usage_threshold,
            });
        }
    }

    pub(crate) fn analyze_service_bottlenecks(
        &self,
        analysis: &mut BottleneckAnalysis,
        history: &MetricsHistory,
        _current: &SystemPerformanceMetrics,
    ) {
        // Check if any services are consistently slow
        for (service_id, service_metrics) in &history.service_metrics {
            if let Some(recent_metric) = service_metrics.back() {
                if recent_metric.response_time_p95.as_millis() > 1000 {
                    analysis.contributing_factors.push(BottleneckFactor {
                        factor_type: FactorType::Latency,
                        description: format!("Service {service_id} has high response times"),
                        weight: recent_metric.response_time_p95.as_millis() as f64 / 100.0,
                        metric_value: recent_metric.response_time_p95.as_millis() as f64,
                        threshold: 1000.0,
                    });
                }
            }
        }
    }

    pub(crate) fn analyze_cache_performance(
        &self,
        analysis: &mut BottleneckAnalysis,
        current: &SystemPerformanceMetrics,
        _baseline: &SystemPerformanceMetrics,
    ) {
        if current.cache_hit_rate
            < self
                .bottleneck_detector
                .detection_thresholds
                .cache_hit_rate_threshold
        {
            analysis.contributing_factors.push(BottleneckFactor {
                factor_type: FactorType::CachePerformance,
                description: "Low cache hit rate affecting performance".to_string(),
                weight: (1.0 - current.cache_hit_rate) * 100.0,
                metric_value: current.cache_hit_rate,
                threshold: self
                    .bottleneck_detector
                    .detection_thresholds
                    .cache_hit_rate_threshold,
            });
        }
    }

    pub(crate) fn calculate_confidence_level(&self, analysis: &BottleneckAnalysis) -> f64 {
        let factor_count = analysis.contributing_factors.len() as f64;
        let max_weight = analysis
            .contributing_factors
            .iter()
            .map(|f| f.weight)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        // Higher confidence with more factors and higher weights
        ((factor_count * 0.2) + (max_weight / 200.0)).min(1.0)
    }

    pub(crate) async fn generate_bottleneck_recommendations(
        &self,
        analysis: &BottleneckAnalysis,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match analysis.primary_bottleneck {
            BottleneckType::NetworkLatency => {
                recommendations
                    .push("Enable request batching to reduce network round trips".to_string());
                recommendations.push("Implement more aggressive caching strategies".to_string());
                recommendations
                    .push("Consider query optimization to reduce data transfer".to_string());
            }
            BottleneckType::ServiceResponseTime => {
                recommendations
                    .push("Analyze slow services and optimize their queries".to_string());
                recommendations.push("Implement service-level caching".to_string());
                recommendations.push("Consider load balancing across service replicas".to_string());
            }
            BottleneckType::MemoryPressure => {
                recommendations.push("Implement result streaming for large queries".to_string());
                recommendations.push("Reduce batch sizes to decrease memory usage".to_string());
                recommendations
                    .push("Enable memory-efficient query execution strategies".to_string());
            }
            BottleneckType::CPUUtilization => {
                recommendations.push("Scale CPU resources horizontally or vertically".to_string());
                recommendations.push("Optimize query execution algorithms".to_string());
                recommendations.push("Implement query complexity limits".to_string());
            }
            _ => {
                recommendations.push(
                    "Monitor system metrics more closely to identify bottlenecks".to_string(),
                );
            }
        }

        recommendations
    }

    pub(crate) fn calculate_trend(&self, metric_name: &str, values: &[f64]) -> PerformanceTrend {
        if values.len() < 2 {
            return PerformanceTrend {
                metric_name: metric_name.to_string(),
                trend_direction: TrendDirection::Stable,
                rate_of_change: 0.0,
                confidence: 0.0,
                prediction_accuracy: 0.0,
            };
        }

        // Simple linear regression for trend calculation
        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x_sq_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum.powi(2));

        let direction = if slope > 0.05 {
            TrendDirection::Improving
        } else if slope < -0.05 {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        };

        PerformanceTrend {
            metric_name: metric_name.to_string(),
            trend_direction: direction,
            rate_of_change: slope * 100.0,   // Convert to percentage
            confidence: (n / 20.0).min(1.0), // Higher confidence with more data points
            prediction_accuracy: 0.8,        // Simplified prediction accuracy
        }
    }
}

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
