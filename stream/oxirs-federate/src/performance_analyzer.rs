//! # Comprehensive Performance Analysis Engine
//!
//! This module provides advanced performance analysis, bottleneck identification,
//! and optimization recommendations for the federated query system.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Performance analysis engine for the federation system
pub struct PerformanceAnalyzer {
    config: AnalyzerConfig,
    metrics_history: Arc<RwLock<MetricsHistory>>,
    bottleneck_detector: BottleneckDetector,
    recommendation_engine: RecommendationEngine,
    alert_thresholds: AlertThresholds,
}

/// Configuration for the performance analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyzerConfig {
    /// Enable real-time performance monitoring
    pub enable_real_time_monitoring: bool,
    /// History retention period
    pub history_retention_hours: u64,
    /// Analysis interval
    pub analysis_interval: Duration,
    /// Enable predictive analysis
    pub enable_predictive_analysis: bool,
    /// Minimum data points for reliable analysis
    pub min_data_points: usize,
    /// Performance baseline update frequency
    pub baseline_update_frequency: Duration,
}

impl Default for AnalyzerConfig {
    fn default() -> Self {
        Self {
            enable_real_time_monitoring: true,
            history_retention_hours: 24,
            analysis_interval: Duration::from_secs(60),
            enable_predictive_analysis: true,
            min_data_points: 10,
            baseline_update_frequency: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// System-wide performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPerformanceMetrics {
    pub timestamp: SystemTime,
    pub overall_latency_p50: Duration,
    pub overall_latency_p95: Duration,
    pub overall_latency_p99: Duration,
    pub throughput_qps: f64,
    pub error_rate: f64,
    pub timeout_rate: f64,
    pub cache_hit_rate: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub network_bandwidth_mbps: f64,
    pub active_connections: usize,
    pub queue_depth: usize,
}

/// Service-specific performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePerformanceMetrics {
    pub service_id: String,
    pub timestamp: SystemTime,
    pub response_time_p50: Duration,
    pub response_time_p95: Duration,
    pub response_time_p99: Duration,
    pub requests_per_second: f64,
    pub error_rate: f64,
    pub timeout_rate: f64,
    pub availability: f64,
    pub data_transfer_kb: f64,
    pub connection_pool_utilization: f64,
}

/// Query execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExecutionMetrics {
    pub query_id: String,
    pub timestamp: SystemTime,
    pub total_execution_time: Duration,
    pub planning_time: Duration,
    pub execution_time: Duration,
    pub result_serialization_time: Duration,
    pub services_involved: Vec<String>,
    pub result_size_bytes: u64,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub parallel_steps: usize,
    pub sequential_steps: usize,
}

/// Bottleneck identification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub primary_bottleneck: BottleneckType,
    pub contributing_factors: Vec<BottleneckFactor>,
    pub severity_score: f64,       // 0.0 - 1.0
    pub confidence_level: f64,     // 0.0 - 1.0
    pub impact_on_throughput: f64, // percentage
    pub recommended_actions: Vec<String>,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BottleneckType {
    NetworkLatency,
    ServiceResponseTime,
    MemoryPressure,
    CPUUtilization,
    ConnectionPoolExhaustion,
    QueryComplexity,
    DataTransferVolume,
    CacheInefficiency,
    ParallelizationLimits,
    Unknown,
}

/// Factors contributing to bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckFactor {
    pub factor_type: FactorType,
    pub description: String,
    pub weight: f64, // contribution weight
    pub metric_value: f64,
    pub threshold: f64,
}

/// Types of bottleneck factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactorType {
    Latency,
    Throughput,
    ErrorRate,
    ResourceUtilization,
    QueueDepth,
    CachePerformance,
}

/// Performance optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendations {
    pub high_priority: Vec<Recommendation>,
    pub medium_priority: Vec<Recommendation>,
    pub low_priority: Vec<Recommendation>,
    pub long_term: Vec<Recommendation>,
}

/// Individual optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub title: String,
    pub description: String,
    pub category: RecommendationCategory,
    pub expected_improvement: String,
    pub implementation_effort: ImplementationEffort,
    pub estimated_impact_score: f64, // 0.0 - 1.0
}

/// Recommendation categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Configuration,
    Architecture,
    QueryOptimization,
    ResourceScaling,
    CachingStrategy,
    NetworkOptimization,
}

/// Implementation effort levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Low,    // Configuration changes
    Medium, // Code changes
    High,   // Architecture changes
}

/// Performance trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrend {
    pub metric_name: String,
    pub trend_direction: TrendDirection,
    pub rate_of_change: f64,      // percentage per hour
    pub confidence: f64,          // 0.0 - 1.0
    pub prediction_accuracy: f64, // 0.0 - 1.0
}

/// Trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Degrading,
    Stable,
    Volatile,
}

/// Historical metrics storage
#[derive(Debug)]
pub struct MetricsHistory {
    pub system_metrics: VecDeque<SystemPerformanceMetrics>,
    pub service_metrics: HashMap<String, VecDeque<ServicePerformanceMetrics>>,
    pub query_metrics: VecDeque<QueryExecutionMetrics>,
    pub max_entries: usize,
}

/// Bottleneck detection engine
#[derive(Debug)]
pub struct BottleneckDetector {
    baseline_metrics: Option<SystemPerformanceMetrics>,
    detection_thresholds: DetectionThresholds,
}

/// Detection thresholds for bottlenecks
#[derive(Debug, Clone)]
pub struct DetectionThresholds {
    pub latency_degradation_threshold: f64, // percentage increase
    pub throughput_degradation_threshold: f64, // percentage decrease
    pub error_rate_threshold: f64,          // error rate threshold
    pub memory_usage_threshold: f64,        // percentage of total memory
    pub cpu_usage_threshold: f64,           // percentage
    pub cache_hit_rate_threshold: f64,      // minimum cache hit rate
}

/// Recommendation engine
#[derive(Debug)]
pub struct RecommendationEngine {
    rule_base: Vec<OptimizationRule>,
}

/// Optimization rule
#[derive(Debug, Clone)]
pub struct OptimizationRule {
    pub condition: RuleCondition,
    pub recommendation: Recommendation,
    pub priority: f64,
}

/// Rule condition
#[derive(Debug, Clone)]
pub struct RuleCondition {
    pub metric_type: String,
    pub operator: ComparisonOperator,
    pub threshold: f64,
}

/// Comparison operators for rules
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equals,
    NotEquals,
}

/// Alert thresholds
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    pub critical_latency_ms: u128,
    pub critical_error_rate: f64,
    pub critical_memory_usage: f64,
    pub critical_cpu_usage: f64,
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

        let recent_metrics = history.system_metrics.back().unwrap();
        let baseline = self
            .bottleneck_detector
            .baseline_metrics
            .as_ref()
            .unwrap_or(history.system_metrics.front().unwrap());

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
                .max_by(|a, b| a.weight.partial_cmp(&b.weight).unwrap())
                .unwrap();

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

    /// Generate comprehensive optimization recommendations
    pub async fn generate_recommendations(&self) -> Result<OptimizationRecommendations> {
        let analysis = self.analyze_performance().await?;
        let history = self.metrics_history.read().await;

        let mut recommendations = OptimizationRecommendations {
            high_priority: Vec::new(),
            medium_priority: Vec::new(),
            low_priority: Vec::new(),
            long_term: Vec::new(),
        };

        // Apply rule-based recommendations
        for rule in &self.recommendation_engine.rule_base {
            if self.evaluate_rule_condition(&rule.condition, &history) {
                match rule.priority {
                    p if p >= 0.8 => recommendations
                        .high_priority
                        .push(rule.recommendation.clone()),
                    p if p >= 0.6 => recommendations
                        .medium_priority
                        .push(rule.recommendation.clone()),
                    p if p >= 0.4 => recommendations
                        .low_priority
                        .push(rule.recommendation.clone()),
                    _ => recommendations.long_term.push(rule.recommendation.clone()),
                }
            }
        }

        // Add bottleneck-specific recommendations
        let bottleneck_recommendations =
            self.generate_bottleneck_specific_recommendations(&analysis);
        recommendations
            .high_priority
            .extend(bottleneck_recommendations);

        info!(
            "Generated {} high-priority, {} medium-priority, {} low-priority, and {} long-term recommendations",
            recommendations.high_priority.len(),
            recommendations.medium_priority.len(),
            recommendations.low_priority.len(),
            recommendations.long_term.len()
        );

        Ok(recommendations)
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

    // Private helper methods

    fn analyze_network_bottlenecks(
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

    fn analyze_resource_bottlenecks(
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

    fn analyze_service_bottlenecks(
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

    fn analyze_cache_performance(
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

    fn calculate_confidence_level(&self, analysis: &BottleneckAnalysis) -> f64 {
        let factor_count = analysis.contributing_factors.len() as f64;
        let max_weight = analysis
            .contributing_factors
            .iter()
            .map(|f| f.weight)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        // Higher confidence with more factors and higher weights
        ((factor_count * 0.2) + (max_weight / 200.0)).min(1.0)
    }

    async fn generate_bottleneck_recommendations(
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

    fn generate_bottleneck_specific_recommendations(
        &self,
        analysis: &BottleneckAnalysis,
    ) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        for factor in &analysis.contributing_factors {
            match factor.factor_type {
                FactorType::Latency => {
                    recommendations.push(Recommendation {
                        title: "Optimize Network Performance".to_string(),
                        description:
                            "Implement compression and request batching to reduce network overhead"
                                .to_string(),
                        category: RecommendationCategory::NetworkOptimization,
                        expected_improvement: "20-40% reduction in response times".to_string(),
                        implementation_effort: ImplementationEffort::Medium,
                        estimated_impact_score: 0.7,
                    });
                }
                FactorType::ResourceUtilization => {
                    recommendations.push(Recommendation {
                        title: "Scale System Resources".to_string(),
                        description: "Increase CPU and memory allocation to handle current load"
                            .to_string(),
                        category: RecommendationCategory::ResourceScaling,
                        expected_improvement: "Improved response times and throughput".to_string(),
                        implementation_effort: ImplementationEffort::Low,
                        estimated_impact_score: 0.8,
                    });
                }
                FactorType::CachePerformance => {
                    recommendations.push(Recommendation {
                        title: "Improve Caching Strategy".to_string(),
                        description: "Optimize cache size, TTL, and eviction policies".to_string(),
                        category: RecommendationCategory::CachingStrategy,
                        expected_improvement: "30-50% reduction in backend service load"
                            .to_string(),
                        implementation_effort: ImplementationEffort::Medium,
                        estimated_impact_score: 0.6,
                    });
                }
                _ => {}
            }
        }

        recommendations
    }

    fn calculate_trend(&self, metric_name: &str, values: &[f64]) -> PerformanceTrend {
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

    fn evaluate_rule_condition(&self, condition: &RuleCondition, history: &MetricsHistory) -> bool {
        if let Some(recent_metrics) = history.system_metrics.back() {
            let metric_value = match condition.metric_type.as_str() {
                "latency_p95" => recent_metrics.overall_latency_p95.as_millis() as f64,
                "error_rate" => recent_metrics.error_rate,
                "throughput" => recent_metrics.throughput_qps,
                "memory_usage" => recent_metrics.memory_usage_mb,
                "cpu_usage" => recent_metrics.cpu_usage_percent,
                _ => return false,
            };

            match condition.operator {
                ComparisonOperator::GreaterThan => metric_value > condition.threshold,
                ComparisonOperator::LessThan => metric_value < condition.threshold,
                ComparisonOperator::Equals => (metric_value - condition.threshold).abs() < 0.01,
                ComparisonOperator::NotEquals => (metric_value - condition.threshold).abs() >= 0.01,
            }
        } else {
            false
        }
    }

    fn initialize_rule_base() -> Vec<OptimizationRule> {
        vec![
            OptimizationRule {
                condition: RuleCondition {
                    metric_type: "latency_p95".to_string(),
                    operator: ComparisonOperator::GreaterThan,
                    threshold: 1000.0,
                },
                recommendation: Recommendation {
                    title: "High Latency Alert".to_string(),
                    description: "P95 latency exceeds 1000ms, consider optimization".to_string(),
                    category: RecommendationCategory::QueryOptimization,
                    expected_improvement: "Reduce latency by 30-50%".to_string(),
                    implementation_effort: ImplementationEffort::Medium,
                    estimated_impact_score: 0.8,
                },
                priority: 0.9,
            },
            OptimizationRule {
                condition: RuleCondition {
                    metric_type: "error_rate".to_string(),
                    operator: ComparisonOperator::GreaterThan,
                    threshold: 0.05,
                },
                recommendation: Recommendation {
                    title: "High Error Rate".to_string(),
                    description: "Error rate exceeds 5%, investigate service health".to_string(),
                    category: RecommendationCategory::Configuration,
                    expected_improvement: "Reduce error rate to < 1%".to_string(),
                    implementation_effort: ImplementationEffort::High,
                    estimated_impact_score: 0.9,
                },
                priority: 0.95,
            },
            // Add more rules as needed
        ]
    }
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub severity: AlertSeverity,
    pub title: String,
    pub description: String,
    pub timestamp: SystemTime,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

impl Default for PerformanceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_performance_analyzer_creation() {
        let analyzer = PerformanceAnalyzer::new();
        assert!(analyzer.config.enable_real_time_monitoring);
    }

    #[tokio::test]
    async fn test_metrics_recording() {
        let analyzer = PerformanceAnalyzer::new();

        let system_metrics = SystemPerformanceMetrics {
            timestamp: SystemTime::now(),
            overall_latency_p50: Duration::from_millis(100),
            overall_latency_p95: Duration::from_millis(200),
            overall_latency_p99: Duration::from_millis(500),
            throughput_qps: 100.0,
            error_rate: 0.01,
            timeout_rate: 0.001,
            cache_hit_rate: 0.85,
            memory_usage_mb: 512.0,
            cpu_usage_percent: 65.0,
            network_bandwidth_mbps: 100.0,
            active_connections: 50,
            queue_depth: 10,
        };

        let result = analyzer.record_system_metrics(system_metrics).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_trend_analysis() {
        let analyzer = PerformanceAnalyzer::new();

        // Record multiple metrics to enable trend analysis
        for i in 0..15 {
            let metrics = SystemPerformanceMetrics {
                timestamp: SystemTime::now(),
                overall_latency_p50: Duration::from_millis(100 + i * 10),
                overall_latency_p95: Duration::from_millis(200 + i * 20),
                overall_latency_p99: Duration::from_millis(500 + i * 50),
                throughput_qps: 100.0 - i as f64,
                error_rate: 0.01,
                timeout_rate: 0.001,
                cache_hit_rate: 0.85,
                memory_usage_mb: 512.0,
                cpu_usage_percent: 65.0,
                network_bandwidth_mbps: 100.0,
                active_connections: 50,
                queue_depth: 10,
            };

            analyzer.record_system_metrics(metrics).await.unwrap();
        }

        let trends = analyzer.analyze_trends().await.unwrap();
        assert!(!trends.is_empty());
    }
}

/// Intelligent Query Plan Optimizer using ML-driven predictions
pub struct QueryPlanOptimizer {
    config: QueryOptimizerConfig,
    historical_plans: Arc<RwLock<VecDeque<QueryPlanExecution>>>,
    plan_model: Arc<RwLock<QueryPlanModel>>,
    optimization_stats: Arc<QueryOptimizationStats>,
    plan_cache: Arc<RwLock<HashMap<String, OptimalPlan>>>,
}

/// Configuration for query plan optimization
#[derive(Debug, Clone)]
pub struct QueryOptimizerConfig {
    pub enable_ml_optimization: bool,
    pub max_historical_plans: usize,
    pub retraining_interval: usize,
    pub plan_cache_size: usize,
    pub enable_adaptive_timeout: bool,
    pub enable_cost_estimation: bool,
    pub enable_parallel_execution: bool,
    pub confidence_threshold: f64,
}

impl Default for QueryOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_ml_optimization: true,
            max_historical_plans: 10000,
            retraining_interval: 100,
            plan_cache_size: 1000,
            enable_adaptive_timeout: true,
            enable_cost_estimation: true,
            enable_parallel_execution: true,
            confidence_threshold: 0.75,
        }
    }
}

/// Historical execution data for a query plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlanExecution {
    pub query_hash: String,
    pub plan_type: PlanType,
    pub services_involved: Vec<String>,
    pub execution_time: Duration,
    pub result_count: usize,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub network_io_mb: f64,
    pub cache_hit_rate: f64,
    pub timestamp: SystemTime,
    pub query_complexity: QueryComplexity,
    pub success: bool,
}

/// Types of query execution plans
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanType {
    Sequential,
    Parallel,
    Hybrid,
    CacheFirst,
    BindJoin,
    HashJoin,
    NestedLoop,
}

/// Query complexity assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryComplexity {
    pub triple_patterns: usize,
    pub join_count: usize,
    pub optional_patterns: usize,
    pub filter_count: usize,
    pub union_count: usize,
    pub service_count: usize,
    pub complexity_score: f64,
}

/// ML model for query plan prediction
#[derive(Debug, Clone)]
pub struct QueryPlanModel {
    /// Feature weights for execution time prediction
    execution_time_weights: Vec<f64>,
    /// Feature weights for memory usage prediction
    memory_usage_weights: Vec<f64>,
    /// Feature weights for success rate prediction
    success_rate_weights: Vec<f64>,
    /// Model accuracy metrics
    model_accuracy: ModelAccuracy,
    /// Number of training samples
    training_samples: usize,
}

/// Model accuracy tracking
#[derive(Debug, Clone)]
pub struct ModelAccuracy {
    pub execution_time_r_squared: f64,
    pub memory_usage_r_squared: f64,
    pub success_rate_accuracy: f64,
    pub last_training: SystemTime,
}

impl Default for QueryPlanModel {
    fn default() -> Self {
        Self {
            execution_time_weights: vec![0.0; 15], // 15 features
            memory_usage_weights: vec![0.0; 15],
            success_rate_weights: vec![0.0; 15],
            model_accuracy: ModelAccuracy {
                execution_time_r_squared: 0.0,
                memory_usage_r_squared: 0.0,
                success_rate_accuracy: 0.0,
                last_training: SystemTime::now(),
            },
            training_samples: 0,
        }
    }
}

/// Optimal plan recommendation
#[derive(Debug, Clone)]
pub struct OptimalPlan {
    pub plan_type: PlanType,
    pub predicted_execution_time: Duration,
    pub predicted_memory_usage: f64,
    pub predicted_success_rate: f64,
    pub confidence: f64,
    pub service_execution_order: Vec<String>,
    pub parallel_groups: Vec<Vec<String>>,
    pub timeout_recommendation: Duration,
    pub cache_strategy: CacheStrategy,
}

/// Cache strategy recommendation
#[derive(Debug, Clone)]
pub enum CacheStrategy {
    NoCache,
    ResultCache,
    IntermediateCache,
    AggressiveCache,
}

/// Statistics for query optimization
#[derive(Debug, Default)]
pub struct QueryOptimizationStats {
    pub total_optimizations: AtomicU64,
    pub successful_predictions: AtomicU64,
    pub model_retrainings: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub average_improvement: Arc<RwLock<f64>>,
    pub total_time_saved: Arc<RwLock<Duration>>,
}

impl QueryPlanOptimizer {
    /// Create a new query plan optimizer
    pub fn new(config: QueryOptimizerConfig) -> Self {
        Self {
            config,
            historical_plans: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            plan_model: Arc::new(RwLock::new(QueryPlanModel::default())),
            optimization_stats: Arc::new(QueryOptimizationStats::default()),
            plan_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record execution of a query plan
    pub async fn record_execution(&self, execution: QueryPlanExecution) -> Result<()> {
        let mut historical_plans = self.historical_plans.write().await;

        // Keep only recent executions
        if historical_plans.len() >= self.config.max_historical_plans {
            historical_plans.pop_front();
        }

        historical_plans.push_back(execution);

        // Retrain model if enough new data
        if historical_plans.len() % self.config.retraining_interval == 0 {
            drop(historical_plans);
            self.retrain_model().await?;
        }

        Ok(())
    }

    /// Get optimal execution plan for a query
    pub async fn optimize_query_plan(
        &self,
        query_hash: &str,
        complexity: &QueryComplexity,
        available_services: &[String],
    ) -> Result<OptimalPlan> {
        // Check cache first
        if let Some(cached_plan) = self.plan_cache.read().await.get(query_hash) {
            self.optimization_stats
                .cache_hits
                .fetch_add(1, Ordering::Relaxed);
            return Ok(cached_plan.clone());
        }

        self.optimization_stats
            .cache_misses
            .fetch_add(1, Ordering::Relaxed);

        let model = self.plan_model.read().await;

        if model.training_samples < 10 {
            // Not enough training data, use heuristics
            return self
                .generate_heuristic_plan(complexity, available_services)
                .await;
        }

        // Generate candidate plans
        let candidate_plans = self
            .generate_candidate_plans(complexity, available_services)
            .await;

        let mut best_plan = None;
        let mut best_score = f64::MIN;

        for plan_type in candidate_plans {
            let features = self.extract_plan_features(complexity, &plan_type, available_services);

            // Predict performance for this plan
            let predicted_time = self.predict_execution_time(&features, &model);
            let predicted_memory = self.predict_memory_usage(&features, &model);
            let predicted_success = self.predict_success_rate(&features, &model);

            // Calculate composite score (lower is better for time/memory, higher for success)
            let score = predicted_success / (1.0 + predicted_time + predicted_memory * 0.01);

            if score > best_score {
                best_score = score;
                best_plan = Some(OptimalPlan {
                    plan_type: plan_type.clone(),
                    predicted_execution_time: Duration::from_millis(predicted_time as u64),
                    predicted_memory_usage: predicted_memory,
                    predicted_success_rate: predicted_success,
                    confidence: model.model_accuracy.execution_time_r_squared,
                    service_execution_order: self
                        .optimize_service_order(available_services, &plan_type),
                    parallel_groups: self.determine_parallel_groups(available_services, &plan_type),
                    timeout_recommendation: self.calculate_adaptive_timeout(predicted_time),
                    cache_strategy: self.recommend_cache_strategy(complexity, predicted_time),
                });
            }
        }

        let optimal_plan = best_plan.ok_or_else(|| anyhow!("Failed to generate optimal plan"))?;

        // Cache the result
        self.plan_cache
            .write()
            .await
            .insert(query_hash.to_string(), optimal_plan.clone());

        self.optimization_stats
            .total_optimizations
            .fetch_add(1, Ordering::Relaxed);

        Ok(optimal_plan)
    }

    /// Retrain the ML model with historical data
    async fn retrain_model(&self) -> Result<()> {
        let historical_plans = self.historical_plans.read().await;

        if historical_plans.len() < 10 {
            return Ok(());
        }

        let mut model = self.plan_model.write().await;

        // Prepare training data
        let mut features_matrix = Vec::new();
        let mut execution_time_targets = Vec::new();
        let mut memory_usage_targets = Vec::new();
        let mut success_targets = Vec::new();

        for plan in historical_plans.iter() {
            let features = self.extract_execution_features(plan);
            features_matrix.push(features);
            execution_time_targets.push(plan.execution_time.as_millis() as f64);
            memory_usage_targets.push(plan.memory_usage_mb);
            success_targets.push(if plan.success { 1.0 } else { 0.0 });
        }

        // Train models using simple linear regression
        model.execution_time_weights =
            self.train_linear_regression(&features_matrix, &execution_time_targets);
        model.memory_usage_weights =
            self.train_linear_regression(&features_matrix, &memory_usage_targets);
        model.success_rate_weights =
            self.train_linear_regression(&features_matrix, &success_targets);

        // Calculate accuracy metrics
        model.model_accuracy.execution_time_r_squared = self.calculate_r_squared(
            &features_matrix,
            &execution_time_targets,
            &model.execution_time_weights,
        );
        model.model_accuracy.memory_usage_r_squared = self.calculate_r_squared(
            &features_matrix,
            &memory_usage_targets,
            &model.memory_usage_weights,
        );
        model.model_accuracy.success_rate_accuracy = self.calculate_classification_accuracy(
            &features_matrix,
            &success_targets,
            &model.success_rate_weights,
        );

        model.model_accuracy.last_training = SystemTime::now();
        model.training_samples = historical_plans.len();

        self.optimization_stats
            .model_retrainings
            .fetch_add(1, Ordering::Relaxed);

        info!(
            "Retrained query plan model - Samples: {}, Time R²: {:.3}, Memory R²: {:.3}, Success Acc: {:.3}",
            model.training_samples,
            model.model_accuracy.execution_time_r_squared,
            model.model_accuracy.memory_usage_r_squared,
            model.model_accuracy.success_rate_accuracy
        );

        Ok(())
    }

    /// Generate heuristic plan when ML model is not ready
    async fn generate_heuristic_plan(
        &self,
        complexity: &QueryComplexity,
        available_services: &[String],
    ) -> Result<OptimalPlan> {
        let plan_type = if complexity.service_count > 3 && self.config.enable_parallel_execution {
            PlanType::Parallel
        } else if complexity.join_count > 5 {
            PlanType::HashJoin
        } else {
            PlanType::Sequential
        };

        Ok(OptimalPlan {
            plan_type,
            predicted_execution_time: Duration::from_millis(
                (complexity.complexity_score * 100.0) as u64,
            ),
            predicted_memory_usage: complexity.triple_patterns as f64 * 10.0,
            predicted_success_rate: 0.9,
            confidence: 0.5, // Low confidence for heuristics
            service_execution_order: available_services.to_vec(),
            parallel_groups: vec![available_services.to_vec()],
            timeout_recommendation: Duration::from_secs(30),
            cache_strategy: CacheStrategy::ResultCache,
        })
    }

    /// Generate candidate plan types to evaluate
    async fn generate_candidate_plans(
        &self,
        complexity: &QueryComplexity,
        _services: &[String],
    ) -> Vec<PlanType> {
        let mut candidates = vec![PlanType::Sequential];

        if complexity.service_count > 1 && self.config.enable_parallel_execution {
            candidates.push(PlanType::Parallel);
            candidates.push(PlanType::Hybrid);
        }

        if complexity.join_count > 2 {
            candidates.push(PlanType::HashJoin);
            candidates.push(PlanType::BindJoin);
        }

        if complexity.complexity_score > 0.7 {
            candidates.push(PlanType::CacheFirst);
        }

        candidates
    }

    /// Extract features for plan prediction
    fn extract_plan_features(
        &self,
        complexity: &QueryComplexity,
        plan_type: &PlanType,
        services: &[String],
    ) -> Vec<f64> {
        vec![
            complexity.triple_patterns as f64,
            complexity.join_count as f64,
            complexity.optional_patterns as f64,
            complexity.filter_count as f64,
            complexity.union_count as f64,
            complexity.service_count as f64,
            complexity.complexity_score,
            services.len() as f64,
            match plan_type {
                PlanType::Sequential => 1.0,
                PlanType::Parallel => 2.0,
                PlanType::Hybrid => 3.0,
                PlanType::CacheFirst => 4.0,
                PlanType::BindJoin => 5.0,
                PlanType::HashJoin => 6.0,
                PlanType::NestedLoop => 7.0,
            },
            // Additional derived features
            complexity.join_count as f64 / complexity.triple_patterns.max(1) as f64,
            complexity.filter_count as f64 / complexity.triple_patterns.max(1) as f64,
            if complexity.optional_patterns > 0 {
                1.0
            } else {
                0.0
            },
            if complexity.union_count > 0 { 1.0 } else { 0.0 },
            (complexity.service_count as f64).ln(),
            complexity.complexity_score.powi(2),
        ]
    }

    /// Extract features from historical execution
    fn extract_execution_features(&self, execution: &QueryPlanExecution) -> Vec<f64> {
        self.extract_plan_features(
            &execution.query_complexity,
            &execution.plan_type,
            &execution.services_involved,
        )
    }

    /// Predict execution time using the model
    fn predict_execution_time(&self, features: &[f64], model: &QueryPlanModel) -> f64 {
        self.linear_prediction(features, &model.execution_time_weights)
            .max(0.0)
    }

    /// Predict memory usage using the model
    fn predict_memory_usage(&self, features: &[f64], model: &QueryPlanModel) -> f64 {
        self.linear_prediction(features, &model.memory_usage_weights)
            .max(0.0)
    }

    /// Predict success rate using the model
    fn predict_success_rate(&self, features: &[f64], model: &QueryPlanModel) -> f64 {
        self.linear_prediction(features, &model.success_rate_weights)
            .clamp(0.0, 1.0)
    }

    /// Make linear prediction
    fn linear_prediction(&self, features: &[f64], weights: &[f64]) -> f64 {
        features
            .iter()
            .zip(weights.iter())
            .map(|(f, w)| f * w)
            .sum()
    }

    /// Train linear regression model
    fn train_linear_regression(&self, features: &[Vec<f64>], targets: &[f64]) -> Vec<f64> {
        if features.is_empty() || targets.is_empty() {
            return vec![0.0; 15];
        }

        let n = features.len();
        let feature_count = features[0].len();
        let mut weights = vec![0.0; feature_count];

        // Simple least squares implementation
        #[allow(clippy::needless_range_loop)]
        for i in 0..feature_count {
            let mut sum_xy = 0.0;
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_x2 = 0.0;

            for j in 0..n {
                let x = features[j][i];
                let y = targets[j];
                sum_xy += x * y;
                sum_x += x;
                sum_y += y;
                sum_x2 += x * x;
            }

            let denominator = n as f64 * sum_x2 - sum_x * sum_x;
            if denominator.abs() > 1e-10 {
                weights[i] = (n as f64 * sum_xy - sum_x * sum_y) / denominator;
            }
        }

        weights
    }

    /// Calculate R-squared for regression models
    fn calculate_r_squared(&self, features: &[Vec<f64>], targets: &[f64], weights: &[f64]) -> f64 {
        if features.is_empty() || targets.is_empty() {
            return 0.0;
        }

        let mean_target: f64 = targets.iter().sum::<f64>() / targets.len() as f64;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for (i, target) in targets.iter().enumerate() {
            let predicted = self.linear_prediction(&features[i], weights);
            ss_res += (target - predicted).powi(2);
            ss_tot += (target - mean_target).powi(2);
        }

        if ss_tot > 1e-10 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        }
    }

    /// Calculate classification accuracy
    fn calculate_classification_accuracy(
        &self,
        features: &[Vec<f64>],
        targets: &[f64],
        weights: &[f64],
    ) -> f64 {
        if features.is_empty() || targets.is_empty() {
            return 0.0;
        }

        let mut correct = 0;
        for (i, target) in targets.iter().enumerate() {
            let predicted = self.linear_prediction(&features[i], weights);
            let predicted_class = if predicted > 0.5 { 1.0 } else { 0.0 };
            if (predicted_class - target).abs() < 0.1 {
                correct += 1;
            }
        }

        correct as f64 / targets.len() as f64
    }

    /// Optimize service execution order
    fn optimize_service_order(&self, services: &[String], plan_type: &PlanType) -> Vec<String> {
        let mut ordered = services.to_vec();

        match plan_type {
            PlanType::Sequential | PlanType::BindJoin => {
                // Keep original order for sequential execution
            }
            PlanType::Parallel | PlanType::HashJoin => {
                // Randomize for parallel execution to balance load
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                ordered.sort_by_key(|s| {
                    let mut hasher = DefaultHasher::new();
                    s.hash(&mut hasher);
                    hasher.finish()
                });
            }
            _ => {
                // Default ordering
            }
        }

        ordered
    }

    /// Determine parallel execution groups
    fn determine_parallel_groups(
        &self,
        services: &[String],
        plan_type: &PlanType,
    ) -> Vec<Vec<String>> {
        match plan_type {
            PlanType::Parallel => {
                // Split into groups of 2-3 services
                services.chunks(3).map(|chunk| chunk.to_vec()).collect()
            }
            PlanType::Hybrid => {
                // First service alone, rest in parallel
                if services.len() > 1 {
                    vec![vec![services[0].clone()], services[1..].to_vec()]
                } else {
                    vec![services.to_vec()]
                }
            }
            _ => {
                // Sequential execution
                services.iter().map(|s| vec![s.clone()]).collect()
            }
        }
    }

    /// Calculate adaptive timeout based on predicted execution time
    fn calculate_adaptive_timeout(&self, predicted_time: f64) -> Duration {
        // Add buffer based on prediction confidence
        let buffer_factor = 2.5;
        let timeout_ms = (predicted_time * buffer_factor).clamp(1000.0, 300000.0); // 1s to 5min
        Duration::from_millis(timeout_ms as u64)
    }

    /// Recommend cache strategy based on query characteristics
    fn recommend_cache_strategy(
        &self,
        complexity: &QueryComplexity,
        predicted_time: f64,
    ) -> CacheStrategy {
        if predicted_time > 10000.0 {
            // > 10 seconds
            CacheStrategy::AggressiveCache
        } else if complexity.service_count > 3 {
            CacheStrategy::IntermediateCache
        } else if predicted_time > 1000.0 {
            // > 1 second
            CacheStrategy::ResultCache
        } else {
            CacheStrategy::NoCache
        }
    }

    /// Get optimization statistics
    pub async fn get_optimization_stats(&self) -> QueryOptimizationStats {
        QueryOptimizationStats {
            total_optimizations: AtomicU64::new(
                self.optimization_stats
                    .total_optimizations
                    .load(Ordering::Relaxed),
            ),
            successful_predictions: AtomicU64::new(
                self.optimization_stats
                    .successful_predictions
                    .load(Ordering::Relaxed),
            ),
            model_retrainings: AtomicU64::new(
                self.optimization_stats
                    .model_retrainings
                    .load(Ordering::Relaxed),
            ),
            cache_hits: AtomicU64::new(self.optimization_stats.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(
                self.optimization_stats.cache_misses.load(Ordering::Relaxed),
            ),
            average_improvement: Arc::new(RwLock::new(
                *self.optimization_stats.average_improvement.read().await,
            )),
            total_time_saved: Arc::new(RwLock::new(
                *self.optimization_stats.total_time_saved.read().await,
            )),
        }
    }

    /// Clear plan cache
    pub async fn clear_cache(&self) {
        self.plan_cache.write().await.clear();
    }
}

#[cfg(test)]
mod query_optimizer_tests {
    use super::*;

    #[tokio::test]
    async fn test_query_optimizer_creation() {
        let config = QueryOptimizerConfig::default();
        let optimizer = QueryPlanOptimizer::new(config);

        let stats = optimizer.get_optimization_stats().await;
        assert_eq!(stats.total_optimizations.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_heuristic_plan_generation() {
        let config = QueryOptimizerConfig::default();
        let optimizer = QueryPlanOptimizer::new(config);

        let complexity = QueryComplexity {
            triple_patterns: 5,
            join_count: 3,
            optional_patterns: 1,
            filter_count: 2,
            union_count: 0,
            service_count: 2,
            complexity_score: 0.7,
        };

        let services = vec!["service1".to_string(), "service2".to_string()];
        let plan = optimizer
            .optimize_query_plan("test_query", &complexity, &services)
            .await
            .unwrap();

        assert!(plan.confidence > 0.0);
        assert!(plan.predicted_execution_time > Duration::from_millis(0));
    }

    #[tokio::test]
    async fn test_execution_recording() {
        let config = QueryOptimizerConfig::default();
        let optimizer = QueryPlanOptimizer::new(config);

        let execution = QueryPlanExecution {
            query_hash: "test_hash".to_string(),
            plan_type: PlanType::Sequential,
            services_involved: vec!["service1".to_string()],
            execution_time: Duration::from_millis(1000),
            result_count: 100,
            memory_usage_mb: 50.0,
            cpu_usage_percent: 25.0,
            network_io_mb: 10.0,
            cache_hit_rate: 0.8,
            timestamp: SystemTime::now(),
            query_complexity: QueryComplexity {
                triple_patterns: 3,
                join_count: 1,
                optional_patterns: 0,
                filter_count: 1,
                union_count: 0,
                service_count: 1,
                complexity_score: 0.3,
            },
            success: true,
        };

        let result = optimizer.record_execution(execution).await;
        assert!(result.is_ok());
    }
}
