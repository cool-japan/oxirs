//! # Comprehensive Performance Analysis Engine
//!
//! This module provides advanced performance analysis, bottleneck identification,
//! and optimization recommendations for the federated query system.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::{FederatedService, ServiceRegistry};

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
    pub timestamp: Instant,
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
    pub timestamp: Instant,
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
    pub timestamp: Instant,
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
    pub severity_score: f64, // 0.0 - 1.0
    pub confidence_level: f64, // 0.0 - 1.0
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
    pub rate_of_change: f64, // percentage per hour
    pub confidence: f64,     // 0.0 - 1.0
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
    pub latency_degradation_threshold: f64,   // percentage increase
    pub throughput_degradation_threshold: f64, // percentage decrease
    pub error_rate_threshold: f64,             // error rate threshold
    pub memory_usage_threshold: f64,           // percentage of total memory
    pub cpu_usage_threshold: f64,              // percentage
    pub cache_hit_rate_threshold: f64,         // minimum cache hit rate
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
                latency_degradation_threshold: 50.0, // 50% increase
                throughput_degradation_threshold: 20.0, // 20% decrease
                error_rate_threshold: 0.05,           // 5%
                memory_usage_threshold: 0.85,         // 85%
                cpu_usage_threshold: 0.90,            // 90%
                cache_hit_rate_threshold: 0.80,       // 80%
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
        
        let service_history = history.service_metrics
            .entry(metrics.service_id.clone())
            .or_insert_with(VecDeque::new);
        
        service_history.push_back(metrics.clone());
        
        // Maintain size limit
        while service_history.len() > history.max_entries {
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
        let baseline = self.bottleneck_detector.baseline_metrics.as_ref()
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
            let primary_factor = analysis.contributing_factors
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
                    p if p >= 0.8 => recommendations.high_priority.push(rule.recommendation.clone()),
                    p if p >= 0.6 => recommendations.medium_priority.push(rule.recommendation.clone()),
                    p if p >= 0.4 => recommendations.low_priority.push(rule.recommendation.clone()),
                    _ => recommendations.long_term.push(rule.recommendation.clone()),
                }
            }
        }

        // Add bottleneck-specific recommendations
        let bottleneck_recommendations = self.generate_bottleneck_specific_recommendations(&analysis);
        recommendations.high_priority.extend(bottleneck_recommendations);

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
        let latency_values: Vec<f64> = history.system_metrics
            .iter()
            .map(|m| m.overall_latency_p95.as_millis() as f64)
            .collect();
        
        trends.push(self.calculate_trend("latency_p95", &latency_values));

        // Analyze throughput trend
        let throughput_values: Vec<f64> = history.system_metrics
            .iter()
            .map(|m| m.throughput_qps)
            .collect();
        
        trends.push(self.calculate_trend("throughput_qps", &throughput_values));

        // Analyze error rate trend
        let error_rate_values: Vec<f64> = history.system_metrics
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
            if recent_metrics.overall_latency_p95.as_millis() > self.alert_thresholds.critical_latency_ms {
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

        if latency_increase > self.bottleneck_detector.detection_thresholds.latency_degradation_threshold / 100.0 {
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
        if current.cpu_usage_percent > self.bottleneck_detector.detection_thresholds.cpu_usage_threshold {
            analysis.contributing_factors.push(BottleneckFactor {
                factor_type: FactorType::ResourceUtilization,
                description: "High CPU utilization detected".to_string(),
                weight: current.cpu_usage_percent,
                metric_value: current.cpu_usage_percent,
                threshold: self.bottleneck_detector.detection_thresholds.cpu_usage_threshold,
            });
        }

        // Check memory usage
        if current.memory_usage_mb > self.bottleneck_detector.detection_thresholds.memory_usage_threshold {
            analysis.contributing_factors.push(BottleneckFactor {
                factor_type: FactorType::ResourceUtilization,
                description: "High memory utilization detected".to_string(),
                weight: (current.memory_usage_mb / 1024.0) * 100.0, // Convert to percentage
                metric_value: current.memory_usage_mb,
                threshold: self.bottleneck_detector.detection_thresholds.memory_usage_threshold,
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
                        description: format!("Service {} has high response times", service_id),
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
        if current.cache_hit_rate < self.bottleneck_detector.detection_thresholds.cache_hit_rate_threshold {
            analysis.contributing_factors.push(BottleneckFactor {
                factor_type: FactorType::CachePerformance,
                description: "Low cache hit rate affecting performance".to_string(),
                weight: (1.0 - current.cache_hit_rate) * 100.0,
                metric_value: current.cache_hit_rate,
                threshold: self.bottleneck_detector.detection_thresholds.cache_hit_rate_threshold,
            });
        }
    }

    fn calculate_confidence_level(&self, analysis: &BottleneckAnalysis) -> f64 {
        let factor_count = analysis.contributing_factors.len() as f64;
        let max_weight = analysis.contributing_factors
            .iter()
            .map(|f| f.weight)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        // Higher confidence with more factors and higher weights
        ((factor_count * 0.2) + (max_weight / 200.0)).min(1.0)
    }

    async fn generate_bottleneck_recommendations(&self, analysis: &BottleneckAnalysis) -> Vec<String> {
        let mut recommendations = Vec::new();

        match analysis.primary_bottleneck {
            BottleneckType::NetworkLatency => {
                recommendations.push("Enable request batching to reduce network round trips".to_string());
                recommendations.push("Implement more aggressive caching strategies".to_string());
                recommendations.push("Consider query optimization to reduce data transfer".to_string());
            }
            BottleneckType::ServiceResponseTime => {
                recommendations.push("Analyze slow services and optimize their queries".to_string());
                recommendations.push("Implement service-level caching".to_string());
                recommendations.push("Consider load balancing across service replicas".to_string());
            }
            BottleneckType::MemoryPressure => {
                recommendations.push("Implement result streaming for large queries".to_string());
                recommendations.push("Reduce batch sizes to decrease memory usage".to_string());
                recommendations.push("Enable memory-efficient query execution strategies".to_string());
            }
            BottleneckType::CPUUtilization => {
                recommendations.push("Scale CPU resources horizontally or vertically".to_string());
                recommendations.push("Optimize query execution algorithms".to_string());
                recommendations.push("Implement query complexity limits".to_string());
            }
            _ => {
                recommendations.push("Monitor system metrics more closely to identify bottlenecks".to_string());
            }
        }

        recommendations
    }

    fn generate_bottleneck_specific_recommendations(&self, analysis: &BottleneckAnalysis) -> Vec<Recommendation> {
        let mut recommendations = Vec::new();

        for factor in &analysis.contributing_factors {
            match factor.factor_type {
                FactorType::Latency => {
                    recommendations.push(Recommendation {
                        title: "Optimize Network Performance".to_string(),
                        description: "Implement compression and request batching to reduce network overhead".to_string(),
                        category: RecommendationCategory::NetworkOptimization,
                        expected_improvement: "20-40% reduction in response times".to_string(),
                        implementation_effort: ImplementationEffort::Medium,
                        estimated_impact_score: 0.7,
                    });
                }
                FactorType::ResourceUtilization => {
                    recommendations.push(Recommendation {
                        title: "Scale System Resources".to_string(),
                        description: "Increase CPU and memory allocation to handle current load".to_string(),
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
                        expected_improvement: "30-50% reduction in backend service load".to_string(),
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
            rate_of_change: slope * 100.0, // Convert to percentage
            confidence: (n / 20.0).min(1.0), // Higher confidence with more data points
            prediction_accuracy: 0.8, // Simplified prediction accuracy
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
    pub timestamp: Instant,
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
            timestamp: Instant::now(),
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
                timestamp: Instant::now(),
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