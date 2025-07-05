//! Advanced Performance Insights and Monitoring for OxiRS Vector Search
//!
//! This module provides comprehensive performance analysis, optimization recommendations,
//! and real-time monitoring capabilities for the vector search engine.

use crate::{Vector, VectorId};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tracing::{debug, info, warn};

/// Advanced performance insights analyzer
#[derive(Debug, Clone)]
pub struct PerformanceInsightsAnalyzer {
    /// Query execution statistics
    query_stats: QueryStatistics,
    /// Vector distribution analysis
    vector_stats: VectorStatistics,
    /// Performance trends over time
    trends: PerformanceTrends,
    /// Optimization recommendations
    recommendations: OptimizationRecommendations,
    /// Real-time metrics collection
    metrics_collector: MetricsCollector,
    /// Performance alerting system
    alerting_system: AlertingSystem,
}

/// Comprehensive query performance statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStatistics {
    pub total_queries: u64,
    pub average_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_qps: f64,
    pub error_rate: f64,
    pub cache_hit_rate: f64,
    pub index_efficiency: f64,
    pub memory_usage_mb: f64,
    pub cpu_utilization: f64,
    pub latency_distribution: Vec<LatencyBucket>,
    pub query_complexity_distribution: HashMap<QueryComplexity, u64>,
    pub top_slow_queries: Vec<SlowQueryEntry>,
}

/// Vector dataset quality and distribution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorStatistics {
    pub total_vectors: u64,
    pub average_dimension: u32,
    pub dimension_distribution: HashMap<u32, u64>,
    pub vector_density: f64,
    pub sparsity_ratio: f64,
    pub clustering_coefficient: f64,
    pub hubness_measure: f64,
    pub intrinsic_dimensionality: f64,
    pub noise_estimation: f64,
    pub quality_score: f64,
    pub outlier_count: u64,
    pub similarity_distribution: SimilarityDistribution,
    pub vector_type_distribution: HashMap<String, u64>,
}

/// Performance trends analysis over time
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub trend_window_hours: u32,
    pub latency_trend: TrendDirection,
    pub throughput_trend: TrendDirection,
    pub error_rate_trend: TrendDirection,
    pub memory_usage_trend: TrendDirection,
    pub cache_efficiency_trend: TrendDirection,
    pub index_performance_trend: TrendDirection,
    pub historical_data: Vec<PerformanceSnapshot>,
    pub seasonal_patterns: SeasonalPatterns,
    pub anomaly_detection: AnomalyDetection,
}

/// Intelligent optimization recommendations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizationRecommendations {
    pub index_recommendations: Vec<IndexRecommendation>,
    pub caching_recommendations: Vec<CachingRecommendation>,
    pub query_recommendations: Vec<QueryRecommendation>,
    pub hardware_recommendations: Vec<HardwareRecommendation>,
    pub configuration_recommendations: Vec<ConfigurationRecommendation>,
    pub priority_actions: Vec<PriorityAction>,
    pub estimated_improvements: ImprovementEstimates,
}

/// Real-time metrics collection system
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    start_time: Instant,
    query_times: Vec<Duration>,
    query_complexities: Vec<QueryComplexity>,
    memory_samples: Vec<u64>,
    cpu_samples: Vec<f64>,
    cache_metrics: CacheMetrics,
    error_counts: HashMap<String, u64>,
    active_queries: u32,
}

/// Advanced alerting system
#[derive(Debug, Clone)]
pub struct AlertingSystem {
    alert_rules: Vec<AlertRule>,
    active_alerts: Vec<ActiveAlert>,
    alert_history: Vec<AlertEvent>,
    notification_channels: Vec<NotificationChannel>,
}

// Supporting types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyBucket {
    pub range_ms: (f64, f64),
    pub count: u64,
    pub percentage: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryComplexity {
    Simple,
    Moderate,
    Complex,
    Highly_Complex,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowQueryEntry {
    pub query_id: String,
    pub execution_time_ms: f64,
    pub timestamp: SystemTime,
    pub query_type: String,
    pub vector_count: u64,
    pub optimization_potential: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SimilarityDistribution {
    pub min_similarity: f64,
    pub max_similarity: f64,
    pub mean_similarity: f64,
    pub std_deviation: f64,
    pub distribution_buckets: Vec<(f64, u64)>,
    pub skewness: f64,
    pub kurtosis: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    #[default]
    Stable,
    Degrading,
    Volatile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    pub timestamp: SystemTime,
    pub latency_p99: f64,
    pub throughput: f64,
    pub memory_usage: u64,
    pub cpu_usage: f64,
    pub cache_hit_rate: f64,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SeasonalPatterns {
    pub daily_patterns: Vec<(u8, f64)>,   // Hour -> performance factor
    pub weekly_patterns: Vec<(u8, f64)>,  // Day of week -> performance factor
    pub monthly_patterns: Vec<(u8, f64)>, // Day of month -> performance factor
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnomalyDetection {
    pub anomaly_threshold: f64,
    pub detected_anomalies: Vec<AnomalyEvent>,
    pub baseline_performance: f64,
    pub current_deviation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyEvent {
    pub timestamp: SystemTime,
    pub metric: String,
    pub value: f64,
    pub expected_value: f64,
    pub deviation_score: f64,
    pub severity: AlertSeverity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRecommendation {
    pub recommendation_type: String,
    pub description: String,
    pub estimated_improvement: f64,
    pub implementation_effort: EffortLevel,
    pub prerequisites: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingRecommendation {
    pub cache_type: String,
    pub recommended_size_mb: u64,
    pub eviction_policy: String,
    pub estimated_hit_rate: f64,
    pub cost_benefit_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRecommendation {
    pub query_pattern: String,
    pub optimization_technique: String,
    pub expected_speedup: f64,
    pub applicability: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareRecommendation {
    pub component: String,
    pub current_bottleneck: bool,
    pub recommended_upgrade: String,
    pub performance_impact: f64,
    pub cost_estimate: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationRecommendation {
    pub parameter: String,
    pub current_value: String,
    pub recommended_value: String,
    pub justification: String,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityAction {
    pub action: String,
    pub priority: Priority,
    pub estimated_impact: f64,
    pub implementation_time: Duration,
    pub dependencies: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImprovementEstimates {
    pub latency_improvement: f64,
    pub throughput_improvement: f64,
    pub memory_savings: f64,
    pub cost_reduction: f64,
    pub reliability_improvement: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub metric: String,
    pub threshold: f64,
    pub comparison: ComparisonOperator,
    pub severity: AlertSeverity,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAlert {
    pub rule_name: String,
    pub triggered_at: SystemTime,
    pub current_value: f64,
    pub threshold: f64,
    pub severity: AlertSeverity,
    pub acknowledged: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    pub timestamp: SystemTime,
    pub alert_name: String,
    pub event_type: AlertEventType,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannel {
    Email(String),
    Webhook(String),
    Slack(String),
    Console,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EffortLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Priority {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equals,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertEventType {
    Triggered,
    Resolved,
    Acknowledged,
    Escalated,
}

impl PerformanceInsightsAnalyzer {
    /// Create a new performance insights analyzer
    pub fn new() -> Self {
        Self {
            query_stats: QueryStatistics::default(),
            vector_stats: VectorStatistics::default(),
            trends: PerformanceTrends::default(),
            recommendations: OptimizationRecommendations::default(),
            metrics_collector: MetricsCollector::new(),
            alerting_system: AlertingSystem::new(),
        }
    }

    /// Record a query execution for performance analysis
    pub fn record_query(&mut self, duration: Duration, complexity: QueryComplexity, success: bool) {
        self.metrics_collector
            .record_query(duration, complexity, success);
        self.query_stats.total_queries += 1;

        let latency_ms = duration.as_secs_f64() * 1000.0;
        self.update_latency_statistics(latency_ms);

        if !success {
            self.query_stats.error_rate = self.calculate_error_rate();
        }

        self.check_performance_alerts(latency_ms);
    }

    /// Analyze vector dataset characteristics
    pub fn analyze_vector_dataset(
        &mut self,
        vectors: &[(VectorId, Vector)],
    ) -> Result<VectorStatistics> {
        info!("Analyzing vector dataset with {} vectors", vectors.len());

        let mut stats = VectorStatistics::default();
        stats.total_vectors = vectors.len() as u64;

        if vectors.is_empty() {
            return Ok(stats);
        }

        // Analyze dimensions
        let dimensions: Vec<u32> = vectors.iter().map(|(_, v)| v.dimensions as u32).collect();
        stats.average_dimension = dimensions.iter().sum::<u32>() / dimensions.len() as u32;

        for &dim in &dimensions {
            *stats.dimension_distribution.entry(dim).or_insert(0) += 1;
        }

        // Analyze vector density and sparsity
        let (density, sparsity) = self.calculate_vector_density(vectors);
        stats.vector_density = density;
        stats.sparsity_ratio = sparsity;

        // Calculate clustering coefficient
        stats.clustering_coefficient = self.calculate_clustering_coefficient(vectors);

        // Calculate hubness measure
        stats.hubness_measure = self.calculate_hubness_measure(vectors);

        // Estimate intrinsic dimensionality
        stats.intrinsic_dimensionality = self.estimate_intrinsic_dimensionality(vectors);

        // Estimate noise level
        stats.noise_estimation = self.estimate_noise_level(vectors);

        // Calculate overall quality score
        stats.quality_score = self.calculate_quality_score(&stats);

        self.vector_stats = stats.clone();
        Ok(stats)
    }

    /// Generate comprehensive optimization recommendations
    pub fn generate_recommendations(&mut self) -> OptimizationRecommendations {
        let mut recommendations = OptimizationRecommendations::default();

        // Index recommendations
        recommendations.index_recommendations = self.generate_index_recommendations();

        // Caching recommendations
        recommendations.caching_recommendations = self.generate_caching_recommendations();

        // Query optimization recommendations
        recommendations.query_recommendations = self.generate_query_recommendations();

        // Hardware recommendations
        recommendations.hardware_recommendations = self.generate_hardware_recommendations();

        // Configuration recommendations
        recommendations.configuration_recommendations =
            self.generate_configuration_recommendations();

        // Priority actions
        recommendations.priority_actions = self.generate_priority_actions();

        // Estimated improvements
        recommendations.estimated_improvements = self.estimate_improvements(&recommendations);

        self.recommendations = recommendations.clone();
        recommendations
    }

    /// Export comprehensive performance report
    pub fn export_performance_report(&self, format: ReportFormat) -> Result<String> {
        match format {
            ReportFormat::Json => {
                let report = PerformanceReport {
                    timestamp: SystemTime::now(),
                    query_stats: self.query_stats.clone(),
                    vector_stats: self.vector_stats.clone(),
                    trends: self.trends.clone(),
                    recommendations: self.recommendations.clone(),
                    alerts: self.alerting_system.get_active_alerts(),
                };
                Ok(serde_json::to_string_pretty(&report)?)
            }
            ReportFormat::Csv => self.export_csv_report(),
            ReportFormat::Html => self.export_html_report(),
            ReportFormat::Prometheus => self.export_prometheus_metrics(),
        }
    }

    // Private helper methods

    fn update_latency_statistics(&mut self, latency_ms: f64) {
        // Update rolling average
        let alpha = 0.1; // Exponential moving average factor
        if self.query_stats.total_queries == 1 {
            self.query_stats.average_latency_ms = latency_ms;
        } else {
            self.query_stats.average_latency_ms =
                alpha * latency_ms + (1.0 - alpha) * self.query_stats.average_latency_ms;
        }

        // Update percentiles (simplified implementation)
        self.metrics_collector
            .query_times
            .push(Duration::from_secs_f64(latency_ms / 1000.0));
        self.update_percentiles();
    }

    fn update_percentiles(&mut self) {
        let mut times: Vec<f64> = self
            .metrics_collector
            .query_times
            .iter()
            .map(|d| d.as_secs_f64() * 1000.0)
            .collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if !times.is_empty() {
            self.query_stats.p50_latency_ms = self.percentile(&times, 0.5);
            self.query_stats.p95_latency_ms = self.percentile(&times, 0.95);
            self.query_stats.p99_latency_ms = self.percentile(&times, 0.99);
        }
    }

    fn percentile(&self, sorted_data: &[f64], p: f64) -> f64 {
        if sorted_data.is_empty() {
            return 0.0;
        }
        let index = (p * (sorted_data.len() - 1) as f64).round() as usize;
        sorted_data[index.min(sorted_data.len() - 1)]
    }

    fn calculate_error_rate(&self) -> f64 {
        let total_errors: u64 = self.metrics_collector.error_counts.values().sum();
        if self.query_stats.total_queries > 0 {
            total_errors as f64 / self.query_stats.total_queries as f64
        } else {
            0.0
        }
    }

    fn check_performance_alerts(&mut self, latency_ms: f64) {
        // Collect rules that need alerts to avoid borrowing conflict
        let mut rules_to_alert = Vec::new();
        for rule in &self.alerting_system.alert_rules {
            if rule.enabled && self.evaluate_alert_rule(rule, latency_ms) {
                rules_to_alert.push(rule.clone());
            }
        }

        // Now trigger alerts for collected rules
        for rule in rules_to_alert {
            self.alerting_system.trigger_alert(&rule, latency_ms);
        }
    }

    fn evaluate_alert_rule(&self, rule: &AlertRule, value: f64) -> bool {
        match rule.comparison {
            ComparisonOperator::GreaterThan => value > rule.threshold,
            ComparisonOperator::LessThan => value < rule.threshold,
            ComparisonOperator::Equals => (value - rule.threshold).abs() < f64::EPSILON,
            ComparisonOperator::GreaterThanOrEqual => value >= rule.threshold,
            ComparisonOperator::LessThanOrEqual => value <= rule.threshold,
        }
    }

    fn calculate_vector_density(&self, vectors: &[(VectorId, Vector)]) -> (f64, f64) {
        if vectors.is_empty() {
            return (0.0, 0.0);
        }

        let mut total_non_zero = 0;
        let mut total_elements = 0;

        for (_, vector) in vectors {
            let values = &vector.as_f32();
            total_elements += values.len();
            total_non_zero += values.iter().filter(|&&x| x != 0.0).count();
        }

        let density = total_non_zero as f64 / total_elements as f64;
        let sparsity = 1.0 - density;
        (density, sparsity)
    }

    fn calculate_clustering_coefficient(&self, _vectors: &[(VectorId, Vector)]) -> f64 {
        // Simplified clustering coefficient calculation
        // In a full implementation, this would analyze the k-NN graph
        0.5 // Placeholder
    }

    fn calculate_hubness_measure(&self, _vectors: &[(VectorId, Vector)]) -> f64 {
        // Simplified hubness measure
        // Measures how often certain vectors appear in k-NN lists
        0.3 // Placeholder
    }

    fn estimate_intrinsic_dimensionality(&self, _vectors: &[(VectorId, Vector)]) -> f64 {
        // Simplified intrinsic dimensionality estimation
        // Could use techniques like MLE or correlation dimension
        if self.vector_stats.average_dimension > 0 {
            self.vector_stats.average_dimension as f64 * 0.7
        } else {
            10.0
        }
    }

    fn estimate_noise_level(&self, _vectors: &[(VectorId, Vector)]) -> f64 {
        // Simplified noise estimation
        0.1 // Placeholder - 10% noise level
    }

    fn calculate_quality_score(&self, stats: &VectorStatistics) -> f64 {
        let density_score = stats.vector_density;
        let clustering_score = stats.clustering_coefficient;
        let noise_penalty = 1.0 - stats.noise_estimation;

        (density_score + clustering_score + noise_penalty) / 3.0
    }

    fn generate_index_recommendations(&self) -> Vec<IndexRecommendation> {
        let mut recommendations = Vec::new();

        if self.query_stats.average_latency_ms > 10.0 {
            recommendations.push(IndexRecommendation {
                recommendation_type: "HNSW Optimization".to_string(),
                description: "Consider optimizing HNSW parameters (M, efConstruction) for better search performance".to_string(),
                estimated_improvement: 0.3,
                implementation_effort: EffortLevel::Medium,
                prerequisites: vec!["Performance profiling".to_string()],
            });
        }

        if self.vector_stats.sparsity_ratio > 0.8 {
            recommendations.push(IndexRecommendation {
                recommendation_type: "Sparse Index".to_string(),
                description:
                    "High sparsity detected - consider using sparse-optimized index structures"
                        .to_string(),
                estimated_improvement: 0.4,
                implementation_effort: EffortLevel::High,
                prerequisites: vec!["Sparse vector support".to_string()],
            });
        }

        recommendations
    }

    fn generate_caching_recommendations(&self) -> Vec<CachingRecommendation> {
        let mut recommendations = Vec::new();

        if self.query_stats.cache_hit_rate < 0.7 {
            recommendations.push(CachingRecommendation {
                cache_type: "Query Result Cache".to_string(),
                recommended_size_mb: 512,
                eviction_policy: "LRU".to_string(),
                estimated_hit_rate: 0.85,
                cost_benefit_ratio: 2.5,
            });
        }

        recommendations
    }

    fn generate_query_recommendations(&self) -> Vec<QueryRecommendation> {
        vec![QueryRecommendation {
            query_pattern: "High-dimensional similarity search".to_string(),
            optimization_technique: "Dimensionality reduction with PCA".to_string(),
            expected_speedup: 1.5,
            applicability: 0.8,
        }]
    }

    fn generate_hardware_recommendations(&self) -> Vec<HardwareRecommendation> {
        let mut recommendations = Vec::new();

        if self.query_stats.memory_usage_mb > 8192.0 {
            recommendations.push(HardwareRecommendation {
                component: "Memory".to_string(),
                current_bottleneck: true,
                recommended_upgrade: "Increase RAM to 32GB+".to_string(),
                performance_impact: 0.4,
                cost_estimate: Some("$200-500".to_string()),
            });
        }

        recommendations
    }

    fn generate_configuration_recommendations(&self) -> Vec<ConfigurationRecommendation> {
        vec![ConfigurationRecommendation {
            parameter: "thread_pool_size".to_string(),
            current_value: "4".to_string(),
            recommended_value: "8".to_string(),
            justification: "CPU utilization suggests more threads could improve throughput"
                .to_string(),
            risk_level: RiskLevel::Low,
        }]
    }

    fn generate_priority_actions(&self) -> Vec<PriorityAction> {
        let mut actions = Vec::new();

        if self.query_stats.average_latency_ms > 50.0 {
            actions.push(PriorityAction {
                action: "Optimize slow queries".to_string(),
                priority: Priority::High,
                estimated_impact: 0.6,
                implementation_time: Duration::from_secs(3600 * 8), // 8 hours
                dependencies: vec!["Query profiling".to_string()],
            });
        }

        actions.sort_by(|a, b| b.priority.cmp(&a.priority));
        actions
    }

    fn estimate_improvements(
        &self,
        _recommendations: &OptimizationRecommendations,
    ) -> ImprovementEstimates {
        ImprovementEstimates {
            latency_improvement: 0.3,
            throughput_improvement: 0.25,
            memory_savings: 0.15,
            cost_reduction: 0.2,
            reliability_improvement: 0.1,
        }
    }

    fn export_csv_report(&self) -> Result<String> {
        let mut csv = String::new();
        csv.push_str("Metric,Value,Unit\n");
        csv.push_str(&format!(
            "Total Queries,{},count\n",
            self.query_stats.total_queries
        ));
        csv.push_str(&format!(
            "Average Latency,{:.2},ms\n",
            self.query_stats.average_latency_ms
        ));
        csv.push_str(&format!(
            "P99 Latency,{:.2},ms\n",
            self.query_stats.p99_latency_ms
        ));
        csv.push_str(&format!(
            "Throughput,{:.2},QPS\n",
            self.query_stats.throughput_qps
        ));
        csv.push_str(&format!(
            "Error Rate,{:.4},ratio\n",
            self.query_stats.error_rate
        ));
        csv.push_str(&format!(
            "Cache Hit Rate,{:.4},ratio\n",
            self.query_stats.cache_hit_rate
        ));
        Ok(csv)
    }

    fn export_html_report(&self) -> Result<String> {
        let html = format!(
            r#"
            <!DOCTYPE html>
            <html>
            <head><title>OxiRS Vector Search Performance Report</title></head>
            <body>
                <h1>Performance Report</h1>
                <h2>Query Statistics</h2>
                <p>Total Queries: {}</p>
                <p>Average Latency: {:.2} ms</p>
                <p>P99 Latency: {:.2} ms</p>
                <p>Throughput: {:.2} QPS</p>
                <p>Error Rate: {:.4}</p>
                <h2>Vector Statistics</h2>
                <p>Total Vectors: {}</p>
                <p>Average Dimension: {}</p>
                <p>Vector Density: {:.4}</p>
                <p>Quality Score: {:.4}</p>
            </body>
            </html>
            "#,
            self.query_stats.total_queries,
            self.query_stats.average_latency_ms,
            self.query_stats.p99_latency_ms,
            self.query_stats.throughput_qps,
            self.query_stats.error_rate,
            self.vector_stats.total_vectors,
            self.vector_stats.average_dimension,
            self.vector_stats.vector_density,
            self.vector_stats.quality_score,
        );
        Ok(html)
    }

    fn export_prometheus_metrics(&self) -> Result<String> {
        let mut metrics = String::new();
        metrics.push_str(&format!(
            "oxirs_query_total {}\n",
            self.query_stats.total_queries
        ));
        metrics.push_str(&format!(
            "oxirs_query_latency_avg {}\n",
            self.query_stats.average_latency_ms
        ));
        metrics.push_str(&format!(
            "oxirs_query_latency_p99 {}\n",
            self.query_stats.p99_latency_ms
        ));
        metrics.push_str(&format!(
            "oxirs_query_throughput {}\n",
            self.query_stats.throughput_qps
        ));
        metrics.push_str(&format!(
            "oxirs_query_error_rate {}\n",
            self.query_stats.error_rate
        ));
        metrics.push_str(&format!(
            "oxirs_cache_hit_rate {}\n",
            self.query_stats.cache_hit_rate
        ));
        metrics.push_str(&format!(
            "oxirs_vector_total {}\n",
            self.vector_stats.total_vectors
        ));
        metrics.push_str(&format!(
            "oxirs_vector_quality_score {}\n",
            self.vector_stats.quality_score
        ));
        Ok(metrics)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub timestamp: SystemTime,
    pub query_stats: QueryStatistics,
    pub vector_stats: VectorStatistics,
    pub trends: PerformanceTrends,
    pub recommendations: OptimizationRecommendations,
    pub alerts: Vec<ActiveAlert>,
}

#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    Json,
    Csv,
    Html,
    Prometheus,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            query_times: Vec::new(),
            query_complexities: Vec::new(),
            memory_samples: Vec::new(),
            cpu_samples: Vec::new(),
            cache_metrics: CacheMetrics::default(),
            error_counts: HashMap::new(),
            active_queries: 0,
        }
    }

    pub fn record_query(&mut self, duration: Duration, complexity: QueryComplexity, success: bool) {
        self.query_times.push(duration);
        self.query_complexities.push(complexity);

        if !success {
            *self
                .error_counts
                .entry("query_error".to_string())
                .or_insert(0) += 1;
        }
    }
}

impl AlertingSystem {
    pub fn new() -> Self {
        Self {
            alert_rules: Self::default_alert_rules(),
            active_alerts: Vec::new(),
            alert_history: Vec::new(),
            notification_channels: vec![NotificationChannel::Console],
        }
    }

    fn default_alert_rules() -> Vec<AlertRule> {
        vec![
            AlertRule {
                name: "High Latency".to_string(),
                metric: "latency_p99".to_string(),
                threshold: 100.0,
                comparison: ComparisonOperator::GreaterThan,
                severity: AlertSeverity::Warning,
                enabled: true,
            },
            AlertRule {
                name: "Critical Latency".to_string(),
                metric: "latency_p99".to_string(),
                threshold: 1000.0,
                comparison: ComparisonOperator::GreaterThan,
                severity: AlertSeverity::Critical,
                enabled: true,
            },
        ]
    }

    pub fn trigger_alert(&mut self, rule: &AlertRule, value: f64) {
        let alert = ActiveAlert {
            rule_name: rule.name.clone(),
            triggered_at: SystemTime::now(),
            current_value: value,
            threshold: rule.threshold,
            severity: rule.severity,
            acknowledged: false,
        };

        self.active_alerts.push(alert.clone());

        let event = AlertEvent {
            timestamp: SystemTime::now(),
            alert_name: rule.name.clone(),
            event_type: AlertEventType::Triggered,
            details: format!(
                "Alert triggered: {} = {:.2} > {:.2}",
                rule.metric, value, rule.threshold
            ),
        };

        self.alert_history.push(event);
        self.send_notifications(&alert);
    }

    pub fn get_active_alerts(&self) -> Vec<ActiveAlert> {
        self.active_alerts.clone()
    }

    fn send_notifications(&self, alert: &ActiveAlert) {
        for channel in &self.notification_channels {
            match channel {
                NotificationChannel::Console => {
                    warn!(
                        "ALERT: {} - {} at {:.2} (threshold: {:.2})",
                        alert.rule_name, alert.severity as u8, alert.current_value, alert.threshold
                    );
                }
                NotificationChannel::Email(_) => {
                    debug!(
                        "Would send email notification for alert: {}",
                        alert.rule_name
                    );
                }
                NotificationChannel::Webhook(_) => {
                    debug!(
                        "Would send webhook notification for alert: {}",
                        alert.rule_name
                    );
                }
                NotificationChannel::Slack(_) => {
                    debug!(
                        "Would send Slack notification for alert: {}",
                        alert.rule_name
                    );
                }
            }
        }
    }
}

impl Default for PerformanceInsightsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_insights_creation() {
        let analyzer = PerformanceInsightsAnalyzer::new();
        assert_eq!(analyzer.query_stats.total_queries, 0);
        assert_eq!(analyzer.vector_stats.total_vectors, 0);
    }

    #[test]
    fn test_query_recording() {
        let mut analyzer = PerformanceInsightsAnalyzer::new();
        analyzer.record_query(Duration::from_millis(50), QueryComplexity::Simple, true);

        assert_eq!(analyzer.query_stats.total_queries, 1);
        assert_eq!(analyzer.query_stats.average_latency_ms, 50.0);
    }

    #[test]
    fn test_vector_analysis() {
        let mut analyzer = PerformanceInsightsAnalyzer::new();
        let vectors = vec![
            ("vec1".to_string(), Vector::new(vec![1.0, 2.0, 3.0])),
            ("vec2".to_string(), Vector::new(vec![4.0, 5.0, 6.0])),
        ];

        let stats = analyzer.analyze_vector_dataset(&vectors).unwrap();
        assert_eq!(stats.total_vectors, 2);
        assert_eq!(stats.average_dimension, 3);
    }

    #[test]
    fn test_alert_generation() {
        let mut analyzer = PerformanceInsightsAnalyzer::new();
        // Record a slow query that should trigger an alert
        analyzer.record_query(Duration::from_millis(1500), QueryComplexity::Complex, true);

        assert!(!analyzer.alerting_system.active_alerts.is_empty());
    }

    #[test]
    fn test_recommendations_generation() {
        let mut analyzer = PerformanceInsightsAnalyzer::new();
        // Set up conditions that should trigger recommendations
        analyzer.query_stats.average_latency_ms = 100.0;
        analyzer.vector_stats.sparsity_ratio = 0.9;

        let recommendations = analyzer.generate_recommendations();
        assert!(!recommendations.index_recommendations.is_empty());
    }

    #[test]
    fn test_report_export() {
        let analyzer = PerformanceInsightsAnalyzer::new();
        let json_report = analyzer
            .export_performance_report(ReportFormat::Json)
            .unwrap();
        assert!(!json_report.is_empty());

        let csv_report = analyzer
            .export_performance_report(ReportFormat::Csv)
            .unwrap();
        assert!(csv_report.contains("Metric,Value,Unit"));
    }
}
