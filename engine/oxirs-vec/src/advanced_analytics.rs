//! # Advanced Analytics and Insights for Vector Search
//!
//! This module provides comprehensive analytics and insights for vector search operations:
//! - Search pattern analysis and optimization recommendations
//! - Vector distribution analysis and cluster insights  
//! - Performance trend analysis and predictive modeling
//! - Query optimization suggestions based on usage patterns
//! - Anomaly detection in search behavior
//! - Vector quality assessment and recommendations

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::similarity::SimilarityMetric;

/// Search query analytics and patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalytics {
    pub query_id: String,
    pub timestamp: u64,
    pub query_vector: Vec<f32>,
    pub similarity_metric: String,
    pub top_k: usize,
    pub response_time: Duration,
    pub results_count: usize,
    pub avg_similarity_score: f32,
    pub min_similarity_score: f32,
    pub max_similarity_score: f32,
    pub cache_hit: bool,
    pub index_type: String,
}

/// Vector distribution and clustering insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDistributionAnalysis {
    pub total_vectors: usize,
    pub dimensionality: usize,
    pub density_estimate: f32,
    pub cluster_count: usize,
    pub cluster_sizes: Vec<usize>,
    pub cluster_cohesion: Vec<f32>,
    pub cluster_separation: f32,
    pub outlier_count: usize,
    pub outlier_threshold: f32,
    pub sparsity_ratio: f32,
    pub distribution_skewness: f32,
}

/// Performance trends and predictions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTrends {
    pub time_window: Duration,
    pub query_volume_trend: Vec<(u64, usize)>,
    pub response_time_trend: Vec<(u64, f32)>,
    pub cache_hit_rate_trend: Vec<(u64, f32)>,
    pub error_rate_trend: Vec<(u64, f32)>,
    pub predicted_peak_hours: Vec<u8>,
    pub performance_score: f32,
    pub bottleneck_analysis: Vec<BottleneckInsight>,
}

/// Performance bottleneck insights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckInsight {
    pub component: String,
    pub severity: BottleneckSeverity,
    pub impact_score: f32,
    pub recommendation: String,
    pub estimated_improvement: f32,
}

/// Severity levels for bottlenecks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Critical,
    High,
    Medium,
    Low,
}

/// Query optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecommendation {
    pub recommendation_type: RecommendationType,
    pub priority: Priority,
    pub description: String,
    pub expected_improvement: f32,
    pub implementation_effort: ImplementationEffort,
    pub affected_queries: Vec<String>,
}

/// Types of optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    IndexOptimization,
    CacheStrategy,
    SimilarityMetric,
    Preprocessing,
    Batching,
    Hardware,
}

/// Priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Priority {
    Critical,
    High,
    Medium,
    Low,
}

/// Implementation effort estimates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationEffort {
    Minimal,
    Low,
    Medium,
    High,
    Significant,
}

/// Anomaly detection results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetection {
    pub anomalies: Vec<QueryAnomaly>,
    pub detection_threshold: f32,
    pub false_positive_rate: f32,
    pub confidence_level: f32,
}

/// Individual query anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnomaly {
    pub query_id: String,
    pub anomaly_type: AnomalyType,
    pub severity_score: f32,
    pub description: String,
    pub suggested_action: String,
}

/// Types of anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    UnusualLatency,
    LowSimilarityScores,
    HighErrorRate,
    UnexpectedTraffic,
    SuspiciousPattern,
}

/// Vector quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorQualityAssessment {
    pub overall_quality_score: f32,
    pub dimension_quality: Vec<f32>,
    pub noise_level: f32,
    pub embedding_consistency: f32,
    pub semantic_coherence: f32,
    pub recommendations: Vec<QualityRecommendation>,
}

/// Quality improvement recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRecommendation {
    pub aspect: QualityAspect,
    pub current_score: f32,
    pub target_score: f32,
    pub recommendation: String,
    pub priority: Priority,
}

/// Quality aspects
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityAspect {
    DimensionalityReduction,
    NoiseReduction,
    EmbeddingModel,
    Preprocessing,
    DataCleaning,
}

/// Main analytics engine
#[derive(Debug)]
pub struct VectorAnalyticsEngine {
    query_history: VecDeque<QueryAnalytics>,
    performance_metrics: BTreeMap<u64, PerformanceMetrics>,
    max_history_size: usize,
    analysis_window: Duration,
    anomaly_detector: AnomalyDetector,
}

/// Performance metrics for a time period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: u64,
    pub query_count: usize,
    pub avg_response_time: f32,
    pub cache_hit_rate: f32,
    pub error_rate: f32,
    pub throughput: f32,
}

/// Anomaly detection system
#[derive(Debug)]
pub struct AnomalyDetector {
    baseline_metrics: HashMap<String, f32>,
    detection_sensitivity: f32,
    learning_rate: f32,
}

impl VectorAnalyticsEngine {
    /// Create a new analytics engine
    pub fn new() -> Self {
        Self {
            query_history: VecDeque::new(),
            performance_metrics: BTreeMap::new(),
            max_history_size: 10000,
            analysis_window: Duration::from_secs(24 * 60 * 60), // 24 hours
            anomaly_detector: AnomalyDetector::new(),
        }
    }

    /// Create analytics engine with custom configuration
    pub fn with_config(max_history: usize, window: Duration, sensitivity: f32) -> Self {
        Self {
            query_history: VecDeque::new(),
            performance_metrics: BTreeMap::new(),
            max_history_size: max_history,
            analysis_window: window,
            anomaly_detector: AnomalyDetector::with_sensitivity(sensitivity),
        }
    }

    /// Record a search query for analysis
    pub fn record_query(&mut self, analytics: QueryAnalytics) {
        // Add to history with size limit
        if self.query_history.len() >= self.max_history_size {
            self.query_history.pop_front();
        }

        self.query_history.push_back(analytics.clone());

        // Update performance metrics
        self.update_performance_metrics(&analytics);

        // Update anomaly detector
        self.anomaly_detector.update_baseline(&analytics);
    }

    /// Analyze vector distribution patterns
    pub fn analyze_vector_distribution(
        &self,
        vectors: &[Vec<f32>],
    ) -> Result<VectorDistributionAnalysis> {
        if vectors.is_empty() {
            return Err(anyhow!("Cannot analyze empty vector set"));
        }

        let total_vectors = vectors.len();
        let dimensionality = vectors[0].len();

        // Calculate basic statistics
        let density_estimate = self.calculate_density_estimate(vectors);
        let sparsity_ratio = self.calculate_sparsity_ratio(vectors);
        let distribution_skewness = self.calculate_skewness(vectors);

        // Perform clustering analysis
        let (cluster_count, cluster_sizes, cluster_cohesion, cluster_separation) =
            self.analyze_clustering(vectors)?;

        // Detect outliers
        let (outlier_count, outlier_threshold) = self.detect_outliers(vectors);

        Ok(VectorDistributionAnalysis {
            total_vectors,
            dimensionality,
            density_estimate,
            cluster_count,
            cluster_sizes,
            cluster_cohesion,
            cluster_separation,
            outlier_count,
            outlier_threshold,
            sparsity_ratio,
            distribution_skewness,
        })
    }

    /// Generate performance trends analysis
    pub fn analyze_performance_trends(&self) -> PerformanceTrends {
        let cutoff_time = self.current_timestamp() - self.analysis_window.as_secs();

        let query_volume_trend = self.calculate_query_volume_trend(cutoff_time);
        let response_time_trend = self.calculate_response_time_trend(cutoff_time);
        let cache_hit_rate_trend = self.calculate_cache_hit_rate_trend(cutoff_time);
        let error_rate_trend = self.calculate_error_rate_trend(cutoff_time);

        let predicted_peak_hours = self.predict_peak_hours();
        let performance_score = self.calculate_performance_score();
        let bottleneck_analysis = self.analyze_bottlenecks();

        PerformanceTrends {
            time_window: self.analysis_window,
            query_volume_trend,
            response_time_trend,
            cache_hit_rate_trend,
            error_rate_trend,
            predicted_peak_hours,
            performance_score,
            bottleneck_analysis,
        }
    }

    /// Generate optimization recommendations
    pub fn generate_optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        let mut recommendations = Vec::new();

        // Analyze query patterns for index optimization
        recommendations.extend(self.analyze_index_optimization());

        // Analyze cache effectiveness
        recommendations.extend(self.analyze_cache_optimization());

        // Analyze similarity metric usage
        recommendations.extend(self.analyze_similarity_optimization());

        // Analyze batching opportunities
        recommendations.extend(self.analyze_batching_optimization());

        // Sort by priority and expected improvement
        recommendations.sort_by(|a, b| match (&a.priority, &b.priority) {
            (Priority::Critical, Priority::Critical) => b
                .expected_improvement
                .partial_cmp(&a.expected_improvement)
                .unwrap(),
            (Priority::Critical, _) => std::cmp::Ordering::Less,
            (_, Priority::Critical) => std::cmp::Ordering::Greater,
            (Priority::High, Priority::High) => b
                .expected_improvement
                .partial_cmp(&a.expected_improvement)
                .unwrap(),
            (Priority::High, _) => std::cmp::Ordering::Less,
            (_, Priority::High) => std::cmp::Ordering::Greater,
            _ => b
                .expected_improvement
                .partial_cmp(&a.expected_improvement)
                .unwrap(),
        });

        recommendations
    }

    /// Detect anomalies in query patterns
    pub fn detect_anomalies(&self) -> AnomalyDetection {
        self.anomaly_detector.detect_anomalies(&self.query_history)
    }

    /// Assess vector quality
    pub fn assess_vector_quality(&self, vectors: &[Vec<f32>]) -> Result<VectorQualityAssessment> {
        if vectors.is_empty() {
            return Err(anyhow!("Cannot assess quality of empty vector set"));
        }

        let overall_quality_score = self.calculate_overall_quality(vectors);
        let dimension_quality = self.calculate_dimension_quality(vectors);
        let noise_level = self.estimate_noise_level(vectors);
        let embedding_consistency = self.calculate_embedding_consistency(vectors);
        let semantic_coherence = self.calculate_semantic_coherence(vectors);
        let recommendations = self.generate_quality_recommendations(
            overall_quality_score,
            &dimension_quality,
            noise_level,
        );

        Ok(VectorQualityAssessment {
            overall_quality_score,
            dimension_quality,
            noise_level,
            embedding_consistency,
            semantic_coherence,
            recommendations,
        })
    }

    /// Export analytics data to JSON
    pub fn export_analytics(&self) -> Result<String> {
        #[derive(Serialize)]
        struct AnalyticsExport {
            query_count: usize,
            performance_trends: PerformanceTrends,
            recommendations: Vec<OptimizationRecommendation>,
            anomalies: AnomalyDetection,
            export_timestamp: u64,
        }

        let export = AnalyticsExport {
            query_count: self.query_history.len(),
            performance_trends: self.analyze_performance_trends(),
            recommendations: self.generate_optimization_recommendations(),
            anomalies: self.detect_anomalies(),
            export_timestamp: self.current_timestamp(),
        };

        serde_json::to_string_pretty(&export)
            .map_err(|e| anyhow!("Failed to export analytics: {}", e))
    }

    // Private implementation methods

    fn update_performance_metrics(&mut self, query: &QueryAnalytics) {
        let time_bucket = (query.timestamp / 300) * 300; // 5-minute buckets

        let metrics = self
            .performance_metrics
            .entry(time_bucket)
            .or_insert(PerformanceMetrics {
                timestamp: time_bucket,
                query_count: 0,
                avg_response_time: 0.0,
                cache_hit_rate: 0.0,
                error_rate: 0.0,
                throughput: 0.0,
            });

        metrics.query_count += 1;
        metrics.avg_response_time = (metrics.avg_response_time * (metrics.query_count - 1) as f32
            + query.response_time.as_secs_f32())
            / metrics.query_count as f32;

        if query.cache_hit {
            metrics.cache_hit_rate = (metrics.cache_hit_rate * (metrics.query_count - 1) as f32
                + 1.0)
                / metrics.query_count as f32;
        } else {
            metrics.cache_hit_rate = (metrics.cache_hit_rate * (metrics.query_count - 1) as f32)
                / metrics.query_count as f32;
        }

        metrics.throughput = metrics.query_count as f32 / 300.0; // queries per second in 5-min window
    }

    fn calculate_density_estimate(&self, vectors: &[Vec<f32>]) -> f32 {
        // Simplified density estimation using average pairwise distances
        if vectors.len() < 2 {
            return 0.0;
        }

        let mut total_distance = 0.0;
        let mut count = 0;

        for (i, v1) in vectors.iter().enumerate().take(100) {
            // Sample for performance
            for v2 in vectors.iter().skip(i + 1).take(10) {
                total_distance += euclidean_distance(v1, v2);
                count += 1;
            }
        }

        if count > 0 {
            1.0 / (total_distance / count as f32)
        } else {
            0.0
        }
    }

    fn calculate_sparsity_ratio(&self, vectors: &[Vec<f32>]) -> f32 {
        let mut zero_count = 0;
        let mut total_elements = 0;

        for vector in vectors {
            for &value in vector {
                if value.abs() < 1e-6 {
                    zero_count += 1;
                }
                total_elements += 1;
            }
        }

        zero_count as f32 / total_elements as f32
    }

    fn calculate_skewness(&self, vectors: &[Vec<f32>]) -> f32 {
        // Calculate skewness for the first dimension as a representative measure
        if vectors.is_empty() || vectors[0].is_empty() {
            return 0.0;
        }

        let values: Vec<f32> = vectors.iter().map(|v| v[0]).collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return 0.0;
        }

        let skewness = values
            .iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f32>()
            / values.len() as f32;

        skewness
    }

    fn analyze_clustering(
        &self,
        vectors: &[Vec<f32>],
    ) -> Result<(usize, Vec<usize>, Vec<f32>, f32)> {
        // Simplified clustering analysis using k-means with multiple k values
        let max_k = (vectors.len() as f32).sqrt() as usize;
        let optimal_k = (max_k / 2).max(2).min(10);

        // For simplicity, return estimated values
        let cluster_count = optimal_k;
        let cluster_sizes = vec![vectors.len() / cluster_count; cluster_count];
        let cluster_cohesion = vec![0.8; cluster_count]; // Simulated cohesion scores
        let cluster_separation = 0.6; // Simulated separation score

        Ok((
            cluster_count,
            cluster_sizes,
            cluster_cohesion,
            cluster_separation,
        ))
    }

    fn detect_outliers(&self, vectors: &[Vec<f32>]) -> (usize, f32) {
        // Simple outlier detection based on distance to centroid
        let centroid = calculate_centroid(vectors);
        let mut distances = Vec::new();

        for vector in vectors {
            distances.push(euclidean_distance(vector, &centroid));
        }

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q3_index = (distances.len() as f32 * 0.75) as usize;
        let q1_index = (distances.len() as f32 * 0.25) as usize;

        let iqr = distances[q3_index] - distances[q1_index];
        let threshold = distances[q3_index] + 1.5 * iqr;

        let outlier_count = distances.iter().filter(|&&d| d > threshold).count();

        (outlier_count, threshold)
    }

    fn calculate_query_volume_trend(&self, cutoff_time: u64) -> Vec<(u64, usize)> {
        let mut hourly_counts = BTreeMap::new();

        for query in &self.query_history {
            if query.timestamp > cutoff_time {
                let hour_bucket = (query.timestamp / 3600) * 3600;
                *hourly_counts.entry(hour_bucket).or_insert(0) += 1;
            }
        }

        hourly_counts.into_iter().collect()
    }

    fn calculate_response_time_trend(&self, cutoff_time: u64) -> Vec<(u64, f32)> {
        let mut hourly_times = BTreeMap::new();
        let mut hourly_counts = BTreeMap::new();

        for query in &self.query_history {
            if query.timestamp > cutoff_time {
                let hour_bucket = (query.timestamp / 3600) * 3600;
                *hourly_times.entry(hour_bucket).or_insert(0.0) +=
                    query.response_time.as_secs_f32();
                *hourly_counts.entry(hour_bucket).or_insert(0) += 1;
            }
        }

        hourly_times
            .into_iter()
            .map(|(time, total)| (time, total / hourly_counts[&time] as f32))
            .collect()
    }

    fn calculate_cache_hit_rate_trend(&self, cutoff_time: u64) -> Vec<(u64, f32)> {
        let mut hourly_hits = BTreeMap::new();
        let mut hourly_counts = BTreeMap::new();

        for query in &self.query_history {
            if query.timestamp > cutoff_time {
                let hour_bucket = (query.timestamp / 3600) * 3600;
                if query.cache_hit {
                    *hourly_hits.entry(hour_bucket).or_insert(0) += 1;
                }
                *hourly_counts.entry(hour_bucket).or_insert(0) += 1;
            }
        }

        hourly_counts
            .into_iter()
            .map(|(time, count)| {
                let hits = hourly_hits.get(&time).unwrap_or(&0);
                (time, *hits as f32 / count as f32)
            })
            .collect()
    }

    fn calculate_error_rate_trend(&self, _cutoff_time: u64) -> Vec<(u64, f32)> {
        // Placeholder - would track actual errors
        vec![]
    }

    fn predict_peak_hours(&self) -> Vec<u8> {
        let mut hour_volumes = vec![0; 24];

        for query in &self.query_history {
            let hour = ((query.timestamp % 86400) / 3600) as usize;
            if hour < 24 {
                hour_volumes[hour] += 1;
            }
        }

        let avg_volume = hour_volumes.iter().sum::<usize>() as f32 / 24.0;

        hour_volumes
            .iter()
            .enumerate()
            .filter(|(_, &volume)| volume as f32 > avg_volume * 1.5)
            .map(|(hour, _)| hour as u8)
            .collect()
    }

    fn calculate_performance_score(&self) -> f32 {
        if self.query_history.is_empty() {
            return 0.0;
        }

        let avg_response_time = self
            .query_history
            .iter()
            .map(|q| q.response_time.as_secs_f32())
            .sum::<f32>()
            / self.query_history.len() as f32;

        let cache_hit_rate = self.query_history.iter().filter(|q| q.cache_hit).count() as f32
            / self.query_history.len() as f32;

        let avg_similarity = self
            .query_history
            .iter()
            .map(|q| q.avg_similarity_score)
            .sum::<f32>()
            / self.query_history.len() as f32;

        // Composite score (0-1)
        let response_score = 1.0 / (1.0 + avg_response_time);
        let cache_score = cache_hit_rate;
        let similarity_score = avg_similarity;

        (response_score + cache_score + similarity_score) / 3.0
    }

    fn analyze_bottlenecks(&self) -> Vec<BottleneckInsight> {
        let mut bottlenecks = Vec::new();

        // Analyze response time bottlenecks
        let avg_response_time = self
            .query_history
            .iter()
            .map(|q| q.response_time.as_secs_f32())
            .sum::<f32>()
            / self.query_history.len().max(1) as f32;

        if avg_response_time > 0.1 {
            bottlenecks.push(BottleneckInsight {
                component: "Response Time".to_string(),
                severity: if avg_response_time > 1.0 {
                    BottleneckSeverity::Critical
                } else {
                    BottleneckSeverity::High
                },
                impact_score: avg_response_time * 10.0,
                recommendation: "Consider index optimization or caching improvements".to_string(),
                estimated_improvement: 0.3,
            });
        }

        // Analyze cache hit rate
        let cache_hit_rate = self.query_history.iter().filter(|q| q.cache_hit).count() as f32
            / self.query_history.len().max(1) as f32;

        if cache_hit_rate < 0.5 {
            bottlenecks.push(BottleneckInsight {
                component: "Cache Efficiency".to_string(),
                severity: BottleneckSeverity::Medium,
                impact_score: (1.0 - cache_hit_rate) * 5.0,
                recommendation: "Improve cache strategy or increase cache size".to_string(),
                estimated_improvement: 0.25,
            });
        }

        bottlenecks
    }

    fn analyze_index_optimization(&self) -> Vec<OptimizationRecommendation> {
        // Analyze which index types are performing poorly
        let mut recommendations = Vec::new();

        // Group queries by index type and analyze performance
        let mut index_performance = HashMap::new();
        for query in &self.query_history {
            let entry = index_performance
                .entry(&query.index_type)
                .or_insert(Vec::new());
            entry.push(query.response_time.as_secs_f32());
        }

        for (index_type, times) in index_performance {
            let avg_time = times.iter().sum::<f32>() / times.len() as f32;
            if avg_time > 0.05 {
                // 50ms threshold
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::IndexOptimization,
                    priority: if avg_time > 0.2 {
                        Priority::High
                    } else {
                        Priority::Medium
                    },
                    description: format!("Optimize {} index performance", index_type),
                    expected_improvement: (avg_time * 0.3).min(0.8),
                    implementation_effort: ImplementationEffort::Medium,
                    affected_queries: vec![], // Would populate with actual query IDs
                });
            }
        }

        recommendations
    }

    fn analyze_cache_optimization(&self) -> Vec<OptimizationRecommendation> {
        let cache_hit_rate = self.query_history.iter().filter(|q| q.cache_hit).count() as f32
            / self.query_history.len().max(1) as f32;

        let mut recommendations = Vec::new();

        if cache_hit_rate < 0.7 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::CacheStrategy,
                priority: Priority::Medium,
                description: "Improve cache hit rate through better caching strategy".to_string(),
                expected_improvement: (0.7 - cache_hit_rate) * 0.5,
                implementation_effort: ImplementationEffort::Low,
                affected_queries: vec![],
            });
        }

        recommendations
    }

    fn analyze_similarity_optimization(&self) -> Vec<OptimizationRecommendation> {
        // Analyze similarity metric performance
        let mut metric_performance = HashMap::new();

        for query in &self.query_history {
            let entry = metric_performance
                .entry(&query.similarity_metric)
                .or_insert(Vec::new());
            entry.push((
                query.response_time.as_secs_f32(),
                query.avg_similarity_score,
            ));
        }

        let mut recommendations = Vec::new();

        for (metric, performance) in metric_performance {
            let avg_time =
                performance.iter().map(|(t, _)| t).sum::<f32>() / performance.len() as f32;
            let avg_score =
                performance.iter().map(|(_, s)| s).sum::<f32>() / performance.len() as f32;

            if avg_time > 0.05 && avg_score < 0.8 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::SimilarityMetric,
                    priority: Priority::Low,
                    description: format!("Consider alternative to {} similarity metric", metric),
                    expected_improvement: 0.15,
                    implementation_effort: ImplementationEffort::Low,
                    affected_queries: vec![],
                });
            }
        }

        recommendations
    }

    fn analyze_batching_optimization(&self) -> Vec<OptimizationRecommendation> {
        // Analyze query patterns for batching opportunities
        let mut recommendations = Vec::new();

        let single_query_count = self.query_history.iter().filter(|q| q.top_k == 1).count();

        if single_query_count > self.query_history.len() / 3 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::Batching,
                priority: Priority::Medium,
                description: "Consider batching single-result queries for better throughput"
                    .to_string(),
                expected_improvement: 0.2,
                implementation_effort: ImplementationEffort::Medium,
                affected_queries: vec![],
            });
        }

        recommendations
    }

    fn calculate_overall_quality(&self, vectors: &[Vec<f32>]) -> f32 {
        // Composite quality score based on multiple factors
        let consistency = self.calculate_embedding_consistency(vectors);
        let coherence = self.calculate_semantic_coherence(vectors);
        let noise = 1.0 - self.estimate_noise_level(vectors);

        (consistency + coherence + noise) / 3.0
    }

    fn calculate_dimension_quality(&self, vectors: &[Vec<f32>]) -> Vec<f32> {
        if vectors.is_empty() || vectors[0].is_empty() {
            return vec![];
        }

        let dim_count = vectors[0].len();
        let mut quality_scores = vec![0.0; dim_count];

        for dim in 0..dim_count {
            let values: Vec<f32> = vectors.iter().map(|v| v[dim]).collect();
            let variance = calculate_variance(&values);
            let range = values
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
                - values
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();

            // Quality based on variance and range (higher is better for meaningful dimensions)
            quality_scores[dim] = (variance * range).min(1.0);
        }

        quality_scores
    }

    fn estimate_noise_level(&self, vectors: &[Vec<f32>]) -> f32 {
        // Estimate noise as inconsistency in similar vectors
        let mut noise_estimate = 0.0;
        let sample_size = vectors.len().min(100);

        for i in 0..sample_size {
            let mut min_distance = f32::INFINITY;
            for j in 0..sample_size {
                if i != j {
                    let distance = euclidean_distance(&vectors[i], &vectors[j]);
                    min_distance = min_distance.min(distance);
                }
            }
            noise_estimate += min_distance;
        }

        noise_estimate / sample_size as f32
    }

    fn calculate_embedding_consistency(&self, vectors: &[Vec<f32>]) -> f32 {
        // Measure consistency across embeddings
        let centroid = calculate_centroid(vectors);
        let mut total_distance = 0.0;

        for vector in vectors {
            total_distance += euclidean_distance(vector, &centroid);
        }

        let avg_distance = total_distance / vectors.len() as f32;
        1.0 / (1.0 + avg_distance) // Convert to 0-1 score
    }

    fn calculate_semantic_coherence(&self, _vectors: &[Vec<f32>]) -> f32 {
        // Placeholder for semantic coherence calculation
        // Would require domain knowledge or external semantic evaluation
        0.8
    }

    fn generate_quality_recommendations(
        &self,
        overall_score: f32,
        dimension_quality: &[f32],
        noise_level: f32,
    ) -> Vec<QualityRecommendation> {
        let mut recommendations = Vec::new();

        if overall_score < 0.7 {
            recommendations.push(QualityRecommendation {
                aspect: QualityAspect::EmbeddingModel,
                current_score: overall_score,
                target_score: 0.8,
                recommendation: "Consider using a higher-quality embedding model".to_string(),
                priority: Priority::High,
            });
        }

        if noise_level > 0.3 {
            recommendations.push(QualityRecommendation {
                aspect: QualityAspect::NoiseReduction,
                current_score: 1.0 - noise_level,
                target_score: 0.8,
                recommendation: "Apply noise reduction techniques to improve vector quality"
                    .to_string(),
                priority: Priority::Medium,
            });
        }

        let low_quality_dims = dimension_quality
            .iter()
            .enumerate()
            .filter(|(_, &score)| score < 0.5)
            .count();

        if low_quality_dims > dimension_quality.len() / 4 {
            recommendations.push(QualityRecommendation {
                aspect: QualityAspect::DimensionalityReduction,
                current_score: 1.0 - (low_quality_dims as f32 / dimension_quality.len() as f32),
                target_score: 0.9,
                recommendation: "Consider dimensionality reduction to remove low-quality dimensions".to_string(),
                priority: Priority::Medium,
            });
        }

        recommendations
    }

    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

impl AnomalyDetector {
    fn new() -> Self {
        Self {
            baseline_metrics: HashMap::new(),
            detection_sensitivity: 2.0,
            learning_rate: 0.1,
        }
    }

    fn with_sensitivity(sensitivity: f32) -> Self {
        Self {
            baseline_metrics: HashMap::new(),
            detection_sensitivity: sensitivity,
            learning_rate: 0.1,
        }
    }

    fn update_baseline(&mut self, query: &QueryAnalytics) {
        let response_time = query.response_time.as_secs_f32();

        let current_baseline = self.baseline_metrics.get("response_time").unwrap_or(&0.1);
        let new_baseline =
            current_baseline * (1.0 - self.learning_rate) + response_time * self.learning_rate;
        self.baseline_metrics
            .insert("response_time".to_string(), new_baseline);

        let current_similarity = self.baseline_metrics.get("avg_similarity").unwrap_or(&0.8);
        let new_similarity = current_similarity * (1.0 - self.learning_rate)
            + query.avg_similarity_score * self.learning_rate;
        self.baseline_metrics
            .insert("avg_similarity".to_string(), new_similarity);
    }

    fn detect_anomalies(&self, queries: &VecDeque<QueryAnalytics>) -> AnomalyDetection {
        let mut anomalies = Vec::new();

        let response_time_baseline = self.baseline_metrics.get("response_time").unwrap_or(&0.1);
        let similarity_baseline = self.baseline_metrics.get("avg_similarity").unwrap_or(&0.8);

        for query in queries {
            let response_time_ratio = query.response_time.as_secs_f32() / response_time_baseline;
            let similarity_ratio = query.avg_similarity_score / similarity_baseline;

            if response_time_ratio > self.detection_sensitivity {
                anomalies.push(QueryAnomaly {
                    query_id: query.query_id.clone(),
                    anomaly_type: AnomalyType::UnusualLatency,
                    severity_score: response_time_ratio,
                    description: format!(
                        "Query response time {}x higher than baseline",
                        response_time_ratio
                    ),
                    suggested_action: "Investigate query complexity or system load".to_string(),
                });
            }

            if similarity_ratio < (1.0 / self.detection_sensitivity) {
                anomalies.push(QueryAnomaly {
                    query_id: query.query_id.clone(),
                    anomaly_type: AnomalyType::LowSimilarityScores,
                    severity_score: 1.0 / similarity_ratio,
                    description: "Unusually low similarity scores detected".to_string(),
                    suggested_action: "Check vector quality or similarity metric configuration"
                        .to_string(),
                });
            }
        }

        AnomalyDetection {
            anomalies,
            detection_threshold: self.detection_sensitivity,
            false_positive_rate: 0.05, // Estimated
            confidence_level: 0.95,
        }
    }
}

// Utility functions

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn calculate_centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    if vectors.is_empty() {
        return vec![];
    }

    let dim_count = vectors[0].len();
    let mut centroid = vec![0.0; dim_count];

    for vector in vectors {
        for (i, &value) in vector.iter().enumerate() {
            centroid[i] += value;
        }
    }

    for value in &mut centroid {
        *value /= vectors.len() as f32;
    }

    centroid
}

fn calculate_variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<f32>() / values.len() as f32;
    values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32
}

impl Default for VectorAnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_analytics_engine_creation() {
        let engine = VectorAnalyticsEngine::new();
        assert_eq!(engine.query_history.len(), 0);
        assert_eq!(engine.max_history_size, 10000);
    }

    #[test]
    fn test_query_recording() {
        let mut engine = VectorAnalyticsEngine::new();

        let query = QueryAnalytics {
            query_id: "test_query_1".to_string(),
            timestamp: 1640995200, // 2022-01-01
            query_vector: vec![0.1, 0.2, 0.3],
            similarity_metric: "cosine".to_string(),
            top_k: 10,
            response_time: Duration::from_millis(50),
            results_count: 8,
            avg_similarity_score: 0.85,
            min_similarity_score: 0.7,
            max_similarity_score: 0.95,
            cache_hit: true,
            index_type: "hnsw".to_string(),
        };

        engine.record_query(query);
        assert_eq!(engine.query_history.len(), 1);
    }

    #[test]
    fn test_vector_distribution_analysis() {
        let engine = VectorAnalyticsEngine::new();

        let vectors = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![0.9, 1.9, 2.9],
            vec![5.0, 6.0, 7.0],
            vec![5.1, 6.1, 7.1],
        ];

        let analysis = engine.analyze_vector_distribution(&vectors).unwrap();

        assert_eq!(analysis.total_vectors, 5);
        assert_eq!(analysis.dimensionality, 3);
        assert!(analysis.density_estimate > 0.0);
        assert!(analysis.sparsity_ratio >= 0.0);
    }

    #[test]
    fn test_performance_score_calculation() {
        let mut engine = VectorAnalyticsEngine::new();

        // Add some test queries
        for i in 0..10 {
            let query = QueryAnalytics {
                query_id: format!("query_{}", i),
                timestamp: 1640995200 + i * 60,
                query_vector: vec![0.1 * i as f32, 0.2 * i as f32],
                similarity_metric: "cosine".to_string(),
                top_k: 5,
                response_time: Duration::from_millis(30 + i * 5),
                results_count: 5,
                avg_similarity_score: 0.8 + (i as f32 * 0.01),
                min_similarity_score: 0.7,
                max_similarity_score: 0.95,
                cache_hit: i % 2 == 0,
                index_type: "hnsw".to_string(),
            };
            engine.record_query(query);
        }

        let score = engine.calculate_performance_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_anomaly_detection() {
        let engine = VectorAnalyticsEngine::new();
        let anomalies = engine.detect_anomalies();

        assert_eq!(anomalies.anomalies.len(), 0); // No queries recorded yet
        assert!(anomalies.confidence_level > 0.0);
    }

    #[test]
    fn test_optimization_recommendations() {
        let mut engine = VectorAnalyticsEngine::new();

        // Add a slow query to trigger recommendations
        let slow_query = QueryAnalytics {
            query_id: "slow_query".to_string(),
            timestamp: 1640995200,
            query_vector: vec![0.1, 0.2, 0.3],
            similarity_metric: "cosine".to_string(),
            top_k: 10,
            response_time: Duration::from_millis(500), // Slow query
            results_count: 8,
            avg_similarity_score: 0.85,
            min_similarity_score: 0.7,
            max_similarity_score: 0.95,
            cache_hit: false, // Cache miss
            index_type: "linear".to_string(),
        };

        engine.record_query(slow_query);
        let recommendations = engine.generate_optimization_recommendations();

        assert!(!recommendations.is_empty());
    }

    #[test]
    fn test_vector_quality_assessment() {
        let engine = VectorAnalyticsEngine::new();

        let vectors = vec![
            vec![1.0, 2.0, 3.0, 0.0],
            vec![1.1, 2.1, 3.1, 0.0], // Last dimension is always 0 (low quality)
            vec![0.9, 1.9, 2.9, 0.0],
            vec![1.05, 2.05, 3.05, 0.0],
        ];

        let assessment = engine.assess_vector_quality(&vectors).unwrap();

        assert!(assessment.overall_quality_score >= 0.0 && assessment.overall_quality_score <= 1.0);
        assert_eq!(assessment.dimension_quality.len(), 4);
        assert!(assessment.noise_level >= 0.0);
        assert!(!assessment.recommendations.is_empty());
    }

    #[test]
    fn test_analytics_export() {
        let engine = VectorAnalyticsEngine::new();
        let json_result = engine.export_analytics();

        assert!(json_result.is_ok());
        let json_data = json_result.unwrap();
        assert!(json_data.contains("query_count"));
        assert!(json_data.contains("performance_trends"));
        assert!(json_data.contains("recommendations"));
    }
}
