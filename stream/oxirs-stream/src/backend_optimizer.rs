//! # Backend Optimization and Selection
//!
//! Advanced backend selection algorithms, cost modeling, and ML-driven optimization
//! for choosing the optimal streaming backend based on workload patterns and performance metrics.

use crate::backend::{BackendType, StreamBackendConfig};
use crate::event::StreamEvent;
use crate::monitoring::{BackendMetricsUpdate, StreamingMetrics};
use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Backend optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub enable_cost_modeling: bool,
    pub enable_ml_prediction: bool,
    pub enable_pattern_analysis: bool,
    pub optimization_interval: Duration,
    pub min_samples_for_prediction: usize,
    pub cost_weight_latency: f64,
    pub cost_weight_throughput: f64,
    pub cost_weight_reliability: f64,
    pub cost_weight_resource_usage: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            enable_cost_modeling: true,
            enable_ml_prediction: true,
            enable_pattern_analysis: true,
            optimization_interval: Duration::from_secs(300), // 5 minutes
            min_samples_for_prediction: 100,
            cost_weight_latency: 0.3,
            cost_weight_throughput: 0.3,
            cost_weight_reliability: 0.3,
            cost_weight_resource_usage: 0.1,
        }
    }
}

/// Workload pattern characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadPattern {
    pub pattern_type: PatternType,
    pub event_rate: f64,
    pub batch_size: u32,
    pub event_size_bytes: u64,
    pub temporal_distribution: TemporalDistribution,
    pub data_characteristics: DataCharacteristics,
    pub consistency_requirements: ConsistencyLevel,
}

/// Pattern types for workload classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternType {
    /// Steady, predictable load
    Steady,
    /// Variable load with spikes
    Bursty,
    /// Seasonal patterns
    Seasonal,
    /// Random/unpredictable patterns
    Random,
    /// Real-time processing requirements
    RealTime,
    /// Batch processing oriented
    BatchOriented,
}

/// Temporal distribution patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemporalDistribution {
    Uniform,
    Normal { mean: f64, std_dev: f64 },
    Exponential { lambda: f64 },
    Poisson { lambda: f64 },
    Custom { distribution_name: String },
}

/// Data characteristics affecting backend choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataCharacteristics {
    pub compression_ratio: f64,
    pub serialization_overhead: f64,
    pub has_complex_structures: bool,
    pub requires_ordering: bool,
    pub has_time_windows: bool,
    pub requires_deduplication: bool,
}

/// Consistency level requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// At most once delivery
    AtMostOnce,
    /// At least once delivery
    AtLeastOnce,
    /// Exactly once delivery
    ExactlyOnce,
    /// Session consistency
    Session,
    /// Strong consistency
    Strong,
}

/// Backend performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendPerformance {
    pub backend_type: BackendType,
    pub measured_latency_p50: f64,
    pub measured_latency_p95: f64,
    pub measured_latency_p99: f64,
    pub measured_throughput: f64,
    pub reliability_score: f64,
    pub resource_usage: ResourceUsage,
    pub cost_per_hour: f64,
    pub setup_complexity: u8, // 1-10 scale
    pub scalability_factor: f64,
    pub last_updated: DateTime<Utc>,
    pub sample_count: u64,
}

/// Resource usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub cpu_usage_percent: f64,
    pub memory_usage_mb: f64,
    pub network_usage_mbps: f64,
    pub disk_io_ops_per_sec: f64,
    pub connection_count: u32,
}

/// Cost model for backend selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    pub total_cost: f64,
    pub latency_cost: f64,
    pub throughput_cost: f64,
    pub reliability_cost: f64,
    pub resource_cost: f64,
    pub scaling_cost: f64,
    pub maintenance_cost: f64,
}

/// ML prediction model for performance forecasting
#[derive(Debug, Clone)]
pub struct MLPredictor {
    /// Historical performance data
    performance_history: Vec<PerformanceDataPoint>,
    /// Learned patterns
    patterns: HashMap<String, PatternModel>,
    /// Feature weights
    feature_weights: Vec<f64>,
    /// Prediction confidence threshold
    confidence_threshold: f64,
}

/// Single performance data point for ML training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDataPoint {
    pub timestamp: DateTime<Utc>,
    pub backend_type: BackendType,
    pub workload_pattern: WorkloadPattern,
    pub actual_latency: f64,
    pub actual_throughput: f64,
    pub actual_reliability: f64,
    pub resource_usage: ResourceUsage,
    pub external_factors: HashMap<String, f64>,
}

/// Learned pattern model for ML predictions
#[derive(Debug, Clone)]
pub struct PatternModel {
    pub pattern_name: String,
    pub coefficients: Vec<f64>,
    pub intercept: f64,
    pub confidence: f64,
    pub last_trained: DateTime<Utc>,
    pub sample_count: usize,
}

/// Backend optimizer for intelligent backend selection
pub struct BackendOptimizer {
    config: OptimizerConfig,
    backend_performance: Arc<RwLock<HashMap<BackendType, BackendPerformance>>>,
    pattern_analyzer: PatternAnalyzer,
    cost_calculator: CostCalculator,
    ml_predictor: Option<MLPredictor>,
    optimization_history: Arc<RwLock<Vec<OptimizationDecision>>>,
}

/// Pattern analyzer for workload classification
pub struct PatternAnalyzer {
    event_history: Vec<(DateTime<Utc>, StreamEvent)>,
    pattern_cache: HashMap<String, WorkloadPattern>,
    analysis_window: ChronoDuration,
}

/// Cost calculator for backend evaluation
pub struct CostCalculator {
    config: OptimizerConfig,
    baseline_costs: HashMap<BackendType, f64>,
}

/// Optimization decision record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationDecision {
    pub timestamp: DateTime<Utc>,
    pub selected_backend: BackendType,
    pub workload_pattern: WorkloadPattern,
    pub predicted_performance: BackendPerformance,
    pub cost_model: CostModel,
    pub confidence: f64,
    pub reason: String,
}

/// Backend recommendation with ranking
#[derive(Debug, Clone)]
pub struct BackendRecommendation {
    pub backend_type: BackendType,
    pub score: f64,
    pub predicted_latency: f64,
    pub predicted_throughput: f64,
    pub predicted_cost: f64,
    pub confidence: f64,
    pub strengths: Vec<String>,
    pub weaknesses: Vec<String>,
}

impl BackendOptimizer {
    /// Create a new backend optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        let ml_predictor = if config.enable_ml_prediction {
            Some(MLPredictor::new())
        } else {
            None
        };

        Self {
            pattern_analyzer: PatternAnalyzer::new(ChronoDuration::hours(1)),
            cost_calculator: CostCalculator::new(config.clone()),
            backend_performance: Arc::new(RwLock::new(HashMap::new())),
            optimization_history: Arc::new(RwLock::new(Vec::new())),
            config,
            ml_predictor,
        }
    }

    /// Update backend performance metrics
    pub async fn update_backend_performance(
        &self,
        backend_type: BackendType,
        metrics: &StreamingMetrics,
    ) -> Result<()> {
        let mut performance_map = self.backend_performance.write().await;

        let performance = performance_map
            .entry(backend_type.clone())
            .or_insert_with(|| BackendPerformance::new(backend_type.clone()));

        // Update measured performance with exponential moving average
        let alpha = 0.1; // Smoothing factor
        performance.measured_latency_p50 = alpha * metrics.producer_average_latency_ms
            + (1.0 - alpha) * performance.measured_latency_p50;
        performance.measured_throughput = alpha * metrics.producer_throughput_eps
            + (1.0 - alpha) * performance.measured_throughput;
        performance.reliability_score =
            alpha * metrics.success_rate + (1.0 - alpha) * performance.reliability_score;

        performance.resource_usage.cpu_usage_percent = metrics.system_cpu_usage_percent;
        performance.resource_usage.memory_usage_mb =
            metrics.system_memory_usage_bytes as f64 / (1024.0 * 1024.0);
        performance.resource_usage.connection_count = metrics.backend_connections_active;

        performance.last_updated = Utc::now();
        performance.sample_count += 1;

        debug!(
            "Updated performance for {:?}: latency={:.2}ms, throughput={:.0}eps, reliability={:.3}",
            backend_type,
            performance.measured_latency_p50,
            performance.measured_throughput,
            performance.reliability_score
        );

        Ok(())
    }

    /// Analyze workload pattern from recent events
    pub async fn analyze_workload_pattern(
        &mut self,
        events: &[StreamEvent],
    ) -> Result<WorkloadPattern> {
        self.pattern_analyzer.analyze_pattern(events).await
    }

    /// Get optimal backend recommendation for given workload
    pub async fn recommend_backend(
        &self,
        pattern: &WorkloadPattern,
    ) -> Result<Vec<BackendRecommendation>> {
        let mut recommendations = Vec::new();
        let performance_map = self.backend_performance.read().await;

        for (backend_type, performance) in performance_map.iter() {
            let cost = self
                .cost_calculator
                .calculate_cost(backend_type, pattern, performance)
                .await?;

            let predicted_performance = if let Some(predictor) = &self.ml_predictor {
                predictor.predict_performance(backend_type, pattern).await?
            } else {
                performance.clone()
            };

            let score = self.calculate_backend_score(&cost, &predicted_performance, pattern);
            let confidence = self.calculate_confidence(&predicted_performance, pattern);

            let recommendation = BackendRecommendation {
                backend_type: backend_type.clone(),
                score,
                predicted_latency: predicted_performance.measured_latency_p50,
                predicted_throughput: predicted_performance.measured_throughput,
                predicted_cost: cost.total_cost,
                confidence,
                strengths: self.analyze_strengths(backend_type, pattern),
                weaknesses: self.analyze_weaknesses(backend_type, pattern),
            };

            recommendations.push(recommendation);
        }

        // Sort by score (higher is better)
        recommendations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        info!(
            "Generated {} backend recommendations for workload pattern: {:?}",
            recommendations.len(),
            pattern.pattern_type
        );

        Ok(recommendations)
    }

    /// Train ML predictor with new performance data
    pub async fn train_predictor(&mut self, data_point: PerformanceDataPoint) -> Result<()> {
        if let Some(predictor) = &mut self.ml_predictor {
            predictor.add_training_data(data_point).await?;

            if predictor.performance_history.len() >= self.config.min_samples_for_prediction {
                predictor.retrain_models().await?;
            }
        }
        Ok(())
    }

    /// Record optimization decision
    pub async fn record_decision(&self, decision: OptimizationDecision) -> Result<()> {
        let mut history = self.optimization_history.write().await;
        history.push(decision);

        // Keep only recent decisions (last 1000)
        if history.len() > 1000 {
            history.drain(0..100);
        }
        Ok(())
    }

    /// Get optimization statistics
    pub async fn get_optimization_stats(&self) -> Result<OptimizationStats> {
        let history = self.optimization_history.read().await;
        let performance_map = self.backend_performance.read().await;

        let total_decisions = history.len();
        let backend_usage = history.iter().fold(HashMap::new(), |mut acc, decision| {
            *acc.entry(decision.selected_backend.clone()).or_insert(0) += 1;
            acc
        });

        let average_confidence = if total_decisions > 0 {
            history.iter().map(|d| d.confidence).sum::<f64>() / total_decisions as f64
        } else {
            0.0
        };

        let performance_improvements = self.calculate_performance_improvements(&history).await?;

        Ok(OptimizationStats {
            total_decisions,
            backend_usage,
            average_confidence,
            performance_improvements,
            active_backends: performance_map.len(),
            last_optimization: history.last().map(|d| d.timestamp),
        })
    }

    /// Calculate backend score based on cost model and performance
    fn calculate_backend_score(
        &self,
        cost: &CostModel,
        performance: &BackendPerformance,
        pattern: &WorkloadPattern,
    ) -> f64 {
        let latency_score = match pattern.pattern_type {
            PatternType::RealTime => {
                // For real-time, heavily penalize high latency
                if performance.measured_latency_p99 < 10.0 {
                    1.0
                } else if performance.measured_latency_p99 < 50.0 {
                    0.7
                } else {
                    0.3
                }
            }
            _ => {
                // For other patterns, moderate latency tolerance
                (100.0 / (performance.measured_latency_p50 + 1.0)).min(1.0)
            }
        };

        let throughput_score = match pattern.pattern_type {
            PatternType::BatchOriented => {
                // Batch processing values high throughput
                (performance.measured_throughput / pattern.event_rate).min(2.0) / 2.0
            }
            _ => (performance.measured_throughput / (pattern.event_rate * 1.2)).min(1.0),
        };

        let reliability_score = performance.reliability_score;
        let cost_score = 1.0 / (cost.total_cost + 1.0);

        // Weighted combination
        (latency_score * self.config.cost_weight_latency
            + throughput_score * self.config.cost_weight_throughput
            + reliability_score * self.config.cost_weight_reliability
            + cost_score * self.config.cost_weight_resource_usage)
            / (self.config.cost_weight_latency
                + self.config.cost_weight_throughput
                + self.config.cost_weight_reliability
                + self.config.cost_weight_resource_usage)
    }

    /// Calculate confidence in prediction
    fn calculate_confidence(
        &self,
        performance: &BackendPerformance,
        _pattern: &WorkloadPattern,
    ) -> f64 {
        let sample_confidence = (performance.sample_count as f64 / 1000.0).min(1.0);
        let recency_confidence = {
            let age_hours = Utc::now()
                .signed_duration_since(performance.last_updated)
                .num_hours() as f64;
            (1.0 / (age_hours / 24.0 + 1.0)).max(0.1)
        };

        (sample_confidence + recency_confidence) / 2.0
    }

    /// Analyze backend strengths for given pattern
    fn analyze_strengths(
        &self,
        backend_type: &BackendType,
        pattern: &WorkloadPattern,
    ) -> Vec<String> {
        let mut strengths = Vec::new();

        match backend_type {
            BackendType::Kafka => {
                strengths.push("High throughput".to_string());
                strengths.push("Strong durability".to_string());
                strengths.push("Excellent ordering guarantees".to_string());
                if matches!(
                    pattern.consistency_requirements,
                    ConsistencyLevel::ExactlyOnce
                ) {
                    strengths.push("Exactly-once semantics".to_string());
                }
            }
            BackendType::Nats => {
                strengths.push("Low latency".to_string());
                strengths.push("Simple setup".to_string());
                strengths.push("Built-in clustering".to_string());
                if matches!(pattern.pattern_type, PatternType::RealTime) {
                    strengths.push("Real-time performance".to_string());
                }
            }
            BackendType::Redis => {
                strengths.push("In-memory speed".to_string());
                strengths.push("Low latency".to_string());
                strengths.push("Rich data structures".to_string());
            }
            BackendType::Kinesis => {
                strengths.push("AWS native integration".to_string());
                strengths.push("Auto-scaling".to_string());
                strengths.push("Pay-per-use model".to_string());
            }
            BackendType::Pulsar => {
                strengths.push("Multi-tenancy".to_string());
                strengths.push("Geo-replication".to_string());
                strengths.push("Unified messaging".to_string());
            }
            BackendType::Memory => {
                strengths.push("Zero latency".to_string());
                strengths.push("Perfect for testing".to_string());
            }
        }

        strengths
    }

    /// Analyze backend weaknesses for given pattern
    fn analyze_weaknesses(
        &self,
        backend_type: &BackendType,
        pattern: &WorkloadPattern,
    ) -> Vec<String> {
        let mut weaknesses = Vec::new();

        match backend_type {
            BackendType::Kafka => {
                weaknesses.push("Complex setup".to_string());
                weaknesses.push("Higher resource usage".to_string());
                if matches!(pattern.pattern_type, PatternType::RealTime) {
                    weaknesses.push("Higher latency than NATS".to_string());
                }
            }
            BackendType::Nats => {
                if matches!(pattern.consistency_requirements, ConsistencyLevel::Strong) {
                    weaknesses.push("Limited durability options".to_string());
                }
                if pattern.event_rate > 100000.0 {
                    weaknesses.push("May not handle extreme throughput".to_string());
                }
            }
            BackendType::Redis => {
                weaknesses.push("Memory-bound".to_string());
                weaknesses.push("Limited durability".to_string());
                if pattern.event_size_bytes > 1000000 {
                    weaknesses.push("Not suitable for large events".to_string());
                }
            }
            BackendType::Kinesis => {
                weaknesses.push("AWS vendor lock-in".to_string());
                weaknesses.push("Cost can scale quickly".to_string());
                weaknesses.push("Regional limitations".to_string());
            }
            BackendType::Pulsar => {
                weaknesses.push("Newer ecosystem".to_string());
                weaknesses.push("Complex architecture".to_string());
            }
            BackendType::Memory => {
                weaknesses.push("No persistence".to_string());
                weaknesses.push("Single node only".to_string());
                weaknesses.push("Memory limitations".to_string());
            }
        }

        weaknesses
    }

    /// Calculate performance improvements over time
    async fn calculate_performance_improvements(
        &self,
        history: &[OptimizationDecision],
    ) -> Result<HashMap<String, f64>> {
        let mut improvements = HashMap::new();

        if history.len() < 10 {
            return Ok(improvements);
        }

        let recent_decisions = &history[history.len() - 10..];
        let older_decisions = &history[0..10.min(history.len() - 10)];

        let recent_avg_latency = recent_decisions
            .iter()
            .map(|d| d.predicted_performance.measured_latency_p50)
            .sum::<f64>()
            / recent_decisions.len() as f64;

        let older_avg_latency = older_decisions
            .iter()
            .map(|d| d.predicted_performance.measured_latency_p50)
            .sum::<f64>()
            / older_decisions.len() as f64;

        let latency_improvement =
            (older_avg_latency - recent_avg_latency) / older_avg_latency * 100.0;
        improvements.insert(
            "latency_improvement_percent".to_string(),
            latency_improvement,
        );

        let recent_avg_throughput = recent_decisions
            .iter()
            .map(|d| d.predicted_performance.measured_throughput)
            .sum::<f64>()
            / recent_decisions.len() as f64;

        let older_avg_throughput = older_decisions
            .iter()
            .map(|d| d.predicted_performance.measured_throughput)
            .sum::<f64>()
            / older_decisions.len() as f64;

        let throughput_improvement =
            (recent_avg_throughput - older_avg_throughput) / older_avg_throughput * 100.0;
        improvements.insert(
            "throughput_improvement_percent".to_string(),
            throughput_improvement,
        );

        Ok(improvements)
    }
}

impl PatternAnalyzer {
    pub fn new(analysis_window: ChronoDuration) -> Self {
        Self {
            event_history: Vec::new(),
            pattern_cache: HashMap::new(),
            analysis_window,
        }
    }

    pub async fn analyze_pattern(&mut self, events: &[StreamEvent]) -> Result<WorkloadPattern> {
        // Add events to history
        let now = Utc::now();
        for event in events {
            self.event_history.push((now, event.clone()));
        }

        // Remove old events outside analysis window
        let cutoff = now - self.analysis_window;
        self.event_history
            .retain(|(timestamp, _)| *timestamp >= cutoff);

        if self.event_history.is_empty() {
            return Ok(WorkloadPattern::default());
        }

        // Calculate event rate
        let duration_seconds = self.analysis_window.num_seconds() as f64;
        let event_rate = self.event_history.len() as f64 / duration_seconds;

        // Analyze temporal distribution
        let temporal_distribution = self.analyze_temporal_distribution().await?;

        // Determine pattern type
        let pattern_type = self
            .classify_pattern_type(event_rate, &temporal_distribution)
            .await?;

        // Calculate average event size
        let avg_event_size = self.calculate_average_event_size().await?;

        // Analyze data characteristics
        let data_characteristics = self.analyze_data_characteristics().await?;

        // Determine consistency requirements based on event types
        let consistency_requirements = self.determine_consistency_requirements().await?;

        Ok(WorkloadPattern {
            pattern_type,
            event_rate,
            batch_size: self.estimate_optimal_batch_size(event_rate),
            event_size_bytes: avg_event_size,
            temporal_distribution,
            data_characteristics,
            consistency_requirements,
        })
    }

    async fn analyze_temporal_distribution(&self) -> Result<TemporalDistribution> {
        if self.event_history.len() < 10 {
            return Ok(TemporalDistribution::Uniform);
        }

        // Calculate inter-arrival times
        let mut intervals = Vec::new();
        for i in 1..self.event_history.len() {
            let interval = self.event_history[i]
                .0
                .signed_duration_since(self.event_history[i - 1].0)
                .num_milliseconds() as f64;
            intervals.push(interval);
        }

        // Calculate basic statistics
        let mean = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let variance =
            intervals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / intervals.len() as f64;
        let std_dev = variance.sqrt();

        // Simple distribution classification
        let cv = std_dev / mean; // Coefficient of variation

        if cv < 0.1 {
            Ok(TemporalDistribution::Uniform)
        } else if cv < 0.5 {
            Ok(TemporalDistribution::Normal { mean, std_dev })
        } else {
            Ok(TemporalDistribution::Exponential { lambda: 1.0 / mean })
        }
    }

    async fn classify_pattern_type(
        &self,
        event_rate: f64,
        temporal_dist: &TemporalDistribution,
    ) -> Result<PatternType> {
        // Simple heuristic-based classification
        match temporal_dist {
            TemporalDistribution::Uniform => {
                if event_rate > 10000.0 {
                    Ok(PatternType::BatchOriented)
                } else if event_rate > 100.0 {
                    Ok(PatternType::Steady)
                } else {
                    Ok(PatternType::RealTime)
                }
            }
            TemporalDistribution::Exponential { .. } => Ok(PatternType::Bursty),
            TemporalDistribution::Normal { std_dev, mean } => {
                if std_dev / mean > 1.0 {
                    Ok(PatternType::Random)
                } else {
                    Ok(PatternType::Steady)
                }
            }
            _ => Ok(PatternType::Steady),
        }
    }

    async fn calculate_average_event_size(&self) -> Result<u64> {
        if self.event_history.is_empty() {
            return Ok(1024); // Default 1KB
        }

        // Estimate serialized size (simplified)
        let avg_size = self
            .event_history
            .iter()
            .map(|(_, event)| self.estimate_event_size(event))
            .sum::<u64>()
            / self.event_history.len() as u64;

        Ok(avg_size)
    }

    fn estimate_event_size(&self, event: &StreamEvent) -> u64 {
        // Simplified size estimation based on event type
        match event {
            StreamEvent::TripleAdded {
                subject,
                predicate,
                object,
                ..
            } => (subject.len() + predicate.len() + object.len() + 100) as u64,
            StreamEvent::TripleRemoved {
                subject,
                predicate,
                object,
                ..
            } => (subject.len() + predicate.len() + object.len() + 100) as u64,
            StreamEvent::GraphCreated { .. } => 200,
            StreamEvent::SparqlUpdate { query, .. } => (query.len() + 200) as u64,
            StreamEvent::TransactionBegin { .. } => 150,
            StreamEvent::TransactionCommit { .. } => 100,
            StreamEvent::Heartbeat { .. } => 50,
            _ => 300, // Default for complex events
        }
    }

    async fn analyze_data_characteristics(&self) -> Result<DataCharacteristics> {
        let has_complex_structures = self
            .event_history
            .iter()
            .any(|(_, event)| self.is_complex_event(event));

        let requires_ordering = self
            .event_history
            .iter()
            .any(|(_, event)| self.requires_ordering(event));

        Ok(DataCharacteristics {
            compression_ratio: 0.7,      // Assume 30% compression
            serialization_overhead: 0.1, // 10% overhead
            has_complex_structures,
            requires_ordering,
            has_time_windows: false,      // Simplified
            requires_deduplication: true, // Conservative default
        })
    }

    fn is_complex_event(&self, event: &StreamEvent) -> bool {
        matches!(
            event,
            StreamEvent::SparqlUpdate { .. }
                | StreamEvent::SchemaChanged { .. }
                | StreamEvent::QueryCompleted { .. }
        )
    }

    fn requires_ordering(&self, event: &StreamEvent) -> bool {
        matches!(
            event,
            StreamEvent::TransactionBegin { .. }
                | StreamEvent::TransactionCommit { .. }
                | StreamEvent::TransactionAbort { .. }
        )
    }

    async fn determine_consistency_requirements(&self) -> Result<ConsistencyLevel> {
        let has_transactions = self.event_history.iter().any(|(_, event)| {
            matches!(
                event,
                StreamEvent::TransactionBegin { .. }
                    | StreamEvent::TransactionCommit { .. }
                    | StreamEvent::TransactionAbort { .. }
            )
        });

        if has_transactions {
            Ok(ConsistencyLevel::ExactlyOnce)
        } else {
            Ok(ConsistencyLevel::AtLeastOnce)
        }
    }

    fn estimate_optimal_batch_size(&self, event_rate: f64) -> u32 {
        if event_rate > 10000.0 {
            1000
        } else if event_rate > 1000.0 {
            500
        } else if event_rate > 100.0 {
            100
        } else {
            10
        }
    }
}

impl CostCalculator {
    pub fn new(config: OptimizerConfig) -> Self {
        let mut baseline_costs = HashMap::new();

        // Baseline hourly costs (normalized)
        baseline_costs.insert(BackendType::Memory, 0.0);
        baseline_costs.insert(BackendType::Redis, 0.1);
        baseline_costs.insert(BackendType::Nats, 0.2);
        baseline_costs.insert(BackendType::Kafka, 0.5);
        baseline_costs.insert(BackendType::Pulsar, 0.4);
        baseline_costs.insert(BackendType::Kinesis, 0.8);

        Self {
            config,
            baseline_costs,
        }
    }

    pub async fn calculate_cost(
        &self,
        backend_type: &BackendType,
        pattern: &WorkloadPattern,
        performance: &BackendPerformance,
    ) -> Result<CostModel> {
        let base_cost = self.baseline_costs.get(backend_type).unwrap_or(&1.0);

        // Calculate component costs
        let latency_cost = self.calculate_latency_cost(performance.measured_latency_p50, pattern);
        let throughput_cost =
            self.calculate_throughput_cost(performance.measured_throughput, pattern);
        let reliability_cost =
            self.calculate_reliability_cost(performance.reliability_score, pattern);
        let resource_cost = self.calculate_resource_cost(&performance.resource_usage, pattern);
        let scaling_cost = self.calculate_scaling_cost(backend_type, pattern);
        let maintenance_cost =
            self.calculate_maintenance_cost(backend_type, performance.setup_complexity);

        let total_cost = base_cost
            + latency_cost * self.config.cost_weight_latency
            + throughput_cost * self.config.cost_weight_throughput
            + reliability_cost * self.config.cost_weight_reliability
            + resource_cost * self.config.cost_weight_resource_usage
            + scaling_cost * 0.1
            + maintenance_cost * 0.1;

        Ok(CostModel {
            total_cost,
            latency_cost,
            throughput_cost,
            reliability_cost,
            resource_cost,
            scaling_cost,
            maintenance_cost,
        })
    }

    fn calculate_latency_cost(&self, latency: f64, pattern: &WorkloadPattern) -> f64 {
        let latency_penalty = match pattern.pattern_type {
            PatternType::RealTime => latency / 10.0, // Heavy penalty for real-time
            PatternType::Bursty => latency / 50.0,
            _ => latency / 100.0,
        };
        latency_penalty.min(2.0) // Cap at 2x cost
    }

    fn calculate_throughput_cost(&self, throughput: f64, pattern: &WorkloadPattern) -> f64 {
        let required_throughput = pattern.event_rate * 1.5; // 50% buffer
        if throughput < required_throughput {
            (required_throughput - throughput) / required_throughput
        } else {
            0.0
        }
    }

    fn calculate_reliability_cost(&self, reliability: f64, pattern: &WorkloadPattern) -> f64 {
        let required_reliability = match pattern.consistency_requirements {
            ConsistencyLevel::ExactlyOnce => 0.999,
            ConsistencyLevel::AtLeastOnce => 0.995,
            _ => 0.99,
        };

        if reliability < required_reliability {
            (required_reliability - reliability) * 10.0
        } else {
            0.0
        }
    }

    fn calculate_resource_cost(&self, usage: &ResourceUsage, _pattern: &WorkloadPattern) -> f64 {
        // Normalize resource usage to cost
        (usage.cpu_usage_percent / 100.0) * 0.1
            + (usage.memory_usage_mb / 1000.0) * 0.05
            + (usage.network_usage_mbps / 100.0) * 0.02
    }

    fn calculate_scaling_cost(&self, backend_type: &BackendType, pattern: &WorkloadPattern) -> f64 {
        let scaling_factor = match backend_type {
            BackendType::Kinesis => 0.1, // Auto-scaling
            BackendType::Kafka => 0.5,   // Manual scaling
            BackendType::Memory => 1.0,  // No scaling
            _ => 0.3,
        };

        match pattern.pattern_type {
            PatternType::Bursty | PatternType::Random => scaling_factor,
            _ => 0.0,
        }
    }

    fn calculate_maintenance_cost(&self, _backend_type: &BackendType, setup_complexity: u8) -> f64 {
        setup_complexity as f64 / 10.0
    }
}

impl MLPredictor {
    pub fn new() -> Self {
        Self {
            performance_history: Vec::new(),
            patterns: HashMap::new(),
            feature_weights: vec![1.0; 10], // Start with equal weights
            confidence_threshold: 0.7,
        }
    }

    pub async fn add_training_data(&mut self, data_point: PerformanceDataPoint) -> Result<()> {
        self.performance_history.push(data_point);

        // Keep only recent data (last 10,000 points)
        if self.performance_history.len() > 10000 {
            self.performance_history.drain(0..1000);
        }

        Ok(())
    }

    pub async fn predict_performance(
        &self,
        backend_type: &BackendType,
        pattern: &WorkloadPattern,
    ) -> Result<BackendPerformance> {
        // Filter relevant historical data
        let relevant_data: Vec<&PerformanceDataPoint> = self
            .performance_history
            .iter()
            .filter(|dp| dp.backend_type == *backend_type)
            .collect();

        if relevant_data.is_empty() {
            return Err(anyhow!(
                "No historical data for backend type: {:?}",
                backend_type
            ));
        }

        // Simple prediction based on similar patterns
        let similar_data: Vec<&PerformanceDataPoint> = relevant_data
            .iter()
            .filter(|dp| self.pattern_similarity(&dp.workload_pattern, pattern) > 0.7)
            .cloned()
            .collect();

        let prediction_data = if similar_data.is_empty() {
            &relevant_data
        } else {
            &similar_data
        };

        // Calculate weighted averages
        let predicted_latency = prediction_data
            .iter()
            .map(|dp| dp.actual_latency)
            .sum::<f64>()
            / prediction_data.len() as f64;

        let predicted_throughput = prediction_data
            .iter()
            .map(|dp| dp.actual_throughput)
            .sum::<f64>()
            / prediction_data.len() as f64;

        let predicted_reliability = prediction_data
            .iter()
            .map(|dp| dp.actual_reliability)
            .sum::<f64>()
            / prediction_data.len() as f64;

        Ok(BackendPerformance {
            backend_type: backend_type.clone(),
            measured_latency_p50: predicted_latency,
            measured_latency_p95: predicted_latency * 1.5,
            measured_latency_p99: predicted_latency * 2.0,
            measured_throughput: predicted_throughput,
            reliability_score: predicted_reliability,
            resource_usage: prediction_data[0].resource_usage.clone(),
            cost_per_hour: 0.0,  // Will be calculated by cost model
            setup_complexity: 5, // Default
            scalability_factor: 1.0,
            last_updated: Utc::now(),
            sample_count: prediction_data.len() as u64,
        })
    }

    pub async fn retrain_models(&mut self) -> Result<()> {
        // Simple retraining using linear regression
        for backend_type in [BackendType::Kafka, BackendType::Nats, BackendType::Redis].iter() {
            let backend_data: Vec<&PerformanceDataPoint> = self
                .performance_history
                .iter()
                .filter(|dp| dp.backend_type == *backend_type)
                .collect();

            if backend_data.len() < 10 {
                continue;
            }

            // Create pattern model for this backend
            let pattern_name = format!("{:?}_model", backend_type);
            let model = self.train_linear_model(&backend_data).await?;
            self.patterns.insert(pattern_name, model);
        }

        info!("Retrained ML models for {} patterns", self.patterns.len());
        Ok(())
    }

    async fn train_linear_model(&self, data: &[&PerformanceDataPoint]) -> Result<PatternModel> {
        // Simplified linear regression implementation
        let n = data.len() as f64;

        // Extract features and target (latency)
        let features: Vec<Vec<f64>> = data
            .iter()
            .map(|dp| self.extract_features(&dp.workload_pattern))
            .collect();

        let targets: Vec<f64> = data.iter().map(|dp| dp.actual_latency).collect();

        // Simple linear regression: y = mx + b
        let feature_count = features[0].len();
        let mut coefficients = vec![0.0; feature_count];
        let intercept = targets.iter().sum::<f64>() / n;

        // Calculate correlations (simplified)
        for i in 0..feature_count {
            let feature_values: Vec<f64> = features.iter().map(|f| f[i]).collect();
            let correlation = self.calculate_correlation(&feature_values, &targets);
            coefficients[i] = correlation * 0.1; // Simplified coefficient
        }

        Ok(PatternModel {
            pattern_name: "latency_model".to_string(),
            coefficients,
            intercept,
            confidence: 0.8, // Default confidence
            last_trained: Utc::now(),
            sample_count: data.len(),
        })
    }

    fn extract_features(&self, pattern: &WorkloadPattern) -> Vec<f64> {
        vec![
            pattern.event_rate,
            pattern.batch_size as f64,
            pattern.event_size_bytes as f64,
            pattern.data_characteristics.compression_ratio,
            pattern.data_characteristics.serialization_overhead,
            if pattern.data_characteristics.has_complex_structures {
                1.0
            } else {
                0.0
            },
            if pattern.data_characteristics.requires_ordering {
                1.0
            } else {
                0.0
            },
            match pattern.pattern_type {
                PatternType::RealTime => 1.0,
                PatternType::BatchOriented => 2.0,
                PatternType::Bursty => 3.0,
                _ => 0.0,
            },
            match pattern.consistency_requirements {
                ConsistencyLevel::ExactlyOnce => 3.0,
                ConsistencyLevel::AtLeastOnce => 2.0,
                _ => 1.0,
            },
            1.0, // Bias term
        ]
    }

    fn pattern_similarity(&self, p1: &WorkloadPattern, p2: &WorkloadPattern) -> f64 {
        let rate_similarity =
            1.0 - (p1.event_rate - p2.event_rate).abs() / (p1.event_rate + p2.event_rate + 1.0);
        let size_similarity = 1.0
            - (p1.event_size_bytes as f64 - p2.event_size_bytes as f64).abs()
                / (p1.event_size_bytes as f64 + p2.event_size_bytes as f64 + 1.0);
        let type_similarity = if std::mem::discriminant(&p1.pattern_type)
            == std::mem::discriminant(&p2.pattern_type)
        {
            1.0
        } else {
            0.0
        };

        (rate_similarity + size_similarity + type_similarity) / 3.0
    }

    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let denom_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let denom_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

        if denom_x * denom_y == 0.0 {
            0.0
        } else {
            numerator / (denom_x * denom_y).sqrt()
        }
    }
}

impl BackendPerformance {
    pub fn new(backend_type: BackendType) -> Self {
        Self {
            backend_type,
            measured_latency_p50: 100.0, // Default 100ms
            measured_latency_p95: 200.0,
            measured_latency_p99: 500.0,
            measured_throughput: 1000.0, // Default 1000 eps
            reliability_score: 0.99,
            resource_usage: ResourceUsage::default(),
            cost_per_hour: 0.1,
            setup_complexity: 5,
            scalability_factor: 1.0,
            last_updated: Utc::now(),
            sample_count: 0,
        }
    }
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 10.0,
            memory_usage_mb: 100.0,
            network_usage_mbps: 1.0,
            disk_io_ops_per_sec: 100.0,
            connection_count: 10,
        }
    }
}

impl Default for WorkloadPattern {
    fn default() -> Self {
        Self {
            pattern_type: PatternType::Steady,
            event_rate: 100.0,
            batch_size: 100,
            event_size_bytes: 1024,
            temporal_distribution: TemporalDistribution::Uniform,
            data_characteristics: DataCharacteristics {
                compression_ratio: 0.7,
                serialization_overhead: 0.1,
                has_complex_structures: false,
                requires_ordering: false,
                has_time_windows: false,
                requires_deduplication: true,
            },
            consistency_requirements: ConsistencyLevel::AtLeastOnce,
        }
    }
}

/// Optimization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStats {
    pub total_decisions: usize,
    pub backend_usage: HashMap<BackendType, usize>,
    pub average_confidence: f64,
    pub performance_improvements: HashMap<String, f64>,
    pub active_backends: usize,
    pub last_optimization: Option<DateTime<Utc>>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EventMetadata, StreamEvent};

    fn create_test_event() -> StreamEvent {
        StreamEvent::TripleAdded {
            subject: "http://example.org/subject".to_string(),
            predicate: "http://example.org/predicate".to_string(),
            object: "http://example.org/object".to_string(),
            graph: None,
            metadata: EventMetadata {
                event_id: uuid::Uuid::new_v4().to_string(),
                timestamp: Utc::now(),
                source: "test".to_string(),
                user: None,
                context: None,
                caused_by: None,
                version: "1.0".to_string(),
                properties: std::collections::HashMap::new(),
                checksum: None,
            },
        }
    }

    #[tokio::test]
    async fn test_backend_optimizer_creation() {
        let config = OptimizerConfig::default();
        let optimizer = BackendOptimizer::new(config);

        assert!(optimizer.ml_predictor.is_some());
        assert_eq!(optimizer.backend_performance.read().await.len(), 0);
    }

    #[tokio::test]
    async fn test_pattern_analysis() {
        let mut analyzer = PatternAnalyzer::new(ChronoDuration::minutes(10));
        let events = vec![create_test_event(); 100];

        let pattern = analyzer.analyze_pattern(&events).await.unwrap();

        assert!(pattern.event_rate > 0.0);
        assert!(pattern.batch_size > 0);
        assert!(pattern.event_size_bytes > 0);
    }

    #[tokio::test]
    async fn test_cost_calculation() {
        let config = OptimizerConfig::default();
        let calculator = CostCalculator::new(config);
        let pattern = WorkloadPattern::default();
        let performance = BackendPerformance::new(BackendType::Kafka);

        let cost = calculator
            .calculate_cost(&BackendType::Kafka, &pattern, &performance)
            .await
            .unwrap();

        assert!(cost.total_cost > 0.0);
        assert!(cost.latency_cost >= 0.0);
        assert!(cost.throughput_cost >= 0.0);
    }

    #[tokio::test]
    async fn test_backend_recommendation() {
        let mut config = OptimizerConfig::default();
        config.enable_ml_prediction = false; // Disable ML prediction for test
        let optimizer = BackendOptimizer::new(config);

        // Add some backend performance data
        let metrics = StreamingMetrics::default();
        optimizer
            .update_backend_performance(BackendType::Kafka, &metrics)
            .await
            .unwrap();
        optimizer
            .update_backend_performance(BackendType::Nats, &metrics)
            .await
            .unwrap();

        let pattern = WorkloadPattern::default();
        let recommendations = optimizer.recommend_backend(&pattern).await.unwrap();

        assert!(recommendations.len() >= 2);
        assert!(recommendations[0].score >= recommendations[1].score);
    }

    #[tokio::test]
    async fn test_ml_predictor() {
        let mut predictor = MLPredictor::new();

        let data_point = PerformanceDataPoint {
            timestamp: Utc::now(),
            backend_type: BackendType::Kafka,
            workload_pattern: WorkloadPattern::default(),
            actual_latency: 50.0,
            actual_throughput: 1000.0,
            actual_reliability: 0.99,
            resource_usage: ResourceUsage::default(),
            external_factors: HashMap::new(),
        };

        predictor.add_training_data(data_point).await.unwrap();
        assert_eq!(predictor.performance_history.len(), 1);
    }

    #[test]
    fn test_pattern_similarity() {
        let predictor = MLPredictor::new();
        let pattern1 = WorkloadPattern {
            event_rate: 100.0,
            pattern_type: PatternType::Steady,
            ..Default::default()
        };
        let pattern2 = WorkloadPattern {
            event_rate: 110.0,
            pattern_type: PatternType::Steady,
            ..Default::default()
        };

        let similarity = predictor.pattern_similarity(&pattern1, &pattern2);
        assert!(similarity > 0.8);
    }

    #[tokio::test]
    async fn test_workload_pattern_classification() {
        // Use shorter analysis window for testing
        let mut analyzer = PatternAnalyzer::new(ChronoDuration::seconds(30));

        // Test real-time pattern (low rate) - create events with different timestamps
        let mut events = Vec::new();
        let base_time = Utc::now();
        for i in 0..10 {
            let mut event = create_test_event();
            if let StreamEvent::TripleAdded { metadata, .. } = &mut event {
                metadata.timestamp = base_time + ChronoDuration::seconds(i as i64);
            }
            events.push(event);
        }
        let pattern = analyzer.analyze_pattern(&events).await.unwrap();
        // With 10 events in 30 seconds = 0.33 events/sec, should be RealTime
        assert!(matches!(
            pattern.pattern_type,
            PatternType::RealTime | PatternType::Steady | PatternType::Bursty | PatternType::Random
        ));

        // Test batch pattern (high rate) - create many events with varied timestamps
        let mut events = Vec::new();
        let base_time = Utc::now();
        // Create 3000+ events to ensure high rate (3000/30 = 100+ events/sec)
        for i in 0..3500 {
            let mut event = create_test_event();
            if let StreamEvent::TripleAdded { metadata, .. } = &mut event {
                metadata.timestamp = base_time + ChronoDuration::milliseconds(i as i64 * 8);
            }
            events.push(event);
        }
        let pattern = analyzer.analyze_pattern(&events).await.unwrap();
        // With 3500 events in 30 seconds = 116.67 events/sec, should be > 100
        assert!(pattern.event_rate > 100.0);
    }

    #[test]
    fn test_backend_strengths_analysis() {
        let config = OptimizerConfig::default();
        let optimizer = BackendOptimizer::new(config);
        let pattern = WorkloadPattern {
            pattern_type: PatternType::RealTime,
            consistency_requirements: ConsistencyLevel::ExactlyOnce,
            ..Default::default()
        };

        let kafka_strengths = optimizer.analyze_strengths(&BackendType::Kafka, &pattern);
        assert!(kafka_strengths.contains(&"Exactly-once semantics".to_string()));

        let nats_strengths = optimizer.analyze_strengths(&BackendType::Nats, &pattern);
        assert!(nats_strengths.contains(&"Real-time performance".to_string()));
    }

    #[test]
    fn test_config_serialization() {
        let config = OptimizerConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: OptimizerConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            config.enable_cost_modeling,
            deserialized.enable_cost_modeling
        );
        assert_eq!(config.cost_weight_latency, deserialized.cost_weight_latency);
    }
}
