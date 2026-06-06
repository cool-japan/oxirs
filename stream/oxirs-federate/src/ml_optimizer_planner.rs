//! ML Optimizer — Query plan optimization using ML predictions
//!
//! Contains the MLOptimizer struct and its full implementation: feature
//! extraction, query planning with cost-based decisions, source selection,
//! join-order optimization, caching strategy, and anomaly detection.

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use scirs2_core::random::Random;

use super::ml_optimizer_types::{
    AnomalyDetection, AnomalyType, CacheRecommendation, CachingStrategy, JoinOrderAlternative,
    JoinOrderOptimization, LinearRegressionModel, MLConfig, MLStatistics, NeuralNetworkModel,
    PerformanceOutcome, QueryFeatures, SourceAlternative, SourceSelectionPrediction,
    TrainingSample,
};

/// ML-driven query optimizer
#[derive(Clone)]
pub struct MLOptimizer {
    /// Configuration
    pub(super) config: MLConfig,
    /// Linear regression performance model
    pub(super) linear_model: Arc<RwLock<LinearRegressionModel>>,
    /// Neural network performance model
    pub(super) neural_model: Arc<RwLock<NeuralNetworkModel>>,
    /// Source selection model
    pub(super) source_selection_model: Arc<RwLock<HashMap<String, f64>>>,
    /// Join order optimization model
    pub(super) join_order_model: Arc<RwLock<HashMap<String, f64>>>,
    /// Caching strategy model
    pub(super) caching_model: Arc<RwLock<HashMap<String, f64>>>,
    /// Training samples
    pub(super) training_samples: Arc<RwLock<VecDeque<TrainingSample>>>,
    /// Anomaly detection baseline
    pub(super) anomaly_baseline: Arc<RwLock<HashMap<String, f64>>>,
    /// Model statistics
    pub(super) statistics: Arc<RwLock<MLStatistics>>,
}

impl Default for MLOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl MLOptimizer {
    /// Create new ML optimizer
    pub fn new() -> Self {
        Self::with_config(MLConfig::default())
    }

    /// Create ML optimizer with configuration
    pub fn with_config(config: MLConfig) -> Self {
        let feature_count = 13;
        let hidden_size = 16;

        Self {
            config: config.clone(),
            linear_model: Arc::new(RwLock::new(LinearRegressionModel::new(feature_count))),
            neural_model: Arc::new(RwLock::new(NeuralNetworkModel::new(
                feature_count,
                hidden_size,
                config.learning_rate,
            ))),
            source_selection_model: Arc::new(RwLock::new(HashMap::new())),
            join_order_model: Arc::new(RwLock::new(HashMap::new())),
            caching_model: Arc::new(RwLock::new(HashMap::new())),
            training_samples: Arc::new(RwLock::new(VecDeque::new())),
            anomaly_baseline: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(MLStatistics::default())),
        }
    }

    /// Predict query performance using ensemble of linear and neural network models
    pub async fn predict_performance(&self, features: &QueryFeatures) -> Result<f64> {
        if !self.config.enable_performance_prediction {
            return Ok(0.0);
        }

        let linear_prediction = {
            let model = self.linear_model.read().await;
            model.predict(features)
        };
        let neural_prediction = {
            let model = self.neural_model.read().await;
            model.predict(features)
        };
        let linear_accuracy = {
            let model = self.linear_model.read().await;
            model.accuracy
        };
        let neural_accuracy = {
            let model = self.neural_model.read().await;
            model.accuracy
        };

        let ensemble_prediction = if linear_accuracy + neural_accuracy > 0.0 {
            let linear_weight = linear_accuracy / (linear_accuracy + neural_accuracy);
            let neural_weight = neural_accuracy / (linear_accuracy + neural_accuracy);
            linear_prediction * linear_weight + neural_prediction * neural_weight
        } else {
            (linear_prediction + neural_prediction) / 2.0
        };

        let final_prediction = if ensemble_prediction <= 0.0 {
            let base_time = 50.0;
            let pattern_complexity = features.pattern_count as f64 * 20.0;
            let join_complexity = features.join_count as f64 * 100.0;
            let filter_complexity = features.filter_count as f64 * 10.0;
            let service_latency =
                features.avg_service_latency * (features.service_count as f64).max(1.0);
            base_time + pattern_complexity + join_complexity + filter_complexity + service_latency
        } else {
            ensemble_prediction
        };

        {
            let mut stats = self.statistics.write().await;
            stats.total_predictions += 1;
            stats.model_accuracy = (linear_accuracy + neural_accuracy) / 2.0;
        }

        debug!(
            "Performance prediction: {:.2}ms (linear: {:.2}, neural: {:.2}, final: {:.2}) for query with {} patterns",
            ensemble_prediction, linear_prediction, neural_prediction, final_prediction, features.pattern_count
        );

        Ok(final_prediction)
    }

    /// Recommend source selection
    pub async fn recommend_source_selection(
        &self,
        features: &QueryFeatures,
        available_services: &[String],
    ) -> Result<SourceSelectionPrediction> {
        if !self.config.enable_source_selection_learning {
            return Ok(SourceSelectionPrediction {
                recommended_services: available_services.to_vec(),
                confidence_scores: HashMap::new(),
                expected_performance: PerformanceOutcome::default(),
                alternatives: vec![],
            });
        }

        let model = self.source_selection_model.read().await;
        let mut service_scores = HashMap::new();

        for service in available_services {
            let default_score = self.config.confidence_threshold + 0.1;
            let score = model.get(service).copied().unwrap_or(default_score);
            let adjusted_score = self.adjust_service_score(service, features, score).await;
            service_scores.insert(service.clone(), adjusted_score);
        }

        let mut recommended: Vec<_> = service_scores
            .iter()
            .filter(|&(_, &score)| score > self.config.confidence_threshold)
            .map(|(service, &score)| (service.clone(), score))
            .collect();
        recommended.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut recommended_services: Vec<String> = recommended
            .iter()
            .take(3)
            .map(|(service, _)| service.clone())
            .collect();

        if recommended_services.is_empty() {
            let mut all_services: Vec<_> = service_scores
                .iter()
                .map(|(service, &score)| (service.clone(), score))
                .collect();
            all_services.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            recommended_services = all_services
                .iter()
                .take(3)
                .map(|(service, _)| service.clone())
                .collect();
        }

        let alternatives = self
            .generate_source_alternatives(available_services, &service_scores)
            .await;
        let expected_performance = self
            .predict_service_performance(&recommended_services, features)
            .await;

        Ok(SourceSelectionPrediction {
            recommended_services,
            confidence_scores: service_scores,
            expected_performance,
            alternatives,
        })
    }

    /// Optimize join order
    pub async fn optimize_join_order(
        &self,
        join_patterns: &[String],
        features: &QueryFeatures,
    ) -> Result<JoinOrderOptimization> {
        if !self.config.enable_join_order_optimization || join_patterns.is_empty() {
            return Ok(JoinOrderOptimization {
                recommended_order: (0..join_patterns.len()).map(|i| i.to_string()).collect(),
                expected_cost: 1.0,
                alternatives: vec![],
                confidence: 0.5,
            });
        }

        let model = self.join_order_model.read().await;

        let permutations = if join_patterns.len() <= 6 {
            Self::generate_permutations(join_patterns)
        } else {
            self.generate_heuristic_orders(join_patterns)
        };

        let mut order_scores = Vec::new();
        for order in permutations {
            let cost = self.calculate_join_cost(&order, features, &model).await;
            order_scores.push((order, cost));
        }
        order_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_order = order_scores
            .first()
            .map(|(order, cost)| (order.clone(), *cost))
            .unwrap_or_else(|| (join_patterns.to_vec(), 1.0));

        let pattern_to_index: HashMap<String, usize> = join_patterns
            .iter()
            .enumerate()
            .map(|(i, pattern)| (pattern.clone(), i))
            .collect();

        let recommended_indices: Vec<String> = best_order
            .0
            .iter()
            .map(|pattern| {
                pattern_to_index
                    .get(pattern)
                    .map(|&i| i.to_string())
                    .unwrap_or_else(|| "0".to_string())
            })
            .collect();

        let alternatives: Vec<JoinOrderAlternative> = order_scores
            .iter()
            .skip(1)
            .take(3)
            .map(|(order, cost)| JoinOrderAlternative {
                order: order
                    .iter()
                    .map(|pattern| {
                        pattern_to_index
                            .get(pattern)
                            .map(|&i| i.to_string())
                            .unwrap_or_else(|| "0".to_string())
                    })
                    .collect(),
                cost: *cost,
                risk: self.calculate_risk_score(*cost, best_order.1),
            })
            .collect();

        Ok(JoinOrderOptimization {
            recommended_order: recommended_indices,
            expected_cost: best_order.1,
            alternatives,
            confidence: self.calculate_join_confidence(&order_scores),
        })
    }

    /// Recommend caching strategy
    pub async fn recommend_caching_strategy(
        &self,
        query_patterns: &[String],
        features: &QueryFeatures,
    ) -> Result<CachingStrategy> {
        if !self.config.enable_caching_strategy_learning {
            return Ok(CachingStrategy {
                cache_items: HashMap::new(),
                eviction_order: vec![],
                expected_hit_rate: 0.5,
                memory_requirements: 0,
            });
        }

        let model = self.caching_model.read().await;
        let mut cache_items = HashMap::new();
        let mut total_memory = 0u64;

        for pattern in query_patterns {
            let cache_score = model.get(pattern).copied().unwrap_or(0.3);
            let adjusted_score = self
                .adjust_cache_score(pattern, features, cache_score)
                .await;

            if adjusted_score > 0.5 {
                let estimated_size = self.estimate_cache_size(pattern, features);
                let ttl = self.calculate_optimal_ttl(pattern, adjusted_score);

                cache_items.insert(
                    pattern.clone(),
                    CacheRecommendation {
                        should_cache: true,
                        priority: adjusted_score,
                        expected_benefit: adjusted_score * 100.0,
                        ttl_seconds: ttl,
                    },
                );
                total_memory += estimated_size;
            }
        }

        let mut eviction_order: Vec<_> = cache_items
            .iter()
            .map(|(pattern, rec)| (pattern.clone(), rec.priority))
            .collect();
        eviction_order.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let eviction_order: Vec<String> = eviction_order.into_iter().map(|(p, _)| p).collect();

        let expected_hit_rate = self.estimate_cache_hit_rate(&cache_items, features).await;

        Ok(CachingStrategy {
            cache_items,
            eviction_order,
            expected_hit_rate,
            memory_requirements: total_memory,
        })
    }

    /// Detect anomalies in query execution
    pub async fn detect_anomalies(
        &self,
        features: &QueryFeatures,
        outcome: &PerformanceOutcome,
    ) -> Result<AnomalyDetection> {
        if !self.config.enable_anomaly_detection {
            return Ok(AnomalyDetection {
                is_anomalous: false,
                anomaly_score: 0.0,
                anomaly_type: AnomalyType::PerformanceDegradation,
                confidence: 0.0,
                recommendations: vec![],
            });
        }

        let baseline = self.anomaly_baseline.read().await;

        let performance_score = self.calculate_performance_anomaly_score(outcome, &baseline);
        let resource_score = self.calculate_resource_anomaly_score(outcome, &baseline);
        let pattern_score = self.calculate_pattern_anomaly_score(features, &baseline);

        let max_score = performance_score.max(resource_score).max(pattern_score);
        let is_anomalous = max_score > 0.7;

        let anomaly_type = if performance_score == max_score {
            AnomalyType::PerformanceDegradation
        } else if resource_score == max_score {
            AnomalyType::ResourceAnomaly
        } else {
            AnomalyType::PatternAnomaly
        };

        let recommendations = self.generate_anomaly_recommendations(&anomaly_type, max_score);

        if is_anomalous {
            let mut stats = self.statistics.write().await;
            stats.anomalies_detected += 1;
        }

        Ok(AnomalyDetection {
            is_anomalous,
            anomaly_score: max_score,
            anomaly_type,
            confidence: max_score,
            recommendations,
        })
    }

    /// Add training sample
    pub async fn add_training_sample(&self, sample: TrainingSample) {
        let mut samples = self.training_samples.write().await;
        samples.push_back(sample);

        while samples.len() > self.config.feature_history_size {
            samples.pop_front();
        }

        let mut stats = self.statistics.write().await;
        stats.training_samples_count += 1;

        if samples.len() % 100 == 0 {
            drop(samples);
            drop(stats);
            let _ = self.retrain_models().await;
        }
    }

    /// Retrain all models
    pub async fn retrain_models(&self) -> Result<()> {
        info!("Starting ML model retraining");

        let samples: Vec<TrainingSample> = {
            let samples_guard = self.training_samples.read().await;
            samples_guard.iter().cloned().collect()
        };

        if samples.is_empty() {
            warn!("No training samples available for retraining");
            return Ok(());
        }

        {
            let mut linear_model = self.linear_model.write().await;
            linear_model.train(
                &samples,
                self.config.learning_rate,
                self.config.regularization,
            );
        }

        {
            let mut neural_model = self.neural_model.write().await;
            neural_model.train(&samples);
        }

        self.update_source_selection_model(&samples).await;
        self.update_join_order_model(&samples).await;
        self.update_caching_model(&samples).await;
        self.update_anomaly_baseline(&samples).await;

        {
            let mut stats = self.statistics.write().await;
            stats.last_training = Some(SystemTime::now());
            let linear_model = self.linear_model.read().await;
            let neural_model = self.neural_model.read().await;
            stats.model_accuracy = (linear_model.accuracy + neural_model.accuracy) / 2.0;
        }

        info!(
            "ML model retraining completed with {} samples",
            samples.len()
        );
        Ok(())
    }

    /// Get ML optimizer statistics
    pub async fn get_statistics(&self) -> MLStatistics {
        self.statistics.read().await.clone()
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    async fn adjust_service_score(
        &self,
        _service: &str,
        features: &QueryFeatures,
        base_score: f64,
    ) -> f64 {
        let complexity_factor = 1.0 - (features.complexity_score / 20.0).min(0.2);
        let service_factor = 1.0 - (features.service_count as f64 / 20.0).min(0.15);
        let adjusted_score = base_score * complexity_factor * service_factor;
        adjusted_score.max(self.config.confidence_threshold + 0.01)
    }

    async fn generate_source_alternatives(
        &self,
        available_services: &[String],
        scores: &HashMap<String, f64>,
    ) -> Vec<SourceAlternative> {
        let mut alternatives = Vec::new();

        for combination_size in 1..=3.min(available_services.len()) {
            if let Some(combination) =
                self.select_service_combination(available_services, scores, combination_size)
            {
                alternatives.push(SourceAlternative {
                    services: combination,
                    expected_performance: PerformanceOutcome::default(),
                    confidence: 0.7,
                    risk_score: 0.3,
                });
            }
        }

        alternatives
    }

    fn select_service_combination(
        &self,
        services: &[String],
        scores: &HashMap<String, f64>,
        size: usize,
    ) -> Option<Vec<String>> {
        if size > services.len() {
            return None;
        }
        let mut scored_services: Vec<_> = services
            .iter()
            .map(|s| (s.clone(), scores.get(s).copied().unwrap_or(0.0)))
            .collect();
        scored_services.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Some(
            scored_services
                .into_iter()
                .take(size)
                .map(|(s, _)| s)
                .collect(),
        )
    }

    async fn predict_service_performance(
        &self,
        _services: &[String],
        _features: &QueryFeatures,
    ) -> PerformanceOutcome {
        PerformanceOutcome::default()
    }

    fn generate_permutations(items: &[String]) -> Vec<Vec<String>> {
        if items.is_empty() {
            return vec![vec![]];
        }
        let mut permutations = Vec::new();
        for (i, item) in items.iter().enumerate() {
            let mut remaining = items.to_vec();
            remaining.remove(i);
            for mut perm in Self::generate_permutations(&remaining) {
                perm.insert(0, item.clone());
                permutations.push(perm);
            }
        }
        permutations
    }

    fn generate_heuristic_orders(&self, items: &[String]) -> Vec<Vec<String>> {
        vec![
            items.to_vec(),
            {
                let mut reversed = items.to_vec();
                reversed.reverse();
                reversed
            },
            {
                let mut random = items.to_vec();
                let mut rng = Random::default();
                Self::shuffle_optimized(&mut random, &mut rng);
                random
            },
        ]
    }

    async fn calculate_join_cost(
        &self,
        order: &[String],
        features: &QueryFeatures,
        model: &HashMap<String, f64>,
    ) -> f64 {
        let mut cost = 1.0;
        let selectivity_factor = (1.0 - features.selectivity).max(0.1);
        let complexity_factor = features.complexity_score + 1.0;

        for (i, pattern) in order.iter().enumerate() {
            let position_cost = (i + 1) as f64 * 0.2;
            let pattern_specific_cost = model.get(pattern).unwrap_or(&1.0);
            cost += position_cost * pattern_specific_cost * selectivity_factor;
        }

        cost *= complexity_factor;
        cost *= (features.service_count as f64).max(1.0);
        cost.clamp(0.1, 100.0)
    }

    fn calculate_risk_score(&self, cost: f64, best_cost: f64) -> f64 {
        if best_cost == 0.0 {
            return 0.0;
        }
        ((cost - best_cost) / best_cost).min(1.0)
    }

    fn calculate_join_confidence(&self, scores: &[(Vec<String>, f64)]) -> f64 {
        if scores.len() < 2 {
            return 0.5;
        }
        let best_cost = scores[0].1;
        let second_cost = scores[1].1;
        if second_cost == 0.0 {
            return 1.0;
        }
        1.0 - (best_cost / second_cost).min(1.0)
    }

    async fn adjust_cache_score(
        &self,
        _pattern: &str,
        features: &QueryFeatures,
        base_score: f64,
    ) -> f64 {
        let frequency_factor = if features.pattern_count > 5 { 1.2 } else { 1.0 };
        let complexity_factor = if features.complexity_score > 5.0 {
            1.1
        } else {
            1.0
        };
        base_score * frequency_factor * complexity_factor
    }

    fn estimate_cache_size(&self, _pattern: &str, features: &QueryFeatures) -> u64 {
        (features.data_size_estimate / features.pattern_count as u64).max(1024)
    }

    fn calculate_optimal_ttl(&self, _pattern: &str, priority: f64) -> u64 {
        (3600.0 * priority) as u64
    }

    async fn estimate_cache_hit_rate(
        &self,
        cache_items: &HashMap<String, CacheRecommendation>,
        _features: &QueryFeatures,
    ) -> f64 {
        if cache_items.is_empty() {
            return 0.0;
        }
        let avg_priority: f64 =
            cache_items.values().map(|r| r.priority).sum::<f64>() / cache_items.len() as f64;
        avg_priority.min(0.9)
    }

    fn calculate_performance_anomaly_score(
        &self,
        outcome: &PerformanceOutcome,
        baseline: &HashMap<String, f64>,
    ) -> f64 {
        let baseline_time = baseline.get("execution_time").copied().unwrap_or(1000.0);
        let current_time = outcome.execution_time_ms;
        if baseline_time == 0.0 {
            return 0.0;
        }
        ((current_time - baseline_time) / baseline_time).clamp(0.0, 1.0)
    }

    fn calculate_resource_anomaly_score(
        &self,
        outcome: &PerformanceOutcome,
        baseline: &HashMap<String, f64>,
    ) -> f64 {
        let baseline_memory = baseline
            .get("memory_usage")
            .copied()
            .unwrap_or(1024.0 * 1024.0);
        let current_memory = outcome.memory_usage_bytes as f64;
        if baseline_memory == 0.0 {
            return 0.0;
        }
        ((current_memory - baseline_memory) / baseline_memory).clamp(0.0, 1.0)
    }

    fn calculate_pattern_anomaly_score(
        &self,
        _features: &QueryFeatures,
        _baseline: &HashMap<String, f64>,
    ) -> f64 {
        0.1
    }

    fn generate_anomaly_recommendations(
        &self,
        anomaly_type: &AnomalyType,
        score: f64,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();
        match anomaly_type {
            AnomalyType::PerformanceDegradation => {
                recommendations.push("Consider optimizing query patterns".to_string());
                if score > 0.8 {
                    recommendations.push("Review service selection strategy".to_string());
                }
            }
            AnomalyType::ResourceAnomaly => {
                recommendations.push("Monitor memory usage patterns".to_string());
                recommendations.push("Consider implementing result streaming".to_string());
            }
            AnomalyType::PatternAnomaly => {
                recommendations.push("Review query structure".to_string());
            }
            _ => {
                recommendations.push("Monitor system behavior".to_string());
            }
        }
        recommendations
    }

    async fn update_source_selection_model(&self, samples: &[TrainingSample]) {
        let mut model = self.source_selection_model.write().await;
        for sample in samples {
            for service in &sample.service_selections {
                let success_rate = sample.outcome.success_rate;
                let current_score = model.get(service).copied().unwrap_or(0.5);
                let new_score = current_score * 0.9 + success_rate * 0.1;
                model.insert(service.clone(), new_score);
            }
        }
    }

    async fn update_join_order_model(&self, samples: &[TrainingSample]) {
        let mut model = self.join_order_model.write().await;
        for sample in samples {
            for (i, pattern) in sample.join_order.iter().enumerate() {
                let position_score = 1.0 - (i as f64 / sample.join_order.len() as f64);
                let performance_factor = 1.0 / sample.outcome.execution_time_ms.max(1.0);
                let score = position_score * performance_factor * 1000.0;
                let current_score = model.get(pattern).copied().unwrap_or(0.5);
                let new_score = current_score * 0.9 + score * 0.1;
                model.insert(pattern.clone(), new_score);
            }
        }
    }

    async fn update_caching_model(&self, samples: &[TrainingSample]) {
        let mut model = self.caching_model.write().await;
        for sample in samples {
            for (item, &should_cache) in &sample.caching_decisions {
                let cache_benefit = if should_cache {
                    sample.outcome.cache_hit_rate
                } else {
                    1.0 - sample.outcome.cache_hit_rate
                };
                let current_score = model.get(item).copied().unwrap_or(0.5);
                let new_score = current_score * 0.9 + cache_benefit * 0.1;
                model.insert(item.clone(), new_score);
            }
        }
    }

    async fn update_anomaly_baseline(&self, samples: &[TrainingSample]) {
        let mut baseline = self.anomaly_baseline.write().await;
        if samples.is_empty() {
            return;
        }
        let avg_execution_time = samples
            .iter()
            .map(|s| s.outcome.execution_time_ms)
            .sum::<f64>()
            / samples.len() as f64;
        let avg_memory_usage = samples
            .iter()
            .map(|s| s.outcome.memory_usage_bytes as f64)
            .sum::<f64>()
            / samples.len() as f64;
        baseline.insert("execution_time".to_string(), avg_execution_time);
        baseline.insert("memory_usage".to_string(), avg_memory_usage);
    }

    /// scirs2-optimized Fisher-Yates shuffle
    fn shuffle_optimized<T>(items: &mut [T], rng: &mut Random) {
        if items.len() <= 1 {
            return;
        }
        for i in (1..items.len()).rev() {
            let j = rng.random_range(0..i + 1);
            if i != j {
                items.swap(i, j);
            }
        }
    }
}
