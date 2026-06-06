//! Performance analyzer reporting and ML-driven query plan optimizer.
//!
//! Provides recommendation generation for the
//! [`PerformanceAnalyzer`](crate::performance_analyzer_collector::PerformanceAnalyzer)
//! as well as the [`QueryPlanOptimizer`], an ML-driven query-plan selection
//! engine that predicts execution time, memory usage, and success rate.

use anyhow::{anyhow, Result};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::info;

use crate::performance_analyzer_collector::PerformanceAnalyzer;
use crate::performance_analyzer_types::*;

// ---------------------------------------------------------------------------
// Recommendation reporting for PerformanceAnalyzer
// ---------------------------------------------------------------------------

impl PerformanceAnalyzer {
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

    pub(crate) fn generate_bottleneck_specific_recommendations(
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

    pub(crate) fn evaluate_rule_condition(
        &self,
        condition: &RuleCondition,
        history: &MetricsHistory,
    ) -> bool {
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

    pub(crate) fn initialize_rule_base() -> Vec<OptimizationRule> {
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

// ---------------------------------------------------------------------------
// QueryPlanOptimizer — ML-driven plan selection
// ---------------------------------------------------------------------------

/// Intelligent Query Plan Optimizer using ML-driven predictions
pub struct QueryPlanOptimizer {
    config: QueryOptimizerConfig,
    historical_plans: Arc<RwLock<VecDeque<QueryPlanExecution>>>,
    plan_model: Arc<RwLock<QueryPlanModel>>,
    optimization_stats: Arc<QueryOptimizationStats>,
    plan_cache: Arc<RwLock<HashMap<String, OptimalPlan>>>,
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
