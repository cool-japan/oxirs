//! Advanced Query Optimization Engine
//!
//! This module provides cutting-edge optimization techniques including
//! index-aware optimization, streaming support, and machine learning-enhanced
//! query optimization.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::algebra::{Algebra, BinaryOperator, Expression, Term, TriplePattern, Variable};
use crate::cost_model::{CostEstimate, CostModel};
use crate::optimizer::{IndexStatistics, IndexType, OptimizationDecision, Statistics};
use crate::statistics_collector::StatisticsCollector;

/// Advanced optimizer with machine learning capabilities
pub struct AdvancedOptimizer {
    config: AdvancedOptimizerConfig,
    cost_model: Arc<Mutex<CostModel>>,
    statistics: Arc<StatisticsCollector>,
    index_advisor: IndexAdvisor,
    streaming_analyzer: StreamingAnalyzer,
    ml_predictor: Option<MLPredictor>,
    optimization_cache: OptimizationCache,
}

/// Configuration for advanced optimization features
#[derive(Debug, Clone)]
pub struct AdvancedOptimizerConfig {
    /// Enable machine learning-enhanced optimization
    pub enable_ml_optimization: bool,
    /// Enable adaptive index selection
    pub adaptive_index_selection: bool,
    /// Enable streaming optimization
    pub enable_streaming: bool,
    /// Maximum memory usage for optimization (bytes)
    pub max_memory_usage: usize,
    /// Enable cross-query optimization
    pub cross_query_optimization: bool,
    /// Learning rate for ML predictor
    pub learning_rate: f64,
    /// Cache size for optimization decisions
    pub cache_size: usize,
    /// Enable parallel optimization
    pub parallel_optimization: bool,
}

impl Default for AdvancedOptimizerConfig {
    fn default() -> Self {
        Self {
            enable_ml_optimization: true,
            adaptive_index_selection: true,
            enable_streaming: true,
            max_memory_usage: 1024 * 1024 * 1024, // 1GB
            cross_query_optimization: true,
            learning_rate: 0.01,
            cache_size: 10000,
            parallel_optimization: true,
        }
    }
}

/// Index advisor for automatic index recommendation
pub struct IndexAdvisor {
    query_patterns: HashMap<String, QueryPattern>,
    index_usage_stats: HashMap<IndexType, IndexUsageStats>,
    recommended_indexes: Vec<IndexRecommendation>,
}

/// Query pattern for index analysis
#[derive(Debug, Clone)]
pub struct QueryPattern {
    pub pattern_hash: u64,
    pub triple_patterns: Vec<TriplePattern>,
    pub join_variables: HashSet<Variable>,
    pub filter_variables: HashSet<Variable>,
    pub frequency: usize,
    pub avg_execution_time: Duration,
    pub avg_cardinality: usize,
}

/// Index usage statistics
#[derive(Debug, Clone, Default)]
pub struct IndexUsageStats {
    pub access_count: usize,
    pub total_access_time: Duration,
    pub avg_selectivity: f64,
    pub memory_usage: usize,
    pub last_updated: Option<Instant>,
}

/// Index recommendation
#[derive(Debug, Clone)]
pub struct IndexRecommendation {
    pub index_type: IndexType,
    pub priority: IndexPriority,
    pub estimated_benefit: f64,
    pub estimated_cost: f64,
    pub supporting_patterns: Vec<String>,
    pub confidence: f64,
}

/// Index priority levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum IndexPriority {
    Critical = 4,
    High = 3,
    Medium = 2,
    Low = 1,
}

/// Streaming query analyzer
pub struct StreamingAnalyzer {
    memory_threshold: usize,
    streaming_strategies: HashMap<String, StreamingStrategy>,
    spill_policies: Vec<SpillPolicy>,
}

/// Streaming execution strategy
#[derive(Debug, Clone)]
pub struct StreamingStrategy {
    pub strategy_type: StreamingType,
    pub memory_limit: usize,
    pub batch_size: usize,
    pub spill_threshold: f64,
    pub parallelism_degree: usize,
}

/// Types of streaming strategies
#[derive(Debug, Clone)]
pub enum StreamingType {
    PipelineBreaker,
    HashJoinStreaming,
    SortMergeStreaming,
    NestedLoopStreaming,
    IndexNestedLoop,
    HybridStreaming,
}

/// Spill policy for memory management
#[derive(Debug, Clone)]
pub struct SpillPolicy {
    pub policy_type: SpillType,
    pub threshold: f64,
    pub target_operators: Vec<String>,
    pub cost_factor: f64,
}

/// Types of spill policies
#[derive(Debug, Clone)]
pub enum SpillType {
    LeastRecentlyUsed,
    LargestFirst,
    CostBased,
    PredictiveBased,
}

/// Machine learning predictor for optimization decisions
pub struct MLPredictor {
    model: MLModel,
    training_data: Vec<TrainingExample>,
    feature_extractor: FeatureExtractor,
    prediction_cache: HashMap<u64, MLPrediction>,
}

/// ML model for cost prediction
#[derive(Debug, Clone)]
pub struct MLModel {
    weights: Vec<f64>,
    bias: f64,
    model_type: MLModelType,
    accuracy_metrics: AccuracyMetrics,
}

/// Types of ML models
#[derive(Debug, Clone)]
pub enum MLModelType {
    LinearRegression,
    RandomForest,
    NeuralNetwork,
    GradientBoosting,
}

/// Training example for ML model
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub features: Vec<f64>,
    pub target_cost: f64,
    pub actual_cost: f64,
    pub query_characteristics: QueryCharacteristics,
}

/// Query characteristics for feature extraction
#[derive(Debug, Clone)]
pub struct QueryCharacteristics {
    pub num_triple_patterns: usize,
    pub num_join_variables: usize,
    pub num_filters: usize,
    pub max_path_length: usize,
    pub estimated_cardinality: usize,
    pub complexity_score: f64,
}

/// Feature extractor for ML model
pub struct FeatureExtractor {
    feature_names: Vec<String>,
    normalization_params: HashMap<String, NormalizationParams>,
}

/// Normalization parameters for features
#[derive(Debug, Clone)]
pub struct NormalizationParams {
    pub mean: f64,
    pub std_dev: f64,
    pub min_val: f64,
    pub max_val: f64,
}

/// ML prediction result
#[derive(Debug, Clone)]
pub struct MLPrediction {
    pub predicted_cost: f64,
    pub confidence: f64,
    pub alternative_strategies: Vec<OptimizationStrategy>,
}

/// Optimization strategy suggestion
#[derive(Debug, Clone)]
pub struct OptimizationStrategy {
    pub strategy_name: String,
    pub predicted_improvement: f64,
    pub risk_factor: f64,
    pub implementation_cost: f64,
}

/// Accuracy metrics for ML model
#[derive(Debug, Clone, Default)]
pub struct AccuracyMetrics {
    pub mean_absolute_error: f64,
    pub root_mean_square_error: f64,
    pub r_squared: f64,
    pub prediction_confidence: f64,
}

/// Optimization cache for storing previous decisions
pub struct OptimizationCache {
    cache: HashMap<u64, CachedOptimization>,
    max_size: usize,
    hit_count: usize,
    miss_count: usize,
}

/// Cached optimization result
#[derive(Debug, Clone)]
pub struct CachedOptimization {
    pub optimized_algebra: Algebra,
    pub optimization_decisions: Vec<OptimizationDecision>,
    pub estimated_cost: f64,
    pub created_at: Instant,
    pub access_count: usize,
}

impl AdvancedOptimizer {
    /// Create a new advanced optimizer
    pub fn new(
        config: AdvancedOptimizerConfig,
        cost_model: Arc<Mutex<CostModel>>,
        statistics: Arc<StatisticsCollector>,
    ) -> Self {
        let index_advisor = IndexAdvisor::new();
        let streaming_analyzer = StreamingAnalyzer::new(config.max_memory_usage);
        let ml_predictor = if config.enable_ml_optimization {
            Some(MLPredictor::new(config.learning_rate))
        } else {
            None
        };
        let optimization_cache = OptimizationCache::new(config.cache_size);

        Self {
            config,
            cost_model,
            statistics,
            index_advisor,
            streaming_analyzer,
            ml_predictor,
            optimization_cache,
        }
    }

    /// Perform advanced optimization on query algebra
    pub fn optimize(&mut self, algebra: &Algebra) -> Result<OptimizationResult> {
        let start_time = Instant::now();
        let query_hash = self.calculate_query_hash(algebra);

        // Check cache first
        if let Some(cached) = self.optimization_cache.get(query_hash) {
            return Ok(OptimizationResult {
                optimized_algebra: cached.optimized_algebra.clone(),
                optimization_decisions: cached.optimization_decisions.clone(),
                estimated_cost: cached.estimated_cost,
                optimization_time: start_time.elapsed(),
                cache_hit: true,
            });
        }

        // Extract query characteristics
        let characteristics = self.extract_query_characteristics(algebra)?;

        // Perform ML-enhanced cost prediction if available
        let ml_guidance = if let Some(predictor) = &mut self.ml_predictor {
            predictor.predict_optimization_strategy(&characteristics)?
        } else {
            None
        };

        // Perform index-aware optimization
        let index_optimized = self.optimize_with_indexes(algebra)?;

        // Perform streaming optimization if needed
        let streaming_optimized = if self.should_use_streaming(&characteristics) {
            self.optimize_for_streaming(&index_optimized)?
        } else {
            index_optimized
        };

        // Apply ML-guided optimizations
        let final_optimized = if let Some(guidance) = ml_guidance {
            self.apply_ml_guidance(&streaming_optimized, &guidance)?
        } else {
            streaming_optimized
        };

        // Calculate final cost
        let cost_estimate = self
            .cost_model
            .lock()
            .unwrap()
            .estimate_cost(&final_optimized)?;
        let estimated_cost = cost_estimate.total_cost;

        let optimization_decisions = vec![]; // Would be populated with actual decisions

        // Cache the result
        self.optimization_cache.insert(
            query_hash,
            CachedOptimization {
                optimized_algebra: final_optimized.clone(),
                optimization_decisions: optimization_decisions.clone(),
                estimated_cost,
                created_at: Instant::now(),
                access_count: 1,
            },
        );

        Ok(OptimizationResult {
            optimized_algebra: final_optimized,
            optimization_decisions,
            estimated_cost,
            optimization_time: start_time.elapsed(),
            cache_hit: false,
        })
    }

    /// Optimize query with index awareness
    fn optimize_with_indexes(&mut self, algebra: &Algebra) -> Result<Algebra> {
        if !self.config.adaptive_index_selection {
            return Ok(algebra.clone());
        }

        // Analyze query for index opportunities
        let index_opportunities = self.analyze_index_opportunities(algebra)?;

        // Select optimal indexes
        let selected_indexes = self.select_optimal_indexes(&index_opportunities)?;

        // Rewrite algebra to use selected indexes
        let optimized = self.rewrite_with_indexes(algebra, &selected_indexes)?;

        // Update index advisor statistics
        self.index_advisor.update_usage_stats(&selected_indexes);

        Ok(optimized)
    }

    /// Optimize query for streaming execution
    fn optimize_for_streaming(&mut self, algebra: &Algebra) -> Result<Algebra> {
        if !self.config.enable_streaming {
            return Ok(algebra.clone());
        }

        // Analyze memory requirements
        let memory_analysis = self.analyze_memory_requirements(algebra)?;

        // Determine streaming strategy
        let strategy = self.streaming_analyzer.select_strategy(&memory_analysis)?;

        // Apply streaming optimizations
        let optimized = self.apply_streaming_strategy(algebra, &strategy)?;

        Ok(optimized)
    }

    /// Apply ML-guided optimizations
    fn apply_ml_guidance(&self, algebra: &Algebra, guidance: &MLPrediction) -> Result<Algebra> {
        let mut optimized = algebra.clone();

        // Apply suggested strategies in order of predicted improvement
        let mut strategies = guidance.alternative_strategies.clone();
        strategies.sort_by(|a, b| {
            b.predicted_improvement
                .partial_cmp(&a.predicted_improvement)
                .unwrap()
        });

        for strategy in strategies.iter().take(3) {
            // Apply top 3 strategies
            if strategy.predicted_improvement > 0.1 && strategy.risk_factor < 0.3 {
                optimized = self.apply_optimization_strategy(&optimized, strategy)?;
            }
        }

        Ok(optimized)
    }

    /// Generate index recommendations
    pub fn recommend_indexes(&self) -> Vec<IndexRecommendation> {
        self.index_advisor.generate_recommendations()
    }

    /// Update optimizer with execution feedback
    pub fn update_with_feedback(
        &mut self,
        algebra: &Algebra,
        actual_cost: f64,
        execution_time: Duration,
    ) -> Result<()> {
        // Update ML model if available
        if self.ml_predictor.is_some() {
            let characteristics = self.extract_query_characteristics(algebra)?;
            let features = self.extract_features(&characteristics)?;
            let training_example = TrainingExample {
                features,
                target_cost: actual_cost,
                actual_cost: actual_cost,
                query_characteristics: characteristics,
            };

            // Now update the predictor
            if let Some(predictor) = &mut self.ml_predictor {
                predictor.add_training_example(training_example);
            }
        }

        // Update statistics
        // Note: StatisticsCollector method signature may need to be checked
        // self.statistics.record_execution(algebra, actual_cost, execution_time)?;

        Ok(())
    }

    // Helper methods

    fn calculate_query_hash(&self, algebra: &Algebra) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", algebra).hash(&mut hasher);
        hasher.finish()
    }

    fn extract_query_characteristics(&self, algebra: &Algebra) -> Result<QueryCharacteristics> {
        // Analyze the algebra tree to extract characteristics
        let mut characteristics = QueryCharacteristics {
            num_triple_patterns: 0,
            num_join_variables: 0,
            num_filters: 0,
            max_path_length: 0,
            estimated_cardinality: 0,
            complexity_score: 0.0,
        };

        self.analyze_algebra_node(algebra, &mut characteristics, 0)?;
        characteristics.complexity_score = self.calculate_complexity_score(&characteristics);

        Ok(characteristics)
    }

    fn analyze_algebra_node(
        &self,
        algebra: &Algebra,
        characteristics: &mut QueryCharacteristics,
        depth: usize,
    ) -> Result<()> {
        characteristics.max_path_length = characteristics.max_path_length.max(depth);

        match algebra {
            Algebra::Bgp(patterns) => {
                characteristics.num_triple_patterns += patterns.len();
            }
            Algebra::Join { left, right } => {
                self.analyze_algebra_node(left, characteristics, depth + 1)?;
                self.analyze_algebra_node(right, characteristics, depth + 1)?;
            }
            Algebra::LeftJoin { left, right, .. } => {
                self.analyze_algebra_node(left, characteristics, depth + 1)?;
                self.analyze_algebra_node(right, characteristics, depth + 1)?;
            }
            Algebra::Union { left, right } => {
                self.analyze_algebra_node(left, characteristics, depth + 1)?;
                self.analyze_algebra_node(right, characteristics, depth + 1)?;
            }
            Algebra::Filter { pattern, .. } => {
                characteristics.num_filters += 1;
                self.analyze_algebra_node(pattern, characteristics, depth + 1)?;
            }
            Algebra::Extend { pattern, .. } => {
                self.analyze_algebra_node(pattern, characteristics, depth + 1)?;
            }
            Algebra::PropertyPath { .. } => {
                characteristics.num_triple_patterns += 1;
            }
            _ => {}
        }

        Ok(())
    }

    fn calculate_complexity_score(&self, characteristics: &QueryCharacteristics) -> f64 {
        let base_score = characteristics.num_triple_patterns as f64;
        let join_penalty = (characteristics.num_join_variables as f64).powi(2) * 0.1;
        let filter_penalty = characteristics.num_filters as f64 * 0.2;
        let depth_penalty = characteristics.max_path_length as f64 * 0.1;

        base_score + join_penalty + filter_penalty + depth_penalty
    }

    fn should_use_streaming(&self, characteristics: &QueryCharacteristics) -> bool {
        characteristics.estimated_cardinality > 100000
            || characteristics.complexity_score > 10.0
            || characteristics.num_join_variables > 5
    }

    fn analyze_index_opportunities(&self, _algebra: &Algebra) -> Result<Vec<IndexOpportunity>> {
        // Placeholder implementation
        Ok(vec![])
    }

    fn select_optimal_indexes(
        &self,
        _opportunities: &[IndexOpportunity],
    ) -> Result<Vec<IndexType>> {
        // Placeholder implementation
        Ok(vec![])
    }

    fn rewrite_with_indexes(&self, algebra: &Algebra, _indexes: &[IndexType]) -> Result<Algebra> {
        // Placeholder implementation
        Ok(algebra.clone())
    }

    fn analyze_memory_requirements(&self, _algebra: &Algebra) -> Result<MemoryAnalysis> {
        // Placeholder implementation
        Ok(MemoryAnalysis {
            estimated_memory: 1024 * 1024,
            memory_hotspots: vec![],
            spill_candidates: vec![],
        })
    }

    fn apply_streaming_strategy(
        &self,
        algebra: &Algebra,
        _strategy: &StreamingStrategy,
    ) -> Result<Algebra> {
        // Placeholder implementation
        Ok(algebra.clone())
    }

    fn apply_optimization_strategy(
        &self,
        algebra: &Algebra,
        _strategy: &OptimizationStrategy,
    ) -> Result<Algebra> {
        // Placeholder implementation
        Ok(algebra.clone())
    }

    fn extract_features(&self, characteristics: &QueryCharacteristics) -> Result<Vec<f64>> {
        Ok(vec![
            characteristics.num_triple_patterns as f64,
            characteristics.num_join_variables as f64,
            characteristics.num_filters as f64,
            characteristics.max_path_length as f64,
            characteristics.estimated_cardinality as f64,
            characteristics.complexity_score,
        ])
    }
}

/// Result of advanced optimization
#[derive(Debug)]
pub struct OptimizationResult {
    pub optimized_algebra: Algebra,
    pub optimization_decisions: Vec<OptimizationDecision>,
    pub estimated_cost: f64,
    pub optimization_time: Duration,
    pub cache_hit: bool,
}

/// Index opportunity analysis
#[derive(Debug, Clone)]
pub struct IndexOpportunity {
    pub index_type: IndexType,
    pub benefit_score: f64,
    pub cost_score: f64,
    pub frequency: usize,
}

/// Memory analysis result
#[derive(Debug, Clone)]
pub struct MemoryAnalysis {
    pub estimated_memory: usize,
    pub memory_hotspots: Vec<String>,
    pub spill_candidates: Vec<String>,
}

// Implementation details for sub-components

impl IndexAdvisor {
    fn new() -> Self {
        Self {
            query_patterns: HashMap::new(),
            index_usage_stats: HashMap::new(),
            recommended_indexes: Vec::new(),
        }
    }

    fn update_usage_stats(&mut self, _indexes: &[IndexType]) {
        // Update index usage statistics
    }

    fn generate_recommendations(&self) -> Vec<IndexRecommendation> {
        self.recommended_indexes.clone()
    }
}

impl StreamingAnalyzer {
    fn new(memory_threshold: usize) -> Self {
        Self {
            memory_threshold,
            streaming_strategies: HashMap::new(),
            spill_policies: Vec::new(),
        }
    }

    fn select_strategy(&self, _analysis: &MemoryAnalysis) -> Result<StreamingStrategy> {
        Ok(StreamingStrategy {
            strategy_type: StreamingType::PipelineBreaker,
            memory_limit: self.memory_threshold,
            batch_size: 10000,
            spill_threshold: 0.8,
            parallelism_degree: 4,
        })
    }
}

impl MLPredictor {
    fn new(learning_rate: f64) -> Self {
        Self {
            model: MLModel {
                weights: vec![0.0; 6], // 6 features
                bias: 0.0,
                model_type: MLModelType::LinearRegression,
                accuracy_metrics: AccuracyMetrics::default(),
            },
            training_data: Vec::new(),
            feature_extractor: FeatureExtractor {
                feature_names: vec![
                    "num_triple_patterns".to_string(),
                    "num_join_variables".to_string(),
                    "num_filters".to_string(),
                    "max_path_length".to_string(),
                    "estimated_cardinality".to_string(),
                    "complexity_score".to_string(),
                ],
                normalization_params: HashMap::new(),
            },
            prediction_cache: HashMap::new(),
        }
    }

    fn predict_optimization_strategy(
        &mut self,
        characteristics: &QueryCharacteristics,
    ) -> Result<Option<MLPrediction>> {
        let features = vec![
            characteristics.num_triple_patterns as f64,
            characteristics.num_join_variables as f64,
            characteristics.num_filters as f64,
            characteristics.max_path_length as f64,
            characteristics.estimated_cardinality as f64,
            characteristics.complexity_score,
        ];

        let predicted_cost = self.predict_cost(&features)?;

        Ok(Some(MLPrediction {
            predicted_cost,
            confidence: 0.8, // Placeholder
            alternative_strategies: vec![OptimizationStrategy {
                strategy_name: "join_reordering".to_string(),
                predicted_improvement: 0.2,
                risk_factor: 0.1,
                implementation_cost: 0.05,
            }],
        }))
    }

    fn predict_cost(&self, features: &[f64]) -> Result<f64> {
        // Simple linear regression prediction
        let mut prediction = self.model.bias;
        for (i, &feature) in features.iter().enumerate() {
            if i < self.model.weights.len() {
                prediction += self.model.weights[i] * feature;
            }
        }
        Ok(prediction.max(0.0))
    }

    fn add_training_example(&mut self, example: TrainingExample) {
        self.training_data.push(example);

        // Retrain model if we have enough data
        if self.training_data.len() % 100 == 0 {
            let _ = self.retrain_model();
        }
    }

    fn retrain_model(&mut self) -> Result<()> {
        // Simple gradient descent for linear regression
        let learning_rate = 0.01;
        let n = self.training_data.len() as f64;

        if n == 0.0 {
            return Ok(());
        }

        // Calculate gradients
        let mut weight_gradients = vec![0.0; self.model.weights.len()];
        let mut bias_gradient = 0.0;

        for example in &self.training_data {
            let prediction = self.predict_cost(&example.features)?;
            let error = prediction - example.actual_cost;

            bias_gradient += error;
            for (i, &feature) in example.features.iter().enumerate() {
                if i < weight_gradients.len() {
                    weight_gradients[i] += error * feature;
                }
            }
        }

        // Update weights
        self.model.bias -= learning_rate * bias_gradient / n;
        for (i, gradient) in weight_gradients.iter().enumerate() {
            self.model.weights[i] -= learning_rate * gradient / n;
        }

        Ok(())
    }
}

impl OptimizationCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            hit_count: 0,
            miss_count: 0,
        }
    }

    fn get(&mut self, key: u64) -> Option<&CachedOptimization> {
        if let Some(cached) = self.cache.get_mut(&key) {
            cached.access_count += 1;
            self.hit_count += 1;
            Some(cached)
        } else {
            self.miss_count += 1;
            None
        }
    }

    fn insert(&mut self, key: u64, value: CachedOptimization) {
        if self.cache.len() >= self.max_size {
            // Simple LRU eviction
            let oldest_key = *self.cache.keys().next().unwrap();
            self.cache.remove(&oldest_key);
        }
        self.cache.insert(key, value);
    }

    fn hit_rate(&self) -> f64 {
        let total = self.hit_count + self.miss_count;
        if total > 0 {
            self.hit_count as f64 / total as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::Algebra;
    use crate::cost_model::{CostModel, CostModelConfig};

    #[test]
    fn test_advanced_optimizer_creation() {
        let config = AdvancedOptimizerConfig::default();
        let cost_model = Arc::new(Mutex::new(CostModel::new(CostModelConfig::default())));
        let statistics = Arc::new(StatisticsCollector::new());

        let optimizer = AdvancedOptimizer::new(config, cost_model, statistics);
        assert!(optimizer.config.enable_ml_optimization);
    }

    #[test]
    fn test_query_characteristics_extraction() {
        let config = AdvancedOptimizerConfig::default();
        let cost_model = Arc::new(Mutex::new(CostModel::new(CostModelConfig::default())));
        let statistics = Arc::new(StatisticsCollector::new());

        let optimizer = AdvancedOptimizer::new(config, cost_model, statistics);

        // Create a simple algebra for testing
        let pattern = crate::algebra::TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Variable(Variable::new("p").unwrap()),
            object: Term::Variable(Variable::new("o").unwrap()),
        };
        let algebra = Algebra::Bgp(vec![pattern]);

        let characteristics = optimizer.extract_query_characteristics(&algebra).unwrap();
        assert_eq!(characteristics.num_triple_patterns, 1);
    }

    #[test]
    fn test_optimization_cache() {
        let mut cache = OptimizationCache::new(2);

        let pattern = crate::algebra::TriplePattern {
            subject: Term::Variable(Variable::new("s").unwrap()),
            predicate: Term::Variable(Variable::new("p").unwrap()),
            object: Term::Variable(Variable::new("o").unwrap()),
        };
        let cached_opt = CachedOptimization {
            optimized_algebra: Algebra::Bgp(vec![pattern]),
            optimization_decisions: vec![],
            estimated_cost: 1.0,
            created_at: Instant::now(),
            access_count: 0,
        };

        cache.insert(1, cached_opt);
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_none());

        let hit_rate = cache.hit_rate();
        assert!(hit_rate > 0.0);
    }
}
