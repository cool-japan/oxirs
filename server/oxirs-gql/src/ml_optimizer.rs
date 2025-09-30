//! Machine Learning-Enhanced Query Optimizer
//!
//! This module provides ML-powered query optimization capabilities that learn from
//! historical query performance to make intelligent optimization decisions.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::info;

use crate::ast::{Document, OperationType, Selection, SelectionSet};
use crate::optimizer::{QueryComplexity, QueryOptimizer};
use crate::performance::{OperationMetrics, PerformanceTracker};
use crate::system_monitor;

/// ML optimizer configuration
#[derive(Debug, Clone)]
pub struct MLOptimizerConfig {
    pub learning_enabled: bool,
    pub min_samples_for_learning: usize,
    pub feature_extraction_enabled: bool,
    pub prediction_threshold: f64,
    pub model_update_interval: Duration,
    pub max_training_samples: usize,
    pub performance_history_window: Duration,
    pub use_neural_network: bool,
    pub neural_network_layers: Vec<usize>,
    pub neural_learning_rate: f64,
    pub enable_reinforcement_learning: bool,
    pub enable_semantic_analysis: bool,
    pub adaptive_optimization: bool,
}

impl Default for MLOptimizerConfig {
    fn default() -> Self {
        Self {
            learning_enabled: true,
            min_samples_for_learning: 100,
            feature_extraction_enabled: true,
            prediction_threshold: 0.7,
            model_update_interval: Duration::from_secs(3600), // 1 hour
            max_training_samples: 10000,
            performance_history_window: Duration::from_secs(86400), // 24 hours
            use_neural_network: false,
            neural_network_layers: vec![64, 32, 16],
            neural_learning_rate: 0.001,
            enable_reinforcement_learning: false,
            enable_semantic_analysis: true,
            adaptive_optimization: true,
        }
    }
}

/// Query features extracted for ML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    pub field_count: f64,
    pub max_depth: f64,
    pub complexity_score: f64,
    pub selection_count: f64,
    pub has_fragments: f64,
    pub has_variables: f64,
    pub operation_type: f64, // 0 = Query, 1 = Mutation, 2 = Subscription
    pub unique_field_types: f64,
    pub nested_list_count: f64,
    pub argument_count: f64,
    pub directive_count: f64,
    pub estimated_result_size: f64,
}

impl QueryFeatures {
    /// Convert features to a vector for ML algorithms
    pub fn to_vector(&self) -> Vec<f64> {
        vec![
            self.field_count,
            self.max_depth,
            self.complexity_score,
            self.selection_count,
            self.has_fragments,
            self.has_variables,
            self.operation_type,
            self.unique_field_types,
            self.nested_list_count,
            self.argument_count,
            self.directive_count,
            self.estimated_result_size,
        ]
    }

    /// Create features from a vector
    pub fn from_vector(vector: &[f64]) -> Result<Self> {
        if vector.len() != 12 {
            return Err(anyhow!(
                "Invalid feature vector length: expected 12, got {}",
                vector.len()
            ));
        }

        Ok(Self {
            field_count: vector[0],
            max_depth: vector[1],
            complexity_score: vector[2],
            selection_count: vector[3],
            has_fragments: vector[4],
            has_variables: vector[5],
            operation_type: vector[6],
            unique_field_types: vector[7],
            nested_list_count: vector[8],
            argument_count: vector[9],
            directive_count: vector[10],
            estimated_result_size: vector[11],
        })
    }
}

/// Training sample for the ML model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub features: QueryFeatures,
    pub execution_time_ms: f64,
    pub memory_usage_mb: f64,
    pub cache_hit: bool,
    pub error_occurred: bool,
    pub timestamp: SystemTime,
}

/// Performance prediction from the ML model
#[derive(Debug, Clone)]
pub struct PerformancePrediction {
    pub predicted_execution_time: Duration,
    pub predicted_memory_usage: f64,
    pub cache_hit_probability: f64,
    pub error_probability: f64,
    pub confidence_score: f64,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub estimated_improvement: f64, // Percentage improvement
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum RecommendationType {
    ReduceDepth,
    AddCaching,
    BreakIntoSubqueries,
    AddIndexes,
    OptimizeFragments,
    ParallelizeFields,
    ReduceComplexity,
}

/// Advanced neural network model for performance prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkModel {
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
    pub training_iterations: usize,
    pub accuracy: f64,
    pub last_trained: SystemTime,
}

/// Neural network layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: ActivationFunction,
}

/// Activation functions for neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    Softmax,
}

/// Simple linear regression model for performance prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegressionModel {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub training_samples: usize,
    pub last_updated: SystemTime,
}

impl LinearRegressionModel {
    pub fn new(feature_count: usize) -> Self {
        Self {
            weights: vec![0.0; feature_count],
            bias: 0.0,
            training_samples: 0,
            last_updated: SystemTime::now(),
        }
    }

    /// Predict execution time based on features
    pub fn predict(&self, features: &[f64]) -> f64 {
        if features.len() != self.weights.len() {
            return 1000.0; // Default prediction
        }

        let prediction = self.bias
            + features
                .iter()
                .zip(&self.weights)
                .map(|(f, w)| f * w)
                .sum::<f64>();

        prediction.max(0.0) // Ensure non-negative prediction
    }

    /// Train the model using gradient descent
    pub fn train(&mut self, samples: &[TrainingSample], learning_rate: f64, iterations: usize) {
        if samples.is_empty() {
            return;
        }

        for _ in 0..iterations {
            let mut weight_gradients = vec![0.0; self.weights.len()];
            let mut bias_gradient = 0.0;

            for sample in samples {
                let features = sample.features.to_vector();

                // Safety check: ensure features match model dimensions
                if features.len() != self.weights.len() {
                    continue; // Skip this sample if dimensions don't match
                }

                let prediction = self.predict(&features);
                let error = prediction - sample.execution_time_ms;

                // Calculate gradients
                bias_gradient += error;
                for (i, &feature) in features.iter().enumerate() {
                    weight_gradients[i] += error * feature;
                }
            }

            // Update weights and bias
            let sample_count = samples.len() as f64;
            self.bias -= learning_rate * bias_gradient / sample_count;
            for (i, gradient) in weight_gradients.iter().enumerate() {
                self.weights[i] -= learning_rate * gradient / sample_count;
            }
        }

        self.training_samples += samples.len();
        self.last_updated = SystemTime::now();
    }
}

/// ML-enhanced query optimizer
pub struct MLQueryOptimizer {
    config: MLOptimizerConfig,
    base_optimizer: QueryOptimizer,
    #[allow(dead_code)]
    performance_tracker: Arc<PerformanceTracker>,
    execution_time_model: Arc<RwLock<LinearRegressionModel>>,
    memory_model: Arc<RwLock<LinearRegressionModel>>,
    training_samples: Arc<RwLock<VecDeque<TrainingSample>>>,
    feature_stats: Arc<RwLock<FeatureStatistics>>,
}

/// Statistics for feature normalization
#[derive(Debug, Clone, Default)]
pub struct FeatureStatistics {
    pub feature_means: Vec<f64>,
    pub feature_stds: Vec<f64>,
    pub sample_count: usize,
}

impl FeatureStatistics {
    /// Update statistics with new samples
    pub fn update(&mut self, samples: &[TrainingSample]) {
        if samples.is_empty() {
            return;
        }

        let feature_count = samples[0].features.to_vector().len();

        if self.feature_means.is_empty() {
            self.feature_means = vec![0.0; feature_count];
            self.feature_stds = vec![1.0; feature_count];
        }

        // Calculate means
        let mut sums = vec![0.0; feature_count];
        for sample in samples {
            let features = sample.features.to_vector();
            for (i, &feature) in features.iter().enumerate() {
                sums[i] += feature;
            }
        }

        let total_samples = self.sample_count + samples.len();
        #[allow(clippy::needless_range_loop)]
        for i in 0..feature_count {
            let new_mean = sums[i] / samples.len() as f64;
            self.feature_means[i] = (self.feature_means[i] * self.sample_count as f64
                + new_mean * samples.len() as f64)
                / total_samples as f64;
        }

        // Calculate standard deviations
        let mut var_sums = vec![0.0; feature_count];
        for sample in samples {
            let features = sample.features.to_vector();
            for (i, &feature) in features.iter().enumerate() {
                let diff = feature - self.feature_means[i];
                var_sums[i] += diff * diff;
            }
        }

        #[allow(clippy::needless_range_loop)]
        for i in 0..feature_count {
            self.feature_stds[i] = (var_sums[i] / samples.len() as f64).sqrt().max(1e-6);
        }

        self.sample_count = total_samples;
    }

    /// Normalize features using z-score normalization
    pub fn normalize(&self, features: &[f64]) -> Vec<f64> {
        if self.feature_means.is_empty() || features.len() != self.feature_means.len() {
            return features.to_vec();
        }

        features
            .iter()
            .zip(&self.feature_means)
            .zip(&self.feature_stds)
            .map(|((&feature, &mean), &std)| (feature - mean) / std)
            .collect()
    }
}

impl MLQueryOptimizer {
    /// Create a new ML-enhanced query optimizer
    pub fn new(config: MLOptimizerConfig, performance_tracker: Arc<PerformanceTracker>) -> Self {
        let feature_count = 12; // Number of features in QueryFeatures

        Self {
            config,
            base_optimizer: QueryOptimizer::new(crate::optimizer::OptimizationConfig::default()),
            performance_tracker,
            execution_time_model: Arc::new(RwLock::new(LinearRegressionModel::new(feature_count))),
            memory_model: Arc::new(RwLock::new(LinearRegressionModel::new(feature_count))),
            training_samples: Arc::new(RwLock::new(VecDeque::new())),
            feature_stats: Arc::new(RwLock::new(FeatureStatistics::default())),
        }
    }

    /// Extract features from a GraphQL document
    pub fn extract_features(&self, document: &Document) -> Result<QueryFeatures> {
        let mut field_count = 0;
        let mut max_depth = 0;
        let mut selection_count = 0;
        let mut has_fragments = false;
        let mut has_variables = false;
        let mut operation_type = 0.0;
        let mut unique_field_types = HashSet::new();
        let mut nested_list_count = 0;
        let mut argument_count = 0;
        let mut directive_count = 0;

        use std::collections::HashSet;

        // Analyze each definition in the document
        for definition in &document.definitions {
            match definition {
                crate::ast::Definition::Operation(operation) => {
                    operation_type = match operation.operation_type {
                        OperationType::Query => 0.0,
                        OperationType::Mutation => 1.0,
                        OperationType::Subscription => 2.0,
                    };

                    if !operation.variable_definitions.is_empty() {
                        has_variables = true;
                    }

                    // Analyze selection set
                    let (fc, md, sc, uft, nlc, ac, dc) =
                        self.analyze_selection_set(&operation.selection_set, 1)?;
                    field_count += fc;
                    max_depth = max_depth.max(md);
                    selection_count += sc;
                    unique_field_types.extend(uft);
                    nested_list_count += nlc;
                    argument_count += ac;
                    directive_count += dc;
                }
                crate::ast::Definition::Fragment(_) => {
                    has_fragments = true;
                }
                crate::ast::Definition::Schema(_) => {
                    // Schema definitions don't affect query complexity
                }
                crate::ast::Definition::Type(_) => {
                    // Type definitions don't affect query complexity
                }
                crate::ast::Definition::Directive(_) => {
                    directive_count += 1;
                }
                crate::ast::Definition::SchemaExtension(_) => {
                    // Schema extensions don't affect query complexity
                }
                crate::ast::Definition::TypeExtension(_) => {
                    // Type extensions don't affect query complexity
                }
            }
        }

        let complexity = self.base_optimizer.analyze_complexity(document)?;

        Ok(QueryFeatures {
            field_count: field_count as f64,
            max_depth: max_depth as f64,
            complexity_score: complexity.complexity_score as f64,
            selection_count: selection_count as f64,
            has_fragments: if has_fragments { 1.0 } else { 0.0 },
            has_variables: if has_variables { 1.0 } else { 0.0 },
            operation_type,
            unique_field_types: unique_field_types.len() as f64,
            nested_list_count: nested_list_count as f64,
            argument_count: argument_count as f64,
            directive_count: directive_count as f64,
            estimated_result_size: self.estimate_result_size(&complexity),
        })
    }

    /// Predict query performance using ML models
    pub async fn predict_performance(&self, document: &Document) -> Result<PerformancePrediction> {
        let features = self.extract_features(document)?;
        let feature_vector = features.to_vector();

        // Normalize features
        let stats = self.feature_stats.read().await;
        let normalized_features = stats.normalize(&feature_vector);
        drop(stats);

        // Get predictions from models
        let execution_time_model = self.execution_time_model.read().await;
        let memory_model = self.memory_model.read().await;

        let predicted_execution_ms = execution_time_model.predict(&normalized_features);
        let predicted_memory_mb = memory_model.predict(&normalized_features);

        // Calculate confidence based on training samples
        let confidence = self
            .calculate_confidence(&execution_time_model, &memory_model)
            .await;

        // Estimate cache hit probability and error probability based on complexity
        let cache_hit_probability = self.estimate_cache_hit_probability(&features);
        let error_probability = self.estimate_error_probability(&features);

        Ok(PerformancePrediction {
            predicted_execution_time: Duration::from_millis(predicted_execution_ms as u64),
            predicted_memory_usage: predicted_memory_mb,
            cache_hit_probability,
            error_probability,
            confidence_score: confidence,
        })
    }

    /// Generate optimization recommendations based on ML predictions
    pub async fn recommend_optimizations(
        &self,
        document: &Document,
    ) -> Result<Vec<OptimizationRecommendation>> {
        let features = self.extract_features(document)?;
        let prediction = self.predict_performance(document).await?;
        let mut recommendations = Vec::new();

        // Recommend based on predicted performance
        if prediction.predicted_execution_time > Duration::from_millis(1000) {
            if features.max_depth > 5.0 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::ReduceDepth,
                    description: "Consider reducing query depth to improve performance".to_string(),
                    estimated_improvement: 15.0,
                    confidence: 0.8,
                });
            }

            if features.complexity_score > 100.0 {
                recommendations.push(OptimizationRecommendation {
                    recommendation_type: RecommendationType::ReduceComplexity,
                    description: "Query complexity is high, consider simplifying".to_string(),
                    estimated_improvement: 20.0,
                    confidence: 0.7,
                });
            }
        }

        if prediction.cache_hit_probability < 0.3 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::AddCaching,
                description: "Low cache hit probability, consider adding caching strategy"
                    .to_string(),
                estimated_improvement: 30.0,
                confidence: 0.6,
            });
        }

        if features.field_count > 20.0 {
            recommendations.push(OptimizationRecommendation {
                recommendation_type: RecommendationType::ParallelizeFields,
                description: "Many fields requested, consider parallelizing field resolution"
                    .to_string(),
                estimated_improvement: 25.0,
                confidence: 0.7,
            });
        }

        Ok(recommendations)
    }

    /// Record query execution for learning
    pub async fn record_execution(
        &self,
        document: &Document,
        metrics: &OperationMetrics,
    ) -> Result<()> {
        if !self.config.learning_enabled {
            return Ok(());
        }

        let features = self.extract_features(document)?;

        // Get real memory usage measurement
        let memory_usage_mb = system_monitor::get_current_memory_usage_mb().await;

        let sample = TrainingSample {
            features,
            execution_time_ms: metrics.execution_time.as_millis() as f64,
            memory_usage_mb,
            cache_hit: metrics.cache_hit,
            error_occurred: metrics.error_count > 0,
            timestamp: metrics.timestamp,
        };

        // Add to training samples
        let mut samples = self.training_samples.write().await;
        samples.push_back(sample);

        // Limit sample size
        while samples.len() > self.config.max_training_samples {
            samples.pop_front();
        }

        // Trigger model update if we have enough samples
        if samples.len() >= self.config.min_samples_for_learning {
            self.update_models().await?;
        }

        Ok(())
    }

    /// Update ML models with new training data
    pub async fn update_models(&self) -> Result<()> {
        let samples = self.training_samples.read().await;
        if samples.len() < self.config.min_samples_for_learning {
            return Ok(());
        }

        let recent_samples: Vec<_> = samples
            .iter()
            .filter(|sample| {
                sample.timestamp.elapsed().unwrap_or(Duration::from_secs(0))
                    < self.config.performance_history_window
            })
            .cloned()
            .collect();

        if recent_samples.is_empty() {
            return Ok(());
        }

        drop(samples);

        // Update feature statistics
        {
            let mut stats = self.feature_stats.write().await;
            stats.update(&recent_samples);
        }

        // Train execution time model
        {
            let mut model = self.execution_time_model.write().await;
            model.train(&recent_samples, 0.01, 100); // learning_rate=0.01, iterations=100
        }

        // Train memory model
        {
            let mut model = self.memory_model.write().await;
            model.train(&recent_samples, 0.01, 100);
        }

        info!("ML models updated with {} samples", recent_samples.len());
        Ok(())
    }

    /// Analyze a selection set recursively
    #[allow(clippy::only_used_in_recursion)]
    #[allow(clippy::type_complexity)]
    fn analyze_selection_set(
        &self,
        selection_set: &SelectionSet,
        depth: usize,
    ) -> Result<(usize, usize, usize, HashSet<String>, usize, usize, usize)> {
        let mut field_count = 0;
        let mut max_depth = depth;
        let mut selection_count = selection_set.selections.len();
        let mut unique_field_types = HashSet::new();
        let mut nested_list_count = 0;
        let mut argument_count = 0;
        let mut directive_count = 0;

        for selection in &selection_set.selections {
            match selection {
                Selection::Field(field) => {
                    field_count += 1;
                    unique_field_types.insert(field.name.clone());
                    argument_count += field.arguments.len();
                    directive_count += field.directives.len();

                    if let Some(ref sub_selection_set) = field.selection_set {
                        let (fc, md, sc, uft, nlc, ac, dc) =
                            self.analyze_selection_set(sub_selection_set, depth + 1)?;
                        field_count += fc;
                        max_depth = max_depth.max(md);
                        selection_count += sc;
                        unique_field_types.extend(uft);
                        nested_list_count += nlc;
                        argument_count += ac;
                        directive_count += dc;

                        // Count nested lists (simplified heuristic)
                        if field.name.ends_with("s") || field.name.contains("list") {
                            nested_list_count += 1;
                        }
                    }
                }
                Selection::InlineFragment(fragment) => {
                    directive_count += fragment.directives.len();
                    let (fc, md, sc, uft, nlc, ac, dc) =
                        self.analyze_selection_set(&fragment.selection_set, depth)?;
                    field_count += fc;
                    max_depth = max_depth.max(md);
                    selection_count += sc;
                    unique_field_types.extend(uft);
                    nested_list_count += nlc;
                    argument_count += ac;
                    directive_count += dc;
                }
                Selection::FragmentSpread(spread) => {
                    directive_count += spread.directives.len();
                    // Fragment spread doesn't directly contribute to depth analysis
                }
            }
        }

        Ok((
            field_count,
            max_depth,
            selection_count,
            unique_field_types,
            nested_list_count,
            argument_count,
            directive_count,
        ))
    }

    /// Estimate result size based on query complexity
    fn estimate_result_size(&self, complexity: &QueryComplexity) -> f64 {
        // Simple heuristic based on field count and depth
        (complexity.field_count as f64 * complexity.depth as f64)
            .log10()
            .max(1.0)
    }

    /// Estimate cache hit probability based on query features
    fn estimate_cache_hit_probability(&self, features: &QueryFeatures) -> f64 {
        // Simple heuristic: less complex queries are more likely to be cached
        let complexity_factor = (features.complexity_score / 100.0).min(1.0);
        (1.0 - complexity_factor).max(0.1)
    }

    /// Estimate error probability based on query features
    fn estimate_error_probability(&self, features: &QueryFeatures) -> f64 {
        // Simple heuristic: more complex queries are more likely to have errors
        let complexity_factor = (features.complexity_score / 200.0).min(1.0);
        complexity_factor.max(0.01)
    }

    /// Calculate confidence score based on model training
    async fn calculate_confidence(
        &self,
        execution_model: &LinearRegressionModel,
        memory_model: &LinearRegressionModel,
    ) -> f64 {
        let min_samples = execution_model
            .training_samples
            .min(memory_model.training_samples);
        let confidence =
            (min_samples as f64 / self.config.min_samples_for_learning as f64).min(1.0);
        confidence.max(0.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::performance::PerformanceTracker;

    #[test]
    fn test_linear_regression_model() {
        let feature_count = 12; // Match QueryFeatures vector length
        let mut model = LinearRegressionModel::new(feature_count);

        // Create simple training data
        let samples = vec![
            TrainingSample {
                features: QueryFeatures::from_vector(&[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ])
                .unwrap(),
                execution_time_ms: 100.0,
                memory_usage_mb: 0.0,
                cache_hit: false,
                error_occurred: false,
                timestamp: SystemTime::now(),
            },
            TrainingSample {
                features: QueryFeatures::from_vector(&[
                    2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
                ])
                .unwrap(),
                execution_time_ms: 200.0,
                memory_usage_mb: 0.0,
                cache_hit: false,
                error_occurred: false,
                timestamp: SystemTime::now(),
            },
        ];

        model.train(&samples, 0.01, 100);

        // Test prediction with a feature vector
        let test_features = [
            1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0,
        ];
        let prediction = model.predict(&test_features);

        // Since we're training with limited data, just ensure we get a reasonable prediction
        assert!(prediction >= 0.0, "Prediction should be non-negative");
        assert!(prediction < 1000.0, "Prediction should be reasonable");
    }

    #[test]
    fn test_feature_statistics() {
        let mut stats = FeatureStatistics::default();

        let samples = vec![
            TrainingSample {
                features: QueryFeatures::from_vector(&[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ])
                .unwrap(),
                execution_time_ms: 100.0,
                memory_usage_mb: 50.0, // Realistic test value
                cache_hit: false,
                error_occurred: false,
                timestamp: SystemTime::now(),
            },
            TrainingSample {
                features: QueryFeatures::from_vector(&[
                    2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0,
                ])
                .unwrap(),
                execution_time_ms: 200.0,
                memory_usage_mb: 20.0,
                cache_hit: true,
                error_occurred: false,
                timestamp: SystemTime::now(),
            },
        ];

        stats.update(&samples);

        assert_eq!(stats.feature_means.len(), 12);
        assert_eq!(stats.feature_stds.len(), 12);
        assert_eq!(stats.sample_count, 2);

        // Test normalization
        let normalized = stats.normalize(&[
            1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0, 16.5, 18.0,
        ]);
        assert_eq!(normalized.len(), 12);
    }

    #[tokio::test]
    async fn test_ml_optimizer_creation() {
        let config = MLOptimizerConfig::default();
        let performance_tracker = Arc::new(PerformanceTracker::new());

        let optimizer = MLQueryOptimizer::new(config, performance_tracker);

        // Test with a simple document
        let document = Document {
            definitions: vec![Definition::Operation(OperationDefinition {
                operation_type: OperationType::Query,
                name: Some("TestQuery".to_string()),
                variable_definitions: vec![],
                directives: vec![],
                selection_set: SelectionSet {
                    selections: vec![Selection::Field(Field {
                        alias: None,
                        name: "user".to_string(),
                        arguments: vec![],
                        directives: vec![],
                        selection_set: Some(SelectionSet {
                            selections: vec![Selection::Field(Field {
                                alias: None,
                                name: "id".to_string(),
                                arguments: vec![],
                                directives: vec![],
                                selection_set: None,
                            })],
                        }),
                    })],
                },
            })],
        };

        let features = optimizer.extract_features(&document).unwrap();
        assert!(features.field_count > 0.0);
        assert!(features.max_depth > 0.0);
    }
}
