//! ML predictor: core `MLPredictor` orchestration, training, and prediction logic.
//!
//! This module owns the `MLPredictor` struct that ties feature extraction,
//! the regression model, online learning, prediction caching, and histogram-
//! based cardinality estimation together. It also defines `TrainingExample`,
//! which is shared with `ml_predictor_features` for histogram construction.
//!
//! Split from `ml_predictor.rs` to keep the file below the workspace
//! 2000-line refactor threshold.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// SciRS2 imports for ML and statistics
use scirs2_core::metrics::{Counter, Histogram, MetricsRegistry, Timer};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_stats::regression::{linear_regression, ridge_regression, RegressionResults};

use crate::algebra::Algebra;

use super::ml_predictor_features::{
    FeatureExtractor, HistogramCardinalityEstimator, HistogramConfig, HistogramStatistics,
    NormalizationParams, QueryCharacteristics,
};
use super::ml_predictor_model::{
    AccuracyMetrics, MLConfig, MLModel, MLModelType, MLPrediction, OptimizationRecommendation,
    SerializableRegressionResults,
};

// ---------------------------------------------------------------------------
// Training example
// ---------------------------------------------------------------------------

/// Training example for ML model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingExample {
    pub features: Vec<f64>,
    pub target_cost: f64,
    pub actual_cost: f64,
    pub query_characteristics: QueryCharacteristics,
    pub timestamp: SystemTime,
}

// ---------------------------------------------------------------------------
// Machine learning predictor
// ---------------------------------------------------------------------------

/// Machine learning predictor for optimization decisions
pub struct MLPredictor {
    pub(super) model: MLModel,
    pub(super) training_data: Vec<TrainingExample>,
    pub(super) feature_extractor: FeatureExtractor,
    pub(super) prediction_cache: HashMap<u64, MLPrediction>,
    pub(super) config: MLConfig,
    pub(super) metrics_collector: Arc<MetricsRegistry>,
    pub(super) last_training: Option<SystemTime>,
    pub(super) histogram_estimator: HistogramCardinalityEstimator,

    // Metrics
    pub(super) prediction_counter: Counter,
    pub(super) prediction_timer: Timer,
    pub(super) prediction_histogram: Histogram,
}

impl MLPredictor {
    /// Create a new ML predictor with configuration
    pub fn new(config: MLConfig) -> Result<Self> {
        let metrics_collector = Arc::new(MetricsRegistry::new());

        let prediction_counter = Counter::new("ml_predictor_predictions_total".to_string());
        let prediction_timer = Timer::new("ml_predictor_prediction_duration_seconds".to_string());
        let prediction_histogram =
            Histogram::new("ml_predictor_prediction_distribution".to_string());

        Ok(Self {
            model: MLModel {
                regression_results: None,
                model_type: config.model_type.clone(),
                accuracy_metrics: AccuracyMetrics {
                    mean_absolute_error: 0.0,
                    root_mean_square_error: 0.0,
                    r_squared: 0.0,
                    confidence_interval: (0.0, 0.0),
                },
                normalization_params: None,
            },
            training_data: Vec::new(),
            feature_extractor: FeatureExtractor::new(),
            prediction_cache: HashMap::new(),
            config,
            metrics_collector,
            last_training: None,
            histogram_estimator: HistogramCardinalityEstimator::new(HistogramConfig::default()),
            prediction_counter,
            prediction_timer,
            prediction_histogram,
        })
    }

    /// Create predictor from model type (backward compatibility)
    pub fn from_model_type(model_type: MLModelType) -> Result<Self> {
        let config = MLConfig {
            model_type,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Load model from disk
    pub fn load_model(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read model from {:?}", path))?;

        let predictor: MLPredictor = serde_json::from_str(&contents)
            .with_context(|| format!("Failed to deserialize model from {:?}", path))?;

        Ok(predictor)
    }

    /// Save model to disk
    pub fn save_model(&self, path: &Path) -> Result<()> {
        // Create parent directory if it doesn't exist
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory {:?}", parent))?;
        }

        let contents = serde_json::to_string_pretty(self).context("Failed to serialize model")?;

        std::fs::write(path, contents)
            .with_context(|| format!("Failed to write model to {:?}", path))?;

        Ok(())
    }

    /// Get model confidence score (0.0-1.0)
    pub fn confidence(&self) -> f64 {
        // Confidence based on R² score
        let r_squared = self.model.accuracy_metrics.r_squared;
        r_squared.clamp(0.0, 1.0)
    }

    /// Check if ML model should be used
    pub fn should_use_ml(&self) -> bool {
        self.confidence() >= self.config.confidence_threshold
            && self.model.regression_results.is_some()
    }

    /// Extract 13 features from query plan
    pub fn extract_features(&self, query: &Algebra) -> Vec<f64> {
        let mut features = Vec::with_capacity(13);

        // Extract structural features
        let characteristics = self.analyze_query_structure(query);

        // 1. Number of triple patterns
        features.push(characteristics.triple_pattern_count as f64);

        // 2. Number of joins
        features.push(characteristics.join_count as f64);

        // 3. Number of filters
        features.push(characteristics.filter_count as f64);

        // 4. Number of optional patterns
        features.push(characteristics.optional_count as f64);

        // 5. Has aggregation
        features.push(if characteristics.has_aggregation {
            1.0
        } else {
            0.0
        });

        // 6. Has sorting
        features.push(if characteristics.has_sorting {
            1.0
        } else {
            0.0
        });

        // 7. Estimated cardinality
        features.push(characteristics.estimated_cardinality as f64);

        // 8. Query graph diameter
        features.push(characteristics.query_graph_diameter as f64);

        // 9. Average degree
        features.push(characteristics.avg_degree);

        // 10. Max degree
        features.push(characteristics.max_degree as f64);

        // 11. Has cross product (detected from join pattern)
        features.push(self.calculate_cross_product_likelihood(query));

        // 12. Subquery depth
        features.push(self.calculate_subquery_depth(query));

        // 13. Aggregation complexity
        features.push(self.calculate_aggregation_complexity(query));

        // Normalize features if enabled
        if self.config.feature_normalization {
            self.normalize_features(features)
        } else {
            features
        }
    }

    /// Analyze query structure to extract characteristics
    fn analyze_query_structure(&self, query: &Algebra) -> QueryCharacteristics {
        let mut characteristics = QueryCharacteristics {
            triple_pattern_count: 0,
            join_count: 0,
            filter_count: 0,
            optional_count: 0,
            has_aggregation: false,
            has_sorting: false,
            estimated_cardinality: 1000,
            complexity_score: 0.0,
            query_graph_diameter: 1,
            avg_degree: 1.0,
            max_degree: 1,
        };

        self.traverse_algebra(query, &mut characteristics);

        // Calculate derived metrics
        characteristics.complexity_score = self.calculate_complexity_score(&characteristics);
        characteristics.query_graph_diameter = self.calculate_graph_diameter(&characteristics);
        let (avg_deg, max_deg) = self.calculate_degree_metrics(&characteristics);
        characteristics.avg_degree = avg_deg;
        characteristics.max_degree = max_deg;

        characteristics
    }

    /// Traverse algebra tree to extract features
    fn traverse_algebra(&self, algebra: &Algebra, characteristics: &mut QueryCharacteristics) {
        use crate::algebra::Algebra;

        match algebra {
            Algebra::Service { .. } => characteristics.triple_pattern_count += 1,
            Algebra::PropertyPath { .. } => characteristics.triple_pattern_count += 1,
            Algebra::Join { left, right, .. } => {
                characteristics.join_count += 1;
                self.traverse_algebra(left, characteristics);
                self.traverse_algebra(right, characteristics);
            }
            Algebra::LeftJoin { left, right, .. } => {
                characteristics.join_count += 1;
                characteristics.optional_count += 1;
                self.traverse_algebra(left, characteristics);
                self.traverse_algebra(right, characteristics);
            }
            Algebra::Filter { pattern, .. } => {
                characteristics.filter_count += 1;
                self.traverse_algebra(pattern, characteristics);
            }
            Algebra::Union { left, right } => {
                self.traverse_algebra(left, characteristics);
                self.traverse_algebra(right, characteristics);
            }
            Algebra::Extend { pattern, .. } => {
                self.traverse_algebra(pattern, characteristics);
            }
            Algebra::OrderBy { pattern, .. } => {
                characteristics.has_sorting = true;
                self.traverse_algebra(pattern, characteristics);
            }
            Algebra::Project { pattern, .. } => {
                self.traverse_algebra(pattern, characteristics);
            }
            Algebra::Distinct { pattern } => {
                self.traverse_algebra(pattern, characteristics);
            }
            Algebra::Reduced { pattern } => {
                self.traverse_algebra(pattern, characteristics);
            }
            Algebra::Slice { pattern, .. } => {
                self.traverse_algebra(pattern, characteristics);
            }
            Algebra::Group { pattern, .. } => {
                characteristics.has_aggregation = true;
                self.traverse_algebra(pattern, characteristics);
            }
            _ => {}
        }
    }

    /// Calculate complexity score from characteristics
    fn calculate_complexity_score(&self, characteristics: &QueryCharacteristics) -> f64 {
        let mut score = 0.0;

        // Exponential growth for joins (most expensive operation)
        score += (characteristics.join_count as f64).powi(2) * 3.0;

        // Linear growth for triple patterns
        score += characteristics.triple_pattern_count as f64 * 1.0;

        // Filter complexity
        score += characteristics.filter_count as f64 * 0.5;

        // Optional patterns
        score += characteristics.optional_count as f64 * 2.0;

        // Aggregation penalty
        if characteristics.has_aggregation {
            score += 5.0;
        }

        // Sorting penalty
        if characteristics.has_sorting {
            score += 2.0;
        }

        // Cardinality factor
        let cardinality_log = (characteristics.estimated_cardinality as f64)
            .log10()
            .max(1.0);
        score *= cardinality_log;

        score
    }

    /// Calculate query graph diameter
    fn calculate_graph_diameter(&self, characteristics: &QueryCharacteristics) -> usize {
        // Simplified diameter calculation based on structure
        if characteristics.join_count == 0 {
            1
        } else {
            (characteristics.join_count as f64).sqrt().ceil() as usize + 1
        }
    }

    /// Calculate degree metrics (average, max)
    fn calculate_degree_metrics(&self, characteristics: &QueryCharacteristics) -> (f64, usize) {
        if characteristics.triple_pattern_count == 0 {
            return (0.0, 0);
        }

        let avg_degree = if characteristics.triple_pattern_count > 0 {
            (characteristics.join_count as f64 * 2.0) / characteristics.triple_pattern_count as f64
        } else {
            0.0
        };

        let max_degree = characteristics
            .join_count
            .min(characteristics.triple_pattern_count);

        (avg_degree, max_degree)
    }

    /// Calculate cross product likelihood
    fn calculate_cross_product_likelihood(&self, _query: &Algebra) -> f64 {
        // Simplified heuristic - would need join graph analysis for accuracy
        0.0
    }

    /// Calculate subquery nesting depth
    fn calculate_subquery_depth(&self, algebra: &Algebra) -> f64 {
        self.calculate_depth_recursive(algebra, 0) as f64
    }

    fn calculate_depth_recursive(&self, algebra: &Algebra, current_depth: usize) -> usize {
        use crate::algebra::Algebra;

        match algebra {
            Algebra::Join { left, right, .. }
            | Algebra::LeftJoin { left, right, .. }
            | Algebra::Union { left, right } => {
                let left_depth = self.calculate_depth_recursive(left, current_depth + 1);
                let right_depth = self.calculate_depth_recursive(right, current_depth + 1);
                left_depth.max(right_depth)
            }
            Algebra::Filter { pattern, .. }
            | Algebra::Extend { pattern, .. }
            | Algebra::OrderBy { pattern, .. }
            | Algebra::Project { pattern, .. }
            | Algebra::Distinct { pattern }
            | Algebra::Reduced { pattern }
            | Algebra::Slice { pattern, .. }
            | Algebra::Group { pattern, .. } => {
                self.calculate_depth_recursive(pattern, current_depth + 1)
            }
            _ => current_depth,
        }
    }

    /// Calculate aggregation complexity
    fn calculate_aggregation_complexity(&self, algebra: &Algebra) -> f64 {
        self.count_aggregations(algebra) as f64
    }

    fn count_aggregations(&self, algebra: &Algebra) -> usize {
        use crate::algebra::Algebra;

        match algebra {
            Algebra::Group { .. } => 1,
            Algebra::Join { left, right, .. }
            | Algebra::LeftJoin { left, right, .. }
            | Algebra::Union { left, right } => {
                self.count_aggregations(left) + self.count_aggregations(right)
            }
            Algebra::Filter { pattern, .. }
            | Algebra::Extend { pattern, .. }
            | Algebra::OrderBy { pattern, .. }
            | Algebra::Project { pattern, .. }
            | Algebra::Distinct { pattern }
            | Algebra::Reduced { pattern }
            | Algebra::Slice { pattern, .. } => self.count_aggregations(pattern),
            _ => 0,
        }
    }

    /// Normalize features using z-score normalization
    fn normalize_features(&self, features: Vec<f64>) -> Vec<f64> {
        if let Some(params) = self.feature_extractor.normalization_params.as_ref() {
            features
                .iter()
                .enumerate()
                .map(|(i, &value)| {
                    let mean_slice = params.mean();
                    let std_slice = params.std_dev();
                    if i < mean_slice.len() && i < std_slice.len() {
                        let mean = mean_slice[i];
                        let std_dev = std_slice[i];
                        if std_dev > 1e-10 {
                            (value - mean) / std_dev
                        } else {
                            value
                        }
                    } else {
                        value
                    }
                })
                .collect()
        } else {
            features
        }
    }

    /// Make cost prediction
    pub fn predict_cost(&mut self, query: &Algebra) -> Result<MLPrediction> {
        let _guard = self.prediction_timer.start();
        self.prediction_counter.inc();

        let features = self.extract_features(query);
        let query_hash = self.hash_query(query);

        // Check cache
        if let Some(cached) = self.prediction_cache.get(&query_hash) {
            return Ok(cached.clone());
        }

        // Make prediction
        let (predicted_cost, confidence) = if self.should_use_ml() {
            self.predict_with_model(&features)?
        } else {
            self.heuristic_prediction(&features)?
        };

        self.prediction_histogram.observe(predicted_cost);

        // Generate recommendations and feature importance
        let recommendation = self.generate_recommendation(&features, predicted_cost);
        let feature_importance = self.calculate_feature_importance(&features);

        let prediction = MLPrediction {
            predicted_cost,
            confidence,
            recommendation,
            feature_importance,
        };

        // Cache prediction
        self.prediction_cache.insert(query_hash, prediction.clone());

        Ok(prediction)
    }

    /// Predict using trained model
    fn predict_with_model(&self, features: &[f64]) -> Result<(f64, f64)> {
        if let Some(ref results) = self.model.regression_results {
            // Linear prediction: y = b0 + b1*x1 + b2*x2 + ... + bn*xn
            let mut prediction = 0.0;

            for (i, &coef) in results.coefficients.iter().enumerate() {
                if i < features.len() {
                    prediction += coef * features[i];
                } else if i == features.len() {
                    // Intercept (if present as last coefficient)
                    prediction += coef;
                }
            }

            // Ensure non-negative prediction
            prediction = prediction.max(0.1);

            // Confidence from R²
            let confidence = self.confidence();

            Ok((prediction, confidence))
        } else {
            // Fall back to heuristic if model not trained
            self.heuristic_prediction(features)
        }
    }

    /// Heuristic-based prediction when ML model is not available
    fn heuristic_prediction(&self, features: &[f64]) -> Result<(f64, f64)> {
        let mut cost = 0.0;

        if features.len() >= 13 {
            let triple_patterns = features[0];
            let joins = features[1];
            let filters = features[2];
            let optional = features[3];
            let has_aggregation = features[4];
            let has_sorting = features[5];
            let cardinality = features[6];

            // Heuristic cost calculation
            cost += triple_patterns * 10.0;
            cost += joins * joins * 50.0;
            cost += filters * 5.0;
            cost += optional * 15.0;
            cost += has_aggregation * 100.0;
            cost += has_sorting * 20.0;
            cost += (cardinality / 1000.0) * 2.0;
        }

        cost = cost.max(1.0);
        let confidence = 0.5; // Lower confidence for heuristic

        Ok((cost, confidence))
    }

    /// Generate optimization recommendations
    fn generate_recommendation(
        &self,
        features: &[f64],
        predicted_cost: f64,
    ) -> OptimizationRecommendation {
        if features.len() < 7 {
            return OptimizationRecommendation::NoChange;
        }

        let joins = features[1];
        let has_aggregation = features[4];
        let cardinality = features[6];

        if predicted_cost > 1000.0 {
            if joins > 3.0 {
                return OptimizationRecommendation::ReorderJoins(vec![0, 1, 2]);
            }
            if cardinality > 10000.0 {
                return OptimizationRecommendation::EnableParallelism(4);
            }
            if has_aggregation > 0.0 {
                return OptimizationRecommendation::MaterializeSubquery;
            }
        }

        if predicted_cost > 100.0 && cardinality > 5000.0 {
            return OptimizationRecommendation::ApplyStreaming;
        }

        OptimizationRecommendation::NoChange
    }

    /// Calculate feature importance
    fn calculate_feature_importance(&self, features: &[f64]) -> Vec<(String, f64)> {
        let feature_names = vec![
            "triple_patterns",
            "joins",
            "filters",
            "optional",
            "has_aggregation",
            "has_sorting",
            "cardinality",
            "graph_diameter",
            "avg_degree",
            "max_degree",
            "cross_product",
            "subquery_depth",
            "aggregation_complexity",
        ];

        let total: f64 = features.iter().map(|x| x.abs()).sum();

        let mut importance: Vec<(String, f64)> = feature_names
            .iter()
            .zip(features.iter())
            .map(|(name, &value)| {
                let normalized = if total > 1e-10 {
                    (value.abs() / total).min(1.0)
                } else {
                    0.0
                };
                (name.to_string(), normalized)
            })
            .collect();

        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        importance
    }

    /// Add training example for online learning
    pub fn add_training_example(&mut self, example: TrainingExample) {
        self.training_data.push(example);

        // Limit training data size
        if self.training_data.len() > self.config.max_training_examples {
            self.training_data.remove(0);
        }

        // Clear prediction cache when new data arrives
        self.prediction_cache.clear();
    }

    /// Update from actual execution results (online learning)
    pub fn update_from_execution(&mut self, query: &Algebra, actual_cost: f64) -> Result<()> {
        let features = self.extract_features(query);
        let characteristics = self.analyze_query_structure(query);

        let example = TrainingExample {
            features: features.clone(),
            target_cost: actual_cost,
            actual_cost,
            query_characteristics: characteristics,
            timestamp: SystemTime::now(),
        };

        self.add_training_example(example);

        // Check if we should retrain
        if self.should_retrain() {
            self.train_model()?;
        }

        Ok(())
    }

    /// Check if model should be retrained
    fn should_retrain(&self) -> bool {
        if !self.config.auto_retraining {
            return false;
        }

        // Need minimum examples
        if self.training_data.len() < self.config.min_examples_for_training {
            return false;
        }

        // Check time since last training
        if let Some(last_training) = self.last_training {
            if let Ok(elapsed) = SystemTime::now().duration_since(last_training) {
                let hours_elapsed = elapsed.as_secs() / 3600;
                if hours_elapsed < self.config.training_interval_hours {
                    return false;
                }
            }
        }

        true
    }

    /// Train the model with collected data
    pub fn train_model(&mut self) -> Result<()> {
        if self.training_data.len() < self.config.min_examples_for_training {
            return Err(anyhow::anyhow!(
                "Insufficient training data: {} < {}",
                self.training_data.len(),
                self.config.min_examples_for_training
            ));
        }

        // Extract features and targets
        let n_samples = self.training_data.len();
        let n_features = self.training_data[0].features.len();

        // Build feature matrix
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        let mut y_data = Vec::with_capacity(n_samples);

        for example in &self.training_data {
            x_data.extend_from_slice(&example.features);
            y_data.push(example.actual_cost);
        }

        let x = Array2::from_shape_vec((n_samples, n_features), x_data)
            .map_err(|e| anyhow::anyhow!("Failed to create feature matrix: {}", e))?;
        let y = Array1::from_vec(y_data);

        // Calculate normalization parameters
        if self.config.feature_normalization {
            self.calculate_normalization_params(&x);
        }

        // Train model based on type
        let results = match self.model.model_type {
            MLModelType::LinearRegression => linear_regression(&x.view(), &y.view(), None)
                .map_err(|e| anyhow::anyhow!("Linear regression failed: {:?}", e))?,
            MLModelType::Ridge => {
                let alpha = Some(1.0); // Regularization parameter
                ridge_regression(
                    &x.view(),
                    &y.view(),
                    alpha,
                    None, // fit_intercept
                    None, // normalize
                    None, // tol
                    None, // max_iter
                    None, // conf_level
                )
                .map_err(|e| anyhow::anyhow!("Ridge regression failed: {:?}", e))?
            }
            _ => {
                // Fall back to linear regression for unsupported types
                linear_regression(&x.view(), &y.view(), None)
                    .map_err(|e| anyhow::anyhow!("Linear regression failed: {:?}", e))?
            }
        };

        // Store serializable results
        self.model.regression_results = Some(SerializableRegressionResults::from(&results));

        // Update accuracy metrics
        self.update_accuracy_metrics(&results, &x, &y)?;

        // Build histograms for improved cardinality estimation
        self.histogram_estimator
            .build_from_training_data(&self.training_data);

        // Update last training time
        self.last_training = Some(SystemTime::now());

        // Save model if persistence enabled
        if let Some(ref path) = self.config.model_persistence_path {
            self.save_model(path)?;
        }

        Ok(())
    }

    /// Calculate normalization parameters from training data
    fn calculate_normalization_params(&mut self, x: &Array2<f64>) {
        let n_features = x.ncols();

        let mut means = vec![0.0; n_features];
        let mut std_devs = vec![0.0; n_features];
        let mut mins = vec![f64::MAX; n_features];
        let mut maxs = vec![f64::MIN; n_features];

        // Calculate statistics for each feature
        for j in 0..n_features {
            let column = x.column(j);
            let n = column.len() as f64;

            // Mean
            let mean: f64 = column.iter().sum::<f64>() / n;
            means[j] = mean;

            // Std dev
            let variance: f64 = column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
            std_devs[j] = variance.sqrt();

            // Min/Max
            for &val in column.iter() {
                if val < mins[j] {
                    mins[j] = val;
                }
                if val > maxs[j] {
                    maxs[j] = val;
                }
            }
        }

        let params = NormalizationParams::new(means, std_devs, mins, maxs);

        self.feature_extractor.normalization_params = Some(params.clone());
        self.model.normalization_params = Some(params);
    }

    /// Update accuracy metrics after training
    fn update_accuracy_metrics(
        &mut self,
        results: &RegressionResults<f64>,
        _x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<()> {
        // Use metrics from RegressionResults
        let r_squared = results.r_squared;
        let residual_std_error = results.residual_std_error;

        // Calculate MAE and RMSE from residuals
        let residuals = &results.residuals;
        let n = residuals.len() as f64;

        let mae = residuals.iter().map(|&r| r.abs()).sum::<f64>() / n;
        let rmse = (residuals.iter().map(|&r| r * r).sum::<f64>() / n).sqrt();

        // Calculate confidence interval (simplified)
        let confidence_interval = (
            r_squared - 1.96 * residual_std_error / n.sqrt(),
            r_squared + 1.96 * residual_std_error / n.sqrt(),
        );

        self.model.accuracy_metrics = AccuracyMetrics {
            mean_absolute_error: mae,
            root_mean_square_error: rmse,
            r_squared,
            confidence_interval,
        };

        Ok(())
    }

    /// Get model accuracy metrics
    pub fn accuracy_metrics(&self) -> &AccuracyMetrics {
        &self.model.accuracy_metrics
    }

    /// Get the number of predictions made
    pub fn predictions_count(&self) -> usize {
        self.prediction_counter.get() as usize
    }

    /// Get training data count
    pub fn training_data_count(&self) -> usize {
        self.training_data.len()
    }

    /// Get improved cardinality estimate using histogram statistics
    pub fn estimate_cardinality_with_histogram(&self, features: &[f64]) -> Option<f64> {
        self.histogram_estimator
            .estimate_cardinality_with_histogram(features)
    }

    /// Get histogram statistics
    pub fn get_histogram_statistics(&self) -> HistogramStatistics {
        self.histogram_estimator.get_statistics()
    }

    /// Predict with enhanced histogram-based cardinality estimation
    pub fn predict_cost_with_histogram(&mut self, query: &Algebra) -> Result<MLPrediction> {
        let _guard = self.prediction_timer.start();
        self.prediction_counter.inc();

        let features = self.extract_features(query);
        let query_hash = self.hash_query(query);

        // Check cache
        if let Some(cached) = self.prediction_cache.get(&query_hash) {
            return Ok(cached.clone());
        }

        // Make prediction with histogram enhancement
        let (mut predicted_cost, mut confidence) = if self.should_use_ml() {
            self.predict_with_model(&features)?
        } else {
            self.heuristic_prediction(&features)?
        };

        // Enhance with histogram-based cardinality if available
        if let Some(histogram_cardinality) = self.estimate_cardinality_with_histogram(&features) {
            // Blend ML prediction with histogram-based estimate
            let blend_weight = 0.3; // 30% histogram, 70% ML
            predicted_cost =
                predicted_cost * (1.0 - blend_weight) + histogram_cardinality * blend_weight;

            // Increase confidence if histogram estimate agrees
            let agreement =
                1.0 - ((predicted_cost - histogram_cardinality).abs() / (predicted_cost + 1.0));
            confidence = (confidence + agreement * 0.2).min(1.0);
        }

        self.prediction_histogram.observe(predicted_cost);

        // Generate recommendations and feature importance
        let recommendation = self.generate_recommendation(&features, predicted_cost);
        let feature_importance = self.calculate_feature_importance(&features);

        let prediction = MLPrediction {
            predicted_cost,
            confidence,
            recommendation,
            feature_importance,
        };

        // Cache prediction
        self.prediction_cache.insert(query_hash, prediction.clone());

        Ok(prediction)
    }

    /// Hash query for caching
    fn hash_query(&self, query: &Algebra) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        let query_string = self.algebra_to_string(query);
        query_string.hash(&mut hasher);
        hasher.finish()
    }

    /// Convert algebra to string for hashing
    fn algebra_to_string(&self, algebra: &Algebra) -> String {
        use crate::algebra::Algebra;

        match algebra {
            Algebra::Service { .. } => "Service".to_string(),
            Algebra::PropertyPath { .. } => "PropertyPath".to_string(),
            Algebra::Join { left, right, .. } => {
                format!(
                    "Join({},{})",
                    self.algebra_to_string(left),
                    self.algebra_to_string(right)
                )
            }
            Algebra::LeftJoin { left, right, .. } => {
                format!(
                    "LeftJoin({},{})",
                    self.algebra_to_string(left),
                    self.algebra_to_string(right)
                )
            }
            Algebra::Filter { pattern, .. } => {
                format!("Filter({})", self.algebra_to_string(pattern))
            }
            Algebra::Union { left, right } => {
                format!(
                    "Union({},{})",
                    self.algebra_to_string(left),
                    self.algebra_to_string(right)
                )
            }
            Algebra::Extend { pattern, .. } => {
                format!("Extend({})", self.algebra_to_string(pattern))
            }
            Algebra::OrderBy { pattern, .. } => {
                format!("OrderBy({})", self.algebra_to_string(pattern))
            }
            Algebra::Project { pattern, .. } => {
                format!("Project({})", self.algebra_to_string(pattern))
            }
            Algebra::Distinct { pattern } => {
                format!("Distinct({})", self.algebra_to_string(pattern))
            }
            Algebra::Reduced { pattern } => {
                format!("Reduced({})", self.algebra_to_string(pattern))
            }
            Algebra::Slice { pattern, .. } => {
                format!("Slice({})", self.algebra_to_string(pattern))
            }
            Algebra::Group { pattern, .. } => {
                format!("Group({})", self.algebra_to_string(pattern))
            }
            _ => "Unknown".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Manual Serialize / Deserialize / Clone for MLPredictor
// ---------------------------------------------------------------------------

impl Serialize for MLPredictor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut state = serializer.serialize_struct("MLPredictor", 5)?;
        state.serialize_field("model", &self.model)?;
        state.serialize_field("training_data", &self.training_data)?;
        state.serialize_field("feature_extractor", &self.feature_extractor)?;
        state.serialize_field("config", &self.config)?;
        state.serialize_field("histogram_estimator", &self.histogram_estimator)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for MLPredictor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct MLPredictorData {
            model: MLModel,
            training_data: Vec<TrainingExample>,
            feature_extractor: FeatureExtractor,
            config: MLConfig,
            #[serde(default)]
            histogram_estimator: Option<HistogramCardinalityEstimator>,
        }

        let data = MLPredictorData::deserialize(deserializer)?;

        let metrics_collector = Arc::new(MetricsRegistry::new());
        let prediction_counter = Counter::new("ml_predictor_predictions_total".to_string());
        let prediction_timer = Timer::new("ml_predictor_prediction_duration_seconds".to_string());
        let prediction_histogram =
            Histogram::new("ml_predictor_prediction_distribution".to_string());

        Ok(MLPredictor {
            model: data.model,
            training_data: data.training_data,
            feature_extractor: data.feature_extractor,
            prediction_cache: HashMap::new(),
            config: data.config,
            metrics_collector,
            last_training: None,
            histogram_estimator: data
                .histogram_estimator
                .unwrap_or_else(|| HistogramCardinalityEstimator::new(HistogramConfig::default())),
            prediction_counter,
            prediction_timer,
            prediction_histogram,
        })
    }
}

impl Clone for MLPredictor {
    fn clone(&self) -> Self {
        MLPredictor {
            model: self.model.clone(),
            training_data: self.training_data.clone(),
            feature_extractor: self.feature_extractor.clone(),
            prediction_cache: HashMap::new(), // Don't clone cache
            config: self.config.clone(),
            metrics_collector: Arc::clone(&self.metrics_collector),
            last_training: self.last_training,
            histogram_estimator: self.histogram_estimator.clone(),
            prediction_counter: Counter::new("ml_predictor_predictions_total".to_string()),
            prediction_timer: Timer::new("ml_predictor_prediction_duration_seconds".to_string()),
            prediction_histogram: Histogram::new(
                "ml_predictor_prediction_distribution".to_string(),
            ),
        }
    }
}
