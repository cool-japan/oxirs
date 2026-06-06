//! ML predictor model and configuration types.
//!
//! This module contains the configuration knobs, model parameter container,
//! prediction record, and accuracy metrics for the ML cost predictor.
//!
//! These types are split from `ml_predictor.rs` to keep that file (and its
//! sibling units) below the workspace 2000-line refactor threshold while
//! preserving the public API surface re-exported through
//! `super::ml_predictor`.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

// SciRS2 imports for ML and statistics
use scirs2_stats::regression::RegressionResults;

use super::ml_predictor_features::NormalizationParams;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for ML predictor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLConfig {
    /// Model type to use
    pub model_type: MLModelType,
    /// Confidence threshold for using ML predictions (0.0-1.0)
    pub confidence_threshold: f64,
    /// Training interval in hours
    pub training_interval_hours: u64,
    /// Maximum number of training examples to keep
    pub max_training_examples: usize,
    /// Minimum examples required before training
    pub min_examples_for_training: usize,
    /// Whether to normalize features
    pub feature_normalization: bool,
    /// Enable auto-retraining
    pub auto_retraining: bool,
    /// Path to save/load model
    pub model_persistence_path: Option<PathBuf>,
}

impl Default for MLConfig {
    fn default() -> Self {
        Self {
            model_type: MLModelType::Ridge,
            confidence_threshold: 0.7,
            training_interval_hours: 24,
            max_training_examples: 10000,
            min_examples_for_training: 100,
            feature_normalization: true,
            auto_retraining: true,
            model_persistence_path: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Model parameter container
// ---------------------------------------------------------------------------

/// ML model for cost prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLModel {
    /// Regression results (coefficients, statistics)
    pub(super) regression_results: Option<SerializableRegressionResults>,
    /// Model type
    pub(super) model_type: MLModelType,
    /// Accuracy metrics
    pub(super) accuracy_metrics: AccuracyMetrics,
    /// Normalization parameters
    pub(super) normalization_params: Option<NormalizationParams>,
}

/// Serializable version of RegressionResults
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableRegressionResults {
    pub coefficients: Vec<f64>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    pub residual_std_error: f64,
    pub std_errors: Vec<f64>,
}

impl From<&RegressionResults<f64>> for SerializableRegressionResults {
    fn from(results: &RegressionResults<f64>) -> Self {
        Self {
            coefficients: results.coefficients.to_vec(),
            r_squared: results.r_squared,
            adj_r_squared: results.adj_r_squared,
            residual_std_error: results.residual_std_error,
            std_errors: results.std_errors.to_vec(),
        }
    }
}

/// Types of ML models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MLModelType {
    LinearRegression,
    Ridge,
    RandomForest,
    NeuralNetwork,
    GradientBoosting,
}

// ---------------------------------------------------------------------------
// Prediction outputs
// ---------------------------------------------------------------------------

/// ML prediction result
#[derive(Debug, Clone)]
pub struct MLPrediction {
    pub predicted_cost: f64,
    pub confidence: f64,
    pub recommendation: OptimizationRecommendation,
    pub feature_importance: Vec<(String, f64)>,
}

/// Optimization recommendation from ML
#[derive(Debug, Clone)]
pub enum OptimizationRecommendation {
    UseIndex(String),
    EnableParallelism(usize),
    ApplyStreaming,
    MaterializeSubquery,
    ReorderJoins(Vec<usize>),
    NoChange,
}

/// Accuracy metrics for model evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub mean_absolute_error: f64,
    pub root_mean_square_error: f64,
    pub r_squared: f64,
    pub confidence_interval: (f64, f64),
}
