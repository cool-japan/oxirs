//! ML Model Lifecycle Management
//!
//! This module manages ML model training, quality tracking, retraining,
//! and rollback capabilities for the query cost predictor.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::advanced_optimizer::ml_predictor::MLPredictor;
use crate::advanced_optimizer::training_collector::TrainingCollector;

/// Configuration for model manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerConfig {
    /// Retraining interval in hours
    pub retraining_interval_hours: u64,
    /// Minimum examples required before training
    pub min_examples_for_training: usize,
    /// Quality threshold for R² score
    pub quality_threshold_r2: f64,
    /// Quality threshold for MAE (as percentage)
    pub quality_threshold_mae: f64,
    /// Enable automatic retraining
    pub enable_auto_retraining: bool,
    /// Enable model rollback on quality degradation
    pub enable_rollback: bool,
    /// Maximum prediction tracking buffer size
    pub max_prediction_buffer: usize,
}

impl Default for ManagerConfig {
    fn default() -> Self {
        Self {
            retraining_interval_hours: 24,
            min_examples_for_training: 100,
            quality_threshold_r2: 0.8,
            quality_threshold_mae: 0.2, // 20%
            enable_auto_retraining: true,
            enable_rollback: true,
            max_prediction_buffer: 1000,
        }
    }
}

/// Model manager for ML predictor lifecycle
pub struct ModelManager {
    active_model: Arc<RwLock<MLPredictor>>,
    previous_model: Option<Arc<RwLock<MLPredictor>>>,
    training_collector: Option<Arc<RwLock<TrainingCollector>>>,
    config: ManagerConfig,
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    retraining_in_progress: Arc<AtomicBool>,
    last_retraining: Option<SystemTime>,
}

impl ModelManager {
    /// Create a new model manager
    pub fn new(predictor: MLPredictor, config: ManagerConfig) -> Self {
        let performance_tracker = Arc::new(RwLock::new(PerformanceTracker::new(
            config.max_prediction_buffer,
        )));

        Self {
            active_model: Arc::new(RwLock::new(predictor)),
            previous_model: None,
            training_collector: None,
            config,
            performance_tracker,
            retraining_in_progress: Arc::new(AtomicBool::new(false)),
            last_retraining: None,
        }
    }

    /// Create model manager with training collector
    pub fn with_training_collector(mut self, collector: Arc<RwLock<TrainingCollector>>) -> Self {
        self.training_collector = Some(collector);
        self
    }

    /// Get the active model
    pub fn get_predictor(&self) -> Arc<RwLock<MLPredictor>> {
        Arc::clone(&self.active_model)
    }

    /// Record a prediction result
    pub fn record_prediction(&self, predicted: f64, actual: f64) -> Result<()> {
        let mut tracker = self
            .performance_tracker
            .write()
            .map_err(|e| anyhow::anyhow!("Failed to acquire write lock: {}", e))?;

        tracker.record(PredictionResult {
            predicted_cost: predicted,
            actual_cost: Some(actual),
            timestamp: SystemTime::now(),
            error: Some((predicted - actual).abs()),
        });

        Ok(())
    }

    /// Evaluate current model quality
    pub fn evaluate_model_quality(&self) -> Result<ModelQuality> {
        let tracker = self
            .performance_tracker
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;

        let quality = tracker.calculate_quality();

        Ok(quality)
    }

    /// Check if ML model should be used
    pub fn should_use_ml(&self) -> bool {
        if let Ok(quality) = self.evaluate_model_quality() {
            quality.is_acceptable
        } else {
            false
        }
    }

    /// Check if model should be retrained
    pub fn should_retrain(&self) -> bool {
        if !self.config.enable_auto_retraining {
            return false;
        }

        // Don't retrain if already in progress
        if self.retraining_in_progress.load(Ordering::Relaxed) {
            return false;
        }

        // Check if enough time has passed
        if let Some(last_training) = self.last_retraining {
            if let Ok(elapsed) = SystemTime::now().duration_since(last_training) {
                let hours_elapsed = elapsed.as_secs() / 3600;
                if hours_elapsed < self.config.retraining_interval_hours {
                    return false;
                }
            }
        }

        // Check if we have enough training data
        if let Some(ref collector) = self.training_collector {
            if let Ok(collector_guard) = collector.read() {
                if collector_guard.len() < self.config.min_examples_for_training {
                    return false;
                }
            }
        }

        true
    }

    /// Trigger model retraining
    pub fn trigger_retraining(&mut self) -> Result<()> {
        // Set retraining flag
        if self
            .retraining_in_progress
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(anyhow::anyhow!("Retraining already in progress"));
        }

        // Perform retraining
        let result = self.retrain_internal();

        // Clear retraining flag
        self.retraining_in_progress.store(false, Ordering::SeqCst);

        result
    }

    /// Internal retraining implementation
    fn retrain_internal(&mut self) -> Result<()> {
        // Get training data
        let training_collector = self
            .training_collector
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("No training collector available"))?;

        let examples = {
            let collector = training_collector
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;
            collector.get_all_examples()?
        };

        if examples.len() < self.config.min_examples_for_training {
            return Err(anyhow::anyhow!(
                "Insufficient training examples: {} < {}",
                examples.len(),
                self.config.min_examples_for_training
            ));
        }

        // Save current model as previous (for rollback)
        if self.config.enable_rollback {
            let current = self
                .active_model
                .read()
                .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;

            self.previous_model = Some(Arc::new(RwLock::new(current.clone())));
        }

        // Get current quality before retraining
        let old_quality = self.evaluate_model_quality()?;

        // Train new model
        {
            let mut model = self
                .active_model
                .write()
                .map_err(|e| anyhow::anyhow!("Failed to acquire write lock: {}", e))?;

            // Add training examples to model
            for example in examples {
                model.add_training_example(example);
            }

            // Train
            model.train_model().context("Failed to train model")?;
        }

        // Evaluate new model quality
        let new_quality = self.evaluate_model_quality()?;

        // Check if new model is better
        if self.config.enable_rollback && new_quality.r_squared < old_quality.r_squared {
            tracing::warn!(
                "New model quality degraded (R²: {} → {}). Rolling back.",
                old_quality.r_squared,
                new_quality.r_squared
            );
            self.rollback_to_previous()?;
        } else {
            tracing::info!(
                "Model retrained successfully. R²: {} → {}, MAE: {} → {}",
                old_quality.r_squared,
                new_quality.r_squared,
                old_quality.mae,
                new_quality.mae
            );
        }

        // Update last retraining time
        self.last_retraining = Some(SystemTime::now());

        Ok(())
    }

    /// Rollback to previous model
    pub fn rollback_to_previous(&mut self) -> Result<()> {
        let previous = self
            .previous_model
            .take()
            .ok_or_else(|| anyhow::anyhow!("No previous model available for rollback"))?;

        self.active_model = previous;

        tracing::info!("Rolled back to previous model");

        Ok(())
    }

    /// Save model checkpoint
    pub fn save_checkpoint(&self, path: &Path) -> Result<()> {
        let model = self
            .active_model
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;

        model
            .save_model(path)
            .context("Failed to save model checkpoint")?;

        Ok(())
    }

    /// Load model from checkpoint
    pub fn load_checkpoint(path: &Path, config: ManagerConfig) -> Result<Self> {
        let predictor =
            MLPredictor::load_model(path).context("Failed to load model from checkpoint")?;

        Ok(Self::new(predictor, config))
    }

    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> Result<PerformanceMetrics> {
        let tracker = self
            .performance_tracker
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;

        let model = self
            .active_model
            .read()
            .map_err(|e| anyhow::anyhow!("Failed to acquire read lock: {}", e))?;

        Ok(PerformanceMetrics {
            predictions_made: model.predictions_count(),
            training_examples: model.training_data_count(),
            mae: tracker.mae,
            rmse: tracker.rmse,
            r_squared: tracker.r_squared,
            is_using_ml: self.should_use_ml(),
        })
    }
}

/// Performance tracker for model predictions
pub struct PerformanceTracker {
    predictions: VecDeque<PredictionResult>,
    max_buffer: usize,
    pub mae: f64,
    pub rmse: f64,
    pub r_squared: f64,
    last_update: SystemTime,
}

impl PerformanceTracker {
    /// Create a new performance tracker
    pub fn new(max_buffer: usize) -> Self {
        Self {
            predictions: VecDeque::with_capacity(max_buffer.min(1000)),
            max_buffer,
            mae: 0.0,
            rmse: 0.0,
            r_squared: 0.0,
            last_update: SystemTime::now(),
        }
    }

    /// Record a prediction result
    pub fn record(&mut self, result: PredictionResult) {
        self.predictions.push_back(result);

        // Remove oldest if over capacity
        if self.predictions.len() > self.max_buffer {
            self.predictions.pop_front();
        }

        // Recalculate metrics
        self.update_metrics();
    }

    /// Update performance metrics
    pub fn update_metrics(&mut self) {
        let valid_predictions: Vec<&PredictionResult> = self
            .predictions
            .iter()
            .filter(|p| p.actual_cost.is_some())
            .collect();

        if valid_predictions.is_empty() {
            return;
        }

        let n = valid_predictions.len() as f64;

        // Calculate MAE
        let total_error: f64 = valid_predictions.iter().filter_map(|p| p.error).sum();
        self.mae = total_error / n;

        // Calculate RMSE
        let squared_errors: f64 = valid_predictions
            .iter()
            .filter_map(|p| p.error.map(|e| e * e))
            .sum();
        self.rmse = (squared_errors / n).sqrt();

        // Calculate R²
        let mean_actual: f64 = valid_predictions
            .iter()
            .filter_map(|p| p.actual_cost)
            .sum::<f64>()
            / n;

        let ss_tot: f64 = valid_predictions
            .iter()
            .filter_map(|p| p.actual_cost.map(|a| (a - mean_actual).powi(2)))
            .sum();

        let ss_res: f64 = valid_predictions
            .iter()
            .filter_map(|p| {
                if let (Some(_actual), Some(error)) = (p.actual_cost, p.error) {
                    Some(error.powi(2))
                } else {
                    None
                }
            })
            .sum();

        self.r_squared = if ss_tot > 1e-10 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        self.last_update = SystemTime::now();
    }

    /// Calculate model quality
    pub fn calculate_quality(&self) -> ModelQuality {
        let is_acceptable = self.r_squared >= 0.8 && self.mae <= 0.2;

        let recommendation = if self.r_squared < 0.5 {
            QualityRecommendation::UseFallback
        } else if self.r_squared < 0.8 {
            QualityRecommendation::NeedsRetraining
        } else {
            QualityRecommendation::UseMl
        };

        ModelQuality {
            r_squared: self.r_squared,
            mae: self.mae,
            rmse: self.rmse,
            is_acceptable,
            recommendation,
        }
    }

    /// Get number of tracked predictions
    pub fn prediction_count(&self) -> usize {
        self.predictions.len()
    }
}

/// Prediction result for tracking
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub predicted_cost: f64,
    pub actual_cost: Option<f64>,
    pub timestamp: SystemTime,
    pub error: Option<f64>,
}

/// Model quality assessment
#[derive(Debug, Clone)]
pub struct ModelQuality {
    pub r_squared: f64,
    pub mae: f64,
    pub rmse: f64,
    pub is_acceptable: bool,
    pub recommendation: QualityRecommendation,
}

/// Quality-based recommendation
#[derive(Debug, Clone, PartialEq)]
pub enum QualityRecommendation {
    /// Use ML predictor (high quality)
    UseMl,
    /// Fall back to heuristic (poor quality)
    UseFallback,
    /// Model needs retraining (degraded quality)
    NeedsRetraining,
}

/// Performance metrics for monitoring
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub predictions_made: usize,
    pub training_examples: usize,
    pub mae: f64,
    pub rmse: f64,
    pub r_squared: f64,
    pub is_using_ml: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::advanced_optimizer::ml_predictor::MLModelType;

    #[test]
    fn test_model_manager_creation() -> Result<()> {
        let predictor = MLPredictor::from_model_type(MLModelType::LinearRegression)?;
        let config = ManagerConfig::default();
        let manager = ModelManager::new(predictor, config);

        assert!(!manager.should_retrain()); // No training data yet

        Ok(())
    }

    #[test]
    fn test_performance_tracker() {
        let mut tracker = PerformanceTracker::new(10);

        // Add some predictions
        for i in 1..=5 {
            let predicted = i as f64 * 10.0;
            let actual = i as f64 * 10.0 + 5.0; // Error of 5.0

            tracker.record(PredictionResult {
                predicted_cost: predicted,
                actual_cost: Some(actual),
                timestamp: SystemTime::now(),
                error: Some(5.0),
            });
        }

        assert_eq!(tracker.prediction_count(), 5);
        assert!((tracker.mae - 5.0).abs() < 1e-6); // MAE should be 5.0
    }

    #[test]
    fn test_model_quality_assessment() {
        let mut tracker = PerformanceTracker::new(10);

        // Add perfect predictions
        for i in 1..=10 {
            let cost = i as f64 * 10.0;
            tracker.record(PredictionResult {
                predicted_cost: cost,
                actual_cost: Some(cost),
                timestamp: SystemTime::now(),
                error: Some(0.0),
            });
        }

        let quality = tracker.calculate_quality();
        assert_eq!(quality.mae, 0.0);
        assert_eq!(quality.rmse, 0.0);
        assert!(quality.is_acceptable);
        assert_eq!(quality.recommendation, QualityRecommendation::UseMl);
    }

    #[test]
    fn test_quality_recommendation_poor() {
        let mut tracker = PerformanceTracker::new(10);

        // Add predictions with large errors
        for i in 1..=5 {
            let predicted = i as f64 * 10.0;
            let actual = i as f64 * 50.0; // Large difference

            tracker.record(PredictionResult {
                predicted_cost: predicted,
                actual_cost: Some(actual),
                timestamp: SystemTime::now(),
                error: Some((predicted - actual).abs()),
            });
        }

        let quality = tracker.calculate_quality();
        assert!(!quality.is_acceptable);
        // R² should be very low or negative
        assert!(quality.r_squared < 0.8);
    }

    #[test]
    fn test_buffer_limit() {
        let max_buffer = 5;
        let mut tracker = PerformanceTracker::new(max_buffer);

        // Add more predictions than buffer size
        for i in 1..=10 {
            tracker.record(PredictionResult {
                predicted_cost: i as f64,
                actual_cost: Some(i as f64),
                timestamp: SystemTime::now(),
                error: Some(0.0),
            });
        }

        // Should only keep last 5
        assert_eq!(tracker.prediction_count(), max_buffer);
    }

    #[test]
    fn test_record_prediction() -> Result<()> {
        let predictor = MLPredictor::from_model_type(MLModelType::LinearRegression)?;
        let config = ManagerConfig::default();
        let manager = ModelManager::new(predictor, config);

        manager.record_prediction(100.0, 105.0)?;
        manager.record_prediction(200.0, 195.0)?;

        let metrics = manager.get_performance_metrics()?;
        assert_eq!(metrics.predictions_made, 0); // Predictor hasn't been used yet

        Ok(())
    }
}
