//! Uncertainty quantification for neural cost estimation

use ndarray::{Array1, Array2};

use super::{config::*, types::*};
use crate::{Result, ShaclAiError};

/// Uncertainty quantifier
#[derive(Debug)]
pub struct UncertaintyQuantifier {
    /// Configuration
    config: UncertaintyConfig,

    /// Bootstrap samples storage
    bootstrap_samples: Vec<f64>,

    /// Uncertainty model
    uncertainty_model: UncertaintyModel,

    /// Statistics
    uncertainty_stats: UncertaintyStatistics,
}

/// Uncertainty model
#[derive(Debug)]
pub struct UncertaintyModel {
    /// Model parameters
    parameters: Array1<f64>,

    /// Confidence intervals cache
    confidence_intervals: Vec<ConfidenceInterval>,

    /// Historical uncertainty data
    historical_uncertainties: Vec<UncertaintyRecord>,
}

/// Confidence interval
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    pub confidence_level: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub point_estimate: f64,
}

/// Uncertainty record
#[derive(Debug, Clone)]
pub struct UncertaintyRecord {
    pub predicted_uncertainty: f64,
    pub actual_error: f64,
    pub features: Array1<f64>,
    pub timestamp: std::time::SystemTime,
}

/// Uncertainty statistics
#[derive(Debug, Clone)]
pub struct UncertaintyStatistics {
    pub total_predictions: usize,
    pub average_uncertainty: f64,
    pub uncertainty_calibration: f64,
    pub confidence_accuracy: f64,
}

impl UncertaintyQuantifier {
    pub fn new(config: UncertaintyConfig) -> Self {
        Self {
            config,
            bootstrap_samples: Vec::new(),
            uncertainty_model: UncertaintyModel::new(),
            uncertainty_stats: UncertaintyStatistics::default(),
        }
    }

    /// Quantify uncertainty for a prediction
    pub fn quantify_uncertainty(
        &mut self,
        base_prediction: &CostPrediction,
        features: &Array1<f64>,
    ) -> Result<CostPrediction> {
        if !self.config.enable_uncertainty {
            return Ok(base_prediction.clone());
        }

        let uncertainty = match self.config.uncertainty_method {
            UncertaintyMethod::Bootstrap => {
                self.bootstrap_uncertainty(base_prediction, features)?
            }
            UncertaintyMethod::Bayesian => self.bayesian_uncertainty(base_prediction, features)?,
            UncertaintyMethod::Ensemble => self.ensemble_uncertainty(base_prediction, features)?,
            UncertaintyMethod::Dropout => self.dropout_uncertainty(base_prediction, features)?,
            UncertaintyMethod::Gaussian => self.gaussian_uncertainty(base_prediction, features)?,
        };

        let confidence_intervals =
            self.calculate_confidence_intervals(base_prediction.estimated_cost, uncertainty)?;

        let mut enhanced_prediction = base_prediction.clone();
        enhanced_prediction.uncertainty = uncertainty;
        enhanced_prediction.confidence = 1.0 - uncertainty.min(1.0);

        // Add confidence intervals to contributing factors
        for interval in confidence_intervals {
            enhanced_prediction
                .contributing_factors
                .push(ContributingFactor {
                    factor_type: FactorType::PatternComplexity, // Placeholder
                    importance: interval.confidence_level,
                    description: format!(
                        "{}% CI: [{:.3}, {:.3}]",
                        interval.confidence_level * 100.0,
                        interval.lower_bound,
                        interval.upper_bound
                    ),
                });
        }

        // Update statistics
        self.update_uncertainty_stats(uncertainty);

        Ok(enhanced_prediction)
    }

    fn bootstrap_uncertainty(
        &mut self,
        prediction: &CostPrediction,
        features: &Array1<f64>,
    ) -> Result<f64> {
        // Generate bootstrap samples
        let num_samples = self.config.bootstrap_samples;
        let mut samples = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            // Add noise to features for bootstrap sampling
            let noisy_features = self.add_noise_to_features(features, 0.1);

            // Simple bootstrap estimate (in practice, this would retrain models)
            let sample_prediction = prediction.estimated_cost * (0.8 + 0.4 * fastrand::f64());
            samples.push(sample_prediction);
        }

        // Calculate uncertainty as standard deviation of samples
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;

        self.bootstrap_samples = samples;
        Ok(variance.sqrt() / mean) // Coefficient of variation
    }

    fn bayesian_uncertainty(
        &self,
        prediction: &CostPrediction,
        _features: &Array1<f64>,
    ) -> Result<f64> {
        // Simplified Bayesian uncertainty
        // In practice, this would use proper Bayesian inference
        let epistemic_uncertainty = 0.1 * prediction.estimated_cost;
        let aleatoric_uncertainty = 0.05 * prediction.estimated_cost;

        Ok(
            (epistemic_uncertainty.powi(2) + aleatoric_uncertainty.powi(2)).sqrt()
                / prediction.estimated_cost,
        )
    }

    fn ensemble_uncertainty(
        &self,
        prediction: &CostPrediction,
        _features: &Array1<f64>,
    ) -> Result<f64> {
        // Use existing prediction uncertainty as ensemble uncertainty
        Ok(prediction.uncertainty)
    }

    fn dropout_uncertainty(
        &self,
        prediction: &CostPrediction,
        _features: &Array1<f64>,
    ) -> Result<f64> {
        // Simplified Monte Carlo dropout
        let num_samples = 50;
        let mut predictions = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            // Simulate dropout by scaling prediction randomly
            let dropout_prediction = prediction.estimated_cost * (0.7 + 0.6 * fastrand::f64());
            predictions.push(dropout_prediction);
        }

        let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance =
            predictions.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / predictions.len() as f64;

        Ok(variance.sqrt() / mean)
    }

    fn gaussian_uncertainty(
        &self,
        prediction: &CostPrediction,
        features: &Array1<f64>,
    ) -> Result<f64> {
        // Simple Gaussian uncertainty based on feature complexity
        let feature_complexity =
            features.iter().map(|x| x.abs()).sum::<f64>() / features.len() as f64;
        let base_uncertainty = 0.05 + 0.1 * feature_complexity.min(1.0);

        Ok(base_uncertainty)
    }

    fn calculate_confidence_intervals(
        &self,
        point_estimate: f64,
        uncertainty: f64,
    ) -> Result<Vec<ConfidenceInterval>> {
        let mut intervals = Vec::new();

        for &confidence_level in &self.config.confidence_levels {
            // Calculate z-score for confidence level (simplified)
            let z_score = match confidence_level {
                0.95 => 1.96,
                0.99 => 2.576,
                0.90 => 1.645,
                _ => 1.96, // Default to 95%
            };

            let margin = z_score * uncertainty * point_estimate;

            intervals.push(ConfidenceInterval {
                confidence_level,
                lower_bound: (point_estimate - margin).max(0.0),
                upper_bound: point_estimate + margin,
                point_estimate,
            });
        }

        Ok(intervals)
    }

    fn add_noise_to_features(&self, features: &Array1<f64>, noise_level: f64) -> Array1<f64> {
        features.map(|&x| x + noise_level * (fastrand::f64() - 0.5) * 2.0)
    }

    fn update_uncertainty_stats(&mut self, uncertainty: f64) {
        self.uncertainty_stats.total_predictions += 1;

        let n = self.uncertainty_stats.total_predictions as f64;
        self.uncertainty_stats.average_uncertainty =
            (self.uncertainty_stats.average_uncertainty * (n - 1.0) + uncertainty) / n;
    }

    /// Update uncertainty model with actual error
    pub fn update_with_actual_error(
        &mut self,
        predicted_uncertainty: f64,
        actual_error: f64,
        features: &Array1<f64>,
    ) -> Result<()> {
        let record = UncertaintyRecord {
            predicted_uncertainty,
            actual_error,
            features: features.clone(),
            timestamp: std::time::SystemTime::now(),
        };

        self.uncertainty_model.historical_uncertainties.push(record);

        // Update calibration statistics
        self.update_calibration_stats(predicted_uncertainty, actual_error);

        // Keep only recent records
        if self.uncertainty_model.historical_uncertainties.len() > 1000 {
            self.uncertainty_model
                .historical_uncertainties
                .drain(0..100);
        }

        Ok(())
    }

    fn update_calibration_stats(&mut self, predicted_uncertainty: f64, actual_error: f64) {
        // Simple calibration metric
        let calibration_error = (predicted_uncertainty - actual_error).abs();
        let n = self.uncertainty_stats.total_predictions as f64;

        self.uncertainty_stats.uncertainty_calibration =
            (self.uncertainty_stats.uncertainty_calibration * (n - 1.0) + calibration_error) / n;
    }

    /// Get uncertainty statistics
    pub fn get_stats(&self) -> &UncertaintyStatistics {
        &self.uncertainty_stats
    }

    /// Get bootstrap samples
    pub fn get_bootstrap_samples(&self) -> &[f64] {
        &self.bootstrap_samples
    }
}

impl UncertaintyModel {
    pub fn new() -> Self {
        Self {
            parameters: Array1::zeros(10), // Default parameter size
            confidence_intervals: Vec::new(),
            historical_uncertainties: Vec::new(),
        }
    }
}

impl Default for UncertaintyStatistics {
    fn default() -> Self {
        Self {
            total_predictions: 0,
            average_uncertainty: 0.0,
            uncertainty_calibration: 0.0,
            confidence_accuracy: 0.0,
        }
    }
}
