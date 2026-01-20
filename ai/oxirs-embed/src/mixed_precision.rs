//! Mixed Precision Training for Knowledge Graph Embeddings
//!
//! This module provides mixed precision training support to accelerate training
//! and reduce memory usage while maintaining numerical stability. Uses float16
//! for forward/backward passes and float32 for parameter updates.

use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Mixed precision training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    /// Enable mixed precision training
    pub enabled: bool,
    /// Initial loss scale factor
    pub init_scale: f32,
    /// Scale factor growth rate
    pub scale_growth_factor: f32,
    /// Backoff factor when overflow detected
    pub scale_backoff_factor: f32,
    /// Number of successful steps before increasing scale
    pub scale_growth_interval: usize,
    /// Use dynamic loss scaling
    pub dynamic_loss_scale: bool,
    /// Gradient clipping threshold
    pub grad_clip_threshold: f32,
    /// Enable gradient accumulation
    pub gradient_accumulation: bool,
    /// Number of steps to accumulate gradients
    pub accumulation_steps: usize,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            init_scale: 65536.0, // 2^16
            scale_growth_factor: 2.0,
            scale_backoff_factor: 0.5,
            scale_growth_interval: 2000,
            dynamic_loss_scale: true,
            grad_clip_threshold: 1.0,
            gradient_accumulation: false,
            accumulation_steps: 1,
        }
    }
}

/// Mixed precision training statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionStats {
    /// Current loss scale
    pub current_scale: f32,
    /// Number of overflow events
    pub num_overflows: usize,
    /// Number of successful steps
    pub num_successful_steps: usize,
    /// Number of scale updates
    pub num_scale_updates: usize,
    /// Average gradient norm
    pub avg_gradient_norm: f32,
    /// Memory saved (estimated, in bytes)
    pub memory_saved_bytes: usize,
}

impl Default for MixedPrecisionStats {
    fn default() -> Self {
        Self {
            current_scale: 1.0,
            num_overflows: 0,
            num_successful_steps: 0,
            num_scale_updates: 0,
            avg_gradient_norm: 0.0,
            memory_saved_bytes: 0,
        }
    }
}

/// Mixed precision trainer for embeddings
pub struct MixedPrecisionTrainer {
    config: MixedPrecisionConfig,
    stats: MixedPrecisionStats,
    steps_since_overflow: usize,
    accumulated_gradients: HashMap<String, Array1<f32>>,
    accumulation_count: usize,
}

impl MixedPrecisionTrainer {
    /// Create new mixed precision trainer
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let initial_scale = if config.enabled {
            config.init_scale
        } else {
            1.0
        };

        info!(
            "Initialized mixed precision trainer: enabled={}, init_scale={}",
            config.enabled, initial_scale
        );

        Self {
            config,
            stats: MixedPrecisionStats {
                current_scale: initial_scale,
                ..Default::default()
            },
            steps_since_overflow: 0,
            accumulated_gradients: HashMap::new(),
            accumulation_count: 0,
        }
    }

    /// Convert float32 tensor to float16 (simulated with f32)
    ///
    /// Note: Rust doesn't have native float16, so we simulate by clamping range
    pub fn to_fp16(&self, tensor: &Array1<f32>) -> Array1<f32> {
        if !self.config.enabled {
            return tensor.clone();
        }

        // Simulate FP16 range: approximately [-65504, 65504]
        const FP16_MAX: f32 = 65504.0;
        const FP16_MIN: f32 = -65504.0;

        tensor.mapv(|x| x.clamp(FP16_MIN, FP16_MAX))
    }

    /// Convert float16 back to float32 (no-op in simulation)
    pub fn to_fp32(&self, tensor: &Array1<f32>) -> Array1<f32> {
        tensor.clone()
    }

    /// Scale loss for backward pass
    pub fn scale_loss(&self, loss: f32) -> f32 {
        if !self.config.enabled {
            return loss;
        }

        loss * self.stats.current_scale
    }

    /// Unscale gradients after backward pass
    pub fn unscale_gradients(&self, gradients: &Array1<f32>) -> Result<Array1<f32>> {
        if !self.config.enabled {
            return Ok(gradients.clone());
        }

        // Check for overflow/underflow
        if self.has_inf_or_nan(gradients) {
            return Err(anyhow!("Gradient overflow detected"));
        }

        // Unscale
        let unscaled = gradients / self.stats.current_scale;

        // Gradient clipping
        let grad_norm = self.compute_gradient_norm(&unscaled);

        if grad_norm > self.config.grad_clip_threshold {
            let scale_factor = self.config.grad_clip_threshold / grad_norm;
            Ok(&unscaled * scale_factor)
        } else {
            Ok(unscaled)
        }
    }

    /// Update parameters with mixed precision
    pub fn update_parameters(
        &mut self,
        parameters: &mut Array1<f32>,
        gradients: &Array1<f32>,
        learning_rate: f32,
    ) -> Result<()> {
        if !self.config.enabled {
            // Standard update
            *parameters = &*parameters - &(gradients * learning_rate);
            return Ok(());
        }

        // Unscale gradients
        let unscaled_grads = match self.unscale_gradients(gradients) {
            Ok(grads) => grads,
            Err(_) => {
                self.handle_overflow();
                return Ok(()); // Skip this update
            }
        };

        if self.config.gradient_accumulation {
            // Accumulate gradients
            let param_key = format!("{:p}", parameters);

            let accumulated = self
                .accumulated_gradients
                .entry(param_key)
                .or_insert_with(|| Array1::zeros(parameters.len()));

            *accumulated = &*accumulated + &unscaled_grads;
            self.accumulation_count += 1;

            // Only update when we've accumulated enough
            if self.accumulation_count >= self.config.accumulation_steps {
                let avg_grad = &*accumulated / (self.config.accumulation_steps as f32);

                // Update in FP32
                *parameters = &*parameters - &(&avg_grad * learning_rate);

                // Reset accumulation
                self.accumulated_gradients.clear();
                self.accumulation_count = 0;

                self.on_successful_step();
            }
        } else {
            // Direct update in FP32
            *parameters = &*parameters - &(&unscaled_grads * learning_rate);

            self.on_successful_step();
        }

        Ok(())
    }

    /// Handle gradient overflow
    fn handle_overflow(&mut self) {
        self.stats.num_overflows += 1;
        self.steps_since_overflow = 0;

        if self.config.dynamic_loss_scale {
            self.stats.current_scale *= self.config.scale_backoff_factor;
            self.stats.num_scale_updates += 1;

            warn!(
                "Gradient overflow detected! Reducing loss scale to {}",
                self.stats.current_scale
            );
        }
    }

    /// Called after successful parameter update
    fn on_successful_step(&mut self) {
        self.stats.num_successful_steps += 1;
        self.steps_since_overflow += 1;

        // Increase scale if we've had many successful steps
        if self.config.dynamic_loss_scale
            && self.steps_since_overflow >= self.config.scale_growth_interval
        {
            self.stats.current_scale *= self.config.scale_growth_factor;
            self.stats.num_scale_updates += 1;
            self.steps_since_overflow = 0;

            debug!(
                "Increasing loss scale to {} after {} successful steps",
                self.stats.current_scale, self.config.scale_growth_interval
            );
        }
    }

    /// Check if tensor contains inf or nan
    fn has_inf_or_nan(&self, tensor: &Array1<f32>) -> bool {
        tensor.iter().any(|&x| x.is_infinite() || x.is_nan())
    }

    /// Compute gradient norm
    fn compute_gradient_norm(&self, gradients: &Array1<f32>) -> f32 {
        gradients.dot(gradients).sqrt()
    }

    /// Get current statistics
    pub fn get_stats(&self) -> &MixedPrecisionStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = MixedPrecisionStats {
            current_scale: self.config.init_scale,
            ..Default::default()
        };
        self.steps_since_overflow = 0;
    }

    /// Estimate memory savings
    pub fn estimate_memory_savings(&mut self, num_parameters: usize) {
        // FP16 uses 2 bytes vs FP32's 4 bytes
        if self.config.enabled {
            self.stats.memory_saved_bytes = num_parameters * 2;
        } else {
            self.stats.memory_saved_bytes = 0;
        }
    }

    /// Update average gradient norm
    pub fn update_gradient_stats(&mut self, gradients: &Array1<f32>) {
        let norm = self.compute_gradient_norm(gradients);
        let n = self.stats.num_successful_steps as f32;

        if n > 0.0 {
            self.stats.avg_gradient_norm = (self.stats.avg_gradient_norm * (n - 1.0) + norm) / n;
        } else {
            self.stats.avg_gradient_norm = norm;
        }
    }

    /// Check if training is stable
    pub fn is_stable(&self) -> bool {
        if !self.config.enabled {
            return true;
        }

        // Consider training unstable if too many overflows
        let overflow_rate =
            self.stats.num_overflows as f32 / (self.stats.num_successful_steps + 1) as f32;

        overflow_rate < 0.1 // Less than 10% overflow rate
    }

    /// Get configuration
    pub fn config(&self) -> &MixedPrecisionConfig {
        &self.config
    }
}

/// Helper trait for mixed precision operations on embeddings
pub trait MixedPrecisionEmbedding {
    /// Convert embeddings to mixed precision format
    fn to_mixed_precision(&self, trainer: &MixedPrecisionTrainer) -> Self;

    /// Convert back to full precision
    fn to_full_precision(&self, trainer: &MixedPrecisionTrainer) -> Self;
}

impl MixedPrecisionEmbedding for Array1<f32> {
    fn to_mixed_precision(&self, trainer: &MixedPrecisionTrainer) -> Self {
        trainer.to_fp16(self)
    }

    fn to_full_precision(&self, trainer: &MixedPrecisionTrainer) -> Self {
        trainer.to_fp32(self)
    }
}

impl MixedPrecisionEmbedding for HashMap<String, Array1<f32>> {
    fn to_mixed_precision(&self, trainer: &MixedPrecisionTrainer) -> Self {
        self.iter()
            .map(|(k, v)| (k.clone(), trainer.to_fp16(v)))
            .collect()
    }

    fn to_full_precision(&self, trainer: &MixedPrecisionTrainer) -> Self {
        self.iter()
            .map(|(k, v)| (k.clone(), trainer.to_fp32(v)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray_ext::array;

    #[test]
    fn test_mixed_precision_creation() {
        let config = MixedPrecisionConfig::default();
        let trainer = MixedPrecisionTrainer::new(config);

        assert_eq!(trainer.stats.current_scale, 65536.0);
        assert_eq!(trainer.stats.num_overflows, 0);
    }

    #[test]
    fn test_fp16_conversion() {
        let config = MixedPrecisionConfig::default();
        let trainer = MixedPrecisionTrainer::new(config);

        let tensor = array![1.0, 2.0, 3.0];
        let fp16 = trainer.to_fp16(&tensor);
        let fp32 = trainer.to_fp32(&fp16);

        assert_eq!(tensor.len(), fp32.len());
    }

    #[test]
    fn test_loss_scaling() {
        let config = MixedPrecisionConfig {
            enabled: true,
            init_scale: 1024.0,
            ..Default::default()
        };

        let trainer = MixedPrecisionTrainer::new(config);

        let loss = 0.5;
        let scaled_loss = trainer.scale_loss(loss);

        assert_eq!(scaled_loss, 512.0);
    }

    #[test]
    fn test_gradient_unscaling() {
        let config = MixedPrecisionConfig {
            enabled: true,
            init_scale: 1024.0,
            grad_clip_threshold: 10.0,
            ..Default::default()
        };

        let trainer = MixedPrecisionTrainer::new(config);

        let scaled_grads = array![1024.0, 2048.0, 512.0];
        let unscaled = trainer.unscale_gradients(&scaled_grads).unwrap();

        // Should be divided by scale (1024.0)
        assert!((unscaled[0] - 1.0).abs() < 1e-5);
        assert!((unscaled[1] - 2.0).abs() < 1e-5);
        assert!((unscaled[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_gradient_clipping() {
        let config = MixedPrecisionConfig {
            enabled: true,
            init_scale: 1.0,
            grad_clip_threshold: 1.0,
            ..Default::default()
        };

        let trainer = MixedPrecisionTrainer::new(config.clone());

        // Large gradients that exceed threshold
        let grads = array![10.0, 10.0, 10.0];
        let clipped = trainer.unscale_gradients(&grads).unwrap();

        let norm = clipped.dot(&clipped).sqrt();
        assert!(norm <= config.grad_clip_threshold + 1e-5);
    }

    #[test]
    fn test_overflow_handling() {
        let config = MixedPrecisionConfig {
            enabled: true,
            init_scale: 1024.0,
            dynamic_loss_scale: true,
            scale_backoff_factor: 0.5,
            ..Default::default()
        };

        let mut trainer = MixedPrecisionTrainer::new(config.clone());

        // Simulate overflow with inf gradients
        let bad_grads = array![f32::INFINITY, 1.0, 2.0];

        let result = trainer.unscale_gradients(&bad_grads);
        assert!(result.is_err());

        // Manually handle overflow
        trainer.handle_overflow();

        // Scale should be reduced
        assert_eq!(trainer.stats.current_scale, 512.0);
        assert_eq!(trainer.stats.num_overflows, 1);
    }

    #[test]
    fn test_parameter_update() {
        let config = MixedPrecisionConfig {
            enabled: true,
            init_scale: 1.0,
            ..Default::default()
        };

        let mut trainer = MixedPrecisionTrainer::new(config);

        let mut params = array![1.0, 2.0, 3.0];
        let grads = array![0.1, 0.2, 0.3];
        let lr = 0.1;

        trainer.update_parameters(&mut params, &grads, lr).unwrap();

        // params should be updated: params -= lr * grads
        assert!((params[0] - 0.99).abs() < 1e-5);
        assert!((params[1] - 1.98).abs() < 1e-5);
        assert!((params[2] - 2.97).abs() < 1e-5);
    }

    #[test]
    fn test_stability_check() {
        let config = MixedPrecisionConfig::default();
        let mut trainer = MixedPrecisionTrainer::new(config);

        trainer.stats.num_successful_steps = 100;
        trainer.stats.num_overflows = 5; // 5% overflow rate

        assert!(trainer.is_stable());

        trainer.stats.num_overflows = 15; // 15% overflow rate
        assert!(!trainer.is_stable());
    }
}
