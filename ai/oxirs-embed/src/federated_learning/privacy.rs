//! Privacy-preserving mechanisms for federated learning
//!
//! This module implements differential privacy, gradient clipping, noise mechanisms,
//! and privacy budget accounting for secure federated learning.

use super::config::{NoiseMechanism, PrivacyConfig};
use chrono::{DateTime, Utc};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Privacy engine for differential privacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyEngine {
    /// Privacy configuration
    pub config: PrivacyConfig,
    /// Privacy accountant
    pub privacy_accountant: PrivacyAccountant,
    /// Noise generator
    pub noise_generator: NoiseGenerator,
    /// Clipping mechanisms
    pub clipping_mechanisms: ClippingMechanisms,
}

impl PrivacyEngine {
    /// Create new privacy engine
    pub fn new(config: PrivacyConfig) -> Self {
        Self {
            privacy_accountant: PrivacyAccountant::new(config.epsilon, config.delta),
            noise_generator: NoiseGenerator::new(config.noise_mechanism.clone()),
            clipping_mechanisms: ClippingMechanisms::new(config.clipping_threshold),
            config,
        }
    }

    /// Process gradients with privacy mechanisms
    pub fn process_gradients(
        &mut self,
        gradients: &Array2<f32>,
        participant_id: Uuid,
    ) -> anyhow::Result<Array2<f32>> {
        // Clip gradients first
        let clipped_gradients = self.clipping_mechanisms.clip_gradients(gradients);
        
        // Add noise for differential privacy
        let noisy_gradients = if self.config.enable_differential_privacy {
            self.noise_generator.add_noise(&clipped_gradients)
        } else {
            clipped_gradients
        };

        // Update privacy budget
        let privacy_cost = self.calculate_privacy_cost(&noisy_gradients);
        self.privacy_accountant.consume_budget(participant_id, privacy_cost)?;

        Ok(noisy_gradients)
    }

    /// Calculate privacy cost for an operation
    fn calculate_privacy_cost(&self, _gradients: &Array2<f32>) -> f64 {
        // Simplified privacy cost calculation
        // In practice, this would be more sophisticated
        self.config.local_epsilon / 100.0
    }
}

/// Privacy budget accounting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyAccountant {
    /// Total epsilon budget
    pub total_epsilon: f64,
    /// Used epsilon budget
    pub used_epsilon: f64,
    /// Delta parameter
    pub delta: f64,
    /// Privacy budget per participant
    pub participant_budgets: HashMap<Uuid, f64>,
    /// Budget tracking per round
    pub round_budgets: Vec<f64>,
    /// Budget history
    pub budget_history: Vec<BudgetEntry>,
}

/// Budget entry for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetEntry {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Participant ID
    pub participant_id: Option<Uuid>,
    /// Privacy cost
    pub privacy_cost: f64,
    /// Operation type
    pub operation: String,
    /// Remaining budget
    pub remaining_budget: f64,
}

impl PrivacyAccountant {
    /// Create new privacy accountant
    pub fn new(total_epsilon: f64, delta: f64) -> Self {
        Self {
            total_epsilon,
            used_epsilon: 0.0,
            delta,
            participant_budgets: HashMap::new(),
            round_budgets: Vec::new(),
            budget_history: Vec::new(),
        }
    }

    /// Consume privacy budget
    pub fn consume_budget(&mut self, participant_id: Uuid, cost: f64) -> anyhow::Result<()> {
        if self.used_epsilon + cost > self.total_epsilon {
            return Err(anyhow::anyhow!("Privacy budget exceeded"));
        }

        self.used_epsilon += cost;
        *self.participant_budgets.entry(participant_id).or_insert(0.0) += cost;

        // Record budget entry
        self.budget_history.push(BudgetEntry {
            timestamp: Utc::now(),
            participant_id: Some(participant_id),
            privacy_cost: cost,
            operation: "gradient_update".to_string(),
            remaining_budget: self.total_epsilon - self.used_epsilon,
        });

        Ok(())
    }

    /// Get remaining budget
    pub fn remaining_budget(&self) -> f64 {
        self.total_epsilon - self.used_epsilon
    }

    /// Check if budget is available
    pub fn is_budget_available(&self, required_budget: f64) -> bool {
        self.remaining_budget() >= required_budget
    }
}

/// Noise generation for differential privacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseGenerator {
    /// Noise mechanism
    pub mechanism: NoiseMechanism,
    /// Noise scale
    pub scale: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl NoiseGenerator {
    /// Create new noise generator
    pub fn new(mechanism: NoiseMechanism) -> Self {
        Self {
            mechanism,
            scale: 1.0,
            seed: None,
        }
    }

    /// Set noise scale
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Add noise to parameters for differential privacy
    pub fn add_noise(&self, parameters: &Array2<f32>) -> Array2<f32> {
        match self.mechanism {
            NoiseMechanism::Gaussian => self.add_gaussian_noise(parameters),
            NoiseMechanism::Laplace => self.add_laplace_noise(parameters),
            NoiseMechanism::Exponential => self.add_exponential_noise(parameters),
            NoiseMechanism::SparseVector => self.add_sparse_vector_noise(parameters),
        }
    }

    /// Add Gaussian noise
    fn add_gaussian_noise(&self, parameters: &Array2<f32>) -> Array2<f32> {
        let noise = Array2::from_shape_fn(parameters.raw_dim(), |_| {
            // Box-Muller transform for Gaussian noise
            let u1: f32 = rand::random();
            let u2: f32 = rand::random();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            z * self.scale as f32
        });
        parameters + &noise
    }

    /// Add Laplace noise
    fn add_laplace_noise(&self, parameters: &Array2<f32>) -> Array2<f32> {
        let noise = Array2::from_shape_fn(parameters.raw_dim(), |_| {
            let u: f32 = rand::random::<f32>() - 0.5;
            let sign = if u > 0.0 { 1.0 } else { -1.0 };
            -sign * (1.0 - 2.0 * u.abs()).ln() * self.scale as f32
        });
        parameters + &noise
    }

    /// Add exponential mechanism noise (simplified)
    fn add_exponential_noise(&self, parameters: &Array2<f32>) -> Array2<f32> {
        // Simplified implementation - would need proper exponential mechanism
        self.add_laplace_noise(parameters)
    }

    /// Add sparse vector technique noise
    fn add_sparse_vector_noise(&self, parameters: &Array2<f32>) -> Array2<f32> {
        // Simplified implementation - would need proper sparse vector technique
        let mut result = self.add_gaussian_noise(parameters);
        
        // Apply sparsity by zeroing out small values
        result.mapv_inplace(|x| {
            if x.abs() < self.scale as f32 * 0.1 {
                0.0
            } else {
                x
            }
        });
        
        result
    }
}

/// Gradient clipping mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClippingMechanisms {
    /// Clipping threshold
    pub threshold: f64,
    /// Clipping method
    pub method: ClippingMethod,
    /// Adaptive clipping
    pub adaptive_clipping: bool,
    /// Adaptive threshold history
    pub threshold_history: Vec<f64>,
}

/// Gradient clipping methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClippingMethod {
    /// L2 norm clipping
    L2Norm,
    /// L1 norm clipping
    L1Norm,
    /// Element-wise clipping
    ElementWise,
    /// Adaptive clipping
    Adaptive,
}

impl ClippingMechanisms {
    /// Create new clipping mechanisms
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            method: ClippingMethod::L2Norm,
            adaptive_clipping: false,
            threshold_history: Vec::new(),
        }
    }

    /// Set clipping method
    pub fn with_method(mut self, method: ClippingMethod) -> Self {
        self.method = method;
        self
    }

    /// Enable adaptive clipping
    pub fn with_adaptive_clipping(mut self, adaptive: bool) -> Self {
        self.adaptive_clipping = adaptive;
        self
    }

    /// Clip gradients based on configured method
    pub fn clip_gradients(&mut self, gradients: &Array2<f32>) -> Array2<f32> {
        let result = match self.method {
            ClippingMethod::L2Norm => self.clip_l2_norm(gradients),
            ClippingMethod::L1Norm => self.clip_l1_norm(gradients),
            ClippingMethod::ElementWise => self.clip_element_wise(gradients),
            ClippingMethod::Adaptive => self.clip_adaptive(gradients),
        };

        // Update threshold history for adaptive clipping
        if self.adaptive_clipping {
            let current_norm = self.calculate_norm(gradients);
            self.threshold_history.push(current_norm);
            
            // Adapt threshold based on history (simplified)
            if self.threshold_history.len() > 10 {
                let avg_norm: f64 = self.threshold_history.iter().sum::<f64>() / self.threshold_history.len() as f64;
                self.threshold = avg_norm * 1.2; // Allow 20% above average
                self.threshold_history.remove(0); // Keep only recent history
            }
        }

        result
    }

    /// Calculate gradient norm
    fn calculate_norm(&self, gradients: &Array2<f32>) -> f64 {
        match self.method {
            ClippingMethod::L2Norm | ClippingMethod::Adaptive => {
                gradients.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt()
            }
            ClippingMethod::L1Norm => {
                gradients.iter().map(|x| (*x as f64).abs()).sum::<f64>()
            }
            ClippingMethod::ElementWise => {
                gradients.iter().map(|x| (*x as f64).abs()).fold(0.0, f64::max)
            }
        }
    }

    /// L2 norm clipping
    fn clip_l2_norm(&self, gradients: &Array2<f32>) -> Array2<f32> {
        let norm = gradients.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > self.threshold as f32 {
            gradients * (self.threshold as f32 / norm)
        } else {
            gradients.clone()
        }
    }

    /// L1 norm clipping
    fn clip_l1_norm(&self, gradients: &Array2<f32>) -> Array2<f32> {
        let norm = gradients.iter().map(|x| x.abs()).sum::<f32>();
        if norm > self.threshold as f32 {
            gradients * (self.threshold as f32 / norm)
        } else {
            gradients.clone()
        }
    }

    /// Element-wise clipping
    fn clip_element_wise(&self, gradients: &Array2<f32>) -> Array2<f32> {
        gradients.mapv(|x| x.max(-self.threshold as f32).min(self.threshold as f32))
    }

    /// Adaptive clipping
    fn clip_adaptive(&self, gradients: &Array2<f32>) -> Array2<f32> {
        // Use L2 norm clipping with adaptive threshold
        self.clip_l2_norm(gradients)
    }
}

/// Privacy parameters for different mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyParams {
    /// Epsilon value for differential privacy
    pub epsilon: f64,
    /// Delta value for differential privacy
    pub delta: f64,
    /// Sensitivity of the query/function
    pub sensitivity: f64,
    /// Composition method for privacy accounting
    pub composition_method: CompositionMethod,
}

/// Methods for privacy composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompositionMethod {
    /// Basic composition
    Basic,
    /// Advanced composition
    Advanced,
    /// Renyi differential privacy
    RenyiDP { alpha: f64 },
    /// Privacy loss distribution
    PLD,
    /// Gaussian differential privacy
    GDP,
}

/// Privacy accountant with advanced composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedPrivacyAccountant {
    /// Privacy parameters
    pub privacy_params: PrivacyParams,
    /// Composition tracking
    pub compositions: Vec<CompositionEntry>,
    /// Current privacy guarantees
    pub current_guarantees: PrivacyGuarantees,
}

/// Privacy composition entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionEntry {
    /// Operation timestamp
    pub timestamp: DateTime<Utc>,
    /// Privacy cost
    pub privacy_cost: (f64, f64), // (epsilon, delta)
    /// Mechanism used
    pub mechanism: String,
    /// Participant involved
    pub participant_id: Option<Uuid>,
}

/// Current privacy guarantees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyGuarantees {
    /// Total epsilon consumed
    pub total_epsilon: f64,
    /// Total delta consumed
    pub total_delta: f64,
    /// Worst-case privacy loss
    pub worst_case_loss: f64,
    /// Expected privacy loss
    pub expected_loss: f64,
    /// Confidence intervals
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}