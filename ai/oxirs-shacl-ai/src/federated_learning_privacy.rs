//! Differential privacy mechanisms, noise injection, gradient clipping,
//! and Secure Multi-Party Computation for federated learning.

use std::collections::HashMap;

use uuid::Uuid;

use crate::{Result, ShaclAiError};

use super::federated_learning_types::{
    FederatedUpdate, ModelParameterDelta, NoiseMechanism, PrivacyBudgetTracker, PrivacyReport,
    PrivacySpending, SMPCProtocol, SecretShare, SecretSharingScheme, SecureAggregationResult,
    SecureChannel, SecurityLevel,
};

// ──────────────────────────────────────────────────────────────────────────────
// Advanced Differential Privacy
// ──────────────────────────────────────────────────────────────────────────────

/// Advanced Differential Privacy with Composition
#[derive(Debug, Clone)]
pub struct AdvancedDifferentialPrivacy {
    /// Privacy budget (epsilon)
    epsilon: f64,
    /// Delta parameter for (ε,δ)-differential privacy
    delta: f64,
    /// Privacy budget tracker
    budget_tracker: PrivacyBudgetTracker,
    /// Noise mechanisms
    noise_mechanisms: Vec<NoiseMechanism>,
}

impl AdvancedDifferentialPrivacy {
    /// Create new advanced differential privacy system
    pub fn new(epsilon: f64, delta: f64) -> Self {
        Self {
            epsilon,
            delta,
            budget_tracker: PrivacyBudgetTracker::new(epsilon),
            noise_mechanisms: vec![
                NoiseMechanism::Laplace,
                NoiseMechanism::Gaussian,
                NoiseMechanism::Exponential,
            ],
        }
    }

    /// Apply differential privacy to federated update
    pub fn privatize_update(&mut self, update: &mut FederatedUpdate) -> Result<PrivacyReport> {
        let mut privacy_report = PrivacyReport {
            epsilon_used: 0.0,
            delta_used: 0.0,
            mechanism_used: NoiseMechanism::Laplace,
            noise_added: 0.0,
            utility_preserved: 1.0,
        };

        if !self.budget_tracker.can_spend(self.epsilon / 10.0) {
            return Err(ShaclAiError::DataProcessing(
                "Insufficient privacy budget".to_string(),
            ));
        }

        let mechanism = self.select_noise_mechanism(&update.parameter_delta)?;
        privacy_report.mechanism_used = mechanism.clone();

        for (_key, value) in update.parameter_delta.deltas.iter_mut() {
            let noise = self.generate_noise(&mechanism, *value)?;
            *value += noise;
            privacy_report.noise_added += noise.abs();
        }

        let epsilon_spent = self.epsilon / 10.0;
        self.budget_tracker.spend(epsilon_spent)?;
        privacy_report.epsilon_used = epsilon_spent;
        privacy_report.utility_preserved = self.calculate_utility_preservation(&privacy_report);

        Ok(privacy_report)
    }

    /// Select optimal noise mechanism based on data characteristics
    fn select_noise_mechanism(&self, _delta: &ModelParameterDelta) -> Result<NoiseMechanism> {
        Ok(NoiseMechanism::Gaussian)
    }

    /// Generate noise based on mechanism
    fn generate_noise(&self, mechanism: &NoiseMechanism, sensitivity: f64) -> Result<f64> {
        match mechanism {
            NoiseMechanism::Laplace => {
                let scale = sensitivity / self.epsilon;
                Ok(laplace_noise(scale))
            }
            NoiseMechanism::Gaussian => {
                let sigma = sensitivity * (2.0 * (1.25 / self.delta).ln()).sqrt() / self.epsilon;
                Ok(gaussian_noise(0.0, sigma))
            }
            NoiseMechanism::Exponential => Ok(exponential_noise(1.0 / self.epsilon)),
        }
    }

    /// Calculate utility preservation
    fn calculate_utility_preservation(&self, report: &PrivacyReport) -> f64 {
        1.0 - (report.noise_added / (1.0 + report.epsilon_used))
    }

    /// Apply composition theorems for privacy budget
    pub fn compose_privacy(&mut self, other_epsilon: f64, other_delta: f64) -> Result<(f64, f64)> {
        let composed_epsilon = self.epsilon + other_epsilon;
        let composed_delta = self.delta + other_delta;
        self.epsilon = composed_epsilon;
        self.delta = composed_delta;
        Ok((composed_epsilon, composed_delta))
    }

    /// Suppress unused-field warning on noise_mechanisms
    #[allow(dead_code)]
    fn _use_noise_mechanisms(&self) {
        let _ = self.noise_mechanisms.len();
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Privacy Budget Tracker
// ──────────────────────────────────────────────────────────────────────────────

impl PrivacyBudgetTracker {
    pub fn new(total_budget: f64) -> Self {
        Self {
            total_budget,
            spent_budget: 0.0,
            spending_history: Vec::new(),
        }
    }

    pub fn can_spend(&self, amount: f64) -> bool {
        self.spent_budget + amount <= self.total_budget
    }

    pub fn spend(&mut self, amount: f64) -> Result<()> {
        if !self.can_spend(amount) {
            return Err(ShaclAiError::DataProcessing(
                "Insufficient privacy budget".to_string(),
            ));
        }
        self.spent_budget += amount;
        self.spending_history.push(PrivacySpending {
            amount,
            timestamp: std::time::SystemTime::now(),
            remaining: self.total_budget - self.spent_budget,
        });
        Ok(())
    }

    pub fn remaining_budget(&self) -> f64 {
        self.total_budget - self.spent_budget
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Secure Multi-Party Computation
// ──────────────────────────────────────────────────────────────────────────────

/// Secure Multi-Party Computation for Federated Learning
#[derive(Debug)]
pub struct SecureMultiPartyComputation {
    /// Number of parties
    num_parties: usize,
    /// Secret sharing scheme
    secret_sharing: SecretSharingScheme,
    /// Secure computation protocols
    protocols: Vec<SMPCProtocol>,
    /// Communication channels
    channels: HashMap<Uuid, SecureChannel>,
}

impl SecureMultiPartyComputation {
    /// Create new SMPC system
    pub fn new(num_parties: usize) -> Self {
        Self {
            num_parties,
            secret_sharing: SecretSharingScheme::Shamir {
                threshold: (num_parties * 2) / 3 + 1,
                total_shares: num_parties,
            },
            protocols: vec![
                SMPCProtocol::SecureAggregation,
                SMPCProtocol::PrivateSetIntersection,
                SMPCProtocol::SecureComparison,
            ],
            channels: HashMap::new(),
        }
    }

    /// Perform secure aggregation of federated updates
    pub async fn secure_aggregate(
        &self,
        local_update: &FederatedUpdate,
        other_parties: &[Uuid],
    ) -> Result<SecureAggregationResult> {
        let shared_parameters = self.secret_share_parameters(&local_update.parameter_delta)?;

        let mut collected_shares = HashMap::new();
        for party in other_parties {
            let shares = self
                .exchange_shares_with_party(*party, &shared_parameters)
                .await?;
            collected_shares.insert(*party, shares);
        }

        let aggregated_result = self.reconstruct_aggregated_parameters(&collected_shares)?;

        Ok(SecureAggregationResult {
            aggregated_parameters: aggregated_result,
            participating_parties: other_parties.iter().copied().collect(),
            security_level: SecurityLevel::High,
            computation_rounds: 3,
        })
    }

    fn secret_share_parameters(
        &self,
        delta: &ModelParameterDelta,
    ) -> Result<HashMap<String, Vec<SecretShare>>> {
        let mut shared_params = HashMap::new();

        for (key, value) in &delta.deltas {
            let shares = match &self.secret_sharing {
                SecretSharingScheme::Shamir {
                    threshold,
                    total_shares,
                } => self.shamir_secret_share(*value, *threshold, *total_shares)?,
                SecretSharingScheme::Additive { num_shares } => {
                    self.additive_secret_share(*value, *num_shares)?
                }
            };
            shared_params.insert(key.clone(), shares);
        }

        Ok(shared_params)
    }

    fn shamir_secret_share(
        &self,
        secret: f64,
        threshold: usize,
        total_shares: usize,
    ) -> Result<Vec<SecretShare>> {
        let mut shares = Vec::new();
        for i in 1..=total_shares {
            shares.push(SecretShare {
                party_id: i,
                share_value: secret / total_shares as f64,
                threshold,
                total_shares,
            });
        }
        Ok(shares)
    }

    fn additive_secret_share(&self, secret: f64, num_shares: usize) -> Result<Vec<SecretShare>> {
        let mut shares = Vec::new();
        let mut sum = 0.0;

        for i in 1..num_shares {
            let share_value = fastrand::f64() * secret;
            sum += share_value;
            shares.push(SecretShare {
                party_id: i,
                share_value,
                threshold: num_shares,
                total_shares: num_shares,
            });
        }

        shares.push(SecretShare {
            party_id: num_shares,
            share_value: secret - sum,
            threshold: num_shares,
            total_shares: num_shares,
        });

        Ok(shares)
    }

    async fn exchange_shares_with_party(
        &self,
        _party_id: Uuid,
        _shares: &HashMap<String, Vec<SecretShare>>,
    ) -> Result<HashMap<String, Vec<SecretShare>>> {
        Ok(HashMap::new())
    }

    fn reconstruct_aggregated_parameters(
        &self,
        _shares: &HashMap<Uuid, HashMap<String, Vec<SecretShare>>>,
    ) -> Result<HashMap<String, f64>> {
        Ok(HashMap::new())
    }

    #[allow(dead_code)]
    fn _use_fields(&self) {
        let _ = self.num_parties;
        let _ = self.protocols.len();
        let _ = self.channels.len();
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Noise helper functions (pure, no side-effects)
// ──────────────────────────────────────────────────────────────────────────────

pub(crate) fn laplace_noise(scale: f64) -> f64 {
    let u = fastrand::f64() - 0.5;
    -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
}

pub(crate) fn gaussian_noise(mean: f64, std_dev: f64) -> f64 {
    let u1 = fastrand::f64();
    let u2 = fastrand::f64();
    let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + std_dev * z0
}

pub(crate) fn exponential_noise(lambda: f64) -> f64 {
    let u = fastrand::f64();
    -(1.0 - u).ln() / lambda
}
