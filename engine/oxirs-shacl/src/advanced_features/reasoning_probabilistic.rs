//! Probabilistic SHACL validation using Bayesian inference.
//!
//! Provides [`ProbabilisticValidator`], which quantifies uncertainty in SHACL
//! validation results with Bayesian updates, confidence intervals, Shannon
//! entropy, and Monte Carlo aggregation drawn from SciRS2 statistics.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Probabilistic validation using Bayesian inference
///
/// Provides uncertainty quantification for SHACL validation results
/// using statistical methods from SciRS2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticValidator {
    /// Prior probabilities for constraint satisfaction
    priors: HashMap<String, f64>,
    /// Evidence history for Bayesian updates
    evidence_history: Vec<EvidenceData>,
    /// Configuration
    config: ProbabilisticConfig,
    /// Statistics
    stats: ProbabilisticStats,
}

/// Configuration for probabilistic validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticConfig {
    /// Default prior probability for unknown constraints
    pub default_prior: f64,
    /// Confidence level for intervals (e.g., 0.95 for 95% CI)
    pub confidence_level: f64,
    /// Minimum evidence count for reliable estimates
    pub min_evidence_count: usize,
    /// Learning rate for Bayesian updates
    pub learning_rate: f64,
    /// Use Monte Carlo sampling for complex probabilities
    pub use_monte_carlo: bool,
    /// Number of Monte Carlo samples
    pub mc_sample_count: usize,
}

impl Default for ProbabilisticConfig {
    fn default() -> Self {
        Self {
            default_prior: 0.5,
            confidence_level: 0.95,
            min_evidence_count: 10,
            learning_rate: 0.1,
            use_monte_carlo: false,
            mc_sample_count: 1000,
        }
    }
}

/// Evidence data for Bayesian updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceData {
    /// Constraint being evaluated
    pub constraint_id: String,
    /// Whether constraint was satisfied
    pub satisfied: bool,
    /// Confidence in this evidence
    pub confidence: f64,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Probabilistic validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticValidationResult {
    /// Whether constraint conforms (deterministic)
    pub conforms: bool,
    /// Probability that constraint is satisfied
    pub satisfaction_probability: f64,
    /// Confidence interval (lower, upper)
    pub confidence_interval: (f64, f64),
    /// Uncertainty measure (entropy or variance)
    pub uncertainty: f64,
    /// Evidence count used for calculation
    pub evidence_count: usize,
    /// Bayesian posterior probability
    pub posterior_probability: f64,
}

/// Statistics for probabilistic validation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProbabilisticStats {
    /// Total probabilistic validations
    pub total_validations: usize,
    /// Total evidence collected
    pub total_evidence: usize,
    /// Average uncertainty
    pub avg_uncertainty: f64,
    /// Bayesian updates performed
    pub bayesian_updates: usize,
}

impl ProbabilisticValidator {
    /// Create a new probabilistic validator
    pub fn new(config: ProbabilisticConfig) -> Self {
        Self {
            priors: HashMap::new(),
            evidence_history: Vec::new(),
            config,
            stats: ProbabilisticStats::default(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(ProbabilisticConfig::default())
    }

    /// Validate with probabilistic reasoning
    pub fn validate_probabilistic(
        &mut self,
        constraint_id: &str,
        observed_satisfaction: bool,
        observation_confidence: f64,
    ) -> ProbabilisticValidationResult {
        self.stats.total_validations += 1;

        // Get prior probability
        let prior = *self
            .priors
            .get(constraint_id)
            .unwrap_or(&self.config.default_prior);

        // Collect evidence
        self.add_evidence(EvidenceData {
            constraint_id: constraint_id.to_string(),
            satisfied: observed_satisfaction,
            confidence: observation_confidence,
            timestamp: chrono::Utc::now(),
        });

        // Compute posterior using Bayesian inference
        let posterior = self.compute_posterior(
            constraint_id,
            observed_satisfaction,
            observation_confidence,
            prior,
        );

        // Update prior for next iteration (Bayesian learning)
        self.priors.insert(constraint_id.to_string(), posterior);
        self.stats.bayesian_updates += 1;

        // Compute confidence interval
        let evidence_for_constraint = self.get_evidence_for_constraint(constraint_id);
        let ci = self.compute_confidence_interval(&evidence_for_constraint);

        // Compute uncertainty (entropy)
        let uncertainty = self.compute_uncertainty(posterior);

        ProbabilisticValidationResult {
            conforms: observed_satisfaction,
            satisfaction_probability: posterior,
            confidence_interval: ci,
            uncertainty,
            evidence_count: evidence_for_constraint.len(),
            posterior_probability: posterior,
        }
    }

    /// Compute posterior probability using Bayes' theorem
    /// P(H|E) = P(E|H) * P(H) / P(E)
    fn compute_posterior(
        &self,
        _constraint_id: &str,
        observed_satisfaction: bool,
        observation_confidence: f64,
        prior: f64,
    ) -> f64 {
        // Likelihood: P(E|H) - probability of observation given hypothesis
        let likelihood = if observed_satisfaction {
            observation_confidence
        } else {
            1.0 - observation_confidence
        };

        // Prior: P(H)
        let p_hypothesis = prior;

        // Marginal probability: P(E)
        // P(E) = P(E|H) * P(H) + P(E|¬H) * P(¬H)
        let p_evidence = likelihood * p_hypothesis + (1.0 - likelihood) * (1.0 - p_hypothesis);

        // Avoid division by zero
        if p_evidence == 0.0 {
            return prior;
        }

        // Posterior: P(H|E)
        let posterior = (likelihood * p_hypothesis) / p_evidence;

        // Apply learning rate for smoothing
        let smoothed =
            prior * (1.0 - self.config.learning_rate) + posterior * self.config.learning_rate;

        smoothed.clamp(0.0, 1.0)
    }

    /// Compute confidence interval using normal approximation
    fn compute_confidence_interval(&self, evidence: &[EvidenceData]) -> (f64, f64) {
        if evidence.len() < self.config.min_evidence_count {
            // Not enough evidence - return wide interval
            return (0.0, 1.0);
        }

        // Count satisfactions
        let n = evidence.len() as f64;
        let successes = evidence.iter().filter(|e| e.satisfied).count() as f64;
        let p = successes / n;

        // Use normal approximation for binomial proportion
        // CI = p ± z * sqrt(p(1-p)/n)
        // For 95% CI, z ≈ 1.96
        let z = match self.config.confidence_level {
            0.90 => 1.645,
            0.95 => 1.96,
            0.99 => 2.576,
            _ => 1.96, // default to 95%
        };

        let standard_error = (p * (1.0 - p) / n).sqrt();
        let margin = z * standard_error;

        let lower = (p - margin).max(0.0);
        let upper = (p + margin).min(1.0);

        (lower, upper)
    }

    /// Compute uncertainty using Shannon entropy
    /// H(X) = -Σ p(x) * log2(p(x))
    fn compute_uncertainty(&self, probability: f64) -> f64 {
        if probability == 0.0 || probability == 1.0 {
            return 0.0; // No uncertainty for certain events
        }

        let p = probability;
        let q = 1.0 - probability;

        // Binary entropy (normalized to [0, 1], max entropy for binary is 1.0)
        -(p * p.log2() + q * q.log2())
    }

    /// Get all evidence for a specific constraint
    fn get_evidence_for_constraint(&self, constraint_id: &str) -> Vec<EvidenceData> {
        self.evidence_history
            .iter()
            .filter(|e| e.constraint_id == constraint_id)
            .cloned()
            .collect()
    }

    /// Add new evidence
    fn add_evidence(&mut self, evidence: EvidenceData) {
        self.evidence_history.push(evidence);
        self.stats.total_evidence += 1;
    }

    /// Compute aggregate probability for multiple constraints using Monte Carlo
    pub fn compute_aggregate_probability(&self, constraint_ids: &[String]) -> f64 {
        if !self.config.use_monte_carlo {
            // Simple multiplication (assumes independence)
            return constraint_ids
                .iter()
                .map(|id| *self.priors.get(id).unwrap_or(&self.config.default_prior))
                .product();
        }

        // Monte Carlo simulation for complex joint probabilities
        use scirs2_core::random::Random;

        let mut rng_state = Random::seed(42);
        let mut successes = 0;

        for _ in 0..self.config.mc_sample_count {
            let all_satisfied = constraint_ids.iter().all(|id| {
                let p = *self.priors.get(id).unwrap_or(&self.config.default_prior);
                rng_state.gen_range(0.0..1.0) < p
            });

            if all_satisfied {
                successes += 1;
            }
        }

        successes as f64 / self.config.mc_sample_count as f64
    }

    /// Estimate probability distribution using kernel density estimation
    pub fn estimate_probability_distribution(
        &self,
        constraint_id: &str,
        num_bins: usize,
    ) -> Vec<(f64, f64)> {
        let evidence = self.get_evidence_for_constraint(constraint_id);

        if evidence.is_empty() {
            // Return uniform prior
            let bin_width = 1.0 / num_bins as f64;
            return (0..num_bins)
                .map(|i| {
                    let x = i as f64 * bin_width;
                    (x, self.config.default_prior)
                })
                .collect();
        }

        // Simple histogram-based estimation
        let successes = evidence.iter().filter(|e| e.satisfied).count();
        let total = evidence.len();
        let _success_rate = successes as f64 / total as f64;

        // Beta distribution approximation
        // Using (successes + 1) and (failures + 1) for Laplace smoothing
        let alpha = successes as f64 + 1.0;
        let beta = (total - successes) as f64 + 1.0;

        // Generate beta distribution points
        let bin_width = 1.0 / num_bins as f64;
        (0..num_bins)
            .map(|i| {
                let x = (i as f64 + 0.5) * bin_width; // Mid-point of bin
                let density = self.beta_pdf(x, alpha, beta);
                (x, density)
            })
            .collect()
    }

    /// Beta distribution PDF (simplified)
    fn beta_pdf(&self, x: f64, alpha: f64, beta: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }

        // Simplified beta PDF (not normalized, for relative comparison)
        x.powf(alpha - 1.0) * (1.0 - x).powf(beta - 1.0)
    }

    /// Get validation statistics
    pub fn stats(&self) -> &ProbabilisticStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = ProbabilisticStats::default();
    }

    /// Clear evidence history
    pub fn clear_evidence(&mut self) {
        self.evidence_history.clear();
    }

    /// Get current priors
    pub fn get_priors(&self) -> &HashMap<String, f64> {
        &self.priors
    }

    /// Set prior for a constraint
    pub fn set_prior(&mut self, constraint_id: String, prior: f64) {
        self.priors.insert(constraint_id, prior.clamp(0.0, 1.0));
    }

    // ---- crate-internal test accessors ------------------------------------
    // These expose otherwise-private helpers to the sibling `reasoning_tests`
    // module so the unit tests can exercise them directly.

    /// Test accessor for [`Self::compute_uncertainty`].
    #[doc(hidden)]
    pub(crate) fn compute_uncertainty_pub(&self, probability: f64) -> f64 {
        self.compute_uncertainty(probability)
    }

    /// Test accessor for [`Self::add_evidence`].
    #[doc(hidden)]
    pub(crate) fn add_evidence_pub(&mut self, evidence: EvidenceData) {
        self.add_evidence(evidence)
    }

    /// Test accessor for [`Self::get_evidence_for_constraint`].
    #[doc(hidden)]
    pub(crate) fn get_evidence_for_constraint_pub(&self, constraint_id: &str) -> Vec<EvidenceData> {
        self.get_evidence_for_constraint(constraint_id)
    }

    /// Test accessor for [`Self::compute_confidence_interval`].
    #[doc(hidden)]
    pub(crate) fn compute_confidence_interval_pub(&self, evidence: &[EvidenceData]) -> (f64, f64) {
        self.compute_confidence_interval(evidence)
    }
}

impl Default for ProbabilisticValidator {
    fn default() -> Self {
        Self::default_config()
    }
}
