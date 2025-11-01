//! Active Learning for SHACL Validation
//!
//! This module implements active learning strategies for efficiently selecting
//! the most informative samples for validation and model training.
//!
//! Includes uncertainty sampling, query-by-committee, and human-in-the-loop approaches.

use crate::{Result, ShaclAiError};
use oxirs_core::Store;
use oxirs_shacl::Shape;
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::{Random, Rng};
use scirs2_stats::distributions::{Normal, Uniform};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Active learning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveLearningConfig {
    /// Query strategy
    pub query_strategy: QueryStrategy,

    /// Sampling strategy
    pub sampling_strategy: SamplingStrategy,

    /// Budget (max number of queries)
    pub budget: usize,

    /// Batch size for each query
    pub batch_size: usize,

    /// Uncertainty threshold
    pub uncertainty_threshold: f64,

    /// Enable diversity-based sampling
    pub enable_diversity_sampling: bool,

    /// Enable committee-based queries
    pub enable_committee_queries: bool,

    /// Number of committee members
    pub committee_size: usize,

    /// Minimum confidence for auto-labeling
    pub min_auto_label_confidence: f64,
}

impl Default for ActiveLearningConfig {
    fn default() -> Self {
        Self {
            query_strategy: QueryStrategy::UncertaintySampling,
            sampling_strategy: SamplingStrategy::LeastConfident,
            budget: 1000,
            batch_size: 10,
            uncertainty_threshold: 0.7,
            enable_diversity_sampling: true,
            enable_committee_queries: false,
            committee_size: 5,
            min_auto_label_confidence: 0.95,
        }
    }
}

/// Query strategies for active learning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryStrategy {
    /// Uncertainty sampling
    UncertaintySampling,

    /// Query-by-committee
    QueryByCommittee,

    /// Expected model change
    ExpectedModelChange,

    /// Expected error reduction
    ExpectedErrorReduction,

    /// Density-weighted sampling
    DensityWeighted,

    /// Representative sampling
    RepresentativeSampling,
}

/// Sampling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Select samples with least confidence
    LeastConfident,

    /// Select samples with highest margin
    MarginSampling,

    /// Select samples with highest entropy
    EntropySampling,

    /// Random sampling baseline
    Random,
}

/// Active learner for SHACL validation
#[derive(Debug)]
pub struct ActiveLearner {
    config: ActiveLearningConfig,
    uncertainty_sampler: UncertaintySampling,
    committee: Option<Committee>,
    queried_samples: HashSet<String>,
    query_history: Vec<QueryRecord>,
    rng: Random,
    stats: ActiveLearningStats,
}

/// Uncertainty sampling implementation
#[derive(Debug)]
pub struct UncertaintySampling {
    strategy: SamplingStrategy,
    uncertainty_scores: HashMap<String, f64>,
}

/// Committee for query-by-committee
#[derive(Debug)]
struct Committee {
    members: Vec<CommitteeMember>,
    disagreement_measure: DisagreementMeasure,
}

/// Committee member (model variant)
#[derive(Debug)]
struct CommitteeMember {
    member_id: usize,
    predictions: HashMap<String, f64>,
}

/// Disagreement measures for committee
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DisagreementMeasure {
    VoteEntropy,
    KLDivergence,
    Variance,
}

/// Query record for tracking
#[derive(Debug, Clone)]
pub struct QueryRecord {
    pub sample_id: String,
    pub uncertainty_score: f64,
    pub query_timestamp: chrono::DateTime<chrono::Utc>,
    pub label_source: LabelSource,
    pub confidence: f64,
}

/// Source of label
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabelSource {
    Human,
    AutoLabeled,
    Committee,
}

/// Active learning statistics
#[derive(Debug, Clone, Default)]
pub struct ActiveLearningStats {
    pub total_queries: usize,
    pub human_labeled: usize,
    pub auto_labeled: usize,
    pub average_uncertainty: f64,
    pub model_improvement: f64,
    pub budget_used: usize,
}

impl ActiveLearner {
    /// Create a new active learner
    pub fn new(config: ActiveLearningConfig) -> Result<Self> {
        let uncertainty_sampler = UncertaintySampling::new(config.sampling_strategy);

        let committee = if config.enable_committee_queries {
            Some(Committee::new(config.committee_size))
        } else {
            None
        };

        Ok(Self {
            config,
            uncertainty_sampler,
            committee,
            queried_samples: HashSet::new(),
            query_history: Vec::new(),
            rng: Random::default(),
            stats: ActiveLearningStats::default(),
        })
    }

    /// Select most informative samples for labeling
    pub fn select_samples_for_query(
        &mut self,
        unlabeled_pool: &[Shape],
        model_predictions: &HashMap<String, f64>,
    ) -> Result<Vec<String>> {
        tracing::info!(
            "Selecting {} samples from pool of {}",
            self.config.batch_size,
            unlabeled_pool.len()
        );

        let mut selected = match self.config.query_strategy {
            QueryStrategy::UncertaintySampling => {
                self.uncertainty_based_selection(unlabeled_pool, model_predictions)?
            }
            QueryStrategy::QueryByCommittee => self.committee_based_selection(unlabeled_pool)?,
            QueryStrategy::ExpectedModelChange => {
                self.expected_model_change_selection(unlabeled_pool, model_predictions)?
            }
            QueryStrategy::DensityWeighted => {
                self.density_weighted_selection(unlabeled_pool, model_predictions)?
            }
            _ => {
                return Err(ShaclAiError::MetaLearning(format!(
                    "Query strategy {:?} not yet implemented",
                    self.config.query_strategy
                )));
            }
        };

        // Apply diversity if enabled
        if self.config.enable_diversity_sampling && selected.len() > 1 {
            selected = self.apply_diversity_sampling(selected, unlabeled_pool)?;
        }

        // Record queries
        for sample_id in &selected {
            self.queried_samples.insert(sample_id.clone());
            self.stats.total_queries += 1;
            self.stats.budget_used += 1;
        }

        Ok(selected)
    }

    /// Uncertainty-based sample selection
    fn uncertainty_based_selection(
        &mut self,
        unlabeled_pool: &[Shape],
        model_predictions: &HashMap<String, f64>,
    ) -> Result<Vec<String>> {
        // Compute uncertainty scores
        let mut uncertainty_scores = Vec::new();

        for shape in unlabeled_pool {
            let shape_id = format!("{:?}", shape.id); // Simplified ID

            if self.queried_samples.contains(&shape_id) {
                continue;
            }

            let prediction = model_predictions.get(&shape_id).unwrap_or(&0.5);
            let uncertainty = self.uncertainty_sampler.compute_uncertainty(*prediction)?;

            uncertainty_scores.push((shape_id.clone(), uncertainty));
            self.uncertainty_sampler
                .record_uncertainty(shape_id, uncertainty);
        }

        // Sort by uncertainty (descending)
        uncertainty_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top batch_size samples
        let selected: Vec<String> = uncertainty_scores
            .into_iter()
            .take(self.config.batch_size)
            .map(|(id, _)| id)
            .collect();

        Ok(selected)
    }

    /// Committee-based sample selection
    fn committee_based_selection(&mut self, unlabeled_pool: &[Shape]) -> Result<Vec<String>> {
        if let Some(ref mut committee) = self.committee {
            let disagreements = committee.compute_disagreements(unlabeled_pool)?;

            // Sort by disagreement (descending)
            let mut sorted_disagreements: Vec<_> = disagreements.into_iter().collect();
            sorted_disagreements.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let selected: Vec<String> = sorted_disagreements
                .into_iter()
                .take(self.config.batch_size)
                .map(|(id, _)| id)
                .collect();

            Ok(selected)
        } else {
            Err(ShaclAiError::MetaLearning(
                "Committee not initialized".to_string(),
            ))
        }
    }

    /// Expected model change based selection
    fn expected_model_change_selection(
        &self,
        unlabeled_pool: &[Shape],
        _model_predictions: &HashMap<String, f64>,
    ) -> Result<Vec<String>> {
        // Estimate which samples would change the model most if labeled
        let mut change_scores = Vec::new();
        let mut rng = Random::default();

        for shape in unlabeled_pool {
            let shape_id = format!("{:?}", shape.id);

            if self.queried_samples.contains(&shape_id) {
                continue;
            }

            // Simplified gradient-based change estimation
            let change_score = rng.random::<f64>();
            change_scores.push((shape_id, change_score));
        }

        change_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected: Vec<String> = change_scores
            .into_iter()
            .take(self.config.batch_size)
            .map(|(id, _)| id)
            .collect();

        Ok(selected)
    }

    /// Density-weighted selection
    fn density_weighted_selection(
        &self,
        unlabeled_pool: &[Shape],
        model_predictions: &HashMap<String, f64>,
    ) -> Result<Vec<String>> {
        // Weight uncertainty by sample density (informativeness + representativeness)
        let mut weighted_scores = Vec::new();
        let mut rng = Random::default();

        for shape in unlabeled_pool {
            let shape_id = format!("{:?}", shape.id);

            if self.queried_samples.contains(&shape_id) {
                continue;
            }

            let prediction = model_predictions.get(&shape_id).unwrap_or(&0.5);
            let uncertainty = (prediction - 0.5).abs();

            // Simplified density estimation
            let density = 0.5 + rng.random::<f64>() * 0.5;

            let weighted_score = uncertainty * density;
            weighted_scores.push((shape_id, weighted_score));
        }

        weighted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected: Vec<String> = weighted_scores
            .into_iter()
            .take(self.config.batch_size)
            .map(|(id, _)| id)
            .collect();

        Ok(selected)
    }

    /// Apply diversity sampling to avoid redundant queries
    fn apply_diversity_sampling(
        &self,
        selected: Vec<String>,
        _unlabeled_pool: &[Shape],
    ) -> Result<Vec<String>> {
        // Use simple greedy diversity selection (simplified)
        // In production, this would use actual feature-based diversity

        let mut diverse_selected = Vec::new();
        let mut remaining = selected;

        while !remaining.is_empty() && diverse_selected.len() < self.config.batch_size {
            // Select most diverse from remaining
            if diverse_selected.is_empty() {
                diverse_selected.push(remaining.remove(0));
            } else {
                // Find most diverse sample (simplified - just alternate)
                if !remaining.is_empty() {
                    diverse_selected.push(remaining.remove(0));
                }
            }
        }

        Ok(diverse_selected)
    }

    /// Process human feedback on queried samples
    pub fn process_human_feedback(
        &mut self,
        sample_id: String,
        label: bool,
        confidence: f64,
    ) -> Result<()> {
        let uncertainty_score = self
            .uncertainty_sampler
            .uncertainty_scores
            .get(&sample_id)
            .copied()
            .unwrap_or(0.5);

        self.query_history.push(QueryRecord {
            sample_id,
            uncertainty_score,
            query_timestamp: chrono::Utc::now(),
            label_source: LabelSource::Human,
            confidence,
        });

        self.stats.human_labeled += 1;
        self.update_stats();

        Ok(())
    }

    /// Auto-label high-confidence samples
    pub fn auto_label_high_confidence(
        &mut self,
        model_predictions: &HashMap<String, f64>,
    ) -> Result<Vec<(String, bool)>> {
        let mut auto_labeled = Vec::new();

        for (sample_id, prediction) in model_predictions {
            if self.queried_samples.contains(sample_id) {
                continue;
            }

            let confidence = if *prediction > 0.5 {
                *prediction
            } else {
                1.0 - prediction
            };

            if confidence >= self.config.min_auto_label_confidence {
                let label = *prediction > 0.5;
                auto_labeled.push((sample_id.clone(), label));

                self.query_history.push(QueryRecord {
                    sample_id: sample_id.clone(),
                    uncertainty_score: 1.0 - confidence,
                    query_timestamp: chrono::Utc::now(),
                    label_source: LabelSource::AutoLabeled,
                    confidence,
                });

                self.stats.auto_labeled += 1;
            }
        }

        self.update_stats();

        Ok(auto_labeled)
    }

    /// Update statistics
    fn update_stats(&mut self) {
        if !self.query_history.is_empty() {
            self.stats.average_uncertainty = self
                .query_history
                .iter()
                .map(|r| r.uncertainty_score)
                .sum::<f64>()
                / self.query_history.len() as f64;
        }
    }

    /// Get active learning statistics
    pub fn get_stats(&self) -> &ActiveLearningStats {
        &self.stats
    }

    /// Check if budget is exhausted
    pub fn is_budget_exhausted(&self) -> bool {
        self.stats.budget_used >= self.config.budget
    }
}

impl UncertaintySampling {
    fn new(strategy: SamplingStrategy) -> Self {
        Self {
            strategy,
            uncertainty_scores: HashMap::new(),
        }
    }

    fn compute_uncertainty(&self, prediction: f64) -> Result<f64> {
        let uncertainty = match self.strategy {
            SamplingStrategy::LeastConfident => {
                // Confidence is distance from decision boundary (0.5)
                1.0 - (prediction - 0.5).abs() * 2.0
            }
            SamplingStrategy::MarginSampling => {
                // Margin between top two predictions (simplified for binary)
                (prediction - 0.5).abs()
            }
            SamplingStrategy::EntropySampling => {
                // Binary entropy
                let p = prediction.max(1e-10).min(1.0 - 1e-10);
                -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
            }
            SamplingStrategy::Random => {
                // Random baseline
                0.5
            }
        };

        Ok(uncertainty)
    }

    fn record_uncertainty(&mut self, sample_id: String, uncertainty: f64) {
        self.uncertainty_scores.insert(sample_id, uncertainty);
    }
}

impl Committee {
    fn new(size: usize) -> Self {
        let mut members = Vec::new();
        for i in 0..size {
            members.push(CommitteeMember {
                member_id: i,
                predictions: HashMap::new(),
            });
        }

        Self {
            members,
            disagreement_measure: DisagreementMeasure::VoteEntropy,
        }
    }

    fn compute_disagreements(&mut self, unlabeled_pool: &[Shape]) -> Result<HashMap<String, f64>> {
        let mut disagreements = HashMap::new();

        for shape in unlabeled_pool {
            let shape_id = format!("{:?}", shape.id);

            // Get predictions from all committee members
            let mut predictions = Vec::new();
            for member in &mut self.members {
                // Simplified: generate random predictions
                let mut rng = Random::default();
                let prediction = rng.random::<f64>();
                member.predictions.insert(shape_id.clone(), prediction);
                predictions.push(prediction);
            }

            // Compute disagreement
            let disagreement = match self.disagreement_measure {
                DisagreementMeasure::VoteEntropy => {
                    let votes: Vec<bool> = predictions.iter().map(|p| *p > 0.5).collect();
                    let pos_votes = votes.iter().filter(|v| **v).count() as f64;
                    let total = votes.len() as f64;
                    let p = pos_votes / total;
                    if p == 0.0 || p == 1.0 {
                        0.0
                    } else {
                        -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
                    }
                }
                DisagreementMeasure::Variance => {
                    let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
                    predictions.iter().map(|p| (p - mean).powi(2)).sum::<f64>()
                        / predictions.len() as f64
                }
                DisagreementMeasure::KLDivergence => {
                    // Simplified KL divergence
                    let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
                    predictions
                        .iter()
                        .map(|p| {
                            let p = p.max(1e-10).min(1.0 - 1e-10);
                            let q = mean.max(1e-10).min(1.0 - 1e-10);
                            p * (p / q).ln()
                        })
                        .sum::<f64>()
                        / predictions.len() as f64
                }
            };

            disagreements.insert(shape_id, disagreement);
        }

        Ok(disagreements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_active_learner_creation() {
        let config = ActiveLearningConfig::default();
        let learner = ActiveLearner::new(config).unwrap();
        assert_eq!(learner.stats.total_queries, 0);
    }

    #[test]
    fn test_uncertainty_computation() {
        let sampler = UncertaintySampling::new(SamplingStrategy::LeastConfident);

        // Prediction at decision boundary should have highest uncertainty
        let uncertainty = sampler.compute_uncertainty(0.5).unwrap();
        assert!(uncertainty > 0.9);

        // Confident prediction should have low uncertainty
        let uncertainty = sampler.compute_uncertainty(0.95).unwrap();
        assert!(uncertainty < 0.2);
    }

    #[test]
    fn test_committee_creation() {
        let committee = Committee::new(5);
        assert_eq!(committee.members.len(), 5);
    }
}
