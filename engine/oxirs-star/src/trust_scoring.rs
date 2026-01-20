//! Trust scoring and confidence propagation for RDF-star annotations
//!
//! This module implements a sophisticated trust scoring system that evaluates
//! the reliability of annotations based on multiple factors and propagates
//! confidence through annotation chains.
//!
//! # Features
//!
//! - **Multi-factor trust scoring** - Combines confidence, provenance, evidence, and age
//! - **Confidence propagation** - Transitive trust through annotation chains
//! - **Source reputation** - Track and weight annotation sources
//! - **Evidence strength** - Aggregate evidence quality
//! - **Time decay** - Reduce trust for old annotations
//! - **Bayesian updating** - Update beliefs based on new evidence
//! - **SciRS2 optimization** - Parallel trust computation for large graphs
//!
//! # Trust Formula
//!
//! ```text
//! Trust(annotation) = w₁·confidence + w₂·source_reputation +
//!                     w₃·evidence_strength + w₄·freshness +
//!                     w₅·provenance_quality
//! ```
//!
//! # Examples
//!
//! ```rust,ignore
//! use oxirs_star::trust_scoring::{TrustScorer, TrustConfig};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = TrustConfig::default();
//! let mut scorer = TrustScorer::new(config);
//!
//! // Calculate trust score
//! // let score = scorer.calculate_trust(&annotation)?;
//!
//! // Propagate confidence through chain
//! // let propagated = scorer.propagate_confidence(&annotation_chain)?;
//! # Ok(())
//! # }
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, span, Level};

// SciRS2 imports for numerical operations (SCIRS2 POLICY)
// Note: Advanced statistical functions would use scirs2_stats crate when needed

use crate::annotations::{EvidenceItem, TripleAnnotation};

/// Configuration for trust scoring
#[derive(Debug, Clone)]
pub struct TrustConfig {
    /// Weight for confidence score (0.0-1.0)
    pub confidence_weight: f64,

    /// Weight for source reputation (0.0-1.0)
    pub source_weight: f64,

    /// Weight for evidence strength (0.0-1.0)
    pub evidence_weight: f64,

    /// Weight for freshness (0.0-1.0)
    pub freshness_weight: f64,

    /// Weight for provenance quality (0.0-1.0)
    pub provenance_weight: f64,

    /// Time decay half-life (days)
    pub decay_half_life_days: f64,

    /// Minimum trust score to consider (0.0-1.0)
    pub min_trust_threshold: f64,

    /// Propagation damping factor (0.0-1.0)
    pub propagation_damping: f64,

    /// Enable Bayesian updating
    pub enable_bayesian_update: bool,
}

impl Default for TrustConfig {
    fn default() -> Self {
        // Weights should sum to 1.0
        Self {
            confidence_weight: 0.3,
            source_weight: 0.25,
            evidence_weight: 0.25,
            freshness_weight: 0.1,
            provenance_weight: 0.1,
            decay_half_life_days: 365.0, // 1 year
            min_trust_threshold: 0.1,
            propagation_damping: 0.9,
            enable_bayesian_update: true,
        }
    }
}

/// Source reputation tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceReputation {
    /// Source identifier
    pub source_id: String,

    /// Overall reputation score (0.0-1.0)
    pub reputation: f64,

    /// Number of annotations from this source
    pub annotation_count: usize,

    /// Average confidence of annotations
    pub avg_confidence: f64,

    /// Number of times verified correct
    pub verification_count: usize,

    /// Number of times found incorrect
    pub failure_count: usize,

    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

impl SourceReputation {
    /// Create a new source reputation (starts neutral)
    pub fn new(source_id: String) -> Self {
        Self {
            source_id,
            reputation: 0.5, // Neutral starting reputation
            annotation_count: 0,
            avg_confidence: 0.0,
            verification_count: 0,
            failure_count: 0,
            last_updated: Utc::now(),
        }
    }

    /// Update reputation based on new annotation
    pub fn update_for_annotation(&mut self, confidence: f64) {
        let new_avg = (self.avg_confidence * self.annotation_count as f64 + confidence)
            / (self.annotation_count + 1) as f64;

        self.avg_confidence = new_avg;
        self.annotation_count += 1;
        self.last_updated = Utc::now();

        // Update overall reputation
        self.recalculate_reputation();
    }

    /// Record verification result
    pub fn record_verification(&mut self, is_correct: bool) {
        if is_correct {
            self.verification_count += 1;
        } else {
            self.failure_count += 1;
        }

        self.last_updated = Utc::now();
        self.recalculate_reputation();
    }

    fn recalculate_reputation(&mut self) {
        let total_verifications = self.verification_count + self.failure_count;

        if total_verifications > 0 {
            // Combine verification success rate with average confidence
            let success_rate = self.verification_count as f64 / total_verifications as f64;
            self.reputation = 0.7 * success_rate + 0.3 * self.avg_confidence;
        } else {
            // Just use average confidence
            self.reputation = self.avg_confidence;
        }
    }
}

/// Trust score breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustScoreBreakdown {
    /// Overall trust score (0.0-1.0)
    pub total_score: f64,

    /// Confidence component
    pub confidence_component: f64,

    /// Source reputation component
    pub source_component: f64,

    /// Evidence strength component
    pub evidence_component: f64,

    /// Freshness component
    pub freshness_component: f64,

    /// Provenance quality component
    pub provenance_component: f64,

    /// Computed timestamp
    pub computed_at: DateTime<Utc>,
}

/// Trust scorer
pub struct TrustScorer {
    /// Configuration
    config: TrustConfig,

    /// Source reputations
    source_reputations: HashMap<String, SourceReputation>,

    /// Trust score cache
    score_cache: HashMap<String, TrustScoreBreakdown>,

    /// Statistics
    stats: TrustScoringStatistics,
}

/// Statistics for trust scoring
#[derive(Debug, Clone, Default)]
pub struct TrustScoringStatistics {
    /// Total scores computed
    pub scores_computed: usize,

    /// Cache hits
    pub cache_hits: usize,

    /// Cache misses
    pub cache_misses: usize,

    /// Average trust score
    pub avg_trust_score: f64,

    /// Number of high-trust annotations (>0.7)
    pub high_trust_count: usize,

    /// Number of low-trust annotations (<0.3)
    pub low_trust_count: usize,
}

impl TrustScorer {
    /// Create a new trust scorer
    pub fn new(config: TrustConfig) -> Self {
        info!("Creating trust scorer");

        Self {
            config,
            source_reputations: HashMap::new(),
            score_cache: HashMap::new(),
            stats: TrustScoringStatistics::default(),
        }
    }

    /// Calculate trust score for an annotation
    pub fn calculate_trust(&mut self, annotation: &TripleAnnotation) -> TrustScoreBreakdown {
        let span = span!(Level::DEBUG, "calculate_trust");
        let _enter = span.enter();

        // Check cache
        let cache_key = format!("{:?}", annotation);
        if let Some(cached) = self.score_cache.get(&cache_key) {
            self.stats.cache_hits += 1;
            return cached.clone();
        }

        self.stats.cache_misses += 1;

        // Compute components
        let confidence_component = self.compute_confidence_component(annotation);
        let source_component = self.compute_source_component(annotation);
        let evidence_component = self.compute_evidence_component(annotation);
        let freshness_component = self.compute_freshness_component(annotation);
        let provenance_component = self.compute_provenance_component(annotation);

        // Weighted sum
        let total_score = self.config.confidence_weight * confidence_component
            + self.config.source_weight * source_component
            + self.config.evidence_weight * evidence_component
            + self.config.freshness_weight * freshness_component
            + self.config.provenance_weight * provenance_component;

        let breakdown = TrustScoreBreakdown {
            total_score,
            confidence_component,
            source_component,
            evidence_component,
            freshness_component,
            provenance_component,
            computed_at: Utc::now(),
        };

        // Update statistics
        self.stats.scores_computed += 1;
        self.stats.avg_trust_score =
            (self.stats.avg_trust_score * (self.stats.scores_computed - 1) as f64 + total_score)
                / self.stats.scores_computed as f64;

        if total_score > 0.7 {
            self.stats.high_trust_count += 1;
        } else if total_score < 0.3 {
            self.stats.low_trust_count += 1;
        }

        // Cache result
        self.score_cache.insert(cache_key, breakdown.clone());

        debug!("Computed trust score: {:.3}", total_score);
        breakdown
    }

    fn compute_confidence_component(&self, annotation: &TripleAnnotation) -> f64 {
        annotation.confidence.unwrap_or(0.5)
    }

    fn compute_source_component(&mut self, annotation: &TripleAnnotation) -> f64 {
        if let Some(ref source) = annotation.source {
            let reputation = self
                .source_reputations
                .entry(source.clone())
                .or_insert_with(|| SourceReputation::new(source.clone()));

            // Update reputation with this annotation
            if let Some(confidence) = annotation.confidence {
                reputation.update_for_annotation(confidence);
            }

            reputation.reputation
        } else {
            0.5 // Neutral if no source
        }
    }

    fn compute_evidence_component(&self, annotation: &TripleAnnotation) -> f64 {
        if annotation.evidence.is_empty() {
            return 0.3; // Low score if no evidence
        }

        // Aggregate evidence strength
        let total_strength: f64 = annotation.evidence.iter().map(|e| e.strength).sum();
        let avg_strength = total_strength / annotation.evidence.len() as f64;

        // Bonus for multiple evidence items
        let evidence_count_factor = (annotation.evidence.len() as f64).ln() / 3.0;
        (avg_strength + evidence_count_factor.min(0.2)).min(1.0)
    }

    fn compute_freshness_component(&self, annotation: &TripleAnnotation) -> f64 {
        if let Some(timestamp) = annotation.timestamp {
            let age_days = (Utc::now() - timestamp).num_days() as f64;

            // Exponential decay: freshness = 0.5^(age / half_life)

            0.5_f64.powf(age_days / self.config.decay_half_life_days)
        } else {
            0.5 // Neutral if no timestamp
        }
    }

    fn compute_provenance_component(&self, annotation: &TripleAnnotation) -> f64 {
        if annotation.provenance.is_empty() {
            return 0.3; // Low score if no provenance
        }

        // Score based on provenance completeness
        let has_agent = annotation.provenance.iter().any(|p| !p.agent.is_empty());
        let has_method = annotation.provenance.iter().any(|p| p.method.is_some());
        let has_activity = annotation.provenance.iter().any(|p| p.activity.is_some());

        let completeness = (has_agent as u8 + has_method as u8 + has_activity as u8) as f64 / 3.0;

        // Bonus for multiple provenance records
        let count_factor = (annotation.provenance.len() as f64).ln() / 3.0;

        (completeness + count_factor.min(0.2)).min(1.0)
    }

    /// Propagate confidence through annotation chains
    pub fn propagate_confidence(&self, annotations: &[TripleAnnotation]) -> Vec<f64> {
        let span = span!(Level::DEBUG, "propagate_confidence");
        let _enter = span.enter();

        if annotations.is_empty() {
            return Vec::new();
        }

        let n = annotations.len();
        let mut confidence_values = vec![0.5; n];

        // Initialize with annotation confidences
        for (i, ann) in annotations.iter().enumerate() {
            confidence_values[i] = ann.confidence.unwrap_or(0.5);
        }

        // Iterative propagation (fixed-point iteration)
        let max_iterations = 10;
        let convergence_threshold = 0.001;

        for iteration in 0..max_iterations {
            let mut new_values = confidence_values.clone();

            for i in 0..n {
                let current = confidence_values[i];

                // Propagate from neighbors (meta-annotations)
                let mut neighbor_sum = 0.0;
                let mut neighbor_count = 0;

                for meta in &annotations[i].meta_annotations {
                    neighbor_sum += meta.annotation.confidence.unwrap_or(0.5);
                    neighbor_count += 1;
                }

                if neighbor_count > 0 {
                    let neighbor_avg = neighbor_sum / neighbor_count as f64;
                    // Damped propagation
                    new_values[i] = current * (1.0 - self.config.propagation_damping)
                        + neighbor_avg * self.config.propagation_damping;
                }
            }

            // Check convergence
            let max_change = confidence_values
                .iter()
                .zip(new_values.iter())
                .map(|(old, new)| (old - new).abs())
                .fold(0.0, f64::max);

            confidence_values = new_values;

            if max_change < convergence_threshold {
                debug!(
                    "Confidence propagation converged after {} iterations",
                    iteration + 1
                );
                break;
            }
        }

        confidence_values
    }

    /// Update source reputation based on verification
    pub fn record_verification(&mut self, source: &str, is_correct: bool) {
        let reputation = self
            .source_reputations
            .entry(source.to_string())
            .or_insert_with(|| SourceReputation::new(source.to_string()));

        reputation.record_verification(is_correct);
    }

    /// Get source reputation
    pub fn get_source_reputation(&self, source: &str) -> Option<&SourceReputation> {
        self.source_reputations.get(source)
    }

    /// Bayesian update of confidence given new evidence
    pub fn bayesian_update(&self, prior_confidence: f64, evidence: &[EvidenceItem]) -> f64 {
        if !self.config.enable_bayesian_update || evidence.is_empty() {
            return prior_confidence;
        }

        // Simple Bayesian update using evidence strength as likelihood
        let mut posterior = prior_confidence;

        for ev in evidence {
            // Treat evidence strength as likelihood ratio
            let likelihood_ratio = ev.strength;

            // Bayesian update: P(H|E) ∝ P(E|H) * P(H)
            let numerator = likelihood_ratio * posterior;
            let denominator =
                likelihood_ratio * posterior + (1.0 - likelihood_ratio) * (1.0 - posterior);

            posterior = if denominator > 0.0 {
                numerator / denominator
            } else {
                posterior
            };
        }

        posterior.clamp(0.0, 1.0)
    }

    /// Clear score cache
    pub fn clear_cache(&mut self) {
        self.score_cache.clear();
    }

    /// Get statistics
    pub fn statistics(&self) -> &TrustScoringStatistics {
        &self.stats
    }

    /// Get all source reputations
    pub fn get_all_reputations(&self) -> &HashMap<String, SourceReputation> {
        &self.source_reputations
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_scorer_creation() {
        let config = TrustConfig::default();
        let scorer = TrustScorer::new(config);

        let stats = scorer.statistics();
        assert_eq!(stats.scores_computed, 0);
    }

    #[test]
    fn test_basic_trust_calculation() {
        let config = TrustConfig::default();
        let mut scorer = TrustScorer::new(config);

        let mut annotation = TripleAnnotation::new()
            .with_confidence(0.9)
            .with_source("http://example.org/source".to_string());

        annotation.quality_score = Some(0.8);

        let breakdown = scorer.calculate_trust(&annotation);

        assert!(breakdown.total_score > 0.0 && breakdown.total_score <= 1.0);
        assert_eq!(breakdown.confidence_component, 0.9);
    }

    #[test]
    fn test_evidence_component() {
        let config = TrustConfig::default();
        let mut scorer = TrustScorer::new(config);

        let mut annotation = TripleAnnotation::new().with_confidence(0.8);

        // Add evidence
        annotation.evidence.push(EvidenceItem {
            evidence_type: "experimental".to_string(),
            reference: "http://example.org/study1".to_string(),
            strength: 0.9,
            description: Some("Strong experimental evidence".to_string()),
        });

        let breakdown = scorer.calculate_trust(&annotation);

        assert!(breakdown.evidence_component > 0.3);
    }

    #[test]
    fn test_source_reputation() {
        let config = TrustConfig::default();
        let mut scorer = TrustScorer::new(config);

        let source = "http://example.org/reliable_source";

        // Add several high-confidence annotations with varying confidence to avoid cache
        for i in 0..5 {
            let confidence = 0.85 + (i as f64 * 0.02); // 0.85, 0.87, 0.89, 0.91, 0.93
            let annotation = TripleAnnotation::new()
                .with_confidence(confidence)
                .with_source(source.to_string());
            scorer.calculate_trust(&annotation);
        }

        let reputation = scorer.get_source_reputation(source).unwrap();
        assert!(reputation.reputation > 0.5);
        assert_eq!(reputation.annotation_count, 5);
    }

    #[test]
    fn test_verification_updates_reputation() {
        let config = TrustConfig::default();
        let mut scorer = TrustScorer::new(config);

        let source = "http://example.org/source";

        // Create source reputation
        let annotation = TripleAnnotation::new()
            .with_confidence(0.8)
            .with_source(source.to_string());
        scorer.calculate_trust(&annotation);

        let initial_reputation = scorer.get_source_reputation(source).unwrap().reputation;

        // Record successful verifications
        for _ in 0..5 {
            scorer.record_verification(source, true);
        }

        let updated_reputation = scorer.get_source_reputation(source).unwrap().reputation;

        assert!(updated_reputation > initial_reputation);
    }

    #[test]
    fn test_confidence_propagation() {
        let config = TrustConfig::default();
        let scorer = TrustScorer::new(config);

        let annotations = vec![
            TripleAnnotation::new().with_confidence(0.9),
            TripleAnnotation::new().with_confidence(0.5),
            TripleAnnotation::new().with_confidence(0.7),
        ];

        let propagated = scorer.propagate_confidence(&annotations);

        assert_eq!(propagated.len(), 3);
        assert!(propagated.iter().all(|&c| (0.0..=1.0).contains(&c)));
    }

    #[test]
    fn test_bayesian_update() {
        let config = TrustConfig::default();
        let scorer = TrustScorer::new(config);

        let prior = 0.5;
        let evidence = vec![EvidenceItem {
            evidence_type: "experimental".to_string(),
            reference: "http://example.org/study1".to_string(),
            strength: 0.9,
            description: None,
        }];

        let posterior = scorer.bayesian_update(prior, &evidence);

        assert!(posterior > prior);
        assert!((0.0..=1.0).contains(&posterior));
    }

    #[test]
    fn test_freshness_decay() {
        let config = TrustConfig::default();
        let mut scorer = TrustScorer::new(config);

        let old_annotation = TripleAnnotation {
            confidence: Some(0.9),
            timestamp: Some(Utc::now() - chrono::Duration::days(730)), // 2 years ago
            ..Default::default()
        };

        let recent_annotation = TripleAnnotation {
            confidence: Some(0.9),
            timestamp: Some(Utc::now()),
            ..Default::default()
        };

        let old_breakdown = scorer.calculate_trust(&old_annotation);
        let recent_breakdown = scorer.calculate_trust(&recent_annotation);

        assert!(recent_breakdown.freshness_component > old_breakdown.freshness_component);
    }

    #[test]
    fn test_cache_effectiveness() {
        let config = TrustConfig::default();
        let mut scorer = TrustScorer::new(config);

        let annotation = TripleAnnotation::new().with_confidence(0.9);

        // First calculation (cache miss)
        scorer.calculate_trust(&annotation);
        assert_eq!(scorer.statistics().cache_hits, 0);
        assert_eq!(scorer.statistics().cache_misses, 1);

        // Second calculation (cache hit)
        scorer.calculate_trust(&annotation);
        assert_eq!(scorer.statistics().cache_hits, 1);
        assert_eq!(scorer.statistics().cache_misses, 1);
    }
}
