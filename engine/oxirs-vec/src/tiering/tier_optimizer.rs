//! Tier optimization using ML and statistical analysis

use super::config::TieringConfig;
use super::policies::{PolicyEvaluator, TierTransitionReason};
use super::types::{IndexMetadata, StorageTier, TierStatistics};
use scirs2_core::ndarray_ext::{Array1, Array2};
use std::collections::HashMap;
use std::time::SystemTime;

/// Tier optimizer using ML and statistical analysis
pub struct TierOptimizer {
    /// Policy evaluator
    policy_evaluator: PolicyEvaluator,
    /// Configuration
    config: TieringConfig,
    /// Historical decisions
    decision_history: Vec<OptimizationDecision>,
    /// Feature importance weights (learned)
    feature_weights: Array1<f64>,
}

/// Optimization decision record
#[derive(Debug, Clone)]
struct OptimizationDecision {
    index_id: String,
    from_tier: StorageTier,
    to_tier: StorageTier,
    reason: TierTransitionReason,
    timestamp: SystemTime,
    features: Array1<f64>,
    outcome_score: f64,
}

impl TierOptimizer {
    /// Create a new tier optimizer
    pub fn new(config: TieringConfig) -> Self {
        let policy = config.policy;
        let feature_weights = Array1::from_vec(vec![1.0; 10]); // 10 features

        Self {
            policy_evaluator: PolicyEvaluator::new(policy),
            config,
            decision_history: Vec::new(),
            feature_weights,
        }
    }

    /// Optimize tier placements for all indices
    pub fn optimize_tier_placements(
        &mut self,
        indices: &[IndexMetadata],
        tier_stats: &[TierStatistics; 3],
    ) -> Vec<TierOptimizationRecommendation> {
        let mut recommendations = Vec::new();

        for metadata in indices {
            if let Some(recommendation) = self.evaluate_index(metadata, tier_stats) {
                recommendations.push(recommendation);
            }
        }

        // Sort by priority (highest first)
        recommendations.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap());

        recommendations
    }

    /// Evaluate a single index for tier optimization
    fn evaluate_index(
        &mut self,
        metadata: &IndexMetadata,
        tier_stats: &[TierStatistics; 3],
    ) -> Option<TierOptimizationRecommendation> {
        let current_tier = metadata.current_tier;
        let (optimal_tier, reason) =
            self.policy_evaluator
                .evaluate_optimal_tier(metadata, tier_stats, SystemTime::now());

        // Check if transition is needed
        if optimal_tier == current_tier {
            return None;
        }

        // Extract features
        let features = self.extract_features(metadata, tier_stats);

        // Calculate priority using learned weights
        let priority = self.calculate_priority(&features, current_tier, optimal_tier);

        // Estimate benefit
        let benefit = self.estimate_benefit(metadata, current_tier, optimal_tier);

        Some(TierOptimizationRecommendation {
            index_id: metadata.index_id.clone(),
            current_tier,
            recommended_tier: optimal_tier,
            reason,
            priority,
            estimated_benefit: benefit,
            confidence: self.calculate_confidence(&features),
        })
    }

    /// Extract features for ML-based optimization
    fn extract_features(
        &self,
        metadata: &IndexMetadata,
        tier_stats: &[TierStatistics; 3],
    ) -> Array1<f64> {
        let mut features = Vec::with_capacity(10);

        // Feature 1: Access frequency (QPS)
        features.push(metadata.access_stats.avg_qps.ln_1p());

        // Feature 2: Index size (GB)
        let size_gb = metadata.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        features.push(size_gb.ln_1p());

        // Feature 3: Recency (hours since last access)
        let recency = metadata
            .access_stats
            .last_access_time
            .and_then(|t| SystemTime::now().duration_since(t).ok())
            .map(|d| d.as_secs() as f64 / 3600.0)
            .unwrap_or(1000.0);
        features.push(recency.ln_1p());

        // Feature 4: Query latency (p95)
        features.push((metadata.access_stats.query_latencies.p95 as f64).ln_1p());

        // Feature 5: Current tier utilization
        let tier_idx = match metadata.current_tier {
            StorageTier::Hot => 0,
            StorageTier::Warm => 1,
            StorageTier::Cold => 2,
        };
        features.push(tier_stats[tier_idx].utilization());

        // Feature 6: Peak QPS
        features.push(metadata.access_stats.peak_qps.ln_1p());

        // Feature 7: Total queries
        features.push((metadata.access_stats.total_queries as f64).ln_1p());

        // Feature 8: Memory footprint
        let memory_gb =
            metadata.performance_metrics.memory_footprint_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        features.push(memory_gb.ln_1p());

        // Feature 9: Cache hit rate
        features.push(metadata.performance_metrics.cache_hit_rate);

        // Feature 10: Time in current tier (hours)
        let time_in_tier = metadata
            .last_modified
            .elapsed()
            .map(|d| d.as_secs() as f64 / 3600.0)
            .unwrap_or(0.0);
        features.push(time_in_tier.ln_1p());

        Array1::from_vec(features)
    }

    /// Calculate priority for tier transition
    fn calculate_priority(
        &self,
        features: &Array1<f64>,
        from_tier: StorageTier,
        to_tier: StorageTier,
    ) -> f64 {
        // Weighted sum of features
        let mut priority = features.dot(&self.feature_weights);

        // Boost priority for promotions (moving to faster tier)
        if matches!(
            (from_tier, to_tier),
            (StorageTier::Cold, StorageTier::Warm)
                | (StorageTier::Cold, StorageTier::Hot)
                | (StorageTier::Warm, StorageTier::Hot)
        ) {
            priority *= 1.5;
        }

        // Reduce priority for demotions
        if matches!(
            (from_tier, to_tier),
            (StorageTier::Hot, StorageTier::Warm)
                | (StorageTier::Hot, StorageTier::Cold)
                | (StorageTier::Warm, StorageTier::Cold)
        ) {
            priority *= 0.7;
        }

        priority.max(0.0)
    }

    /// Estimate benefit of tier transition
    fn estimate_benefit(
        &self,
        metadata: &IndexMetadata,
        from_tier: StorageTier,
        to_tier: StorageTier,
    ) -> f64 {
        // Latency improvement
        let latency_improvement = from_tier.typical_latency().as_micros() as f64
            - to_tier.typical_latency().as_micros() as f64;
        let latency_benefit = latency_improvement * metadata.access_stats.avg_qps;

        // Cost change
        let from_cost = from_tier.cost_factor();
        let to_cost = to_tier.cost_factor();
        let size_gb = metadata.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let cost_change = (from_cost - to_cost) * size_gb;

        // Combined benefit (latency savings - cost increase)
        latency_benefit + cost_change * 1000.0 // Weight cost changes
    }

    /// Calculate confidence in recommendation
    fn calculate_confidence(&self, features: &Array1<f64>) -> f64 {
        // Use variance of features as uncertainty measure
        let mean = features.mean().unwrap_or(0.0);
        let variance =
            features.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / features.len() as f64;

        // Higher variance = lower confidence
        let confidence = 1.0 / (1.0 + variance);
        confidence.clamp(0.0, 1.0)
    }

    /// Learn from optimization outcomes
    pub fn update_from_feedback(
        &mut self,
        index_id: &str,
        from_tier: StorageTier,
        to_tier: StorageTier,
        reason: TierTransitionReason,
        features: Array1<f64>,
        outcome_score: f64,
    ) {
        let decision = OptimizationDecision {
            index_id: index_id.to_string(),
            from_tier,
            to_tier,
            reason,
            timestamp: SystemTime::now(),
            features,
            outcome_score,
        };

        self.decision_history.push(decision);

        // Update feature weights periodically
        if self.decision_history.len() % 100 == 0 {
            self.update_feature_weights();
        }
    }

    /// Update feature weights based on historical decisions
    fn update_feature_weights(&mut self) {
        if self.decision_history.len() < 10 {
            return; // Need minimum history
        }

        // Build feature matrix and outcome vector
        let n = self.decision_history.len();
        let mut features = Array2::zeros((n, 10));
        let mut outcomes = Array1::zeros(n);

        for (i, decision) in self.decision_history.iter().enumerate() {
            for (j, &feature) in decision.features.iter().enumerate() {
                features[[i, j]] = feature;
            }
            outcomes[i] = decision.outcome_score;
        }

        // Simple gradient descent update (learning rate = 0.01)
        let learning_rate = 0.01;
        for i in 0..10 {
            let feature_col = features.column(i);
            let correlation = feature_col.dot(&outcomes) / n as f64;
            self.feature_weights[i] += learning_rate * correlation;
        }

        // Normalize weights
        let sum: f64 = self.feature_weights.iter().map(|w| w.abs()).sum();
        if sum > 0.0 {
            self.feature_weights /= sum;
        }
    }

    /// Predict optimal tier for a new index
    pub fn predict_optimal_tier(
        &mut self,
        metadata: &IndexMetadata,
        tier_stats: &[TierStatistics; 3],
    ) -> (StorageTier, f64) {
        let features = self.extract_features(metadata, tier_stats);

        // Calculate scores for each tier
        let mut best_tier = StorageTier::Cold;
        let mut best_score = f64::NEG_INFINITY;

        for &tier in &[StorageTier::Hot, StorageTier::Warm, StorageTier::Cold] {
            let score = self.calculate_tier_score(&features, tier);
            if score > best_score {
                best_score = score;
                best_tier = tier;
            }
        }

        (best_tier, best_score)
    }

    /// Calculate score for placing index in a tier
    fn calculate_tier_score(&self, features: &Array1<f64>, tier: StorageTier) -> f64 {
        let base_score = features.dot(&self.feature_weights);

        // Adjust score based on tier characteristics
        let tier_multiplier = match tier {
            StorageTier::Hot => 1.2,
            StorageTier::Warm => 1.0,
            StorageTier::Cold => 0.8,
        };

        base_score * tier_multiplier
    }

    /// Get optimization statistics
    pub fn get_optimization_stats(&self) -> OptimizationStats {
        let total_decisions = self.decision_history.len();
        let avg_outcome = if total_decisions > 0 {
            self.decision_history
                .iter()
                .map(|d| d.outcome_score)
                .sum::<f64>()
                / total_decisions as f64
        } else {
            0.0
        };

        let tier_transitions = self.count_tier_transitions();

        OptimizationStats {
            total_decisions,
            avg_outcome_score: avg_outcome,
            tier_transitions,
            feature_importance: self.feature_weights.clone(),
        }
    }

    /// Count transitions between tiers
    fn count_tier_transitions(&self) -> HashMap<(StorageTier, StorageTier), usize> {
        let mut transitions = HashMap::new();

        for decision in &self.decision_history {
            *transitions
                .entry((decision.from_tier, decision.to_tier))
                .or_insert(0) += 1;
        }

        transitions
    }
}

/// Tier optimization recommendation
#[derive(Debug, Clone)]
pub struct TierOptimizationRecommendation {
    /// Index identifier
    pub index_id: String,
    /// Current tier
    pub current_tier: StorageTier,
    /// Recommended tier
    pub recommended_tier: StorageTier,
    /// Reason for recommendation
    pub reason: TierTransitionReason,
    /// Priority (higher = more important)
    pub priority: f64,
    /// Estimated benefit of transition
    pub estimated_benefit: f64,
    /// Confidence in recommendation (0.0 - 1.0)
    pub confidence: f64,
}

/// Optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Total number of decisions made
    pub total_decisions: usize,
    /// Average outcome score
    pub avg_outcome_score: f64,
    /// Transition counts between tiers
    pub tier_transitions: HashMap<(StorageTier, StorageTier), usize>,
    /// Learned feature importance weights
    pub feature_importance: Array1<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiering::types::{
        AccessPattern, AccessStatistics, IndexType, LatencyPercentiles, PerformanceMetrics,
    };
    use std::collections::HashMap;

    fn create_test_metadata() -> IndexMetadata {
        IndexMetadata {
            index_id: "test_index".to_string(),
            current_tier: StorageTier::Warm,
            size_bytes: 1024 * 1024 * 1024, // 1 GB
            compressed_size_bytes: 512 * 1024 * 1024,
            vector_count: 1_000_000,
            dimension: 768,
            index_type: IndexType::Hnsw,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            last_modified: SystemTime::now(),
            access_stats: AccessStatistics {
                total_queries: 100_000,
                queries_last_hour: 3600,
                queries_last_day: 86400,
                queries_last_week: 604800,
                avg_qps: 10.0,
                peak_qps: 20.0,
                last_access_time: Some(SystemTime::now()),
                access_pattern: AccessPattern::Hot,
                query_latencies: LatencyPercentiles::default(),
            },
            performance_metrics: PerformanceMetrics::default(),
            storage_path: None,
            custom_metadata: HashMap::new(),
        }
    }

    fn create_test_tier_stats() -> [TierStatistics; 3] {
        [
            TierStatistics {
                capacity_bytes: 16 * 1024 * 1024 * 1024,
                used_bytes: 8 * 1024 * 1024 * 1024,
                ..Default::default()
            },
            TierStatistics {
                capacity_bytes: 128 * 1024 * 1024 * 1024,
                used_bytes: 64 * 1024 * 1024 * 1024,
                ..Default::default()
            },
            TierStatistics {
                capacity_bytes: 1024 * 1024 * 1024 * 1024,
                used_bytes: 256 * 1024 * 1024 * 1024,
                ..Default::default()
            },
        ]
    }

    #[test]
    fn test_tier_optimizer_creation() {
        let config = TieringConfig::default();
        let optimizer = TierOptimizer::new(config);

        assert_eq!(optimizer.feature_weights.len(), 10);
    }

    #[test]
    fn test_feature_extraction() {
        let config = TieringConfig::default();
        let optimizer = TierOptimizer::new(config);

        let metadata = create_test_metadata();
        let tier_stats = create_test_tier_stats();

        let features = optimizer.extract_features(&metadata, &tier_stats);
        assert_eq!(features.len(), 10);
        assert!(features.iter().all(|&f| f.is_finite()));
    }

    #[test]
    fn test_optimization_recommendations() {
        let config = TieringConfig::default();
        let mut optimizer = TierOptimizer::new(config);

        let metadata = create_test_metadata();
        let tier_stats = create_test_tier_stats();

        let recommendations = optimizer.optimize_tier_placements(&[metadata], &tier_stats);
        assert!(recommendations.len() <= 1);
    }
}
