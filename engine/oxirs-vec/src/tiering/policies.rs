//! Tiering policies for index placement and transition decisions

use super::types::{AccessStatistics, IndexMetadata, StorageTier, TierStatistics};
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

/// Tiering policy for determining index placement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TieringPolicy {
    /// Least Recently Used (LRU)
    Lru,
    /// Least Frequently Used (LFU)
    Lfu,
    /// Cost-based optimization
    CostBased,
    /// Size-aware placement
    SizeBased,
    /// Latency-sensitive optimization
    LatencyOptimized,
    /// Adaptive policy (ML-driven)
    Adaptive,
    /// Custom policy with user-defined rules
    Custom,
}

impl Default for TieringPolicy {
    fn default() -> Self {
        Self::Adaptive
    }
}

/// Reason for tier transition
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TierTransitionReason {
    /// High access frequency (promotion)
    HighAccessFrequency,
    /// Low access frequency (demotion)
    LowAccessFrequency,
    /// Tier capacity exceeded
    CapacityExceeded,
    /// Cost optimization
    CostOptimization,
    /// Latency optimization
    LatencyOptimization,
    /// Manual intervention
    Manual,
    /// Predictive (ML-driven)
    Predictive,
    /// SLA requirements
    SlaRequirement,
    /// Emergency (e.g., out of memory)
    Emergency,
}

/// Policy evaluator for tier placement decisions
pub struct PolicyEvaluator {
    policy: TieringPolicy,
}

impl PolicyEvaluator {
    /// Create a new policy evaluator
    pub fn new(policy: TieringPolicy) -> Self {
        Self { policy }
    }

    /// Evaluate optimal tier for an index
    pub fn evaluate_optimal_tier(
        &self,
        metadata: &IndexMetadata,
        tier_stats: &[TierStatistics; 3],
        current_time: SystemTime,
    ) -> (StorageTier, TierTransitionReason) {
        match self.policy {
            TieringPolicy::Lru => self.evaluate_lru(metadata, tier_stats, current_time),
            TieringPolicy::Lfu => self.evaluate_lfu(metadata, tier_stats),
            TieringPolicy::CostBased => self.evaluate_cost_based(metadata, tier_stats),
            TieringPolicy::SizeBased => self.evaluate_size_based(metadata, tier_stats),
            TieringPolicy::LatencyOptimized => {
                self.evaluate_latency_optimized(metadata, tier_stats)
            }
            TieringPolicy::Adaptive => self.evaluate_adaptive(metadata, tier_stats, current_time),
            TieringPolicy::Custom => {
                // Default to adaptive for custom policy
                self.evaluate_adaptive(metadata, tier_stats, current_time)
            }
        }
    }

    /// LRU policy: Place in tier based on recency of access
    fn evaluate_lru(
        &self,
        metadata: &IndexMetadata,
        tier_stats: &[TierStatistics; 3],
        current_time: SystemTime,
    ) -> (StorageTier, TierTransitionReason) {
        let time_since_access = metadata
            .access_stats
            .last_access_time
            .and_then(|t| current_time.duration_since(t).ok())
            .unwrap_or(Duration::from_secs(u64::MAX));

        // Hot: accessed in last 1 hour
        // Warm: accessed in last 24 hours
        // Cold: accessed more than 24 hours ago
        if time_since_access < Duration::from_secs(3600) {
            if self.has_capacity(&tier_stats[0], metadata.size_bytes) {
                (StorageTier::Hot, TierTransitionReason::HighAccessFrequency)
            } else {
                (StorageTier::Warm, TierTransitionReason::CapacityExceeded)
            }
        } else if time_since_access < Duration::from_secs(86400) {
            if self.has_capacity(&tier_stats[1], metadata.size_bytes) {
                (StorageTier::Warm, TierTransitionReason::LowAccessFrequency)
            } else {
                (StorageTier::Cold, TierTransitionReason::CapacityExceeded)
            }
        } else {
            (StorageTier::Cold, TierTransitionReason::LowAccessFrequency)
        }
    }

    /// LFU policy: Place in tier based on access frequency
    fn evaluate_lfu(
        &self,
        metadata: &IndexMetadata,
        tier_stats: &[TierStatistics; 3],
    ) -> (StorageTier, TierTransitionReason) {
        let qps = metadata.access_stats.avg_qps;

        // Hot: > 10 QPS
        // Warm: 1-10 QPS
        // Cold: < 1 QPS
        if qps > 10.0 {
            if self.has_capacity(&tier_stats[0], metadata.size_bytes) {
                (StorageTier::Hot, TierTransitionReason::HighAccessFrequency)
            } else {
                (StorageTier::Warm, TierTransitionReason::CapacityExceeded)
            }
        } else if qps > 1.0 {
            if self.has_capacity(&tier_stats[1], metadata.size_bytes) {
                (StorageTier::Warm, TierTransitionReason::HighAccessFrequency)
            } else {
                (StorageTier::Cold, TierTransitionReason::CapacityExceeded)
            }
        } else {
            (StorageTier::Cold, TierTransitionReason::LowAccessFrequency)
        }
    }

    /// Cost-based policy: Minimize cost while meeting performance requirements
    fn evaluate_cost_based(
        &self,
        metadata: &IndexMetadata,
        tier_stats: &[TierStatistics; 3],
    ) -> (StorageTier, TierTransitionReason) {
        // Calculate cost for each tier
        let hot_cost = self.calculate_tier_cost(metadata, StorageTier::Hot);
        let warm_cost = self.calculate_tier_cost(metadata, StorageTier::Warm);
        let cold_cost = self.calculate_tier_cost(metadata, StorageTier::Cold);

        // Choose tier with minimum cost that has capacity
        if cold_cost <= warm_cost
            && cold_cost <= hot_cost
            && self.has_capacity(&tier_stats[2], metadata.size_bytes)
        {
            (StorageTier::Cold, TierTransitionReason::CostOptimization)
        } else if warm_cost <= hot_cost && self.has_capacity(&tier_stats[1], metadata.size_bytes) {
            (StorageTier::Warm, TierTransitionReason::CostOptimization)
        } else if self.has_capacity(&tier_stats[0], metadata.size_bytes) {
            (StorageTier::Hot, TierTransitionReason::CostOptimization)
        } else {
            // Fall back to cold if no capacity
            (StorageTier::Cold, TierTransitionReason::CapacityExceeded)
        }
    }

    /// Size-based policy: Large indices in cold tier, small in hot tier
    fn evaluate_size_based(
        &self,
        metadata: &IndexMetadata,
        tier_stats: &[TierStatistics; 3],
    ) -> (StorageTier, TierTransitionReason) {
        let size_gb = metadata.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        // Hot: < 1 GB
        // Warm: 1-10 GB
        // Cold: > 10 GB
        if size_gb < 1.0 && self.has_capacity(&tier_stats[0], metadata.size_bytes) {
            (StorageTier::Hot, TierTransitionReason::LatencyOptimization)
        } else if size_gb < 10.0 && self.has_capacity(&tier_stats[1], metadata.size_bytes) {
            (StorageTier::Warm, TierTransitionReason::CostOptimization)
        } else {
            (StorageTier::Cold, TierTransitionReason::CostOptimization)
        }
    }

    /// Latency-optimized policy: Prioritize query latency
    fn evaluate_latency_optimized(
        &self,
        metadata: &IndexMetadata,
        tier_stats: &[TierStatistics; 3],
    ) -> (StorageTier, TierTransitionReason) {
        // If actively queried, keep in fastest tier available
        if metadata.access_stats.avg_qps > 0.1 {
            if self.has_capacity(&tier_stats[0], metadata.size_bytes) {
                (StorageTier::Hot, TierTransitionReason::LatencyOptimization)
            } else if self.has_capacity(&tier_stats[1], metadata.size_bytes) {
                (StorageTier::Warm, TierTransitionReason::LatencyOptimization)
            } else {
                (StorageTier::Cold, TierTransitionReason::CapacityExceeded)
            }
        } else {
            (StorageTier::Cold, TierTransitionReason::LowAccessFrequency)
        }
    }

    /// Adaptive policy: Combine multiple factors
    fn evaluate_adaptive(
        &self,
        metadata: &IndexMetadata,
        tier_stats: &[TierStatistics; 3],
        current_time: SystemTime,
    ) -> (StorageTier, TierTransitionReason) {
        // Calculate scores for each tier (higher is better)
        let hot_score = self.calculate_adaptive_score(metadata, StorageTier::Hot, current_time);
        let warm_score = self.calculate_adaptive_score(metadata, StorageTier::Warm, current_time);
        let cold_score = self.calculate_adaptive_score(metadata, StorageTier::Cold, current_time);

        // Choose tier with highest score that has capacity
        if hot_score >= warm_score
            && hot_score >= cold_score
            && self.has_capacity(&tier_stats[0], metadata.size_bytes)
        {
            (StorageTier::Hot, TierTransitionReason::HighAccessFrequency)
        } else if warm_score >= cold_score && self.has_capacity(&tier_stats[1], metadata.size_bytes)
        {
            (StorageTier::Warm, TierTransitionReason::CostOptimization)
        } else {
            (StorageTier::Cold, TierTransitionReason::LowAccessFrequency)
        }
    }

    /// Calculate adaptive score for tier placement
    fn calculate_adaptive_score(
        &self,
        metadata: &IndexMetadata,
        tier: StorageTier,
        current_time: SystemTime,
    ) -> f64 {
        let mut score = 0.0;

        // Factor 1: Access frequency (40% weight)
        let qps_factor = metadata.access_stats.avg_qps.min(100.0) / 100.0;
        score += qps_factor * 0.4;

        // Factor 2: Recency (30% weight)
        let recency_factor = metadata
            .access_stats
            .last_access_time
            .and_then(|t| current_time.duration_since(t).ok())
            .map(|d| {
                let hours = d.as_secs() as f64 / 3600.0;
                1.0 / (1.0 + hours)
            })
            .unwrap_or(0.0);
        score += recency_factor * 0.3;

        // Factor 3: Cost efficiency (20% weight)
        let cost = self.calculate_tier_cost(metadata, tier);
        let cost_factor = 1.0 / (1.0 + cost);
        score += cost_factor * 0.2;

        // Factor 4: Latency sensitivity (10% weight)
        let latency_factor = match tier {
            StorageTier::Hot => 1.0,
            StorageTier::Warm => 0.5,
            StorageTier::Cold => 0.1,
        };
        score += latency_factor * 0.1;

        score
    }

    /// Calculate cost for placing index in a tier
    fn calculate_tier_cost(&self, metadata: &IndexMetadata, tier: StorageTier) -> f64 {
        let size_gb = metadata.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let storage_cost = size_gb * tier.cost_factor();
        let query_cost = metadata.access_stats.avg_qps * tier.typical_latency().as_secs_f64();
        storage_cost + query_cost
    }

    /// Check if tier has capacity for index
    fn has_capacity(&self, tier_stats: &TierStatistics, size_bytes: u64) -> bool {
        tier_stats.available_bytes() >= size_bytes
    }
}

/// Calculate access score for tier promotion/demotion decisions
pub fn calculate_access_score(stats: &AccessStatistics) -> f64 {
    let mut score = 0.0;

    // Recent access frequency (50% weight)
    score += (stats.avg_qps.min(100.0) / 100.0) * 0.5;

    // Peak QPS (20% weight)
    score += (stats.peak_qps.min(1000.0) / 1000.0) * 0.2;

    // Total queries (15% weight)
    let total_queries_normalized = stats.total_queries.min(1_000_000) as f64 / 1_000_000.0;
    score += total_queries_normalized * 0.15;

    // Recent activity (15% weight)
    let recent_activity =
        (stats.queries_last_hour as f64).max((stats.queries_last_day as f64) / 24.0);
    score += (recent_activity.min(1000.0) / 1000.0) * 0.15;

    score.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiering::types::{AccessPattern, IndexType, LatencyPercentiles, PerformanceMetrics};
    use std::collections::HashMap;

    fn create_test_metadata(qps: f64, size_gb: f64) -> IndexMetadata {
        IndexMetadata {
            index_id: "test_index".to_string(),
            current_tier: StorageTier::Warm,
            size_bytes: (size_gb * 1024.0 * 1024.0 * 1024.0) as u64,
            compressed_size_bytes: (size_gb * 512.0 * 1024.0 * 1024.0) as u64,
            vector_count: 1_000_000,
            dimension: 768,
            index_type: IndexType::Hnsw,
            created_at: SystemTime::now(),
            last_accessed: SystemTime::now(),
            last_modified: SystemTime::now(),
            access_stats: AccessStatistics {
                total_queries: 100_000,
                queries_last_hour: (qps * 3600.0) as u64,
                queries_last_day: (qps * 86400.0) as u64,
                queries_last_week: (qps * 604800.0) as u64,
                avg_qps: qps,
                peak_qps: qps * 2.0,
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
                capacity_bytes: 16 * 1024 * 1024 * 1024, // 16 GB
                used_bytes: 8 * 1024 * 1024 * 1024,      // 8 GB used
                ..Default::default()
            },
            TierStatistics {
                capacity_bytes: 128 * 1024 * 1024 * 1024, // 128 GB
                used_bytes: 64 * 1024 * 1024 * 1024,      // 64 GB used
                ..Default::default()
            },
            TierStatistics {
                capacity_bytes: 1024 * 1024 * 1024 * 1024, // 1 TB
                used_bytes: 256 * 1024 * 1024 * 1024,      // 256 GB used
                ..Default::default()
            },
        ]
    }

    #[test]
    fn test_lfu_policy_high_qps() {
        let evaluator = PolicyEvaluator::new(TieringPolicy::Lfu);
        let metadata = create_test_metadata(20.0, 1.0); // 20 QPS, 1 GB
        let tier_stats = create_test_tier_stats();

        let (tier, _reason) = evaluator.evaluate_lfu(&metadata, &tier_stats);
        assert_eq!(tier, StorageTier::Hot);
    }

    #[test]
    fn test_lfu_policy_medium_qps() {
        let evaluator = PolicyEvaluator::new(TieringPolicy::Lfu);
        let metadata = create_test_metadata(5.0, 1.0); // 5 QPS, 1 GB
        let tier_stats = create_test_tier_stats();

        let (tier, _reason) = evaluator.evaluate_lfu(&metadata, &tier_stats);
        assert_eq!(tier, StorageTier::Warm);
    }

    #[test]
    fn test_lfu_policy_low_qps() {
        let evaluator = PolicyEvaluator::new(TieringPolicy::Lfu);
        let metadata = create_test_metadata(0.5, 1.0); // 0.5 QPS, 1 GB
        let tier_stats = create_test_tier_stats();

        let (tier, _reason) = evaluator.evaluate_lfu(&metadata, &tier_stats);
        assert_eq!(tier, StorageTier::Cold);
    }

    #[test]
    fn test_size_based_policy() {
        let evaluator = PolicyEvaluator::new(TieringPolicy::SizeBased);
        let tier_stats = create_test_tier_stats();

        // Small index -> Hot
        let small_metadata = create_test_metadata(1.0, 0.5);
        let (tier, _) = evaluator.evaluate_size_based(&small_metadata, &tier_stats);
        assert_eq!(tier, StorageTier::Hot);

        // Medium index -> Warm
        let medium_metadata = create_test_metadata(1.0, 5.0);
        let (tier, _) = evaluator.evaluate_size_based(&medium_metadata, &tier_stats);
        assert_eq!(tier, StorageTier::Warm);

        // Large index -> Cold
        let large_metadata = create_test_metadata(1.0, 20.0);
        let (tier, _) = evaluator.evaluate_size_based(&large_metadata, &tier_stats);
        assert_eq!(tier, StorageTier::Cold);
    }

    #[test]
    fn test_access_score_calculation() {
        let stats = AccessStatistics {
            total_queries: 500_000,
            queries_last_hour: 3600,
            queries_last_day: 86400,
            queries_last_week: 604800,
            avg_qps: 10.0,
            peak_qps: 20.0,
            last_access_time: Some(SystemTime::now()),
            access_pattern: AccessPattern::Hot,
            query_latencies: LatencyPercentiles::default(),
        };

        let score = calculate_access_score(&stats);
        assert!(score > 0.0 && score <= 1.0);
    }
}
