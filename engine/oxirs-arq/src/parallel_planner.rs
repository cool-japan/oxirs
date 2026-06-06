//! Parallel plan generation, partition strategies, and cost-based parallelism decisions.
//!
//! This module provides utilities for determining when and how to apply parallel
//! execution to SPARQL algebra plans, including partition strategies for BGP
//! evaluation, cost-based thresholds for switching between sequential and
//! parallel execution, and helpers for decomposing complex patterns.

use crate::executor::ParallelConfig;

/// Strategy for partitioning BGP patterns across worker threads.
#[derive(Debug, Clone, PartialEq)]
pub enum BgpPartitionStrategy {
    /// Divide patterns into equal-sized chunks, one per thread.
    ChunkByPattern,
    /// Each thread handles a single predicate family for index locality.
    ByPredicateFamily,
    /// Adaptively choose chunk size based on estimated pattern selectivity.
    AdaptiveSelectivity,
}

/// Cost-based decision record for a parallel plan segment.
#[derive(Debug, Clone)]
pub struct ParallelPlanCost {
    /// Estimated number of result bindings.
    pub estimated_cardinality: usize,
    /// Number of worker threads recommended for this sub-plan.
    pub recommended_threads: usize,
    /// Whether parallel execution is estimated to be beneficial.
    pub parallel_beneficial: bool,
    /// Threshold at which switching to parallel processing is worthwhile.
    pub parallelism_threshold: usize,
}

impl ParallelPlanCost {
    /// Compute a cost estimate given configuration and an estimated input size.
    pub fn estimate(config: &ParallelConfig, estimated_input_size: usize) -> Self {
        let parallel_beneficial = estimated_input_size >= config.parallel_threshold;
        let recommended_threads = if parallel_beneficial {
            config.max_threads.min(
                // Never use more threads than we have work items
                estimated_input_size.max(1),
            )
        } else {
            1
        };

        Self {
            estimated_cardinality: estimated_input_size,
            recommended_threads,
            parallel_beneficial,
            parallelism_threshold: config.parallel_threshold,
        }
    }
}

/// Decide which BGP partition strategy to use for a given pattern count and config.
pub fn select_bgp_partition_strategy(
    pattern_count: usize,
    config: &ParallelConfig,
) -> BgpPartitionStrategy {
    if pattern_count <= config.max_threads {
        // Few patterns: assign one per thread, no benefit from sub-chunk splitting
        BgpPartitionStrategy::ChunkByPattern
    } else if pattern_count > config.parallel_threshold {
        // Large pattern sets: use adaptive strategy to balance load
        BgpPartitionStrategy::AdaptiveSelectivity
    } else {
        BgpPartitionStrategy::ByPredicateFamily
    }
}

/// Calculate the optimal chunk size for dividing `item_count` items across
/// `max_threads` worker threads, with a minimum chunk size of 1.
pub fn compute_chunk_size(item_count: usize, max_threads: usize) -> usize {
    let threads = max_threads.max(1);
    std::cmp::max(1, (item_count + threads - 1) / threads)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::ParallelConfig;

    #[test]
    fn test_compute_chunk_size_basic() {
        assert_eq!(compute_chunk_size(10, 4), 3);
        assert_eq!(compute_chunk_size(0, 4), 1);
        assert_eq!(compute_chunk_size(4, 4), 1);
        assert_eq!(compute_chunk_size(100, 10), 10);
    }

    #[test]
    fn test_parallel_plan_cost_below_threshold() {
        let config = ParallelConfig {
            parallel_threshold: 100,
            max_threads: 4,
            ..ParallelConfig::default()
        };
        let cost = ParallelPlanCost::estimate(&config, 50);
        assert!(!cost.parallel_beneficial);
        assert_eq!(cost.recommended_threads, 1);
    }

    #[test]
    fn test_parallel_plan_cost_above_threshold() {
        let config = ParallelConfig {
            parallel_threshold: 100,
            max_threads: 4,
            ..ParallelConfig::default()
        };
        let cost = ParallelPlanCost::estimate(&config, 200);
        assert!(cost.parallel_beneficial);
        assert!(cost.recommended_threads > 1);
    }

    #[test]
    fn test_select_bgp_partition_strategy() {
        let config = ParallelConfig {
            parallel_threshold: 50,
            max_threads: 8,
            ..ParallelConfig::default()
        };
        // Few patterns → ChunkByPattern
        assert_eq!(
            select_bgp_partition_strategy(3, &config),
            BgpPartitionStrategy::ChunkByPattern
        );
        // Above threshold → AdaptiveSelectivity
        assert_eq!(
            select_bgp_partition_strategy(100, &config),
            BgpPartitionStrategy::AdaptiveSelectivity
        );
    }
}
