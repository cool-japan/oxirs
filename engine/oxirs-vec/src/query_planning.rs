//! Query planning and cost estimation for vector search operations
//!
//! This module provides intelligent query planning to select the optimal
//! search strategy based on query characteristics and index statistics.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Query execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QueryStrategy {
    /// Exhaustive linear scan (most accurate, slowest)
    ExhaustiveScan,
    /// HNSW approximate search
    HnswApproximate,
    /// NSG (Navigable Small World Graph) approximate search
    NsgApproximate,
    /// IVF with coarse quantization
    IvfCoarse,
    /// Product quantization with refinement
    ProductQuantization,
    /// Scalar quantization
    ScalarQuantization,
    /// LSH approximate search
    LocalitySensitiveHashing,
    /// GPU-accelerated search
    GpuAccelerated,
    /// Hybrid strategy (multiple indices)
    Hybrid,
}

/// Cost model for query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    /// Cost per distance computation (CPU microseconds)
    pub distance_computation_cost_us: f64,
    /// Cost per index lookup (CPU microseconds)
    pub index_lookup_cost_us: f64,
    /// Cost per memory access (nanoseconds)
    pub memory_access_cost_ns: f64,
    /// GPU availability and cost multiplier
    pub gpu_available: bool,
    pub gpu_cost_multiplier: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            distance_computation_cost_us: 0.5,
            index_lookup_cost_us: 0.1,
            memory_access_cost_ns: 50.0,
            gpu_available: false,
            gpu_cost_multiplier: 0.1, // GPU is 10x faster
        }
    }
}

/// Query characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCharacteristics {
    /// Number of results requested (k)
    pub k: usize,
    /// Vector dimensionality
    pub dimensions: usize,
    /// Minimum acceptable recall
    pub min_recall: f32,
    /// Maximum acceptable latency
    pub max_latency_ms: f64,
    /// Query type
    pub query_type: VectorQueryType,
}

/// Type of vector query
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VectorQueryType {
    /// Single vector query
    Single,
    /// Batch of queries
    Batch(usize),
    /// Streaming queries
    Streaming,
}

/// Index statistics for planning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatistics {
    /// Total number of vectors
    pub vector_count: usize,
    /// Vector dimensionality
    pub dimensions: usize,
    /// Available index types
    pub available_indices: Vec<QueryStrategy>,
    /// Average query latencies by strategy (milliseconds)
    pub avg_latencies: HashMap<QueryStrategy, f64>,
    /// Average recalls by strategy
    pub avg_recalls: HashMap<QueryStrategy, f32>,
}

/// Query execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    /// Selected strategy
    pub strategy: QueryStrategy,
    /// Estimated cost (microseconds)
    pub estimated_cost_us: f64,
    /// Estimated recall
    pub estimated_recall: f32,
    /// Confidence in plan (0.0 to 1.0)
    pub confidence: f32,
    /// Alternative strategies considered
    pub alternatives: Vec<(QueryStrategy, f64, f32)>, // (strategy, cost, recall)
    /// Recommended parameters
    pub parameters: HashMap<String, String>,
}

/// Query planner
pub struct QueryPlanner {
    cost_model: CostModel,
    index_stats: IndexStatistics,
}

impl QueryPlanner {
    /// Create a new query planner
    pub fn new(cost_model: CostModel, index_stats: IndexStatistics) -> Self {
        Self {
            cost_model,
            index_stats,
        }
    }

    /// Plan optimal query execution strategy
    pub fn plan(&self, query: &QueryCharacteristics) -> Result<QueryPlan> {
        let mut candidates = Vec::new();

        // Evaluate each available strategy
        for strategy in &self.index_stats.available_indices {
            let (cost, recall) = self.estimate_strategy(*strategy, query);
            candidates.push((*strategy, cost, recall));
        }

        // Sort by cost
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Find best strategy that meets recall requirement
        let best = candidates
            .iter()
            .find(|(_, _, recall)| *recall >= query.min_recall)
            .or_else(|| candidates.first())
            .ok_or_else(|| anyhow::anyhow!("No suitable strategy found"))?;

        let (strategy, cost, recall) = *best;

        // Generate parameters for selected strategy
        let parameters = self.generate_parameters(strategy, query);

        // Calculate confidence based on historical data
        let confidence = self.calculate_confidence(strategy);

        Ok(QueryPlan {
            strategy,
            estimated_cost_us: cost,
            estimated_recall: recall,
            confidence,
            alternatives: candidates
                .iter()
                .filter(|(s, _, _)| *s != strategy)
                .take(3)
                .copied()
                .collect(),
            parameters,
        })
    }

    /// Estimate cost and recall for a strategy
    fn estimate_strategy(
        &self,
        strategy: QueryStrategy,
        query: &QueryCharacteristics,
    ) -> (f64, f32) {
        let base_cost = match strategy {
            QueryStrategy::ExhaustiveScan => {
                // Cost = number of vectors * distance computation cost
                self.index_stats.vector_count as f64 * self.cost_model.distance_computation_cost_us
            }
            QueryStrategy::HnswApproximate => {
                // Cost ≈ log(N) * M * distance computation
                let hnsw_complexity = (self.index_stats.vector_count as f64).ln() * 16.0;
                hnsw_complexity * self.cost_model.distance_computation_cost_us
            }
            QueryStrategy::NsgApproximate => {
                // NSG is typically more efficient than HNSW due to monotonic search
                // Cost ≈ log(N) * out_degree (typically 32) * distance computation
                let nsg_complexity = (self.index_stats.vector_count as f64).ln() * 12.0;
                nsg_complexity * self.cost_model.distance_computation_cost_us
            }
            QueryStrategy::IvfCoarse => {
                // Cost ≈ sqrt(N) * distance computation
                let ivf_probes = (self.index_stats.vector_count as f64).sqrt();
                ivf_probes * self.cost_model.distance_computation_cost_us
            }
            QueryStrategy::ProductQuantization => {
                // Lower cost due to compressed distance computations
                let pq_cost = self.index_stats.vector_count as f64 * 0.1;
                pq_cost * self.cost_model.distance_computation_cost_us
            }
            QueryStrategy::ScalarQuantization => {
                // Similar to PQ but slightly faster
                let sq_cost = self.index_stats.vector_count as f64 * 0.08;
                sq_cost * self.cost_model.distance_computation_cost_us
            }
            QueryStrategy::LocalitySensitiveHashing => {
                // Cost ≈ number of hash tables * bucket size
                let lsh_cost = 10.0 * 100.0; // Example: 10 tables, 100 vectors per bucket
                lsh_cost * self.cost_model.distance_computation_cost_us
            }
            QueryStrategy::GpuAccelerated => {
                if self.cost_model.gpu_available {
                    let cpu_cost = self.index_stats.vector_count as f64
                        * self.cost_model.distance_computation_cost_us;
                    cpu_cost * self.cost_model.gpu_cost_multiplier
                } else {
                    f64::INFINITY // Not available
                }
            }
            QueryStrategy::Hybrid => {
                // Combine HNSW + refinement
                let hnsw_cost = (self.index_stats.vector_count as f64).ln() * 16.0;
                let refinement_cost = query.k as f64 * 10.0;
                (hnsw_cost + refinement_cost) * self.cost_model.distance_computation_cost_us
            }
        };

        // Adjust for batch queries
        let cost = match query.query_type {
            VectorQueryType::Single => base_cost,
            VectorQueryType::Batch(n) => base_cost * n as f64 * 0.8, // 20% batch efficiency
            VectorQueryType::Streaming => base_cost * 1.2,           // 20% overhead for streaming
        };

        // Get historical recall or estimate
        let recall = self
            .index_stats
            .avg_recalls
            .get(&strategy)
            .copied()
            .unwrap_or_else(|| self.estimate_recall(strategy));

        (cost, recall)
    }

    /// Estimate recall for a strategy
    fn estimate_recall(&self, strategy: QueryStrategy) -> f32 {
        match strategy {
            QueryStrategy::ExhaustiveScan => 1.0,
            QueryStrategy::HnswApproximate => 0.95,
            QueryStrategy::NsgApproximate => 0.96, // NSG typically has slightly better recall than HNSW
            QueryStrategy::IvfCoarse => 0.85,
            QueryStrategy::ProductQuantization => 0.90,
            QueryStrategy::ScalarQuantization => 0.92,
            QueryStrategy::LocalitySensitiveHashing => 0.80,
            QueryStrategy::GpuAccelerated => 0.95,
            QueryStrategy::Hybrid => 0.98,
        }
    }

    /// Generate recommended parameters for strategy
    fn generate_parameters(
        &self,
        strategy: QueryStrategy,
        query: &QueryCharacteristics,
    ) -> HashMap<String, String> {
        let mut params = HashMap::new();

        match strategy {
            QueryStrategy::HnswApproximate => {
                // Adaptive ef_search based on k and recall requirement
                let ef_search = if query.min_recall >= 0.95 {
                    (query.k * 4).max(64)
                } else {
                    (query.k * 2).max(32)
                };
                params.insert("ef_search".to_string(), ef_search.to_string());
            }
            QueryStrategy::NsgApproximate => {
                // NSG search length based on k and recall requirement
                let search_length = if query.min_recall >= 0.95 {
                    (query.k * 5).max(50)
                } else {
                    (query.k * 3).max(30)
                };
                params.insert("search_length".to_string(), search_length.to_string());
                params.insert("out_degree".to_string(), "32".to_string());
            }
            QueryStrategy::IvfCoarse => {
                let nprobe = if query.min_recall >= 0.90 { 16 } else { 8 };
                params.insert("nprobe".to_string(), nprobe.to_string());
            }
            QueryStrategy::LocalitySensitiveHashing => {
                params.insert("num_probes".to_string(), "3".to_string());
            }
            _ => {}
        }

        params
    }

    /// Calculate confidence in plan based on historical data
    fn calculate_confidence(&self, strategy: QueryStrategy) -> f32 {
        // Higher confidence if we have historical data
        if self.index_stats.avg_latencies.contains_key(&strategy) {
            0.9
        } else {
            0.5 // Lower confidence for estimated values
        }
    }

    /// Update index statistics with observed performance
    pub fn update_statistics(&mut self, strategy: QueryStrategy, latency_ms: f64, recall: f32) {
        self.index_stats.avg_latencies.insert(strategy, latency_ms);
        self.index_stats.avg_recalls.insert(strategy, recall);
    }

    /// Update index metadata (vector count, dimensions)
    pub fn update_index_metadata(&mut self, vector_count: usize, dimensions: usize) {
        self.index_stats.vector_count = vector_count;
        self.index_stats.dimensions = dimensions;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_stats() -> IndexStatistics {
        IndexStatistics {
            vector_count: 100_000,
            dimensions: 128,
            available_indices: vec![
                QueryStrategy::ExhaustiveScan,
                QueryStrategy::HnswApproximate,
                QueryStrategy::IvfCoarse,
            ],
            avg_latencies: HashMap::new(),
            avg_recalls: HashMap::new(),
        }
    }

    #[test]
    fn test_query_planner_creation() {
        let cost_model = CostModel::default();
        let stats = create_test_stats();
        let _planner = QueryPlanner::new(cost_model, stats);
    }

    #[test]
    fn test_query_planning() {
        let planner = QueryPlanner::new(CostModel::default(), create_test_stats());

        let query = QueryCharacteristics {
            k: 10,
            dimensions: 128,
            min_recall: 0.90,
            max_latency_ms: 100.0,
            query_type: VectorQueryType::Single,
        };

        let plan = planner.plan(&query);
        assert!(plan.is_ok());

        let plan = plan.unwrap();
        assert!(plan.estimated_recall >= query.min_recall);
        assert!(!plan.alternatives.is_empty());
    }

    #[test]
    fn test_exhaustive_vs_approximate() {
        let planner = QueryPlanner::new(CostModel::default(), create_test_stats());

        // High recall requirement should avoid exhaustive if approximate is available
        let query = QueryCharacteristics {
            k: 10,
            dimensions: 128,
            min_recall: 0.95,
            max_latency_ms: 10.0,
            query_type: VectorQueryType::Single,
        };

        let plan = planner.plan(&query).unwrap();
        // Should prefer HNSW over exhaustive for speed
        assert_ne!(plan.strategy, QueryStrategy::ExhaustiveScan);
    }

    #[test]
    fn test_batch_query_planning() {
        let planner = QueryPlanner::new(CostModel::default(), create_test_stats());

        let query = QueryCharacteristics {
            k: 10,
            dimensions: 128,
            min_recall: 0.90,
            max_latency_ms: 100.0,
            query_type: VectorQueryType::Batch(100),
        };

        let plan = planner.plan(&query).unwrap();
        assert!(plan.estimated_cost_us > 0.0);
    }

    #[test]
    fn test_statistics_update() {
        let mut planner = QueryPlanner::new(CostModel::default(), create_test_stats());

        planner.update_statistics(QueryStrategy::HnswApproximate, 5.0, 0.96);

        assert_eq!(
            planner
                .index_stats
                .avg_latencies
                .get(&QueryStrategy::HnswApproximate),
            Some(&5.0)
        );
        assert_eq!(
            planner
                .index_stats
                .avg_recalls
                .get(&QueryStrategy::HnswApproximate),
            Some(&0.96)
        );
    }
}
