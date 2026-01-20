//! Shard routing for distributed query execution
//!
//! This module handles intelligent query routing across shards,
//! optimizing for data locality and minimizing network traffic.

use crate::shard::{ShardId, ShardRouter};
use crate::shard_manager::ShardManager;
use crate::{ClusterError, Result};
use oxirs_core::model::Triple;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, warn};

/// Query routing plan
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Query identifier
    pub query_id: String,
    /// Shards to query
    pub shard_targets: Vec<ShardTarget>,
    /// Query optimization hints
    pub optimization_hints: QueryOptimizationHints,
    /// Estimated cost
    pub estimated_cost: f64,
}

/// Target shard for query execution
#[derive(Debug, Clone)]
pub struct ShardTarget {
    /// Shard identifier
    pub shard_id: ShardId,
    /// Preferred node for query execution
    pub preferred_node: u64,
    /// Alternative nodes
    pub alternative_nodes: Vec<u64>,
    /// Query selectivity estimate (0.0 to 1.0)
    pub selectivity: f64,
}

/// Query optimization hints
#[derive(Debug, Clone, Default)]
pub struct QueryOptimizationHints {
    /// Use index if available
    pub use_index: bool,
    /// Enable parallel execution
    pub parallel_execution: bool,
    /// Maximum results to return
    pub limit: Option<usize>,
    /// Result ordering
    pub order_by: Option<String>,
    /// Enable caching
    pub enable_cache: bool,
    /// Query timeout in milliseconds
    pub timeout_ms: Option<u64>,
}

/// Query routing statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RoutingStatistics {
    /// Total queries routed
    pub total_queries: u64,
    /// Queries hitting single shard
    pub single_shard_queries: u64,
    /// Queries spanning multiple shards
    pub multi_shard_queries: u64,
    /// Average shards per query
    pub avg_shards_per_query: f64,
    /// Cache hit rate
    pub cache_hit_rate: f64,
    /// Average query latency in ms
    pub avg_latency_ms: f64,
}

/// Query router for optimizing distributed queries
pub struct QueryRouter {
    /// Shard router
    shard_router: Arc<ShardRouter>,
    /// Shard manager
    shard_manager: Arc<ShardManager>,
    /// Query cache
    query_cache: Arc<RwLock<QueryCache>>,
    /// Routing statistics
    statistics: Arc<RwLock<RoutingStatistics>>,
    /// Cost model for query optimization
    cost_model: Arc<dyn CostModel>,
}

/// Query cache for recent routing decisions
struct QueryCache {
    /// Cache entries
    entries: HashMap<String, CacheEntry>,
    /// Maximum cache size
    max_size: usize,
    /// Cache hits
    hits: u64,
    /// Cache misses
    misses: u64,
}

/// Cache entry
struct CacheEntry {
    /// Query plan
    plan: QueryPlan,
    /// Timestamp
    #[allow(dead_code)]
    timestamp: u64,
    /// Access count
    access_count: u64,
}

/// Cost model for query optimization
pub trait CostModel: Send + Sync {
    /// Estimate query cost for a shard
    fn estimate_shard_cost(&self, shard_id: ShardId, selectivity: f64) -> f64;

    /// Estimate network transfer cost
    fn estimate_network_cost(&self, data_size: usize, hop_count: u32) -> f64;

    /// Estimate merge cost for multi-shard results
    fn estimate_merge_cost(&self, shard_count: usize, result_size: usize) -> f64;
}

/// Default cost model implementation
pub struct DefaultCostModel;

impl CostModel for DefaultCostModel {
    fn estimate_shard_cost(&self, _shard_id: ShardId, selectivity: f64) -> f64 {
        // Base cost + selectivity factor
        10.0 + (100.0 * selectivity)
    }

    fn estimate_network_cost(&self, data_size: usize, hop_count: u32) -> f64 {
        // Network latency + bandwidth cost
        (hop_count as f64 * 5.0) + (data_size as f64 / 1_000_000.0)
    }

    fn estimate_merge_cost(&self, shard_count: usize, result_size: usize) -> f64 {
        // Logarithmic merge cost
        (shard_count as f64).log2() * (result_size as f64 / 1000.0)
    }
}

impl QueryRouter {
    /// Create a new query router
    pub fn new(shard_router: Arc<ShardRouter>, shard_manager: Arc<ShardManager>) -> Self {
        Self {
            shard_router,
            shard_manager,
            query_cache: Arc::new(RwLock::new(QueryCache {
                entries: HashMap::new(),
                max_size: 1000,
                hits: 0,
                misses: 0,
            })),
            statistics: Arc::new(RwLock::new(RoutingStatistics::default())),
            cost_model: Arc::new(DefaultCostModel),
        }
    }

    /// Set custom cost model
    pub fn with_cost_model(mut self, cost_model: Arc<dyn CostModel>) -> Self {
        self.cost_model = cost_model;
        self
    }

    /// Create query plan for pattern query
    pub async fn plan_query(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
        hints: QueryOptimizationHints,
    ) -> Result<QueryPlan> {
        let query_id = self.generate_query_id(subject, predicate, object);

        // Check cache first
        if hints.enable_cache {
            if let Some(plan) = self.check_cache(&query_id).await {
                return Ok(plan);
            }
        }

        // Determine target shards
        let shard_ids = self
            .shard_router
            .route_query_pattern(subject, predicate, object)
            .await?;

        // Update statistics
        {
            let mut stats = self.statistics.write().await;
            stats.total_queries += 1;
            if shard_ids.len() == 1 {
                stats.single_shard_queries += 1;
            } else {
                stats.multi_shard_queries += 1;
            }
            stats.avg_shards_per_query = (stats.avg_shards_per_query
                * (stats.total_queries - 1) as f64
                + shard_ids.len() as f64)
                / stats.total_queries as f64;
        }

        // Build shard targets
        let mut shard_targets = Vec::new();
        let mut total_cost = 0.0;

        for &shard_id in &shard_ids {
            if let Some(metadata) = self.shard_router.get_shard_metadata(shard_id).await {
                // Estimate selectivity based on query pattern
                let selectivity = self.estimate_selectivity(subject, predicate, object);

                let target = ShardTarget {
                    shard_id,
                    preferred_node: metadata.primary_node,
                    alternative_nodes: metadata
                        .node_ids
                        .iter()
                        .filter(|&&id| id != metadata.primary_node)
                        .copied()
                        .collect(),
                    selectivity,
                };

                // Calculate cost for this shard
                let shard_cost = self.cost_model.estimate_shard_cost(shard_id, selectivity);
                total_cost += shard_cost;

                shard_targets.push(target);
            }
        }

        // Add merge cost for multi-shard queries
        if shard_targets.len() > 1 {
            let merge_cost = self.cost_model.estimate_merge_cost(
                shard_targets.len(),
                1000, // Estimated result size
            );
            total_cost += merge_cost;
        }

        let plan = QueryPlan {
            query_id: query_id.clone(),
            shard_targets,
            optimization_hints: hints,
            estimated_cost: total_cost,
        };

        // Cache the plan
        self.cache_plan(query_id, plan.clone()).await;

        Ok(plan)
    }

    /// Plan federated query across multiple datasets
    pub async fn plan_federated_query(
        &self,
        local_pattern: (Option<&str>, Option<&str>, Option<&str>),
        remote_endpoints: Vec<String>,
        hints: QueryOptimizationHints,
    ) -> Result<FederatedQueryPlan> {
        // Create local query plan
        let local_plan = self
            .plan_query(
                local_pattern.0,
                local_pattern.1,
                local_pattern.2,
                hints.clone(),
            )
            .await?;

        // Create remote query plans
        let mut remote_plans = Vec::new();
        for endpoint in remote_endpoints {
            remote_plans.push(RemoteQueryPlan {
                endpoint: endpoint.clone(),
                estimated_latency_ms: 100.0, // Estimate based on endpoint
                estimated_result_size: 1000,
            });
        }

        Ok(FederatedQueryPlan {
            local_plan,
            remote_plans,
            merge_strategy: MergeStrategy::Union,
        })
    }

    /// Optimize query plan based on current cluster state
    pub async fn optimize_plan(&self, plan: &mut QueryPlan) -> Result<()> {
        // Check shard health and adjust targets
        for target in &mut plan.shard_targets {
            if let Some(metadata) = self.shard_router.get_shard_metadata(target.shard_id).await {
                // If primary is offline, use alternative
                if metadata.state != crate::shard::ShardState::Active {
                    if let Some(alt) = target.alternative_nodes.first() {
                        target.preferred_node = *alt;
                        warn!(
                            "Shard {} primary offline, using alternative node {}",
                            target.shard_id, alt
                        );
                    }
                }
            }
        }

        // Enable parallel execution for multi-shard queries
        if plan.shard_targets.len() > 1 && plan.optimization_hints.parallel_execution {
            debug!(
                "Enabling parallel execution for {} shards",
                plan.shard_targets.len()
            );
        }

        Ok(())
    }

    /// Execute query plan
    pub async fn execute_plan(&self, plan: QueryPlan) -> Result<Vec<Triple>> {
        let start_time = std::time::Instant::now();

        // Execute queries on target shards
        let mut all_results = Vec::new();

        if plan.optimization_hints.parallel_execution && plan.shard_targets.len() > 1 {
            // Parallel execution
            let mut handles = Vec::new();

            for _target in plan.shard_targets {
                let _shard_manager = self.shard_manager.clone();
                let handle = tokio::spawn(async move {
                    // Execute query on shard
                    // In real implementation, this would query the specific shard
                    Vec::<Triple>::new()
                });
                handles.push(handle);
            }

            for handle in handles {
                let results = handle
                    .await
                    .map_err(|e| ClusterError::Runtime(format!("Query execution failed: {e}")))?;
                all_results.extend(results);
            }
        } else {
            // Sequential execution
            for _target in plan.shard_targets {
                // Execute query on shard
                // In real implementation, this would query the specific shard
            }
        }

        // Apply post-processing
        if let Some(limit) = plan.optimization_hints.limit {
            all_results.truncate(limit);
        }

        // Update latency statistics
        let latency_ms = start_time.elapsed().as_millis() as f64;
        {
            let mut stats = self.statistics.write().await;
            stats.avg_latency_ms = (stats.avg_latency_ms * (stats.total_queries - 1) as f64
                + latency_ms)
                / stats.total_queries as f64;
        }

        Ok(all_results)
    }

    /// Generate query identifier
    fn generate_query_id(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> String {
        format!("{subject:?}:{predicate:?}:{object:?}")
    }

    /// Estimate query selectivity
    fn estimate_selectivity(
        &self,
        subject: Option<&str>,
        predicate: Option<&str>,
        object: Option<&str>,
    ) -> f64 {
        // Simple heuristic: more specific patterns have lower selectivity
        let specificity = [subject.is_some(), predicate.is_some(), object.is_some()]
            .iter()
            .filter(|&&x| x)
            .count();

        match specificity {
            3 => 0.01, // Very specific
            2 => 0.1,  // Moderately specific
            1 => 0.5,  // Less specific
            _ => 1.0,  // Scan everything
        }
    }

    /// Check query cache
    async fn check_cache(&self, query_id: &str) -> Option<QueryPlan> {
        let mut cache = self.query_cache.write().await;

        if let Some(entry) = cache.entries.get_mut(query_id) {
            entry.access_count += 1;
            let plan = entry.plan.clone();
            cache.hits += 1;

            // Update cache hit rate
            let hit_rate = cache.hits as f64 / (cache.hits + cache.misses) as f64;
            self.statistics.write().await.cache_hit_rate = hit_rate;

            Some(plan)
        } else {
            cache.misses += 1;
            None
        }
    }

    /// Cache query plan
    async fn cache_plan(&self, query_id: String, plan: QueryPlan) {
        let mut cache = self.query_cache.write().await;

        // Evict old entries if cache is full
        if cache.entries.len() >= cache.max_size {
            // Simple LRU eviction
            if let Some((evict_id, _)) = cache
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.access_count)
            {
                let evict_id = evict_id.clone();
                cache.entries.remove(&evict_id);
            }
        }

        cache.entries.insert(
            query_id,
            CacheEntry {
                plan,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("SystemTime should be after UNIX_EPOCH")
                    .as_secs(),
                access_count: 1,
            },
        );
    }

    /// Get routing statistics
    pub async fn get_statistics(&self) -> RoutingStatistics {
        self.statistics.read().await.clone()
    }
}

/// Federated query plan
#[derive(Debug, Clone)]
pub struct FederatedQueryPlan {
    /// Local query plan
    pub local_plan: QueryPlan,
    /// Remote query plans
    pub remote_plans: Vec<RemoteQueryPlan>,
    /// Result merge strategy
    pub merge_strategy: MergeStrategy,
}

/// Remote query plan
#[derive(Debug, Clone)]
pub struct RemoteQueryPlan {
    /// SPARQL endpoint
    pub endpoint: String,
    /// Estimated latency in milliseconds
    pub estimated_latency_ms: f64,
    /// Estimated result size
    pub estimated_result_size: usize,
}

/// Result merge strategy
#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Union all results
    Union,
    /// Intersection of results
    Intersection,
    /// Join on specific variables
    Join(Vec<String>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::{NetworkConfig, NetworkService};
    use crate::shard::ShardingStrategy;
    use crate::shard_manager::ShardManagerConfig;
    use crate::storage::mock::MockStorageBackend;

    #[tokio::test]
    async fn test_query_planning() {
        let strategy = ShardingStrategy::Hash { num_shards: 4 };
        let shard_router = Arc::new(ShardRouter::new(strategy));
        shard_router.init_shards(4, 3).await.unwrap();

        let storage = Arc::new(MockStorageBackend::new());
        let network = Arc::new(NetworkService::new(1, NetworkConfig::default()));
        let shard_manager = Arc::new(ShardManager::new(
            1,
            shard_router.clone(),
            ShardManagerConfig::default(),
            storage,
            network,
        ));

        let query_router = QueryRouter::new(shard_router, shard_manager);

        let hints = QueryOptimizationHints {
            parallel_execution: true,
            enable_cache: true,
            ..Default::default()
        };

        let plan = query_router
            .plan_query(Some("http://example.org/subject"), None, None, hints)
            .await
            .unwrap();

        assert_eq!(plan.shard_targets.len(), 1);
        assert!(plan.estimated_cost > 0.0);
    }

    #[test]
    fn test_selectivity_estimation() {
        let query_router = QueryRouter::new(
            Arc::new(ShardRouter::new(ShardingStrategy::Hash { num_shards: 1 })),
            Arc::new(ShardManager::new(
                1,
                Arc::new(ShardRouter::new(ShardingStrategy::Hash { num_shards: 1 })),
                ShardManagerConfig::default(),
                Arc::new(MockStorageBackend::new()),
                Arc::new(NetworkService::new(1, NetworkConfig::default())),
            )),
        );

        assert_eq!(
            query_router.estimate_selectivity(Some("s"), Some("p"), Some("o")),
            0.01
        );
        assert_eq!(
            query_router.estimate_selectivity(Some("s"), Some("p"), None),
            0.1
        );
        assert_eq!(
            query_router.estimate_selectivity(Some("s"), None, None),
            0.5
        );
        assert_eq!(query_router.estimate_selectivity(None, None, None), 1.0);
    }

    #[test]
    fn test_cost_model() {
        let cost_model = DefaultCostModel;

        let shard_cost = cost_model.estimate_shard_cost(0, 0.5);
        assert!(shard_cost > 0.0);

        let network_cost = cost_model.estimate_network_cost(1_000_000, 2);
        assert!(network_cost > 0.0);

        let merge_cost = cost_model.estimate_merge_cost(4, 1000);
        assert!(merge_cost > 0.0);
    }
}
