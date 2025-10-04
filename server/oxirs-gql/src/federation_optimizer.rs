//! Federation-Aware Query Optimization
//!
//! Optimizes federated GraphQL queries with intelligent caching, batching,
//! and cost-based query planning for distributed subgraph architectures.
//!
//! ## Features
//!
//! - **Query Result Caching**: Cache federated query results with TTL
//! - **Request Batching**: Batch multiple subgraph requests into single calls
//! - **Cost-Based Optimization**: Choose optimal execution strategies
//! - **Query Deduplication**: Eliminate redundant subgraph queries
//! - **Parallel Execution**: Maximize parallelism across subgraphs

use crate::federation_composer::{QueryPlan, QueryPlanNode, Supergraph};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Cached query result with metadata
#[derive(Debug, Clone)]
struct CachedResult {
    /// The cached data
    data: serde_json::Value,
    /// When this entry was created
    created_at: Instant,
    /// Time-to-live for this entry
    ttl: Duration,
    /// Number of cache hits
    hit_count: u64,
    /// Size in bytes (approximate)
    size_bytes: usize,
}

impl CachedResult {
    fn new(data: serde_json::Value, ttl: Duration) -> Self {
        let size_bytes = serde_json::to_string(&data).unwrap_or_default().len();
        Self {
            data,
            created_at: Instant::now(),
            ttl,
            hit_count: 0,
            size_bytes,
        }
    }

    fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }

    fn record_hit(&mut self) {
        self.hit_count += 1;
    }
}

/// Query result cache for federated queries
#[derive(Debug)]
pub struct FederationCache {
    /// Cache storage
    cache: Arc<RwLock<HashMap<String, CachedResult>>>,
    /// Default TTL for cache entries
    default_ttl: Duration,
    /// Maximum cache size in bytes
    max_size_bytes: usize,
    /// Current cache size in bytes
    current_size_bytes: Arc<RwLock<usize>>,
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
}

#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub total_requests: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.total_requests == 0 {
            0.0
        } else {
            self.hits as f64 / self.total_requests as f64
        }
    }
}

impl FederationCache {
    pub fn new(default_ttl: Duration, max_size_mb: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            default_ttl,
            max_size_bytes: max_size_mb * 1024 * 1024,
            current_size_bytes: Arc::new(RwLock::new(0)),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Get cached result for a query
    pub fn get(&self, query_key: &str) -> Option<serde_json::Value> {
        let mut cache = self.cache.write().unwrap();
        let mut stats = self.stats.write().unwrap();
        stats.total_requests += 1;

        if let Some(entry) = cache.get_mut(query_key) {
            if entry.is_expired() {
                // Remove expired entry
                let size = entry.size_bytes;
                cache.remove(query_key);
                let mut current_size = self.current_size_bytes.write().unwrap();
                *current_size = current_size.saturating_sub(size);
                stats.misses += 1;
                None
            } else {
                entry.record_hit();
                stats.hits += 1;
                Some(entry.data.clone())
            }
        } else {
            stats.misses += 1;
            None
        }
    }

    /// Store query result in cache
    pub fn set(&self, query_key: String, data: serde_json::Value) {
        self.set_with_ttl(query_key, data, self.default_ttl);
    }

    /// Store query result with custom TTL
    pub fn set_with_ttl(&self, query_key: String, data: serde_json::Value, ttl: Duration) {
        let entry = CachedResult::new(data, ttl);
        let entry_size = entry.size_bytes;

        let mut cache = self.cache.write().unwrap();
        let mut current_size = self.current_size_bytes.write().unwrap();

        // Evict entries if necessary
        while *current_size + entry_size > self.max_size_bytes && !cache.is_empty() {
            self.evict_lru(&mut cache, &mut current_size);
        }

        // Add new entry
        if let Some(old_entry) = cache.insert(query_key, entry) {
            *current_size = current_size.saturating_sub(old_entry.size_bytes);
        }
        *current_size += entry_size;
    }

    fn evict_lru(
        &self,
        cache: &mut HashMap<String, CachedResult>,
        current_size: &mut usize,
    ) {
        // Find LRU entry (oldest with lowest hit count)
        let lru_key = cache
            .iter()
            .min_by_key(|(_, entry)| (entry.hit_count, entry.created_at))
            .map(|(k, _)| k.clone());

        if let Some(key) = lru_key {
            if let Some(removed) = cache.remove(&key) {
                *current_size = current_size.saturating_sub(removed.size_bytes);
                let mut stats = self.stats.write().unwrap();
                stats.evictions += 1;
            }
        }
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        let mut cache = self.cache.write().unwrap();
        cache.clear();
        let mut current_size = self.current_size_bytes.write().unwrap();
        *current_size = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    /// Get current cache size in bytes
    pub fn size_bytes(&self) -> usize {
        *self.current_size_bytes.read().unwrap()
    }
}

impl Default for FederationCache {
    fn default() -> Self {
        Self::new(Duration::from_secs(300), 100) // 5 min TTL, 100MB max
    }
}

/// Request batching coordinator for subgraph queries
#[derive(Debug)]
pub struct RequestBatcher {
    /// Batch window duration
    _batch_window: Duration,
    /// Maximum batch size
    max_batch_size: usize,
}

impl RequestBatcher {
    pub fn new(batch_window: Duration, max_batch_size: usize) -> Self {
        Self {
            _batch_window: batch_window,
            max_batch_size,
        }
    }

    /// Check if queries can be batched together
    pub fn can_batch(&self, queries: &[String]) -> bool {
        queries.len() <= self.max_batch_size
    }

    /// Combine multiple queries into a single batched query
    pub fn batch_queries(&self, queries: Vec<String>) -> Result<String> {
        if queries.is_empty() {
            return Ok(String::new());
        }

        if queries.len() == 1 {
            return Ok(queries[0].clone());
        }

        // Create a batched query with multiple operations
        let mut batched = String::from("{\n");
        for (i, query) in queries.iter().enumerate() {
            // Extract operation from each query and alias it
            let aliased = format!("  query{}: {}\n", i, self.extract_fields(query));
            batched.push_str(&aliased);
        }
        batched.push('}');

        Ok(batched)
    }

    fn extract_fields<'a>(&self, query: &'a str) -> &'a str {
        // Simplified extraction - production would parse AST
        query.trim()
    }
}

impl Default for RequestBatcher {
    fn default() -> Self {
        Self::new(Duration::from_millis(10), 100)
    }
}

/// Optimization strategy for query execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    /// Maximize parallelism, minimize latency
    MinLatency,
    /// Minimize subgraph requests, maximize caching
    MinRequests,
    /// Balance between latency and requests
    Balanced,
    /// Optimize for cost (minimize expensive operations)
    MinCost,
}

/// Federation-aware query optimizer
#[derive(Debug)]
pub struct FederationOptimizer {
    /// Result cache
    cache: Arc<FederationCache>,
    /// Request batcher
    batcher: RequestBatcher,
    /// Optimization strategy
    strategy: OptimizationStrategy,
    /// Supergraph schema
    _supergraph: Supergraph,
}

impl FederationOptimizer {
    pub fn new(
        supergraph: Supergraph,
        cache: Arc<FederationCache>,
        strategy: OptimizationStrategy,
    ) -> Self {
        Self {
            cache,
            batcher: RequestBatcher::default(),
            strategy,
            _supergraph: supergraph,
        }
    }

    /// Optimize a query plan based on the configured strategy
    pub fn optimize(&self, plan: QueryPlan) -> Result<OptimizedQueryPlan> {
        match self.strategy {
            OptimizationStrategy::MinLatency => self.optimize_for_latency(plan),
            OptimizationStrategy::MinRequests => self.optimize_for_requests(plan),
            OptimizationStrategy::Balanced => self.optimize_balanced(plan),
            OptimizationStrategy::MinCost => self.optimize_for_cost(plan),
        }
    }

    fn optimize_for_latency(&self, plan: QueryPlan) -> Result<OptimizedQueryPlan> {
        // Maximize parallelism
        let estimated_latency = self.estimate_latency(&plan.root);
        let optimized_root = self.parallelize_node(plan.root);

        Ok(OptimizedQueryPlan {
            root: optimized_root,
            estimated_cost: plan.estimated_cost,
            estimated_latency,
            cache_hits_expected: 0,
            subgraphs: plan.subgraphs,
            optimizations_applied: vec!["parallelization".to_string()],
        })
    }

    fn optimize_for_requests(&self, plan: QueryPlan) -> Result<OptimizedQueryPlan> {
        // Deduplicate and batch requests
        let estimated_latency = self.estimate_latency(&plan.root);
        let deduplicated_root = self.deduplicate_node(plan.root);
        let batched_root = self.batch_node(deduplicated_root);

        Ok(OptimizedQueryPlan {
            root: batched_root,
            estimated_cost: plan.estimated_cost * 0.7, // Batching reduces cost
            estimated_latency,
            cache_hits_expected: 0,
            subgraphs: plan.subgraphs,
            optimizations_applied: vec!["deduplication".to_string(), "batching".to_string()],
        })
    }

    fn optimize_balanced(&self, plan: QueryPlan) -> Result<OptimizedQueryPlan> {
        // Balance parallelism and request reduction
        let estimated_latency = self.estimate_latency(&plan.root);
        let deduplicated_root = self.deduplicate_node(plan.root);
        let optimized_root = self.parallelize_node(deduplicated_root);

        Ok(OptimizedQueryPlan {
            root: optimized_root,
            estimated_cost: plan.estimated_cost * 0.85,
            estimated_latency,
            cache_hits_expected: 0,
            subgraphs: plan.subgraphs,
            optimizations_applied: vec!["deduplication".to_string(), "parallelization".to_string()],
        })
    }

    fn optimize_for_cost(&self, plan: QueryPlan) -> Result<OptimizedQueryPlan> {
        // Minimize expensive operations, maximize caching
        let estimated_latency = self.estimate_latency(&plan.root) * 1.2; // Slight latency increase
        let cached_root = self.apply_caching(plan.root);
        let deduplicated_root = self.deduplicate_node(cached_root);

        Ok(OptimizedQueryPlan {
            root: deduplicated_root,
            estimated_cost: plan.estimated_cost * 0.6, // Caching significantly reduces cost
            estimated_latency,
            cache_hits_expected: 3,
            subgraphs: plan.subgraphs,
            optimizations_applied: vec!["caching".to_string(), "deduplication".to_string()],
        })
    }

    fn parallelize_node(&self, node: QueryPlanNode) -> QueryPlanNode {
        match node {
            QueryPlanNode::Sequence { nodes } => {
                // Convert sequence to parallel where safe
                if self.can_parallelize(&nodes) {
                    QueryPlanNode::Parallel {
                        nodes: nodes.into_iter().map(|n| self.parallelize_node(n)).collect(),
                    }
                } else {
                    QueryPlanNode::Sequence {
                        nodes: nodes.into_iter().map(|n| self.parallelize_node(n)).collect(),
                    }
                }
            }
            QueryPlanNode::Parallel { nodes } => QueryPlanNode::Parallel {
                nodes: nodes.into_iter().map(|n| self.parallelize_node(n)).collect(),
            },
            QueryPlanNode::Flatten { path, node } => QueryPlanNode::Flatten {
                path,
                node: Box::new(self.parallelize_node(*node)),
            },
            other => other,
        }
    }

    fn can_parallelize(&self, nodes: &[QueryPlanNode]) -> bool {
        // Check if nodes have dependencies - simplified check
        let subgraphs: HashSet<_> = nodes
            .iter()
            .filter_map(|n| match n {
                QueryPlanNode::Fetch { subgraph, .. } => Some(subgraph.clone()),
                _ => None,
            })
            .collect();

        // If all nodes target different subgraphs, they can be parallelized
        subgraphs.len() == nodes.len()
    }

    fn deduplicate_node(&self, node: QueryPlanNode) -> QueryPlanNode {
        match node {
            QueryPlanNode::Sequence { nodes } => {
                let deduplicated = self.deduplicate_fetches(nodes);
                QueryPlanNode::Sequence {
                    nodes: deduplicated.into_iter().map(|n| self.deduplicate_node(n)).collect(),
                }
            }
            QueryPlanNode::Parallel { nodes } => {
                let deduplicated = self.deduplicate_fetches(nodes);
                QueryPlanNode::Parallel {
                    nodes: deduplicated.into_iter().map(|n| self.deduplicate_node(n)).collect(),
                }
            }
            other => other,
        }
    }

    fn deduplicate_fetches(&self, nodes: Vec<QueryPlanNode>) -> Vec<QueryPlanNode> {
        let mut seen = HashMap::new();
        let mut deduplicated = Vec::new();

        for node in nodes {
            if let QueryPlanNode::Fetch { ref subgraph, ref query, .. } = node {
                let key = format!("{}:{}", subgraph, query);
                if !seen.contains_key(&key) {
                    seen.insert(key, true);
                    deduplicated.push(node);
                }
            } else {
                deduplicated.push(node);
            }
        }

        deduplicated
    }

    fn batch_node(&self, node: QueryPlanNode) -> QueryPlanNode {
        // Simplified batching - combine fetches to same subgraph
        match node {
            QueryPlanNode::Parallel { nodes } => {
                let batched = self.batch_parallel_fetches(nodes);
                QueryPlanNode::Parallel { nodes: batched }
            }
            other => other,
        }
    }

    fn batch_parallel_fetches(&self, nodes: Vec<QueryPlanNode>) -> Vec<QueryPlanNode> {
        let mut by_subgraph: HashMap<String, Vec<String>> = HashMap::new();
        let mut other_nodes = Vec::new();

        for node in nodes {
            match node {
                QueryPlanNode::Fetch { subgraph, query, requires } if requires.is_empty() => {
                    by_subgraph.entry(subgraph).or_insert_with(Vec::new).push(query);
                }
                other => other_nodes.push(other),
            }
        }

        let mut result = other_nodes;
        for (subgraph, queries) in by_subgraph {
            if queries.len() > 1 && self.batcher.can_batch(&queries) {
                if let Ok(batched_query) = self.batcher.batch_queries(queries) {
                    result.push(QueryPlanNode::Fetch {
                        subgraph,
                        query: batched_query,
                        requires: Vec::new(),
                    });
                }
            } else {
                for query in queries {
                    result.push(QueryPlanNode::Fetch {
                        subgraph: subgraph.clone(),
                        query,
                        requires: Vec::new(),
                    });
                }
            }
        }

        result
    }

    fn apply_caching(&self, node: QueryPlanNode) -> QueryPlanNode {
        // Mark cacheable operations - simplified
        node
    }

    fn estimate_latency(&self, _node: &QueryPlanNode) -> f64 {
        // Simplified latency estimation
        1.0
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }
}

/// Optimized query plan with metadata
#[derive(Debug, Clone)]
pub struct OptimizedQueryPlan {
    /// Optimized execution root
    pub root: QueryPlanNode,
    /// Estimated execution cost
    pub estimated_cost: f64,
    /// Estimated latency in milliseconds
    pub estimated_latency: f64,
    /// Expected cache hits
    pub cache_hits_expected: usize,
    /// Subgraphs involved
    pub subgraphs: HashSet<String>,
    /// Optimizations applied
    pub optimizations_applied: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federation_cache_basic() {
        let cache = FederationCache::new(Duration::from_secs(10), 10);
        let query_key = "{ user { id } }";
        let data = serde_json::json!({"user": {"id": "123"}});

        assert!(cache.get(query_key).is_none());

        cache.set(query_key.to_string(), data.clone());
        assert_eq!(cache.get(query_key), Some(data));
    }

    #[test]
    fn test_cache_expiration() {
        let cache = FederationCache::new(Duration::from_millis(50), 10);
        let query_key = "{ user { id } }";
        let data = serde_json::json!({"user": {"id": "123"}});

        cache.set(query_key.to_string(), data.clone());
        assert_eq!(cache.get(query_key), Some(data));

        std::thread::sleep(Duration::from_millis(100));
        assert!(cache.get(query_key).is_none());
    }

    #[test]
    fn test_cache_stats() {
        let cache = FederationCache::default();
        let data = serde_json::json!({"test": "data"});

        cache.set("key1".to_string(), data.clone());
        cache.get("key1");
        cache.get("key2");

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.total_requests, 2);
        assert_eq!(stats.hit_rate(), 0.5);
    }

    #[test]
    fn test_request_batcher() {
        let batcher = RequestBatcher::default();

        let queries = vec![
            "{ user { id } }".to_string(),
            "{ product { sku } }".to_string(),
        ];

        assert!(batcher.can_batch(&queries));
        let batched = batcher.batch_queries(queries).unwrap();
        assert!(batched.contains("query0"));
        assert!(batched.contains("query1"));
    }

    #[test]
    fn test_optimization_strategies() {
        let supergraph = Supergraph {
            sdl: String::new(),
            entities: HashMap::new(),
            field_ownership: HashMap::new(),
            subgraphs: vec!["users".to_string()],
        };

        let cache = Arc::new(FederationCache::default());

        let optimizer = FederationOptimizer::new(
            supergraph,
            cache,
            OptimizationStrategy::MinLatency,
        );

        let plan = QueryPlan {
            root: QueryPlanNode::Fetch {
                subgraph: "users".to_string(),
                query: "{ user { id } }".to_string(),
                requires: Vec::new(),
            },
            estimated_cost: 1.0,
            subgraphs: HashSet::new(),
        };

        let optimized = optimizer.optimize(plan).unwrap();
        assert!(!optimized.optimizations_applied.is_empty());
    }

    #[test]
    fn test_cache_eviction() {
        let cache = FederationCache::new(Duration::from_secs(10), 1); // 1MB max
        let large_data = serde_json::json!({"data": "x".repeat(700_000)});

        cache.set("key1".to_string(), large_data.clone());
        assert!(cache.get("key1").is_some());

        cache.set("key2".to_string(), large_data.clone());

        // Should evict key1 due to size limit
        let stats = cache.stats();
        assert!(stats.evictions > 0);
    }
}
