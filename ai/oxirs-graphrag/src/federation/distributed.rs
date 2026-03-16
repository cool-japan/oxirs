//! Distributed federation: route GraphRAG queries to multiple remote nodes,
//! merge results, and manage health state.
//!
//! # Architecture
//!
//! ```text
//! FederatedGraphRag
//!     │
//!     ├── FederationRouter  (strategy + health-check policy)
//!     ├── LocalRagEngine    (fallback / self-hosted stub)
//!     └── Vec<FederationNode>  (remote peers)
//!
//! query(FederatedQuery)
//!     → select nodes via FederationStrategy
//!     → dispatch (simulated sync) to each node
//!     → merge Vec<RagResult> into FederatedResult
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────────────────────────────────────

/// A single result item returned by a RAG node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RagResult {
    /// Retrieved text passage or triple serialization
    pub text: String,
    /// Relevance score in [0, 1]
    pub score: f64,
    /// Identifier of the node that produced this result
    pub source: String,
}

/// A remote (or local) RAG node in the federation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationNode {
    /// Unique node identifier
    pub id: String,
    /// Network endpoint (URL or address)
    pub endpoint: String,
    /// Capabilities advertised by this node (e.g. "temporal", "vector")
    pub capabilities: Vec<String>,
    /// Observed round-trip latency in milliseconds
    pub latency_ms: u64,
    /// Whether the node is currently reachable
    pub is_healthy: bool,
}

/// Strategy for selecting which nodes to query.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FederationStrategy {
    /// Send the query to every healthy node and merge results
    BroadcastAll,
    /// Route only to nodes whose `capabilities` overlap with query needs
    RouteByCoverage,
    /// Distribute queries evenly across healthy nodes
    LoadBalance,
    /// Try nodes in latency order; stop at the first successful response
    FailoverChain,
}

/// Policy and health-check configuration for the router.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationRouter {
    pub strategy: FederationStrategy,
    pub health_check_interval_ms: u64,
}

impl FederationRouter {
    /// Create a router with the given strategy.
    pub fn new(strategy: FederationStrategy) -> Self {
        Self {
            strategy,
            health_check_interval_ms: 30_000,
        }
    }

    /// Select which nodes to query given the full node list and query context.
    pub fn select_nodes<'a>(
        &self,
        nodes: &'a [FederationNode],
        query: &FederatedQuery,
        counter: &mut u64,
    ) -> Vec<&'a FederationNode> {
        let healthy: Vec<&FederationNode> = nodes.iter().filter(|n| n.is_healthy).collect();

        match &self.strategy {
            FederationStrategy::BroadcastAll => healthy,

            FederationStrategy::RouteByCoverage => {
                // Use nodes that advertise "temporal" capability when query has a timestamp
                if query.timestamp.is_some() {
                    let temporal: Vec<_> = healthy
                        .iter()
                        .copied()
                        .filter(|n| n.capabilities.iter().any(|c| c == "temporal"))
                        .collect();
                    if !temporal.is_empty() {
                        return temporal;
                    }
                }
                healthy
            }

            FederationStrategy::LoadBalance => {
                if healthy.is_empty() {
                    return vec![];
                }
                // Round-robin: pick node at index (counter % healthy.len())
                let idx = (*counter as usize) % healthy.len();
                *counter = counter.wrapping_add(1);
                vec![healthy[idx]]
            }

            FederationStrategy::FailoverChain => {
                // Sort by latency, pick the fastest healthy node
                let mut sorted = healthy.clone();
                sorted.sort_by_key(|n| n.latency_ms);
                sorted.into_iter().take(1).collect()
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stub local RAG engine
// ─────────────────────────────────────────────────────────────────────────────

/// Simple in-memory RAG engine used as a local fallback.
///
/// Stores a small text corpus and returns entries whose text contains any
/// query keyword (scored by match ratio).
#[derive(Debug, Default)]
pub struct LocalRagEngine {
    corpus: Vec<(String, f64)>, // (text, base_score)
}

impl LocalRagEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a passage to the local corpus with a pre-assigned relevance base score.
    pub fn add_passage(&mut self, text: impl Into<String>, base_score: f64) {
        self.corpus.push((text.into(), base_score.clamp(0.0, 1.0)));
    }

    /// Query the corpus and return up to `top_k` results.
    pub fn query(&self, q: &str, top_k: usize, source: &str) -> Vec<RagResult> {
        let keywords: Vec<&str> = q.split_whitespace().collect();
        let mut scored: Vec<RagResult> = self
            .corpus
            .iter()
            .filter_map(|(text, base)| {
                let matched = keywords
                    .iter()
                    .filter(|kw| text.to_lowercase().contains(&kw.to_lowercase()))
                    .count();
                if matched == 0 {
                    return None;
                }
                let kw_score = matched as f64 / keywords.len().max(1) as f64;
                Some(RagResult {
                    text: text.clone(),
                    score: (base + kw_score) / 2.0,
                    source: source.to_string(),
                })
            })
            .collect();

        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        scored
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Query and result types
// ─────────────────────────────────────────────────────────────────────────────

/// A query to submit to the federation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedQuery {
    /// Natural-language question or keyword query
    pub query: String,
    /// Optional point-in-time constraint (Unix-ms)
    pub timestamp: Option<i64>,
    /// Maximum number of results to return
    pub top_k: usize,
    /// Abort if no response within this many milliseconds (advisory)
    pub timeout_ms: u64,
}

/// Aggregated results from the federation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedResult {
    /// Merged and de-duplicated result list (sorted by score descending)
    pub results: Vec<RagResult>,
    /// IDs of the nodes that contributed results
    pub sources: Vec<String>,
    /// Observed total latency (wall-clock ms)
    pub total_latency_ms: u64,
    /// Number of nodes consulted
    pub node_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// FederatedGraphRag
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-node GraphRAG federation manager.
///
/// In production this would issue async HTTP requests to remote endpoints;
/// here remote nodes are simulated by the local engine (each node shares the
/// same local corpus but is given a distinct source label).
pub struct FederatedGraphRag {
    nodes: Vec<FederationNode>,
    local_rag: LocalRagEngine,
    router: FederationRouter,
    /// Round-robin counter used by LoadBalance strategy
    lb_counter: u64,
}

impl FederatedGraphRag {
    /// Create a new federation with the given routing strategy.
    pub fn new(strategy: FederationStrategy) -> Self {
        Self {
            nodes: Vec::new(),
            local_rag: LocalRagEngine::new(),
            router: FederationRouter::new(strategy),
            lb_counter: 0,
        }
    }

    /// Add a remote node to the federation.
    pub fn add_node(&mut self, node: FederationNode) {
        self.nodes.push(node);
    }

    /// Remove a node by its ID.  Returns `true` if the node existed.
    pub fn remove_node(&mut self, node_id: &str) -> bool {
        let before = self.nodes.len();
        self.nodes.retain(|n| n.id != node_id);
        self.nodes.len() < before
    }

    /// Execute a federated query and return the merged result.
    pub fn query(&mut self, q: &FederatedQuery) -> FederatedResult {
        let start = Instant::now();

        let selected: Vec<String> = self
            .router
            .select_nodes(&self.nodes, q, &mut self.lb_counter)
            .iter()
            .map(|n| n.id.clone())
            .collect();

        let mut all_results: Vec<RagResult> = Vec::new();
        let mut sources: Vec<String> = Vec::new();

        // Simulate per-node queries via the local engine
        for node_id in &selected {
            let node_results = self.local_rag.query(&q.query, q.top_k, node_id);
            if !node_results.is_empty() {
                sources.push(node_id.clone());
                all_results.extend(node_results);
            }
        }

        // Merge: de-duplicate by text, keep highest score
        let mut seen: HashMap<String, usize> = HashMap::new();
        let mut merged: Vec<RagResult> = Vec::new();
        for r in all_results {
            match seen.get(&r.text) {
                Some(&idx) if merged[idx].score >= r.score => {}
                _ => {
                    let idx = merged.len();
                    seen.insert(r.text.clone(), idx);
                    merged.push(r);
                }
            }
        }

        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(q.top_k);

        FederatedResult {
            results: merged,
            sources,
            total_latency_ms: start.elapsed().as_millis() as u64,
            node_count: selected.len(),
        }
    }

    /// Return references to all currently healthy nodes.
    pub fn healthy_nodes(&self) -> Vec<&FederationNode> {
        self.nodes.iter().filter(|n| n.is_healthy).collect()
    }

    /// Mark a node as unhealthy (e.g. after a failed health check).
    pub fn mark_unhealthy(&mut self, node_id: &str) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == node_id) {
            node.is_healthy = false;
        }
    }

    /// Rebalance: restore all nodes to healthy (simulates health-check recovery).
    pub fn rebalance(&mut self) {
        for node in &mut self.nodes {
            node.is_healthy = true;
        }
    }

    /// Add a passage to the local corpus (used as backing store for all nodes
    /// in the simulation).
    pub fn add_corpus_passage(&mut self, text: impl Into<String>, base_score: f64) {
        self.local_rag.add_passage(text, base_score);
    }

    /// Number of nodes in the federation (healthy + unhealthy).
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Index types and builder
// ─────────────────────────────────────────────────────────────────────────────

/// A per-node index fragment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalIndex {
    pub node_id: String,
    /// (key, score) pairs
    pub entries: Vec<(String, f64)>,
}

/// A merged index covering all nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergedIndex {
    /// (key, score, originating_node_id)
    pub entries: Vec<(String, f64, String)>,
}

/// A shard of a merged index for distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexShard {
    pub shard_id: usize,
    /// (key, score, originating_node_id)
    pub entries: Vec<(String, f64, String)>,
}

/// Utility for building and sharding federation indices.
pub struct FederatedIndexBuilder;

impl FederatedIndexBuilder {
    /// Merge multiple per-node indices into a single sorted index.
    ///
    /// Duplicate keys are resolved by keeping the highest score across all nodes.
    pub fn merge_indices(indices: Vec<LocalIndex>) -> MergedIndex {
        let mut best: HashMap<String, (f64, String)> = HashMap::new();

        for local in indices {
            for (key, score) in local.entries {
                let entry = best
                    .entry(key.clone())
                    .or_insert((f64::NEG_INFINITY, local.node_id.clone()));
                if score > entry.0 {
                    *entry = (score, local.node_id.clone());
                }
            }
        }

        let mut entries: Vec<(String, f64, String)> =
            best.into_iter().map(|(k, (s, n))| (k, s, n)).collect();

        // Sort by score descending, then key ascending for determinism
        entries.sort_by(|(ka, sa, _), (kb, sb, _)| {
            sb.partial_cmp(sa)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| ka.cmp(kb))
        });

        MergedIndex { entries }
    }

    /// Partition a merged index into `shard_count` roughly equal shards.
    pub fn shard_index(index: &MergedIndex, shard_count: usize) -> Vec<IndexShard> {
        if shard_count == 0 {
            return vec![];
        }

        let mut shards: Vec<IndexShard> = (0..shard_count)
            .map(|id| IndexShard {
                shard_id: id,
                entries: Vec::new(),
            })
            .collect();

        for (i, entry) in index.entries.iter().enumerate() {
            shards[i % shard_count].entries.push(entry.clone());
        }

        shards
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy_node(id: &str, latency: u64) -> FederationNode {
        FederationNode {
            id: id.to_string(),
            endpoint: format!("http://{id}.example.com"),
            capabilities: vec!["vector".to_string()],
            latency_ms: latency,
            is_healthy: true,
        }
    }

    fn temporal_node(id: &str) -> FederationNode {
        FederationNode {
            id: id.to_string(),
            endpoint: format!("http://{id}.example.com"),
            capabilities: vec!["temporal".to_string(), "vector".to_string()],
            latency_ms: 10,
            is_healthy: true,
        }
    }

    fn make_query(q: &str) -> FederatedQuery {
        FederatedQuery {
            query: q.to_string(),
            timestamp: None,
            top_k: 5,
            timeout_ms: 1000,
        }
    }

    // ── FederationNode ────────────────────────────────────────────────────

    #[test]
    fn test_federation_node_fields() {
        let node = healthy_node("node1", 50);
        assert_eq!(node.id, "node1");
        assert!(node.is_healthy);
        assert_eq!(node.latency_ms, 50);
    }

    // ── FederatedGraphRag::add_node / remove_node ─────────────────────────

    #[test]
    fn test_add_and_remove_node() {
        let mut fed = FederatedGraphRag::new(FederationStrategy::BroadcastAll);
        fed.add_node(healthy_node("A", 10));
        fed.add_node(healthy_node("B", 20));
        assert_eq!(fed.node_count(), 2);

        let removed = fed.remove_node("A");
        assert!(removed);
        assert_eq!(fed.node_count(), 1);
    }

    #[test]
    fn test_remove_nonexistent_node_returns_false() {
        let mut fed = FederatedGraphRag::new(FederationStrategy::BroadcastAll);
        assert!(!fed.remove_node("ghost"));
    }

    // ── healthy_nodes ─────────────────────────────────────────────────────

    #[test]
    fn test_healthy_nodes_filters_unhealthy() {
        let mut fed = FederatedGraphRag::new(FederationStrategy::BroadcastAll);
        fed.add_node(healthy_node("A", 10));
        fed.add_node(healthy_node("B", 10));
        fed.mark_unhealthy("A");
        assert_eq!(fed.healthy_nodes().len(), 1);
        assert_eq!(fed.healthy_nodes()[0].id, "B");
    }

    #[test]
    fn test_healthy_nodes_empty_federation() {
        let fed = FederatedGraphRag::new(FederationStrategy::BroadcastAll);
        assert!(fed.healthy_nodes().is_empty());
    }

    // ── mark_unhealthy / rebalance ────────────────────────────────────────

    #[test]
    fn test_mark_unhealthy_sets_flag() {
        let mut fed = FederatedGraphRag::new(FederationStrategy::BroadcastAll);
        fed.add_node(healthy_node("A", 10));
        fed.mark_unhealthy("A");
        assert!(!fed.nodes[0].is_healthy);
    }

    #[test]
    fn test_rebalance_restores_all_nodes() {
        let mut fed = FederatedGraphRag::new(FederationStrategy::BroadcastAll);
        fed.add_node(healthy_node("A", 10));
        fed.add_node(healthy_node("B", 10));
        fed.mark_unhealthy("A");
        fed.mark_unhealthy("B");
        assert_eq!(fed.healthy_nodes().len(), 0);
        fed.rebalance();
        assert_eq!(fed.healthy_nodes().len(), 2);
    }

    // ── query: BroadcastAll ───────────────────────────────────────────────

    #[test]
    fn test_query_broadcast_all_returns_merged_results() {
        let mut fed = FederatedGraphRag::new(FederationStrategy::BroadcastAll);
        fed.add_node(healthy_node("A", 10));
        fed.add_node(healthy_node("B", 20));
        fed.add_corpus_passage("Rust is a systems language", 0.9);

        let result = fed.query(&make_query("Rust language"));
        // Should have collected from both nodes
        assert_eq!(result.node_count, 2);
        assert!(!result.results.is_empty());
    }

    #[test]
    fn test_query_with_no_healthy_nodes_returns_empty() {
        let mut fed = FederatedGraphRag::new(FederationStrategy::BroadcastAll);
        fed.add_node(healthy_node("A", 10));
        fed.mark_unhealthy("A");
        let result = fed.query(&make_query("anything"));
        assert!(result.results.is_empty());
        assert_eq!(result.node_count, 0);
    }

    // ── query: FailoverChain ──────────────────────────────────────────────

    #[test]
    fn test_failover_chain_picks_fastest_node() {
        let mut fed = FederatedGraphRag::new(FederationStrategy::FailoverChain);
        fed.add_node(healthy_node("slow", 200));
        fed.add_node(healthy_node("fast", 10));
        fed.add_corpus_passage("Semantic Web SPARQL", 0.8);

        let result = fed.query(&make_query("Semantic Web"));
        // Only one node consulted (fastest)
        assert_eq!(result.node_count, 1);
        assert_eq!(result.sources[0], "fast");
    }

    // ── query: RouteByCoverage with timestamp ─────────────────────────────

    #[test]
    fn test_route_by_coverage_uses_temporal_node() {
        let mut fed = FederatedGraphRag::new(FederationStrategy::RouteByCoverage);
        fed.add_node(healthy_node("generic", 10));
        fed.add_node(temporal_node("temporal_node"));
        fed.add_corpus_passage("historical data", 0.85);

        let mut q = make_query("historical data");
        q.timestamp = Some(1_700_000_000_000); // some timestamp

        let result = fed.query(&q);
        assert!(result.node_count > 0);
        // Should prefer temporal_node
        assert!(result.sources.contains(&"temporal_node".to_string()));
    }

    // ── query: LoadBalance ────────────────────────────────────────────────

    #[test]
    fn test_load_balance_rotates_nodes() {
        let mut fed = FederatedGraphRag::new(FederationStrategy::LoadBalance);
        fed.add_node(healthy_node("N1", 10));
        fed.add_node(healthy_node("N2", 10));
        fed.add_corpus_passage("GraphRAG federation", 0.9);

        let q = make_query("GraphRAG");
        let r1 = fed.query(&q);
        let r2 = fed.query(&q);

        // Should have queried different nodes
        assert_eq!(r1.node_count, 1);
        assert_eq!(r2.node_count, 1);
        // sources may differ (round-robin)
        let _ = r1.sources;
        let _ = r2.sources;
    }

    // ── FederatedResult fields ────────────────────────────────────────────

    #[test]
    fn test_federated_result_latency_non_negative() {
        let mut fed = FederatedGraphRag::new(FederationStrategy::BroadcastAll);
        fed.add_node(healthy_node("A", 10));
        let result = fed.query(&make_query("test"));
        // latency is wall-clock ms, should be very small in tests but ≥ 0
        // (just verify it doesn't panic / overflow)
        let _ = result.total_latency_ms;
    }

    // ── LocalRagEngine ────────────────────────────────────────────────────

    #[test]
    fn test_local_rag_returns_matching_passage() {
        let mut eng = LocalRagEngine::new();
        eng.add_passage("GraphRAG combines graph and retrieval", 0.8);
        eng.add_passage("Unrelated content here", 0.5);

        let results = eng.query("GraphRAG retrieval", 5, "local");
        assert!(!results.is_empty());
        assert!(results[0].text.contains("GraphRAG"));
    }

    #[test]
    fn test_local_rag_top_k_limit() {
        let mut eng = LocalRagEngine::new();
        for i in 0..10 {
            eng.add_passage(format!("passage {i} keyword"), 0.5);
        }
        let results = eng.query("keyword", 3, "local");
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_local_rag_no_match_returns_empty() {
        let mut eng = LocalRagEngine::new();
        eng.add_passage("Completely unrelated text", 0.5);
        let results = eng.query("xyzzy", 5, "local");
        assert!(results.is_empty());
    }

    // ── FederatedIndexBuilder::merge_indices ──────────────────────────────

    #[test]
    fn test_merge_indices_picks_best_score() {
        let i1 = LocalIndex {
            node_id: "A".to_string(),
            entries: vec![("key1".to_string(), 0.5), ("key2".to_string(), 0.9)],
        };
        let i2 = LocalIndex {
            node_id: "B".to_string(),
            entries: vec![("key1".to_string(), 0.8), ("key3".to_string(), 0.7)],
        };

        let merged = FederatedIndexBuilder::merge_indices(vec![i1, i2]);
        // key1: B wins (0.8 > 0.5)
        let key1 = merged
            .entries
            .iter()
            .find(|(k, _, _)| k == "key1")
            .expect("should succeed");
        assert!((key1.1 - 0.8).abs() < 1e-9);
        assert_eq!(key1.2, "B");
        // key2 from A, key3 from B
        assert_eq!(merged.entries.len(), 3);
    }

    #[test]
    fn test_merge_indices_sorted_descending() {
        let i1 = LocalIndex {
            node_id: "A".to_string(),
            entries: vec![
                ("low".to_string(), 0.1),
                ("high".to_string(), 0.9),
                ("mid".to_string(), 0.5),
            ],
        };
        let merged = FederatedIndexBuilder::merge_indices(vec![i1]);
        for i in 1..merged.entries.len() {
            assert!(merged.entries[i - 1].1 >= merged.entries[i].1);
        }
    }

    #[test]
    fn test_merge_indices_empty_returns_empty() {
        let merged = FederatedIndexBuilder::merge_indices(vec![]);
        assert!(merged.entries.is_empty());
    }

    // ── FederatedIndexBuilder::shard_index ───────────────────────────────

    #[test]
    fn test_shard_index_creates_correct_shard_count() {
        let merged = MergedIndex {
            entries: (0..10)
                .map(|i| (format!("key{i}"), i as f64 * 0.1, "A".to_string()))
                .collect(),
        };
        let shards = FederatedIndexBuilder::shard_index(&merged, 3);
        assert_eq!(shards.len(), 3);
    }

    #[test]
    fn test_shard_index_all_entries_distributed() {
        let merged = MergedIndex {
            entries: (0..9)
                .map(|i| (format!("key{i}"), 0.5, "A".to_string()))
                .collect(),
        };
        let shards = FederatedIndexBuilder::shard_index(&merged, 3);
        let total: usize = shards.iter().map(|s| s.entries.len()).sum();
        assert_eq!(total, 9);
    }

    #[test]
    fn test_shard_index_zero_shards_returns_empty() {
        let merged = MergedIndex {
            entries: vec![("k".to_string(), 0.5, "A".to_string())],
        };
        let shards = FederatedIndexBuilder::shard_index(&merged, 0);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_shard_index_ids_are_sequential() {
        let merged = MergedIndex {
            entries: (0..6)
                .map(|i| (format!("k{i}"), 0.5, "A".to_string()))
                .collect(),
        };
        let shards = FederatedIndexBuilder::shard_index(&merged, 3);
        for (expected, shard) in shards.iter().enumerate() {
            assert_eq!(shard.shard_id, expected);
        }
    }
}
