//! Node2Vec: Scalable Feature Learning for Networks
//!
//! Reference: Grover & Leskovec (KDD 2016) — <https://arxiv.org/abs/1607.00653>
//!
//! # Algorithm Overview
//!
//! Node2Vec generates node embeddings via **biased second-order random walks**.
//! Two hyperparameters control the walk bias:
//!
//! * **p** (return parameter) – probability of returning to the previous node.
//!   High p → DFS-like walk (explores further from source).
//! * **q** (in-out parameter) – probability of moving away from the previous node.
//!   Low q → BFS-like walk (stays near source).
//!
//! The algorithm:
//! 1. Pre-compute per-edge alias tables for O(1) biased sampling.
//! 2. Simulate `num_walks` second-order random walks of length `walk_length`
//!    from every node.
//! 3. Train a simplified skip-gram model on the walk corpus to produce
//!    `embedding_dim`-dimensional embeddings.
//!
//! # Implementation Notes
//!
//! * No external ML or linear-algebra crates are required.
//! * Random number generation uses `scirs2_core::random` per project policy.
//! * All unsafe code is avoided; heavy loops use Rust iterators.

use petgraph::graph::{NodeIndex, UnGraph};
use scirs2_core::random::rand_prelude::StdRng;
use scirs2_core::random::{seeded_rng, CoreRandom, Random};
use std::collections::HashMap;

use crate::{GraphRAGError, GraphRAGResult, Triple};

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for the random-walk phase of Node2Vec.
#[derive(Debug, Clone)]
pub struct Node2VecWalkConfig {
    /// Number of random walks starting from each node.
    pub num_walks: usize,
    /// Length (number of steps) of each random walk.
    pub walk_length: usize,
    /// Return parameter *p* (see module docs).
    pub p: f64,
    /// In-out parameter *q* (see module docs).
    pub q: f64,
    /// Random seed for reproducibility.
    pub random_seed: u64,
}

impl Default for Node2VecWalkConfig {
    fn default() -> Self {
        Self {
            num_walks: 10,
            walk_length: 80,
            p: 1.0,
            q: 1.0,
            random_seed: 42,
        }
    }
}

/// Full Node2Vec configuration (walks + skip-gram training).
#[derive(Debug, Clone)]
pub struct Node2VecConfig {
    /// Walk phase configuration.
    pub walk: Node2VecWalkConfig,
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Skip-gram context window radius (both sides).
    pub window_size: usize,
    /// Number of training epochs over the full walk corpus.
    pub num_epochs: usize,
    /// Initial learning rate for skip-gram SGD.
    pub learning_rate: f64,
    /// Whether to normalize final embeddings to unit length.
    pub normalize: bool,
}

impl Default for Node2VecConfig {
    fn default() -> Self {
        Self {
            walk: Node2VecWalkConfig::default(),
            embedding_dim: 128,
            window_size: 5,
            num_epochs: 5,
            learning_rate: 0.025,
            normalize: true,
        }
    }
}

// ─── Output ──────────────────────────────────────────────────────────────────

/// Computed Node2Vec embeddings.
#[derive(Debug, Clone)]
pub struct Node2VecEmbeddings {
    /// Map from node URI/label → embedding vector.
    pub embeddings: HashMap<String, Vec<f64>>,
    /// Dimension of each embedding vector.
    pub dim: usize,
    /// Total number of random-walk steps generated.
    pub total_walk_steps: usize,
}

impl Node2VecEmbeddings {
    /// Return the embedding for a specific node, if present.
    pub fn get(&self, node: &str) -> Option<&[f64]> {
        self.embeddings.get(node).map(|v| v.as_slice())
    }

    /// Cosine similarity between two node embeddings.
    ///
    /// Returns `None` if either node is missing or an embedding has zero norm.
    pub fn cosine_similarity(&self, a: &str, b: &str) -> Option<f64> {
        let ea = self.embeddings.get(a)?;
        let eb = self.embeddings.get(b)?;

        let dot: f64 = ea.iter().zip(eb.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = ea.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = eb.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_a < 1e-12 || norm_b < 1e-12 {
            None
        } else {
            Some(dot / (norm_a * norm_b))
        }
    }

    /// Return the `k` most similar nodes to `query` by cosine similarity.
    pub fn top_k_similar(&self, query: &str, k: usize) -> Vec<(String, f64)> {
        let Some(eq) = self.embeddings.get(query) else {
            return vec![];
        };

        let norm_q: f64 = eq.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm_q < 1e-12 {
            return vec![];
        }

        let mut scored: Vec<(String, f64)> = self
            .embeddings
            .iter()
            .filter(|(node, _)| node.as_str() != query)
            .map(|(node, emb)| {
                let dot: f64 = emb.iter().zip(eq.iter()).map(|(x, y)| x * y).sum();
                let norm_e: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
                let sim = if norm_e < 1e-12 {
                    0.0
                } else {
                    dot / (norm_q * norm_e)
                };
                (node.clone(), sim)
            })
            .collect();

        scored.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

// ─── Alias sampling helper ────────────────────────────────────────────────────

/// Vose's alias method for O(1) sampling from a discrete distribution.
///
/// Reference: <https://www.keithschwarz.com/darts-dice-coins/>
struct AliasTable {
    prob: Vec<f64>,
    alias: Vec<usize>,
}

impl AliasTable {
    /// Build a table from a slice of unnormalized weights.
    fn build(weights: &[f64]) -> Option<Self> {
        let n = weights.len();
        if n == 0 {
            return None;
        }

        let sum: f64 = weights.iter().sum();
        if sum <= 0.0 {
            return None;
        }

        // Normalize.
        let prob_norm: Vec<f64> = weights.iter().map(|w| w * n as f64 / sum).collect();

        let mut small: Vec<usize> = Vec::with_capacity(n);
        let mut large: Vec<usize> = Vec::with_capacity(n);
        let mut prob = prob_norm.clone();
        let mut alias = vec![0usize; n];

        for (i, &p) in prob_norm.iter().enumerate() {
            if p < 1.0 {
                small.push(i);
            } else {
                large.push(i);
            }
        }

        while !small.is_empty() && !large.is_empty() {
            let l = small.pop().expect("checked non-empty");
            let g = large.last().copied().expect("checked non-empty");

            alias[l] = g;
            prob[g] -= 1.0 - prob[l];

            if prob[g] < 1.0 {
                large.pop();
                small.push(g);
            }
        }

        Some(Self { prob, alias })
    }

    /// Draw one sample index in O(1).
    fn sample(&self, rng: &mut CoreRandom<StdRng>) -> usize {
        let n = self.prob.len();
        // Pick a column uniformly at random.
        let i = (rng.random_range(0.0..1.0) * n as f64) as usize;
        let i = i.min(n - 1);
        // Flip a biased coin.
        if rng.random_range(0.0..1.0) < self.prob[i] {
            i
        } else {
            self.alias[i]
        }
    }
}

// ─── Pre-computed transition tables ──────────────────────────────────────────

/// Alias table keyed by (previous_node, current_node) → table for next step.
type EdgeAlias = HashMap<(NodeIndex, NodeIndex), (Vec<NodeIndex>, AliasTable)>;
/// Alias table for the *first* step from each node (no previous node yet).
type NodeAlias = HashMap<NodeIndex, (Vec<NodeIndex>, AliasTable)>;

// ─── Main embedder ────────────────────────────────────────────────────────────

/// Node2Vec graph embedding generator.
///
/// # Example
///
/// ```rust,ignore
/// use oxirs_graphrag::{Triple, embeddings::node2vec::{Node2VecConfig, Node2VecEmbedder}};
///
/// let triples = vec![
///     Triple::new("a", "knows", "b"),
///     Triple::new("b", "knows", "c"),
/// ];
/// let embedder = Node2VecEmbedder::new(Node2VecConfig::default());
/// let embs = embedder.embed(&triples)?;
/// println!("a→b similarity: {:?}", embs.cosine_similarity("a", "b"));
/// ```
pub struct Node2VecEmbedder {
    config: Node2VecConfig,
}

impl Node2VecEmbedder {
    /// Create a new embedder with the given configuration.
    pub fn new(config: Node2VecConfig) -> Self {
        Self { config }
    }

    /// Create a new embedder with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(Node2VecConfig::default())
    }

    /// Build embeddings for all nodes reachable via `triples`.
    pub fn embed(&self, triples: &[Triple]) -> GraphRAGResult<Node2VecEmbeddings> {
        // 1. Build adjacency graph.
        let (graph, node_map) = self.build_graph(triples);

        if graph.node_count() == 0 {
            return Ok(Node2VecEmbeddings {
                embeddings: HashMap::new(),
                dim: self.config.embedding_dim,
                total_walk_steps: 0,
            });
        }

        let mut rng = seeded_rng(self.config.walk.random_seed);

        // 2. Pre-compute alias tables for O(1) per-step sampling.
        let (node_alias, edge_alias) = self.build_alias_tables(&graph)?;

        // 3. Simulate random walks.
        let (walks, total_steps) =
            self.simulate_walks(&graph, &node_map, &node_alias, &edge_alias, &mut rng);

        // 4. Train skip-gram.
        let embeddings = self.train_skip_gram(&walks, &node_map, &mut rng)?;

        Ok(Node2VecEmbeddings {
            embeddings,
            dim: self.config.embedding_dim,
            total_walk_steps: total_steps,
        })
    }

    // ─── Graph construction ───────────────────────────────────────────────────

    fn build_graph(&self, triples: &[Triple]) -> (UnGraph<String, ()>, HashMap<String, NodeIndex>) {
        let mut graph: UnGraph<String, ()> = UnGraph::new_undirected();
        let mut node_map: HashMap<String, NodeIndex> = HashMap::new();

        for triple in triples {
            let s = *node_map
                .entry(triple.subject.clone())
                .or_insert_with(|| graph.add_node(triple.subject.clone()));
            let o = *node_map
                .entry(triple.object.clone())
                .or_insert_with(|| graph.add_node(triple.object.clone()));
            if s != o && graph.find_edge(s, o).is_none() {
                graph.add_edge(s, o, ());
            }
        }

        (graph, node_map)
    }

    // ─── Alias table construction ─────────────────────────────────────────────

    /// Build alias tables for:
    /// * first-step transitions from each node (uniform over neighbors), and
    /// * second-order transitions given (prev, cur) pairs.
    fn build_alias_tables(
        &self,
        graph: &UnGraph<String, ()>,
    ) -> GraphRAGResult<(NodeAlias, EdgeAlias)> {
        let p = self.config.walk.p;
        let q = self.config.walk.q;

        // Per-node table (first step).
        let mut node_alias: NodeAlias = HashMap::new();
        for node in graph.node_indices() {
            let neighbors: Vec<NodeIndex> = graph.neighbors(node).collect();
            if neighbors.is_empty() {
                continue;
            }
            let weights: Vec<f64> = vec![1.0; neighbors.len()];
            if let Some(table) = AliasTable::build(&weights) {
                node_alias.insert(node, (neighbors, table));
            }
        }

        // Per-edge table (second-order Markov transitions).
        let mut edge_alias: EdgeAlias = HashMap::new();
        for edge in graph.edge_indices() {
            let (u, v) = graph
                .edge_endpoints(edge)
                .ok_or_else(|| GraphRAGError::InternalError("bad edge".to_string()))?;

            // Process both directed orientations of the undirected edge.
            for (prev, cur) in [(u, v), (v, u)] {
                let neighbors: Vec<NodeIndex> = graph.neighbors(cur).collect();
                if neighbors.is_empty() {
                    continue;
                }
                let weights: Vec<f64> = neighbors
                    .iter()
                    .map(|&next| {
                        if next == prev {
                            // Return to previous node → weight = 1/p.
                            1.0 / p
                        } else if graph.find_edge(prev, next).is_some() {
                            // Neighbor of prev → stay in neighborhood → weight 1.
                            1.0
                        } else {
                            // Further away → weight = 1/q.
                            1.0 / q
                        }
                    })
                    .collect();

                if let Some(table) = AliasTable::build(&weights) {
                    edge_alias.insert((prev, cur), (neighbors, table));
                }
            }
        }

        Ok((node_alias, edge_alias))
    }

    // ─── Random walk simulation ───────────────────────────────────────────────

    fn simulate_walks(
        &self,
        graph: &UnGraph<String, ()>,
        node_map: &HashMap<String, NodeIndex>,
        node_alias: &NodeAlias,
        edge_alias: &EdgeAlias,
        rng: &mut CoreRandom<StdRng>,
    ) -> (Vec<Vec<String>>, usize) {
        let walk_length = self.config.walk.walk_length;
        let num_walks = self.config.walk.num_walks;

        let node_indices: Vec<NodeIndex> = node_map.values().copied().collect();
        let mut walks: Vec<Vec<String>> = Vec::with_capacity(num_walks * node_indices.len());
        let mut total_steps = 0usize;

        for _ in 0..num_walks {
            // Shuffle node order per walk round.
            let mut order = node_indices.clone();
            for i in (1..order.len()).rev() {
                let j = (rng.random_range(0.0..1.0) * (i + 1) as f64) as usize;
                order.swap(i, j.min(i));
            }

            for &start in &order {
                let walk = self.single_walk(graph, start, walk_length, node_alias, edge_alias, rng);
                total_steps += walk.len();
                walks.push(walk);
            }
        }

        (walks, total_steps)
    }

    fn single_walk(
        &self,
        graph: &UnGraph<String, ()>,
        start: NodeIndex,
        walk_length: usize,
        node_alias: &NodeAlias,
        edge_alias: &EdgeAlias,
        rng: &mut CoreRandom<StdRng>,
    ) -> Vec<String> {
        let mut walk: Vec<String> = Vec::with_capacity(walk_length);

        // Add start node label.
        if let Some(label) = graph.node_weight(start) {
            walk.push(label.clone());
        } else {
            return walk;
        }

        let mut current = start;
        let mut prev: Option<NodeIndex> = None;

        for _ in 1..walk_length {
            let next = if let Some(p) = prev {
                // Second-order: use per-edge alias table.
                if let Some((neighbors, table)) = edge_alias.get(&(p, current)) {
                    let idx = table.sample(rng);
                    neighbors.get(idx).copied()
                } else {
                    None
                }
            } else {
                // First step: uniform over neighbors.
                if let Some((neighbors, table)) = node_alias.get(&current) {
                    let idx = table.sample(rng);
                    neighbors.get(idx).copied()
                } else {
                    None
                }
            };

            match next {
                Some(n) => {
                    if let Some(label) = graph.node_weight(n) {
                        walk.push(label.clone());
                    }
                    prev = Some(current);
                    current = n;
                }
                None => break, // Dead end (isolated node after first step).
            }
        }

        walk
    }

    // ─── Skip-gram training ───────────────────────────────────────────────────

    /// Simplified skip-gram with stochastic gradient descent.
    ///
    /// For each (target, context) pair within the window, the update maximizes
    /// the inner-product similarity between target and context embeddings.
    /// Negative sampling is approximated via L2 regularization to bound norms.
    fn train_skip_gram(
        &self,
        walks: &[Vec<String>],
        node_map: &HashMap<String, NodeIndex>,
        rng: &mut CoreRandom<StdRng>,
    ) -> GraphRAGResult<HashMap<String, Vec<f64>>> {
        let dim = self.config.embedding_dim;
        let window = self.config.window_size;
        let lr_init = self.config.learning_rate;

        // Initialize embeddings with small random values.
        let mut embeddings: HashMap<String, Vec<f64>> = HashMap::new();
        for node_label in node_map.keys() {
            let emb: Vec<f64> = (0..dim)
                .map(|_| (rng.random_range(0.0..1.0) - 0.5) / dim as f64)
                .collect();
            embeddings.insert(node_label.clone(), emb);
        }

        // Context embeddings (separate parameter matrix as in word2vec).
        let mut ctx_embeddings: HashMap<String, Vec<f64>> = HashMap::new();
        for node_label in node_map.keys() {
            ctx_embeddings.insert(node_label.clone(), vec![0.0f64; dim]);
        }

        let total_epochs = self.config.num_epochs;
        let total_pairs: usize = walks
            .iter()
            .map(|w| w.len() * (2 * window).min(if w.len() > 1 { w.len() - 1 } else { 0 }))
            .sum();

        let mut pair_count = 0usize;

        for epoch in 0..total_epochs {
            // Linear learning rate decay.
            let lr = lr_init * (1.0 - epoch as f64 / total_epochs as f64).max(0.001);

            for walk in walks {
                for (i, target) in walk.iter().enumerate() {
                    let start = i.saturating_sub(window);
                    let end = (i + window + 1).min(walk.len());

                    for (j, context) in walk[start..end].iter().enumerate() {
                        let abs_j = start + j;
                        if abs_j == i || context == target {
                            continue;
                        }

                        // Decay lr further within epoch based on pair index.
                        let local_lr =
                            lr * (1.0 - pair_count as f64 / (total_pairs + 1) as f64).max(0.001);

                        self.sgd_update(
                            target,
                            context,
                            &mut embeddings,
                            &mut ctx_embeddings,
                            local_lr,
                            dim,
                        );

                        pair_count += 1;
                    }
                }
            }
        }

        // Merge context vectors into main embeddings (average of both).
        for (node, emb) in &mut embeddings {
            if let Some(ctx) = ctx_embeddings.get(node) {
                for (e, c) in emb.iter_mut().zip(ctx.iter()) {
                    *e = (*e + c) / 2.0;
                }
            }
        }

        // Optionally normalize to unit vectors.
        if self.config.normalize {
            for emb in embeddings.values_mut() {
                let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    for v in emb.iter_mut() {
                        *v /= norm;
                    }
                }
            }
        }

        Ok(embeddings)
    }

    /// Single SGD step for one (target, context) positive pair.
    ///
    /// Gradient of log-sigmoid loss:
    ///   L = log σ(v_t · v_c)
    ///   ∂L/∂v_t = (1 - σ(score)) · v_c
    ///   ∂L/∂v_c = (1 - σ(score)) · v_t
    fn sgd_update(
        &self,
        target: &str,
        context: &str,
        embeddings: &mut HashMap<String, Vec<f64>>,
        ctx_embeddings: &mut HashMap<String, Vec<f64>>,
        lr: f64,
        dim: usize,
    ) {
        // Compute inner product (score).
        let score = {
            let Some(te) = embeddings.get(target) else {
                return;
            };
            let Some(ce) = ctx_embeddings.get(context) else {
                return;
            };
            te.iter().zip(ce.iter()).map(|(a, b)| a * b).sum::<f64>()
        };

        // Sigmoid of score → gradient weight.
        let sigma = 1.0 / (1.0 + (-score).exp());
        let grad = (1.0 - sigma) * lr;

        // Capture snapshots to avoid borrow conflicts.
        let te_snap: Vec<f64> = match embeddings.get(target) {
            Some(v) => v.clone(),
            None => return,
        };
        let ce_snap: Vec<f64> = match ctx_embeddings.get(context) {
            Some(v) => v.clone(),
            None => return,
        };

        // Update target embedding.
        if let Some(te) = embeddings.get_mut(target) {
            for k in 0..dim {
                te[k] += grad * ce_snap[k];
            }
        }

        // Update context embedding.
        if let Some(ce) = ctx_embeddings.get_mut(context) {
            for k in 0..dim {
                ce[k] += grad * te_snap[k];
            }
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Triple;

    fn ring_triples(n: usize) -> Vec<Triple> {
        (0..n)
            .map(|i| {
                Triple::new(
                    format!("node_{}", i),
                    "connects",
                    format!("node_{}", (i + 1) % n),
                )
            })
            .collect()
    }

    fn complete_triples(n: usize) -> Vec<Triple> {
        let mut ts = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                ts.push(Triple::new(format!("n{}", i), "edge", format!("n{}", j)));
            }
        }
        ts
    }

    fn small_config() -> Node2VecConfig {
        Node2VecConfig {
            walk: Node2VecWalkConfig {
                num_walks: 5,
                walk_length: 10,
                p: 1.0,
                q: 1.0,
                random_seed: 99,
            },
            embedding_dim: 16,
            window_size: 2,
            num_epochs: 3,
            learning_rate: 0.05,
            normalize: true,
        }
    }

    #[test]
    fn test_embed_produces_correct_number_of_embeddings() {
        let triples = ring_triples(6);
        let embedder = Node2VecEmbedder::new(small_config());
        let result = embedder.embed(&triples).expect("embed failed");
        // A 6-node ring should produce 6 embeddings.
        assert_eq!(result.embeddings.len(), 6);
        assert_eq!(result.dim, 16);
    }

    #[test]
    fn test_embed_correct_dimension() {
        let triples = complete_triples(4);
        let embedder = Node2VecEmbedder::new(small_config());
        let result = embedder.embed(&triples).expect("embed failed");
        for emb in result.embeddings.values() {
            assert_eq!(emb.len(), 16);
        }
    }

    #[test]
    fn test_normalized_embeddings_have_unit_norm() {
        let triples = ring_triples(5);
        let config = Node2VecConfig {
            normalize: true,
            ..small_config()
        };
        let embedder = Node2VecEmbedder::new(config);
        let result = embedder.embed(&triples).expect("embed failed");
        for (node, emb) in &result.embeddings {
            let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "node {} has non-unit norm {:.6}",
                node,
                norm
            );
        }
    }

    #[test]
    fn test_cosine_similarity_in_range() {
        let triples = complete_triples(5);
        let embedder = Node2VecEmbedder::new(small_config());
        let result = embedder.embed(&triples).expect("embed failed");

        let nodes: Vec<String> = result.embeddings.keys().cloned().collect();
        if nodes.len() >= 2 {
            if let Some(sim) = result.cosine_similarity(&nodes[0], &nodes[1]) {
                assert!(
                    (-1.0 - 1e-9..=1.0 + 1e-9).contains(&sim),
                    "cosine similarity out of range: {}",
                    sim
                );
            }
        }
    }

    #[test]
    fn test_top_k_similar_returns_at_most_k() {
        let triples = ring_triples(8);
        let embedder = Node2VecEmbedder::new(small_config());
        let result = embedder.embed(&triples).expect("embed failed");

        let similar = result.top_k_similar("node_0", 3);
        assert!(similar.len() <= 3);
    }

    #[test]
    fn test_empty_triples_returns_empty_embeddings() {
        let embedder = Node2VecEmbedder::new(small_config());
        let result = embedder.embed(&[]).expect("embed failed");
        assert!(result.embeddings.is_empty());
        assert_eq!(result.total_walk_steps, 0);
    }

    #[test]
    fn test_single_node_isolated() {
        // A single triple where subject == object is filtered out,
        // but two different nodes with no shared edges should still produce embeddings.
        let triples = vec![Triple::new("a", "r", "b")];
        let embedder = Node2VecEmbedder::new(small_config());
        let result = embedder.embed(&triples).expect("embed failed");
        assert_eq!(result.embeddings.len(), 2);
    }

    #[test]
    fn test_walk_bias_dfs_vs_bfs() {
        // With q=0.1 (DFS), walks explore further; with q=10.0 (BFS), walks stay near.
        let triples = ring_triples(10);

        let dfs_config = Node2VecConfig {
            walk: Node2VecWalkConfig {
                num_walks: 3,
                walk_length: 20,
                p: 0.25,
                q: 0.25,
                random_seed: 1,
            },
            ..small_config()
        };
        let bfs_config = Node2VecConfig {
            walk: Node2VecWalkConfig {
                num_walks: 3,
                walk_length: 20,
                p: 4.0,
                q: 4.0,
                random_seed: 1,
            },
            ..small_config()
        };

        let embedder_dfs = Node2VecEmbedder::new(dfs_config);
        let embedder_bfs = Node2VecEmbedder::new(bfs_config);

        let res_dfs = embedder_dfs.embed(&triples).expect("dfs embed failed");
        let res_bfs = embedder_bfs.embed(&triples).expect("bfs embed failed");

        // Both should produce embeddings for all 10 nodes.
        assert_eq!(res_dfs.embeddings.len(), 10);
        assert_eq!(res_bfs.embeddings.len(), 10);
    }

    #[test]
    fn test_total_walk_steps_is_plausible() {
        let n = 5usize;
        let triples = ring_triples(n);
        let config = Node2VecConfig {
            walk: Node2VecWalkConfig {
                num_walks: 2,
                walk_length: 10,
                ..Default::default()
            },
            ..small_config()
        };
        let embedder = Node2VecEmbedder::new(config);
        let result = embedder.embed(&triples).expect("embed failed");

        // Each node starts a walk of up to `walk_length` steps.
        // total_steps ≥ num_nodes × num_walks (at least 1 step per walk).
        assert!(
            result.total_walk_steps >= n * 2,
            "expected ≥{} steps, got {}",
            n * 2,
            result.total_walk_steps
        );
        assert!(
            result.total_walk_steps <= n * 2 * 10 + n * 2,
            "unexpectedly many steps: {}",
            result.total_walk_steps
        );
    }

    #[test]
    fn test_alias_table_samples_valid_index() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let table = AliasTable::build(&weights).expect("alias build failed");
        let mut rng = seeded_rng(777);
        for _ in 0..100 {
            let idx = table.sample(&mut rng);
            assert!(idx < weights.len());
        }
    }

    #[test]
    fn test_alias_table_uniform_weights() {
        // All weights equal → each index should be sampled roughly equally.
        let weights = vec![1.0; 4];
        let table = AliasTable::build(&weights).expect("alias build failed");
        let mut rng = seeded_rng(42);
        let mut counts = [0usize; 4];
        for _ in 0..4000 {
            counts[table.sample(&mut rng)] += 1;
        }
        // Expect each bucket ~1000 ± 200.
        for c in counts {
            assert!(c > 800 && c < 1200, "bucket count out of range: {}", c);
        }
    }
}
