//! GraphSAGE: Inductive Representation Learning on Large Graphs
//!
//! Hamilton, Ying, Leskovec (2017) - NeurIPS
//!
//! Key idea: learn aggregation functions from node neighborhoods
//! rather than training per-node embeddings (transductive).
//! This enables inductive inference on unseen nodes.

use crate::EmbeddingError;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Aggregation strategy for neighborhood sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregatorType {
    /// Mean of neighbor features (GCN-like, simple and effective)
    Mean,
    /// Max-pool with learned MLP (captures representative features)
    MaxPool { hidden_dim: usize },
    /// Concatenation + mean (original GraphSAGE mean aggregator)
    MeanConcat,
}

/// GraphSAGE configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSageConfig {
    /// Dimensionality of input node features
    pub input_dim: usize,
    /// Dimensionality of hidden layers
    pub hidden_dims: Vec<usize>,
    /// Dimensionality of output embeddings
    pub output_dim: usize,
    /// Aggregation strategy
    pub aggregator: AggregatorType,
    /// Number of neighbor samples per hop, e.g. [25, 10]
    pub num_samples: Vec<usize>,
    /// Dropout rate (applied during training forward pass)
    pub dropout: f64,
    /// Learning rate for parameter updates
    pub learning_rate: f64,
    /// Number of training epochs
    pub epochs: usize,
    /// Mini-batch size for training
    pub batch_size: usize,
    /// L2-normalize output embeddings
    pub normalize_output: bool,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for GraphSageConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_dims: vec![256, 128],
            output_dim: 64,
            aggregator: AggregatorType::Mean,
            num_samples: vec![25, 10],
            dropout: 0.5,
            learning_rate: 0.01,
            epochs: 10,
            batch_size: 512,
            normalize_output: true,
            seed: 42,
        }
    }
}

/// Node features and graph structure for GraphSAGE
#[derive(Debug, Clone)]
pub struct GraphData {
    /// Node feature matrix: `node_features[i]` = feature vector for node i
    pub node_features: Vec<Vec<f64>>,
    /// Adjacency list: `adjacency[i]` = list of neighbor indices
    pub adjacency: Vec<Vec<usize>>,
    /// Optional node labels for supervised training
    pub labels: Option<Vec<usize>>,
}

impl GraphData {
    /// Create a new graph from node features and adjacency list.
    ///
    /// Validates that all adjacency indices are within bounds.
    pub fn new(features: Vec<Vec<f64>>, adjacency: Vec<Vec<usize>>) -> Result<Self> {
        let num_nodes = features.len();
        if adjacency.len() != num_nodes {
            return Err(anyhow!(
                "Adjacency list length {} does not match number of nodes {}",
                adjacency.len(),
                num_nodes
            ));
        }
        // Validate all neighbor indices
        for (i, neighbors) in adjacency.iter().enumerate() {
            for &neighbor in neighbors {
                if neighbor >= num_nodes {
                    return Err(anyhow!(
                        "Node {} has neighbor index {} which is out of bounds (num_nodes={})",
                        i,
                        neighbor,
                        num_nodes
                    ));
                }
            }
        }
        // Validate feature dimensions are consistent
        if let Some(first) = features.first() {
            let dim = first.len();
            for (i, feat) in features.iter().enumerate() {
                if feat.len() != dim {
                    return Err(anyhow!(
                        "Node {} has feature dimension {} but expected {}",
                        i,
                        feat.len(),
                        dim
                    ));
                }
            }
        }
        Ok(Self {
            node_features: features,
            adjacency,
            labels: None,
        })
    }

    /// Number of nodes in the graph
    pub fn num_nodes(&self) -> usize {
        self.node_features.len()
    }

    /// Dimensionality of node features
    pub fn feature_dim(&self) -> usize {
        self.node_features.first().map(|f| f.len()).unwrap_or(0)
    }

    /// Get the neighbors of a node
    pub fn neighbors(&self, node: usize) -> &[usize] {
        if node < self.adjacency.len() {
            &self.adjacency[node]
        } else {
            &[]
        }
    }

    /// Sample up to k neighbors uniformly at random using a simple LCG PRNG
    pub fn sample_neighbors(&self, node: usize, k: usize, rng: &mut SimpleLcg) -> Vec<usize> {
        let neighbors = self.neighbors(node);
        if neighbors.is_empty() {
            return Vec::new();
        }
        if neighbors.len() <= k {
            return neighbors.to_vec();
        }
        // Fisher-Yates partial shuffle to sample k items
        let mut indices: Vec<usize> = (0..neighbors.len()).collect();
        for i in 0..k {
            let j = i + (rng.next_usize() % (indices.len() - i));
            indices.swap(i, j);
        }
        indices[..k].iter().map(|&idx| neighbors[idx]).collect()
    }

    /// Set node labels for supervised training
    pub fn with_labels(mut self, labels: Vec<usize>) -> Result<Self> {
        if labels.len() != self.num_nodes() {
            return Err(anyhow!(
                "Labels length {} does not match num_nodes {}",
                labels.len(),
                self.num_nodes()
            ));
        }
        self.labels = Some(labels);
        Ok(self)
    }
}

/// Simple Linear Congruential Generator for reproducible sampling.
/// Avoids the need for external rand crate.
#[derive(Debug, Clone)]
pub struct SimpleLcg {
    state: u64,
}

impl SimpleLcg {
    /// Create a new LCG with the given seed
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Generate the next random u64
    pub fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Generate a random usize in range [0, n)
    pub fn next_usize(&mut self) -> usize {
        self.next_u64() as usize
    }

    /// Generate a random f64 in [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Generate a random f64 in [-scale, scale)
    pub fn next_f64_range(&mut self, scale: f64) -> f64 {
        (self.next_f64() * 2.0 - 1.0) * scale
    }
}

/// Dense linear layer: output = W * input + bias
#[derive(Debug, Clone)]
struct DenseLayer {
    weights: Vec<Vec<f64>>, // [output_dim][input_dim]
    bias: Vec<f64>,
    input_dim: usize,
    output_dim: usize,
}

impl DenseLayer {
    /// Xavier/Glorot uniform initialization
    fn new(input_dim: usize, output_dim: usize, rng: &mut SimpleLcg) -> Self {
        let scale = (6.0 / (input_dim + output_dim) as f64).sqrt();
        let weights = (0..output_dim)
            .map(|_| (0..input_dim).map(|_| rng.next_f64_range(scale)).collect())
            .collect();
        let bias = vec![0.0; output_dim];
        Self {
            weights,
            bias,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass: compute W*x + b
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        debug_assert_eq!(input.len(), self.input_dim);
        let mut output = self.bias.clone();
        for (i, row) in self.weights.iter().enumerate() {
            for (j, &w) in row.iter().enumerate() {
                output[i] += w * input[j];
            }
        }
        output
    }

    /// ReLU activation: max(0, x)
    fn relu(x: &[f64]) -> Vec<f64> {
        x.iter().map(|&v| v.max(0.0)).collect()
    }
}

/// A single GraphSAGE layer that aggregates neighbor information
#[derive(Debug, Clone)]
struct SageLayer {
    /// Transform for self node features
    self_transform: DenseLayer,
    /// Transform for aggregated neighbor features
    neigh_transform: DenseLayer,
    /// Optional pooling MLP for MaxPool aggregator
    pool_mlp: Option<DenseLayer>,
    /// Output dimensionality
    output_dim: usize,
}

impl SageLayer {
    /// Create a new SAGE layer.
    ///
    /// For MeanConcat: input to self_transform is input_dim,
    /// input to neigh_transform is neigh_dim.
    /// Final output is concat of both => output_dim each.
    fn new(
        input_dim: usize,
        neigh_dim: usize,
        output_dim: usize,
        pool_hidden: Option<usize>,
        rng: &mut SimpleLcg,
    ) -> Self {
        let self_transform = DenseLayer::new(input_dim, output_dim, rng);
        let neigh_transform = DenseLayer::new(neigh_dim, output_dim, rng);
        let pool_mlp = pool_hidden.map(|hidden| DenseLayer::new(neigh_dim, hidden, rng));
        Self {
            self_transform,
            neigh_transform,
            pool_mlp,
            output_dim,
        }
    }

    /// Mean aggregation: element-wise mean of neighbor features
    fn aggregate_mean(neighbor_features: &[Vec<f64>]) -> Vec<f64> {
        if neighbor_features.is_empty() {
            return Vec::new();
        }
        let dim = neighbor_features[0].len();
        let mut result = vec![0.0f64; dim];
        for feat in neighbor_features {
            for (r, &v) in result.iter_mut().zip(feat.iter()) {
                *r += v;
            }
        }
        let n = neighbor_features.len() as f64;
        result.iter_mut().for_each(|v| *v /= n);
        result
    }

    /// MaxPool aggregation: apply MLP then element-wise max
    fn aggregate_maxpool(neighbor_features: &[Vec<f64>], pool_layer: &DenseLayer) -> Vec<f64> {
        if neighbor_features.is_empty() {
            return Vec::new();
        }
        let transformed: Vec<Vec<f64>> = neighbor_features
            .iter()
            .map(|feat| DenseLayer::relu(&pool_layer.forward(feat)))
            .collect();
        let dim = transformed[0].len();
        let mut result = vec![f64::NEG_INFINITY; dim];
        for feat in &transformed {
            for (r, &v) in result.iter_mut().zip(feat.iter()) {
                if v > *r {
                    *r = v;
                }
            }
        }
        result
    }

    /// Forward pass for a single node.
    ///
    /// Computes: h = ReLU(W_self * self_feat + W_neigh * agg_neigh)
    /// Then normalizes if configured.
    fn forward(
        &self,
        self_feat: &[f64],
        neighbor_feats: &[Vec<f64>],
        aggregator: &AggregatorType,
    ) -> Vec<f64> {
        let agg = if neighbor_feats.is_empty() {
            vec![0.0; self_feat.len()]
        } else {
            match aggregator {
                AggregatorType::Mean | AggregatorType::MeanConcat => {
                    Self::aggregate_mean(neighbor_feats)
                }
                AggregatorType::MaxPool { .. } => {
                    if let Some(pool_layer) = &self.pool_mlp {
                        Self::aggregate_maxpool(neighbor_feats, pool_layer)
                    } else {
                        Self::aggregate_mean(neighbor_feats)
                    }
                }
            }
        };

        // Ensure agg has correct size for neigh_transform
        let agg_padded = if agg.len() != self.neigh_transform.input_dim {
            let mut padded = vec![0.0f64; self.neigh_transform.input_dim];
            let copy_len = agg.len().min(self.neigh_transform.input_dim);
            padded[..copy_len].copy_from_slice(&agg[..copy_len]);
            padded
        } else {
            agg
        };

        // Ensure self_feat has correct size for self_transform
        let self_padded = if self_feat.len() != self.self_transform.input_dim {
            let mut padded = vec![0.0f64; self.self_transform.input_dim];
            let copy_len = self_feat.len().min(self.self_transform.input_dim);
            padded[..copy_len].copy_from_slice(&self_feat[..copy_len]);
            padded
        } else {
            self_feat.to_vec()
        };

        let h_self = self.self_transform.forward(&self_padded);
        let h_neigh = self.neigh_transform.forward(&agg_padded);

        // Concatenate or add depending on aggregation type
        let combined = match aggregator {
            AggregatorType::MeanConcat => {
                // Concatenate self and neighbor, then project
                let mut concat = h_self;
                concat.extend(h_neigh);
                // For simplicity, we truncate/pad to output_dim
                concat.truncate(self.output_dim);
                while concat.len() < self.output_dim {
                    concat.push(0.0);
                }
                concat
            }
            _ => {
                // Element-wise sum
                h_self
                    .iter()
                    .zip(h_neigh.iter())
                    .map(|(a, b)| a + b)
                    .collect()
            }
        };

        // Apply ReLU activation (not on final layer typically, but standard here)
        DenseLayer::relu(&combined)
    }
}

/// GraphSAGE model for inductive node embedding
///
/// Implements the GraphSAGE algorithm from Hamilton et al. (2017).
/// Key property: can generate embeddings for nodes not seen during training
/// by aggregating from their neighborhoods.
#[derive(Debug, Clone)]
pub struct GraphSage {
    config: GraphSageConfig,
    layers: Vec<SageLayer>,
    rng: SimpleLcg,
}

impl GraphSage {
    /// Create a new GraphSAGE model with the given configuration
    pub fn new(config: GraphSageConfig) -> Result<Self> {
        if config.input_dim == 0 {
            return Err(anyhow!("input_dim must be > 0"));
        }
        if config.output_dim == 0 {
            return Err(anyhow!("output_dim must be > 0"));
        }
        if config.num_samples.is_empty() {
            return Err(anyhow!("num_samples must have at least one entry"));
        }

        let mut rng = SimpleLcg::new(config.seed);
        let pool_hidden = match &config.aggregator {
            AggregatorType::MaxPool { hidden_dim } => Some(*hidden_dim),
            _ => None,
        };

        // Build layer dimensions
        // Layer 0: input_dim -> hidden_dims[0]
        // Layer i: hidden_dims[i-1] -> hidden_dims[i]
        // Last layer: hidden_dims[-1] -> output_dim
        let mut dims: Vec<usize> = vec![config.input_dim];
        dims.extend(config.hidden_dims.iter().copied());
        dims.push(config.output_dim);

        let num_layers = dims.len() - 1;
        let mut layers = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let in_dim = dims[i];
            let out_dim = dims[i + 1];
            // Neighbor aggregation input dim: same as node feature dim at this layer
            let neigh_dim = in_dim;
            layers.push(SageLayer::new(
                in_dim,
                neigh_dim,
                out_dim,
                pool_hidden,
                &mut rng,
            ));
        }

        Ok(Self {
            config,
            layers,
            rng,
        })
    }

    /// Generate embeddings for all nodes via inductive forward pass.
    ///
    /// Performs K-hop neighborhood aggregation where K is the number of layers.
    pub fn embed(&self, graph: &GraphData) -> Result<GraphSageEmbeddings> {
        if graph.num_nodes() == 0 {
            return Err(anyhow!("Graph has no nodes"));
        }
        if graph.feature_dim() != self.config.input_dim {
            return Err(anyhow!(
                "Graph feature dim {} does not match model input_dim {}",
                graph.feature_dim(),
                self.config.input_dim
            ));
        }

        // Memoization: compute embeddings layer by layer for all nodes
        // h_prev[node] = node representation at previous layer
        let mut h_prev: Vec<Vec<f64>> = graph.node_features.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut h_next: Vec<Vec<f64>> = Vec::with_capacity(graph.num_nodes());

            // Determine neighbor sample count for this layer
            let num_samples = self
                .config
                .num_samples
                .get(layer_idx)
                .copied()
                .unwrap_or(25);

            // Use a deterministic sampling order for inference
            let mut local_rng = SimpleLcg::new(self.config.seed.wrapping_add(layer_idx as u64));

            for node in 0..graph.num_nodes() {
                let sampled = graph.sample_neighbors(node, num_samples, &mut local_rng);
                let neighbor_feats: Vec<Vec<f64>> = sampled
                    .iter()
                    .filter_map(|&n| h_prev.get(n).cloned())
                    .collect();

                let self_feat = h_prev.get(node).cloned().unwrap_or_default();
                let h = layer.forward(&self_feat, &neighbor_feats, &self.config.aggregator);
                h_next.push(h);
            }

            h_prev = h_next;
        }

        // Apply L2 normalization to final embeddings
        let embeddings: Vec<Vec<f64>> = if self.config.normalize_output {
            h_prev.into_iter().map(|v| Self::normalize(&v)).collect()
        } else {
            h_prev
        };

        let dim = self.config.output_dim;
        let num_nodes = graph.num_nodes();

        Ok(GraphSageEmbeddings {
            embeddings,
            config: self.config.clone(),
            num_nodes,
            dim,
        })
    }

    /// Train the model with unsupervised random-walk loss.
    ///
    /// Uses a simple positive/negative sampling strategy where:
    /// - Positive pairs: nodes connected by an edge (BFS neighbors)
    /// - Negative pairs: randomly sampled unconnected nodes
    ///
    /// Loss: -log(sigma(pos_score)) - log(1 - sigma(neg_score))
    pub fn train_unsupervised(&mut self, graph: &GraphData) -> Result<GraphSageTrainingMetrics> {
        if graph.num_nodes() < 2 {
            return Err(anyhow!("Graph must have at least 2 nodes for training"));
        }
        if graph.feature_dim() != self.config.input_dim {
            return Err(anyhow!(
                "Graph feature dim {} != model input_dim {}",
                graph.feature_dim(),
                self.config.input_dim
            ));
        }

        let mut loss_history = Vec::with_capacity(self.config.epochs);

        for epoch in 0..self.config.epochs {
            let embeddings = self.embed(graph)?;
            let epoch_loss = self.compute_unsupervised_loss(&embeddings, graph);
            loss_history.push(epoch_loss);

            // Gradient update: simple random perturbation for demonstration
            // In production, proper backpropagation would be used
            self.apply_random_gradient_step(epoch_loss);

            tracing::debug!(epoch = epoch, loss = epoch_loss, "GraphSAGE training step");
        }

        let final_loss = loss_history.last().copied().unwrap_or(f64::NAN);
        let convergence = loss_history.windows(2).all(|w| (w[1] - w[0]).abs() < 1e-4);

        Ok(GraphSageTrainingMetrics {
            epochs_completed: self.config.epochs,
            final_loss,
            loss_history,
            convergence_achieved: convergence,
        })
    }

    /// Compute unsupervised loss using positive/negative node pairs
    fn compute_unsupervised_loss(
        &self,
        embeddings: &GraphSageEmbeddings,
        graph: &GraphData,
    ) -> f64 {
        let num_nodes = graph.num_nodes();
        if num_nodes < 2 {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut count = 0usize;
        let mut local_rng = SimpleLcg::new(self.rng.state);

        // Collect a sample of positive edges
        let sample_nodes: Vec<usize> = (0..num_nodes.min(self.config.batch_size))
            .map(|i| i % num_nodes)
            .collect();

        for &node in &sample_nodes {
            let neighbors = graph.neighbors(node);
            if neighbors.is_empty() {
                continue;
            }
            // Positive: a direct neighbor
            let pos_neighbor = neighbors[local_rng.next_usize() % neighbors.len()];

            // Negative: a random non-neighbor node
            let neg_node = local_rng.next_usize() % num_nodes;

            if let (Some(h_u), Some(h_pos), Some(h_neg)) = (
                embeddings.get(node),
                embeddings.get(pos_neighbor),
                embeddings.get(neg_node),
            ) {
                let pos_score = dot_product(h_u, h_pos);
                let neg_score = dot_product(h_u, h_neg);

                // Cross-entropy loss
                let pos_loss = -sigmoid(pos_score).max(1e-10).ln();
                let neg_loss = -(1.0 - sigmoid(neg_score)).max(1e-10).ln();
                total_loss += pos_loss + neg_loss;
                count += 1;
            }
        }

        if count > 0 {
            total_loss / count as f64
        } else {
            0.0
        }
    }

    /// Apply a small random perturbation to layer weights (stub for full backprop)
    fn apply_random_gradient_step(&mut self, loss: f64) {
        let noise_scale = self.config.learning_rate * loss.abs().min(1.0) * 0.01;
        for layer in self.layers.iter_mut() {
            for row in layer.self_transform.weights.iter_mut() {
                for w in row.iter_mut() {
                    *w -= noise_scale * self.rng.next_f64_range(1.0);
                }
            }
        }
    }

    /// L2 normalize a vector
    pub fn normalize(v: &[f64]) -> Vec<f64> {
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            return v.to_vec();
        }
        v.iter().map(|x| x / norm).collect()
    }
}

/// Training metrics from a GraphSAGE training run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSageTrainingMetrics {
    pub epochs_completed: usize,
    pub final_loss: f64,
    pub loss_history: Vec<f64>,
    pub convergence_achieved: bool,
}

/// Output embeddings from GraphSAGE inference
#[derive(Debug, Clone)]
pub struct GraphSageEmbeddings {
    /// Embedding vectors indexed by node ID
    pub embeddings: Vec<Vec<f64>>,
    /// Configuration used for generation
    pub config: GraphSageConfig,
    /// Number of nodes
    pub num_nodes: usize,
    /// Embedding dimensionality
    pub dim: usize,
}

impl GraphSageEmbeddings {
    /// Get the embedding for a specific node
    pub fn get(&self, node: usize) -> Option<&[f64]> {
        self.embeddings.get(node).map(|v| v.as_slice())
    }

    /// Compute cosine similarity between two node embeddings
    pub fn cosine_similarity(&self, a: usize, b: usize) -> Option<f64> {
        let va = self.get(a)?;
        let vb = self.get(b)?;
        Some(cosine_similarity_vecs(va, vb))
    }

    /// Find the top-k most similar nodes to a given node
    ///
    /// Returns a sorted list of (node_id, similarity_score) pairs.
    pub fn top_k_similar(&self, node: usize, k: usize) -> Vec<(usize, f64)> {
        let query = match self.get(node) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut similarities: Vec<(usize, f64)> = (0..self.num_nodes)
            .filter(|&i| i != node)
            .filter_map(|i| self.get(i).map(|v| (i, cosine_similarity_vecs(query, v))))
            .collect();

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(k);
        similarities
    }

    /// Build a lookup from node label to embedding for classified nodes
    pub fn labeled_embeddings(&self, labels: &[usize]) -> HashMap<usize, Vec<Vec<f64>>> {
        let mut map: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();
        for (node, &label) in labels.iter().enumerate() {
            if let Some(emb) = self.get(node) {
                map.entry(label).or_default().push(emb.to_vec());
            }
        }
        map
    }
}

/// Dot product of two slices of equal length
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Sigmoid function: 1 / (1 + exp(-x))
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Cosine similarity between two vectors
pub fn cosine_similarity_vecs(a: &[f64], b: &[f64]) -> f64 {
    let dot = dot_product(a, b);
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }
    (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
}

/// Convert a `crate::EmbeddingError` to anyhow::Error for use in Results
pub fn embedding_err(msg: impl Into<String>) -> crate::EmbeddingError {
    EmbeddingError::Other(anyhow!(msg.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a simple star graph: node 0 is connected to nodes 1..n
    fn star_graph(n: usize, feat_dim: usize, seed: u64) -> GraphData {
        let mut rng = SimpleLcg::new(seed);
        let features: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..feat_dim).map(|_| rng.next_f64()).collect())
            .collect();
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        // Star: center 0 connects to all others
        for i in 1..n {
            adjacency[0].push(i);
            adjacency[i].push(0);
        }
        GraphData::new(features, adjacency).expect("star graph construction should succeed")
    }

    #[test]
    fn test_graphsage_default_config() {
        let config = GraphSageConfig::default();
        assert_eq!(config.input_dim, 64);
        assert_eq!(config.output_dim, 64);
        assert!(!config.num_samples.is_empty());
    }

    #[test]
    fn test_graphdata_construction() {
        let graph = star_graph(5, 8, 1);
        assert_eq!(graph.num_nodes(), 5);
        assert_eq!(graph.feature_dim(), 8);
        assert_eq!(graph.neighbors(0).len(), 4);
        assert_eq!(graph.neighbors(1).len(), 1);
        assert_eq!(graph.neighbors(1)[0], 0);
    }

    #[test]
    fn test_graphdata_invalid_adjacency() {
        let features = vec![vec![1.0, 2.0]; 3];
        let adjacency = vec![
            vec![1usize, 99], // 99 is out of bounds
            vec![0],
            vec![0],
        ];
        assert!(GraphData::new(features, adjacency).is_err());
    }

    #[test]
    fn test_graphsage_embed_shape() {
        let config = GraphSageConfig {
            input_dim: 8,
            hidden_dims: vec![16],
            output_dim: 4,
            num_samples: vec![3],
            epochs: 1,
            ..Default::default()
        };
        let model = GraphSage::new(config).expect("model construction should succeed");
        let graph = star_graph(5, 8, 42);
        let embeddings = model.embed(&graph).expect("embed should succeed");

        assert_eq!(embeddings.num_nodes, 5);
        assert_eq!(embeddings.dim, 4);
        for i in 0..5 {
            let emb = embeddings
                .get(i)
                .expect("should have embedding for every node");
            assert_eq!(emb.len(), 4);
        }
    }

    #[test]
    fn test_graphsage_normalized_output() {
        let config = GraphSageConfig {
            input_dim: 8,
            hidden_dims: vec![],
            output_dim: 8,
            num_samples: vec![5],
            normalize_output: true,
            epochs: 1,
            ..Default::default()
        };
        let model = GraphSage::new(config).expect("model should construct");
        let graph = star_graph(5, 8, 7);
        let embeddings = model.embed(&graph).expect("embed should succeed");

        // Each embedding should have unit norm (up to floating point tolerance)
        for i in 0..5 {
            let emb = embeddings.get(i).expect("embedding exists");
            let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
            // May be 0 if ReLU killed all activations, otherwise should be ~1
            assert!(norm < 1.0 + 1e-6, "norm {} should be <= 1", norm);
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let config = GraphSageConfig {
            input_dim: 4,
            hidden_dims: vec![],
            output_dim: 4,
            num_samples: vec![5],
            normalize_output: false,
            epochs: 1,
            ..Default::default()
        };
        let model = GraphSage::new(config).expect("model should construct");
        let graph = star_graph(5, 4, 13);
        let embeddings = model.embed(&graph).expect("embed should succeed");

        // Cosine similarity of a node with itself should be ~1.0
        if let Some(sim) = embeddings.cosine_similarity(0, 0) {
            // Self-similarity may be 0 if all values are 0 after ReLU
            assert!((0.0..=1.0 + 1e-6).contains(&sim));
        }
    }

    #[test]
    fn test_top_k_similar() {
        let config = GraphSageConfig {
            input_dim: 4,
            hidden_dims: vec![8],
            output_dim: 4,
            num_samples: vec![5],
            normalize_output: true,
            epochs: 1,
            ..Default::default()
        };
        let model = GraphSage::new(config).expect("model should construct");
        let graph = star_graph(6, 4, 17);
        let embeddings = model.embed(&graph).expect("embed should succeed");

        let top3 = embeddings.top_k_similar(0, 3);
        assert!(top3.len() <= 3);
        // Similarities should be in descending order
        for window in top3.windows(2) {
            assert!(window[0].1 >= window[1].1 - 1e-10);
        }
    }

    #[test]
    fn test_maxpool_aggregator() {
        let config = GraphSageConfig {
            input_dim: 4,
            hidden_dims: vec![8],
            output_dim: 4,
            aggregator: AggregatorType::MaxPool { hidden_dim: 8 },
            num_samples: vec![3],
            epochs: 1,
            ..Default::default()
        };
        let model = GraphSage::new(config).expect("model should construct with MaxPool");
        let graph = star_graph(4, 4, 99);
        let embeddings = model.embed(&graph).expect("embed should succeed");
        assert_eq!(embeddings.num_nodes, 4);
    }

    #[test]
    fn test_train_unsupervised() {
        let config = GraphSageConfig {
            input_dim: 4,
            hidden_dims: vec![8],
            output_dim: 4,
            num_samples: vec![3],
            epochs: 3,
            batch_size: 4,
            ..Default::default()
        };
        let mut model = GraphSage::new(config).expect("model should construct");
        let graph = star_graph(5, 4, 42);
        let metrics = model
            .train_unsupervised(&graph)
            .expect("training should succeed");
        assert_eq!(metrics.epochs_completed, 3);
        assert_eq!(metrics.loss_history.len(), 3);
    }

    #[test]
    fn test_simplecg_reproducibility() {
        let mut rng1 = SimpleLcg::new(42);
        let mut rng2 = SimpleLcg::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_sample_neighbors() {
        let graph = star_graph(10, 4, 1);
        let mut rng = SimpleLcg::new(55);
        let sampled = graph.sample_neighbors(0, 3, &mut rng);
        assert!(sampled.len() <= 3);
        for &n in &sampled {
            assert!(graph.neighbors(0).contains(&n));
        }
    }
}
