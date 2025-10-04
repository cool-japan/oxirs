//! Graph Neural Networks for RDF Knowledge Graphs
//!
//! This module implements various Graph Neural Network architectures optimized
//! for RDF knowledge graphs, including GCN, GraphSAGE, GAT, and custom architectures.

use crate::model::Triple;
use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::{Random, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Graph Neural Network trait
#[async_trait::async_trait]
pub trait GraphNeuralNetwork: Send + Sync {
    /// Forward pass through the network
    async fn forward(&self, graph: &RdfGraph, features: &Array2<f32>) -> Result<Array2<f32>>;

    /// Train the network on labeled data
    async fn train(
        &mut self,
        graph: &RdfGraph,
        features: &Array2<f32>,
        labels: &Array2<f32>,
        config: &TrainingConfig,
    ) -> Result<TrainingMetrics>;

    /// Get node embeddings
    async fn get_embeddings(&self, graph: &RdfGraph, features: &Array2<f32>)
        -> Result<Array2<f32>>;

    /// Predict links between entities
    async fn predict_links(
        &self,
        graph: &RdfGraph,
        source_nodes: &[usize],
        target_nodes: &[usize],
    ) -> Result<Array1<f32>>;

    /// Get model parameters
    fn get_parameters(&self) -> Result<Vec<Array2<f32>>>;

    /// Set model parameters
    fn set_parameters(&mut self, parameters: &[Array2<f32>]) -> Result<()>;

    /// Extract node features from RDF graph
    async fn extract_node_features(&self, graph: &RdfGraph) -> Result<Array2<f32>>;

    /// Compute loss between predictions and labels
    fn compute_loss(&self, predictions: &Array2<f32>, labels: &Array2<f32>) -> Result<f32>;

    /// Compute gradients for backpropagation
    async fn compute_gradients(
        &self,
        predictions: &Array2<f32>,
        labels: &Array2<f32>,
        graph: &RdfGraph,
        features: &Array2<f32>,
    ) -> Result<Vec<Array2<f32>>>;

    /// Apply gradient clipping to prevent exploding gradients
    fn clip_gradients(&self, gradients: Vec<Array2<f32>>, clip_value: f32) -> Vec<Array2<f32>>;

    /// Update parameters using the configured optimizer
    fn update_parameters(
        &mut self,
        gradients: &[Array2<f32>],
        momentum_buffers: &mut [Array2<f32>],
        velocity_buffers: &mut [Array2<f32>],
        config: &TrainingConfig,
        step: f32,
    ) -> Result<()>;

    /// Compute accuracy between predictions and labels
    fn compute_accuracy(&self, predictions: &Array2<f32>, labels: &Array2<f32>) -> Result<f32>;
}

/// GNN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnConfig {
    /// Network architecture type
    pub architecture: GnnArchitecture,

    /// Input feature dimension
    pub input_dim: usize,

    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,

    /// Output dimension
    pub output_dim: usize,

    /// Number of message passing layers
    pub num_layers: usize,

    /// Dropout rate
    pub dropout: f32,

    /// Activation function
    pub activation: ActivationFunction,

    /// Aggregation function for message passing
    pub aggregation: Aggregation,

    /// Whether to use residual connections
    pub use_residual: bool,

    /// Whether to use batch normalization
    pub use_batch_norm: bool,

    /// Learning rate
    pub learning_rate: f32,

    /// L2 regularization weight
    pub l2_weight: f32,

    /// Link prediction method
    pub link_prediction_method: LinkPredictionMethod,
}

impl Default for GnnConfig {
    fn default() -> Self {
        Self {
            architecture: GnnArchitecture::GraphConvolutionalNetwork,
            input_dim: 100,
            hidden_dims: vec![256, 128],
            output_dim: 64,
            num_layers: 3,
            dropout: 0.1,
            activation: ActivationFunction::ReLU,
            aggregation: Aggregation::Mean,
            use_residual: true,
            use_batch_norm: true,
            learning_rate: 0.001,
            l2_weight: 1e-4,
            link_prediction_method: LinkPredictionMethod::DotProduct,
        }
    }
}

/// GNN architecture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GnnArchitecture {
    /// Graph Convolutional Network
    GraphConvolutionalNetwork,

    /// GraphSAGE (Sample and Aggregate)
    GraphSage,

    /// Graph Attention Network
    GraphAttentionNetwork,

    /// Graph Isomorphism Network
    GraphIsomorphismNetwork,

    /// Relational Graph Convolutional Network
    RelationalGraphConvolutionalNetwork,

    /// Knowledge Graph Transformer
    KnowledgeGraphTransformer,

    /// Custom architecture
    Custom(CustomArchitectureConfig),
}

/// Custom architecture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomArchitectureConfig {
    /// Layer types and configurations
    pub layers: Vec<LayerConfig>,

    /// Skip connections
    pub skip_connections: Vec<(usize, usize)>,

    /// Custom message passing function
    pub message_passing: MessagePassingType,
}

/// Layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Layer type
    pub layer_type: LayerType,

    /// Input dimension
    pub input_dim: usize,

    /// Output dimension
    pub output_dim: usize,

    /// Layer-specific parameters
    pub parameters: HashMap<String, f32>,
}

/// Layer types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    /// Graph convolution layer
    GraphConvolution,

    /// Self-attention layer
    SelfAttention,

    /// Multi-head attention
    MultiHeadAttention { num_heads: usize },

    /// Feed-forward layer
    FeedForward,

    /// Residual layer
    Residual,

    /// Batch normalization
    BatchNorm,

    /// Dropout layer
    Dropout { rate: f32 },
}

/// Activation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU { negative_slope: f32 },
    ELU { alpha: f32 },
    GELU,
    Swish,
    Tanh,
    Sigmoid,
}

/// Message passing types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePassingType {
    /// Standard message passing
    Standard,

    /// Attention-based message passing
    Attention,

    /// Relational message passing
    Relational,

    /// Custom message function
    Custom(String),
}

/// Aggregation functions for message passing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Aggregation {
    /// Mean aggregation
    Mean,

    /// Sum aggregation
    Sum,

    /// Max pooling
    Max,

    /// Attention-weighted aggregation
    Attention,

    /// LSTM aggregation
    Lstm,

    /// Set2Set aggregation
    Set2Set,
}

/// Link prediction methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LinkPredictionMethod {
    /// Simple dot product similarity
    DotProduct,

    /// Cosine similarity
    Cosine,

    /// Negative L2 distance
    L2Distance,

    /// Bilinear transformation
    Bilinear,

    /// Multi-layer perceptron approach
    MLP,
}

/// RDF Graph representation for GNN
#[derive(Debug, Clone)]
pub struct RdfGraph {
    /// Number of nodes (entities)
    pub num_nodes: usize,

    /// Number of edges (relations)
    pub num_edges: usize,

    /// Edge index (source, target) pairs
    pub edge_index: Array2<usize>,

    /// Edge types (relation types)
    pub edge_types: Array1<usize>,

    /// Node features
    pub node_features: Option<Array2<f32>>,

    /// Edge features
    pub edge_features: Option<Array2<f32>>,

    /// Node to entity mapping
    pub node_to_entity: HashMap<usize, String>,

    /// Entity to node mapping
    pub entity_to_node: HashMap<String, usize>,

    /// Relation to type mapping
    pub relation_to_type: HashMap<String, usize>,

    /// Type to relation mapping
    pub type_to_relation: HashMap<usize, String>,
}

impl RdfGraph {
    /// Create RDF graph from triples
    pub fn from_triples(triples: &[Triple]) -> Result<Self> {
        let mut entities = HashSet::new();
        let mut relations = HashSet::new();

        // Collect entities and relations
        for triple in triples {
            entities.insert(triple.subject().to_string());
            entities.insert(triple.object().to_string());
            relations.insert(triple.predicate().to_string());
        }

        let num_nodes = entities.len();
        let num_edges = triples.len();

        // Create mappings
        let entity_to_node: HashMap<String, usize> = entities
            .iter()
            .enumerate()
            .map(|(i, entity)| (entity.clone(), i))
            .collect();

        let node_to_entity: HashMap<usize, String> = entity_to_node
            .iter()
            .map(|(entity, &node)| (node, entity.clone()))
            .collect();

        let relation_to_type: HashMap<String, usize> = relations
            .iter()
            .enumerate()
            .map(|(i, relation)| (relation.clone(), i))
            .collect();

        let type_to_relation: HashMap<usize, String> = relation_to_type
            .iter()
            .map(|(relation, &type_id)| (type_id, relation.clone()))
            .collect();

        // Build edge index and types
        let mut edge_sources = Vec::new();
        let mut edge_targets = Vec::new();
        let mut edge_types = Vec::new();

        for triple in triples {
            let source = entity_to_node[&triple.subject().to_string()];
            let target = entity_to_node[&triple.object().to_string()];
            let edge_type = relation_to_type[&triple.predicate().to_string()];

            edge_sources.push(source);
            edge_targets.push(target);
            edge_types.push(edge_type);
        }

        let edge_index =
            Array2::from_shape_vec((2, num_edges), [edge_sources, edge_targets].concat())?;

        let edge_types = Array1::from_vec(edge_types);

        Ok(Self {
            num_nodes,
            num_edges,
            edge_index,
            edge_types,
            node_features: None,
            edge_features: None,
            node_to_entity,
            entity_to_node,
            relation_to_type,
            type_to_relation,
        })
    }

    /// Add node features
    pub fn set_node_features(&mut self, features: Array2<f32>) -> Result<()> {
        if features.nrows() != self.num_nodes {
            return Err(anyhow!(
                "Node features shape mismatch: expected {} nodes, got {}",
                self.num_nodes,
                features.nrows()
            ));
        }
        self.node_features = Some(features);
        Ok(())
    }

    /// Add edge features
    pub fn set_edge_features(&mut self, features: Array2<f32>) -> Result<()> {
        if features.nrows() != self.num_edges {
            return Err(anyhow!(
                "Edge features shape mismatch: expected {} edges, got {}",
                self.num_edges,
                features.nrows()
            ));
        }
        self.edge_features = Some(features);
        Ok(())
    }

    /// Get node neighbors
    pub fn get_neighbors(&self, node: usize) -> Vec<usize> {
        let mut neighbors = Vec::new();

        for edge_idx in 0..self.num_edges {
            if self.edge_index[[0, edge_idx]] == node {
                neighbors.push(self.edge_index[[1, edge_idx]]);
            }
            // For undirected graphs, also check reverse direction
            if self.edge_index[[1, edge_idx]] == node {
                neighbors.push(self.edge_index[[0, edge_idx]]);
            }
        }

        neighbors
    }

    /// Get adjacency matrix
    pub fn get_adjacency_matrix(&self) -> Array2<f32> {
        let mut adj = Array2::zeros((self.num_nodes, self.num_nodes));

        for edge_idx in 0..self.num_edges {
            let source = self.edge_index[[0, edge_idx]];
            let target = self.edge_index[[1, edge_idx]];
            adj[[source, target]] = 1.0;
            // For undirected graphs
            adj[[target, source]] = 1.0;
        }

        adj
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.num_nodes
    }

    /// Get iterator over node indices
    pub fn nodes(&self) -> impl Iterator<Item = usize> {
        0..self.num_nodes
    }

    /// Get degree of a node
    pub fn degree(&self, node: usize) -> usize {
        self.get_neighbors(node).len()
    }

    /// Get in-degree of a node
    pub fn in_degree(&self, node: usize) -> Option<usize> {
        if node >= self.num_nodes {
            return None;
        }

        let mut count = 0;
        for edge_idx in 0..self.num_edges {
            if self.edge_index[[1, edge_idx]] == node {
                count += 1;
            }
        }
        Some(count)
    }

    /// Get out-degree of a node
    pub fn out_degree(&self, node: usize) -> Option<usize> {
        if node >= self.num_nodes {
            return None;
        }

        let mut count = 0;
        for edge_idx in 0..self.num_edges {
            if self.edge_index[[0, edge_idx]] == node {
                count += 1;
            }
        }
        Some(count)
    }
}

/// Graph Convolutional Network implementation
pub struct GraphConvolutionalNetwork {
    /// Model configuration
    config: GnnConfig,

    /// Layer weights
    layers: Vec<GraphConvLayer>,

    /// Model state
    trained: bool,
}

impl GraphConvolutionalNetwork {
    /// Create new GCN
    pub fn new(config: GnnConfig) -> Self {
        let mut layers = Vec::new();

        // Input layer
        let mut input_dim = config.input_dim;

        // Hidden layers
        for &hidden_dim in &config.hidden_dims {
            layers.push(GraphConvLayer::new(input_dim, hidden_dim));
            input_dim = hidden_dim;
        }

        // Output layer
        layers.push(GraphConvLayer::new(input_dim, config.output_dim));

        Self {
            config,
            layers,
            trained: false,
        }
    }
}

#[async_trait::async_trait]
impl GraphNeuralNetwork for GraphConvolutionalNetwork {
    async fn forward(&self, graph: &RdfGraph, features: &Array2<f32>) -> Result<Array2<f32>> {
        let mut x = features.clone();
        let adj = graph.get_adjacency_matrix();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &adj)?;

            // Apply activation function (except for last layer)
            if i < self.layers.len() - 1 {
                x = apply_activation(&x, &self.config.activation);

                // Apply dropout during training
                if !self.trained {
                    x = apply_dropout(&x, self.config.dropout);
                }
            }
        }

        Ok(x)
    }

    async fn train(
        &mut self,
        graph: &RdfGraph,
        features: &Array2<f32>,
        labels: &Array2<f32>,
        config: &TrainingConfig,
    ) -> Result<TrainingMetrics> {
        let start_time = std::time::Instant::now();
        let mut _total_loss = 0.0;
        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;

        tracing::info!("Starting GNN training for {} epochs", config.max_epochs);

        // Initialize optimizer state (for Adam-like optimizers)
        let mut momentum_buffers: Vec<Array2<f32>> = self
            .layers
            .iter()
            .map(|layer| Array2::zeros(layer.weight.dim()))
            .collect();
        let mut velocity_buffers: Vec<Array2<f32>> = self
            .layers
            .iter()
            .map(|layer| Array2::zeros(layer.weight.dim()))
            .collect();

        for epoch in 0..config.max_epochs {
            let epoch_start = std::time::Instant::now();

            // Forward pass
            let predictions = self.forward(graph, features).await?;

            // Compute loss (using mean squared error for simplicity)
            let loss = self.compute_loss(&predictions, labels)?;
            _total_loss += loss;

            // Backward pass: compute gradients
            let gradients = self
                .compute_gradients(&predictions, labels, graph, features)
                .await?;

            // Apply gradient clipping if configured
            let clipped_gradients = if let Some(clip_value) = config.gradient_clipping {
                self.clip_gradients(gradients, clip_value)
            } else {
                gradients
            };

            // Update parameters using optimizer
            self.update_parameters(
                &clipped_gradients,
                &mut momentum_buffers,
                &mut velocity_buffers,
                config,
                epoch as f32 + 1.0,
            )?;

            // Compute accuracy
            let accuracy = self.compute_accuracy(&predictions, labels)?;

            // Early stopping check
            if loss < best_loss {
                best_loss = loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= config.early_stopping.patience {
                    tracing::info!("Early stopping triggered at epoch {}", epoch);
                    break;
                }
            }

            let epoch_time = epoch_start.elapsed();

            if epoch % 10 == 0 {
                tracing::info!(
                    "Epoch {}: loss={:.6}, accuracy={:.4}, time={:?}",
                    epoch,
                    loss,
                    accuracy,
                    epoch_time
                );
            }
        }

        self.trained = true;
        let total_time = start_time.elapsed();

        tracing::info!(
            "GNN training completed. Final loss: {:.6}, Time: {:?}",
            best_loss,
            total_time
        );

        Ok(TrainingMetrics {
            loss: best_loss,
            accuracy: self.compute_accuracy(&self.forward(graph, features).await?, labels)?,
            epochs: config.max_epochs,
            time_elapsed: total_time,
        })
    }

    async fn get_embeddings(
        &self,
        graph: &RdfGraph,
        features: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        self.forward(graph, features).await
    }

    async fn predict_links(
        &self,
        graph: &RdfGraph,
        source_nodes: &[usize],
        target_nodes: &[usize],
    ) -> Result<Array1<f32>> {
        if source_nodes.len() != target_nodes.len() {
            return Err(anyhow!(
                "Source and target node arrays must have the same length"
            ));
        }

        if !self.trained {
            tracing::warn!("Model not trained yet, predictions may be inaccurate");
        }

        // Extract node features from the graph
        let node_features = self.extract_node_features(graph).await?;

        // Get embeddings for all nodes
        let embeddings = self.get_embeddings(graph, &node_features).await?;

        let mut predictions = Array1::zeros(source_nodes.len());

        for (i, (&src, &tgt)) in source_nodes.iter().zip(target_nodes.iter()).enumerate() {
            if src >= embeddings.nrows() || tgt >= embeddings.nrows() {
                return Err(anyhow!("Node index out of bounds"));
            }

            // Extract embeddings for source and target nodes
            let src_embedding = embeddings.row(src);
            let tgt_embedding = embeddings.row(tgt);

            // Compute link prediction score using different methods
            let score = match self.config.link_prediction_method {
                LinkPredictionMethod::DotProduct => {
                    // Simple dot product similarity
                    src_embedding
                        .iter()
                        .zip(tgt_embedding.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>()
                }
                LinkPredictionMethod::Cosine => {
                    // Cosine similarity
                    let dot_product = src_embedding
                        .iter()
                        .zip(tgt_embedding.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>();

                    let src_norm = src_embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let tgt_norm = tgt_embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();

                    if src_norm > 0.0 && tgt_norm > 0.0 {
                        dot_product / (src_norm * tgt_norm)
                    } else {
                        0.0
                    }
                }
                LinkPredictionMethod::L2Distance => {
                    // Negative L2 distance (higher score = more similar)
                    let distance = src_embedding
                        .iter()
                        .zip(tgt_embedding.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    -distance
                }
                LinkPredictionMethod::Bilinear => {
                    // Bilinear transformation: src^T * W * tgt
                    // For simplicity, use identity matrix (equivalent to dot product)
                    src_embedding
                        .iter()
                        .zip(tgt_embedding.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>()
                }
                LinkPredictionMethod::MLP => {
                    // Multi-layer perceptron approach
                    // Concatenate embeddings and pass through a simple network
                    let concat_features: Vec<f32> = src_embedding
                        .iter()
                        .chain(tgt_embedding.iter())
                        .cloned()
                        .collect();

                    // Simple linear transformation (placeholder for full MLP)
                    let hidden_size = concat_features.len() / 2;
                    let mut hidden: Vec<f32> = vec![0.0; hidden_size];

                    for (i, &feature) in concat_features.iter().enumerate() {
                        hidden[i % hidden_size] += feature * 0.1; // Simple weight
                    }

                    // Apply activation and output layer
                    let activated: f32 = hidden.iter().map(|&x| x.tanh()).sum();
                    activated / hidden_size as f32
                }
            };

            // Apply sigmoid to get probability-like score
            predictions[i] = 1.0 / (1.0 + (-score).exp());
        }

        Ok(predictions)
    }

    /// Extract node features from RDF graph
    async fn extract_node_features(&self, graph: &RdfGraph) -> Result<Array2<f32>> {
        let node_count = graph.node_count();
        let feature_dim = 64; // Default feature dimension

        let mut features = Array2::zeros((node_count, feature_dim));

        // Simple feature extraction based on node properties
        for node_idx in graph.nodes() {
            // Extract features based on node type, degree, and properties
            let degree = graph.degree(node_idx) as f32;
            let in_degree = graph.in_degree(node_idx).unwrap_or(0) as f32;
            let out_degree = graph.out_degree(node_idx).unwrap_or(0) as f32;

            // Basic structural features
            features[[node_idx, 0]] = degree.ln_1p(); // Log-normalized degree
            features[[node_idx, 1]] = in_degree.ln_1p();
            features[[node_idx, 2]] = out_degree.ln_1p();
            features[[node_idx, 3]] = (in_degree / (degree + 1.0)).clamp(0.0, 1.0); // In-degree ratio
            features[[node_idx, 4]] = (out_degree / (degree + 1.0)).clamp(0.0, 1.0); // Out-degree ratio

            // Node type encoding based on entity IRI
            if let Some(entity_iri) = graph.node_to_entity.get(&node_idx) {
                let node_type_hash = entity_iri.len() % 10;
                features[[node_idx, 5]] = node_type_hash as f32 / 10.0;

                // Random features for remaining dimensions (placeholder for actual properties)
                for i in 6..feature_dim {
                    features[[node_idx, i]] = (entity_iri.len() * i) as f32 / 1000.0 % 1.0;
                }
            }
        }

        Ok(features)
    }

    fn get_parameters(&self) -> Result<Vec<Array2<f32>>> {
        Ok(self
            .layers
            .iter()
            .map(|layer| layer.weight.clone())
            .collect())
    }

    fn set_parameters(&mut self, parameters: &[Array2<f32>]) -> Result<()> {
        if parameters.len() != self.layers.len() {
            return Err(anyhow!("Parameter count mismatch"));
        }

        for (layer, param) in self.layers.iter_mut().zip(parameters.iter()) {
            layer.weight = param.clone();
        }

        Ok(())
    }

    /// Compute loss between predictions and labels
    fn compute_loss(&self, predictions: &Array2<f32>, labels: &Array2<f32>) -> Result<f32> {
        if predictions.dim() != labels.dim() {
            return Err(anyhow!("Predictions and labels dimension mismatch"));
        }

        // Mean Squared Error loss
        let diff = predictions - labels;
        let squared_diff = &diff * &diff;
        let mse = squared_diff.mean().unwrap_or(0.0);

        Ok(mse)
    }

    /// Compute gradients for backpropagation
    async fn compute_gradients(
        &self,
        predictions: &Array2<f32>,
        labels: &Array2<f32>,
        graph: &RdfGraph,
        features: &Array2<f32>,
    ) -> Result<Vec<Array2<f32>>> {
        let mut gradients = Vec::with_capacity(self.layers.len());

        // Output layer gradient (MSE derivative)
        let _output_grad = 2.0 * (predictions - labels) / (predictions.len() as f32);

        // For simplicity, we'll compute approximate gradients using finite differences
        // In a full implementation, this would use automatic differentiation
        let epsilon = 1e-5;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut layer_grad = Array2::zeros(layer.weight.dim());

            // Compute gradient for each weight using finite differences
            for i in 0..layer.weight.nrows() {
                for j in 0..layer.weight.ncols() {
                    // Forward pass with perturbed weight
                    let mut perturbed_layers = self.layers.clone();
                    perturbed_layers[layer_idx].weight[[i, j]] += epsilon;

                    let perturbed_network = GraphConvolutionalNetwork {
                        layers: perturbed_layers,
                        config: self.config.clone(),
                        trained: self.trained,
                    };

                    let perturbed_output = perturbed_network.forward(graph, features).await?;
                    let perturbed_loss = self.compute_loss(&perturbed_output, labels)?;
                    let original_loss = self.compute_loss(predictions, labels)?;

                    // Approximate gradient using finite difference
                    layer_grad[[i, j]] = (perturbed_loss - original_loss) / epsilon;
                }
            }

            gradients.push(layer_grad);
        }

        Ok(gradients)
    }

    /// Apply gradient clipping to prevent exploding gradients
    fn clip_gradients(&self, gradients: Vec<Array2<f32>>, clip_value: f32) -> Vec<Array2<f32>> {
        let mut clipped_gradients = Vec::with_capacity(gradients.len());

        for grad in gradients {
            // Compute L2 norm of gradient
            let grad_norm = grad.iter().map(|&x| x * x).sum::<f32>().sqrt();

            if grad_norm > clip_value {
                // Scale gradients to clip_value
                let scale_factor = clip_value / grad_norm;
                clipped_gradients.push(&grad * scale_factor);
            } else {
                clipped_gradients.push(grad);
            }
        }

        clipped_gradients
    }

    /// Update parameters using the configured optimizer
    fn update_parameters(
        &mut self,
        gradients: &[Array2<f32>],
        momentum_buffers: &mut [Array2<f32>],
        velocity_buffers: &mut [Array2<f32>],
        config: &TrainingConfig,
        step: f32,
    ) -> Result<()> {
        if gradients.len() != self.layers.len() {
            return Err(anyhow!("Gradient count mismatch"));
        }

        match &config.optimizer {
            Optimizer::SGD {
                momentum,
                weight_decay,
                nesterov,
            } => {
                for (i, (layer, grad)) in self.layers.iter_mut().zip(gradients.iter()).enumerate() {
                    // Apply weight decay
                    let mut effective_grad = grad.clone();
                    if weight_decay > &0.0 {
                        effective_grad = &effective_grad + &(&layer.weight * *weight_decay);
                    }

                    // Update momentum buffer
                    momentum_buffers[i] = &(&momentum_buffers[i] * *momentum) + &effective_grad;

                    // Apply Nesterov acceleration if enabled
                    let update = if *nesterov {
                        &effective_grad + &(&momentum_buffers[i] * *momentum)
                    } else {
                        momentum_buffers[i].clone()
                    };

                    // Update parameters
                    layer.weight = &layer.weight - &(&update * config.learning_rate);
                }
            }
            Optimizer::Adam {
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => {
                for (i, (layer, grad)) in self.layers.iter_mut().zip(gradients.iter()).enumerate() {
                    // Apply weight decay
                    let mut effective_grad = grad.clone();
                    if weight_decay > &0.0 {
                        effective_grad = &effective_grad + &(&layer.weight * *weight_decay);
                    }

                    // Update biased first moment estimate
                    momentum_buffers[i] =
                        &(&momentum_buffers[i] * *beta1) + &(&effective_grad * (1.0 - beta1));

                    // Update biased second raw moment estimate
                    let grad_squared = &effective_grad * &effective_grad;
                    velocity_buffers[i] =
                        &(&velocity_buffers[i] * *beta2) + &(&grad_squared * (1.0 - beta2));

                    // Compute bias-corrected first moment estimate
                    let bias_correction1 = 1.0 - beta1.powf(step);
                    let bias_correction2 = 1.0 - beta2.powf(step);

                    let m_hat = &momentum_buffers[i] / bias_correction1;
                    let v_hat = &velocity_buffers[i] / bias_correction2;

                    // Update parameters
                    let denominator = v_hat.mapv(|x| x.sqrt() + epsilon);
                    let update = &m_hat / &denominator;
                    layer.weight = &layer.weight - &(&update * config.learning_rate);
                }
            }
            _ => {
                // Default to simple gradient descent for other optimizers
                for (layer, grad) in self.layers.iter_mut().zip(gradients.iter()) {
                    layer.weight = &layer.weight - &(grad * config.learning_rate);
                }
            }
        }

        Ok(())
    }

    /// Compute accuracy between predictions and labels
    fn compute_accuracy(&self, predictions: &Array2<f32>, labels: &Array2<f32>) -> Result<f32> {
        if predictions.dim() != labels.dim() {
            return Err(anyhow!("Predictions and labels dimension mismatch"));
        }

        // For regression tasks, use R² score (coefficient of determination)
        let labels_mean = labels.mean().unwrap_or(0.0);
        let ss_res: f32 = predictions
            .iter()
            .zip(labels.iter())
            .map(|(&pred, &label)| (label - pred).powi(2))
            .sum();
        let ss_tot: f32 = labels
            .iter()
            .map(|&label| (label - labels_mean).powi(2))
            .sum();

        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        Ok(r_squared.max(0.0)) // Clamp to [0, 1]
    }
}

/// Graph convolution layer
#[derive(Debug, Clone)]
pub struct GraphConvLayer {
    pub weight: Array2<f32>,
    pub bias: Array1<f32>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl GraphConvLayer {
    /// Create new graph convolution layer
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        // Initialize weights with Xavier initialization
        let weight = Array2::from_shape_simple_fn((input_dim, output_dim), || {
            ({
                let mut rng = Random::default();
                rng.random::<f32>()
            }) * 2.0
                / (input_dim as f32).sqrt()
                - 1.0 / (input_dim as f32).sqrt()
        });

        let bias = Array1::zeros(output_dim);

        Self {
            weight,
            bias,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Array2<f32>, adj: &Array2<f32>) -> Result<Array2<f32>> {
        // Graph convolution: A * X * W + b
        let ax = adj.dot(x);
        let axw = ax.dot(&self.weight);
        let output = axw + &self.bias;

        Ok(output)
    }
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Maximum number of epochs
    pub max_epochs: usize,

    /// Batch size
    pub batch_size: usize,

    /// Learning rate
    pub learning_rate: f32,

    /// Early stopping patience
    pub patience: usize,

    /// Validation split ratio
    pub validation_split: f32,

    /// Loss function
    pub loss_function: LossFunction,

    /// Optimizer
    pub optimizer: Optimizer,

    /// Gradient clipping threshold
    pub gradient_clipping: Option<f32>,

    /// Early stopping configuration
    pub early_stopping: EarlyStoppingConfig,
}

/// Early stopping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Whether early stopping is enabled
    pub enabled: bool,

    /// Patience (number of epochs without improvement)
    pub patience: usize,

    /// Minimum improvement threshold
    pub min_delta: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 1000,
            batch_size: 256,
            learning_rate: 0.001,
            patience: 50,
            validation_split: 0.2,
            loss_function: LossFunction::CrossEntropy,
            optimizer: Optimizer::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 1e-4,
            },
            gradient_clipping: Some(1.0),
            early_stopping: EarlyStoppingConfig {
                enabled: true,
                patience: 50,
                min_delta: 1e-6,
            },
        }
    }
}

/// Loss functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LossFunction {
    /// Cross-entropy loss
    CrossEntropy,

    /// Mean squared error
    MeanSquaredError,

    /// Binary cross-entropy
    BinaryCrossEntropy,

    /// Hinge loss
    HingeLoss,

    /// Contrastive loss
    ContrastiveLoss { margin: f32 },
}

/// Optimizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Optimizer {
    /// Stochastic Gradient Descent
    SGD {
        momentum: f32,
        weight_decay: f32,
        nesterov: bool,
    },

    /// Adam optimizer
    Adam {
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },

    /// AdaGrad optimizer
    AdaGrad { epsilon: f32 },

    /// RMSprop optimizer
    RMSprop { decay: f32, epsilon: f32 },
}

/// Training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Final loss value
    pub loss: f32,

    /// Final accuracy
    pub accuracy: f32,

    /// Number of epochs trained
    pub epochs: usize,

    /// Total training time
    pub time_elapsed: std::time::Duration,
}

/// Apply activation function
pub fn apply_activation(x: &Array2<f32>, activation: &ActivationFunction) -> Array2<f32> {
    match activation {
        ActivationFunction::ReLU => x.mapv(|v| v.max(0.0)),
        ActivationFunction::LeakyReLU { negative_slope } => {
            x.mapv(|v| if v > 0.0 { v } else { v * negative_slope })
        }
        ActivationFunction::ELU { alpha } => {
            x.mapv(|v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) })
        }
        ActivationFunction::GELU => {
            x.mapv(|v| 0.5 * v * (1.0 + (v * 0.797_884_6 * (1.0 + 0.044715 * v * v)).tanh()))
        }
        ActivationFunction::Swish => x.mapv(|v| v * (1.0 / (1.0 + (-v).exp()))),
        ActivationFunction::Tanh => x.mapv(|v| v.tanh()),
        ActivationFunction::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
    }
}

/// Apply dropout
pub fn apply_dropout(x: &Array2<f32>, rate: f32) -> Array2<f32> {
    if rate <= 0.0 {
        return x.clone();
    }

    let keep_prob = 1.0 - rate;
    x.mapv(|v| {
        if {
            let mut rng = Random::default();
            rng.random::<f32>()
        } < keep_prob
        {
            v / keep_prob
        } else {
            0.0
        }
    })
}

/// Create GNN based on configuration
pub fn create_gnn(config: GnnConfig) -> Result<Arc<dyn GraphNeuralNetwork>> {
    match config.architecture {
        GnnArchitecture::GraphConvolutionalNetwork => {
            Ok(Arc::new(GraphConvolutionalNetwork::new(config)))
        }
        GnnArchitecture::GraphSage => {
            // TODO: Implement GraphSAGE
            Err(anyhow!("GraphSAGE not yet implemented"))
        }
        GnnArchitecture::GraphAttentionNetwork => {
            // TODO: Implement GAT
            Err(anyhow!("GAT not yet implemented"))
        }
        _ => Err(anyhow!("Unsupported GNN architecture")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Literal, NamedNode};

    #[test]
    fn test_rdf_graph_creation() {
        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/person1").unwrap(),
                NamedNode::new("http://example.org/name").unwrap(),
                Literal::new("Alice"),
            ),
            Triple::new(
                NamedNode::new("http://example.org/person1").unwrap(),
                NamedNode::new("http://example.org/age").unwrap(),
                Literal::new("30"),
            ),
        ];

        let graph = RdfGraph::from_triples(&triples).unwrap();
        assert_eq!(graph.num_nodes, 3); // person1, Alice, 30 (predicates are not counted as nodes)
        assert_eq!(graph.num_edges, 2);
    }

    #[test]
    fn test_gcn_creation() {
        let config = GnnConfig::default();
        let gcn = GraphConvolutionalNetwork::new(config);
        assert_eq!(gcn.layers.len(), 3); // 2 hidden + 1 output
    }

    #[tokio::test]
    async fn test_gcn_forward() {
        let config = GnnConfig {
            input_dim: 10,
            hidden_dims: vec![20],
            output_dim: 5,
            ..Default::default()
        };

        let gcn = GraphConvolutionalNetwork::new(config);

        let triples = vec![Triple::new(
            NamedNode::new("http://example.org/a").unwrap(),
            NamedNode::new("http://example.org/rel").unwrap(),
            NamedNode::new("http://example.org/b").unwrap(),
        )];

        let graph = RdfGraph::from_triples(&triples).unwrap();
        let features = Array2::ones((graph.num_nodes, 10));

        let output = gcn.forward(&graph, &features).await.unwrap();
        assert_eq!(output.shape(), &[graph.num_nodes, 5]);
    }

    #[test]
    fn test_activation_functions() {
        let x = Array2::from_shape_vec((2, 2), vec![-1.0, 0.0, 1.0, 2.0]).unwrap();

        let relu = apply_activation(&x, &ActivationFunction::ReLU);
        assert_eq!(relu[[0, 0]], 0.0);
        assert_eq!(relu[[1, 1]], 2.0);

        let sigmoid = apply_activation(&x, &ActivationFunction::Sigmoid);
        assert!(sigmoid[[0, 0]] > 0.0 && sigmoid[[0, 0]] < 1.0);
    }
}
