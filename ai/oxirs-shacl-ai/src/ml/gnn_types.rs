//! GNN type definitions: node/edge features, layer configs, aggregation types.

use serde::{Deserialize, Serialize};

/// GNN architecture types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GNNArchitecture {
    /// Graph Convolutional Network
    GCN,
    /// Graph Attention Network
    GAT,
    /// Graph Isomorphism Network
    GIN,
    /// GraphSAGE
    GraphSAGE,
    /// Message Passing Neural Network
    MPNN,
    /// Graph Completion Network for link prediction
    GraphCompletion,
    /// Entity Completion Network
    EntityCompletion,
    /// Relation Completion Network
    RelationCompletion,
    /// Graph Transformer for advanced pattern recognition
    GraphTransformer,
    /// Hierarchical Graph Transformer
    HierarchicalGraphTransformer,
}

/// Activation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationFunction {
    ReLU,
    LeakyReLU(f64),
    ELU(f64),
    Tanh,
    Sigmoid,
    GELU,
}

/// Aggregation functions for message passing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationFunction {
    Sum,
    Mean,
    Max,
    Min,
    Attention,
}

/// Scoring functions for graph completion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoringFunction {
    /// TransE scoring function
    TransE,
    /// DistMult scoring function
    DistMult,
    /// ComplEx scoring function
    ComplEx,
    /// RotatE scoring function
    RotatE,
    /// ConvE scoring function
    ConvE,
}

/// GNN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GNNConfig {
    pub architecture: GNNArchitecture,
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub activation: ActivationFunction,
    pub aggregation: AggregationFunction,
    pub attention_heads: usize,
    pub edge_features: bool,
    pub global_features: bool,
    pub residual_connections: bool,
    pub batch_normalization: bool,
}

/// Default GNN configuration
impl Default for GNNConfig {
    fn default() -> Self {
        Self {
            architecture: GNNArchitecture::GCN,
            num_layers: 3,
            hidden_dim: 128,
            output_dim: 64,
            activation: ActivationFunction::ReLU,
            aggregation: AggregationFunction::Mean,
            attention_heads: 4,
            edge_features: true,
            global_features: true,
            residual_connections: true,
            batch_normalization: true,
        }
    }
}
