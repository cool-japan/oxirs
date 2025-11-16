//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array1, Array2};
use scirs2_core::random::{Random, Rng};
use std::sync::Arc;

#[cfg(test)]
use crate::Triple;
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
        GnnArchitecture::GraphSage => Ok(Arc::new(GraphSageNetwork::new(config))),
        GnnArchitecture::GraphAttentionNetwork => Ok(Arc::new(GraphAttentionNetwork::new(config))),
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
        assert_eq!(graph.num_nodes, 3);
        assert_eq!(graph.num_edges, 2);
    }
    #[test]
    fn test_gcn_creation() {
        let config = GnnConfig::default();
        let gcn = GraphConvolutionalNetwork::new(config);
        assert_eq!(gcn.layers.len(), 3);
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
    #[test]
    fn test_graphsage_creation() {
        let config = GnnConfig {
            architecture: GnnArchitecture::GraphSage,
            input_dim: 10,
            hidden_dims: vec![20, 15],
            output_dim: 5,
            num_layers: 3,
            ..Default::default()
        };
        let graphsage = GraphSageNetwork::new(config.clone());
        assert_eq!(graphsage.layers.len(), 3);
        assert_eq!(graphsage.num_samples.len(), 3);
        assert!(!graphsage.trained);
    }
    #[tokio::test]
    async fn test_graphsage_forward() {
        let config = GnnConfig {
            architecture: GnnArchitecture::GraphSage,
            input_dim: 10,
            hidden_dims: vec![20],
            output_dim: 5,
            num_layers: 2,
            aggregation: Aggregation::Mean,
            ..Default::default()
        };
        let graphsage = GraphSageNetwork::new(config);
        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/a").unwrap(),
                NamedNode::new("http://example.org/rel").unwrap(),
                NamedNode::new("http://example.org/b").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/b").unwrap(),
                NamedNode::new("http://example.org/rel").unwrap(),
                NamedNode::new("http://example.org/c").unwrap(),
            ),
        ];
        let graph = RdfGraph::from_triples(&triples).unwrap();
        let features = Array2::ones((graph.num_nodes, 10));
        let output = graphsage.forward(&graph, &features).await.unwrap();
        assert_eq!(output.shape()[0], graph.num_nodes);
        assert_eq!(output.shape()[1], 5);
    }
    #[test]
    fn test_graphsage_layer_aggregation() {
        let layer = GraphSageLayer::new(10, 5, Aggregation::Mean);
        let node_features = Array1::ones(10);
        let neighbor1 = Array1::from_elem(10, 0.5);
        let neighbor2 = Array1::from_elem(10, 1.5);
        let neighbors = vec![neighbor1, neighbor2];
        let result = layer
            .aggregate_and_combine(&node_features, &neighbors)
            .unwrap();
        assert_eq!(result.len(), 5);
    }
    #[test]
    fn test_graphsage_aggregation_mean() {
        let layer = GraphSageLayer::new(5, 3, Aggregation::Mean);
        let features = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]),
        ];
        let aggregated = layer.aggregate(&features).unwrap();
        assert_eq!(aggregated.len(), 5);
        assert!((aggregated[0] - 1.5).abs() < 1e-5);
        assert!((aggregated[4] - 5.5).abs() < 1e-5);
    }
    #[test]
    fn test_graphsage_aggregation_max() {
        let layer = GraphSageLayer::new(5, 3, Aggregation::Max);
        let features = vec![
            Array1::from_vec(vec![1.0, 5.0, 3.0, 2.0, 4.0]),
            Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 1.0]),
        ];
        let aggregated = layer.aggregate(&features).unwrap();
        assert_eq!(aggregated.len(), 5);
        assert_eq!(aggregated[0], 2.0);
        assert_eq!(aggregated[1], 5.0);
        assert_eq!(aggregated[3], 5.0);
    }
    #[tokio::test]
    async fn test_graphsage_link_prediction() {
        let config = GnnConfig {
            architecture: GnnArchitecture::GraphSage,
            input_dim: 10,
            hidden_dims: vec![8],
            output_dim: 4,
            num_layers: 2,
            ..Default::default()
        };
        let graphsage = GraphSageNetwork::new(config);
        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/a").unwrap(),
                NamedNode::new("http://example.org/rel").unwrap(),
                NamedNode::new("http://example.org/b").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/b").unwrap(),
                NamedNode::new("http://example.org/rel").unwrap(),
                NamedNode::new("http://example.org/c").unwrap(),
            ),
        ];
        let graph = RdfGraph::from_triples(&triples).unwrap();
        let source_nodes = vec![0, 1];
        let target_nodes = vec![1, 2];
        let predictions = graphsage
            .predict_links(&graph, &source_nodes, &target_nodes)
            .await
            .unwrap();
        assert_eq!(predictions.len(), 2);
        assert!(predictions[0] >= 0.0 && predictions[0] <= 1.0);
        assert!(predictions[1] >= 0.0 && predictions[1] <= 1.0);
    }
    #[test]
    fn test_create_gnn_graphsage() {
        let config = GnnConfig {
            architecture: GnnArchitecture::GraphSage,
            ..Default::default()
        };
        let gnn = create_gnn(config).unwrap();
        assert!(gnn.get_parameters().is_ok());
    }
    #[test]
    fn test_gat_creation() {
        let config = GnnConfig {
            architecture: GnnArchitecture::GraphAttentionNetwork,
            input_dim: 10,
            hidden_dims: vec![20, 15],
            output_dim: 5,
            num_layers: 3,
            ..Default::default()
        };
        let gat = GraphAttentionNetwork::new(config.clone());
        assert_eq!(gat.layers.len(), 3);
        assert_eq!(gat.num_heads.len(), 3);
        assert!(!gat.trained);
    }
    #[tokio::test]
    async fn test_gat_forward() {
        let config = GnnConfig {
            architecture: GnnArchitecture::GraphAttentionNetwork,
            input_dim: 10,
            hidden_dims: vec![16],
            output_dim: 8,
            num_layers: 2,
            ..Default::default()
        };
        let gat = GraphAttentionNetwork::new(config);
        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/a").unwrap(),
                NamedNode::new("http://example.org/rel").unwrap(),
                NamedNode::new("http://example.org/b").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/b").unwrap(),
                NamedNode::new("http://example.org/rel").unwrap(),
                NamedNode::new("http://example.org/c").unwrap(),
            ),
        ];
        let graph = RdfGraph::from_triples(&triples).unwrap();
        let features = Array2::ones((graph.num_nodes, 10));
        let output = gat.forward(&graph, &features).await.unwrap();
        assert_eq!(output.shape()[0], graph.num_nodes);
        assert_eq!(output.shape()[1], 8);
    }
    #[test]
    fn test_gat_layer_creation() {
        let layer = GraphAttentionLayer::new(10, 8, 4);
        assert_eq!(layer.num_heads, 4);
        assert_eq!(layer.attention_weights.len(), 4);
        assert_eq!(layer.weight_matrices.len(), 4);
        assert_eq!(layer.input_dim, 10);
        assert_eq!(layer.output_dim, 8);
    }
    #[test]
    fn test_gat_layer_forward() {
        let layer = GraphAttentionLayer::new(5, 4, 2);
        let features = Array2::ones((3, 5));
        let adj = Array2::from_shape_vec((3, 3), vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .unwrap();
        let output = layer.forward(&features, &adj).unwrap();
        assert_eq!(output.shape(), &[3, 4]);
    }
    #[tokio::test]
    async fn test_gat_link_prediction() {
        let config = GnnConfig {
            architecture: GnnArchitecture::GraphAttentionNetwork,
            input_dim: 10,
            hidden_dims: vec![8],
            output_dim: 4,
            num_layers: 2,
            ..Default::default()
        };
        let gat = GraphAttentionNetwork::new(config);
        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/a").unwrap(),
                NamedNode::new("http://example.org/rel").unwrap(),
                NamedNode::new("http://example.org/b").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/b").unwrap(),
                NamedNode::new("http://example.org/rel").unwrap(),
                NamedNode::new("http://example.org/c").unwrap(),
            ),
        ];
        let graph = RdfGraph::from_triples(&triples).unwrap();
        let source_nodes = vec![0, 1];
        let target_nodes = vec![1, 2];
        let predictions = gat
            .predict_links(&graph, &source_nodes, &target_nodes)
            .await
            .unwrap();
        assert_eq!(predictions.len(), 2);
        assert!(predictions[0] >= 0.0 && predictions[0] <= 1.0);
        assert!(predictions[1] >= 0.0 && predictions[1] <= 1.0);
    }
    #[test]
    fn test_create_gnn_gat() {
        let config = GnnConfig {
            architecture: GnnArchitecture::GraphAttentionNetwork,
            ..Default::default()
        };
        let gnn = create_gnn(config).unwrap();
        assert!(gnn.get_parameters().is_ok());
    }
    #[tokio::test]
    async fn test_gat_attention_mechanism() {
        let config = GnnConfig {
            architecture: GnnArchitecture::GraphAttentionNetwork,
            input_dim: 5,
            hidden_dims: vec![4],
            output_dim: 3,
            num_layers: 2,
            ..Default::default()
        };
        let gat = GraphAttentionNetwork::new(config);
        let triples = vec![
            Triple::new(
                NamedNode::new("http://example.org/a").unwrap(),
                NamedNode::new("http://example.org/rel").unwrap(),
                NamedNode::new("http://example.org/b").unwrap(),
            ),
            Triple::new(
                NamedNode::new("http://example.org/a").unwrap(),
                NamedNode::new("http://example.org/rel").unwrap(),
                NamedNode::new("http://example.org/c").unwrap(),
            ),
        ];
        let graph = RdfGraph::from_triples(&triples).unwrap();
        let features = Array2::ones((graph.num_nodes, 5));
        let output = gat.forward(&graph, &features).await.unwrap();
        assert_eq!(output.shape()[0], graph.num_nodes);
        for val in output.iter() {
            assert!(val.is_finite());
        }
    }
    #[test]
    fn test_all_gnn_architectures() {
        let architectures = vec![
            GnnArchitecture::GraphConvolutionalNetwork,
            GnnArchitecture::GraphSage,
            GnnArchitecture::GraphAttentionNetwork,
        ];
        for arch in architectures {
            let config = GnnConfig {
                architecture: arch,
                input_dim: 10,
                hidden_dims: vec![8],
                output_dim: 4,
                num_layers: 2,
                ..Default::default()
            };
            let gnn = create_gnn(config);
            assert!(gnn.is_ok(), "Failed to create GNN architecture");
        }
    }
}
