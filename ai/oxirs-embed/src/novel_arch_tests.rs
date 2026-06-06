//! Tests for the novel architectures module.
//!
//! Covers the default configuration trees, parameter constructors, model
//! creation, Poincaré distance computations, the quantum-inspired forward
//! pass, the asynchronous training loop, the 2-D softmax helper, GPU-free
//! architecture initialisation and end-to-end text encoding.

#![cfg(test)]

use crate::novel_arch_types::*;
use crate::{EmbeddingModel, ModelConfig, NamedNode, Triple};
use scirs2_core::ndarray_ext::{Array1, Array2};

#[test]
fn test_novel_architecture_config_default() {
    let config = NovelArchitectureConfig::default();
    assert_eq!(config.base_config.dimensions, 100);
    assert!(matches!(
        config.architecture,
        ArchitectureType::GraphTransformer
    ));
}

#[test]
fn test_graph_transformer_params() {
    let params = GraphTransformerParams::default();
    assert_eq!(params.num_heads, 8);
    assert_eq!(params.num_layers, 6);
    assert_eq!(params.attention_dim, 512);
}

#[test]
fn test_hyperbolic_params() {
    let params = HyperbolicParams::default();
    assert_eq!(params.curvature, -1.0);
    assert_eq!(params.manifold_dim, 128);
    assert!(matches!(params.manifold, HyperbolicManifold::Poincare));
}

#[test]
fn test_neural_ode_params() {
    let params = NeuralODEParams::default();
    assert_eq!(params.time_steps, 100);
    assert_eq!(params.tolerance, 1e-6);
    assert!(matches!(params.solver_type, ODESolverType::DormandPrince));
}

#[test]
fn test_quantum_params() {
    let params = QuantumParams::default();
    assert_eq!(params.num_qubits, 10);
    assert!(matches!(params.gate_set, QuantumGateSet::Universal));
    assert!(params.hybrid_layers);
}

#[test]
fn test_novel_architecture_model_creation() {
    let config = NovelArchitectureConfig::default();
    let model = NovelArchitectureModel::new(config);

    assert_eq!(model.entities.len(), 0);
    assert_eq!(model.relations.len(), 0);
    assert!(!model.is_trained);
}

#[test]
fn test_poincare_distance() {
    let config = NovelArchitectureConfig {
        architecture: ArchitectureType::HyperbolicEmbedding,
        ..Default::default()
    };
    let model = NovelArchitectureModel::new(config);

    let x = Array1::from_vec(vec![0.1, 0.2]);
    let y = Array1::from_vec(vec![0.3, 0.4]);

    let distance = model.poincare_distance(&x, &y);
    assert!(distance > 0.0);
    assert!(distance.is_finite());
}

#[test]
fn test_quantum_forward() {
    // Configure quantum system with 3 qubits to match input dimension
    let config = NovelArchitectureConfig {
        architecture: ArchitectureType::QuantumInspired,
        base_config: ModelConfig {
            dimensions: 3, // Match the input dimension
            ..Default::default()
        },
        architecture_params: ArchitectureParams {
            quantum_params: QuantumParams {
                num_qubits: 3, // Set to match input dimension
                ..Default::default()
            },
            ..Default::default()
        },
        ..Default::default()
    };
    let mut model = NovelArchitectureModel::new(config);

    // Initialize quantum state
    model.initialize_architecture().expect("should succeed");

    let input = Array1::from_vec(vec![0.5, 0.3, 0.8]);
    let output = model.quantum_forward(&input).expect("should succeed");

    assert_eq!(output.len(), input.len());

    // Check values are in expected range with floating-point tolerance
    const TOLERANCE: f64 = 1e-10;
    assert!(output
        .iter()
        .all(|&x| (-1.0 - TOLERANCE..=1.0 + TOLERANCE).contains(&x)));
}

#[tokio::test]
async fn test_novel_architecture_training() {
    let config = NovelArchitectureConfig::default();
    let mut model = NovelArchitectureModel::new(config);

    // Add some test data
    let triple = Triple::new(
        NamedNode::new("http://example.org/alice").expect("should succeed"),
        NamedNode::new("http://example.org/knows").expect("should succeed"),
        NamedNode::new("http://example.org/bob").expect("should succeed"),
    );
    model.add_triple(triple).expect("should succeed");

    let stats = model.train(Some(5)).await.expect("should succeed");
    assert_eq!(stats.epochs_completed, 5);
    assert!(model.is_trained());
}

#[test]
fn test_softmax_2d() {
    let config = NovelArchitectureConfig::default();
    let model = NovelArchitectureModel::new(config);

    let input =
        Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("should succeed");
    let output = model.softmax_2d(&input);

    // Check that rows sum to 1
    for row in output.rows() {
        let sum: f64 = row.sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}

#[test]
fn test_architecture_initialization() {
    let mut model = NovelArchitectureModel::new(NovelArchitectureConfig {
        architecture: ArchitectureType::GraphTransformer,
        ..Default::default()
    });

    // Add entity first
    let triple = Triple::new(
        NamedNode::new("http://example.org/alice").expect("should succeed"),
        NamedNode::new("http://example.org/knows").expect("should succeed"),
        NamedNode::new("http://example.org/bob").expect("should succeed"),
    );
    model.add_triple(triple).expect("should succeed");

    model.initialize_architecture().expect("should succeed");
    assert!(model.architecture_state.transformer_state.is_some());
}

#[tokio::test]
async fn test_novel_architecture_encoding() {
    let config = NovelArchitectureConfig {
        architecture: ArchitectureType::QuantumInspired,
        base_config: ModelConfig {
            dimensions: 16, // Use smaller dimensions for quantum operations
            ..Default::default()
        },
        ..Default::default()
    };
    let mut model = NovelArchitectureModel::new(config);
    model.initialize_architecture().expect("should succeed");

    let texts = vec!["hello".to_string(), "world".to_string()];
    let embeddings = model.encode(&texts).await.expect("should succeed");

    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), model.config.base_config.dimensions);
}
