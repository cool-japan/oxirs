//! Integration tests for the hybrid GNN+LLM head (block 6, phase b).

use std::sync::Arc;

use oxirs_graphrag::gnn_encoder::{GraphSageConfig, GraphSageEncoder, KgGraph};
use oxirs_graphrag::hybrid::{
    HybridLlmHead, KgqaExample, LlmProvider, LocalProvider, SoftPromptProjector,
};
use scirs2_core::ndarray_ext::Array2;

fn toy_kg() -> KgGraph {
    KgGraph {
        num_nodes: 4,
        edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        node_features: Array2::zeros((4, 8)),
    }
}

fn toy_config() -> GraphSageConfig {
    GraphSageConfig {
        input_dim: 8,
        hidden_dim: 8,
        output_dim: 8,
        num_layers: 1,
        dropout: 0.0,
        k_neighbors: 2,
        learning_rate: 0.0,
    }
}

fn make_head() -> HybridLlmHead<LocalProvider> {
    let encoder =
        Arc::new(GraphSageEncoder::new_with_seed(&toy_config(), 1).expect("construct encoder"));
    let projector = SoftPromptProjector::new(8, 8, 42);
    HybridLlmHead::new(encoder, projector, LocalProvider::new())
}

#[test]
fn test_projector_forward_shape() {
    let mut projector = SoftPromptProjector::new(8, 16, 0);
    let input = Array2::zeros((4, 8));
    let out = projector.forward(&input);
    assert_eq!(out.nrows(), 4);
    assert_eq!(out.ncols(), 16);
}

#[test]
fn test_frozen_encoder_weights_unchanged_after_projector_step() {
    let config = toy_config();
    let encoder = Arc::new(GraphSageEncoder::new_with_seed(&config, 1).expect("construct encoder"));
    let projector = SoftPromptProjector::new(8, 8, 42);
    let mut head = HybridLlmHead::new(Arc::clone(&encoder), projector, LocalProvider::new());

    // Capture encoder state before training.
    let before = encoder.encode(&toy_kg()).expect("encode before");

    let examples = vec![KgqaExample {
        question: "What is entity 0?".to_string(),
        answer: "Entity A".to_string(),
        entity_ids: vec![0, 1],
    }];
    head.train_projector(&toy_kg(), &examples, 5)
        .expect("train projector");

    // Encoder state should be unchanged (frozen).
    let after = encoder.encode(&toy_kg()).expect("encode after");
    for i in 0..4 {
        for j in 0..8 {
            assert!(
                (before.embeddings[[i, j]] - after.embeddings[[i, j]]).abs() < 1e-12,
                "encoder weights changed at [{i},{j}]"
            );
        }
    }
}

#[tokio::test]
async fn test_local_provider_answers_kgqa() {
    let mut head = make_head();
    let kg = toy_kg();
    let answer = head
        .answer("Who is entity 1?", &kg)
        .await
        .expect("should answer");
    assert!(!answer.is_empty(), "answer should not be empty");
}

#[test]
fn test_projector_loss_decreases() {
    let config = toy_config();
    let encoder = Arc::new(GraphSageEncoder::new_with_seed(&config, 2).expect("construct encoder"));
    let projector = SoftPromptProjector::new(8, 8, 99);
    let mut head =
        HybridLlmHead::new(encoder, projector, LocalProvider::new()).with_learning_rate(0.1);
    let examples = vec![
        KgqaExample {
            question: "q0".to_string(),
            answer: "a0".to_string(),
            entity_ids: vec![0],
        },
        KgqaExample {
            question: "q1".to_string(),
            answer: "a1".to_string(),
            entity_ids: vec![1],
        },
    ];
    let history = head
        .train_projector(&toy_kg(), &examples, 20)
        .expect("train");
    assert_eq!(history.epoch_losses.len(), 20);
    let first = history.epoch_losses[0];
    let last = *history.epoch_losses.last().expect("non-empty");
    assert!(last <= first, "loss should not increase: {first} → {last}");
}

#[test]
fn test_capabilities_surface() {
    let provider = LocalProvider::new();
    let caps = provider.capabilities();
    assert_eq!(caps.max_context_tokens, 4096);
    assert!(!caps.supports_streaming);
}
