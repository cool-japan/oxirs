//! Hybrid GNN+LLM KGQA demo.
//!
//! Demonstrates a complete knowledge-graph question-answering pipeline:
//!
//! 1. Build a toy knowledge graph.
//! 2. Create a frozen GraphSAGE encoder (block 5).
//! 3. Create a learnable SoftPromptProjector (block 6, phase b).
//! 4. Assemble the HybridLlmHead with a local deterministic provider.
//! 5. Train the projector for a few epochs.
//! 6. Answer a natural-language question.

use std::sync::Arc;

use oxirs_graphrag::gnn_encoder::{GraphSageConfig, GraphSageEncoder, KgGraph};
use oxirs_graphrag::hybrid::{HybridLlmHead, KgqaExample, LocalProvider, SoftPromptProjector};
use scirs2_core::ndarray_ext::Array2;

#[tokio::main]
async fn main() {
    // ── Step 1: construct a toy KG ────────────────────────────────────────────
    let kg = KgGraph {
        num_nodes: 6,
        edges: vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)],
        node_features: Array2::zeros((6, 16)),
    };

    println!(
        "Knowledge graph: {} nodes, {} edges",
        kg.num_nodes,
        kg.edges.len()
    );

    // ── Step 2: build frozen GNN encoder ──────────────────────────────────────
    let config = GraphSageConfig {
        input_dim: 16,
        hidden_dim: 16,
        output_dim: 16,
        num_layers: 2,
        dropout: 0.0,
        k_neighbors: 3,
        learning_rate: 0.0, // frozen — no gradient updates on the encoder
    };
    let encoder =
        Arc::new(GraphSageEncoder::new_with_seed(&config, 42).expect("construct encoder"));

    let initial_embeddings = encoder.encode(&kg).expect("initial encode");
    println!(
        "Initial GNN embeddings shape: [{} x {}]",
        initial_embeddings.embeddings.nrows(),
        initial_embeddings.embeddings.ncols()
    );

    // ── Step 3: build learnable soft-prompt projector ─────────────────────────
    let projector = SoftPromptProjector::new(16, 32, 7);
    println!(
        "Projector: {} → {} dim",
        projector.gnn_dim(),
        projector.prompt_dim()
    );

    // ── Step 4: assemble the hybrid head ─────────────────────────────────────
    let provider = LocalProvider::with_response("Entities are connected via shared predicates.");
    let mut head =
        HybridLlmHead::new(Arc::clone(&encoder), projector, provider).with_learning_rate(0.05);

    // ── Step 5: train the projector ───────────────────────────────────────────
    let training_examples = vec![
        KgqaExample {
            question: "What is the role of entity 0?".to_string(),
            answer: "Hub node connecting entities 1 and 3.".to_string(),
            entity_ids: vec![0, 1, 3],
        },
        KgqaExample {
            question: "How are entities 2 and 4 related?".to_string(),
            answer: "They are connected via entity 3.".to_string(),
            entity_ids: vec![2, 3, 4],
        },
        KgqaExample {
            question: "What closes the ring?".to_string(),
            answer: "Edge from entity 5 back to entity 0.".to_string(),
            entity_ids: vec![5, 0],
        },
    ];

    let history = head
        .train_projector(&kg, &training_examples, 30)
        .expect("projector training");

    println!(
        "Training complete — initial loss: {:.6}, final loss: {:.6}",
        history.epoch_losses.first().copied().unwrap_or(0.0),
        history.epoch_losses.last().copied().unwrap_or(0.0)
    );

    // Verify encoder is still frozen.
    let after_embeddings = encoder.encode(&kg).expect("post-training encode");
    let mut max_drift = 0.0_f64;
    for i in 0..kg.num_nodes {
        for j in 0..16 {
            let drift =
                (initial_embeddings.embeddings[[i, j]] - after_embeddings.embeddings[[i, j]]).abs();
            if drift > max_drift {
                max_drift = drift;
            }
        }
    }
    println!("Max encoder weight drift (should be 0): {max_drift:.2e}");

    // ── Step 6: answer a question ─────────────────────────────────────────────
    let question = "What entities are most central in the knowledge graph?";
    let answer = head.answer(question, &kg).await.expect("answer");
    println!("\nQuestion: {question}");
    println!("Answer:   {answer}");
}
