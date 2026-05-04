//! Demonstrates the joint GNN+LLM training scaffold (phase c).
//!
//! Trains a [`JointTrainer`] with the [`Schedule::Curriculum`] schedule on a
//! synthetic 4-entity knowledge graph.  The first 10 epochs warm up only the
//! projector (GNN frozen); the remaining 40 epochs train both jointly.

use oxirs_graphrag::gnn_encoder::{GraphSageConfig, GraphSageEncoder, KgGraph};
use oxirs_graphrag::hybrid::{
    JointTrainer, KgqaExample, LocalProvider, Schedule, SoftPromptProjector,
};
use scirs2_core::ndarray_ext::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Build a tiny 4-node knowledge graph ──────────────────────────────────
    let kg = KgGraph {
        num_nodes: 4,
        edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        node_features: Array2::zeros((4, 8)),
    };

    // ── Encoder config — learning_rate=0 keeps GNN weights stable in the demo
    let config = GraphSageConfig {
        input_dim: 8,
        hidden_dim: 8,
        output_dim: 8,
        num_layers: 1,
        dropout: 0.0,
        k_neighbors: 2,
        learning_rate: 0.0,
    };
    let encoder = GraphSageEncoder::new_with_seed(&config, 1)?;

    // ── Projector: 8-dim GNN → 8-dim prompt space ────────────────────────────
    let projector = SoftPromptProjector::new(8, 8, 42);

    // ── Four KGQA examples (one per entity) ──────────────────────────────────
    let examples: Vec<KgqaExample> = (0..4_usize)
        .map(|i| KgqaExample {
            question: format!("Who is entity {i}?"),
            answer: format!("Entity {i}"),
            entity_ids: vec![i],
        })
        .collect();

    // ── Joint trainer with curriculum schedule ───────────────────────────────
    let mut trainer = JointTrainer::new(
        encoder,
        projector,
        LocalProvider::new(),
        Schedule::Curriculum {
            warmup_epochs: 10,
            joint_epochs: 40,
        },
    )
    .with_learning_rates(0.0, 0.05);

    // ── Run 50 epochs ────────────────────────────────────────────────────────
    let history = trainer.train(&kg, &examples, 50)?;

    println!("Initial loss: {:.6}", history.epochs[0].loss);
    println!("Final loss:   {:.6}", history.epochs[49].loss);
    println!("Training complete over {} epochs.", history.epochs.len());

    // Print a brief schedule overview.
    for (i, m) in history.epochs.iter().enumerate() {
        if !(3..47).contains(&i) {
            println!(
                "  epoch {:>2}  loss={:.6}  gnn_frozen={}  proj_frozen={}",
                m.epoch, m.loss, m.gnn_frozen, m.projector_frozen
            );
        } else if i == 3 {
            println!("  … (epochs 3–46 omitted) …");
        }
    }

    Ok(())
}
