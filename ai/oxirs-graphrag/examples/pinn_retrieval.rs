//! Demonstration of physics-informed entity scoring via the neuro-symbolic module.
//!
//! Builds a 4-node toy knowledge graph representing thermal components,
//! scores entities with a PINN scorer (damping = 0.4), and prints ranked results.
//!
//! Run with:
//! ```not_rust
//! cargo run --example pinn_retrieval -p oxirs-graphrag
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use scirs2_core::ndarray_ext::{Array1, Array2};

use oxirs_graphrag::gnn_encoder::{GraphSageConfig, GraphSageEncoder, KgGraph};
use oxirs_graphrag::neuro_symbolic::{KgEntity, PhysicsContext, PhysicsDomain, PinnEntityScorer};

fn main() {
    // ── 1. Build a 4-node KG representing thermal components ─────────────────

    // Nodes: steel_plate (0), copper_rod (1), ceramic_tile (2), unknown_part (3)
    // Edges form a ring: 0→1→2→3→0
    let kg = KgGraph {
        num_nodes: 4,
        edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        // 4-dimensional feature vectors (all zeros for this demo)
        node_features: Array2::zeros((4, 4)),
    };

    // ── 2. Construct the GraphSAGE encoder ───────────────────────────────────

    let config = GraphSageConfig {
        input_dim: 4,
        hidden_dim: 8,
        output_dim: 8,
        num_layers: 2,
        dropout: 0.0,
        k_neighbors: 3,
        learning_rate: 0.0,
    };
    let encoder = Arc::new(GraphSageEncoder::new_with_seed(&config, 42).expect("create encoder"));

    // ── 3. Set up the thermal-diffusion physics context ──────────────────────

    // Steel: α ≈ 1.2e-5 m²/s (realistic for carbon steel)
    let ctx = PhysicsContext::new(PhysicsDomain::ThermalDiffusion {
        thermal_diffusivity: 1.2e-5,
    });

    // ── 4. Create PINN scorer with damping = 0.4 (40% physics, 60% neural) ──

    let scorer = PinnEntityScorer::new(Arc::clone(&encoder), ctx, 0.4);

    // ── 5. Define entities with named thermal properties ─────────────────────

    // For steel_plate: Fo = α·t/L² = 1.2e-5 × 0.5 / 0.1² ≈ 0.6 → score ≈ high
    let mut steel_props = HashMap::new();
    steel_props.insert("time_s".to_string(), 0.5);
    steel_props.insert("length_m".to_string(), 0.1);

    // For copper_rod: Fo ≈ 1.2e-5 × 1.0 / 0.05² = 4.8 → |log10(4.8)|/3 ≈ 0.23 → moderate
    let mut copper_props = HashMap::new();
    copper_props.insert("time_s".to_string(), 1.0);
    copper_props.insert("length_m".to_string(), 0.05);

    // For ceramic_tile: very small Fo due to long length → low score
    let mut ceramic_props = HashMap::new();
    ceramic_props.insert("time_s".to_string(), 0.01);
    ceramic_props.insert("length_m".to_string(), 1.0);

    // unknown_part: no thermal data → neutral score 0.5
    let unknown_props = HashMap::new();

    let entities = vec![
        KgEntity {
            id: "steel_plate".to_string(),
            embedding_idx: 0,
            properties: steel_props,
        },
        KgEntity {
            id: "copper_rod".to_string(),
            embedding_idx: 1,
            properties: copper_props,
        },
        KgEntity {
            id: "ceramic_tile".to_string(),
            embedding_idx: 2,
            properties: ceramic_props,
        },
        KgEntity {
            id: "unknown_part".to_string(),
            embedding_idx: 3,
            properties: unknown_props,
        },
    ];

    // ── 6. Construct a query embedding ───────────────────────────────────────

    // Simulate a query embedding (all 0.5 components, normalised direction)
    let query_embedding = Array1::from_vec(vec![0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]);

    // ── 7. Run rank() ─────────────────────────────────────────────────────────

    let ranked = scorer
        .rank(&kg, &entities, &query_embedding)
        .expect("rank entities");

    // ── 8. Print results ─────────────────────────────────────────────────────

    println!("PINN Entity Ranking (damping=0.4)");
    println!("───────────────────────────────────────────────────────────────");
    println!(
        "{:<20} {:>8} {:>8} {:>10}  Physics reason",
        "Entity", "Neural", "Physics", "Combined"
    );
    println!("───────────────────────────────────────────────────────────────");

    for (rank, se) in ranked.iter().enumerate() {
        println!(
            "#{:<2} {:<18} {:>8.4} {:>8.4} {:>10.4}  {}",
            rank + 1,
            se.entity_id,
            se.neural_score,
            se.physics_score,
            se.combined_score,
            se.plausibility.reason
        );
    }
    println!("───────────────────────────────────────────────────────────────");

    // Verify at least some ordering signal.
    assert!(ranked.len() == 4, "should have 4 ranked entities");
    let first = &ranked[0];
    let last = ranked.last().expect("non-empty");
    println!(
        "\nBest entity:  {} (combined={:.4})",
        first.entity_id, first.combined_score
    );
    println!(
        "Worst entity: {} (combined={:.4})",
        last.entity_id, last.combined_score
    );
}
