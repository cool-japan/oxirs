//! Integration tests for the neuro-symbolic PINN entity scoring module.
//!
//! Covers physics-context scoring, PINN scorer mechanics, and the
//! neuro-symbolic retriever end-to-end.

use std::collections::HashMap;
use std::sync::Arc;

use scirs2_core::ndarray_ext::{Array1, Array2};

use oxirs_graphrag::gnn_encoder::{EntityEmbeddings, GraphSageConfig, GraphSageEncoder, KgGraph};
use oxirs_graphrag::hybrid::provider::LocalProvider;
use oxirs_graphrag::neuro_symbolic::physics_context::PhysicsContext as Ctx;
use oxirs_graphrag::neuro_symbolic::{
    FlowRegime, KgEntity, NeuroSymbolicRetriever, PhysicsContext, PhysicsDomain, PinnEntityScorer,
    PinnScorerError, PlausibilityScore,
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn props(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
    pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

fn thermal_ctx(alpha: f64) -> PhysicsContext {
    Ctx::new(PhysicsDomain::ThermalDiffusion {
        thermal_diffusivity: alpha,
    })
}

fn fluid_ctx(regime: FlowRegime) -> PhysicsContext {
    Ctx::new(PhysicsDomain::FluidFlow {
        kinematic_viscosity: 1e-6,
        expected_regime: regime,
    })
}

fn structural_ctx(e: f64, yield_pa: f64) -> PhysicsContext {
    Ctx::new(PhysicsDomain::StructuralMechanics {
        youngs_modulus_pa: e,
        yield_stress_pa: yield_pa,
    })
}

fn em_ctx() -> PhysicsContext {
    Ctx::new(PhysicsDomain::Electromagnetic)
}

/// Build entity embeddings manually so tests are deterministic.
fn manual_embeddings(rows: Vec<Vec<f64>>) -> EntityEmbeddings {
    let n = rows.len();
    let dim = rows.first().map(Vec::len).unwrap_or(0);
    let mut mat = Array2::zeros((n, dim));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            mat[[i, j]] = v;
        }
    }
    EntityEmbeddings {
        embeddings: mat,
        node_ids: (0..n).map(|i| i.to_string()).collect(),
    }
}

fn toy_encoder_4d() -> Arc<GraphSageEncoder> {
    let config = GraphSageConfig {
        input_dim: 4,
        hidden_dim: 4,
        output_dim: 4,
        num_layers: 2,
        dropout: 0.0,
        k_neighbors: 2,
        learning_rate: 0.0,
    };
    Arc::new(GraphSageEncoder::new_with_seed(&config, 1).expect("encoder"))
}

fn toy_kg_4() -> KgGraph {
    KgGraph {
        num_nodes: 4,
        edges: vec![(0, 1), (1, 2), (2, 3), (3, 0)],
        node_features: Array2::zeros((4, 4)),
    }
}

// ─── Physics context tests ────────────────────────────────────────────────────

/// Fourier number Fo = α·t/L² = 1e-5 × 1.0 / (√1e-5)² = 1.0 → log10(1)=0 → score≈1
#[test]
fn test_thermal_fourier_in_range_scores_high() {
    let ctx = thermal_ctx(1e-5);
    let p: PlausibilityScore =
        ctx.plausibility_score(&props(&[("time_s", 1.0), ("length_m", (1e-5_f64).sqrt())]));
    assert!(
        p.score > 0.95,
        "expected high score for Fo≈1, got {}",
        p.score
    );
    assert!(p.dimensionless_param.is_some());
}

/// Fo = 1e-5 × 1e-6 / 1² = 1e-11 → log10 = -11 → |−11|/3 >> 1 → score = 0
#[test]
fn test_thermal_fourier_extreme_scores_low() {
    let ctx = thermal_ctx(1e-5);
    let p = ctx.plausibility_score(&props(&[("time_s", 1e-6), ("length_m", 1.0)]));
    assert!(
        p.score < 0.1,
        "expected low score for extreme Fo, got {}",
        p.score
    );
}

/// Re = 0.001 × 0.001 / 1e-6 = 1 → laminar → score = 1.0
#[test]
fn test_fluid_laminar_correct_regime_scores_high() {
    let ctx = fluid_ctx(FlowRegime::Laminar);
    // Re = v*L/nu = 0.001 * 0.1 / 1e-6 = 100 — well inside laminar
    let p = ctx.plausibility_score(&props(&[("velocity_ms", 0.001), ("length_m", 0.1)]));
    assert!(
        p.score > 0.9,
        "expected high laminar score, got {}",
        p.score
    );
}

/// Expected laminar but Re > 10 000 → score ≈ 0
#[test]
fn test_fluid_laminar_wrong_regime_scores_low() {
    let ctx = fluid_ctx(FlowRegime::Laminar);
    // Re = 10.0 * 1.0 / 1e-6 = 10_000_000 — deep turbulent
    let p = ctx.plausibility_score(&props(&[("velocity_ms", 10.0), ("length_m", 1.0)]));
    assert!(
        p.score < 0.05,
        "expected low score for wrong regime, got {}",
        p.score
    );
}

/// Strain exactly matches Hooke's law, stress below yield → score = 1.0
#[test]
fn test_structural_hookes_consistent_scores_high() {
    let ctx = structural_ctx(2e11, 2.5e8);
    let sigma = 1e8_f64; // below yield
    let eps = sigma / 2e11; // Hooke-consistent strain
    let p = ctx.plausibility_score(&props(&[("stress_pa", sigma), ("strain", eps)]));
    assert!(
        p.score > 0.95,
        "expected high structural score, got {}",
        p.score
    );
}

/// Stress exceeds yield stress → 0.1× penalty applied
#[test]
fn test_structural_yield_exceeded_scores_low() {
    let ctx = structural_ctx(2e11, 2.5e8);
    let sigma = 5e8_f64; // above yield
    let eps = sigma / 2e11;
    let p = ctx.plausibility_score(&props(&[("stress_pa", sigma), ("strain", eps)]));
    assert!(
        p.score < 0.2,
        "expected low score due to yield, got {}",
        p.score
    );
}

/// Missing required properties → neutral score 0.5
#[test]
fn test_missing_props_neutral_score() {
    let thermal = thermal_ctx(1e-5);
    let pt = thermal.plausibility_score(&props(&[]));
    assert!(
        (pt.score - 0.5).abs() < 1e-10,
        "thermal missing → 0.5, got {}",
        pt.score
    );

    let fluid = fluid_ctx(FlowRegime::Laminar);
    let pf = fluid.plausibility_score(&props(&[]));
    assert!(
        (pf.score - 0.5).abs() < 1e-10,
        "fluid missing → 0.5, got {}",
        pf.score
    );

    let em = em_ctx();
    let pe = em.plausibility_score(&props(&[]));
    assert!(
        (pe.score - 0.5).abs() < 1e-10,
        "em missing → 0.5, got {}",
        pe.score
    );

    let st = structural_ctx(2e11, 2.5e8);
    let ps = st.plausibility_score(&props(&[]));
    assert!(
        (ps.score - 0.5).abs() < 1e-10,
        "structural missing → 0.5, got {}",
        ps.score
    );
}

/// V = I·R exactly → score = 1.0
#[test]
fn test_electromagnetic_ohms_law_consistent() {
    let ctx = em_ctx();
    let p = ctx.plausibility_score(&props(&[
        ("voltage_v", 24.0),
        ("current_a", 4.0),
        ("resistance_ohm", 6.0),
    ]));
    assert!(
        p.score > 0.99,
        "expected ~1.0 for exact Ohm, got {}",
        p.score
    );
}

// ─── Scorer tests ─────────────────────────────────────────────────────────────

/// cosine_similarity_normalized of identical non-zero vectors is 1.0
#[test]
fn test_cosine_normalized_range() {
    // Test via the scorer's output: entity embedding == query embedding → neural ≈ 1.0
    let encoder = toy_encoder_4d();
    let ctx = thermal_ctx(1e-5);
    let scorer = PinnEntityScorer::new(Arc::clone(&encoder), ctx, 0.0);

    let embs = manual_embeddings(vec![vec![1.0, 0.0, 0.0, 0.0]]);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let entities = vec![KgEntity {
        id: "e0".into(),
        embedding_idx: 0,
        properties: HashMap::new(),
    }];
    let results = scorer
        .score_entities(&embs, &entities, &query)
        .expect("score");
    let se = &results[0];
    assert!(
        (0.0..=1.0).contains(&se.neural_score),
        "neural_score out of range: {}",
        se.neural_score
    );
    assert!(
        (0.0..=1.0).contains(&se.physics_score),
        "physics_score out of range: {}",
        se.physics_score
    );
    assert!(
        (0.0..=1.0).contains(&se.combined_score),
        "combined_score out of range: {}",
        se.combined_score
    );
    assert!(
        se.neural_score > 0.99,
        "identical embedding/query → neural ≈ 1.0, got {}",
        se.neural_score
    );
}

/// damping=0 → combined_score == neural_score
#[test]
fn test_damping_zero_pure_neural() {
    let scorer = PinnEntityScorer::new(toy_encoder_4d(), thermal_ctx(1e-5), 0.0);
    let embs = manual_embeddings(vec![vec![1.0, 0.0, 0.0, 0.0]]);
    let query = Array1::from_vec(vec![0.5, 0.5, 0.0, 0.0]);
    let entities = vec![KgEntity {
        id: "e0".into(),
        embedding_idx: 0,
        properties: HashMap::new(),
    }];
    let r = scorer
        .score_entities(&embs, &entities, &query)
        .expect("score");
    assert!(
        (r[0].combined_score - r[0].neural_score).abs() < 1e-12,
        "damping=0 → combined==neural: combined={}, neural={}",
        r[0].combined_score,
        r[0].neural_score
    );
}

/// damping=1 → combined_score == physics_score
#[test]
fn test_damping_one_pure_physics() {
    let scorer = PinnEntityScorer::new(toy_encoder_4d(), thermal_ctx(1e-5), 1.0);
    let embs = manual_embeddings(vec![vec![1.0, 0.0, 0.0, 0.0]]);
    let query = Array1::from_vec(vec![0.5, 0.5, 0.0, 0.0]);
    let mut p = HashMap::new();
    p.insert("time_s".to_string(), 1.0);
    p.insert("length_m".to_string(), (1e-5_f64).sqrt());
    let entities = vec![KgEntity {
        id: "e0".into(),
        embedding_idx: 0,
        properties: p,
    }];
    let r = scorer
        .score_entities(&embs, &entities, &query)
        .expect("score");
    assert!(
        (r[0].combined_score - r[0].physics_score).abs() < 1e-12,
        "damping=1 → combined==physics: combined={}, physics={}",
        r[0].combined_score,
        r[0].physics_score
    );
}

/// rank() returns entities in descending combined_score order
#[test]
fn test_rank_descending_order() {
    let scorer = PinnEntityScorer::new(toy_encoder_4d(), thermal_ctx(1e-5), 0.3);
    let kg = toy_kg_4();
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let entities: Vec<KgEntity> = (0..4)
        .map(|i| KgEntity {
            id: format!("e{i}"),
            embedding_idx: i,
            properties: HashMap::new(),
        })
        .collect();
    let ranked = scorer.rank(&kg, &entities, &query).expect("rank");
    assert_eq!(ranked.len(), 4);
    for w in ranked.windows(2) {
        assert!(
            w[0].combined_score >= w[1].combined_score,
            "rank not descending: {} then {}",
            w[0].combined_score,
            w[1].combined_score
        );
    }
}

/// An entity with good physics properties should be promoted when damping > 0
#[test]
fn test_high_physics_entity_promoted() {
    // Use damping=1.0 so physics dominates entirely
    let scorer = PinnEntityScorer::new(toy_encoder_4d(), thermal_ctx(1e-5), 1.0);

    // Two entities with identical embeddings; e0 has ideal physics, e1 has none
    let embs = manual_embeddings(vec![vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]]);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);

    let mut good_props = HashMap::new();
    good_props.insert("time_s".to_string(), 1.0);
    good_props.insert("length_m".to_string(), (1e-5_f64).sqrt()); // Fo ≈ 1 → score ≈ 1

    let entities = vec![
        KgEntity {
            id: "good_physics".into(),
            embedding_idx: 0,
            properties: good_props,
        },
        KgEntity {
            id: "no_props".into(),
            embedding_idx: 1,
            properties: HashMap::new(), // missing → 0.5
        },
    ];

    let r = scorer
        .score_entities(&embs, &entities, &query)
        .expect("score");
    let good = r
        .iter()
        .find(|s| s.entity_id == "good_physics")
        .expect("good");
    let neutral = r
        .iter()
        .find(|s| s.entity_id == "no_props")
        .expect("neutral");

    assert!(
        good.combined_score > neutral.combined_score,
        "entity with ideal Fourier physics should score higher: good={} neutral={}",
        good.combined_score,
        neutral.combined_score
    );
}

/// Toy KG with 4 entities, varied physics properties, verify score structure
#[test]
fn test_toy_kg_four_entities() {
    let scorer = PinnEntityScorer::new(toy_encoder_4d(), thermal_ctx(1e-5), 0.5);
    let kg = toy_kg_4();
    let query = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);

    // Four entities with varying thermal properties
    let entities: Vec<KgEntity> = vec![
        // Fo ≈ 1 → physics ≈ 1
        KgEntity {
            id: "steel_plate".into(),
            embedding_idx: 0,
            properties: props(&[("time_s", 1.0), ("length_m", (1e-5_f64).sqrt())]),
        },
        // Fo very small → physics low
        KgEntity {
            id: "fast_pulse".into(),
            embedding_idx: 1,
            properties: props(&[("time_s", 1e-10), ("length_m", 1.0)]),
        },
        // Missing props → physics = 0.5 (neutral)
        KgEntity {
            id: "unknown_material".into(),
            embedding_idx: 2,
            properties: HashMap::new(),
        },
        // Fo ≈ 100 → |log10(100)|/3 = 2/3 → score ≈ 0.33
        KgEntity {
            id: "medium_fo".into(),
            embedding_idx: 3,
            properties: props(&[("time_s", 100.0), ("length_m", (1e-5_f64).sqrt())]),
        },
    ];

    let results = scorer
        .encode_and_score(&kg, &entities, &query)
        .expect("score");
    assert_eq!(results.len(), 4);

    for r in &results {
        assert!((0.0..=1.0).contains(&r.neural_score), "neural out of [0,1]");
        assert!(
            (0.0..=1.0).contains(&r.physics_score),
            "physics out of [0,1]"
        );
        assert!(
            (0.0..=1.0).contains(&r.combined_score),
            "combined out of [0,1]"
        );
    }

    // steel_plate (best physics) should have highest physics_score
    let steel = results
        .iter()
        .find(|r| r.entity_id == "steel_plate")
        .expect("steel");
    let fast = results
        .iter()
        .find(|r| r.entity_id == "fast_pulse")
        .expect("fast");
    assert!(
        steel.physics_score > fast.physics_score,
        "steel_plate physics {} should beat fast_pulse physics {}",
        steel.physics_score,
        fast.physics_score
    );
}

// ─── Retriever integration test ───────────────────────────────────────────────

#[tokio::test]
async fn test_retriever_answer_end_to_end() {
    let encoder = toy_encoder_4d();
    let ctx = thermal_ctx(1e-5);
    let scorer = PinnEntityScorer::new(Arc::clone(&encoder), ctx, 0.3);
    let mut retriever =
        NeuroSymbolicRetriever::from_parts(scorer, encoder, 4, 4, LocalProvider::new(), 2);

    let query = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
    let entities: Vec<KgEntity> = (0..4)
        .map(|i| KgEntity {
            id: format!("e{i}"),
            embedding_idx: i,
            properties: HashMap::new(),
        })
        .collect();

    let answer = retriever
        .answer(
            "Which entity has the best thermal properties?",
            &toy_kg_4(),
            &entities,
            &query,
        )
        .await
        .expect("answer");

    assert!(!answer.is_empty(), "answer should not be empty");
}

/// scorer error propagates correctly through PinnScorerError::EmbeddingIndexOutOfBounds
#[test]
fn test_scorer_out_of_bounds_error_message() {
    let scorer = PinnEntityScorer::new(toy_encoder_4d(), thermal_ctx(1e-5), 0.3);
    let embs = manual_embeddings(vec![vec![1.0, 0.0, 0.0, 0.0]]);
    let query = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
    let entities = vec![KgEntity {
        id: "oob".into(),
        embedding_idx: 42,
        properties: HashMap::new(),
    }];
    let err = scorer
        .score_entities(&embs, &entities, &query)
        .expect_err("expected error");
    assert!(
        matches!(
            err,
            PinnScorerError::EmbeddingIndexOutOfBounds { idx: 42, max: 1 }
        ),
        "unexpected error variant: {err:?}"
    );
    // Display should be non-empty
    assert!(!err.to_string().is_empty());
}
