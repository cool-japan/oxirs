//! Tests for the kg_embeddings module.
//! Imported via `#[path]` attribute in kg_embeddings.rs.

use super::*;
use std::collections::HashSet;

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

/// Build a small synthetic knowledge graph with `n_ent` entities,
/// `n_rel` relation types, and a cycle of triples.
fn make_synthetic_kg(n_ent: usize, n_rel: usize) -> Vec<KgTriple> {
    let mut triples = Vec::new();
    for r in 0..n_rel {
        for h in 0..n_ent {
            let t = (h + 1) % n_ent;
            triples.push(KgTriple::new(h, r, t));
        }
    }
    triples
}

fn default_config() -> KgEmbeddingConfig {
    KgEmbeddingConfig {
        embedding_dim: 10,
        learning_rate: 0.05,
        num_epochs: 20,
        batch_size: 8,
        neg_samples: 1,
        margin: 1.0,
        regularization: 1e-4,
        seed: 12345,
    }
}

// -----------------------------------------------------------------------
// Lcg tests
// -----------------------------------------------------------------------

#[test]
fn test_lcg_output_in_range() {
    let mut rng = Lcg::new(0);
    for _ in 0..1000 {
        let v = rng.next_f64();
        assert!((0.0..1.0).contains(&v), "LCG out of range: {v}");
    }
}

#[test]
fn test_lcg_usize_in_range() {
    let mut rng = Lcg::new(99);
    for n in [1usize, 2, 5, 10, 100] {
        for _ in 0..200 {
            let v = rng.next_usize(n);
            assert!(v < n, "usize {v} >= n={n}");
        }
    }
}

#[test]
fn test_lcg_deterministic() {
    let seed = 777;
    let mut rng1 = Lcg::new(seed);
    let mut rng2 = Lcg::new(seed);
    for _ in 0..50 {
        assert_eq!(
            rng1.next_f64().to_bits(),
            rng2.next_f64().to_bits(),
            "LCG not deterministic"
        );
    }
}

// -----------------------------------------------------------------------
// KgTriple tests
// -----------------------------------------------------------------------

#[test]
fn test_kg_triple_construction() {
    let t = KgTriple::new(0, 1, 2);
    assert_eq!(t.head, 0);
    assert_eq!(t.relation, 1);
    assert_eq!(t.tail, 2);
}

#[test]
fn test_kg_triple_equality() {
    let a = KgTriple::new(3, 0, 7);
    let b = KgTriple::new(3, 0, 7);
    let c = KgTriple::new(3, 0, 8);
    assert_eq!(a, b);
    assert_ne!(a, c);
}

#[test]
fn test_kg_triple_hashing() {
    let mut set = HashSet::new();
    set.insert(KgTriple::new(0, 0, 1));
    set.insert(KgTriple::new(0, 0, 1)); // duplicate
    set.insert(KgTriple::new(1, 0, 0));
    assert_eq!(set.len(), 2);
}

// -----------------------------------------------------------------------
// Math helpers
// -----------------------------------------------------------------------

#[test]
fn test_l2_norm_zero_vector() {
    let v = vec![0.0_f64; 5];
    assert_eq!(l2_norm(&v), 0.0);
}

#[test]
fn test_l2_norm_unit_vector() {
    let v = vec![1.0_f64, 0.0, 0.0];
    assert!((l2_norm(&v) - 1.0).abs() < 1e-12);
}

#[test]
fn test_l2_norm_general() {
    let v = vec![3.0_f64, 4.0];
    assert!((l2_norm(&v) - 5.0).abs() < 1e-12);
}

#[test]
fn test_l2_dist() {
    let a = vec![1.0_f64, 2.0, 3.0];
    let b = vec![4.0_f64, 6.0, 3.0];
    let d = l2_dist(&a, &b);
    assert!((d - 5.0).abs() < 1e-12);
}

#[test]
fn test_dot_product() {
    let a = vec![1.0_f64, 2.0, 3.0];
    let b = vec![4.0_f64, 5.0, 6.0];
    assert!((dot(&a, &b) - 32.0).abs() < 1e-12);
}

#[test]
fn test_normalize_vec() {
    let mut v = vec![3.0_f64, 4.0];
    normalize_vec(&mut v);
    assert!((l2_norm(&v) - 1.0).abs() < 1e-12);
}

#[test]
fn test_normalize_zero_vec_no_panic() {
    let mut v = vec![0.0_f64; 3];
    normalize_vec(&mut v); // should not panic or produce NaN
    for x in &v {
        assert!(x.is_finite());
    }
}

#[test]
fn test_sigmoid_boundary() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
    assert!(sigmoid(100.0) > 0.99);
    assert!(sigmoid(-100.0) < 0.01);
}

// -----------------------------------------------------------------------
// TransE tests
// -----------------------------------------------------------------------

#[test]
fn test_transe_score_fn_zero_distance() {
    // h + r - t = 0 when h = t and r = 0.
    let h = vec![0.5, 0.5];
    let r = vec![0.0, 0.0];
    let t = vec![0.5, 0.5];
    let dist = TransE::score_fn(&h, &r, &t);
    assert!(dist.abs() < 1e-12);
}

#[test]
fn test_transe_score_fn_known_value() {
    // h + r - t = (3,4) → norm = 5.
    let h = vec![1.0, 0.0];
    let r = vec![2.0, 4.0];
    let t = vec![0.0, 0.0];
    let dist = TransE::score_fn(&h, &r, &t);
    assert!((dist - 5.0).abs() < 1e-12);
}

#[test]
fn test_transe_not_trained_error() {
    let model = TransE::new(default_config());
    let err = model.score(&KgTriple::new(0, 0, 1));
    assert!(matches!(err, Err(KgError::NotTrained)));
}

#[test]
fn test_transe_no_data_error() {
    let mut model = TransE::new(default_config());
    let err = model.train(&[], 5, 2);
    assert!(matches!(err, Err(KgError::NoTrainingData)));
}

#[test]
fn test_transe_invalid_dim_error() {
    let cfg = KgEmbeddingConfig {
        embedding_dim: 0,
        ..default_config()
    };
    let mut model = TransE::new(cfg);
    let triples = vec![KgTriple::new(0, 0, 1)];
    let err = model.train(&triples, 2, 1);
    assert!(matches!(err, Err(KgError::InvalidDimension)));
}

#[test]
fn test_transe_trains_without_error() {
    let triples = make_synthetic_kg(5, 2);
    let mut model = TransE::new(default_config());
    let history = model
        .train(&triples, 5, 2)
        .expect("training should succeed");
    assert_eq!(history.epochs_trained, 20);
    assert!(history.final_loss.is_finite());
}

#[test]
fn test_transe_loss_recorded_per_epoch() {
    let triples = make_synthetic_kg(4, 1);
    let cfg = KgEmbeddingConfig {
        num_epochs: 10,
        ..default_config()
    };
    let mut model = TransE::new(cfg);
    let history = model.train(&triples, 4, 1).expect("should succeed");
    assert_eq!(history.losses.len(), 10);
}

#[test]
fn test_transe_score_is_finite_after_training() {
    let triples = make_synthetic_kg(4, 1);
    let mut model = TransE::new(default_config());
    model.train(&triples, 4, 1).expect("should succeed");
    let s = model
        .score(&KgTriple::new(0, 0, 1))
        .expect("should succeed");
    assert!(s.is_finite());
}

#[test]
fn test_transe_score_negative_for_trained_pair() {
    let triples = make_synthetic_kg(6, 1);
    let cfg = KgEmbeddingConfig {
        embedding_dim: 20,
        learning_rate: 0.1,
        num_epochs: 50,
        neg_samples: 2,
        ..default_config()
    };
    let mut model = TransE::new(cfg);
    model.train(&triples, 6, 1).expect("should succeed");
    let s_pos = model
        .score(&KgTriple::new(0, 0, 1))
        .expect("should succeed");
    let s_other = model
        .score(&KgTriple::new(0, 0, 3))
        .expect("should succeed");
    assert!(s_pos.is_finite());
    assert!(s_other.is_finite());
}

#[test]
fn test_transe_unknown_entity_error() {
    let triples = make_synthetic_kg(3, 1);
    let mut model = TransE::new(default_config());
    model.train(&triples, 3, 1).expect("should succeed");
    let err = model.score(&KgTriple::new(99, 0, 0));
    assert!(matches!(err, Err(KgError::UnknownEntity(99))));
}

#[test]
fn test_transe_unknown_relation_error() {
    let triples = make_synthetic_kg(3, 1);
    let mut model = TransE::new(default_config());
    model.train(&triples, 3, 1).expect("should succeed");
    let err = model.score(&KgTriple::new(0, 99, 1));
    assert!(matches!(err, Err(KgError::UnknownRelation(99))));
}

#[test]
fn test_transe_predict_tail_length() {
    let triples = make_synthetic_kg(5, 1);
    let mut model = TransE::new(default_config());
    model.train(&triples, 5, 1).expect("should succeed");
    let preds = model.predict_tail(0, 0, 3).expect("should succeed");
    assert_eq!(preds.len(), 3);
}

#[test]
fn test_transe_predict_tail_sorted_descending() {
    let triples = make_synthetic_kg(5, 1);
    let mut model = TransE::new(default_config());
    model.train(&triples, 5, 1).expect("should succeed");
    let preds = model.predict_tail(0, 0, 5).expect("should succeed");
    for w in preds.windows(2) {
        assert!(w[0].1 >= w[1].1, "predictions not sorted: {:?}", preds);
    }
}

#[test]
fn test_transe_predict_head_length() {
    let triples = make_synthetic_kg(5, 1);
    let mut model = TransE::new(default_config());
    model.train(&triples, 5, 1).expect("should succeed");
    let preds = model.predict_head(0, 1, 5).expect("should succeed");
    assert_eq!(preds.len(), 5);
}

#[test]
fn test_transe_predict_tail_top_k_zero_error() {
    let triples = make_synthetic_kg(3, 1);
    let mut model = TransE::new(default_config());
    model.train(&triples, 3, 1).expect("should succeed");
    let err = model.predict_tail(0, 0, 0);
    assert!(matches!(err, Err(KgError::InvalidTopK)));
}

#[test]
fn test_transe_normalize_entities() {
    let triples = make_synthetic_kg(4, 1);
    let mut model = TransE::new(default_config());
    model.train(&triples, 4, 1).expect("should succeed");
    model.normalize_entities();
    let emb = model.embeddings.as_ref().expect("should succeed");
    for v in &emb.entity_embeddings {
        let n = l2_norm(v);
        assert!((n - 1.0).abs() < 1e-10, "entity not unit norm: {n}");
    }
}

#[test]
fn test_transe_embedding_dimensions() {
    let triples = make_synthetic_kg(4, 2);
    let cfg = KgEmbeddingConfig {
        embedding_dim: 25,
        ..default_config()
    };
    let mut model = TransE::new(cfg);
    model.train(&triples, 4, 2).expect("should succeed");
    let emb = model.embeddings.as_ref().expect("should succeed");
    assert_eq!(emb.entity_embeddings.len(), 4);
    assert_eq!(emb.relation_embeddings.len(), 2);
    for v in &emb.entity_embeddings {
        assert_eq!(v.len(), 25);
    }
    for v in &emb.relation_embeddings {
        assert_eq!(v.len(), 25);
    }
}

#[test]
fn test_transe_deterministic_with_same_seed() {
    let triples = make_synthetic_kg(4, 1);
    let mut m1 = TransE::new(default_config());
    let mut m2 = TransE::new(default_config());
    let h1 = m1.train(&triples, 4, 1).expect("should succeed");
    let h2 = m2.train(&triples, 4, 1).expect("should succeed");
    assert_eq!(h1.final_loss.to_bits(), h2.final_loss.to_bits());
}

// -----------------------------------------------------------------------
// DistMult tests
// -----------------------------------------------------------------------

#[test]
fn test_distmult_score_fn_known_value() {
    let h = vec![1.0, 2.0];
    let r = vec![3.0, 4.0];
    let t = vec![5.0, 6.0];
    // Σ = 1*3*5 + 2*4*6 = 15 + 48 = 63.
    let s = DistMult::score_fn(&h, &r, &t);
    assert!((s - 63.0).abs() < 1e-12);
}

#[test]
fn test_distmult_score_fn_symmetry() {
    // DistMult score is symmetric in h and t.
    let h = vec![1.0, -2.0, 3.0];
    let r = vec![0.5, 1.0, 2.0];
    let t = vec![2.0, 3.0, -1.0];
    let s1 = DistMult::score_fn(&h, &r, &t);
    let s2 = DistMult::score_fn(&t, &r, &h);
    assert!(
        (s1 - s2).abs() < 1e-12,
        "DistMult not symmetric: {s1} vs {s2}"
    );
}

#[test]
fn test_distmult_not_trained_error() {
    let model = DistMult::new(default_config());
    assert!(matches!(
        model.score(&KgTriple::new(0, 0, 1)),
        Err(KgError::NotTrained)
    ));
}

#[test]
fn test_distmult_trains_without_error() {
    let triples = make_synthetic_kg(5, 2);
    let mut model = DistMult::new(default_config());
    let history = model.train(&triples, 5, 2).expect("should succeed");
    assert_eq!(history.epochs_trained, 20);
    assert!(history.final_loss.is_finite());
}

#[test]
fn test_distmult_score_finite_after_training() {
    let triples = make_synthetic_kg(4, 1);
    let mut model = DistMult::new(default_config());
    model.train(&triples, 4, 1).expect("should succeed");
    let s = model
        .score(&KgTriple::new(0, 0, 1))
        .expect("should succeed");
    assert!(s.is_finite());
}

#[test]
fn test_distmult_loss_per_epoch() {
    let triples = make_synthetic_kg(4, 1);
    let cfg = KgEmbeddingConfig {
        num_epochs: 15,
        ..default_config()
    };
    let mut model = DistMult::new(cfg);
    let h = model.train(&triples, 4, 1).expect("should succeed");
    assert_eq!(h.losses.len(), 15);
}

#[test]
fn test_distmult_predict_tail_returns_k() {
    let triples = make_synthetic_kg(6, 1);
    let mut model = DistMult::new(default_config());
    model.train(&triples, 6, 1).expect("should succeed");
    let preds = model.predict_tail(0, 0, 4).expect("should succeed");
    assert_eq!(preds.len(), 4);
}

#[test]
fn test_distmult_predict_tail_sorted() {
    let triples = make_synthetic_kg(6, 1);
    let mut model = DistMult::new(default_config());
    model.train(&triples, 6, 1).expect("should succeed");
    let preds = model.predict_tail(0, 0, 6).expect("should succeed");
    for w in preds.windows(2) {
        assert!(w[0].1 >= w[1].1);
    }
}

#[test]
fn test_distmult_embedding_shape() {
    let triples = make_synthetic_kg(3, 2);
    let cfg = KgEmbeddingConfig {
        embedding_dim: 8,
        ..default_config()
    };
    let mut model = DistMult::new(cfg);
    model.train(&triples, 3, 2).expect("should succeed");
    let emb = model.embeddings.as_ref().expect("should succeed");
    assert_eq!(emb.entity_embeddings.len(), 3);
    assert_eq!(emb.relation_embeddings.len(), 2);
    for v in &emb.entity_embeddings {
        assert_eq!(v.len(), 8);
    }
}

#[test]
fn test_distmult_unknown_entity_error() {
    let triples = make_synthetic_kg(3, 1);
    let mut model = DistMult::new(default_config());
    model.train(&triples, 3, 1).expect("should succeed");
    assert!(matches!(
        model.score(&KgTriple::new(50, 0, 0)),
        Err(KgError::UnknownEntity(50))
    ));
}

#[test]
fn test_distmult_predict_head_sorted() {
    let triples = make_synthetic_kg(5, 1);
    let mut model = DistMult::new(default_config());
    model.train(&triples, 5, 1).expect("should succeed");
    let preds = model.predict_head(0, 2, 5).expect("should succeed");
    for w in preds.windows(2) {
        assert!(w[0].1 >= w[1].1);
    }
}

#[test]
fn test_distmult_convergence_trend() {
    let triples = make_synthetic_kg(6, 2);
    let cfg = KgEmbeddingConfig {
        embedding_dim: 15,
        learning_rate: 0.05,
        num_epochs: 30,
        neg_samples: 2,
        ..default_config()
    };
    let mut model = DistMult::new(cfg);
    let history = model.train(&triples, 6, 2).expect("should succeed");
    let early: f64 = history.losses[..5].iter().sum::<f64>() / 5.0;
    let late: f64 = history.losses[25..].iter().sum::<f64>() / 5.0;
    assert!(
        late <= early * 1.5,
        "loss did not decrease: early={early:.4} late={late:.4}"
    );
}

// -----------------------------------------------------------------------
// RotatE tests
// -----------------------------------------------------------------------

#[test]
fn test_rotate_complex_multiply_identity() {
    let (re, im) = RotatE::complex_multiply(1.0, 0.0, 1.0, 0.0);
    assert!((re - 1.0).abs() < 1e-12);
    assert!(im.abs() < 1e-12);
}

#[test]
fn test_rotate_complex_multiply_rotation_90_deg() {
    let (re, im) = RotatE::complex_multiply(1.0, 0.0, 0.0, 1.0);
    assert!(re.abs() < 1e-12);
    assert!((im - 1.0).abs() < 1e-12);
}

#[test]
fn test_rotate_complex_multiply_general() {
    // (1+i)(2+3i) = -1 + 5i.
    let (re, im) = RotatE::complex_multiply(1.0, 1.0, 2.0, 3.0);
    assert!((re - (-1.0)).abs() < 1e-12);
    assert!((im - 5.0).abs() < 1e-12);
}

#[test]
fn test_rotate_not_trained_error() {
    let model = RotatE::new(default_config());
    assert!(matches!(
        model.score(&KgTriple::new(0, 0, 1)),
        Err(KgError::NotTrained)
    ));
}

#[test]
fn test_rotate_trains_without_error() {
    let triples = make_synthetic_kg(5, 2);
    let mut model = RotatE::new(default_config());
    let history = model.train(&triples, 5, 2).expect("should succeed");
    assert_eq!(history.epochs_trained, 20);
    assert!(history.final_loss.is_finite());
}

#[test]
fn test_rotate_score_finite_after_training() {
    let triples = make_synthetic_kg(4, 1);
    let mut model = RotatE::new(default_config());
    model.train(&triples, 4, 1).expect("should succeed");
    let s = model
        .score(&KgTriple::new(0, 0, 1))
        .expect("should succeed");
    assert!(s.is_finite());
}

#[test]
fn test_rotate_embedding_count() {
    let triples = make_synthetic_kg(5, 3);
    let mut model = RotatE::new(default_config());
    model.train(&triples, 5, 3).expect("should succeed");
    assert_eq!(model.entity_re.as_ref().expect("should succeed").len(), 5);
    assert_eq!(model.entity_im.as_ref().expect("should succeed").len(), 5);
    assert_eq!(
        model
            .relation_phases
            .as_ref()
            .expect("should succeed")
            .len(),
        3
    );
}

#[test]
fn test_rotate_predict_tail_length() {
    let triples = make_synthetic_kg(5, 1);
    let mut model = RotatE::new(default_config());
    model.train(&triples, 5, 1).expect("should succeed");
    let preds = model.predict_tail(0, 0, 3).expect("should succeed");
    assert_eq!(preds.len(), 3);
}

#[test]
fn test_rotate_predict_tail_sorted() {
    let triples = make_synthetic_kg(5, 1);
    let mut model = RotatE::new(default_config());
    model.train(&triples, 5, 1).expect("should succeed");
    let preds = model.predict_tail(0, 0, 5).expect("should succeed");
    for w in preds.windows(2) {
        assert!(w[0].1 >= w[1].1, "not sorted: {:?}", preds);
    }
}

#[test]
fn test_rotate_predict_head_sorted() {
    let triples = make_synthetic_kg(5, 1);
    let mut model = RotatE::new(default_config());
    model.train(&triples, 5, 1).expect("should succeed");
    let preds = model.predict_head(0, 2, 5).expect("should succeed");
    for w in preds.windows(2) {
        assert!(w[0].1 >= w[1].1, "not sorted: {:?}", preds);
    }
}

#[test]
fn test_rotate_loss_per_epoch() {
    let triples = make_synthetic_kg(4, 1);
    let cfg = KgEmbeddingConfig {
        num_epochs: 12,
        ..default_config()
    };
    let mut model = RotatE::new(cfg);
    let h = model.train(&triples, 4, 1).expect("should succeed");
    assert_eq!(h.losses.len(), 12);
}

#[test]
fn test_rotate_unknown_entity_error() {
    let triples = make_synthetic_kg(3, 1);
    let mut model = RotatE::new(default_config());
    model.train(&triples, 3, 1).expect("should succeed");
    assert!(matches!(
        model.score(&KgTriple::new(99, 0, 0)),
        Err(KgError::UnknownEntity(99))
    ));
}

#[test]
fn test_rotate_unknown_relation_error() {
    let triples = make_synthetic_kg(3, 1);
    let mut model = RotatE::new(default_config());
    model.train(&triples, 3, 1).expect("should succeed");
    assert!(matches!(
        model.score(&KgTriple::new(0, 99, 1)),
        Err(KgError::UnknownRelation(99))
    ));
}

#[test]
fn test_rotate_zero_top_k_error() {
    let triples = make_synthetic_kg(3, 1);
    let mut model = RotatE::new(default_config());
    model.train(&triples, 3, 1).expect("should succeed");
    assert!(matches!(
        model.predict_tail(0, 0, 0),
        Err(KgError::InvalidTopK)
    ));
}

// -----------------------------------------------------------------------
// Negative sampling tests
// -----------------------------------------------------------------------

#[test]
fn test_corrupt_triple_changes_triple() {
    let positive: HashSet<(usize, usize, usize)> = [(0, 0, 1)].iter().cloned().collect();
    let original = KgTriple::new(0, 0, 1);
    let mut rng = Lcg::new(42);
    let corrupted = corrupt_triple(&original, 10, &positive, &mut rng);
    assert_ne!(
        (corrupted.head, corrupted.relation, corrupted.tail),
        (original.head, original.relation, original.tail)
    );
}

#[test]
fn test_corrupt_triple_keeps_relation() {
    let positive: HashSet<(usize, usize, usize)> = HashSet::new();
    let original = KgTriple::new(2, 5, 3);
    let mut rng = Lcg::new(7);
    for _ in 0..100 {
        let c = corrupt_triple(&original, 20, &positive, &mut rng);
        assert_eq!(c.relation, original.relation);
    }
}

// -----------------------------------------------------------------------
// KgModel trait tests
// -----------------------------------------------------------------------

#[test]
fn test_kg_model_trait_transe() {
    let triples = make_synthetic_kg(4, 1);
    let mut model = TransE::new(default_config());
    model.train(&triples, 4, 1).expect("should succeed");
    let dyn_model: &dyn KgModel = &model;
    let s = dyn_model
        .score(&KgTriple::new(0, 0, 1))
        .expect("should succeed");
    assert!(s.is_finite());
}

#[test]
fn test_kg_model_trait_distmult() {
    let triples = make_synthetic_kg(4, 1);
    let mut model = DistMult::new(default_config());
    model.train(&triples, 4, 1).expect("should succeed");
    let dyn_model: &dyn KgModel = &model;
    let s = dyn_model
        .score(&KgTriple::new(0, 0, 1))
        .expect("should succeed");
    assert!(s.is_finite());
}

#[test]
fn test_kg_model_trait_rotate() {
    let triples = make_synthetic_kg(4, 1);
    let mut model = RotatE::new(default_config());
    model.train(&triples, 4, 1).expect("should succeed");
    let dyn_model: &dyn KgModel = &model;
    let s = dyn_model
        .score(&KgTriple::new(0, 0, 1))
        .expect("should succeed");
    assert!(s.is_finite());
}

// -----------------------------------------------------------------------
// Evaluation metrics tests
// -----------------------------------------------------------------------

#[test]
fn test_hits_at_k_perfect() {
    let triples = make_synthetic_kg(5, 1);
    let cfg = KgEmbeddingConfig {
        embedding_dim: 20,
        learning_rate: 0.1,
        num_epochs: 50,
        neg_samples: 2,
        ..default_config()
    };
    let mut model = TransE::new(cfg);
    model.train(&triples, 5, 1).expect("should succeed");
    let h5 = LinkPredictionEvaluator::hits_at_k(&model, &triples, 5);
    assert!((0.0..=1.0).contains(&h5), "Hits@5 out of range: {h5}");
}

#[test]
fn test_hits_at_k_empty() {
    let model = TransE::new(default_config());
    let h = LinkPredictionEvaluator::hits_at_k(&model, &[], 5);
    assert_eq!(h, 0.0);
}

#[test]
fn test_hits_at_k_zero_k() {
    let model = TransE::new(default_config());
    let triples = vec![KgTriple::new(0, 0, 1)];
    let h = LinkPredictionEvaluator::hits_at_k(&model, &triples, 0);
    assert_eq!(h, 0.0);
}

#[test]
fn test_mean_rank_empty() {
    let model = TransE::new(default_config());
    let mr = LinkPredictionEvaluator::mean_rank(&model, &[], 5);
    assert_eq!(mr, 0.0);
}

#[test]
fn test_mean_rank_trained() {
    let triples = make_synthetic_kg(6, 1);
    let mut model = TransE::new(default_config());
    model.train(&triples, 6, 1).expect("should succeed");
    let mr = LinkPredictionEvaluator::mean_rank(&model, &triples, 6);
    assert!((1.0..=7.0).contains(&mr), "mean rank out of range: {mr}");
}

#[test]
fn test_mrr_empty() {
    let model = DistMult::new(default_config());
    let mrr = LinkPredictionEvaluator::mrr(&model, &[], 5);
    assert_eq!(mrr, 0.0);
}

#[test]
fn test_mrr_trained_in_range() {
    let triples = make_synthetic_kg(5, 1);
    let mut model = DistMult::new(default_config());
    model.train(&triples, 5, 1).expect("should succeed");
    let mrr = LinkPredictionEvaluator::mrr(&model, &triples, 5);
    assert!((0.0..=1.0).contains(&mrr), "MRR out of [0,1]: {mrr}");
}

#[test]
fn test_hits_at_1_leq_hits_at_3_leq_hits_at_10() {
    let triples = make_synthetic_kg(10, 2);
    let cfg = KgEmbeddingConfig {
        embedding_dim: 20,
        num_epochs: 30,
        ..default_config()
    };
    let mut model = TransE::new(cfg);
    model.train(&triples, 10, 2).expect("should succeed");
    let h1 = LinkPredictionEvaluator::hits_at_k(&model, &triples, 1);
    let h3 = LinkPredictionEvaluator::hits_at_k(&model, &triples, 3);
    let h10 = LinkPredictionEvaluator::hits_at_k(&model, &triples, 10);
    assert!(h1 <= h3 + 1e-9, "Hits@1 > Hits@3: {h1} > {h3}");
    assert!(h3 <= h10 + 1e-9, "Hits@3 > Hits@10: {h3} > {h10}");
}

// -----------------------------------------------------------------------
// Serialisation / deserialisation tests
// -----------------------------------------------------------------------

#[test]
fn test_serialize_round_trip() {
    let triples = make_synthetic_kg(4, 2);
    let mut model = TransE::new(default_config());
    model.train(&triples, 4, 2).expect("should succeed");
    let emb = model.embeddings.as_ref().expect("should succeed");
    let bytes = serialize_embeddings(emb);
    let restored = deserialize_embeddings(&bytes).expect("should succeed");
    assert_eq!(
        restored.entity_embeddings.len(),
        emb.entity_embeddings.len()
    );
    assert_eq!(
        restored.relation_embeddings.len(),
        emb.relation_embeddings.len()
    );
    for (orig, rest) in emb
        .entity_embeddings
        .iter()
        .zip(restored.entity_embeddings.iter())
    {
        for (&o, &r) in orig.iter().zip(rest.iter()) {
            assert!((o - r).abs() < 1e-6, "mismatch: {o} vs {r}");
        }
    }
}

#[test]
fn test_deserialize_invalid_utf8() {
    let bad: Vec<u8> = vec![0xFF, 0xFE, 0x00];
    let err = deserialize_embeddings(&bad);
    assert!(err.is_err());
}

#[test]
fn test_serialize_empty_embeddings() {
    let emb = KgEmbeddings {
        entity_embeddings: vec![],
        relation_embeddings: vec![],
        entity_to_id: HashMap::new(),
        relation_to_id: HashMap::new(),
    };
    let bytes = serialize_embeddings(&emb);
    let restored = deserialize_embeddings(&bytes).expect("should succeed");
    assert!(restored.entity_embeddings.is_empty());
    assert!(restored.relation_embeddings.is_empty());
}

#[test]
fn test_serialize_preserves_relation_count() {
    let triples = make_synthetic_kg(3, 3);
    let mut model = DistMult::new(default_config());
    model.train(&triples, 3, 3).expect("should succeed");
    let emb = model.embeddings.as_ref().expect("should succeed");
    let bytes = serialize_embeddings(emb);
    let restored = deserialize_embeddings(&bytes).expect("should succeed");
    assert_eq!(restored.relation_embeddings.len(), 3);
}

// -----------------------------------------------------------------------
// Config default tests
// -----------------------------------------------------------------------

#[test]
fn test_config_defaults() {
    let cfg = KgEmbeddingConfig::default();
    assert_eq!(cfg.embedding_dim, 50);
    assert!(cfg.learning_rate > 0.0);
    assert!(cfg.num_epochs > 0);
    assert!(cfg.neg_samples > 0);
    assert!(cfg.margin > 0.0);
}

// -----------------------------------------------------------------------
// Larger-scale convergence smoke tests
// -----------------------------------------------------------------------

#[test]
fn test_transe_larger_kg_convergence() {
    let triples = make_synthetic_kg(15, 3);
    let cfg = KgEmbeddingConfig {
        embedding_dim: 20,
        learning_rate: 0.05,
        num_epochs: 40,
        neg_samples: 2,
        ..default_config()
    };
    let mut model = TransE::new(cfg);
    let h = model.train(&triples, 15, 3).expect("should succeed");
    assert!(h.final_loss.is_finite());
    assert!(h.final_loss >= 0.0);
}

#[test]
fn test_distmult_larger_kg_convergence() {
    let triples = make_synthetic_kg(15, 3);
    let cfg = KgEmbeddingConfig {
        embedding_dim: 20,
        learning_rate: 0.05,
        num_epochs: 40,
        neg_samples: 2,
        ..default_config()
    };
    let mut model = DistMult::new(cfg);
    let h = model.train(&triples, 15, 3).expect("should succeed");
    assert!(h.final_loss.is_finite());
    assert!(h.final_loss >= 0.0);
}

#[test]
fn test_rotate_larger_kg_convergence() {
    let triples = make_synthetic_kg(15, 3);
    let cfg = KgEmbeddingConfig {
        embedding_dim: 20,
        learning_rate: 0.01,
        num_epochs: 40,
        neg_samples: 1,
        ..default_config()
    };
    let mut model = RotatE::new(cfg);
    let h = model.train(&triples, 15, 3).expect("should succeed");
    assert!(h.final_loss.is_finite());
    assert!(h.final_loss >= 0.0);
}

#[test]
fn test_all_scores_finite_for_all_triples() {
    let triples = make_synthetic_kg(6, 2);
    let mut model = TransE::new(default_config());
    model.train(&triples, 6, 2).expect("should succeed");
    for t in &triples {
        let s = model.score(t).expect("should succeed");
        assert!(s.is_finite(), "non-finite score for triple {t:?}: {s}");
    }
}
