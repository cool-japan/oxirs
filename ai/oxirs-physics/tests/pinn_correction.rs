//! Integration tests for the PINN residual corrector.
//!
//! Both the model construction APIs (`ResidualModel`, `DenseLayer`,
//! `Activation`) and the loader (JSON round-trip via `temp_dir`) are
//! exercised here. The corrector itself is feature-gated; tests assert the
//! correct behaviour with and without the `pinn_correction` feature.

use std::env::temp_dir;
use std::fs;
use std::process;

use oxirs_physics::pinn::loader::{
    dump_residual_model_to_json, load_residual_model_from_json, load_residual_model_from_path,
};
use oxirs_physics::pinn::residual_model::DenseLayer;
use oxirs_physics::pinn::{
    Activation, PinnCorrector, PinnError, ResidualModel, ResidualModelConfig,
};

fn tiny_identity_model() -> ResidualModel {
    let cfg = ResidualModelConfig {
        input_dim: 2,
        output_dim: 2,
        hidden_widths: vec![2],
        hidden_activation: Activation::Relu,
        output_activation: Activation::Identity,
        description: "identity".to_string(),
    };
    let h = DenseLayer::new(
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![0.0, 0.0],
        Activation::Identity,
    )
    .expect("layer ok");
    let o = DenseLayer::new(
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![0.0, 0.0],
        Activation::Identity,
    )
    .expect("layer ok");
    ResidualModel::from_layers(cfg, vec![h, o]).expect("model ok")
}

#[test]
fn json_round_trip_through_temp_dir() {
    let model = tiny_identity_model();
    let bytes = dump_residual_model_to_json(&model).expect("dump ok");
    let mut path = temp_dir();
    path.push(format!(
        "oxirs_physics_pinn_test_{}_{}.json",
        process::id(),
        chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
    ));
    fs::write(&path, &bytes).expect("write ok");
    let loaded = load_residual_model_from_path(&path).expect("load ok");
    assert_eq!(loaded, model);
    let _ = fs::remove_file(&path);
}

#[test]
fn json_bytes_round_trip() {
    let model = tiny_identity_model();
    let bytes = dump_residual_model_to_json(&model).expect("dump ok");
    let loaded = load_residual_model_from_json(&bytes).expect("load ok");
    assert_eq!(loaded, model);
}

#[test]
fn parameter_count_under_10k() {
    // The plan requires "~10K params"; tiny model has well under.
    let model = tiny_identity_model();
    assert!(model.parameter_count() < 10_000);
}

#[test]
fn corrector_apply_matches_feature() {
    let model = tiny_identity_model();
    let corrector = PinnCorrector::new(model);
    let prev = vec![1.0, 2.0];
    let step = vec![3.0, 4.0];
    let result = corrector.apply(&prev, &step);

    if cfg!(feature = "pinn_correction") {
        // Identity nn(prev) = prev, residual_scale defaults to 1.0,
        // so corrected = step + prev = [4.0, 6.0].
        let v = result.expect("apply ok");
        assert_eq!(v, vec![4.0, 6.0]);
        assert!(corrector.is_active());
    } else {
        assert!(matches!(result, Err(PinnError::FeatureDisabled)));
        assert!(!corrector.is_active());
    }
}

#[test]
fn empty_corrector_always_returns_disabled() {
    let c = PinnCorrector::empty();
    let r = c.apply(&[1.0], &[1.0]);
    assert!(matches!(r, Err(PinnError::FeatureDisabled)));
}

#[test]
fn validate_rejects_inconsistent_layer_count() {
    let cfg = ResidualModelConfig {
        input_dim: 2,
        output_dim: 2,
        hidden_widths: vec![2, 2, 2], // expects 4 layers
        hidden_activation: Activation::Relu,
        output_activation: Activation::Identity,
        description: "bad".to_string(),
    };
    let h = DenseLayer::new(
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![0.0, 0.0],
        Activation::Relu,
    )
    .expect("layer ok");
    let o = DenseLayer::new(
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        vec![0.0, 0.0],
        Activation::Identity,
    )
    .expect("layer ok");
    // Only 2 layers — much fewer than the config demands.
    let result = ResidualModel::from_layers(cfg, vec![h, o]);
    assert!(matches!(result, Err(PinnError::InvalidModel(_))));
}

#[test]
fn synthetic_residual_correction_matches_hand_computation() {
    // Build a network whose residual is +1 on input 0 and 0 on input 1.
    let cfg = ResidualModelConfig {
        input_dim: 2,
        output_dim: 2,
        hidden_widths: vec![2],
        hidden_activation: Activation::Identity,
        output_activation: Activation::Identity,
        description: "synthetic".to_string(),
    };
    let h = DenseLayer::new(
        vec![vec![1.0, 0.0], vec![0.0, 0.0]],
        vec![1.0, 0.0],
        Activation::Identity,
    )
    .expect("layer ok");
    let o = DenseLayer::new(
        vec![vec![1.0, 0.0], vec![0.0, 0.0]],
        vec![0.0, 0.0],
        Activation::Identity,
    )
    .expect("layer ok");
    let model = ResidualModel::from_layers(cfg, vec![h, o]).expect("model ok");

    // Hand computation:
    //   hidden = [1*x0 + 1, 0]
    //   output = [hidden[0], 0] = [x0 + 1, 0]
    //   residual = output (because skip connection lives in apply, not raw)
    let raw = model.forward_raw(&[5.0, 7.0]).expect("forward ok");
    assert_eq!(raw, vec![6.0, 0.0]);

    // apply_residual returns input + 1.0 * raw
    let out = model.apply_residual(&[5.0, 7.0]).expect("apply ok");
    assert_eq!(out, vec![5.0 + 6.0, 7.0 + 0.0]);
}

#[test]
fn loader_rejects_garbage() {
    let result = load_residual_model_from_json(b"not a model");
    assert!(matches!(result, Err(PinnError::Deserialise(_))));
}
