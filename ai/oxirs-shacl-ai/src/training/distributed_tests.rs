//! Tests for the distributed training subsystem.
//!
//! Consolidates the test suites originally embedded inside `distributed.rs`
//! covering parameter vectors, gradient accumulation, optimisers (SGD, Adam),
//! finite-difference gradients, gradient compression, the AllReduce-based
//! data-parallel training runner, federated learning (FedAvg), and the
//! Gaussian differential-privacy mechanism.

#![cfg(test)]

use crate::training::distributed::*;

// ---------------------------------------------------------------------------
// ParameterVector
// ---------------------------------------------------------------------------

#[test]
fn test_param_vector_zeros() {
    let p = ParameterVector::zeros(4);
    assert_eq!(p.len(), 4);
    assert!(p.values.iter().all(|&v| v == 0.0));
}

#[test]
fn test_param_vector_add_assign() {
    let mut a = ParameterVector::from_vec(vec![1.0, 2.0, 3.0]);
    let b = ParameterVector::from_vec(vec![0.5, 0.5, 0.5]);
    a.add_assign(&b);
    assert!((a.values[0] - 1.5).abs() < 1e-12);
    assert!((a.values[1] - 2.5).abs() < 1e-12);
    assert!((a.values[2] - 3.5).abs() < 1e-12);
}

#[test]
fn test_param_vector_scale() {
    let mut p = ParameterVector::from_vec(vec![2.0, 4.0, 6.0]);
    p.scale(0.5);
    assert!((p.values[0] - 1.0).abs() < 1e-12);
    assert!((p.values[2] - 3.0).abs() < 1e-12);
}

#[test]
fn test_param_vector_norm() {
    let p = ParameterVector::from_vec(vec![3.0, 4.0]);
    assert!((p.norm() - 5.0).abs() < 1e-12);
}

#[test]
fn test_param_vector_clip_norm() {
    let mut p = ParameterVector::from_vec(vec![3.0, 4.0]); // norm = 5
    p.clip_norm(2.5);
    assert!(
        (p.norm() - 2.5).abs() < 1e-10,
        "norm after clip: {}",
        p.norm()
    );
}

#[test]
fn test_param_vector_clip_norm_no_effect_when_below() {
    let mut p = ParameterVector::from_vec(vec![1.0, 1.0]); // norm ~= 1.41
    p.clip_norm(5.0);
    assert!((p.norm() - std::f64::consts::SQRT_2).abs() < 1e-10);
}

#[test]
fn test_param_vector_mse_diff_zeros() {
    let a = ParameterVector::zeros(4);
    let b = ParameterVector::zeros(4);
    assert_eq!(a.mse_diff(&b), 0.0);
}

#[test]
fn test_param_vector_mse_diff_nonzero() {
    let a = ParameterVector::from_vec(vec![0.0, 0.0]);
    let b = ParameterVector::from_vec(vec![1.0, 1.0]);
    assert!((a.mse_diff(&b) - 1.0).abs() < 1e-12);
}

// ---------------------------------------------------------------------------
// GradientAccumulator
// ---------------------------------------------------------------------------

#[test]
fn test_grad_accumulator_new() {
    let acc = GradientAccumulator::new(8);
    assert_eq!(acc.grads.len(), 8);
    assert_eq!(acc.sample_count, 0);
}

#[test]
fn test_grad_accumulator_accumulate_and_average() {
    let mut acc = GradientAccumulator::new(2);
    acc.accumulate(&ParameterVector::from_vec(vec![2.0, 4.0]));
    acc.accumulate(&ParameterVector::from_vec(vec![4.0, 8.0]));
    acc.average();
    // After 2 samples with sum [6, 12] → average [3, 6]
    assert!((acc.grads.values[0] - 3.0).abs() < 1e-12);
    assert!((acc.grads.values[1] - 6.0).abs() < 1e-12);
    assert_eq!(acc.sample_count, 0);
}

#[test]
fn test_grad_accumulator_reset() {
    let mut acc = GradientAccumulator::new(3);
    acc.accumulate(&ParameterVector::from_vec(vec![1.0, 1.0, 1.0]));
    acc.reset();
    assert!(acc.grads.values.iter().all(|&v| v == 0.0));
    assert_eq!(acc.sample_count, 0);
}

// ---------------------------------------------------------------------------
// SGD Optimiser
// ---------------------------------------------------------------------------

#[test]
fn test_sgd_converges_quadratic() {
    let n = 4;
    let mut opt = SgdOptimiser::new(n, 0.1, 0.0, 0.0);
    let mut params = ParameterVector::from_vec(vec![1.0_f64; n]);

    // Gradient of 0.5 * ||params||^2 is params itself.
    // With lr=0.1 and n=4 dimensions, 50 iterations yields
    // norm = sqrt(n) * 0.9^50 ≈ 0.0103 which narrowly exceeds 0.01.
    // 100 iterations gives norm ≈ 5e-6, well within the threshold.
    for _ in 0..100 {
        let grad = params.clone();
        opt.step(&mut params, &grad);
    }
    let norm = params.norm();
    assert!(norm < 0.01, "SGD did not converge, norm={norm}");
}

#[test]
fn test_sgd_momentum() {
    let n = 2;
    let mut opt = SgdOptimiser::new(n, 0.01, 0.9, 0.0);
    let mut params = ParameterVector::from_vec(vec![1.0_f64; n]);
    for _ in 0..100 {
        let grad = params.clone();
        opt.step(&mut params, &grad);
    }
    assert!(params.norm() < 1.0, "momentum SGD norm={}", params.norm());
}

// ---------------------------------------------------------------------------
// Adam Optimiser
// ---------------------------------------------------------------------------

#[test]
fn test_adam_converges_quadratic() {
    let n = 4;
    let mut opt = AdamOptimiser::default_config(n, 0.01);
    let mut params = ParameterVector::from_vec(vec![1.0_f64; n]);

    for _ in 0..200 {
        let grad = params.clone();
        opt.step(&mut params, &grad);
    }
    let norm = params.norm();
    assert!(norm < 0.1, "Adam did not converge, norm={norm}");
}

#[test]
fn test_adam_step_counter() {
    let n = 2;
    let mut opt = AdamOptimiser::default_config(n, 0.001);
    let mut params = ParameterVector::from_vec(vec![1.0_f64; n]);
    let grad = ParameterVector::from_vec(vec![0.1_f64; n]);
    opt.step(&mut params, &grad);
    opt.step(&mut params, &grad);
    assert_eq!(opt.step_count(), 2);
}

// ---------------------------------------------------------------------------
// Finite difference
// ---------------------------------------------------------------------------

#[test]
fn test_finite_difference_quadratic() {
    // f(x) = 0.5 * ||x||^2  =>  grad = x
    let params = vec![1.0_f64, 2.0, 3.0];
    let f = |p: &[f64]| -> f64 { 0.5 * p.iter().map(|&v| v * v).sum::<f64>() };
    let grad = finite_difference_grad(&params, &f, 1e-5);
    for (i, &g) in grad.iter().enumerate() {
        assert!(
            (g - params[i]).abs() < 1e-6,
            "grad[{i}]={g} expected {}",
            params[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Gradient compression
// ---------------------------------------------------------------------------

#[test]
fn test_sparsify_gradients() {
    let g = ParameterVector::from_vec(vec![0.001, 0.5, 0.002, 0.8]);
    let (sparse, sparsity) = sparsify_gradients(&g, 0.01);
    assert_eq!(sparse.values[0], 0.0);
    assert!((sparse.values[1] - 0.5).abs() < 1e-12);
    assert_eq!(sparse.values[2], 0.0);
    assert!((sparse.values[3] - 0.8).abs() < 1e-12);
    assert!((sparsity - 0.5).abs() < 1e-12, "sparsity={sparsity}");
}

#[test]
fn test_sparsify_empty() {
    let g = ParameterVector::zeros(0);
    let (sparse, sparsity) = sparsify_gradients(&g, 0.1);
    assert!(sparse.is_empty());
    assert_eq!(sparsity, 0.0);
}

#[test]
fn test_quantise_dequantise_roundtrip() {
    let original = ParameterVector::from_vec(vec![-1.0, 0.0, 0.5, 1.0]);
    let (q, scale, zp) = quantise_gradients_i8(&original);
    let recovered = dequantise_gradients_i8(&q, scale, zp);
    for (a, b) in original.values.iter().zip(&recovered.values) {
        assert!(
            (a - b).abs() < 0.05,
            "roundtrip error too large: {a} vs {b}"
        );
    }
}

#[test]
fn test_quantise_empty() {
    let g = ParameterVector::zeros(0);
    let (q, scale, zp) = quantise_gradients_i8(&g);
    assert!(q.is_empty());
    assert_eq!(scale, 1.0);
    assert_eq!(zp, 0.0);
}

// ---------------------------------------------------------------------------
// Distributed training runner
// ---------------------------------------------------------------------------

#[test]
fn test_distributed_training_single_worker() {
    let cfg = DistributedTrainingConfig {
        num_workers: 1,
        num_steps: 5,
        param_count: 16,
        ..Default::default()
    };
    let trainer = DistributedTrainer::new(cfg);
    let stats = trainer.run().expect("single-worker training ok");
    assert_eq!(stats.num_workers, 1);
    assert_eq!(stats.communication_rounds, 5);
    assert!(stats.total_loss >= 0.0);
}

#[test]
fn test_distributed_training_multi_worker() {
    let cfg = DistributedTrainingConfig {
        num_workers: 3,
        num_steps: 4,
        param_count: 8,
        ..Default::default()
    };
    let trainer = DistributedTrainer::new(cfg);
    let stats = trainer.run().expect("multi-worker training ok");
    assert_eq!(stats.num_workers, 3);
    // The 4 step losses are recorded by rank 0
    assert_eq!(stats.step_losses.len(), 4);
}

#[test]
fn test_distributed_training_zero_workers_error() {
    let cfg = DistributedTrainingConfig {
        num_workers: 0,
        ..Default::default()
    };
    let trainer = DistributedTrainer::new(cfg);
    assert!(trainer.run().is_err());
}

#[test]
fn test_distributed_training_loss_decreases() {
    let cfg = DistributedTrainingConfig {
        num_workers: 2,
        num_steps: 8,
        param_count: 4,
        learning_rate: 0.1,
        use_adam: false,
        ..Default::default()
    };
    let trainer = DistributedTrainer::new(cfg);
    let stats = trainer.run().expect("loss decrease test ok");
    let losses = &stats.step_losses;
    if losses.len() >= 2 {
        // Generally the first loss should be larger than the last
        // (not strictly guaranteed with threading nondeterminism but very likely)
        assert!(
            losses[0] >= losses[losses.len() - 1],
            "loss should not increase: first={} last={}",
            losses[0],
            losses[losses.len() - 1]
        );
    }
}

#[test]
fn test_distributed_training_sgd() {
    let cfg = DistributedTrainingConfig {
        num_workers: 2,
        num_steps: 3,
        param_count: 4,
        use_adam: false,
        strategy: AllReduceStrategy::Sum,
        ..Default::default()
    };
    let trainer = DistributedTrainer::new(cfg);
    let stats = trainer.run().expect("SGD run ok");
    assert!(stats.total_time_ms < 10_000); // should finish quickly
}

#[test]
fn test_all_reduce_strategies() {
    for strategy in [
        AllReduceStrategy::Mean,
        AllReduceStrategy::Sum,
        AllReduceStrategy::WeightedMean,
    ] {
        let cfg = DistributedTrainingConfig {
            num_workers: 2,
            num_steps: 2,
            param_count: 4,
            strategy,
            ..Default::default()
        };
        let trainer = DistributedTrainer::new(cfg);
        trainer.run().expect("strategy variant should run");
    }
}

#[test]
fn test_training_stats_serialization() {
    let stats = DistributedTrainingStats {
        total_time_ms: 150,
        communication_rounds: 10,
        total_loss: 3.5,
        final_param_norm: 0.12,
        num_workers: 4,
        step_losses: vec![0.5, 0.4, 0.35, 0.3],
    };
    let json = serde_json::to_string(&stats).expect("serialize ok");
    let s2: DistributedTrainingStats = serde_json::from_str(&json).expect("deserialize ok");
    assert_eq!(s2.num_workers, 4);
    assert_eq!(s2.communication_rounds, 10);
}

#[test]
fn test_worker_config_default() {
    let wc = WorkerConfig::default();
    assert_eq!(wc.rank, 0);
    assert_eq!(wc.world_size, 1);
    assert!(wc.learning_rate > 0.0);
}

// ---------------------------------------------------------------------------
// ModelWeights
// ---------------------------------------------------------------------------

#[test]
fn test_model_weights_zeros() {
    let mw = ModelWeights::zeros(4, 2);
    assert_eq!(mw.weights, vec![0.0; 4]);
    assert_eq!(mw.biases, vec![0.0; 2]);
    assert_eq!(mw.version, 0);
}

#[test]
fn test_model_weights_norm() {
    let mw = ModelWeights::from_vecs(vec![3.0, 4.0], vec![0.0]);
    // ‖[3,4,0]‖ = 5.0
    assert!((mw.norm() - 5.0).abs() < 1e-9);
}

#[test]
fn test_model_weights_add_assign() {
    let mut a = ModelWeights::from_vecs(vec![1.0, 2.0], vec![0.5]);
    let b = ModelWeights::from_vecs(vec![0.5, 0.5], vec![0.5]);
    a.add_assign(&b);
    assert!((a.weights[0] - 1.5).abs() < 1e-9);
    assert!((a.weights[1] - 2.5).abs() < 1e-9);
    assert!((a.biases[0] - 1.0).abs() < 1e-9);
}

#[test]
fn test_model_weights_scale() {
    let mut mw = ModelWeights::from_vecs(vec![2.0, 4.0], vec![1.0]);
    mw.scale(0.5);
    assert!((mw.weights[0] - 1.0).abs() < 1e-9);
    assert!((mw.weights[1] - 2.0).abs() < 1e-9);
    assert!((mw.biases[0] - 0.5).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// FederatedRound
// ---------------------------------------------------------------------------

#[test]
fn test_federated_round_total_samples() {
    let global = ModelWeights::zeros(2, 1);
    let mut round = FederatedRound::new(0, global);
    round.add_update(LocalUpdate::new("p1", ModelWeights::zeros(2, 1), 100, 0.5));
    round.add_update(LocalUpdate::new("p2", ModelWeights::zeros(2, 1), 200, 0.4));
    assert_eq!(round.total_samples(), 300);
    assert_eq!(round.participant_count, 2);
}

#[test]
fn test_federated_round_average_loss() {
    let global = ModelWeights::zeros(2, 1);
    let mut round = FederatedRound::new(0, global);
    round.add_update(LocalUpdate::new("p1", ModelWeights::zeros(2, 1), 50, 0.6));
    round.add_update(LocalUpdate::new("p2", ModelWeights::zeros(2, 1), 50, 0.4));
    let avg = round.average_local_loss();
    assert!((avg - 0.5).abs() < 1e-9);
}

#[test]
fn test_federated_round_empty_loss() {
    let round = FederatedRound::new(0, ModelWeights::zeros(2, 1));
    assert_eq!(round.average_local_loss(), 0.0);
}

// ---------------------------------------------------------------------------
// FederatedShapeTrainer — federated_averaging
// ---------------------------------------------------------------------------

#[test]
fn test_federated_averaging_equal_samples() {
    let trainer = FederatedShapeTrainer::new(2, 1);
    let updates = vec![
        LocalUpdate::new(
            "p1",
            ModelWeights::from_vecs(vec![2.0, 0.0], vec![1.0]),
            50,
            0.5,
        ),
        LocalUpdate::new(
            "p2",
            ModelWeights::from_vecs(vec![0.0, 2.0], vec![1.0]),
            50,
            0.5,
        ),
    ];
    let result = trainer.federated_averaging(&updates);
    // With equal samples each participant contributes 0.5
    // delta_w = 0.5*[2,0] + 0.5*[0,2] = [1,1]
    // global is zeros → new global = [1,1]
    assert!((result.weights[0] - 1.0).abs() < 1e-9);
    assert!((result.weights[1] - 1.0).abs() < 1e-9);
}

#[test]
fn test_federated_averaging_weighted_by_samples() {
    let trainer = FederatedShapeTrainer::new(1, 0);
    let updates = vec![
        LocalUpdate::new("p1", ModelWeights::from_vecs(vec![3.0], vec![]), 100, 0.3),
        LocalUpdate::new("p2", ModelWeights::from_vecs(vec![1.0], vec![]), 100, 0.7),
    ];
    let result = trainer.federated_averaging(&updates);
    // delta = (100/200)*3 + (100/200)*1 = 1.5 + 0.5 = 2.0
    assert!((result.weights[0] - 2.0).abs() < 1e-9);
}

#[test]
fn test_federated_averaging_empty_updates() {
    let trainer =
        FederatedShapeTrainer::with_initial_model(ModelWeights::from_vecs(vec![5.0], vec![1.0]));
    let result = trainer.federated_averaging(&[]);
    // Should return the current global model unchanged
    assert!((result.weights[0] - 5.0).abs() < 1e-9);
}

#[test]
fn test_execute_round_updates_global_model() {
    let mut trainer = FederatedShapeTrainer::new(2, 1);
    let mut round = FederatedRound::new(0, trainer.global_model().clone());
    round.add_update(LocalUpdate::new(
        "p1",
        ModelWeights::from_vecs(vec![1.0, 0.5], vec![0.1]),
        80,
        0.4,
    ));
    round.add_update(LocalUpdate::new(
        "p2",
        ModelWeights::from_vecs(vec![0.5, 1.0], vec![0.2]),
        120,
        0.6,
    ));
    trainer.execute_round(round);
    assert_eq!(trainer.rounds_completed(), 1);
    // Global model should have changed
    assert!(trainer.global_model().norm() > 0.0);
}

#[test]
fn test_version_increments_per_round() {
    let mut trainer = FederatedShapeTrainer::new(2, 1);
    assert_eq!(trainer.global_model().version, 0);
    for r in 0..3 {
        let mut round = FederatedRound::new(r, trainer.global_model().clone());
        round.add_update(LocalUpdate::new("p1", ModelWeights::zeros(2, 1), 10, 0.5));
        trainer.execute_round(round);
    }
    assert_eq!(trainer.global_model().version, 3);
}

#[test]
fn test_average_round_loss_no_rounds() {
    let trainer = FederatedShapeTrainer::new(2, 1);
    assert_eq!(trainer.average_round_loss(), 0.0);
}

#[test]
fn test_average_round_loss_after_rounds() {
    let mut trainer = FederatedShapeTrainer::new(1, 0);
    for loss in [0.8, 0.6, 0.4] {
        let mut round = FederatedRound::new(0, trainer.global_model().clone());
        round.add_update(LocalUpdate::new("p1", ModelWeights::zeros(1, 0), 10, loss));
        trainer.execute_round(round);
    }
    // Average of 0.8, 0.6, 0.4 = 0.6
    let avg = trainer.average_round_loss();
    assert!((avg - 0.6).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// GradientPrivacy
// ---------------------------------------------------------------------------

#[test]
fn test_clip_gradients_below_max_norm_unchanged() {
    let mut grad = vec![3.0, 4.0]; // norm = 5.0
    GradientPrivacy::clip_gradients(&mut grad, 10.0);
    // norm < max_norm → unchanged
    assert!((grad[0] - 3.0).abs() < 1e-9);
    assert!((grad[1] - 4.0).abs() < 1e-9);
}

#[test]
fn test_clip_gradients_above_max_norm_scaled() {
    let mut grad = vec![3.0, 4.0]; // norm = 5.0
    GradientPrivacy::clip_gradients(&mut grad, 2.5);
    let new_norm: f64 = grad.iter().map(|&v| v * v).sum::<f64>().sqrt();
    assert!((new_norm - 2.5).abs() < 1e-6);
}

#[test]
fn test_clip_gradients_zero_vector() {
    let mut grad = vec![0.0, 0.0];
    GradientPrivacy::clip_gradients(&mut grad, 1.0);
    assert_eq!(grad, vec![0.0, 0.0]);
}

#[test]
fn test_compute_noise_scale_positive() {
    let sigma = GradientPrivacy::compute_noise_scale(1.0, 1.0, 1e-5);
    assert!(sigma > 0.0 && sigma.is_finite());
}

#[test]
fn test_compute_noise_scale_large_epsilon() {
    // Larger epsilon → smaller noise
    let sigma_small_eps = GradientPrivacy::compute_noise_scale(1.0, 0.1, 1e-5);
    let sigma_large_eps = GradientPrivacy::compute_noise_scale(1.0, 10.0, 1e-5);
    assert!(sigma_large_eps < sigma_small_eps);
}

#[test]
fn test_compute_noise_scale_invalid_inputs() {
    assert!(GradientPrivacy::compute_noise_scale(1.0, 0.0, 1e-5).is_infinite());
    assert!(GradientPrivacy::compute_noise_scale(1.0, -1.0, 1e-5).is_infinite());
    assert!(GradientPrivacy::compute_noise_scale(1.0, 1.0, 0.0).is_infinite());
    assert!(GradientPrivacy::compute_noise_scale(1.0, 1.0, 1.5).is_infinite());
}

#[test]
fn test_add_gaussian_noise_changes_gradient() {
    let original = vec![1.0, 2.0, 3.0];
    let mut grad = original.clone();
    GradientPrivacy::add_gaussian_noise(&mut grad, 1.0, 1.0, 1e-5);
    // With noise added, at least one element should differ
    let changed = grad
        .iter()
        .zip(original.iter())
        .any(|(&a, &b)| (a - b).abs() > 1e-10);
    assert!(changed, "noise should modify gradient values");
}

#[test]
fn test_add_gaussian_noise_zero_epsilon_no_panic() {
    let mut grad = vec![1.0, 2.0];
    // epsilon=0 → sigma=inf → no-op (early return)
    GradientPrivacy::add_gaussian_noise(&mut grad, 1.0, 0.0, 1e-5);
    // Should not panic; values may remain unchanged
}

#[test]
fn test_clip_then_noise_pipeline() {
    let mut grad = vec![100.0, 200.0, 300.0];
    // First clip to reasonable norm
    GradientPrivacy::clip_gradients(&mut grad, 1.0);
    let norm_after_clip: f64 = grad.iter().map(|&v| v * v).sum::<f64>().sqrt();
    assert!((norm_after_clip - 1.0).abs() < 1e-6);
    // Then add noise
    GradientPrivacy::add_gaussian_noise(&mut grad, 1.0, 2.0, 1e-5);
    // Should not panic; gradient values altered
}

#[test]
fn test_local_update_serialization() {
    let update = LocalUpdate::new(
        "participant-1",
        ModelWeights::from_vecs(vec![0.1, 0.2], vec![0.05]),
        42,
        0.35,
    );
    let json = serde_json::to_string(&update).expect("serialize ok");
    let back: LocalUpdate = serde_json::from_str(&json).expect("deserialize ok");
    assert_eq!(back.participant_id, "participant-1");
    assert_eq!(back.sample_count, 42);
    assert!((back.local_loss - 0.35).abs() < 1e-9);
}

#[test]
fn test_federated_round_serialization() {
    let global = ModelWeights::from_vecs(vec![1.0, 2.0], vec![0.5]);
    let mut round = FederatedRound::new(3, global);
    round.add_update(LocalUpdate::new("p", ModelWeights::zeros(2, 1), 10, 0.2));
    let json = serde_json::to_string(&round).expect("serialize ok");
    let back: FederatedRound = serde_json::from_str(&json).expect("deserialize ok");
    assert_eq!(back.round_id, 3);
    assert_eq!(back.local_updates.len(), 1);
}
