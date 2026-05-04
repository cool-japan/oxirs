//! Integration tests for the joint GNN+LLM training scaffold (phase c).

use oxirs_graphrag::gnn_encoder::{GraphSageConfig, GraphSageEncoder, KgGraph};
use oxirs_graphrag::hybrid::{
    JointTrainer, KgqaExample, LocalProvider, Schedule, SoftPromptProjector,
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

fn toy_examples() -> Vec<KgqaExample> {
    (0..4)
        .map(|i| KgqaExample {
            question: format!("q{i}"),
            answer: format!("a{i}"),
            entity_ids: vec![i % 4],
        })
        .collect()
}

fn make_trainer(schedule: Schedule) -> JointTrainer<LocalProvider> {
    let encoder = GraphSageEncoder::new_with_seed(&toy_config(), 1).expect("construct encoder");
    let projector = SoftPromptProjector::new(8, 8, 42);
    JointTrainer::new(encoder, projector, LocalProvider::new(), schedule)
}

#[test]
fn test_freeze_gnn_toggle() {
    let mut trainer = make_trainer(Schedule::AlternateEpoch { gnn_first: true });
    trainer.freeze_gnn(true);
    assert!(trainer.is_gnn_frozen());
    trainer.freeze_gnn(false);
    assert!(!trainer.is_gnn_frozen());
}

#[test]
fn test_freeze_projector_toggle() {
    let mut trainer = make_trainer(Schedule::AlternateEpoch { gnn_first: true });
    trainer.freeze_projector(true);
    assert!(trainer.is_projector_frozen());
}

#[test]
fn test_alternate_schedule_gnn_first() {
    let mut trainer = make_trainer(Schedule::AlternateEpoch { gnn_first: true });
    let history = trainer
        .train(&toy_kg(), &toy_examples(), 4)
        .expect("should train");
    assert_eq!(history.epochs.len(), 4);
    // Epoch 0: GNN trains (gnn_first), projector frozen.
    assert!(
        !history.epochs[0].gnn_frozen,
        "epoch 0 GNN should not be frozen"
    );
    assert!(
        history.epochs[0].projector_frozen,
        "epoch 0 projector should be frozen"
    );
    // Epoch 1: projector trains, GNN frozen.
    assert!(history.epochs[1].gnn_frozen, "epoch 1 GNN should be frozen");
    assert!(
        !history.epochs[1].projector_frozen,
        "epoch 1 projector should not be frozen"
    );
}

#[test]
fn test_curriculum_warmup() {
    let mut trainer = make_trainer(Schedule::Curriculum {
        warmup_epochs: 3,
        joint_epochs: 2,
    });
    let history = trainer
        .train(&toy_kg(), &toy_examples(), 5)
        .expect("should train");
    // Warmup epochs: GNN frozen, projector trains.
    for epoch in 0..3 {
        assert!(
            history.epochs[epoch].gnn_frozen,
            "warmup epoch {epoch} GNN should be frozen"
        );
        assert!(
            !history.epochs[epoch].projector_frozen,
            "warmup epoch {epoch} projector should train"
        );
    }
    // Joint epochs: both unfrozen.
    for epoch in 3..5 {
        assert!(
            !history.epochs[epoch].gnn_frozen,
            "joint epoch {epoch} GNN should train"
        );
        assert!(
            !history.epochs[epoch].projector_frozen,
            "joint epoch {epoch} projector should train"
        );
    }
}

#[test]
fn test_frozen_projector_grad_norm_zero() {
    let mut trainer = make_trainer(Schedule::AlternateEpoch { gnn_first: true });
    let history = trainer
        .train(&toy_kg(), &toy_examples(), 2)
        .expect("should train");
    // Epoch 0: projector is frozen → grad norm must be 0.
    assert_eq!(
        history.epochs[0].projector_grad_norm, 0.0,
        "frozen projector should have zero grad norm"
    );
}

#[test]
fn test_joint_loss_over_50_epochs() {
    let encoder = GraphSageEncoder::new_with_seed(&toy_config(), 1).expect("construct encoder");
    let projector = SoftPromptProjector::new(8, 8, 42);
    let mut trainer = JointTrainer::new(
        encoder,
        projector,
        LocalProvider::new(),
        Schedule::AlternateEpoch { gnn_first: false },
    )
    .with_learning_rates(0.0, 0.05);

    let history = trainer
        .train(&toy_kg(), &toy_examples(), 50)
        .expect("should train");
    assert_eq!(history.epochs.len(), 50);
    let first = history.epochs[0].loss;
    let last = history.epochs[49].loss;
    // Alternating schedule may cause transient increases on GNN-only epochs;
    // allow a small tolerance.
    assert!(
        last <= first + 0.01,
        "loss should not significantly increase over 50 epochs: {} → {}",
        first,
        last
    );
}

#[test]
fn test_history_has_correct_epoch_count() {
    let mut trainer = make_trainer(Schedule::Curriculum {
        warmup_epochs: 2,
        joint_epochs: 3,
    });
    let history = trainer.train(&toy_kg(), &toy_examples(), 5).expect("train");
    assert_eq!(history.epochs.len(), 5);
}
