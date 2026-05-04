//! Joint training scaffold for GNN encoder + LLM head soft-prompt projector.
//!
//! # Design note
//!
//! `JointTrainer` takes `GraphSageEncoder` **by value** (not wrapped in `Arc`)
//! so that it can call `encoder.train()` which requires `&mut self`.  The
//! `HybridLlmHead` wraps the encoder in `Arc` only after training is complete
//! (the "done training → frozen → inference" workflow).
//!
//! When the GNN is frozen (`gnn_frozen == true`), the encoder's `train()` call
//! is skipped entirely.  When it is unfrozen, we call
//! `encoder.train(kg, 1).map_err(|e| e.to_string())?` once per epoch so that
//! the encoder weights receive a genuine gradient step.  Because the toy config
//! sets `learning_rate = 0.0`, the call acts as a forward-only pass in the
//! demo and tests, but the hook is real and will fire with any non-zero lr.
//!
//! The `gnn_grad_norm` field in [`EpochMetrics`] is populated as the L2 norm
//! of the upstream gradient flowing from the projector backward pass into the
//! GNN output space — a proxy that is always available without requiring
//! end-to-end autograd.

use scirs2_core::ndarray_ext::Array2;

use crate::gnn_encoder::{GraphSageEncoder, KgGraph};
use crate::hybrid::llm_head::KgqaExample;
use crate::hybrid::provider::LlmProvider;
use crate::hybrid::soft_prompt::SoftPromptProjector;

// ─── Schedule ────────────────────────────────────────────────────────────────

/// Training schedule for the joint training loop.
#[derive(Debug, Clone)]
pub enum Schedule {
    /// Alternate between GNN and projector training each epoch.
    ///
    /// When `gnn_first == true`, epoch 0 trains the GNN (projector frozen),
    /// epoch 1 trains the projector (GNN frozen), and so on.
    AlternateEpoch { gnn_first: bool },
    /// Train only the projector for `warmup_epochs`, then train both jointly.
    Curriculum {
        warmup_epochs: usize,
        joint_epochs: usize,
    },
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

/// Metrics recorded for one training epoch.
#[derive(Debug, Clone, Default)]
pub struct EpochMetrics {
    /// Zero-based epoch index.
    pub epoch: usize,
    /// Mean MSE loss across all examples.
    pub loss: f64,
    /// L2 norm of the gradient proxy at the GNN output boundary.
    /// Zero when the GNN is frozen.
    pub gnn_grad_norm: f64,
    /// L2 norm of the upstream gradient flowing into the projector backward.
    /// Zero when the projector is frozen.
    pub projector_grad_norm: f64,
    /// Whether the GNN encoder was frozen during this epoch.
    pub gnn_frozen: bool,
    /// Whether the soft-prompt projector was frozen during this epoch.
    pub projector_frozen: bool,
}

/// Complete training history returned by [`JointTrainer::train`].
#[derive(Debug, Default)]
pub struct TrainingHistory {
    /// Per-epoch metrics, one entry per epoch.
    pub epochs: Vec<EpochMetrics>,
    /// Loss from the last epoch (convenience copy).
    pub final_loss: f64,
}

// ─── JointTrainer ─────────────────────────────────────────────────────────────

/// Controls joint training of the GNN encoder and LLM head projector.
///
/// Takes ownership of the `GraphSageEncoder` so that it can call
/// `encoder.train()` — which requires `&mut self` — without wrapping in
/// `Arc<Mutex<…>>`.
pub struct JointTrainer<P: LlmProvider> {
    encoder: GraphSageEncoder,
    projector: SoftPromptProjector,
    /// LLM provider — held here to mirror `HybridLlmHead`; not called during
    /// training but required to support future gradient-capable providers.
    #[allow(dead_code)]
    provider: P,
    schedule: Schedule,
    gnn_frozen: bool,
    projector_frozen: bool,
    gnn_learning_rate: f64,
    projector_learning_rate: f64,
}

impl<P: LlmProvider> JointTrainer<P> {
    /// Create a new joint trainer.
    pub fn new(
        encoder: GraphSageEncoder,
        projector: SoftPromptProjector,
        provider: P,
        schedule: Schedule,
    ) -> Self {
        Self {
            encoder,
            projector,
            provider,
            schedule,
            gnn_frozen: false,
            projector_frozen: false,
            gnn_learning_rate: 0.01,
            projector_learning_rate: 0.01,
        }
    }

    /// Override learning rates (builder-style).
    pub fn with_learning_rates(mut self, gnn_lr: f64, proj_lr: f64) -> Self {
        self.gnn_learning_rate = gnn_lr;
        self.projector_learning_rate = proj_lr;
        self
    }

    // ─── Freeze toggles ───────────────────────────────────────────────────

    /// Set whether the GNN encoder is frozen (no parameter updates).
    pub fn freeze_gnn(&mut self, frozen: bool) {
        self.gnn_frozen = frozen;
    }

    /// Set whether the soft-prompt projector is frozen.
    pub fn freeze_projector(&mut self, frozen: bool) {
        self.projector_frozen = frozen;
    }

    /// Return the current GNN freeze state.
    pub fn is_gnn_frozen(&self) -> bool {
        self.gnn_frozen
    }

    /// Return the current projector freeze state.
    pub fn is_projector_frozen(&self) -> bool {
        self.projector_frozen
    }

    // ─── Main training loop ───────────────────────────────────────────────

    /// Run the joint training loop for `total_epochs` epochs.
    ///
    /// Returns a [`TrainingHistory`] with per-epoch metrics.
    pub fn train(
        &mut self,
        kg: &KgGraph,
        examples: &[KgqaExample],
        total_epochs: usize,
    ) -> Result<TrainingHistory, String> {
        let mut history = TrainingHistory::default();

        let freeze_schedule = self.build_freeze_schedule(total_epochs);

        for (epoch, (freeze_gnn, freeze_projector)) in freeze_schedule.into_iter().enumerate() {
            self.gnn_frozen = freeze_gnn;
            self.projector_frozen = freeze_projector;

            // GNN training step (when unfrozen) — one epoch of link-prediction.
            // With learning_rate == 0.0 this is effectively a no-op parameter update,
            // but the call is real and would fire with any non-zero lr.
            if !self.gnn_frozen {
                self.encoder.train(kg, 1).map_err(|e| e.to_string())?;
            }

            let metrics = self.train_projector_epoch(epoch, kg, examples)?;
            history.epochs.push(metrics);
        }

        history.final_loss = history.epochs.last().map(|m| m.loss).unwrap_or(0.0);
        Ok(history)
    }

    // ─── Internal helpers ─────────────────────────────────────────────────

    /// Build the per-epoch `(freeze_gnn, freeze_projector)` schedule.
    fn build_freeze_schedule(&self, total_epochs: usize) -> Vec<(bool, bool)> {
        match &self.schedule {
            Schedule::AlternateEpoch { gnn_first } => (0..total_epochs)
                .map(|epoch| {
                    // When gnn_first: epoch 0 → train GNN (projector frozen),
                    //                 epoch 1 → train projector (GNN frozen), …
                    let train_gnn_this_epoch = if *gnn_first {
                        epoch % 2 == 0
                    } else {
                        epoch % 2 == 1
                    };
                    let freeze_gnn = !train_gnn_this_epoch;
                    let freeze_proj = train_gnn_this_epoch;
                    (freeze_gnn, freeze_proj)
                })
                .collect(),

            Schedule::Curriculum {
                warmup_epochs,
                joint_epochs: _,
            } => (0..total_epochs)
                .map(|epoch| {
                    if epoch < *warmup_epochs {
                        // Warmup: only train projector; GNN frozen.
                        (true, false)
                    } else {
                        // Joint: both unfrozen.
                        (false, false)
                    }
                })
                .collect(),
        }
    }

    /// Run one projector epoch: forward, loss, backward, optional weight update.
    fn train_projector_epoch(
        &mut self,
        epoch: usize,
        kg: &KgGraph,
        examples: &[KgqaExample],
    ) -> Result<EpochMetrics, String> {
        let embeddings = self.encoder.encode(kg).map_err(|e| e.to_string())?;

        let mut total_loss = 0.0_f64;
        let mut gnn_grad_norm_acc = 0.0_f64;
        let mut projector_grad_norm_acc = 0.0_f64;

        for ex in examples {
            // Collect valid row indices; fall back to node 0 if none match.
            let rows: Vec<usize> = {
                let candidate: Vec<usize> = ex
                    .entity_ids
                    .iter()
                    .copied()
                    .filter(|&id| id < embeddings.embeddings.nrows())
                    .collect();
                if candidate.is_empty() {
                    vec![0]
                } else {
                    candidate
                }
            };
            let n = rows.len();
            let dim = embeddings.embeddings.ncols();

            // Build input matrix [n, dim].
            let mut input_data = vec![0.0_f64; n * dim];
            for (i, &row_idx) in rows.iter().enumerate() {
                for j in 0..dim {
                    input_data[i * dim + j] = embeddings.embeddings[[row_idx, j]];
                }
            }
            let input = Array2::from_shape_vec((n, dim), input_data).map_err(|e| e.to_string())?;

            // Forward through projector.
            let projected = self.projector.forward(&input);
            let prompt_dim = projected.ncols();

            // MSE loss toward unit-vector (first component = 1, rest = 0).
            let mut loss = 0.0_f64;
            let mut d_output: Array2<f64> = Array2::zeros((n, prompt_dim));
            let scale = (n * prompt_dim).max(1) as f64;
            for i in 0..n {
                for j in 0..prompt_dim {
                    let target = if j == 0 { 1.0_f64 } else { 0.0_f64 };
                    let diff = projected[[i, j]] - target;
                    loss += diff * diff;
                    d_output[[i, j]] = 2.0 * diff / scale;
                }
            }
            total_loss += loss / scale;

            // Projector backward + SGD update (lr = 0 when frozen).
            let proj_lr = if self.projector_frozen {
                0.0
            } else {
                self.projector_learning_rate
            };
            let d_input = self.projector.backward(&d_output, proj_lr);

            // Compute gradient norm at the GNN–projector boundary.
            let boundary_norm: f64 = {
                let mut sq_sum = 0.0_f64;
                for i in 0..n {
                    for j in 0..dim {
                        let v = d_input[[i, j]];
                        sq_sum += v * v;
                    }
                }
                sq_sum.sqrt()
            };

            // The boundary norm is a proxy for the upstream GNN gradient;
            // it is non-zero only when the projector is actively propagating gradients.
            if !self.gnn_frozen {
                gnn_grad_norm_acc += boundary_norm;
            }
            if !self.projector_frozen {
                projector_grad_norm_acc += boundary_norm;
            }
        }

        let n_ex = examples.len().max(1) as f64;
        Ok(EpochMetrics {
            epoch,
            loss: total_loss / n_ex,
            gnn_grad_norm: gnn_grad_norm_acc / n_ex,
            projector_grad_norm: projector_grad_norm_acc / n_ex,
            gnn_frozen: self.gnn_frozen,
            projector_frozen: self.projector_frozen,
        })
    }
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gnn_encoder::{GraphSageConfig, GraphSageEncoder, KgGraph};
    use crate::hybrid::provider::LocalProvider;
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
    fn test_freeze_toggle() {
        let mut trainer = make_trainer(Schedule::AlternateEpoch { gnn_first: true });
        trainer.freeze_gnn(true);
        assert!(trainer.is_gnn_frozen());
        trainer.freeze_gnn(false);
        assert!(!trainer.is_gnn_frozen());
        trainer.freeze_projector(true);
        assert!(trainer.is_projector_frozen());
        trainer.freeze_projector(false);
        assert!(!trainer.is_projector_frozen());
    }

    #[test]
    fn test_alternate_schedule_gnn_first() {
        let mut trainer = make_trainer(Schedule::AlternateEpoch { gnn_first: true });
        let history = trainer.train(&toy_kg(), &toy_examples(), 4).expect("train");
        assert_eq!(history.epochs.len(), 4);
        assert!(
            !history.epochs[0].gnn_frozen,
            "epoch 0 GNN should not be frozen"
        );
        assert!(
            history.epochs[0].projector_frozen,
            "epoch 0 projector should be frozen"
        );
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
        let history = trainer.train(&toy_kg(), &toy_examples(), 5).expect("train");
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
        let history = trainer.train(&toy_kg(), &toy_examples(), 2).expect("train");
        assert_eq!(
            history.epochs[0].projector_grad_norm, 0.0,
            "frozen projector should have zero grad norm"
        );
    }

    #[test]
    fn test_history_epoch_count() {
        let mut trainer = make_trainer(Schedule::Curriculum {
            warmup_epochs: 2,
            joint_epochs: 3,
        });
        let history = trainer.train(&toy_kg(), &toy_examples(), 5).expect("train");
        assert_eq!(history.epochs.len(), 5);
    }
}
