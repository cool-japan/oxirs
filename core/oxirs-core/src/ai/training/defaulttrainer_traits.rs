//! # DefaultTrainer - Trait Implementations
//!
//! This module contains trait implementations for `DefaultTrainer`.
//!
//! ## Implemented Traits
//!
//! - `Trainer`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use super::types::*;
use crate::ai::{GraphNeuralNetwork, KnowledgeGraphEmbedding};
use crate::Triple;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

#[async_trait::async_trait]
impl Trainer for DefaultTrainer {
    async fn train_embedding_model(
        &mut self,
        model: Arc<dyn KnowledgeGraphEmbedding>,
        training_data: &[Triple],
        validation_data: &[Triple],
    ) -> Result<TrainingMetrics> {
        let mut metrics = TrainingMetrics::new();
        let start_time = Instant::now();
        let batch_size = self.config.batch_size;
        let num_batches = (training_data.len() + batch_size - 1) / batch_size;
        for epoch in 0..self.config.max_epochs {
            let epoch_start = Instant::now();
            let mut epoch_loss = 0.0;
            for batch_idx in 0..num_batches {
                let start_idx = batch_idx * batch_size;
                let end_idx = (start_idx + batch_size).min(training_data.len());
                let batch = &training_data[start_idx..end_idx];
                let _negatives = self.generate_negative_samples(batch, 1.0);
                let mut positive_scores = Vec::new();
                let negative_scores = Vec::new();
                for triple in batch {
                    if let Ok(score) = model
                        .score_triple(
                            &triple.subject().to_string(),
                            &triple.predicate().to_string(),
                            &triple.object().to_string(),
                        )
                        .await
                    {
                        positive_scores.push(score);
                    }
                }
                let batch_loss = self.compute_loss(&positive_scores, &negative_scores);
                epoch_loss += batch_loss;
                if let Err(e) = self
                    .backward_pass(&positive_scores, &negative_scores, model.as_ref())
                    .await
                {
                    tracing::warn!("Backward pass failed: {}", e);
                    continue;
                }
                if let Some(clip_value) = self.config.gradient_clipping {
                    self.clip_gradients(model.as_ref(), clip_value).await;
                }
                self.update_parameters(model.as_ref(), epoch as f32).await;
            }
            epoch_loss /= num_batches as f32;
            let val_loss = if epoch % self.config.validation.validation_frequency == 0 {
                let mut val_loss = 0.0;
                let val_batches = (validation_data.len() + batch_size - 1) / batch_size;
                for batch_idx in 0..val_batches {
                    let start_idx = batch_idx * batch_size;
                    let end_idx = (start_idx + batch_size).min(validation_data.len());
                    let batch = &validation_data[start_idx..end_idx];
                    for triple in batch {
                        if let Ok(score) = model
                            .score_triple(
                                &triple.subject().to_string(),
                                &triple.predicate().to_string(),
                                &triple.object().to_string(),
                            )
                            .await
                        {
                            val_loss += score;
                        }
                    }
                }
                Some(val_loss / validation_data.len() as f32)
            } else {
                None
            };
            let epoch_time = epoch_start.elapsed();
            let train_accuracy = if epoch % self.config.logging.log_frequency == 0 {
                Some(
                    self.compute_accuracy(training_data, model.as_ref())
                        .await
                        .unwrap_or(0.0),
                )
            } else {
                None
            };
            let val_accuracy = if epoch % self.config.validation.validation_frequency == 0 {
                Some(
                    self.compute_accuracy(validation_data, model.as_ref())
                        .await
                        .unwrap_or(0.0),
                )
            } else {
                None
            };
            metrics.update_epoch(
                epoch,
                epoch_loss,
                val_loss,
                train_accuracy,
                val_accuracy,
                self.current_lr,
                epoch_time,
            );
            if let Some(val_loss) = val_loss {
                if self.check_early_stopping(val_loss) {
                    metrics.early_stopped = true;
                    break;
                }
            }
            self.update_learning_rate(epoch, val_loss);
            if epoch % self.config.logging.log_frequency == 0 {
                println!(
                    "Epoch {}: train_loss={:.4}, val_loss={:.4}, lr={:.6}",
                    epoch,
                    epoch_loss,
                    val_loss.unwrap_or(0.0),
                    self.current_lr
                );
            }
        }
        metrics.total_time = start_time.elapsed();
        Ok(metrics)
    }
    async fn train_gnn(
        &mut self,
        model: Arc<dyn GraphNeuralNetwork>,
        training_data: &[Triple],
        _validation_data: &[Triple],
    ) -> Result<TrainingMetrics> {
        use crate::ai::gnn::{RdfGraph, TrainingConfig as GnnTrainingConfig};
        let gnn_config = GnnTrainingConfig {
            max_epochs: self.config.max_epochs,
            batch_size: self.config.batch_size,
            learning_rate: self.config.learning_rate,
            patience: self.config.early_stopping.patience,
            validation_split: self.config.validation.validation_split,
            loss_function: match &self.config.loss_function {
                LossFunction::CrossEntropy => crate::ai::gnn::LossFunction::CrossEntropy,
                LossFunction::BinaryCrossEntropy => {
                    crate::ai::gnn::LossFunction::BinaryCrossEntropy
                }
                LossFunction::MeanSquaredError => crate::ai::gnn::LossFunction::MeanSquaredError,
                LossFunction::ContrastiveLoss { margin } => {
                    crate::ai::gnn::LossFunction::ContrastiveLoss { margin: *margin }
                }
                _ => crate::ai::gnn::LossFunction::CrossEntropy,
            },
            optimizer: match &self.config.optimizer {
                Optimizer::SGD {
                    momentum,
                    weight_decay,
                    nesterov,
                } => crate::ai::gnn::Optimizer::SGD {
                    momentum: *momentum,
                    weight_decay: *weight_decay,
                    nesterov: *nesterov,
                },
                Optimizer::Adam {
                    beta1,
                    beta2,
                    epsilon,
                    weight_decay,
                } => crate::ai::gnn::Optimizer::Adam {
                    beta1: *beta1,
                    beta2: *beta2,
                    epsilon: *epsilon,
                    weight_decay: *weight_decay,
                },
                Optimizer::AdaGrad {
                    epsilon,
                    weight_decay: _,
                } => crate::ai::gnn::Optimizer::AdaGrad { epsilon: *epsilon },
                Optimizer::RMSprop {
                    alpha,
                    epsilon,
                    weight_decay: _,
                    momentum: _,
                } => crate::ai::gnn::Optimizer::RMSprop {
                    decay: *alpha,
                    epsilon: *epsilon,
                },
                _ => crate::ai::gnn::Optimizer::Adam {
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    weight_decay: 1e-4,
                },
            },
            gradient_clipping: self.config.gradient_clipping,
            early_stopping: crate::ai::gnn::EarlyStoppingConfig {
                enabled: self.config.early_stopping.enabled,
                patience: self.config.early_stopping.patience,
                min_delta: self.config.early_stopping.min_delta,
            },
        };
        let graph = RdfGraph::from_triples(training_data)?;
        let features = model.extract_node_features(&graph).await?;
        let labels = {
            use scirs2_core::ndarray_ext::Array2;
            let num_nodes = graph.num_nodes;
            let output_dim = 64;
            let mut labels = Array2::zeros((num_nodes, output_dim));
            for i in 0..num_nodes.min(output_dim) {
                labels[[i, i]] = 1.0;
            }
            labels
        };
        let mut model_mut = model;
        let gnn_metrics = if let Some(model_ref) = Arc::get_mut(&mut model_mut) {
            model_ref
                .train(&graph, &features, &labels, &gnn_config)
                .await?
        } else {
            return Err(anyhow!(
                "Cannot train GNN: model has multiple references. \
                 Clone the model or ensure exclusive ownership before training."
            ));
        };
        let mut metrics = TrainingMetrics::new();
        metrics.update_epoch(
            0,
            gnn_metrics.loss,
            Some(gnn_metrics.loss),
            Some(gnn_metrics.accuracy),
            Some(gnn_metrics.accuracy),
            self.config.learning_rate,
            gnn_metrics.time_elapsed,
        );
        metrics.final_epoch = gnn_metrics.epochs;
        metrics.total_time = gnn_metrics.time_elapsed;
        Ok(metrics)
    }
    async fn resume_training(
        &mut self,
        checkpoint_path: &str,
        model: Arc<dyn KnowledgeGraphEmbedding>,
        _training_data: &[Triple],
        _validation_data: &[Triple],
    ) -> Result<TrainingMetrics> {
        use std::path::Path;
        let path = Path::new(checkpoint_path);
        if !path.exists() {
            return Err(anyhow!("Checkpoint file not found: {}", checkpoint_path));
        }
        let checkpoint_data = std::fs::read_to_string(checkpoint_path)?;
        let checkpoint: CheckpointData = serde_json::from_str(&checkpoint_data).map_err(|e| {
            anyhow!(
                "Failed to parse checkpoint file {}: {}. \
                 Expected JSON format with model state and training progress.",
                checkpoint_path,
                e
            )
        })?;
        tracing::info!(
            "Loaded checkpoint from epoch {}, best validation score: {:.6}",
            checkpoint.epoch,
            checkpoint.best_val_score
        );
        self.current_lr = checkpoint.current_lr;
        self.early_stopping_state = EarlyStoppingState {
            best_score: checkpoint.best_val_score,
            patience_counter: 0,
            should_stop: false,
        };
        let metrics = TrainingMetrics {
            train_loss: checkpoint.train_loss_history.clone(),
            val_loss: checkpoint.val_loss_history.clone(),
            train_accuracy: checkpoint.train_accuracy_history.clone(),
            val_accuracy: checkpoint.val_accuracy_history.clone(),
            learning_rate: checkpoint.lr_history.clone(),
            epoch_times: checkpoint
                .epoch_times_ms
                .iter()
                .map(|&ms| Duration::from_millis(ms))
                .collect(),
            best_val_score: checkpoint.best_val_score,
            best_epoch: checkpoint.best_epoch,
            total_time: Duration::from_millis(checkpoint.total_time_ms),
            final_epoch: checkpoint.epoch,
            early_stopped: false,
            additional_metrics: checkpoint.additional_metrics.clone(),
        };
        tracing::info!(
            "Resuming training from epoch {} with {} remaining epochs",
            checkpoint.epoch + 1,
            self.config.max_epochs.saturating_sub(checkpoint.epoch + 1)
        );

        // Restore the actual model parameters. Without this, "resuming"
        // would silently restart optimization from randomly-initialized
        // weights while reporting the prior run's metrics/LR schedule as
        // if training had truly continued - fail loudly instead of
        // fabricating a successful resume.
        match &checkpoint.model_state {
            Some(state_bytes) => {
                let mut model_mut = model;
                let model_ref = Arc::get_mut(&mut model_mut).ok_or_else(|| {
                    anyhow!(
                        "Cannot restore model parameters from checkpoint {}: the model \
                         handle has other outstanding references. Pass an exclusively-owned \
                         model into resume_training() (clone/drop other references first).",
                        checkpoint_path
                    )
                })?;

                let temp_path = std::env::temp_dir().join(format!(
                    "oxirs-resume-checkpoint-{}-{}.bin",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_nanos())
                        .unwrap_or(0)
                ));
                std::fs::write(&temp_path, state_bytes).map_err(|e| {
                    anyhow!(
                        "Failed to stage checkpoint model state for restoration: {}",
                        e
                    )
                })?;
                let load_result = model_ref.load(temp_path.to_string_lossy().as_ref()).await;
                let _ = std::fs::remove_file(&temp_path);
                load_result.map_err(|e| {
                    anyhow!(
                        "Failed to restore model parameters from checkpoint {}: {}",
                        checkpoint_path,
                        e
                    )
                })?;
                tracing::info!(
                    "Restored model parameters from checkpoint model_state ({} bytes)",
                    state_bytes.len()
                );
            }
            None => {
                return Err(anyhow!(
                    "Checkpoint {} has no `model_state`; cannot resume training without the \
                     model's parameters. Save checkpoints with `model_state` populated (e.g. \
                     serialize the model via KnowledgeGraphEmbedding::save into the checkpoint) \
                     to support resume_training().",
                    checkpoint_path
                ));
            }
        }

        Ok(metrics)
    }
    async fn evaluate(
        &self,
        model: Arc<dyn KnowledgeGraphEmbedding>,
        test_data: &[Triple],
        _metrics: &[TrainingMetric],
    ) -> Result<HashMap<String, f32>> {
        let computed_metrics = self.compute_metrics(test_data, model.as_ref()).await?;
        Ok(computed_metrics)
    }
}

#[cfg(test)]
mod resume_training_tests {
    use super::*;
    use crate::ai::embeddings::simple::SimplE;
    use crate::ai::embeddings::EmbeddingConfig;
    use crate::ai::training::types::CheckpointData;
    use crate::model::{Literal, NamedNode};

    fn unique_temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "oxirs-core-resume-training-test-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            name
        ))
    }

    fn sample_triples() -> Vec<Triple> {
        vec![Triple::new(
            NamedNode::new("http://example.org/alice").expect("valid IRI"),
            NamedNode::new("http://example.org/knows").expect("valid IRI"),
            Literal::new("bob"),
        )]
    }

    fn empty_checkpoint(model_state: Option<Vec<u8>>) -> CheckpointData {
        CheckpointData {
            epoch: 0,
            current_lr: 0.001,
            best_val_score: 0.0,
            best_epoch: 0,
            train_loss_history: Vec::new(),
            val_loss_history: Vec::new(),
            train_accuracy_history: Vec::new(),
            val_accuracy_history: Vec::new(),
            lr_history: Vec::new(),
            epoch_times_ms: Vec::new(),
            total_time_ms: 0,
            model_state,
            optimizer_state: None,
            additional_metrics: HashMap::new(),
            config: TrainingConfig::default(),
        }
    }

    #[tokio::test]
    async fn test_resume_training_restores_model_state() {
        // Train a small model, save its state into a checkpoint's
        // `model_state`, mutate the in-memory model so it no longer
        // matches, then verify resume_training() restores it.
        let triples = sample_triples();
        let mut model = SimplE::new(EmbeddingConfig {
            embedding_dim: 4,
            ..Default::default()
        });
        model
            .train(&triples, &crate::ai::embeddings::TrainingConfig::default())
            .await
            .expect("initial training succeeds");

        let saved_model_path = unique_temp_path("model.json");
        model
            .save(saved_model_path.to_str().expect("valid utf8 path"))
            .await
            .expect("save succeeds");
        let model_state = std::fs::read(&saved_model_path).expect("read back saved model bytes");
        let _ = std::fs::remove_file(&saved_model_path);

        let checkpoint = empty_checkpoint(Some(model_state));
        let checkpoint_json = serde_json::to_string(&checkpoint).expect("checkpoint serializes");
        let checkpoint_path = unique_temp_path("checkpoint.json");
        std::fs::write(&checkpoint_path, checkpoint_json).expect("write checkpoint");

        // Fresh, untrained model instance standing in for "resuming into".
        let fresh_model: Arc<dyn KnowledgeGraphEmbedding> =
            Arc::new(SimplE::new(EmbeddingConfig {
                embedding_dim: 4,
                ..Default::default()
            }));

        let mut trainer = DefaultTrainer::new(TrainingConfig::default());
        let result = trainer
            .resume_training(
                checkpoint_path.to_str().expect("valid utf8 path"),
                fresh_model,
                &triples,
                &triples,
            )
            .await;

        let _ = std::fs::remove_file(&checkpoint_path);
        result.expect("resume_training should restore model parameters and succeed");
    }

    #[tokio::test]
    async fn test_resume_training_fails_loudly_without_model_state() {
        // A checkpoint with no serialized model_state must not silently
        // "succeed" - resume_training() must fail rather than fabricate
        // a resumed-training result with un-restored parameters.
        let checkpoint = empty_checkpoint(None);
        assert!(checkpoint.model_state.is_none());
        let checkpoint_json = serde_json::to_string(&checkpoint).expect("checkpoint serializes");
        let checkpoint_path = unique_temp_path("checkpoint_no_state.json");
        std::fs::write(&checkpoint_path, checkpoint_json).expect("write checkpoint");

        let model: Arc<dyn KnowledgeGraphEmbedding> =
            Arc::new(SimplE::new(EmbeddingConfig::default()));
        let mut trainer = DefaultTrainer::new(TrainingConfig::default());
        let triples = sample_triples();
        let result = trainer
            .resume_training(
                checkpoint_path.to_str().expect("valid utf8 path"),
                model,
                &triples,
                &triples,
            )
            .await;

        let _ = std::fs::remove_file(&checkpoint_path);
        assert!(
            result.is_err(),
            "resume_training must fail when the checkpoint has no model_state"
        );
    }

    #[tokio::test]
    async fn test_resume_training_fails_on_shared_model_reference() {
        // A non-exclusively-owned model handle cannot have its
        // parameters overwritten in place; resume_training() must
        // surface that as an error instead of quietly skipping restore.
        let model: Arc<dyn KnowledgeGraphEmbedding> =
            Arc::new(SimplE::new(EmbeddingConfig::default()));
        let _extra_reference = model.clone(); // keep a second Arc reference

        let checkpoint = empty_checkpoint(Some(vec![1, 2, 3]));
        let checkpoint_json = serde_json::to_string(&checkpoint).expect("checkpoint serializes");
        let checkpoint_path = unique_temp_path("checkpoint_shared.json");
        std::fs::write(&checkpoint_path, checkpoint_json).expect("write checkpoint");

        let mut trainer = DefaultTrainer::new(TrainingConfig::default());
        let triples = sample_triples();
        let result = trainer
            .resume_training(
                checkpoint_path.to_str().expect("valid utf8 path"),
                model,
                &triples,
                &triples,
            )
            .await;

        let _ = std::fs::remove_file(&checkpoint_path);
        assert!(
            result.is_err(),
            "resume_training must fail when the model handle is not exclusively owned"
        );
    }
}
