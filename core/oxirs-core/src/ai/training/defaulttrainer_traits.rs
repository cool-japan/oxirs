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
        tracing::warn!(
            "Checkpoint loaded successfully, but model parameter restoration is not yet implemented. \
             To fully resume training, implement model state serialization/deserialization."
        );
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
