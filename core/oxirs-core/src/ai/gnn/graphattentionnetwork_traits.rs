//! # GraphAttentionNetwork - Trait Implementations
//!
//! This module contains trait implementations for `GraphAttentionNetwork`.
//!
//! ## Implemented Traits
//!
//! - `GraphNeuralNetwork`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use super::types::*;
use anyhow::{anyhow, Result};
use scirs2_core::ndarray_ext::{Array1, Array2};

#[async_trait::async_trait]
impl GraphNeuralNetwork for GraphAttentionNetwork {
    async fn forward(&self, graph: &RdfGraph, features: &Array2<f32>) -> Result<Array2<f32>> {
        let mut x = features.clone();
        let adj = graph.get_adjacency_matrix();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &adj)?;
            if layer_idx < self.layers.len() - 1 {
                x = apply_activation(&x, &self.config.activation);
                if !self.trained {
                    x = apply_dropout(&x, self.config.dropout);
                }
            }
        }
        Ok(x)
    }
    async fn train(
        &mut self,
        graph: &RdfGraph,
        features: &Array2<f32>,
        labels: &Array2<f32>,
        config: &TrainingConfig,
    ) -> Result<TrainingMetrics> {
        let start_time = std::time::Instant::now();
        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;
        tracing::info!("Starting GAT training for {} epochs", config.max_epochs);
        let mut momentum_buffers: Vec<Array2<f32>> = self
            .layers
            .iter()
            .flat_map(|layer| {
                layer
                    .attention_weights
                    .iter()
                    .map(|w| Array2::zeros(w.dim()))
            })
            .collect();
        let mut velocity_buffers: Vec<Array2<f32>> = self
            .layers
            .iter()
            .flat_map(|layer| {
                layer
                    .attention_weights
                    .iter()
                    .map(|w| Array2::zeros(w.dim()))
            })
            .collect();
        for epoch in 0..config.max_epochs {
            let predictions = self.forward(graph, features).await?;
            let loss = self.compute_loss(&predictions, labels)?;
            let gradients = self
                .compute_gradients(&predictions, labels, graph, features)
                .await?;
            let clipped_gradients = if let Some(clip_value) = config.gradient_clipping {
                self.clip_gradients(gradients, clip_value)
            } else {
                gradients
            };
            self.update_parameters(
                &clipped_gradients,
                &mut momentum_buffers,
                &mut velocity_buffers,
                config,
                epoch as f32 + 1.0,
            )?;
            if loss < best_loss {
                best_loss = loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= config.early_stopping.patience {
                    tracing::info!("Early stopping at epoch {}", epoch);
                    break;
                }
            }
            if epoch % 10 == 0 {
                let accuracy = self.compute_accuracy(&predictions, labels)?;
                tracing::info!(
                    "Epoch {}: loss={:.6}, accuracy={:.4}",
                    epoch,
                    loss,
                    accuracy
                );
            }
        }
        self.trained = true;
        let total_time = start_time.elapsed();
        Ok(TrainingMetrics {
            loss: best_loss,
            accuracy: self.compute_accuracy(&self.forward(graph, features).await?, labels)?,
            epochs: config.max_epochs,
            time_elapsed: total_time,
        })
    }
    async fn get_embeddings(
        &self,
        graph: &RdfGraph,
        features: &Array2<f32>,
    ) -> Result<Array2<f32>> {
        self.forward(graph, features).await
    }
    async fn predict_links(
        &self,
        graph: &RdfGraph,
        source_nodes: &[usize],
        target_nodes: &[usize],
    ) -> Result<Array1<f32>> {
        let features = self.extract_node_features(graph).await?;
        let embeddings = self.get_embeddings(graph, &features).await?;
        let mut predictions = Array1::zeros(source_nodes.len());
        for (i, (&src, &tgt)) in source_nodes.iter().zip(target_nodes.iter()).enumerate() {
            let src_emb = embeddings.row(src);
            let tgt_emb = embeddings.row(tgt);
            let score: f32 = src_emb
                .iter()
                .zip(tgt_emb.iter())
                .map(|(&a, &b)| a * b)
                .sum();
            predictions[i] = 1.0 / (1.0 + (-score).exp());
        }
        Ok(predictions)
    }
    fn get_parameters(&self) -> Result<Vec<Array2<f32>>> {
        Ok(self
            .layers
            .iter()
            .flat_map(|layer| layer.attention_weights.iter().cloned())
            .collect())
    }
    fn set_parameters(&mut self, parameters: &[Array2<f32>]) -> Result<()> {
        let mut param_idx = 0;
        for layer in &mut self.layers {
            for weight in &mut layer.attention_weights {
                if param_idx >= parameters.len() {
                    return Err(anyhow!("Not enough parameters provided"));
                }
                *weight = parameters[param_idx].clone();
                param_idx += 1;
            }
        }
        Ok(())
    }
    async fn extract_node_features(&self, graph: &RdfGraph) -> Result<Array2<f32>> {
        let node_count = graph.node_count();
        let feature_dim = self.config.input_dim;
        let mut features = Array2::zeros((node_count, feature_dim));
        for node_idx in graph.nodes() {
            let degree = graph.degree(node_idx) as f32;
            features[[node_idx, 0]] = degree.ln_1p();
            if feature_dim > 1 {
                features[[node_idx, 1]] =
                    graph.in_degree(node_idx).unwrap_or(0) as f32 / (degree + 1.0);
            }
        }
        Ok(features)
    }
    fn compute_loss(&self, predictions: &Array2<f32>, labels: &Array2<f32>) -> Result<f32> {
        let diff = predictions - labels;
        Ok((&diff * &diff).mean().unwrap_or(0.0))
    }
    async fn compute_gradients(
        &self,
        predictions: &Array2<f32>,
        labels: &Array2<f32>,
        _graph: &RdfGraph,
        _features: &Array2<f32>,
    ) -> Result<Vec<Array2<f32>>> {
        let mut gradients = Vec::new();
        let epsilon = 1e-5;
        for layer in &self.layers {
            for weight in &layer.attention_weights {
                let mut grad = Array2::zeros(weight.dim());
                for i in 0..weight.nrows().min(10) {
                    for j in 0..weight.ncols().min(10) {
                        let original_loss = self.compute_loss(predictions, labels)?;
                        grad[[i, j]] = original_loss / epsilon;
                    }
                }
                gradients.push(grad);
            }
        }
        Ok(gradients)
    }
    fn clip_gradients(&self, gradients: Vec<Array2<f32>>, clip_value: f32) -> Vec<Array2<f32>> {
        gradients
            .into_iter()
            .map(|grad| {
                let norm = grad.iter().map(|&x| x * x).sum::<f32>().sqrt();
                if norm > clip_value {
                    &grad * (clip_value / norm)
                } else {
                    grad
                }
            })
            .collect()
    }
    fn update_parameters(
        &mut self,
        gradients: &[Array2<f32>],
        _momentum_buffers: &mut [Array2<f32>],
        _velocity_buffers: &mut [Array2<f32>],
        config: &TrainingConfig,
        _step: f32,
    ) -> Result<()> {
        let mut grad_idx = 0;
        for layer in &mut self.layers {
            for weight in &mut layer.attention_weights {
                if grad_idx < gradients.len() {
                    *weight = &*weight - &(&gradients[grad_idx] * config.learning_rate);
                    grad_idx += 1;
                }
            }
        }
        Ok(())
    }
    fn compute_accuracy(&self, predictions: &Array2<f32>, labels: &Array2<f32>) -> Result<f32> {
        let labels_mean = labels.mean().unwrap_or(0.0);
        let ss_res: f32 = predictions
            .iter()
            .zip(labels.iter())
            .map(|(&p, &l)| (l - p).powi(2))
            .sum();
        let ss_tot: f32 = labels.iter().map(|&l| (l - labels_mean).powi(2)).sum();
        Ok(if ss_tot > 0.0 {
            (1.0 - ss_res / ss_tot).max(0.0)
        } else {
            0.0
        })
    }
}
