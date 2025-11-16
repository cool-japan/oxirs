//! # GraphConvolutionalNetwork - Trait Implementations
//!
//! This module contains trait implementations for `GraphConvolutionalNetwork`.
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
impl GraphNeuralNetwork for GraphConvolutionalNetwork {
    async fn forward(&self, graph: &RdfGraph, features: &Array2<f32>) -> Result<Array2<f32>> {
        let mut x = features.clone();
        let adj = graph.get_adjacency_matrix();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &adj)?;
            if i < self.layers.len() - 1 {
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
        let mut _total_loss = 0.0;
        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;
        tracing::info!("Starting GNN training for {} epochs", config.max_epochs);
        let mut momentum_buffers: Vec<Array2<f32>> = self
            .layers
            .iter()
            .map(|layer| Array2::zeros(layer.weight.dim()))
            .collect();
        let mut velocity_buffers: Vec<Array2<f32>> = self
            .layers
            .iter()
            .map(|layer| Array2::zeros(layer.weight.dim()))
            .collect();
        for epoch in 0..config.max_epochs {
            let epoch_start = std::time::Instant::now();
            let predictions = self.forward(graph, features).await?;
            let loss = self.compute_loss(&predictions, labels)?;
            _total_loss += loss;
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
            let accuracy = self.compute_accuracy(&predictions, labels)?;
            if loss < best_loss {
                best_loss = loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= config.early_stopping.patience {
                    tracing::info!("Early stopping triggered at epoch {}", epoch);
                    break;
                }
            }
            let epoch_time = epoch_start.elapsed();
            if epoch % 10 == 0 {
                tracing::info!(
                    "Epoch {}: loss={:.6}, accuracy={:.4}, time={:?}",
                    epoch,
                    loss,
                    accuracy,
                    epoch_time
                );
            }
        }
        self.trained = true;
        let total_time = start_time.elapsed();
        tracing::info!(
            "GNN training completed. Final loss: {:.6}, Time: {:?}",
            best_loss,
            total_time
        );
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
        if source_nodes.len() != target_nodes.len() {
            return Err(anyhow!(
                "Source and target node arrays must have the same length"
            ));
        }
        if !self.trained {
            tracing::warn!("Model not trained yet, predictions may be inaccurate");
        }
        let node_features = self.extract_node_features(graph).await?;
        let embeddings = self.get_embeddings(graph, &node_features).await?;
        let mut predictions = Array1::zeros(source_nodes.len());
        for (i, (&src, &tgt)) in source_nodes.iter().zip(target_nodes.iter()).enumerate() {
            if src >= embeddings.nrows() || tgt >= embeddings.nrows() {
                return Err(anyhow!("Node index out of bounds"));
            }
            let src_embedding = embeddings.row(src);
            let tgt_embedding = embeddings.row(tgt);
            let score = match self.config.link_prediction_method {
                LinkPredictionMethod::DotProduct => src_embedding
                    .iter()
                    .zip(tgt_embedding.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>(),
                LinkPredictionMethod::Cosine => {
                    let dot_product = src_embedding
                        .iter()
                        .zip(tgt_embedding.iter())
                        .map(|(&a, &b)| a * b)
                        .sum::<f32>();
                    let src_norm = src_embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    let tgt_norm = tgt_embedding.iter().map(|&x| x * x).sum::<f32>().sqrt();
                    if src_norm > 0.0 && tgt_norm > 0.0 {
                        dot_product / (src_norm * tgt_norm)
                    } else {
                        0.0
                    }
                }
                LinkPredictionMethod::L2Distance => {
                    let distance = src_embedding
                        .iter()
                        .zip(tgt_embedding.iter())
                        .map(|(&a, &b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    -distance
                }
                LinkPredictionMethod::Bilinear => src_embedding
                    .iter()
                    .zip(tgt_embedding.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f32>(),
                LinkPredictionMethod::MLP => {
                    let concat_features: Vec<f32> = src_embedding
                        .iter()
                        .chain(tgt_embedding.iter())
                        .cloned()
                        .collect();
                    let hidden_size = concat_features.len() / 2;
                    let mut hidden: Vec<f32> = vec![0.0; hidden_size];
                    for (i, &feature) in concat_features.iter().enumerate() {
                        hidden[i % hidden_size] += feature * 0.1;
                    }
                    let activated: f32 = hidden.iter().map(|&x| x.tanh()).sum();
                    activated / hidden_size as f32
                }
            };
            predictions[i] = 1.0 / (1.0 + (-score).exp());
        }
        Ok(predictions)
    }
    /// Extract node features from RDF graph
    async fn extract_node_features(&self, graph: &RdfGraph) -> Result<Array2<f32>> {
        let node_count = graph.node_count();
        let feature_dim = 64;
        let mut features = Array2::zeros((node_count, feature_dim));
        for node_idx in graph.nodes() {
            let degree = graph.degree(node_idx) as f32;
            let in_degree = graph.in_degree(node_idx).unwrap_or(0) as f32;
            let out_degree = graph.out_degree(node_idx).unwrap_or(0) as f32;
            features[[node_idx, 0]] = degree.ln_1p();
            features[[node_idx, 1]] = in_degree.ln_1p();
            features[[node_idx, 2]] = out_degree.ln_1p();
            features[[node_idx, 3]] = (in_degree / (degree + 1.0)).clamp(0.0, 1.0);
            features[[node_idx, 4]] = (out_degree / (degree + 1.0)).clamp(0.0, 1.0);
            if let Some(entity_iri) = graph.node_to_entity.get(&node_idx) {
                let node_type_hash = entity_iri.len() % 10;
                features[[node_idx, 5]] = node_type_hash as f32 / 10.0;
                for i in 6..feature_dim {
                    features[[node_idx, i]] = (entity_iri.len() * i) as f32 / 1000.0 % 1.0;
                }
            }
        }
        Ok(features)
    }
    fn get_parameters(&self) -> Result<Vec<Array2<f32>>> {
        Ok(self
            .layers
            .iter()
            .map(|layer| layer.weight.clone())
            .collect())
    }
    fn set_parameters(&mut self, parameters: &[Array2<f32>]) -> Result<()> {
        if parameters.len() != self.layers.len() {
            return Err(anyhow!("Parameter count mismatch"));
        }
        for (layer, param) in self.layers.iter_mut().zip(parameters.iter()) {
            layer.weight = param.clone();
        }
        Ok(())
    }
    /// Compute loss between predictions and labels
    fn compute_loss(&self, predictions: &Array2<f32>, labels: &Array2<f32>) -> Result<f32> {
        if predictions.dim() != labels.dim() {
            return Err(anyhow!("Predictions and labels dimension mismatch"));
        }
        let diff = predictions - labels;
        let squared_diff = &diff * &diff;
        let mse = squared_diff.mean().unwrap_or(0.0);
        Ok(mse)
    }
    /// Compute gradients for backpropagation
    async fn compute_gradients(
        &self,
        predictions: &Array2<f32>,
        labels: &Array2<f32>,
        graph: &RdfGraph,
        features: &Array2<f32>,
    ) -> Result<Vec<Array2<f32>>> {
        let mut gradients = Vec::with_capacity(self.layers.len());
        let _output_grad = 2.0 * (predictions - labels) / (predictions.len() as f32);
        let epsilon = 1e-5;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut layer_grad = Array2::zeros(layer.weight.dim());
            for i in 0..layer.weight.nrows() {
                for j in 0..layer.weight.ncols() {
                    let mut perturbed_layers = self.layers.clone();
                    perturbed_layers[layer_idx].weight[[i, j]] += epsilon;
                    let perturbed_network = GraphConvolutionalNetwork {
                        layers: perturbed_layers,
                        config: self.config.clone(),
                        trained: self.trained,
                    };
                    let perturbed_output = perturbed_network.forward(graph, features).await?;
                    let perturbed_loss = self.compute_loss(&perturbed_output, labels)?;
                    let original_loss = self.compute_loss(predictions, labels)?;
                    layer_grad[[i, j]] = (perturbed_loss - original_loss) / epsilon;
                }
            }
            gradients.push(layer_grad);
        }
        Ok(gradients)
    }
    /// Apply gradient clipping to prevent exploding gradients
    fn clip_gradients(&self, gradients: Vec<Array2<f32>>, clip_value: f32) -> Vec<Array2<f32>> {
        let mut clipped_gradients = Vec::with_capacity(gradients.len());
        for grad in gradients {
            let grad_norm = grad.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if grad_norm > clip_value {
                let scale_factor = clip_value / grad_norm;
                clipped_gradients.push(&grad * scale_factor);
            } else {
                clipped_gradients.push(grad);
            }
        }
        clipped_gradients
    }
    /// Update parameters using the configured optimizer
    fn update_parameters(
        &mut self,
        gradients: &[Array2<f32>],
        momentum_buffers: &mut [Array2<f32>],
        velocity_buffers: &mut [Array2<f32>],
        config: &TrainingConfig,
        step: f32,
    ) -> Result<()> {
        if gradients.len() != self.layers.len() {
            return Err(anyhow!("Gradient count mismatch"));
        }
        match &config.optimizer {
            Optimizer::SGD {
                momentum,
                weight_decay,
                nesterov,
            } => {
                for (i, (layer, grad)) in self.layers.iter_mut().zip(gradients.iter()).enumerate() {
                    let mut effective_grad = grad.clone();
                    if weight_decay > &0.0 {
                        effective_grad = &effective_grad + &(&layer.weight * *weight_decay);
                    }
                    momentum_buffers[i] = &(&momentum_buffers[i] * *momentum) + &effective_grad;
                    let update = if *nesterov {
                        &effective_grad + &(&momentum_buffers[i] * *momentum)
                    } else {
                        momentum_buffers[i].clone()
                    };
                    layer.weight = &layer.weight - &(&update * config.learning_rate);
                }
            }
            Optimizer::Adam {
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => {
                for (i, (layer, grad)) in self.layers.iter_mut().zip(gradients.iter()).enumerate() {
                    let mut effective_grad = grad.clone();
                    if weight_decay > &0.0 {
                        effective_grad = &effective_grad + &(&layer.weight * *weight_decay);
                    }
                    momentum_buffers[i] =
                        &(&momentum_buffers[i] * *beta1) + &(&effective_grad * (1.0 - beta1));
                    let grad_squared = &effective_grad * &effective_grad;
                    velocity_buffers[i] =
                        &(&velocity_buffers[i] * *beta2) + &(&grad_squared * (1.0 - beta2));
                    let bias_correction1 = 1.0 - beta1.powf(step);
                    let bias_correction2 = 1.0 - beta2.powf(step);
                    let m_hat = &momentum_buffers[i] / bias_correction1;
                    let v_hat = &velocity_buffers[i] / bias_correction2;
                    let denominator = v_hat.mapv(|x| x.sqrt() + epsilon);
                    let update = &m_hat / &denominator;
                    layer.weight = &layer.weight - &(&update * config.learning_rate);
                }
            }
            _ => {
                for (layer, grad) in self.layers.iter_mut().zip(gradients.iter()) {
                    layer.weight = &layer.weight - &(grad * config.learning_rate);
                }
            }
        }
        Ok(())
    }
    /// Compute accuracy between predictions and labels
    fn compute_accuracy(&self, predictions: &Array2<f32>, labels: &Array2<f32>) -> Result<f32> {
        if predictions.dim() != labels.dim() {
            return Err(anyhow!("Predictions and labels dimension mismatch"));
        }
        let labels_mean = labels.mean().unwrap_or(0.0);
        let ss_res: f32 = predictions
            .iter()
            .zip(labels.iter())
            .map(|(&pred, &label)| (label - pred).powi(2))
            .sum();
        let ss_tot: f32 = labels
            .iter()
            .map(|&label| (label - labels_mean).powi(2))
            .sum();
        let r_squared = if ss_tot > 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };
        Ok(r_squared.max(0.0))
    }
}
