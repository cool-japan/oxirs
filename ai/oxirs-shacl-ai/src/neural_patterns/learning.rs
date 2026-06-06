//! Neural pattern learning — thin facade.

use scirs2_core::ndarray_ext::Array2;
use std::collections::HashMap;
use std::time::Instant;

pub use crate::neural_patterns::learning_engine::{
    adaptive_optimization_step, apply_activation_fn, apply_attention, apply_batch_normalization,
    backward_pass, compute_adaptive_step_size, compute_gradients, compute_loss,
    continual_learning_update, extract_pattern_features, forward_pass, forward_pass_with_dropout,
    gradient_step, patterns_to_embeddings, softmax, update_learning_rate, update_weights,
    ReplayBufferData,
};
pub use crate::neural_patterns::learning_eval::{
    compute_accuracy, compute_auc_roc, compute_binary_auc, compute_comprehensive_metrics,
    compute_confusion_matrix, compute_per_class_metrics, correlation_type_to_index,
    index_to_correlation_type, predict_correlations,
};
pub use crate::neural_patterns::learning_types::{NetworkWeights, OptimizerState, TrainingHistory};

use crate::ml::ModelMetrics;
use crate::neural_patterns::types::{CorrelationType, NeuralPatternConfig};
use crate::patterns::Pattern;
use crate::Result;

/// Neural pattern learning engine for discovering complex pattern relationships
#[derive(Debug)]
pub struct NeuralPatternLearner {
    /// Configuration for learning
    pub config: NeuralPatternConfig,
    /// Neural network weights
    pub weights: NetworkWeights,
    /// Learning optimizer state
    pub optimizer: OptimizerState,
    /// Training history
    pub training_history: TrainingHistory,
    /// Current learning rate
    pub current_learning_rate: f64,
}

impl NeuralPatternLearner {
    /// Create new neural pattern learner
    pub fn new(config: NeuralPatternConfig) -> Self {
        let weights = NetworkWeights::new(&config);
        let optimizer = OptimizerState::new();

        Self {
            current_learning_rate: config.learning_rate,
            config,
            weights,
            optimizer,
            training_history: TrainingHistory::default(),
        }
    }

    /// Train the neural pattern recognition model
    pub async fn train(
        &mut self,
        training_patterns: &[Pattern],
        validation_patterns: &[Pattern],
        target_correlations: &HashMap<(String, String), CorrelationType>,
    ) -> Result<ModelMetrics> {
        let training_start_time = Instant::now();
        let mut best_validation_loss = f64::INFINITY;
        let mut epochs_without_improvement = 0;
        let patience = 10;

        for epoch in 0..self.config.max_epochs {
            let epoch_start = Instant::now();

            let training_loss = self
                .train_epoch(training_patterns, target_correlations)
                .await?;
            let validation_loss = self
                .validate_epoch(validation_patterns, target_correlations)
                .await?;
            let accuracy = compute_accuracy(
                &self.config,
                &self.weights,
                validation_patterns,
                target_correlations,
            )
            .await?;

            self.current_learning_rate = update_learning_rate(
                self.current_learning_rate,
                epoch,
                validation_loss,
                &self.training_history,
            );

            self.training_history.loss_history.push(training_loss);
            self.training_history
                .validation_loss_history
                .push(validation_loss);
            self.training_history.accuracy_history.push(accuracy);
            self.training_history
                .learning_rate_history
                .push(self.current_learning_rate);
            self.training_history
                .epoch_times
                .push(epoch_start.elapsed());

            if validation_loss < best_validation_loss {
                best_validation_loss = validation_loss;
                epochs_without_improvement = 0;
            } else {
                epochs_without_improvement += 1;
            }

            if epochs_without_improvement >= patience {
                tracing::info!("Early stopping at epoch {} due to no improvement", epoch);
                break;
            }

            tracing::info!(
                "Epoch {}: training_loss={:.4}, validation_loss={:.4}, accuracy={:.4}",
                epoch,
                training_loss,
                validation_loss,
                accuracy
            );
        }

        let training_time = training_start_time.elapsed();
        compute_comprehensive_metrics(
            &self.config,
            &self.weights,
            validation_patterns,
            target_correlations,
            training_time,
        )
        .await
    }

    async fn train_epoch(
        &mut self,
        patterns: &[Pattern],
        target_correlations: &HashMap<(String, String), CorrelationType>,
    ) -> Result<f64> {
        let batch_size = self.config.batch_size;
        let num_batches = (patterns.len() + batch_size - 1) / batch_size;
        let mut total_loss = 0.0;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * batch_size;
            let batch_end = std::cmp::min(batch_start + batch_size, patterns.len());
            let batch_patterns = &patterns[batch_start..batch_end];

            let predictions = forward_pass(&self.config, &self.weights, batch_patterns).await?;
            let loss = compute_loss(
                &predictions,
                batch_patterns,
                target_correlations,
                correlation_type_to_index,
            )?;

            backward_pass(
                &self.config,
                &mut self.weights,
                &mut self.optimizer,
                &predictions,
                batch_patterns,
                target_correlations,
                correlation_type_to_index,
            )
            .await?;

            update_weights(
                &mut self.weights,
                &mut self.optimizer,
                self.current_learning_rate,
            )?;
            total_loss += loss;
        }

        Ok(total_loss / num_batches as f64)
    }

    async fn validate_epoch(
        &self,
        patterns: &[Pattern],
        target_correlations: &HashMap<(String, String), CorrelationType>,
    ) -> Result<f64> {
        let predictions = forward_pass(&self.config, &self.weights, patterns).await?;
        compute_loss(
            &predictions,
            patterns,
            target_correlations,
            correlation_type_to_index,
        )
    }

    /// Self-attention forward pass
    pub async fn self_attention_forward(
        &self,
        pattern_embeddings: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let embed_dim = pattern_embeddings.ncols();
        let head_dim = embed_dim / self.config.attention_heads;

        let mut attention_outputs = Vec::new();

        for head in 0..self.config.attention_heads {
            let head_name = format!("attention_head_{head}");
            if let Some(weights) = self.weights.attention_weights.get(&head_name) {
                let q = pattern_embeddings.dot(weights);
                let k = pattern_embeddings.dot(weights);
                let v = pattern_embeddings.dot(weights);
                let attention_scores = q.dot(&k.t()) / (head_dim as f64).sqrt();
                let attention_probs = softmax(&attention_scores);
                let attention_output = attention_probs.dot(&v);
                attention_outputs.push(attention_output);
            }
        }

        if let Some(first_output) = attention_outputs.first() {
            let mut combined = first_output.clone();
            for output in attention_outputs.iter().skip(1) {
                combined += output;
            }
            let n_heads = attention_outputs.len() as f64;
            combined.mapv_inplace(|x| x / n_heads);
            Ok(combined)
        } else {
            Ok(pattern_embeddings.clone())
        }
    }

    /// Meta-learning update
    pub async fn meta_learning_update(
        &mut self,
        support_patterns: &[Pattern],
        query_patterns: &[Pattern],
        support_correlations: &HashMap<(String, String), CorrelationType>,
    ) -> Result<()> {
        let support_predictions =
            forward_pass(&self.config, &self.weights, support_patterns).await?;
        let support_loss = compute_loss(
            &support_predictions,
            support_patterns,
            support_correlations,
            correlation_type_to_index,
        )?;

        let original_weights = self.weights.clone_weights();
        let inner_lr = self.config.learning_rate * 0.1;

        gradient_step(
            &self.config,
            &mut self.weights,
            &mut self.optimizer,
            &support_predictions,
            support_patterns,
            support_correlations,
            inner_lr,
            correlation_type_to_index,
        )
        .await?;

        let query_predictions = forward_pass(&self.config, &self.weights, query_patterns).await?;
        let query_loss = compute_loss(
            &query_predictions,
            query_patterns,
            support_correlations,
            correlation_type_to_index,
        )?;

        self.meta_gradient_update(&original_weights, query_loss)
            .await?;

        tracing::info!(
            "Meta-learning update: support_loss={:.4}, query_loss={:.4}",
            support_loss,
            query_loss
        );
        Ok(())
    }

    async fn meta_gradient_update(
        &mut self,
        original_weights: &NetworkWeights,
        _query_loss: f64,
    ) -> Result<()> {
        let meta_lr = self.config.learning_rate * 0.01;

        let embedding_diff = &self.weights.embedding_weights - &original_weights.embedding_weights;
        let classification_diff =
            &self.weights.classification_weights - &original_weights.classification_weights;

        self.weights.embedding_weights =
            &original_weights.embedding_weights - &(embedding_diff * meta_lr);
        self.weights.classification_weights =
            &original_weights.classification_weights - &(classification_diff * meta_lr);

        for (head_name, original_weight) in &original_weights.attention_weights {
            if let Some(current_weight) = self.weights.attention_weights.get(head_name) {
                let diff = current_weight - original_weight;
                let updated = original_weight - &(diff * meta_lr);
                self.weights
                    .attention_weights
                    .insert(head_name.clone(), updated);
            }
        }

        tracing::debug!("Meta-gradient update applied with meta_lr={:.6}", meta_lr);
        Ok(())
    }

    /// Predict with uncertainty via MC dropout
    pub async fn predict_with_uncertainty(
        &self,
        patterns: &[Pattern],
        num_samples: usize,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let mut predictions_samples = Vec::new();

        for _ in 0..num_samples {
            let predictions = forward_pass_with_dropout(&self.config, patterns, true).await?;
            predictions_samples.push(predictions);
        }

        let num_patterns = patterns.len();
        let output_dim = 10;

        let mut mean_predictions = Array2::zeros((num_patterns, output_dim));
        let mut var_predictions = Array2::zeros((num_patterns, output_dim));

        for predictions in &predictions_samples {
            mean_predictions += predictions;
        }
        mean_predictions /= num_samples as f64;

        for predictions in &predictions_samples {
            let diff = predictions - &mean_predictions;
            var_predictions += &diff.mapv(|x| x * x);
        }
        var_predictions /= (num_samples - 1) as f64;

        Ok((mean_predictions, var_predictions))
    }

    /// Continual learning update
    pub async fn continual_learning_update(
        &mut self,
        new_patterns: &[Pattern],
        new_correlations: &HashMap<(String, String), CorrelationType>,
        replay_buffer: &[ReplayBufferData],
    ) -> Result<()> {
        continual_learning_update(
            &self.config,
            &mut self.weights,
            &mut self.optimizer,
            self.current_learning_rate,
            new_patterns,
            new_correlations,
            replay_buffer,
            correlation_type_to_index,
        )
        .await
    }

    /// Adaptive optimization step
    pub async fn adaptive_optimization_step(
        &mut self,
        gradients: &HashMap<String, Array2<f64>>,
    ) -> Result<()> {
        let grad_norm: f64 = gradients
            .values()
            .map(|g| g.mapv(|x| x * x).sum())
            .sum::<f64>()
            .sqrt();
        let step_size =
            compute_adaptive_step_size(self.current_learning_rate, grad_norm, self.optimizer.step);
        adaptive_optimization_step(&mut self.weights, &mut self.optimizer, step_size, gradients)
            .await
    }

    /// Predict correlations for new patterns
    pub async fn predict_correlations(
        &self,
        patterns: &[Pattern],
    ) -> Result<HashMap<(String, String), (CorrelationType, f64)>> {
        predict_correlations(&self.config, &self.weights, patterns).await
    }

    /// Get training history
    pub fn get_training_history(&self) -> &TrainingHistory {
        &self.training_history
    }

    /// Save model weights to file
    pub fn save_weights(&self, path: &str) -> Result<()> {
        use std::io::Write;

        let (emb_rows, emb_cols) = self.weights.embedding_weights.dim();
        let (cls_rows, cls_cols) = self.weights.classification_weights.dim();

        let mut obj = serde_json::json!({
            "embedding": {
                "shape": [emb_rows, emb_cols],
                "data": self.weights.embedding_weights.iter().cloned().collect::<Vec<_>>()
            },
            "classification": {
                "shape": [cls_rows, cls_cols],
                "data": self.weights.classification_weights.iter().cloned().collect::<Vec<_>>()
            },
            "attention": {}
        });

        for (head_name, head_weights) in &self.weights.attention_weights {
            let (hr, hc) = head_weights.dim();
            obj["attention"][head_name] = serde_json::json!({
                "shape": [hr, hc],
                "data": head_weights.iter().cloned().collect::<Vec<_>>()
            });
        }

        let mut file = std::fs::File::create(path)
            .map_err(|e| crate::ShaclAiError::ProcessingError(e.to_string()))?;
        file.write_all(obj.to_string().as_bytes())
            .map_err(|e| crate::ShaclAiError::ProcessingError(e.to_string()))?;

        tracing::debug!("Model weights saved to '{}'", path);
        Ok(())
    }

    /// Load model weights from file
    pub fn load_weights(&mut self, path: &str) -> Result<()> {
        use scirs2_core::ndarray_ext::Array2;

        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::ShaclAiError::ProcessingError(e.to_string()))?;
        let obj: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| crate::ShaclAiError::ProcessingError(e.to_string()))?;

        let parse_matrix = |v: &serde_json::Value| -> Result<Array2<f64>> {
            let shape = v["shape"]
                .as_array()
                .ok_or_else(|| crate::ShaclAiError::ProcessingError("missing shape".into()))?;
            let rows = shape[0].as_u64().unwrap_or(0) as usize;
            let cols = shape[1].as_u64().unwrap_or(0) as usize;
            let data: Vec<f64> = v["data"]
                .as_array()
                .ok_or_else(|| crate::ShaclAiError::ProcessingError("missing data".into()))?
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0))
                .collect();
            Array2::from_shape_vec((rows, cols), data)
                .map_err(|e| crate::ShaclAiError::ProcessingError(e.to_string()))
        };

        self.weights.embedding_weights = parse_matrix(&obj["embedding"])?;
        self.weights.classification_weights = parse_matrix(&obj["classification"])?;

        if let Some(attention) = obj["attention"].as_object() {
            for (head_name, head_val) in attention {
                let matrix = parse_matrix(head_val)?;
                self.weights
                    .attention_weights
                    .insert(head_name.clone(), matrix);
            }
        }

        tracing::debug!("Model weights loaded from '{}'", path);
        Ok(())
    }
}
