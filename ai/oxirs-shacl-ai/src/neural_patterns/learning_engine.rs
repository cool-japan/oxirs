//! Training loop, gradient update, convergence detection.

use scirs2_core::ndarray_ext::{Array1, Array2, Axis};
use scirs2_core::random::{Random, Rng, RngExt};
use std::collections::HashMap;

use crate::neural_patterns::learning_types::{NetworkWeights, OptimizerState, TrainingHistory};
use crate::neural_patterns::types::{ActivationFunction, CorrelationType, NeuralPatternConfig};
use crate::patterns::Pattern;
use crate::{Result, ShaclAiError};

/// Type alias for complex replay buffer data
pub type ReplayBufferData = (Pattern, HashMap<(String, String), CorrelationType>);

// ─── Forward pass ────────────────────────────────────────────────────────────

/// Convert patterns to embeddings
pub async fn patterns_to_embeddings(
    config: &NeuralPatternConfig,
    weights: &NetworkWeights,
    patterns: &[Pattern],
) -> Result<Array2<f64>> {
    let num_patterns = patterns.len();
    let embedding_dim = config.embedding_dim;

    let mut embeddings = Array2::zeros((num_patterns, embedding_dim));

    for (i, pattern) in patterns.iter().enumerate() {
        let features = extract_pattern_features(config, pattern).await?;
        let embedding = weights.embedding_weights.dot(&features);
        embeddings.row_mut(i).assign(&embedding);
    }

    if config.enable_batch_norm {
        apply_batch_normalization(&mut embeddings)?;
    }

    Ok(embeddings)
}

/// Extract deterministic features from a pattern
pub async fn extract_pattern_features(
    config: &NeuralPatternConfig,
    pattern: &Pattern,
) -> Result<Array1<f64>> {
    use crate::patterns::PatternType;

    let feature_dim = config.embedding_dim;
    let mut features = Array1::zeros(feature_dim);

    if feature_dim == 0 {
        return Ok(features);
    }

    features[0] = pattern.support().clamp(0.0, 1.0);
    if feature_dim > 1 {
        features[1] = pattern.confidence().clamp(0.0, 1.0);
    }
    if feature_dim > 2 {
        let type_idx: usize = match pattern.pattern_type() {
            PatternType::Structural => 0,
            PatternType::Temporal => 1,
            PatternType::Association => 2,
            PatternType::Constraint => 3,
            PatternType::Cardinality => 4,
            PatternType::Datatype => 5,
            PatternType::ShapeComposition => 6,
            PatternType::Usage => 7,
            PatternType::Anomalous => 8,
            PatternType::Range => 9,
        };
        features[2] = type_idx as f64 / 10.0;
    }

    let id_bytes = pattern.id().as_bytes();
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in id_bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    for i in 3..feature_dim {
        let rotated = h.rotate_left((i * 7 % 64) as u32);
        features[i] = (rotated as f64 / u64::MAX as f64) * 2.0 - 1.0;
        h = h.wrapping_add(0x9e37_79b9_7f4a_7c15);
    }

    Ok(features)
}

/// Forward pass through the network
pub async fn forward_pass(
    config: &NeuralPatternConfig,
    weights: &NetworkWeights,
    patterns: &[Pattern],
) -> Result<Array2<f64>> {
    let embeddings = patterns_to_embeddings(config, weights, patterns).await?;
    let attention_output = apply_attention(config, weights, &embeddings)?;
    apply_classification(config, weights, &attention_output)
}

/// Apply attention mechanism
pub fn apply_attention(
    config: &NeuralPatternConfig,
    weights: &NetworkWeights,
    embeddings: &Array2<f64>,
) -> Result<Array2<f64>> {
    let num_heads = config.attention_heads;
    let head_dim = config.embedding_dim / num_heads;

    let mut attention_outputs = Vec::new();

    for head in 0..num_heads {
        let head_name = format!("attention_head_{head}");
        if let Some(attention_weights) = weights.attention_weights.get(&head_name) {
            let head_output = compute_attention_head(embeddings, attention_weights, head_dim)?;
            attention_outputs.push(head_output);
        }
    }

    if attention_outputs.is_empty() {
        return Ok(embeddings.clone());
    }

    let mut aggregated = attention_outputs[0].clone();
    for output in attention_outputs.iter().skip(1) {
        aggregated += output;
    }
    let n = attention_outputs.len() as f64;
    aggregated.mapv_inplace(|x| x / n);

    if config.enable_residual_connections {
        aggregated += embeddings;
    }

    Ok(aggregated)
}

/// Compute a single attention head
pub fn compute_attention_head(
    embeddings: &Array2<f64>,
    attention_weights: &Array2<f64>,
    head_dim: usize,
) -> Result<Array2<f64>> {
    let q = embeddings.dot(attention_weights);
    let k = embeddings.dot(attention_weights);
    let v = embeddings.dot(attention_weights);

    let scale = (head_dim as f64).sqrt().max(1.0);
    let scores = q.dot(&k.t()) / scale;

    let mut attention_probs = scores.clone();
    for mut row in attention_probs.rows_mut() {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|x| (x - max_val).exp());
        let row_sum: f64 = row.iter().sum();
        if row_sum > 0.0 {
            row.mapv_inplace(|x| x / row_sum);
        }
    }

    Ok(attention_probs.dot(&v))
}

/// Apply classification layer
pub fn apply_classification(
    config: &NeuralPatternConfig,
    weights: &NetworkWeights,
    features: &Array2<f64>,
) -> Result<Array2<f64>> {
    let output = features.dot(&weights.classification_weights);
    apply_activation_fn(config, &output)
}

/// Apply activation function
pub fn apply_activation_fn(
    config: &NeuralPatternConfig,
    input: &Array2<f64>,
) -> Result<Array2<f64>> {
    let mut output = input.clone();
    match config.activation_function {
        ActivationFunction::ReLU => {
            output.mapv_inplace(|x| x.max(0.0));
        }
        ActivationFunction::LeakyReLU => {
            output.mapv_inplace(|x| if x > 0.0 { x } else { 0.01 * x });
        }
        ActivationFunction::Sigmoid => {
            output.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
        }
        ActivationFunction::Tanh => {
            output.mapv_inplace(|x| x.tanh());
        }
        ActivationFunction::GELU => {
            output.mapv_inplace(|x| {
                0.5 * x
                    * (1.0
                        + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            });
        }
        ActivationFunction::ELU => {
            output.mapv_inplace(|x| if x > 0.0 { x } else { x.exp() - 1.0 });
        }
        ActivationFunction::Swish => {
            output.mapv_inplace(|x| x / (1.0 + (-x).exp()));
        }
    }
    Ok(output)
}

/// Apply batch normalization in-place
pub fn apply_batch_normalization(input: &mut Array2<f64>) -> Result<()> {
    let mean = input
        .mean_axis(Axis(0))
        .expect("input array should have valid axis");
    let variance = input.var_axis(Axis(0), 1.0);

    for mut row in input.axis_iter_mut(Axis(0)) {
        for (i, elem) in row.iter_mut().enumerate() {
            *elem = (*elem - mean[i]) / (variance[i] + 1e-8).sqrt();
        }
    }

    Ok(())
}

// ─── Loss ────────────────────────────────────────────────────────────────────

/// Compute cross-entropy loss
pub fn compute_loss(
    predictions: &Array2<f64>,
    patterns: &[Pattern],
    target_correlations: &HashMap<(String, String), CorrelationType>,
    correlation_type_to_index: impl Fn(&CorrelationType) -> usize,
) -> Result<f64> {
    let num_classes = predictions.ncols();
    let mut total_loss = 0.0;
    let mut num_labelled = 0usize;

    for (i, pattern) in patterns.iter().enumerate() {
        for (j, other_pattern) in patterns.iter().enumerate() {
            if i == j {
                continue;
            }
            let key = (pattern.id().to_string(), other_pattern.id().to_string());
            if let Some(target) = target_correlations.get(&key) {
                let target_idx = correlation_type_to_index(target);
                let row = predictions.row(i);
                let max_logit = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = row.iter().map(|&x| (x - max_logit).exp()).sum();
                let log_prob = if exp_sum > 0.0 && target_idx < num_classes {
                    let logit = row[target_idx];
                    (logit - max_logit) - exp_sum.ln()
                } else {
                    -(num_classes as f64).ln()
                };
                total_loss -= log_prob;
                num_labelled += 1;
            }
        }
    }

    Ok(if num_labelled > 0 {
        total_loss / num_labelled as f64
    } else {
        (patterns.len() as f64).ln().max(1.0)
    })
}

// ─── Backward pass ───────────────────────────────────────────────────────────

/// Finite-difference backward pass
pub async fn backward_pass(
    config: &NeuralPatternConfig,
    weights: &mut NetworkWeights,
    optimizer: &mut OptimizerState,
    predictions: &Array2<f64>,
    patterns: &[Pattern],
    target_correlations: &HashMap<(String, String), CorrelationType>,
    correlation_type_to_index: impl Fn(&CorrelationType) -> usize,
) -> Result<()> {
    let base_loss = compute_loss(
        predictions,
        patterns,
        target_correlations,
        &correlation_type_to_index,
    )?;
    let epsilon = 1e-5;

    // Embedding weight gradients
    let (emb_rows, emb_cols) = weights.embedding_weights.dim();
    let mut emb_grad = Array2::zeros((emb_rows, emb_cols));
    for r in 0..emb_rows {
        for c in 0..emb_cols {
            weights.embedding_weights[[r, c]] += epsilon;
            let fwd = forward_pass(config, weights, patterns).await?;
            let loss_plus = compute_loss(
                &fwd,
                patterns,
                target_correlations,
                &correlation_type_to_index,
            )?;
            weights.embedding_weights[[r, c]] -= epsilon;
            emb_grad[[r, c]] = (loss_plus - base_loss) / epsilon;
        }
    }
    optimizer
        .momentum
        .insert("embedding".to_string(), emb_grad.clone());
    optimizer
        .squared_gradients
        .insert("embedding".to_string(), emb_grad.mapv(|x| x * x));

    // Classification weight gradients
    let (cls_rows, cls_cols) = weights.classification_weights.dim();
    let mut cls_grad = Array2::zeros((cls_rows, cls_cols));
    for r in 0..cls_rows {
        for c in 0..cls_cols {
            weights.classification_weights[[r, c]] += epsilon;
            let fwd = forward_pass(config, weights, patterns).await?;
            let loss_plus = compute_loss(
                &fwd,
                patterns,
                target_correlations,
                &correlation_type_to_index,
            )?;
            weights.classification_weights[[r, c]] -= epsilon;
            cls_grad[[r, c]] = (loss_plus - base_loss) / epsilon;
        }
    }
    optimizer
        .momentum
        .insert("classification".to_string(), cls_grad.clone());
    optimizer
        .squared_gradients
        .insert("classification".to_string(), cls_grad.mapv(|x| x * x));

    Ok(())
}

// ─── Adam weight update ──────────────────────────────────────────────────────

/// Apply Adam optimizer updates to all weights
pub fn update_weights(
    weights: &mut NetworkWeights,
    optimizer: &mut OptimizerState,
    current_learning_rate: f64,
) -> Result<()> {
    let step = optimizer.step + 1;
    optimizer.step = step;

    let lr = current_learning_rate;
    let beta1 = optimizer.bias_correction_1;
    let beta2 = optimizer.bias_correction_2;
    let epsilon = 1e-8;

    let bc1 = 1.0 - beta1.powi(step as i32);
    let bc2 = 1.0 - beta2.powi(step as i32);

    let adam_update = |w: &mut Array2<f64>, m: &mut Array2<f64>, v: &mut Array2<f64>| {
        let grad = m.clone();
        w.zip_mut_with(&grad, |_, _| {});
        let rows = w.nrows();
        let cols = w.ncols();
        for r in 0..rows {
            for c in 0..cols {
                let g = grad[[r, c]];
                let m_new = beta1 * m[[r, c]] + (1.0 - beta1) * g;
                let v_new = beta2 * v[[r, c]] + (1.0 - beta2) * g * g;
                m[[r, c]] = m_new;
                v[[r, c]] = v_new;
                let m_hat = m_new / bc1;
                let v_hat = v_new / bc2;
                w[[r, c]] -= lr * m_hat / (v_hat.sqrt() + epsilon);
            }
        }
    };

    if let Some(m_emb) = optimizer.momentum.get("embedding").cloned() {
        let mut m_emb_mut = m_emb;
        let v_emb = optimizer
            .squared_gradients
            .entry("embedding".to_string())
            .or_insert_with(|| Array2::zeros(weights.embedding_weights.dim()));
        adam_update(&mut weights.embedding_weights, &mut m_emb_mut, v_emb);
        optimizer
            .momentum
            .insert("embedding".to_string(), m_emb_mut);
    }

    if let Some(m_cls) = optimizer.momentum.get("classification").cloned() {
        let mut m_cls_mut = m_cls;
        let v_cls = optimizer
            .squared_gradients
            .entry("classification".to_string())
            .or_insert_with(|| Array2::zeros(weights.classification_weights.dim()));
        adam_update(&mut weights.classification_weights, &mut m_cls_mut, v_cls);
        optimizer
            .momentum
            .insert("classification".to_string(), m_cls_mut);
    }

    let head_names: Vec<String> = weights.attention_weights.keys().cloned().collect();
    for head_name in head_names {
        if let Some(m_head) = optimizer.momentum.get(&head_name).cloned() {
            let mut m_head_mut = m_head;
            if let Some(w_head) = weights.attention_weights.get_mut(&head_name) {
                let dim = w_head.dim();
                let v_head = optimizer
                    .squared_gradients
                    .entry(head_name.clone())
                    .or_insert_with(|| Array2::zeros(dim));
                adam_update(w_head, &mut m_head_mut, v_head);
            }
            optimizer.momentum.insert(head_name, m_head_mut);
        }
    }

    Ok(())
}

// ─── Adaptive gradient step ───────────────────────────────────────────────────

/// Single gradient step for meta-learning inner loop
#[allow(clippy::too_many_arguments)]
pub async fn gradient_step(
    config: &NeuralPatternConfig,
    weights: &mut NetworkWeights,
    optimizer: &mut OptimizerState,
    predictions: &Array2<f64>,
    patterns: &[Pattern],
    target_correlations: &HashMap<(String, String), CorrelationType>,
    learning_rate: f64,
    correlation_type_to_index: impl Fn(&CorrelationType) -> usize,
) -> Result<()> {
    let gradients =
        compute_gradients(config, weights, predictions, patterns, target_correlations).await?;

    for (param_name, grad) in gradients {
        match param_name.as_str() {
            "embedding" => {
                weights.embedding_weights = &weights.embedding_weights - &(grad * learning_rate);
            }
            "classification" => {
                weights.classification_weights =
                    &weights.classification_weights - &(grad * learning_rate);
            }
            _ => {
                if let Some(attention_weight) = weights.attention_weights.get_mut(&param_name) {
                    *attention_weight = attention_weight.clone() - &(grad * learning_rate);
                }
            }
        }
    }

    Ok(())
}

/// Compute gradients (simplified finite-diff placeholder)
pub async fn compute_gradients(
    config: &NeuralPatternConfig,
    _weights: &NetworkWeights,
    _predictions: &Array2<f64>,
    _patterns: &[Pattern],
    _target_correlations: &HashMap<(String, String), CorrelationType>,
) -> Result<HashMap<String, Array2<f64>>> {
    let mut gradients = HashMap::new();
    let embedding_dim = config.embedding_dim;

    gradients.insert(
        "embedding".to_string(),
        Array2::zeros((embedding_dim, embedding_dim)),
    );
    gradients.insert(
        "classification".to_string(),
        Array2::zeros((embedding_dim, 10)),
    );

    for head in 0..config.attention_heads {
        let head_name = format!("attention_head_{head}");
        let head_dim = embedding_dim / config.attention_heads;
        gradients.insert(head_name, Array2::zeros((head_dim, head_dim)));
    }

    Ok(gradients)
}

// ─── Learning rate scheduler ──────────────────────────────────────────────────

/// Update learning rate with reduce-on-plateau / exponential fallback
pub fn update_learning_rate(
    current_lr: f64,
    epoch: usize,
    validation_loss: f64,
    history: &TrainingHistory,
) -> f64 {
    let patience = 5usize;
    if history.validation_loss_history.len() >= patience {
        let recent_min = history
            .validation_loss_history
            .iter()
            .rev()
            .take(patience)
            .cloned()
            .fold(f64::INFINITY, f64::min);
        if validation_loss >= recent_min {
            let new_lr = (current_lr * 0.5).max(1e-7);
            if new_lr < current_lr {
                tracing::debug!(
                    "Epoch {}: reducing LR {:.2e} → {:.2e} (plateau detected)",
                    epoch,
                    current_lr,
                    new_lr
                );
                return new_lr;
            }
        }
    }
    let decay_rate = 0.95;
    (current_lr * decay_rate).max(1e-7)
}

/// Adaptive step size based on gradient norm and training progress
pub fn compute_adaptive_step_size(current_lr: f64, grad_norm: f64, step: usize) -> f64 {
    let adaptive_factor = (1.0 + grad_norm).recip();
    let warmup_factor = if step < 1000 {
        step as f64 / 1000.0
    } else {
        1.0
    };
    current_lr * adaptive_factor * warmup_factor
}

// ─── Continual learning ──────────────────────────────────────────────────────

/// Continual learning with experience replay
#[allow(clippy::too_many_arguments)]
pub async fn continual_learning_update(
    config: &NeuralPatternConfig,
    weights: &mut NetworkWeights,
    optimizer: &mut OptimizerState,
    current_lr: f64,
    new_patterns: &[Pattern],
    new_correlations: &HashMap<(String, String), CorrelationType>,
    replay_buffer: &[ReplayBufferData],
    correlation_type_to_index: impl Fn(&CorrelationType) -> usize,
) -> Result<()> {
    let replay_ratio = 0.3;
    let batch_size = config.batch_size;
    let replay_size = (batch_size as f64 * replay_ratio) as usize;

    let mut rng = Random::default();
    let mut replay_patterns = Vec::new();
    let mut replay_correlations = HashMap::new();

    for _ in 0..replay_size.min(replay_buffer.len()) {
        let idx = rng.random_range(0..replay_buffer.len());
        let (pattern, correlations) = &replay_buffer[idx];
        replay_patterns.push(pattern.clone());
        replay_correlations.extend(correlations.clone());
    }

    let mut combined_patterns = new_patterns.to_vec();
    combined_patterns.extend(replay_patterns);

    let mut combined_correlations = new_correlations.clone();
    combined_correlations.extend(replay_correlations);

    let predictions = forward_pass(config, weights, &combined_patterns).await?;
    let loss = compute_loss(
        &predictions,
        &combined_patterns,
        &combined_correlations,
        &correlation_type_to_index,
    )?;

    backward_pass(
        config,
        weights,
        optimizer,
        &predictions,
        &combined_patterns,
        &combined_correlations,
        &correlation_type_to_index,
    )
    .await?;
    update_weights(weights, optimizer, current_lr)?;

    tracing::info!(
        "Continual learning update: loss={:.4}, replay_ratio={:.2}",
        loss,
        replay_ratio
    );
    Ok(())
}

// ─── Softmax ─────────────────────────────────────────────────────────────────

pub fn softmax(x: &Array2<f64>) -> Array2<f64> {
    let mut result = x.clone();
    for mut row in result.rows_mut() {
        let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|x| (x - max_val).exp());
        let sum = row.sum();
        if sum > 0.0 {
            row.mapv_inplace(|x| x / sum);
        }
    }
    result
}

// ─── Forward with dropout ─────────────────────────────────────────────────────

pub async fn forward_pass_with_dropout(
    config: &NeuralPatternConfig,
    patterns: &[Pattern],
    use_dropout: bool,
) -> Result<Array2<f64>> {
    let num_patterns = patterns.len();
    let output_dim = 10;
    let mut predictions = Array2::zeros((num_patterns, output_dim));

    if use_dropout {
        let dropout_rate = config.dropout_rate;
        let mut rng = Random::default();
        for mut row in predictions.rows_mut() {
            for elem in row.iter_mut() {
                if rng.random::<f64>() < dropout_rate {
                    *elem = 0.0;
                } else {
                    *elem = rng.random::<f64>() / (1.0 - dropout_rate);
                }
            }
        }
    }

    Ok(predictions)
}

/// Adaptive optimization step (gradient clipping + Adam)
pub async fn adaptive_optimization_step(
    weights: &mut NetworkWeights,
    optimizer: &mut OptimizerState,
    step_size: f64,
    gradients: &HashMap<String, Array2<f64>>,
) -> Result<()> {
    let clip_norm = 1.0;
    let mut total_norm = 0.0;
    for grad in gradients.values() {
        total_norm += grad.mapv(|x| x * x).sum();
    }
    let grad_norm = total_norm.sqrt();
    let clip_coeff = if grad_norm > clip_norm {
        clip_norm / grad_norm
    } else {
        1.0
    };

    optimizer.step += 1;

    for (param_name, grad) in gradients {
        let clipped_grad = grad.mapv(|x| x * clip_coeff);
        let momentum = optimizer
            .momentum
            .entry(param_name.clone())
            .or_insert_with(|| Array2::zeros(grad.dim()));
        let squared_grad = optimizer
            .squared_gradients
            .entry(param_name.clone())
            .or_insert_with(|| Array2::zeros(grad.dim()));

        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        *momentum = momentum.mapv(|m| m * beta1) + clipped_grad.mapv(|g| g * (1.0 - beta1));
        *squared_grad =
            squared_grad.mapv(|v| v * beta2) + clipped_grad.mapv(|g| g * g * (1.0 - beta2));

        let bias_correction_1 = 1.0 - beta1.powi(optimizer.step as i32);
        let bias_correction_2 = 1.0 - beta2.powi(optimizer.step as i32);
        let corrected_momentum = momentum.mapv(|m| m / bias_correction_1);
        let corrected_squared_grad = squared_grad.mapv(|v| v / bias_correction_2);
        let update =
            corrected_momentum.mapv(|m| m) / corrected_squared_grad.mapv(|v| v.sqrt() + eps);

        if param_name == "embedding" {
            weights.embedding_weights = &weights.embedding_weights - &(update * step_size);
        } else if param_name == "classification" {
            weights.classification_weights =
                &weights.classification_weights - &(update * step_size);
        }
    }

    Ok(())
}
