//! GNN training loop, loss computation, optimizer step, gradient handling.

use std::collections::HashMap;

use scirs2_core::ndarray_ext::{Array2, Axis};

use super::{GraphData, LearnedConstraint, LearnedShape, ModelError, ModelMetrics, ModelParams};
use crate::ml::gnn_layers::{
    apply_activation, graph_transformer_forward, hierarchical_graph_transformer_forward, GNNLayer,
    GraphEmbedding, LayerGradients, OutputLayer,
};
use crate::ml::gnn_types::{ActivationFunction, GNNConfig};

/// Optimizer state for GNN training
#[derive(Debug)]
pub struct OptimizerState {
    pub learning_rate: f64,
    pub momentum: HashMap<String, Array2<f64>>,
    pub velocity: HashMap<String, Array2<f64>>,
    pub iteration: usize,
}

/// Training history
#[derive(Debug, Default)]
pub struct TrainingHistory {
    pub loss_history: Vec<f64>,
    pub metric_history: Vec<ModelMetrics>,
    pub best_weights: Option<HashMap<String, Array2<f64>>>,
    pub best_epoch: usize,
}

/// Predict shapes from a graph embedding using the output layer.
pub fn predict_from_embedding(
    embedding: &GraphEmbedding,
    output_layer: &OutputLayer,
) -> Result<Vec<LearnedShape>, ModelError> {
    let graph_emb = &embedding.graph_embedding;
    let shape_logits = graph_emb.dot(&output_layer.shape_classifier);
    let shape_probs = softmax_vec(&shape_logits);

    let mut shape_scores: Vec<(usize, f64)> = shape_probs
        .iter()
        .enumerate()
        .map(|(i, &score)| (i, score))
        .collect();
    shape_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut learned_shapes = Vec::new();

    for (shape_idx, shape_confidence) in shape_scores.iter().take(3) {
        if *shape_confidence < 0.1 {
            continue;
        }

        let mut constraints = Vec::new();

        for (constraint_type, constraint_head) in &output_layer.constraint_heads {
            let constraint_logits = graph_emb.dot(constraint_head);
            let constraint_probs = softmax_vec(&constraint_logits);

            let max_idx = constraint_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let constraint_confidence = constraint_probs[max_idx];

            if constraint_confidence > 0.3 {
                let mut parameters = HashMap::new();
                match constraint_type.as_str() {
                    "minCount" | "maxCount" => {
                        parameters.insert("value".to_string(), serde_json::json!(max_idx + 1));
                    }
                    "datatype" => {
                        let datatypes = ["xsd:string", "xsd:integer", "xsd:boolean", "xsd:date"];
                        if max_idx < datatypes.len() {
                            parameters
                                .insert("value".to_string(), serde_json::json!(datatypes[max_idx]));
                        }
                    }
                    _ => {
                        parameters.insert("value".to_string(), serde_json::json!(max_idx));
                    }
                }
                constraints.push(LearnedConstraint {
                    constraint_type: constraint_type.clone(),
                    parameters,
                    confidence: constraint_confidence,
                    support: 0.8,
                });
            }
        }

        learned_shapes.push(LearnedShape {
            shape_id: format!("learned_shape_{shape_idx}"),
            constraints,
            confidence: *shape_confidence,
            feature_importance: HashMap::new(),
        });
    }

    Ok(learned_shapes)
}

/// Compute loss and output-layer gradients.
pub fn compute_loss_and_gradients(
    embedding: &GraphEmbedding,
    target: &[f64],
    output_layer: &OutputLayer,
) -> Result<(f64, Array2<f64>), ModelError> {
    let predictions = embedding
        .graph_embedding
        .dot(&output_layer.shape_classifier);
    let softmax_preds = softmax_2d(&predictions);

    let mut loss = 0.0;
    for (i, &target_val) in target.iter().enumerate() {
        if target_val > 0.0 && i < softmax_preds.ncols() {
            loss -= target_val * softmax_preds[[0, i]].ln();
        }
    }

    let mut gradients = Array2::zeros(predictions.raw_dim());
    for (i, &target_val) in target.iter().enumerate() {
        if i < gradients.ncols() {
            gradients[[0, i]] = softmax_preds[[0, i]] - target_val;
        }
    }

    Ok((loss, gradients))
}

/// Backward pass: build per-layer gradient list.
pub fn backward_pass(
    _embedding: &GraphEmbedding,
    output_gradients: &Array2<f64>,
    layers: &[GNNLayer],
    config: &GNNConfig,
) -> Result<Vec<LayerGradients>, ModelError> {
    let mut layer_gradients = Vec::new();

    let output_grad = LayerGradients {
        weight_gradients: output_gradients.clone(),
        bias_gradients: Some(output_gradients.clone()),
    };
    layer_gradients.push(output_grad);

    for _ in 0..layers.len() {
        let layer_grad = LayerGradients {
            weight_gradients: Array2::zeros((config.hidden_dim, config.hidden_dim)),
            bias_gradients: Some(Array2::zeros((1, config.hidden_dim))),
        };
        layer_gradients.push(layer_grad);
    }

    Ok(layer_gradients)
}

/// Apply SGD weight updates to all layers.
pub fn update_weights(
    layers: &mut [GNNLayer],
    output_layer: &mut OutputLayer,
    optimizer_state: &mut OptimizerState,
    gradients: &[LayerGradients],
) -> Result<(), ModelError> {
    let learning_rate = optimizer_state.learning_rate;

    if let Some(grad) = gradients.first() {
        let update = &grad.weight_gradients * learning_rate;
        output_layer.shape_classifier = &output_layer.shape_classifier - &update;
    }

    for (i, grad) in gradients.iter().skip(1).enumerate() {
        if i < layers.len() {
            match &mut layers[i] {
                GNNLayer::Gcn(layer) => {
                    let update = &grad.weight_gradients * learning_rate;
                    layer.weight = &layer.weight - &update;
                }
                GNNLayer::Gat(layer) => {
                    let update = &grad.weight_gradients * learning_rate;
                    layer.weight = &layer.weight - &update;
                }
                GNNLayer::Gin(layer) => {
                    for mlp_weight in &mut layer.mlp_layers {
                        let update = &grad.weight_gradients * learning_rate;
                        *mlp_weight = &*mlp_weight - &update;
                    }
                }
                GNNLayer::GraphSAGE(layer) => {
                    let update = &grad.weight_gradients * learning_rate;
                    layer.weight_self = &layer.weight_self - &update;
                    layer.weight_neighbor = &layer.weight_neighbor - &update;
                }
                GNNLayer::Mpnn(layer) => {
                    for weight in &mut layer.message_net {
                        let update = &grad.weight_gradients * learning_rate;
                        *weight = &*weight - &update;
                    }
                    for weight in &mut layer.update_net {
                        let update = &grad.weight_gradients * learning_rate;
                        *weight = &*weight - &update;
                    }
                }
                GNNLayer::GraphCompletion(layer) => {
                    let update = &grad.weight_gradients * learning_rate;
                    layer.entity_encoder = &layer.entity_encoder - &update;
                    layer.relation_encoder = &layer.relation_encoder - &update;
                }
                GNNLayer::EntityCompletion(layer) => {
                    let update = &grad.weight_gradients * learning_rate;
                    layer.entity_embedding = &layer.entity_embedding - &update;
                }
                GNNLayer::RelationCompletion(layer) => {
                    let update = &grad.weight_gradients * learning_rate;
                    layer.pattern_encoder = &layer.pattern_encoder - &update;
                    layer.relation_embedding = &layer.relation_embedding - &update;
                }
                GNNLayer::GraphTransformer(layer) => {
                    let update = &grad.weight_gradients * learning_rate;
                    layer.attention_weights = &layer.attention_weights - &update;
                    layer.feed_forward = &layer.feed_forward - &update;
                }
                GNNLayer::HierarchicalGraphTransformer(layer) => {
                    let update = &grad.weight_gradients * learning_rate;
                    layer.hierarchical_attention = &layer.hierarchical_attention - &update;
                    layer.level_encoders = layer
                        .level_encoders
                        .iter()
                        .map(|encoder| encoder - &update)
                        .collect();
                }
            }
        }
    }

    optimizer_state.iteration += 1;
    Ok(())
}

/// Accuracy helper
pub fn calculate_batch_accuracy(
    predictions: &[LearnedShape],
    shape_label: &super::super::ml::ShapeLabel,
) -> (usize, usize) {
    let mut correct = 0;
    let total = shape_label.shapes.len().max(1);
    for predicted_shape in predictions {
        for actual_shape in &shape_label.shapes {
            if predicted_shape.shape_id.contains(&actual_shape.shape_type) {
                correct += 1;
                break;
            }
        }
    }
    (correct, total)
}

/// Hash string helper for shape type encoding
pub fn hash_string(s: &str) -> usize {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish() as usize
}

/// Create target vector from shape label
pub fn create_target_vector(shape_label: &super::super::ml::ShapeLabel) -> Vec<f64> {
    let mut target = vec![0.0; 10];
    for shape in &shape_label.shapes {
        let shape_hash = hash_string(&shape.shape_type) % 10;
        target[shape_hash] = 1.0;
    }
    target
}

// ─── Softmax helpers ─────────────────────────────────────────────────────────

pub fn softmax_vec(logits: &Array2<f64>) -> Vec<f64> {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_logits: Vec<f64> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp: f64 = exp_logits.iter().sum();
    exp_logits.iter().map(|&x| x / sum_exp).collect()
}

pub fn softmax_2d(logits: &Array2<f64>) -> Array2<f64> {
    let mut result = logits.clone();
    for mut row in result.rows_mut() {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        row.mapv_inplace(|x| (x - max_val).exp());
        let sum: f64 = row.sum();
        if sum > 0.0 {
            row.mapv_inplace(|x| x / sum);
        }
    }
    result
}
