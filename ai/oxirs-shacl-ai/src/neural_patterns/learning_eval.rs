//! Evaluation metrics: precision/recall/F1 for neural pattern learning.

use scirs2_core::ndarray_ext::Array2;
use std::collections::HashMap;
use std::time::Duration;

use crate::ml::{ClassMetrics, ModelMetrics};
use crate::neural_patterns::types::CorrelationType;
use crate::patterns::Pattern;
use crate::{Result, ShaclAiError};

use super::learning_engine::{forward_pass, softmax};
use super::learning_types::NetworkWeights;
use super::types::NeuralPatternConfig;

// ─── Index ↔ CorrelationType conversions ─────────────────────────────────────

pub fn index_to_correlation_type(index: usize) -> CorrelationType {
    match index {
        0 => CorrelationType::Structural,
        1 => CorrelationType::Semantic,
        2 => CorrelationType::Temporal,
        3 => CorrelationType::Causal,
        4 => CorrelationType::Hierarchical,
        5 => CorrelationType::Functional,
        6 => CorrelationType::Contextual,
        7 => CorrelationType::CrossDomain,
        _ => CorrelationType::Structural,
    }
}

pub fn correlation_type_to_index(corr_type: &CorrelationType) -> usize {
    match corr_type {
        CorrelationType::Structural => 0,
        CorrelationType::Semantic => 1,
        CorrelationType::Temporal => 2,
        CorrelationType::Causal => 3,
        CorrelationType::Hierarchical => 4,
        CorrelationType::Functional => 5,
        CorrelationType::Contextual => 6,
        CorrelationType::CrossDomain => 7,
    }
}

// ─── Accuracy ────────────────────────────────────────────────────────────────

/// Compute accuracy on validation set
pub async fn compute_accuracy(
    config: &NeuralPatternConfig,
    weights: &NetworkWeights,
    patterns: &[Pattern],
    target_correlations: &HashMap<(String, String), CorrelationType>,
) -> Result<f64> {
    let predictions = forward_pass(config, weights, patterns).await?;

    let mut correct_predictions = 0;
    let mut total_predictions = 0;

    for (i, pattern) in patterns.iter().enumerate() {
        for (j, other_pattern) in patterns.iter().enumerate() {
            if i != j {
                let pattern_pair = (pattern.id().to_string(), other_pattern.id().to_string());
                if let Some(expected_correlation) = target_correlations.get(&pattern_pair) {
                    let pred_row = predictions.row(i);
                    let predicted_type_idx = pred_row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    let predicted_correlation = index_to_correlation_type(predicted_type_idx);
                    if predicted_correlation == *expected_correlation {
                        correct_predictions += 1;
                    }
                    total_predictions += 1;
                }
            }
        }
    }

    let accuracy = if total_predictions > 0 {
        correct_predictions as f64 / total_predictions as f64
    } else {
        0.0
    };

    Ok(accuracy)
}

// ─── Confusion matrix ────────────────────────────────────────────────────────

pub fn compute_confusion_matrix(
    predictions: &[usize],
    true_labels: &[usize],
    num_classes: usize,
) -> Vec<Vec<usize>> {
    let mut matrix = vec![vec![0; num_classes]; num_classes];
    for (pred, true_label) in predictions.iter().zip(true_labels.iter()) {
        if *pred < num_classes && *true_label < num_classes {
            matrix[*true_label][*pred] += 1;
        }
    }
    matrix
}

// ─── Per-class metrics ───────────────────────────────────────────────────────

pub fn compute_per_class_metrics(
    confusion_matrix: &[Vec<usize>],
    num_classes: usize,
) -> (HashMap<String, ClassMetrics>, f64, f64, f64) {
    let mut per_class_metrics = HashMap::new();
    let mut precision_sum = 0.0;
    let mut recall_sum = 0.0;
    let mut f1_sum = 0.0;
    let mut valid_classes = 0;

    #[allow(clippy::needless_range_loop)]
    for class_idx in 0..num_classes {
        let tp = confusion_matrix[class_idx][class_idx];
        let fp: usize = (0..num_classes)
            .filter(|&i| i != class_idx)
            .map(|i| confusion_matrix[i][class_idx])
            .sum();
        let fn_count: usize = (0..num_classes)
            .filter(|&j| j != class_idx)
            .map(|j| confusion_matrix[class_idx][j])
            .sum();
        let support = tp + fn_count;

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_count > 0 {
            tp as f64 / (tp + fn_count) as f64
        } else {
            0.0
        };
        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        let class_name = format!("{:?}", index_to_correlation_type(class_idx));
        per_class_metrics.insert(
            class_name,
            ClassMetrics {
                precision,
                recall,
                f1_score,
                support,
            },
        );

        if support > 0 {
            precision_sum += precision;
            recall_sum += recall;
            f1_sum += f1_score;
            valid_classes += 1;
        }
    }

    let macro_precision = if valid_classes > 0 {
        precision_sum / valid_classes as f64
    } else {
        0.0
    };
    let macro_recall = if valid_classes > 0 {
        recall_sum / valid_classes as f64
    } else {
        0.0
    };
    let macro_f1 = if valid_classes > 0 {
        f1_sum / valid_classes as f64
    } else {
        0.0
    };

    (per_class_metrics, macro_precision, macro_recall, macro_f1)
}

// ─── AUC-ROC ─────────────────────────────────────────────────────────────────

pub fn compute_auc_roc(
    prediction_scores: &[Vec<f64>],
    true_labels: &[usize],
    num_classes: usize,
) -> f64 {
    let mut auc_sum = 0.0;
    let mut valid_classes = 0;

    for class_idx in 0..num_classes {
        let mut positive_scores = Vec::new();
        let mut negative_scores = Vec::new();

        for (scores, &true_label) in prediction_scores.iter().zip(true_labels.iter()) {
            if class_idx < scores.len() {
                let score = scores[class_idx];
                if true_label == class_idx {
                    positive_scores.push(score);
                } else {
                    negative_scores.push(score);
                }
            }
        }

        if !positive_scores.is_empty() && !negative_scores.is_empty() {
            let auc = compute_binary_auc(&positive_scores, &negative_scores);
            auc_sum += auc;
            valid_classes += 1;
        }
    }

    if valid_classes > 0 {
        auc_sum / valid_classes as f64
    } else {
        0.5
    }
}

pub fn compute_binary_auc(positive_scores: &[f64], negative_scores: &[f64]) -> f64 {
    let mut comparison_sum = 0.0;
    for &pos_score in positive_scores {
        for &neg_score in negative_scores {
            if pos_score > neg_score {
                comparison_sum += 1.0;
            } else if (pos_score - neg_score).abs() < 1e-10 {
                comparison_sum += 0.5;
            }
        }
    }
    let total_comparisons = (positive_scores.len() * negative_scores.len()) as f64;
    if total_comparisons > 0.0 {
        comparison_sum / total_comparisons
    } else {
        0.5
    }
}

// ─── Comprehensive metrics ───────────────────────────────────────────────────

pub async fn compute_comprehensive_metrics(
    config: &NeuralPatternConfig,
    weights: &NetworkWeights,
    patterns: &[Pattern],
    target_correlations: &HashMap<(String, String), CorrelationType>,
    training_time: Duration,
) -> Result<ModelMetrics> {
    let predictions = forward_pass(config, weights, patterns).await?;

    let mut all_predictions = Vec::new();
    let mut all_true_labels = Vec::new();
    let mut all_prediction_scores = Vec::new();

    for (i, pattern) in patterns.iter().enumerate() {
        for (j, other_pattern) in patterns.iter().enumerate() {
            if i != j {
                let pattern_pair = (pattern.id().to_string(), other_pattern.id().to_string());
                if let Some(expected_correlation) = target_correlations.get(&pattern_pair) {
                    let pred_row = predictions.row(i);
                    let predicted_type_idx = pred_row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);

                    let true_label_idx = correlation_type_to_index(expected_correlation);
                    all_predictions.push(predicted_type_idx);
                    all_true_labels.push(true_label_idx);
                    all_prediction_scores.push(pred_row.to_vec());
                }
            }
        }
    }

    let num_classes = 8;
    let confusion_matrix =
        compute_confusion_matrix(&all_predictions, &all_true_labels, num_classes);
    let (per_class_metrics, macro_precision, macro_recall, macro_f1) =
        compute_per_class_metrics(&confusion_matrix, num_classes);

    let total_correct: usize = confusion_matrix
        .iter()
        .enumerate()
        .map(|(i, row)| row[i])
        .sum();
    let total_samples: usize = confusion_matrix
        .iter()
        .map(|row| row.iter().sum::<usize>())
        .sum();
    let accuracy = if total_samples > 0 {
        total_correct as f64 / total_samples as f64
    } else {
        0.0
    };

    let auc_roc = compute_auc_roc(&all_prediction_scores, &all_true_labels, num_classes);

    Ok(ModelMetrics {
        accuracy,
        precision: macro_precision,
        recall: macro_recall,
        f1_score: macro_f1,
        auc_roc,
        confusion_matrix,
        per_class_metrics,
        training_time,
    })
}

// ─── Predict correlations ────────────────────────────────────────────────────

pub async fn predict_correlations(
    config: &NeuralPatternConfig,
    weights: &NetworkWeights,
    patterns: &[Pattern],
) -> Result<HashMap<(String, String), (CorrelationType, f64)>> {
    let predictions = forward_pass(config, weights, patterns).await?;
    let mut correlations = HashMap::new();

    for (i, pattern_i) in patterns.iter().enumerate() {
        for (j, pattern_j) in patterns.iter().enumerate() {
            if i == j {
                continue;
            }

            let row = predictions.row(i);
            let max_logit = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row.iter().map(|&x| (x - max_logit).exp()).collect();
            let exp_sum: f64 = exps.iter().sum();

            let (best_idx, best_prob) = if exp_sum > 0.0 {
                exps.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, &exp_val)| (idx, exp_val / exp_sum))
                    .unwrap_or((0, 1.0 / exps.len().max(1) as f64))
            } else {
                (0, 0.0)
            };

            let corr_type = index_to_correlation_type(best_idx);
            let key = (pattern_i.id().to_string(), pattern_j.id().to_string());
            correlations.insert(key, (corr_type, best_prob));
        }
    }

    Ok(correlations)
}
