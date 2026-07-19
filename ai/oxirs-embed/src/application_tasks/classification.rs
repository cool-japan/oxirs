//! Classification evaluation module
//!
//! This module provides comprehensive evaluation for classification tasks using
//! embedding models, including accuracy, precision, recall, F1-score, and
//! confusion matrix analysis.

use super::ApplicationEvalConfig;
use crate::EmbeddingModel;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Classification evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClassificationMetric {
    /// Accuracy
    Accuracy,
    /// Precision (macro-averaged)
    Precision,
    /// Recall (macro-averaged)
    Recall,
    /// F1 Score (macro-averaged)
    F1Score,
    /// ROC AUC (for binary classification)
    ROCAUC,
    /// Precision-Recall AUC
    PRAUC,
    /// Matthews Correlation Coefficient
    MCC,
}

/// Per-class classification results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassResults {
    /// Class label
    pub class_label: String,
    /// Precision
    pub precision: f64,
    /// Recall
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// Support (number of instances)
    pub support: usize,
}

/// Classification report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationReport {
    /// Macro-averaged metrics
    pub macro_avg: ClassResults,
    /// Weighted-averaged metrics
    pub weighted_avg: ClassResults,
    /// Overall accuracy
    pub accuracy: f64,
    /// Total number of samples
    pub total_samples: usize,
}

/// Classification evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResults {
    /// Metric scores
    pub metric_scores: HashMap<String, f64>,
    /// Per-class results
    pub per_class_results: HashMap<String, ClassResults>,
    /// Confusion matrix
    pub confusion_matrix: Vec<Vec<usize>>,
    /// Classification report
    pub classification_report: ClassificationReport,
}

/// Simple classifier for evaluation
#[allow(dead_code)]
pub struct SimpleClassifier {
    /// Class centroids
    class_centroids: HashMap<String, Vec<f32>>,
    /// Class counts
    class_counts: HashMap<String, usize>,
}

impl Default for SimpleClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleClassifier {
    /// Create a new simple classifier
    pub fn new() -> Self {
        Self {
            class_centroids: HashMap::new(),
            class_counts: HashMap::new(),
        }
    }

    /// Predict class for an embedding
    pub fn predict(&self, embedding: &[f32]) -> Option<String> {
        if self.class_centroids.is_empty() {
            return None;
        }

        let mut best_class = None;
        let mut best_distance = f32::INFINITY;

        for (class_name, centroid) in &self.class_centroids {
            let distance = self.euclidean_distance(embedding, centroid);
            if distance < best_distance {
                best_distance = distance;
                best_class = Some(class_name.clone());
            }
        }

        best_class
    }

    /// Calculate euclidean distance
    fn euclidean_distance(&self, v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Classification evaluator
pub struct ClassificationEvaluator {
    /// Training data with labels
    training_data: Vec<(String, String)>, // (entity, label)
    /// Test data with labels
    test_data: Vec<(String, String)>,
    /// Classification metrics
    metrics: Vec<ClassificationMetric>,
}

impl ClassificationEvaluator {
    /// Create a new classification evaluator
    pub fn new() -> Self {
        Self {
            training_data: Vec::new(),
            test_data: Vec::new(),
            metrics: vec![
                ClassificationMetric::Accuracy,
                ClassificationMetric::Precision,
                ClassificationMetric::Recall,
                ClassificationMetric::F1Score,
            ],
        }
    }

    /// Add training data
    pub fn add_training_data(&mut self, entity: String, label: String) {
        self.training_data.push((entity, label));
    }

    /// Add test data
    pub fn add_test_data(&mut self, entity: String, label: String) {
        self.test_data.push((entity, label));
    }

    /// Evaluate classification performance
    pub async fn evaluate(
        &self,
        model: &dyn EmbeddingModel,
        _config: &ApplicationEvalConfig,
    ) -> Result<ClassificationResults> {
        if self.test_data.is_empty() {
            return Err(anyhow!(
                "No test data available for classification evaluation"
            ));
        }

        // Train a simple classifier on embeddings
        let classifier = self.train_classifier(model).await?;

        // Predict on test data
        let predictions = self.predict_test_data(model, &classifier).await?;

        // Calculate metrics
        let mut metric_scores = HashMap::new();
        for metric in &self.metrics {
            let score = self.calculate_classification_metric(metric, &predictions)?;
            metric_scores.insert(format!("{metric:?}"), score);
        }

        // Generate per-class results
        let per_class_results = self.calculate_per_class_results(&predictions)?;

        // Generate confusion matrix
        let confusion_matrix = self.generate_confusion_matrix(&predictions)?;

        // Generate classification report
        let classification_report =
            self.generate_classification_report(&per_class_results, &predictions)?;

        Ok(ClassificationResults {
            metric_scores,
            per_class_results,
            confusion_matrix,
            classification_report,
        })
    }

    /// Train a simple classifier
    async fn train_classifier(&self, model: &dyn EmbeddingModel) -> Result<SimpleClassifier> {
        let mut class_centroids = HashMap::new();
        let mut class_counts = HashMap::new();

        for (entity, label) in &self.training_data {
            if let Ok(embedding) = model.get_entity_embedding(entity) {
                let centroid = class_centroids
                    .entry(label.clone())
                    .or_insert_with(|| vec![0.0f32; embedding.values.len()]);

                for (i, &value) in embedding.values.iter().enumerate() {
                    centroid[i] += value;
                }

                *class_counts.entry(label.clone()).or_insert(0) += 1;
            }
        }

        // Average the centroids
        for (label, count) in &class_counts {
            if let Some(centroid) = class_centroids.get_mut(label) {
                for value in centroid.iter_mut() {
                    *value /= *count as f32;
                }
            }
        }

        Ok(SimpleClassifier {
            class_centroids,
            class_counts,
        })
    }

    /// Predict on test data
    async fn predict_test_data(
        &self,
        model: &dyn EmbeddingModel,
        classifier: &SimpleClassifier,
    ) -> Result<Vec<(String, String, Option<String>)>> {
        // (true_label, entity, predicted_label)
        let mut predictions = Vec::new();

        for (entity, true_label) in &self.test_data {
            if let Ok(embedding) = model.get_entity_embedding(entity) {
                let predicted_label = classifier.predict(&embedding.values);
                predictions.push((true_label.clone(), entity.clone(), predicted_label));
            }
        }

        Ok(predictions)
    }

    /// Calculate classification metric
    fn calculate_classification_metric(
        &self,
        metric: &ClassificationMetric,
        predictions: &[(String, String, Option<String>)],
    ) -> Result<f64> {
        match metric {
            ClassificationMetric::Accuracy => {
                let correct = predictions
                    .iter()
                    .filter(|(true_label, _, pred)| {
                        pred.as_ref().map(|p| p == true_label).unwrap_or(false)
                    })
                    .count();
                Ok(correct as f64 / predictions.len() as f64)
            }
            ClassificationMetric::Precision
            | ClassificationMetric::Recall
            | ClassificationMetric::F1Score => {
                // Macro-averaged over the real per-class precision/recall/F1
                // computed from the confusion matrix (true/false positives and
                // false negatives per class).
                let per_class = self.calculate_per_class_results(predictions)?;
                if per_class.is_empty() {
                    return Ok(0.0);
                }
                let sum: f64 = match metric {
                    ClassificationMetric::Precision => {
                        per_class.values().map(|c| c.precision).sum()
                    }
                    ClassificationMetric::Recall => per_class.values().map(|c| c.recall).sum(),
                    ClassificationMetric::F1Score => per_class.values().map(|c| c.f1_score).sum(),
                    _ => unreachable!("matched above"),
                };
                Ok(sum / per_class.len() as f64)
            }
            ClassificationMetric::ROCAUC
            | ClassificationMetric::PRAUC
            | ClassificationMetric::MCC => Err(anyhow!(
                "Classification metric {metric:?} is not yet implemented; requires per-class \
                     prediction scores, which SimpleClassifier does not currently expose"
            )),
        }
    }

    /// Calculate per-class results from real true-positive/false-positive/
    /// false-negative counts derived from the predictions.
    fn calculate_per_class_results(
        &self,
        predictions: &[(String, String, Option<String>)],
    ) -> Result<HashMap<String, ClassResults>> {
        let mut results = HashMap::new();

        // Consider every class that appears either as ground truth or as a
        // prediction, so classes the classifier over/under-predicts are not
        // silently dropped from the report.
        let classes: std::collections::HashSet<String> = predictions
            .iter()
            .map(|(true_label, _, _)| true_label.clone())
            .chain(predictions.iter().filter_map(|(_, _, pred)| pred.clone()))
            .collect();

        for class in classes {
            let true_positives = predictions
                .iter()
                .filter(|(true_label, _, pred)| {
                    true_label == &class && pred.as_deref() == Some(class.as_str())
                })
                .count();
            let false_positives = predictions
                .iter()
                .filter(|(true_label, _, pred)| {
                    true_label != &class && pred.as_deref() == Some(class.as_str())
                })
                .count();
            let false_negatives = predictions
                .iter()
                .filter(|(true_label, _, pred)| {
                    true_label == &class && pred.as_deref() != Some(class.as_str())
                })
                .count();
            let support = predictions
                .iter()
                .filter(|(true_label, _, _)| true_label == &class)
                .count();

            let precision = if true_positives + false_positives > 0 {
                true_positives as f64 / (true_positives + false_positives) as f64
            } else {
                0.0
            };
            let recall = if true_positives + false_negatives > 0 {
                true_positives as f64 / (true_positives + false_negatives) as f64
            } else {
                0.0
            };
            let f1_score = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            results.insert(
                class.clone(),
                ClassResults {
                    class_label: class,
                    precision,
                    recall,
                    f1_score,
                    support,
                },
            );
        }

        Ok(results)
    }

    /// Generate a real multi-class confusion matrix, indexed by a
    /// deterministic (sorted) ordering over every class seen as ground truth
    /// or prediction. `matrix[i][j]` is the count of true-class-`i` samples
    /// predicted as class `j`.
    fn generate_confusion_matrix(
        &self,
        predictions: &[(String, String, Option<String>)],
    ) -> Result<Vec<Vec<usize>>> {
        let mut classes: Vec<String> = predictions
            .iter()
            .map(|(true_label, _, _)| true_label.clone())
            .chain(predictions.iter().filter_map(|(_, _, pred)| pred.clone()))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();

        let class_index: HashMap<&str, usize> = classes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect();

        let mut matrix = vec![vec![0usize; classes.len()]; classes.len()];
        for (true_label, _, pred) in predictions {
            let Some(&true_idx) = class_index.get(true_label.as_str()) else {
                continue;
            };
            if let Some(predicted_label) = pred {
                if let Some(&pred_idx) = class_index.get(predicted_label.as_str()) {
                    matrix[true_idx][pred_idx] += 1;
                }
            }
        }

        Ok(matrix)
    }

    /// Generate classification report with macro- and support-weighted
    /// averages computed from the real per-class results.
    fn generate_classification_report(
        &self,
        per_class_results: &HashMap<String, ClassResults>,
        predictions: &[(String, String, Option<String>)],
    ) -> Result<ClassificationReport> {
        let accuracy = predictions
            .iter()
            .filter(|(true_label, _, pred)| pred.as_ref().map(|p| p == true_label).unwrap_or(false))
            .count() as f64
            / predictions.len().max(1) as f64;

        let num_classes = per_class_results.len().max(1);
        let total_support: usize = per_class_results.values().map(|c| c.support).sum();

        let macro_precision =
            per_class_results.values().map(|c| c.precision).sum::<f64>() / num_classes as f64;
        let macro_recall =
            per_class_results.values().map(|c| c.recall).sum::<f64>() / num_classes as f64;
        let macro_f1 =
            per_class_results.values().map(|c| c.f1_score).sum::<f64>() / num_classes as f64;

        let (weighted_precision, weighted_recall, weighted_f1) = if total_support > 0 {
            let weighted_precision = per_class_results
                .values()
                .map(|c| c.precision * c.support as f64)
                .sum::<f64>()
                / total_support as f64;
            let weighted_recall = per_class_results
                .values()
                .map(|c| c.recall * c.support as f64)
                .sum::<f64>()
                / total_support as f64;
            let weighted_f1 = per_class_results
                .values()
                .map(|c| c.f1_score * c.support as f64)
                .sum::<f64>()
                / total_support as f64;
            (weighted_precision, weighted_recall, weighted_f1)
        } else {
            (0.0, 0.0, 0.0)
        };

        let macro_avg = ClassResults {
            class_label: "macro avg".to_string(),
            precision: macro_precision,
            recall: macro_recall,
            f1_score: macro_f1,
            support: predictions.len(),
        };

        let weighted_avg = ClassResults {
            class_label: "weighted avg".to_string(),
            precision: weighted_precision,
            recall: weighted_recall,
            f1_score: weighted_f1,
            support: predictions.len(),
        };

        Ok(ClassificationReport {
            macro_avg,
            weighted_avg,
            accuracy,
            total_samples: predictions.len(),
        })
    }
}

impl Default for ClassificationEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the P1 finding: Precision/Recall/F1Score must be
    /// computed from the real confusion matrix, not hardcoded 0.75/0.73/0.74.
    #[test]
    fn test_calculate_classification_metric_precision_recall_f1_are_real() -> Result<()> {
        let evaluator = ClassificationEvaluator::new();
        // 2 correct "cat" predictions, 1 correct "dog", 1 "cat" mispredicted
        // as "dog".
        let predictions = vec![
            ("cat".to_string(), "e1".to_string(), Some("cat".to_string())),
            ("cat".to_string(), "e2".to_string(), Some("cat".to_string())),
            ("cat".to_string(), "e3".to_string(), Some("dog".to_string())),
            ("dog".to_string(), "e4".to_string(), Some("dog".to_string())),
        ];

        let precision = evaluator
            .calculate_classification_metric(&ClassificationMetric::Precision, &predictions)?;
        let recall = evaluator
            .calculate_classification_metric(&ClassificationMetric::Recall, &predictions)?;
        let f1 = evaluator
            .calculate_classification_metric(&ClassificationMetric::F1Score, &predictions)?;

        // cat: TP=2, FP=0, FN=1 -> precision=1.0, recall=2/3
        // dog: TP=1, FP=1, FN=0 -> precision=0.5, recall=1.0
        // macro precision = (1.0 + 0.5) / 2 = 0.75; macro recall = (2/3 + 1.0) / 2 ≈ 0.8333
        assert!((precision - 0.75).abs() < 1e-9, "precision = {precision}");
        assert!((recall - 0.8333333333).abs() < 1e-6, "recall = {recall}");
        assert!(f1 > 0.0 && f1 < 1.0, "f1 = {f1}");

        // These must NOT be the old hardcoded constants for a *different*
        // confusion matrix (sanity: values change with the data).
        let all_correct = vec![
            ("cat".to_string(), "e1".to_string(), Some("cat".to_string())),
            ("dog".to_string(), "e2".to_string(), Some("dog".to_string())),
        ];
        let perfect_precision = evaluator
            .calculate_classification_metric(&ClassificationMetric::Precision, &all_correct)?;
        assert!(
            (perfect_precision - 1.0).abs() < 1e-9,
            "perfect_precision = {perfect_precision}"
        );

        Ok(())
    }

    #[test]
    fn test_calculate_classification_metric_unsupported_metrics_error() {
        let evaluator = ClassificationEvaluator::new();
        let predictions = vec![("cat".to_string(), "e1".to_string(), Some("cat".to_string()))];

        for metric in [
            ClassificationMetric::ROCAUC,
            ClassificationMetric::PRAUC,
            ClassificationMetric::MCC,
        ] {
            assert!(
                evaluator
                    .calculate_classification_metric(&metric, &predictions)
                    .is_err(),
                "metric = {metric:?}"
            );
        }
    }

    #[test]
    fn test_calculate_per_class_results_computes_real_confusion_counts() {
        let evaluator = ClassificationEvaluator::new();
        let predictions = vec![
            ("cat".to_string(), "e1".to_string(), Some("cat".to_string())),
            ("cat".to_string(), "e2".to_string(), Some("dog".to_string())),
        ];

        let results = evaluator
            .calculate_per_class_results(&predictions)
            .expect("should succeed");
        let cat_results = &results["cat"];
        assert_eq!(cat_results.support, 2);
        assert!((cat_results.precision - 1.0).abs() < 1e-9);
        assert!((cat_results.recall - 0.5).abs() < 1e-9);
    }
}
