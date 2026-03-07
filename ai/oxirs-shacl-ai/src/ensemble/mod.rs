//! Ensemble methods for SHACL shape inference
//!
//! Combines multiple ML models for better shape learning accuracy through
//! various aggregation strategies including majority voting, weighted averaging,
//! max-confidence selection, and unanimous agreement.

pub mod stacking;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ShaclAiError;

/// A single shape prediction from one model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapePrediction {
    /// Unique identifier for the predicted shape
    pub shape_id: String,
    /// Type of SHACL constraint (e.g., "sh:minCount", "sh:datatype")
    pub constraint_type: String,
    /// Confidence score in [0.0, 1.0]
    pub confidence: f64,
    /// Triple IRIs that support this prediction
    pub supporting_triples: Vec<String>,
}

/// Features extracted from an RDF graph for ML inference
#[derive(Debug, Clone)]
pub struct GraphFeatures {
    /// Per-node feature vectors
    pub node_features: Vec<NodeFeature>,
    /// Per-edge feature vectors
    pub edge_features: Vec<EdgeFeature>,
    /// Aggregate graph statistics
    pub graph_stats: GraphStats,
}

/// Features for a single RDF node
#[derive(Debug, Clone)]
pub struct NodeFeature {
    /// Node identifier (IRI or blank node)
    pub node_id: String,
    /// One-hot encoded type vector
    pub type_vec: Vec<f64>,
    /// Histogram of predicates used by this node
    pub predicate_histogram: Vec<f64>,
    /// Number of incoming edges
    pub degree_in: usize,
    /// Number of outgoing edges
    pub degree_out: usize,
}

/// Features for a single RDF edge (predicate occurrence)
#[derive(Debug, Clone)]
pub struct EdgeFeature {
    /// Predicate IRI
    pub predicate: String,
    /// RDF type of the source node
    pub source_type: String,
    /// RDF type of the target node
    pub target_type: String,
    /// Relative frequency of this predicate in the graph
    pub frequency: f64,
}

/// High-level statistics for an entire RDF graph
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Total number of distinct nodes
    pub node_count: usize,
    /// Total number of triples (edges)
    pub edge_count: usize,
    /// Number of distinct rdf:type values
    pub type_count: usize,
    /// Number of distinct predicates
    pub predicate_count: usize,
    /// Mean degree across all nodes
    pub avg_degree: f64,
    /// Graph density: edge_count / (node_count * (node_count - 1))
    pub density: f64,
}

/// A labelled training example consisting of graph features and ground-truth shapes
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Input features extracted from the RDF graph
    pub features: GraphFeatures,
    /// Ground-truth SHACL shape predictions for supervised learning
    pub shapes: Vec<ShapePrediction>,
}

/// Metrics produced after training a base learner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Name of the model that produced these metrics
    pub model_name: String,
    /// Number of training epochs completed
    pub epochs: usize,
    /// Final training loss value
    pub final_loss: f64,
    /// Precision on the validation set
    pub precision: f64,
    /// Recall on the validation set
    pub recall: f64,
    /// Harmonic mean of precision and recall
    pub f1_score: f64,
}

/// Strategy used to combine predictions from ensemble members
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleStrategy {
    /// Each model casts a vote; the most-voted prediction wins
    MajorityVoting,
    /// Each model's vote is weighted by its validation performance
    WeightedAverage {
        /// Per-model weights (must align with the order learners were added)
        weights: Vec<f64>,
    },
    /// Select the prediction with the highest confidence per constraint type
    MaxConfidence,
    /// Require all models to agree above a minimum confidence threshold
    Unanimous {
        /// Minimum confidence required from every model
        min_confidence: f64,
    },
}

/// Trait for a base shape learner that can participate in an ensemble
pub trait ShapeLearner: Send + Sync {
    /// Human-readable name for this learner (used in logging and metrics)
    fn name(&self) -> &str;

    /// Produce shape predictions for the given graph features
    fn predict(&self, graph_features: &GraphFeatures)
        -> Result<Vec<ShapePrediction>, ShaclAiError>;

    /// Train (or update) this learner on a set of labelled examples
    fn train(&mut self, examples: &[TrainingExample]) -> Result<TrainingMetrics, ShaclAiError>;
}

/// Ensemble that aggregates predictions from multiple base `ShapeLearner` implementations
pub struct ShapeLearnerEnsemble {
    /// Ordered list of (learner, weight) pairs
    learners: Vec<(Box<dyn ShapeLearner>, f64)>,
    /// How predictions are combined
    strategy: EnsembleStrategy,
    /// Minimum confidence required to include a combined prediction
    min_ensemble_confidence: f64,
}

impl ShapeLearnerEnsemble {
    /// Create a new empty ensemble with the given combination strategy
    pub fn new(strategy: EnsembleStrategy) -> Self {
        Self {
            learners: Vec::new(),
            strategy,
            min_ensemble_confidence: 0.5,
        }
    }

    /// Set the minimum confidence threshold for emitted predictions
    pub fn with_min_confidence(mut self, threshold: f64) -> Self {
        self.min_ensemble_confidence = threshold;
        self
    }

    /// Add a base learner with an initial weight
    pub fn add_learner(&mut self, learner: Box<dyn ShapeLearner>, weight: f64) {
        self.learners.push((learner, weight));
    }

    /// Number of base learners in the ensemble
    pub fn learner_count(&self) -> usize {
        self.learners.len()
    }

    /// Predict using all models and combine results according to the current strategy
    pub fn predict(&self, features: &GraphFeatures) -> Result<Vec<ShapePrediction>, ShaclAiError> {
        if self.learners.is_empty() {
            return Err(ShaclAiError::ModelTraining(
                "Ensemble has no learners".to_string(),
            ));
        }

        // Collect predictions from every learner
        let mut all_predictions: Vec<Vec<ShapePrediction>> = Vec::new();
        for (learner, _weight) in &self.learners {
            let preds = learner.predict(features)?;
            all_predictions.push(preds);
        }

        // Combine using the configured strategy
        let combined = self.combine_predictions(all_predictions);

        // Filter by minimum ensemble confidence
        let filtered = combined
            .into_iter()
            .filter(|p| p.confidence >= self.min_ensemble_confidence)
            .collect();

        Ok(filtered)
    }

    /// Train all member learners sequentially and return per-model metrics
    pub fn train_all(
        &mut self,
        examples: &[TrainingExample],
    ) -> Result<Vec<TrainingMetrics>, ShaclAiError> {
        let mut all_metrics = Vec::with_capacity(self.learners.len());
        for (learner, _weight) in &mut self.learners {
            let metrics = learner.train(examples)?;
            all_metrics.push(metrics);
        }
        Ok(all_metrics)
    }

    /// Update per-learner weights based on their F1 scores from validation
    ///
    /// Weights are normalised to sum to 1.0.  A learner with F1 = 0 receives a
    /// small epsilon weight so it is never completely silenced.
    pub fn update_weights(&mut self, validation_metrics: &[TrainingMetrics]) {
        let eps = 1e-6_f64;
        let n = self.learners.len().min(validation_metrics.len());

        // Accumulate raw F1 scores
        let mut raw_weights: Vec<f64> = (0..n)
            .map(|i| validation_metrics[i].f1_score.max(eps))
            .collect();

        // Pad remaining learners (if any) with equal shares
        while raw_weights.len() < self.learners.len() {
            raw_weights.push(eps);
        }

        let total: f64 = raw_weights.iter().sum();
        for (i, (_learner, weight)) in self.learners.iter_mut().enumerate() {
            *weight = raw_weights[i] / total;
        }

        // Also update WeightedAverage strategy weights to stay in sync.
        // Collect the new weights from learners first (immutable borrow) to avoid
        // a simultaneous mutable borrow of self.strategy (E0502).
        if matches!(self.strategy, EnsembleStrategy::WeightedAverage { .. }) {
            let new_weights: Vec<f64> = self.learners.iter().map(|(_, w)| *w).collect();
            if let EnsembleStrategy::WeightedAverage { ref mut weights } = self.strategy {
                *weights = new_weights;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Dispatch to the correct combination method
    fn combine_predictions(&self, all_preds: Vec<Vec<ShapePrediction>>) -> Vec<ShapePrediction> {
        match &self.strategy {
            EnsembleStrategy::MajorityVoting => Self::majority_vote(&all_preds),
            EnsembleStrategy::WeightedAverage { weights } => {
                Self::weighted_average(&all_preds, weights)
            }
            EnsembleStrategy::MaxConfidence => Self::max_confidence(&all_preds),
            EnsembleStrategy::Unanimous { min_confidence } => {
                self.unanimous(&all_preds, *min_confidence)
            }
        }
    }

    /// Each (shape_id, constraint_type) pair is counted across models.
    /// The most-voted pairs win; confidence is the fraction of models that voted for it.
    fn majority_vote(all_preds: &[Vec<ShapePrediction>]) -> Vec<ShapePrediction> {
        let n_models = all_preds.len();
        if n_models == 0 {
            return Vec::new();
        }

        // Count votes per (shape_id, constraint_type)
        let mut vote_map: HashMap<(String, String), (usize, Vec<String>, f64)> = HashMap::new();
        for preds in all_preds {
            for pred in preds {
                let key = (pred.shape_id.clone(), pred.constraint_type.clone());
                let entry = vote_map.entry(key).or_insert((0, Vec::new(), 0.0));
                entry.0 += 1;
                entry.1.extend(pred.supporting_triples.iter().cloned());
                entry.2 += pred.confidence;
            }
        }

        // Include predictions supported by a strict majority of models
        let majority = (n_models as f64 / 2.0).ceil() as usize;
        vote_map
            .into_iter()
            .filter(|(_, (count, _, _))| *count >= majority)
            .map(
                |((shape_id, constraint_type), (count, supporting, total_conf))| ShapePrediction {
                    shape_id,
                    constraint_type,
                    confidence: total_conf / count as f64,
                    supporting_triples: {
                        let mut v = supporting;
                        v.sort_unstable();
                        v.dedup();
                        v
                    },
                },
            )
            .collect()
    }

    /// Predictions are weighted by each model's weight.
    /// For a (shape_id, constraint_type) pair the combined confidence is the
    /// sum of (weight * confidence) normalised by the total weight of models
    /// that produced that prediction.
    fn weighted_average(
        all_preds: &[Vec<ShapePrediction>],
        weights: &[f64],
    ) -> Vec<ShapePrediction> {
        // Effective weight for model i
        let eff_weight = |i: usize| -> f64 {
            if i < weights.len() {
                weights[i].max(0.0)
            } else {
                1.0
            }
        };

        let total_weight: f64 = (0..all_preds.len()).map(eff_weight).sum();
        if total_weight == 0.0 {
            return Self::majority_vote(all_preds);
        }

        // Accumulate weighted confidence per (shape_id, constraint_type)
        let mut accum: HashMap<(String, String), (f64, f64, Vec<String>)> = HashMap::new();
        for (i, preds) in all_preds.iter().enumerate() {
            let w = eff_weight(i);
            for pred in preds {
                let key = (pred.shape_id.clone(), pred.constraint_type.clone());
                let entry = accum.entry(key).or_insert((0.0, 0.0, Vec::new()));
                entry.0 += pred.confidence * w; // weighted confidence sum
                entry.1 += w; // weight sum
                entry.2.extend(pred.supporting_triples.iter().cloned());
            }
        }

        accum
            .into_iter()
            .map(
                |((shape_id, constraint_type), (conf_sum, weight_sum, supporting))| {
                    ShapePrediction {
                        shape_id,
                        constraint_type,
                        confidence: if weight_sum > 0.0 {
                            conf_sum / weight_sum
                        } else {
                            0.0
                        },
                        supporting_triples: {
                            let mut v = supporting;
                            v.sort_unstable();
                            v.dedup();
                            v
                        },
                    }
                },
            )
            .collect()
    }

    /// For each (shape_id, constraint_type) pair select the prediction with
    /// the highest confidence across all models.
    fn max_confidence(all_preds: &[Vec<ShapePrediction>]) -> Vec<ShapePrediction> {
        let mut best: HashMap<(String, String), ShapePrediction> = HashMap::new();
        for preds in all_preds {
            for pred in preds {
                let key = (pred.shape_id.clone(), pred.constraint_type.clone());
                let entry = best.entry(key).or_insert_with(|| pred.clone());
                if pred.confidence > entry.confidence {
                    *entry = pred.clone();
                }
            }
        }
        best.into_values().collect()
    }

    /// A prediction is included only when *every* model produced it with
    /// confidence >= `min_confidence`.
    fn unanimous(
        &self,
        all_preds: &[Vec<ShapePrediction>],
        min_confidence: f64,
    ) -> Vec<ShapePrediction> {
        let n_models = all_preds.len();
        if n_models == 0 {
            return Vec::new();
        }

        // Count qualifying votes per (shape_id, constraint_type)
        let mut vote_map: HashMap<(String, String), (usize, f64, Vec<String>)> = HashMap::new();
        for preds in all_preds {
            for pred in preds {
                if pred.confidence >= min_confidence {
                    let key = (pred.shape_id.clone(), pred.constraint_type.clone());
                    let entry = vote_map.entry(key).or_insert((0, 0.0, Vec::new()));
                    entry.0 += 1;
                    entry.1 += pred.confidence;
                    entry.2.extend(pred.supporting_triples.iter().cloned());
                }
            }
        }

        vote_map
            .into_iter()
            .filter(|(_, (count, _, _))| *count == n_models)
            .map(
                |((shape_id, constraint_type), (count, conf_sum, supporting))| ShapePrediction {
                    shape_id,
                    constraint_type,
                    confidence: conf_sum / count as f64,
                    supporting_triples: {
                        let mut v = supporting;
                        v.sort_unstable();
                        v.dedup();
                        v
                    },
                },
            )
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Mock learner
    // -----------------------------------------------------------------------

    struct ConstantLearner {
        name: String,
        predictions: Vec<ShapePrediction>,
    }

    impl ShapeLearner for ConstantLearner {
        fn name(&self) -> &str {
            &self.name
        }

        fn predict(&self, _features: &GraphFeatures) -> Result<Vec<ShapePrediction>, ShaclAiError> {
            Ok(self.predictions.clone())
        }

        fn train(
            &mut self,
            _examples: &[TrainingExample],
        ) -> Result<TrainingMetrics, ShaclAiError> {
            Ok(TrainingMetrics {
                model_name: self.name.clone(),
                epochs: 1,
                final_loss: 0.1,
                precision: 0.9,
                recall: 0.8,
                f1_score: 0.85,
            })
        }
    }

    fn dummy_features() -> GraphFeatures {
        GraphFeatures {
            node_features: Vec::new(),
            edge_features: Vec::new(),
            graph_stats: GraphStats {
                node_count: 10,
                edge_count: 20,
                type_count: 3,
                predicate_count: 5,
                avg_degree: 4.0,
                density: 0.22,
            },
        }
    }

    fn make_prediction(shape_id: &str, constraint_type: &str, confidence: f64) -> ShapePrediction {
        ShapePrediction {
            shape_id: shape_id.to_string(),
            constraint_type: constraint_type.to_string(),
            confidence,
            supporting_triples: vec!["<s> <p> <o>".to_string()],
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_majority_voting_basic() {
        let pred_a = make_prediction("PersonShape", "sh:minCount", 0.9);
        let pred_b = make_prediction("PersonShape", "sh:minCount", 0.8);
        let pred_c = make_prediction("PersonShape", "sh:minCount", 0.7);

        let all_preds = vec![
            vec![pred_a.clone()],
            vec![pred_b.clone()],
            vec![pred_c.clone()],
        ];

        let result = ShapeLearnerEnsemble::majority_vote(&all_preds);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].shape_id, "PersonShape");
        // Confidence should be average of the three
        let expected_conf = (0.9 + 0.8 + 0.7) / 3.0;
        assert!((result[0].confidence - expected_conf).abs() < 1e-9);
    }

    #[test]
    fn test_majority_voting_minority_excluded() {
        // Only 1 out of 3 models votes for "OtherShape"
        let pred_common = make_prediction("PersonShape", "sh:minCount", 0.9);
        let pred_rare = make_prediction("OtherShape", "sh:maxCount", 0.95);

        let all_preds = vec![
            vec![pred_common.clone()],
            vec![pred_common.clone()],
            vec![pred_rare.clone()],
        ];

        let result = ShapeLearnerEnsemble::majority_vote(&all_preds);
        assert!(result.iter().all(|p| p.shape_id == "PersonShape"));
        assert!(!result.iter().any(|p| p.shape_id == "OtherShape"));
    }

    #[test]
    fn test_weighted_average() {
        let pred = make_prediction("S", "sh:datatype", 1.0);
        let all_preds = vec![vec![pred.clone()], vec![pred.clone()]];
        let weights = vec![0.8, 0.2];

        let result = ShapeLearnerEnsemble::weighted_average(&all_preds, &weights);
        assert_eq!(result.len(), 1);
        // Both have confidence 1.0 so weighted avg should be 1.0
        assert!((result[0].confidence - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_max_confidence_selects_highest() {
        let low = make_prediction("S", "sh:minCount", 0.4);
        let high = make_prediction("S", "sh:minCount", 0.95);
        let all_preds = vec![vec![low], vec![high]];

        let result = ShapeLearnerEnsemble::max_confidence(&all_preds);
        assert_eq!(result.len(), 1);
        assert!((result[0].confidence - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_unanimous_requires_all_models() {
        let pred = make_prediction("S", "sh:minCount", 0.9);
        // Model 2 does NOT predict this shape
        let all_preds = vec![vec![pred], vec![]];

        let mut ensemble = ShapeLearnerEnsemble::new(EnsembleStrategy::Unanimous {
            min_confidence: 0.5,
        });
        ensemble.add_learner(
            Box::new(ConstantLearner {
                name: "m1".to_string(),
                predictions: vec![make_prediction("S", "sh:minCount", 0.9)],
            }),
            1.0,
        );
        ensemble.add_learner(
            Box::new(ConstantLearner {
                name: "m2".to_string(),
                predictions: Vec::new(),
            }),
            1.0,
        );

        let result = ensemble.unanimous(&all_preds, 0.5);
        assert!(
            result.is_empty(),
            "Unanimous should require all models to agree"
        );
    }

    #[test]
    fn test_ensemble_predict_end_to_end() {
        let predictions = vec![
            make_prediction("PersonShape", "sh:minCount", 0.9),
            make_prediction("PersonShape", "sh:datatype", 0.8),
        ];
        let mut ensemble =
            ShapeLearnerEnsemble::new(EnsembleStrategy::MajorityVoting).with_min_confidence(0.5);

        for i in 0..3 {
            ensemble.add_learner(
                Box::new(ConstantLearner {
                    name: format!("model_{i}"),
                    predictions: predictions.clone(),
                }),
                1.0,
            );
        }

        let result = ensemble.predict(&dummy_features()).expect("predict failed");
        assert!(!result.is_empty());
        assert!(result.iter().all(|p| p.confidence >= 0.5));
    }

    #[test]
    fn test_train_all_returns_metrics() {
        let mut ensemble = ShapeLearnerEnsemble::new(EnsembleStrategy::MajorityVoting);
        ensemble.add_learner(
            Box::new(ConstantLearner {
                name: "m1".to_string(),
                predictions: Vec::new(),
            }),
            1.0,
        );
        ensemble.add_learner(
            Box::new(ConstantLearner {
                name: "m2".to_string(),
                predictions: Vec::new(),
            }),
            1.0,
        );

        let metrics = ensemble.train_all(&[]).expect("train_all failed");
        assert_eq!(metrics.len(), 2);
        assert_eq!(metrics[0].model_name, "m1");
        assert_eq!(metrics[1].model_name, "m2");
    }

    #[test]
    fn test_update_weights_normalises() {
        let mut ensemble = ShapeLearnerEnsemble::new(EnsembleStrategy::MajorityVoting);
        for i in 0..3 {
            ensemble.add_learner(
                Box::new(ConstantLearner {
                    name: format!("m{i}"),
                    predictions: Vec::new(),
                }),
                1.0,
            );
        }

        let metrics = vec![
            TrainingMetrics {
                model_name: "m0".to_string(),
                epochs: 1,
                final_loss: 0.1,
                precision: 0.9,
                recall: 0.9,
                f1_score: 0.9,
            },
            TrainingMetrics {
                model_name: "m1".to_string(),
                epochs: 1,
                final_loss: 0.2,
                precision: 0.6,
                recall: 0.6,
                f1_score: 0.6,
            },
            TrainingMetrics {
                model_name: "m2".to_string(),
                epochs: 1,
                final_loss: 0.3,
                precision: 0.3,
                recall: 0.3,
                f1_score: 0.3,
            },
        ];

        ensemble.update_weights(&metrics);

        let total: f64 = ensemble.learners.iter().map(|(_, w)| w).sum();
        assert!((total - 1.0).abs() < 1e-9, "Weights should sum to 1.0");

        // m0 should have the highest weight
        let w0 = ensemble.learners[0].1;
        let w1 = ensemble.learners[1].1;
        let w2 = ensemble.learners[2].1;
        assert!(w0 > w1, "m0 should outweigh m1");
        assert!(w1 > w2, "m1 should outweigh m2");
    }

    #[test]
    fn test_empty_ensemble_error() {
        let ensemble = ShapeLearnerEnsemble::new(EnsembleStrategy::MajorityVoting);
        let result = ensemble.predict(&dummy_features());
        assert!(result.is_err());
    }
}
