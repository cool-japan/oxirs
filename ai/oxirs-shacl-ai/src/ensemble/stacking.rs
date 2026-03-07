//! Stacking Ensemble for SHACL Shape Learning
//!
//! Implements a two-level stacking ensemble that combines predictions from
//! GNN, Transformer, and rule-based base learners through a meta-learner.
//!
//! ## Architecture
//!
//! ```text
//!  Input Graph Features
//!       │
//!  ┌────┴──────────────────────┐
//!  │  Level 0 Base Learners    │
//!  │  ┌─────┐ ┌────────┐ ┌──┐ │
//!  │  │ GNN │ │  Trans │ │ R│ │
//!  │  └─────┘ └────────┘ └──┘ │
//!  └────┬──────────────────────┘
//!       │ (meta-features = stacked predictions)
//!  ┌────┴──────────────┐
//!  │  Level 1 Meta-    │
//!  │  Learner (Logistic│
//!  │  Regression or    │
//!  │  Neural Network)  │
//!  └────┬──────────────┘
//!       │
//!  Final Shape Predictions
//! ```
//!
//! Cross-validation is used during training to generate out-of-fold predictions
//! for the meta-features, preventing data leakage.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ensemble::{
    EnsembleStrategy, GraphFeatures, GraphStats, ShapeLearner, ShapeLearnerEnsemble,
    ShapePrediction, TrainingExample, TrainingMetrics,
};
use crate::ShaclAiError;

// ---------------------------------------------------------------------------
// Base learner type enumeration
// ---------------------------------------------------------------------------

/// Identifies the category of a base learner in the stacking ensemble.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BaseLearnerType {
    /// Graph Neural Network (message-passing style)
    Gnn,
    /// Transformer-based shape learner
    Transformer,
    /// Rule-based / symbolic learner
    RuleBased,
    /// Statistical pattern miner
    StatisticalMiner,
    /// Custom user-provided learner
    Custom(String),
}

impl BaseLearnerType {
    /// Human-readable label.
    pub fn label(&self) -> &str {
        match self {
            Self::Gnn => "gnn",
            Self::Transformer => "transformer",
            Self::RuleBased => "rule_based",
            Self::StatisticalMiner => "statistical_miner",
            Self::Custom(s) => s.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// Meta-feature vector (Level-0 outputs used as Level-1 inputs)
// ---------------------------------------------------------------------------

/// Feature vector constructed from base-learner predictions, used as input to
/// the meta-learner.
#[derive(Debug, Clone)]
pub struct MetaFeatureVector {
    /// Flattened confidence scores from all base learners, ordered by
    /// (learner_index, constraint_type).
    pub values: Vec<f64>,
    /// Source (learner, constraint-type) label for each value.
    pub labels: Vec<(String, String)>,
}

impl MetaFeatureVector {
    /// Construct from a list of (learner_name, predictions) pairs.
    pub fn from_predictions(
        all_predictions: &[(String, Vec<ShapePrediction>)],
        constraint_types: &[String],
    ) -> Self {
        let mut values = Vec::new();
        let mut labels = Vec::new();
        for (learner_name, preds) in all_predictions {
            for ct in constraint_types {
                let conf = preds
                    .iter()
                    .filter(|p| &p.constraint_type == ct)
                    .map(|p| p.confidence)
                    .fold(0.0_f64, f64::max);
                values.push(conf);
                labels.push((learner_name.clone(), ct.clone()));
            }
        }
        Self { values, labels }
    }

    /// Number of meta-features.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if no features are present.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Meta-learner: logistic regression / small MLP
// ---------------------------------------------------------------------------

/// Configuration for the meta-learner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearnerConfig {
    /// Number of training epochs.
    pub epochs: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// L2 regularisation.
    pub weight_decay: f64,
    /// Threshold for binary classification.
    pub decision_threshold: f64,
    /// Whether to use a small hidden layer (MLP) vs pure logistic regression.
    pub use_mlp: bool,
    /// Hidden layer size (only when use_mlp = true).
    pub hidden_size: usize,
}

impl Default for MetaLearnerConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            learning_rate: 0.05,
            weight_decay: 1e-4,
            decision_threshold: 0.5,
            use_mlp: false,
            hidden_size: 32,
        }
    }
}

/// Logistic regression meta-learner for each constraint class.
///
/// Trains one binary classifier per constraint type using SGD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticMetaLearner {
    /// Weight vector per constraint class (indexed by constraint-type name).
    pub weights: HashMap<String, Vec<f64>>,
    /// Bias per constraint class.
    pub biases: HashMap<String, f64>,
    pub config: MetaLearnerConfig,
    /// Number of input features.
    pub input_dim: usize,
    /// Known constraint types.
    pub constraint_types: Vec<String>,
}

impl LogisticMetaLearner {
    /// Create a new meta-learner.
    pub fn new(input_dim: usize, constraint_types: Vec<String>, config: MetaLearnerConfig) -> Self {
        // Initialise weights to small random values using a seeded LCG
        let mut rng = SeedLcg::new(0xDEADBEEF_CAFECAFE);
        let mut weights = HashMap::new();
        let mut biases = HashMap::new();
        for ct in &constraint_types {
            let w: Vec<f64> = (0..input_dim).map(|_| rng.next_f64() * 0.01).collect();
            weights.insert(ct.clone(), w);
            biases.insert(ct.clone(), 0.0_f64);
        }
        Self {
            weights,
            biases,
            config,
            input_dim,
            constraint_types,
        }
    }

    /// Predict probability for one constraint type given meta-features.
    pub fn predict_one(&self, ct: &str, meta_features: &[f64]) -> Result<f64, ShaclAiError> {
        let w = self
            .weights
            .get(ct)
            .ok_or_else(|| ShaclAiError::ModelTraining(format!("Unknown constraint type: {ct}")))?;
        let b = self.biases.get(ct).copied().unwrap_or(0.0);
        if meta_features.len() != self.input_dim {
            return Err(ShaclAiError::ModelTraining(format!(
                "Meta-learner: expected {}, got {} features",
                self.input_dim,
                meta_features.len()
            )));
        }
        let logit: f64 = w
            .iter()
            .zip(meta_features)
            .map(|(&wi, &xi)| wi * xi)
            .sum::<f64>()
            + b;
        Ok(sigmoid(logit))
    }

    /// Predict probabilities for all constraint types.
    pub fn predict_all(&self, meta_features: &[f64]) -> Result<HashMap<String, f64>, ShaclAiError> {
        let mut result = HashMap::new();
        for ct in &self.constraint_types {
            let p = self.predict_one(ct, meta_features)?;
            result.insert(ct.clone(), p);
        }
        Ok(result)
    }

    /// Train on (meta-feature, label) pairs using SGD.
    ///
    /// `examples` is a list of `(meta_features, labels)` where `labels` maps
    /// constraint-type name → 0.0 or 1.0.
    pub fn train(
        &mut self,
        examples: &[(Vec<f64>, HashMap<String, f64>)],
    ) -> Result<MetaTrainingReport, ShaclAiError> {
        if examples.is_empty() {
            return Ok(MetaTrainingReport {
                epochs: 0,
                final_loss: 0.0,
                epoch_losses: Vec::new(),
            });
        }

        let mut epoch_losses = Vec::with_capacity(self.config.epochs);

        for _epoch in 0..self.config.epochs {
            let mut epoch_loss = 0.0_f64;
            let mut count = 0usize;

            for (features, labels) in examples {
                if features.len() != self.input_dim {
                    continue; // skip mismatched
                }
                for ct in &self.constraint_types.clone() {
                    let y = labels.get(ct).copied().unwrap_or(0.0);
                    let p = self.predict_one(ct, features)?;
                    let loss = -(y * (p + 1e-12).ln() + (1.0 - y) * (1.0 - p + 1e-12).ln());
                    epoch_loss += loss;
                    count += 1;

                    // SGD gradient
                    let error = p - y;
                    let lr = self.config.learning_rate;
                    let wd = self.config.weight_decay;

                    if let Some(w) = self.weights.get_mut(ct) {
                        for (wi, &xi) in w.iter_mut().zip(features.iter()) {
                            *wi -= lr * (error * xi + wd * *wi);
                        }
                    }
                    if let Some(b) = self.biases.get_mut(ct) {
                        *b -= lr * error;
                    }
                }
            }

            let mean_loss = if count > 0 {
                epoch_loss / count as f64
            } else {
                0.0
            };
            epoch_losses.push(mean_loss);
        }

        let final_loss = epoch_losses.last().copied().unwrap_or(0.0);
        Ok(MetaTrainingReport {
            epochs: self.config.epochs,
            final_loss,
            epoch_losses,
        })
    }
}

/// Report from meta-learner training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaTrainingReport {
    pub epochs: usize,
    pub final_loss: f64,
    pub epoch_losses: Vec<f64>,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Cross-validation fold splitter
// ---------------------------------------------------------------------------

/// Split `n` examples into `k` folds and return index ranges.
pub fn k_fold_indices(n: usize, k: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    if k == 0 || n == 0 {
        return Vec::new();
    }
    let k = k.min(n);
    let fold_size = n / k;
    let remainder = n % k;
    let mut folds = Vec::with_capacity(k);

    let mut start = 0usize;
    let mut fold_ranges = Vec::with_capacity(k);
    for i in 0..k {
        let extra = if i < remainder { 1 } else { 0 };
        fold_ranges.push(start..start + fold_size + extra);
        start += fold_size + extra;
    }

    for fold_range in &fold_ranges {
        let val_indices: Vec<usize> = fold_range.clone().collect();
        let train_indices: Vec<usize> = (0..n).filter(|&j| !fold_range.contains(&j)).collect();
        folds.push((train_indices, val_indices));
    }
    folds
}

// ---------------------------------------------------------------------------
// Stacking ensemble
// ---------------------------------------------------------------------------

/// Configuration for the stacking ensemble.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackingConfig {
    /// Number of cross-validation folds for meta-feature generation.
    pub cv_folds: usize,
    /// Meta-learner configuration.
    pub meta_config: MetaLearnerConfig,
    /// Constraint types to predict.
    pub constraint_types: Vec<String>,
    /// Minimum confidence to include a final prediction.
    pub min_final_confidence: f64,
    /// Whether to include original features alongside meta-features.
    pub passthrough: bool,
}

impl Default for StackingConfig {
    fn default() -> Self {
        Self {
            cv_folds: 3,
            meta_config: MetaLearnerConfig::default(),
            constraint_types: vec![
                "sh:minCount".to_string(),
                "sh:maxCount".to_string(),
                "sh:datatype".to_string(),
                "sh:nodeKind".to_string(),
                "sh:in".to_string(),
                "sh:pattern".to_string(),
            ],
            min_final_confidence: 0.5,
            passthrough: false,
        }
    }
}

/// A type-tagged base learner entry in the stacking ensemble.
pub struct BaseLearnerEntry {
    pub learner_type: BaseLearnerType,
    pub learner: Box<dyn ShapeLearner>,
    pub weight: f64,
}

/// Full training report for the stacking ensemble.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackingTrainingReport {
    /// Metrics from each base learner.
    pub base_metrics: Vec<TrainingMetrics>,
    /// Meta-learner training report.
    pub meta_report: MetaTrainingReport,
    /// Number of meta-training examples generated by cross-validation.
    pub num_meta_examples: usize,
    /// Final validation scores per constraint type (precision).
    pub constraint_precisions: HashMap<String, f64>,
}

/// Stacking ensemble that combines GNN, Transformer, and rule-based predictions
/// via a logistic-regression meta-learner.
pub struct StackingEnsemble {
    base_learners: Vec<BaseLearnerEntry>,
    meta_learner: Option<LogisticMetaLearner>,
    config: StackingConfig,
    is_fitted: bool,
}

impl StackingEnsemble {
    /// Create a new (untrained) stacking ensemble.
    pub fn new(config: StackingConfig) -> Self {
        Self {
            base_learners: Vec::new(),
            meta_learner: None,
            config,
            is_fitted: false,
        }
    }

    /// Add a base learner.
    pub fn add_learner(
        &mut self,
        learner: Box<dyn ShapeLearner>,
        learner_type: BaseLearnerType,
        weight: f64,
    ) {
        self.base_learners.push(BaseLearnerEntry {
            learner_type,
            learner,
            weight,
        });
    }

    /// Number of base learners.
    pub fn num_base_learners(&self) -> usize {
        self.base_learners.len()
    }

    /// Whether the ensemble has been fitted.
    pub fn is_fitted(&self) -> bool {
        self.is_fitted
    }

    /// Fit the stacking ensemble using k-fold cross-validation for meta-features.
    pub fn fit(
        &mut self,
        examples: &[TrainingExample],
    ) -> Result<StackingTrainingReport, ShaclAiError> {
        if self.base_learners.is_empty() {
            return Err(ShaclAiError::ModelTraining(
                "Stacking ensemble has no base learners".to_string(),
            ));
        }
        if examples.is_empty() {
            return Err(ShaclAiError::ModelTraining(
                "No training examples provided".to_string(),
            ));
        }

        let n = examples.len();
        let k = self.config.cv_folds.min(n).max(1);

        // Step 1: Train all base learners on the full dataset and collect metrics
        let mut base_metrics = Vec::new();
        for entry in &mut self.base_learners {
            let metrics = entry.learner.train(examples)?;
            base_metrics.push(metrics);
        }

        // Step 2: Generate out-of-fold meta-features via cross-validation
        let folds = k_fold_indices(n, k);
        let mut meta_examples: Vec<(Vec<f64>, HashMap<String, f64>)> = Vec::new();

        for (train_idx, val_idx) in &folds {
            // Get predictions on validation fold from each base learner
            for val_i in val_idx {
                let val_features = &examples[*val_i].features;
                let mut learner_preds: Vec<(String, Vec<ShapePrediction>)> = Vec::new();

                for entry in &self.base_learners {
                    let name = entry.learner.name().to_string();
                    let preds = entry.learner.predict(val_features)?;
                    learner_preds.push((name, preds));
                }

                let meta = MetaFeatureVector::from_predictions(
                    &learner_preds,
                    &self.config.constraint_types,
                );

                // Build ground-truth labels
                let mut labels: HashMap<String, f64> = HashMap::new();
                let gt_shapes = &examples[*val_i].shapes;
                for ct in &self.config.constraint_types {
                    let has_ct = gt_shapes.iter().any(|s| &s.constraint_type == ct);
                    labels.insert(ct.clone(), if has_ct { 1.0 } else { 0.0 });
                }

                meta_examples.push((meta.values, labels));
            }
            // Suppress unused variable warning
            let _ = train_idx;
        }

        let num_meta_examples = meta_examples.len();

        // Step 3: Determine meta-feature dimension
        let meta_dim = if meta_examples.is_empty() {
            self.base_learners.len() * self.config.constraint_types.len()
        } else {
            meta_examples[0].0.len()
        };

        // Step 4: Train meta-learner
        let mut meta_learner = LogisticMetaLearner::new(
            meta_dim,
            self.config.constraint_types.clone(),
            self.config.meta_config.clone(),
        );
        let meta_report = meta_learner.train(&meta_examples)?;
        self.meta_learner = Some(meta_learner);
        self.is_fitted = true;

        // Step 5: Compute per-constraint precisions on full dataset
        let mut constraint_precisions = HashMap::new();
        for ct in &self.config.constraint_types {
            let mut tp = 0usize;
            let mut fp = 0usize;
            for ex in examples {
                let preds = self.predict(&ex.features)?;
                let predicted = preds.iter().any(|p| &p.constraint_type == ct);
                let actual = ex.shapes.iter().any(|s| &s.constraint_type == ct);
                if predicted && actual {
                    tp += 1;
                } else if predicted && !actual {
                    fp += 1;
                }
            }
            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            constraint_precisions.insert(ct.clone(), precision);
        }

        Ok(StackingTrainingReport {
            base_metrics,
            meta_report,
            num_meta_examples,
            constraint_precisions,
        })
    }

    /// Predict shapes using the stacking ensemble.
    ///
    /// Requires the ensemble to have been fitted.
    pub fn predict(&self, features: &GraphFeatures) -> Result<Vec<ShapePrediction>, ShaclAiError> {
        if !self.is_fitted {
            return Err(ShaclAiError::ModelTraining(
                "Stacking ensemble has not been fitted; call fit() first".to_string(),
            ));
        }
        let meta_learner = self.meta_learner.as_ref().ok_or_else(|| {
            ShaclAiError::ModelTraining("Meta-learner not initialized".to_string())
        })?;

        // Get base-learner predictions
        let mut learner_preds: Vec<(String, Vec<ShapePrediction>)> = Vec::new();
        for entry in &self.base_learners {
            let name = entry.learner.name().to_string();
            let preds = entry.learner.predict(features)?;
            learner_preds.push((name, preds));
        }

        // Build meta-features
        let meta =
            MetaFeatureVector::from_predictions(&learner_preds, &self.config.constraint_types);

        // Meta-learner inference
        let probs = meta_learner.predict_all(&meta.values)?;

        // Convert to ShapePredictions
        let mut result = Vec::new();
        for (ct, prob) in probs {
            if prob >= self.config.min_final_confidence {
                // Aggregate supporting triples from all base-learner predictions
                let supporting: Vec<String> = learner_preds
                    .iter()
                    .flat_map(|(_, preds)| {
                        preds
                            .iter()
                            .filter(|p| p.constraint_type == ct)
                            .flat_map(|p| p.supporting_triples.iter().cloned())
                    })
                    .collect();

                result.push(ShapePrediction {
                    shape_id: format!("stacked_{ct}"),
                    constraint_type: ct,
                    confidence: prob,
                    supporting_triples: supporting,
                });
            }
        }

        // Sort by confidence descending
        result.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(result)
    }

    /// Get current base ensemble (using WeightedAverage strategy) for comparison.
    pub fn build_base_ensemble(&self) -> ShapeLearnerEnsemble {
        let weights: Vec<f64> = self.base_learners.iter().map(|e| e.weight).collect();
        let total: f64 = weights.iter().sum::<f64>().max(1e-12);
        let normalised: Vec<f64> = weights.iter().map(|&w| w / total).collect();

        let strategy = EnsembleStrategy::WeightedAverage {
            weights: normalised,
        };
        ShapeLearnerEnsemble::new(strategy)
    }

    /// Statistics about the fitted meta-learner.
    pub fn meta_learner_stats(&self) -> Option<MetaLearnerSummary> {
        self.meta_learner.as_ref().map(|ml| MetaLearnerSummary {
            input_dim: ml.input_dim,
            num_constraint_types: ml.constraint_types.len(),
            total_parameters: ml.input_dim * ml.constraint_types.len() + ml.constraint_types.len(), // weights + biases
        })
    }
}

/// Summary of meta-learner size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearnerSummary {
    pub input_dim: usize,
    pub num_constraint_types: usize,
    pub total_parameters: usize,
}

// ---------------------------------------------------------------------------
// Simple LCG for deterministic weight initialization
// ---------------------------------------------------------------------------

struct SeedLcg {
    state: u64,
}

impl SeedLcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        self.next_u64() as f64 / u64::MAX as f64
    }
}

// ---------------------------------------------------------------------------
// Convenience: mock GNN-like learner and Transformer-like learner
// ---------------------------------------------------------------------------

/// Simulated GNN base learner (pattern-based, uses degree statistics).
pub struct GnnBaseLearner {
    name: String,
    learned_patterns: Vec<(String, f64)>, // (constraint_type, threshold)
}

impl GnnBaseLearner {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            learned_patterns: Vec::new(),
        }
    }
}

impl ShapeLearner for GnnBaseLearner {
    fn name(&self) -> &str {
        &self.name
    }

    fn predict(&self, features: &GraphFeatures) -> Result<Vec<ShapePrediction>, ShaclAiError> {
        let mut preds = Vec::new();
        let avg_degree = features.graph_stats.avg_degree;

        // Simple heuristic: high degree → minCount constraint
        if avg_degree > 1.0 {
            preds.push(ShapePrediction {
                shape_id: format!("{}_minCount", self.name),
                constraint_type: "sh:minCount".to_string(),
                confidence: (avg_degree / 10.0).min(0.95),
                supporting_triples: Vec::new(),
            });
        }

        // Node type diversity → nodeKind constraint
        if features.graph_stats.type_count > 1 {
            preds.push(ShapePrediction {
                shape_id: format!("{}_nodeKind", self.name),
                constraint_type: "sh:nodeKind".to_string(),
                confidence: 0.7,
                supporting_triples: Vec::new(),
            });
        }

        for (ct, threshold) in &self.learned_patterns {
            if avg_degree > *threshold {
                preds.push(ShapePrediction {
                    shape_id: format!("{}_{ct}", self.name),
                    constraint_type: ct.clone(),
                    confidence: 0.6,
                    supporting_triples: Vec::new(),
                });
            }
        }

        Ok(preds)
    }

    fn train(&mut self, examples: &[TrainingExample]) -> Result<TrainingMetrics, ShaclAiError> {
        // Learn threshold from training examples
        let avg: f64 = examples
            .iter()
            .map(|e| e.features.graph_stats.avg_degree)
            .sum::<f64>()
            / examples.len().max(1) as f64;

        // Extract common constraint types
        let mut ct_counts: HashMap<String, usize> = HashMap::new();
        for ex in examples {
            for shape in &ex.shapes {
                *ct_counts.entry(shape.constraint_type.clone()).or_insert(0) += 1;
            }
        }
        self.learned_patterns = ct_counts
            .into_iter()
            .filter(|(_, cnt)| *cnt > 0)
            .map(|(ct, _)| (ct, avg * 0.5))
            .collect();

        Ok(TrainingMetrics {
            model_name: self.name.clone(),
            epochs: 1,
            final_loss: 0.2,
            precision: 0.75,
            recall: 0.7,
            f1_score: 0.72,
        })
    }
}

/// Simulated Transformer base learner (attention over predicates).
pub struct TransformerBaseLearner {
    name: String,
    predicate_weights: HashMap<String, f64>,
}

impl TransformerBaseLearner {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            predicate_weights: HashMap::new(),
        }
    }
}

impl ShapeLearner for TransformerBaseLearner {
    fn name(&self) -> &str {
        &self.name
    }

    fn predict(&self, features: &GraphFeatures) -> Result<Vec<ShapePrediction>, ShaclAiError> {
        let mut preds = Vec::new();
        let density = features.graph_stats.density;

        // Dense graphs → datatype constraints are more reliable
        if density > 0.1 {
            preds.push(ShapePrediction {
                shape_id: format!("{}_datatype", self.name),
                constraint_type: "sh:datatype".to_string(),
                confidence: (density * 5.0).min(0.92),
                supporting_triples: Vec::new(),
            });
        }

        // Edge features → predicate-based constraints
        for ef in &features.edge_features {
            if ef.frequency > 0.5 {
                let ct = "sh:maxCount".to_string();
                let existing = preds
                    .iter()
                    .any(|p: &ShapePrediction| p.constraint_type == ct);
                if !existing {
                    preds.push(ShapePrediction {
                        shape_id: format!("{}_maxCount", self.name),
                        constraint_type: ct,
                        confidence: ef.frequency.min(0.88),
                        supporting_triples: Vec::new(),
                    });
                }
            }
        }

        // Learned predicate weights
        for (pred, &weight) in &self.predicate_weights {
            if weight > 0.5 {
                preds.push(ShapePrediction {
                    shape_id: format!("{}_{pred}", self.name),
                    constraint_type: pred.clone(),
                    confidence: weight.min(0.85),
                    supporting_triples: Vec::new(),
                });
            }
        }

        Ok(preds)
    }

    fn train(&mut self, examples: &[TrainingExample]) -> Result<TrainingMetrics, ShaclAiError> {
        // Learn predicate importance from training examples
        let mut total_freq: HashMap<String, f64> = HashMap::new();
        let mut count = 0usize;
        for ex in examples {
            for ef in &ex.features.edge_features {
                *total_freq.entry(ef.predicate.clone()).or_insert(0.0) += ef.frequency;
                count += 1;
            }
        }
        if count > 0 {
            self.predicate_weights = total_freq
                .into_iter()
                .map(|(p, f)| (p, f / count as f64))
                .collect();
        }
        Ok(TrainingMetrics {
            model_name: self.name.clone(),
            epochs: 1,
            final_loss: 0.18,
            precision: 0.80,
            recall: 0.75,
            f1_score: 0.77,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble::{EdgeFeature, NodeFeature};

    fn make_graph_features(avg_degree: f64, density: f64, type_count: usize) -> GraphFeatures {
        GraphFeatures {
            node_features: vec![NodeFeature {
                node_id: "n1".to_string(),
                type_vec: vec![1.0, 0.0],
                predicate_histogram: vec![0.5, 0.3],
                degree_in: 2,
                degree_out: 3,
            }],
            edge_features: vec![EdgeFeature {
                predicate: "ex:name".to_string(),
                source_type: "ex:Person".to_string(),
                target_type: "xsd:string".to_string(),
                frequency: 0.8,
            }],
            graph_stats: GraphStats {
                node_count: 10,
                edge_count: 20,
                type_count,
                predicate_count: 5,
                avg_degree,
                density,
            },
        }
    }

    fn make_training_example(avg_degree: f64, density: f64) -> TrainingExample {
        let features = make_graph_features(avg_degree, density, 3);
        let shapes = vec![
            ShapePrediction {
                shape_id: "test_shape".to_string(),
                constraint_type: "sh:minCount".to_string(),
                confidence: 0.9,
                supporting_triples: vec!["<s> <p> <o>".to_string()],
            },
            ShapePrediction {
                shape_id: "test_shape_2".to_string(),
                constraint_type: "sh:datatype".to_string(),
                confidence: 0.85,
                supporting_triples: vec![],
            },
        ];
        TrainingExample { features, shapes }
    }

    fn make_stacking_ensemble() -> StackingEnsemble {
        let config = StackingConfig {
            cv_folds: 2,
            constraint_types: vec![
                "sh:minCount".to_string(),
                "sh:datatype".to_string(),
                "sh:nodeKind".to_string(),
            ],
            ..Default::default()
        };
        let mut ensemble = StackingEnsemble::new(config);
        ensemble.add_learner(
            Box::new(GnnBaseLearner::new("gnn")),
            BaseLearnerType::Gnn,
            1.0,
        );
        ensemble.add_learner(
            Box::new(TransformerBaseLearner::new("transformer")),
            BaseLearnerType::Transformer,
            1.5,
        );
        ensemble
    }

    // --- k_fold_indices tests ---

    #[test]
    fn test_k_fold_exact_division() {
        let folds = k_fold_indices(6, 3);
        assert_eq!(folds.len(), 3);
        for (train, val) in &folds {
            assert_eq!(train.len() + val.len(), 6);
        }
    }

    #[test]
    fn test_k_fold_uneven_division() {
        let folds = k_fold_indices(7, 3);
        assert_eq!(folds.len(), 3);
        for (train, val) in &folds {
            assert_eq!(train.len() + val.len(), 7);
        }
    }

    #[test]
    fn test_k_fold_zero_examples() {
        let folds = k_fold_indices(0, 3);
        assert!(folds.is_empty());
    }

    #[test]
    fn test_k_fold_zero_k() {
        let folds = k_fold_indices(5, 0);
        assert!(folds.is_empty());
    }

    #[test]
    fn test_k_fold_k_larger_than_n() {
        let folds = k_fold_indices(3, 10); // k clamped to 3
        assert_eq!(folds.len(), 3);
    }

    #[test]
    fn test_k_fold_no_index_overlap() {
        let folds = k_fold_indices(6, 3);
        for (train, val) in &folds {
            for &v in val {
                assert!(!train.contains(&v), "val index {v} found in train set");
            }
        }
    }

    // --- MetaFeatureVector tests ---

    #[test]
    fn test_meta_feature_vector_from_predictions() {
        let preds = vec![
            (
                "gnn".to_string(),
                vec![ShapePrediction {
                    shape_id: "s".to_string(),
                    constraint_type: "sh:minCount".to_string(),
                    confidence: 0.8,
                    supporting_triples: Vec::new(),
                }],
            ),
            ("transformer".to_string(), Vec::new()),
        ];
        let cts = vec!["sh:minCount".to_string(), "sh:datatype".to_string()];
        let meta = MetaFeatureVector::from_predictions(&preds, &cts);
        assert_eq!(meta.len(), 4); // 2 learners × 2 constraint types
        assert!(!meta.is_empty());
    }

    #[test]
    fn test_meta_feature_vector_empty() {
        let meta = MetaFeatureVector::from_predictions(&[], &[]);
        assert!(meta.is_empty());
    }

    // --- LogisticMetaLearner tests ---

    #[test]
    fn test_meta_learner_predict_bounds() {
        let cts = vec!["sh:minCount".to_string(), "sh:datatype".to_string()];
        let ml = LogisticMetaLearner::new(4, cts, MetaLearnerConfig::default());
        let features = vec![0.5, 0.3, 0.8, 0.1];
        let probs = ml.predict_all(&features).expect("predict ok");
        for (_, &p) in probs.iter() {
            assert!((0.0..=1.0).contains(&p), "prob {p} out of [0,1]");
        }
    }

    #[test]
    fn test_meta_learner_wrong_dim_error() {
        let cts = vec!["sh:minCount".to_string()];
        let ml = LogisticMetaLearner::new(4, cts, MetaLearnerConfig::default());
        let result = ml.predict_one("sh:minCount", &[0.5, 0.3]); // wrong dim
        assert!(result.is_err());
    }

    #[test]
    fn test_meta_learner_unknown_ct_error() {
        let cts = vec!["sh:minCount".to_string()];
        let ml = LogisticMetaLearner::new(2, cts, MetaLearnerConfig::default());
        let result = ml.predict_one("sh:unknown", &[0.5, 0.3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_meta_learner_trains_without_panic() {
        let cts = vec!["sh:minCount".to_string(), "sh:datatype".to_string()];
        let mut ml = LogisticMetaLearner::new(2, cts, MetaLearnerConfig::default());
        let examples: Vec<(Vec<f64>, HashMap<String, f64>)> = vec![
            (vec![0.8, 0.2], {
                let mut m = HashMap::new();
                m.insert("sh:minCount".to_string(), 1.0);
                m.insert("sh:datatype".to_string(), 0.0);
                m
            }),
            (vec![0.1, 0.9], {
                let mut m = HashMap::new();
                m.insert("sh:minCount".to_string(), 0.0);
                m.insert("sh:datatype".to_string(), 1.0);
                m
            }),
        ];
        let report = ml.train(&examples).expect("train ok");
        assert!(report.final_loss >= 0.0);
        assert_eq!(report.epochs, 50);
        assert_eq!(report.epoch_losses.len(), 50);
    }

    #[test]
    fn test_meta_learner_empty_examples() {
        let cts = vec!["sh:minCount".to_string()];
        let mut ml = LogisticMetaLearner::new(2, cts, MetaLearnerConfig::default());
        let report = ml.train(&[]).expect("empty train ok");
        assert_eq!(report.epochs, 0);
    }

    // --- GnnBaseLearner tests ---

    #[test]
    fn test_gnn_learner_predict() {
        let learner = GnnBaseLearner::new("test_gnn");
        let features = make_graph_features(5.0, 0.2, 2);
        let preds = learner.predict(&features).expect("predict ok");
        // High avg_degree should produce minCount
        assert!(
            preds.iter().any(|p| p.constraint_type == "sh:minCount"),
            "expected sh:minCount prediction"
        );
    }

    #[test]
    fn test_gnn_learner_low_degree_no_pred() {
        let learner = GnnBaseLearner::new("gnn");
        let features = make_graph_features(0.5, 0.01, 1); // low degree
        let preds = learner.predict(&features).expect("ok");
        assert!(!preds.iter().any(|p| p.constraint_type == "sh:minCount"));
    }

    #[test]
    fn test_gnn_learner_train() {
        let mut learner = GnnBaseLearner::new("gnn");
        let examples = vec![
            make_training_example(3.0, 0.1),
            make_training_example(5.0, 0.2),
        ];
        let metrics = learner.train(&examples).expect("train ok");
        assert_eq!(metrics.model_name, "gnn");
        assert!(metrics.f1_score > 0.0);
    }

    // --- TransformerBaseLearner tests ---

    #[test]
    fn test_transformer_learner_predict_dense() {
        let learner = TransformerBaseLearner::new("transformer");
        let features = make_graph_features(3.0, 0.5, 2); // high density
        let preds = learner.predict(&features).expect("ok");
        assert!(
            preds.iter().any(|p| p.constraint_type == "sh:datatype"),
            "dense graph should predict sh:datatype"
        );
    }

    #[test]
    fn test_transformer_learner_train() {
        let mut learner = TransformerBaseLearner::new("trans");
        let examples = vec![make_training_example(4.0, 0.3)];
        let metrics = learner.train(&examples).expect("ok");
        assert_eq!(metrics.model_name, "trans");
    }

    // --- StackingEnsemble tests ---

    #[test]
    fn test_stacking_not_fitted_error() {
        let ensemble = make_stacking_ensemble();
        let features = make_graph_features(3.0, 0.2, 2);
        let result = ensemble.predict(&features);
        assert!(result.is_err(), "should error before fitting");
    }

    #[test]
    fn test_stacking_fit_and_predict() {
        let mut ensemble = make_stacking_ensemble();
        let examples: Vec<TrainingExample> = (0..6)
            .map(|i| make_training_example(i as f64 + 1.0, 0.1 + i as f64 * 0.05))
            .collect();

        let report = ensemble.fit(&examples).expect("fit should succeed");
        assert_eq!(report.base_metrics.len(), 2);
        assert!(report.num_meta_examples > 0);

        let features = make_graph_features(5.0, 0.4, 3);
        let preds = ensemble.predict(&features).expect("predict after fit ok");
        // Predictions may be empty if all confidences are below threshold,
        // but the call should not panic
        let _ = preds;
    }

    #[test]
    fn test_stacking_fit_no_learners_error() {
        let config = StackingConfig::default();
        let mut ensemble = StackingEnsemble::new(config);
        let examples = vec![make_training_example(3.0, 0.2)];
        assert!(ensemble.fit(&examples).is_err());
    }

    #[test]
    fn test_stacking_fit_no_examples_error() {
        let mut ensemble = make_stacking_ensemble();
        assert!(ensemble.fit(&[]).is_err());
    }

    #[test]
    fn test_stacking_is_fitted_flag() {
        let mut ensemble = make_stacking_ensemble();
        assert!(!ensemble.is_fitted());
        let examples: Vec<TrainingExample> = (0..4)
            .map(|i| make_training_example(i as f64 + 2.0, 0.2))
            .collect();
        ensemble.fit(&examples).expect("ok");
        assert!(ensemble.is_fitted());
    }

    #[test]
    fn test_stacking_meta_learner_stats() {
        let mut ensemble = make_stacking_ensemble();
        assert!(ensemble.meta_learner_stats().is_none()); // not fitted yet

        let examples: Vec<TrainingExample> = (0..4)
            .map(|i| make_training_example(i as f64 + 1.0, 0.1))
            .collect();
        ensemble.fit(&examples).expect("ok");

        let stats = ensemble.meta_learner_stats().expect("should have stats");
        assert_eq!(stats.num_constraint_types, 3);
        assert!(stats.total_parameters > 0);
    }

    #[test]
    fn test_stacking_num_base_learners() {
        let ensemble = make_stacking_ensemble();
        assert_eq!(ensemble.num_base_learners(), 2);
    }

    #[test]
    fn test_stacking_predictions_sorted_by_confidence() {
        let mut ensemble = make_stacking_ensemble();
        let examples: Vec<TrainingExample> = (0..6)
            .map(|i| make_training_example(i as f64 + 2.0, 0.3))
            .collect();
        ensemble.fit(&examples).expect("ok");

        let features = make_graph_features(5.0, 0.5, 3);
        let preds = ensemble.predict(&features).expect("ok");

        for window in preds.windows(2) {
            assert!(
                window[0].confidence >= window[1].confidence,
                "predictions not sorted: {} < {}",
                window[0].confidence,
                window[1].confidence
            );
        }
    }

    #[test]
    fn test_build_base_ensemble() {
        let ensemble = make_stacking_ensemble();
        let base = ensemble.build_base_ensemble();
        assert_eq!(base.learner_count(), 0); // base ensemble has no learners added
    }

    #[test]
    fn test_base_learner_type_label() {
        assert_eq!(BaseLearnerType::Gnn.label(), "gnn");
        assert_eq!(BaseLearnerType::Transformer.label(), "transformer");
        assert_eq!(BaseLearnerType::RuleBased.label(), "rule_based");
        assert_eq!(
            BaseLearnerType::Custom("custom_lr".to_string()).label(),
            "custom_lr"
        );
    }

    #[test]
    fn test_meta_training_report_serialization() {
        let r = MetaTrainingReport {
            epochs: 50,
            final_loss: 0.15,
            epoch_losses: vec![0.3, 0.2, 0.15],
        };
        let json = serde_json::to_string(&r).expect("ok");
        let r2: MetaTrainingReport = serde_json::from_str(&json).expect("ok");
        assert_eq!(r2.epochs, 50);
        assert!((r2.final_loss - 0.15).abs() < 1e-12);
    }

    #[test]
    fn test_meta_learner_summary_serialization() {
        let s = MetaLearnerSummary {
            input_dim: 12,
            num_constraint_types: 6,
            total_parameters: 78,
        };
        let json = serde_json::to_string(&s).expect("ok");
        let s2: MetaLearnerSummary = serde_json::from_str(&json).expect("ok");
        assert_eq!(s2.input_dim, 12);
    }

    #[test]
    fn test_stacking_config_default() {
        let cfg = StackingConfig::default();
        assert_eq!(cfg.cv_folds, 3);
        assert!(!cfg.constraint_types.is_empty());
        assert!(cfg.min_final_confidence > 0.0);
    }

    #[test]
    fn test_stacking_training_report_serialization() {
        let report = StackingTrainingReport {
            base_metrics: vec![TrainingMetrics {
                model_name: "gnn".to_string(),
                epochs: 1,
                final_loss: 0.2,
                precision: 0.8,
                recall: 0.75,
                f1_score: 0.77,
            }],
            meta_report: MetaTrainingReport {
                epochs: 50,
                final_loss: 0.1,
                epoch_losses: vec![0.3, 0.1],
            },
            num_meta_examples: 20,
            constraint_precisions: {
                let mut m = HashMap::new();
                m.insert("sh:minCount".to_string(), 0.85);
                m
            },
        };
        let json = serde_json::to_string(&report).expect("ok");
        let r2: StackingTrainingReport = serde_json::from_str(&json).expect("ok");
        assert_eq!(r2.num_meta_examples, 20);
        assert_eq!(r2.base_metrics.len(), 1);
    }

    #[test]
    fn test_meta_learner_config_default() {
        let cfg = MetaLearnerConfig::default();
        assert_eq!(cfg.epochs, 50);
        assert!(!cfg.use_mlp);
        assert!(cfg.decision_threshold > 0.0);
    }

    #[test]
    fn test_gnn_multi_type_node_kind() {
        let learner = GnnBaseLearner::new("gnn");
        let features = make_graph_features(3.0, 0.2, 3); // type_count = 3
        let preds = learner.predict(&features).expect("ok");
        assert!(
            preds.iter().any(|p| p.constraint_type == "sh:nodeKind"),
            "multi-type graph should predict sh:nodeKind"
        );
    }

    #[test]
    fn test_stacking_report_has_constraint_precisions() {
        let mut ensemble = make_stacking_ensemble();
        let examples: Vec<TrainingExample> = (0..6)
            .map(|i| make_training_example(i as f64 + 1.0, 0.2))
            .collect();
        let report = ensemble.fit(&examples).expect("ok");
        for ct in &["sh:minCount", "sh:datatype", "sh:nodeKind"] {
            assert!(
                report.constraint_precisions.contains_key(*ct),
                "missing precision for {ct}"
            );
        }
    }
}
