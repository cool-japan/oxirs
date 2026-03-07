//! Rule-based shape learner
//!
//! Mines frequent (predicate, constraint) association rules from training
//! data.  Fast, fully interpretable and suitable as a baseline in ensembles.

use std::collections::HashMap;

use crate::ensemble::{
    GraphFeatures, ShapeLearner, ShapePrediction, TrainingExample, TrainingMetrics,
};
use crate::ShaclAiError;

// -------------------------------------------------------------------------
// Types
// -------------------------------------------------------------------------

/// A single mined association rule
#[derive(Debug, Clone)]
pub struct LearnedRule {
    /// The RDF predicate this rule was mined from
    pub predicate: String,
    /// SHACL constraint type derived from the rule (e.g. "sh:minCount")
    pub constraint_type: String,
    /// Shape identifier for the predicted shape
    pub shape_id: String,
    /// Support: fraction of training examples containing this rule
    pub support: f64,
    /// Confidence: fraction of times this rule led to a correct prediction
    pub confidence: f64,
    /// Number of training examples used to derive this rule
    pub examples_seen: usize,
}

// -------------------------------------------------------------------------
// Learner
// -------------------------------------------------------------------------

/// Rule-based shape learner that mines frequent predicate patterns
///
/// The learner builds rules of the form:
///   `(predicate, constraint_type) → shape_id`
///
/// Each rule has a *support* (how often the predicate occurs in training
/// data) and a *confidence* (fraction of occurrences that match the expected
/// constraint in the ground-truth shapes).
pub struct RuleBasedShapeLearner {
    /// Minimum support threshold for a rule to be retained
    min_support: f64,
    /// Minimum confidence threshold for a rule to be retained
    min_confidence: f64,
    /// Mined rules (populated by `train`)
    rules: Vec<LearnedRule>,
}

impl RuleBasedShapeLearner {
    /// Create a new learner with explicit thresholds
    pub fn new(min_support: f64, min_confidence: f64) -> Self {
        Self {
            min_support,
            min_confidence,
            rules: Vec::new(),
        }
    }

    /// Inspect the currently mined rules
    pub fn rules(&self) -> &[LearnedRule] {
        &self.rules
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Mine association rules from labelled training examples.
    ///
    /// Strategy
    /// --------
    /// 1. For every training example, collect (predicate → set of shape/constraint pairs)
    ///    from the edge features and the ground-truth shapes.
    /// 2. Count occurrences across all examples.
    /// 3. Derive support and confidence; keep only rules above the thresholds.
    fn mine_rules(&mut self, examples: &[TrainingExample]) {
        if examples.is_empty() {
            return;
        }

        let n = examples.len() as f64;

        // predicate -> {(shape_id, constraint_type) -> (seen_in_example, confirmed)}
        // "confirmed" means both the predicate appeared AND the shape/constraint appeared in truth
        type RuleKey = (String, String);
        type RuleStats = (usize, usize); // (total_examples_with_predicate, confirmed_matches)

        let mut predicate_count: HashMap<String, usize> = HashMap::new();
        let mut rule_stats: HashMap<(String, RuleKey), RuleStats> = HashMap::new();

        for example in examples {
            // Collect distinct predicates in this example
            let predicates: Vec<String> = example
                .features
                .edge_features
                .iter()
                .map(|e| e.predicate.clone())
                .collect();

            // Collect distinct (shape_id, constraint_type) pairs from ground truth
            let ground_truth: Vec<(String, String)> = example
                .shapes
                .iter()
                .map(|s| (s.shape_id.clone(), s.constraint_type.clone()))
                .collect();

            let seen_predicates: std::collections::HashSet<String> =
                predicates.into_iter().collect();

            for pred in &seen_predicates {
                *predicate_count.entry(pred.clone()).or_insert(0) += 1;

                // For each ground-truth rule key, check if we have a hit
                for (shape_id, constraint_type) in &ground_truth {
                    let outer_key = (pred.clone(), (shape_id.clone(), constraint_type.clone()));
                    let entry = rule_stats.entry(outer_key).or_insert((0, 0));
                    entry.0 += 1; // predicate appeared in this example
                    entry.1 += 1; // and the constraint was also in ground truth
                }
            }
        }

        // Build rules from statistics
        let mut new_rules: Vec<LearnedRule> = Vec::new();
        for ((predicate, (shape_id, constraint_type)), (pred_examples, confirmed)) in &rule_stats {
            let total_pred = predicate_count.get(predicate).copied().unwrap_or(1) as f64;
            let support = total_pred / n;
            let confidence = *confirmed as f64 / (*pred_examples).max(1) as f64;

            if support >= self.min_support && confidence >= self.min_confidence {
                new_rules.push(LearnedRule {
                    predicate: predicate.clone(),
                    constraint_type: constraint_type.clone(),
                    shape_id: shape_id.clone(),
                    support,
                    confidence,
                    examples_seen: *pred_examples,
                });
            }
        }

        self.rules = new_rules;
    }

    /// Compute precision, recall, and F1 on a validation set
    ///
    /// Returns `(precision, recall, f1)`.
    fn evaluate(&self, examples: &[TrainingExample]) -> (f64, f64, f64) {
        if examples.is_empty() || self.rules.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let mut true_positives: usize = 0;
        let mut false_positives: usize = 0;
        let mut false_negatives: usize = 0;

        for example in examples {
            let predicted = self.predict_raw(&example.features);
            let ground_truth: std::collections::HashSet<(String, String)> = example
                .shapes
                .iter()
                .map(|s| (s.shape_id.clone(), s.constraint_type.clone()))
                .collect();

            let predicted_keys: std::collections::HashSet<(String, String)> = predicted
                .iter()
                .map(|p| (p.shape_id.clone(), p.constraint_type.clone()))
                .collect();

            for pk in &predicted_keys {
                if ground_truth.contains(pk) {
                    true_positives += 1;
                } else {
                    false_positives += 1;
                }
            }
            for gt in &ground_truth {
                if !predicted_keys.contains(gt) {
                    false_negatives += 1;
                }
            }
        }

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

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        (precision, recall, f1)
    }

    /// Internal prediction that skips the confidence filter
    fn predict_raw(&self, features: &GraphFeatures) -> Vec<ShapePrediction> {
        let predicates: std::collections::HashSet<String> = features
            .edge_features
            .iter()
            .map(|e| e.predicate.clone())
            .collect();

        self.rules
            .iter()
            .filter(|r| predicates.contains(&r.predicate))
            .map(|r| ShapePrediction {
                shape_id: r.shape_id.clone(),
                constraint_type: r.constraint_type.clone(),
                confidence: r.confidence,
                supporting_triples: Vec::new(),
            })
            .collect()
    }
}

// -------------------------------------------------------------------------
// ShapeLearner trait implementation
// -------------------------------------------------------------------------

impl ShapeLearner for RuleBasedShapeLearner {
    fn name(&self) -> &str {
        "rule_based"
    }

    fn predict(&self, features: &GraphFeatures) -> Result<Vec<ShapePrediction>, ShaclAiError> {
        Ok(self.predict_raw(features))
    }

    fn train(&mut self, examples: &[TrainingExample]) -> Result<TrainingMetrics, ShaclAiError> {
        self.mine_rules(examples);

        let (precision, recall, f1) = self.evaluate(examples);

        // Approximate loss as 1 - F1
        let final_loss = 1.0 - f1;

        Ok(TrainingMetrics {
            model_name: self.name().to_string(),
            epochs: 1, // Rule mining is a single-pass algorithm
            final_loss,
            precision,
            recall,
            f1_score: f1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ensemble::{EdgeFeature, GraphFeatures, GraphStats, NodeFeature, TrainingExample};

    fn make_features(predicates: &[&str]) -> GraphFeatures {
        let edge_features: Vec<EdgeFeature> = predicates
            .iter()
            .map(|p| EdgeFeature {
                predicate: p.to_string(),
                source_type: "Subject".to_string(),
                target_type: "Object".to_string(),
                frequency: 1.0,
            })
            .collect();

        GraphFeatures {
            node_features: vec![NodeFeature {
                node_id: "n1".to_string(),
                type_vec: vec![1.0],
                predicate_histogram: vec![1.0; predicates.len()],
                degree_in: 0,
                degree_out: predicates.len(),
            }],
            edge_features,
            graph_stats: GraphStats {
                node_count: 1,
                edge_count: predicates.len(),
                type_count: 1,
                predicate_count: predicates.len(),
                avg_degree: predicates.len() as f64,
                density: 1.0,
            },
        }
    }

    fn make_example(predicates: &[&str], shape_id: &str, constraint: &str) -> TrainingExample {
        TrainingExample {
            features: make_features(predicates),
            shapes: vec![ShapePrediction {
                shape_id: shape_id.to_string(),
                constraint_type: constraint.to_string(),
                confidence: 1.0,
                supporting_triples: Vec::new(),
            }],
        }
    }

    #[test]
    fn test_rule_learner_mines_rules() {
        let mut learner = RuleBasedShapeLearner::new(0.0, 0.0);

        let examples = vec![
            make_example(&["foaf:name"], "PersonShape", "sh:minCount"),
            make_example(&["foaf:name"], "PersonShape", "sh:minCount"),
            make_example(&["foaf:name"], "PersonShape", "sh:minCount"),
        ];

        let metrics = learner.train(&examples).expect("train failed");
        assert!(
            !learner.rules().is_empty(),
            "Should have mined at least one rule"
        );
        assert_eq!(metrics.model_name, "rule_based");
        assert_eq!(metrics.epochs, 1);
    }

    #[test]
    fn test_rule_learner_predict() {
        let mut learner = RuleBasedShapeLearner::new(0.0, 0.0);

        let examples = vec![
            make_example(&["foaf:name"], "PersonShape", "sh:minCount"),
            make_example(&["foaf:name"], "PersonShape", "sh:minCount"),
        ];
        learner.train(&examples).expect("train failed");

        let features = make_features(&["foaf:name"]);
        let preds = learner.predict(&features).expect("predict failed");

        assert!(!preds.is_empty(), "Should predict at least one shape");
        assert!(preds.iter().any(|p| p.shape_id == "PersonShape"));
    }

    #[test]
    fn test_rule_learner_min_support_threshold() {
        // Set high min_support so rare predicates are excluded
        let mut learner = RuleBasedShapeLearner::new(0.9, 0.0);

        let examples = vec![
            make_example(&["common:pred"], "ShapeA", "sh:minCount"),
            make_example(&["common:pred"], "ShapeA", "sh:minCount"),
            make_example(&["common:pred"], "ShapeA", "sh:minCount"),
            make_example(&["rare:pred"], "ShapeB", "sh:datatype"), // appears only once
        ];
        learner.train(&examples).expect("train failed");

        let features = make_features(&["rare:pred"]);
        let preds = learner.predict(&features).expect("predict failed");
        // ShapeB should not appear because rare:pred has low support
        assert!(
            preds.iter().all(|p| p.shape_id != "ShapeB"),
            "Low-support rule should be filtered out"
        );
    }

    #[test]
    fn test_rule_learner_empty_training() {
        let mut learner = RuleBasedShapeLearner::new(0.1, 0.5);
        let metrics = learner.train(&[]).expect("train on empty should succeed");
        assert!(learner.rules().is_empty());
        assert_eq!(metrics.epochs, 1);
    }

    #[test]
    fn test_rule_learner_f1_reasonable() {
        let mut learner = RuleBasedShapeLearner::new(0.0, 0.0);

        // Clear signal: foaf:name always means PersonShape/sh:minCount
        let examples: Vec<TrainingExample> = (0..10)
            .map(|_| make_example(&["foaf:name"], "PersonShape", "sh:minCount"))
            .collect();

        let metrics = learner.train(&examples).expect("train failed");
        assert!(
            metrics.f1_score > 0.0,
            "F1 should be > 0 when rules clearly match training data"
        );
    }

    #[test]
    fn test_rule_learner_no_match_for_unseen_predicate() {
        let mut learner = RuleBasedShapeLearner::new(0.0, 0.0);
        let examples = vec![make_example(&["foaf:name"], "PersonShape", "sh:minCount")];
        learner.train(&examples).expect("train failed");

        // Query with a predicate that never appeared in training
        let features = make_features(&["schema:email"]);
        let preds = learner.predict(&features).expect("predict failed");
        assert!(preds.is_empty(), "No rules should match unseen predicates");
    }
}
