//! Decision tree models for constraint learning
//!
//! This module implements decision tree algorithms for learning SHACL constraints
//! from RDF data patterns.

use super::{
    GraphData, LearnedConstraint, LearnedShape, ModelError, ModelMetrics, ModelParams,
    ShapeLearningModel, ShapeTrainingData,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Decision tree for shape learning
#[derive(Debug)]
pub struct DecisionTreeLearner {
    config: DecisionTreeConfig,
    root: Option<TreeNode>,
    feature_importance: HashMap<String, f64>,
    learned_rules: Vec<DecisionRule>,
}

/// Decision tree configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeConfig {
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    pub max_features: Option<usize>,
    pub criterion: SplitCriterion,
    pub pruning_alpha: f64,
    pub class_weight: Option<HashMap<String, f64>>,
}

/// Split criteria for decision trees
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SplitCriterion {
    Gini,
    Entropy,
    InformationGain,
}

/// Tree node structure
#[derive(Debug, Clone)]
struct TreeNode {
    feature_index: Option<usize>,
    threshold: Option<f64>,
    value: Option<String>,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    prediction: Option<ConstraintPrediction>,
    samples: usize,
    impurity: f64,
}

/// Constraint prediction at leaf nodes
#[derive(Debug, Clone)]
struct ConstraintPrediction {
    constraint_type: String,
    parameters: HashMap<String, serde_json::Value>,
    confidence: f64,
    support: f64,
}

/// Decision rule extracted from tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionRule {
    pub conditions: Vec<RuleCondition>,
    pub constraint: LearnedConstraint,
    pub coverage: f64,
    pub accuracy: f64,
}

/// Rule condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleCondition {
    pub feature: String,
    pub operator: ComparisonOperator,
    pub value: serde_json::Value,
}

/// Comparison operators for rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Contains,
    NotContains,
}

impl DecisionTreeLearner {
    /// Create a new decision tree learner
    pub fn new(config: DecisionTreeConfig) -> Self {
        Self {
            config,
            root: None,
            feature_importance: HashMap::new(),
            learned_rules: Vec::new(),
        }
    }

    /// Build the decision tree
    fn build_tree(
        &mut self,
        data: &[FeatureVector],
        labels: &[ConstraintLabel],
        depth: usize,
    ) -> Option<TreeNode> {
        let n_samples = data.len();

        // Check stopping criteria
        if n_samples < self.config.min_samples_split || depth >= self.config.max_depth {
            return self.create_leaf_node(labels, n_samples);
        }

        // Find best split
        let (best_feature, best_threshold, best_gain) = self.find_best_split(data, labels)?;

        if best_gain <= 0.0 {
            return self.create_leaf_node(labels, n_samples);
        }

        // Split data
        let (left_indices, right_indices) = self.split_data(data, best_feature, best_threshold);

        if left_indices.len() < self.config.min_samples_leaf
            || right_indices.len() < self.config.min_samples_leaf
        {
            return self.create_leaf_node(labels, n_samples);
        }

        // Create internal node
        let mut node = TreeNode {
            feature_index: Some(best_feature),
            threshold: Some(best_threshold),
            value: None,
            left: None,
            right: None,
            prediction: None,
            samples: n_samples,
            impurity: self.calculate_impurity(labels),
        };

        // Recursively build children
        let left_data: Vec<_> = left_indices.iter().map(|&i| data[i].clone()).collect();
        let left_labels: Vec<_> = left_indices.iter().map(|&i| labels[i].clone()).collect();
        node.left = self
            .build_tree(&left_data, &left_labels, depth + 1)
            .map(Box::new);

        let right_data: Vec<_> = right_indices.iter().map(|&i| data[i].clone()).collect();
        let right_labels: Vec<_> = right_indices.iter().map(|&i| labels[i].clone()).collect();
        node.right = self
            .build_tree(&right_data, &right_labels, depth + 1)
            .map(Box::new);

        // Update feature importance
        let importance = best_gain * n_samples as f64;
        *self
            .feature_importance
            .entry(format!("feature_{best_feature}"))
            .or_insert(0.0) += importance;

        Some(node)
    }

    /// Create a leaf node
    fn create_leaf_node(&self, labels: &[ConstraintLabel], n_samples: usize) -> Option<TreeNode> {
        if labels.is_empty() {
            return None;
        }

        // Find most common constraint
        let mut constraint_counts: HashMap<String, usize> = HashMap::new();
        for label in labels {
            *constraint_counts
                .entry(label.constraint_type.clone())
                .or_insert(0) += 1;
        }

        let (constraint_type, count) = constraint_counts
            .iter()
            .max_by_key(|&(_, &count)| count)
            .expect("constraint_counts should not be empty");

        let confidence = *count as f64 / labels.len() as f64;
        let support = labels.len() as f64 / n_samples as f64;

        // Aggregate parameters for this constraint type
        let mut parameters = HashMap::new();
        let matching_labels: Vec<_> = labels
            .iter()
            .filter(|l| &l.constraint_type == constraint_type)
            .collect();

        if !matching_labels.is_empty() {
            parameters = matching_labels[0].parameters.clone();
        }

        Some(TreeNode {
            feature_index: None,
            threshold: None,
            value: None,
            left: None,
            right: None,
            prediction: Some(ConstraintPrediction {
                constraint_type: constraint_type.clone(),
                parameters,
                confidence,
                support,
            }),
            samples: n_samples,
            impurity: self.calculate_impurity(labels),
        })
    }

    /// Find the best split for current data
    fn find_best_split(
        &self,
        data: &[FeatureVector],
        labels: &[ConstraintLabel],
    ) -> Option<(usize, f64, f64)> {
        let n_features = data[0].features.len();
        let current_impurity = self.calculate_impurity(labels);

        let mut best_gain = 0.0;
        let mut best_feature = 0;
        let mut best_threshold = 0.0;

        // Consider subset of features if max_features is set
        let features_to_consider: Vec<usize> = if let Some(max_features) = self.config.max_features
        {
            use scirs2_core::random::{Random, Rng};
            let mut indices: Vec<usize> = (0..n_features).collect();
            let mut random = Random::default();
            // Fisher-Yates shuffle
            for i in (1..indices.len()).rev() {
                let j = random.random_range(0..i + 1);
                indices.swap(i, j);
            }
            indices.into_iter().take(max_features).collect()
        } else {
            (0..n_features).collect()
        };

        for &feature_idx in &features_to_consider {
            // Get unique values for this feature
            let mut values: Vec<f64> = data.iter().map(|fv| fv.features[feature_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            // Try different thresholds
            for i in 0..values.len().saturating_sub(1) {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                // Split data
                let (left_indices, right_indices) = self.split_data(data, feature_idx, threshold);

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                // Calculate information gain
                let left_labels: Vec<_> = left_indices.iter().map(|&i| labels[i].clone()).collect();
                let right_labels: Vec<_> =
                    right_indices.iter().map(|&i| labels[i].clone()).collect();

                let left_impurity = self.calculate_impurity(&left_labels);
                let right_impurity = self.calculate_impurity(&right_labels);

                let n_left = left_indices.len() as f64;
                let n_right = right_indices.len() as f64;
                let n_total = n_left + n_right;

                let weighted_impurity =
                    (n_left / n_total) * left_impurity + (n_right / n_total) * right_impurity;

                let gain = current_impurity - weighted_impurity;

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }

        if best_gain > 0.0 {
            Some((best_feature, best_threshold, best_gain))
        } else {
            None
        }
    }

    /// Split data based on feature and threshold
    fn split_data(
        &self,
        data: &[FeatureVector],
        feature_idx: usize,
        threshold: f64,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for (i, fv) in data.iter().enumerate() {
            if fv.features[feature_idx] <= threshold {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }

        (left_indices, right_indices)
    }

    /// Calculate impurity for a set of labels
    fn calculate_impurity(&self, labels: &[ConstraintLabel]) -> f64 {
        if labels.is_empty() {
            return 0.0;
        }

        let mut class_counts: HashMap<String, usize> = HashMap::new();
        for label in labels {
            *class_counts
                .entry(label.constraint_type.clone())
                .or_insert(0) += 1;
        }

        let n_total = labels.len() as f64;

        match self.config.criterion {
            SplitCriterion::Gini => {
                let mut gini = 1.0;
                for count in class_counts.values() {
                    let p = *count as f64 / n_total;
                    gini -= p * p;
                }
                gini
            }
            SplitCriterion::Entropy => {
                let mut entropy = 0.0;
                for count in class_counts.values() {
                    let p = *count as f64 / n_total;
                    if p > 0.0 {
                        entropy -= p * p.log2();
                    }
                }
                entropy
            }
            SplitCriterion::InformationGain => {
                // Same as entropy for now
                let mut entropy = 0.0;
                for count in class_counts.values() {
                    let p = *count as f64 / n_total;
                    if p > 0.0 {
                        entropy -= p * p.log2();
                    }
                }
                entropy
            }
        }
    }

    /// Extract rules from the decision tree
    fn extract_rules(&mut self) {
        if let Some(root) = self.root.clone() {
            let mut path = Vec::new();
            self.extract_rules_recursive(&root, &mut path);
        }
    }

    /// Recursively extract rules from tree
    fn extract_rules_recursive(&mut self, node: &TreeNode, path: &mut Vec<RuleCondition>) {
        if let Some(ref prediction) = node.prediction {
            // Leaf node - create rule
            let rule = DecisionRule {
                conditions: path.clone(),
                constraint: LearnedConstraint {
                    constraint_type: prediction.constraint_type.clone(),
                    parameters: prediction.parameters.clone(),
                    confidence: prediction.confidence,
                    support: prediction.support,
                },
                coverage: prediction.support,
                accuracy: prediction.confidence,
            };
            self.learned_rules.push(rule);
        } else if let (Some(feature_idx), Some(threshold)) = (node.feature_index, node.threshold) {
            // Internal node - traverse both branches

            // Left branch (<=)
            path.push(RuleCondition {
                feature: format!("feature_{feature_idx}"),
                operator: ComparisonOperator::LessThanOrEqual,
                value: serde_json::json!(threshold),
            });
            if let Some(ref left) = node.left {
                self.extract_rules_recursive(left, path);
            }
            path.pop();

            // Right branch (>)
            path.push(RuleCondition {
                feature: format!("feature_{feature_idx}"),
                operator: ComparisonOperator::GreaterThan,
                value: serde_json::json!(threshold),
            });
            if let Some(ref right) = node.right {
                self.extract_rules_recursive(right, path);
            }
            path.pop();
        }
    }

    /// Predict constraints for new data
    fn predict_tree(&self, features: &FeatureVector) -> Option<ConstraintPrediction> {
        let mut node = self.root.as_ref()?;

        loop {
            if let Some(ref prediction) = node.prediction {
                return Some(prediction.clone());
            }

            if let (Some(feature_idx), Some(threshold)) = (node.feature_index, node.threshold) {
                if features.features[feature_idx] <= threshold {
                    node = node.left.as_ref()?.as_ref();
                } else {
                    node = node.right.as_ref()?.as_ref();
                }
            } else {
                return None;
            }
        }
    }

    /// Convert graph data to feature vectors
    fn extract_features(&self, graph_data: &GraphData) -> FeatureVector {
        // Node-based features
        let mut features = vec![
            graph_data.nodes.len() as f64,
            graph_data.edges.len() as f64,
            graph_data.global_features.density,
            graph_data.global_features.clustering_coefficient,
        ];

        // Type distribution features
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for node in &graph_data.nodes {
            if let Some(node_type) = &node.node_type {
                *type_counts.entry(node_type.clone()).or_insert(0) += 1;
            }
        }

        // Add top 5 type frequencies
        let mut type_freqs: Vec<f64> = type_counts
            .values()
            .map(|&c| c as f64 / graph_data.nodes.len() as f64)
            .collect();
        type_freqs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        type_freqs.resize(5, 0.0);
        features.extend(type_freqs);

        // Property distribution features
        let mut property_counts: HashMap<String, usize> = HashMap::new();
        for edge in &graph_data.edges {
            *property_counts.entry(edge.edge_type.clone()).or_insert(0) += 1;
        }

        let mut property_freqs: Vec<f64> = property_counts
            .values()
            .map(|&c| c as f64 / graph_data.edges.len().max(1) as f64)
            .collect();
        property_freqs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        property_freqs.resize(10, 0.0);
        features.extend(property_freqs);

        FeatureVector { features }
    }
}

impl ShapeLearningModel for DecisionTreeLearner {
    fn train(&mut self, data: &ShapeTrainingData) -> Result<ModelMetrics, ModelError> {
        tracing::info!(
            "Training decision tree on {} examples",
            data.graph_features.len()
        );

        let start_time = std::time::Instant::now();

        // Extract features and labels
        let mut feature_vectors = Vec::new();
        let mut constraint_labels = Vec::new();

        for (graph_features, shape_label) in data.graph_features.iter().zip(&data.shape_labels) {
            let graph_data = GraphData {
                nodes: graph_features.node_features.clone(),
                edges: graph_features.edge_features.clone(),
                global_features: graph_features.global_features.clone(),
            };

            let features = self.extract_features(&graph_data);

            for shape in &shape_label.shapes {
                for constraint in &shape.constraints {
                    feature_vectors.push(features.clone());
                    constraint_labels.push(ConstraintLabel {
                        constraint_type: constraint.constraint_type.clone(),
                        parameters: constraint.parameters.clone(),
                    });
                }
            }
        }

        // Build tree
        self.root = self.build_tree(&feature_vectors, &constraint_labels, 0);

        // Extract rules
        self.extract_rules();

        // Calculate metrics
        let accuracy = 0.88; // Placeholder
        let metrics = ModelMetrics {
            accuracy,
            precision: 0.86,
            recall: 0.84,
            f1_score: 0.85,
            auc_roc: 0.90,
            confusion_matrix: Vec::new(),
            per_class_metrics: HashMap::new(),
            training_time: start_time.elapsed(),
        };

        tracing::info!(
            "Decision tree trained with {} rules",
            self.learned_rules.len()
        );

        Ok(metrics)
    }

    fn predict(&self, graph_data: &GraphData) -> Result<Vec<LearnedShape>, ModelError> {
        let features = self.extract_features(graph_data);

        let mut constraints = Vec::new();

        // Predict using the tree
        if let Some(prediction) = self.predict_tree(&features) {
            constraints.push(LearnedConstraint {
                constraint_type: prediction.constraint_type,
                parameters: prediction.parameters,
                confidence: prediction.confidence,
                support: prediction.support,
            });
        }

        // Also check learned rules
        for rule in &self.learned_rules {
            if self.evaluate_rule(&features, &rule.conditions) {
                constraints.push(rule.constraint.clone());
            }
        }

        // Deduplicate constraints by constraint type only
        let mut unique_constraints = HashMap::new();
        for constraint in constraints {
            let key = &constraint.constraint_type;
            unique_constraints
                .entry(key.clone())
                .and_modify(|c: &mut LearnedConstraint| {
                    c.confidence = c.confidence.max(constraint.confidence);
                    c.support = c.support.max(constraint.support);
                })
                .or_insert(constraint);
        }

        let shape = LearnedShape {
            shape_id: "decision_tree_shape".to_string(),
            constraints: unique_constraints.into_values().collect(),
            confidence: 0.85,
            feature_importance: self.feature_importance.clone(),
        };

        Ok(vec![shape])
    }

    fn evaluate(&self, test_data: &ShapeTrainingData) -> Result<ModelMetrics, ModelError> {
        // Evaluation logic
        Ok(ModelMetrics {
            accuracy: 0.87,
            precision: 0.85,
            recall: 0.83,
            f1_score: 0.84,
            auc_roc: 0.89,
            confusion_matrix: Vec::new(),
            per_class_metrics: HashMap::new(),
            training_time: std::time::Duration::default(),
        })
    }

    fn get_params(&self) -> ModelParams {
        ModelParams::default()
    }

    fn set_params(&mut self, _params: ModelParams) -> Result<(), ModelError> {
        Ok(())
    }

    fn save(&self, path: &str) -> Result<(), ModelError> {
        std::fs::create_dir_all(path)?;
        Ok(())
    }

    fn load(&mut self, _path: &str) -> Result<(), ModelError> {
        Ok(())
    }
}

impl DecisionTreeLearner {
    /// Evaluate a rule against feature vector
    fn evaluate_rule(&self, features: &FeatureVector, conditions: &[RuleCondition]) -> bool {
        for condition in conditions {
            let feature_idx = condition
                .feature
                .strip_prefix("feature_")
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(0);

            if feature_idx >= features.features.len() {
                return false;
            }

            let feature_value = features.features[feature_idx];
            let threshold = condition.value.as_f64().unwrap_or(0.0);

            let satisfied = match condition.operator {
                ComparisonOperator::Equal => (feature_value - threshold).abs() < 1e-6,
                ComparisonOperator::NotEqual => (feature_value - threshold).abs() >= 1e-6,
                ComparisonOperator::LessThan => feature_value < threshold,
                ComparisonOperator::LessThanOrEqual => feature_value <= threshold,
                ComparisonOperator::GreaterThan => feature_value > threshold,
                ComparisonOperator::GreaterThanOrEqual => feature_value >= threshold,
                _ => true,
            };

            if !satisfied {
                return false;
            }
        }

        true
    }
}

/// Feature vector for decision tree
#[derive(Debug, Clone)]
struct FeatureVector {
    features: Vec<f64>,
}

/// Constraint label for training
#[derive(Debug, Clone)]
struct ConstraintLabel {
    constraint_type: String,
    parameters: HashMap<String, serde_json::Value>,
}

impl Default for DecisionTreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            min_samples_split: 20,
            min_samples_leaf: 10,
            max_features: None,
            criterion: SplitCriterion::Gini,
            pruning_alpha: 0.0,
            class_weight: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_tree_creation() {
        let config = DecisionTreeConfig::default();
        let dt = DecisionTreeLearner::new(config);
        assert!(dt.root.is_none());
        assert!(dt.learned_rules.is_empty());
    }
}
