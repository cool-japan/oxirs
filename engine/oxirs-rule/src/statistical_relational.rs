//! # Statistical Relational Learning Module
//!
//! This module provides Statistical Relational Learning (SRL) capabilities,
//! combining statistical machine learning with relational logic programming.
//!
//! ## Features
//!
//! - **Relational Feature Extraction**: Extracting features from relational data
//! - **Structure Learning**: Learning rule structures from data
//! - **Parameter Learning**: Learning weights using maximum likelihood
//! - **Probabilistic Inference**: Combining statistical and logical reasoning
//! - **Relational Classification**: Classifying entities based on relationships
//! - **Link Prediction**: Predicting missing relationships
//! - **Collective Classification**: Joint inference over related entities
//!
//! ## Example
//!
//! ```rust
//! use oxirs_rule::statistical_relational::*;
//! use oxirs_rule::{RuleAtom, Term};
//!
//! // Create an SRL learner
//! let mut srl = StatisticalRelationalLearner::new();
//!
//! // Add training examples
//! let facts = vec![
//!     RuleAtom::Triple {
//!         subject: Term::Constant("alice".to_string()),
//!         predicate: Term::Constant("friend".to_string()),
//!         object: Term::Constant("bob".to_string()),
//!     },
//!     RuleAtom::Triple {
//!         subject: Term::Constant("bob".to_string()),
//!         predicate: Term::Constant("likes".to_string()),
//!         object: Term::Constant("sports".to_string()),
//!     },
//! ];
//!
//! // Learn relational patterns
//! srl.fit(&facts).unwrap();
//!
//! // Extract features for prediction
//! let features = srl.extract_features(&facts).unwrap();
//! println!("Extracted {} relational features", features.len());
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

/// Relational feature extracted from data
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RelationalFeature {
    /// Feature type (e.g., "path", "neighborhood", "aggregate")
    pub feature_type: String,
    /// Feature pattern (e.g., predicate sequence)
    pub pattern: Vec<String>,
    /// Feature value or count
    pub value: String,
}

impl RelationalFeature {
    /// Create a new relational feature
    pub fn new(feature_type: String, pattern: Vec<String>, value: String) -> Self {
        Self {
            feature_type,
            pattern,
            value,
        }
    }

    /// Create a path feature (A -pred1-> B -pred2-> C)
    pub fn path_feature(predicates: Vec<String>) -> Self {
        Self {
            feature_type: "path".to_string(),
            pattern: predicates,
            value: "exists".to_string(),
        }
    }

    /// Create a neighborhood feature (entity has N neighbors via predicate)
    pub fn neighborhood_feature(predicate: String, count: usize) -> Self {
        Self {
            feature_type: "neighborhood".to_string(),
            pattern: vec![predicate],
            value: count.to_string(),
        }
    }
}

/// Statistical relational model combining rules with learned weights
#[derive(Debug, Clone)]
pub struct StatisticalRelationalModel {
    /// Rules with learned weights
    pub weighted_rules: Vec<(Rule, f64)>,
    /// Feature weights for classification
    pub feature_weights: HashMap<RelationalFeature, f64>,
    /// Class prior probabilities
    pub class_priors: HashMap<String, f64>,
}

impl StatisticalRelationalModel {
    /// Create a new SRL model
    pub fn new() -> Self {
        Self {
            weighted_rules: Vec::new(),
            feature_weights: HashMap::new(),
            class_priors: HashMap::new(),
        }
    }

    /// Add a weighted rule
    pub fn add_weighted_rule(&mut self, rule: Rule, weight: f64) {
        self.weighted_rules.push((rule, weight));
    }

    /// Predict class probability using feature weights
    pub fn predict_class_probability(&self, features: &[RelationalFeature], class: &str) -> f64 {
        let mut score = self.class_priors.get(class).copied().unwrap_or(0.5);

        for feature in features {
            if let Some(&weight) = self.feature_weights.get(feature) {
                score += weight;
            }
        }

        // Apply sigmoid to get probability
        1.0 / (1.0 + (-score).exp())
    }

    /// Get most likely class
    pub fn predict_class(&self, features: &[RelationalFeature]) -> Option<String> {
        let classes: Vec<_> = self.class_priors.keys().cloned().collect();

        classes
            .into_iter()
            .map(|class| {
                let prob = self.predict_class_probability(features, &class);
                (class, prob)
            })
            .max_by(|(_, p1), (_, p2)| p1.partial_cmp(p2).unwrap())
            .map(|(class, _)| class)
    }
}

impl Default for StatisticalRelationalModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistical relational learner
#[derive(Debug)]
pub struct StatisticalRelationalLearner {
    /// Learned model
    model: StatisticalRelationalModel,
    /// Training data
    training_data: Vec<RuleAtom>,
    /// Entity types discovered from data
    entity_types: HashMap<String, HashSet<String>>,
    /// Predicate statistics
    predicate_counts: HashMap<String, usize>,
}

impl StatisticalRelationalLearner {
    /// Create a new SRL learner
    pub fn new() -> Self {
        Self {
            model: StatisticalRelationalModel::new(),
            training_data: Vec::new(),
            entity_types: HashMap::new(),
            predicate_counts: HashMap::new(),
        }
    }

    /// Fit the model to training data
    pub fn fit(&mut self, data: &[RuleAtom]) -> Result<()> {
        self.training_data = data.to_vec();

        // Discover entity types and predicate statistics
        self.discover_schema(data);

        // Learn structure (frequent patterns)
        self.learn_structure(data)?;

        // Learn parameters (weights)
        self.learn_parameters(data)?;

        Ok(())
    }

    /// Discover schema from data
    fn discover_schema(&mut self, data: &[RuleAtom]) {
        for atom in data {
            if let RuleAtom::Triple {
                subject,
                predicate,
                object,
            } = atom
            {
                // Extract predicate name
                let pred_name = match predicate {
                    Term::Constant(p) => p.clone(),
                    _ => continue,
                };

                // Count predicate occurrences
                *self.predicate_counts.entry(pred_name.clone()).or_insert(0) += 1;

                // Track entities for each predicate
                if let Term::Constant(subj) = subject {
                    self.entity_types
                        .entry(pred_name.clone())
                        .or_default()
                        .insert(subj.clone());
                }

                if let Term::Constant(obj) = object {
                    self.entity_types
                        .entry(pred_name)
                        .or_default()
                        .insert(obj.clone());
                }
            }
        }
    }

    /// Learn rule structure from data using frequent pattern mining
    fn learn_structure(&mut self, data: &[RuleAtom]) -> Result<()> {
        // Extract frequent predicate pairs (length-2 paths)
        let mut path_counts: HashMap<(String, String), usize> = HashMap::new();

        // Build adjacency map for path extraction
        let mut adjacency: HashMap<String, Vec<(String, String)>> = HashMap::new();

        for atom in data {
            if let RuleAtom::Triple {
                subject,
                predicate,
                object,
            } = atom
            {
                if let (Term::Constant(s), Term::Constant(p), Term::Constant(o)) =
                    (subject, predicate, object)
                {
                    adjacency
                        .entry(s.clone())
                        .or_default()
                        .push((p.clone(), o.clone()));
                }
            }
        }

        // Find length-2 paths
        for neighbors in adjacency.values() {
            for (pred1, obj1) in neighbors {
                if let Some(obj_neighbors) = adjacency.get(obj1) {
                    for (pred2, _) in obj_neighbors {
                        let path = (pred1.clone(), pred2.clone());
                        *path_counts.entry(path).or_insert(0) += 1;
                    }
                }
            }
        }

        // Create rules from frequent paths (support >= 1)
        const MIN_SUPPORT: usize = 1;
        for ((pred1, pred2), count) in path_counts {
            if count >= MIN_SUPPORT {
                // Create rule: pred1(X,Y) ∧ pred2(Y,Z) → related(X,Z)
                let rule = Rule {
                    name: format!("path_{}_{}", pred1, pred2),
                    body: vec![
                        RuleAtom::Triple {
                            subject: Term::Variable("X".to_string()),
                            predicate: Term::Constant(pred1),
                            object: Term::Variable("Y".to_string()),
                        },
                        RuleAtom::Triple {
                            subject: Term::Variable("Y".to_string()),
                            predicate: Term::Constant(pred2),
                            object: Term::Variable("Z".to_string()),
                        },
                    ],
                    head: vec![RuleAtom::Triple {
                        subject: Term::Variable("X".to_string()),
                        predicate: Term::Constant("related".to_string()),
                        object: Term::Variable("Z".to_string()),
                    }],
                };

                // Weight based on frequency (normalized)
                let weight = (count as f64).ln() + 1.0;
                self.model.add_weighted_rule(rule, weight);
            }
        }

        Ok(())
    }

    /// Learn parameters using maximum likelihood estimation
    fn learn_parameters(&mut self, data: &[RuleAtom]) -> Result<()> {
        // Count class occurrences for priors
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        let mut total_count = 0;

        for atom in data {
            if let RuleAtom::Triple {
                subject: _,
                predicate,
                object,
            } = atom
            {
                if let (Term::Constant(pred), Term::Constant(obj)) = (predicate, object) {
                    if pred == "rdf:type" || pred.contains("type") {
                        *class_counts.entry(obj.clone()).or_insert(0) += 1;
                        total_count += 1;
                    }
                }
            }
        }

        // Compute class priors
        for (class, count) in class_counts {
            let prior = count as f64 / total_count.max(1) as f64;
            self.model.class_priors.insert(class, prior);
        }

        // Learn feature weights using simple frequency-based approach
        let features = self.extract_features(data)?;
        let feature_counts: HashMap<RelationalFeature, usize> =
            features.iter().fold(HashMap::new(), |mut acc, f| {
                *acc.entry(f.clone()).or_insert(0) += 1;
                acc
            });

        for (feature, count) in feature_counts {
            // Weight = log(count + 1) for smoothing
            let weight = (count as f64 + 1.0).ln();
            self.model.feature_weights.insert(feature, weight);
        }

        Ok(())
    }

    /// Extract relational features from data
    pub fn extract_features(&self, data: &[RuleAtom]) -> Result<Vec<RelationalFeature>> {
        let mut features = Vec::new();

        // Build entity neighborhood map
        let mut neighborhoods: HashMap<String, HashMap<String, usize>> = HashMap::new();

        for atom in data {
            if let RuleAtom::Triple {
                subject,
                predicate,
                object: _,
            } = atom
            {
                if let (Term::Constant(s), Term::Constant(p)) = (subject, predicate) {
                    *neighborhoods
                        .entry(s.clone())
                        .or_default()
                        .entry(p.clone())
                        .or_insert(0) += 1;
                }
            }
        }

        // Extract neighborhood features
        for (entity, pred_counts) in neighborhoods {
            for (predicate, count) in pred_counts {
                features.push(RelationalFeature::neighborhood_feature(
                    format!("{}_{}", entity, predicate),
                    count,
                ));
            }
        }

        // Extract path features (simplified: just collect predicate sequences)
        let predicates: Vec<_> = data
            .iter()
            .filter_map(|atom| {
                if let RuleAtom::Triple {
                    predicate: Term::Constant(p),
                    ..
                } = atom
                {
                    Some(p.clone())
                } else {
                    None
                }
            })
            .collect();

        // Create bigram path features
        for window in predicates.windows(2) {
            if let [p1, p2] = window {
                features.push(RelationalFeature::path_feature(vec![
                    p1.clone(),
                    p2.clone(),
                ]));
            }
        }

        Ok(features)
    }

    /// Predict link between two entities
    pub fn predict_link(&self, _entity1: &str, _entity2: &str, predicate: &str) -> f64 {
        // Simplified link prediction using rule weights
        let mut score = 0.0;

        for (rule, weight) in &self.model.weighted_rules {
            // Check if rule head matches the query predicate
            if let Some(RuleAtom::Triple {
                predicate: Term::Constant(head_pred),
                ..
            }) = rule.head.first()
            {
                if head_pred == predicate || head_pred == "related" {
                    score += weight;
                }
            }
        }

        // Normalize to probability
        1.0 / (1.0 + (-score).exp())
    }

    /// Classify entity based on relational features
    pub fn classify_entity(&self, entity: &str, data: &[RuleAtom]) -> Result<Option<String>> {
        // Extract features for this entity
        let entity_features: Vec<_> = data
            .iter()
            .filter(|atom| {
                if let RuleAtom::Triple {
                    subject: Term::Constant(s),
                    ..
                } = atom
                {
                    s == entity
                } else {
                    false
                }
            })
            .collect();

        if entity_features.is_empty() {
            return Ok(None);
        }

        // Extract features
        let atoms: Vec<_> = entity_features.into_iter().cloned().collect();
        let features = self.extract_features(&atoms)?;

        // Predict class
        Ok(self.model.predict_class(&features))
    }

    /// Get the learned model
    pub fn get_model(&self) -> &StatisticalRelationalModel {
        &self.model
    }

    /// Get predicate statistics
    pub fn get_predicate_stats(&self) -> &HashMap<String, usize> {
        &self.predicate_counts
    }
}

impl Default for StatisticalRelationalLearner {
    fn default() -> Self {
        Self::new()
    }
}

/// Collective classification for jointly classifying related entities
#[derive(Debug)]
pub struct CollectiveClassifier {
    /// Base learner
    learner: StatisticalRelationalLearner,
    /// Maximum iterations for collective inference
    max_iterations: usize,
    /// Convergence threshold
    #[allow(dead_code)]
    convergence_threshold: f64,
}

impl CollectiveClassifier {
    /// Create a new collective classifier
    pub fn new(max_iterations: usize, convergence_threshold: f64) -> Self {
        Self {
            learner: StatisticalRelationalLearner::new(),
            max_iterations,
            convergence_threshold,
        }
    }

    /// Fit the model
    pub fn fit(&mut self, data: &[RuleAtom]) -> Result<()> {
        self.learner.fit(data)
    }

    /// Perform collective classification using iterative refinement
    pub fn classify_collectively(
        &self,
        entities: &[String],
        data: &[RuleAtom],
    ) -> Result<HashMap<String, String>> {
        let mut classifications: HashMap<String, String> = HashMap::new();

        // Initialize with individual classifications
        for entity in entities {
            if let Ok(Some(class)) = self.learner.classify_entity(entity, data) {
                classifications.insert(entity.clone(), class);
            }
        }

        // Iterative refinement
        for _iter in 0..self.max_iterations {
            let mut changed = false;

            for entity in entities {
                // Get neighbor classifications
                let neighbor_classes = self.get_neighbor_classes(entity, data, &classifications);

                // Re-classify based on neighbors (majority voting)
                if let Some(majority_class) = self.majority_vote(&neighbor_classes) {
                    if classifications.get(entity) != Some(&majority_class) {
                        classifications.insert(entity.clone(), majority_class);
                        changed = true;
                    }
                }
            }

            // Check convergence
            if !changed {
                break;
            }
        }

        Ok(classifications)
    }

    /// Get classifications of neighboring entities
    fn get_neighbor_classes(
        &self,
        entity: &str,
        data: &[RuleAtom],
        current_classifications: &HashMap<String, String>,
    ) -> Vec<String> {
        let mut neighbor_classes = Vec::new();

        for atom in data {
            if let RuleAtom::Triple {
                subject: Term::Constant(s),
                object: Term::Constant(o),
                ..
            } = atom
            {
                if s == entity {
                    if let Some(class) = current_classifications.get(o) {
                        neighbor_classes.push(class.clone());
                    }
                } else if o == entity {
                    if let Some(class) = current_classifications.get(s) {
                        neighbor_classes.push(class.clone());
                    }
                }
            }
        }

        neighbor_classes
    }

    /// Majority vote among classes
    fn majority_vote(&self, classes: &[String]) -> Option<String> {
        if classes.is_empty() {
            return None;
        }

        let mut counts: HashMap<String, usize> = HashMap::new();
        for class in classes {
            *counts.entry(class.clone()).or_insert(0) += 1;
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(class, _)| class)
    }
}

impl Default for CollectiveClassifier {
    fn default() -> Self {
        Self::new(10, 0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> Vec<RuleAtom> {
        vec![
            RuleAtom::Triple {
                subject: Term::Constant("alice".to_string()),
                predicate: Term::Constant("friend".to_string()),
                object: Term::Constant("bob".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("bob".to_string()),
                predicate: Term::Constant("friend".to_string()),
                object: Term::Constant("charlie".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("alice".to_string()),
                predicate: Term::Constant("rdf:type".to_string()),
                object: Term::Constant("Person".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("bob".to_string()),
                predicate: Term::Constant("rdf:type".to_string()),
                object: Term::Constant("Person".to_string()),
            },
        ]
    }

    #[test]
    fn test_relational_feature_creation() {
        let feature =
            RelationalFeature::path_feature(vec!["friend".to_string(), "likes".to_string()]);

        assert_eq!(feature.feature_type, "path");
        assert_eq!(feature.pattern.len(), 2);
    }

    #[test]
    fn test_srl_learner_basic() {
        let mut learner = StatisticalRelationalLearner::new();
        let data = create_test_data();

        let result = learner.fit(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_schema_discovery() {
        let mut learner = StatisticalRelationalLearner::new();
        let data = create_test_data();

        learner.fit(&data).unwrap();

        // Should discover 'friend' and 'rdf:type' predicates
        let stats = learner.get_predicate_stats();
        assert!(stats.contains_key("friend"));
        assert!(stats.contains_key("rdf:type"));
    }

    #[test]
    fn test_structure_learning() {
        let mut learner = StatisticalRelationalLearner::new();
        let data = create_test_data();

        learner.fit(&data).unwrap();

        // Should learn some rules
        let model = learner.get_model();
        assert!(!model.weighted_rules.is_empty());
    }

    #[test]
    fn test_parameter_learning() {
        let mut learner = StatisticalRelationalLearner::new();
        let data = create_test_data();

        learner.fit(&data).unwrap();

        // Should learn class priors
        let model = learner.get_model();
        assert!(!model.class_priors.is_empty());
        assert!(model.class_priors.contains_key("Person"));
    }

    #[test]
    fn test_feature_extraction() {
        let learner = StatisticalRelationalLearner::new();
        let data = create_test_data();

        let features = learner.extract_features(&data).unwrap();

        // Should extract some features
        assert!(!features.is_empty());
    }

    #[test]
    fn test_link_prediction() {
        let mut learner = StatisticalRelationalLearner::new();
        let data = create_test_data();

        learner.fit(&data).unwrap();

        let prob = learner.predict_link("alice", "charlie", "friend");

        // Should return a probability in [0, 1]
        assert!((0.0..=1.0).contains(&prob));
    }

    #[test]
    fn test_entity_classification() {
        let mut learner = StatisticalRelationalLearner::new();
        let data = create_test_data();

        learner.fit(&data).unwrap();

        let classification = learner.classify_entity("alice", &data).unwrap();

        // alice should be classified (has rdf:type in data)
        assert!(classification.is_some());
    }

    #[test]
    fn test_collective_classification() {
        let mut classifier = CollectiveClassifier::new(10, 0.01);
        let data = create_test_data();

        classifier.fit(&data).unwrap();

        let entities = vec!["alice".to_string(), "bob".to_string()];
        let classifications = classifier.classify_collectively(&entities, &data).unwrap();

        // Should classify both entities
        assert!(classifications.contains_key("alice"));
        assert!(classifications.contains_key("bob"));
    }

    #[test]
    fn test_statistical_relational_model() {
        let mut model = StatisticalRelationalModel::new();

        let rule = Rule {
            name: "test_rule".to_string(),
            body: vec![],
            head: vec![],
        };

        model.add_weighted_rule(rule, 0.8);

        assert_eq!(model.weighted_rules.len(), 1);
        assert!((model.weighted_rules[0].1 - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_class_probability_prediction() {
        let mut model = StatisticalRelationalModel::new();

        model.class_priors.insert("Person".to_string(), 0.6);

        let feature = RelationalFeature::path_feature(vec!["friend".to_string()]);
        model.feature_weights.insert(feature.clone(), 0.5);

        let prob = model.predict_class_probability(&[feature], "Person");

        // Should return a valid probability
        assert!((0.0..=1.0).contains(&prob));
    }
}
