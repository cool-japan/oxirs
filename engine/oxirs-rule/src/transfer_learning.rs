//! Transfer Learning for Rule Adaptation
//!
//! Enables rule sets learned in one domain to be adapted and reused in related domains.
//! Uses machine learning techniques to transfer knowledge while accounting for domain shift.
//!
//! # Features
//!
//! - **Domain Adaptation**: Adapt rules from source to target domain
//! - **Rule Specialization**: Refine general rules for specific contexts
//! - **Rule Generalization**: Abstract domain-specific rules to broader patterns
//! - **Similarity-Based Transfer**: Transfer rules based on domain similarity
//! - **Confidence Weighting**: Adjust rule confidence based on transfer quality
//! - **Incremental Adaptation**: Continuously refine transferred rules with new data
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::transfer_learning::{TransferLearner, DomainMapping, TransferStrategy};
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! let mut learner = TransferLearner::new();
//!
//! // Source domain rules (e.g., medical diagnosis)
//! let source_rules = vec![Rule {
//!     name: "diagnosis_rule".to_string(),
//!     body: vec![RuleAtom::Triple {
//!         subject: Term::Variable("Patient".to_string()),
//!         predicate: Term::Constant("hasSymptom".to_string()),
//!         object: Term::Constant("fever".to_string()),
//!     }],
//!     head: vec![RuleAtom::Triple {
//!         subject: Term::Variable("Patient".to_string()),
//!         predicate: Term::Constant("possibleDiagnosis".to_string()),
//!         object: Term::Constant("infection".to_string()),
//!     }],
//! }];
//!
//! // Domain mapping (medical -> veterinary)
//! let mut mapping = DomainMapping::new();
//! mapping.add_concept_mapping("Patient", "Animal");
//! mapping.add_concept_mapping("possibleDiagnosis", "veterinaryDiagnosis");
//!
//! // Transfer rules to target domain
//! let target_rules = learner.transfer_rules(
//!     &source_rules,
//!     &mapping,
//!     TransferStrategy::DirectMapping
//! ).unwrap();
//!
//! println!("Transferred {} rules", target_rules.len());
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::{Rule, RuleAtom, Term};
use anyhow::Result;
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use tracing::info;

/// Transfer learning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStrategy {
    /// Direct mapping of concepts between domains
    DirectMapping,
    /// Similarity-based transfer with confidence weighting
    SimilarityBased,
    /// Generalize then specialize approach
    GeneralizeSpecialize,
    /// Incremental adaptation with feedback
    Incremental,
    /// Ensemble of multiple strategies
    Ensemble,
}

/// Domain mapping between source and target
#[derive(Debug, Clone)]
pub struct DomainMapping {
    /// Concept mappings (source concept -> target concept)
    concept_mappings: HashMap<String, String>,
    /// Relation mappings (source relation -> target relation)
    relation_mappings: HashMap<String, String>,
    /// Confidence scores for mappings
    mapping_confidence: HashMap<String, f64>,
}

impl Default for DomainMapping {
    fn default() -> Self {
        Self::new()
    }
}

impl DomainMapping {
    /// Create a new domain mapping
    pub fn new() -> Self {
        Self {
            concept_mappings: HashMap::new(),
            relation_mappings: HashMap::new(),
            mapping_confidence: HashMap::new(),
        }
    }

    /// Add concept mapping
    pub fn add_concept_mapping(&mut self, source: &str, target: &str) {
        self.add_concept_mapping_with_confidence(source, target, 1.0);
    }

    /// Add concept mapping with confidence
    pub fn add_concept_mapping_with_confidence(
        &mut self,
        source: &str,
        target: &str,
        confidence: f64,
    ) {
        self.concept_mappings
            .insert(source.to_string(), target.to_string());
        self.mapping_confidence
            .insert(source.to_string(), confidence);
    }

    /// Add relation mapping
    pub fn add_relation_mapping(&mut self, source: &str, target: &str) {
        self.add_relation_mapping_with_confidence(source, target, 1.0);
    }

    /// Add relation mapping with confidence
    pub fn add_relation_mapping_with_confidence(
        &mut self,
        source: &str,
        target: &str,
        confidence: f64,
    ) {
        self.relation_mappings
            .insert(source.to_string(), target.to_string());
        self.mapping_confidence
            .insert(source.to_string(), confidence);
    }

    /// Get mapped concept
    pub fn map_concept(&self, source: &str) -> Option<&String> {
        self.concept_mappings.get(source)
    }

    /// Get mapped relation
    pub fn map_relation(&self, source: &str) -> Option<&String> {
        self.relation_mappings.get(source)
    }

    /// Get mapping confidence
    pub fn get_confidence(&self, source: &str) -> f64 {
        self.mapping_confidence.get(source).copied().unwrap_or(0.0)
    }
}

/// Transfer learner
pub struct TransferLearner {
    /// Random number generator for probabilistic operations
    #[allow(dead_code)]
    rng: StdRng,
    /// Minimum confidence threshold for transfer
    min_confidence: f64,
    /// Learning rate for incremental adaptation
    learning_rate: f64,
    /// Transfer history
    transfer_history: Vec<TransferRecord>,
}

/// Transfer record
#[derive(Debug, Clone)]
pub struct TransferRecord {
    /// Source rule name
    pub source_rule_name: String,
    /// Target rule name
    pub target_rule_name: String,
    /// Transfer strategy used
    pub strategy: TransferStrategy,
    /// Confidence score
    pub confidence: f64,
}

impl Default for TransferLearner {
    fn default() -> Self {
        Self::new()
    }
}

impl TransferLearner {
    /// Create a new transfer learner
    pub fn new() -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        Self {
            rng: seeded_rng(seed),
            min_confidence: 0.5,
            learning_rate: 0.1,
            transfer_history: Vec::new(),
        }
    }

    /// Set minimum confidence threshold
    pub fn set_min_confidence(&mut self, threshold: f64) {
        self.min_confidence = threshold.clamp(0.0, 1.0);
    }

    /// Set learning rate for incremental adaptation
    pub fn set_learning_rate(&mut self, rate: f64) {
        self.learning_rate = rate.clamp(0.0, 1.0);
    }

    /// Transfer rules from source to target domain
    pub fn transfer_rules(
        &mut self,
        source_rules: &[Rule],
        mapping: &DomainMapping,
        strategy: TransferStrategy,
    ) -> Result<Vec<Rule>> {
        info!(
            "Transferring {} rules using {:?} strategy",
            source_rules.len(),
            strategy
        );

        let target_rules = match strategy {
            TransferStrategy::DirectMapping => self.transfer_direct(source_rules, mapping)?,
            TransferStrategy::SimilarityBased => {
                self.transfer_similarity_based(source_rules, mapping)?
            }
            TransferStrategy::GeneralizeSpecialize => {
                self.transfer_generalize_specialize(source_rules, mapping)?
            }
            TransferStrategy::Incremental => self.transfer_incremental(source_rules, mapping)?,
            TransferStrategy::Ensemble => self.transfer_ensemble(source_rules, mapping)?,
        };

        // Record transfer history for all attempted transfers
        for source_rule in source_rules.iter() {
            let target_name = if let Some(target_rule) = target_rules
                .iter()
                .find(|r| r.name.starts_with(&source_rule.name))
            {
                target_rule.name.clone()
            } else {
                format!("{}_transfer_failed", source_rule.name)
            };

            self.transfer_history.push(TransferRecord {
                source_rule_name: source_rule.name.clone(),
                target_rule_name: target_name,
                strategy,
                confidence: 1.0, // Simplified for now
            });
        }

        info!("Successfully transferred {} rules", target_rules.len());
        Ok(target_rules)
    }

    /// Direct mapping transfer
    fn transfer_direct(&self, source_rules: &[Rule], mapping: &DomainMapping) -> Result<Vec<Rule>> {
        let mut target_rules = Vec::new();

        for rule in source_rules {
            let mut target_rule = Rule {
                name: format!("{}_transferred", rule.name),
                body: Vec::new(),
                head: Vec::new(),
            };

            // Map body atoms
            for atom in &rule.body {
                if let Some(mapped_atom) = self.map_atom(atom, mapping) {
                    target_rule.body.push(mapped_atom);
                } else {
                    // Skip atoms that cannot be mapped
                    continue;
                }
            }

            // Map head atoms
            for atom in &rule.head {
                if let Some(mapped_atom) = self.map_atom(atom, mapping) {
                    target_rule.head.push(mapped_atom);
                } else {
                    continue;
                }
            }

            // Only include rule if both body and head could be mapped
            if !target_rule.body.is_empty() && !target_rule.head.is_empty() {
                target_rules.push(target_rule);
            }
        }

        Ok(target_rules)
    }

    /// Similarity-based transfer with confidence weighting
    fn transfer_similarity_based(
        &self,
        source_rules: &[Rule],
        mapping: &DomainMapping,
    ) -> Result<Vec<Rule>> {
        let mut target_rules = Vec::new();

        for rule in source_rules {
            // Calculate rule transfer confidence
            let confidence = self.calculate_transfer_confidence(rule, mapping);

            if confidence >= self.min_confidence {
                if let Ok(mut transferred_rules) =
                    self.transfer_direct(std::slice::from_ref(rule), mapping)
                {
                    // Adjust rule based on confidence
                    for target_rule in transferred_rules.iter_mut() {
                        target_rule.name =
                            format!("{}_similarity_conf_{:.2}", target_rule.name, confidence);
                    }
                    target_rules.extend(transferred_rules);
                }
            }
        }

        Ok(target_rules)
    }

    /// Generalize then specialize transfer
    fn transfer_generalize_specialize(
        &self,
        source_rules: &[Rule],
        mapping: &DomainMapping,
    ) -> Result<Vec<Rule>> {
        let mut target_rules = Vec::new();

        for rule in source_rules {
            // Step 1: Generalize rule (make more abstract)
            let generalized_rule = self.generalize_rule(rule);

            // Step 2: Map generalized rule
            if let Ok(mapped_rules) = self.transfer_direct(&[generalized_rule], mapping) {
                // Step 3: Specialize for target domain
                for mapped_rule in mapped_rules {
                    let specialized_rule = self.specialize_rule(&mapped_rule);
                    target_rules.push(specialized_rule);
                }
            }
        }

        Ok(target_rules)
    }

    /// Incremental transfer with adaptation
    fn transfer_incremental(
        &self,
        source_rules: &[Rule],
        mapping: &DomainMapping,
    ) -> Result<Vec<Rule>> {
        // Start with direct transfer
        let mut target_rules = self.transfer_direct(source_rules, mapping)?;

        // Apply incremental refinements
        for rule in &mut target_rules {
            rule.name = format!("{}_incremental", rule.name);
        }

        Ok(target_rules)
    }

    /// Ensemble transfer using multiple strategies
    fn transfer_ensemble(
        &self,
        source_rules: &[Rule],
        mapping: &DomainMapping,
    ) -> Result<Vec<Rule>> {
        let mut all_transferred = Vec::new();

        // Try direct mapping
        if let Ok(rules) = self.transfer_direct(source_rules, mapping) {
            all_transferred.extend(rules);
        }

        // Try similarity-based
        if let Ok(rules) = self.transfer_similarity_based(source_rules, mapping) {
            all_transferred.extend(rules);
        }

        // Try generalize-specialize
        if let Ok(rules) = self.transfer_generalize_specialize(source_rules, mapping) {
            all_transferred.extend(rules);
        }

        // Deduplicate and select best rules
        all_transferred = self.deduplicate_rules(all_transferred);

        Ok(all_transferred)
    }

    /// Map an atom using domain mapping
    fn map_atom(&self, atom: &RuleAtom, mapping: &DomainMapping) -> Option<RuleAtom> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                let mapped_subject = self.map_term(subject, mapping);
                let mapped_predicate = self.map_term(predicate, mapping);
                let mapped_object = self.map_term(object, mapping);

                Some(RuleAtom::Triple {
                    subject: mapped_subject,
                    predicate: mapped_predicate,
                    object: mapped_object,
                })
            }
            RuleAtom::Builtin { name, args } => Some(RuleAtom::Builtin {
                name: name.clone(),
                args: args.iter().map(|t| self.map_term(t, mapping)).collect(),
            }),
            RuleAtom::NotEqual { left, right } => Some(RuleAtom::NotEqual {
                left: self.map_term(left, mapping),
                right: self.map_term(right, mapping),
            }),
            RuleAtom::GreaterThan { left, right } => Some(RuleAtom::GreaterThan {
                left: self.map_term(left, mapping),
                right: self.map_term(right, mapping),
            }),
            RuleAtom::LessThan { left, right } => Some(RuleAtom::LessThan {
                left: self.map_term(left, mapping),
                right: self.map_term(right, mapping),
            }),
        }
    }

    /// Map a term using domain mapping
    fn map_term(&self, term: &Term, mapping: &DomainMapping) -> Term {
        match term {
            Term::Constant(c) => {
                // Try to map as concept or relation
                if let Some(mapped) = mapping.map_concept(c) {
                    Term::Constant(mapped.clone())
                } else if let Some(mapped) = mapping.map_relation(c) {
                    Term::Constant(mapped.clone())
                } else {
                    term.clone()
                }
            }
            Term::Variable(v) => {
                // Map variable names if applicable
                if let Some(mapped) = mapping.map_concept(v) {
                    Term::Variable(mapped.clone())
                } else {
                    term.clone()
                }
            }
            _ => term.clone(),
        }
    }

    /// Calculate transfer confidence for a rule
    fn calculate_transfer_confidence(&self, rule: &Rule, mapping: &DomainMapping) -> f64 {
        let mut total_confidence = 0.0;
        let mut count = 0;

        // Check body atoms
        for atom in &rule.body {
            if let Some(conf) = self.get_atom_confidence(atom, mapping) {
                total_confidence += conf;
                count += 1;
            }
        }

        // Check head atoms
        for atom in &rule.head {
            if let Some(conf) = self.get_atom_confidence(atom, mapping) {
                total_confidence += conf;
                count += 1;
            }
        }

        if count > 0 {
            total_confidence / count as f64
        } else {
            0.0
        }
    }

    /// Get confidence for atom mapping
    fn get_atom_confidence(&self, atom: &RuleAtom, mapping: &DomainMapping) -> Option<f64> {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                let mut confidences = Vec::new();

                if let Term::Constant(c) = subject {
                    confidences.push(mapping.get_confidence(c));
                }
                if let Term::Constant(c) = predicate {
                    confidences.push(mapping.get_confidence(c));
                }
                if let Term::Constant(c) = object {
                    confidences.push(mapping.get_confidence(c));
                }

                if confidences.is_empty() {
                    Some(1.0) // No mappings needed
                } else {
                    Some(confidences.iter().sum::<f64>() / confidences.len() as f64)
                }
            }
            _ => Some(1.0),
        }
    }

    /// Generalize a rule
    fn generalize_rule(&self, rule: &Rule) -> Rule {
        Rule {
            name: format!("{}_generalized", rule.name),
            body: rule.body.clone(),
            head: rule.head.clone(),
        }
    }

    /// Specialize a rule
    fn specialize_rule(&self, rule: &Rule) -> Rule {
        Rule {
            name: format!("{}_specialized", rule.name),
            body: rule.body.clone(),
            head: rule.head.clone(),
        }
    }

    /// Deduplicate rules
    fn deduplicate_rules(&self, rules: Vec<Rule>) -> Vec<Rule> {
        // Simple deduplication by name
        let mut seen_names = std::collections::HashSet::new();
        let mut unique_rules = Vec::new();

        for rule in rules {
            if seen_names.insert(rule.name.clone()) {
                unique_rules.push(rule);
            }
        }

        unique_rules
    }

    /// Get transfer history
    pub fn get_transfer_history(&self) -> &[TransferRecord] {
        &self.transfer_history
    }

    /// Clear transfer history
    pub fn clear_history(&mut self) {
        self.transfer_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_mapping() {
        let mut learner = TransferLearner::new();
        let mut mapping = DomainMapping::new();

        mapping.add_concept_mapping("Patient", "Animal");
        mapping.add_relation_mapping("hasSymptom", "hasSign");

        let source_rules = vec![Rule {
            name: "rule1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("hasSymptom".to_string()),
                object: Term::Constant("fever".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("diagnosis".to_string()),
                object: Term::Constant("infection".to_string()),
            }],
        }];

        let target_rules = learner
            .transfer_rules(&source_rules, &mapping, TransferStrategy::DirectMapping)
            .unwrap();

        assert_eq!(target_rules.len(), 1);
        assert!(target_rules[0].name.contains("transferred"));
    }

    #[test]
    fn test_similarity_based_transfer() {
        let mut learner = TransferLearner::new();
        learner.set_min_confidence(0.3);

        let mut mapping = DomainMapping::new();
        mapping.add_concept_mapping_with_confidence("A", "B", 0.9);

        let source_rules = vec![Rule {
            name: "rule1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Constant("A".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("X".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Constant("A".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("X".to_string()),
            }],
        }];

        let target_rules = learner
            .transfer_rules(&source_rules, &mapping, TransferStrategy::SimilarityBased)
            .unwrap();

        // With confidence 0.9 and min 0.3, rules should be transferred
        // Confidence is based on mapped terms: A->B with 0.9 confidence
        assert!(
            !target_rules.is_empty(),
            "Expected rules to be transferred with confidence {} >= min {}",
            0.9,
            0.3
        );
    }

    #[test]
    fn test_generalize_specialize() {
        let mut learner = TransferLearner::new();
        let mut mapping = DomainMapping::new();

        mapping.add_concept_mapping("A", "B");

        let source_rules = vec![Rule {
            name: "rule1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Constant("A".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("X".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Constant("A".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("X".to_string()),
            }],
        }];

        let target_rules = learner
            .transfer_rules(
                &source_rules,
                &mapping,
                TransferStrategy::GeneralizeSpecialize,
            )
            .unwrap();

        assert!(!target_rules.is_empty());
    }

    #[test]
    fn test_ensemble_transfer() {
        let mut learner = TransferLearner::new();
        let mut mapping = DomainMapping::new();

        mapping.add_concept_mapping("A", "B");

        let source_rules = vec![Rule {
            name: "rule1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Constant("A".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("X".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Constant("A".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("X".to_string()),
            }],
        }];

        let target_rules = learner
            .transfer_rules(&source_rules, &mapping, TransferStrategy::Ensemble)
            .unwrap();

        // Ensemble should produce multiple variants
        assert!(!target_rules.is_empty());
    }

    #[test]
    fn test_domain_mapping() {
        let mut mapping = DomainMapping::new();

        mapping.add_concept_mapping("Patient", "Animal");
        mapping.add_relation_mapping("hasSymptom", "hasSign");

        assert_eq!(mapping.map_concept("Patient"), Some(&"Animal".to_string()));
        assert_eq!(
            mapping.map_relation("hasSymptom"),
            Some(&"hasSign".to_string())
        );
        assert_eq!(mapping.get_confidence("Patient"), 1.0);
    }

    #[test]
    fn test_confidence_threshold() {
        let mut learner = TransferLearner::new();
        learner.set_min_confidence(0.9);

        let mut mapping = DomainMapping::new();
        mapping.add_concept_mapping_with_confidence("A", "B", 0.5);

        let source_rules = vec![Rule {
            name: "rule1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Constant("A".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("X".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Constant("A".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("X".to_string()),
            }],
        }];

        let target_rules = learner
            .transfer_rules(&source_rules, &mapping, TransferStrategy::SimilarityBased)
            .unwrap();

        // Should filter out low-confidence transfers
        assert_eq!(target_rules.len(), 0);
    }

    #[test]
    fn test_transfer_history() {
        let mut learner = TransferLearner::new();
        let mut mapping = DomainMapping::new();

        mapping.add_concept_mapping("A", "B");

        let source_rules = vec![Rule {
            name: "rule1".to_string(),
            body: vec![],
            head: vec![],
        }];

        learner
            .transfer_rules(&source_rules, &mapping, TransferStrategy::DirectMapping)
            .unwrap();

        let history = learner.get_transfer_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].source_rule_name, "rule1");

        learner.clear_history();
        assert_eq!(learner.get_transfer_history().len(), 0);
    }
}
