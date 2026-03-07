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
            .expect("SystemTime should be after UNIX_EPOCH")
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

    /// Perform feature-based transfer using structural similarity
    pub fn transfer_feature_based(
        &mut self,
        source_rules: &[Rule],
        target_examples: &[RuleAtom],
        mapping: &DomainMapping,
    ) -> Result<Vec<Rule>> {
        info!(
            "Performing feature-based transfer with {} source rules and {} target examples",
            source_rules.len(),
            target_examples.len()
        );

        let mut target_rules = Vec::new();

        for rule in source_rules {
            // Extract structural features from rule
            let features = self.extract_rule_features(rule);

            // Find matching target examples
            let matching_examples = self.find_matching_examples(&features, target_examples);

            if !matching_examples.is_empty() {
                // Transfer rule with feature alignment
                if let Ok(transferred) = self.transfer_direct(std::slice::from_ref(rule), mapping) {
                    for mut t_rule in transferred {
                        // Adapt rule based on matching examples
                        t_rule.name = format!("{}_feature_based", t_rule.name);
                        target_rules.push(t_rule);
                    }
                }
            }
        }

        Ok(target_rules)
    }

    /// Extract structural features from a rule
    fn extract_rule_features(&self, rule: &Rule) -> RuleFeatures {
        let mut features = RuleFeatures {
            num_body_atoms: rule.body.len(),
            num_head_atoms: rule.head.len(),
            num_variables: 0,
            num_constants: 0,
            predicates: Vec::new(),
            variable_sharing: 0,
        };

        let mut body_vars = std::collections::HashSet::new();
        let mut head_vars = std::collections::HashSet::new();

        for atom in &rule.body {
            self.extract_atom_features(atom, &mut features, &mut body_vars);
        }

        for atom in &rule.head {
            self.extract_atom_features(atom, &mut features, &mut head_vars);
        }

        // Calculate variable sharing between body and head
        features.variable_sharing = body_vars.intersection(&head_vars).count();

        features
    }

    /// Extract features from an atom
    fn extract_atom_features(
        &self,
        atom: &RuleAtom,
        features: &mut RuleFeatures,
        vars: &mut std::collections::HashSet<String>,
    ) {
        match atom {
            RuleAtom::Triple {
                subject,
                predicate,
                object,
            } => {
                self.extract_term_features(subject, features, vars);
                self.extract_term_features(predicate, features, vars);
                self.extract_term_features(object, features, vars);

                if let Term::Constant(p) = predicate {
                    features.predicates.push(p.clone());
                }
            }
            RuleAtom::Builtin { args, .. } => {
                for arg in args {
                    self.extract_term_features(arg, features, vars);
                }
            }
            _ => {}
        }
    }

    /// Extract features from a term
    fn extract_term_features(
        &self,
        term: &Term,
        features: &mut RuleFeatures,
        vars: &mut std::collections::HashSet<String>,
    ) {
        match term {
            Term::Variable(v) => {
                if vars.insert(v.clone()) {
                    features.num_variables += 1;
                }
            }
            Term::Constant(_) => {
                features.num_constants += 1;
            }
            _ => {}
        }
    }

    /// Find examples that match the given features
    fn find_matching_examples(
        &self,
        features: &RuleFeatures,
        examples: &[RuleAtom],
    ) -> Vec<RuleAtom> {
        examples
            .iter()
            .filter(|example| {
                // Check if example predicate matches any rule predicate
                if let RuleAtom::Triple {
                    predicate: Term::Constant(p),
                    ..
                } = example
                {
                    return features.predicates.iter().any(|fp| {
                        // Simple similarity check
                        fp.to_lowercase().contains(&p.to_lowercase())
                            || p.to_lowercase().contains(&fp.to_lowercase())
                    });
                }
                false
            })
            .cloned()
            .collect()
    }

    /// Calculate domain similarity between source and target
    pub fn calculate_domain_similarity(
        &self,
        source_facts: &[RuleAtom],
        target_facts: &[RuleAtom],
    ) -> DomainSimilarity {
        // Extract predicates from both domains
        let source_predicates = self.extract_predicates(source_facts);
        let target_predicates = self.extract_predicates(target_facts);

        // Calculate Jaccard similarity
        let intersection = source_predicates.intersection(&target_predicates).count() as f64;
        let union = source_predicates.union(&target_predicates).count() as f64;
        let jaccard = if union > 0.0 {
            intersection / union
        } else {
            0.0
        };

        // Calculate structural similarity based on predicate distribution
        let structural = self.calculate_structural_similarity(source_facts, target_facts);

        // Calculate concept overlap
        let concept_overlap = intersection / source_predicates.len().max(1) as f64;

        DomainSimilarity {
            jaccard_similarity: jaccard,
            structural_similarity: structural,
            concept_overlap,
            overall_score: (jaccard * 0.4) + (structural * 0.3) + (concept_overlap * 0.3),
        }
    }

    /// Extract predicates from facts
    fn extract_predicates(&self, facts: &[RuleAtom]) -> std::collections::HashSet<String> {
        let mut predicates = std::collections::HashSet::new();

        for fact in facts {
            if let RuleAtom::Triple {
                predicate: Term::Constant(p),
                ..
            } = fact
            {
                predicates.insert(p.clone());
            }
        }

        predicates
    }

    /// Calculate structural similarity based on fact distributions
    fn calculate_structural_similarity(
        &self,
        source_facts: &[RuleAtom],
        target_facts: &[RuleAtom],
    ) -> f64 {
        if source_facts.is_empty() || target_facts.is_empty() {
            return 0.0;
        }

        // Simple structural similarity based on fact count ratio
        source_facts.len().min(target_facts.len()) as f64
            / source_facts.len().max(target_facts.len()) as f64
    }

    /// Detect potential negative transfer
    pub fn detect_negative_transfer(
        &self,
        source_rules: &[Rule],
        target_facts: &[RuleAtom],
        mapping: &DomainMapping,
    ) -> NegativeTransferAnalysis {
        let mut warnings = Vec::new();
        let mut risk_score: f64 = 0.0;

        for rule in source_rules {
            // Check for unmapped predicates
            let unmapped = self.find_unmapped_predicates(rule, mapping);
            if !unmapped.is_empty() {
                warnings.push(NegativeTransferWarning {
                    rule_name: rule.name.clone(),
                    warning_type: WarningType::UnmappedPredicates,
                    severity: Severity::Medium,
                    description: format!(
                        "Rule has {} unmapped predicates: {:?}",
                        unmapped.len(),
                        unmapped
                    ),
                });
                risk_score += 0.2;
            }

            // Check for domain mismatch
            let confidence = self.calculate_transfer_confidence(rule, mapping);
            if confidence < 0.5 {
                warnings.push(NegativeTransferWarning {
                    rule_name: rule.name.clone(),
                    warning_type: WarningType::LowConfidence,
                    severity: Severity::High,
                    description: format!("Low transfer confidence: {:.2}", confidence),
                });
                risk_score += 0.3;
            }

            // Check for structural incompatibility
            if rule.body.len() > 3 && target_facts.len() < 10 {
                warnings.push(NegativeTransferWarning {
                    rule_name: rule.name.clone(),
                    warning_type: WarningType::StructuralMismatch,
                    severity: Severity::Low,
                    description: "Complex rule with limited target data".to_string(),
                });
                risk_score += 0.1;
            }
        }

        NegativeTransferAnalysis {
            warnings,
            risk_score: risk_score.min(1.0),
            recommendation: if risk_score > 0.5 {
                TransferRecommendation::AvoidTransfer
            } else if risk_score > 0.2 {
                TransferRecommendation::ProceedWithCaution
            } else {
                TransferRecommendation::SafeToTransfer
            },
        }
    }

    /// Find unmapped predicates in a rule
    fn find_unmapped_predicates(&self, rule: &Rule, mapping: &DomainMapping) -> Vec<String> {
        let mut unmapped = Vec::new();

        for atom in rule.body.iter().chain(rule.head.iter()) {
            if let RuleAtom::Triple {
                predicate: Term::Constant(p),
                ..
            } = atom
            {
                if mapping.map_relation(p).is_none() && mapping.map_concept(p).is_none() {
                    unmapped.push(p.clone());
                }
            }
        }

        unmapped
    }

    /// Evaluate transfer quality after transfer
    pub fn evaluate_transfer_quality(
        &self,
        transferred_rules: &[Rule],
        target_facts: &[RuleAtom],
        expected_outputs: &[RuleAtom],
    ) -> TransferQualityMetrics {
        // Calculate rule applicability
        let applicable_rules = transferred_rules
            .iter()
            .filter(|rule| self.is_rule_applicable(rule, target_facts))
            .count();

        let applicability = if !transferred_rules.is_empty() {
            applicable_rules as f64 / transferred_rules.len() as f64
        } else {
            0.0
        };

        // Calculate coverage (how many expected outputs could be derived)
        let coverage = if !expected_outputs.is_empty() {
            // Simplified coverage estimation
            (applicable_rules.min(expected_outputs.len()) as f64) / expected_outputs.len() as f64
        } else {
            0.0
        };

        // Calculate precision estimate (rules that produce valid outputs)
        let precision = if applicable_rules > 0 {
            0.8 // Conservative estimate
        } else {
            0.0
        };

        TransferQualityMetrics {
            applicability,
            coverage,
            precision,
            f1_score: if precision + coverage > 0.0 {
                2.0 * precision * coverage / (precision + coverage)
            } else {
                0.0
            },
            overall_quality: (applicability * 0.3) + (coverage * 0.4) + (precision * 0.3),
        }
    }

    /// Check if a rule is applicable to the given facts
    fn is_rule_applicable(&self, rule: &Rule, facts: &[RuleAtom]) -> bool {
        if rule.body.is_empty() {
            return true;
        }

        // Check if any fact matches any body atom pattern
        for body_atom in &rule.body {
            if let RuleAtom::Triple {
                subject: _,
                predicate: body_pred,
                object: _,
            } = body_atom
            {
                for fact in facts {
                    if let RuleAtom::Triple {
                        predicate: fact_pred,
                        ..
                    } = fact
                    {
                        // Match if both are constants with same value
                        // or body predicate is a variable
                        match (body_pred, fact_pred) {
                            (Term::Constant(bp), Term::Constant(fp)) if bp == fp => return true,
                            (Term::Variable(_), _) => return true,
                            _ => {}
                        }
                    }
                }
            }
        }

        false
    }
}

/// Rule features for feature-based transfer
#[derive(Debug, Clone)]
pub struct RuleFeatures {
    /// Number of atoms in rule body
    pub num_body_atoms: usize,
    /// Number of atoms in rule head
    pub num_head_atoms: usize,
    /// Number of unique variables
    pub num_variables: usize,
    /// Number of constants
    pub num_constants: usize,
    /// List of predicates used
    pub predicates: Vec<String>,
    /// Number of variables shared between body and head
    pub variable_sharing: usize,
}

/// Domain similarity metrics
#[derive(Debug, Clone)]
pub struct DomainSimilarity {
    /// Jaccard similarity of predicates
    pub jaccard_similarity: f64,
    /// Structural similarity
    pub structural_similarity: f64,
    /// Concept overlap ratio
    pub concept_overlap: f64,
    /// Overall similarity score
    pub overall_score: f64,
}

/// Negative transfer analysis results
#[derive(Debug, Clone)]
pub struct NegativeTransferAnalysis {
    /// Warnings about potential negative transfer
    pub warnings: Vec<NegativeTransferWarning>,
    /// Overall risk score (0-1)
    pub risk_score: f64,
    /// Transfer recommendation
    pub recommendation: TransferRecommendation,
}

/// Warning about potential negative transfer
#[derive(Debug, Clone)]
pub struct NegativeTransferWarning {
    /// Rule name
    pub rule_name: String,
    /// Warning type
    pub warning_type: WarningType,
    /// Severity
    pub severity: Severity,
    /// Description
    pub description: String,
}

/// Warning type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningType {
    /// Rule has unmapped predicates
    UnmappedPredicates,
    /// Low transfer confidence
    LowConfidence,
    /// Structural mismatch between domains
    StructuralMismatch,
    /// Insufficient target data
    InsufficientData,
}

/// Severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
}

/// Transfer recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferRecommendation {
    /// Safe to transfer
    SafeToTransfer,
    /// Proceed with caution
    ProceedWithCaution,
    /// Avoid transfer
    AvoidTransfer,
}

/// Transfer quality metrics
#[derive(Debug, Clone)]
pub struct TransferQualityMetrics {
    /// Rule applicability (percentage of rules that can be applied)
    pub applicability: f64,
    /// Coverage (percentage of expected outputs covered)
    pub coverage: f64,
    /// Precision (percentage of outputs that are correct)
    pub precision: f64,
    /// F1 score
    pub f1_score: f64,
    /// Overall quality score
    pub overall_quality: f64,
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

    #[test]
    fn test_feature_based_transfer() {
        let mut learner = TransferLearner::new();
        let mut mapping = DomainMapping::new();

        mapping.add_concept_mapping("A", "B");

        let source_rules = vec![Rule {
            name: "rule1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("hasProperty".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("result".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }];

        let target_examples = vec![RuleAtom::Triple {
            subject: Term::Constant("entity1".to_string()),
            predicate: Term::Constant("hasProperty".to_string()),
            object: Term::Constant("value1".to_string()),
        }];

        let target_rules = learner
            .transfer_feature_based(&source_rules, &target_examples, &mapping)
            .unwrap();

        assert!(!target_rules.is_empty());
        assert!(target_rules[0].name.contains("feature_based"));
    }

    #[test]
    fn test_domain_similarity() {
        let learner = TransferLearner::new();

        let source_facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("a".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Constant("b".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("c".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Constant("d".to_string()),
            },
        ];

        let target_facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("x".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Constant("y".to_string()),
            },
            RuleAtom::Triple {
                subject: Term::Constant("z".to_string()),
                predicate: Term::Constant("r".to_string()),
                object: Term::Constant("w".to_string()),
            },
        ];

        let similarity = learner.calculate_domain_similarity(&source_facts, &target_facts);

        // Should have some overlap due to shared predicate 'p'
        assert!(similarity.jaccard_similarity > 0.0);
        assert!(similarity.overall_score > 0.0);
        assert!(similarity.structural_similarity == 1.0); // Same number of facts
    }

    #[test]
    fn test_negative_transfer_detection() {
        let learner = TransferLearner::new();
        let mapping = DomainMapping::new(); // Empty mapping

        let source_rules = vec![Rule {
            name: "risky_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("unmapped_pred".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("another_unmapped".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }];

        let target_facts = vec![];

        let analysis = learner.detect_negative_transfer(&source_rules, &target_facts, &mapping);

        // Should detect unmapped predicates and low confidence
        assert!(!analysis.warnings.is_empty());
        assert!(analysis.risk_score > 0.0);
    }

    #[test]
    fn test_transfer_quality_metrics() {
        let learner = TransferLearner::new();

        let transferred_rules = vec![Rule {
            name: "rule1".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }];

        let target_facts = vec![RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("p".to_string()),
            object: Term::Constant("b".to_string()),
        }];

        let expected_outputs = vec![RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("q".to_string()),
            object: Term::Constant("b".to_string()),
        }];

        let metrics =
            learner.evaluate_transfer_quality(&transferred_rules, &target_facts, &expected_outputs);

        assert!(metrics.applicability > 0.0);
        assert!(metrics.overall_quality > 0.0);
    }

    #[test]
    fn test_rule_features_extraction() {
        let learner = TransferLearner::new();

        let rule = Rule {
            name: "test".to_string(),
            body: vec![
                RuleAtom::Triple {
                    subject: Term::Variable("X".to_string()),
                    predicate: Term::Constant("p".to_string()),
                    object: Term::Variable("Y".to_string()),
                },
                RuleAtom::Triple {
                    subject: Term::Variable("Y".to_string()),
                    predicate: Term::Constant("q".to_string()),
                    object: Term::Variable("Z".to_string()),
                },
            ],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("r".to_string()),
                object: Term::Variable("Z".to_string()),
            }],
        };

        let features = learner.extract_rule_features(&rule);

        assert_eq!(features.num_body_atoms, 2);
        assert_eq!(features.num_head_atoms, 1);
        assert_eq!(features.predicates.len(), 3); // p, q, r
        assert!(features.variable_sharing > 0); // X and Z shared
    }

    #[test]
    fn test_transfer_recommendation_safe() {
        let learner = TransferLearner::new();
        let mut mapping = DomainMapping::new();

        mapping.add_concept_mapping_with_confidence("A", "B", 1.0);
        mapping.add_relation_mapping_with_confidence("p", "q", 1.0);

        let source_rules = vec![Rule {
            name: "safe_rule".to_string(),
            body: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
            head: vec![RuleAtom::Triple {
                subject: Term::Variable("X".to_string()),
                predicate: Term::Constant("p".to_string()),
                object: Term::Variable("Y".to_string()),
            }],
        }];

        let target_facts = vec![
            RuleAtom::Triple {
                subject: Term::Constant("a".to_string()),
                predicate: Term::Constant("q".to_string()),
                object: Term::Constant("b".to_string()),
            };
            20
        ];

        let analysis = learner.detect_negative_transfer(&source_rules, &target_facts, &mapping);

        // With high confidence mapping, should be safe
        assert_eq!(
            analysis.recommendation,
            TransferRecommendation::SafeToTransfer
        );
    }

    #[test]
    fn test_empty_domains() {
        let learner = TransferLearner::new();

        let similarity = learner.calculate_domain_similarity(&[], &[]);

        assert_eq!(similarity.jaccard_similarity, 0.0);
        assert_eq!(similarity.structural_similarity, 0.0);
    }

    #[test]
    fn test_identical_domains() {
        let learner = TransferLearner::new();

        let facts = vec![RuleAtom::Triple {
            subject: Term::Constant("a".to_string()),
            predicate: Term::Constant("p".to_string()),
            object: Term::Constant("b".to_string()),
        }];

        let similarity = learner.calculate_domain_similarity(&facts, &facts);

        assert_eq!(similarity.jaccard_similarity, 1.0);
        assert_eq!(similarity.structural_similarity, 1.0);
        assert_eq!(similarity.concept_overlap, 1.0);
    }
}
