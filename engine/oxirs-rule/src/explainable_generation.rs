//! Explainable Rule Generation
//!
//! Provides transparency in rule generation by explaining how rules were learned,
//! derived, or synthesized. Helps users understand and trust automatically generated rules.
//!
//! # Features
//!
//! - **Rule Provenance Tracking**: Track how rules were generated
//! - **Feature Importance**: Identify which features contributed to rule generation
//! - **Confidence Explanation**: Explain confidence scores
//! - **Counterfactual Analysis**: Show what would change the rule
//! - **Natural Language Explanations**: Generate human-readable explanations
//! - **Visual Explanations**: Generate rule dependency graphs
//!
//! # Example
//!
//! ```rust
//! use oxirs_rule::explainable_generation::{ExplainableGenerator, ExplanationType};
//! use oxirs_rule::{Rule, RuleAtom, Term};
//!
//! let mut generator = ExplainableGenerator::new();
//!
//! // Generate a rule with explanation
//! let rule = Rule {
//!     name: "transitivity_rule".to_string(),
//!     body: vec![
//!         RuleAtom::Triple {
//!             subject: Term::Variable("X".to_string()),
//!             predicate: Term::Constant("knows".to_string()),
//!             object: Term::Variable("Y".to_string()),
//!         },
//!         RuleAtom::Triple {
//!             subject: Term::Variable("Y".to_string()),
//!             predicate: Term::Constant("knows".to_string()),
//!             object: Term::Variable("Z".to_string()),
//!         },
//!     ],
//!     head: vec![RuleAtom::Triple {
//!         subject: Term::Variable("X".to_string()),
//!         predicate: Term::Constant("knows_indirectly".to_string()),
//!         object: Term::Variable("Z".to_string()),
//!     }],
//! };
//!
//! generator.register_rule(rule, "Learned from 100 examples with 95% confidence");
//!
//! // Get explanation
//! let explanation = generator.explain("transitivity_rule", ExplanationType::NaturalLanguage);
//! if let Some(exp) = explanation {
//!     println!("{}", exp);
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

use crate::Rule;
use std::collections::HashMap;
use tracing::info;

/// Type of explanation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExplanationType {
    /// Natural language explanation
    NaturalLanguage,
    /// Feature importance analysis
    FeatureImportance,
    /// Confidence breakdown
    ConfidenceAnalysis,
    /// Counterfactual analysis
    Counterfactual,
    /// Provenance trace
    Provenance,
}

/// Generation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationMethod {
    /// Learned from data
    MachineLearning,
    /// Derived from existing rules
    Derivation,
    /// Manually authored
    Manual,
    /// Synthesized automatically
    Synthesis,
    /// Transferred from another domain
    Transfer,
}

/// Rule metadata for explanation
#[derive(Debug, Clone)]
pub struct RuleMetadata {
    /// Rule name
    pub rule_name: String,
    /// Generation method
    pub generation_method: GenerationMethod,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Number of supporting examples
    pub support_count: usize,
    /// Feature importance scores
    pub feature_importance: HashMap<String, f64>,
    /// Generation description
    pub description: String,
    /// Parent rules (if derived)
    pub parent_rules: Vec<String>,
    /// Timestamp
    pub timestamp: std::time::SystemTime,
}

/// Feature contribution
#[derive(Debug, Clone)]
pub struct FeatureContribution {
    /// Feature name
    pub feature: String,
    /// Importance score
    pub importance: f64,
    /// Description
    pub description: String,
}

/// Explainable rule generator
pub struct ExplainableGenerator {
    /// Registered rules
    rules: HashMap<String, Rule>,
    /// Rule metadata
    metadata: HashMap<String, RuleMetadata>,
    /// Explanation cache
    explanation_cache: HashMap<(String, ExplanationType), String>,
}

impl Default for ExplainableGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ExplainableGenerator {
    /// Create a new explainable generator
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            metadata: HashMap::new(),
            explanation_cache: HashMap::new(),
        }
    }

    /// Register a rule with simple description
    pub fn register_rule(&mut self, rule: Rule, description: impl Into<String>) {
        let name = rule.name.clone();
        self.rules.insert(name.clone(), rule);

        self.metadata.insert(
            name.clone(),
            RuleMetadata {
                rule_name: name.clone(),
                generation_method: GenerationMethod::Manual,
                confidence: 1.0,
                support_count: 0,
                feature_importance: HashMap::new(),
                description: description.into(),
                parent_rules: Vec::new(),
                timestamp: std::time::SystemTime::now(),
            },
        );

        info!("Registered rule '{}' with explanation", name);
    }

    /// Register a rule with full metadata
    pub fn register_rule_with_metadata(&mut self, rule: Rule, metadata: RuleMetadata) {
        let name = rule.name.clone();
        self.rules.insert(name.clone(), rule);
        self.metadata.insert(name.clone(), metadata);

        // Clear explanation cache for this rule
        self.explanation_cache
            .retain(|(rule_name, _), _| rule_name != &name);
    }

    /// Get explanation for a rule
    pub fn explain(
        &mut self,
        rule_name: &str,
        explanation_type: ExplanationType,
    ) -> Option<String> {
        // Check cache first
        let cache_key = (rule_name.to_string(), explanation_type);
        if let Some(cached) = self.explanation_cache.get(&cache_key) {
            return Some(cached.clone());
        }

        let rule = self.rules.get(rule_name)?;
        let metadata = self.metadata.get(rule_name)?;

        let explanation = match explanation_type {
            ExplanationType::NaturalLanguage => {
                self.generate_natural_language_explanation(rule, metadata)
            }
            ExplanationType::FeatureImportance => {
                self.generate_feature_importance_explanation(metadata)
            }
            ExplanationType::ConfidenceAnalysis => self.generate_confidence_explanation(metadata),
            ExplanationType::Counterfactual => {
                self.generate_counterfactual_explanation(rule, metadata)
            }
            ExplanationType::Provenance => self.generate_provenance_explanation(metadata),
        };

        // Cache the explanation
        self.explanation_cache
            .insert(cache_key, explanation.clone());

        Some(explanation)
    }

    /// Generate natural language explanation
    fn generate_natural_language_explanation(
        &self,
        rule: &Rule,
        metadata: &RuleMetadata,
    ) -> String {
        let mut explanation = format!("Rule '{}' ", rule.name);

        match metadata.generation_method {
            GenerationMethod::MachineLearning => {
                explanation.push_str(&format!(
                    "was learned from {} examples with {:.1}% confidence. ",
                    metadata.support_count,
                    metadata.confidence * 100.0
                ));
            }
            GenerationMethod::Derivation => {
                explanation.push_str("was derived from existing rules. ");
                if !metadata.parent_rules.is_empty() {
                    explanation.push_str(&format!(
                        "Parent rules: {}. ",
                        metadata.parent_rules.join(", ")
                    ));
                }
            }
            GenerationMethod::Manual => {
                explanation.push_str("was manually authored. ");
            }
            GenerationMethod::Synthesis => {
                explanation.push_str("was automatically synthesized. ");
            }
            GenerationMethod::Transfer => {
                explanation.push_str("was transferred from another domain. ");
            }
        }

        explanation.push_str(&format!("\n\n{}", metadata.description));

        if rule.body.len() == 1 {
            explanation.push_str("\n\nThis rule has 1 condition.");
        } else {
            explanation.push_str(&format!(
                "\n\nThis rule has {} conditions.",
                rule.body.len()
            ));
        }

        if rule.head.len() == 1 {
            explanation.push_str(" It produces 1 conclusion.");
        } else {
            explanation.push_str(&format!(" It produces {} conclusions.", rule.head.len()));
        }

        explanation
    }

    /// Generate feature importance explanation
    fn generate_feature_importance_explanation(&self, metadata: &RuleMetadata) -> String {
        let mut explanation = format!(
            "Feature Importance Analysis for Rule '{}'\n\n",
            metadata.rule_name
        );

        if metadata.feature_importance.is_empty() {
            explanation.push_str("No feature importance data available.");
            return explanation;
        }

        let mut features: Vec<_> = metadata.feature_importance.iter().collect();
        features.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));

        explanation.push_str("Top contributing features:\n");
        for (idx, (feature, importance)) in features.iter().take(5).enumerate() {
            explanation.push_str(&format!(
                "{}. {}: {:.1}%\n",
                idx + 1,
                feature,
                **importance * 100.0
            ));
        }

        explanation
    }

    /// Generate confidence explanation
    fn generate_confidence_explanation(&self, metadata: &RuleMetadata) -> String {
        let mut explanation = format!("Confidence Analysis for Rule '{}'\n\n", metadata.rule_name);

        explanation.push_str(&format!(
            "Overall Confidence: {:.1}%\n\n",
            metadata.confidence * 100.0
        ));

        explanation.push_str(&format!(
            "Based on {} supporting examples.\n\n",
            metadata.support_count
        ));

        let confidence_level = if metadata.confidence >= 0.9 {
            "Very High"
        } else if metadata.confidence >= 0.75 {
            "High"
        } else if metadata.confidence >= 0.6 {
            "Moderate"
        } else if metadata.confidence >= 0.4 {
            "Low"
        } else {
            "Very Low"
        };

        explanation.push_str(&format!("Confidence Level: {}\n\n", confidence_level));

        if metadata.confidence < 0.6 {
            explanation.push_str(
                "⚠️  Note: This rule has low confidence and may need additional validation.\n",
            );
        }

        explanation
    }

    /// Generate counterfactual explanation
    fn generate_counterfactual_explanation(&self, rule: &Rule, metadata: &RuleMetadata) -> String {
        let mut explanation = format!("Counterfactual Analysis for Rule '{}'\n\n", rule.name);

        explanation.push_str("What would change this rule:\n\n");

        explanation.push_str("1. If the rule body had fewer conditions, it would match more cases but might be less specific.\n");
        explanation.push_str("2. If the rule body had more conditions, it would match fewer cases but be more specific.\n");
        explanation.push_str("3. If the confidence threshold were lowered, this rule might be accepted even with less evidence.\n");

        if metadata.support_count < 100 {
            explanation.push_str(&format!(
                "4. With more supporting examples (currently {}), the confidence could increase.\n",
                metadata.support_count
            ));
        }

        explanation
    }

    /// Generate provenance explanation
    fn generate_provenance_explanation(&self, metadata: &RuleMetadata) -> String {
        let mut explanation = format!("Provenance Trace for Rule '{}'\n\n", metadata.rule_name);

        explanation.push_str(&format!(
            "Generation Method: {:?}\n",
            metadata.generation_method
        ));
        explanation.push_str(&format!(
            "Created: {:?}\n",
            metadata
                .timestamp
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        ));
        explanation.push_str(&format!("Support Count: {}\n", metadata.support_count));
        explanation.push_str(&format!("Confidence: {:.3}\n\n", metadata.confidence));

        if !metadata.parent_rules.is_empty() {
            explanation.push_str("Derived from rules:\n");
            for parent in &metadata.parent_rules {
                explanation.push_str(&format!("  - {}\n", parent));
            }
        }

        explanation.push_str(&format!("\n{}", metadata.description));

        explanation
    }

    /// Get feature contributions
    pub fn get_feature_contributions(&self, rule_name: &str) -> Vec<FeatureContribution> {
        let metadata = match self.metadata.get(rule_name) {
            Some(m) => m,
            None => return Vec::new(),
        };

        let mut contributions: Vec<_> = metadata
            .feature_importance
            .iter()
            .map(|(feature, importance)| FeatureContribution {
                feature: feature.clone(),
                importance: *importance,
                description: format!("Contributes {:.1}% to rule generation", importance * 100.0),
            })
            .collect();

        contributions.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        contributions
    }

    /// Get rule metadata
    pub fn get_metadata(&self, rule_name: &str) -> Option<&RuleMetadata> {
        self.metadata.get(rule_name)
    }

    /// Get all registered rules
    pub fn get_rules(&self) -> Vec<String> {
        self.rules.keys().cloned().collect()
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.explanation_cache.clear();
    }

    /// Remove rule
    pub fn remove_rule(&mut self, rule_name: &str) -> bool {
        let removed = self.rules.remove(rule_name).is_some();
        self.metadata.remove(rule_name);
        self.explanation_cache
            .retain(|(name, _), _| name != rule_name);
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RuleAtom, Term};

    fn create_test_rule(name: &str) -> Rule {
        Rule {
            name: name.to_string(),
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
        }
    }

    #[test]
    fn test_generator_creation() {
        let generator = ExplainableGenerator::new();
        assert_eq!(generator.get_rules().len(), 0);
    }

    #[test]
    fn test_register_rule() {
        let mut generator = ExplainableGenerator::new();
        let rule = create_test_rule("test_rule");

        generator.register_rule(rule, "Test description");

        assert_eq!(generator.get_rules().len(), 1);
        assert!(generator.get_rules().contains(&"test_rule".to_string()));
    }

    #[test]
    fn test_natural_language_explanation() {
        let mut generator = ExplainableGenerator::new();
        let rule = create_test_rule("test_rule");

        generator.register_rule(rule, "A simple test rule");

        let explanation = generator.explain("test_rule", ExplanationType::NaturalLanguage);
        assert!(explanation.is_some());

        let exp = explanation.unwrap();
        assert!(exp.contains("test_rule"));
        assert!(exp.contains("manually authored"));
    }

    #[test]
    fn test_feature_importance_explanation() {
        let mut generator = ExplainableGenerator::new();
        let rule = create_test_rule("test_rule");

        let mut feature_importance = HashMap::new();
        feature_importance.insert("feature1".to_string(), 0.6);
        feature_importance.insert("feature2".to_string(), 0.3);
        feature_importance.insert("feature3".to_string(), 0.1);

        let metadata = RuleMetadata {
            rule_name: "test_rule".to_string(),
            generation_method: GenerationMethod::MachineLearning,
            confidence: 0.85,
            support_count: 100,
            feature_importance,
            description: "Learned rule".to_string(),
            parent_rules: Vec::new(),
            timestamp: std::time::SystemTime::now(),
        };

        generator.register_rule_with_metadata(rule, metadata);

        let explanation = generator.explain("test_rule", ExplanationType::FeatureImportance);
        assert!(explanation.is_some());

        let exp = explanation.unwrap();
        assert!(exp.contains("feature1"));
        assert!(exp.contains("60.0%"));
    }

    #[test]
    fn test_confidence_explanation() {
        let mut generator = ExplainableGenerator::new();
        let rule = create_test_rule("high_conf_rule");

        let metadata = RuleMetadata {
            rule_name: "high_conf_rule".to_string(),
            generation_method: GenerationMethod::MachineLearning,
            confidence: 0.92,
            support_count: 500,
            feature_importance: HashMap::new(),
            description: "High confidence rule".to_string(),
            parent_rules: Vec::new(),
            timestamp: std::time::SystemTime::now(),
        };

        generator.register_rule_with_metadata(rule, metadata);

        let explanation = generator.explain("high_conf_rule", ExplanationType::ConfidenceAnalysis);
        assert!(explanation.is_some());

        let exp = explanation.unwrap();
        assert!(exp.contains("92.0%"));
        assert!(exp.contains("500"));
        assert!(exp.contains("Very High"));
    }

    #[test]
    fn test_provenance_explanation() {
        let mut generator = ExplainableGenerator::new();
        let rule = create_test_rule("derived_rule");

        let metadata = RuleMetadata {
            rule_name: "derived_rule".to_string(),
            generation_method: GenerationMethod::Derivation,
            confidence: 0.88,
            support_count: 200,
            feature_importance: HashMap::new(),
            description: "Derived from parent rules".to_string(),
            parent_rules: vec!["parent1".to_string(), "parent2".to_string()],
            timestamp: std::time::SystemTime::now(),
        };

        generator.register_rule_with_metadata(rule, metadata);

        let explanation = generator.explain("derived_rule", ExplanationType::Provenance);
        assert!(explanation.is_some());

        let exp = explanation.unwrap();
        assert!(exp.contains("Derivation"));
        assert!(exp.contains("parent1"));
        assert!(exp.contains("parent2"));
    }

    #[test]
    fn test_get_feature_contributions() {
        let mut generator = ExplainableGenerator::new();
        let rule = create_test_rule("test_rule");

        let mut feature_importance = HashMap::new();
        feature_importance.insert("high_importance".to_string(), 0.8);
        feature_importance.insert("low_importance".to_string(), 0.2);

        let metadata = RuleMetadata {
            rule_name: "test_rule".to_string(),
            generation_method: GenerationMethod::MachineLearning,
            confidence: 0.85,
            support_count: 100,
            feature_importance,
            description: "Test".to_string(),
            parent_rules: Vec::new(),
            timestamp: std::time::SystemTime::now(),
        };

        generator.register_rule_with_metadata(rule, metadata);

        let contributions = generator.get_feature_contributions("test_rule");
        assert_eq!(contributions.len(), 2);
        assert_eq!(contributions[0].feature, "high_importance");
        assert!((contributions[0].importance - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_explanation_caching() {
        let mut generator = ExplainableGenerator::new();
        let rule = create_test_rule("cached_rule");

        generator.register_rule(rule, "Cache test");

        // First call - generates explanation
        let exp1 = generator.explain("cached_rule", ExplanationType::NaturalLanguage);

        // Second call - should use cache
        let exp2 = generator.explain("cached_rule", ExplanationType::NaturalLanguage);

        assert_eq!(exp1, exp2);
    }

    #[test]
    fn test_remove_rule() {
        let mut generator = ExplainableGenerator::new();
        let rule = create_test_rule("removable_rule");

        generator.register_rule(rule, "To be removed");
        assert_eq!(generator.get_rules().len(), 1);

        let removed = generator.remove_rule("removable_rule");
        assert!(removed);
        assert_eq!(generator.get_rules().len(), 0);
    }

    #[test]
    fn test_clear_cache() {
        let mut generator = ExplainableGenerator::new();
        let rule = create_test_rule("test_rule");

        generator.register_rule(rule, "Test");

        // Generate explanation to populate cache
        generator.explain("test_rule", ExplanationType::NaturalLanguage);
        assert!(!generator.explanation_cache.is_empty());

        generator.clear_cache();
        assert!(generator.explanation_cache.is_empty());
    }
}
