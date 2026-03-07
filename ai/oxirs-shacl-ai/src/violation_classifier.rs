//! ML-style SHACL violation severity classification.
//!
//! Classifies SHACL violations into semantic categories using a rule-based
//! engine.  Rules are ordered by priority; the first matching rule wins.

// ---------------------------------------------------------------------------
// ViolationClass
// ---------------------------------------------------------------------------

/// Semantic category of a SHACL violation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ViolationClass {
    /// Data is missing, malformed, or outside valid bounds in a critical way.
    CriticalDataQuality,
    /// Minor data quality issue (e.g., pattern mismatch).
    MinorDataQuality,
    /// The graph structure does not conform to the shape (e.g., missing node).
    StructuralError,
    /// Semantically incorrect value (e.g., wrong class or property).
    SemanticError,
    /// Violation of an organizational or business policy.
    PolicyViolation,
}

impl std::fmt::Display for ViolationClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CriticalDataQuality => write!(f, "CriticalDataQuality"),
            Self::MinorDataQuality => write!(f, "MinorDataQuality"),
            Self::StructuralError => write!(f, "StructuralError"),
            Self::SemanticError => write!(f, "SemanticError"),
            Self::PolicyViolation => write!(f, "PolicyViolation"),
        }
    }
}

// ---------------------------------------------------------------------------
// ViolationFeatures
// ---------------------------------------------------------------------------

/// Feature vector extracted from a SHACL violation report.
#[derive(Debug, Clone)]
pub struct ViolationFeatures {
    /// Type of the violated SHACL constraint (e.g. `"minCount"`, `"datatype"`).
    pub constraint_type: String,
    /// Property path of the constraint, if any.
    pub property_path: Option<String>,
    /// Number of values observed.
    pub value_count: usize,
    /// Whether the focal node has a value at all.
    pub has_value: bool,
    /// A keyword extracted from the `sh:message` (e.g. `"required"`, `"invalid"`).
    pub message_keyword: String,
}

// ---------------------------------------------------------------------------
// ClassificationResult
// ---------------------------------------------------------------------------

/// Result of classifying one SHACL violation.
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    /// The assigned class.
    pub class: ViolationClass,
    /// Confidence in `[0.0, 1.0]`.
    pub confidence: f64,
    /// Human-readable explanation of why this class was chosen.
    pub reasoning: String,
}

// ---------------------------------------------------------------------------
// ClassificationRule
// ---------------------------------------------------------------------------

/// A single rule in the classifier's rule-base.
#[derive(Debug, Clone)]
pub struct ClassificationRule {
    /// Unique rule name.
    pub name: String,
    /// Constraint types this rule matches (case-insensitive, prefix match).
    pub constraint_types: Vec<String>,
    /// Class to assign when this rule matches.
    pub class: ViolationClass,
    /// Confidence to assign when this rule matches.
    pub confidence: f64,
}

impl ClassificationRule {
    /// `true` when the rule fires for the given `features`.
    fn matches(&self, features: &ViolationFeatures) -> bool {
        let ct_lower = features.constraint_type.to_lowercase();
        self.constraint_types
            .iter()
            .any(|t| ct_lower.starts_with(&t.to_lowercase()))
    }
}

// ---------------------------------------------------------------------------
// ViolationClassifier
// ---------------------------------------------------------------------------

/// Rule-based SHACL violation severity classifier.
pub struct ViolationClassifier {
    rules: Vec<ClassificationRule>,
}

impl Default for ViolationClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ViolationClassifier {
    /// Create an empty classifier (no rules).
    pub fn new() -> Self {
        ViolationClassifier { rules: Vec::new() }
    }

    /// Create a classifier pre-loaded with default rules covering the most
    /// common SHACL constraint types.
    pub fn with_defaults() -> Self {
        let mut c = Self::new();

        // Structural rules
        c.add_rule(ClassificationRule {
            name: "minCount".into(),
            constraint_types: vec!["mincount".into()],
            class: ViolationClass::StructuralError,
            confidence: 0.90,
        });
        c.add_rule(ClassificationRule {
            name: "maxCount".into(),
            constraint_types: vec!["maxcount".into()],
            class: ViolationClass::StructuralError,
            confidence: 0.85,
        });
        c.add_rule(ClassificationRule {
            name: "nodeKind".into(),
            constraint_types: vec!["nodekind".into()],
            class: ViolationClass::StructuralError,
            confidence: 0.80,
        });

        // Data quality rules
        c.add_rule(ClassificationRule {
            name: "datatype".into(),
            constraint_types: vec!["datatype".into()],
            class: ViolationClass::MinorDataQuality,
            confidence: 0.88,
        });
        c.add_rule(ClassificationRule {
            name: "pattern".into(),
            constraint_types: vec!["pattern".into()],
            class: ViolationClass::MinorDataQuality,
            confidence: 0.82,
        });
        c.add_rule(ClassificationRule {
            name: "minLength".into(),
            constraint_types: vec!["minlength".into(), "maxlength".into()],
            class: ViolationClass::MinorDataQuality,
            confidence: 0.75,
        });
        c.add_rule(ClassificationRule {
            name: "minInclusive".into(),
            constraint_types: vec![
                "mininclusive".into(),
                "maxinclusive".into(),
                "minexclusive".into(),
                "maxexclusive".into(),
            ],
            class: ViolationClass::CriticalDataQuality,
            confidence: 0.87,
        });

        // Semantic rules
        c.add_rule(ClassificationRule {
            name: "class".into(),
            constraint_types: vec!["class".into()],
            class: ViolationClass::SemanticError,
            confidence: 0.85,
        });
        c.add_rule(ClassificationRule {
            name: "in".into(),
            constraint_types: vec!["in".into()],
            class: ViolationClass::SemanticError,
            confidence: 0.80,
        });
        c.add_rule(ClassificationRule {
            name: "hasValue".into(),
            constraint_types: vec!["hasvalue".into()],
            class: ViolationClass::SemanticError,
            confidence: 0.78,
        });

        // Semantic rules (continued)
        c.add_rule(ClassificationRule {
            name: "uniqueLang".into(),
            constraint_types: vec!["uniquelang".into()],
            class: ViolationClass::SemanticError,
            confidence: 0.76,
        });

        // Policy rules
        c.add_rule(ClassificationRule {
            name: "sparqlConstraint".into(),
            constraint_types: vec!["sparql".into()],
            class: ViolationClass::PolicyViolation,
            confidence: 0.70,
        });

        c
    }

    /// Append a rule to the rule base.
    pub fn add_rule(&mut self, rule: ClassificationRule) {
        self.rules.push(rule);
    }

    /// Classify a single violation.
    ///
    /// Tries each rule in order; the first matching rule wins.
    /// Falls back to `MinorDataQuality` with low confidence when no rule matches.
    pub fn classify(&self, features: &ViolationFeatures) -> ClassificationResult {
        for rule in &self.rules {
            if rule.matches(features) {
                return ClassificationResult {
                    class: rule.class.clone(),
                    confidence: rule.confidence.clamp(0.0, 1.0),
                    reasoning: format!(
                        "Rule '{}' matched constraint type '{}'",
                        rule.name, features.constraint_type
                    ),
                };
            }
        }
        // Default fallback
        ClassificationResult {
            class: ViolationClass::MinorDataQuality,
            confidence: 0.40,
            reasoning: format!(
                "No rule matched constraint type '{}'; defaulting to MinorDataQuality",
                features.constraint_type
            ),
        }
    }

    /// Classify a batch of violations.
    pub fn classify_batch(&self, features: &[ViolationFeatures]) -> Vec<ClassificationResult> {
        features.iter().map(|f| self.classify(f)).collect()
    }

    /// Number of rules in the classifier.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Return the most common `ViolationClass` in a slice of results,
    /// or `None` if `results` is empty.
    pub fn top_class(results: &[ClassificationResult]) -> Option<ViolationClass> {
        if results.is_empty() {
            return None;
        }
        let mut counts: std::collections::HashMap<String, (ViolationClass, usize)> =
            std::collections::HashMap::new();
        for r in results {
            let key = r.class.to_string();
            counts
                .entry(key)
                .and_modify(|(_, c)| *c += 1)
                .or_insert_with(|| (r.class.clone(), 1));
        }
        counts
            .into_values()
            .max_by_key(|(_, count)| *count)
            .map(|(class, _)| class)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn feat(constraint_type: &str) -> ViolationFeatures {
        ViolationFeatures {
            constraint_type: constraint_type.to_string(),
            property_path: None,
            value_count: 0,
            has_value: false,
            message_keyword: String::new(),
        }
    }

    // --- ViolationClass variants ---

    #[test]
    fn test_all_violation_class_variants_reachable() {
        let variants = [
            ViolationClass::CriticalDataQuality,
            ViolationClass::MinorDataQuality,
            ViolationClass::StructuralError,
            ViolationClass::SemanticError,
            ViolationClass::PolicyViolation,
        ];
        assert_eq!(variants.len(), 5);
    }

    #[test]
    fn test_violation_class_display() {
        assert_eq!(
            ViolationClass::StructuralError.to_string(),
            "StructuralError"
        );
        assert_eq!(
            ViolationClass::PolicyViolation.to_string(),
            "PolicyViolation"
        );
    }

    // --- with_defaults has rules ---

    #[test]
    fn test_with_defaults_has_rules() {
        let c = ViolationClassifier::with_defaults();
        assert!(c.rule_count() > 0);
    }

    #[test]
    fn test_with_defaults_rule_count_is_expected() {
        let c = ViolationClassifier::with_defaults();
        assert_eq!(c.rule_count(), 12);
    }

    // --- classify: minCount → StructuralError ---

    #[test]
    fn test_classify_mincount_structural_error() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("minCount"));
        assert_eq!(result.class, ViolationClass::StructuralError);
    }

    #[test]
    fn test_classify_maxcount_structural_error() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("maxCount"));
        assert_eq!(result.class, ViolationClass::StructuralError);
    }

    // --- classify: datatype → MinorDataQuality ---

    #[test]
    fn test_classify_datatype_minor_data_quality() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("datatype"));
        assert_eq!(result.class, ViolationClass::MinorDataQuality);
    }

    // --- classify: pattern → MinorDataQuality ---

    #[test]
    fn test_classify_pattern_minor_data_quality() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("pattern"));
        assert_eq!(result.class, ViolationClass::MinorDataQuality);
    }

    // --- classify: class → SemanticError ---

    #[test]
    fn test_classify_class_constraint_semantic_error() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("class"));
        assert_eq!(result.class, ViolationClass::SemanticError);
    }

    // --- classify: in → SemanticError ---

    #[test]
    fn test_classify_in_constraint_semantic_error() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("in"));
        assert_eq!(result.class, ViolationClass::SemanticError);
    }

    // --- classify: sparql → PolicyViolation ---

    #[test]
    fn test_classify_sparql_policy_violation() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("sparqlConstraint"));
        assert_eq!(result.class, ViolationClass::PolicyViolation);
    }

    // --- classify: minInclusive → CriticalDataQuality ---

    #[test]
    fn test_classify_min_inclusive_critical_data_quality() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("minInclusive"));
        assert_eq!(result.class, ViolationClass::CriticalDataQuality);
    }

    // --- confidence range ---

    #[test]
    fn test_classify_confidence_in_range() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("minCount"));
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    // --- no matching rule → default ---

    #[test]
    fn test_classify_no_matching_rule_default() {
        let c = ViolationClassifier::new(); // empty rule base
        let result = c.classify(&feat("unknownConstraint"));
        assert_eq!(result.class, ViolationClass::MinorDataQuality);
        assert!(result.confidence < 0.5);
    }

    // --- classify_batch ---

    #[test]
    fn test_classify_batch_length_matches_input() {
        let c = ViolationClassifier::with_defaults();
        let features = vec![feat("minCount"), feat("datatype"), feat("pattern")];
        let results = c.classify_batch(&features);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_classify_batch_correct_classes() {
        let c = ViolationClassifier::with_defaults();
        let features = vec![feat("minCount"), feat("class")];
        let results = c.classify_batch(&features);
        assert_eq!(results[0].class, ViolationClass::StructuralError);
        assert_eq!(results[1].class, ViolationClass::SemanticError);
    }

    // --- rule_count ---

    #[test]
    fn test_rule_count_empty() {
        let c = ViolationClassifier::new();
        assert_eq!(c.rule_count(), 0);
    }

    #[test]
    fn test_rule_count_after_add() {
        let mut c = ViolationClassifier::new();
        c.add_rule(ClassificationRule {
            name: "test".into(),
            constraint_types: vec!["foo".into()],
            class: ViolationClass::PolicyViolation,
            confidence: 0.5,
        });
        assert_eq!(c.rule_count(), 1);
    }

    // --- add_rule ---

    #[test]
    fn test_add_rule_is_used_in_classify() {
        let mut c = ViolationClassifier::new();
        c.add_rule(ClassificationRule {
            name: "custom".into(),
            constraint_types: vec!["myConstraint".into()],
            class: ViolationClass::PolicyViolation,
            confidence: 0.95,
        });
        let result = c.classify(&feat("myConstraint"));
        assert_eq!(result.class, ViolationClass::PolicyViolation);
    }

    // --- top_class ---

    #[test]
    fn test_top_class_most_frequent() {
        let results = vec![
            ClassificationResult {
                class: ViolationClass::StructuralError,
                confidence: 0.9,
                reasoning: String::new(),
            },
            ClassificationResult {
                class: ViolationClass::StructuralError,
                confidence: 0.8,
                reasoning: String::new(),
            },
            ClassificationResult {
                class: ViolationClass::SemanticError,
                confidence: 0.7,
                reasoning: String::new(),
            },
        ];
        let top = ViolationClassifier::top_class(&results).expect("should have top class");
        assert_eq!(top, ViolationClass::StructuralError);
    }

    #[test]
    fn test_top_class_empty_returns_none() {
        assert!(ViolationClassifier::top_class(&[]).is_none());
    }

    #[test]
    fn test_top_class_single_element() {
        let results = vec![ClassificationResult {
            class: ViolationClass::PolicyViolation,
            confidence: 0.6,
            reasoning: String::new(),
        }];
        let top = ViolationClassifier::top_class(&results).expect("should have top class");
        assert_eq!(top, ViolationClass::PolicyViolation);
    }

    // --- reasoning field ---

    #[test]
    fn test_classify_reasoning_non_empty() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("minCount"));
        assert!(!result.reasoning.is_empty());
    }

    // --- Case insensitivity ---

    #[test]
    fn test_classify_case_insensitive_constraint_type() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("MINCOUNT"));
        assert_eq!(result.class, ViolationClass::StructuralError);
    }

    // --- default() ---

    #[test]
    fn test_default_is_same_as_new() {
        let c = ViolationClassifier::default();
        assert_eq!(c.rule_count(), 0);
    }

    // --- hasValue ---

    #[test]
    fn test_classify_hasvalue_semantic_error() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("hasValue"));
        assert_eq!(result.class, ViolationClass::SemanticError);
    }

    // --- nodeKind ---

    #[test]
    fn test_classify_nodekind_structural_error() {
        let c = ViolationClassifier::with_defaults();
        let result = c.classify(&feat("nodeKind"));
        assert_eq!(result.class, ViolationClass::StructuralError);
    }
}
