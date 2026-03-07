//! # SHACL Rule Generator
//!
//! Generates SHACL shape rules from observed RDF data patterns.
//!
//! The generator analyses collections of [`ObservedPattern`] records (each
//! describing how often a subject of a given class uses a given predicate),
//! and emits [`GeneratedRule`] structures that encode SHACL constraints such as
//! `sh:minCount`, `sh:maxCount`, `sh:nodeKind`, `sh:datatype`, and `sh:class`.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_shacl_ai::rule_generator::{
//!     ObservedPattern, RuleConfig, RuleGenerator,
//! };
//!
//! let config = RuleConfig {
//!     min_frequency_ratio: 0.5,
//!     min_confidence: 0.5,
//!     generate_counts: true,
//! };
//! let mut gen = RuleGenerator::new(config);
//!
//! gen.add_observation(ObservedPattern {
//!     subject_type: "Person".to_string(),
//!     predicate: "ex:name".to_string(),
//!     object_type: "xsd:string".to_string(),
//!     frequency: 9,
//!     total_subjects: 10,
//! });
//!
//! let rules = gen.generate_rules();
//! assert!(!rules.is_empty());
//! ```

use std::collections::HashMap;

// ─── Observable input ─────────────────────────────────────────────────────────

/// A single observation: how often subjects of a given type use a predicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedPattern {
    /// The RDF class (or type label) of the subject.
    pub subject_type: String,
    /// The predicate IRI or local name observed.
    pub predicate: String,
    /// The range type/datatype of the object.
    pub object_type: String,
    /// Number of subjects of `subject_type` that used `predicate`.
    pub frequency: usize,
    /// Total number of distinct subjects of `subject_type`.
    pub total_subjects: usize,
}

impl ObservedPattern {
    /// Compute the frequency ratio (frequency / total_subjects).
    ///
    /// Returns 0.0 if `total_subjects` is 0.
    pub fn frequency_ratio(&self) -> f64 {
        if self.total_subjects == 0 {
            0.0
        } else {
            self.frequency as f64 / self.total_subjects as f64
        }
    }
}

// ─── Rule output ─────────────────────────────────────────────────────────────

/// The kind of SHACL constraint to be generated.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RuleType {
    /// `sh:minCount` — property must appear at least N times.
    MinCount,
    /// `sh:maxCount` — property may appear at most N times.
    MaxCount,
    /// `sh:minCount == sh:maxCount` — property must appear exactly N times.
    ExactCount,
    /// `sh:nodeKind` — restricts the kind of the value node (IRI / Literal / …).
    NodeKind,
    /// `sh:datatype` — restricts the datatype of a literal value.
    Datatype,
    /// `sh:class` — restricts the class of the value node.
    Class,
}

impl RuleType {
    /// Human-readable SHACL keyword for this constraint type.
    pub fn shacl_keyword(&self) -> &'static str {
        match self {
            Self::MinCount => "sh:minCount",
            Self::MaxCount => "sh:maxCount",
            Self::ExactCount => "sh:minCount/sh:maxCount",
            Self::NodeKind => "sh:nodeKind",
            Self::Datatype => "sh:datatype",
            Self::Class => "sh:class",
        }
    }
}

/// A SHACL rule derived from observed data patterns.
#[derive(Debug, Clone)]
pub struct GeneratedRule {
    /// Target class URI / label for `sh:targetClass`.
    pub target_class: String,
    /// Predicate IRI or local name (used in `sh:path`).
    pub predicate: String,
    /// The kind of constraint.
    pub rule_type: RuleType,
    /// Confidence in the rule (0.0 – 1.0): derived from frequency ratio.
    pub confidence: f64,
    /// Optional lower count bound (for `MinCount` / `ExactCount`).
    pub min_count: Option<usize>,
    /// Optional upper count bound (for `MaxCount` / `ExactCount`).
    pub max_count: Option<usize>,
}

impl GeneratedRule {
    /// Create a rule with the given fields.
    pub fn new(
        target_class: impl Into<String>,
        predicate: impl Into<String>,
        rule_type: RuleType,
        confidence: f64,
        min_count: Option<usize>,
        max_count: Option<usize>,
    ) -> Self {
        Self {
            target_class: target_class.into(),
            predicate: predicate.into(),
            rule_type,
            confidence,
            min_count,
            max_count,
        }
    }

    /// Return true if this rule targets a count-based constraint.
    pub fn is_count_rule(&self) -> bool {
        matches!(
            self.rule_type,
            RuleType::MinCount | RuleType::MaxCount | RuleType::ExactCount
        )
    }
}

// ─── Generator configuration ─────────────────────────────────────────────────

/// Configuration that controls which rules are emitted.
#[derive(Debug, Clone)]
pub struct RuleConfig {
    /// Minimum frequency ratio required for a predicate to generate rules.
    ///
    /// A value of 0.5 means the predicate must appear in at least 50 % of subjects.
    pub min_frequency_ratio: f64,

    /// Minimum confidence score required for a rule to be included.
    pub min_confidence: f64,

    /// Whether to emit `MinCount`/`MaxCount` rules alongside type-based rules.
    pub generate_counts: bool,
}

impl Default for RuleConfig {
    fn default() -> Self {
        Self {
            min_frequency_ratio: 0.5,
            min_confidence: 0.5,
            generate_counts: true,
        }
    }
}

// ─── Generator ───────────────────────────────────────────────────────────────

/// Aggregation key for grouping observations by (subject_type, predicate).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PatternKey {
    subject_type: String,
    predicate: String,
}

/// Accumulated statistics for a (subject_type, predicate) pair.
#[derive(Debug, Default)]
struct PatternStats {
    /// Total frequency across all observations for this key.
    total_frequency: usize,
    /// The maximum `total_subjects` seen for this pair (used for ratio).
    max_total_subjects: usize,
    /// All unique object types observed for this predicate.
    object_types: Vec<String>,
}

/// Generates SHACL rules from accumulated [`ObservedPattern`] data.
pub struct RuleGenerator {
    config: RuleConfig,
    observations: Vec<ObservedPattern>,
}

impl RuleGenerator {
    /// Create a new generator with the given configuration.
    pub fn new(config: RuleConfig) -> Self {
        Self {
            config,
            observations: Vec::new(),
        }
    }

    /// Record an observed predicate usage pattern.
    pub fn add_observation(&mut self, pattern: ObservedPattern) {
        self.observations.push(pattern);
    }

    /// Generate SHACL rules from all recorded observations.
    ///
    /// For each (subject_type, predicate) pair whose combined frequency ratio
    /// meets `min_frequency_ratio`, this method emits:
    ///
    /// - A `MinCount` rule if `generate_counts` is true.
    /// - A `MaxCount` rule if `generate_counts` is true.
    /// - An appropriate type rule (`Datatype`, `Class`, or `NodeKind`) based on
    ///   the object type strings observed.
    ///
    /// All emitted rules satisfy `confidence >= min_confidence`.
    pub fn generate_rules(&self) -> Vec<GeneratedRule> {
        // Aggregate observations by (subject_type, predicate).
        let mut stats_map: HashMap<PatternKey, PatternStats> = HashMap::new();

        for obs in &self.observations {
            let key = PatternKey {
                subject_type: obs.subject_type.clone(),
                predicate: obs.predicate.clone(),
            };
            let entry = stats_map.entry(key).or_default();
            entry.total_frequency += obs.frequency;
            entry.max_total_subjects = entry.max_total_subjects.max(obs.total_subjects);
            if !entry.object_types.contains(&obs.object_type) {
                entry.object_types.push(obs.object_type.clone());
            }
        }

        let mut rules: Vec<GeneratedRule> = Vec::new();

        for (key, stats) in &stats_map {
            let ratio = if stats.max_total_subjects == 0 {
                0.0
            } else {
                stats.total_frequency as f64 / stats.max_total_subjects as f64
            };

            if ratio < self.config.min_frequency_ratio {
                continue;
            }

            let confidence = ratio.min(1.0);
            if confidence < self.config.min_confidence {
                continue;
            }

            // ── Count-based rules ────────────────────────────────────────────
            if self.config.generate_counts {
                // sh:minCount 1 — property must appear at least once
                rules.push(GeneratedRule::new(
                    &key.subject_type,
                    &key.predicate,
                    RuleType::MinCount,
                    confidence,
                    Some(1),
                    None,
                ));

                // sh:maxCount — conservative upper bound from observed frequency
                let max_count = (stats.total_frequency / stats.max_total_subjects.max(1)).max(1);
                rules.push(GeneratedRule::new(
                    &key.subject_type,
                    &key.predicate,
                    RuleType::MaxCount,
                    confidence,
                    None,
                    Some(max_count),
                ));
            }

            // ── Type-based rules ─────────────────────────────────────────────
            for obj_type in &stats.object_types {
                let rule_type = classify_object_type(obj_type);
                rules.push(GeneratedRule::new(
                    &key.subject_type,
                    &key.predicate,
                    rule_type,
                    confidence,
                    None,
                    None,
                ));
            }
        }

        rules
    }

    /// Generate all rules and then filter by `min` confidence.
    pub fn filter_by_confidence(&self, min: f64) -> Vec<GeneratedRule> {
        self.generate_rules()
            .into_iter()
            .filter(|r| r.confidence >= min)
            .collect()
    }

    /// Total number of recorded observations.
    pub fn rule_count(&self) -> usize {
        self.observations.len()
    }

    /// All recorded observations.
    pub fn observations(&self) -> &[ObservedPattern] {
        &self.observations
    }

    /// Clear all observations.
    pub fn clear(&mut self) {
        self.observations.clear();
    }
}

/// Classify an object type string into a SHACL constraint type.
///
/// Strings beginning with `"xsd:"` are treated as datatypes.
/// Strings beginning with lowercase `"sh:"` are treated as node-kind hints.
/// Everything else is treated as a class reference.
fn classify_object_type(obj_type: &str) -> RuleType {
    if obj_type.starts_with("xsd:") {
        RuleType::Datatype
    } else if obj_type.starts_with("sh:") {
        RuleType::NodeKind
    } else {
        RuleType::Class
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> RuleConfig {
        RuleConfig {
            min_frequency_ratio: 0.5,
            min_confidence: 0.5,
            generate_counts: true,
        }
    }

    fn make_pattern(
        subject_type: &str,
        predicate: &str,
        object_type: &str,
        frequency: usize,
        total: usize,
    ) -> ObservedPattern {
        ObservedPattern {
            subject_type: subject_type.to_string(),
            predicate: predicate.to_string(),
            object_type: object_type.to_string(),
            frequency,
            total_subjects: total,
        }
    }

    // ── basic construction ────────────────────────────────────────────────

    #[test]
    fn test_new_generator_empty() {
        let gen = RuleGenerator::new(default_config());
        assert_eq!(gen.rule_count(), 0);
        assert!(gen.generate_rules().is_empty());
    }

    #[test]
    fn test_add_observation_increments_count() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 8, 10));
        assert_eq!(gen.rule_count(), 1);
    }

    #[test]
    fn test_add_multiple_observations() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 8, 10));
        gen.add_observation(make_pattern("Person", "ex:age", "xsd:integer", 7, 10));
        assert_eq!(gen.rule_count(), 2);
    }

    // ── frequency ratio ───────────────────────────────────────────────────

    #[test]
    fn test_frequency_ratio_normal() {
        let p = make_pattern("T", "p", "xsd:string", 7, 10);
        assert!((p.frequency_ratio() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_frequency_ratio_zero_total() {
        let p = make_pattern("T", "p", "xsd:string", 0, 0);
        assert_eq!(p.frequency_ratio(), 0.0);
    }

    #[test]
    fn test_frequency_ratio_full() {
        let p = make_pattern("T", "p", "xsd:string", 10, 10);
        assert!((p.frequency_ratio() - 1.0).abs() < 1e-9);
    }

    // ── generate_rules: basic ─────────────────────────────────────────────

    #[test]
    fn test_generate_produces_min_count() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 9, 10));
        let rules = gen.generate_rules();
        let has_min = rules.iter().any(|r| r.rule_type == RuleType::MinCount);
        assert!(has_min);
    }

    #[test]
    fn test_generate_produces_max_count() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 9, 10));
        let rules = gen.generate_rules();
        let has_max = rules.iter().any(|r| r.rule_type == RuleType::MaxCount);
        assert!(has_max);
    }

    #[test]
    fn test_generate_produces_datatype_for_xsd() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 9, 10));
        let rules = gen.generate_rules();
        let has_datatype = rules.iter().any(|r| r.rule_type == RuleType::Datatype);
        assert!(has_datatype);
    }

    #[test]
    fn test_generate_produces_class_for_non_xsd() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:knows", "Person", 9, 10));
        let rules = gen.generate_rules();
        let has_class = rules.iter().any(|r| r.rule_type == RuleType::Class);
        assert!(has_class);
    }

    #[test]
    fn test_generate_produces_nodekind_for_sh_prefix() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:ref", "sh:IRI", 9, 10));
        let rules = gen.generate_rules();
        let has_nodekind = rules.iter().any(|r| r.rule_type == RuleType::NodeKind);
        assert!(has_nodekind);
    }

    // ── threshold effects ─────────────────────────────────────────────────

    #[test]
    fn test_below_frequency_threshold_no_rules() {
        let mut gen = RuleGenerator::new(RuleConfig {
            min_frequency_ratio: 0.8,
            ..default_config()
        });
        // ratio = 3/10 = 0.3 < 0.8
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 3, 10));
        assert!(gen.generate_rules().is_empty());
    }

    #[test]
    fn test_exactly_at_frequency_threshold_produces_rules() {
        let mut gen = RuleGenerator::new(RuleConfig {
            min_frequency_ratio: 0.5,
            min_confidence: 0.5,
            generate_counts: true,
        });
        // ratio = 5/10 = 0.5 == threshold -> should pass
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 5, 10));
        assert!(!gen.generate_rules().is_empty());
    }

    #[test]
    fn test_below_confidence_threshold_no_rules() {
        let mut gen = RuleGenerator::new(RuleConfig {
            min_frequency_ratio: 0.1,
            min_confidence: 0.9,
            generate_counts: true,
        });
        // ratio = 0.5 -> confidence = 0.5 < 0.9
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 5, 10));
        assert!(gen.generate_rules().is_empty());
    }

    // ── generate_counts = false ───────────────────────────────────────────

    #[test]
    fn test_no_count_rules_when_disabled() {
        let mut gen = RuleGenerator::new(RuleConfig {
            generate_counts: false,
            ..default_config()
        });
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 9, 10));
        let rules = gen.generate_rules();
        let has_count = rules
            .iter()
            .any(|r| r.rule_type == RuleType::MinCount || r.rule_type == RuleType::MaxCount);
        assert!(!has_count);
    }

    #[test]
    fn test_type_rules_still_generated_when_counts_disabled() {
        let mut gen = RuleGenerator::new(RuleConfig {
            generate_counts: false,
            ..default_config()
        });
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 9, 10));
        let rules = gen.generate_rules();
        assert!(!rules.is_empty());
        assert!(rules.iter().any(|r| r.rule_type == RuleType::Datatype));
    }

    // ── confidence values ─────────────────────────────────────────────────

    #[test]
    fn test_confidence_equals_frequency_ratio() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 7, 10));
        let rules = gen.generate_rules();
        for rule in &rules {
            assert!(
                (rule.confidence - 0.7).abs() < 1e-9,
                "confidence mismatch: {}",
                rule.confidence
            );
        }
    }

    #[test]
    fn test_confidence_capped_at_one() {
        let mut gen = RuleGenerator::new(RuleConfig {
            min_frequency_ratio: 0.1,
            min_confidence: 0.1,
            generate_counts: true,
        });
        // frequency > total_subjects (edge case)
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 15, 10));
        let rules = gen.generate_rules();
        for rule in &rules {
            assert!(rule.confidence <= 1.0, "confidence should not exceed 1.0");
        }
    }

    // ── filter_by_confidence ─────────────────────────────────────────────

    #[test]
    fn test_filter_by_confidence_removes_low() {
        let mut gen = RuleGenerator::new(RuleConfig {
            min_frequency_ratio: 0.1,
            min_confidence: 0.1,
            generate_counts: true,
        });
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 5, 10)); // conf=0.5
        gen.add_observation(make_pattern("Person", "ex:age", "xsd:integer", 2, 10)); // conf=0.2
        let high_confidence = gen.filter_by_confidence(0.45);
        let low_target = high_confidence
            .iter()
            .filter(|r| r.predicate == "ex:age")
            .count();
        assert_eq!(low_target, 0, "low-confidence predicate should be filtered");
    }

    #[test]
    fn test_filter_by_confidence_keeps_high() {
        let mut gen = RuleGenerator::new(RuleConfig {
            min_frequency_ratio: 0.1,
            min_confidence: 0.1,
            generate_counts: true,
        });
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 9, 10));
        let rules = gen.filter_by_confidence(0.8);
        assert!(!rules.is_empty());
    }

    // ── multiple subjects / predicates ────────────────────────────────────

    #[test]
    fn test_multiple_types_separate_rules() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 9, 10));
        gen.add_observation(make_pattern("Company", "ex:name", "xsd:string", 9, 10));
        let rules = gen.generate_rules();
        let person_rules: Vec<_> = rules
            .iter()
            .filter(|r| r.target_class == "Person")
            .collect();
        let company_rules: Vec<_> = rules
            .iter()
            .filter(|r| r.target_class == "Company")
            .collect();
        assert!(!person_rules.is_empty());
        assert!(!company_rules.is_empty());
    }

    #[test]
    fn test_multiple_predicates_same_type() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 9, 10));
        gen.add_observation(make_pattern("Person", "ex:age", "xsd:integer", 8, 10));
        let rules = gen.generate_rules();
        let predicates: std::collections::HashSet<_> =
            rules.iter().map(|r| r.predicate.as_str()).collect();
        assert!(predicates.contains("ex:name"));
        assert!(predicates.contains("ex:age"));
    }

    // ── clear ─────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_removes_all_observations() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 9, 10));
        gen.clear();
        assert_eq!(gen.rule_count(), 0);
        assert!(gen.generate_rules().is_empty());
    }

    // ── observations accessor ─────────────────────────────────────────────

    #[test]
    fn test_observations_accessor() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 9, 10));
        assert_eq!(gen.observations().len(), 1);
    }

    // ── RuleType helpers ──────────────────────────────────────────────────

    #[test]
    fn test_rule_type_shacl_keyword() {
        assert_eq!(RuleType::MinCount.shacl_keyword(), "sh:minCount");
        assert_eq!(RuleType::MaxCount.shacl_keyword(), "sh:maxCount");
        assert_eq!(RuleType::NodeKind.shacl_keyword(), "sh:nodeKind");
        assert_eq!(RuleType::Datatype.shacl_keyword(), "sh:datatype");
        assert_eq!(RuleType::Class.shacl_keyword(), "sh:class");
    }

    #[test]
    fn test_is_count_rule() {
        assert!(
            GeneratedRule::new("T", "p", RuleType::MinCount, 0.9, Some(1), None).is_count_rule()
        );
        assert!(
            GeneratedRule::new("T", "p", RuleType::MaxCount, 0.9, None, Some(1)).is_count_rule()
        );
        assert!(
            GeneratedRule::new("T", "p", RuleType::ExactCount, 0.9, Some(1), Some(1))
                .is_count_rule()
        );
        assert!(!GeneratedRule::new("T", "p", RuleType::Datatype, 0.9, None, None).is_count_rule());
        assert!(!GeneratedRule::new("T", "p", RuleType::Class, 0.9, None, None).is_count_rule());
        assert!(!GeneratedRule::new("T", "p", RuleType::NodeKind, 0.9, None, None).is_count_rule());
    }

    // ── classify_object_type ──────────────────────────────────────────────

    #[test]
    fn test_classify_xsd_is_datatype() {
        assert_eq!(classify_object_type("xsd:integer"), RuleType::Datatype);
        assert_eq!(classify_object_type("xsd:string"), RuleType::Datatype);
        assert_eq!(classify_object_type("xsd:boolean"), RuleType::Datatype);
    }

    #[test]
    fn test_classify_sh_is_nodekind() {
        assert_eq!(classify_object_type("sh:IRI"), RuleType::NodeKind);
        assert_eq!(classify_object_type("sh:Literal"), RuleType::NodeKind);
    }

    #[test]
    fn test_classify_other_is_class() {
        assert_eq!(classify_object_type("Person"), RuleType::Class);
        assert_eq!(classify_object_type("ex:Organization"), RuleType::Class);
        assert_eq!(
            classify_object_type("http://example.org/Thing"),
            RuleType::Class
        );
    }

    // ── min_count value in generated rule ────────────────────────────────

    #[test]
    fn test_min_count_rule_has_value_1() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 9, 10));
        let rules = gen.generate_rules();
        let min_rule = rules
            .iter()
            .find(|r| r.rule_type == RuleType::MinCount)
            .expect("min count rule");
        assert_eq!(min_rule.min_count, Some(1));
        assert!(min_rule.max_count.is_none());
    }

    #[test]
    fn test_max_count_rule_has_none_min() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "ex:name", "xsd:string", 9, 10));
        let rules = gen.generate_rules();
        let max_rule = rules
            .iter()
            .find(|r| r.rule_type == RuleType::MaxCount)
            .expect("max count rule");
        assert!(max_rule.min_count.is_none());
        assert!(max_rule.max_count.is_some());
    }

    // ── target_class and predicate in rules ──────────────────────────────

    #[test]
    fn test_generated_rule_target_class() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Employee", "ex:name", "xsd:string", 8, 10));
        let rules = gen.generate_rules();
        assert!(rules.iter().all(|r| r.target_class == "Employee"));
    }

    #[test]
    fn test_generated_rule_predicate() {
        let mut gen = RuleGenerator::new(default_config());
        gen.add_observation(make_pattern("Person", "schema:email", "xsd:string", 8, 10));
        let rules = gen.generate_rules();
        assert!(rules.iter().all(|r| r.predicate == "schema:email"));
    }
}
