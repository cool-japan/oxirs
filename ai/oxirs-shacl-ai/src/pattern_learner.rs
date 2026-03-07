//! # ML-Based SHACL Pattern Learner
//!
//! Learns SHACL property shapes from RDF fact triples by analysing
//! cardinality, data type, and node kind distributions across subjects of
//! a given class.
//!
//! ## Overview
//!
//! Given a set of `RdfFact` triples and a class IRI, the learner:
//! 1. Identifies all subjects typed with the class.
//! 2. Counts per-property occurrences.
//! 3. Estimates min/max observed cardinality.
//! 4. Derives SHACL constraints when evidence is strong enough.
//! 5. Renders the result as Turtle (SHACL compact notation).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────
// RDF fact
// ─────────────────────────────────────────────

/// A single RDF triple with optional datatype annotation on the object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdfFact {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    /// XSD datatype IRI for literal objects, e.g. `xsd:string`.
    pub object_datatype: Option<String>,
}

// ─────────────────────────────────────────────
// Property frequency
// ─────────────────────────────────────────────

/// Frequency statistics for a single property in a class pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertyFrequency {
    pub property: String,
    /// Total triples with this property for subjects of the class.
    pub count: u64,
    /// Number of distinct subjects of the class.
    pub total_subjects: u64,
    /// `count / total_subjects` — fraction of subjects having this property.
    pub frequency: f64,
}

// ─────────────────────────────────────────────
// Class pattern
// ─────────────────────────────────────────────

/// Aggregated statistics for one SHACL node shape class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassPattern {
    pub class_iri: String,
    pub subject_count: u64,
    pub property_frequencies: Vec<PropertyFrequency>,
    /// Per-property `(min_observed, max_observed)` cardinality.
    pub cardinality_estimates: HashMap<String, (u64, u64)>,
}

// ─────────────────────────────────────────────
// Learned constraint
// ─────────────────────────────────────────────

/// Type of a single learned SHACL constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearnedConstraintType {
    MinCount(u64),
    MaxCount(u64),
    ExactCount(u64),
    DataType(String),
    NodeKind(String),
    MinLength(u64),
    MaxLength(u64),
}

/// A derived SHACL constraint with confidence and support metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedConstraint {
    pub constraint_type: LearnedConstraintType,
    /// Property this constraint applies to.
    pub property: String,
    /// Confidence in [0.0, 1.0].
    pub confidence: f64,
    /// Support: fraction of subjects covered by this constraint.
    pub support: f64,
}

// ─────────────────────────────────────────────
// Pattern learner
// ─────────────────────────────────────────────

/// ML-based SHACL pattern learner operating on RDF facts.
#[derive(Debug, Clone)]
pub struct PatternLearner {
    /// Minimum confidence threshold for emitting a constraint.
    min_confidence: f64,
    /// Minimum support threshold for emitting a constraint.
    min_support: f64,
}

impl PatternLearner {
    /// Create a new learner with the given thresholds.
    pub fn new(min_confidence: f64, min_support: f64) -> Self {
        Self {
            min_confidence: min_confidence.clamp(0.0, 1.0),
            min_support: min_support.clamp(0.0, 1.0),
        }
    }

    /// Analyse `facts` for subjects typed with `class_iri` and build a
    /// [`ClassPattern`].
    pub fn learn_class_patterns(&self, facts: &[RdfFact], class_iri: &str) -> ClassPattern {
        // RDF type predicate
        const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

        // 1. Collect all subjects that have rdf:type == class_iri
        let subjects: std::collections::HashSet<&str> = facts
            .iter()
            .filter(|f| f.predicate == RDF_TYPE && f.object == class_iri)
            .map(|f| f.subject.as_str())
            .collect();

        let subject_count = subjects.len() as u64;

        // 2. For each non-type triple whose subject is in the class, accumulate stats
        // per-subject-per-property count: HashMap<property, HashMap<subject, u64>>
        let mut prop_subj_count: HashMap<&str, HashMap<&str, u64>> = HashMap::new();

        for fact in facts {
            if fact.predicate == RDF_TYPE {
                continue;
            }
            if !subjects.contains(fact.subject.as_str()) {
                continue;
            }
            prop_subj_count
                .entry(fact.predicate.as_str())
                .or_default()
                .entry(fact.subject.as_str())
                .and_modify(|c| *c += 1)
                .or_insert(1);
        }

        // 3. Build PropertyFrequency and cardinality estimates
        let mut property_frequencies: Vec<PropertyFrequency> = Vec::new();
        let mut cardinality_estimates: HashMap<String, (u64, u64)> = HashMap::new();

        for (prop, subj_map) in &prop_subj_count {
            let total_triples: u64 = subj_map.values().sum();
            let frequency = if subject_count > 0 {
                subj_map.len() as f64 / subject_count as f64
            } else {
                0.0
            };

            property_frequencies.push(PropertyFrequency {
                property: prop.to_string(),
                count: total_triples,
                total_subjects: subject_count,
                frequency,
            });

            let min_card = *subj_map.values().min().unwrap_or(&0);
            let max_card = *subj_map.values().max().unwrap_or(&0);
            cardinality_estimates.insert(prop.to_string(), (min_card, max_card));
        }

        // Sort by property name for deterministic output
        property_frequencies.sort_by(|a, b| a.property.cmp(&b.property));

        ClassPattern {
            class_iri: class_iri.to_string(),
            subject_count,
            property_frequencies,
            cardinality_estimates,
        }
    }

    /// Derive SHACL constraints from a [`ClassPattern`].
    pub fn learn_constraints(&self, pattern: &ClassPattern) -> Vec<LearnedConstraint> {
        let mut constraints: Vec<LearnedConstraint> = Vec::new();

        for freq in &pattern.property_frequencies {
            let support = freq.frequency;
            if support < self.min_support {
                continue;
            }

            let (card_min, card_max) = pattern
                .cardinality_estimates
                .get(&freq.property)
                .copied()
                .unwrap_or((0, 0));

            // ── MinCount(1): all subjects with the property have ≥1 occurrence
            // Confidence proportional to frequency (how many subjects have it)
            if freq.frequency >= self.min_confidence && card_min >= 1 {
                let conf = freq.frequency;
                constraints.push(LearnedConstraint {
                    constraint_type: LearnedConstraintType::MinCount(1),
                    property: freq.property.clone(),
                    confidence: conf,
                    support,
                });
            }

            // ── MaxCount(1): observed max cardinality is 1 for ≥95% of subjects
            if card_max == 1 {
                let conf = Self::constraint_confidence(freq, card_min, card_max);
                if conf >= self.min_confidence {
                    constraints.push(LearnedConstraint {
                        constraint_type: LearnedConstraintType::MaxCount(1),
                        property: freq.property.clone(),
                        confidence: conf,
                        support,
                    });
                }
            }

            // ── ExactCount: min == max
            if card_min == card_max && card_min > 0 && card_min > 1 {
                let conf = Self::constraint_confidence(freq, card_min, card_max);
                if conf >= self.min_confidence {
                    constraints.push(LearnedConstraint {
                        constraint_type: LearnedConstraintType::ExactCount(card_min),
                        property: freq.property.clone(),
                        confidence: conf,
                        support,
                    });
                }
            }
        }

        constraints
    }

    /// Derive constraints related to datatype and node kind from facts.
    ///
    /// This is a separate analysis pass that requires the raw facts in addition
    /// to the pattern, since the pattern itself only stores aggregated counts.
    pub fn learn_datatype_constraints(
        &self,
        facts: &[RdfFact],
        pattern: &ClassPattern,
    ) -> Vec<LearnedConstraint> {
        const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";

        let subjects: std::collections::HashSet<&str> = facts
            .iter()
            .filter(|f| f.predicate == RDF_TYPE && f.object == pattern.class_iri)
            .map(|f| f.subject.as_str())
            .collect();

        // Group object datatypes per property
        let mut prop_types: HashMap<String, Vec<Option<String>>> = HashMap::new();
        let mut prop_node_kinds: HashMap<String, (u64, u64)> = HashMap::new(); // (uri_count, bnode_count)

        for fact in facts {
            if fact.predicate == RDF_TYPE {
                continue;
            }
            if !subjects.contains(fact.subject.as_str()) {
                continue;
            }
            prop_types
                .entry(fact.predicate.clone())
                .or_default()
                .push(fact.object_datatype.clone());

            let (uris, bnodes) = prop_node_kinds
                .entry(fact.predicate.clone())
                .or_insert((0, 0));
            if fact.object_datatype.is_none() {
                if fact.object.starts_with("_:") {
                    *bnodes += 1;
                } else {
                    *uris += 1;
                }
            }
        }

        let mut constraints: Vec<LearnedConstraint> = Vec::new();

        // DataType constraint
        for (prop, types) in &prop_types {
            let non_null: Vec<&str> = types.iter().filter_map(|t| t.as_deref()).collect();
            if non_null.is_empty() {
                continue;
            }
            let first = non_null[0];
            let all_same = non_null.iter().all(|&t| t == first);
            if all_same {
                let support = pattern
                    .property_frequencies
                    .iter()
                    .find(|f| f.property == *prop)
                    .map(|f| f.frequency)
                    .unwrap_or(0.0);
                let conf = non_null.len() as f64 / types.len() as f64;
                if conf >= self.min_confidence && support >= self.min_support {
                    constraints.push(LearnedConstraint {
                        constraint_type: LearnedConstraintType::DataType(first.to_string()),
                        property: prop.clone(),
                        confidence: conf,
                        support,
                    });
                }
            }
        }

        // NodeKind constraint
        for (prop, (uris, bnodes)) in &prop_node_kinds {
            let total = uris + bnodes;
            if total == 0 {
                continue;
            }
            let support = pattern
                .property_frequencies
                .iter()
                .find(|f| f.property == *prop)
                .map(|f| f.frequency)
                .unwrap_or(0.0);

            if *uris == total {
                let conf = *uris as f64 / total as f64;
                if conf >= self.min_confidence && support >= self.min_support {
                    constraints.push(LearnedConstraint {
                        constraint_type: LearnedConstraintType::NodeKind("IRI".to_string()),
                        property: prop.clone(),
                        confidence: conf,
                        support,
                    });
                }
            } else if *bnodes == total {
                let conf = *bnodes as f64 / total as f64;
                if conf >= self.min_confidence && support >= self.min_support {
                    constraints.push(LearnedConstraint {
                        constraint_type: LearnedConstraintType::NodeKind("BlankNode".to_string()),
                        property: prop.clone(),
                        confidence: conf,
                        support,
                    });
                }
            }
        }

        constraints
    }

    /// Generate Turtle serialisation of a learned shape.
    pub fn render_shacl_ttl(
        &self,
        pattern: &ClassPattern,
        constraints: &[LearnedConstraint],
    ) -> String {
        let mut ttl = String::new();
        ttl.push_str("@prefix sh: <http://www.w3.org/ns/shacl#> .\n");
        ttl.push_str("@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n\n");

        // Use a safe shape name derived from the class IRI
        let shape_name = class_to_shape_name(&pattern.class_iri);
        ttl.push_str(&format!("<{shape_name}Shape>\n"));
        ttl.push_str("    a sh:NodeShape ;\n");
        ttl.push_str(&format!("    sh:targetClass <{}> ;\n", pattern.class_iri));

        // Group constraints by property
        let mut by_prop: HashMap<&str, Vec<&LearnedConstraint>> = HashMap::new();
        for c in constraints {
            by_prop.entry(c.property.as_str()).or_default().push(c);
        }

        let mut props: Vec<&str> = by_prop.keys().copied().collect();
        props.sort_unstable();

        for (i, prop) in props.iter().enumerate() {
            let cs = &by_prop[prop];
            let sep = if i + 1 < props.len() { ";" } else { "." };
            ttl.push_str("    sh:property [\n");
            ttl.push_str(&format!("        sh:path <{prop}> ;\n"));
            for c in cs.iter() {
                match &c.constraint_type {
                    LearnedConstraintType::MinCount(n) => {
                        ttl.push_str(&format!("        sh:minCount {n} ;\n"));
                    }
                    LearnedConstraintType::MaxCount(n) => {
                        ttl.push_str(&format!("        sh:maxCount {n} ;\n"));
                    }
                    LearnedConstraintType::ExactCount(n) => {
                        ttl.push_str(&format!("        sh:minCount {n} ;\n"));
                        ttl.push_str(&format!("        sh:maxCount {n} ;\n"));
                    }
                    LearnedConstraintType::DataType(dt) => {
                        ttl.push_str(&format!("        sh:datatype <{dt}> ;\n"));
                    }
                    LearnedConstraintType::NodeKind(nk) => {
                        ttl.push_str(&format!("        sh:nodeKind sh:{nk} ;\n"));
                    }
                    LearnedConstraintType::MinLength(n) => {
                        ttl.push_str(&format!("        sh:minLength {n} ;\n"));
                    }
                    LearnedConstraintType::MaxLength(n) => {
                        ttl.push_str(&format!("        sh:maxLength {n} ;\n"));
                    }
                }
            }
            ttl.push_str(&format!("    ] {sep}\n"));
        }

        ttl
    }

    /// Merge multiple class patterns from different sources into one.
    ///
    /// Accumulates subject counts, property counts, and expands cardinality
    /// ranges.
    pub fn merge_patterns(patterns: &[ClassPattern]) -> ClassPattern {
        if patterns.is_empty() {
            return ClassPattern {
                class_iri: String::new(),
                subject_count: 0,
                property_frequencies: vec![],
                cardinality_estimates: HashMap::new(),
            };
        }

        let class_iri = patterns[0].class_iri.clone();
        let subject_count: u64 = patterns.iter().map(|p| p.subject_count).sum();

        let mut prop_count: HashMap<String, u64> = HashMap::new();
        let mut cardinality_estimates: HashMap<String, (u64, u64)> = HashMap::new();

        for pat in patterns {
            for freq in &pat.property_frequencies {
                *prop_count.entry(freq.property.clone()).or_insert(0) += freq.count;
            }
            for (prop, (min_c, max_c)) in &pat.cardinality_estimates {
                let entry = cardinality_estimates
                    .entry(prop.clone())
                    .or_insert((*min_c, *max_c));
                entry.0 = entry.0.min(*min_c);
                entry.1 = entry.1.max(*max_c);
            }
        }

        let property_frequencies: Vec<PropertyFrequency> = {
            let mut v: Vec<PropertyFrequency> = prop_count
                .into_iter()
                .map(|(prop, count)| {
                    let freq = if subject_count > 0 {
                        count as f64 / subject_count as f64
                    } else {
                        0.0
                    };
                    PropertyFrequency {
                        property: prop,
                        count,
                        total_subjects: subject_count,
                        frequency: freq,
                    }
                })
                .collect();
            v.sort_by(|a, b| a.property.cmp(&b.property));
            v
        };

        ClassPattern {
            class_iri,
            subject_count,
            property_frequencies,
            cardinality_estimates,
        }
    }

    /// Compute confidence for a cardinality constraint.
    ///
    /// Uses the frequency multiplied by a consistency factor based on whether
    /// min == max.
    pub fn constraint_confidence(freq: &PropertyFrequency, card_min: u64, card_max: u64) -> f64 {
        let consistency = if card_min == card_max { 1.0 } else { 0.7 };
        (freq.frequency * consistency).min(1.0)
    }
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

fn class_to_shape_name(class_iri: &str) -> String {
    // Use the local name after the last '#' or '/'
    class_iri
        .rsplit_once('#')
        .map(|(_, local)| local)
        .or_else(|| class_iri.rsplit_once('/').map(|(_, local)| local))
        .unwrap_or(class_iri)
        .to_string()
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const PERSON_CLASS: &str = "http://example.org/Person";
    const RDF_TYPE: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type";
    const KNOWS: &str = "http://xmlns.com/foaf/0.1/knows";
    const NAME: &str = "http://xmlns.com/foaf/0.1/name";
    const AGE: &str = "http://xmlns.com/foaf/0.1/age";
    const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";
    const XSD_INT: &str = "http://www.w3.org/2001/XMLSchema#integer";

    fn typed(subj: &str) -> RdfFact {
        RdfFact {
            subject: subj.to_string(),
            predicate: RDF_TYPE.to_string(),
            object: PERSON_CLASS.to_string(),
            object_datatype: None,
        }
    }

    fn triple(subj: &str, pred: &str, obj: &str, dt: Option<&str>) -> RdfFact {
        RdfFact {
            subject: subj.to_string(),
            predicate: pred.to_string(),
            object: obj.to_string(),
            object_datatype: dt.map(|s| s.to_string()),
        }
    }

    fn learner() -> PatternLearner {
        PatternLearner::new(0.5, 0.3)
    }

    // ── learn_class_patterns ────────────────────────────────────────────

    #[test]
    fn test_learn_class_patterns_basic() {
        let facts = vec![
            typed("alice"),
            typed("bob"),
            triple("alice", NAME, "Alice", Some(XSD_STRING)),
            triple("bob", NAME, "Bob", Some(XSD_STRING)),
        ];
        let l = learner();
        let pat = l.learn_class_patterns(&facts, PERSON_CLASS);
        assert_eq!(pat.subject_count, 2);
        assert_eq!(pat.class_iri, PERSON_CLASS);
        let name_freq = pat.property_frequencies.iter().find(|f| f.property == NAME);
        assert!(name_freq.is_some());
        assert!((name_freq.unwrap().frequency - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_learn_class_patterns_partial_property() {
        let facts = vec![
            typed("alice"),
            typed("bob"),
            typed("carol"),
            triple("alice", AGE, "30", Some(XSD_INT)),
        ];
        let l = learner();
        let pat = l.learn_class_patterns(&facts, PERSON_CLASS);
        assert_eq!(pat.subject_count, 3);
        let age_freq = pat
            .property_frequencies
            .iter()
            .find(|f| f.property == AGE)
            .unwrap();
        assert!((age_freq.frequency - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_learn_class_patterns_no_subjects() {
        let facts = vec![triple("alice", NAME, "Alice", Some(XSD_STRING))];
        let l = learner();
        let pat = l.learn_class_patterns(&facts, PERSON_CLASS);
        assert_eq!(pat.subject_count, 0);
        assert!(pat.property_frequencies.is_empty());
    }

    #[test]
    fn test_learn_class_patterns_cardinality() {
        let facts = vec![
            typed("alice"),
            triple("alice", KNOWS, "bob", None),
            triple("alice", KNOWS, "carol", None),
        ];
        let l = learner();
        let pat = l.learn_class_patterns(&facts, PERSON_CLASS);
        let (min_c, max_c) = pat
            .cardinality_estimates
            .get(KNOWS)
            .copied()
            .unwrap_or((0, 0));
        assert_eq!(min_c, 2);
        assert_eq!(max_c, 2);
    }

    // ── learn_constraints ────────────────────────────────────────────────

    #[test]
    fn test_learn_constraints_min_count() {
        // All subjects have the property → MinCount(1) should be derived
        let facts = vec![
            typed("alice"),
            typed("bob"),
            triple("alice", NAME, "Alice", Some(XSD_STRING)),
            triple("bob", NAME, "Bob", Some(XSD_STRING)),
        ];
        let l = PatternLearner::new(0.5, 0.5);
        let pat = l.learn_class_patterns(&facts, PERSON_CLASS);
        let cs = l.learn_constraints(&pat);
        let has_min = cs.iter().any(|c| {
            matches!(&c.constraint_type, LearnedConstraintType::MinCount(1)) && c.property == NAME
        });
        assert!(has_min, "expected MinCount(1) for {NAME}");
    }

    #[test]
    fn test_learn_constraints_max_count_1() {
        let facts = vec![
            typed("alice"),
            typed("bob"),
            triple("alice", NAME, "Alice", Some(XSD_STRING)),
            triple("bob", NAME, "Bob", Some(XSD_STRING)),
        ];
        let l = PatternLearner::new(0.5, 0.5);
        let pat = l.learn_class_patterns(&facts, PERSON_CLASS);
        let cs = l.learn_constraints(&pat);
        let has_max = cs.iter().any(|c| {
            matches!(&c.constraint_type, LearnedConstraintType::MaxCount(1)) && c.property == NAME
        });
        assert!(has_max, "expected MaxCount(1) for {NAME}");
    }

    #[test]
    fn test_learn_constraints_below_threshold_skipped() {
        // Only 1 out of 10 subjects has the property → below min_support 0.5
        let mut facts: Vec<RdfFact> = (0..10).map(|i| typed(&format!("s{i}"))).collect();
        facts.push(triple("s0", NAME, "S0", Some(XSD_STRING)));
        let l = PatternLearner::new(0.5, 0.5);
        let pat = l.learn_class_patterns(&facts, PERSON_CLASS);
        let cs = l.learn_constraints(&pat);
        let name_cs: Vec<_> = cs.iter().filter(|c| c.property == NAME).collect();
        assert!(
            name_cs.is_empty(),
            "low-frequency property should be skipped"
        );
    }

    // ── render_shacl_ttl ────────────────────────────────────────────────

    #[test]
    fn test_render_shacl_ttl_contains_shape() {
        let facts = vec![
            typed("alice"),
            triple("alice", NAME, "Alice", Some(XSD_STRING)),
        ];
        let l = PatternLearner::new(0.5, 0.3);
        let pat = l.learn_class_patterns(&facts, PERSON_CLASS);
        let cs = l.learn_constraints(&pat);
        let ttl = l.render_shacl_ttl(&pat, &cs);
        assert!(ttl.contains("sh:NodeShape"));
        assert!(ttl.contains(PERSON_CLASS));
    }

    #[test]
    fn test_render_shacl_ttl_contains_prefix() {
        let l = learner();
        let pat = ClassPattern {
            class_iri: PERSON_CLASS.to_string(),
            subject_count: 0,
            property_frequencies: vec![],
            cardinality_estimates: HashMap::new(),
        };
        let ttl = l.render_shacl_ttl(&pat, &[]);
        assert!(ttl.contains("@prefix sh:"));
    }

    #[test]
    fn test_render_shacl_ttl_with_min_count() {
        let constraint = LearnedConstraint {
            constraint_type: LearnedConstraintType::MinCount(1),
            property: NAME.to_string(),
            confidence: 1.0,
            support: 1.0,
        };
        let pat = ClassPattern {
            class_iri: PERSON_CLASS.to_string(),
            subject_count: 1,
            property_frequencies: vec![],
            cardinality_estimates: HashMap::new(),
        };
        let l = learner();
        let ttl = l.render_shacl_ttl(&pat, &[constraint]);
        assert!(ttl.contains("sh:minCount 1"));
    }

    // ── merge_patterns ───────────────────────────────────────────────────

    #[test]
    fn test_merge_patterns_empty() {
        let merged = PatternLearner::merge_patterns(&[]);
        assert_eq!(merged.subject_count, 0);
        assert!(merged.property_frequencies.is_empty());
    }

    #[test]
    fn test_merge_patterns_sums_subjects() {
        let p1 = ClassPattern {
            class_iri: PERSON_CLASS.to_string(),
            subject_count: 5,
            property_frequencies: vec![PropertyFrequency {
                property: NAME.to_string(),
                count: 5,
                total_subjects: 5,
                frequency: 1.0,
            }],
            cardinality_estimates: [(NAME.to_string(), (1u64, 1u64))].into(),
        };
        let p2 = ClassPattern {
            class_iri: PERSON_CLASS.to_string(),
            subject_count: 3,
            property_frequencies: vec![PropertyFrequency {
                property: NAME.to_string(),
                count: 3,
                total_subjects: 3,
                frequency: 1.0,
            }],
            cardinality_estimates: [(NAME.to_string(), (1u64, 2u64))].into(),
        };
        let merged = PatternLearner::merge_patterns(&[p1, p2]);
        assert_eq!(merged.subject_count, 8);
        let name_freq = merged
            .property_frequencies
            .iter()
            .find(|f| f.property == NAME)
            .unwrap();
        assert_eq!(name_freq.count, 8);
        let (min_c, max_c) = merged.cardinality_estimates.get(NAME).copied().unwrap();
        assert_eq!(min_c, 1);
        assert_eq!(max_c, 2);
    }

    // ── constraint_confidence ────────────────────────────────────────────

    #[test]
    fn test_constraint_confidence_equal_card() {
        let freq = PropertyFrequency {
            property: NAME.to_string(),
            count: 10,
            total_subjects: 10,
            frequency: 1.0,
        };
        let conf = PatternLearner::constraint_confidence(&freq, 1, 1);
        assert!((conf - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_constraint_confidence_unequal_card() {
        let freq = PropertyFrequency {
            property: NAME.to_string(),
            count: 10,
            total_subjects: 10,
            frequency: 1.0,
        };
        let conf = PatternLearner::constraint_confidence(&freq, 1, 3);
        assert!((conf - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_constraint_confidence_capped_at_1() {
        let freq = PropertyFrequency {
            property: NAME.to_string(),
            count: 10,
            total_subjects: 10,
            frequency: 2.0, // artificially > 1
        };
        let conf = PatternLearner::constraint_confidence(&freq, 1, 1);
        assert!(conf <= 1.0);
    }

    // ── learn_datatype_constraints ───────────────────────────────────────

    #[test]
    fn test_learn_datatype_constraints_detects_xsd_string() {
        let facts = vec![
            typed("alice"),
            typed("bob"),
            triple("alice", NAME, "Alice", Some(XSD_STRING)),
            triple("bob", NAME, "Bob", Some(XSD_STRING)),
        ];
        let l = PatternLearner::new(0.5, 0.5);
        let pat = l.learn_class_patterns(&facts, PERSON_CLASS);
        let cs = l.learn_datatype_constraints(&facts, &pat);
        let dt_cs: Vec<_> = cs
            .iter()
            .filter(|c| matches!(&c.constraint_type, LearnedConstraintType::DataType(dt) if dt == XSD_STRING))
            .collect();
        assert!(!dt_cs.is_empty(), "expected DataType(xsd:string)");
    }

    #[test]
    fn test_learn_datatype_constraints_iri_node_kind() {
        let facts = vec![
            typed("alice"),
            typed("bob"),
            triple("alice", KNOWS, "http://example.org/bob", None),
            triple("bob", KNOWS, "http://example.org/carol", None),
        ];
        let l = PatternLearner::new(0.5, 0.5);
        let pat = l.learn_class_patterns(&facts, PERSON_CLASS);
        let cs = l.learn_datatype_constraints(&facts, &pat);
        let iri_cs: Vec<_> = cs
            .iter()
            .filter(
                |c| matches!(&c.constraint_type, LearnedConstraintType::NodeKind(k) if k == "IRI"),
            )
            .collect();
        assert!(!iri_cs.is_empty(), "expected NodeKind(IRI) for knows");
    }
}
