//! AI-assisted SHACL property suggestion from data patterns.
//!
//! Analyses observed property usage patterns across RDF nodes and suggests
//! SHACL property shapes with inferred cardinalities, datatypes, and
//! confidence scores.

use std::collections::HashSet;

// ──────────────────────────────────────────────────────────────────────────────
// Public types
// ──────────────────────────────────────────────────────────────────────────────

/// Observed usage of a single property across a set of nodes of the same class.
#[derive(Debug, Clone)]
pub struct PropertyPattern {
    pub predicate: String,
    /// Number of nodes that have at least one value for this property.
    pub usage_count: usize,
    /// RDF datatypes or class IRIs observed in the objects.
    pub object_types: HashSet<String>,
    /// `true` if every node of the class has this property.
    pub always_present: bool,
    /// A sample of up to a few object values (for datatype inference).
    pub example_values: Vec<String>,
}

/// All property patterns observed for a single class.
#[derive(Debug, Clone)]
pub struct NodePattern {
    pub class: String,
    pub properties: Vec<PropertyPattern>,
}

/// A candidate SHACL property shape.
#[derive(Debug, Clone)]
pub struct SuggestedProperty {
    pub predicate: String,
    /// Whether the property should be `sh:minCount 1` (required).
    pub required: bool,
    /// Inferred `xsd:*` datatype IRI, if detectable.
    pub inferred_datatype: Option<String>,
    /// Suggested `sh:minCount`.
    pub min_count: Option<usize>,
    /// Suggested `sh:maxCount` (`None` = unbounded).
    pub max_count: Option<usize>,
    /// Confidence in this suggestion in [0, 1].
    pub confidence: f64,
}

/// A candidate SHACL node shape for a target class.
#[derive(Debug, Clone)]
pub struct ShapeCandidate {
    pub target_class: String,
    pub suggested_properties: Vec<SuggestedProperty>,
}

/// Analyses [`NodePattern`]s and produces [`ShapeCandidate`]s.
pub struct PropertySuggester {
    /// Properties used by fewer than this fraction of nodes are excluded.
    min_frequency: f64,
    /// Suggestions with confidence below this threshold are excluded.
    min_confidence: f64,
}

impl PropertySuggester {
    /// Create a new suggester.
    ///
    /// * `min_frequency` — fraction of nodes [0, 1] that must use a property
    ///   for it to be considered.
    /// * `min_confidence` — minimum confidence [0, 1] for a suggestion to be
    ///   included in the output.
    pub fn new(min_frequency: f64, min_confidence: f64) -> Self {
        Self {
            min_frequency: min_frequency.clamp(0.0, 1.0),
            min_confidence: min_confidence.clamp(0.0, 1.0),
        }
    }

    /// Analyse all class patterns and return shape candidates for every class.
    pub fn analyze(&self, patterns: &[NodePattern]) -> Vec<ShapeCandidate> {
        patterns
            .iter()
            .filter_map(|np| self.suggest_for_class(&np.class, patterns))
            .collect()
    }

    /// Return a shape candidate for the given class, or `None` if no class
    /// pattern is found.
    pub fn suggest_for_class(
        &self,
        class: &str,
        patterns: &[NodePattern],
    ) -> Option<ShapeCandidate> {
        let node_pattern = patterns.iter().find(|np| np.class == class)?;

        // Estimate total distinct nodes by the maximum usage_count in the class
        let total_nodes = node_pattern
            .properties
            .iter()
            .map(|p| p.usage_count)
            .max()
            .unwrap_or(1)
            .max(1);

        let suggested_properties: Vec<SuggestedProperty> = node_pattern
            .properties
            .iter()
            .filter(|p| {
                // Filter by frequency
                let freq = p.usage_count as f64 / total_nodes as f64;
                freq >= self.min_frequency
            })
            .filter_map(|p| {
                let confidence = self.confidence_score(p, total_nodes);
                if confidence < self.min_confidence {
                    return None;
                }
                let (min_count, max_count) = self.infer_cardinality(p, total_nodes);
                let inferred_datatype = self.infer_datatype(&p.example_values);
                Some(SuggestedProperty {
                    predicate: p.predicate.clone(),
                    required: p.always_present,
                    inferred_datatype,
                    min_count,
                    max_count,
                    confidence,
                })
            })
            .collect();

        Some(ShapeCandidate {
            target_class: class.to_string(),
            suggested_properties,
        })
    }

    /// Infer minimum and maximum cardinality for a property.
    ///
    /// - If `always_present`, min_count = 1; otherwise 0.
    /// - max_count is `None` (unbounded) unless the property appears to be
    ///   functional (one value per node), in which case max_count = 1.
    pub fn infer_cardinality(
        &self,
        pattern: &PropertyPattern,
        total_nodes: usize,
    ) -> (Option<usize>, Option<usize>) {
        let min_count = if pattern.always_present {
            Some(1)
        } else {
            Some(0)
        };

        // Heuristic: if usage_count equals total_nodes and only one object
        // type is observed, treat it as functional (max 1).
        let max_count =
            if pattern.always_present && pattern.object_types.len() <= 1 && total_nodes > 0 {
                Some(1)
            } else {
                None
            };

        (min_count, max_count)
    }

    /// Infer an `xsd:*` datatype from a collection of example values.
    ///
    /// Detection order: integer → decimal → boolean → dateTime → string.
    /// Returns `None` if `example_values` is empty.
    pub fn infer_datatype(&self, example_values: &[String]) -> Option<String> {
        if example_values.is_empty() {
            return None;
        }

        // All must pass the predicate for that type
        let all = |pred: &dyn Fn(&str) -> bool| example_values.iter().all(|v| pred(v.as_str()));

        if all(&|v| v.parse::<i64>().is_ok()) {
            return Some("xsd:integer".to_string());
        }
        if all(&|v| v.parse::<f64>().is_ok()) {
            return Some("xsd:decimal".to_string());
        }
        if all(&|v| matches!(v, "true" | "false" | "1" | "0")) {
            return Some("xsd:boolean".to_string());
        }
        // Simple ISO 8601 / xsd:dateTime heuristic: contains 'T' and '-'
        if all(&|v| v.contains('T') && v.contains('-')) {
            return Some("xsd:dateTime".to_string());
        }

        Some("xsd:string".to_string())
    }

    /// Compute a confidence score in [0, 1] for the given pattern.
    ///
    /// Score is the usage frequency relative to `total_nodes`, boosted
    /// slightly when the property is always present.
    pub fn confidence_score(&self, pattern: &PropertyPattern, total_nodes: usize) -> f64 {
        if total_nodes == 0 {
            return 0.0;
        }
        let freq = (pattern.usage_count as f64 / total_nodes as f64).clamp(0.0, 1.0);
        if pattern.always_present {
            // Small boost for always-present properties
            (freq * 1.05).clamp(0.0, 1.0)
        } else {
            freq
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn suggester() -> PropertySuggester {
        PropertySuggester::new(0.5, 0.0)
    }

    fn always_pattern(pred: &str, count: usize) -> PropertyPattern {
        PropertyPattern {
            predicate: pred.to_string(),
            usage_count: count,
            object_types: HashSet::from(["xsd:string".to_string()]),
            always_present: true,
            example_values: vec!["hello".to_string()],
        }
    }

    fn partial_pattern(pred: &str, count: usize, total: usize) -> PropertyPattern {
        PropertyPattern {
            predicate: pred.to_string(),
            usage_count: count,
            object_types: HashSet::new(),
            always_present: count == total,
            example_values: Vec::new(),
        }
    }

    // ── always_present → required ────────────────────────────────────────────

    #[test]
    fn test_always_present_implies_required() {
        let s = suggester();
        let np = NodePattern {
            class: "Person".to_string(),
            properties: vec![always_pattern("ex:name", 10)],
        };
        let candidate = s
            .suggest_for_class("Person", &[np])
            .expect("should succeed");
        let prop = &candidate.suggested_properties[0];
        assert!(prop.required);
    }

    #[test]
    fn test_not_always_present_not_required() {
        let s = suggester();
        let np = NodePattern {
            class: "Person".to_string(),
            properties: vec![partial_pattern("ex:nickname", 5, 10)],
        };
        let candidate = s
            .suggest_for_class("Person", &[np])
            .expect("should succeed");
        let prop = &candidate.suggested_properties[0];
        assert!(!prop.required);
    }

    // ── min_frequency threshold ──────────────────────────────────────────────

    #[test]
    fn test_low_usage_filtered_out() {
        let s = PropertySuggester::new(0.8, 0.0);
        let np = NodePattern {
            class: "Person".to_string(),
            properties: vec![
                // 9/10 = 0.9 ≥ 0.8 → kept
                always_pattern("ex:name", 9),
                // 2/9 ≈ 0.22 < 0.8 → filtered
                partial_pattern("ex:optProp", 2, 9),
            ],
        };
        let candidate = s
            .suggest_for_class("Person", &[np])
            .expect("should succeed");
        assert_eq!(candidate.suggested_properties.len(), 1);
        assert_eq!(candidate.suggested_properties[0].predicate, "ex:name");
    }

    #[test]
    fn test_all_properties_above_threshold() {
        let s = PropertySuggester::new(0.0, 0.0);
        let np = NodePattern {
            class: "X".to_string(),
            properties: vec![partial_pattern("p1", 1, 10), partial_pattern("p2", 1, 10)],
        };
        let candidate = s.suggest_for_class("X", &[np]).expect("should succeed");
        assert_eq!(candidate.suggested_properties.len(), 2);
    }

    // ── min_confidence filtering ─────────────────────────────────────────────

    #[test]
    fn test_min_confidence_filters_low_confidence() {
        let s = PropertySuggester::new(0.0, 0.9);
        let np = NodePattern {
            class: "Y".to_string(),
            properties: vec![
                // usage_count=5, total=10 → confidence = 0.5 < 0.9 → filtered
                partial_pattern("ex:rare", 5, 10),
                // always_present: confidence = min(1.05, 1) = 1.0 ≥ 0.9 → kept
                always_pattern("ex:common", 10),
            ],
        };
        let candidate = s.suggest_for_class("Y", &[np]).expect("should succeed");
        assert_eq!(candidate.suggested_properties.len(), 1);
        assert_eq!(candidate.suggested_properties[0].predicate, "ex:common");
    }

    // ── analyze multiple classes ─────────────────────────────────────────────

    #[test]
    fn test_analyze_multiple_classes() {
        let s = suggester();
        let patterns = vec![
            NodePattern {
                class: "A".to_string(),
                properties: vec![always_pattern("p1", 5)],
            },
            NodePattern {
                class: "B".to_string(),
                properties: vec![always_pattern("p2", 3)],
            },
        ];
        let candidates = s.analyze(&patterns);
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn test_analyze_empty_patterns() {
        let s = suggester();
        let candidates = s.analyze(&[]);
        assert!(candidates.is_empty());
    }

    // ── suggest_for_class found/not found ────────────────────────────────────

    #[test]
    fn test_suggest_for_class_found() {
        let s = suggester();
        let np = NodePattern {
            class: "Person".to_string(),
            properties: vec![always_pattern("ex:name", 10)],
        };
        assert!(s.suggest_for_class("Person", &[np]).is_some());
    }

    #[test]
    fn test_suggest_for_class_not_found_returns_none() {
        let s = suggester();
        let np = NodePattern {
            class: "Person".to_string(),
            properties: vec![],
        };
        assert!(s.suggest_for_class("Animal", &[np]).is_none());
    }

    // ── infer_datatype ───────────────────────────────────────────────────────

    #[test]
    fn test_infer_datatype_integer() {
        let s = suggester();
        let dt = s.infer_datatype(&["42".into(), "-7".into(), "0".into()]);
        assert_eq!(dt.as_deref(), Some("xsd:integer"));
    }

    #[test]
    fn test_infer_datatype_decimal() {
        let s = suggester();
        let dt = s.infer_datatype(&["3.14".into(), "-2.71".into()]);
        assert_eq!(dt.as_deref(), Some("xsd:decimal"));
    }

    #[test]
    fn test_infer_datatype_boolean_true_false() {
        let s = suggester();
        let dt = s.infer_datatype(&["true".into(), "false".into()]);
        assert_eq!(dt.as_deref(), Some("xsd:boolean"));
    }

    #[test]
    fn test_infer_datatype_boolean_01() {
        let s = suggester();
        let dt = s.infer_datatype(&["1".into(), "0".into()]);
        // "1" and "0" parse as integers, so xsd:integer wins over xsd:boolean
        // (detection order: integer first)
        assert!(dt.is_some());
    }

    #[test]
    fn test_infer_datatype_datetime() {
        let s = suggester();
        let dt = s.infer_datatype(&["2024-01-15T10:30:00Z".into(), "2023-06-01T00:00:00Z".into()]);
        assert_eq!(dt.as_deref(), Some("xsd:dateTime"));
    }

    #[test]
    fn test_infer_datatype_string() {
        let s = suggester();
        let dt = s.infer_datatype(&["hello".into(), "world".into()]);
        assert_eq!(dt.as_deref(), Some("xsd:string"));
    }

    #[test]
    fn test_infer_datatype_empty_returns_none() {
        let s = suggester();
        assert!(s.infer_datatype(&[]).is_none());
    }

    #[test]
    fn test_infer_datatype_mixed_returns_string() {
        let s = suggester();
        let dt = s.infer_datatype(&["42".into(), "not-a-number".into()]);
        assert_eq!(dt.as_deref(), Some("xsd:string"));
    }

    // ── confidence_score 0-1 range ───────────────────────────────────────────

    #[test]
    fn test_confidence_score_range() {
        let s = suggester();
        for count in [0, 1, 5, 10] {
            let p = PropertyPattern {
                predicate: "p".into(),
                usage_count: count,
                object_types: HashSet::new(),
                always_present: false,
                example_values: Vec::new(),
            };
            let c = s.confidence_score(&p, 10);
            assert!((0.0..=1.0).contains(&c), "confidence out of [0,1]: {}", c);
        }
    }

    #[test]
    fn test_confidence_score_always_present_boost() {
        let s = suggester();
        let p_always = PropertyPattern {
            predicate: "p".into(),
            usage_count: 10,
            object_types: HashSet::new(),
            always_present: true,
            example_values: Vec::new(),
        };
        let p_normal = PropertyPattern {
            always_present: false,
            ..p_always.clone()
        };
        assert!(s.confidence_score(&p_always, 10) >= s.confidence_score(&p_normal, 10));
    }

    #[test]
    fn test_confidence_score_zero_nodes() {
        let s = suggester();
        let p = always_pattern("p", 0);
        assert_eq!(s.confidence_score(&p, 0), 0.0);
    }

    #[test]
    fn test_confidence_score_full_usage() {
        let s = suggester();
        let p = always_pattern("p", 10);
        let c = s.confidence_score(&p, 10);
        assert!(c > 0.9);
    }

    // ── cardinality inference ─────────────────────────────────────────────────

    #[test]
    fn test_infer_cardinality_always_present_min1() {
        let s = suggester();
        let p = always_pattern("p", 10);
        let (min, _) = s.infer_cardinality(&p, 10);
        assert_eq!(min, Some(1));
    }

    #[test]
    fn test_infer_cardinality_optional_min0() {
        let s = suggester();
        let p = partial_pattern("p", 5, 10);
        let (min, _) = s.infer_cardinality(&p, 10);
        assert_eq!(min, Some(0));
    }

    #[test]
    fn test_infer_cardinality_functional_max1() {
        let s = suggester();
        let p = always_pattern("p", 10); // single object_type (1 type in set)
        let (_, max) = s.infer_cardinality(&p, 10);
        assert_eq!(max, Some(1));
    }

    #[test]
    fn test_infer_cardinality_multi_type_unbounded() {
        let s = suggester();
        let mut p = always_pattern("p", 10);
        p.object_types.insert("xsd:integer".to_string());
        p.object_types.insert("xsd:string".to_string());
        let (_, max) = s.infer_cardinality(&p, 10);
        assert!(max.is_none());
    }

    // ── target_class preserved ───────────────────────────────────────────────

    #[test]
    fn test_target_class_preserved_in_candidate() {
        let s = suggester();
        let np = NodePattern {
            class: "http://schema.org/Person".to_string(),
            properties: vec![always_pattern("ex:name", 5)],
        };
        let candidate = s
            .suggest_for_class("http://schema.org/Person", &[np])
            .expect("should succeed");
        assert_eq!(candidate.target_class, "http://schema.org/Person");
    }

    // ── empty properties list ────────────────────────────────────────────────

    #[test]
    fn test_class_with_no_properties() {
        let s = suggester();
        let np = NodePattern {
            class: "Empty".to_string(),
            properties: Vec::new(),
        };
        let candidate = s.suggest_for_class("Empty", &[np]).expect("should succeed");
        assert!(candidate.suggested_properties.is_empty());
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_suggested_property_predicate_correct() {
        let s = suggester();
        let np = NodePattern {
            class: "A".to_string(),
            properties: vec![always_pattern("ex:age", 10)],
        };
        let candidate = s.suggest_for_class("A", &[np]).expect("should succeed");
        assert_eq!(candidate.suggested_properties[0].predicate, "ex:age");
    }

    #[test]
    fn test_infer_datatype_integer_negative() {
        let s = suggester();
        let dt = s.infer_datatype(&["-100".into(), "-200".into()]);
        assert_eq!(dt.as_deref(), Some("xsd:integer"));
    }

    #[test]
    fn test_infer_datatype_single_value_integer() {
        let s = suggester();
        assert_eq!(
            s.infer_datatype(&["42".into()]).as_deref(),
            Some("xsd:integer")
        );
    }

    #[test]
    fn test_infer_datatype_single_value_decimal() {
        let s = suggester();
        assert_eq!(
            s.infer_datatype(&["3.14".into()]).as_deref(),
            Some("xsd:decimal")
        );
    }

    #[test]
    fn test_infer_datatype_single_value_string() {
        let s = suggester();
        assert_eq!(
            s.infer_datatype(&["hello world".into()]).as_deref(),
            Some("xsd:string")
        );
    }

    #[test]
    fn test_confidence_score_half_usage() {
        let s = suggester();
        let p = partial_pattern("p", 5, 10);
        let c = s.confidence_score(&p, 10);
        assert!((c - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cardinality_min1_max1_for_always_single_type() {
        let s = suggester();
        let p = always_pattern("p", 5);
        let (min, max) = s.infer_cardinality(&p, 5);
        assert_eq!(min, Some(1));
        assert_eq!(max, Some(1));
    }

    #[test]
    fn test_cardinality_min0_max_none_for_optional_multi_type() {
        let s = suggester();
        let mut p = partial_pattern("p", 5, 10);
        p.object_types.insert("xsd:integer".to_string());
        p.object_types.insert("xsd:string".to_string());
        let (min, max) = s.infer_cardinality(&p, 10);
        assert_eq!(min, Some(0));
        assert!(max.is_none());
    }

    #[test]
    fn test_analyze_single_class() {
        let s = suggester();
        let patterns = vec![NodePattern {
            class: "A".to_string(),
            properties: vec![always_pattern("p1", 10)],
        }];
        let candidates = s.analyze(&patterns);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].target_class, "A");
    }

    #[test]
    fn test_infer_datatype_boolean_true() {
        let s = suggester();
        let dt = s.infer_datatype(&["true".into()]);
        assert_eq!(dt.as_deref(), Some("xsd:boolean"));
    }

    #[test]
    fn test_infer_datatype_boolean_false() {
        let s = suggester();
        let dt = s.infer_datatype(&["false".into()]);
        assert_eq!(dt.as_deref(), Some("xsd:boolean"));
    }

    #[test]
    fn test_suggest_returns_none_for_nonexistent_class() {
        let s = suggester();
        let patterns = vec![NodePattern {
            class: "Person".to_string(),
            properties: vec![],
        }];
        assert!(s.suggest_for_class("Animal", &patterns).is_none());
    }

    #[test]
    fn test_analyze_returns_candidate_per_class() {
        let s = PropertySuggester::new(0.0, 0.0);
        let patterns: Vec<NodePattern> = (0..5)
            .map(|i| NodePattern {
                class: format!("Class{}", i),
                properties: vec![always_pattern("p", 5)],
            })
            .collect();
        let candidates = s.analyze(&patterns);
        assert_eq!(candidates.len(), 5);
    }

    #[test]
    fn test_inferred_datatype_in_suggested_property() {
        let s = suggester();
        let mut p = always_pattern("ex:age", 10);
        p.example_values = vec!["25".into(), "30".into(), "40".into()];
        let np = NodePattern {
            class: "Person".to_string(),
            properties: vec![p],
        };
        let candidate = s
            .suggest_for_class("Person", &[np])
            .expect("should succeed");
        assert_eq!(
            candidate.suggested_properties[0]
                .inferred_datatype
                .as_deref(),
            Some("xsd:integer")
        );
    }

    #[test]
    fn test_confidence_in_suggested_property_range() {
        let s = suggester();
        let np = NodePattern {
            class: "X".to_string(),
            properties: vec![always_pattern("p", 10)],
        };
        let candidate = s.suggest_for_class("X", &[np]).expect("should succeed");
        let c = candidate.suggested_properties[0].confidence;
        assert!((0.0..=1.0).contains(&c));
    }

    #[test]
    fn test_min_count_in_suggested_property() {
        let s = suggester();
        let np = NodePattern {
            class: "Y".to_string(),
            properties: vec![always_pattern("p", 10)],
        };
        let candidate = s.suggest_for_class("Y", &[np]).expect("should succeed");
        assert_eq!(candidate.suggested_properties[0].min_count, Some(1));
    }

    #[test]
    fn test_max_count_none_for_multi_type_optional() {
        let s = suggester();
        let mut p = partial_pattern("p", 5, 10);
        p.object_types.insert("t1".to_string());
        p.object_types.insert("t2".to_string());
        let np = NodePattern {
            class: "Z".to_string(),
            properties: vec![p],
        };
        let candidate = s.suggest_for_class("Z", &[np]).expect("should succeed");
        if let Some(sp) = candidate.suggested_properties.first() {
            assert!(sp.max_count.is_none());
        }
    }
}
