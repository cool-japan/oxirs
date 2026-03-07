//! Automated SHACL constraint synthesis from data samples.
//!
//! `ConstraintSynthesizer` analyses a collection of `DataSample` values and
//! generates SHACL-compatible constraints with confidence scores.  Constraints
//! whose confidence falls below `min_confidence` or which are derived from
//! fewer than `min_samples` observations are discarded.

use std::collections::HashMap;

// ──────────────────────────────────────────────────────────────────────────────
// Types
// ──────────────────────────────────────────────────────────────────────────────

/// A single observation: one node has the given property with one or more values.
#[derive(Debug, Clone, PartialEq)]
pub struct DataSample {
    /// Node IRI or blank-node label.
    pub node: String,
    /// Property IRI.
    pub property: String,
    /// Observed values for this (node, property) pair.
    pub values: Vec<String>,
}

impl DataSample {
    /// Convenience constructor.
    pub fn new(
        node: impl Into<String>,
        property: impl Into<String>,
        values: Vec<impl Into<String>>,
    ) -> Self {
        Self {
            node: node.into(),
            property: property.into(),
            values: values.into_iter().map(Into::into).collect(),
        }
    }
}

/// A generated SHACL-compatible constraint.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthesizedConstraint {
    /// Property IRI the constraint applies to.
    pub property: String,
    /// Kind of constraint.
    pub constraint_type: ConstraintType,
    /// Fraction of samples that support this constraint (0.0–1.0).
    pub confidence: f64,
}

/// Supported SHACL constraint shapes.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintType {
    /// sh:minCount
    MinCount(usize),
    /// sh:maxCount
    MaxCount(usize),
    /// sh:datatype
    Datatype(String),
    /// sh:pattern (simple regex-like string)
    Pattern(String),
    /// sh:minInclusive
    MinInclusive(f64),
    /// sh:maxInclusive
    MaxInclusive(f64),
    /// sh:nodeKind ("IRI", "Literal", "BlankNode", …)
    NodeKind(String),
    /// sh:in — closed vocabulary
    In(Vec<String>),
}

// ──────────────────────────────────────────────────────────────────────────────
// ConstraintSynthesizer
// ──────────────────────────────────────────────────────────────────────────────

/// Analyses data samples and generates probable SHACL constraints.
#[derive(Debug, Clone)]
pub struct ConstraintSynthesizer {
    /// Minimum confidence fraction required to emit a constraint.
    pub min_confidence: f64,
    /// Minimum number of distinct nodes needed to derive a constraint.
    pub min_samples: usize,
}

impl ConstraintSynthesizer {
    /// Create a synthesizer with the given thresholds.
    ///
    /// * `min_confidence` – fraction of samples that must support a constraint
    ///   before it is emitted (0.0–1.0).
    /// * `min_samples` – minimum number of distinct nodes observed for a
    ///   property before constraints are generated.
    pub fn new(min_confidence: f64, min_samples: usize) -> Self {
        Self {
            min_confidence,
            min_samples,
        }
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Generate constraints from a slice of data samples.
    pub fn synthesize(&self, samples: &[DataSample]) -> Vec<SynthesizedConstraint> {
        if samples.is_empty() {
            return Vec::new();
        }

        // Group samples by property.
        let mut by_property: HashMap<&str, Vec<&DataSample>> = HashMap::new();
        for sample in samples {
            by_property
                .entry(sample.property.as_str())
                .or_default()
                .push(sample);
        }

        let mut constraints = Vec::new();

        for (property, group) in &by_property {
            if group.len() < self.min_samples {
                continue;
            }

            // Collect all values across nodes.
            let all_values: Vec<&str> = group
                .iter()
                .flat_map(|s| s.values.iter().map(String::as_str))
                .collect();

            let total_nodes = group.len() as f64;

            // ── cardinality constraints ───────────────────────────────────────
            let (min_count, max_count) = self.infer_cardinality(samples, property);

            // MinCount: supported when all observed nodes have >= min_count values.
            if min_count > 0 {
                let supporting = group.iter().filter(|s| s.values.len() >= min_count).count();
                let confidence = supporting as f64 / total_nodes;
                if confidence >= self.min_confidence {
                    constraints.push(SynthesizedConstraint {
                        property: property.to_string(),
                        constraint_type: ConstraintType::MinCount(min_count),
                        confidence,
                    });
                }
            }

            // MaxCount: supported when all nodes have <= max_count values.
            if max_count > 0 {
                let supporting = group.iter().filter(|s| s.values.len() <= max_count).count();
                let confidence = supporting as f64 / total_nodes;
                if confidence >= self.min_confidence {
                    constraints.push(SynthesizedConstraint {
                        property: property.to_string(),
                        constraint_type: ConstraintType::MaxCount(max_count),
                        confidence,
                    });
                }
            }

            // ── datatype inference ────────────────────────────────────────────
            if let Some(dtype) = self.infer_datatype(&all_values) {
                let supporting = group
                    .iter()
                    .filter(|s| {
                        s.values
                            .iter()
                            .all(|v| Self::value_matches_datatype(v, &dtype))
                    })
                    .count();
                let confidence = supporting as f64 / total_nodes;
                if confidence >= self.min_confidence {
                    constraints.push(SynthesizedConstraint {
                        property: property.to_string(),
                        constraint_type: ConstraintType::Datatype(dtype),
                        confidence,
                    });
                }
            }

            // ── pattern inference ─────────────────────────────────────────────
            if let Some(pattern) = self.infer_pattern(&all_values) {
                let supporting = group
                    .iter()
                    .filter(|s| s.values.iter().all(|v| Self::matches_pattern(v, &pattern)))
                    .count();
                let confidence = supporting as f64 / total_nodes;
                if confidence >= self.min_confidence {
                    constraints.push(SynthesizedConstraint {
                        property: property.to_string(),
                        constraint_type: ConstraintType::Pattern(pattern),
                        confidence,
                    });
                }
            }

            // ── numeric range inference ───────────────────────────────────────
            if let Some((min_val, max_val)) = self.infer_range(&all_values) {
                let supporting_min = group
                    .iter()
                    .filter(|s| {
                        s.values
                            .iter()
                            .all(|v| v.parse::<f64>().is_ok_and(|n| n >= min_val))
                    })
                    .count();
                let conf_min = supporting_min as f64 / total_nodes;
                if conf_min >= self.min_confidence {
                    constraints.push(SynthesizedConstraint {
                        property: property.to_string(),
                        constraint_type: ConstraintType::MinInclusive(min_val),
                        confidence: conf_min,
                    });
                }

                let supporting_max = group
                    .iter()
                    .filter(|s| {
                        s.values
                            .iter()
                            .all(|v| v.parse::<f64>().is_ok_and(|n| n <= max_val))
                    })
                    .count();
                let conf_max = supporting_max as f64 / total_nodes;
                if conf_max >= self.min_confidence {
                    constraints.push(SynthesizedConstraint {
                        property: property.to_string(),
                        constraint_type: ConstraintType::MaxInclusive(max_val),
                        confidence: conf_max,
                    });
                }
            }

            // ── node kind inference ───────────────────────────────────────────
            let node_kind = Self::infer_node_kind(&all_values);
            {
                let supporting = group
                    .iter()
                    .filter(|s| {
                        s.values
                            .iter()
                            .all(|v| Self::value_node_kind(v) == node_kind)
                    })
                    .count();
                let confidence = supporting as f64 / total_nodes;
                if confidence >= self.min_confidence {
                    constraints.push(SynthesizedConstraint {
                        property: property.to_string(),
                        constraint_type: ConstraintType::NodeKind(node_kind.to_string()),
                        confidence,
                    });
                }
            }

            // ── closed vocabulary ─────────────────────────────────────────────
            if let Some(vocab) = self.is_closed_vocabulary(&all_values) {
                let supporting = group
                    .iter()
                    .filter(|s| s.values.iter().all(|v| vocab.contains(v)))
                    .count();
                let confidence = supporting as f64 / total_nodes;
                if confidence >= self.min_confidence {
                    constraints.push(SynthesizedConstraint {
                        property: property.to_string(),
                        constraint_type: ConstraintType::In(vocab),
                        confidence,
                    });
                }
            }
        }

        constraints
    }

    /// Infer the XSD datatype for a set of values.
    ///
    /// Returns:
    /// - `"xsd:integer"` if all values parse as integers.
    /// - `"xsd:decimal"` if all values parse as floats (but not all integers).
    /// - `"xsd:boolean"` if all values are `"true"` or `"false"`.
    /// - `"xsd:anyURI"` if all values look like IRIs (`http://…` or `https://…`).
    /// - `None` otherwise.
    pub fn infer_datatype(&self, values: &[&str]) -> Option<String> {
        if values.is_empty() {
            return None;
        }

        if values.iter().all(|v| v.parse::<i64>().is_ok()) {
            return Some("xsd:integer".to_owned());
        }

        if values.iter().all(|v| v.parse::<f64>().is_ok()) {
            return Some("xsd:decimal".to_owned());
        }

        if values
            .iter()
            .all(|v| *v == "true" || *v == "false" || *v == "1" || *v == "0")
        {
            // Only emit boolean if they are literal true/false strings
            if values.iter().all(|v| *v == "true" || *v == "false") {
                return Some("xsd:boolean".to_owned());
            }
        }

        if values
            .iter()
            .all(|v| v.starts_with("http://") || v.starts_with("https://"))
        {
            return Some("xsd:anyURI".to_owned());
        }

        None
    }

    /// Infer a simple character-class pattern shared by all values.
    ///
    /// Returns:
    /// - `"^[0-9]+$"` if all values are digit-only.
    /// - `"^[a-zA-Z]+$"` if all values are letter-only.
    /// - `"^[a-zA-Z0-9]+$"` if all values are alphanumeric.
    /// - `None` otherwise.
    pub fn infer_pattern(&self, values: &[&str]) -> Option<String> {
        if values.is_empty() {
            return None;
        }

        if values
            .iter()
            .all(|v| !v.is_empty() && v.chars().all(|c| c.is_ascii_digit()))
        {
            return Some("^[0-9]+$".to_owned());
        }

        if values
            .iter()
            .all(|v| !v.is_empty() && v.chars().all(|c| c.is_ascii_alphabetic()))
        {
            return Some("^[a-zA-Z]+$".to_owned());
        }

        if values
            .iter()
            .all(|v| !v.is_empty() && v.chars().all(|c| c.is_ascii_alphanumeric()))
        {
            return Some("^[a-zA-Z0-9]+$".to_owned());
        }

        None
    }

    /// Infer the numeric range `(min, max)` observed across all values.
    ///
    /// Returns `None` if any value fails to parse as `f64`.
    pub fn infer_range(&self, values: &[&str]) -> Option<(f64, f64)> {
        if values.is_empty() {
            return None;
        }

        let nums: Vec<f64> = values
            .iter()
            .map(|v| v.parse::<f64>().ok())
            .collect::<Option<Vec<_>>>()?;

        let min = nums.iter().copied().fold(f64::INFINITY, f64::min);
        let max = nums.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        Some((min, max))
    }

    /// Compute the observed (min_count, max_count) of `property` values across
    /// all samples that mention the property.
    ///
    /// Returns `(0, 0)` if no relevant samples exist.
    pub fn infer_cardinality(&self, samples: &[DataSample], property: &str) -> (usize, usize) {
        let counts: Vec<usize> = samples
            .iter()
            .filter(|s| s.property == property)
            .map(|s| s.values.len())
            .collect();

        if counts.is_empty() {
            return (0, 0);
        }

        let min = counts.iter().copied().min().unwrap_or(0);
        let max = counts.iter().copied().max().unwrap_or(0);
        (min, max)
    }

    /// Determine whether the values form a closed vocabulary.
    ///
    /// Returns the unique set of values if there are no more than 10 distinct
    /// values and each distinct value appears more than once (i.e. the set is
    /// reused); returns `None` otherwise.
    pub fn is_closed_vocabulary(&self, values: &[&str]) -> Option<Vec<String>> {
        if values.is_empty() {
            return None;
        }

        let mut freq: HashMap<&str, usize> = HashMap::new();
        for v in values {
            *freq.entry(v).or_insert(0) += 1;
        }

        let distinct_count = freq.len();

        // Closed vocabulary heuristic: ≤ 10 distinct values, each repeated.
        if distinct_count <= 10 && freq.values().all(|&c| c > 1) {
            let mut vocab: Vec<String> = freq.keys().map(|s| s.to_string()).collect();
            vocab.sort_unstable();
            Some(vocab)
        } else {
            None
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn value_matches_datatype(value: &str, dtype: &str) -> bool {
        match dtype {
            "xsd:integer" => value.parse::<i64>().is_ok(),
            "xsd:decimal" => value.parse::<f64>().is_ok(),
            "xsd:boolean" => value == "true" || value == "false",
            "xsd:anyURI" => value.starts_with("http://") || value.starts_with("https://"),
            _ => true,
        }
    }

    fn matches_pattern(value: &str, pattern: &str) -> bool {
        match pattern {
            "^[0-9]+$" => !value.is_empty() && value.chars().all(|c| c.is_ascii_digit()),
            "^[a-zA-Z]+$" => !value.is_empty() && value.chars().all(|c| c.is_ascii_alphabetic()),
            "^[a-zA-Z0-9]+$" => {
                !value.is_empty() && value.chars().all(|c| c.is_ascii_alphanumeric())
            }
            _ => true,
        }
    }

    fn value_node_kind(value: &str) -> &'static str {
        if value.starts_with("http://") || value.starts_with("https://") {
            "IRI"
        } else if value.starts_with("_:") {
            "BlankNode"
        } else {
            "Literal"
        }
    }

    fn infer_node_kind(values: &[&str]) -> &'static str {
        if values.is_empty() {
            return "Literal";
        }
        let kinds: Vec<&str> = values.iter().map(|v| Self::value_node_kind(v)).collect();
        if kinds.iter().all(|k| *k == "IRI") {
            "IRI"
        } else if kinds.iter().all(|k| *k == "BlankNode") {
            "BlankNode"
        } else {
            "Literal"
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(min_conf: f64, min_samples: usize) -> ConstraintSynthesizer {
        ConstraintSynthesizer::new(min_conf, min_samples)
    }

    fn sample(node: &str, prop: &str, values: &[&str]) -> DataSample {
        DataSample::new(node, prop, values.to_vec())
    }

    // ── DataSample ────────────────────────────────────────────────────────────

    #[test]
    fn test_data_sample_new() {
        let s = sample("node1", "prop1", &["v1", "v2"]);
        assert_eq!(s.node, "node1");
        assert_eq!(s.property, "prop1");
        assert_eq!(s.values, vec!["v1", "v2"]);
    }

    #[test]
    fn test_data_sample_empty_values() {
        let s = sample("n", "p", &[]);
        assert!(s.values.is_empty());
    }

    // ── infer_datatype ────────────────────────────────────────────────────────

    #[test]
    fn test_infer_datatype_integer() {
        let s = synth(0.5, 1);
        let vals = &["1", "2", "-3", "100"];
        assert_eq!(s.infer_datatype(vals), Some("xsd:integer".to_string()));
    }

    #[test]
    fn test_infer_datatype_decimal() {
        let s = synth(0.5, 1);
        let vals = &["1.5", "2.7", "-0.1"];
        assert_eq!(s.infer_datatype(vals), Some("xsd:decimal".to_string()));
    }

    #[test]
    fn test_infer_datatype_boolean() {
        let s = synth(0.5, 1);
        let vals = &["true", "false", "true"];
        assert_eq!(s.infer_datatype(vals), Some("xsd:boolean".to_string()));
    }

    #[test]
    fn test_infer_datatype_uri() {
        let s = synth(0.5, 1);
        let vals = &["http://example.org/a", "https://example.org/b"];
        assert_eq!(s.infer_datatype(vals), Some("xsd:anyURI".to_string()));
    }

    #[test]
    fn test_infer_datatype_mixed_returns_none() {
        let s = synth(0.5, 1);
        let vals = &["hello", "42", "true"];
        assert!(s.infer_datatype(vals).is_none());
    }

    #[test]
    fn test_infer_datatype_empty() {
        let s = synth(0.5, 1);
        assert!(s.infer_datatype(&[]).is_none());
    }

    // ── infer_pattern ─────────────────────────────────────────────────────────

    #[test]
    fn test_infer_pattern_digits() {
        let s = synth(0.5, 1);
        let vals = &["123", "456", "789"];
        assert_eq!(s.infer_pattern(vals), Some("^[0-9]+$".to_string()));
    }

    #[test]
    fn test_infer_pattern_alpha() {
        let s = synth(0.5, 1);
        let vals = &["abc", "DEF", "xyz"];
        assert_eq!(s.infer_pattern(vals), Some("^[a-zA-Z]+$".to_string()));
    }

    #[test]
    fn test_infer_pattern_alphanumeric() {
        let s = synth(0.5, 1);
        let vals = &["abc123", "XYZ789"];
        assert_eq!(s.infer_pattern(vals), Some("^[a-zA-Z0-9]+$".to_string()));
    }

    #[test]
    fn test_infer_pattern_mixed_returns_none() {
        let s = synth(0.5, 1);
        let vals = &["hello world", "foo-bar"];
        assert!(s.infer_pattern(vals).is_none());
    }

    #[test]
    fn test_infer_pattern_empty() {
        let s = synth(0.5, 1);
        assert!(s.infer_pattern(&[]).is_none());
    }

    // ── infer_range ───────────────────────────────────────────────────────────

    #[test]
    fn test_infer_range_integers() {
        let s = synth(0.5, 1);
        let vals = &["3", "1", "5", "2"];
        let (min, max) = s.infer_range(vals).unwrap();
        assert!((min - 1.0).abs() < 1e-9);
        assert!((max - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_infer_range_floats() {
        let s = synth(0.5, 1);
        let vals = &["-1.5", "0.0", "2.5"];
        let (min, max) = s.infer_range(vals).unwrap();
        assert!((min - (-1.5)).abs() < 1e-9);
        assert!((max - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_infer_range_non_numeric_returns_none() {
        let s = synth(0.5, 1);
        let vals = &["foo", "bar"];
        assert!(s.infer_range(vals).is_none());
    }

    #[test]
    fn test_infer_range_empty() {
        let s = synth(0.5, 1);
        assert!(s.infer_range(&[]).is_none());
    }

    #[test]
    fn test_infer_range_single_value() {
        let s = synth(0.5, 1);
        let (min, max) = s.infer_range(&["42"]).unwrap();
        assert!((min - 42.0).abs() < 1e-9);
        assert!((max - 42.0).abs() < 1e-9);
    }

    // ── infer_cardinality ─────────────────────────────────────────────────────

    #[test]
    fn test_infer_cardinality_basic() {
        let s = synth(0.5, 1);
        let samples = vec![
            sample("n1", "p", &["a"]),
            sample("n2", "p", &["b", "c"]),
            sample("n3", "p", &["d", "e", "f"]),
        ];
        let (min, max) = s.infer_cardinality(&samples, "p");
        assert_eq!(min, 1);
        assert_eq!(max, 3);
    }

    #[test]
    fn test_infer_cardinality_no_samples() {
        let s = synth(0.5, 1);
        let (min, max) = s.infer_cardinality(&[], "p");
        assert_eq!((min, max), (0, 0));
    }

    #[test]
    fn test_infer_cardinality_different_property_ignored() {
        let s = synth(0.5, 1);
        let samples = vec![
            sample("n1", "p", &["a"]),
            sample("n2", "q", &["b", "c", "d"]),
        ];
        let (min, max) = s.infer_cardinality(&samples, "p");
        assert_eq!((min, max), (1, 1));
    }

    #[test]
    fn test_infer_cardinality_empty_values() {
        let s = synth(0.5, 1);
        let samples = vec![sample("n1", "p", &[]), sample("n2", "p", &["a"])];
        let (min, max) = s.infer_cardinality(&samples, "p");
        assert_eq!(min, 0);
        assert_eq!(max, 1);
    }

    // ── is_closed_vocabulary ──────────────────────────────────────────────────

    #[test]
    fn test_is_closed_vocabulary_yes() {
        let s = synth(0.5, 1);
        let vals = &["low", "medium", "high", "low", "medium", "high"];
        let vocab = s.is_closed_vocabulary(vals);
        assert!(vocab.is_some());
        let v = vocab.unwrap();
        assert!(v.contains(&"low".to_string()));
        assert!(v.contains(&"medium".to_string()));
        assert!(v.contains(&"high".to_string()));
    }

    #[test]
    fn test_is_closed_vocabulary_too_many_distinct() {
        let s = synth(0.5, 1);
        let vals: Vec<String> = (0..11).map(|i| format!("val{}", i)).collect();
        let refs: Vec<&str> = vals.iter().map(String::as_str).collect();
        assert!(s.is_closed_vocabulary(&refs).is_none());
    }

    #[test]
    fn test_is_closed_vocabulary_single_occurrence() {
        let s = synth(0.5, 1);
        // Each value appears only once → not a closed vocabulary
        let vals = &["a", "b", "c"];
        assert!(s.is_closed_vocabulary(vals).is_none());
    }

    #[test]
    fn test_is_closed_vocabulary_empty() {
        let s = synth(0.5, 1);
        assert!(s.is_closed_vocabulary(&[]).is_none());
    }

    // ── synthesize ────────────────────────────────────────────────────────────

    #[test]
    fn test_synthesize_empty() {
        let s = synth(0.5, 1);
        assert!(s.synthesize(&[]).is_empty());
    }

    #[test]
    fn test_synthesize_below_min_samples_skipped() {
        let s = synth(0.5, 5); // need at least 5 nodes
        let samples = vec![sample("n1", "age", &["25"]), sample("n2", "age", &["30"])];
        let constraints = s.synthesize(&samples);
        // Only 2 samples < 5 min_samples → no constraints
        assert!(constraints.is_empty());
    }

    #[test]
    fn test_synthesize_integer_datatype() {
        let s = synth(0.8, 2);
        let samples: Vec<DataSample> = (0..5)
            .map(|i| sample(&format!("n{}", i), "age", &[&format!("{}", 20 + i)]))
            .collect();
        let constraints = s.synthesize(&samples);
        let has_integer = constraints.iter().any(
            |c| matches!(&c.constraint_type, ConstraintType::Datatype(dt) if dt == "xsd:integer"),
        );
        assert!(has_integer);
    }

    #[test]
    fn test_synthesize_min_max_count() {
        let s = synth(0.9, 2);
        let samples: Vec<DataSample> = (0..4)
            .map(|i| sample(&format!("n{}", i), "tags", &["a", "b"]))
            .collect();
        let constraints = s.synthesize(&samples);
        let has_min = constraints
            .iter()
            .any(|c| matches!(c.constraint_type, ConstraintType::MinCount(2)));
        let has_max = constraints
            .iter()
            .any(|c| matches!(c.constraint_type, ConstraintType::MaxCount(2)));
        assert!(has_min, "expected MinCount(2)");
        assert!(has_max, "expected MaxCount(2)");
    }

    #[test]
    fn test_synthesize_iri_node_kind() {
        let s = synth(0.8, 2);
        let samples: Vec<DataSample> = (0..3)
            .map(|i| {
                sample(
                    &format!("n{}", i),
                    "seeAlso",
                    &[&format!("http://example.org/{}", i)],
                )
            })
            .collect();
        let constraints = s.synthesize(&samples);
        let has_iri = constraints
            .iter()
            .any(|c| matches!(&c.constraint_type, ConstraintType::NodeKind(k) if k == "IRI"));
        assert!(has_iri);
    }

    #[test]
    fn test_synthesize_confidence_above_threshold() {
        let s = synth(0.5, 1);
        let samples: Vec<DataSample> = (0..4)
            .map(|i| sample(&format!("n{}", i), "count", &[&format!("{}", i)]))
            .collect();
        let constraints = s.synthesize(&samples);
        for c in &constraints {
            assert!(
                c.confidence >= 0.5,
                "confidence {} < min_confidence",
                c.confidence
            );
        }
    }

    #[test]
    fn test_synthesize_closed_vocabulary() {
        let s = synth(0.7, 2);
        let values_by_node = [
            ("n1", "status", vec!["active"]),
            ("n2", "status", vec!["inactive"]),
            ("n3", "status", vec!["active"]),
            ("n4", "status", vec!["inactive"]),
            ("n5", "status", vec!["active"]),
        ];
        let samples: Vec<DataSample> = values_by_node
            .iter()
            .map(|(n, p, v)| sample(n, p, &v.to_vec()))
            .collect();
        let constraints = s.synthesize(&samples);
        let has_in = constraints
            .iter()
            .any(|c| matches!(&c.constraint_type, ConstraintType::In(v) if v.contains(&"active".to_string())));
        assert!(has_in, "expected closed-vocabulary (In) constraint");
    }

    #[test]
    fn test_synthesize_uri_datatype() {
        let s = synth(0.8, 2);
        let samples: Vec<DataSample> = (0..4)
            .map(|i| {
                sample(
                    &format!("n{}", i),
                    "homepage",
                    &[&format!("https://example.org/page{}", i)],
                )
            })
            .collect();
        let constraints = s.synthesize(&samples);
        let has_uri = constraints.iter().any(
            |c| matches!(&c.constraint_type, ConstraintType::Datatype(dt) if dt == "xsd:anyURI"),
        );
        assert!(has_uri);
    }

    #[test]
    fn test_synthesize_multiple_properties() {
        let s = synth(0.5, 2);
        let samples = vec![
            sample("n1", "age", &["25"]),
            sample("n2", "age", &["30"]),
            sample("n1", "name", &["Alice"]),
            sample("n2", "name", &["Bob"]),
        ];
        let constraints = s.synthesize(&samples);
        let props: std::collections::HashSet<&str> =
            constraints.iter().map(|c| c.property.as_str()).collect();
        assert!(props.contains("age"));
        assert!(props.contains("name"));
    }
}
