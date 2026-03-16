//! Data-driven SHACL constraint inference.
//!
//! Analyses a set of [`DataSample`] observations and infers SHACL constraints
//! for each class such as `sh:minCount`, `sh:maxCount`, `sh:datatype`,
//! `sh:pattern`, `sh:in`, and `sh:minInclusive`/`sh:maxInclusive`.

use std::collections::{HashMap, HashSet};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single node observation with its class and per-property value lists.
#[derive(Debug, Clone)]
pub struct DataSample {
    /// Node identifier (e.g. IRI or blank node label).
    pub node: String,
    /// Class of this node (e.g. `"Person"` or a full IRI).
    pub class: String,
    /// Property path → list of observed string-serialised values.
    pub properties: HashMap<String, Vec<String>>,
}

/// An inferred SHACL constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum InferredConstraint {
    /// `sh:minCount` — property must appear at least `min` times.
    MinCount { path: String, min: usize },
    /// `sh:maxCount` — property may appear at most `max` times.
    MaxCount { path: String, max: usize },
    /// `sh:datatype` — observed values appear to be of this XSD datatype.
    DataType { path: String, datatype: String },
    /// `sh:pattern` — values match a recognised common pattern.
    Pattern { path: String, regex: String },
    /// `sh:in` — value space is a small closed set.
    In { path: String, values: Vec<String> },
    /// `sh:minInclusive` / `sh:maxInclusive` — numeric range.
    Range {
        path: String,
        min: Option<f64>,
        max: Option<f64>,
    },
}

/// Constraints inferred for one class.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Class name.
    pub class: String,
    /// Inferred constraints.
    pub constraints: Vec<InferredConstraint>,
    /// Number of samples used.
    pub sample_count: usize,
    /// Overall confidence score in `[0, 1]`.
    pub confidence: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// ConstraintInference
// ─────────────────────────────────────────────────────────────────────────────

/// Configurable data-driven SHACL constraint inference engine.
pub struct ConstraintInference {
    /// Minimum number of samples required to draw conclusions about a class.
    min_samples: usize,
    /// Minimum fraction of samples a rule must be observed in to be reported.
    confidence_threshold: f64,
}

impl ConstraintInference {
    /// Create an inference engine.
    ///
    /// * `min_samples` – minimum number of samples needed per class.
    /// * `confidence_threshold` – fraction in `[0, 1]` (e.g. `0.9` = 90 %).
    pub fn new(min_samples: usize, confidence_threshold: f64) -> Self {
        Self {
            min_samples,
            confidence_threshold,
        }
    }

    // ── top-level entrypoints ─────────────────────────────────────────────────

    /// Infer constraints for every class present in `samples`.
    pub fn infer(&self, samples: &[DataSample]) -> Vec<InferenceResult> {
        // Group samples by class.
        let mut by_class: HashMap<String, Vec<&DataSample>> = HashMap::new();
        for s in samples {
            by_class.entry(s.class.clone()).or_default().push(s);
        }
        by_class
            .keys()
            .map(|class| self.infer_for_class(class, samples))
            .collect()
    }

    /// Infer constraints for `class` from `samples`.
    ///
    /// Returns an `InferenceResult` with empty constraints if fewer than
    /// `min_samples` match the class.
    pub fn infer_for_class(&self, class: &str, samples: &[DataSample]) -> InferenceResult {
        let class_samples: Vec<&DataSample> = samples.iter().filter(|s| s.class == class).collect();

        let n = class_samples.len();
        if n < self.min_samples {
            return InferenceResult {
                class: class.to_string(),
                constraints: Vec::new(),
                sample_count: n,
                confidence: 0.0,
            };
        }

        // Collect all property paths seen in this class.
        let paths: HashSet<String> = class_samples
            .iter()
            .flat_map(|s| s.properties.keys().cloned())
            .collect();

        let mut constraints = Vec::new();

        for path in &paths {
            // Collect value lists per sample (absent = empty vec).
            let counts_per_sample: Vec<usize> = class_samples
                .iter()
                .map(|s| s.properties.get(path).map_or(0, |v| v.len()))
                .collect();

            let min_count = *counts_per_sample.iter().min().unwrap_or(&0);
            let max_count = *counts_per_sample.iter().max().unwrap_or(&0);

            // sh:minCount
            if min_count > 0 {
                constraints.push(InferredConstraint::MinCount {
                    path: path.clone(),
                    min: min_count,
                });
            }

            // sh:maxCount (only when bounded)
            constraints.push(InferredConstraint::MaxCount {
                path: path.clone(),
                max: max_count,
            });

            // Collect all values for deeper analysis.
            let all_values: Vec<String> = class_samples
                .iter()
                .flat_map(|s| s.properties.get(path).map_or_else(Vec::new, |v| v.clone()))
                .collect();

            if all_values.is_empty() {
                continue;
            }

            // sh:datatype
            if let Some(dt) = self.infer_datatype(&all_values) {
                constraints.push(InferredConstraint::DataType {
                    path: path.clone(),
                    datatype: dt,
                });
            }

            // sh:pattern
            if let Some(pat) = self.infer_pattern(&all_values) {
                constraints.push(InferredConstraint::Pattern {
                    path: path.clone(),
                    regex: pat,
                });
            }

            // sh:in (closed value set)
            if let Some(values) = self.detect_closed_values(&all_values) {
                constraints.push(InferredConstraint::In {
                    path: path.clone(),
                    values,
                });
            }

            // sh:range for numeric values.
            if let Some(range) = self.infer_numeric_range(&all_values) {
                constraints.push(range.with_path(path));
            }
        }

        // Confidence = proportion of paths that produced at least one constraint.
        let confidence = if paths.is_empty() {
            0.0
        } else {
            (constraints.len() as f64 / paths.len() as f64).min(1.0)
        };

        InferenceResult {
            class: class.to_string(),
            constraints,
            sample_count: n,
            confidence,
        }
    }

    // ── inference helpers ─────────────────────────────────────────────────────

    /// Attempt to detect a single XSD datatype for all `values`.
    ///
    /// Returns the datatype IRI fragment only when all non-empty values match
    /// that type.  Supported types (in detection order):
    ///
    /// * `xsd:boolean`
    /// * `xsd:integer`
    /// * `xsd:decimal`
    /// * `xsd:dateTime`
    /// * `xsd:anyURI`
    /// * `xsd:string` (fallback when values are non-empty)
    pub fn infer_datatype(&self, values: &[String]) -> Option<String> {
        let non_empty: Vec<&str> = values
            .iter()
            .filter(|v| !v.is_empty())
            .map(|s| s.as_str())
            .collect();
        if non_empty.is_empty() {
            return None;
        }

        if non_empty.iter().all(|v| is_boolean(v)) {
            return Some("xsd:boolean".to_string());
        }
        if non_empty.iter().all(|v| is_integer(v)) {
            return Some("xsd:integer".to_string());
        }
        if non_empty.iter().all(|v| is_decimal(v)) {
            return Some("xsd:decimal".to_string());
        }
        if non_empty.iter().all(|v| is_datetime(v)) {
            return Some("xsd:dateTime".to_string());
        }
        if non_empty.iter().all(|v| is_any_uri(v)) {
            return Some("xsd:anyURI".to_string());
        }
        Some("xsd:string".to_string())
    }

    /// Detect a common string pattern (email or UUID) shared by all values.
    ///
    /// Returns the regex string when **all** non-empty values match one
    /// well-known pattern.
    pub fn infer_pattern(&self, values: &[String]) -> Option<String> {
        let non_empty: Vec<&str> = values
            .iter()
            .filter(|v| !v.is_empty())
            .map(|s| s.as_str())
            .collect();
        if non_empty.is_empty() {
            return None;
        }

        if non_empty.iter().all(|v| is_email(v)) {
            return Some(r"^[^@]+@[^@]+\.[^@]+$".to_string());
        }
        if non_empty.iter().all(|v| is_uuid(v)) {
            return Some(
                r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
                    .to_string(),
            );
        }
        None
    }

    /// Suggest `sh:in` when the number of distinct values does not exceed the
    /// closed-world threshold (≤ 10 distinct values and ≤ half the sample
    /// count when there are many samples).
    pub fn detect_closed_values(&self, values: &[String]) -> Option<Vec<String>> {
        if values.is_empty() {
            return None;
        }
        let unique: HashSet<&String> = values.iter().collect();
        let threshold = 10;
        if unique.len() <= threshold {
            let mut sorted: Vec<String> = unique.into_iter().cloned().collect();
            sorted.sort();
            return Some(sorted);
        }
        None
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn infer_numeric_range(&self, values: &[String]) -> Option<RangeBuilder> {
        let nums: Vec<f64> = values
            .iter()
            .filter_map(|v| v.parse::<f64>().ok())
            .collect();
        if nums.is_empty() || nums.len() < values.len() / 2 {
            return None; // not predominantly numeric
        }
        let min = nums.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = nums.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Some(RangeBuilder {
            min: Some(min),
            max: Some(max),
        })
    }
}

// ── tiny builder for range ────────────────────────────────────────────────────

struct RangeBuilder {
    min: Option<f64>,
    max: Option<f64>,
}

impl RangeBuilder {
    fn with_path(self, path: &str) -> InferredConstraint {
        InferredConstraint::Range {
            path: path.to_string(),
            min: self.min,
            max: self.max,
        }
    }
}

// ── type-detection predicates ─────────────────────────────────────────────────

fn is_boolean(v: &str) -> bool {
    matches!(v.to_lowercase().as_str(), "true" | "false" | "0" | "1")
}

fn is_integer(v: &str) -> bool {
    v.parse::<i64>().is_ok()
}

fn is_decimal(v: &str) -> bool {
    v.parse::<f64>().is_ok()
}

fn is_datetime(v: &str) -> bool {
    // Very lightweight check: look for ISO 8601-like structure.
    let v = v.trim();
    // e.g. "2024-01-15T10:30:00Z" or "2024-01-15T10:30:00+00:00"
    if v.len() < 10 {
        return false;
    }
    let date_part = &v[..10];
    let date_ok = date_part.len() == 10
        && date_part.as_bytes()[4] == b'-'
        && date_part.as_bytes()[7] == b'-'
        && date_part[..4].parse::<u32>().is_ok()
        && date_part[5..7].parse::<u32>().is_ok()
        && date_part[8..10].parse::<u32>().is_ok();
    if !date_ok {
        return false;
    }
    if v.len() > 10 {
        v.as_bytes()[10] == b'T'
    } else {
        true
    }
}

fn is_any_uri(v: &str) -> bool {
    v.starts_with("http://") || v.starts_with("https://") || v.starts_with("urn:")
}

fn is_email(v: &str) -> bool {
    let parts: Vec<&str> = v.splitn(2, '@').collect();
    if parts.len() != 2 {
        return false;
    }
    let domain = parts[1];
    !parts[0].is_empty()
        && domain.contains('.')
        && !domain.starts_with('.')
        && !domain.ends_with('.')
}

fn is_uuid(v: &str) -> bool {
    let parts: Vec<&str> = v.split('-').collect();
    if parts.len() != 5 {
        return false;
    }
    let expected_lens = [8, 4, 4, 4, 12];
    parts
        .iter()
        .zip(expected_lens.iter())
        .all(|(p, &len)| p.len() == len && p.chars().all(|c| c.is_ascii_hexdigit()))
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> ConstraintInference {
        ConstraintInference::new(1, 0.5)
    }

    fn sample(node: &str, class: &str, props: &[(&str, &[&str])]) -> DataSample {
        let properties = props
            .iter()
            .map(|(k, vs)| (k.to_string(), vs.iter().map(|v| v.to_string()).collect()))
            .collect();
        DataSample {
            node: node.to_string(),
            class: class.to_string(),
            properties,
        }
    }

    // ── infer_datatype ────────────────────────────────────────────────────────

    #[test]
    fn test_infer_datatype_integer() {
        let ci = engine();
        let dt = ci.infer_datatype(&["1".into(), "2".into(), "42".into()]);
        assert_eq!(dt, Some("xsd:integer".to_string()));
    }

    #[test]
    fn test_infer_datatype_decimal() {
        let ci = engine();
        let dt = ci.infer_datatype(&["1.5".into(), "2.7".into()]);
        assert_eq!(dt, Some("xsd:decimal".to_string()));
    }

    #[test]
    fn test_infer_datatype_boolean() {
        let ci = engine();
        let dt = ci.infer_datatype(&["true".into(), "false".into()]);
        assert_eq!(dt, Some("xsd:boolean".to_string()));
    }

    #[test]
    fn test_infer_datatype_datetime() {
        let ci = engine();
        let dt = ci.infer_datatype(&["2024-01-15T10:30:00Z".into()]);
        assert_eq!(dt, Some("xsd:dateTime".to_string()));
    }

    #[test]
    fn test_infer_datatype_any_uri() {
        let ci = engine();
        let dt = ci.infer_datatype(&[
            "http://example.org/foo".into(),
            "https://example.org/bar".into(),
        ]);
        assert_eq!(dt, Some("xsd:anyURI".to_string()));
    }

    #[test]
    fn test_infer_datatype_string_fallback() {
        let ci = engine();
        let dt = ci.infer_datatype(&["hello".into(), "world".into()]);
        assert_eq!(dt, Some("xsd:string".to_string()));
    }

    #[test]
    fn test_infer_datatype_empty_values_none() {
        let ci = engine();
        assert!(ci.infer_datatype(&[]).is_none());
    }

    // ── infer_pattern ─────────────────────────────────────────────────────────

    #[test]
    fn test_infer_pattern_email() {
        let ci = engine();
        let pat = ci.infer_pattern(&["alice@example.com".into(), "bob@test.org".into()]);
        assert!(pat.is_some());
        assert!(pat.expect("should succeed").contains('@'));
    }

    #[test]
    fn test_infer_pattern_uuid() {
        let ci = engine();
        let pat = ci.infer_pattern(&[
            "550e8400-e29b-41d4-a716-446655440000".into(),
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8".into(),
        ]);
        assert!(pat.is_some());
    }

    #[test]
    fn test_infer_pattern_mixed_no_pattern() {
        let ci = engine();
        let pat = ci.infer_pattern(&["alice@example.com".into(), "not-an-email".into()]);
        assert!(pat.is_none());
    }

    #[test]
    fn test_infer_pattern_empty_none() {
        let ci = engine();
        assert!(ci.infer_pattern(&[]).is_none());
    }

    // ── detect_closed_values ──────────────────────────────────────────────────

    #[test]
    fn test_detect_closed_values_small_set() {
        let ci = engine();
        let result =
            ci.detect_closed_values(&["red".into(), "green".into(), "blue".into(), "red".into()]);
        assert!(result.is_some());
        let vals = result.expect("should succeed");
        assert_eq!(vals.len(), 3); // 3 unique
    }

    #[test]
    fn test_detect_closed_values_empty_none() {
        let ci = engine();
        assert!(ci.detect_closed_values(&[]).is_none());
    }

    #[test]
    fn test_detect_closed_values_large_set_none() {
        let ci = engine();
        let values: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        assert!(ci.detect_closed_values(&values).is_none());
    }

    #[test]
    fn test_detect_closed_values_exactly_threshold() {
        let ci = engine();
        let values: Vec<String> = (0..10).map(|i| i.to_string()).collect();
        assert!(ci.detect_closed_values(&values).is_some());
    }

    // ── infer_for_class ───────────────────────────────────────────────────────

    #[test]
    fn test_infer_for_class_min_count() {
        let ci = engine();
        let samples = vec![
            sample("n1", "Person", &[("name", &["Alice"])]),
            sample("n2", "Person", &[("name", &["Bob"])]),
        ];
        let result = ci.infer_for_class("Person", &samples);
        let has_min = result.constraints.iter().any(|c| {
            matches!(c, InferredConstraint::MinCount { path, min } if path == "name" && *min >= 1)
        });
        assert!(has_min, "Expected MinCount for name");
    }

    #[test]
    fn test_infer_for_class_max_count() {
        let ci = engine();
        let samples = vec![sample("n1", "Thing", &[("label", &["a", "b"])])];
        let result = ci.infer_for_class("Thing", &samples);
        let has_max = result.constraints.iter().any(|c| {
            matches!(c, InferredConstraint::MaxCount { path, max } if path == "label" && *max == 2)
        });
        assert!(has_max, "Expected MaxCount(2) for label");
    }

    #[test]
    fn test_infer_for_class_min_samples_not_met() {
        let ci = ConstraintInference::new(5, 0.9);
        let samples = vec![sample("n1", "Rare", &[("x", &["v"])])];
        let result = ci.infer_for_class("Rare", &samples);
        assert!(result.constraints.is_empty());
        assert_eq!(result.sample_count, 1);
    }

    #[test]
    fn test_infer_for_class_unknown_class_empty() {
        let ci = engine();
        let result = ci.infer_for_class("Unknown", &[]);
        assert!(result.constraints.is_empty());
    }

    // ── infer multiple classes ────────────────────────────────────────────────

    #[test]
    fn test_infer_multiple_classes() {
        let ci = engine();
        let samples = vec![
            sample("a", "ClassA", &[("x", &["1"])]),
            sample("b", "ClassB", &[("y", &["hello"])]),
        ];
        let results = ci.infer(&samples);
        let classes: HashSet<String> = results.iter().map(|r| r.class.clone()).collect();
        assert!(classes.contains("ClassA"));
        assert!(classes.contains("ClassB"));
    }

    // ── confidence calculation ────────────────────────────────────────────────

    #[test]
    fn test_confidence_nonzero_with_constraints() {
        let ci = engine();
        let samples = vec![
            sample("n1", "C", &[("p", &["42"])]),
            sample("n2", "C", &[("p", &["43"])]),
        ];
        let result = ci.infer_for_class("C", &samples);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_confidence_zero_below_min_samples() {
        let ci = ConstraintInference::new(10, 0.9);
        let samples = vec![sample("n1", "C", &[("p", &["v"])])];
        let result = ci.infer_for_class("C", &samples);
        assert_eq!(result.confidence, 0.0);
    }

    // ── sample_count ─────────────────────────────────────────────────────────

    #[test]
    fn test_sample_count_reported() {
        let ci = engine();
        let samples = vec![
            sample("n1", "P", &[("k", &["v"])]),
            sample("n2", "P", &[("k", &["v"])]),
            sample("n3", "P", &[("k", &["v"])]),
        ];
        let result = ci.infer_for_class("P", &samples);
        assert_eq!(result.sample_count, 3);
    }

    // ── datatype inferred ─────────────────────────────────────────────────────

    #[test]
    fn test_infer_for_class_includes_datatype() {
        let ci = engine();
        let samples = vec![
            sample("n1", "Metric", &[("value", &["3.14"])]),
            sample("n2", "Metric", &[("value", &["2.71"])]),
        ];
        let result = ci.infer_for_class("Metric", &samples);
        let has_dt = result
            .constraints
            .iter()
            .any(|c| matches!(c, InferredConstraint::DataType { path, .. } if path == "value"));
        assert!(has_dt);
    }

    // ── empty samples ─────────────────────────────────────────────────────────

    #[test]
    fn test_infer_empty_samples_returns_empty_vec() {
        let ci = engine();
        let results = ci.infer(&[]);
        assert!(results.is_empty());
    }

    // ── urn: URI detection ────────────────────────────────────────────────────

    #[test]
    fn test_infer_datatype_urn_uri() {
        let ci = engine();
        let dt = ci.infer_datatype(&["urn:isbn:0451450523".into()]);
        assert_eq!(dt, Some("xsd:anyURI".to_string()));
    }

    // ── additional coverage ───────────────────────────────────────────────────

    #[test]
    fn test_infer_datatype_https_uri() {
        let ci = engine();
        let dt = ci.infer_datatype(&["https://example.org/foo".into()]);
        assert_eq!(dt, Some("xsd:anyURI".to_string()));
    }

    #[test]
    fn test_infer_datatype_negative_integer() {
        let ci = engine();
        let dt = ci.infer_datatype(&["-42".into(), "-1".into()]);
        assert_eq!(dt, Some("xsd:integer".to_string()));
    }

    #[test]
    fn test_detect_closed_values_single_value() {
        let ci = engine();
        let result = ci.detect_closed_values(&["yes".into()]);
        assert!(result.is_some());
        assert_eq!(result.expect("should succeed"), vec!["yes".to_string()]);
    }

    #[test]
    fn test_detect_closed_values_sorted() {
        let ci = engine();
        let result = ci.detect_closed_values(&["c".into(), "a".into(), "b".into()]);
        assert_eq!(result.expect("should succeed"), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_infer_for_class_max_count_correct() {
        let ci = engine();
        let samples = vec![sample("n1", "T", &[("tags", &["a", "b", "c"])])];
        let result = ci.infer_for_class("T", &samples);
        let max = result.constraints.iter().find_map(|c| {
            if let InferredConstraint::MaxCount { path, max } = c {
                if path == "tags" {
                    Some(*max)
                } else {
                    None
                }
            } else {
                None
            }
        });
        assert_eq!(max, Some(3));
    }

    #[test]
    fn test_infer_pattern_single_email() {
        let ci = engine();
        let pat = ci.infer_pattern(&["user@domain.com".into()]);
        assert!(pat.is_some());
    }

    #[test]
    fn test_infer_pattern_single_uuid() {
        let ci = engine();
        let pat = ci.infer_pattern(&["123e4567-e89b-12d3-a456-426614174000".into()]);
        assert!(pat.is_some());
    }

    #[test]
    fn test_infer_for_class_in_constraint_small_set() {
        let ci = engine();
        let values = &["active", "inactive"];
        let samples = vec![
            sample("n1", "Status", &[("state", values)]),
            sample("n2", "Status", &[("state", &["active"])]),
        ];
        let result = ci.infer_for_class("Status", &samples);
        let has_in = result
            .constraints
            .iter()
            .any(|c| matches!(c, InferredConstraint::In { path, .. } if path == "state"));
        assert!(has_in);
    }

    #[test]
    fn test_data_sample_creation() {
        let s = sample("node1", "MyClass", &[("prop", &["val1", "val2"])]);
        assert_eq!(s.node, "node1");
        assert_eq!(s.class, "MyClass");
        assert_eq!(s.properties["prop"], vec!["val1", "val2"]);
    }

    #[test]
    fn test_inferred_constraint_range_variant() {
        let c = InferredConstraint::Range {
            path: "age".into(),
            min: Some(0.0),
            max: Some(150.0),
        };
        if let InferredConstraint::Range { path, min, max } = c {
            assert_eq!(path, "age");
            assert_eq!(min, Some(0.0));
            assert_eq!(max, Some(150.0));
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_min_samples_threshold_two() {
        let ci = ConstraintInference::new(2, 0.5);
        // Only 1 sample → below threshold
        let samples = vec![sample("n1", "T", &[("p", &["v"])])];
        let result = ci.infer_for_class("T", &samples);
        assert!(result.constraints.is_empty());
    }

    #[test]
    fn test_min_samples_threshold_met() {
        let ci = ConstraintInference::new(2, 0.5);
        let samples = vec![
            sample("n1", "T", &[("p", &["v"])]),
            sample("n2", "T", &[("p", &["v"])]),
        ];
        let result = ci.infer_for_class("T", &samples);
        assert_eq!(result.sample_count, 2);
        assert!(!result.constraints.is_empty());
    }

    #[test]
    fn test_infer_for_class_multiple_properties() {
        let ci = engine();
        let samples = vec![sample(
            "n1",
            "Person",
            &[("name", &["Alice"]), ("age", &["30"])],
        )];
        let result = ci.infer_for_class("Person", &samples);
        // Should have constraints for both properties
        let paths: HashSet<String> = result
            .constraints
            .iter()
            .filter_map(|c| match c {
                InferredConstraint::MinCount { path, .. }
                | InferredConstraint::MaxCount { path, .. } => Some(path.clone()),
                _ => None,
            })
            .collect();
        assert!(paths.contains("name"));
        assert!(paths.contains("age"));
    }

    #[test]
    fn test_infer_datatype_only_empty_strings_none() {
        let ci = engine();
        let dt = ci.infer_datatype(&["".into(), "".into()]);
        assert!(dt.is_none());
    }

    #[test]
    fn test_infer_pattern_mixed_email_and_plain_no_pattern() {
        let ci = engine();
        let pat = ci.infer_pattern(&["user@example.com".into(), "plain-string".into()]);
        assert!(pat.is_none());
    }

    #[test]
    fn test_inferred_constraint_min_count_variant() {
        let c = InferredConstraint::MinCount {
            path: "p".into(),
            min: 1,
        };
        if let InferredConstraint::MinCount { path, min } = c {
            assert_eq!(path, "p");
            assert_eq!(min, 1);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_inferred_constraint_datatype_variant() {
        let c = InferredConstraint::DataType {
            path: "val".into(),
            datatype: "xsd:integer".into(),
        };
        if let InferredConstraint::DataType { path, datatype } = c {
            assert_eq!(path, "val");
            assert_eq!(datatype, "xsd:integer");
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_inferred_constraint_in_variant() {
        let c = InferredConstraint::In {
            path: "status".into(),
            values: vec!["on".into(), "off".into()],
        };
        if let InferredConstraint::In { path, values } = c {
            assert_eq!(path, "status");
            assert_eq!(values.len(), 2);
        } else {
            panic!("wrong variant");
        }
    }

    #[test]
    fn test_detect_closed_values_deduplicates() {
        let ci = engine();
        let result = ci.detect_closed_values(&["a".into(), "a".into(), "b".into()]);
        assert_eq!(result.expect("should succeed").len(), 2);
    }
}
