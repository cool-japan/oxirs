//! # Explainable AI for SHACL Violations
//!
//! Generates human-readable explanations for why a SHACL constraint failed,
//! including root cause analysis, fix suggestions, and confidence scores.
//!
//! ## Features
//!
//! - **Natural language explanations**: Convert constraint violations to readable text
//! - **Root cause analysis**: Trace violation to originating data issue
//! - **Fix suggestions**: Recommend concrete data changes to resolve violations
//! - **Severity classification**: Rate violations by impact
//! - **Explanation templates**: Customisable explanation patterns
//! - **Batch explanation**: Explain multiple violations efficiently

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─────────────────────────────────────────────
// Violation Types
// ─────────────────────────────────────────────

/// Type of SHACL constraint that was violated.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintType {
    MinCount,
    MaxCount,
    DataType,
    MinLength,
    MaxLength,
    Pattern,
    MinInclusive,
    MaxInclusive,
    MinExclusive,
    MaxExclusive,
    NodeKind,
    Class,
    In,
    HasValue,
    ClosedShape,
    DisjointProperties,
    Uniqueness,
    QualifiedValueShape,
    Custom(String),
}

impl ConstraintType {
    /// Human-readable name.
    pub fn display_name(&self) -> &str {
        match self {
            ConstraintType::MinCount => "Minimum Count",
            ConstraintType::MaxCount => "Maximum Count",
            ConstraintType::DataType => "Data Type",
            ConstraintType::MinLength => "Minimum Length",
            ConstraintType::MaxLength => "Maximum Length",
            ConstraintType::Pattern => "Pattern Match",
            ConstraintType::MinInclusive => "Minimum Value (inclusive)",
            ConstraintType::MaxInclusive => "Maximum Value (inclusive)",
            ConstraintType::MinExclusive => "Minimum Value (exclusive)",
            ConstraintType::MaxExclusive => "Maximum Value (exclusive)",
            ConstraintType::NodeKind => "Node Kind",
            ConstraintType::Class => "Class Membership",
            ConstraintType::In => "Allowed Values",
            ConstraintType::HasValue => "Required Value",
            ConstraintType::ClosedShape => "Closed Shape",
            ConstraintType::DisjointProperties => "Disjoint Properties",
            ConstraintType::Uniqueness => "Uniqueness",
            ConstraintType::QualifiedValueShape => "Qualified Value Shape",
            ConstraintType::Custom(name) => name.as_str(),
        }
    }
}

/// Severity of a violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ViolationSeverity {
    /// Informational — data quality hint.
    Info,
    /// Warning — may cause issues downstream.
    Warning,
    /// Error — breaks data contract, must be fixed.
    Error,
    /// Critical — security or integrity impact.
    Critical,
}

// ─────────────────────────────────────────────
// Violation Input
// ─────────────────────────────────────────────

/// A SHACL constraint violation to be explained.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShaclViolation {
    /// The focus node URI that failed validation.
    pub focus_node: String,
    /// The property path that was validated.
    pub result_path: Option<String>,
    /// The constraint type that was violated.
    pub constraint_type: ConstraintType,
    /// The expected value/pattern from the shape.
    pub expected: Option<String>,
    /// The actual value found in the data.
    pub actual: Option<String>,
    /// The shape that defined the constraint.
    pub source_shape: Option<String>,
    /// SHACL severity (sh:Violation, sh:Warning, sh:Info).
    pub shacl_severity: Option<String>,
    /// Additional context key-value pairs.
    pub context: HashMap<String, String>,
}

// ─────────────────────────────────────────────
// Explanation Output
// ─────────────────────────────────────────────

/// Explanation of a SHACL violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationExplanation {
    /// Human-readable summary (1-2 sentences).
    pub summary: String,
    /// Detailed explanation.
    pub detail: String,
    /// Root cause description.
    pub root_cause: String,
    /// Suggested fixes (ordered by preference).
    pub suggested_fixes: Vec<SuggestedFix>,
    /// Severity classification.
    pub severity: ViolationSeverity,
    /// Confidence in the explanation (0.0-1.0).
    pub confidence: f64,
    /// Related violations that share the same root cause.
    pub related_violation_indices: Vec<usize>,
    /// Tags for categorisation.
    pub tags: Vec<String>,
}

/// A suggested fix for a violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedFix {
    /// Description of the fix.
    pub description: String,
    /// SPARQL UPDATE query to apply the fix (if applicable).
    pub sparql_update: Option<String>,
    /// Estimated effort to apply (Low, Medium, High).
    pub effort: FixEffort,
    /// Whether this fix is automatically applicable.
    pub auto_applicable: bool,
}

/// Estimated effort for a fix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FixEffort {
    Low,
    Medium,
    High,
}

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

/// Configuration for the explainability engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainabilityConfig {
    /// Whether to generate SPARQL UPDATE fix suggestions (default: true).
    pub generate_sparql_fixes: bool,
    /// Whether to perform root cause grouping (default: true).
    pub group_related_violations: bool,
    /// Language for explanations (default: "en").
    pub language: String,
    /// Maximum number of suggested fixes per violation (default: 3).
    pub max_fixes: usize,
    /// Whether to include verbose detail (default: false).
    pub verbose: bool,
}

impl Default for ExplainabilityConfig {
    fn default() -> Self {
        Self {
            generate_sparql_fixes: true,
            group_related_violations: true,
            language: "en".to_string(),
            max_fixes: 3,
            verbose: false,
        }
    }
}

// ─────────────────────────────────────────────
// Statistics
// ─────────────────────────────────────────────

/// Statistics for the explainability engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExplainabilityStats {
    /// Total violations explained.
    pub violations_explained: u64,
    /// Violations by constraint type.
    pub by_constraint_type: HashMap<String, u64>,
    /// Violations by severity.
    pub info_count: u64,
    pub warning_count: u64,
    pub error_count: u64,
    pub critical_count: u64,
    /// Total fixes suggested.
    pub fixes_suggested: u64,
    /// Auto-applicable fixes.
    pub auto_fixes: u64,
}

// ─────────────────────────────────────────────
// Explainability Engine
// ─────────────────────────────────────────────

/// Engine for generating human-readable SHACL violation explanations.
pub struct ViolationExplainer {
    config: ExplainabilityConfig,
    stats: ExplainabilityStats,
}

impl ViolationExplainer {
    /// Create a new explainer.
    pub fn new(config: ExplainabilityConfig) -> Self {
        Self {
            config,
            stats: ExplainabilityStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ExplainabilityConfig::default())
    }

    /// Explain a single SHACL violation.
    pub fn explain(&mut self, violation: &ShaclViolation) -> ViolationExplanation {
        let severity = self.classify_severity(violation);
        let summary = self.generate_summary(violation);
        let detail = self.generate_detail(violation);
        let root_cause = self.identify_root_cause(violation);
        let fixes = self.suggest_fixes(violation);

        self.stats.violations_explained += 1;
        let type_name = violation.constraint_type.display_name().to_string();
        *self.stats.by_constraint_type.entry(type_name).or_insert(0) += 1;

        match severity {
            ViolationSeverity::Info => self.stats.info_count += 1,
            ViolationSeverity::Warning => self.stats.warning_count += 1,
            ViolationSeverity::Error => self.stats.error_count += 1,
            ViolationSeverity::Critical => self.stats.critical_count += 1,
        }

        self.stats.fixes_suggested += fixes.len() as u64;
        self.stats.auto_fixes += fixes.iter().filter(|f| f.auto_applicable).count() as u64;

        ViolationExplanation {
            summary,
            detail,
            root_cause,
            suggested_fixes: fixes,
            severity,
            confidence: self.compute_confidence(violation),
            related_violation_indices: Vec::new(),
            tags: self.generate_tags(violation),
        }
    }

    /// Explain multiple violations with root cause grouping.
    pub fn explain_batch(&mut self, violations: &[ShaclViolation]) -> Vec<ViolationExplanation> {
        let mut explanations: Vec<ViolationExplanation> =
            violations.iter().map(|v| self.explain(v)).collect();

        if self.config.group_related_violations {
            self.group_related(&mut explanations, violations);
        }

        explanations
    }

    /// Get statistics.
    pub fn stats(&self) -> &ExplainabilityStats {
        &self.stats
    }

    /// Get configuration.
    pub fn config(&self) -> &ExplainabilityConfig {
        &self.config
    }

    // ─── Internal ────────────────────────────

    fn classify_severity(&self, violation: &ShaclViolation) -> ViolationSeverity {
        // Use SHACL severity if available
        if let Some(ref sev) = violation.shacl_severity {
            let lower = sev.to_lowercase();
            if lower.contains("info") {
                return ViolationSeverity::Info;
            } else if lower.contains("warning") {
                return ViolationSeverity::Warning;
            }
        }

        // Heuristic-based severity
        match &violation.constraint_type {
            ConstraintType::DataType | ConstraintType::NodeKind => ViolationSeverity::Error,
            ConstraintType::MinCount | ConstraintType::MaxCount => ViolationSeverity::Error,
            ConstraintType::Class => ViolationSeverity::Error,
            ConstraintType::Pattern | ConstraintType::MinLength | ConstraintType::MaxLength => {
                ViolationSeverity::Warning
            }
            ConstraintType::ClosedShape => ViolationSeverity::Warning,
            ConstraintType::Uniqueness | ConstraintType::DisjointProperties => {
                ViolationSeverity::Critical
            }
            _ => ViolationSeverity::Warning,
        }
    }

    fn generate_summary(&self, v: &ShaclViolation) -> String {
        let path_str = v.result_path.as_deref().unwrap_or("(unknown property)");
        let node_short = shorten_uri(&v.focus_node);

        match &v.constraint_type {
            ConstraintType::MinCount => {
                let expected = v.expected.as_deref().unwrap_or("1");
                let actual = v.actual.as_deref().unwrap_or("0");
                format!(
                    "Node '{node_short}' has too few values for property '{path_str}': \
                     found {actual}, expected at least {expected}."
                )
            }
            ConstraintType::MaxCount => {
                let expected = v.expected.as_deref().unwrap_or("1");
                let actual = v.actual.as_deref().unwrap_or("?");
                format!(
                    "Node '{node_short}' has too many values for property '{path_str}': \
                     found {actual}, expected at most {expected}."
                )
            }
            ConstraintType::DataType => {
                let expected = v.expected.as_deref().unwrap_or("(expected type)");
                let actual = v.actual.as_deref().unwrap_or("(actual type)");
                format!(
                    "Property '{path_str}' on node '{node_short}' has wrong data type: \
                     expected {expected}, got {actual}."
                )
            }
            ConstraintType::Pattern => {
                let pattern = v.expected.as_deref().unwrap_or("(pattern)");
                format!(
                    "Value of '{path_str}' on node '{node_short}' does not match \
                     required pattern '{pattern}'."
                )
            }
            ConstraintType::Class => {
                let expected = v.expected.as_deref().unwrap_or("(class)");
                format!("Node '{node_short}' is not an instance of required class '{expected}'.")
            }
            ConstraintType::MinInclusive | ConstraintType::MinExclusive => {
                let min = v.expected.as_deref().unwrap_or("?");
                let actual = v.actual.as_deref().unwrap_or("?");
                format!("Value {actual} of '{path_str}' on '{node_short}' is below minimum {min}.")
            }
            ConstraintType::MaxInclusive | ConstraintType::MaxExclusive => {
                let max = v.expected.as_deref().unwrap_or("?");
                let actual = v.actual.as_deref().unwrap_or("?");
                format!("Value {actual} of '{path_str}' on '{node_short}' exceeds maximum {max}.")
            }
            _ => {
                format!(
                    "Constraint '{}' violated on node '{node_short}' at property '{path_str}'.",
                    v.constraint_type.display_name()
                )
            }
        }
    }

    fn generate_detail(&self, v: &ShaclViolation) -> String {
        let mut detail = String::new();
        detail.push_str(&format!("Focus Node: {}\n", v.focus_node));
        if let Some(ref path) = v.result_path {
            detail.push_str(&format!("Property Path: {path}\n"));
        }
        detail.push_str(&format!(
            "Constraint: {}\n",
            v.constraint_type.display_name()
        ));
        if let Some(ref expected) = v.expected {
            detail.push_str(&format!("Expected: {expected}\n"));
        }
        if let Some(ref actual) = v.actual {
            detail.push_str(&format!("Actual: {actual}\n"));
        }
        if let Some(ref shape) = v.source_shape {
            detail.push_str(&format!("Source Shape: {shape}\n"));
        }
        detail
    }

    fn identify_root_cause(&self, v: &ShaclViolation) -> String {
        match &v.constraint_type {
            ConstraintType::MinCount => {
                "Missing required data — the property has fewer values than required by the shape."
                    .to_string()
            }
            ConstraintType::MaxCount => {
                "Duplicate or excess data — the property has more values than allowed.".to_string()
            }
            ConstraintType::DataType => {
                "Type mismatch — the literal value has an incompatible XSD datatype.".to_string()
            }
            ConstraintType::Pattern => {
                "Format violation — the value does not conform to the expected string pattern."
                    .to_string()
            }
            ConstraintType::Class => {
                "Classification error — the node is not typed as the required class.".to_string()
            }
            ConstraintType::Uniqueness => {
                "Duplicate value — a value that should be unique appears on multiple nodes."
                    .to_string()
            }
            ConstraintType::ClosedShape => {
                "Extra property — the node has a property not allowed by the closed shape."
                    .to_string()
            }
            _ => "The data does not conform to the constraint defined in the SHACL shape."
                .to_string(),
        }
    }

    fn suggest_fixes(&self, v: &ShaclViolation) -> Vec<SuggestedFix> {
        let mut fixes = Vec::new();
        let max = self.config.max_fixes;

        match &v.constraint_type {
            ConstraintType::MinCount => {
                if let Some(ref path) = v.result_path {
                    if self.config.generate_sparql_fixes {
                        fixes.push(SuggestedFix {
                            description: format!("Add the missing value(s) for property '{path}'"),
                            sparql_update: Some(format!(
                                "INSERT DATA {{ <{}> <{path}> \"VALUE_HERE\" . }}",
                                v.focus_node
                            )),
                            effort: FixEffort::Low,
                            auto_applicable: false,
                        });
                    }
                }
            }
            ConstraintType::MaxCount => {
                if let Some(ref path) = v.result_path {
                    fixes.push(SuggestedFix {
                        description: format!(
                            "Remove excess value(s) for property '{path}'"
                        ),
                        sparql_update: if self.config.generate_sparql_fixes {
                            Some(format!(
                                "DELETE {{ <{}> <{path}> ?v . }} WHERE {{ <{}> <{path}> ?v . }} LIMIT 1",
                                v.focus_node, v.focus_node
                            ))
                        } else {
                            None
                        },
                        effort: FixEffort::Low,
                        auto_applicable: false,
                    });
                }
            }
            ConstraintType::DataType => {
                if let (Some(ref path), Some(ref expected)) = (&v.result_path, &v.expected) {
                    fixes.push(SuggestedFix {
                        description: format!("Cast the value to the correct datatype '{expected}'"),
                        sparql_update: None,
                        effort: FixEffort::Medium,
                        auto_applicable: true,
                    });
                }
            }
            ConstraintType::Class => {
                if let Some(ref expected) = v.expected {
                    fixes.push(SuggestedFix {
                        description: format!("Add rdf:type assertion for class '{expected}'"),
                        sparql_update: if self.config.generate_sparql_fixes {
                            Some(format!(
                                "INSERT DATA {{ <{}> a <{expected}> . }}",
                                v.focus_node
                            ))
                        } else {
                            None
                        },
                        effort: FixEffort::Low,
                        auto_applicable: true,
                    });
                }
            }
            _ => {
                fixes.push(SuggestedFix {
                    description: format!(
                        "Review and correct the {} constraint violation",
                        v.constraint_type.display_name()
                    ),
                    sparql_update: None,
                    effort: FixEffort::Medium,
                    auto_applicable: false,
                });
            }
        }

        fixes.truncate(max);
        fixes
    }

    fn compute_confidence(&self, v: &ShaclViolation) -> f64 {
        let mut confidence: f64 = 0.5;

        // Higher confidence when we have more context
        if v.expected.is_some() {
            confidence += 0.15;
        }
        if v.actual.is_some() {
            confidence += 0.15;
        }
        if v.result_path.is_some() {
            confidence += 0.1;
        }
        if v.source_shape.is_some() {
            confidence += 0.1;
        }

        confidence.min(1.0)
    }

    fn generate_tags(&self, v: &ShaclViolation) -> Vec<String> {
        let mut tags = vec![v.constraint_type.display_name().to_string()];
        if v.constraint_type == ConstraintType::MinCount
            || v.constraint_type == ConstraintType::MaxCount
        {
            tags.push("cardinality".to_string());
        }
        if matches!(
            v.constraint_type,
            ConstraintType::MinInclusive
                | ConstraintType::MaxInclusive
                | ConstraintType::MinExclusive
                | ConstraintType::MaxExclusive
        ) {
            tags.push("range".to_string());
        }
        tags
    }

    fn group_related(
        &self,
        explanations: &mut [ViolationExplanation],
        violations: &[ShaclViolation],
    ) {
        // Group violations by focus node and constraint type
        let mut groups: HashMap<(String, String), Vec<usize>> = HashMap::new();
        for (i, v) in violations.iter().enumerate() {
            let key = (
                v.focus_node.clone(),
                v.constraint_type.display_name().to_string(),
            );
            groups.entry(key).or_default().push(i);
        }

        for indices in groups.values() {
            if indices.len() > 1 {
                for &i in indices {
                    if i < explanations.len() {
                        explanations[i].related_violation_indices =
                            indices.iter().filter(|&&j| j != i).copied().collect();
                    }
                }
            }
        }
    }
}

/// Shorten a URI to its local name.
fn shorten_uri(uri: &str) -> String {
    // Check for '#' first — in RDF URIs the fragment identifier always
    // follows the path, so `rfind('#')` gives the true local-name boundary
    // even when '/' appears earlier in the scheme or authority.
    if let Some(idx) = uri.rfind('#') {
        uri[idx + 1..].to_string()
    } else if let Some(idx) = uri.rfind('/') {
        uri[idx + 1..].to_string()
    } else {
        uri.to_string()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn min_count_violation() -> ShaclViolation {
        ShaclViolation {
            focus_node: "http://example.org/person/42".into(),
            result_path: Some("http://schema.org/name".into()),
            constraint_type: ConstraintType::MinCount,
            expected: Some("1".into()),
            actual: Some("0".into()),
            source_shape: Some("http://example.org/shapes/PersonShape".into()),
            shacl_severity: None,
            context: HashMap::new(),
        }
    }

    fn datatype_violation() -> ShaclViolation {
        ShaclViolation {
            focus_node: "http://example.org/sensor/1".into(),
            result_path: Some("http://example.org/value".into()),
            constraint_type: ConstraintType::DataType,
            expected: Some("xsd:decimal".into()),
            actual: Some("xsd:string".into()),
            source_shape: None,
            shacl_severity: None,
            context: HashMap::new(),
        }
    }

    fn class_violation() -> ShaclViolation {
        ShaclViolation {
            focus_node: "http://example.org/item/99".into(),
            result_path: None,
            constraint_type: ConstraintType::Class,
            expected: Some("http://schema.org/Product".into()),
            actual: None,
            source_shape: None,
            shacl_severity: None,
            context: HashMap::new(),
        }
    }

    #[test]
    fn test_explain_min_count() {
        let mut explainer = ViolationExplainer::with_defaults();
        let explanation = explainer.explain(&min_count_violation());
        assert!(explanation.summary.contains("too few"));
        assert!(!explanation.suggested_fixes.is_empty());
    }

    #[test]
    fn test_explain_datatype() {
        let mut explainer = ViolationExplainer::with_defaults();
        let explanation = explainer.explain(&datatype_violation());
        assert!(explanation.summary.contains("wrong data type"));
    }

    #[test]
    fn test_explain_class() {
        let mut explainer = ViolationExplainer::with_defaults();
        let explanation = explainer.explain(&class_violation());
        assert!(explanation.summary.contains("not an instance"));
    }

    #[test]
    fn test_severity_classification() {
        let mut explainer = ViolationExplainer::with_defaults();
        let e = explainer.explain(&min_count_violation());
        assert_eq!(e.severity, ViolationSeverity::Error);

        let pattern_v = ShaclViolation {
            constraint_type: ConstraintType::Pattern,
            ..min_count_violation()
        };
        let e2 = explainer.explain(&pattern_v);
        assert_eq!(e2.severity, ViolationSeverity::Warning);
    }

    #[test]
    fn test_shacl_severity_override() {
        let mut explainer = ViolationExplainer::with_defaults();
        let mut v = min_count_violation();
        v.shacl_severity = Some("sh:Info".into());
        let e = explainer.explain(&v);
        assert_eq!(e.severity, ViolationSeverity::Info);
    }

    #[test]
    fn test_sparql_fix_generation() {
        let mut explainer = ViolationExplainer::with_defaults();
        let e = explainer.explain(&min_count_violation());
        let sparql_fixes: Vec<_> = e
            .suggested_fixes
            .iter()
            .filter(|f| f.sparql_update.is_some())
            .collect();
        assert!(!sparql_fixes.is_empty());
    }

    #[test]
    fn test_no_sparql_when_disabled() {
        let mut explainer = ViolationExplainer::new(ExplainabilityConfig {
            generate_sparql_fixes: false,
            ..Default::default()
        });
        let e = explainer.explain(&min_count_violation());
        let sparql_fixes: Vec<_> = e
            .suggested_fixes
            .iter()
            .filter(|f| f.sparql_update.is_some())
            .collect();
        assert!(sparql_fixes.is_empty());
    }

    #[test]
    fn test_confidence_calculation() {
        let mut explainer = ViolationExplainer::with_defaults();
        let full_v = min_count_violation(); // Has expected, actual, path, shape
        let e = explainer.explain(&full_v);
        assert!(e.confidence > 0.8);

        let minimal_v = ShaclViolation {
            focus_node: "http://example.org/x".into(),
            result_path: None,
            constraint_type: ConstraintType::Custom("test".into()),
            expected: None,
            actual: None,
            source_shape: None,
            shacl_severity: None,
            context: HashMap::new(),
        };
        let e2 = explainer.explain(&minimal_v);
        assert!(e2.confidence < e.confidence);
    }

    #[test]
    fn test_batch_explanation() {
        let mut explainer = ViolationExplainer::with_defaults();
        let violations = vec![
            min_count_violation(),
            datatype_violation(),
            class_violation(),
        ];
        let explanations = explainer.explain_batch(&violations);
        assert_eq!(explanations.len(), 3);
    }

    #[test]
    fn test_related_violations_grouping() {
        let mut explainer = ViolationExplainer::with_defaults();
        let v1 = min_count_violation();
        let mut v2 = min_count_violation();
        v2.result_path = Some("http://schema.org/email".into());
        // Same focus node, same constraint type → related
        let explanations = explainer.explain_batch(&[v1, v2]);
        assert!(!explanations[0].related_violation_indices.is_empty());
    }

    #[test]
    fn test_stats_tracking() {
        let mut explainer = ViolationExplainer::with_defaults();
        explainer.explain(&min_count_violation());
        explainer.explain(&datatype_violation());

        let stats = explainer.stats();
        assert_eq!(stats.violations_explained, 2);
        assert_eq!(stats.error_count, 2);
    }

    #[test]
    fn test_tags_generation() {
        let mut explainer = ViolationExplainer::with_defaults();
        let e = explainer.explain(&min_count_violation());
        assert!(e.tags.contains(&"cardinality".to_string()));
    }

    #[test]
    fn test_range_tags() {
        let mut explainer = ViolationExplainer::with_defaults();
        let v = ShaclViolation {
            constraint_type: ConstraintType::MinInclusive,
            ..min_count_violation()
        };
        let e = explainer.explain(&v);
        assert!(e.tags.contains(&"range".to_string()));
    }

    #[test]
    fn test_max_count_explanation() {
        let mut explainer = ViolationExplainer::with_defaults();
        let v = ShaclViolation {
            constraint_type: ConstraintType::MaxCount,
            expected: Some("1".into()),
            actual: Some("3".into()),
            ..min_count_violation()
        };
        let e = explainer.explain(&v);
        assert!(e.summary.contains("too many"));
    }

    #[test]
    fn test_pattern_explanation() {
        let mut explainer = ViolationExplainer::with_defaults();
        let v = ShaclViolation {
            constraint_type: ConstraintType::Pattern,
            expected: Some("^[A-Z].*".into()),
            ..min_count_violation()
        };
        let e = explainer.explain(&v);
        assert!(e.summary.contains("pattern"));
    }

    #[test]
    fn test_uniqueness_severity() {
        let mut explainer = ViolationExplainer::with_defaults();
        let v = ShaclViolation {
            constraint_type: ConstraintType::Uniqueness,
            ..min_count_violation()
        };
        let e = explainer.explain(&v);
        assert_eq!(e.severity, ViolationSeverity::Critical);
    }

    #[test]
    fn test_detail_contains_node() {
        let mut explainer = ViolationExplainer::with_defaults();
        let e = explainer.explain(&min_count_violation());
        assert!(e.detail.contains("http://example.org/person/42"));
    }

    #[test]
    fn test_root_cause_min_count() {
        let mut explainer = ViolationExplainer::with_defaults();
        let e = explainer.explain(&min_count_violation());
        assert!(e.root_cause.contains("Missing"));
    }

    #[test]
    fn test_shorten_uri_slash() {
        assert_eq!(shorten_uri("http://example.org/person/42"), "42");
    }

    #[test]
    fn test_shorten_uri_hash() {
        assert_eq!(shorten_uri("http://example.org#name"), "name");
    }

    #[test]
    fn test_shorten_uri_plain() {
        assert_eq!(shorten_uri("plain"), "plain");
    }

    #[test]
    fn test_config_serialization() {
        let config = ExplainabilityConfig::default();
        let json = serde_json::to_string(&config).expect("serialize failed");
        assert!(json.contains("generate_sparql_fixes"));
    }

    #[test]
    fn test_explanation_serialization() {
        let mut explainer = ViolationExplainer::with_defaults();
        let e = explainer.explain(&min_count_violation());
        let json = serde_json::to_string(&e).expect("serialize failed");
        assert!(json.contains("summary"));
    }

    #[test]
    fn test_stats_serialization() {
        let stats = ExplainabilityStats::default();
        let json = serde_json::to_string(&stats).expect("serialize failed");
        assert!(json.contains("violations_explained"));
    }

    #[test]
    fn test_fix_effort_serde() {
        let e = FixEffort::Low;
        let json = serde_json::to_string(&e).expect("serialize failed");
        let deser: FixEffort = serde_json::from_str(&json).expect("deser failed");
        assert_eq!(deser, e);
    }

    #[test]
    fn test_constraint_type_display() {
        assert_eq!(ConstraintType::MinCount.display_name(), "Minimum Count");
        assert_eq!(ConstraintType::Custom("x".into()).display_name(), "x");
    }

    #[test]
    fn test_class_fix_is_auto_applicable() {
        let mut explainer = ViolationExplainer::with_defaults();
        let e = explainer.explain(&class_violation());
        assert!(e.suggested_fixes.iter().any(|f| f.auto_applicable));
    }

    #[test]
    fn test_max_fixes_limit() {
        let mut explainer = ViolationExplainer::new(ExplainabilityConfig {
            max_fixes: 1,
            ..Default::default()
        });
        let e = explainer.explain(&min_count_violation());
        assert!(e.suggested_fixes.len() <= 1);
    }

    #[test]
    fn test_min_exclusive_summary() {
        let mut explainer = ViolationExplainer::with_defaults();
        let v = ShaclViolation {
            constraint_type: ConstraintType::MinExclusive,
            expected: Some("0".into()),
            actual: Some("-1".into()),
            ..min_count_violation()
        };
        let e = explainer.explain(&v);
        assert!(e.summary.contains("below minimum"));
    }

    #[test]
    fn test_max_inclusive_summary() {
        let mut explainer = ViolationExplainer::with_defaults();
        let v = ShaclViolation {
            constraint_type: ConstraintType::MaxInclusive,
            expected: Some("100".into()),
            actual: Some("150".into()),
            ..min_count_violation()
        };
        let e = explainer.explain(&v);
        assert!(e.summary.contains("exceeds maximum"));
    }
}
