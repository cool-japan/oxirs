//! Human-readable explanation generator for SHACL validation violations.
//!
//! Translates structured [`Violation`] values into user-friendly text in
//! multiple formats (short summary, detailed prose, Markdown table, plain text).

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of constraint that was violated.
#[derive(Debug, Clone)]
pub enum ViolationType {
    /// Property appears fewer times than required.
    MinCount { actual: usize, required: usize },
    /// Property appears more times than allowed.
    MaxCount { actual: usize, allowed: usize },
    /// Value has an unexpected datatype.
    Datatype { expected: String, found: String },
    /// Value does not match the required regular expression.
    Pattern { regex: String, value: String },
    /// Numeric value is below the inclusive minimum.
    MinInclusive { threshold: f64, value: f64 },
    /// Numeric value is above the inclusive maximum.
    MaxInclusive { threshold: f64, value: f64 },
    /// Value is not in the allowed list.
    In { allowed: Vec<String>, found: String },
    /// Node does not have the expected RDF class.
    Class {
        expected: String,
        actual_types: Vec<String>,
    },
    /// Node has an unexpected kind (IRI, blank node, literal).
    NodeKind { expected: String, found: String },
    /// A property not permitted by a closed shape was found.
    ClosedShape { unexpected_property: String },
    /// A custom / catch-all violation message.
    Custom { message: String },
}

impl ViolationType {
    /// Return a short label identifying the violation kind (used in summaries).
    pub fn kind_label(&self) -> &'static str {
        match self {
            Self::MinCount { .. } => "MinCount",
            Self::MaxCount { .. } => "MaxCount",
            Self::Datatype { .. } => "Datatype",
            Self::Pattern { .. } => "Pattern",
            Self::MinInclusive { .. } => "MinInclusive",
            Self::MaxInclusive { .. } => "MaxInclusive",
            Self::In { .. } => "In",
            Self::Class { .. } => "Class",
            Self::NodeKind { .. } => "NodeKind",
            Self::ClosedShape { .. } => "ClosedShape",
            Self::Custom { .. } => "Custom",
        }
    }
}

/// A single SHACL constraint violation.
#[derive(Debug, Clone)]
pub struct Violation {
    /// The RDF node that violated the constraint
    pub node: String,
    /// The property path (if applicable)
    pub path: Option<String>,
    /// What went wrong
    pub violation_type: ViolationType,
    /// Which shape triggered the violation
    pub shape_id: String,
}

impl Violation {
    pub fn new(
        node: impl Into<String>,
        path: Option<impl Into<String>>,
        violation_type: ViolationType,
        shape_id: impl Into<String>,
    ) -> Self {
        Self {
            node: node.into(),
            path: path.map(|p| p.into()),
            violation_type,
            shape_id: shape_id.into(),
        }
    }
}

/// A human-readable explanation for a single violation.
#[derive(Debug, Clone)]
pub struct Explanation {
    /// One-line description suitable for error tables
    pub short: String,
    /// Multi-sentence description including context
    pub long: String,
    /// Actionable suggestion for fixing the issue
    pub suggestion: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// ExplanationGenerator
// ─────────────────────────────────────────────────────────────────────────────

/// Produces human-readable explanations for SHACL violations.
pub struct ExplanationGenerator {
    locale: String,
}

impl Default for ExplanationGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl ExplanationGenerator {
    /// Create a generator with the default locale (`en`).
    pub fn new() -> Self {
        Self {
            locale: "en".to_string(),
        }
    }

    /// Create a generator with a specific locale.
    pub fn with_locale(locale: impl Into<String>) -> Self {
        Self {
            locale: locale.into(),
        }
    }

    /// Return the locale identifier.
    pub fn locale(&self) -> &str {
        &self.locale
    }

    /// Generate an explanation for a single violation.
    pub fn explain(&self, v: &Violation) -> Explanation {
        let path_str = v
            .path
            .as_deref()
            .map(|p| format!(" (path: `{p}`)"))
            .unwrap_or_default();

        match &v.violation_type {
            ViolationType::MinCount { actual, required } => Explanation {
                short: format!(
                    "MinCount violation on `{}`: found {actual}, required {required}",
                    v.node
                ),
                long: format!(
                    "The node `{}`{path_str} has {actual} value(s) for shape `{}`, \
                     but the sh:minCount constraint requires at least {required}. \
                     This means not enough values are present.",
                    v.node, v.shape_id
                ),
                suggestion: format!(
                    "Add at least {} more value(s) for the property defined in shape `{}`.",
                    required - actual,
                    v.shape_id
                ),
            },

            ViolationType::MaxCount { actual, allowed } => Explanation {
                short: format!(
                    "MaxCount violation on `{}`: found {actual}, allowed {allowed}",
                    v.node
                ),
                long: format!(
                    "The node `{}`{path_str} has {actual} value(s) for shape `{}`, \
                     but sh:maxCount permits at most {allowed}.",
                    v.node, v.shape_id
                ),
                suggestion: format!(
                    "Remove {} value(s) from the property defined in shape `{}`.",
                    actual - allowed,
                    v.shape_id
                ),
            },

            ViolationType::Datatype { expected, found } => Explanation {
                short: format!(
                    "Datatype violation on `{}`: expected {expected}, found {found}",
                    v.node
                ),
                long: format!(
                    "The node `{}`{path_str} has a value of datatype `{found}`, \
                     but shape `{}` requires datatype `{expected}`.",
                    v.node, v.shape_id
                ),
                suggestion: format!(
                    "Convert the value to datatype `{expected}` as required by shape `{}`.",
                    v.shape_id
                ),
            },

            ViolationType::Pattern { regex, value } => Explanation {
                short: format!(
                    "Pattern violation on `{}`: value does not match /{regex}/",
                    v.node
                ),
                long: format!(
                    "The node `{}`{path_str} has value `{value}` which does not match \
                     the pattern `{regex}` required by shape `{}`.",
                    v.node, v.shape_id
                ),
                suggestion: format!(
                    "Update the value so it conforms to the regular expression `{regex}`."
                ),
            },

            ViolationType::MinInclusive { threshold, value } => Explanation {
                short: format!(
                    "MinInclusive violation on `{}`: value {value} < threshold {threshold}",
                    v.node
                ),
                long: format!(
                    "The node `{}`{path_str} has value {value}, which is below the \
                     sh:minInclusive threshold of {threshold} required by shape `{}`.",
                    v.node, v.shape_id
                ),
                suggestion: format!(
                    "Increase the value to at least {threshold} to satisfy shape `{}`.",
                    v.shape_id
                ),
            },

            ViolationType::MaxInclusive { threshold, value } => Explanation {
                short: format!(
                    "MaxInclusive violation on `{}`: value {value} > threshold {threshold}",
                    v.node
                ),
                long: format!(
                    "The node `{}`{path_str} has value {value}, which exceeds the \
                     sh:maxInclusive threshold of {threshold} required by shape `{}`.",
                    v.node, v.shape_id
                ),
                suggestion: format!(
                    "Reduce the value to at most {threshold} to satisfy shape `{}`.",
                    v.shape_id
                ),
            },

            ViolationType::In { allowed, found } => Explanation {
                short: format!(
                    "In violation on `{}`: value `{found}` not in allowed list",
                    v.node
                ),
                long: format!(
                    "The node `{}`{path_str} has value `{found}`, which is not among the \
                     allowed values {:?} required by shape `{}`.",
                    v.node, allowed, v.shape_id
                ),
                suggestion: format!(
                    "Replace the value with one of the allowed values: {}.",
                    allowed.join(", ")
                ),
            },

            ViolationType::Class {
                expected,
                actual_types,
            } => Explanation {
                short: format!(
                    "Class violation on `{}`: expected {expected}, got {:?}",
                    v.node, actual_types
                ),
                long: format!(
                    "The node `{}`{path_str} is expected to be an instance of `{expected}` \
                     by shape `{}`, but its actual types are: {:?}.",
                    v.node, v.shape_id, actual_types
                ),
                suggestion: format!(
                    "Add `rdf:type {expected}` to the node `{}`, or verify the correct class \
                     is being used.",
                    v.node
                ),
            },

            ViolationType::NodeKind { expected, found } => Explanation {
                short: format!(
                    "NodeKind violation on `{}`: expected {expected}, found {found}",
                    v.node
                ),
                long: format!(
                    "The node `{}`{path_str} has node kind `{found}`, but shape `{}` requires \
                     node kind `{expected}`.",
                    v.node, v.shape_id
                ),
                suggestion: format!(
                    "Ensure the node is a `{expected}` as required by shape `{}`.",
                    v.shape_id
                ),
            },

            ViolationType::ClosedShape {
                unexpected_property,
            } => Explanation {
                short: format!(
                    "ClosedShape violation on `{}`: unexpected property `{unexpected_property}`",
                    v.node
                ),
                long: format!(
                    "The node `{}`{path_str} has property `{unexpected_property}`, which is not \
                     declared in the closed shape `{}`.",
                    v.node, v.shape_id
                ),
                suggestion: format!(
                    "Remove property `{unexpected_property}` from `{}`, or add it to shape `{}`.",
                    v.node, v.shape_id
                ),
            },

            ViolationType::Custom { message } => Explanation {
                short: format!("Custom violation on `{}`: {message}", v.node),
                long: format!(
                    "Shape `{}` reported a custom violation for node `{}`{path_str}: {message}",
                    v.shape_id, v.node
                ),
                suggestion: format!(
                    "Refer to the documentation for shape `{}` to resolve this violation.",
                    v.shape_id
                ),
            },
        }
    }

    /// Generate explanations for every violation.
    pub fn explain_all(&self, violations: &[Violation]) -> Vec<Explanation> {
        violations.iter().map(|v| self.explain(v)).collect()
    }

    /// Produce a one-line summary such as `"3 violations: 2 MinCount, 1 Datatype"`.
    pub fn summary(&self, violations: &[Violation]) -> String {
        if violations.is_empty() {
            return "No violations found.".to_string();
        }
        let mut counts: HashMap<&'static str, usize> = HashMap::new();
        for v in violations {
            *counts.entry(v.violation_type.kind_label()).or_insert(0) += 1;
        }
        let mut parts: Vec<String> = counts
            .iter()
            .map(|(label, count)| format!("{count} {label}"))
            .collect();
        parts.sort(); // stable output
        format!(
            "{} violation{}: {}",
            violations.len(),
            if violations.len() == 1 { "" } else { "s" },
            parts.join(", ")
        )
    }

    /// Render explanations as a Markdown table.
    pub fn to_markdown(&self, violations: &[Violation]) -> String {
        if violations.is_empty() {
            return "No violations found.\n".to_string();
        }
        let mut out = String::new();
        out.push_str("# SHACL Violations\n\n");
        out.push_str("| Node | Shape | Type | Short Description | Suggestion |\n");
        out.push_str("|------|-------|------|-------------------|------------|\n");
        for v in violations {
            let exp = self.explain(v);
            // Escape pipe chars to avoid breaking the table
            let short = exp.short.replace('|', "\\|");
            let suggestion = exp.suggestion.replace('|', "\\|");
            out.push_str(&format!(
                "| `{}` | `{}` | {} | {} | {} |\n",
                v.node,
                v.shape_id,
                v.violation_type.kind_label(),
                short,
                suggestion
            ));
        }
        out
    }

    /// Render explanations as plain text (no Markdown syntax).
    pub fn to_plain_text(&self, violations: &[Violation]) -> String {
        if violations.is_empty() {
            return "No violations found.\n".to_string();
        }
        let mut out = String::new();
        out.push_str("SHACL Violations\n");
        out.push_str(&"=".repeat(60));
        out.push('\n');
        for (i, v) in violations.iter().enumerate() {
            let exp = self.explain(v);
            out.push_str(&format!(
                "\n[{}] {}\n{}\nSuggestion: {}\n",
                i + 1,
                exp.short,
                exp.long,
                exp.suggestion
            ));
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    fn gen() -> ExplanationGenerator {
        ExplanationGenerator::new()
    }

    fn v(node: &str, vtype: ViolationType) -> Violation {
        Violation::new(node, Option::<String>::None, vtype, "TestShape")
    }

    fn v_with_path(node: &str, path: &str, vtype: ViolationType) -> Violation {
        Violation::new(node, Some(path), vtype, "TestShape")
    }

    // ── Short, long and suggestion are non-empty for every ViolationType ──

    // 1. MinCount
    #[test]
    fn test_min_count_non_empty() {
        let exp = gen().explain(&v(
            "ex:node",
            ViolationType::MinCount {
                actual: 0,
                required: 1,
            },
        ));
        assert!(!exp.short.is_empty());
        assert!(!exp.long.is_empty());
        assert!(!exp.suggestion.is_empty());
    }

    // 2. MaxCount
    #[test]
    fn test_max_count_non_empty() {
        let exp = gen().explain(&v(
            "ex:node",
            ViolationType::MaxCount {
                actual: 5,
                allowed: 2,
            },
        ));
        assert!(!exp.short.is_empty());
        assert!(!exp.long.is_empty());
        assert!(!exp.suggestion.is_empty());
    }

    // 3. Datatype
    #[test]
    fn test_datatype_non_empty() {
        let exp = gen().explain(&v(
            "ex:node",
            ViolationType::Datatype {
                expected: "xsd:integer".to_string(),
                found: "xsd:string".to_string(),
            },
        ));
        assert!(!exp.short.is_empty());
        assert!(!exp.long.is_empty());
        assert!(!exp.suggestion.is_empty());
    }

    // 4. Pattern
    #[test]
    fn test_pattern_non_empty() {
        let exp = gen().explain(&v(
            "ex:node",
            ViolationType::Pattern {
                regex: "^[A-Z]+$".to_string(),
                value: "lowercase".to_string(),
            },
        ));
        assert!(!exp.short.is_empty());
        assert!(!exp.long.is_empty());
        assert!(!exp.suggestion.is_empty());
    }

    // 5. MinInclusive
    #[test]
    fn test_min_inclusive_non_empty() {
        let exp = gen().explain(&v(
            "ex:node",
            ViolationType::MinInclusive {
                threshold: 0.0,
                value: -1.5,
            },
        ));
        assert!(!exp.short.is_empty());
        assert!(!exp.long.is_empty());
        assert!(!exp.suggestion.is_empty());
    }

    // 6. MaxInclusive
    #[test]
    fn test_max_inclusive_non_empty() {
        let exp = gen().explain(&v(
            "ex:node",
            ViolationType::MaxInclusive {
                threshold: 100.0,
                value: 150.0,
            },
        ));
        assert!(!exp.short.is_empty());
        assert!(!exp.long.is_empty());
        assert!(!exp.suggestion.is_empty());
    }

    // 7. In
    #[test]
    fn test_in_non_empty() {
        let exp = gen().explain(&v(
            "ex:node",
            ViolationType::In {
                allowed: vec!["A".to_string(), "B".to_string()],
                found: "C".to_string(),
            },
        ));
        assert!(!exp.short.is_empty());
        assert!(!exp.long.is_empty());
        assert!(!exp.suggestion.is_empty());
    }

    // 8. Class
    #[test]
    fn test_class_non_empty() {
        let exp = gen().explain(&v(
            "ex:node",
            ViolationType::Class {
                expected: "ex:Person".to_string(),
                actual_types: vec!["ex:Animal".to_string()],
            },
        ));
        assert!(!exp.short.is_empty());
        assert!(!exp.long.is_empty());
        assert!(!exp.suggestion.is_empty());
    }

    // 9. NodeKind
    #[test]
    fn test_nodekind_non_empty() {
        let exp = gen().explain(&v(
            "ex:node",
            ViolationType::NodeKind {
                expected: "IRI".to_string(),
                found: "Literal".to_string(),
            },
        ));
        assert!(!exp.short.is_empty());
        assert!(!exp.long.is_empty());
        assert!(!exp.suggestion.is_empty());
    }

    // 10. ClosedShape
    #[test]
    fn test_closed_shape_non_empty() {
        let exp = gen().explain(&v(
            "ex:node",
            ViolationType::ClosedShape {
                unexpected_property: "ex:unknownProp".to_string(),
            },
        ));
        assert!(!exp.short.is_empty());
        assert!(!exp.long.is_empty());
        assert!(!exp.suggestion.is_empty());
    }

    // 11. Custom
    #[test]
    fn test_custom_non_empty() {
        let exp = gen().explain(&v(
            "ex:node",
            ViolationType::Custom {
                message: "some custom error".to_string(),
            },
        ));
        assert!(!exp.short.is_empty());
        assert!(!exp.long.is_empty());
        assert!(!exp.suggestion.is_empty());
    }

    // 12. summary correct count
    #[test]
    fn test_summary_correct_count() {
        let violations = vec![
            v(
                "n1",
                ViolationType::MinCount {
                    actual: 0,
                    required: 1,
                },
            ),
            v(
                "n2",
                ViolationType::MinCount {
                    actual: 0,
                    required: 1,
                },
            ),
            v(
                "n3",
                ViolationType::Datatype {
                    expected: "xsd:int".into(),
                    found: "xsd:str".into(),
                },
            ),
        ];
        let s = gen().summary(&violations);
        assert!(s.starts_with("3 violations"), "got: {s}");
        assert!(s.contains("2 MinCount"), "got: {s}");
        assert!(s.contains("1 Datatype"), "got: {s}");
    }

    // 13. summary empty violations
    #[test]
    fn test_summary_empty() {
        let s = gen().summary(&[]);
        assert!(s.contains("No violations"));
    }

    // 14. to_markdown contains markdown syntax
    #[test]
    fn test_to_markdown_contains_syntax() {
        let violations = vec![v(
            "ex:n",
            ViolationType::MaxCount {
                actual: 3,
                allowed: 1,
            },
        )];
        let md = gen().to_markdown(&violations);
        assert!(md.contains('|'));
        assert!(md.contains('#'));
        assert!(md.contains("---"));
    }

    // 15. to_plain_text no markdown
    #[test]
    fn test_to_plain_text_no_markdown() {
        let violations = vec![v(
            "ex:n",
            ViolationType::MinCount {
                actual: 0,
                required: 2,
            },
        )];
        let txt = gen().to_plain_text(&violations);
        assert!(!txt.contains("---"));
        // No Markdown table pipes in expected positions
        // (plain text may contain '|' in URIs but not table rows)
        assert!(!txt.contains('|'));
    }

    // 16. explain_all length matches violations
    #[test]
    fn test_explain_all_length_matches() {
        let violations: Vec<Violation> = (0..7)
            .map(|i| {
                v(
                    &format!("ex:node{i}"),
                    ViolationType::Custom {
                        message: format!("err {i}"),
                    },
                )
            })
            .collect();
        let explanations = gen().explain_all(&violations);
        assert_eq!(explanations.len(), 7);
    }

    // 17. explain_all with empty violations
    #[test]
    fn test_explain_all_empty() {
        let explanations = gen().explain_all(&[]);
        assert!(explanations.is_empty());
    }

    // 18. MinCount suggestion mentions how many to add
    #[test]
    fn test_min_count_suggestion_mentions_delta() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::MinCount {
                actual: 1,
                required: 3,
            },
        ));
        assert!(
            exp.suggestion.contains('2') || exp.suggestion.to_lowercase().contains("more"),
            "suggestion: {}",
            exp.suggestion
        );
    }

    // 19. MaxCount suggestion mentions how many to remove
    #[test]
    fn test_max_count_suggestion_mentions_delta() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::MaxCount {
                actual: 5,
                allowed: 2,
            },
        ));
        assert!(
            exp.suggestion.contains('3') || exp.suggestion.to_lowercase().contains("remov"),
            "suggestion: {}",
            exp.suggestion
        );
    }

    // 20. Datatype short mentions expected and found types
    #[test]
    fn test_datatype_short_mentions_types() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::Datatype {
                expected: "xsd:integer".into(),
                found: "xsd:string".into(),
            },
        ));
        assert!(exp.short.contains("xsd:integer"));
        assert!(exp.short.contains("xsd:string"));
    }

    // 21. Pattern short mentions regex
    #[test]
    fn test_pattern_short_mentions_regex() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::Pattern {
                regex: r"^\d+$".to_string(),
                value: "abc".to_string(),
            },
        ));
        assert!(exp.short.contains(r"^\d+$"));
    }

    // 22. MinInclusive short mentions threshold
    #[test]
    fn test_min_inclusive_short_mentions_threshold() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::MinInclusive {
                threshold: 10.0,
                value: 5.0,
            },
        ));
        assert!(exp.short.contains("10"));
    }

    // 23. MaxInclusive short mentions threshold
    #[test]
    fn test_max_inclusive_short_mentions_threshold() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::MaxInclusive {
                threshold: 50.0,
                value: 75.0,
            },
        ));
        assert!(exp.short.contains("50"));
    }

    // 24. In suggestion lists allowed values
    #[test]
    fn test_in_suggestion_lists_values() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::In {
                allowed: vec!["X".into(), "Y".into(), "Z".into()],
                found: "W".into(),
            },
        ));
        assert!(exp.suggestion.contains("X") || exp.suggestion.contains("Y"));
    }

    // 25. Class long mentions expected class
    #[test]
    fn test_class_long_mentions_expected() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::Class {
                expected: "ex:Vehicle".into(),
                actual_types: vec!["ex:Animal".into()],
            },
        ));
        assert!(exp.long.contains("ex:Vehicle"));
    }

    // 26. NodeKind short mentions found kind
    #[test]
    fn test_nodekind_short_mentions_found() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::NodeKind {
                expected: "IRI".into(),
                found: "BlankNode".into(),
            },
        ));
        assert!(exp.short.contains("BlankNode"));
    }

    // 27. ClosedShape short mentions property
    #[test]
    fn test_closed_shape_short_mentions_property() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::ClosedShape {
                unexpected_property: "ex:secretProp".into(),
            },
        ));
        assert!(exp.short.contains("ex:secretProp"));
    }

    // 28. Custom long mentions message
    #[test]
    fn test_custom_long_mentions_message() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::Custom {
                message: "unique error 42".into(),
            },
        ));
        assert!(exp.long.contains("unique error 42"));
    }

    // 29. Path included in long description when present
    #[test]
    fn test_path_in_long_description() {
        let v = v_with_path(
            "ex:n",
            "ex:name",
            ViolationType::MinCount {
                actual: 0,
                required: 1,
            },
        );
        let exp = gen().explain(&v);
        assert!(exp.long.contains("ex:name"), "long: {}", exp.long);
    }

    // 30. With locale constructor stores locale
    #[test]
    fn test_with_locale() {
        let g = ExplanationGenerator::with_locale("de");
        assert_eq!(g.locale(), "de");
    }

    // 31. Default locale is en
    #[test]
    fn test_default_locale() {
        let g = ExplanationGenerator::new();
        assert_eq!(g.locale(), "en");
    }

    // 32. to_markdown empty violations
    #[test]
    fn test_to_markdown_empty() {
        let md = gen().to_markdown(&[]);
        assert!(md.contains("No violations"));
    }

    // 33. to_plain_text empty violations
    #[test]
    fn test_to_plain_text_empty() {
        let txt = gen().to_plain_text(&[]);
        assert!(txt.contains("No violations"));
    }

    // 34. to_markdown contains node names
    #[test]
    fn test_to_markdown_contains_node_names() {
        let v = v(
            "ex:myNode",
            ViolationType::Custom {
                message: "oops".into(),
            },
        );
        let md = gen().to_markdown(&[v]);
        assert!(md.contains("ex:myNode"));
    }

    // 35. to_plain_text contains node names
    #[test]
    fn test_to_plain_text_contains_node_names() {
        let v = v(
            "ex:anotherNode",
            ViolationType::Custom {
                message: "fail".into(),
            },
        );
        let txt = gen().to_plain_text(&[v]);
        assert!(txt.contains("ex:anotherNode"));
    }

    // 36. summary single violation uses singular
    #[test]
    fn test_summary_single_violation_singular() {
        let violations = vec![v(
            "ex:n",
            ViolationType::Custom {
                message: "x".into(),
            },
        )];
        let s = gen().summary(&violations);
        // Should say "1 violation:" (no trailing 's')
        assert!(s.starts_with("1 violation:"), "got: {s}");
    }

    // 37. explain short contains node
    #[test]
    fn test_explain_short_contains_node() {
        let exp = gen().explain(&v(
            "ex:mySpecialNode",
            ViolationType::Custom {
                message: "test".into(),
            },
        ));
        assert!(exp.short.contains("ex:mySpecialNode"));
    }

    // 38. explain long contains shape_id
    #[test]
    fn test_explain_long_contains_shape_id() {
        let mut v = v(
            "ex:n",
            ViolationType::Custom {
                message: "test".into(),
            },
        );
        v.shape_id = "ex:MyShape".to_string();
        let exp = gen().explain(&v);
        assert!(exp.long.contains("ex:MyShape"), "long: {}", exp.long);
    }

    // 39. ViolationType kind_label
    #[test]
    fn test_violation_type_kind_labels() {
        assert_eq!(
            ViolationType::MinCount {
                actual: 0,
                required: 1
            }
            .kind_label(),
            "MinCount"
        );
        assert_eq!(
            ViolationType::MaxCount {
                actual: 2,
                allowed: 1
            }
            .kind_label(),
            "MaxCount"
        );
        assert_eq!(
            ViolationType::Datatype {
                expected: "".into(),
                found: "".into()
            }
            .kind_label(),
            "Datatype"
        );
        assert_eq!(
            ViolationType::Pattern {
                regex: "".into(),
                value: "".into()
            }
            .kind_label(),
            "Pattern"
        );
        assert_eq!(
            ViolationType::MinInclusive {
                threshold: 0.0,
                value: 0.0
            }
            .kind_label(),
            "MinInclusive"
        );
        assert_eq!(
            ViolationType::MaxInclusive {
                threshold: 0.0,
                value: 0.0
            }
            .kind_label(),
            "MaxInclusive"
        );
        assert_eq!(
            ViolationType::In {
                allowed: vec![],
                found: "".into()
            }
            .kind_label(),
            "In"
        );
        assert_eq!(
            ViolationType::Class {
                expected: "".into(),
                actual_types: vec![]
            }
            .kind_label(),
            "Class"
        );
        assert_eq!(
            ViolationType::NodeKind {
                expected: "".into(),
                found: "".into()
            }
            .kind_label(),
            "NodeKind"
        );
        assert_eq!(
            ViolationType::ClosedShape {
                unexpected_property: "".into()
            }
            .kind_label(),
            "ClosedShape"
        );
        assert_eq!(
            ViolationType::Custom { message: "".into() }.kind_label(),
            "Custom"
        );
    }

    // 40. to_markdown contains header row
    #[test]
    fn test_to_markdown_header_row() {
        let vs = vec![v(
            "n",
            ViolationType::Custom {
                message: "x".into(),
            },
        )];
        let md = gen().to_markdown(&vs);
        assert!(md.contains("Node"), "missing 'Node' column: {md}");
        assert!(md.contains("Shape"), "missing 'Shape' column: {md}");
    }

    // 41. to_plain_text contains separator
    #[test]
    fn test_to_plain_text_separator() {
        let vs = vec![v(
            "n",
            ViolationType::Custom {
                message: "x".into(),
            },
        )];
        let txt = gen().to_plain_text(&vs);
        assert!(txt.contains('='), "separator missing: {txt}");
    }

    // 42. MinCount actual > 0 still a violation
    #[test]
    fn test_min_count_actual_nonzero() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::MinCount {
                actual: 2,
                required: 5,
            },
        ));
        assert!(exp.short.contains('2'));
        assert!(exp.short.contains('5'));
    }

    // 43. MaxCount exceeded: suggestion mentions exact delta
    #[test]
    fn test_max_count_exceeded() {
        let exp = gen().explain(&v(
            "ex:n",
            ViolationType::MaxCount {
                actual: 10,
                allowed: 3,
            },
        ));
        // 10 - 3 = 7
        assert!(
            exp.suggestion.contains('7') || exp.suggestion.to_lowercase().contains("remov"),
            "suggestion: {}",
            exp.suggestion
        );
    }

    // 44. explain with path=None — no path text
    #[test]
    fn test_no_path_no_path_text() {
        let v = Violation::new(
            "ex:n",
            Option::<String>::None,
            ViolationType::Custom {
                message: "test".into(),
            },
            "S",
        );
        let exp = gen().explain(&v);
        assert!(!exp.long.contains("path:"), "long: {}", exp.long);
    }

    // 45. to_markdown with multiple violations has multiple rows
    #[test]
    fn test_to_markdown_multiple_rows() {
        let violations: Vec<Violation> = (0..4)
            .map(|i| {
                v(
                    &format!("ex:n{i}"),
                    ViolationType::Custom {
                        message: "err".into(),
                    },
                )
            })
            .collect();
        let md = gen().to_markdown(&violations);
        // Count the pipe-delimited data rows (excluding header + separator)
        let data_rows = md.lines().filter(|l| l.starts_with("| `ex:")).count();
        assert_eq!(data_rows, 4);
    }
}
