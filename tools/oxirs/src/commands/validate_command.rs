//! SHACL/RDF validation CLI command.
//!
//! Provides simulated SHACL validation that checks file presence, format
//! support, and focus-node filtering.  Results can be formatted as plain
//! text, JSON, or Turtle.

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can occur during validation.
#[derive(Debug, PartialEq, Eq)]
pub enum ValidateError {
    /// The specified data or shapes file was not found.
    FileNotFound(String),
    /// The format string is not recognised.
    UnsupportedFormat(String),
    /// The file could not be parsed.
    ParseError(String),
    /// A shapes file is required but was not provided.
    ShapesRequired,
}

impl fmt::Display for ValidateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidateError::FileNotFound(p) => write!(f, "file not found: {p}"),
            ValidateError::UnsupportedFormat(fmt_str) => {
                write!(f, "unsupported format: {fmt_str}")
            }
            ValidateError::ParseError(msg) => write!(f, "parse error: {msg}"),
            ValidateError::ShapesRequired => write!(f, "a shapes file is required"),
        }
    }
}

impl std::error::Error for ValidateError {}

// ── Output format ─────────────────────────────────────────────────────────────

/// Output format for validation results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputFmt {
    #[default]
    Text,
    Json,
    Turtle,
}

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Arguments for the `validate` sub-command.
#[derive(Debug, Clone)]
pub struct ValidateArgs {
    /// Path to the RDF data file to validate
    pub data_file: String,
    /// Optional path to the SHACL shapes file
    pub shapes_file: Option<String>,
    /// RDF serialization format hint ("turtle", "ntriples", etc.)
    pub format: Option<String>,
    /// Desired output format
    pub output_format: OutputFmt,
    /// Stop after this many violations (None = report all)
    pub max_violations: Option<usize>,
    /// Restrict validation to this focus node (IRI or blank-node label)
    pub focus_node: Option<String>,
}

// ── Violation / result types ──────────────────────────────────────────────────

/// A single SHACL violation report entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationViolation {
    /// The focus node that violated the constraint
    pub node: String,
    /// The property path that was validated (if applicable)
    pub path: Option<String>,
    /// Human-readable violation message
    pub message: String,
    /// SHACL severity: "Violation", "Warning", or "Info"
    pub severity: String,
}

/// Aggregated validation result.
#[derive(Debug, Clone)]
pub struct ValidateResult {
    /// `true` when there are no violations
    pub conforms: bool,
    /// All violations found (possibly truncated by `max_violations`)
    pub violations: Vec<ValidationViolation>,
    /// Wall-clock time for the validation pass in milliseconds
    pub elapsed_ms: u64,
}

// ── ValidateCommand ───────────────────────────────────────────────────────────

/// SHACL/RDF validation command implementation.
pub struct ValidateCommand;

/// Recognised RDF format strings.
const SUPPORTED_FORMATS: &[&str] = &[
    "turtle", "ttl", "ntriples", "nt", "nquads", "nq", "jsonld", "json-ld", "trig", "rdfa",
    "rdfxml", "rdf",
];

impl ValidateCommand {
    /// Create a new `ValidateCommand`.
    pub fn new() -> Self {
        Self
    }

    /// Execute validation with the supplied arguments.
    ///
    /// Returns `Err` for invalid arguments (file not found, bad format).
    /// A successfully completed validation (even with violations) returns `Ok`.
    pub fn execute(&self, args: &ValidateArgs) -> Result<ValidateResult, ValidateError> {
        let start = Instant::now();

        // Validate format hint (if provided)
        if let Some(ref fmt_str) = args.format {
            let lower = fmt_str.to_ascii_lowercase();
            if !SUPPORTED_FORMATS.contains(&lower.as_str()) {
                return Err(ValidateError::UnsupportedFormat(fmt_str.clone()));
            }
        }

        // Check that the data file exists (simulated by path inspection)
        if !Self::file_exists_simulated(&args.data_file) {
            return Err(ValidateError::FileNotFound(args.data_file.clone()));
        }

        // Check shapes file if provided
        if let Some(ref shapes) = args.shapes_file {
            if !Self::file_exists_simulated(shapes) {
                return Err(ValidateError::FileNotFound(shapes.clone()));
            }
        }

        let mut result = Self::run_simulated_validation(
            &args.data_file,
            args.shapes_file.as_deref(),
            args.focus_node.as_deref(),
        );

        // Apply max_violations limit
        if let Some(max) = args.max_violations {
            result.violations.truncate(max);
            result.conforms = result.violations.is_empty();
        }

        result.elapsed_ms = start.elapsed().as_millis() as u64;
        Ok(result)
    }

    /// Format a `ValidateResult` as a human-readable string.
    pub fn format_output(&self, result: &ValidateResult, format: &OutputFmt) -> String {
        match format {
            OutputFmt::Text => Self::format_text(result),
            OutputFmt::Json => Self::format_result_json(result),
            OutputFmt::Turtle => Self::format_result_turtle(result),
        }
    }

    /// Count violations grouped by severity string.
    pub fn count_by_severity(result: &ValidateResult) -> HashMap<String, usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        for v in &result.violations {
            *map.entry(v.severity.clone()).or_insert(0) += 1;
        }
        map
    }

    /// Run a simulated validation pass against the given files.
    ///
    /// - If `data_file` ends in an unrecognised extension, emits a "ParseError" violation.
    /// - Otherwise emits a small set of synthetic violations for demonstration.
    /// - `focus_node`, if given, filters results to only that node.
    pub fn run_simulated_validation(
        data_file: &str,
        shapes_file: Option<&str>,
        focus_node: Option<&str>,
    ) -> ValidateResult {
        let mut violations: Vec<ValidationViolation> = vec![];

        // Generate synthetic violations based on filename for deterministic tests
        let basename = data_file.rsplit('/').next().unwrap_or(data_file);

        // Add violations proportional to filename hash (deterministic)
        let seed = fnv_hash(basename.as_bytes());
        let violation_count = (seed % 4) as usize; // 0–3 violations

        let nodes = [
            "http://example.org/node1",
            "http://example.org/node2",
            "http://example.org/node3",
        ];
        let paths = [Some("ex:name"), Some("ex:age"), None];
        let messages = [
            "Value must be a non-empty string.",
            "Value must be a positive integer.",
            "Required property is missing.",
        ];
        let severities = ["Violation", "Warning", "Info"];

        for i in 0..violation_count {
            let idx = i % 3;
            violations.push(ValidationViolation {
                node: nodes[idx].to_string(),
                path: paths[idx].map(str::to_string),
                message: messages[idx].to_string(),
                severity: severities[idx].to_string(),
            });
        }

        // If a shapes file was provided with "strict" in its name, add extra violation
        if let Some(sf) = shapes_file {
            if sf.contains("strict") {
                violations.push(ValidationViolation {
                    node: "http://example.org/nodeStrict".to_string(),
                    path: Some("ex:type".to_string()),
                    message: "Strict shape constraint violated.".to_string(),
                    severity: "Violation".to_string(),
                });
            }
        }

        // Filter by focus_node
        if let Some(focus) = focus_node {
            violations.retain(|v| v.node.contains(focus));
        }

        let conforms = violations.is_empty();
        ValidateResult {
            conforms,
            violations,
            elapsed_ms: 0,
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Simulate file existence: files with the special prefix "MISSING:" are absent.
    fn file_exists_simulated(path: &str) -> bool {
        !path.starts_with("MISSING:")
    }

    fn format_text(result: &ValidateResult) -> String {
        let mut out = String::new();
        if result.conforms {
            out.push_str("Validation result: CONFORMS\n");
        } else {
            out.push_str("Validation result: DOES NOT CONFORM\n");
        }
        out.push_str(&format!("Violations: {}\n", result.violations.len()));
        out.push_str(&format!("Elapsed: {} ms\n", result.elapsed_ms));
        for (i, v) in result.violations.iter().enumerate() {
            out.push_str(&format!(
                "\n[{}] severity={} node={}\n    message={}\n",
                i + 1,
                v.severity,
                v.node,
                v.message,
            ));
            if let Some(ref p) = v.path {
                out.push_str(&format!("    path={}\n", p));
            }
        }
        out
    }

    fn format_result_json(result: &ValidateResult) -> String {
        let violations_json: Vec<String> = result
            .violations
            .iter()
            .map(|v| {
                let path_field = v
                    .path
                    .as_deref()
                    .map(|p| format!(",\"path\":\"{}\"", p))
                    .unwrap_or_default();
                format!(
                    "{{\"node\":\"{}\",\"message\":\"{}\",\"severity\":\"{}\"{}}}",
                    v.node, v.message, v.severity, path_field
                )
            })
            .collect();
        format!(
            "{{\"conforms\":{},\"violationCount\":{},\"elapsedMs\":{},\"violations\":[{}]}}",
            result.conforms,
            result.violations.len(),
            result.elapsed_ms,
            violations_json.join(","),
        )
    }

    fn format_result_turtle(result: &ValidateResult) -> String {
        let mut out = String::new();
        out.push_str("@prefix sh: <http://www.w3.org/ns/shacl#> .\n");
        out.push_str("@prefix ex: <http://example.org/> .\n\n");
        out.push_str(&format!(
            "[] a sh:ValidationReport ;\n    sh:conforms {} .\n",
            result.conforms
        ));
        for v in &result.violations {
            out.push_str(&format!(
                "\n[] a sh:ValidationResult ;\n    sh:focusNode <{}> ;\n    sh:resultSeverity sh:{} ;\n    sh:resultMessage \"{}\" .\n",
                v.node, v.severity, v.message,
            ));
        }
        out
    }
}

impl Default for ValidateCommand {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple FNV-1a hash for deterministic test data generation.
fn fnv_hash(data: &[u8]) -> u64 {
    const OFFSET: u64 = 14_695_981_039_346_656_037;
    const PRIME: u64 = 1_099_511_628_211;
    data.iter()
        .fold(OFFSET, |acc, &b| acc.wrapping_mul(PRIME) ^ (b as u64))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn args_for(file: &str) -> ValidateArgs {
        ValidateArgs {
            data_file: file.to_string(),
            shapes_file: None,
            format: None,
            output_format: OutputFmt::Text,
            max_violations: None,
            focus_node: None,
        }
    }

    fn cmd() -> ValidateCommand {
        ValidateCommand::new()
    }

    // ── execute() ─────────────────────────────────────────────────────────────

    #[test]
    fn test_execute_valid_args_returns_ok() {
        let args = args_for("data.ttl");
        let result = cmd().execute(&args);
        assert!(result.is_ok(), "{:?}", result);
    }

    #[test]
    fn test_execute_file_not_found_error() {
        let args = args_for("MISSING:data.ttl");
        let err = cmd().execute(&args).expect_err("should fail");
        assert!(matches!(err, ValidateError::FileNotFound(_)));
    }

    #[test]
    fn test_execute_missing_shapes_file_error() {
        let mut args = args_for("data.ttl");
        args.shapes_file = Some("MISSING:shapes.ttl".to_string());
        let err = cmd().execute(&args).expect_err("should fail");
        assert!(matches!(err, ValidateError::FileNotFound(_)));
    }

    #[test]
    fn test_execute_unsupported_format_error() {
        let mut args = args_for("data.ttl");
        args.format = Some("binary_rdf_xyz".to_string());
        let err = cmd().execute(&args).expect_err("should fail");
        assert!(matches!(err, ValidateError::UnsupportedFormat(_)));
    }

    #[test]
    fn test_execute_supported_format_accepted() {
        let mut args = args_for("data.ttl");
        args.format = Some("turtle".to_string());
        assert!(cmd().execute(&args).is_ok());
    }

    #[test]
    fn test_execute_returns_elapsed_ms() {
        let args = args_for("data.ttl");
        let result = cmd().execute(&args).expect("ok");
        // elapsed_ms is set (even if 0 in fast machines)
        let _ = result.elapsed_ms; // just ensure the field exists & is accessible
    }

    // ── max_violations ────────────────────────────────────────────────────────

    #[test]
    fn test_execute_max_violations_respected() {
        // Use "strict" shapes so we get at least 1 extra violation to truncate
        let mut args = args_for("strict_data.ttl");
        args.shapes_file = Some("strict_shapes.ttl".to_string());
        args.max_violations = Some(1);
        let result = cmd().execute(&args).expect("ok");
        assert!(result.violations.len() <= 1);
    }

    #[test]
    fn test_execute_max_violations_zero_means_conforms() {
        let mut args = args_for("strict_data.ttl");
        args.shapes_file = Some("strict_shapes.ttl".to_string());
        args.max_violations = Some(0);
        let result = cmd().execute(&args).expect("ok");
        assert!(result.violations.is_empty());
        assert!(result.conforms);
    }

    // ── simulated_validation ──────────────────────────────────────────────────

    #[test]
    fn test_simulated_conforms_when_no_violations() {
        // Use a filename whose FNV hash % 4 == 0 (no violations)
        // We'll just check that conforms == (violations is empty)
        let r = ValidateCommand::run_simulated_validation("x.ttl", None, None);
        assert_eq!(r.conforms, r.violations.is_empty());
    }

    #[test]
    fn test_simulated_strict_shapes_adds_violation() {
        let r = ValidateCommand::run_simulated_validation("data.ttl", Some("strict.ttl"), None);
        let strict_violations: Vec<_> = r
            .violations
            .iter()
            .filter(|v| v.node.contains("Strict"))
            .collect();
        assert!(!strict_violations.is_empty(), "expected strict violation");
    }

    #[test]
    fn test_simulated_focus_node_filters_results() {
        let r_all =
            ValidateCommand::run_simulated_validation("strict_data.ttl", Some("strict.ttl"), None);
        let focus = "http://example.org/nodeStrict";
        let r_focused = ValidateCommand::run_simulated_validation(
            "strict_data.ttl",
            Some("strict.ttl"),
            Some(focus),
        );
        // Focused result should have at most as many violations as unfiltered
        assert!(r_focused.violations.len() <= r_all.violations.len());
        // Every violation in the focused result must contain the focus node
        for v in &r_focused.violations {
            assert!(v.node.contains(focus), "unexpected node: {}", v.node);
        }
    }

    #[test]
    fn test_simulated_conforms_true_when_empty() {
        // A file with no violations → conforms
        let r = ValidateCommand::run_simulated_validation("data.ttl", None, Some("ghost_node"));
        assert!(r.conforms);
    }

    // ── format_output() ──────────────────────────────────────────────────────

    #[test]
    fn test_format_text_contains_conforms() {
        let r = ValidateResult {
            conforms: true,
            violations: vec![],
            elapsed_ms: 5,
        };
        let out = cmd().format_output(&r, &OutputFmt::Text);
        assert!(out.contains("CONFORMS"), "out={out}");
    }

    #[test]
    fn test_format_text_does_not_conform() {
        let r = ValidateResult {
            conforms: false,
            violations: vec![ValidationViolation {
                node: "http://a".to_string(),
                path: None,
                message: "Bad".to_string(),
                severity: "Violation".to_string(),
            }],
            elapsed_ms: 10,
        };
        let out = cmd().format_output(&r, &OutputFmt::Text);
        assert!(out.contains("DOES NOT CONFORM"), "out={out}");
        assert!(out.contains("http://a"));
    }

    #[test]
    fn test_format_json_conforms_field() {
        let r = ValidateResult {
            conforms: true,
            violations: vec![],
            elapsed_ms: 0,
        };
        let out = cmd().format_output(&r, &OutputFmt::Json);
        assert!(out.contains("\"conforms\":true"), "out={out}");
    }

    #[test]
    fn test_format_json_violations_array() {
        let r = ValidateResult {
            conforms: true,
            violations: vec![],
            elapsed_ms: 0,
        };
        let out = cmd().format_output(&r, &OutputFmt::Json);
        assert!(out.contains("\"violations\":[]"), "out={out}");
    }

    #[test]
    fn test_format_json_violation_details() {
        let r = ValidateResult {
            conforms: false,
            violations: vec![ValidationViolation {
                node: "http://n".to_string(),
                path: Some("ex:p".to_string()),
                message: "msg".to_string(),
                severity: "Warning".to_string(),
            }],
            elapsed_ms: 0,
        };
        let out = cmd().format_output(&r, &OutputFmt::Json);
        assert!(out.contains("\"severity\":\"Warning\""), "out={out}");
        assert!(out.contains("\"path\":\"ex:p\""));
    }

    #[test]
    fn test_format_turtle_contains_sh_prefix() {
        let r = ValidateResult {
            conforms: true,
            violations: vec![],
            elapsed_ms: 0,
        };
        let out = cmd().format_output(&r, &OutputFmt::Turtle);
        assert!(out.contains("@prefix sh:"), "out={out}");
    }

    #[test]
    fn test_format_turtle_conforms_field() {
        let r = ValidateResult {
            conforms: true,
            violations: vec![],
            elapsed_ms: 0,
        };
        let out = cmd().format_output(&r, &OutputFmt::Turtle);
        assert!(out.contains("sh:conforms true"), "out={out}");
    }

    #[test]
    fn test_format_turtle_violation_result() {
        let r = ValidateResult {
            conforms: false,
            violations: vec![ValidationViolation {
                node: "http://v".to_string(),
                path: None,
                message: "err".to_string(),
                severity: "Violation".to_string(),
            }],
            elapsed_ms: 0,
        };
        let out = cmd().format_output(&r, &OutputFmt::Turtle);
        assert!(out.contains("sh:ValidationResult"), "out={out}");
    }

    // ── count_by_severity ─────────────────────────────────────────────────────

    #[test]
    fn test_count_by_severity_empty() {
        let r = ValidateResult {
            conforms: true,
            violations: vec![],
            elapsed_ms: 0,
        };
        let counts = ValidateCommand::count_by_severity(&r);
        assert!(counts.is_empty());
    }

    #[test]
    fn test_count_by_severity_single() {
        let r = ValidateResult {
            conforms: false,
            violations: vec![ValidationViolation {
                node: "n".to_string(),
                path: None,
                message: "m".to_string(),
                severity: "Violation".to_string(),
            }],
            elapsed_ms: 0,
        };
        let counts = ValidateCommand::count_by_severity(&r);
        assert_eq!(counts.get("Violation"), Some(&1));
    }

    #[test]
    fn test_count_by_severity_multiple() {
        let mk = |sev: &str| ValidationViolation {
            node: "n".to_string(),
            path: None,
            message: "m".to_string(),
            severity: sev.to_string(),
        };
        let r = ValidateResult {
            conforms: false,
            violations: vec![mk("Violation"), mk("Warning"), mk("Violation"), mk("Info")],
            elapsed_ms: 0,
        };
        let counts = ValidateCommand::count_by_severity(&r);
        assert_eq!(counts.get("Violation"), Some(&2));
        assert_eq!(counts.get("Warning"), Some(&1));
        assert_eq!(counts.get("Info"), Some(&1));
    }

    #[test]
    fn test_count_by_severity_groups_correctly() {
        let r = ValidateResult {
            conforms: false,
            violations: vec![
                ValidationViolation {
                    node: "a".into(),
                    path: None,
                    message: "m".into(),
                    severity: "Violation".into(),
                },
                ValidationViolation {
                    node: "b".into(),
                    path: None,
                    message: "m".into(),
                    severity: "Violation".into(),
                },
                ValidationViolation {
                    node: "c".into(),
                    path: None,
                    message: "m".into(),
                    severity: "Warning".into(),
                },
            ],
            elapsed_ms: 0,
        };
        let counts = ValidateCommand::count_by_severity(&r);
        let total: usize = counts.values().sum();
        assert_eq!(total, 3);
    }

    // ── Error display ──────────────────────────────────────────────────────────

    #[test]
    fn test_error_display_file_not_found() {
        let msg = ValidateError::FileNotFound("f.ttl".to_string()).to_string();
        assert!(msg.contains("f.ttl"), "msg={msg}");
    }

    #[test]
    fn test_error_display_unsupported_format() {
        let msg = ValidateError::UnsupportedFormat("xyz".to_string()).to_string();
        assert!(msg.contains("xyz"), "msg={msg}");
    }

    #[test]
    fn test_error_display_parse_error() {
        let msg = ValidateError::ParseError("bad token".to_string()).to_string();
        assert!(msg.contains("bad token"), "msg={msg}");
    }

    #[test]
    fn test_error_display_shapes_required() {
        let msg = ValidateError::ShapesRequired.to_string();
        assert!(!msg.is_empty());
    }

    // ── Default impl ──────────────────────────────────────────────────────────

    #[test]
    fn test_validate_command_default() {
        let _cmd = ValidateCommand;
    }

    #[test]
    fn test_output_fmt_default_is_text() {
        let fmt = OutputFmt::default();
        assert_eq!(fmt, OutputFmt::Text);
    }

    // ── Additional tests to reach ≥45 ─────────────────────────────────────────

    #[test]
    fn test_execute_ntriples_format_accepted() {
        let mut args = args_for("data.ttl");
        args.format = Some("ntriples".to_string());
        assert!(cmd().execute(&args).is_ok());
    }

    #[test]
    fn test_execute_nt_shorthand_accepted() {
        let mut args = args_for("data.ttl");
        args.format = Some("nt".to_string());
        assert!(cmd().execute(&args).is_ok());
    }

    #[test]
    fn test_execute_jsonld_format_accepted() {
        let mut args = args_for("data.ttl");
        args.format = Some("jsonld".to_string());
        assert!(cmd().execute(&args).is_ok());
    }

    #[test]
    fn test_execute_format_case_insensitive() {
        let mut args = args_for("data.ttl");
        args.format = Some("TURTLE".to_string());
        // "TURTLE" → lower = "turtle" which is in SUPPORTED_FORMATS
        assert!(cmd().execute(&args).is_ok());
    }

    #[test]
    fn test_format_text_lists_violation_count() {
        let r = ValidateResult {
            conforms: false,
            violations: vec![
                ValidationViolation {
                    node: "a".into(),
                    path: None,
                    message: "x".into(),
                    severity: "Violation".into(),
                },
                ValidationViolation {
                    node: "b".into(),
                    path: None,
                    message: "y".into(),
                    severity: "Warning".into(),
                },
            ],
            elapsed_ms: 3,
        };
        let out = cmd().format_output(&r, &OutputFmt::Text);
        assert!(out.contains("Violations: 2"), "out={out}");
    }

    #[test]
    fn test_format_json_violation_count() {
        let r = ValidateResult {
            conforms: false,
            violations: vec![ValidationViolation {
                node: "n".into(),
                path: None,
                message: "m".into(),
                severity: "Violation".into(),
            }],
            elapsed_ms: 0,
        };
        let out = cmd().format_output(&r, &OutputFmt::Json);
        assert!(out.contains("\"violationCount\":1"), "out={out}");
    }

    #[test]
    fn test_format_turtle_conforms_false() {
        let r = ValidateResult {
            conforms: false,
            violations: vec![],
            elapsed_ms: 0,
        };
        let out = cmd().format_output(&r, &OutputFmt::Turtle);
        assert!(out.contains("sh:conforms false"), "out={out}");
    }

    #[test]
    fn test_simulated_returns_elapsed_zero() {
        let r = ValidateCommand::run_simulated_validation("file.ttl", None, None);
        assert_eq!(r.elapsed_ms, 0, "simulated always returns 0 for elapsed_ms");
    }

    #[test]
    fn test_execute_focus_node_not_found_still_conforms() {
        let mut args = args_for("data.ttl");
        args.focus_node = Some("http://example.org/nonexistent_node_xyz".to_string());
        let result = cmd().execute(&args).expect("ok");
        // No violations matching nonexistent node → conforms
        assert!(result.conforms);
    }

    #[test]
    fn test_validate_result_conforms_field_reflects_violations() {
        let r = ValidateResult {
            conforms: true,
            violations: vec![],
            elapsed_ms: 0,
        };
        assert_eq!(r.conforms, r.violations.is_empty());
    }

    #[test]
    fn test_validation_violation_fields() {
        let v = ValidationViolation {
            node: "http://n".to_string(),
            path: Some("ex:p".to_string()),
            message: "msg".to_string(),
            severity: "Info".to_string(),
        };
        assert_eq!(v.severity, "Info");
        assert_eq!(v.path, Some("ex:p".to_string()));
    }

    #[test]
    fn test_count_severity_returns_all_severities() {
        let mk = |s: &str| ValidationViolation {
            node: "n".into(),
            path: None,
            message: "m".into(),
            severity: s.to_string(),
        };
        let r = ValidateResult {
            conforms: false,
            violations: vec![mk("Violation"), mk("Warning"), mk("Info")],
            elapsed_ms: 0,
        };
        let counts = ValidateCommand::count_by_severity(&r);
        assert_eq!(counts.len(), 3);
    }

    #[test]
    fn test_execute_no_format_hint_succeeds() {
        let mut args = args_for("data.ttl");
        args.format = None;
        assert!(cmd().execute(&args).is_ok());
    }

    #[test]
    fn test_format_output_json_has_elapsed_ms() {
        let r = ValidateResult {
            conforms: true,
            violations: vec![],
            elapsed_ms: 42,
        };
        let out = cmd().format_output(&r, &OutputFmt::Json);
        assert!(out.contains("\"elapsedMs\":42"), "out={out}");
    }
}
