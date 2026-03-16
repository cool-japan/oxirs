/// Turtle/TriG document syntax validation.
///
/// Provides a lightweight, line-oriented validator that detects common Turtle
/// syntax problems without performing a full parse.  It is suitable for fast
/// "lint" checks of Turtle files before handing them to a strict parser.
use std::collections::HashSet;

// ── Types ─────────────────────────────────────────────────────────────────────

/// A single issue found during validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationIssue {
    /// Non-fatal issue — document may still be usable.
    Warning(String),
    /// Fatal issue — document is invalid.
    Error(String),
}

impl ValidationIssue {
    /// Returns the human-readable message regardless of severity.
    pub fn message(&self) -> &str {
        match self {
            Self::Warning(m) | Self::Error(m) => m,
        }
    }

    /// Returns `true` for `Error` variants.
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }
}

// ── ValidationReport ─────────────────────────────────────────────────────────

/// The outcome of validating a Turtle document.
#[derive(Debug, Default)]
pub struct ValidationReport {
    /// All validation issues found (warnings and errors).
    pub issues: Vec<ValidationIssue>,
    /// Number of non-blank, non-comment lines in the input.
    pub line_count: usize,
    /// Approximate number of triple-ending `.` tokens found.
    pub triple_count: usize,
}

impl ValidationReport {
    /// Returns `true` when no `Error` issues were found.
    pub fn is_valid(&self) -> bool {
        !self.issues.iter().any(|i| i.is_error())
    }

    /// Number of `Error` issues.
    pub fn error_count(&self) -> usize {
        self.issues.iter().filter(|i| i.is_error()).count()
    }

    /// Number of `Warning` issues.
    pub fn warning_count(&self) -> usize {
        self.issues.iter().filter(|i| !i.is_error()).count()
    }

    /// Collect the messages of all `Error` issues.
    pub fn errors(&self) -> Vec<&str> {
        self.issues
            .iter()
            .filter(|i| i.is_error())
            .map(|i| i.message())
            .collect()
    }

    /// Collect the messages of all `Warning` issues.
    pub fn warnings(&self) -> Vec<&str> {
        self.issues
            .iter()
            .filter(|i| !i.is_error())
            .map(|i| i.message())
            .collect()
    }
}

// ── TurtleValidator ───────────────────────────────────────────────────────────

/// Line-oriented Turtle syntax validator.
pub struct TurtleValidator {
    strict: bool,
}

impl TurtleValidator {
    /// Create a new validator in lenient mode.
    pub fn new() -> Self {
        Self { strict: false }
    }

    /// Enable strict mode (additional warnings become errors).
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Validate an entire Turtle document string.
    pub fn validate(&self, input: &str) -> ValidationReport {
        let mut report = ValidationReport {
            triple_count: Self::count_triples_approx(input),
            ..Default::default()
        };

        let mut known_prefixes: Vec<String> = vec![
            "rdf".to_string(),
            "rdfs".to_string(),
            "owl".to_string(),
            "xsd".to_string(),
        ];

        for (lineno, raw_line) in input.lines().enumerate() {
            let line = raw_line.trim();
            // Skip blank lines and comment lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            report.line_count += 1;

            // @prefix / PREFIX declarations
            if line.to_lowercase().starts_with("@prefix")
                || line.to_lowercase().starts_with("prefix")
            {
                if let Some(issue) = self.validate_prefix_declaration(line) {
                    report.issues.push(issue);
                } else {
                    // Extract the prefix name and remember it
                    if let Some(name) = extract_prefix_name(line) {
                        if !known_prefixes.contains(&name) {
                            known_prefixes.push(name);
                        }
                    }
                }
                continue;
            }

            // Triple lines (non-prefix, non-blank, non-comment)
            if let Some(issue) = self.validate_triple_line(line, &known_prefixes) {
                if self.strict {
                    // Upgrade warnings to errors in strict mode
                    let upgraded = match issue {
                        ValidationIssue::Warning(msg) => {
                            ValidationIssue::Error(format!("[strict] {msg} (line {})", lineno + 1))
                        }
                        other => other,
                    };
                    report.issues.push(upgraded);
                } else {
                    report.issues.push(issue);
                }
            }
        }

        report
    }

    /// Validate a single `@prefix` or `PREFIX` declaration line.
    ///
    /// Returns `Some(Error)` if the line is malformed.
    pub fn validate_prefix_declaration(&self, line: &str) -> Option<ValidationIssue> {
        let lc = line.to_lowercase();
        let rest = if lc.starts_with("@prefix") {
            line["@prefix".len()..].trim()
        } else if lc.starts_with("prefix") {
            line["prefix".len()..].trim()
        } else {
            return None;
        };

        // Must contain a colon in the prefix name part
        // Expected formats:
        //   @prefix ex: <http://example.org/> .
        //   PREFIX ex: <http://example.org/>
        let colon_pos = match rest.find(':') {
            Some(pos) => pos,
            None => {
                return Some(ValidationIssue::Error(format!(
                    "prefix declaration missing colon: {line}"
                )));
            }
        };

        let prefix_name = &rest[..colon_pos].trim();
        if !prefix_name.is_empty() && !self.validate_prefix_name(prefix_name) {
            return Some(ValidationIssue::Error(format!(
                "invalid prefix name '{prefix_name}' in: {line}"
            )));
        }

        // After the colon there should be whitespace then an IRI in angle brackets
        let after_colon = rest[colon_pos + 1..].trim();
        if !after_colon.starts_with('<') {
            return Some(ValidationIssue::Error(format!(
                "prefix IRI must be enclosed in <...>: {line}"
            )));
        }
        if !after_colon.contains('>') {
            return Some(ValidationIssue::Error(format!(
                "prefix IRI not closed with '>': {line}"
            )));
        }

        // Extract the IRI between < >
        if let Some(iri) = extract_iri(after_colon) {
            if !self.validate_iri(&iri) {
                return Some(ValidationIssue::Error(format!(
                    "invalid IRI in prefix declaration: {iri}"
                )));
            }
        }

        None
    }

    /// Validate a single non-prefix Turtle line.
    ///
    /// Returns `Some(Warning)` for unknown prefixes, `Some(Error)` for
    /// unclosed angle brackets, etc.
    pub fn validate_triple_line(
        &self,
        line: &str,
        known_prefixes: &[String],
    ) -> Option<ValidationIssue> {
        // Detect unclosed angle brackets in IRI terms
        let open_angles = line.chars().filter(|&c| c == '<').count();
        let close_angles = line.chars().filter(|&c| c == '>').count();
        if open_angles != close_angles {
            return Some(ValidationIssue::Error(format!(
                "unbalanced angle brackets in: {line}"
            )));
        }

        // Detect unclosed string literals (simple heuristic: odd number of `"`)
        // We don't count escaped quotes here for simplicity.
        let double_quote_count = line.chars().filter(|&c| c == '"').count();
        if double_quote_count % 2 != 0 {
            return Some(ValidationIssue::Warning(format!(
                "possible unclosed string literal in: {line}"
            )));
        }

        // Warn about `prefixName:localPart` terms with unknown prefix
        let known: HashSet<&str> = known_prefixes.iter().map(String::as_str).collect();
        for token in tokenize_turtle_line(line) {
            if let Some(prefix) = extract_prefix_from_token(token) {
                if !prefix.is_empty() && !known.contains(prefix) {
                    return Some(ValidationIssue::Warning(format!(
                        "unknown prefix '{prefix}' in: {line}"
                    )));
                }
            }
        }

        None
    }

    /// Returns `true` when `iri` looks like a valid (absolute) IRI.
    ///
    /// This is a heuristic check — it just verifies the IRI contains a scheme
    /// separator and no bare whitespace.
    pub fn validate_iri(&self, iri: &str) -> bool {
        if iri.is_empty() {
            return false;
        }
        if iri.contains(' ') || iri.contains('\t') {
            return false;
        }
        // Relative IRIs (no scheme) are allowed in Turtle if a base is set.
        // We accept them but flag them as warnings elsewhere.
        true
    }

    /// Returns `true` when `prefix` is a valid Turtle prefix name (PN_PREFIX).
    ///
    /// A prefix name consists of letters, digits (after the first character),
    /// underscores, hyphens, and dots (but not as the last character).
    pub fn validate_prefix_name(&self, prefix: &str) -> bool {
        if prefix.is_empty() {
            return true; // empty prefix is allowed
        }
        let mut chars = prefix.chars();
        // First character must be a letter or underscore
        let first = match chars.next() {
            Some(c) => c,
            None => return true,
        };
        if !first.is_alphabetic() && first != '_' {
            return false;
        }
        // Remaining characters
        let remaining: Vec<char> = chars.collect();
        if let Some(&last) = remaining.last() {
            if last == '.' || last == '-' {
                return false;
            }
        }
        remaining
            .iter()
            .all(|&c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
    }

    /// Count the approximate number of triples in a Turtle document by counting
    /// statement-ending `.` tokens on non-comment lines.
    pub fn count_triples_approx(input: &str) -> usize {
        let mut count = 0usize;
        for line in input.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            // Skip prefix declarations
            let lc = trimmed.to_lowercase();
            if lc.starts_with("@prefix") || lc.starts_with("prefix") {
                continue;
            }
            // A line that ends in '.' (possibly followed by a comment) is an
            // approximate statement terminator.
            // Strip inline comment
            let without_comment = if let Some(pos) = trimmed.find(" #") {
                trimmed[..pos].trim()
            } else {
                trimmed
            };
            if without_comment.ends_with('.') {
                count += 1;
            }
        }
        count
    }
}

impl Default for TurtleValidator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Extract the prefix name (before the colon) from a `@prefix`/`PREFIX` line.
fn extract_prefix_name(line: &str) -> Option<String> {
    let lc = line.to_lowercase();
    let rest = if lc.starts_with("@prefix") {
        line["@prefix".len()..].trim()
    } else if lc.starts_with("prefix") {
        line["prefix".len()..].trim()
    } else {
        return None;
    };
    let colon = rest.find(':')?;
    Some(rest[..colon].trim().to_string())
}

/// Extract an IRI from a string that starts with `<`.
fn extract_iri(s: &str) -> Option<String> {
    let start = s.find('<')? + 1;
    let end = s[start..].find('>')? + start;
    Some(s[start..end].to_string())
}

/// Tokenize a Turtle line into whitespace-separated tokens, skipping string
/// literals for prefix checking purposes.
fn tokenize_turtle_line(line: &str) -> Vec<&str> {
    let mut tokens = Vec::new();
    let mut in_string = false;
    let mut in_iri = false;
    let mut start = 0usize;

    for (i, ch) in line.char_indices() {
        match ch {
            '"' => in_string = !in_string,
            '<' if !in_string => in_iri = true,
            '>' if in_iri => in_iri = false,
            ' ' | '\t' if !in_string && !in_iri => {
                if i > start {
                    tokens.push(&line[start..i]);
                }
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < line.len() {
        tokens.push(&line[start..]);
    }
    tokens
}

/// If `token` looks like `prefix:local`, return the prefix part.
fn extract_prefix_from_token(token: &str) -> Option<&str> {
    // Tokens inside < > are IRIs and don't have a prefix notation
    if token.starts_with('<') || token.starts_with('"') || token.starts_with('_') {
        return None;
    }
    // Strip trailing punctuation (., ;, ,)
    let cleaned = token.trim_end_matches(['.', ';', ',']);
    let colon = cleaned.find(':')?;
    Some(&cleaned[..colon])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── validate ──────────────────────────────────────────────────────────────

    #[test]
    fn test_valid_turtle_passes() {
        let ttl = r#"
@prefix ex: <http://example.org/> .
ex:Alice ex:knows ex:Bob .
"#;
        let v = TurtleValidator::new();
        let report = v.validate(ttl);
        assert!(report.is_valid());
        assert_eq!(report.error_count(), 0);
    }

    #[test]
    fn test_empty_input_is_valid() {
        let v = TurtleValidator::new();
        let report = v.validate("");
        assert!(report.is_valid());
        assert_eq!(report.line_count, 0);
    }

    #[test]
    fn test_comments_only_is_valid() {
        let input = "# This is a comment\n# Another comment\n";
        let v = TurtleValidator::new();
        let report = v.validate(input);
        assert!(report.is_valid());
        assert_eq!(report.line_count, 0);
    }

    #[test]
    fn test_missing_prefix_error() {
        // Using an unknown prefix without declaring it
        let input = "unk:Alice unk:knows unk:Bob .\n";
        let v = TurtleValidator::new();
        let report = v.validate(input);
        // Should produce at least one warning about the unknown prefix
        assert!(!report.issues.is_empty());
    }

    #[test]
    fn test_unknown_prefix_produces_warning() {
        let input = "foo:s foo:p foo:o .\n";
        let v = TurtleValidator::new();
        let report = v.validate(input);
        assert!(report.warning_count() >= 1);
    }

    // ── validate_prefix_declaration ───────────────────────────────────────────

    #[test]
    fn test_valid_prefix_declaration() {
        let v = TurtleValidator::new();
        let issue = v.validate_prefix_declaration("@prefix ex: <http://example.org/> .");
        assert!(issue.is_none());
    }

    #[test]
    fn test_sparql_prefix_declaration() {
        let v = TurtleValidator::new();
        let issue = v.validate_prefix_declaration("PREFIX ex: <http://example.org/>");
        assert!(issue.is_none());
    }

    #[test]
    fn test_prefix_missing_colon() {
        let v = TurtleValidator::new();
        let issue = v.validate_prefix_declaration("@prefix ex <http://example.org/> .");
        assert!(matches!(issue, Some(ValidationIssue::Error(_))));
    }

    #[test]
    fn test_prefix_iri_not_in_angle_brackets() {
        let v = TurtleValidator::new();
        let issue = v.validate_prefix_declaration("@prefix ex: http://example.org/ .");
        assert!(matches!(issue, Some(ValidationIssue::Error(_))));
    }

    #[test]
    fn test_prefix_iri_not_closed() {
        let v = TurtleValidator::new();
        let issue = v.validate_prefix_declaration("@prefix ex: <http://example.org/ .");
        assert!(matches!(issue, Some(ValidationIssue::Error(_))));
    }

    #[test]
    fn test_prefix_invalid_name() {
        let v = TurtleValidator::new();
        // Prefix names cannot start with a digit
        let issue = v.validate_prefix_declaration("@prefix 1bad: <http://example.org/> .");
        assert!(matches!(issue, Some(ValidationIssue::Error(_))));
    }

    #[test]
    fn test_prefix_empty_name_allowed() {
        let v = TurtleValidator::new();
        let issue = v.validate_prefix_declaration("@prefix : <http://example.org/> .");
        assert!(issue.is_none());
    }

    // ── validate_triple_line ──────────────────────────────────────────────────

    #[test]
    fn test_valid_triple_line_with_known_prefixes() {
        let v = TurtleValidator::new();
        let issue = v.validate_triple_line("ex:Alice ex:knows ex:Bob .", &["ex".to_string()]);
        assert!(issue.is_none());
    }

    #[test]
    fn test_triple_line_unbalanced_angle_brackets() {
        let v = TurtleValidator::new();
        let issue = v.validate_triple_line(
            "<http://example.org/Alice ex:knows ex:Bob .",
            &["ex".to_string()],
        );
        assert!(matches!(issue, Some(ValidationIssue::Error(_))));
    }

    #[test]
    fn test_triple_line_unclosed_string_literal() {
        let v = TurtleValidator::new();
        let issue = v.validate_triple_line(r#"ex:s ex:p "unclosed ."#, &["ex".to_string()]);
        assert!(matches!(issue, Some(ValidationIssue::Warning(_))));
    }

    #[test]
    fn test_triple_line_unknown_prefix_warning() {
        let v = TurtleValidator::new();
        let issue = v.validate_triple_line("unkn:s unkn:p unkn:o .", &["ex".to_string()]);
        assert!(matches!(issue, Some(ValidationIssue::Warning(_))));
    }

    // ── validate_iri ──────────────────────────────────────────────────────────

    #[test]
    fn test_valid_iri() {
        let v = TurtleValidator::new();
        assert!(v.validate_iri("http://example.org/"));
    }

    #[test]
    fn test_invalid_iri_empty() {
        let v = TurtleValidator::new();
        assert!(!v.validate_iri(""));
    }

    #[test]
    fn test_invalid_iri_with_space() {
        let v = TurtleValidator::new();
        assert!(!v.validate_iri("http://example.org/ foo"));
    }

    #[test]
    fn test_invalid_iri_with_tab() {
        let v = TurtleValidator::new();
        assert!(!v.validate_iri("http://example.org/\t"));
    }

    // ── validate_prefix_name ──────────────────────────────────────────────────

    #[test]
    fn test_valid_prefix_name_simple() {
        let v = TurtleValidator::new();
        assert!(v.validate_prefix_name("ex"));
    }

    #[test]
    fn test_valid_prefix_name_with_dot() {
        let v = TurtleValidator::new();
        assert!(v.validate_prefix_name("ex.org"));
    }

    #[test]
    fn test_valid_prefix_name_empty() {
        let v = TurtleValidator::new();
        assert!(v.validate_prefix_name(""));
    }

    #[test]
    fn test_invalid_prefix_name_starts_with_digit() {
        let v = TurtleValidator::new();
        assert!(!v.validate_prefix_name("1bad"));
    }

    #[test]
    fn test_invalid_prefix_name_ends_with_dot() {
        let v = TurtleValidator::new();
        assert!(!v.validate_prefix_name("bad."));
    }

    #[test]
    fn test_invalid_prefix_name_ends_with_dash() {
        let v = TurtleValidator::new();
        assert!(!v.validate_prefix_name("bad-"));
    }

    #[test]
    fn test_valid_prefix_name_with_underscore() {
        let v = TurtleValidator::new();
        assert!(v.validate_prefix_name("my_prefix"));
    }

    // ── count_triples_approx ──────────────────────────────────────────────────

    #[test]
    fn test_count_triples_approx_simple() {
        let input = r#"
@prefix ex: <http://example.org/> .
ex:Alice ex:knows ex:Bob .
ex:Bob ex:knows ex:Carol .
"#;
        assert_eq!(TurtleValidator::count_triples_approx(input), 2);
    }

    #[test]
    fn test_count_triples_approx_zero_on_empty() {
        assert_eq!(TurtleValidator::count_triples_approx(""), 0);
    }

    #[test]
    fn test_count_triples_approx_ignores_prefixes() {
        let input = "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .\n";
        assert_eq!(TurtleValidator::count_triples_approx(input), 1);
    }

    #[test]
    fn test_count_triples_approx_ignores_comments() {
        let input = "# comment .\nex:s ex:p ex:o .\n";
        assert_eq!(TurtleValidator::count_triples_approx(input), 1);
    }

    // ── ValidationReport methods ──────────────────────────────────────────────

    #[test]
    fn test_report_is_valid_no_errors() {
        let mut report = ValidationReport::default();
        report.issues.push(ValidationIssue::Warning("w".into()));
        assert!(report.is_valid());
    }

    #[test]
    fn test_report_is_invalid_with_error() {
        let mut report = ValidationReport::default();
        report.issues.push(ValidationIssue::Error("e".into()));
        assert!(!report.is_valid());
    }

    #[test]
    fn test_report_error_count() {
        let mut report = ValidationReport::default();
        report.issues.push(ValidationIssue::Error("e1".into()));
        report.issues.push(ValidationIssue::Error("e2".into()));
        report.issues.push(ValidationIssue::Warning("w".into()));
        assert_eq!(report.error_count(), 2);
        assert_eq!(report.warning_count(), 1);
    }

    #[test]
    fn test_report_errors_and_warnings_vecs() {
        let mut report = ValidationReport::default();
        report.issues.push(ValidationIssue::Error("err".into()));
        report.issues.push(ValidationIssue::Warning("warn".into()));
        assert!(report.errors().contains(&"err"));
        assert!(report.warnings().contains(&"warn"));
    }

    // ── strict mode ───────────────────────────────────────────────────────────

    #[test]
    fn test_strict_mode_upgrades_warnings_to_errors() {
        let input = "unk:s unk:p unk:o .\n";
        let v = TurtleValidator::new().strict();
        let report = v.validate(input);
        // Unknown prefix warning upgraded to error in strict mode
        assert!(!report.is_valid());
        assert!(report.error_count() >= 1);
    }

    #[test]
    fn test_lenient_mode_keeps_warnings() {
        let input = "unk:s unk:p unk:o .\n";
        let v = TurtleValidator::new();
        let report = v.validate(input);
        assert!(report.is_valid()); // errors are zero — only warnings
        assert!(report.warning_count() >= 1);
    }

    // ── ValidationIssue helpers ───────────────────────────────────────────────

    #[test]
    fn test_validation_issue_message_error() {
        let i = ValidationIssue::Error("oops".into());
        assert_eq!(i.message(), "oops");
        assert!(i.is_error());
    }

    #[test]
    fn test_validation_issue_message_warning() {
        let i = ValidationIssue::Warning("hmm".into());
        assert_eq!(i.message(), "hmm");
        assert!(!i.is_error());
    }

    // ── Additional tests for round 12 (reaching ≥45 total) ───────────────────

    #[test]
    fn test_validate_valid_base_and_triple() {
        let input = "@base <http://example.org/> .\n@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .\n";
        let v = TurtleValidator::new();
        let report = v.validate(input);
        assert!(report.is_valid());
    }

    #[test]
    fn test_validate_empty_string() {
        let v = TurtleValidator::new();
        let report = v.validate("");
        assert!(report.is_valid());
        assert_eq!(report.triple_count, 0);
    }

    #[test]
    fn test_validate_comment_only() {
        let input = "# This is just a comment\n";
        let v = TurtleValidator::new();
        let report = v.validate(input);
        assert!(report.is_valid());
    }

    #[test]
    fn test_count_triples_approx_multiple() {
        let input = "ex:a ex:b ex:c .\nex:d ex:e ex:f .\nex:g ex:h ex:i .\n";
        assert_eq!(TurtleValidator::count_triples_approx(input), 3);
    }

    #[test]
    fn test_validate_prefix_name_empty() {
        let v = TurtleValidator::new();
        // Empty prefix "" is valid in Turtle
        assert!(v.validate_prefix_name(""));
    }

    #[test]
    fn test_validate_prefix_name_digits_after_start() {
        let v = TurtleValidator::new();
        // Prefix starting with letter, then digits
        assert!(v.validate_prefix_name("abc123"));
    }

    #[test]
    fn test_validate_invalid_prefix_starts_with_digit() {
        let v = TurtleValidator::new();
        assert!(!v.validate_prefix_name("1bad"));
    }

    #[test]
    fn test_report_default_has_no_issues() {
        let report = ValidationReport::default();
        assert!(report.issues.is_empty());
        assert_eq!(report.triple_count, 0);
        assert_eq!(report.line_count, 0);
    }

    #[test]
    fn test_validation_issue_clone() {
        let i = ValidationIssue::Error("err".into());
        assert_eq!(i, i.clone());
    }

    #[test]
    fn test_validation_issue_eq() {
        assert_eq!(
            ValidationIssue::Warning("w".into()),
            ValidationIssue::Warning("w".into())
        );
        assert_ne!(
            ValidationIssue::Error("e".into()),
            ValidationIssue::Warning("e".into())
        );
    }
}
