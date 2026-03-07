//! SPARQL query validator with multi-rule syntax checking.
//!
//! This module validates SPARQL query strings against a set of structural
//! and syntactic rules without executing a full SPARQL parser. It detects
//! common mistakes such as unbalanced delimiters, missing query forms,
//! malformed PREFIX declarations, and invalid LIMIT/OFFSET values.
//!
//! # Example
//!
//! ```rust
//! use oxirs::commands::query_validator::{QueryValidator, SparqlSyntaxChecker};
//!
//! let v = QueryValidator::new(false);
//! let result = v.validate("SELECT ?s WHERE { ?s a <http://example.org/T> }");
//! assert!(result.is_valid);
//! assert!(result.issues.is_empty());
//!
//! let bad = v.validate("SELECT WHERE { }");
//! // Missing explicit variable list without *
//! assert!(!bad.issues.is_empty());
//! ```

use std::fmt;

/// Severity level of a validation issue
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    /// The query is likely incorrect or cannot be executed
    Error,
    /// The query may work but has a suspicious construct
    Warning,
    /// Informational note that does not affect correctness
    Info,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Error => write!(f, "ERROR"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Info => write!(f, "INFO"),
        }
    }
}

/// A single validation issue found in a query string
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity of the issue
    pub severity: Severity,
    /// Human-readable description of the problem
    pub message: String,
    /// Optional (line, column) position within the query (1-indexed)
    pub position: Option<(usize, usize)>,
}

impl ValidationIssue {
    /// Create an error issue with optional position
    pub fn error(message: impl Into<String>, position: Option<(usize, usize)>) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
            position,
        }
    }

    /// Create a warning issue with optional position
    pub fn warning(message: impl Into<String>, position: Option<(usize, usize)>) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            position,
        }
    }

    /// Create an info issue
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Info,
            message: message.into(),
            position: None,
        }
    }
}

/// Aggregated result from a validation run
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// `true` if no Error-severity issues were found
    pub is_valid: bool,
    /// All issues found (may be empty)
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    fn new(issues: Vec<ValidationIssue>) -> Self {
        let is_valid = !issues.iter().any(|i| i.severity == Severity::Error);
        Self { is_valid, issues }
    }
}

/// The main SPARQL query validator
#[derive(Debug, Clone)]
pub struct QueryValidator {
    /// When `true`, warnings are escalated to errors
    pub strict_mode: bool,
}

impl QueryValidator {
    /// Create a new validator.
    ///
    /// `strict_mode = true` causes all warnings to become errors, making
    /// `is_valid = false` if any warning is present.
    pub fn new(strict_mode: bool) -> Self {
        Self { strict_mode }
    }

    /// Validate the given SPARQL query string.
    pub fn validate(&self, query: &str) -> ValidationResult {
        let mut issues: Vec<ValidationIssue> = Vec::new();

        issues.extend(self.check_balanced_braces(query));
        issues.extend(self.check_iris(query));
        issues.extend(self.check_prefixes(query));
        issues.extend(self.check_query_form(query));
        issues.extend(self.check_variables(query));
        issues.extend(self.check_limit_offset(query));

        if self.strict_mode {
            // Escalate warnings to errors
            for issue in &mut issues {
                if issue.severity == Severity::Warning {
                    issue.severity = Severity::Error;
                }
            }
        }

        ValidationResult::new(issues)
    }

    // -----------------------------------------------------------------------
    // Check: balanced braces
    // -----------------------------------------------------------------------

    /// Verify that `{` and `}` are balanced in the query (outside string literals).
    pub fn check_balanced_braces(&self, query: &str) -> Vec<ValidationIssue> {
        let mut depth: i64 = 0;
        let mut in_string = false;
        let mut string_char = '"';
        let mut issues = Vec::new();

        let chars: Vec<char> = query.chars().collect();
        let mut i = 0;
        let (mut line, mut col) = (1usize, 1usize);

        while i < chars.len() {
            let ch = chars[i];

            // Track line/column
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }

            // Toggle string context on unescaped quote
            if (ch == '"' || ch == '\'') && !in_string {
                in_string = true;
                string_char = ch;
                i += 1;
                continue;
            }
            if in_string && ch == string_char {
                // Check for escape
                let escaped = i > 0 && chars[i - 1] == '\\';
                if !escaped {
                    in_string = false;
                }
                i += 1;
                continue;
            }
            if in_string {
                i += 1;
                continue;
            }

            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth < 0 {
                        issues.push(ValidationIssue::error(
                            "Unexpected closing brace '}'",
                            Some((line, col)),
                        ));
                        depth = 0; // reset to avoid cascading errors
                    }
                }
                _ => {}
            }
            i += 1;
        }

        if depth > 0 {
            issues.push(ValidationIssue::error(
                format!("{} unclosed brace(s) '{{' in query", depth),
                None,
            ));
        }

        issues
    }

    // -----------------------------------------------------------------------
    // Check: IRI angle brackets
    // -----------------------------------------------------------------------

    /// Verify that `<` and `>` used for IRIs are balanced, and warn on empty `<>`.
    pub fn check_iris(&self, query: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let mut in_string = false;
        let mut string_char = '"';
        let mut open_count: i64 = 0;

        let chars: Vec<char> = query.chars().collect();
        let mut i = 0;
        let (mut line, mut col) = (1usize, 1usize);

        while i < chars.len() {
            let ch = chars[i];
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }

            if (ch == '"' || ch == '\'') && !in_string {
                in_string = true;
                string_char = ch;
                i += 1;
                continue;
            }
            if in_string && ch == string_char {
                let escaped = i > 0 && chars[i - 1] == '\\';
                if !escaped {
                    in_string = false;
                }
                i += 1;
                continue;
            }
            if in_string {
                i += 1;
                continue;
            }

            if ch == '<' {
                // Check for empty IRI <>
                if i + 1 < chars.len() && chars[i + 1] == '>' {
                    issues.push(ValidationIssue::warning(
                        "Empty IRI '<>' found",
                        Some((line, col)),
                    ));
                }
                open_count += 1;
            } else if ch == '>' {
                open_count -= 1;
                if open_count < 0 {
                    issues.push(ValidationIssue::error(
                        "Unexpected '>' without matching '<'",
                        Some((line, col)),
                    ));
                    open_count = 0;
                }
            }

            i += 1;
        }

        if open_count > 0 {
            issues.push(ValidationIssue::error(
                format!("{} unclosed '<' in query", open_count),
                None,
            ));
        }

        issues
    }

    // -----------------------------------------------------------------------
    // Check: PREFIX declarations
    // -----------------------------------------------------------------------

    /// Verify that PREFIX declarations follow the pattern `PREFIX qname: <IRI>`.
    pub fn check_prefixes(&self, query: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let upper = query.to_uppercase();

        // Split by PREFIX keyword
        let mut search_start = 0;
        while let Some(pos) = upper[search_start..].find("PREFIX") {
            let abs_pos = search_start + pos;

            // Skip if inside a string (simplified: check quote balance before this position)
            let before = &query[..abs_pos];
            let quote_count = before.chars().filter(|&c| c == '"').count();
            if quote_count % 2 != 0 {
                search_start = abs_pos + 6;
                continue;
            }

            // Extract the rest of the line after PREFIX
            let after_prefix = query[abs_pos + 6..].trim_start();
            let line_end = after_prefix.find('\n').unwrap_or(after_prefix.len());
            let decl = after_prefix[..line_end].trim();

            // Expected pattern: `qname: <IRI>` where qname may be empty (default prefix)
            // Simple regex-free check:
            //   1. There must be a colon
            //   2. After colon (and whitespace) must be `<...>`
            if let Some(colon_pos) = decl.find(':') {
                let after_colon = decl[colon_pos + 1..].trim();
                if !after_colon.starts_with('<') {
                    issues.push(ValidationIssue::error(
                        format!(
                            "PREFIX declaration IRI must start with '<', got: '{}'",
                            &after_colon.chars().take(20).collect::<String>()
                        ),
                        None,
                    ));
                } else if !after_colon.contains('>') {
                    issues.push(ValidationIssue::error(
                        "PREFIX declaration IRI is not closed with '>'",
                        None,
                    ));
                }
            } else {
                issues.push(ValidationIssue::error(
                    "PREFIX declaration missing ':' after namespace prefix",
                    None,
                ));
            }

            search_start = abs_pos + 6;
        }

        issues
    }

    // -----------------------------------------------------------------------
    // Check: query form
    // -----------------------------------------------------------------------

    /// Verify the query has a recognised query form keyword and WHERE clause.
    pub fn check_query_form(&self, query: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let upper = query.to_uppercase();

        // Strip PREFIX lines for analysis
        let body = strip_prefix_lines(query);
        let body_upper = body.to_uppercase();

        let has_select = body_upper.contains("SELECT");
        let has_ask = body_upper.contains("ASK");
        let has_construct = body_upper.contains("CONSTRUCT");
        let has_describe = body_upper.contains("DESCRIBE");

        if !has_select && !has_ask && !has_construct && !has_describe {
            issues.push(ValidationIssue::error(
                "Query must contain one of: SELECT, ASK, CONSTRUCT, DESCRIBE",
                None,
            ));
            return issues;
        }

        // WHERE clause required for SELECT and CONSTRUCT
        if (has_select || has_construct) && !upper.contains("WHERE") {
            issues.push(ValidationIssue::error(
                "SELECT and CONSTRUCT queries require a WHERE clause",
                None,
            ));
        }

        // SELECT: must have at least `?var` or `*` after SELECT
        if has_select {
            let select_issues = self.check_select_variables(query);
            issues.extend(select_issues);
        }

        issues
    }

    /// Check that SELECT has a variable list (`?var` / `$var` / `*`)
    fn check_select_variables(&self, query: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let upper = query.to_uppercase();

        // Find SELECT position (prefer the one outside PREFIX lines)
        let body = strip_prefix_lines(query);
        let body_upper = body.to_uppercase();

        if let Some(sel_pos) = body_upper.find("SELECT") {
            let after_select = body[sel_pos + 6..].trim_start();
            // Skip DISTINCT/REDUCED modifier
            let after_modifier = after_select
                .trim_start_matches("DISTINCT")
                .trim_start_matches("REDUCED")
                .trim_start();

            // Must start with `*`, `?`, or `$`
            let first_meaningful: char = after_modifier.chars().next().unwrap_or(' ');
            if first_meaningful != '*' && first_meaningful != '?' && first_meaningful != '$' {
                issues.push(ValidationIssue::error(
                    "SELECT must be followed by '*, ?variable, or $variable'",
                    None,
                ));
            }
        }

        let _ = upper;
        issues
    }

    // -----------------------------------------------------------------------
    // Check: variable names
    // -----------------------------------------------------------------------

    /// Verify that all `?` / `$` variable references have valid names.
    ///
    /// Valid: `?name`, `?my_var`, `?var123`
    /// Invalid: `?`, `? name`, `?1bad`
    pub fn check_variables(&self, query: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let chars: Vec<char> = query.chars().collect();
        let (mut line, mut col) = (1usize, 1usize);
        let mut in_string = false;
        let mut string_char = '"';

        let mut i = 0;
        while i < chars.len() {
            let ch = chars[i];
            if ch == '\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }

            if (ch == '"' || ch == '\'') && !in_string {
                in_string = true;
                string_char = ch;
                i += 1;
                continue;
            }
            if in_string && ch == string_char {
                let escaped = i > 0 && chars[i - 1] == '\\';
                if !escaped {
                    in_string = false;
                }
                i += 1;
                continue;
            }
            if in_string {
                i += 1;
                continue;
            }

            if ch == '?' || ch == '$' {
                // Collect the variable name
                let var_start = i;
                let _ = var_start;
                let mut j = i + 1;
                while j < chars.len() {
                    let c = chars[j];
                    if c.is_alphanumeric() || c == '_' {
                        j += 1;
                    } else {
                        break;
                    }
                }
                let var_name: String = chars[i + 1..j].iter().collect();

                if var_name.is_empty() {
                    issues.push(ValidationIssue::error(
                        format!(
                            "Empty variable name after '{}'; expected alphanumeric identifier",
                            ch
                        ),
                        Some((line, col)),
                    ));
                } else {
                    // First character of the variable name must be a letter or underscore
                    let first = var_name.chars().next().unwrap_or('0');
                    if first.is_ascii_digit() {
                        issues.push(ValidationIssue::error(
                            format!(
                                "Variable name '{}{}'  must not start with a digit",
                                ch, var_name
                            ),
                            Some((line, col)),
                        ));
                    }
                }
                i = j;
                continue;
            }

            i += 1;
        }

        issues
    }

    // -----------------------------------------------------------------------
    // Check: LIMIT / OFFSET values
    // -----------------------------------------------------------------------

    /// Verify that LIMIT and OFFSET clauses have non-negative integer values.
    pub fn check_limit_offset(&self, query: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let upper = query.to_uppercase();

        for keyword in &["LIMIT", "OFFSET"] {
            let mut search = upper.as_str();
            while let Some(kw_pos) = search.find(keyword) {
                let remainder = search[kw_pos + keyword.len()..].trim_start();

                // Collect the value token
                let value_str: String = remainder
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '-' || *c == '+')
                    .collect();

                if value_str.is_empty() {
                    issues.push(ValidationIssue::error(
                        format!("{} must be followed by a non-negative integer", keyword),
                        None,
                    ));
                } else {
                    match value_str.parse::<i64>() {
                        Ok(v) if v < 0 => {
                            issues.push(ValidationIssue::error(
                                format!("{} value must be non-negative, got {}", keyword, v),
                                None,
                            ));
                        }
                        Err(_) => {
                            issues.push(ValidationIssue::error(
                                format!("{} value is not a valid integer: {}", keyword, value_str),
                                None,
                            ));
                        }
                        _ => {}
                    }
                }

                search = &search[kw_pos + keyword.len()..];
                if search.is_empty() {
                    break;
                }
            }
        }

        issues
    }

    // -----------------------------------------------------------------------
    // Issue formatter
    // -----------------------------------------------------------------------

    /// Format a single issue as `[SEVERITY] line N:col M: message`
    pub fn format_issue(issue: &ValidationIssue) -> String {
        match issue.position {
            Some((line, col)) => {
                format!(
                    "[{}] line {}:col {}: {}",
                    issue.severity, line, col, issue.message
                )
            }
            None => format!("[{}] {}", issue.severity, issue.message),
        }
    }
}

/// Quick structural check that SELECT and WHERE appear in the correct order.
pub struct SparqlSyntaxChecker;

impl SparqlSyntaxChecker {
    /// Return `true` if the query appears to have a valid keyword sequence.
    ///
    /// Specifically checks that:
    /// 1. Some query form keyword appears (SELECT / ASK / CONSTRUCT / DESCRIBE)
    /// 2. For SELECT and CONSTRUCT, WHERE appears after the form keyword
    pub fn check_keyword_sequence(query: &str) -> bool {
        let upper = query.to_uppercase();
        let body_upper = strip_prefix_lines(query).to_uppercase();

        let has_ask = body_upper.contains("ASK");
        let has_describe = body_upper.contains("DESCRIBE");

        if has_ask || has_describe {
            return true;
        }

        let select_pos = body_upper.find("SELECT");
        let construct_pos = body_upper.find("CONSTRUCT");

        // For SELECT or CONSTRUCT, WHERE must follow
        let form_pos = match (select_pos, construct_pos) {
            (Some(s), Some(c)) => Some(s.min(c)),
            (Some(s), None) => Some(s),
            (None, Some(c)) => Some(c),
            (None, None) => None,
        };

        match form_pos {
            None => false,
            Some(fp) => {
                if let Some(where_pos) = upper.find("WHERE") {
                    where_pos > fp
                } else {
                    false
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Remove PREFIX declaration lines from the query for form-level analysis.
fn strip_prefix_lines(query: &str) -> String {
    query
        .lines()
        .filter(|line| !line.trim().to_uppercase().starts_with("PREFIX"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_select() -> &'static str {
        "SELECT ?s ?p ?o WHERE { ?s ?p ?o }"
    }

    fn valid_ask() -> &'static str {
        "ASK { <http://example.org/a> a <http://example.org/T> }"
    }

    // -----------------------------------------------------------------------
    // full validate()
    // -----------------------------------------------------------------------

    #[test]
    fn test_valid_select_is_valid() {
        let v = QueryValidator::new(false);
        let r = v.validate(valid_select());
        assert!(r.is_valid, "Issues: {:?}", r.issues);
    }

    #[test]
    fn test_valid_ask_is_valid() {
        let v = QueryValidator::new(false);
        let r = v.validate(valid_ask());
        assert!(r.is_valid, "Issues: {:?}", r.issues);
    }

    #[test]
    fn test_valid_construct_is_valid() {
        let v = QueryValidator::new(false);
        let q = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }";
        let r = v.validate(q);
        assert!(r.is_valid, "Issues: {:?}", r.issues);
    }

    #[test]
    fn test_valid_describe_is_valid() {
        let v = QueryValidator::new(false);
        let q = "DESCRIBE <http://example.org/resource>";
        let r = v.validate(q);
        assert!(r.is_valid, "Issues: {:?}", r.issues);
    }

    #[test]
    fn test_valid_query_with_prefix_is_valid() {
        let v = QueryValidator::new(false);
        let q = "PREFIX ex: <http://example.org/>\nSELECT ?s WHERE { ?s a ex:Thing }";
        let r = v.validate(q);
        assert!(r.is_valid, "Issues: {:?}", r.issues);
    }

    #[test]
    fn test_valid_select_star_is_valid() {
        let v = QueryValidator::new(false);
        let q = "SELECT * WHERE { ?s ?p ?o }";
        let r = v.validate(q);
        assert!(r.is_valid, "Issues: {:?}", r.issues);
    }

    #[test]
    fn test_valid_limit_offset() {
        let v = QueryValidator::new(false);
        let q = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 10 OFFSET 0";
        let r = v.validate(q);
        assert!(r.is_valid, "Issues: {:?}", r.issues);
    }

    // -----------------------------------------------------------------------
    // check_balanced_braces
    // -----------------------------------------------------------------------

    #[test]
    fn test_unmatched_open_brace_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_balanced_braces("SELECT ?s WHERE { ?s ?p ?o");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_unmatched_close_brace_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_balanced_braces("SELECT ?s WHERE ?s ?p ?o }");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_balanced_braces_no_issues() {
        let v = QueryValidator::new(false);
        let issues = v.check_balanced_braces("SELECT ?s WHERE { ?s ?p ?o }");
        assert!(issues.is_empty());
    }

    #[test]
    fn test_nested_balanced_braces_no_issues() {
        let v = QueryValidator::new(false);
        let issues = v.check_balanced_braces("SELECT ?s WHERE { { ?s ?p ?o } }");
        assert!(issues.is_empty());
    }

    #[test]
    fn test_brace_inside_string_ignored() {
        let v = QueryValidator::new(false);
        let issues = v.check_balanced_braces(r#"SELECT ?s WHERE { BIND("value{" AS ?v) }"#);
        assert!(issues.is_empty(), "Issues: {:?}", issues);
    }

    // -----------------------------------------------------------------------
    // check_iris
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_iri_is_warning() {
        let v = QueryValidator::new(false);
        let issues = v.check_iris("SELECT ?s WHERE { ?s a <> }");
        assert!(issues.iter().any(|i| i.severity == Severity::Warning));
    }

    #[test]
    fn test_unclosed_iri_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_iris("SELECT ?s WHERE { ?s a <http://example.org/T }");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_valid_iri_no_issues() {
        let v = QueryValidator::new(false);
        let issues = v.check_iris("SELECT ?s WHERE { ?s a <http://example.org/T> }");
        assert!(issues.is_empty(), "Issues: {:?}", issues);
    }

    // -----------------------------------------------------------------------
    // check_prefixes
    // -----------------------------------------------------------------------

    #[test]
    fn test_valid_prefix_no_issues() {
        let v = QueryValidator::new(false);
        let issues = v.check_prefixes("PREFIX ex: <http://example.org/>\nSELECT * WHERE {}");
        assert!(issues.is_empty(), "Issues: {:?}", issues);
    }

    #[test]
    fn test_prefix_missing_colon_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_prefixes("PREFIX ex <http://example.org/>\nSELECT * WHERE {}");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_prefix_iri_not_angle_bracket_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_prefixes("PREFIX ex: http://example.org/\nSELECT * WHERE {}");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_prefix_unclosed_iri_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_prefixes("PREFIX ex: <http://example.org/\nSELECT * WHERE {}");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    // -----------------------------------------------------------------------
    // check_query_form
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_query_form_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_query_form("WHERE { ?s ?p ?o }");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_select_without_where_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_query_form("SELECT ?s { ?s ?p ?o }");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_select_without_variables_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_query_form("SELECT WHERE { ?s ?p ?o }");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_ask_without_where_is_valid() {
        let v = QueryValidator::new(false);
        let issues = v.check_query_form("ASK { ?s ?p ?o }");
        assert!(issues.is_empty(), "Issues: {:?}", issues);
    }

    // -----------------------------------------------------------------------
    // check_variables
    // -----------------------------------------------------------------------

    #[test]
    fn test_valid_variables_no_issues() {
        let v = QueryValidator::new(false);
        let issues = v.check_variables("SELECT ?subject WHERE { ?subject a ?type }");
        assert!(issues.is_empty(), "Issues: {:?}", issues);
    }

    #[test]
    fn test_empty_variable_name_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_variables("SELECT ? WHERE { ? a ?type }");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_variable_starting_with_digit_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_variables("SELECT ?1bad WHERE { ?1bad ?p ?o }");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_dollar_sign_variable_valid() {
        let v = QueryValidator::new(false);
        let issues = v.check_variables("SELECT $s WHERE { $s ?p ?o }");
        assert!(issues.is_empty(), "Issues: {:?}", issues);
    }

    // -----------------------------------------------------------------------
    // check_limit_offset
    // -----------------------------------------------------------------------

    #[test]
    fn test_valid_limit_no_issues() {
        let v = QueryValidator::new(false);
        let issues = v.check_limit_offset("SELECT ?s WHERE { ?s ?p ?o } LIMIT 10");
        assert!(issues.is_empty(), "Issues: {:?}", issues);
    }

    #[test]
    fn test_negative_limit_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_limit_offset("SELECT ?s WHERE { ?s ?p ?o } LIMIT -1");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_negative_offset_is_error() {
        let v = QueryValidator::new(false);
        let issues = v.check_limit_offset("SELECT ?s WHERE { ?s ?p ?o } OFFSET -5");
        assert!(issues.iter().any(|i| i.severity == Severity::Error));
    }

    #[test]
    fn test_limit_zero_is_valid() {
        let v = QueryValidator::new(false);
        let issues = v.check_limit_offset("SELECT ?s WHERE { ?s ?p ?o } LIMIT 0");
        assert!(issues.is_empty(), "Issues: {:?}", issues);
    }

    #[test]
    fn test_limit_offset_both_valid() {
        let v = QueryValidator::new(false);
        let issues = v.check_limit_offset("SELECT ?s WHERE { ?s ?p ?o } LIMIT 25 OFFSET 50");
        assert!(issues.is_empty(), "Issues: {:?}", issues);
    }

    // -----------------------------------------------------------------------
    // format_issue
    // -----------------------------------------------------------------------

    #[test]
    fn test_format_issue_with_position() {
        let issue = ValidationIssue::error("bad braces", Some((3, 10)));
        let s = QueryValidator::format_issue(&issue);
        assert!(s.contains("[ERROR]"));
        assert!(s.contains("line 3"));
        assert!(s.contains("bad braces"));
    }

    #[test]
    fn test_format_issue_without_position() {
        let issue = ValidationIssue::warning("empty IRI", None);
        let s = QueryValidator::format_issue(&issue);
        assert!(s.contains("[WARNING]"));
        assert!(!s.contains("line"));
    }

    #[test]
    fn test_format_issue_info() {
        let issue = ValidationIssue::info("note");
        let s = QueryValidator::format_issue(&issue);
        assert!(s.contains("[INFO]"));
    }

    // -----------------------------------------------------------------------
    // strict_mode
    // -----------------------------------------------------------------------

    #[test]
    fn test_strict_mode_escalates_warning_to_error() {
        let v = QueryValidator::new(true);
        // Empty IRI generates a warning in non-strict mode
        let q = "SELECT ?s WHERE { ?s a <> }";
        let r = v.validate(q);
        // All warnings should now be errors → is_valid = false
        assert!(r.issues.iter().all(|i| i.severity == Severity::Error));
        assert!(!r.is_valid);
    }

    #[test]
    fn test_non_strict_mode_warning_still_valid() {
        let v = QueryValidator::new(false);
        // Empty IRI is only a warning → is_valid remains true
        let q = "SELECT ?s WHERE { ?s a <> }";
        let r = v.validate(q);
        assert!(r.issues.iter().any(|i| i.severity == Severity::Warning));
        assert!(r.is_valid);
    }

    // -----------------------------------------------------------------------
    // SparqlSyntaxChecker
    // -----------------------------------------------------------------------

    #[test]
    fn test_syntax_checker_valid_select() {
        assert!(SparqlSyntaxChecker::check_keyword_sequence(
            "SELECT ?s WHERE { ?s ?p ?o }"
        ));
    }

    #[test]
    fn test_syntax_checker_valid_ask() {
        assert!(SparqlSyntaxChecker::check_keyword_sequence(
            "ASK { ?s ?p ?o }"
        ));
    }

    #[test]
    fn test_syntax_checker_select_without_where() {
        assert!(!SparqlSyntaxChecker::check_keyword_sequence(
            "SELECT ?s { ?s ?p ?o }"
        ));
    }

    #[test]
    fn test_syntax_checker_no_form_keyword() {
        assert!(!SparqlSyntaxChecker::check_keyword_sequence("{ ?s ?p ?o }"));
    }

    #[test]
    fn test_syntax_checker_valid_construct() {
        assert!(SparqlSyntaxChecker::check_keyword_sequence(
            "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"
        ));
    }

    #[test]
    fn test_syntax_checker_describe() {
        assert!(SparqlSyntaxChecker::check_keyword_sequence(
            "DESCRIBE <http://example.org/r>"
        ));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_query_is_invalid() {
        let v = QueryValidator::new(false);
        let r = v.validate("");
        assert!(!r.is_valid);
    }

    #[test]
    fn test_whitespace_only_query_is_invalid() {
        let v = QueryValidator::new(false);
        let r = v.validate("   \n\t  ");
        assert!(!r.is_valid);
    }

    #[test]
    fn test_multiline_valid_query() {
        let v = QueryValidator::new(false);
        let q = r#"
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?s ?label
WHERE {
    ?s rdf:type ex:Person .
    ?s ex:name ?label .
}
LIMIT 100
        "#;
        let r = v.validate(q);
        assert!(r.is_valid, "Issues: {:?}", r.issues);
    }
}
