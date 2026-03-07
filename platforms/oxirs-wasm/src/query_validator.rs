//! # SPARQL Query Validator
//!
//! Client-side SPARQL query validation with syntax and semantic checks.
//! Designed to run in the browser (WASM) or server-side with no heavyweight
//! parser dependency.
//!
//! ## Checks performed
//!
//! - Balanced braces `{}`
//! - Query type detection (SELECT / CONSTRUCT / ASK / DESCRIBE / UPDATE)
//! - Prefix declarations vs. prefix usage
//! - Variable usage (variables in SELECT not bound in WHERE)
//! - Valid IRI syntax
//! - Comment stripping
//!
//! ## Example
//!
//! ```rust
//! use oxirs_wasm::query_validator::{QueryValidator, ValidationLevel};
//!
//! let query = "SELECT ?name WHERE { ?person <http://schema.org/name> ?name }";
//! let report = QueryValidator::validate(query, ValidationLevel::Full);
//! assert!(report.valid);
//! ```

use std::collections::HashSet;

/// Granularity of validation to perform
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationLevel {
    /// Only check structural syntax (braces, basic keyword)
    Syntax,
    /// Syntax + prefix consistency
    Semantic,
    /// Full validation: syntax + semantics + variable binding check
    Full,
}

/// Severity of a query issue
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    /// Hard error — the query is likely invalid
    Error,
    /// Advisory — the query may have a problem
    Warning,
    /// Informational note
    Info,
}

/// A single issue found during validation
#[derive(Debug, Clone, PartialEq)]
pub struct QueryIssue {
    /// 1-based line number (0 if unknown)
    pub line: usize,
    /// 1-based column number (0 if unknown)
    pub column: usize,
    /// Human-readable description
    pub message: String,
    /// Severity
    pub severity: IssueSeverity,
}

impl QueryIssue {
    fn error(message: impl Into<String>) -> Self {
        Self {
            line: 0,
            column: 0,
            message: message.into(),
            severity: IssueSeverity::Error,
        }
    }

    fn warning(message: impl Into<String>) -> Self {
        Self {
            line: 0,
            column: 0,
            message: message.into(),
            severity: IssueSeverity::Warning,
        }
    }

    fn info(message: impl Into<String>) -> Self {
        Self {
            line: 0,
            column: 0,
            message: message.into(),
            severity: IssueSeverity::Info,
        }
    }

    fn with_location(mut self, line: usize, column: usize) -> Self {
        self.line = line;
        self.column = column;
        self
    }
}

/// The type of a SPARQL query
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    Select,
    Construct,
    Ask,
    Describe,
    Update,
}

impl QueryType {
    /// Display name for the query type
    pub fn as_str(self) -> &'static str {
        match self {
            QueryType::Select => "SELECT",
            QueryType::Construct => "CONSTRUCT",
            QueryType::Ask => "ASK",
            QueryType::Describe => "DESCRIBE",
            QueryType::Update => "UPDATE",
        }
    }
}

/// Result of validating a query
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// `true` when no Error-level issues are present
    pub valid: bool,
    /// All issues (errors, warnings, info)
    pub issues: Vec<QueryIssue>,
    /// Detected query type (None if not recognised)
    pub query_type: Option<QueryType>,
}

impl ValidationReport {
    fn from_issues(issues: Vec<QueryIssue>, query_type: Option<QueryType>) -> Self {
        let valid = !issues.iter().any(|i| i.severity == IssueSeverity::Error);
        Self {
            valid,
            issues,
            query_type,
        }
    }
}

/// Stateless SPARQL query validator
pub struct QueryValidator;

impl QueryValidator {
    /// Validate `query` at the requested `level`.
    pub fn validate(query: &str, level: ValidationLevel) -> ValidationReport {
        let stripped = Self::strip_comments(query);
        let mut issues: Vec<QueryIssue> = Vec::new();

        // ── Syntax checks ────────────────────────────────────────────────────
        if !Self::check_balanced_braces(&stripped) {
            issues.push(QueryIssue::error("Unbalanced braces '{}'"));
        }

        let query_type = Self::detect_query_type(&stripped);
        if query_type.is_none() {
            issues.push(
                QueryIssue::warning(
                    "Could not detect query type (SELECT/CONSTRUCT/ASK/DESCRIBE/UPDATE)",
                )
                .with_location(1, 1),
            );
        }

        if level == ValidationLevel::Syntax {
            return ValidationReport::from_issues(issues, query_type);
        }

        // ── Semantic checks ──────────────────────────────────────────────────
        let prefix_issues = Self::check_prefixes(&stripped);
        issues.extend(prefix_issues);

        if level == ValidationLevel::Semantic {
            return ValidationReport::from_issues(issues, query_type);
        }

        // ── Full checks ──────────────────────────────────────────────────────
        let var_issues = Self::check_variables(&stripped);
        issues.extend(var_issues);

        ValidationReport::from_issues(issues, query_type)
    }

    /// Detect the query type from the first keyword (case-insensitive).
    pub fn detect_query_type(query: &str) -> Option<QueryType> {
        let upper = query.trim().to_uppercase();
        let stripped = upper.trim_start_matches(|c: char| c.is_whitespace());

        // Walk through words
        for word in stripped.split_whitespace() {
            match word {
                "SELECT" => return Some(QueryType::Select),
                "CONSTRUCT" => return Some(QueryType::Construct),
                "ASK" => return Some(QueryType::Ask),
                "DESCRIBE" => return Some(QueryType::Describe),
                w if w.starts_with("INSERT")
                    || w.starts_with("DELETE")
                    || w == "LOAD"
                    || w == "CLEAR"
                    || w == "DROP"
                    || w == "CREATE"
                    || w == "COPY"
                    || w == "MOVE"
                    || w == "ADD" =>
                {
                    return Some(QueryType::Update)
                }
                _ => {}
            }
        }
        None
    }

    /// Check that all `{` braces are closed (no unclosed blocks).
    pub fn check_balanced_braces(query: &str) -> bool {
        let mut depth: i64 = 0;
        let mut in_string = false;
        let mut prev_char = '\0';

        for ch in query.chars() {
            if in_string {
                if ch == '"' && prev_char != '\\' {
                    in_string = false;
                }
            } else {
                match ch {
                    '"' => in_string = true,
                    '{' => depth += 1,
                    '}' => {
                        depth -= 1;
                        if depth < 0 {
                            return false;
                        }
                    }
                    _ => {}
                }
            }
            prev_char = ch;
        }

        depth == 0
    }

    /// Check that all used prefixes (e.g. `ex:`) are declared with `PREFIX ex: <...>`.
    pub fn check_prefixes(query: &str) -> Vec<QueryIssue> {
        let declared = Self::extract_prefixes(query);
        let declared_names: HashSet<String> =
            declared.iter().map(|(prefix, _)| prefix.clone()).collect();

        let mut issues: Vec<QueryIssue> = Vec::new();
        let mut used: HashSet<String> = HashSet::new();

        // Find prefixed names: word followed by colon that is not a full IRI
        for token in Self::tokenise_query(query) {
            // Skip PREFIX declaration lines
            if token.to_uppercase().starts_with("PREFIX") {
                continue;
            }
            // A prefixed name looks like `abc:something`
            if let Some(colon_pos) = token.find(':') {
                let prefix = &token[..colon_pos];
                // Exclude IRIs (contain //) and empty prefix used as `:`
                if !token.contains("//") && !prefix.is_empty() {
                    // Check it only contains word characters (basic name check)
                    if prefix.chars().all(|c| c.is_alphanumeric() || c == '_') {
                        used.insert(prefix.to_string());
                    }
                }
            }
        }

        for prefix in &used {
            if !declared_names.contains(prefix) {
                issues.push(QueryIssue::warning(format!(
                    "Prefix '{}:' is used but not declared",
                    prefix
                )));
            }
        }

        issues
    }

    /// Check that variables appearing in SELECT are also present in WHERE.
    pub fn check_variables(query: &str) -> Vec<QueryIssue> {
        let upper = query.to_uppercase();
        let is_select = upper.contains("SELECT");
        if !is_select {
            return vec![];
        }

        let vars = Self::extract_variables(query);
        if vars.is_empty() {
            return vec![];
        }

        // Extract SELECT projection: text between SELECT and WHERE
        let select_vars = Self::extract_select_vars(query);
        let where_vars = Self::extract_where_vars(query);

        let mut issues: Vec<QueryIssue> = Vec::new();

        for var in &select_vars {
            if var == "*" {
                continue; // SELECT * is always valid
            }
            if !where_vars.contains(var) {
                issues.push(QueryIssue::warning(format!(
                    "Variable '?{}' appears in SELECT but not in WHERE clause",
                    var
                )));
            }
        }

        issues
    }

    /// Extract all `PREFIX prefix: <uri>` declarations from the query.
    pub fn extract_prefixes(query: &str) -> Vec<(String, String)> {
        let mut prefixes: Vec<(String, String)> = Vec::new();
        let upper = query.to_uppercase();
        let mut search_from = 0usize;

        while let Some(pos) = upper[search_from..].find("PREFIX") {
            let abs_pos = search_from + pos;
            let rest = &query[abs_pos + 6..]; // skip "PREFIX"

            let rest_trimmed = rest.trim_start();
            // Find the prefix name (ends with ':')
            if let Some(colon_pos) = rest_trimmed.find(':') {
                let prefix_name = rest_trimmed[..colon_pos].trim().to_string();
                let after_colon = &rest_trimmed[colon_pos + 1..].trim_start();

                // Find the IRI in angle brackets
                if let Some(lt_pos) = after_colon.find('<') {
                    if let Some(gt_pos) = after_colon[lt_pos + 1..].find('>') {
                        let iri = after_colon[lt_pos + 1..lt_pos + 1 + gt_pos].to_string();
                        prefixes.push((prefix_name, iri));
                    }
                }
            }

            search_from = abs_pos + 6;
        }

        prefixes
    }

    /// Extract all variable names (without `?` or `$` sigils) from the query.
    pub fn extract_variables(query: &str) -> Vec<String> {
        let mut vars: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        for token in query.split(|c: char| c.is_whitespace() || ",(){}[]".contains(c)) {
            if token.starts_with('?') || token.starts_with('$') {
                let name = token[1..]
                    .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                    .to_string();
                if !name.is_empty() && seen.insert(name.clone()) {
                    vars.push(name);
                }
            }
        }

        vars
    }

    /// Check whether `iri` looks like a valid absolute IRI.
    pub fn is_valid_iri(iri: &str) -> bool {
        let trimmed = iri.trim();
        // An IRI must have a scheme followed by ':'
        if let Some(colon) = trimmed.find(':') {
            let scheme = &trimmed[..colon];
            !scheme.is_empty()
                && scheme
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
                && scheme.starts_with(|c: char| c.is_ascii_alphabetic())
        } else {
            false
        }
    }

    /// Remove `#` comments from the query, preserving line structure.
    pub fn strip_comments(query: &str) -> String {
        query
            .lines()
            .map(|line| {
                let mut in_string = false;
                let mut result = String::new();
                let mut prev = '\0';
                for ch in line.chars() {
                    if in_string {
                        result.push(ch);
                        if ch == '"' && prev != '\\' {
                            in_string = false;
                        }
                    } else if ch == '#' {
                        break; // rest of line is a comment
                    } else {
                        if ch == '"' {
                            in_string = true;
                        }
                        result.push(ch);
                    }
                    prev = ch;
                }
                result
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Simple whitespace + punctuation tokeniser
    fn tokenise_query(query: &str) -> Vec<String> {
        let mut tokens: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut in_string = false;

        for ch in query.chars() {
            if in_string {
                current.push(ch);
                if ch == '"' {
                    in_string = false;
                    tokens.push(current.clone());
                    current.clear();
                }
            } else if ch == '"' {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                current.push(ch);
                in_string = true;
            } else if ch.is_whitespace() || "{}[](),;".contains(ch) {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            } else {
                current.push(ch);
            }
        }

        if !current.is_empty() {
            tokens.push(current);
        }

        tokens
    }

    /// Extract variable names appearing in the SELECT clause (between SELECT and WHERE/FROM/GRAPH)
    fn extract_select_vars(query: &str) -> Vec<String> {
        let upper = query.to_uppercase();
        let select_pos = match upper.find("SELECT") {
            Some(p) => p,
            None => return vec![],
        };

        let after_select = &query[select_pos + 6..];
        let upper_after = after_select.to_uppercase();

        // Find the end of the SELECT projection
        let end_pos = ["WHERE", "FROM", "{"]
            .iter()
            .filter_map(|kw| upper_after.find(kw))
            .min()
            .unwrap_or(after_select.len());

        let projection = &after_select[..end_pos];

        // Check for wildcard
        if projection.trim().starts_with('*') {
            return vec!["*".to_string()];
        }

        let mut vars: Vec<String> = Vec::new();
        for token in projection.split_whitespace() {
            let clean = token.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
            if (token.starts_with('?') || token.starts_with('$')) && !clean.is_empty() {
                vars.push(clean.to_string());
            }
        }

        vars
    }

    /// Extract variable names appearing in WHERE clause (between `{` and final `}`)
    fn extract_where_vars(query: &str) -> HashSet<String> {
        let upper = query.to_uppercase();
        let where_pos = match upper.find("WHERE") {
            Some(p) => p,
            None => return HashSet::new(),
        };

        let after_where = &query[where_pos + 5..];

        // Find opening brace
        let open_pos = match after_where.find('{') {
            Some(p) => p + 1,
            None => return HashSet::new(),
        };

        let body = &after_where[open_pos..];
        let mut vars: HashSet<String> = HashSet::new();

        for token in body.split(|c: char| c.is_whitespace() || ",(){}[]<>".contains(c)) {
            if token.starts_with('?') || token.starts_with('$') {
                let name = token[1..]
                    .trim_matches(|c: char| !c.is_alphanumeric() && c != '_')
                    .to_string();
                if !name.is_empty() {
                    vars.insert(name);
                }
            }
        }

        vars
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── detect_query_type ────────────────────────────────────────────────────

    #[test]
    fn test_detect_select() {
        let q = "SELECT ?x WHERE { ?x ?p ?o }";
        assert_eq!(
            QueryValidator::detect_query_type(q),
            Some(QueryType::Select)
        );
    }

    #[test]
    fn test_detect_construct() {
        let q = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }";
        assert_eq!(
            QueryValidator::detect_query_type(q),
            Some(QueryType::Construct)
        );
    }

    #[test]
    fn test_detect_ask() {
        let q = "ASK { ?x <http://example.org/p> ?y }";
        assert_eq!(QueryValidator::detect_query_type(q), Some(QueryType::Ask));
    }

    #[test]
    fn test_detect_describe() {
        let q = "DESCRIBE <http://example.org/resource>";
        assert_eq!(
            QueryValidator::detect_query_type(q),
            Some(QueryType::Describe)
        );
    }

    #[test]
    fn test_detect_insert_as_update() {
        let q = "INSERT DATA { <http://a> <http://b> <http://c> }";
        assert_eq!(
            QueryValidator::detect_query_type(q),
            Some(QueryType::Update)
        );
    }

    #[test]
    fn test_detect_delete_as_update() {
        let q = "DELETE DATA { <http://a> <http://b> <http://c> }";
        assert_eq!(
            QueryValidator::detect_query_type(q),
            Some(QueryType::Update)
        );
    }

    #[test]
    fn test_detect_unknown_type() {
        let q = "something completely random";
        assert_eq!(QueryValidator::detect_query_type(q), None);
    }

    #[test]
    fn test_detect_case_insensitive() {
        let q = "select ?x where { ?x ?p ?o }";
        assert_eq!(
            QueryValidator::detect_query_type(q),
            Some(QueryType::Select)
        );
    }

    // ── check_balanced_braces ────────────────────────────────────────────────

    #[test]
    fn test_balanced_braces_simple() {
        assert!(QueryValidator::check_balanced_braces(
            "SELECT ?x WHERE { ?x ?p ?o }"
        ));
    }

    #[test]
    fn test_balanced_braces_nested() {
        assert!(QueryValidator::check_balanced_braces(
            "SELECT ?x WHERE { { ?x ?p ?o } UNION { ?x ?p2 ?o } }"
        ));
    }

    #[test]
    fn test_unbalanced_braces_extra_open() {
        assert!(!QueryValidator::check_balanced_braces(
            "SELECT ?x WHERE { ?x ?p ?o"
        ));
    }

    #[test]
    fn test_unbalanced_braces_extra_close() {
        assert!(!QueryValidator::check_balanced_braces(
            "SELECT ?x WHERE { ?x ?p ?o }}"
        ));
    }

    #[test]
    fn test_balanced_braces_empty() {
        assert!(QueryValidator::check_balanced_braces(""));
    }

    #[test]
    fn test_braces_in_string_ignored() {
        // A literal string containing braces should not affect depth count
        assert!(QueryValidator::check_balanced_braces(
            r#"SELECT ?x WHERE { ?x <p> "value {nested}" }"#
        ));
    }

    // ── check_prefixes ───────────────────────────────────────────────────────

    #[test]
    fn test_check_prefixes_declared_used() {
        let q = "PREFIX ex: <http://example.org/> SELECT ?x WHERE { ?x ex:name ?y }";
        let issues = QueryValidator::check_prefixes(q);
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .collect();
        assert!(errors.is_empty(), "{errors:?}");
    }

    #[test]
    fn test_check_prefixes_undeclared_prefix() {
        let q = "SELECT ?x WHERE { ?x ex:name ?y }";
        let issues = QueryValidator::check_prefixes(q);
        assert!(!issues.is_empty(), "expected undeclared prefix warning");
    }

    #[test]
    fn test_check_prefixes_no_prefixes_used() {
        let q = "SELECT ?x WHERE { ?x <http://example.org/p> ?y }";
        let issues = QueryValidator::check_prefixes(q);
        assert!(issues.is_empty(), "{issues:?}");
    }

    // ── check_variables ──────────────────────────────────────────────────────

    #[test]
    fn test_check_variables_bound() {
        let q = "SELECT ?name WHERE { ?person <http://schema.org/name> ?name }";
        let issues = QueryValidator::check_variables(q);
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn test_check_variables_unbound_in_where() {
        let q = "SELECT ?name ?missing WHERE { ?person <http://schema.org/name> ?name }";
        let issues = QueryValidator::check_variables(q);
        assert!(
            issues.iter().any(|i| i.message.contains("missing")),
            "expected warning for ?missing: {issues:?}"
        );
    }

    #[test]
    fn test_check_variables_wildcard() {
        let q = "SELECT * WHERE { ?x ?p ?o }";
        let issues = QueryValidator::check_variables(q);
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn test_check_variables_non_select_no_issues() {
        let q = "ASK { ?x <http://p> ?y }";
        let issues = QueryValidator::check_variables(q);
        assert!(issues.is_empty());
    }

    // ── extract_prefixes ─────────────────────────────────────────────────────

    #[test]
    fn test_extract_prefixes_single() {
        let q = "PREFIX ex: <http://example.org/> SELECT ?x WHERE { ?x ?p ?o }";
        let prefixes = QueryValidator::extract_prefixes(q);
        assert_eq!(prefixes.len(), 1);
        assert_eq!(prefixes[0].0, "ex");
        assert_eq!(prefixes[0].1, "http://example.org/");
    }

    #[test]
    fn test_extract_prefixes_multiple() {
        let q = "PREFIX ex: <http://example.org/>\nPREFIX foaf: <http://xmlns.com/foaf/0.1/>\nSELECT ?x WHERE { ?x foaf:name ?n }";
        let prefixes = QueryValidator::extract_prefixes(q);
        assert_eq!(prefixes.len(), 2, "{prefixes:?}");
    }

    #[test]
    fn test_extract_prefixes_none() {
        let q = "SELECT ?x WHERE { ?x ?p ?o }";
        let prefixes = QueryValidator::extract_prefixes(q);
        assert!(prefixes.is_empty());
    }

    // ── extract_variables ────────────────────────────────────────────────────

    #[test]
    fn test_extract_variables_basic() {
        let q = "SELECT ?name ?age WHERE { ?person ?p ?name }";
        let vars = QueryValidator::extract_variables(q);
        assert!(vars.contains(&"name".to_string()), "{vars:?}");
        assert!(vars.contains(&"age".to_string()), "{vars:?}");
    }

    #[test]
    fn test_extract_variables_deduped() {
        let q = "SELECT ?x WHERE { ?x ?p ?x }";
        let vars = QueryValidator::extract_variables(q);
        let count = vars.iter().filter(|v| v.as_str() == "x").count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_extract_variables_empty() {
        let q = "ASK { <http://s> <http://p> <http://o> }";
        let vars = QueryValidator::extract_variables(q);
        assert!(vars.is_empty());
    }

    // ── is_valid_iri ─────────────────────────────────────────────────────────

    #[test]
    fn test_is_valid_iri_http() {
        assert!(QueryValidator::is_valid_iri("http://example.org/"));
        assert!(QueryValidator::is_valid_iri("https://example.org/resource"));
    }

    #[test]
    fn test_is_valid_iri_urn() {
        assert!(QueryValidator::is_valid_iri("urn:isbn:0451450523"));
    }

    #[test]
    fn test_is_valid_iri_invalid() {
        assert!(!QueryValidator::is_valid_iri("not-a-iri"));
        assert!(!QueryValidator::is_valid_iri(""));
    }

    #[test]
    fn test_is_valid_iri_ftp() {
        assert!(QueryValidator::is_valid_iri("ftp://files.example.org/"));
    }

    // ── strip_comments ───────────────────────────────────────────────────────

    #[test]
    fn test_strip_comments_basic() {
        let q = "SELECT ?x WHERE { ?x ?p ?o } # this is a comment";
        let stripped = QueryValidator::strip_comments(q);
        assert!(!stripped.contains("this is a comment"), "{stripped}");
        assert!(stripped.contains("SELECT"), "{stripped}");
    }

    #[test]
    fn test_strip_comments_full_line() {
        let q = "# full comment line\nSELECT ?x WHERE { ?x ?p ?o }";
        let stripped = QueryValidator::strip_comments(q);
        assert!(!stripped.contains("full comment line"), "{stripped}");
    }

    #[test]
    fn test_strip_comments_preserves_hash_in_literal() {
        let q = r#"SELECT ?x WHERE { ?x <p> "val#ue" }"#;
        let stripped = QueryValidator::strip_comments(q);
        assert!(stripped.contains("#"), "{stripped}");
    }

    #[test]
    fn test_strip_comments_no_comments() {
        let q = "SELECT ?x WHERE { ?x ?p ?o }";
        let stripped = QueryValidator::strip_comments(q);
        assert_eq!(stripped.trim(), q.trim());
    }

    // ── validate (end-to-end) ────────────────────────────────────────────────

    #[test]
    fn test_validate_valid_select() {
        let q = "SELECT ?name WHERE { ?person <http://schema.org/name> ?name }";
        let report = QueryValidator::validate(q, ValidationLevel::Full);
        assert!(report.valid, "issues: {:?}", report.issues);
        assert_eq!(report.query_type, Some(QueryType::Select));
    }

    #[test]
    fn test_validate_unbalanced_braces_error() {
        let q = "SELECT ?x WHERE { ?x ?p ?o";
        let report = QueryValidator::validate(q, ValidationLevel::Syntax);
        assert!(!report.valid);
    }

    #[test]
    fn test_validate_level_syntax_only() {
        let q = "SELECT ?x WHERE { ?x ex:name ?y }"; // ex: not declared
        let report = QueryValidator::validate(q, ValidationLevel::Syntax);
        // At Syntax level, prefix check is skipped
        // The query may still be "valid" at syntax level if braces balance
        assert_eq!(report.query_type, Some(QueryType::Select));
    }

    #[test]
    fn test_validate_level_semantic_catches_prefix() {
        let q = "SELECT ?x WHERE { ?x ex:name ?y }";
        let report = QueryValidator::validate(q, ValidationLevel::Semantic);
        assert!(
            report.issues.iter().any(|i| i.message.contains("ex")),
            "expected prefix issue: {:?}",
            report.issues
        );
    }

    #[test]
    fn test_validate_ask_query() {
        let q = "ASK { <http://example.org/s> <http://example.org/p> <http://example.org/o> }";
        let report = QueryValidator::validate(q, ValidationLevel::Full);
        assert!(report.valid, "{:?}", report.issues);
        assert_eq!(report.query_type, Some(QueryType::Ask));
    }

    #[test]
    fn test_validate_construct_query() {
        let q = "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }";
        let report = QueryValidator::validate(q, ValidationLevel::Syntax);
        assert_eq!(report.query_type, Some(QueryType::Construct));
    }

    #[test]
    fn test_validate_unknown_type_warning() {
        let q = "{ ?x ?p ?o }";
        let report = QueryValidator::validate(q, ValidationLevel::Syntax);
        let has_warning = report
            .issues
            .iter()
            .any(|i| i.severity == IssueSeverity::Warning);
        assert!(has_warning, "{:?}", report.issues);
    }

    // ── QueryType display ────────────────────────────────────────────────────

    #[test]
    fn test_query_type_as_str() {
        assert_eq!(QueryType::Select.as_str(), "SELECT");
        assert_eq!(QueryType::Ask.as_str(), "ASK");
        assert_eq!(QueryType::Construct.as_str(), "CONSTRUCT");
        assert_eq!(QueryType::Describe.as_str(), "DESCRIBE");
        assert_eq!(QueryType::Update.as_str(), "UPDATE");
    }

    // ── ValidationReport ────────────────────────────────────────────────────

    #[test]
    fn test_validation_report_no_errors_valid() {
        let report = ValidationReport::from_issues(vec![], None);
        assert!(report.valid);
    }

    #[test]
    fn test_validation_report_error_invalid() {
        let report = ValidationReport::from_issues(vec![QueryIssue::error("oops")], None);
        assert!(!report.valid);
    }

    #[test]
    fn test_validation_report_warning_still_valid() {
        let report = ValidationReport::from_issues(vec![QueryIssue::warning("advisory")], None);
        assert!(report.valid);
    }

    #[test]
    fn test_query_issue_info_severity() {
        let issue = QueryIssue::info("informational note");
        assert_eq!(issue.severity, IssueSeverity::Info);
    }

    #[test]
    fn test_query_issue_with_location() {
        let issue = QueryIssue::error("bad token").with_location(3, 12);
        assert_eq!(issue.line, 3);
        assert_eq!(issue.column, 12);
    }
}
