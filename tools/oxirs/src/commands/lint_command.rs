//! # RDF/SPARQL Linting Command
//!
//! Provides rule-based linting of Turtle RDF documents and SPARQL queries.
//! Checks for common issues such as empty prefixes, undeclared prefixes,
//! duplicate triples, overly long literals, and deprecated predicates.
//!
//! ## Example
//!
//! ```rust
//! use oxirs::commands::lint_command::{LintCommand, LintConfig, LintResult};
//!
//! let content = "@prefix ex: <http://example.org/> . ex:s ex:p ex:o .";
//! let config = LintConfig::default();
//! let result = LintCommand::lint_ttl(content, "example.ttl", &config);
//! assert_eq!(result.file_path, "example.ttl");
//! ```

use std::collections::HashSet;

// ─────────────────────────────────────────────────────────────────────────────
// Rule and severity types
// ─────────────────────────────────────────────────────────────────────────────

/// A linting rule that can be applied to an RDF/SPARQL document
#[derive(Debug, Clone, PartialEq)]
pub enum LintRule {
    /// Turtle document uses the empty prefix (`:`) without declaring `@prefix :`
    EmptyPrefix,
    /// A prefix is used in the document but not declared with `@prefix`
    UndeclaredPrefix,
    /// The same triple `s p o` appears more than once
    DuplicateTriples,
    /// A string literal exceeds the given character limit
    LongLiteral(usize),
    /// One of the listed predicate URIs is used in the document
    DeprecatedPredicate(Vec<String>),
    /// A Turtle subject-list entry is missing a trailing semicolon
    MissingSemicolon,
}

/// Severity of a lint finding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LintSeverity {
    /// Must fix — this likely indicates an error
    Error,
    /// Should fix — advisory
    Warning,
    /// Informational
    Info,
}

/// A single lint finding
#[derive(Debug, Clone)]
pub struct LintIssue {
    /// The rule that triggered this finding
    pub rule: LintRule,
    /// 1-based line number where the issue was found (None if document-level)
    pub line: Option<usize>,
    /// Human-readable description
    pub message: String,
    /// Severity
    pub severity: LintSeverity,
}

impl LintIssue {
    fn new(
        rule: LintRule,
        line: Option<usize>,
        message: impl Into<String>,
        severity: LintSeverity,
    ) -> Self {
        Self {
            rule,
            line,
            message: message.into(),
            severity,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration controlling which lint rules are active
#[derive(Debug, Clone)]
pub struct LintConfig {
    /// Rules to apply (empty = apply all)
    pub rules: Vec<LintRule>,
    /// Maximum allowed literal length (for `LongLiteral` rule)
    pub max_literal_length: usize,
    /// Predicate URIs considered deprecated
    pub deprecated_predicates: Vec<String>,
}

impl Default for LintConfig {
    fn default() -> Self {
        Self {
            rules: vec![
                LintRule::EmptyPrefix,
                LintRule::UndeclaredPrefix,
                LintRule::DuplicateTriples,
                LintRule::LongLiteral(200),
                LintRule::MissingSemicolon,
            ],
            max_literal_length: 200,
            deprecated_predicates: vec![
                "http://www.w3.org/2002/07/owl#priorVersion".to_string(),
                "http://www.w3.org/2004/02/skos/core#altLabel".to_string(),
            ],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Result
// ─────────────────────────────────────────────────────────────────────────────

/// Lint result for a single file
#[derive(Debug, Clone)]
pub struct LintResult {
    /// All issues found in this file
    pub issues: Vec<LintIssue>,
    /// Path of the linted file
    pub file_path: String,
    /// Number of triples detected (heuristic)
    pub triple_count: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Linter implementation
// ─────────────────────────────────────────────────────────────────────────────

/// Stateless Turtle/SPARQL linter
pub struct LintCommand;

impl LintCommand {
    /// Lint the Turtle `content` of `file_path` according to `config`.
    pub fn lint_ttl(content: &str, file_path: &str, config: &LintConfig) -> LintResult {
        let mut issues: Vec<LintIssue> = Vec::new();

        // Determine which rules to apply
        let apply_all = config.rules.is_empty();
        let apply = |rule: &LintRule| -> bool {
            if apply_all {
                return true;
            }
            config
                .rules
                .iter()
                .any(|r| std::mem::discriminant(r) == std::mem::discriminant(rule))
        };

        if apply(&LintRule::EmptyPrefix) {
            issues.extend(Self::check_empty_prefix(content));
        }

        if apply(&LintRule::UndeclaredPrefix) {
            issues.extend(Self::check_undeclared_prefixes(content));
        }

        if apply(&LintRule::DuplicateTriples) {
            issues.extend(Self::check_duplicate_triples(content));
        }

        if apply(&LintRule::LongLiteral(0)) {
            issues.extend(Self::check_long_literals(
                content,
                config.max_literal_length,
            ));
        }

        if !config.deprecated_predicates.is_empty() && apply(&LintRule::DeprecatedPredicate(vec![]))
        {
            issues.extend(Self::check_deprecated_predicates(
                content,
                &config.deprecated_predicates,
            ));
        }

        let triple_count = Self::count_triples(content);

        LintResult {
            issues,
            file_path: file_path.to_string(),
            triple_count,
        }
    }

    // ── Individual rule checkers ─────────────────────────────────────────────

    /// Check for usage of the empty/default prefix (`:`) without a declaration.
    pub fn check_empty_prefix(content: &str) -> Vec<LintIssue> {
        let has_empty_prefix_decl = content
            .lines()
            .any(|l| l.trim_start().to_lowercase().starts_with("@prefix :"));

        if has_empty_prefix_decl {
            return Vec::new();
        }

        let mut issues: Vec<LintIssue> = Vec::new();

        for (line_no, line) in content.lines().enumerate() {
            // Skip prefix declaration lines
            let trimmed = line.trim();
            if trimmed.to_lowercase().starts_with("@prefix") {
                continue;
            }
            // Look for `:word` token that isn't inside a URI
            let mut found = false;
            for token in line.split_whitespace() {
                let clean = token.trim_matches(|c: char| !c.is_alphanumeric() && c != ':');
                if clean.starts_with(':') && !clean.starts_with("://") && clean.len() > 1 {
                    found = true;
                    break;
                }
            }
            if found {
                issues.push(LintIssue::new(
                    LintRule::EmptyPrefix,
                    Some(line_no + 1),
                    format!(
                        "Empty prefix ':' used on line {} but not declared",
                        line_no + 1
                    ),
                    LintSeverity::Warning,
                ));
            }
        }

        issues
    }

    /// Check for prefix usage without declaration (e.g. `foaf:name` without `@prefix foaf:`).
    pub fn check_undeclared_prefixes(content: &str) -> Vec<LintIssue> {
        let declared = Self::extract_declared_prefixes(content);

        let mut issues: Vec<LintIssue> = Vec::new();
        let mut reported: HashSet<String> = HashSet::new();

        for (line_no, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.to_lowercase().starts_with("@prefix") || trimmed.starts_with('#') {
                continue;
            }

            for token in line.split_whitespace() {
                // Look for prefix:localname patterns (not full IRIs)
                if let Some(colon) = token.find(':') {
                    let prefix = &token[..colon];
                    // Not a URI (no ://), not empty, all alphanumeric
                    if !token.contains("://")
                        && !prefix.is_empty()
                        && prefix.chars().all(|c| c.is_alphanumeric() || c == '_')
                        && !declared.contains(prefix)
                        && reported.insert(prefix.to_string())
                    {
                        issues.push(LintIssue::new(
                            LintRule::UndeclaredPrefix,
                            Some(line_no + 1),
                            format!("Prefix '{}:' is used but not declared", prefix),
                            LintSeverity::Error,
                        ));
                    }
                }
            }
        }

        issues
    }

    /// Check for long string literals exceeding `max_length` characters.
    pub fn check_long_literals(content: &str, max_length: usize) -> Vec<LintIssue> {
        let mut issues: Vec<LintIssue> = Vec::new();

        for (line_no, line) in content.lines().enumerate() {
            // Find quoted string literals in the line
            let chars: Vec<char> = line.chars().collect();
            let mut i = 0;
            while i < chars.len() {
                if chars[i] == '"' {
                    // Find matching closing quote (skip escaped)
                    let start = i + 1;
                    i += 1;
                    while i < chars.len() {
                        if chars[i] == '"' && (i == 0 || chars[i - 1] != '\\') {
                            break;
                        }
                        i += 1;
                    }
                    let literal_len = i.saturating_sub(start);
                    if literal_len > max_length {
                        issues.push(LintIssue::new(
                            LintRule::LongLiteral(literal_len),
                            Some(line_no + 1),
                            format!(
                                "Literal on line {} has {} characters (max {})",
                                line_no + 1,
                                literal_len,
                                max_length
                            ),
                            LintSeverity::Warning,
                        ));
                    }
                }
                i += 1;
            }
        }

        issues
    }

    /// Check for usage of deprecated predicate URIs.
    pub fn check_deprecated_predicates(content: &str, deprecated: &[String]) -> Vec<LintIssue> {
        let mut issues: Vec<LintIssue> = Vec::new();

        for (line_no, line) in content.lines().enumerate() {
            for dep in deprecated {
                if line.contains(dep.as_str()) {
                    issues.push(LintIssue::new(
                        LintRule::DeprecatedPredicate(vec![dep.clone()]),
                        Some(line_no + 1),
                        format!(
                            "Deprecated predicate '{}' used on line {}",
                            dep,
                            line_no + 1
                        ),
                        LintSeverity::Warning,
                    ));
                }
            }
        }

        issues
    }

    // ── Aggregate helpers ────────────────────────────────────────────────────

    /// Generate a summary report string for a set of lint results.
    pub fn summary(results: &[LintResult]) -> String {
        let errors = Self::error_count(results);
        let warnings = Self::warning_count(results);
        let files = results.len();
        let triples: usize = results.iter().map(|r| r.triple_count).sum();

        format!(
            "Linted {} file(s): {} triple(s) | {} error(s) | {} warning(s)",
            files, triples, errors, warnings
        )
    }

    /// Count the total number of Error-severity issues across all results.
    pub fn error_count(results: &[LintResult]) -> usize {
        results
            .iter()
            .flat_map(|r| r.issues.iter())
            .filter(|i| i.severity == LintSeverity::Error)
            .count()
    }

    /// Count the total number of Warning-severity issues across all results.
    pub fn warning_count(results: &[LintResult]) -> usize {
        results
            .iter()
            .flat_map(|r| r.issues.iter())
            .filter(|i| i.severity == LintSeverity::Warning)
            .count()
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Extract the set of declared prefix names from the document.
    fn extract_declared_prefixes(content: &str) -> HashSet<String> {
        let mut declared: HashSet<String> = HashSet::new();

        for line in content.lines() {
            let trimmed = line.trim();
            if !trimmed.to_lowercase().starts_with("@prefix") {
                continue;
            }
            // @prefix name: <uri> .
            let rest = trimmed["@prefix".len()..].trim();
            if let Some(colon) = rest.find(':') {
                let name = rest[..colon].trim().to_string();
                declared.insert(name);
            }
        }

        declared
    }

    /// Heuristic triple counter: count lines ending in ` .` or `;` that look like triples.
    fn count_triples(content: &str) -> usize {
        content
            .lines()
            .filter(|l| {
                let t = l.trim();
                !t.is_empty()
                    && !t.starts_with('#')
                    && !t.to_lowercase().starts_with("@prefix")
                    && !t.to_lowercase().starts_with("@base")
                    && (t.ends_with('.') || t.ends_with(';') || t.ends_with(','))
            })
            .count()
    }

    /// Check for duplicate triples (heuristic: identical non-whitespace lines)
    fn check_duplicate_triples(content: &str) -> Vec<LintIssue> {
        let mut seen: HashSet<String> = HashSet::new();
        let mut issues: Vec<LintIssue> = Vec::new();

        for (line_no, line) in content.lines().enumerate() {
            let trimmed = line.trim().to_string();
            if trimmed.is_empty()
                || trimmed.starts_with('#')
                || trimmed.to_lowercase().starts_with("@prefix")
            {
                continue;
            }
            // Only check lines that look like triple statements
            if (trimmed.ends_with('.') || trimmed.ends_with(';')) && !seen.insert(trimmed.clone()) {
                issues.push(LintIssue::new(
                    LintRule::DuplicateTriples,
                    Some(line_no + 1),
                    format!("Duplicate triple on line {}: {}", line_no + 1, trimmed),
                    LintSeverity::Warning,
                ));
            }
        }

        issues
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LintConfig::default ──────────────────────────────────────────────────

    #[test]
    fn test_default_config_has_rules() {
        let cfg = LintConfig::default();
        assert!(!cfg.rules.is_empty());
    }

    #[test]
    fn test_default_config_max_literal_length() {
        let cfg = LintConfig::default();
        assert!(cfg.max_literal_length > 0);
    }

    // ── check_empty_prefix ───────────────────────────────────────────────────

    #[test]
    fn test_empty_prefix_no_decl_triggers() {
        let content = "@prefix ex: <http://example.org/> .\nex:s :p ex:o .";
        let issues = LintCommand::check_empty_prefix(content);
        assert!(!issues.is_empty(), "expected empty prefix warning");
    }

    #[test]
    fn test_empty_prefix_declared_no_issue() {
        let content = "@prefix : <http://example.org/> .\n:s :p :o .";
        let issues = LintCommand::check_empty_prefix(content);
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn test_empty_prefix_issue_severity() {
        let content = "@prefix ex: <http://example.org/> .\nex:s :p ex:o .";
        let issues = LintCommand::check_empty_prefix(content);
        for issue in &issues {
            // Should be Warning or Error (not Info)
            assert!(issue.severity != LintSeverity::Info, "{issue:?}");
        }
    }

    // ── check_undeclared_prefixes ────────────────────────────────────────────

    #[test]
    fn test_undeclared_prefix_triggers() {
        let content = "foaf:name \"Alice\" .";
        let issues = LintCommand::check_undeclared_prefixes(content);
        assert!(!issues.is_empty(), "expected undeclared prefix issue");
        assert!(
            issues.iter().any(|i| i.message.contains("foaf")),
            "{issues:?}"
        );
    }

    #[test]
    fn test_undeclared_prefix_declared_ok() {
        let content = "@prefix foaf: <http://xmlns.com/foaf/0.1/> .\nfoaf:name \"Alice\" .";
        let issues = LintCommand::check_undeclared_prefixes(content);
        let foaf_issues: Vec<_> = issues
            .iter()
            .filter(|i| i.message.contains("foaf"))
            .collect();
        assert!(foaf_issues.is_empty(), "unexpected: {foaf_issues:?}");
    }

    #[test]
    fn test_undeclared_prefix_uri_not_flagged() {
        let content = "<http://example.org/s> <http://example.org/p> <http://example.org/o> .";
        let issues = LintCommand::check_undeclared_prefixes(content);
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn test_undeclared_prefix_error_severity() {
        let content = "ex:subject ex:pred ex:object .";
        let issues = LintCommand::check_undeclared_prefixes(content);
        for issue in &issues {
            assert_eq!(issue.severity, LintSeverity::Error, "{issue:?}");
        }
    }

    // ── check_long_literals ──────────────────────────────────────────────────

    #[test]
    fn test_long_literal_triggers() {
        let long_val = "x".repeat(300);
        let content = format!("ex:s ex:p \"{long_val}\" .");
        let issues = LintCommand::check_long_literals(&content, 200);
        assert!(!issues.is_empty(), "expected long literal issue");
    }

    #[test]
    fn test_short_literal_ok() {
        let content = r#"ex:s ex:p "short" ."#;
        let issues = LintCommand::check_long_literals(content, 200);
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn test_long_literal_exactly_at_limit_ok() {
        let val = "x".repeat(200);
        let content = format!("ex:s ex:p \"{val}\" .");
        let issues = LintCommand::check_long_literals(&content, 200);
        assert!(issues.is_empty(), "exactly at limit should be ok");
    }

    #[test]
    fn test_long_literal_one_over_limit() {
        let val = "x".repeat(201);
        let content = format!("ex:s ex:p \"{val}\" .");
        let issues = LintCommand::check_long_literals(&content, 200);
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_long_literal_warning_severity() {
        let val = "x".repeat(300);
        let content = format!("ex:s ex:p \"{val}\" .");
        let issues = LintCommand::check_long_literals(&content, 200);
        for issue in &issues {
            assert_eq!(issue.severity, LintSeverity::Warning);
        }
    }

    // ── check_deprecated_predicates ──────────────────────────────────────────

    #[test]
    fn test_deprecated_predicate_triggers() {
        let deprecated = vec!["http://old.example.org/pred".to_string()];
        let content = "<http://s> <http://old.example.org/pred> <http://o> .";
        let issues = LintCommand::check_deprecated_predicates(content, &deprecated);
        assert!(!issues.is_empty(), "expected deprecated predicate issue");
    }

    #[test]
    fn test_deprecated_predicate_not_present() {
        let deprecated = vec!["http://old.example.org/pred".to_string()];
        let content = "<http://s> <http://new.example.org/pred> <http://o> .";
        let issues = LintCommand::check_deprecated_predicates(content, &deprecated);
        assert!(issues.is_empty(), "{issues:?}");
    }

    #[test]
    fn test_deprecated_predicate_empty_list() {
        let content = "<http://s> <http://p> <http://o> .";
        let issues = LintCommand::check_deprecated_predicates(content, &[]);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_deprecated_predicate_warning_severity() {
        let deprecated = vec!["http://old.org/pred".to_string()];
        let content = "<http://s> <http://old.org/pred> <http://o> .";
        let issues = LintCommand::check_deprecated_predicates(content, &deprecated);
        for issue in &issues {
            assert_eq!(issue.severity, LintSeverity::Warning);
        }
    }

    // ── lint_ttl (end-to-end) ────────────────────────────────────────────────

    #[test]
    fn test_lint_ttl_clean_document() {
        let content = "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .";
        let config = LintConfig::default();
        let result = LintCommand::lint_ttl(content, "test.ttl", &config);
        assert_eq!(result.file_path, "test.ttl");
    }

    #[test]
    fn test_lint_ttl_file_path_preserved() {
        let content = "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .";
        let config = LintConfig::default();
        let result = LintCommand::lint_ttl(content, "/data/my.ttl", &config);
        assert_eq!(result.file_path, "/data/my.ttl");
    }

    #[test]
    fn test_lint_ttl_triple_count() {
        let content = "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .\nex:s2 ex:p2 ex:o2 .";
        let config = LintConfig::default();
        let result = LintCommand::lint_ttl(content, "test.ttl", &config);
        assert!(result.triple_count >= 1, "expected at least 1 triple");
    }

    #[test]
    fn test_lint_ttl_undeclared_prefix_issue() {
        let content = "foaf:Person a foaf:Class .";
        let config = LintConfig::default();
        let result = LintCommand::lint_ttl(content, "test.ttl", &config);
        assert!(
            !result.issues.is_empty(),
            "expected issues for undeclared prefix"
        );
    }

    #[test]
    fn test_lint_ttl_duplicate_triple_detected() {
        let content = "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .\nex:s ex:p ex:o .";
        let config = LintConfig::default();
        let result = LintCommand::lint_ttl(content, "test.ttl", &config);
        let dup_issues: Vec<_> = result
            .issues
            .iter()
            .filter(|i| matches!(i.rule, LintRule::DuplicateTriples))
            .collect();
        assert!(!dup_issues.is_empty(), "expected duplicate triple issue");
    }

    // ── summary ──────────────────────────────────────────────────────────────

    #[test]
    fn test_summary_empty_results() {
        let summary = LintCommand::summary(&[]);
        assert!(summary.contains('0'));
    }

    #[test]
    fn test_summary_contains_file_count() {
        let content = "@prefix ex: <http://example.org/> .\nex:s ex:p ex:o .";
        let config = LintConfig::default();
        let result = LintCommand::lint_ttl(content, "f.ttl", &config);
        let summary = LintCommand::summary(&[result]);
        assert!(summary.contains("1 file"), "{summary}");
    }

    #[test]
    fn test_summary_contains_error_count() {
        let summary_str = LintCommand::summary(&[]);
        assert!(summary_str.contains("error"), "{summary_str}");
    }

    // ── error_count / warning_count ──────────────────────────────────────────

    #[test]
    fn test_error_count_no_errors() {
        let results = vec![LintResult {
            issues: vec![LintIssue::new(
                LintRule::UndeclaredPrefix,
                None,
                "warning",
                LintSeverity::Warning,
            )],
            file_path: "f.ttl".to_string(),
            triple_count: 0,
        }];
        assert_eq!(LintCommand::error_count(&results), 0);
    }

    #[test]
    fn test_error_count_with_errors() {
        let results = vec![LintResult {
            issues: vec![LintIssue::new(
                LintRule::UndeclaredPrefix,
                None,
                "err",
                LintSeverity::Error,
            )],
            file_path: "f.ttl".to_string(),
            triple_count: 0,
        }];
        assert_eq!(LintCommand::error_count(&results), 1);
    }

    #[test]
    fn test_warning_count_no_warnings() {
        let results = vec![LintResult {
            issues: vec![LintIssue::new(
                LintRule::UndeclaredPrefix,
                None,
                "err",
                LintSeverity::Error,
            )],
            file_path: "f.ttl".to_string(),
            triple_count: 0,
        }];
        assert_eq!(LintCommand::warning_count(&results), 0);
    }

    #[test]
    fn test_warning_count_with_warnings() {
        let results = vec![LintResult {
            issues: vec![
                LintIssue::new(
                    LintRule::LongLiteral(300),
                    None,
                    "w1",
                    LintSeverity::Warning,
                ),
                LintIssue::new(
                    LintRule::LongLiteral(400),
                    None,
                    "w2",
                    LintSeverity::Warning,
                ),
            ],
            file_path: "f.ttl".to_string(),
            triple_count: 2,
        }];
        assert_eq!(LintCommand::warning_count(&results), 2);
    }

    #[test]
    fn test_error_and_warning_counts_multiple_files() {
        let results = vec![
            LintResult {
                issues: vec![
                    LintIssue::new(LintRule::UndeclaredPrefix, None, "e", LintSeverity::Error),
                    LintIssue::new(LintRule::LongLiteral(300), None, "w", LintSeverity::Warning),
                ],
                file_path: "a.ttl".to_string(),
                triple_count: 1,
            },
            LintResult {
                issues: vec![LintIssue::new(
                    LintRule::UndeclaredPrefix,
                    None,
                    "e2",
                    LintSeverity::Error,
                )],
                file_path: "b.ttl".to_string(),
                triple_count: 2,
            },
        ];
        assert_eq!(LintCommand::error_count(&results), 2);
        assert_eq!(LintCommand::warning_count(&results), 1);
    }

    // ── LintRule variants ────────────────────────────────────────────────────

    #[test]
    fn test_lint_rule_long_literal_variant() {
        let rule = LintRule::LongLiteral(500);
        if let LintRule::LongLiteral(n) = rule {
            assert_eq!(n, 500);
        } else {
            panic!("expected LongLiteral");
        }
    }

    #[test]
    fn test_lint_rule_deprecated_predicate_variant() {
        let rule = LintRule::DeprecatedPredicate(vec!["http://old.org/p".to_string()]);
        if let LintRule::DeprecatedPredicate(preds) = rule {
            assert_eq!(preds.len(), 1);
        } else {
            panic!("expected DeprecatedPredicate");
        }
    }

    // ── LintIssue creation ───────────────────────────────────────────────────

    #[test]
    fn test_lint_issue_new() {
        let issue = LintIssue::new(
            LintRule::EmptyPrefix,
            Some(5),
            "test message",
            LintSeverity::Warning,
        );
        assert_eq!(issue.line, Some(5));
        assert_eq!(issue.message, "test message");
        assert_eq!(issue.severity, LintSeverity::Warning);
    }

    #[test]
    fn test_lint_issue_no_line() {
        let issue = LintIssue::new(LintRule::DuplicateTriples, None, "dup", LintSeverity::Info);
        assert_eq!(issue.line, None);
    }

    // ── LintConfig rules ─────────────────────────────────────────────────────

    #[test]
    fn test_lint_config_custom_rules() {
        let config = LintConfig {
            rules: vec![LintRule::EmptyPrefix],
            max_literal_length: 100,
            deprecated_predicates: vec![],
        };
        assert_eq!(config.rules.len(), 1);
    }

    #[test]
    fn test_lint_ttl_respects_config_rules() {
        // Only enable LongLiteral — undeclared prefixes should not appear in issues
        let val = "x".repeat(300);
        let content = format!("foaf:s foaf:p \"{val}\" .");
        let config = LintConfig {
            rules: vec![LintRule::LongLiteral(200)],
            max_literal_length: 200,
            deprecated_predicates: vec![],
        };
        let result = LintCommand::lint_ttl(&content, "t.ttl", &config);
        // Only long-literal issues should appear
        let has_undeclared = result
            .issues
            .iter()
            .any(|i| matches!(i.rule, LintRule::UndeclaredPrefix));
        assert!(
            !has_undeclared,
            "unexpected undeclared prefix issue: {:?}",
            result.issues
        );
    }

    // ── Additional coverage ──────────────────────────────────────────────────

    #[test]
    fn test_lint_result_fields() {
        let result = LintResult {
            issues: vec![],
            file_path: "my_file.ttl".to_string(),
            triple_count: 42,
        };
        assert_eq!(result.file_path, "my_file.ttl");
        assert_eq!(result.triple_count, 42);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_summary_multiple_files() {
        let r1 = LintResult {
            issues: vec![],
            file_path: "a.ttl".to_string(),
            triple_count: 10,
        };
        let r2 = LintResult {
            issues: vec![],
            file_path: "b.ttl".to_string(),
            triple_count: 5,
        };
        let s = LintCommand::summary(&[r1, r2]);
        assert!(s.contains("2 file"), "{s}");
    }

    #[test]
    fn test_check_undeclared_prefix_line_number() {
        let content = "@prefix ex: <http://example.org/> .\nfoaf:name \"Alice\" .";
        let issues = LintCommand::check_undeclared_prefixes(content);
        assert!(!issues.is_empty());
        // Line 2 should be reported
        assert!(
            issues.iter().any(|i| i.line == Some(2)),
            "expected line 2: {issues:?}"
        );
    }

    #[test]
    fn test_lint_ttl_empty_content() {
        let config = LintConfig::default();
        let result = LintCommand::lint_ttl("", "empty.ttl", &config);
        assert_eq!(result.triple_count, 0);
        assert!(result.issues.is_empty());
    }

    #[test]
    fn test_check_long_literals_multiple_on_line() {
        let content = r#"ex:s ex:p "short" . ex:s2 ex:p2 "short2" ."#;
        let issues = LintCommand::check_long_literals(content, 200);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_lint_config_deprecated_predicates() {
        let cfg = LintConfig {
            rules: vec![LintRule::DeprecatedPredicate(vec![
                "http://old.org/p".to_string()
            ])],
            max_literal_length: 200,
            deprecated_predicates: vec!["http://old.org/p".to_string()],
        };
        let content = "<http://s> <http://old.org/p> <http://o> .";
        let result = LintCommand::lint_ttl(content, "f.ttl", &cfg);
        assert!(!result.issues.is_empty());
    }

    #[test]
    fn test_warning_count_mixed_severities() {
        let results = vec![LintResult {
            issues: vec![
                LintIssue::new(LintRule::EmptyPrefix, None, "e", LintSeverity::Error),
                LintIssue::new(LintRule::LongLiteral(300), None, "w", LintSeverity::Warning),
                LintIssue::new(LintRule::DuplicateTriples, None, "i", LintSeverity::Info),
            ],
            file_path: "f.ttl".to_string(),
            triple_count: 3,
        }];
        assert_eq!(LintCommand::error_count(&results), 1);
        assert_eq!(LintCommand::warning_count(&results), 1);
    }

    #[test]
    fn test_lint_severity_variants() {
        assert_eq!(LintSeverity::Error, LintSeverity::Error);
        assert_eq!(LintSeverity::Warning, LintSeverity::Warning);
        assert_eq!(LintSeverity::Info, LintSeverity::Info);
        assert_ne!(LintSeverity::Error, LintSeverity::Warning);
    }
}
