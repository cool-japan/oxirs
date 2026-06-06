//! SPARQL validation component with comprehensive syntax and semantic checks.

use anyhow::Result;
use regex::Regex;
use std::collections::{HashMap, HashSet};

use super::types::{
    SemanticWarning, SemanticWarningType, SyntaxError, SyntaxErrorType, ValidationResult,
    WarningSeverity,
};

/// SPARQL validation component with comprehensive checks
pub struct SPARQLValidator {
    syntax_patterns: HashMap<String, Regex>,
    common_prefixes: HashMap<String, String>,
}

impl Default for SPARQLValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl SPARQLValidator {
    pub fn new() -> Self {
        let mut syntax_patterns = HashMap::new();

        syntax_patterns.insert(
            "select_pattern".to_string(),
            Regex::new(r"(?i)^\s*SELECT\s+(?:DISTINCT\s+)?(?:\*|\?\w+(?:\s+\?\w+)*)\s+WHERE\s*\{")
                .expect("hardcoded regex should be valid"),
        );
        syntax_patterns.insert(
            "construct_pattern".to_string(),
            Regex::new(r"(?i)^\s*CONSTRUCT\s*\{").expect("hardcoded regex should be valid"),
        );
        syntax_patterns.insert(
            "ask_pattern".to_string(),
            Regex::new(r"(?i)^\s*ASK\s*\{").expect("hardcoded regex should be valid"),
        );
        syntax_patterns.insert(
            "describe_pattern".to_string(),
            Regex::new(r"(?i)^\s*DESCRIBE\s+").expect("hardcoded regex should be valid"),
        );
        syntax_patterns.insert(
            "variable_pattern".to_string(),
            Regex::new(r"\?[a-zA-Z][a-zA-Z0-9_]*").expect("hardcoded regex should be valid"),
        );
        syntax_patterns.insert(
            "iri_pattern".to_string(),
            Regex::new(r"<[^<>\s]+>").expect("hardcoded regex should be valid"),
        );

        let mut common_prefixes = HashMap::new();
        common_prefixes.insert(
            "rdf".to_string(),
            "http://www.w3.org/1999/02/22-rdf-syntax-ns#".to_string(),
        );
        common_prefixes.insert(
            "rdfs".to_string(),
            "http://www.w3.org/2000/01/rdf-schema#".to_string(),
        );
        common_prefixes.insert(
            "owl".to_string(),
            "http://www.w3.org/2002/07/owl#".to_string(),
        );
        common_prefixes.insert(
            "xsd".to_string(),
            "http://www.w3.org/2001/XMLSchema#".to_string(),
        );
        common_prefixes.insert("foaf".to_string(), "http://xmlns.com/foaf/0.1/".to_string());
        common_prefixes.insert(
            "skos".to_string(),
            "http://www.w3.org/2004/02/skos/core#".to_string(),
        );

        Self {
            syntax_patterns,
            common_prefixes,
        }
    }

    pub fn validate(&self, query: &str) -> Result<ValidationResult> {
        let mut syntax_errors = Vec::new();
        let mut semantic_warnings = Vec::new();
        let schema_issues = Vec::new();
        let mut suggestions = Vec::new();

        if !self.validate_basic_syntax(query) {
            syntax_errors.push(SyntaxError {
                message: "Query does not match any valid SPARQL query pattern".to_string(),
                position: Some(0),
                error_type: SyntaxErrorType::InvalidSyntax,
                suggestion: Some(
                    "Ensure query starts with SELECT, CONSTRUCT, ASK, or DESCRIBE".to_string(),
                ),
            });
        }

        self.validate_query_structure(query, &mut syntax_errors, &mut semantic_warnings)?;
        self.check_common_issues(query, &mut semantic_warnings, &mut suggestions)?;
        self.validate_prefixes(query, &mut syntax_errors, &mut suggestions)?;
        self.check_performance_issues(query, &mut semantic_warnings)?;

        let is_valid = syntax_errors.is_empty();

        Ok(ValidationResult {
            is_valid,
            syntax_errors,
            semantic_warnings,
            schema_issues,
            suggestions,
        })
    }

    fn validate_basic_syntax(&self, query: &str) -> bool {
        let query_trimmed = query.trim();
        for pattern in self.syntax_patterns.values() {
            if pattern.is_match(query_trimmed) {
                return true;
            }
        }
        false
    }

    fn validate_query_structure(
        &self,
        query: &str,
        syntax_errors: &mut Vec<SyntaxError>,
        semantic_warnings: &mut Vec<SemanticWarning>,
    ) -> Result<()> {
        let open_braces = query.matches('{').count();
        let close_braces = query.matches('}').count();

        if open_braces != close_braces {
            syntax_errors.push(SyntaxError {
                message: format!("Unbalanced braces: {open_braces} open, {close_braces} close"),
                position: None,
                error_type: SyntaxErrorType::InvalidSyntax,
                suggestion: Some(
                    "Check that all opening braces have matching closing braces".to_string(),
                ),
            });
        }

        if let Some(var_pattern) = self.syntax_patterns.get("variable_pattern") {
            let variables: HashSet<&str> =
                var_pattern.find_iter(query).map(|m| m.as_str()).collect();

            if variables.is_empty()
                && query.to_uppercase().contains("SELECT")
                && !query.contains('*')
            {
                semantic_warnings.push(SemanticWarning {
                    message: "No variables found in SELECT query".to_string(),
                    warning_type: SemanticWarningType::UnboundVariable,
                    severity: WarningSeverity::Medium,
                });
            }
        }

        Ok(())
    }

    fn check_common_issues(
        &self,
        query: &str,
        semantic_warnings: &mut Vec<SemanticWarning>,
        suggestions: &mut Vec<String>,
    ) -> Result<()> {
        let query_upper = query.to_uppercase();

        if !query_upper.contains("FILTER") && query_upper.matches('.').count() > 3 {
            semantic_warnings.push(SemanticWarning {
                message: "Query may produce Cartesian product - consider adding FILTER clauses"
                    .to_string(),
                warning_type: SemanticWarningType::PossibleCartesianProduct,
                severity: WarningSeverity::Medium,
            });
            suggestions.push(
                "Add FILTER clauses to constrain results and improve performance".to_string(),
            );
        }

        if !query_upper.contains("LIMIT") && query_upper.contains("SELECT") {
            suggestions
                .push("Consider adding a LIMIT clause to prevent large result sets".to_string());
        }

        if query.len() > 200 && !query_upper.contains("ORDER BY") && query_upper.contains("SELECT")
        {
            suggestions.push("Consider adding ORDER BY for consistent result ordering".to_string());
        }

        Ok(())
    }

    fn validate_prefixes(
        &self,
        query: &str,
        syntax_errors: &mut Vec<SyntaxError>,
        suggestions: &mut Vec<String>,
    ) -> Result<()> {
        let prefix_usage_pattern = Regex::new(r"(\w+):")?;
        let used_prefixes: HashSet<&str> = prefix_usage_pattern
            .find_iter(query)
            .map(|m| m.as_str().trim_end_matches(':'))
            .collect();

        let prefix_declaration_pattern = Regex::new(r"(?i)PREFIX\s+(\w+):")
            .unwrap_or_else(|_| Regex::new(r"PREFIX").expect("fallback regex should be valid"));
        let declared_prefixes: HashSet<&str> = prefix_declaration_pattern
            .captures_iter(query)
            .filter_map(|cap| cap.get(1))
            .map(|m| m.as_str())
            .collect();

        for prefix in &used_prefixes {
            if !declared_prefixes.contains(prefix) && self.common_prefixes.contains_key(*prefix) {
                suggestions.push(format!(
                    "Add prefix declaration: PREFIX {}: <{}>",
                    prefix,
                    self.common_prefixes
                        .get(*prefix)
                        .expect("prefix exists in common_prefixes")
                ));
            } else if !declared_prefixes.contains(prefix) {
                syntax_errors.push(SyntaxError {
                    message: format!("Undeclared prefix: {prefix}"),
                    position: None,
                    error_type: SyntaxErrorType::UnknownPrefix,
                    suggestion: Some(format!("Declare prefix {prefix} or use full IRI")),
                });
            }
        }

        Ok(())
    }

    fn check_performance_issues(
        &self,
        query: &str,
        semantic_warnings: &mut Vec<SemanticWarning>,
    ) -> Result<()> {
        let query_upper = query.to_uppercase();

        if query_upper.contains("REGEX") {
            semantic_warnings.push(SemanticWarning {
                message: "REGEX operations can be expensive - consider alternatives if possible"
                    .to_string(),
                warning_type: SemanticWarningType::PerformanceIssue,
                severity: WarningSeverity::Low,
            });
        }

        if query_upper.contains("UNION") && query_upper.matches("UNION").count() > 2 {
            semantic_warnings.push(SemanticWarning {
                message: "Multiple UNION clauses may impact performance".to_string(),
                warning_type: SemanticWarningType::PerformanceIssue,
                severity: WarningSeverity::Medium,
            });
        }

        if query.len() > 1000 {
            semantic_warnings.push(SemanticWarning {
                message: "Very long query may be difficult to optimize".to_string(),
                warning_type: SemanticWarningType::ComplexQuery,
                severity: WarningSeverity::Low,
            });
        }

        Ok(())
    }
}
