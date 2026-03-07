//! # SPARQL Query Command
//!
//! CLI command for executing SPARQL queries with configurable output formats,
//! timeout, and query-type detection.  Operates on simulated data so the
//! command can be exercised without a running SPARQL endpoint.
//!
//! ## Supported query types
//!
//! | Type      | Output                                          |
//! |-----------|-------------------------------------------------|
//! | SELECT    | Tabular variable bindings (Table, JSON, CSV, TSV)|
//! | ASK       | Boolean `true` / `false`                        |
//! | CONSTRUCT | Triples in N-Triples format                     |
//! | DESCRIBE  | CBD-like N-Triples description                  |
//!
//! ## Example
//!
//! ```rust
//! use oxirs::commands::query_command::{QueryCommand, QueryArgs, ResultFormat};
//!
//! let cmd = QueryCommand::new();
//! let args = QueryArgs {
//!     query: "SELECT ?s WHERE { ?s ?p ?o }".to_string(),
//!     format: ResultFormat::Table,
//!     timeout_ms: Some(5000),
//! };
//! let result = cmd.execute(&args).expect("query failed");
//! assert!(!result.output.is_empty());
//! ```

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Domain types
// ─────────────────────────────────────────────────────────────────────────────

/// Output format for query results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultFormat {
    /// Human-readable ASCII table.
    Table,
    /// SPARQL Results JSON.
    Json,
    /// Comma-separated values.
    Csv,
    /// Tab-separated values.
    Tsv,
}

impl ResultFormat {
    /// Parse from a string (case-insensitive).
    pub fn from_str_ci(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "table" => Some(ResultFormat::Table),
            "json" => Some(ResultFormat::Json),
            "csv" => Some(ResultFormat::Csv),
            "tsv" => Some(ResultFormat::Tsv),
            _ => None,
        }
    }

    /// MIME content type for the format.
    pub fn content_type(self) -> &'static str {
        match self {
            ResultFormat::Table => "text/plain",
            ResultFormat::Json => "application/sparql-results+json",
            ResultFormat::Csv => "text/csv",
            ResultFormat::Tsv => "text/tab-separated-values",
        }
    }
}

/// SPARQL query type discriminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    Select,
    Ask,
    Construct,
    Describe,
}

/// Arguments for the query command.
#[derive(Debug, Clone)]
pub struct QueryArgs {
    /// The SPARQL query text.
    pub query: String,
    /// Desired output format.
    pub format: ResultFormat,
    /// Optional timeout in milliseconds.
    pub timeout_ms: Option<u64>,
}

/// Result of query execution.
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Detected query type.
    pub query_type: QueryType,
    /// Formatted output string.
    pub output: String,
    /// Number of result rows / triples / boolean value.
    pub result_count: usize,
    /// Variable names (for SELECT queries).
    pub variables: Vec<String>,
}

/// A single binding row (variable name → value).
pub type BindingRow = HashMap<String, String>;

/// Validation issue found during syntax checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    /// Human-readable description.
    pub message: String,
    /// Severity level.
    pub severity: IssueSeverity,
}

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    Error,
    Warning,
}

// ─────────────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────────────

/// Errors returned by the query command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryCommandError {
    /// The query string is empty.
    EmptyQuery,
    /// The query type could not be determined.
    UnknownQueryType,
    /// Syntax validation failed.
    ValidationFailed(Vec<ValidationIssue>),
    /// Query execution timed out.
    Timeout(u64),
    /// Generic execution error.
    ExecutionError(String),
}

impl std::fmt::Display for QueryCommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryCommandError::EmptyQuery => write!(f, "Query string is empty"),
            QueryCommandError::UnknownQueryType => write!(f, "Could not determine query type"),
            QueryCommandError::ValidationFailed(issues) => {
                write!(f, "Validation failed: {} issue(s)", issues.len())
            }
            QueryCommandError::Timeout(ms) => write!(f, "Query timed out after {ms}ms"),
            QueryCommandError::ExecutionError(msg) => write!(f, "Execution error: {msg}"),
        }
    }
}

impl std::error::Error for QueryCommandError {}

// ─────────────────────────────────────────────────────────────────────────────
// QueryCommand
// ─────────────────────────────────────────────────────────────────────────────

/// SPARQL query command implementation.
#[derive(Debug, Default)]
pub struct QueryCommand {
    /// Optional simulated data source name.
    pub dataset: Option<String>,
}

impl QueryCommand {
    /// Create a new query command.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with a named dataset.
    pub fn with_dataset(dataset: impl Into<String>) -> Self {
        Self {
            dataset: Some(dataset.into()),
        }
    }

    /// Execute a SPARQL query.
    pub fn execute(&self, args: &QueryArgs) -> Result<QueryResult, QueryCommandError> {
        if args.query.trim().is_empty() {
            return Err(QueryCommandError::EmptyQuery);
        }

        // Validate syntax
        let issues = self.validate(&args.query);
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .cloned()
            .collect();
        if !errors.is_empty() {
            return Err(QueryCommandError::ValidationFailed(errors));
        }

        let query_type =
            detect_query_type(&args.query).ok_or(QueryCommandError::UnknownQueryType)?;

        match query_type {
            QueryType::Select => self.execute_select(args),
            QueryType::Ask => self.execute_ask(args),
            QueryType::Construct => self.execute_construct(args),
            QueryType::Describe => self.execute_describe(args),
        }
    }

    /// Validate query syntax (basic checks).
    pub fn validate(&self, query: &str) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();
        let trimmed = query.trim();

        if trimmed.is_empty() {
            issues.push(ValidationIssue {
                message: "Query is empty".to_string(),
                severity: IssueSeverity::Error,
            });
            return issues;
        }

        // Check for recognisable query form
        if detect_query_type(trimmed).is_none() {
            issues.push(ValidationIssue {
                message: "No recognisable SPARQL query form (SELECT, ASK, CONSTRUCT, DESCRIBE)"
                    .to_string(),
                severity: IssueSeverity::Error,
            });
        }

        // Check balanced braces
        let open_braces = trimmed.chars().filter(|c| *c == '{').count();
        let close_braces = trimmed.chars().filter(|c| *c == '}').count();
        if open_braces != close_braces {
            issues.push(ValidationIssue {
                message: format!(
                    "Unbalanced braces: {open_braces} opening vs {close_braces} closing"
                ),
                severity: IssueSeverity::Error,
            });
        }

        // Warning: no WHERE clause for SELECT
        let upper = trimmed.to_uppercase();
        if upper.contains("SELECT") && !upper.contains("WHERE") {
            issues.push(ValidationIssue {
                message: "SELECT query missing WHERE clause".to_string(),
                severity: IssueSeverity::Warning,
            });
        }

        // Warning: use of SELECT *
        if upper.contains("SELECT") && upper.contains("SELECT *") {
            issues.push(ValidationIssue {
                message: "SELECT * may return more data than needed".to_string(),
                severity: IssueSeverity::Warning,
            });
        }

        issues
    }

    // ── SELECT ───────────────────────────────────────────────────────────

    fn execute_select(&self, args: &QueryArgs) -> Result<QueryResult, QueryCommandError> {
        let variables = extract_select_variables(&args.query);
        let rows = generate_simulated_select_rows(&variables, 5);
        let output = format_select_results(&variables, &rows, args.format);

        Ok(QueryResult {
            query_type: QueryType::Select,
            output,
            result_count: rows.len(),
            variables,
        })
    }

    // ── ASK ──────────────────────────────────────────────────────────────

    fn execute_ask(&self, args: &QueryArgs) -> Result<QueryResult, QueryCommandError> {
        // Simulated ASK always returns true
        let output = match args.format {
            ResultFormat::Json => r#"{"head":{},"boolean":true}"#.to_string(),
            _ => "true".to_string(),
        };

        Ok(QueryResult {
            query_type: QueryType::Ask,
            output,
            result_count: 1,
            variables: vec![],
        })
    }

    // ── CONSTRUCT ────────────────────────────────────────────────────────

    fn execute_construct(&self, _args: &QueryArgs) -> Result<QueryResult, QueryCommandError> {
        let triples = [
            "<http://example.org/s1> <http://example.org/p1> <http://example.org/o1> .",
            "<http://example.org/s2> <http://example.org/p2> \"value\" .",
            "<http://example.org/s3> <http://example.org/p3> <http://example.org/o3> .",
        ];
        let output = triples.join("\n");
        let count = triples.len();

        Ok(QueryResult {
            query_type: QueryType::Construct,
            output,
            result_count: count,
            variables: vec![],
        })
    }

    // ── DESCRIBE ─────────────────────────────────────────────────────────

    fn execute_describe(&self, args: &QueryArgs) -> Result<QueryResult, QueryCommandError> {
        let resource = extract_describe_resource(&args.query)
            .unwrap_or_else(|| "<http://example.org/resource>".to_string());

        let triples = [format!("{resource} <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Thing> ."),
            format!("{resource} <http://www.w3.org/2000/01/rdf-schema#label> \"Resource\" ."),
            format!("{resource} <http://xmlns.com/foaf/0.1/knows> <http://example.org/other> .")];
        let output = triples.join("\n");
        let count = triples.len();

        Ok(QueryResult {
            query_type: QueryType::Describe,
            output,
            result_count: count,
            variables: vec![],
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Detect the query type from the query text.
pub fn detect_query_type(query: &str) -> Option<QueryType> {
    let upper = query.trim().to_uppercase();
    // Strip PREFIX / BASE declarations
    let body = strip_prologue(&upper);

    if body.starts_with("SELECT") {
        Some(QueryType::Select)
    } else if body.starts_with("ASK") {
        Some(QueryType::Ask)
    } else if body.starts_with("CONSTRUCT") {
        Some(QueryType::Construct)
    } else if body.starts_with("DESCRIBE") {
        Some(QueryType::Describe)
    } else {
        None
    }
}

/// Strip PREFIX and BASE declarations from the beginning of a query.
fn strip_prologue(query: &str) -> &str {
    let mut rest = query.trim();
    loop {
        if rest.starts_with("PREFIX") || rest.starts_with("BASE") {
            // Find end of this declaration (next newline or next PREFIX/BASE/SELECT/ASK/...)
            if let Some(pos) = rest.find('\n') {
                rest = rest[pos + 1..].trim();
            } else {
                // Single-line query with only a prefix — nothing left
                break;
            }
        } else {
            break;
        }
    }
    rest
}

/// Extract projected variable names from a SELECT query.
fn extract_select_variables(query: &str) -> Vec<String> {
    let upper = query.to_uppercase();
    let select_pos = match upper.find("SELECT") {
        Some(p) => p,
        None => return vec![],
    };
    let after_select = &query[select_pos + 6..];

    // Check for DISTINCT / REDUCED
    let after_select = after_select.trim();
    let after_select = if after_select.to_uppercase().starts_with("DISTINCT") {
        after_select[8..].trim()
    } else if after_select.to_uppercase().starts_with("REDUCED") {
        after_select[7..].trim()
    } else {
        after_select
    };

    // Find WHERE or opening brace
    let end = after_select
        .to_uppercase()
        .find("WHERE")
        .or_else(|| after_select.find('{'))
        .unwrap_or(after_select.len());

    let projection = &after_select[..end];

    if projection.trim() == "*" {
        return vec!["*".to_string()];
    }

    projection
        .split_whitespace()
        .filter(|token| token.starts_with('?') || token.starts_with('$'))
        .map(|v| v.to_string())
        .collect()
}

/// Extract the resource IRI from a DESCRIBE query.
fn extract_describe_resource(query: &str) -> Option<String> {
    let upper = query.to_uppercase();
    let desc_pos = upper.find("DESCRIBE")?;
    let after = query[desc_pos + 8..].trim();

    let token: String = after.chars().take_while(|c| !c.is_whitespace()).collect();

    if token.is_empty() {
        None
    } else {
        Some(token)
    }
}

/// Generate simulated SELECT rows for testing.
fn generate_simulated_select_rows(variables: &[String], count: usize) -> Vec<BindingRow> {
    (0..count)
        .map(|i| {
            let mut row = BindingRow::new();
            for var in variables {
                let name = var.trim_start_matches('?').trim_start_matches('$');
                row.insert(var.clone(), format!("<http://example.org/{name}{i}>"));
            }
            row
        })
        .collect()
}

/// Format SELECT results in the requested output format.
fn format_select_results(
    variables: &[String],
    rows: &[BindingRow],
    format: ResultFormat,
) -> String {
    match format {
        ResultFormat::Table => format_table(variables, rows),
        ResultFormat::Json => format_json(variables, rows),
        ResultFormat::Csv => format_dsv(variables, rows, ','),
        ResultFormat::Tsv => format_dsv(variables, rows, '\t'),
    }
}

/// Format as an ASCII table.
fn format_table(variables: &[String], rows: &[BindingRow]) -> String {
    if variables.is_empty() {
        return "No variables in projection.\n".to_string();
    }

    // Compute column widths
    let mut widths: Vec<usize> = variables.iter().map(|v| v.len()).collect();
    for row in rows {
        for (i, var) in variables.iter().enumerate() {
            let val_len = row.get(var).map_or(0, |v| v.len());
            if val_len > widths[i] {
                widths[i] = val_len;
            }
        }
    }

    let mut buf = String::new();

    // Separator line
    let sep: String = widths
        .iter()
        .map(|w| "-".repeat(w + 2))
        .collect::<Vec<_>>()
        .join("+");
    let sep = format!("+{sep}+\n");

    buf.push_str(&sep);

    // Header
    let header: String = variables
        .iter()
        .enumerate()
        .map(|(i, v)| format!(" {:<width$} ", v, width = widths[i]))
        .collect::<Vec<_>>()
        .join("|");
    buf.push_str(&format!("|{header}|\n"));
    buf.push_str(&sep);

    // Rows
    for row in rows {
        let line: String = variables
            .iter()
            .enumerate()
            .map(|(i, var)| {
                let val = row.get(var).map_or("", |v| v.as_str());
                format!(" {:<width$} ", val, width = widths[i])
            })
            .collect::<Vec<_>>()
            .join("|");
        buf.push_str(&format!("|{line}|\n"));
    }
    buf.push_str(&sep);

    // Result count
    buf.push_str(&format!("{} row(s)\n", rows.len()));

    buf
}

/// Format as SPARQL Results JSON.
fn format_json(variables: &[String], rows: &[BindingRow]) -> String {
    let mut buf = String::from("{\n  \"head\": { \"vars\": [");
    let var_names: Vec<String> = variables
        .iter()
        .map(|v| {
            let name = v.trim_start_matches('?').trim_start_matches('$');
            format!("\"{name}\"")
        })
        .collect();
    buf.push_str(&var_names.join(", "));
    buf.push_str("] },\n  \"results\": { \"bindings\": [\n");

    let row_strs: Vec<String> = rows
        .iter()
        .map(|row| {
            let entries: Vec<String> = variables
                .iter()
                .filter_map(|var| {
                    row.get(var).map(|val| {
                        let name = var.trim_start_matches('?').trim_start_matches('$');
                        let rdf_type = if val.starts_with('<') && val.ends_with('>') {
                            "uri"
                        } else {
                            "literal"
                        };
                        let clean_val = val
                            .trim_start_matches('<')
                            .trim_end_matches('>')
                            .trim_start_matches('"')
                            .trim_end_matches('"');
                        format!(
                            "      \"{name}\": {{ \"type\": \"{rdf_type}\", \"value\": \"{clean_val}\" }}"
                        )
                    })
                })
                .collect();
            format!("    {{\n{}\n    }}", entries.join(",\n"))
        })
        .collect();

    buf.push_str(&row_strs.join(",\n"));
    buf.push_str("\n  ] }\n}");
    buf
}

/// Format as CSV or TSV.
fn format_dsv(variables: &[String], rows: &[BindingRow], delimiter: char) -> String {
    let mut buf = String::new();

    // Header
    let header: Vec<&str> = variables
        .iter()
        .map(|v| v.trim_start_matches('?').trim_start_matches('$'))
        .collect();
    buf.push_str(&header.join(&delimiter.to_string()));
    buf.push('\n');

    // Rows
    for row in rows {
        let line: Vec<String> = variables
            .iter()
            .map(|var| row.get(var).cloned().unwrap_or_default())
            .collect();
        buf.push_str(&line.join(&delimiter.to_string()));
        buf.push('\n');
    }

    buf
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn select_query() -> String {
        "SELECT ?s ?p ?o WHERE { ?s ?p ?o }".to_string()
    }

    fn ask_query() -> String {
        "ASK { <http://example.org/a> ?p ?o }".to_string()
    }

    fn construct_query() -> String {
        "CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }".to_string()
    }

    fn describe_query() -> String {
        "DESCRIBE <http://example.org/alice>".to_string()
    }

    // ── Query type detection ─────────────────────────────────────────────

    #[test]
    fn test_detect_select() {
        assert_eq!(detect_query_type(&select_query()), Some(QueryType::Select));
    }

    #[test]
    fn test_detect_ask() {
        assert_eq!(detect_query_type(&ask_query()), Some(QueryType::Ask));
    }

    #[test]
    fn test_detect_construct() {
        assert_eq!(
            detect_query_type(&construct_query()),
            Some(QueryType::Construct)
        );
    }

    #[test]
    fn test_detect_describe() {
        assert_eq!(
            detect_query_type(&describe_query()),
            Some(QueryType::Describe)
        );
    }

    #[test]
    fn test_detect_with_prefix() {
        let q =
            "PREFIX foaf: <http://xmlns.com/foaf/0.1/>\nSELECT ?name WHERE { ?s foaf:name ?name }";
        assert_eq!(detect_query_type(q), Some(QueryType::Select));
    }

    #[test]
    fn test_detect_unknown() {
        assert_eq!(detect_query_type("SOMETHING ELSE"), None);
    }

    #[test]
    fn test_detect_case_insensitive() {
        assert_eq!(
            detect_query_type("select ?x where { ?x ?p ?o }"),
            Some(QueryType::Select)
        );
    }

    // ── Variable extraction ──────────────────────────────────────────────

    #[test]
    fn test_extract_variables() {
        let vars = extract_select_variables("SELECT ?s ?p ?o WHERE { ?s ?p ?o }");
        assert_eq!(vars, vec!["?s", "?p", "?o"]);
    }

    #[test]
    fn test_extract_variables_star() {
        let vars = extract_select_variables("SELECT * WHERE { ?s ?p ?o }");
        assert_eq!(vars, vec!["*"]);
    }

    #[test]
    fn test_extract_variables_distinct() {
        let vars = extract_select_variables("SELECT DISTINCT ?name WHERE { ?s ?p ?name }");
        assert_eq!(vars, vec!["?name"]);
    }

    #[test]
    fn test_extract_variables_dollar_prefix() {
        let vars = extract_select_variables("SELECT $x $y WHERE { $x $y ?z }");
        assert_eq!(vars, vec!["$x", "$y"]);
    }

    // ── DESCRIBE resource extraction ─────────────────────────────────────

    #[test]
    fn test_extract_describe_resource() {
        let resource = extract_describe_resource("DESCRIBE <http://example.org/alice>");
        assert_eq!(resource, Some("<http://example.org/alice>".to_string()));
    }

    #[test]
    fn test_extract_describe_resource_none() {
        let resource = extract_describe_resource("DESCRIBE ");
        assert_eq!(resource, None);
    }

    // ── Query execution ──────────────────────────────────────────────────

    #[test]
    fn test_execute_select() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: select_query(),
            format: ResultFormat::Table,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        assert_eq!(result.query_type, QueryType::Select);
        assert!(!result.output.is_empty());
        assert_eq!(result.result_count, 5);
        assert_eq!(result.variables.len(), 3);
    }

    #[test]
    fn test_execute_ask() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: ask_query(),
            format: ResultFormat::Table,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        assert_eq!(result.query_type, QueryType::Ask);
        assert_eq!(result.output, "true");
    }

    #[test]
    fn test_execute_ask_json() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: ask_query(),
            format: ResultFormat::Json,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        assert!(result.output.contains("\"boolean\":true"));
    }

    #[test]
    fn test_execute_construct() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: construct_query(),
            format: ResultFormat::Table,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        assert_eq!(result.query_type, QueryType::Construct);
        assert_eq!(result.result_count, 3);
        assert!(result.output.contains("<http://example.org/s1>"));
    }

    #[test]
    fn test_execute_describe() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: describe_query(),
            format: ResultFormat::Table,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        assert_eq!(result.query_type, QueryType::Describe);
        assert!(result.output.contains("<http://example.org/alice>"));
        assert_eq!(result.result_count, 3);
    }

    #[test]
    fn test_execute_empty_query() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: "".to_string(),
            format: ResultFormat::Table,
            timeout_ms: None,
        };
        let result = cmd.execute(&args);
        assert!(matches!(result, Err(QueryCommandError::EmptyQuery)));
    }

    #[test]
    fn test_execute_whitespace_only_query() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: "   \n\t  ".to_string(),
            format: ResultFormat::Table,
            timeout_ms: None,
        };
        let result = cmd.execute(&args);
        assert!(matches!(result, Err(QueryCommandError::EmptyQuery)));
    }

    #[test]
    fn test_execute_unknown_query_type() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: "INVALID QUERY FORM".to_string(),
            format: ResultFormat::Table,
            timeout_ms: None,
        };
        let result = cmd.execute(&args);
        assert!(matches!(
            result,
            Err(QueryCommandError::ValidationFailed(_))
        ));
    }

    // ── Format selection ─────────────────────────────────────────────────

    #[test]
    fn test_select_csv_format() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: select_query(),
            format: ResultFormat::Csv,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        assert!(result.output.contains(','));
        // Header should contain variable names without ?
        assert!(result.output.starts_with("s,p,o\n"));
    }

    #[test]
    fn test_select_tsv_format() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: select_query(),
            format: ResultFormat::Tsv,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        assert!(result.output.contains('\t'));
        assert!(result.output.starts_with("s\tp\to\n"));
    }

    #[test]
    fn test_select_json_format() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: select_query(),
            format: ResultFormat::Json,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        assert!(result.output.contains("\"head\""));
        assert!(result.output.contains("\"results\""));
        assert!(result.output.contains("\"bindings\""));
    }

    #[test]
    fn test_select_table_format() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: select_query(),
            format: ResultFormat::Table,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        assert!(result.output.contains('+'));
        assert!(result.output.contains('|'));
        assert!(result.output.contains("5 row(s)"));
    }

    // ── ResultFormat parsing ─────────────────────────────────────────────

    #[test]
    fn test_result_format_from_str() {
        assert_eq!(
            ResultFormat::from_str_ci("table"),
            Some(ResultFormat::Table)
        );
        assert_eq!(ResultFormat::from_str_ci("JSON"), Some(ResultFormat::Json));
        assert_eq!(ResultFormat::from_str_ci("csv"), Some(ResultFormat::Csv));
        assert_eq!(ResultFormat::from_str_ci("TSV"), Some(ResultFormat::Tsv));
        assert_eq!(ResultFormat::from_str_ci("xml"), None);
    }

    #[test]
    fn test_result_format_content_type() {
        assert_eq!(ResultFormat::Table.content_type(), "text/plain");
        assert_eq!(
            ResultFormat::Json.content_type(),
            "application/sparql-results+json"
        );
        assert_eq!(ResultFormat::Csv.content_type(), "text/csv");
        assert_eq!(
            ResultFormat::Tsv.content_type(),
            "text/tab-separated-values"
        );
    }

    // ── Validation ───────────────────────────────────────────────────────

    #[test]
    fn test_validate_good_query() {
        let cmd = QueryCommand::new();
        let issues = cmd.validate("SELECT ?s WHERE { ?s ?p ?o }");
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .collect();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_empty() {
        let cmd = QueryCommand::new();
        let issues = cmd.validate("");
        assert!(!issues.is_empty());
        assert_eq!(issues[0].severity, IssueSeverity::Error);
    }

    #[test]
    fn test_validate_no_query_form() {
        let cmd = QueryCommand::new();
        let issues = cmd.validate("SOMETHING { ?s ?p ?o }");
        let errors: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Error)
            .collect();
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_validate_unbalanced_braces() {
        let cmd = QueryCommand::new();
        let issues = cmd.validate("SELECT ?s WHERE { ?s ?p ?o");
        let brace_errors: Vec<_> = issues
            .iter()
            .filter(|i| i.message.contains("Unbalanced braces"))
            .collect();
        assert!(!brace_errors.is_empty());
    }

    #[test]
    fn test_validate_select_star_warning() {
        let cmd = QueryCommand::new();
        let issues = cmd.validate("SELECT * WHERE { ?s ?p ?o }");
        let warnings: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning)
            .collect();
        assert!(!warnings.is_empty());
    }

    #[test]
    fn test_validate_select_missing_where_warning() {
        let cmd = QueryCommand::new();
        let issues = cmd.validate("SELECT ?s { ?s ?p ?o }");
        let warnings: Vec<_> = issues
            .iter()
            .filter(|i| i.severity == IssueSeverity::Warning && i.message.contains("WHERE"))
            .collect();
        assert!(!warnings.is_empty());
    }

    // ── QueryCommand with dataset ────────────────────────────────────────

    #[test]
    fn test_with_dataset() {
        let cmd = QueryCommand::with_dataset("my-graph");
        assert_eq!(cmd.dataset, Some("my-graph".to_string()));
    }

    // ── Error display ────────────────────────────────────────────────────

    #[test]
    fn test_error_display() {
        let err = QueryCommandError::EmptyQuery;
        assert!(err.to_string().contains("empty"));

        let err = QueryCommandError::UnknownQueryType;
        assert!(err.to_string().contains("query type"));

        let err = QueryCommandError::Timeout(5000);
        assert!(err.to_string().contains("5000"));

        let err = QueryCommandError::ExecutionError("fail".to_string());
        assert!(err.to_string().contains("fail"));

        let err = QueryCommandError::ValidationFailed(vec![ValidationIssue {
            message: "test".to_string(),
            severity: IssueSeverity::Error,
        }]);
        assert!(err.to_string().contains("1 issue"));
    }

    // ── Simulated row generation ─────────────────────────────────────────

    #[test]
    fn test_simulated_rows_count() {
        let vars = vec!["?s".to_string(), "?p".to_string()];
        let rows = generate_simulated_select_rows(&vars, 10);
        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn test_simulated_rows_contain_all_vars() {
        let vars = vec!["?s".to_string(), "?p".to_string(), "?o".to_string()];
        let rows = generate_simulated_select_rows(&vars, 1);
        assert_eq!(rows[0].len(), 3);
        assert!(rows[0].contains_key("?s"));
        assert!(rows[0].contains_key("?p"));
        assert!(rows[0].contains_key("?o"));
    }

    // ── PREFIX handling ──────────────────────────────────────────────────

    #[test]
    fn test_select_with_multiple_prefixes() {
        let q = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nPREFIX foaf: <http://xmlns.com/foaf/0.1/>\nSELECT ?name WHERE { ?s foaf:name ?name }";
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: q.to_string(),
            format: ResultFormat::Table,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        assert_eq!(result.query_type, QueryType::Select);
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn test_construct_output_ntriples() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: construct_query(),
            format: ResultFormat::Table,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        // Each line should end with " ."
        for line in result.output.lines() {
            assert!(line.ends_with(" ."), "Line should end with ' .': {line}");
        }
    }

    #[test]
    fn test_describe_output_contains_rdf_type() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: describe_query(),
            format: ResultFormat::Table,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        assert!(result.output.contains("rdf-syntax-ns#type"));
    }

    #[test]
    fn test_json_output_structure() {
        let cmd = QueryCommand::new();
        let args = QueryArgs {
            query: "SELECT ?x WHERE { ?x ?p ?o }".to_string(),
            format: ResultFormat::Json,
            timeout_ms: None,
        };
        let result = cmd.execute(&args).expect("execute");
        // Must have head and results keys
        assert!(result.output.contains("\"head\""));
        assert!(result.output.contains("\"vars\""));
        assert!(result.output.contains("\"results\""));
        assert!(result.output.contains("\"bindings\""));
    }
}
