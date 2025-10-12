//! Interactive REPL mode for OxiRS (Alpha Implementation)
//!
//! Provides an interactive shell for SPARQL queries with real execution

use crate::cli::formatters::{
    create_formatter, Binding, QueryResults as FormatterQueryResults, RdfTerm,
};
use crate::cli::CliResult;
use oxirs_core::model::{Predicate, Subject, Term};
use oxirs_core::rdf_store::{OxirsQueryResults, QueryResults as CoreQueryResults, RdfStore};
use rustyline::error::ReadlineError;
use rustyline::history::DefaultHistory;
use rustyline::Editor;
use rustyline_derive::{Helper, Highlighter, Validator};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// SPARQL query helper for readline with keyword completion
#[derive(Helper, Highlighter, Validator)]
struct SparqlHelper;

impl rustyline::completion::Completer for SparqlHelper {
    type Candidate = String;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Self::Candidate>)> {
        let keywords = vec![
            "SELECT",
            "CONSTRUCT",
            "ASK",
            "DESCRIBE",
            "WHERE",
            "FILTER",
            "OPTIONAL",
            "UNION",
            "GRAPH",
            "SERVICE",
            "BIND",
            "VALUES",
            "PREFIX",
            "BASE",
            "DISTINCT",
            "REDUCED",
            "ORDER BY",
            "GROUP BY",
            "HAVING",
            "LIMIT",
            "OFFSET",
            "FROM",
            "FROM NAMED",
            "INSERT",
            "DELETE",
            "CLEAR",
            "DROP",
            "CREATE",
            "LOAD",
            "COPY",
            "MOVE",
            "ADD",
        ];

        let line_before = &line[..pos];
        let start = line_before
            .rfind(char::is_whitespace)
            .map(|i| i + 1)
            .unwrap_or(0);

        let prefix = &line_before[start..].to_uppercase();

        if prefix.is_empty() {
            return Ok((start, vec![]));
        }

        let matches: Vec<String> = keywords
            .iter()
            .filter(|k| k.starts_with(prefix))
            .map(|k| k.to_string())
            .collect();

        Ok((start, matches))
    }
}

impl rustyline::hint::Hinter for SparqlHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> Option<Self::Hint> {
        if pos < line.len() {
            return None;
        }

        let line_upper = line.to_uppercase();

        // Provide hints based on current context
        if line_upper.starts_with("SELECT") && !line_upper.contains("WHERE") {
            Some(" WHERE { ?s ?p ?o }".to_string())
        } else if line_upper.starts_with("PREFIX")
            && line.matches(':').count() == 1
            && !line.contains('<')
        {
            Some(" <http://example.org/>".to_string())
        } else if line_upper.ends_with("WHERE") {
            Some(" { ?s ?p ?o }".to_string())
        } else {
            None
        }
    }
}

/// Query session data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuerySession {
    /// Session name
    name: String,
    /// Dataset connected to
    dataset: String,
    /// Queries executed in this session
    queries: Vec<String>,
    /// Timestamp of session creation
    created_at: String,
    /// Last modified timestamp
    modified_at: String,
}

impl QuerySession {
    /// Create a new session
    fn new(name: String, dataset: String) -> Self {
        let now = chrono::Local::now().to_rfc3339();
        Self {
            name,
            dataset,
            queries: Vec::new(),
            created_at: now.clone(),
            modified_at: now,
        }
    }

    /// Add a query to the session
    fn add_query(&mut self, query: String) {
        self.queries.push(query);
        self.modified_at = chrono::Local::now().to_rfc3339();
    }

    /// Save session to file
    fn save_to_file(&self, path: &PathBuf) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize session: {}", e))?;
        fs::write(path, json).map_err(|e| format!("Failed to write session file: {}", e))?;
        Ok(())
    }

    /// Load session from file
    fn load_from_file(path: &PathBuf) -> Result<Self, String> {
        let json =
            fs::read_to_string(path).map_err(|e| format!("Failed to read session file: {}", e))?;
        let session: QuerySession = serde_json::from_str(&json)
            .map_err(|e| format!("Failed to parse session file: {}", e))?;
        Ok(session)
    }

    /// Clear all queries from the session
    fn clear(&mut self) {
        self.queries.clear();
        self.modified_at = chrono::Local::now().to_rfc3339();
    }
}

/// Format SPARQL query for better readability
fn format_sparql_query(query: &str) -> String {
    let mut formatted = String::new();
    let mut indent_level: usize = 0;
    let indent = "  ";

    for line in query.lines() {
        let trimmed = line.trim();

        // Decrease indent before closing braces
        if trimmed.starts_with('}') {
            indent_level = indent_level.saturating_sub(1);
        }

        // Add indentation
        if !trimmed.is_empty() {
            formatted.push_str(&indent.repeat(indent_level));
            formatted.push_str(trimmed);
            formatted.push('\n');
        }

        // Increase indent after opening braces
        if trimmed.ends_with('{') {
            indent_level += 1;
        }
    }

    formatted.trim_end().to_string()
}

/// Get query template by name
fn get_query_template(name: &str) -> Option<String> {
    match name.to_lowercase().as_str() {
        "select" => Some(
            "PREFIX ex: <http://example.org/>\n\
             SELECT ?subject ?predicate ?object\n\
             WHERE {\n\
               ?subject ?predicate ?object .\n\
             }\n\
             LIMIT 10"
                .to_string(),
        ),
        "construct" => Some(
            "PREFIX ex: <http://example.org/>\n\
             CONSTRUCT {\n\
               ?s ex:related ?o .\n\
             }\n\
             WHERE {\n\
               ?s ?p ?o .\n\
             }"
            .to_string(),
        ),
        "ask" => Some(
            "PREFIX ex: <http://example.org/>\n\
             ASK {\n\
               ?s ex:property ?o .\n\
             }"
            .to_string(),
        ),
        "describe" => Some(
            "PREFIX ex: <http://example.org/>\n\
             DESCRIBE ?resource\n\
             WHERE {\n\
               ?resource ex:property ?value .\n\
             }"
            .to_string(),
        ),
        "filter" => Some(
            "PREFIX ex: <http://example.org/>\n\
             SELECT ?item ?value\n\
             WHERE {\n\
               ?item ex:property ?value .\n\
               FILTER (?value > 100)\n\
             }"
            .to_string(),
        ),
        "optional" => Some(
            "PREFIX ex: <http://example.org/>\n\
             SELECT ?person ?name ?email\n\
             WHERE {\n\
               ?person ex:name ?name .\n\
               OPTIONAL { ?person ex:email ?email }\n\
             }"
            .to_string(),
        ),
        _ => None,
    }
}

/// Validate basic SPARQL syntax and return hints
fn validate_sparql_syntax(query: &str) -> Vec<String> {
    let mut hints = Vec::new();
    let query_upper = query.to_uppercase();

    // Check for common SPARQL keywords
    let has_select = query_upper.contains("SELECT");
    let has_construct = query_upper.contains("CONSTRUCT");
    let has_ask = query_upper.contains("ASK");
    let has_describe = query_upper.contains("DESCRIBE");
    let has_where = query_upper.contains("WHERE");

    if !has_select && !has_construct && !has_ask && !has_describe {
        hints.push("Query should start with SELECT, CONSTRUCT, ASK, or DESCRIBE".to_string());
    }

    if (has_select || has_construct || has_describe) && !has_where {
        hints.push("Query should include a WHERE clause".to_string());
    }

    // Check for unbalanced braces in WHERE clause
    let brace_count = query.matches('{').count() as i32 - query.matches('}').count() as i32;
    if brace_count != 0 {
        hints.push(format!(
            "Unbalanced braces (difference: {})",
            brace_count.abs()
        ));
    }

    // Check for common prefixes without PREFIX declarations
    let common_prefixes = [
        ("rdf:", "http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
        ("rdfs:", "http://www.w3.org/2000/01/rdf-schema#"),
        ("owl:", "http://www.w3.org/2002/07/owl#"),
        ("xsd:", "http://www.w3.org/2001/XMLSchema#"),
        ("foaf:", "http://xmlns.com/foaf/0.1/"),
        ("dc:", "http://purl.org/dc/elements/1.1/"),
        ("skos:", "http://www.w3.org/2004/02/skos/core#"),
    ];

    for (prefix, uri) in &common_prefixes {
        if query.contains(prefix)
            && !query_upper.contains(&format!(
                "PREFIX {}",
                prefix.to_uppercase().trim_end_matches(':')
            ))
        {
            hints.push(format!(
                "Consider adding: PREFIX {}: <{}>",
                prefix.trim_end_matches(':'),
                uri
            ));
        }
    }

    // Check for FILTER without parentheses
    if query_upper.contains("FILTER") && !query.contains("FILTER (") && !query.contains("FILTER(") {
        hints.push("FILTER should be followed by parentheses: FILTER (expression)".to_string());
    }

    // Check for missing dots between triples
    if has_where {
        let lines: Vec<&str> = query.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.ends_with('.')
                && !trimmed.ends_with('{')
                && !trimmed.ends_with('}')
                && !trimmed.ends_with(';')
                && !trimmed.starts_with("PREFIX")
                && !trimmed.starts_with("SELECT")
                && !trimmed.starts_with("WHERE")
                && !trimmed.starts_with("FILTER")
                && trimmed.contains("?")
                && i + 1 < lines.len()
            {
                hints.push(format!(
                    "Line {} might be missing a dot (.) at the end",
                    i + 1
                ));
                break; // Only show one hint about this
            }
        }
    }

    hints
}

/// Check if a SPARQL query is complete
/// A query is complete if:
/// - All braces are balanced
/// - All quotes are balanced
/// - It doesn't end with a continuation indicator (backslash)
fn is_query_complete(query: &str) -> bool {
    let trimmed = query.trim();

    // Empty queries are not complete
    if trimmed.is_empty() {
        return false;
    }

    // Check for explicit continuation (backslash at end)
    if trimmed.ends_with('\\') {
        return false;
    }

    // Count braces, brackets, and parentheses
    let mut brace_count = 0;
    let mut bracket_count = 0;
    let mut paren_count = 0;
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    let mut in_triple_single_quote = false;
    let mut in_triple_double_quote = false;
    let mut escape_next = false;

    let chars: Vec<char> = query.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let ch = chars[i];

        if escape_next {
            escape_next = false;
            i += 1;
            continue;
        }

        // Check if we're in any quote mode
        let in_any_quote =
            in_single_quote || in_double_quote || in_triple_single_quote || in_triple_double_quote;

        match ch {
            '\\' if in_any_quote => {
                escape_next = true;
            }
            '\'' if !in_double_quote && !in_triple_double_quote => {
                // Check for triple single quotes
                if i + 2 < chars.len() && chars[i + 1] == '\'' && chars[i + 2] == '\'' {
                    if !in_single_quote {
                        in_triple_single_quote = !in_triple_single_quote;
                        i += 2;
                    }
                } else if !in_triple_single_quote {
                    in_single_quote = !in_single_quote;
                }
            }
            '"' if !in_single_quote && !in_triple_single_quote => {
                // Check for triple double quotes
                if i + 2 < chars.len() && chars[i + 1] == '"' && chars[i + 2] == '"' {
                    if !in_double_quote {
                        in_triple_double_quote = !in_triple_double_quote;
                        i += 2;
                    }
                } else if !in_triple_double_quote {
                    in_double_quote = !in_double_quote;
                }
            }
            '{' if !in_any_quote => {
                brace_count += 1;
            }
            '}' if !in_any_quote => {
                brace_count -= 1;
            }
            '[' if !in_any_quote => {
                bracket_count += 1;
            }
            ']' if !in_any_quote => {
                bracket_count -= 1;
            }
            '(' if !in_any_quote => {
                paren_count += 1;
            }
            ')' if !in_any_quote => {
                paren_count -= 1;
            }
            _ => {}
        }

        i += 1;
    }

    // Query is complete if all delimiters are balanced and not in a quote
    brace_count == 0
        && bracket_count == 0
        && paren_count == 0
        && !in_single_quote
        && !in_double_quote
        && !in_triple_single_quote
        && !in_triple_double_quote
}

/// Convert Term to RdfTerm for formatting
fn term_to_rdf_term(term: &Term) -> RdfTerm {
    match term {
        Term::NamedNode(node) => RdfTerm::Uri {
            value: node.as_str().to_string(),
        },
        Term::BlankNode(bnode) => RdfTerm::Bnode {
            value: bnode.as_str().to_string(),
        },
        Term::Literal(lit) => RdfTerm::Literal {
            value: lit.value().to_string(),
            lang: lit.language().map(|l| l.to_string()),
            datatype: Some(lit.datatype().as_str().to_string()),
        },
        Term::Variable(var) => RdfTerm::Literal {
            value: format!("?{}", var.name()),
            lang: None,
            datatype: None,
        },
        Term::QuotedTriple(triple) => RdfTerm::Literal {
            value: format!(
                "<<{} {} {}>>",
                triple.subject(),
                triple.predicate(),
                triple.object()
            ),
            lang: None,
            datatype: Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement".to_string()),
        },
    }
}

/// Convert Subject to RdfTerm for formatting
fn subject_to_rdf_term(subject: &Subject) -> RdfTerm {
    match subject {
        Subject::NamedNode(node) => RdfTerm::Uri {
            value: node.as_str().to_string(),
        },
        Subject::BlankNode(bnode) => RdfTerm::Bnode {
            value: bnode.as_str().to_string(),
        },
        Subject::Variable(var) => RdfTerm::Literal {
            value: format!("?{}", var.name()),
            lang: None,
            datatype: None,
        },
        Subject::QuotedTriple(triple) => RdfTerm::Literal {
            value: format!(
                "<<{} {} {}>>",
                triple.subject(),
                triple.predicate(),
                triple.object()
            ),
            lang: None,
            datatype: Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement".to_string()),
        },
    }
}

/// Convert Predicate to RdfTerm for formatting
fn predicate_to_rdf_term(predicate: &Predicate) -> RdfTerm {
    match predicate {
        Predicate::NamedNode(node) => RdfTerm::Uri {
            value: node.as_str().to_string(),
        },
        Predicate::Variable(var) => RdfTerm::Literal {
            value: format!("?{}", var.name()),
            lang: None,
            datatype: None,
        },
    }
}

/// Convert Object to RdfTerm for formatting
fn object_to_rdf_term(object: &oxirs_core::model::Object) -> RdfTerm {
    use oxirs_core::model::Object;

    match object {
        Object::NamedNode(node) => RdfTerm::Uri {
            value: node.as_str().to_string(),
        },
        Object::BlankNode(bnode) => RdfTerm::Bnode {
            value: bnode.as_str().to_string(),
        },
        Object::Literal(lit) => RdfTerm::Literal {
            value: lit.value().to_string(),
            lang: lit.language().map(|l| l.to_string()),
            datatype: Some(lit.datatype().as_str().to_string()),
        },
        Object::Variable(var) => RdfTerm::Literal {
            value: format!("?{}", var.name()),
            lang: None,
            datatype: None,
        },
        Object::QuotedTriple(triple) => RdfTerm::Literal {
            value: format!(
                "<<{} {} {}>>",
                triple.subject(),
                triple.predicate(),
                triple.object()
            ),
            lang: None,
            datatype: Some("http://www.w3.org/1999/02/22-rdf-syntax-ns#Statement".to_string()),
        },
    }
}

/// Format and display query results
fn format_and_display_results(
    results: &OxirsQueryResults,
    output_format: &str,
) -> Result<(), String> {
    use std::io;

    // Convert real OxirsQueryResults to formatter QueryResults
    let formatter_results = match results.results() {
        CoreQueryResults::Bindings(bindings) => {
            let variables = results.variables();

            FormatterQueryResults {
                variables: variables.to_vec(),
                bindings: bindings
                    .iter()
                    .map(|var_binding| {
                        // Get values in the order of variables
                        let values: Vec<Option<RdfTerm>> = variables
                            .iter()
                            .map(|var| var_binding.get(var).map(term_to_rdf_term))
                            .collect();

                        Binding { values }
                    })
                    .collect(),
            }
        }
        CoreQueryResults::Boolean(value) => {
            // For ASK queries, return a single binding with the boolean result
            FormatterQueryResults {
                variables: vec!["result".to_string()],
                bindings: vec![Binding {
                    values: vec![Some(RdfTerm::Literal {
                        value: value.to_string(),
                        lang: None,
                        datatype: Some("http://www.w3.org/2001/XMLSchema#boolean".to_string()),
                    })],
                }],
            }
        }
        CoreQueryResults::Graph(quads) => {
            // For CONSTRUCT/DESCRIBE queries, convert quads to bindings
            FormatterQueryResults {
                variables: vec![
                    "subject".to_string(),
                    "predicate".to_string(),
                    "object".to_string(),
                ],
                bindings: quads
                    .iter()
                    .map(|quad| Binding {
                        values: vec![
                            Some(subject_to_rdf_term(quad.subject())),
                            Some(predicate_to_rdf_term(quad.predicate())),
                            Some(object_to_rdf_term(quad.object())),
                        ],
                    })
                    .collect(),
            }
        }
    };

    // Use the comprehensive formatter
    if let Some(formatter) = create_formatter(output_format) {
        let mut stdout = io::stdout();
        formatter
            .format(&formatter_results, &mut stdout)
            .map_err(|e| format!("Failed to format results: {e}"))?;
    } else {
        return Err(format!("Unsupported output format: {output_format}"));
    }

    Ok(())
}

/// Execute interactive mode
pub fn execute(dataset: Option<String>, _config_path: Option<PathBuf>) -> CliResult<()> {
    let dataset_name = dataset.clone().unwrap_or_else(|| "default".to_string());

    // Load dataset
    let dataset_dir = PathBuf::from(&dataset_name);
    let dataset_path = if dataset_dir.join("oxirs.toml").exists() {
        // Dataset with configuration file - extract dataset name from directory name
        let name = dataset_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&dataset_name);
        let (storage_path, _config) = crate::config::load_named_dataset(&dataset_dir, name)
            .map_err(|e| crate::cli::CliError::from(format!("Failed to load dataset: {e}")))?;
        storage_path
    } else {
        // Assume dataset is a directory path
        dataset_dir
    };

    // Open the RDF store
    let store = if dataset_path.is_dir() {
        RdfStore::open(&dataset_path)
            .map_err(|e| crate::cli::CliError::from(format!("Failed to open dataset: {e}")))?
    } else {
        return Err(crate::cli::CliError::from(format!(
            "Dataset not found: {dataset_name}"
        )));
    };

    let mut rl = Editor::<SparqlHelper, DefaultHistory>::new()
        .map_err(|e| crate::cli::CliError::from(format!("Failed to create editor: {}", e)))?;
    rl.set_helper(Some(SparqlHelper));

    // Load history
    let history_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("oxirs");
    let _ = std::fs::create_dir_all(&history_dir);
    let history_file = history_dir.join("history.txt");
    let _ = rl.load_history(&history_file);

    // Initialize session
    let sessions_dir = history_dir.join("sessions");
    let _ = std::fs::create_dir_all(&sessions_dir);
    let mut current_session = QuerySession::new("default".to_string(), dataset_name.clone());

    // Print welcome message
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║              OxiRS Interactive SPARQL Shell             ║");
    println!("║                    Version 0.1.0-alpha.2                 ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("Connected to dataset: {}", dataset_name);
    println!(
        "Session: {} (created: {})",
        current_session.name, current_session.created_at
    );
    println!();
    println!("Commands:");
    println!("  .help             Show help");
    println!("  .quit, .exit      Exit");
    println!("  .stats            Show statistics");
    println!();
    println!("Session Commands:");
    println!("  .session          Show session info");
    println!("  .save <file>      Save session");
    println!("  .load <file>      Load session");
    println!("  .list             List saved sessions");
    println!("  .clear            Clear session");
    println!();
    println!("Query Commands:");
    println!("  .history          Show query history");
    println!("  .show <n>         Show query #n");
    println!("  .replay <n>       Replay query #n");
    println!("  .search <word>    Search queries");
    println!("  .format <n>       Format query #n");
    println!();
    println!("File Operations:");
    println!("  .export <file>    Export queries to file");
    println!("  .import <file>    Import queries from file");
    println!("  .batch <file>     Execute queries from file");
    println!();
    println!("Templates:");
    println!("  .template [name]  Show query templates");
    println!();
    println!("Type your SPARQL query and press Enter");
    println!("Multi-line queries supported - query continues until all braces are balanced");
    println!("Queries are executed immediately and results displayed in table format");
    println!("─────────────────────────────────────────────────────────────");
    println!();

    // Main REPL loop with multi-line support
    let mut accumulated_query = String::new();
    let mut in_multiline = false;

    loop {
        let prompt = if in_multiline {
            "oxirs...> "
        } else {
            "oxirs> "
        };

        let readline = rl.readline(prompt);

        match readline {
            Ok(line) => {
                // Handle Ctrl+C during multi-line input
                if in_multiline && line.trim().is_empty() {
                    // Empty line in multi-line mode - check if user wants to cancel
                    // We'll just add it and continue
                }

                // Accumulate the line
                if in_multiline {
                    accumulated_query.push('\n');
                }
                accumulated_query.push_str(&line);

                // Check if this is a meta-command (only at start of input)
                if !in_multiline && line.trim().starts_with('.') {
                    let trimmed = line.trim();
                    let parts: Vec<&str> = trimmed.splitn(2, ' ').collect();
                    let command = parts[0];

                    match command {
                        ".help" | ".h" => {
                            print_help();
                        }
                        ".quit" | ".q" | ".exit" => {
                            println!("Goodbye!");
                            break;
                        }
                        ".stats" => {
                            println!(
                                "╔═══════════════════════════════════════════════════════════╗"
                            );
                            println!(
                                "║                   Session Statistics                     ║"
                            );
                            println!(
                                "╚═══════════════════════════════════════════════════════════╝"
                            );
                            println!("Dataset:         {}", dataset_name);
                            println!("Status:          Connected");
                            println!("Queries:         {}", current_session.queries.len());
                            println!("Session:         {}", current_session.name);
                            println!("Created:         {}", current_session.created_at);
                            println!("Last Modified:   {}", current_session.modified_at);

                            // Calculate query statistics
                            let total_lines: usize = current_session
                                .queries
                                .iter()
                                .map(|q| q.lines().count())
                                .sum();
                            let avg_lines = if current_session.queries.is_empty() {
                                0.0
                            } else {
                                total_lines as f64 / current_session.queries.len() as f64
                            };

                            let total_chars: usize =
                                current_session.queries.iter().map(|q| q.len()).sum();

                            println!("Total Lines:     {}", total_lines);
                            println!("Avg Lines/Query: {:.1}", avg_lines);
                            println!("Total Chars:     {}", total_chars);
                            println!();
                        }
                        ".session" => {
                            println!(
                                "╔═══════════════════════════════════════════════════════════╗"
                            );
                            println!(
                                "║                   Session Information                    ║"
                            );
                            println!(
                                "╚═══════════════════════════════════════════════════════════╝"
                            );
                            println!("Name:     {}", current_session.name);
                            println!("Dataset:  {}", current_session.dataset);
                            println!("Created:  {}", current_session.created_at);
                            println!("Modified: {}", current_session.modified_at);
                            println!("Queries:  {}", current_session.queries.len());
                            println!();
                        }
                        ".save" => {
                            if parts.len() < 2 {
                                eprintln!("Usage: .save <filename>");
                            } else {
                                let filename = parts[1].trim();
                                let session_path = sessions_dir.join(format!("{}.json", filename));
                                match current_session.save_to_file(&session_path) {
                                    Ok(_) => {
                                        println!("Session saved to: {}", session_path.display())
                                    }
                                    Err(e) => eprintln!("Error saving session: {}", e),
                                }
                            }
                        }
                        ".load" => {
                            if parts.len() < 2 {
                                eprintln!("Usage: .load <filename>");
                            } else {
                                let filename = parts[1].trim();
                                let session_path = sessions_dir.join(format!("{}.json", filename));
                                match QuerySession::load_from_file(&session_path) {
                                    Ok(session) => {
                                        current_session = session;
                                        println!("Session loaded: {}", current_session.name);
                                        println!("Queries: {}", current_session.queries.len());
                                    }
                                    Err(e) => eprintln!("Error loading session: {}", e),
                                }
                            }
                        }
                        ".clear" => {
                            current_session.clear();
                            println!(
                                "Session cleared (queries: {})",
                                current_session.queries.len()
                            );
                        }
                        ".history" => {
                            if current_session.queries.is_empty() {
                                println!("No queries in session history");
                            } else {
                                println!(
                                    "╔═══════════════════════════════════════════════════════════╗"
                                );
                                println!(
                                    "║                   Query History                          ║"
                                );
                                println!(
                                    "╚═══════════════════════════════════════════════════════════╝"
                                );
                                for (i, query) in current_session.queries.iter().enumerate() {
                                    println!("\n[Query #{}]", i + 1);
                                    // Show first 2 lines or full query if short
                                    let lines: Vec<&str> = query.lines().collect();
                                    if lines.len() <= 2 {
                                        println!("{}", query);
                                    } else {
                                        println!("{}", lines[0]);
                                        println!("{}", lines[1]);
                                        println!("... ({} more lines)", lines.len() - 2);
                                    }
                                }
                                println!();
                            }
                        }
                        ".show" => {
                            if parts.len() < 2 {
                                eprintln!("Usage: .show <query_number>");
                            } else if let Ok(n) = parts[1].trim().parse::<usize>() {
                                if n == 0 || n > current_session.queries.len() {
                                    eprintln!(
                                        "Query #{} not found (valid: 1-{})",
                                        n,
                                        current_session.queries.len()
                                    );
                                } else {
                                    println!("\n─── Query #{} ───", n);
                                    println!("{}", current_session.queries[n - 1]);
                                    println!("─────────────────");
                                }
                            } else {
                                eprintln!("Invalid query number: {}", parts[1]);
                            }
                        }
                        ".replay" => {
                            if parts.len() < 2 {
                                eprintln!("Usage: .replay <query_number>");
                            } else if let Ok(n) = parts[1].trim().parse::<usize>() {
                                if n == 0 || n > current_session.queries.len() {
                                    eprintln!(
                                        "Query #{} not found (valid: 1-{})",
                                        n,
                                        current_session.queries.len()
                                    );
                                } else {
                                    let query = current_session.queries[n - 1].clone();
                                    println!("\n╔═══════════════════════════════════════════════════════════╗");
                                    println!(
                                        "║ Replaying Query #{:<37}                              ║",
                                        n
                                    );
                                    println!("╚═══════════════════════════════════════════════════════════╝");

                                    // Show query
                                    let lines: Vec<&str> = query.lines().collect();
                                    for (i, line) in lines.iter().enumerate().take(5) {
                                        println!(" {:3} │ {}", i + 1, line);
                                    }
                                    if lines.len() > 5 {
                                        println!(" ... │ ({} more lines)", lines.len() - 5);
                                    }
                                    println!();

                                    // Execute the query
                                    let start_time = Instant::now();
                                    match store.query(&query) {
                                        Ok(results) => {
                                            let elapsed = start_time.elapsed();
                                            println!(
                                                "✓  Query completed in {:.2}ms",
                                                elapsed.as_secs_f64() * 1000.0
                                            );
                                            println!("   Results: {} solutions", results.len());
                                            println!();

                                            if let Err(e) =
                                                format_and_display_results(&results, "table")
                                            {
                                                eprintln!("⚠  Error formatting results: {}", e);
                                            } else {
                                                println!();
                                            }
                                        }
                                        Err(e) => {
                                            let elapsed = start_time.elapsed();
                                            println!(
                                                "✗  Query failed in {:.2}ms",
                                                elapsed.as_secs_f64() * 1000.0
                                            );
                                            eprintln!("   Error: {}", e);
                                            println!();
                                        }
                                    }
                                }
                            } else {
                                eprintln!("Invalid query number: {}", parts[1]);
                            }
                        }
                        ".export" => {
                            if parts.len() < 2 {
                                eprintln!("Usage: .export <filename.sparql>");
                            } else {
                                let filename = parts[1].trim();
                                let export_path = if filename.contains('/') {
                                    PathBuf::from(filename)
                                } else {
                                    sessions_dir.join(filename)
                                };

                                let queries_text =
                                    current_session.queries.join("\n\n# ───────────\n\n");
                                match fs::write(&export_path, queries_text) {
                                    Ok(_) => {
                                        println!("✓ Queries exported to: {}", export_path.display())
                                    }
                                    Err(e) => eprintln!("✗ Error exporting queries: {}", e),
                                }
                            }
                        }
                        ".import" => {
                            if parts.len() < 2 {
                                eprintln!("Usage: .import <filename.sparql>");
                            } else {
                                let filename = parts[1].trim();
                                let import_path = if filename.contains('/') {
                                    PathBuf::from(filename)
                                } else {
                                    sessions_dir.join(filename)
                                };

                                match fs::read_to_string(&import_path) {
                                    Ok(content) => {
                                        // Split by separator or double newlines
                                        let queries: Vec<String> = if content
                                            .contains("# ───────────")
                                        {
                                            content
                                                .split("# ───────────")
                                                .map(|s| s.trim().to_string())
                                                .filter(|s| !s.is_empty())
                                                .collect()
                                        } else {
                                            // Split by PREFIX blocks or SELECT/ASK/CONSTRUCT/DESCRIBE
                                            content
                                                .split("\n\n")
                                                .map(|s| s.trim().to_string())
                                                .filter(|s| !s.is_empty() && !s.starts_with('#'))
                                                .collect()
                                        };

                                        let count = queries.len();
                                        for query in queries {
                                            current_session.add_query(query);
                                        }

                                        println!(
                                            "✓ Imported {} queries from: {}",
                                            count,
                                            import_path.display()
                                        );
                                        println!(
                                            "  Total queries in session: {}",
                                            current_session.queries.len()
                                        );
                                    }
                                    Err(e) => eprintln!("✗ Error importing file: {}", e),
                                }
                            }
                        }
                        ".batch" => {
                            if parts.len() < 2 {
                                eprintln!("Usage: .batch <filename.sparql>");
                            } else {
                                let filename = parts[1].trim();
                                let batch_path = if filename.contains('/') {
                                    PathBuf::from(filename)
                                } else {
                                    sessions_dir.join(filename)
                                };

                                match fs::read_to_string(&batch_path) {
                                    Ok(content) => {
                                        println!("╔═══════════════════════════════════════════════════════════╗");
                                        println!("║                   Batch Execution                        ║");
                                        println!("╚═══════════════════════════════════════════════════════════╝");
                                        println!("File: {}", batch_path.display());
                                        println!();

                                        let queries: Vec<String> = if content
                                            .contains("# ───────────")
                                        {
                                            content
                                                .split("# ───────────")
                                                .map(|s| s.trim().to_string())
                                                .filter(|s| !s.is_empty())
                                                .collect()
                                        } else {
                                            content
                                                .split("\n\n")
                                                .map(|s| s.trim().to_string())
                                                .filter(|s| !s.is_empty() && !s.starts_with('#'))
                                                .collect()
                                        };

                                        let total_start = Instant::now();
                                        let mut successful = 0;
                                        let mut failed = 0;

                                        for (i, query) in queries.iter().enumerate() {
                                            println!("─── Query {}/{} ───", i + 1, queries.len());

                                            // Show first line
                                            if let Some(first_line) = query.lines().next() {
                                                println!("{}", first_line);
                                                if query.lines().count() > 1 {
                                                    println!(
                                                        "... ({} more lines)",
                                                        query.lines().count() - 1
                                                    );
                                                }
                                            }
                                            println!();

                                            // Execute the query
                                            let start = Instant::now();
                                            match store.query(query) {
                                                Ok(results) => {
                                                    let elapsed = start.elapsed();
                                                    println!(
                                                        "✓ Query completed in {:.2}ms",
                                                        elapsed.as_secs_f64() * 1000.0
                                                    );
                                                    println!(
                                                        "  Results: {} solutions",
                                                        results.len()
                                                    );
                                                    successful += 1;
                                                }
                                                Err(e) => {
                                                    let elapsed = start.elapsed();
                                                    println!(
                                                        "✗ Query failed in {:.2}ms",
                                                        elapsed.as_secs_f64() * 1000.0
                                                    );
                                                    eprintln!("  Error: {}", e);
                                                    failed += 1;
                                                }
                                            }
                                            println!();

                                            current_session.add_query(query.clone());
                                        }

                                        let total_elapsed = total_start.elapsed();
                                        println!("═══════════════════════════════════════════════════════════");
                                        println!(
                                            "Batch complete: {} queries in {:.2}ms",
                                            queries.len(),
                                            total_elapsed.as_secs_f64() * 1000.0
                                        );
                                        println!("  Successful: {}", successful);
                                        println!("  Failed:     {}", failed);
                                        println!(
                                            "  Average:    {:.2}ms per query",
                                            total_elapsed.as_secs_f64() * 1000.0
                                                / queries.len() as f64
                                        );
                                        println!();
                                    }
                                    Err(e) => eprintln!("✗ Error reading batch file: {}", e),
                                }
                            }
                        }
                        ".search" => {
                            if parts.len() < 2 {
                                eprintln!("Usage: .search <keyword>");
                            } else {
                                let keyword = parts[1].trim();
                                let matches: Vec<(usize, &String)> = current_session
                                    .queries
                                    .iter()
                                    .enumerate()
                                    .filter(|(_, q)| {
                                        q.to_lowercase().contains(&keyword.to_lowercase())
                                    })
                                    .collect();

                                if matches.is_empty() {
                                    println!("No queries found matching: {}", keyword);
                                } else {
                                    println!("╔═══════════════════════════════════════════════════════════╗");
                                    println!(
                                        "║         Search Results for: {:<29}                  ║",
                                        keyword
                                    );
                                    println!("╚═══════════════════════════════════════════════════════════╝");
                                    println!("Found {} matches:\n", matches.len());

                                    for (idx, query) in matches {
                                        println!("[Query #{}]", idx + 1);
                                        let lines: Vec<&str> = query.lines().collect();
                                        if lines.len() <= 2 {
                                            println!("{}", query);
                                        } else {
                                            println!("{}", lines[0]);
                                            println!("... ({} more lines)", lines.len() - 1);
                                        }
                                        println!();
                                    }
                                }
                            }
                        }
                        ".format" => {
                            if parts.len() < 2 {
                                eprintln!("Usage: .format <query_number>");
                            } else if let Ok(n) = parts[1].trim().parse::<usize>() {
                                if n == 0 || n > current_session.queries.len() {
                                    eprintln!(
                                        "Query #{} not found (valid: 1-{})",
                                        n,
                                        current_session.queries.len()
                                    );
                                } else {
                                    let query = &current_session.queries[n - 1];
                                    println!("\n─── Formatted Query #{} ───", n);
                                    println!("{}", format_sparql_query(query));
                                    println!("──────────────────────────");
                                }
                            } else {
                                eprintln!("Invalid query number: {}", parts[1]);
                            }
                        }
                        ".template" => {
                            if parts.len() < 2 {
                                println!("Available templates:");
                                println!("  select    - Basic SELECT query");
                                println!("  construct - CONSTRUCT query");
                                println!("  ask       - ASK query");
                                println!("  describe  - DESCRIBE query");
                                println!("  filter    - SELECT with FILTER");
                                println!("  optional  - SELECT with OPTIONAL");
                                println!("\nUsage: .template <name>");
                            } else {
                                let template_name = parts[1].trim();
                                if let Some(template) = get_query_template(template_name) {
                                    println!("\n─── Template: {} ───", template_name);
                                    println!("{}", template);
                                    println!("────────────────────────");
                                    println!("\nType or paste to use this template");
                                } else {
                                    eprintln!("Unknown template: {}", template_name);
                                    eprintln!("Use .template to see available templates");
                                }
                            }
                        }
                        ".list" => {
                            println!(
                                "╔═══════════════════════════════════════════════════════════╗"
                            );
                            println!(
                                "║                   Available Sessions                     ║"
                            );
                            println!(
                                "╚═══════════════════════════════════════════════════════════╝"
                            );
                            match fs::read_dir(&sessions_dir) {
                                Ok(entries) => {
                                    let mut found_any = false;
                                    for entry in entries.flatten() {
                                        if let Some(name) = entry.file_name().to_str() {
                                            if name.ends_with(".json") {
                                                found_any = true;
                                                let session_name = name.trim_end_matches(".json");
                                                println!("  {}", session_name);
                                            }
                                        }
                                    }
                                    if !found_any {
                                        println!("  No saved sessions found");
                                    }
                                }
                                Err(e) => eprintln!("Error reading sessions directory: {}", e),
                            }
                            println!();
                        }
                        _ => {
                            eprintln!("Unknown command: {}", trimmed);
                            println!("Type .help for available commands");
                        }
                    }
                    accumulated_query.clear();
                    continue;
                }

                // Check if the query is complete
                if is_query_complete(&accumulated_query) {
                    let query = accumulated_query.trim().to_string();

                    // Skip empty queries
                    if !query.is_empty() {
                        let start_time = Instant::now();

                        // Validate syntax and show hints
                        let hints = validate_sparql_syntax(&query);

                        // Add complete query to history
                        let _ = rl.add_history_entry(&query);

                        // Add query to session
                        current_session.add_query(query.clone());

                        // Display query header
                        println!("\n╔═══════════════════════════════════════════════════════════╗");
                        println!(
                            "║ Query #{:<49}                                       ║",
                            current_session.queries.len()
                        );
                        println!("╚═══════════════════════════════════════════════════════════╝");

                        // Show query with line numbers (first 5 lines)
                        let lines: Vec<&str> = query.lines().collect();
                        for (i, line) in lines.iter().enumerate().take(5) {
                            println!(" {:3} │ {}", i + 1, line);
                        }
                        if lines.len() > 5 {
                            println!(" ... │ ({} more lines)", lines.len() - 5);
                        }
                        println!();

                        // Show syntax hints if any
                        if !hints.is_empty() {
                            println!("⚠  Syntax Hints:");
                            for hint in &hints {
                                println!("   • {}", hint);
                            }
                            println!();
                        }

                        // Execute the query
                        match store.query(&query) {
                            Ok(results) => {
                                let elapsed = start_time.elapsed();

                                // Display result count
                                println!(
                                    "✓  Query completed in {:.2}ms",
                                    elapsed.as_secs_f64() * 1000.0
                                );
                                println!("   Results: {} solutions", results.len());
                                println!();

                                // Format and display results
                                if let Err(e) = format_and_display_results(&results, "table") {
                                    eprintln!("⚠  Error formatting results: {}", e);
                                } else {
                                    println!();
                                }
                            }
                            Err(e) => {
                                let elapsed = start_time.elapsed();
                                println!(
                                    "✗  Query failed in {:.2}ms",
                                    elapsed.as_secs_f64() * 1000.0
                                );
                                eprintln!("   Error: {}", e);
                                println!();
                            }
                        }
                    }

                    // Reset for next query
                    accumulated_query.clear();
                    in_multiline = false;
                } else {
                    // Query is incomplete, continue in multi-line mode
                    in_multiline = true;
                }
            }
            Err(ReadlineError::Interrupted) => {
                if in_multiline {
                    println!("^C (multi-line input cancelled)");
                    accumulated_query.clear();
                    in_multiline = false;
                } else {
                    println!("^C");
                }
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("Bye!");
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        }
    }

    // Save history
    let _ = rl.save_history(&history_file);

    // Auto-save session on exit if it has queries
    if !current_session.queries.is_empty() {
        let autosave_path = sessions_dir.join("_autosave.json");
        if let Ok(()) = current_session.save_to_file(&autosave_path) {
            println!("\n✓ Session auto-saved to: {}", autosave_path.display());
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_complete_simple() {
        assert!(is_query_complete("SELECT * WHERE { ?s ?p ?o }"));
        assert!(is_query_complete(
            "ASK { ?s a <http://example.org/Person> }"
        ));
        assert!(is_query_complete(
            "PREFIX ex: <http://example.org/> SELECT * WHERE { ?s ?p ?o }"
        ));
    }

    #[test]
    fn test_query_incomplete_braces() {
        assert!(!is_query_complete("SELECT * WHERE {"));
        assert!(!is_query_complete("SELECT * WHERE { ?s ?p ?o"));
        assert!(!is_query_complete("SELECT * WHERE { ?s { ?p ?o }"));
    }

    #[test]
    fn test_query_complete_nested_braces() {
        assert!(is_query_complete(
            "SELECT * WHERE { { ?s ?p ?o } UNION { ?a ?b ?c } }"
        ));
        assert!(is_query_complete(
            "SELECT * WHERE { GRAPH <g> { ?s ?p ?o } }"
        ));
    }

    #[test]
    fn test_query_incomplete_quotes() {
        assert!(!is_query_complete("SELECT * WHERE { ?s ?p \"unclosed"));
        assert!(!is_query_complete("SELECT * WHERE { ?s ?p 'unclosed"));
    }

    #[test]
    fn test_query_complete_quotes() {
        assert!(is_query_complete("SELECT * WHERE { ?s ?p \"value\" }"));
        assert!(is_query_complete("SELECT * WHERE { ?s ?p 'value' }"));
        assert!(is_query_complete(
            r#"SELECT * WHERE { ?s ?p "value with \"escaped\" quotes" }"#
        ));
    }

    #[test]
    fn test_query_complete_triple_quotes() {
        assert!(is_query_complete(
            r#"SELECT * WHERE { ?s ?p """triple quoted value""" }"#
        ));
        assert!(is_query_complete(
            r#"SELECT * WHERE { ?s ?p '''triple quoted value''' }"#
        ));
    }

    #[test]
    fn test_query_incomplete_triple_quotes() {
        assert!(!is_query_complete(r#"SELECT * WHERE { ?s ?p """unclosed"#));
        assert!(!is_query_complete(r#"SELECT * WHERE { ?s ?p '''unclosed"#));
    }

    #[test]
    fn test_query_complete_brackets() {
        assert!(is_query_complete("SELECT * WHERE { ?s [ ?p ?o ] }"));
        assert!(is_query_complete("SELECT * WHERE { [ ?p ?o ] ?p2 ?o2 }"));
    }

    #[test]
    fn test_query_incomplete_brackets() {
        assert!(!is_query_complete("SELECT * WHERE { ?s [ ?p ?o }"));
        assert!(!is_query_complete("SELECT * WHERE { [ ?p ?o ?p2 ?o2 }"));
    }

    #[test]
    fn test_query_complete_parentheses() {
        assert!(is_query_complete("SELECT * WHERE { FILTER (1 + 2) }"));
        assert!(is_query_complete(
            "SELECT * WHERE { BIND ((1 + 2) AS ?sum) }"
        ));
    }

    #[test]
    fn test_query_incomplete_parentheses() {
        assert!(!is_query_complete("SELECT * WHERE { FILTER (1 + 2 }"));
        assert!(!is_query_complete(
            "SELECT * WHERE { BIND ((1 + 2 AS ?sum) }"
        ));
    }

    #[test]
    fn test_query_continuation_backslash() {
        assert!(!is_query_complete("SELECT * WHERE { ?s ?p ?o } \\"));
        assert!(!is_query_complete("PREFIX ex: <http://example.org/> \\"));
    }

    #[test]
    fn test_query_empty() {
        assert!(!is_query_complete(""));
        assert!(!is_query_complete("   "));
        assert!(!is_query_complete("\n\n"));
    }

    #[test]
    fn test_query_complex_multiline() {
        let query = r#"SELECT ?name ?email WHERE {
            ?person foaf:name ?name .
            ?person foaf:mbox ?email
        }"#;
        assert!(is_query_complete(query));
    }

    #[test]
    fn test_query_with_comments() {
        let query = "SELECT * WHERE { # This is a comment\n ?s ?p ?o }";
        assert!(is_query_complete(query));
    }

    #[test]
    fn test_query_braces_in_strings() {
        assert!(is_query_complete(
            r#"SELECT * WHERE { ?s ?p "value with { braces }" }"#
        ));
        assert!(is_query_complete(
            r#"SELECT * WHERE { ?s ?p "value with ( parens )" }"#
        ));
        assert!(is_query_complete(
            r#"SELECT * WHERE { ?s ?p "value with [ brackets ]" }"#
        ));
    }

    #[test]
    fn test_syntax_validation_valid_query() {
        let query = "SELECT * WHERE { ?s ?p ?o }";
        let hints = validate_sparql_syntax(query);
        assert!(hints.is_empty());
    }

    #[test]
    fn test_syntax_validation_missing_where() {
        let query = "SELECT * { ?s ?p ?o }";
        let hints = validate_sparql_syntax(query);
        assert!(!hints.is_empty());
        assert!(hints.iter().any(|h| h.contains("WHERE")));
    }

    #[test]
    fn test_syntax_validation_missing_prefix() {
        let query = "SELECT * WHERE { ?s rdf:type ?o }";
        let hints = validate_sparql_syntax(query);
        assert!(hints.iter().any(|h| h.contains("PREFIX rdf:")));
    }

    #[test]
    fn test_syntax_validation_with_prefix() {
        let query = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\nSELECT * WHERE { ?s rdf:type ?o }";
        let hints = validate_sparql_syntax(query);
        assert!(hints.iter().all(|h| !h.contains("PREFIX rdf:")));
    }

    #[test]
    fn test_syntax_validation_multiple_prefixes() {
        let query = "SELECT * WHERE { ?s rdf:type ?o . ?s foaf:name ?name }";
        let hints = validate_sparql_syntax(query);
        assert!(hints.len() >= 2); // Should suggest both rdf and foaf prefixes
    }

    #[test]
    fn test_syntax_validation_ask_query() {
        let query = "ASK { ?s ?p ?o }";
        let hints = validate_sparql_syntax(query);
        assert!(hints.is_empty());
    }

    #[test]
    fn test_syntax_validation_filter_syntax() {
        let query = "SELECT * WHERE { ?s ?p ?o FILTER ?o > 10 }";
        let hints = validate_sparql_syntax(query);
        assert!(hints.iter().any(|h| h.contains("FILTER")));
    }
}

fn print_help() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║                 OxiRS Interactive Help                   ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("Meta Commands:");
    println!("  .help             Show this help message");
    println!("  .quit, .exit      Exit the shell");
    println!("  .stats            Show dataset and session statistics");
    println!();
    println!("Session Management:");
    println!("  .session          Show current session information");
    println!("  .save <filename>  Save session to file");
    println!("  .load <filename>  Load session from file");
    println!("  .list             List all saved sessions");
    println!("  .clear            Clear current session queries");
    println!();
    println!("Query Management:");
    println!("  .history          Show all queries in session");
    println!("  .show <n>         Display query number n");
    println!("  .replay <n>       Re-execute query number n");
    println!("  .search <keyword> Search queries containing keyword");
    println!("  .format <n>       Format and pretty-print query number n");
    println!();
    println!("File Operations:");
    println!("  .export <file>    Export all queries to SPARQL file");
    println!("  .import <file>    Import queries from SPARQL file");
    println!("  .batch <file>     Execute all queries from file with timing");
    println!();
    println!("Templates:");
    println!("  .template         List available query templates");
    println!("  .template <name>  Show specific template (select, construct, etc.)");
    println!();
    println!("  Sessions are stored in: ~/.local/share/oxirs/sessions/");
    println!("  Each session contains query history and metadata.");
    println!();
    println!("Query Examples:");
    println!("  SELECT * WHERE {{ ?s ?p ?o }} LIMIT 10");
    println!("  ASK {{ ?s a <http://example.org/Person> }}");
    println!();
    println!("Multi-line Queries:");
    println!("  Queries automatically continue over multiple lines until");
    println!("  all braces, brackets, and quotes are balanced.");
    println!("  The prompt changes to 'oxirs...>' for continuation lines.");
    println!();
    println!("  Example:");
    println!("    oxirs> SELECT ?name ?email WHERE {{");
    println!("    oxirs...>   ?person foaf:name ?name .");
    println!("    oxirs...>   ?person foaf:mbox ?email");
    println!("    oxirs...> }} LIMIT 10");
    println!();
    println!("Keyboard Shortcuts:");
    println!("  Ctrl+C            Cancel multi-line input or interrupt");
    println!("  Ctrl+D            Exit the shell");
    println!("  Up/Down           Navigate history");
    println!("  Backslash (\\)     Force continuation to next line");
    println!();
}
