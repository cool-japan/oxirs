//! # Validation Core
//!
//! Internal validation logic: SPARQL query/update validation, IRI validation,
//! RDF data parsing, language tag parsing, and all helper utilities.

use crate::handlers::validation_types::{
    DataValidationRequest, DataValidationResponse, IriValidationRequest, IriValidationResponse,
    IriValidationResult, LangTagValidationRequest, LangTagValidationResponse,
    LangTagValidationResult, PrefixMapping, QueryValidationRequest, QueryValidationResponse,
    UpdateValidationRequest, UpdateValidationResponse, ValidationError, ValidationSummary,
    ValidationWarning,
};

// ============================================================================
// Internal Validation Functions (pub(crate) for use in handlers)
// ============================================================================

/// Internal function to validate SPARQL queries
pub fn validate_sparql_query_internal(request: &QueryValidationRequest) -> QueryValidationResponse {
    let query_str = request.query.trim();

    if query_str.is_empty() {
        return QueryValidationResponse {
            valid: false,
            input: request.query.clone(),
            formatted: None,
            query_type: None,
            algebra: None,
            algebra_optimized: None,
            variables: None,
            prefixes: None,
            errors: vec![ValidationError {
                message: "Empty query string".to_string(),
                line: Some(1),
                column: Some(1),
                code: Some("EMPTY_QUERY".to_string()),
            }],
            warnings: vec![],
        };
    }

    // Validate the query using available validation functions
    // Note: oxirs-core doesn't expose a simple Query::parse API yet,
    // so we use a combination of syntax checking and the existing validate function
    match crate::handlers::sparql::validate_sparql_query(query_str) {
        Ok(_) => {
            // Extract query information
            let query_type = detect_query_type(query_str);
            let variables = extract_query_variables(query_str);
            let prefixes = extract_prefixes(query_str);

            // Generate formatted output
            let formatted = format_query(query_str);

            // Generate algebra representation (simplified)
            let algebra = if request.include_algebra {
                Some(generate_algebra_representation(query_str))
            } else {
                None
            };

            // Generate optimized algebra (placeholder - would need actual optimizer)
            let algebra_optimized = if request.include_optimized {
                Some(generate_optimized_algebra(query_str))
            } else {
                None
            };

            let mut warnings = vec![];

            // Check for potential issues
            if query_str.contains("SELECT *") {
                warnings.push(ValidationWarning {
                    message: "Using SELECT * may return more data than needed".to_string(),
                    line: None,
                    code: Some("SELECT_STAR".to_string()),
                });
            }

            if !query_str.to_uppercase().contains("LIMIT")
                && query_type == Some("SELECT".to_string())
            {
                warnings.push(ValidationWarning {
                    message: "Query has no LIMIT clause, may return large result sets".to_string(),
                    line: None,
                    code: Some("NO_LIMIT".to_string()),
                });
            }

            QueryValidationResponse {
                valid: true,
                input: request.query.clone(),
                formatted: Some(formatted),
                query_type,
                algebra,
                algebra_optimized,
                variables: Some(variables),
                prefixes: Some(prefixes),
                errors: vec![],
                warnings,
            }
        }
        Err(e) => {
            // Parse error occurred
            let (line, column) = extract_error_location(&e.to_string());

            QueryValidationResponse {
                valid: false,
                input: request.query.clone(),
                formatted: None,
                query_type: None,
                algebra: None,
                algebra_optimized: None,
                variables: None,
                prefixes: None,
                errors: vec![ValidationError {
                    message: e.to_string(),
                    line,
                    column,
                    code: Some("PARSE_ERROR".to_string()),
                }],
                warnings: vec![],
            }
        }
    }
}

/// Internal function to validate SPARQL Update operations
pub fn validate_sparql_update_internal(
    request: &UpdateValidationRequest,
) -> UpdateValidationResponse {
    let update_str = request.update.trim();

    if update_str.is_empty() {
        return UpdateValidationResponse {
            valid: false,
            input: request.update.clone(),
            formatted: None,
            operations: vec![],
            affected_graphs: vec![],
            errors: vec![ValidationError {
                message: "Empty update string".to_string(),
                line: Some(1),
                column: Some(1),
                code: Some("EMPTY_UPDATE".to_string()),
            }],
            warnings: vec![],
        };
    }

    // Validate the update using basic syntax checking
    // Note: oxirs-core doesn't expose a simple Update::parse API yet
    let is_valid_update = validate_sparql_update_syntax(update_str);

    match is_valid_update {
        Ok(_) => {
            // Extract operation types
            let operations = extract_update_operations(update_str);

            // Extract affected graphs
            let affected_graphs = extract_affected_graphs(update_str);

            // Format the update
            let formatted = format_update(update_str);

            let mut warnings = vec![];

            // Check for potential issues
            if update_str.to_uppercase().contains("DELETE WHERE")
                && !update_str.to_uppercase().contains("GRAPH")
            {
                warnings.push(ValidationWarning {
                    message: "DELETE WHERE without GRAPH clause affects default graph".to_string(),
                    line: None,
                    code: Some("DEFAULT_GRAPH_DELETE".to_string()),
                });
            }

            if update_str.to_uppercase().contains("DROP ALL") {
                warnings.push(ValidationWarning {
                    message: "DROP ALL will remove all data from all graphs".to_string(),
                    line: None,
                    code: Some("DROP_ALL".to_string()),
                });
            }

            if update_str.to_uppercase().contains("CLEAR ALL") {
                warnings.push(ValidationWarning {
                    message: "CLEAR ALL will remove all triples from all graphs".to_string(),
                    line: None,
                    code: Some("CLEAR_ALL".to_string()),
                });
            }

            UpdateValidationResponse {
                valid: true,
                input: request.update.clone(),
                formatted: Some(formatted),
                operations,
                affected_graphs,
                errors: vec![],
                warnings,
            }
        }
        Err(e) => {
            let (line, column) = extract_error_location(&e.to_string());

            UpdateValidationResponse {
                valid: false,
                input: request.update.clone(),
                formatted: None,
                operations: vec![],
                affected_graphs: vec![],
                errors: vec![ValidationError {
                    message: e.to_string(),
                    line,
                    column,
                    code: Some("PARSE_ERROR".to_string()),
                }],
                warnings: vec![],
            }
        }
    }
}

/// Internal function to validate IRIs
pub fn validate_iris_internal(request: &IriValidationRequest) -> IriValidationResponse {
    let mut results = Vec::new();
    let mut valid_count = 0;
    let mut warning_count = 0;

    for iri_str in &request.iris {
        let iri_str = iri_str.trim();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut is_valid = true;
        let mut is_absolute = false;
        let mut is_relative = false;
        let mut scheme = None;

        if iri_str.is_empty() {
            errors.push("Empty IRI string".to_string());
            is_valid = false;
        } else {
            // Parse the IRI using oxirs-core
            match oxirs_core::model::NamedNode::new(iri_str) {
                Ok(node) => {
                    let iri = node.as_str();

                    // Check if absolute or relative
                    if iri.contains("://") {
                        is_absolute = true;
                        // Extract scheme
                        if let Some(idx) = iri.find("://") {
                            scheme = Some(iri[..idx].to_string());
                        }
                    } else if iri.starts_with("urn:") || iri.starts_with("mailto:") {
                        is_absolute = true;
                        if let Some(idx) = iri.find(':') {
                            scheme = Some(iri[..idx].to_string());
                        }
                    } else {
                        is_relative = true;
                        if request.check_relative {
                            warnings.push("Relative IRI detected".to_string());
                            warning_count += 1;
                        }
                    }

                    // Additional validation checks
                    if iri.contains(' ') {
                        warnings.push("IRI contains spaces".to_string());
                        warning_count += 1;
                    }

                    if iri.contains("..") {
                        warnings.push("IRI contains '..' path segments".to_string());
                        warning_count += 1;
                    }

                    // Check for common issues
                    if iri.ends_with('#') || iri.ends_with('/') {
                        // This is actually fine for namespace IRIs
                    }

                    valid_count += 1;
                }
                Err(e) => {
                    errors.push(format!("Invalid IRI: {}", e));
                    is_valid = false;
                }
            }
        }

        results.push(IriValidationResult {
            iri: iri_str.to_string(),
            valid: is_valid,
            is_absolute,
            is_relative,
            scheme,
            errors,
            warnings,
        });
    }

    IriValidationResponse {
        summary: ValidationSummary {
            total: results.len(),
            valid: valid_count,
            invalid: results.len() - valid_count,
            warnings: warning_count,
        },
        results,
    }
}

/// Internal function to validate RDF data
pub fn validate_rdf_data_internal(request: &DataValidationRequest) -> DataValidationResponse {
    let data_str = request.data.trim();

    if data_str.is_empty() {
        return DataValidationResponse {
            valid: false,
            format: request.format.clone(),
            triple_count: 0,
            graph_count: None,
            subject_count: 0,
            predicate_count: 0,
            object_count: 0,
            blank_node_count: 0,
            literal_count: 0,
            errors: vec![ValidationError {
                message: "Empty data string".to_string(),
                line: Some(1),
                column: Some(1),
                code: Some("EMPTY_DATA".to_string()),
            }],
            warnings: vec![],
            sample_triples: vec![],
        };
    }

    // Determine format
    let format = normalize_format(&request.format);
    let base_iri = request
        .base
        .clone()
        .unwrap_or_else(|| "http://example.org/base/".to_string());

    // Parse RDF data
    match parse_rdf_data(data_str, &format, &base_iri) {
        Ok(parse_result) => {
            let mut warnings = vec![];

            // Check for potential issues
            if parse_result.blank_node_count > parse_result.triple_count / 2 {
                warnings.push(ValidationWarning {
                    message: "High ratio of blank nodes to triples".to_string(),
                    line: None,
                    code: Some("HIGH_BNODE_RATIO".to_string()),
                });
            }

            DataValidationResponse {
                valid: true,
                format,
                triple_count: parse_result.triple_count,
                graph_count: parse_result.graph_count,
                subject_count: parse_result.subject_count,
                predicate_count: parse_result.predicate_count,
                object_count: parse_result.object_count,
                blank_node_count: parse_result.blank_node_count,
                literal_count: parse_result.literal_count,
                errors: vec![],
                warnings,
                sample_triples: parse_result.sample_triples,
            }
        }
        Err(e) => {
            let (line, column) = extract_error_location(&e);

            DataValidationResponse {
                valid: false,
                format,
                triple_count: 0,
                graph_count: None,
                subject_count: 0,
                predicate_count: 0,
                object_count: 0,
                blank_node_count: 0,
                literal_count: 0,
                errors: vec![ValidationError {
                    message: e,
                    line,
                    column,
                    code: Some("PARSE_ERROR".to_string()),
                }],
                warnings: vec![],
                sample_triples: vec![],
            }
        }
    }
}

/// Internal function to validate language tags
pub fn validate_langtags_internal(request: &LangTagValidationRequest) -> LangTagValidationResponse {
    let mut results = Vec::new();
    let mut valid_count = 0;
    let mut warning_count = 0;

    for tag_str in &request.tags {
        let tag_str = tag_str.trim();
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut is_valid = true;

        let mut language = None;
        let mut script = None;
        let mut region = None;
        let mut variants = Vec::new();
        let mut extensions = Vec::new();
        let mut private_use = None;

        if tag_str.is_empty() {
            errors.push("Empty language tag".to_string());
            is_valid = false;
        } else {
            // Parse BCP 47 language tag
            let result = parse_language_tag(tag_str);

            match result {
                Ok(parsed) => {
                    language = parsed.language;
                    script = parsed.script;
                    region = parsed.region;
                    variants = parsed.variants;
                    extensions = parsed.extensions;
                    private_use = parsed.private_use;

                    // Validate components
                    if language.is_none() && private_use.is_none() {
                        errors.push("Missing primary language subtag".to_string());
                        is_valid = false;
                    }

                    // Check for deprecated tags
                    if let Some(ref lang) = language {
                        if is_deprecated_language(lang) {
                            warnings.push(format!("Language subtag '{}' is deprecated", lang));
                            warning_count += 1;
                        }
                    }

                    // Check for unusual combinations
                    if script.is_some() && language.is_none() {
                        warnings.push("Script subtag without language subtag".to_string());
                        warning_count += 1;
                    }

                    if is_valid {
                        valid_count += 1;
                    }
                }
                Err(e) => {
                    errors.push(e);
                    is_valid = false;
                }
            }
        }

        results.push(LangTagValidationResult {
            tag: tag_str.to_string(),
            valid: is_valid,
            language,
            script,
            region,
            variants,
            extensions,
            private_use,
            errors,
            warnings,
        });
    }

    LangTagValidationResponse {
        summary: ValidationSummary {
            total: results.len(),
            valid: valid_count,
            invalid: results.len() - valid_count,
            warnings: warning_count,
        },
        results,
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate SPARQL Update syntax
pub fn validate_sparql_update_syntax(update: &str) -> Result<(), String> {
    if update.trim().is_empty() {
        return Err("Empty update string".to_string());
    }

    // Use regex for word-boundary matching to avoid false positives like "INSERTT"
    let operations_patterns = [
        r"(?i)\bINSERT\s+DATA\b",
        r"(?i)\bDELETE\s+DATA\b",
        r"(?i)\bINSERT\s*\{", // INSERT { ... } or INSERT WHERE { ... }
        r"(?i)\bDELETE\s*\{", // DELETE { ... } or DELETE WHERE { ... }
        r"(?i)\bLOAD\b",
        r"(?i)\bCLEAR\b",
        r"(?i)\bDROP\b",
        r"(?i)\bCREATE\b",
        r"(?i)\bCOPY\b",
        r"(?i)\bMOVE\b",
        r"(?i)\bADD\b",
        r"(?i)\bDELETE\s+WHERE\b",
        r"(?i)\bINSERT\s+WHERE\b",
        r"(?i)\bWITH\s+", // WITH <graph> ... INSERT/DELETE
    ];

    let has_valid_operation = operations_patterns.iter().any(|pattern| {
        regex::Regex::new(pattern)
            .map(|re| re.is_match(update))
            .unwrap_or(false)
    });

    if !has_valid_operation {
        return Err("Update must contain a valid SPARQL Update operation (INSERT, DELETE, LOAD, CLEAR, DROP, CREATE, COPY, MOVE, ADD)".to_string());
    }

    // Basic brace matching check
    let open_braces = update.matches('{').count();
    let close_braces = update.matches('}').count();
    if open_braces != close_braces {
        return Err(format!(
            "Mismatched braces: {} open, {} close",
            open_braces, close_braces
        ));
    }

    Ok(())
}

/// Detect query type from SPARQL query string
pub fn detect_query_type(query: &str) -> Option<String> {
    let upper = query.to_uppercase();
    if upper.contains("SELECT") {
        Some("SELECT".to_string())
    } else if upper.contains("CONSTRUCT") {
        Some("CONSTRUCT".to_string())
    } else if upper.contains("ASK") {
        Some("ASK".to_string())
    } else if upper.contains("DESCRIBE") {
        Some("DESCRIBE".to_string())
    } else {
        None
    }
}

/// Extract variable names from query
pub fn extract_query_variables(query: &str) -> Vec<String> {
    let mut variables = Vec::new();
    let mut chars = query.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '?' || c == '$' {
            let mut var_name = String::new();
            while let Some(&nc) = chars.peek() {
                if nc.is_alphanumeric() || nc == '_' {
                    var_name.push(chars.next().expect("char should exist after peek"));
                } else {
                    break;
                }
            }
            if !var_name.is_empty() && !variables.contains(&var_name) {
                variables.push(var_name);
            }
        }
    }

    variables
}

/// Extract prefix declarations from query
pub fn extract_prefixes(query: &str) -> Vec<PrefixMapping> {
    let mut prefixes = Vec::new();
    let prefix_regex =
        regex::Regex::new(r"(?i)PREFIX\s+(\w*):\s*<([^>]+)>").expect("valid regex pattern");

    for cap in prefix_regex.captures_iter(query) {
        prefixes.push(PrefixMapping {
            prefix: cap.get(1).map_or("", |m| m.as_str()).to_string(),
            iri: cap.get(2).map_or("", |m| m.as_str()).to_string(),
        });
    }

    prefixes
}

/// Format/pretty-print a SPARQL query
pub fn format_query(query: &str) -> String {
    // Simple formatting - indent WHERE clause, etc.
    let mut formatted = query.to_string();

    // Add newlines before main keywords
    let keywords = [
        "SELECT",
        "CONSTRUCT",
        "ASK",
        "DESCRIBE",
        "WHERE",
        "ORDER BY",
        "LIMIT",
        "OFFSET",
        "FILTER",
        "OPTIONAL",
        "UNION",
        "BIND",
        "VALUES",
        "GROUP BY",
        "HAVING",
    ];

    for keyword in keywords {
        let pattern = format!(r"(?i)\b{}\b", keyword);
        let re = regex::Regex::new(&pattern).expect("valid regex pattern");
        formatted = re
            .replace_all(&formatted, |caps: &regex::Captures| {
                format!(
                    "\n{}",
                    caps.get(0).expect("capture group 0 should exist").as_str()
                )
            })
            .to_string();
    }

    // Clean up multiple newlines
    let multi_newline = regex::Regex::new(r"\n\s*\n").expect("valid regex pattern");
    formatted = multi_newline.replace_all(&formatted, "\n").to_string();

    formatted.trim().to_string()
}

/// Format SPARQL Update
pub fn format_update(update: &str) -> String {
    let mut formatted = update.to_string();

    let keywords = [
        "INSERT", "DELETE", "WHERE", "DATA", "GRAPH", "WITH", "USING", "CREATE", "DROP", "COPY",
        "MOVE", "ADD", "CLEAR", "LOAD",
    ];

    for keyword in keywords {
        let pattern = format!(r"(?i)\b{}\b", keyword);
        let re = regex::Regex::new(&pattern).expect("valid regex pattern");
        formatted = re
            .replace_all(&formatted, |caps: &regex::Captures| {
                format!(
                    "\n{}",
                    caps.get(0).expect("capture group 0 should exist").as_str()
                )
            })
            .to_string();
    }

    let multi_newline = regex::Regex::new(r"\n\s*\n").expect("valid regex pattern");
    formatted = multi_newline.replace_all(&formatted, "\n").to_string();

    formatted.trim().to_string()
}

/// Generate algebra representation (simplified)
pub fn generate_algebra_representation(query: &str) -> String {
    // This is a simplified representation
    // A full implementation would use the actual SPARQL algebra
    let query_type = detect_query_type(query).unwrap_or_else(|| "UNKNOWN".to_string());

    let variables = extract_query_variables(query);
    let var_str = variables
        .iter()
        .map(|v| format!("?{}", v))
        .collect::<Vec<_>>()
        .join(" ");

    // Extract basic triple patterns
    let triple_pattern_re = regex::Regex::new(r"\{([^{}]+)\}").expect("valid regex pattern");
    let patterns: Vec<String> = triple_pattern_re
        .captures_iter(query)
        .filter_map(|cap| cap.get(1).map(|m| m.as_str().trim().to_string()))
        .collect();

    format!(
        "({}
  (project ({})
    (bgp
      {}
    )))",
        query_type.to_lowercase(),
        var_str,
        patterns.join("\n      ")
    )
}

/// Generate optimized algebra representation
pub fn generate_optimized_algebra(query: &str) -> String {
    // For now, return same as regular algebra
    // A full implementation would apply optimizations
    let algebra = generate_algebra_representation(query);
    format!("; Optimized\n{}", algebra)
}

/// Extract update operation types
pub fn extract_update_operations(update: &str) -> Vec<String> {
    let mut operations = Vec::new();
    let upper = update.to_uppercase();

    let op_types = [
        "INSERT DATA",
        "DELETE DATA",
        "INSERT",
        "DELETE",
        "CLEAR",
        "DROP",
        "CREATE",
        "COPY",
        "MOVE",
        "ADD",
        "LOAD",
    ];

    for op in op_types {
        if upper.contains(op) {
            operations.push(op.to_string());
        }
    }

    // Remove duplicates while preserving order
    let mut seen = std::collections::HashSet::new();
    operations.retain(|op| seen.insert(op.clone()));

    operations
}

/// Extract affected graphs from update
pub fn extract_affected_graphs(update: &str) -> Vec<String> {
    let mut graphs = Vec::new();

    // Look for GRAPH <uri> patterns
    let graph_re = regex::Regex::new(r"(?i)GRAPH\s*<([^>]+)>").expect("valid regex pattern");
    for cap in graph_re.captures_iter(update) {
        if let Some(g) = cap.get(1) {
            let graph_uri = g.as_str().to_string();
            if !graphs.contains(&graph_uri) {
                graphs.push(graph_uri);
            }
        }
    }

    // Check for DEFAULT keyword
    if update.to_uppercase().contains("DEFAULT") && !graphs.contains(&"default".to_string()) {
        graphs.push("default".to_string());
    }

    // Check for ALL keyword
    if update.to_uppercase().contains(" ALL") && !graphs.contains(&"all".to_string()) {
        graphs.push("all".to_string());
    }

    graphs
}

/// Extract error location from error message
pub fn extract_error_location(error: &str) -> (Option<usize>, Option<usize>) {
    // Try to extract line and column from error message
    let line_re = regex::Regex::new(r"line\s*(\d+)").expect("valid regex pattern");
    let col_re = regex::Regex::new(r"col(?:umn)?\s*(\d+)").expect("valid regex pattern");

    let line = line_re
        .captures(error)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse().ok());
    let column = col_re
        .captures(error)
        .and_then(|c| c.get(1))
        .and_then(|m| m.as_str().parse().ok());

    (line, column)
}

/// Normalize RDF format name
pub fn normalize_format(format: &str) -> String {
    match format.to_lowercase().as_str() {
        "turtle" | "ttl" | "text/turtle" => "turtle".to_string(),
        "ntriples" | "nt" | "application/n-triples" => "ntriples".to_string(),
        "nquads" | "nq" | "application/n-quads" => "nquads".to_string(),
        "rdfxml" | "rdf/xml" | "application/rdf+xml" => "rdfxml".to_string(),
        "jsonld" | "json-ld" | "application/ld+json" => "jsonld".to_string(),
        "trig" | "application/trig" => "trig".to_string(),
        _ => format.to_lowercase(),
    }
}

/// RDF parse result
pub struct RdfParseResult {
    pub triple_count: usize,
    pub graph_count: Option<usize>,
    pub subject_count: usize,
    pub predicate_count: usize,
    pub object_count: usize,
    pub blank_node_count: usize,
    pub literal_count: usize,
    pub sample_triples: Vec<String>,
}

/// Parse RDF data and collect statistics
pub fn parse_rdf_data(data: &str, format: &str, _base_iri: &str) -> Result<RdfParseResult, String> {
    use std::collections::HashSet;

    let mut subjects = HashSet::new();
    let mut predicates = HashSet::new();
    let mut objects = HashSet::new();
    let mut blank_nodes = HashSet::new();
    let mut literal_count = 0;
    let mut sample_triples = Vec::new();

    // Parse based on format using oxirs-core parsers
    let triples = match format {
        "turtle" | "ttl" => oxirs_core::format::turtle::TurtleParser::new()
            .parse_str(data)
            .map_err(|e| format!("Turtle parse error: {}", e))?,
        "ntriples" | "nt" => oxirs_core::format::ntriples::NTriplesParser::new()
            .parse_str(data)
            .map_err(|e| format!("N-Triples parse error: {}", e))?,
        "rdfxml" | "rdf/xml" => oxirs_core::format::rdfxml::RdfXmlParser::new()
            .parse_str(data)
            .map_err(|e| format!("RDF/XML parse error: {}", e))?,
        _ => {
            return Err(format!("Unsupported format: {}", format));
        }
    };

    let triple_count = triples.len();

    for triple in &triples {
        // Track subjects
        let subject_str = format!("{:?}", triple.subject());
        if subject_str.contains("BlankNode") {
            blank_nodes.insert(subject_str.clone());
        }
        subjects.insert(subject_str);

        // Track predicates
        predicates.insert(format!("{:?}", triple.predicate()));

        // Track objects
        let object_str = format!("{:?}", triple.object());
        if object_str.contains("BlankNode") {
            blank_nodes.insert(object_str.clone());
        } else if object_str.contains("Literal") {
            literal_count += 1;
        }
        objects.insert(object_str);

        // Sample triples
        if sample_triples.len() < 10 {
            sample_triples.push(format!(
                "{:?} {:?} {:?}",
                triple.subject(),
                triple.predicate(),
                triple.object()
            ));
        }
    }

    Ok(RdfParseResult {
        triple_count,
        graph_count: None, // Only for quads
        subject_count: subjects.len(),
        predicate_count: predicates.len(),
        object_count: objects.len(),
        blank_node_count: blank_nodes.len(),
        literal_count,
        sample_triples,
    })
}

/// Parsed language tag
pub struct ParsedLanguageTag {
    pub language: Option<String>,
    pub script: Option<String>,
    pub region: Option<String>,
    pub variants: Vec<String>,
    pub extensions: Vec<String>,
    pub private_use: Option<String>,
}

/// Parse BCP 47 language tag
pub fn parse_language_tag(tag: &str) -> Result<ParsedLanguageTag, String> {
    let parts: Vec<&str> = tag.split('-').collect();

    if parts.is_empty() {
        return Err("Empty language tag".to_string());
    }

    let mut language = None;
    let mut script = None;
    let mut region = None;
    let mut variants = Vec::new();
    let mut extensions = Vec::new();
    let mut private_use = None;

    let mut i = 0;

    // Check for private use tag
    if parts[0].to_lowercase() == "x" {
        if parts.len() > 1 {
            private_use = Some(parts[1..].join("-"));
        }
        return Ok(ParsedLanguageTag {
            language,
            script,
            region,
            variants,
            extensions,
            private_use,
        });
    }

    // Primary language subtag (2-3 letters or 4 letters for reserved)
    if parts[i].len() >= 2
        && parts[i].len() <= 3
        && parts[i].chars().all(|c| c.is_ascii_alphabetic())
    {
        language = Some(parts[i].to_lowercase());
        i += 1;
    } else if parts[i].len() == 4 && parts[i].chars().all(|c| c.is_ascii_alphabetic()) {
        // Reserved for future use
        language = Some(parts[i].to_lowercase());
        i += 1;
    } else {
        return Err(format!("Invalid primary language subtag: {}", parts[0]));
    }

    // Extended language subtags (3 letters each, up to 3)
    while i < parts.len()
        && parts[i].len() == 3
        && parts[i].chars().all(|c| c.is_ascii_alphabetic())
    {
        // Extended language subtags are appended to language
        if let Some(ref mut lang) = language {
            *lang = format!("{}-{}", lang, parts[i].to_lowercase());
        }
        i += 1;
        if i >= 4 {
            break; // Max 3 extended subtags
        }
    }

    // Script subtag (4 letters)
    if i < parts.len() && parts[i].len() == 4 && parts[i].chars().all(|c| c.is_ascii_alphabetic()) {
        let s = parts[i];
        // Capitalize first letter, lowercase rest
        script = Some(format!(
            "{}{}",
            s.chars()
                .next()
                .expect("script subtag should have at least one char")
                .to_uppercase(),
            s[1..].to_lowercase()
        ));
        i += 1;
    }

    // Region subtag (2 letters or 3 digits)
    if i < parts.len() {
        let p = parts[i];
        if (p.len() == 2 && p.chars().all(|c| c.is_ascii_alphabetic()))
            || (p.len() == 3 && p.chars().all(|c| c.is_ascii_digit()))
        {
            region = Some(p.to_uppercase());
            i += 1;
        }
    }

    // Variant subtags (5-8 alphanum or 4 starting with digit)
    while i < parts.len() {
        let p = parts[i];
        if (p.len() >= 5 && p.len() <= 8 && p.chars().all(|c| c.is_ascii_alphanumeric()))
            || (p.len() == 4
                && p.chars().next().is_some_and(|c| c.is_ascii_digit())
                && p.chars().all(|c| c.is_ascii_alphanumeric()))
        {
            variants.push(p.to_lowercase());
            i += 1;
        } else {
            break;
        }
    }

    // Extension subtags (singleton followed by 2-8 alphanum)
    while i < parts.len() {
        let p = parts[i];
        if p.len() == 1 && p.chars().all(|c| c.is_ascii_alphanumeric()) && p.to_lowercase() != "x" {
            let singleton = p.to_lowercase();
            let mut ext_parts = vec![singleton];
            i += 1;
            while i < parts.len() {
                let ep = parts[i];
                if ep.len() >= 2 && ep.len() <= 8 && ep.chars().all(|c| c.is_ascii_alphanumeric()) {
                    ext_parts.push(ep.to_lowercase());
                    i += 1;
                } else {
                    break;
                }
            }
            if ext_parts.len() > 1 {
                extensions.push(ext_parts.join("-"));
            }
        } else {
            break;
        }
    }

    // Private use subtag
    if i < parts.len() && parts[i].to_lowercase() == "x" {
        i += 1;
        if i < parts.len() {
            private_use = Some(parts[i..].join("-"));
        }
    }

    Ok(ParsedLanguageTag {
        language,
        script,
        region,
        variants,
        extensions,
        private_use,
    })
}

/// Check if a language subtag is deprecated
pub fn is_deprecated_language(lang: &str) -> bool {
    // List of deprecated language subtags (partial)
    let deprecated = [
        "iw",
        "ji",
        "in",
        "no-bok",
        "no-nyn",
        "sgn-be-fr",
        "sgn-be-nl",
        "sgn-ch-de",
    ];
    deprecated.contains(&lang.to_lowercase().as_str())
}
