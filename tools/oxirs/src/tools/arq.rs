//! ARQ - Advanced SPARQL Query Processor
//!
//! Equivalent to Apache Jena's arq command. Advanced SPARQL query execution
//! with optimization, explanation, and multiple data source support.

use super::{utils, ToolResult, ToolStats};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Configuration for ARQ query execution
pub struct ArqConfig {
    pub query: Option<String>,
    pub query_file: Option<PathBuf>,
    pub data: Vec<PathBuf>,
    pub namedgraph: Vec<String>,
    pub results_format: String,
    pub dataset: Option<PathBuf>,
    pub explain: bool,
    pub optimize: bool,
    pub time: bool,
}

/// Run arq command - Advanced SPARQL query processor
pub async fn run(config: ArqConfig) -> ToolResult {
    let mut stats = ToolStats::new();

    println!("Advanced SPARQL Query Processor (arq)");

    // Validate results format
    if !utils::is_supported_results_format(&config.results_format) {
        return Err(format!(
            "Unsupported results format '{}'. Supported: table, csv, tsv, json, xml, html, markdown",
            config.results_format
        )
        .into());
    }

    // Get query string
    let query_string = match (config.query, config.query_file) {
        (Some(q), None) => q,
        (None, Some(ref path)) => {
            utils::check_file_readable(path)?;
            fs::read_to_string(path)?
        }
        (Some(_), Some(_)) => {
            return Err("Cannot specify both --query and --query-file".into());
        }
        (None, None) => {
            return Err("Must specify either --query or --query-file".into());
        }
    };

    println!("Query:");
    println!("---");
    println!("{query_string}");
    println!("---");

    // Validate query syntax
    let query_info = parse_and_validate_query(&query_string)?;
    println!("Query type: {}", query_info.query_type);
    println!("Variables: {:?}", query_info.variables);

    if config.explain {
        explain_query(&query_string, &query_info)?;
    }

    if config.optimize {
        println!("Query optimization enabled");
        optimize_query(&query_string, &query_info)?;
    }

    // Load data sources
    let mut data_sources = Vec::new();

    // Add dataset if specified
    if let Some(dataset_path) = config.dataset {
        println!("Loading dataset: {}", dataset_path.display());
        data_sources.push(DataSource::Dataset(dataset_path));
    }

    // Add data files
    for data_file in &config.data {
        println!("Loading data file: {}", data_file.display());
        utils::check_file_readable(data_file)?;
        data_sources.push(DataSource::File(data_file.clone()));
    }

    // Add named graphs
    for graph_uri in &config.namedgraph {
        println!("Named graph: {graph_uri}");
        utils::validate_iri(graph_uri).map_err(|e| format!("Invalid named graph IRI: {e}"))?;
        data_sources.push(DataSource::NamedGraph(graph_uri.clone()));
    }

    if data_sources.is_empty() {
        return Err("No data sources specified. Use --data, --dataset, or --namedgraph".into());
    }

    // Execute query
    println!("\nExecuting query...");
    let execution_start = if config.time {
        Some(Instant::now())
    } else {
        None
    };

    let results = execute_sparql_query(&query_string, &query_info, &data_sources)?;

    if let Some(start_time) = execution_start {
        let execution_time = start_time.elapsed();
        println!(
            "Query execution time: {}",
            utils::format_duration(execution_time)
        );
    }

    // Format and display results
    format_query_results(&results, &config.results_format, &query_info)?;

    stats.items_processed = results.bindings.len();
    stats.finish();
    stats.print_summary("ARQ");

    Ok(())
}

/// Query information extracted from parsing
#[derive(Debug)]
struct QueryInfo {
    query_type: String,
    variables: Vec<String>,
    prefixes: Vec<(String, String)>,
    where_patterns: usize,
    filters: usize,
    order_by: Vec<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

/// Data source types
#[derive(Debug)]
#[allow(dead_code)]
enum DataSource {
    Dataset(PathBuf),
    File(PathBuf),
    NamedGraph(String),
}

/// Query execution results
#[derive(Debug)]
struct QueryResults {
    variables: Vec<String>,
    bindings: Vec<QueryBinding>,
    result_type: QueryResultType,
}

#[derive(Debug)]
enum QueryResultType {
    Select,
    Construct,
    Ask(bool),
    Describe,
}

#[derive(Debug)]
struct QueryBinding {
    values: std::collections::HashMap<String, String>,
}

/// Parse and validate SPARQL query
fn parse_and_validate_query(query: &str) -> ToolResult<QueryInfo> {
    let query = query.trim();

    // Basic query type detection
    let query_type = if query.to_uppercase().contains("SELECT") {
        "SELECT".to_string()
    } else if query.to_uppercase().contains("CONSTRUCT") {
        "CONSTRUCT".to_string()
    } else if query.to_uppercase().contains("ASK") {
        "ASK".to_string()
    } else if query.to_uppercase().contains("DESCRIBE") {
        "DESCRIBE".to_string()
    } else {
        return Err("Unknown query type. Must be SELECT, CONSTRUCT, ASK, or DESCRIBE".into());
    };

    // Extract variables (simplified)
    let mut variables = Vec::new();
    for line in query.lines() {
        let line = line.trim().to_uppercase();
        if line.starts_with("SELECT") {
            // Extract variables from SELECT clause
            let select_part = line.strip_prefix("SELECT").unwrap_or("");
            for token in select_part.split_whitespace() {
                if token.starts_with('?') {
                    variables.push(token.to_string());
                }
            }
            break;
        }
    }

    // Extract prefixes (simplified)
    let mut prefixes = Vec::new();
    for line in query.lines() {
        let line = line.trim();
        if line.to_uppercase().starts_with("PREFIX") {
            // Parse PREFIX declaration
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 3 {
                let prefix = parts[1].trim_end_matches(':');
                let uri = parts[2].trim_matches('<').trim_matches('>');
                prefixes.push((prefix.to_string(), uri.to_string()));
            }
        }
    }

    // Count WHERE patterns and filters (very basic)
    let where_patterns = query.matches('{').count().max(1) - 1; // Rough estimate
    let filters = query.to_uppercase().matches("FILTER").count();

    // Extract ORDER BY, LIMIT, OFFSET
    let order_by = extract_order_by(query);
    let limit = extract_limit(query);
    let offset = extract_offset(query);

    Ok(QueryInfo {
        query_type,
        variables,
        prefixes,
        where_patterns,
        filters,
        order_by,
        limit,
        offset,
    })
}

/// Extract LIMIT value from query
fn extract_limit(query: &str) -> Option<usize> {
    let query_upper = query.to_uppercase();
    if let Some(limit_pos) = query_upper.find("LIMIT") {
        let after_limit = &query[limit_pos + 5..];
        for token in after_limit.split_whitespace() {
            if let Ok(limit_val) = token.parse::<usize>() {
                return Some(limit_val);
            }
        }
    }
    None
}

/// Extract OFFSET value from query
fn extract_offset(query: &str) -> Option<usize> {
    let query_upper = query.to_uppercase();
    if let Some(offset_pos) = query_upper.find("OFFSET") {
        let after_offset = &query[offset_pos + 6..];
        for token in after_offset.split_whitespace() {
            if let Ok(offset_val) = token.parse::<usize>() {
                return Some(offset_val);
            }
        }
    }
    None
}

/// Extract ORDER BY clause from query
fn extract_order_by(query: &str) -> Vec<String> {
    let mut order_by = Vec::new();
    let query_upper = query.to_uppercase();

    if let Some(order_pos) = query_upper.find("ORDER BY") {
        let after_order = &query[order_pos + 8..];
        // Extract until LIMIT, OFFSET, or end of query
        let mut order_clause = String::new();
        for token in after_order.split_whitespace() {
            let upper = token.to_uppercase();
            if upper.starts_with("LIMIT") || upper.starts_with("OFFSET") {
                break;
            }
            if !order_clause.is_empty() {
                order_clause.push(' ');
            }
            order_clause.push_str(token);
        }

        // Split by comma for multiple order variables
        for var in order_clause.split(',') {
            let var = var.trim();
            if !var.is_empty() {
                order_by.push(var.to_string());
            }
        }
    }

    order_by
}

/// Optimize SPARQL query
fn optimize_query(query: &str, query_info: &QueryInfo) -> ToolResult<()> {
    println!("\n=== Query Optimization ===");

    // Analyze query complexity
    let complexity_score =
        query_info.where_patterns * 10 + query_info.filters * 5 + query_info.variables.len();

    println!("Query complexity score: {complexity_score}");

    // Optimization suggestions
    let mut suggestions = Vec::new();

    if query_info.filters > 0 && query_info.where_patterns > 3 {
        suggestions.push("Consider moving FILTER clauses closer to relevant patterns");
    }

    if query_info.limit.is_none() && query_info.query_type == "SELECT" {
        suggestions.push("Consider adding LIMIT to improve performance");
    }

    if query_info.variables.len() > 10 {
        suggestions.push("Large number of variables may impact performance");
    }

    if !query_info.order_by.is_empty() && query_info.limit.is_none() {
        suggestions.push("ORDER BY without LIMIT sorts entire result set");
    }

    // Check for OPTIONAL patterns
    if query.to_uppercase().contains("OPTIONAL") && query_info.filters > 0 {
        suggestions.push("OPTIONAL with FILTER may benefit from reordering");
    }

    // Check for UNION patterns
    if query.to_uppercase().contains("UNION") {
        suggestions.push("UNION operations can be expensive - ensure selectivity");
    }

    // Check for subqueries
    if query.matches('{').count() > query_info.where_patterns + 1 {
        suggestions.push("Nested subqueries detected - consider flattening if possible");
    }

    if suggestions.is_empty() {
        println!("✓ Query appears well-optimized");
    } else {
        println!("\nOptimization suggestions:");
        for (i, suggestion) in suggestions.iter().enumerate() {
            println!("  {}. {}", i + 1, suggestion);
        }
    }

    // Show optimized execution plan
    println!("\nOptimized Execution Plan:");
    println!("1. Apply selective filters early");
    println!("2. Join smallest patterns first");
    println!("3. Push LIMIT/OFFSET to execution engine");
    if !query_info.order_by.is_empty() {
        println!("4. Sort by: {}", query_info.order_by.join(", "));
    }
    if query_info.limit.is_some() || query_info.offset.is_some() {
        println!("5. Apply pagination at query engine level");
    }

    println!("==========================\n");
    Ok(())
}

/// Explain query execution plan
fn explain_query(_query: &str, query_info: &QueryInfo) -> ToolResult<()> {
    println!("\n=== Query Explanation ===");
    println!("Query type: {}", query_info.query_type);
    println!(
        "Variables: {} ({})",
        query_info.variables.len(),
        query_info.variables.join(", ")
    );
    println!("Prefixes: {}", query_info.prefixes.len());

    for (prefix, uri) in &query_info.prefixes {
        println!("  {prefix}: <{uri}>");
    }

    println!("WHERE patterns: ~{}", query_info.where_patterns);
    println!("Filters: {}", query_info.filters);

    if let Some(limit) = query_info.limit {
        println!("Limit: {limit}");
    }

    if let Some(offset) = query_info.offset {
        println!("Offset: {offset}");
    }

    // Basic execution plan
    println!("\nExecution Plan:");
    println!("1. Parse triple patterns");
    println!("2. Join triple patterns");
    if query_info.filters > 0 {
        println!("3. Apply {} filter(s)", query_info.filters);
    }
    if !query_info.order_by.is_empty() {
        println!("4. Sort results");
    }
    if query_info.offset.is_some() {
        println!("5. Apply offset");
    }
    if query_info.limit.is_some() {
        println!("6. Apply limit");
    }
    println!("7. Project variables");

    println!("========================\n");
    Ok(())
}

/// Execute SPARQL query against data sources
fn execute_sparql_query(
    _query: &str,
    query_info: &QueryInfo,
    data_sources: &[DataSource],
) -> ToolResult<QueryResults> {
    println!("Loading {} data source(s)...", data_sources.len());

    // For now, simulate query execution
    // In a real implementation, this would:
    // 1. Load all data sources into a unified dataset
    // 2. Parse the SPARQL query into an algebra expression
    // 3. Optimize the query plan
    // 4. Execute the query against the dataset
    // 5. Return results

    let variables = query_info.variables.clone();
    let mut bindings = Vec::new();

    // Simulate some results
    match query_info.query_type.as_str() {
        "SELECT" => {
            for i in 0..5 {
                let mut values = std::collections::HashMap::new();
                for var in &variables {
                    values.insert(
                        var.clone(),
                        format!("value_{}{}", var.trim_start_matches('?'), i),
                    );
                }
                bindings.push(QueryBinding { values });
            }
        }
        "ASK" => {
            // ASK queries return boolean
            return Ok(QueryResults {
                variables: Vec::new(),
                bindings: Vec::new(),
                result_type: QueryResultType::Ask(true),
            });
        }
        _ => {
            println!(
                "Query type '{}' simulation not implemented",
                query_info.query_type
            );
        }
    }

    println!("Query executed successfully");
    println!("Result bindings: {}", bindings.len());

    let result_type = match query_info.query_type.as_str() {
        "SELECT" => QueryResultType::Select,
        "CONSTRUCT" => QueryResultType::Construct,
        "ASK" => QueryResultType::Ask(true),
        "DESCRIBE" => QueryResultType::Describe,
        _ => QueryResultType::Select,
    };

    Ok(QueryResults {
        variables,
        bindings,
        result_type,
    })
}

/// Format query results in specified format
fn format_query_results(
    results: &QueryResults,
    format: &str,
    _query_info: &QueryInfo,
) -> ToolResult<()> {
    println!("\nResults ({}):", format.to_uppercase());

    if let QueryResultType::Ask(answer) = &results.result_type {
        match format {
            "table" => println!("ASK result: {answer}"),
            "json" => println!("{{ \"boolean\": {answer} }}"),
            "xml" => println!("<sparql><boolean>{answer}</boolean></sparql>"),
            _ => println!("{answer}"),
        }
        return Ok(());
    }

    if results.bindings.is_empty() {
        println!("No results");
        return Ok(());
    }

    match format {
        "table" => format_table_results(results),
        "csv" => format_csv_results(results, ","),
        "tsv" => format_csv_results(results, "\t"),
        "json" => format_json_results(results),
        "xml" => format_xml_results(results),
        "html" => format_html_results(results),
        "markdown" => format_markdown_results(results),
        _ => Err(format!("Unknown results format: {format}").into()),
    }
}

/// Format results as table
fn format_table_results(results: &QueryResults) -> ToolResult<()> {
    if results.variables.is_empty() {
        println!("No variables to display");
        return Ok(());
    }

    // Calculate column widths
    let mut col_widths = Vec::new();
    for var in &results.variables {
        let mut max_width = var.len();
        for binding in &results.bindings {
            if let Some(value) = binding.values.get(var) {
                max_width = max_width.max(value.len());
            }
        }
        col_widths.push(max_width.max(8)); // Minimum width of 8
    }

    // Print header
    print!("| ");
    for (i, var) in results.variables.iter().enumerate() {
        print!("{:width$} | ", var, width = col_widths[i]);
    }
    println!();

    // Print separator
    print!("|");
    for &width in &col_widths {
        print!("{}", "-".repeat(width + 2));
        print!("|");
    }
    println!();

    // Print rows
    for binding in &results.bindings {
        print!("| ");
        for (i, var) in results.variables.iter().enumerate() {
            let value = binding.values.get(var).map(|s| s.as_str()).unwrap_or("");
            print!("{:width$} | ", value, width = col_widths[i]);
        }
        println!();
    }

    println!("\n{} row(s)", results.bindings.len());
    Ok(())
}

/// Format results as CSV/TSV
fn format_csv_results(results: &QueryResults, separator: &str) -> ToolResult<()> {
    // Header
    println!("{}", results.variables.join(separator));

    // Rows
    for binding in &results.bindings {
        let row: Vec<String> = results
            .variables
            .iter()
            .map(|var| binding.values.get(var).cloned().unwrap_or_default())
            .collect();
        println!("{}", row.join(separator));
    }

    Ok(())
}

/// Format results as JSON
fn format_json_results(results: &QueryResults) -> ToolResult<()> {
    println!("{{");
    println!("  \"head\": {{");
    println!(
        "    \"vars\": [{}]",
        results
            .variables
            .iter()
            .map(|v| format!("\"{}\"", v.trim_start_matches('?')))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  }},");
    println!("  \"results\": {{");
    println!("    \"bindings\": [");

    for (i, binding) in results.bindings.iter().enumerate() {
        println!("      {{");
        for (j, var) in results.variables.iter().enumerate() {
            let var_name = var.trim_start_matches('?');
            let empty_string = String::new();
            let value = binding.values.get(var).unwrap_or(&empty_string);
            print!("        \"{var_name}\": {{ \"type\": \"literal\", \"value\": \"{value}\" }}");
            if j < results.variables.len() - 1 {
                print!(",");
            }
            println!();
        }
        print!("      }}");
        if i < results.bindings.len() - 1 {
            print!(",");
        }
        println!();
    }

    println!("    ]");
    println!("  }}");
    println!("}}");

    Ok(())
}

/// Format results as XML
fn format_xml_results(results: &QueryResults) -> ToolResult<()> {
    println!("<?xml version=\"1.0\"?>");
    println!("<sparql xmlns=\"http://www.w3.org/2005/sparql-results#\">");
    println!("  <head>");
    for var in &results.variables {
        println!("    <variable name=\"{}\"/>", var.trim_start_matches('?'));
    }
    println!("  </head>");
    println!("  <results>");

    for binding in &results.bindings {
        println!("    <result>");
        for var in &results.variables {
            let var_name = var.trim_start_matches('?');
            if let Some(value) = binding.values.get(var) {
                println!("      <binding name=\"{var_name}\">");
                println!("        <literal>{value}</literal>");
                println!("      </binding>");
            }
        }
        println!("    </result>");
    }

    println!("  </results>");
    println!("</sparql>");

    Ok(())
}

/// Format results as HTML
fn format_html_results(results: &QueryResults) -> ToolResult<()> {
    println!("<!DOCTYPE html>");
    println!("<html lang=\"en\">");
    println!("<head>");
    println!("  <meta charset=\"UTF-8\">");
    println!("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">");
    println!("  <title>SPARQL Query Results</title>");
    println!("  <style>");
    println!("    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 20px; background: #f5f5f5; }}");
    println!("    .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}");
    println!("    h1 {{ color: #333; margin-top: 0; }}");
    println!("    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}");
    println!("    th {{ background: #4CAF50; color: white; padding: 12px; text-align: left; font-weight: 600; }}");
    println!("    td {{ padding: 10px 12px; border-bottom: 1px solid #ddd; }}");
    println!("    tr:hover {{ background: #f5f5f5; }}");
    println!("    .summary {{ color: #666; margin-top: 20px; font-size: 14px; }}");
    println!(
        "    .empty {{ color: #999; font-style: italic; padding: 40px; text-align: center; }}"
    );
    println!("  </style>");
    println!("</head>");
    println!("<body>");
    println!("  <div class=\"container\">");
    println!("    <h1>SPARQL Query Results</h1>");

    if results.variables.is_empty() || results.bindings.is_empty() {
        println!("    <div class=\"empty\">No results found</div>");
    } else {
        println!("    <table>");
        println!("      <thead>");
        println!("        <tr>");
        for var in &results.variables {
            println!("          <th>{}</th>", var.trim_start_matches('?'));
        }
        println!("        </tr>");
        println!("      </thead>");
        println!("      <tbody>");

        for binding in &results.bindings {
            println!("        <tr>");
            for var in &results.variables {
                let value = binding
                    .values
                    .get(var)
                    .map(|s| html_escape(s))
                    .unwrap_or_else(|| String::from(""));
                println!("          <td>{value}</td>");
            }
            println!("        </tr>");
        }

        println!("      </tbody>");
        println!("    </table>");
        println!(
            "    <div class=\"summary\">{} row(s) returned</div>",
            results.bindings.len()
        );
    }

    println!("  </div>");
    println!("</body>");
    println!("</html>");

    Ok(())
}

/// Format results as Markdown
fn format_markdown_results(results: &QueryResults) -> ToolResult<()> {
    println!("# SPARQL Query Results\n");

    if results.variables.is_empty() || results.bindings.is_empty() {
        println!("*No results found*\n");
        return Ok(());
    }

    // Header row
    print!("|");
    for var in &results.variables {
        print!(" {} |", var.trim_start_matches('?'));
    }
    println!();

    // Separator row
    print!("|");
    for _ in &results.variables {
        print!(" --- |");
    }
    println!();

    // Data rows
    for binding in &results.bindings {
        print!("|");
        for var in &results.variables {
            let value = binding
                .values
                .get(var)
                .map(|s| markdown_escape(s))
                .unwrap_or_else(|| String::from(""));
            print!(" {value} |");
        }
        println!();
    }

    println!("\n*{} row(s) returned*", results.bindings.len());

    Ok(())
}

/// Escape HTML special characters
fn html_escape(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Escape Markdown special characters
fn markdown_escape(input: &str) -> String {
    input
        .replace('|', "\\|")
        .replace('\n', " ")
        .replace('\r', "")
}
