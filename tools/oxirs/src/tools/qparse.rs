//! SPARQL Query Parse tool
//!
//! Parse a SPARQL query and optionally print the parsed AST and/or algebra.

use super::{utils, ToolResult};
use oxirs_arq::query::parse_query;
use std::fs;

/// Run qparse command — parse and optionally print SPARQL query AST/algebra.
pub async fn run(
    _query: String,
    _file: bool,
    _print_ast: bool,
    _print_algebra: bool,
) -> ToolResult {
    let query_string = if _file {
        let path = std::path::Path::new(&_query);
        utils::check_file_readable(path)?;
        fs::read_to_string(path)?
    } else {
        _query.clone()
    };

    println!("--- SPARQL Query ---");
    println!("{query_string}");
    println!();

    let parsed = parse_query(&query_string).map_err(|e| format!("Parse error: {e}"))?;

    println!("Query type : {:?}", parsed.query_type);

    if !parsed.prefixes.is_empty() {
        println!("Prefixes   :");
        for (prefix, iri) in &parsed.prefixes {
            println!("  {prefix}: <{iri}>");
        }
    }

    if !parsed.select_variables.is_empty() {
        // The parsed Variable Display impl already includes the leading
        // sigil (e.g. "?s" or "$s"); guarding against a double-prefix
        // ensures the printed list matches Jena's `qparse` output.
        let vars: Vec<String> = parsed
            .select_variables
            .iter()
            .map(|v| {
                let rendered = format!("{v}");
                if rendered.starts_with('?') || rendered.starts_with('$') {
                    rendered
                } else {
                    format!("?{rendered}")
                }
            })
            .collect();
        println!("Variables  : {}", vars.join(", "));
    }

    if let Some(limit) = parsed.limit {
        println!("LIMIT      : {limit}");
    }

    if let Some(offset) = parsed.offset {
        println!("OFFSET     : {offset}");
    }

    if parsed.distinct {
        println!("DISTINCT   : true");
    }

    if _print_ast {
        println!();
        println!("--- Parsed AST ---");
        println!("{parsed:#?}");
    }

    if _print_algebra {
        println!();
        println!("--- WHERE Algebra ---");
        println!("{:#?}", parsed.where_clause);
    }

    println!();
    println!("Parse OK");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[tokio::test]
    async fn test_run_simple_select() {
        let query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10".to_string();
        let result = run(query, false, false, false).await;
        assert!(result.is_ok(), "simple select should parse OK: {result:?}");
    }

    #[tokio::test]
    async fn test_run_with_ast() {
        let query = "SELECT ?s WHERE { ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Class> }".to_string();
        // _file=false: query string is the inline SPARQL text
        let result = run(query, false, true, false).await;
        assert!(result.is_ok(), "should print AST: {result:?}");
    }

    #[tokio::test]
    async fn test_run_with_algebra() {
        let query = "ASK WHERE { ?s ?p ?o }".to_string();
        // _file=false: query string is the inline SPARQL text
        let result = run(query, false, false, true).await;
        assert!(result.is_ok(), "should print algebra: {result:?}");
    }

    #[tokio::test]
    async fn test_run_invalid_query() {
        let query = "this is not sparql".to_string();
        let result = run(query, false, false, false).await;
        assert!(result.is_err(), "invalid query should fail");
    }

    #[tokio::test]
    async fn test_run_from_file() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        write!(tmp, "SELECT ?x WHERE {{ ?x ?y ?z }}").expect("write");
        let path = tmp.path().to_string_lossy().to_string();
        let result = run(path, true, false, false).await;
        assert!(
            result.is_ok(),
            "file-based parse should succeed: {result:?}"
        );
    }

    #[tokio::test]
    async fn test_run_missing_file() {
        let result = run("/nonexistent/query.sparql".to_string(), true, false, false).await;
        assert!(result.is_err());
    }
}
