//! Remote SPARQL tool
//!
//! Send SPARQL SELECT (and other query types) via HTTP to a remote SPARQL endpoint
//! using the SPARQL 1.1 Protocol, then print results in the requested format.

use super::ToolResult;
use std::path::PathBuf;
use std::time::Duration;

/// Accepted result MIME types for SPARQL queries.
const ACCEPT_SPARQL_JSON: &str = "application/sparql-results+json";
const ACCEPT_SPARQL_XML: &str = "application/sparql-results+xml";
const ACCEPT_CSV: &str = "text/csv";
const ACCEPT_TSV: &str = "text/tab-separated-values";

/// Run rsparql command — execute a SPARQL query against a remote endpoint.
pub async fn run(
    _service: String,
    _query: Option<String>,
    _query_file: Option<PathBuf>,
    _results: String,
    _timeout: u64,
) -> ToolResult {
    let query_string = resolve_query_string(_query, _query_file)?;
    let accept = results_format_to_accept(&_results);

    println!("Endpoint : {_service}");
    println!("Format   : {_results} ({accept})");
    println!("Query    :");
    println!("{query_string}");
    println!();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(_timeout))
        .build()
        .map_err(|e| format!("Failed to build HTTP client: {e}"))?;

    // Use POST with application/sparql-query body per SPARQL Protocol §2.1.2.
    let response = client
        .post(&_service)
        .header("Content-Type", "application/sparql-query")
        .header("Accept", accept)
        .body(query_string)
        .send()
        .await
        .map_err(|e| format!("HTTP request failed: {e}"))?;

    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();

    println!("HTTP     : {}", status.as_u16());
    println!("Content-Type: {content_type}");
    println!();

    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(format!("SPARQL query failed: HTTP {} — {body}", status.as_u16()).into());
    }

    let body = response
        .text()
        .await
        .map_err(|e| format!("Failed to read response body: {e}"))?;

    // Print result body.  For tabular formats (json/csv/tsv) we try to render
    // a human-readable table; for anything else we just dump the raw body.
    match _results.to_lowercase().as_str() {
        "json" | "srj" => print_json_results(&body),
        "csv" => print_csv_results(&body),
        "tsv" => print_tsv_results(&body),
        _ => {
            // Raw dump (xml / unknown)
            println!("{body}");
        }
    }

    Ok(())
}

/// Resolve the query string from an inline value or a file path.
fn resolve_query_string(query: Option<String>, query_file: Option<PathBuf>) -> ToolResult<String> {
    match (query, query_file) {
        (Some(q), None) => Ok(q),
        (None, Some(path)) => {
            if !path.exists() {
                return Err(format!("Query file not found: {}", path.display()).into());
            }
            std::fs::read_to_string(&path)
                .map_err(|e| format!("Cannot read query file {}: {e}", path.display()).into())
        }
        (Some(_), Some(_)) => Err("Specify either --query or --query-file, not both".into()),
        (None, None) => Err("No query provided. Use --query or --query-file".into()),
    }
}

/// Map a user-friendly results format name to an HTTP Accept header value.
fn results_format_to_accept(format: &str) -> &'static str {
    match format.to_lowercase().as_str() {
        "json" | "srj" => ACCEPT_SPARQL_JSON,
        "xml" | "srx" => ACCEPT_SPARQL_XML,
        "csv" => ACCEPT_CSV,
        "tsv" => ACCEPT_TSV,
        _ => ACCEPT_SPARQL_JSON,
    }
}

/// Print SPARQL Results JSON in a human-readable table form.
fn print_json_results(body: &str) {
    let json: serde_json::Value = match serde_json::from_str(body) {
        Ok(j) => j,
        Err(_) => {
            println!("{body}");
            return;
        }
    };

    // Boolean result (ASK)?
    if let Some(b) = json.get("boolean").and_then(serde_json::Value::as_bool) {
        println!("Result: {b}");
        return;
    }

    // Tabular result (SELECT)?
    let vars = match json
        .pointer("/head/vars")
        .and_then(serde_json::Value::as_array)
    {
        Some(v) => v.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>(),
        None => {
            println!("{body}");
            return;
        }
    };

    let bindings = match json
        .pointer("/results/bindings")
        .and_then(serde_json::Value::as_array)
    {
        Some(b) => b,
        None => {
            println!("{body}");
            return;
        }
    };

    // Header
    println!("| {} |", vars.join(" | "));
    println!(
        "|{}|",
        vars.iter()
            .map(|v| "-".repeat(v.len() + 2))
            .collect::<Vec<_>>()
            .join("|")
    );

    for row in bindings {
        let cells: Vec<String> = vars
            .iter()
            .map(|var| {
                row.get(var)
                    .and_then(|cell| cell.get("value"))
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("")
                    .to_string()
            })
            .collect();
        println!("| {} |", cells.join(" | "));
    }

    println!();
    println!("{} result(s)", bindings.len());
}

/// Print CSV results.
fn print_csv_results(body: &str) {
    println!("{body}");
}

/// Print TSV results.
fn print_tsv_results(body: &str) {
    println!("{body}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_resolve_query_inline() {
        let q = resolve_query_string(Some("SELECT ?s WHERE { ?s ?p ?o }".to_string()), None)
            .expect("inline query");
        assert!(q.contains("SELECT"));
    }

    #[test]
    fn test_resolve_query_file() {
        let mut tmp = tempfile::NamedTempFile::new().expect("tempfile");
        write!(tmp, "ASK {{ ?s ?p ?o }}").expect("write");
        let q = resolve_query_string(None, Some(tmp.path().to_path_buf())).expect("file query");
        assert!(q.contains("ASK"));
    }

    #[test]
    fn test_resolve_both_is_error() {
        let result = resolve_query_string(
            Some("SELECT ?s WHERE { ?s ?p ?o }".to_string()),
            Some(std::env::temp_dir().join("oxirs_rsparql_both.sparql")),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_neither_is_error() {
        let result = resolve_query_string(None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_accept_header_mapping() {
        assert_eq!(results_format_to_accept("json"), ACCEPT_SPARQL_JSON);
        assert_eq!(results_format_to_accept("csv"), ACCEPT_CSV);
        assert_eq!(results_format_to_accept("tsv"), ACCEPT_TSV);
        assert_eq!(results_format_to_accept("xml"), ACCEPT_SPARQL_XML);
        // Unknown formats default to JSON
        assert_eq!(results_format_to_accept("parquet"), ACCEPT_SPARQL_JSON);
    }

    #[tokio::test]
    async fn test_run_with_mock_server() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        let json_response = r#"{
            "head": {"vars": ["s"]},
            "results": {"bindings": [
                {"s": {"type": "uri", "value": "http://example.org/subject"}}
            ]}
        }"#;

        Mock::given(method("POST"))
            .and(path("/sparql"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(json_response)
                    .insert_header("content-type", "application/sparql-results+json"),
            )
            .mount(&server)
            .await;

        let url = format!("{}/sparql", server.uri());
        let result = run(
            url,
            Some("SELECT ?s WHERE { ?s ?p ?o }".to_string()),
            None,
            "json".to_string(),
            30,
        )
        .await;
        assert!(result.is_ok(), "mock server should succeed: {result:?}");
    }

    #[tokio::test]
    async fn test_run_server_error() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/sparql"))
            .respond_with(ResponseTemplate::new(400).set_body_string("Bad Request"))
            .mount(&server)
            .await;

        let url = format!("{}/sparql", server.uri());
        let result = run(
            url,
            Some("INVALID SPARQL".to_string()),
            None,
            "json".to_string(),
            30,
        )
        .await;
        assert!(result.is_err(), "HTTP 400 should fail");
    }
}
