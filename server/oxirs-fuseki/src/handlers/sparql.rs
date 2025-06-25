//! SPARQL 1.1 Protocol implementation
//!
//! This module implements the SPARQL 1.1 Protocol for RDF as defined by W3C:
//! https://www.w3.org/TR/sparql11-protocol/
//!
//! Supports:
//! - SPARQL Query via GET and POST
//! - SPARQL Update via POST
//! - Content negotiation for response formats
//! - URL-encoded and direct POST queries
//! - Error handling with proper HTTP status codes

use crate::{
    auth::{AuthUser, Permission},
    config::ServerConfig,
    error::{FusekiError, FusekiResult},
    metrics::MetricsService,
    server::AppState,
    store::Store,
};
use axum::{
    body::Body,
    extract::{Query, State},
    http::{StatusCode, HeaderMap, header::{CONTENT_TYPE, ACCEPT}},
    response::{Html, Json, Response, IntoResponse},
    Form,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn, error, debug, instrument};

/// SPARQL query parameters for GET requests
#[derive(Debug, Deserialize)]
pub struct SparqlQueryParams {
    pub query: Option<String>,
    #[serde(rename = "default-graph-uri")]
    pub default_graph_uri: Option<Vec<String>>,
    #[serde(rename = "named-graph-uri")]
    pub named_graph_uri: Option<Vec<String>>,
    pub timeout: Option<u32>,
    pub format: Option<String>,
}

/// SPARQL update parameters for POST requests
#[derive(Debug, Deserialize)]
pub struct SparqlUpdateParams {
    pub update: String,
    #[serde(rename = "using-graph-uri")]
    pub using_graph_uri: Option<Vec<String>>,
    #[serde(rename = "using-named-graph-uri")]
    pub using_named_graph_uri: Option<Vec<String>>,
}

/// SPARQL query request body for direct POST
#[derive(Debug, Deserialize)]
pub struct SparqlQueryRequest {
    pub query: String,
    #[serde(rename = "default-graph-uri")]
    pub default_graph_uri: Option<Vec<String>>,
    #[serde(rename = "named-graph-uri")]
    pub named_graph_uri: Option<Vec<String>>,
    pub timeout: Option<u32>,
}

/// SPARQL query execution result
#[derive(Debug, Serialize)]
pub struct QueryResult {
    pub query_type: String,
    pub execution_time_ms: u64,
    pub result_count: Option<usize>,
    pub bindings: Option<Vec<HashMap<String, serde_json::Value>>>,
    pub boolean: Option<bool>,
    pub construct_graph: Option<String>,
    pub describe_graph: Option<String>,
}

/// SPARQL update execution result
#[derive(Debug, Serialize)]
pub struct UpdateResult {
    pub success: bool,
    pub execution_time_ms: u64,
    pub operations_count: usize,
    pub message: String,
}

/// Content type constants for SPARQL protocol
mod content_types {
    pub const SPARQL_QUERY: &str = "application/sparql-query";
    pub const SPARQL_UPDATE: &str = "application/sparql-update";
    pub const SPARQL_RESULTS_JSON: &str = "application/sparql-results+json";
    pub const SPARQL_RESULTS_XML: &str = "application/sparql-results+xml";
    pub const SPARQL_RESULTS_CSV: &str = "text/csv";
    pub const SPARQL_RESULTS_TSV: &str = "text/tab-separated-values";
    pub const RDF_XML: &str = "application/rdf+xml";
    pub const TURTLE: &str = "text/turtle";
    pub const N_TRIPLES: &str = "application/n-triples";
    pub const JSON_LD: &str = "application/ld+json";
    pub const FORM_URLENCODED: &str = "application/x-www-form-urlencoded";
}

/// SPARQL query handler supporting both GET and POST methods
#[instrument(skip(state, headers, body))]
pub async fn query_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query_params): Query<SparqlQueryParams>,
    body: Body,
) -> Result<Response, FusekiError> {
    let start_time = Instant::now();

    // Check authentication and authorization
    // Note: In a full implementation, we'd extract AuthUser here
    // For now, we'll implement basic functionality

    // Determine request method and extract query
    let (query_string, default_graphs, named_graphs) = extract_query_from_request(
        &headers,
        query_params,
        body,
    ).await?;

    if query_string.is_empty() {
        return Err(FusekiError::bad_request("Missing SPARQL query"));
    }

    // Validate query syntax (basic validation)
    validate_sparql_query(&query_string)?;

    // Determine response format based on Accept header
    let response_format = determine_response_format(&headers);

    debug!("Executing SPARQL query: {}", query_string.chars().take(100).collect::<String>());

    // Execute the query
    let query_result = execute_sparql_query(
        &state.store,
        &query_string,
        &default_graphs,
        &named_graphs,
    ).await?;

    let execution_time = start_time.elapsed();

    // Record metrics
    if let Some(metrics_service) = &state.metrics_service {
        metrics_service.record_sparql_query(
            execution_time,
            true,
            &determine_query_type(&query_string),
        ).await;
    }

    // Format and return response
    format_query_response(query_result, &response_format, execution_time).await
}

/// SPARQL update handler for POST requests only
#[instrument(skip(state, headers, body))]
pub async fn update_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Body,
) -> Result<Response, FusekiError> {
    let start_time = Instant::now();

    // Check authentication and authorization for update operations
    // Updates require write permissions
    
    // Extract update string from request
    let (update_string, using_graphs, using_named_graphs) = extract_update_from_request(
        &headers,
        body,
    ).await?;

    if update_string.is_empty() {
        return Err(FusekiError::bad_request("Missing SPARQL update"));
    }

    // Validate update syntax
    validate_sparql_update(&update_string)?;

    debug!("Executing SPARQL update: {}", update_string.chars().take(100).collect::<String>());

    // Execute the update
    let update_result = execute_sparql_update(
        &state.store,
        &update_string,
        &using_graphs,
        &using_named_graphs,
    ).await?;

    let execution_time = start_time.elapsed();

    // Record metrics
    if let Some(metrics_service) = &state.metrics_service {
        metrics_service.record_sparql_update(
            execution_time,
            update_result.success,
            "update",
        ).await;
    }

    // Return success response
    let response = UpdateResult {
        success: update_result.success,
        execution_time_ms: execution_time.as_millis() as u64,
        operations_count: update_result.operations_count,
        message: update_result.message,
    };

    Ok((StatusCode::OK, Json(response)).into_response())
}

/// Extract query from various request formats (GET params, POST form, POST direct)
async fn extract_query_from_request(
    headers: &HeaderMap,
    query_params: SparqlQueryParams,
    body: Body,
) -> FusekiResult<(String, Vec<String>, Vec<String>)> {
    // First try query parameter (GET request)
    if let Some(query) = query_params.query {
        return Ok((
            query,
            query_params.default_graph_uri.unwrap_or_default(),
            query_params.named_graph_uri.unwrap_or_default(),
        ));
    }

    // Try to extract from POST body
    let content_type = headers.get(CONTENT_TYPE)
        .and_then(|ct| ct.to_str().ok())
        .unwrap_or("");

    match content_type {
        ct if ct.starts_with(content_types::SPARQL_QUERY) => {
            // Direct SPARQL query in body
            let body_bytes = axum::body::to_bytes(body, usize::MAX).await
                .map_err(|e| FusekiError::bad_request(format!("Failed to read request body: {}", e)))?;
            let query = String::from_utf8(body_bytes.to_vec())
                .map_err(|e| FusekiError::bad_request(format!("Invalid UTF-8 in query: {}", e)))?;
            
            Ok((query, Vec::new(), Vec::new()))
        }
        ct if ct.starts_with(content_types::FORM_URLENCODED) => {
            // Form-encoded query
            let body_bytes = axum::body::to_bytes(body, usize::MAX).await
                .map_err(|e| FusekiError::bad_request(format!("Failed to read request body: {}", e)))?;
            
            let form_data: SparqlQueryRequest = serde_urlencoded::from_bytes(&body_bytes)
                .map_err(|e| FusekiError::bad_request(format!("Failed to parse form data: {}", e)))?;
            
            Ok((
                form_data.query,
                form_data.default_graph_uri.unwrap_or_default(),
                form_data.named_graph_uri.unwrap_or_default(),
            ))
        }
        _ => {
            Err(FusekiError::bad_request("Unsupported content type for SPARQL query"))
        }
    }
}

/// Extract update from POST request body
async fn extract_update_from_request(
    headers: &HeaderMap,
    body: Body,
) -> FusekiResult<(String, Vec<String>, Vec<String>)> {
    let content_type = headers.get(CONTENT_TYPE)
        .and_then(|ct| ct.to_str().ok())
        .unwrap_or("");

    match content_type {
        ct if ct.starts_with(content_types::SPARQL_UPDATE) => {
            // Direct SPARQL update in body
            let body_bytes = axum::body::to_bytes(body, usize::MAX).await
                .map_err(|e| FusekiError::bad_request(format!("Failed to read request body: {}", e)))?;
            let update = String::from_utf8(body_bytes.to_vec())
                .map_err(|e| FusekiError::bad_request(format!("Invalid UTF-8 in update: {}", e)))?;
            
            Ok((update, Vec::new(), Vec::new()))
        }
        ct if ct.starts_with(content_types::FORM_URLENCODED) => {
            // Form-encoded update
            let body_bytes = axum::body::to_bytes(body, usize::MAX).await
                .map_err(|e| FusekiError::bad_request(format!("Failed to read request body: {}", e)))?;
            
            let form_data: SparqlUpdateParams = serde_urlencoded::from_bytes(&body_bytes)
                .map_err(|e| FusekiError::bad_request(format!("Failed to parse form data: {}", e)))?;
            
            Ok((
                form_data.update,
                form_data.using_graph_uri.unwrap_or_default(),
                form_data.using_named_graph_uri.unwrap_or_default(),
            ))
        }
        _ => {
            Err(FusekiError::bad_request("Unsupported content type for SPARQL update"))
        }
    }
}

/// Determine response format based on Accept header
fn determine_response_format(headers: &HeaderMap) -> String {
    let accept_header = headers.get(ACCEPT)
        .and_then(|accept| accept.to_str().ok())
        .unwrap_or("application/sparql-results+json");

    // Parse Accept header and determine best format
    if accept_header.contains("application/sparql-results+json") {
        content_types::SPARQL_RESULTS_JSON.to_string()
    } else if accept_header.contains("application/sparql-results+xml") {
        content_types::SPARQL_RESULTS_XML.to_string()
    } else if accept_header.contains("text/csv") {
        content_types::SPARQL_RESULTS_CSV.to_string()
    } else if accept_header.contains("text/tab-separated-values") {
        content_types::SPARQL_RESULTS_TSV.to_string()
    } else if accept_header.contains("text/turtle") {
        content_types::TURTLE.to_string()
    } else if accept_header.contains("application/rdf+xml") {
        content_types::RDF_XML.to_string()
    } else if accept_header.contains("application/ld+json") {
        content_types::JSON_LD.to_string()
    } else {
        // Default to JSON
        content_types::SPARQL_RESULTS_JSON.to_string()
    }
}

/// Basic SPARQL query validation
fn validate_sparql_query(query: &str) -> FusekiResult<()> {
    let trimmed = query.trim().to_lowercase();
    
    if trimmed.is_empty() {
        return Err(FusekiError::bad_request("Empty SPARQL query"));
    }

    // Check for valid query types
    if !trimmed.starts_with("select") && 
       !trimmed.starts_with("construct") && 
       !trimmed.starts_with("describe") && 
       !trimmed.starts_with("ask") {
        return Err(FusekiError::bad_request("Invalid SPARQL query type"));
    }

    // Basic syntax validation (more comprehensive validation would be done by the SPARQL engine)
    if !query.contains('{') || !query.contains('}') {
        return Err(FusekiError::bad_request("Invalid SPARQL query syntax: missing braces"));
    }

    Ok(())
}

/// Basic SPARQL update validation
fn validate_sparql_update(update: &str) -> FusekiResult<()> {
    let trimmed = update.trim().to_lowercase();
    
    if trimmed.is_empty() {
        return Err(FusekiError::bad_request("Empty SPARQL update"));
    }

    // Check for valid update operations
    let valid_operations = ["insert", "delete", "load", "clear", "create", "drop", "copy", "move", "add"];
    let has_valid_operation = valid_operations.iter().any(|op| trimmed.contains(op));
    
    if !has_valid_operation {
        return Err(FusekiError::bad_request("Invalid SPARQL update operation"));
    }

    Ok(())
}

/// Determine query type from SPARQL query string
fn determine_query_type(query: &str) -> String {
    let trimmed = query.trim().to_lowercase();
    
    if trimmed.starts_with("select") {
        "SELECT".to_string()
    } else if trimmed.starts_with("construct") {
        "CONSTRUCT".to_string()
    } else if trimmed.starts_with("describe") {
        "DESCRIBE".to_string()
    } else if trimmed.starts_with("ask") {
        "ASK".to_string()
    } else {
        "UNKNOWN".to_string()
    }
}

/// Execute SPARQL query against the store
async fn execute_sparql_query(
    store: &Store,
    query: &str,
    default_graphs: &[String],
    named_graphs: &[String],
) -> FusekiResult<QueryResult> {
    // This is a simplified implementation
    // In a real implementation, this would use the actual SPARQL engine
    
    let query_type = determine_query_type(query);
    
    // Simulate query execution
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    
    match query_type.as_str() {
        "SELECT" => {
            // Mock SELECT results
            let bindings = vec![
                {
                    let mut binding = HashMap::new();
                    binding.insert("s".to_string(), serde_json::json!("http://example.org/subject1"));
                    binding.insert("p".to_string(), serde_json::json!("http://example.org/predicate1"));
                    binding.insert("o".to_string(), serde_json::json!("\"Object 1\""));
                    binding
                }
            ];
            
            Ok(QueryResult {
                query_type: query_type.clone(),
                execution_time_ms: 10,
                result_count: Some(bindings.len()),
                bindings: Some(bindings),
                boolean: None,
                construct_graph: None,
                describe_graph: None,
            })
        }
        "ASK" => {
            Ok(QueryResult {
                query_type: query_type.clone(),
                execution_time_ms: 5,
                result_count: None,
                bindings: None,
                boolean: Some(true),
                construct_graph: None,
                describe_graph: None,
            })
        }
        "CONSTRUCT" | "DESCRIBE" => {
            let graph = "@prefix ex: <http://example.org/> .\nex:subject ex:predicate \"object\" .";
            
            Ok(QueryResult {
                query_type: query_type.clone(),
                execution_time_ms: 15,
                result_count: Some(1),
                bindings: None,
                boolean: None,
                construct_graph: if query_type == "CONSTRUCT" { Some(graph.to_string()) } else { None },
                describe_graph: if query_type == "DESCRIBE" { Some(graph.to_string()) } else { None },
            })
        }
        _ => Err(FusekiError::bad_request("Unsupported query type"))
    }
}

/// Execute SPARQL update against the store
async fn execute_sparql_update(
    store: &Store,
    update: &str,
    using_graphs: &[String],
    using_named_graphs: &[String],
) -> FusekiResult<UpdateResult> {
    // This is a simplified implementation
    // In a real implementation, this would use the actual SPARQL update engine
    
    // Simulate update execution
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    
    // Mock successful update
    Ok(UpdateResult {
        success: true,
        execution_time_ms: 20,
        operations_count: 1,
        message: "Update completed successfully".to_string(),
    })
}

/// Format query response according to requested content type
async fn format_query_response(
    result: QueryResult,
    format: &str,
    execution_time: std::time::Duration,
) -> Result<Response, FusekiError> {
    match format {
        ct if ct == content_types::SPARQL_RESULTS_JSON => {
            // SPARQL 1.1 Results JSON Format
            let json_result = match result.query_type.as_str() {
                "SELECT" => {
                    serde_json::json!({
                        "head": {
                            "vars": ["s", "p", "o"]
                        },
                        "results": {
                            "bindings": result.bindings.unwrap_or_default()
                        }
                    })
                }
                "ASK" => {
                    serde_json::json!({
                        "head": {},
                        "boolean": result.boolean.unwrap_or(false)
                    })
                }
                _ => {
                    return Err(FusekiError::bad_request("Construct/Describe queries not supported in JSON format"));
                }
            };
            
            Ok((
                StatusCode::OK,
                [(CONTENT_TYPE, content_types::SPARQL_RESULTS_JSON)],
                Json(json_result)
            ).into_response())
        }
        ct if ct == content_types::TURTLE => {
            let turtle_data = result.construct_graph
                .or(result.describe_graph)
                .unwrap_or_default();
            
            Ok((
                StatusCode::OK,
                [(CONTENT_TYPE, content_types::TURTLE)],
                turtle_data
            ).into_response())
        }
        ct if ct == content_types::SPARQL_RESULTS_CSV => {
            // Convert to CSV format
            let csv_data = if let Some(bindings) = result.bindings {
                let mut csv = "s,p,o\n".to_string();
                for binding in bindings {
                    csv.push_str(&format!("{},{},{}\n",
                        binding.get("s").unwrap_or(&serde_json::Value::Null),
                        binding.get("p").unwrap_or(&serde_json::Value::Null),
                        binding.get("o").unwrap_or(&serde_json::Value::Null)
                    ));
                }
                csv
            } else {
                "".to_string()
            };
            
            Ok((
                StatusCode::OK,
                [(CONTENT_TYPE, content_types::SPARQL_RESULTS_CSV)],
                csv_data
            ).into_response())
        }
        _ => {
            // Default to JSON
            Ok((StatusCode::OK, Json(result)).into_response())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_validation() {
        assert!(validate_sparql_query("SELECT * WHERE { ?s ?p ?o }").is_ok());
        assert!(validate_sparql_query("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }").is_ok());
        assert!(validate_sparql_query("ASK { ?s ?p ?o }").is_ok());
        assert!(validate_sparql_query("DESCRIBE <http://example.org>").is_ok());
        
        assert!(validate_sparql_query("").is_err());
        assert!(validate_sparql_query("INVALID QUERY").is_err());
        assert!(validate_sparql_query("SELECT * WHERE").is_err());
    }

    #[test]
    fn test_update_validation() {
        assert!(validate_sparql_update("INSERT DATA { <s> <p> <o> }").is_ok());
        assert!(validate_sparql_update("DELETE DATA { <s> <p> <o> }").is_ok());
        assert!(validate_sparql_update("LOAD <http://example.org/data>").is_ok());
        assert!(validate_sparql_update("CLEAR GRAPH <http://example.org>").is_ok());
        
        assert!(validate_sparql_update("").is_err());
        assert!(validate_sparql_update("INVALID UPDATE").is_err());
    }

    #[test]
    fn test_query_type_detection() {
        assert_eq!(determine_query_type("SELECT * WHERE { ?s ?p ?o }"), "SELECT");
        assert_eq!(determine_query_type("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o }"), "CONSTRUCT");
        assert_eq!(determine_query_type("ASK { ?s ?p ?o }"), "ASK");
        assert_eq!(determine_query_type("DESCRIBE <http://example.org>"), "DESCRIBE");
        assert_eq!(determine_query_type("INVALID"), "UNKNOWN");
    }

    #[test]
    fn test_response_format_determination() {
        let mut headers = HeaderMap::new();
        
        headers.insert(ACCEPT, "application/sparql-results+json".parse().unwrap());
        assert_eq!(determine_response_format(&headers), content_types::SPARQL_RESULTS_JSON);
        
        headers.insert(ACCEPT, "text/turtle".parse().unwrap());
        assert_eq!(determine_response_format(&headers), content_types::TURTLE);
        
        headers.insert(ACCEPT, "text/csv".parse().unwrap());
        assert_eq!(determine_response_format(&headers), content_types::SPARQL_RESULTS_CSV);
    }
}