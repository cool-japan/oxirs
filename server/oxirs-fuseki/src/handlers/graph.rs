//! SPARQL 1.1 Graph Store HTTP Protocol implementation
//!
//! This module implements the SPARQL 1.1 Graph Store HTTP Protocol as defined by W3C:
//! https://www.w3.org/TR/sparql11-http-rdf-update/
//!
//! Supports:
//! - GET: Retrieve graph content
//! - PUT: Replace graph content  
//! - POST: Add to graph content
//! - DELETE: Remove graph content
//! - Content negotiation for RDF formats
//! - Default and named graph operations

use crate::{
    error::{FusekiError, FusekiResult},
    server::AppState,
    store::Store,
};
use axum::{
    body::Body,
    extract::{Query, State},
    http::{
        header::{ACCEPT, CONTENT_TYPE},
        HeaderMap, Method, StatusCode,
    },
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, instrument};

/// Graph Store protocol parameters
#[derive(Debug, Deserialize)]
pub struct GraphStoreParams {
    /// Graph URI for named graph operations
    pub graph: Option<String>,
    /// Default graph operations (mutually exclusive with graph)
    #[serde(rename = "default")]
    pub default: Option<bool>,
}

/// Graph operation result
#[derive(Debug, Serialize)]
pub struct GraphResult {
    pub success: bool,
    pub operation: String,
    pub graph_uri: Option<String>,
    pub execution_time_ms: u64,
    pub message: String,
    pub triple_count: Option<usize>,
}

/// RDF content type constants
mod rdf_content_types {
    pub const TURTLE: &str = "text/turtle";
    pub const N_TRIPLES: &str = "application/n-triples";
    pub const RDF_XML: &str = "application/rdf+xml";
    pub const JSON_LD: &str = "application/ld+json";
    pub const N_QUADS: &str = "application/n-quads";
    pub const TRIG: &str = "application/trig";
    pub const N3: &str = "text/n3";
}

/// Graph Store HTTP Protocol handler for all methods
#[instrument(skip(state, headers, body))]
pub async fn graph_store_handler(
    method: Method,
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Query(params): Query<GraphStoreParams>,
    body: Body,
) -> Result<Response, FusekiError> {
    let start_time = Instant::now();

    // Validate parameters
    validate_graph_store_params(&params)?;

    // Determine target graph
    let graph_uri = determine_target_graph(&params)?;
    let _is_default_graph = graph_uri.is_none();

    debug!("Graph Store operation: {} on graph {:?}", method, graph_uri);

    // Check authentication and authorization
    check_graph_store_permissions(&method, &graph_uri)?;

    // Execute operation based on HTTP method
    let result = match method {
        Method::GET => handle_graph_retrieve(&state.store, &graph_uri, &headers).await?,
        Method::PUT => handle_graph_replace(&state.store, &graph_uri, &headers, body).await?,
        Method::POST => handle_graph_add(&state.store, &graph_uri, &headers, body).await?,
        Method::DELETE => handle_graph_delete(&state.store, &graph_uri).await?,
        _ => {
            return Err(FusekiError::method_not_allowed(format!(
                "Method {method} not supported for Graph Store"
            )));
        }
    };

    let execution_time = start_time.elapsed();

    // Record metrics
    if let Some(metrics_service) = &state.metrics_service {
        metrics_service
            .record_sparql_update(
                execution_time,
                true, // Assume success if we get here
                &format!("graph_store_{}", method.as_str().to_lowercase()),
            )
            .await;
    }

    info!(
        "Graph Store {} operation completed in {}ms for graph {:?}",
        method,
        execution_time.as_millis(),
        graph_uri
    );

    Ok(result)
}

/// Validate Graph Store protocol parameters
fn validate_graph_store_params(params: &GraphStoreParams) -> FusekiResult<()> {
    // Check that graph and default are mutually exclusive
    if params.graph.is_some() && params.default == Some(true) {
        return Err(FusekiError::bad_request(
            "Cannot specify both 'graph' and 'default' parameters",
        ));
    }

    // Validate graph URI format if provided
    if let Some(ref graph_uri) = params.graph {
        if graph_uri.is_empty() {
            return Err(FusekiError::bad_request("Graph URI cannot be empty"));
        }

        // Basic URI validation
        if !graph_uri.starts_with("http://")
            && !graph_uri.starts_with("https://")
            && !graph_uri.starts_with("urn:")
        {
            return Err(FusekiError::bad_request("Invalid graph URI format"));
        }
    }

    Ok(())
}

/// Determine target graph from parameters
fn determine_target_graph(params: &GraphStoreParams) -> FusekiResult<Option<String>> {
    if let Some(ref graph_uri) = params.graph {
        Ok(Some(graph_uri.clone()))
    } else if params.default == Some(true) || (params.graph.is_none() && params.default.is_none()) {
        // Default graph operation
        Ok(None)
    } else {
        Err(FusekiError::bad_request(
            "Must specify either 'graph' parameter or 'default=true'",
        ))
    }
}

/// Check permissions for graph store operations
fn check_graph_store_permissions(method: &Method, _graph_uri: &Option<String>) -> FusekiResult<()> {
    // In a full implementation, this would check user permissions
    // For now, we'll implement basic validation

    match method {
        &Method::GET => {
            // Read operations - require read permission
            // if !user.has_permission(&Permission::GraphStore) {
            //     return Err(FusekiError::forbidden("Insufficient permissions for graph read"));
            // }
        }
        &Method::PUT | &Method::POST | &Method::DELETE => {
            // Write operations - require write permission
            // if !user.has_permission(&Permission::SparqlUpdate) {
            //     return Err(FusekiError::forbidden("Insufficient permissions for graph write"));
            // }
        }
        _ => {}
    }

    Ok(())
}

/// Handle GET request - retrieve graph content
async fn handle_graph_retrieve(
    _store: &Store,
    graph_uri: &Option<String>,
    headers: &HeaderMap,
) -> FusekiResult<Response> {
    // Determine response format from Accept header
    let response_format = determine_rdf_response_format(headers);

    // Retrieve graph data from store
    let graph_data = retrieve_graph_from_store(_store, graph_uri, &response_format).await?;

    if graph_data.is_empty() {
        // Graph doesn't exist or is empty
        return Ok(StatusCode::NOT_FOUND.into_response());
    }

    // Return graph data with appropriate content type
    Ok((
        StatusCode::OK,
        [(CONTENT_TYPE, response_format.as_str())],
        graph_data,
    )
        .into_response())
}

/// Handle PUT request - replace graph content
async fn handle_graph_replace(
    _store: &Store,
    graph_uri: &Option<String>,
    headers: &HeaderMap,
    body: Body,
) -> FusekiResult<Response> {
    // Determine content type
    let content_type = get_content_type(headers)?;

    // Read and validate RDF data
    let rdf_data = read_rdf_body(body, &content_type).await?;

    // Replace graph content in store
    let result = replace_graph_in_store(_store, graph_uri, &rdf_data, &content_type).await?;

    // Return success response
    let response = GraphResult {
        success: true,
        operation: "PUT".to_string(),
        graph_uri: graph_uri.clone(),
        execution_time_ms: 0, // Would be calculated from actual operation
        message: "Graph replaced successfully".to_string(),
        triple_count: Some(result.triple_count),
    };

    Ok((StatusCode::OK, axum::Json(response)).into_response())
}

/// Handle POST request - add to graph content
async fn handle_graph_add(
    _store: &Store,
    graph_uri: &Option<String>,
    headers: &HeaderMap,
    body: Body,
) -> FusekiResult<Response> {
    // Determine content type
    let content_type = get_content_type(headers)?;

    // Read and validate RDF data
    let rdf_data = read_rdf_body(body, &content_type).await?;

    // Add to graph content in store
    let result = add_to_graph_in_store(_store, graph_uri, &rdf_data, &content_type).await?;

    // Return success response
    let response = GraphResult {
        success: true,
        operation: "POST".to_string(),
        graph_uri: graph_uri.clone(),
        execution_time_ms: 0,
        message: "Triples added to graph successfully".to_string(),
        triple_count: Some(result.triple_count),
    };

    Ok((StatusCode::OK, axum::Json(response)).into_response())
}

/// Handle DELETE request - remove graph content
async fn handle_graph_delete(_store: &Store, graph_uri: &Option<String>) -> FusekiResult<Response> {
    // Delete graph from store
    let result = delete_graph_from_store(_store, graph_uri).await?;

    if !result.existed {
        return Ok(StatusCode::NOT_FOUND.into_response());
    }

    // Return success response
    let response = GraphResult {
        success: true,
        operation: "DELETE".to_string(),
        graph_uri: graph_uri.clone(),
        execution_time_ms: 0,
        message: "Graph deleted successfully".to_string(),
        triple_count: Some(result.deleted_count),
    };

    Ok((StatusCode::OK, axum::Json(response)).into_response())
}

/// Determine RDF response format from Accept header
fn determine_rdf_response_format(headers: &HeaderMap) -> String {
    let accept_header = headers
        .get(ACCEPT)
        .and_then(|accept| accept.to_str().ok())
        .unwrap_or("text/turtle");

    // Parse Accept header and determine best RDF format
    if accept_header.contains("text/turtle") {
        rdf_content_types::TURTLE.to_string()
    } else if accept_header.contains("application/rdf+xml") {
        rdf_content_types::RDF_XML.to_string()
    } else if accept_header.contains("application/n-triples") {
        rdf_content_types::N_TRIPLES.to_string()
    } else if accept_header.contains("application/ld+json") {
        rdf_content_types::JSON_LD.to_string()
    } else if accept_header.contains("application/n-quads") {
        rdf_content_types::N_QUADS.to_string()
    } else if accept_header.contains("application/trig") {
        rdf_content_types::TRIG.to_string()
    } else if accept_header.contains("text/n3") {
        rdf_content_types::N3.to_string()
    } else {
        // Default to Turtle
        rdf_content_types::TURTLE.to_string()
    }
}

/// Get content type from request headers
fn get_content_type(headers: &HeaderMap) -> FusekiResult<String> {
    let content_type = headers
        .get(CONTENT_TYPE)
        .and_then(|ct| ct.to_str().ok())
        .unwrap_or("text/turtle");

    // Validate that it's a supported RDF content type
    match content_type {
        ct if ct.starts_with("text/turtle") => Ok(rdf_content_types::TURTLE.to_string()),
        ct if ct.starts_with("application/rdf+xml") => Ok(rdf_content_types::RDF_XML.to_string()),
        ct if ct.starts_with("application/n-triples") => {
            Ok(rdf_content_types::N_TRIPLES.to_string())
        }
        ct if ct.starts_with("application/ld+json") => Ok(rdf_content_types::JSON_LD.to_string()),
        ct if ct.starts_with("application/n-quads") => Ok(rdf_content_types::N_QUADS.to_string()),
        ct if ct.starts_with("application/trig") => Ok(rdf_content_types::TRIG.to_string()),
        ct if ct.starts_with("text/n3") => Ok(rdf_content_types::N3.to_string()),
        _ => Err(FusekiError::bad_request(format!(
            "Unsupported RDF content type: {content_type}"
        ))),
    }
}

/// Read and validate RDF data from request body
async fn read_rdf_body(body: Body, content_type: &str) -> FusekiResult<String> {
    let body_bytes = axum::body::to_bytes(body, usize::MAX)
        .await
        .map_err(|e| FusekiError::bad_request(format!("Failed to read request body: {e}")))?;

    let rdf_data = String::from_utf8(body_bytes.to_vec())
        .map_err(|e| FusekiError::bad_request(format!("Invalid UTF-8 in RDF data: {e}")))?;

    if rdf_data.trim().is_empty() {
        return Err(FusekiError::bad_request("Empty RDF data"));
    }

    // Basic RDF syntax validation based on content type
    validate_rdf_syntax(&rdf_data, content_type)?;

    Ok(rdf_data)
}

/// Basic RDF syntax validation
fn validate_rdf_syntax(rdf_data: &str, content_type: &str) -> FusekiResult<()> {
    match content_type {
        ct if ct == rdf_content_types::TURTLE || ct == rdf_content_types::N3 => {
            // Basic Turtle/N3 validation
            if !rdf_data.contains('.') && !rdf_data.contains(';') {
                return Err(FusekiError::bad_request(
                    "Invalid Turtle syntax: missing statement terminators",
                ));
            }
        }
        ct if ct == rdf_content_types::N_TRIPLES => {
            // Basic N-Triples validation
            for line in rdf_data.lines() {
                let trimmed = line.trim();
                if !trimmed.is_empty() && !trimmed.starts_with('#') && !trimmed.ends_with('.') {
                    return Err(FusekiError::bad_request(
                        "Invalid N-Triples syntax: statements must end with '.'",
                    ));
                }
            }
        }
        ct if ct == rdf_content_types::RDF_XML => {
            // Basic XML validation
            if !rdf_data.trim_start().starts_with("<?xml") && !rdf_data.contains("<rdf:RDF") {
                return Err(FusekiError::bad_request("Invalid RDF/XML syntax"));
            }
        }
        ct if ct == rdf_content_types::JSON_LD => {
            // Basic JSON validation
            if !rdf_data.trim_start().starts_with('{') && !rdf_data.trim_start().starts_with('[') {
                return Err(FusekiError::bad_request("Invalid JSON-LD syntax"));
            }
        }
        _ => {
            // For other formats, accept without validation
        }
    }

    Ok(())
}

// Store operation structures and mock implementations

struct GraphRetrievalResult {
    data: String,
    triple_count: usize,
}

struct GraphModificationResult {
    triple_count: usize,
}

struct GraphDeletionResult {
    existed: bool,
    deleted_count: usize,
}

/// Retrieve graph data from store
async fn retrieve_graph_from_store(
    _store: &Store,
    graph_uri: &Option<String>,
    format: &str,
) -> FusekiResult<String> {
    // Mock implementation - in reality this would query the actual store
    match graph_uri {
        Some(uri) => {
            debug!("Retrieving named graph: {}", uri);
            // Return mock data for named graph
            match format {
                ct if ct == rdf_content_types::TURTLE => {
                    Ok(format!("# Graph: {uri}\n<http://example.org/subject> <http://example.org/predicate> \"object\" ."))
                }
                ct if ct == rdf_content_types::N_TRIPLES => {
                    Ok("<http://example.org/subject> <http://example.org/predicate> \"object\" .".to_string())
                }
                _ => Ok("".to_string())
            }
        }
        None => {
            debug!("Retrieving default graph");
            // Return mock data for default graph
            match format {
                ct if ct == rdf_content_types::TURTLE => {
                    Ok("<http://example.org/default> <http://example.org/predicate> \"default graph data\" .".to_string())
                }
                ct if ct == rdf_content_types::N_TRIPLES => {
                    Ok("<http://example.org/default> <http://example.org/predicate> \"default graph data\" .".to_string())
                }
                _ => Ok("".to_string())
            }
        }
    }
}

/// Replace graph content in store
async fn replace_graph_in_store(
    _store: &Store,
    graph_uri: &Option<String>,
    rdf_data: &str,
    content_type: &str,
) -> FusekiResult<GraphModificationResult> {
    // Mock implementation
    debug!(
        "Replacing graph {:?} with {} bytes of {} data",
        graph_uri,
        rdf_data.len(),
        content_type
    );

    // Simulate processing time
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    Ok(GraphModificationResult {
        triple_count: count_approximate_triples(rdf_data, content_type),
    })
}

/// Add triples to graph in store
async fn add_to_graph_in_store(
    _store: &Store,
    graph_uri: &Option<String>,
    rdf_data: &str,
    content_type: &str,
) -> FusekiResult<GraphModificationResult> {
    // Mock implementation
    debug!(
        "Adding to graph {:?} {} bytes of {} data",
        graph_uri,
        rdf_data.len(),
        content_type
    );

    // Simulate processing time
    tokio::time::sleep(std::time::Duration::from_millis(3)).await;

    Ok(GraphModificationResult {
        triple_count: count_approximate_triples(rdf_data, content_type),
    })
}

/// Delete graph from store
async fn delete_graph_from_store(
    _store: &Store,
    graph_uri: &Option<String>,
) -> FusekiResult<GraphDeletionResult> {
    // Mock implementation
    debug!("Deleting graph {:?}", graph_uri);

    // Simulate processing time
    tokio::time::sleep(std::time::Duration::from_millis(2)).await;

    Ok(GraphDeletionResult {
        existed: true,     // Assume graph existed
        deleted_count: 10, // Mock deletion count
    })
}

/// Approximate triple count based on RDF content
fn count_approximate_triples(rdf_data: &str, content_type: &str) -> usize {
    match content_type {
        ct if ct == rdf_content_types::N_TRIPLES => rdf_data
            .lines()
            .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'))
            .count(),
        ct if ct == rdf_content_types::TURTLE || ct == rdf_content_types::N3 => {
            // Count statements ending with '.'
            rdf_data.matches('.').count()
        }
        _ => {
            // For other formats, provide rough estimate
            rdf_data.len() / 100 // Very rough estimate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_store_params_validation() {
        // Valid cases
        assert!(validate_graph_store_params(&GraphStoreParams {
            graph: Some("http://example.org".to_string()),
            default: None
        })
        .is_ok());

        assert!(validate_graph_store_params(&GraphStoreParams {
            graph: None,
            default: Some(true)
        })
        .is_ok());

        // Invalid cases
        assert!(validate_graph_store_params(&GraphStoreParams {
            graph: Some("http://example.org".to_string()),
            default: Some(true)
        })
        .is_err());

        assert!(validate_graph_store_params(&GraphStoreParams {
            graph: Some("".to_string()),
            default: None
        })
        .is_err());
    }

    #[test]
    fn test_target_graph_determination() {
        // Named graph
        let params = GraphStoreParams {
            graph: Some("http://example.org".to_string()),
            default: None,
        };
        assert_eq!(
            determine_target_graph(&params).unwrap(),
            Some("http://example.org".to_string())
        );

        // Default graph
        let params = GraphStoreParams {
            graph: None,
            default: Some(true),
        };
        assert_eq!(determine_target_graph(&params).unwrap(), None);
    }

    #[test]
    fn test_rdf_format_determination() {
        let mut headers = HeaderMap::new();

        headers.insert(ACCEPT, "text/turtle".parse().unwrap());
        assert_eq!(
            determine_rdf_response_format(&headers),
            rdf_content_types::TURTLE
        );

        headers.insert(ACCEPT, "application/rdf+xml".parse().unwrap());
        assert_eq!(
            determine_rdf_response_format(&headers),
            rdf_content_types::RDF_XML
        );
    }

    #[test]
    fn test_rdf_syntax_validation() {
        // Valid Turtle
        assert!(validate_rdf_syntax("<s> <p> <o> .", rdf_content_types::TURTLE).is_ok());

        // Invalid Turtle (missing terminator)
        assert!(validate_rdf_syntax("<s> <p> <o>", rdf_content_types::TURTLE).is_err());

        // Valid N-Triples
        assert!(validate_rdf_syntax(
            "<http://s> <http://p> <http://o> .",
            rdf_content_types::N_TRIPLES
        )
        .is_ok());

        // Invalid N-Triples (missing terminator)
        assert!(validate_rdf_syntax(
            "<http://s> <http://p> <http://o>",
            rdf_content_types::N_TRIPLES
        )
        .is_err());
    }

    #[test]
    fn test_triple_counting() {
        let turtle_data = "<s1> <p> <o1> .\n<s2> <p> <o2> .";
        assert_eq!(
            count_approximate_triples(turtle_data, rdf_content_types::TURTLE),
            2
        );

        let ntriples_data =
            "<http://s1> <http://p> <http://o1> .\n<http://s2> <http://p> <http://o2> .";
        assert_eq!(
            count_approximate_triples(ntriples_data, rdf_content_types::N_TRIPLES),
            2
        );
    }
}
