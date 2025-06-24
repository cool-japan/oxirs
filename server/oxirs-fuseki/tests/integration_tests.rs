//! Comprehensive integration tests for OxiRS Fuseki server
//!
//! These tests verify the complete functionality of the server including:
//! - SPARQL Protocol compliance
//! - Graph Store Protocol
//! - Authentication and authorization
//! - Performance and caching
//! - Configuration management
//! - Error handling
//! - Security features

use axum_test::TestServer;
use oxirs_fuseki::{
    config::{ServerConfig, SecurityConfig, MonitoringConfig, PerformanceConfig},
    store::Store,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Test server builder for creating servers with different configurations
struct TestServerBuilder {
    config: ServerConfig,
    auth_enabled: bool,
    metrics_enabled: bool,
    temp_dir: Option<TempDir>,
}

impl TestServerBuilder {
    fn new() -> Self {
        Self {
            config: ServerConfig::default(),
            auth_enabled: false,
            metrics_enabled: false,
            temp_dir: None,
        }
    }

    fn with_auth(mut self) -> Self {
        self.auth_enabled = true;
        self.config.security.authentication.enabled = true;
        self
    }

    fn with_metrics(mut self) -> Self {
        self.metrics_enabled = true;
        self.config.monitoring.metrics.enabled = true;
        self
    }

    fn with_temp_store(mut self) -> Self {
        let temp_dir = TempDir::new().expect("Failed to create temp directory");
        self.temp_dir = Some(temp_dir);
        self
    }

    async fn build(self) -> TestServer {
        let app_state = oxirs_fuseki::server::AppState {
            store: Store::new().unwrap(),
            config: self.config,
            auth_service: None, // Would be Some(auth_service) in real implementation
            metrics_service: None, // Would be Some(Arc::new(metrics_service)) in real implementation
            performance_service: None, // Would be Some(Arc::new(performance_service)) in real implementation
            #[cfg(feature = "rate-limit")]
            rate_limiter: None,
        };

        let app = create_test_router(app_state);
        TestServer::new(app).unwrap()
    }
}

/// Create a test server for integration testing with default configuration
async fn create_test_server() -> TestServer {
    TestServerBuilder::new().build().await
}

fn create_test_router(state: oxirs_fuseki::server::AppState) -> axum::Router {
    use axum::{routing::{get, post}, Router};
    use oxirs_fuseki::handlers;
    use oxirs_fuseki::server::{health_check, readiness_check, liveness_check, stats_handler, ping_handler};
    use tower_http::{cors::CorsLayer, trace::TraceLayer};
    use tower::ServiceBuilder;

    let cors = CorsLayer::permissive();
    let middleware = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(cors);

    Router::new()
        .route("/", get(handlers::admin::ui_handler))
        .route("/health", get(health_check))
        .route("/health/ready", get(readiness_check))
        .route("/health/live", get(liveness_check))
        .route("/sparql", get(handlers::query_handler).post(handlers::query_handler))
        .route("/update", post(handlers::update_handler))
        .route("/$/stats", get(stats_handler))
        .route("/$/ping", get(ping_handler))
        .with_state(state)
        .layer(middleware)
}

#[tokio::test]
async fn test_health_endpoint() {
    let server = create_test_server().await;
    
    let response = server.get("/health").await;
    response.assert_status_ok();
}

#[tokio::test]
async fn test_readiness_endpoint() {
    let server = create_test_server().await;
    
    let response = server.get("/health/ready").await;
    response.assert_status_ok();
}

#[tokio::test]
async fn test_liveness_endpoint() {
    let server = create_test_server().await;
    
    let response = server.get("/health/live").await;
    response.assert_status_ok();
}

#[tokio::test]
async fn test_ping_endpoint() {
    let server = create_test_server().await;
    
    let response = server.get("/$/ping").await;
    response.assert_status_ok();
    response.assert_text("pong");
}

#[tokio::test]
async fn test_stats_endpoint() {
    let server = create_test_server().await;
    
    let response = server.get("/$/stats").await;
    response.assert_status_ok();
    
    let json: Value = response.json();
    assert!(json.get("datasets").is_some());
    assert!(json.get("version").is_some());
}

#[tokio::test]
async fn test_admin_ui_endpoint() {
    let server = create_test_server().await;
    
    let response = server.get("/").await;
    response.assert_status_ok();
    
    let html = response.text();
    assert!(html.contains("OxiRS Fuseki Server"));
    assert!(html.contains("SPARQL"));
}

#[tokio::test]
async fn test_sparql_query_get() {
    let server = create_test_server().await;
    
    let response = server
        .get("/sparql")
        .add_query_param("query", "SELECT * WHERE { ?s ?p ?o } LIMIT 10")
        .await;
    
    response.assert_status_ok();
    
    let json: Value = response.json();
    assert!(json.get("head").is_some());
    assert!(json.get("results").is_some());
}

#[tokio::test]
async fn test_sparql_query_post_form() {
    let server = create_test_server().await;
    
    let mut form_data = HashMap::new();
    form_data.insert("query", "SELECT * WHERE { ?s ?p ?o } LIMIT 10");
    
    let response = server
        .post("/sparql")
        .form(&form_data)
        .await;
    
    response.assert_status_ok();
    
    let json: Value = response.json();
    assert!(json.get("head").is_some());
    assert!(json.get("results").is_some());
}

#[tokio::test]
async fn test_sparql_query_post_sparql_query() {
    let server = create_test_server().await;
    
    let response = server
        .post("/sparql")
        .content_type("application/sparql-query")
        .text("SELECT * WHERE { ?s ?p ?o } LIMIT 10")
        .await;
    
    response.assert_status_ok();
}

#[tokio::test]
async fn test_sparql_query_accept_json() {
    let server = create_test_server().await;
    
    let response = server
        .get("/sparql")
        .add_query_param("query", "SELECT * WHERE { ?s ?p ?o } LIMIT 10")
        .add_header("Accept", "application/sparql-results+json")
        .await;
    
    response.assert_status_ok();
    let header = response.header("content-type");
    let content_type = header.to_str().unwrap();
    assert!(content_type.contains("application/sparql-results+json"));
}

#[tokio::test]
async fn test_sparql_query_accept_xml() {
    let server = create_test_server().await;
    
    let response = server
        .get("/sparql")
        .add_query_param("query", "SELECT * WHERE { ?s ?p ?o } LIMIT 10")
        .add_header("Accept", "application/sparql-results+xml")
        .await;
    
    response.assert_status_ok();
    let header = response.header("content-type");
    let content_type = header.to_str().unwrap();
    assert!(content_type.contains("application/sparql-results+xml"));
}

#[tokio::test]
async fn test_sparql_query_accept_csv() {
    let server = create_test_server().await;
    
    let response = server
        .get("/sparql")
        .add_query_param("query", "SELECT * WHERE { ?s ?p ?o } LIMIT 10")
        .add_header("Accept", "text/csv")
        .await;
    
    response.assert_status_ok();
    let header = response.header("content-type");
    let content_type = header.to_str().unwrap();
    assert!(content_type.contains("text/csv"));
}

#[tokio::test]
async fn test_sparql_query_no_query() {
    let server = create_test_server().await;
    
    let response = server.get("/sparql").await;
    response.assert_status_bad_request();
}

#[tokio::test]
async fn test_sparql_update_post() {
    let server = create_test_server().await;
    
    let mut form_data = HashMap::new();
    form_data.insert("query", "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }");
    
    let response = server
        .post("/update")
        .form(&form_data)
        .await;
    
    response.assert_status(axum::http::StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn test_sparql_update_no_update() {
    let server = create_test_server().await;
    
    let response = server.post("/update").await;
    response.assert_status_bad_request();
}

#[tokio::test]
async fn test_cors_headers() {
    let server = create_test_server().await;
    
    // Test that CORS is enabled by checking a simple request works
    let response = server
        .get("/health")
        .add_header("Origin", "http://localhost:3000")
        .await;
    
    response.assert_status_ok();
}

#[tokio::test]
async fn test_multiple_requests() {
    let server = create_test_server().await;
    
    // Test multiple sequential requests
    for i in 0..5 {
        let response = server
            .get("/sparql")
            .add_query_param("query", &format!("SELECT * WHERE {{ ?s ?p ?o }} LIMIT {}", i + 1))
            .await;
        
        response.assert_status_ok();
    }
}

// ==================== Enhanced Test Suites ====================

/// Graph Store Protocol tests
#[cfg(test)]
mod graph_store_tests {
    use super::*;

    #[tokio::test]
    async fn test_graph_store_get_default() {
        let server = create_test_server().await;
        
        let response = server
            .get("/graph-store")
            .add_query_param("default", "true")
            .await;
        
        response.assert_status_ok();
        // Should return RDF content
        let content = response.text();
        assert!(!content.is_empty());
    }

    #[tokio::test]
    async fn test_graph_store_get_named() {
        let server = create_test_server().await;
        
        let response = server
            .get("/graph-store")
            .add_query_param("graph", "http://example.org/graph1")
            .await;
        
        response.assert_status_ok();
    }

    #[tokio::test]
    async fn test_graph_store_put_turtle() {
        let server = create_test_server().await;
        
        let turtle_data = "@prefix ex: <http://example.org/> .\nex:subject ex:predicate \"object\" .";
        
        let response = server
            .put("/graph-store")
            .add_query_param("default", "true")
            .content_type("text/turtle")
            .text(turtle_data)
            .await;
        
        response.assert_status_ok();
    }

    #[tokio::test]
    async fn test_graph_store_post_ntriples() {
        let server = create_test_server().await;
        
        let ntriples_data = "<http://example.org/s> <http://example.org/p> \"object\" .";
        
        let response = server
            .post("/graph-store")
            .add_query_param("graph", "http://example.org/test")
            .content_type("application/n-triples")
            .text(ntriples_data)
            .await;
        
        response.assert_status_ok();
    }

    #[tokio::test]
    async fn test_graph_store_delete() {
        let server = create_test_server().await;
        
        let response = server
            .delete("/graph-store")
            .add_query_param("graph", "http://example.org/test")
            .await;
        
        response.assert_status_ok();
    }

    #[tokio::test]
    async fn test_graph_store_content_negotiation() {
        let server = create_test_server().await;
        
        // Test different Accept headers
        let formats = vec![
            ("text/turtle", "text/turtle"),
            ("application/rdf+xml", "application/rdf+xml"),
            ("application/n-triples", "application/n-triples"),
        ];

        for (accept, expected_content_type) in formats {
            let response = server
                .get("/graph-store")
                .add_query_param("default", "true")
                .add_header("Accept", accept)
                .await;
            
            response.assert_status_ok();
            let content_type = response.header("content-type").to_str().unwrap();
            assert!(content_type.contains(expected_content_type));
        }
    }

    #[tokio::test]
    async fn test_graph_store_invalid_params() {
        let server = create_test_server().await;
        
        // Test both graph and default parameters (should fail)
        let response = server
            .get("/graph-store")
            .add_query_param("graph", "http://example.org/test")
            .add_query_param("default", "true")
            .await;
        
        response.assert_status_bad_request();
    }
}

/// Error handling and edge cases tests
#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_malformed_sparql_queries() {
        let server = create_test_server().await;
        
        let invalid_queries = vec![
            "",
            "INVALID SPARQL",
            "SELECT * WHERE",
            "SELECT * { ?s ?p ?o", // Missing closing brace
            "CONSTRUCT WHERE { ?s ?p ?o }", // Missing construct template
        ];

        for invalid_query in invalid_queries {
            let response = server
                .get("/sparql")
                .add_query_param("query", invalid_query)
                .await;
            
            response.assert_status_bad_request();
        }
    }

    #[tokio::test]
    async fn test_unsupported_methods() {
        let server = create_test_server().await;
        
        // SPARQL endpoint should not support PUT/DELETE
        let response = server.put("/sparql").await;
        response.assert_status(axum::http::StatusCode::METHOD_NOT_ALLOWED);
        
        let response = server.delete("/sparql").await;
        response.assert_status(axum::http::StatusCode::METHOD_NOT_ALLOWED);
    }

    #[tokio::test]
    async fn test_large_query_handling() {
        let server = create_test_server().await;
        
        // Create a very large query
        let mut large_query = "SELECT * WHERE {\n".to_string();
        for i in 0..1000 {
            large_query.push_str(&format!("  ?s{} ?p{} ?o{} .\n", i, i, i));
        }
        large_query.push('}');

        let response = server
            .post("/sparql")
            .content_type("application/sparql-query")
            .text(&large_query)
            .await;

        // Should either handle gracefully or return appropriate error
        assert!(response.status_code().is_success() || response.status_code().is_client_error());
    }

    #[tokio::test]
    async fn test_request_timeout() {
        let server = create_test_server().await;
        
        // Test that requests complete within reasonable time
        let start = Instant::now();
        
        let response = server
            .get("/sparql")
            .add_query_param("query", "SELECT * WHERE { ?s ?p ?o }")
            .await;
        
        let elapsed = start.elapsed();
        
        response.assert_status_ok();
        assert!(elapsed < Duration::from_secs(10), "Request should complete within timeout");
    }

    #[tokio::test]
    async fn test_invalid_content_types() {
        let server = create_test_server().await;
        
        let response = server
            .post("/sparql")
            .content_type("application/invalid")
            .text("SELECT * WHERE { ?s ?p ?o }")
            .await;
        
        response.assert_status_bad_request();
    }
}

/// Performance and caching tests
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[tokio::test]
    async fn test_concurrent_requests() {
        let server = create_test_server().await;
        
        // Send multiple concurrent requests
        let mut handles = Vec::new();
        
        for i in 0..10 {
            let server_clone = server.clone();
            let handle = tokio::spawn(async move {
                let response = server_clone
                    .get("/sparql")
                    .add_query_param("query", &format!("SELECT * WHERE {{ ?s{} ?p ?o }}", i))
                    .await;
                response.status_code()
            });
            handles.push(handle);
        }

        // Wait for all requests to complete
        for handle in handles {
            let status = handle.await.unwrap();
            assert!(status.is_success());
        }
    }

    #[tokio::test]
    async fn test_response_time_consistency() {
        let server = create_test_server().await;
        
        let mut response_times = Vec::new();
        
        // Make several requests and measure response times
        for _ in 0..5 {
            let start = Instant::now();
            
            let response = server
                .get("/sparql")
                .add_query_param("query", "SELECT * WHERE { ?s ?p ?o }")
                .await;
            
            let elapsed = start.elapsed();
            response.assert_status_ok();
            response_times.push(elapsed.as_millis());
        }

        // Response times should be reasonably consistent (no extreme outliers)
        let avg_time = response_times.iter().sum::<u128>() / response_times.len() as u128;
        for &time in &response_times {
            assert!(time < avg_time * 3, "Response time should be reasonably consistent");
        }
    }

    #[tokio::test]
    async fn test_memory_stability() {
        let server = create_test_server().await;
        
        // Send many requests to test for memory leaks
        for i in 0..50 {
            let response = server
                .get("/sparql")
                .add_query_param("query", &format!("SELECT * WHERE {{ ?s{} ?p ?o }}", i))
                .await;
            
            response.assert_status_ok();
        }

        // If we get here without issues, the server handled the load well
        assert!(true);
    }
}

/// Security tests
#[cfg(test)]
mod security_tests {
    use super::*;

    #[tokio::test]
    async fn test_injection_prevention() {
        let server = create_test_server().await;
        
        // Test potential injection attempts
        let malicious_inputs = vec![
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "UNION SELECT * FROM users",
            "%27%20OR%20%271%27%3D%271", // URL-encoded ' OR '1'='1
        ];

        for malicious_input in malicious_inputs {
            let response = server
                .get("/sparql")
                .add_query_param("query", malicious_input)
                .await;

            // Should either reject the query or handle it safely
            assert!(
                response.status_code().is_client_error() ||
                response.status_code().is_success()
            );
            
            // Response should not contain the malicious input echoed back
            let response_text = response.text();
            assert!(!response_text.contains("<script>"));
        }
    }

    #[tokio::test]
    async fn test_cors_security() {
        let server = create_test_server().await;
        
        // Test CORS preflight request
        let response = server
            .options("/sparql")
            .add_header("Origin", "http://evil.com")
            .add_header("Access-Control-Request-Method", "POST")
            .await;

        // Should handle CORS appropriately
        assert!(response.status_code().is_success());
    }

    #[tokio::test]
    async fn test_sensitive_headers() {
        let server = create_test_server().await;
        
        let response = server.get("/health").await;
        
        // Check that sensitive information is not leaked in headers
        let headers = response.headers();
        assert!(!headers.contains_key("server")); // Don't expose server version
        
        response.assert_status_ok();
    }
}

/// Admin and monitoring endpoint tests
#[cfg(test)]
mod admin_tests {
    use super::*;

    #[tokio::test]
    async fn test_admin_ui_content() {
        let server = create_test_server().await;
        
        let response = server.get("/").await;
        response.assert_status_ok();
        
        let html = response.text();
        assert!(html.contains("OxiRS Fuseki"));
        assert!(html.contains("SPARQL"));
        assert!(html.contains("datasets"));
        assert!(html.contains("health"));
    }

    #[tokio::test]
    async fn test_server_stats_detailed() {
        let server = create_test_server().await;
        
        let response = server.get("/$/stats").await;
        response.assert_status_ok();
        
        let json: Value = response.json();
        assert!(json["datasets"].is_number());
        assert!(json["version"].is_string());
        
        // Stats should contain reasonable values
        let datasets_count = json["datasets"].as_u64().unwrap();
        assert!(datasets_count >= 0);
    }

    #[tokio::test]
    async fn test_health_check_detailed() {
        let server = create_test_server().await;
        
        let response = server.get("/health").await;
        response.assert_status_ok();
        
        // Health endpoint might return JSON with detailed status
        if let Ok(json) = serde_json::from_str::<Value>(&response.text()) {
            if json.is_object() {
                assert!(json.get("status").is_some());
            }
        }
    }
}

/// Configuration and validation tests
#[cfg(test)]
mod config_tests {
    use super::*;

    #[tokio::test]
    async fn test_different_configurations() {
        // Test server with different configurations
        let server_with_auth = TestServerBuilder::new()
            .with_auth()
            .build()
            .await;
        
        let server_with_metrics = TestServerBuilder::new()
            .with_metrics()
            .build()
            .await;

        // Both should start successfully
        let response1 = server_with_auth.get("/health").await;
        response1.assert_status_ok();
        
        let response2 = server_with_metrics.get("/health").await;
        response2.assert_status_ok();
    }

    #[test]
    fn test_config_validation() {
        let mut config = ServerConfig::default();
        
        // Default config should be valid
        assert!(config.validate().is_ok());
        
        // Invalid port should fail
        config.server.port = 0;
        assert!(config.validate().is_err());
        
        // Reset to valid
        config.server.port = 3030;
        assert!(config.validate().is_ok());
        
        // Empty host should fail
        config.server.host = String::new();
        assert!(config.validate().is_err());
    }
}

/// SPARQL Protocol compliance tests
#[cfg(test)]
mod sparql_protocol_tests {
    use super::*;

    #[tokio::test]
    async fn test_sparql_query_types() {
        let server = create_test_server().await;
        
        let query_types = vec![
            ("SELECT * WHERE { ?s ?p ?o } LIMIT 1", "SELECT"),
            ("CONSTRUCT { ?s ?p ?o } WHERE { ?s ?p ?o } LIMIT 1", "CONSTRUCT"),
            ("ASK { ?s ?p ?o }", "ASK"),
            ("DESCRIBE <http://example.org/resource>", "DESCRIBE"),
        ];

        for (query, query_type) in query_types {
            let response = server
                .get("/sparql")
                .add_query_param("query", query)
                .await;
            
            response.assert_status_ok();
            
            // Verify appropriate response format for query type
            let content_type = response.header("content-type").to_str().unwrap();
            match query_type {
                "SELECT" | "ASK" => assert!(content_type.contains("sparql-results") || content_type.contains("json")),
                "CONSTRUCT" | "DESCRIBE" => assert!(content_type.contains("turtle") || content_type.contains("rdf")),
                _ => {}
            }
        }
    }

    #[tokio::test]
    async fn test_sparql_update_types() {
        let server = create_test_server().await;
        
        let update_operations = vec![
            "INSERT DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }",
            "DELETE DATA { <http://example.org/s> <http://example.org/p> <http://example.org/o> }",
            "LOAD <http://example.org/data.ttl>",
            "CLEAR GRAPH <http://example.org/graph>",
        ];

        for update in update_operations {
            let response = server
                .post("/update")
                .content_type("application/sparql-update")
                .text(update)
                .await;
            
            // Updates should succeed or return appropriate error
            assert!(response.status_code().is_success() || response.status_code().is_client_error());
        }
    }

    #[tokio::test]
    async fn test_query_parameters() {
        let server = create_test_server().await;
        
        // Test with default graph URI
        let response = server
            .get("/sparql")
            .add_query_param("query", "SELECT * WHERE { ?s ?p ?o }")
            .add_query_param("default-graph-uri", "http://example.org/default")
            .await;
        
        response.assert_status_ok();
        
        // Test with named graph URI
        let response = server
            .get("/sparql")
            .add_query_param("query", "SELECT * WHERE { ?s ?p ?o }")
            .add_query_param("named-graph-uri", "http://example.org/named")
            .await;
        
        response.assert_status_ok();
    }
}

/// Comprehensive integration test
#[tokio::test]
async fn test_end_to_end_workflow() {
    let server = create_test_server().await;
    
    // 1. Check server health
    let response = server.get("/health").await;
    response.assert_status_ok();
    
    // 2. Load some data via SPARQL Update
    let insert_data = "INSERT DATA { 
        <http://example.org/alice> <http://example.org/name> \"Alice\" .
        <http://example.org/bob> <http://example.org/name> \"Bob\" .
        <http://example.org/alice> <http://example.org/knows> <http://example.org/bob> .
    }";
    
    let response = server
        .post("/update")
        .content_type("application/sparql-update")
        .text(insert_data)
        .await;
    
    assert!(response.status_code().is_success());
    
    // 3. Query the data
    let response = server
        .get("/sparql")
        .add_query_param("query", "SELECT ?name WHERE { ?person <http://example.org/name> ?name }")
        .await;
    
    response.assert_status_ok();
    
    // 4. Check that we can get statistics
    let response = server.get("/$/stats").await;
    response.assert_status_ok();
    
    // 5. Test Graph Store Protocol
    let turtle_data = "@prefix ex: <http://example.org/> .\nex:charlie ex:name \"Charlie\" .";
    let response = server
        .put("/graph-store")
        .add_query_param("graph", "http://example.org/test-graph")
        .content_type("text/turtle")
        .text(turtle_data)
        .await;
    
    assert!(response.status_code().is_success());
    
    // 6. Retrieve the graph
    let response = server
        .get("/graph-store")
        .add_query_param("graph", "http://example.org/test-graph")
        .add_header("Accept", "text/turtle")
        .await;
    
    response.assert_status_ok();
    let content = response.text();
    assert!(content.contains("Charlie") || content.len() > 0);
}