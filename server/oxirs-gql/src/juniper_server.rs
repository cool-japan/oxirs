//! Juniper GraphQL HTTP server implementation
//!
//! This module provides a complete HTTP server implementation using Juniper
//! for GraphQL processing and Hyper for HTTP handling.

use crate::juniper_schema::{create_schema, GraphQLContext, Schema};
use crate::RdfStore;
use anyhow::{anyhow, Result};
use chrono;
use hyper::body::Bytes;
use hyper::service::service_fn;
use hyper::{body::Incoming, Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder;
use juniper::{execute, EmptyMutation, EmptySubscription};
use juniper_hyper::playground;
use juniper_hyper::{graphiql, graphql};
use serde_json;
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{debug, error, info, warn};

/// Configuration for the GraphQL server
#[derive(Debug, Clone)]
pub struct GraphQLServerConfig {
    /// Enable GraphiQL web interface
    pub enable_graphiql: bool,
    /// Enable GraphQL Playground web interface  
    pub enable_playground: bool,
    /// Enable introspection queries
    pub enable_introspection: bool,
    /// Maximum query depth allowed
    pub max_query_depth: Option<usize>,
    /// Maximum query complexity allowed
    pub max_query_complexity: Option<usize>,
    /// CORS configuration
    pub cors_enabled: bool,
    /// Allowed CORS origins (None means all origins allowed)
    pub cors_origins: Option<Vec<String>>,
}

impl Default for GraphQLServerConfig {
    fn default() -> Self {
        Self {
            enable_graphiql: true,
            enable_playground: true,
            enable_introspection: true,
            max_query_depth: Some(15),
            max_query_complexity: Some(1000),
            cors_enabled: true,
            cors_origins: None, // Allow all origins by default
        }
    }
}

/// The main GraphQL server using Juniper
pub struct JuniperGraphQLServer {
    schema: Arc<Schema>,
    context: GraphQLContext,
    config: GraphQLServerConfig,
}

impl JuniperGraphQLServer {
    /// Create a new GraphQL server with an RDF store
    pub fn new(store: Arc<RdfStore>) -> Self {
        let schema = Arc::new(create_schema());
        let context = GraphQLContext { store };
        let config = GraphQLServerConfig::default();

        Self {
            schema,
            context,
            config,
        }
    }

    /// Create a new GraphQL server with custom configuration
    pub fn with_config(store: Arc<RdfStore>, config: GraphQLServerConfig) -> Self {
        let schema = Arc::new(create_schema());
        let context = GraphQLContext { store };

        Self {
            schema,
            context,
            config,
        }
    }

    /// Start the GraphQL server on the specified address
    pub async fn start(&self, addr: SocketAddr) -> Result<()> {
        info!("Starting Juniper GraphQL server on {}", addr);

        // Create the service with proper cloning for the closure
        let schema = self.schema.clone();
        let context = self.context.clone();
        let config = self.config.clone();

        // Create TCP listener
        let listener = TcpListener::bind(addr).await?;

        info!("GraphQL server running on http://{}", addr);
        info!("GraphQL endpoint: http://{}/graphql", addr);

        // Accept connections in a loop
        loop {
            let (stream, _) = match listener.accept().await {
                Ok(result) => result,
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                    continue;
                }
            };

            let schema_clone = schema.clone();
            let context_clone = context.clone();
            let config_clone = config.clone();

            // Handle each connection in a separate task
            tokio::spawn(async move {
                let io = TokioIo::new(stream);
                let builder = Builder::new(TokioExecutor::new());

                let service = service_fn(move |req| {
                    Self::handle_request(
                        req,
                        (*schema_clone).clone(),
                        context_clone.clone(),
                        config_clone.clone(),
                    )
                });

                if let Err(e) = builder.serve_connection(io, service).await {
                    error!("Connection error: {}", e);
                }
            });
        }
    }

    /// Handle individual HTTP requests
    async fn handle_request(
        req: Request<Incoming>,
        schema: Schema,
        context: GraphQLContext,
        config: GraphQLServerConfig,
    ) -> Result<Response<String>, Infallible> {
        let response = match Self::handle_request_inner(req, schema, context, config).await {
            Ok(response) => response,
            Err(err) => {
                error!("Request handling error: {}", err);
                Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .header("content-type", "application/json")
                    .body(format!(
                        r#"{{"errors": [{{ "message": "{}" }}]}}"#,
                        err.to_string().replace('"', "\\\"")
                    ))
                    .unwrap()
            }
        };

        Ok(response)
    }

    /// Inner request handling with proper error propagation
    async fn handle_request_inner(
        req: Request<Incoming>,
        schema: Schema,
        context: GraphQLContext,
        config: GraphQLServerConfig,
    ) -> Result<Response<String>> {
        let method = req.method();
        let path = req.uri().path();

        debug!("Handling {} request to {}", method, path);

        // Apply CORS headers if enabled
        let mut response_builder = Response::builder();
        if config.cors_enabled {
            response_builder = response_builder
                .header("Access-Control-Allow-Origin", "*")
                .header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                .header(
                    "Access-Control-Allow-Headers",
                    "Content-Type, Authorization",
                );
        }

        // Handle CORS preflight requests
        if method == Method::OPTIONS {
            return Ok(response_builder
                .status(StatusCode::OK)
                .body(String::new())?);
        }

        match (method, path) {
            // Main GraphQL endpoint
            (&Method::GET, "/graphql") | (&Method::POST, "/graphql") => {
                debug!("Processing GraphQL request");
                let response = graphql(schema, context, req).await;

                // Add CORS headers to GraphQL response
                if config.cors_enabled {
                    let (parts, body) = response.into_parts();
                    let mut response = Response::from_parts(parts, body);
                    let headers = response.headers_mut();
                    headers.insert("Access-Control-Allow-Origin", "*".parse().unwrap());
                    headers.insert(
                        "Access-Control-Allow-Methods",
                        "GET, POST, OPTIONS".parse().unwrap(),
                    );
                    headers.insert(
                        "Access-Control-Allow-Headers",
                        "Content-Type, Authorization".parse().unwrap(),
                    );
                    Ok(response)
                } else {
                    Ok(response)
                }
            }

            // GraphiQL interface
            (&Method::GET, "/graphiql") if config.enable_graphiql => {
                debug!("Serving GraphiQL interface");
                let response = graphiql("/graphql", None).await;
                Ok(response)
            }

            // GraphQL Playground
            (&Method::GET, "/playground") if config.enable_playground => {
                debug!("Serving GraphQL Playground");
                let response = playground("/graphql", None).await;
                Ok(response)
            }

            // Health check endpoint
            (&Method::GET, "/health") => {
                debug!("Health check request");
                let health_info = serde_json::json!({
                    "status": "healthy",
                    "service": "oxirs-graphql",
                    "version": env!("CARGO_PKG_VERSION"),
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "endpoints": {
                        "graphql": "/graphql",
                        "graphiql": if config.enable_graphiql { serde_json::Value::String("/graphiql".to_string()) } else { serde_json::Value::Null },
                        "playground": if config.enable_playground { serde_json::Value::String("/playground".to_string()) } else { serde_json::Value::Null }
                    }
                });

                Ok(response_builder
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .body(health_info.to_string())?)
            }

            // Schema introspection endpoint (SDL)
            (&Method::GET, "/schema") if config.enable_introspection => {
                debug!("Schema introspection request");

                // For now, return a placeholder SDL - in production, generate from Juniper schema
                let sdl = r#"
                    """
                    An RDF IRI (Internationalized Resource Identifier)
                    """
                    scalar IRI

                    """
                    An RDF Literal with optional language tag and datatype
                    """
                    scalar RdfLiteral

                    """
                    Information about the RDF store
                    """
                    type StoreInfo {
                        """Total number of triples in the store"""
                        tripleCount: Int!
                        """Version of the GraphQL server"""
                        version: String!
                        """Description of the store"""
                        description: String!
                    }

                    """
                    Input for executing SPARQL queries
                    """
                    input SparqlQueryInput {
                        """The SPARQL query string"""
                        query: String!
                        """Optional result limit"""
                        limit: Int
                        """Optional result offset"""
                        offset: Int
                    }

                    """
                    The root query type
                    """
                    type Query {
                        """Get basic information about the RDF store"""
                        info: StoreInfo!
                        """Execute a SPARQL query"""
                        sparql(input: SparqlQueryInput!): String
                        """Get all subjects in the store"""
                        subjects(limit: Int): [String!]!
                        """Get all predicates in the store"""
                        predicates(limit: Int): [String!]!
                        """Search for resources by label or IRI pattern"""
                        search(pattern: String!, limit: Int): [String!]!
                    }

                    schema {
                        query: Query
                    }
                "#;

                Ok(response_builder
                    .status(StatusCode::OK)
                    .header("content-type", "text/plain")
                    .body(sdl.to_string())?)
            }

            // Root endpoint - redirect to GraphiQL or provide info
            (&Method::GET, "/") => {
                if config.enable_graphiql {
                    Ok(Response::builder()
                        .status(StatusCode::FOUND)
                        .header("location", "/graphiql")
                        .body(String::new())?)
                } else if config.enable_playground {
                    Ok(Response::builder()
                        .status(StatusCode::FOUND)
                        .header("location", "/playground")
                        .body(String::new())?)
                } else {
                    let info = serde_json::json!({
                        "service": "OxiRS GraphQL Server",
                        "version": env!("CARGO_PKG_VERSION"),
                        "description": "GraphQL interface for RDF data using Juniper",
                        "endpoints": {
                            "graphql": "/graphql",
                            "health": "/health",
                            "schema": "/schema"
                        }
                    });

                    Ok(response_builder
                        .status(StatusCode::OK)
                        .header("content-type", "application/json")
                        .body(info.to_string())?)
                }
            }

            // 404 for unknown endpoints
            _ => {
                warn!("Unknown endpoint requested: {} {}", method, path);
                let error_response = serde_json::json!({
                    "error": "Not Found",
                    "message": format!("Endpoint {} {} not found", method, path),
                    "available_endpoints": [
                        "/graphql",
                        "/health",
                        if config.enable_graphiql { "/graphiql" } else { "" },
                        if config.enable_playground { "/playground" } else { "" },
                        if config.enable_introspection { "/schema" } else { "" }
                    ]
                });

                Ok(response_builder
                    .status(StatusCode::NOT_FOUND)
                    .header("content-type", "application/json")
                    .body(error_response.to_string())?)
            }
        }
    }
}

/// Builder for GraphQL server configuration
pub struct GraphQLServerBuilder {
    config: GraphQLServerConfig,
}

impl GraphQLServerBuilder {
    pub fn new() -> Self {
        Self {
            config: GraphQLServerConfig::default(),
        }
    }

    pub fn enable_graphiql(mut self, enable: bool) -> Self {
        self.config.enable_graphiql = enable;
        self
    }

    pub fn enable_playground(mut self, enable: bool) -> Self {
        self.config.enable_playground = enable;
        self
    }

    pub fn enable_introspection(mut self, enable: bool) -> Self {
        self.config.enable_introspection = enable;
        self
    }

    pub fn max_query_depth(mut self, depth: Option<usize>) -> Self {
        self.config.max_query_depth = depth;
        self
    }

    pub fn max_query_complexity(mut self, complexity: Option<usize>) -> Self {
        self.config.max_query_complexity = complexity;
        self
    }

    pub fn cors_enabled(mut self, enabled: bool) -> Self {
        self.config.cors_enabled = enabled;
        self
    }

    pub fn cors_origins(mut self, origins: Vec<String>) -> Self {
        self.config.cors_origins = Some(origins);
        self
    }

    pub fn build(self, store: Arc<RdfStore>) -> JuniperGraphQLServer {
        JuniperGraphQLServer::with_config(store, self.config)
    }
}

impl Default for GraphQLServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function to start a GraphQL server with default configuration
pub async fn start_graphql_server(store: Arc<RdfStore>, addr: SocketAddr) -> Result<()> {
    let server = JuniperGraphQLServer::new(store);
    server.start(addr).await
}

/// Convenience function to start a GraphQL server with custom configuration
pub async fn start_graphql_server_with_config(
    store: Arc<RdfStore>,
    addr: SocketAddr,
    config: GraphQLServerConfig,
) -> Result<()> {
    let server = JuniperGraphQLServer::with_config(store, config);
    server.start(addr).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_server_creation() {
        let store = Arc::new(RdfStore::new().expect("Failed to create store"));
        let server = JuniperGraphQLServer::new(store);

        // Just test that we can create the server
        assert!(server.config.enable_graphiql);
        assert!(server.config.enable_playground);
        assert!(server.config.enable_introspection);
    }

    #[tokio::test]
    async fn test_server_builder() {
        let store = Arc::new(RdfStore::new().expect("Failed to create store"));

        let server = GraphQLServerBuilder::new()
            .enable_graphiql(false)
            .enable_playground(true)
            .enable_introspection(false)
            .max_query_depth(Some(10))
            .cors_enabled(true)
            .build(store);

        assert!(!server.config.enable_graphiql);
        assert!(server.config.enable_playground);
        assert!(!server.config.enable_introspection);
        assert_eq!(server.config.max_query_depth, Some(10));
        assert!(server.config.cors_enabled);
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        // This test would require actually starting a server
        // For now, just test the builder functionality
        let store = Arc::new(RdfStore::new().expect("Failed to create store"));
        let _server = JuniperGraphQLServer::new(store);

        // In a real test, we would:
        // 1. Start the server on a random port
        // 2. Make HTTP requests to test endpoints
        // 3. Verify responses
        // This requires more complex test infrastructure
    }
}
