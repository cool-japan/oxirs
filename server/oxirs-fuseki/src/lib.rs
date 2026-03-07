//! # OxiRS Fuseki - SPARQL HTTP Server
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-fuseki/badge.svg)](https://docs.rs/oxirs-fuseki)
//!
//! **Status**: Production Release (v0.1.0)
//! **Stability**: Public APIs are stable. Production-ready with comprehensive testing.
//!
//! SPARQL 1.1/1.2 HTTP protocol server with Apache Fuseki compatibility.
//! Provides a production-ready HTTP interface for RDF data with query and update operations.
//!
//! ## Features
//!
//! - **SPARQL Protocol** - Full SPARQL 1.1 HTTP protocol implementation
//! - **Query Endpoints** - SPARQL query execution via HTTP
//! - **Update Operations** - SPARQL Update for data modification
//! - **Dataset Management** - RESTful API for managing datasets
//! - **Authentication** - Basic auth and OAuth2/OIDC support (in progress)
//! - **Fuseki Compatibility** - Compatible with Fuseki configuration formats
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use oxirs_fuseki::Server;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let server = Server::builder()
//!         .port(3030)
//!         .dataset_path("/data")
//!         .build()
//!         .await?;
//!
//!     server.run().await?;
//!     Ok(())
//! }
//! ```
//!
//! ## See Also
//!
//! - [`oxirs-arq`](https://docs.rs/oxirs-arq) - SPARQL query engine
//! - [`oxirs-core`](https://docs.rs/oxirs-core) - RDF data model

use std::net::SocketAddr;

pub mod adaptive_execution;
pub mod admin_ui;
pub mod aggregation;
pub mod analytics;
pub mod api;
pub mod auth;
pub mod backup;
pub mod batch_execution;
pub mod bind_values_enhanced;
pub mod cache;
pub mod clustering;
pub mod concurrent;
pub mod config;
#[cfg(feature = "hot-reload")]
pub mod config_reload;
pub mod connection_pool;
pub mod consciousness;
pub mod dataset_catalog;
pub mod dataset_management;
pub mod ddos_protection;
pub mod disaster_recovery;
pub mod error;
pub mod federated_query_optimizer;
pub mod federation;
pub mod gpu_kg_embeddings;
pub mod graph_analytics;
pub mod graphql_integration;
pub mod handlers;
pub mod health;
pub mod http_protocol;
pub mod ids;
pub mod k8s_operator;
pub mod memory_pool;
pub mod metrics;
pub mod middleware;
pub mod middleware_integration;
pub mod optimization;
pub mod performance;
pub mod performance_profiler;
pub mod pool;
pub mod production;
pub mod property_path_optimizer;
pub mod query_cache;
pub mod realtime_notifications;
pub mod recovery;
pub mod rest_api_v2;
pub mod search;
pub mod security_audit;
pub mod server;
pub mod service_description;
pub mod simd_triple_matcher;
pub mod sparql_protocol;
pub mod sparql_simd_integration;
pub mod store;
pub mod store_ext;
pub mod store_health;
pub mod store_impl;
pub mod streaming;
pub mod streaming_results;
pub mod subquery_optimizer;
pub mod tls;
pub mod tls_rotation;
pub mod vector_search;
pub mod websocket;

// v1.1.0 SPARQL Subscription Protocol
pub mod sparql_subscription;

// v1.1.0 HTTP/2 Server Push for SPARQL results
pub mod http2_push;

// v1.2.0 Structured query audit log
pub mod query_log;

// Additional Production Features
pub mod api_keys;
pub mod audit;
pub mod cdn_static;
pub mod edge_caching;
pub mod ldp;
pub mod load_balancing;
pub mod rate_limit;

// v1.2.0 SPARQL Query Explanation / Plan Visualization
pub mod query_explain;

// v1.2.0 SPARQL Update operation executor
pub mod update_processor;

// v1.5.0 Content-addressed SPARQL result cache
pub mod sparql_result_cache;

// v1.6.0 Token-bucket rate limiter for SPARQL endpoints
pub mod request_limiter;

// v1.7.0 Endpoint health check with configurable probes
pub mod endpoint_health;

// v1.8.0 Authentication middleware (bearer/API-key/session/RBAC)
pub mod auth_middleware;

// v1.9.0 HTTP content negotiation for RDF format selection
pub mod content_negotiation;

// v1.10.0 HTTP request validation for SPARQL endpoints
pub mod request_validator;

// v1.11.0 Dataset lifecycle management (create/delete/backup/restore/list)
pub mod dataset_manager;

// v1.1.0 round 15 HTTP endpoint routing for SPARQL/GraphQL/REST paths
pub mod endpoint_router;

// v1.1.0 round 16 SPARQL query audit logger with ring-buffer and statistics
pub mod query_logger;

pub use ldp::{LdpContainer, LdpRequest, LdpResourceType, LdpResponse, LdpService};

use store::Store;

/// SPARQL HTTP server implementation
pub struct Server {
    addr: SocketAddr,
    store: Store,
    config: config::ServerConfig,
}

impl Server {
    /// Create a new server builder
    pub fn builder() -> ServerBuilder {
        ServerBuilder::new()
    }

    /// Run the server
    pub async fn run(self) -> Result<(), Box<dyn std::error::Error>> {
        let runtime = server::Runtime::new(self.addr, self.store, self.config);
        runtime
            .run()
            .await
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
}

/// Server builder for configuration
pub struct ServerBuilder {
    port: u16,
    host: String,
    dataset_path: Option<String>,
}

impl ServerBuilder {
    pub fn new() -> Self {
        ServerBuilder {
            port: 3030,
            host: "localhost".to_string(),
            dataset_path: None,
        }
    }

    pub fn port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    pub fn dataset_path(mut self, path: impl Into<String>) -> Self {
        self.dataset_path = Some(path.into());
        self
    }

    pub async fn build(self) -> Result<Server, Box<dyn std::error::Error>> {
        let addr: SocketAddr = format!("{}:{}", self.host, self.port).parse()?;
        let store = if let Some(path) = self.dataset_path {
            Store::open(path)?
        } else {
            Store::new()?
        };

        let config = config::ServerConfig::default();

        Ok(Server {
            addr,
            store,
            config,
        })
    }
}

impl Default for ServerBuilder {
    fn default() -> Self {
        Self::new()
    }
}
