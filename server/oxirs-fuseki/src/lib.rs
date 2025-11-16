//! # OxiRS Fuseki - SPARQL HTTP Server
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--beta.1-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-fuseki/badge.svg)](https://docs.rs/oxirs-fuseki)
//!
//! **Status**: Beta Release (v0.1.0-beta.1)
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

pub mod admin_ui; // Admin UI enhancements (Beta.2/RC.1)
pub mod aggregation;
pub mod analytics;
pub mod api; // 🆕 REST API endpoints (Phase 3)
pub mod auth;
pub mod backup; // Automatic backup and restore
pub mod batch_execution; // Request batching and parallel execution
pub mod bind_values_enhanced;
pub mod clustering;
pub mod concurrent; // Advanced concurrent request handling
pub mod config;
pub mod connection_pool; // Connection pooling optimization
pub mod consciousness;
pub mod dataset_management; // Enhanced dataset management API
pub mod ddos_protection; // DDoS protection and traffic analysis
pub mod disaster_recovery; // Disaster recovery and failover
pub mod error;
pub mod federated_query_optimizer;
pub mod federation;
pub mod graph_analytics;
pub mod graphql_integration; // GraphQL API integration (Beta.2/RC.1)
pub mod handlers;
pub mod health; // Enhanced health checks
pub mod http_protocol; // HTTP/2 and HTTP/3 support
pub mod k8s_operator; // Kubernetes operator for managing Fuseki instances
pub mod memory_pool; // Memory pooling and optimization
pub mod metrics;
pub mod middleware;
pub mod optimization;
pub mod performance;
pub mod performance_profiler; // Performance profiling tools (Beta.2/RC.1)
pub mod production; // Production hardening features (Beta.1)
pub mod property_path_optimizer;
pub mod query_cache; // Query result caching with intelligent invalidation
pub mod realtime_notifications; // Real-time update notifications (Beta.2/RC.1)
pub mod recovery; // Automatic recovery mechanisms
pub mod rest_api_v2; // REST API v2 with OpenAPI (Beta.2/RC.1)
pub mod security_audit; // Security auditing and vulnerability scanning
pub mod server;
pub mod store;
pub mod store_ext; // Extension trait for Store convenience methods
pub mod store_impl;
pub mod streaming;
pub mod streaming_results; // Memory-efficient result streaming
pub mod subquery_optimizer;
pub mod tls; // TLS/SSL support
pub mod tls_rotation; // TLS certificate rotation
pub mod vector_search;
pub mod websocket;

// v0.1.0 Final - Additional Production Features
pub mod edge_caching;
pub mod load_balancing; // Advanced load balancing strategies // Edge caching integration framework

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
