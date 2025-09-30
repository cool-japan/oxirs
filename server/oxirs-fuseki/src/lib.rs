//! # OxiRS Fuseki - SPARQL HTTP Server
//!
//! [![Version](https://img.shields.io/badge/version-0.1.0--alpha.1-orange)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-fuseki/badge.svg)](https://docs.rs/oxirs-fuseki)
//!
//! **Status**: Alpha Release (v0.1.0-alpha.1)
//! ⚠️ APIs may change. Not recommended for production use.
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

pub mod aggregation;
pub mod analytics;
pub mod auth;
pub mod bind_values_enhanced;
pub mod clustering;
pub mod config;
pub mod consciousness;
pub mod error;
pub mod federated_query_optimizer;
pub mod federation;
pub mod graph_analytics;
pub mod handlers;
pub mod metrics;
pub mod optimization;
pub mod performance;
pub mod property_path_optimizer;
pub mod server;
pub mod store;
pub mod streaming;
pub mod subquery_optimizer;
pub mod vector_search;
pub mod websocket;

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
