//! # OxiRS Fuseki
//!
//! SPARQL 1.1/1.2 HTTP protocol server with Fuseki-compatible configuration.
//!
//! This crate provides a HTTP server that implements the SPARQL Protocol for RDF
//! with full compatibility with Apache Jena Fuseki configuration formats.
//!
//! ## Features
//!
//! - SPARQL 1.1 Query and Update endpoints
//! - SPARQL 1.2 protocol support (when available)
//! - Fuseki-compatible YAML/JSON configuration
//! - RESTful dataset management API
//! - Basic authentication and authorization
//! - Admin web interface
//!
//! ## Examples
//!
//! ```rust
//! use oxirs_fuseki::Server;
//! 
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let server = Server::builder()
//!     .port(3030)
//!     .dataset_path(\"/data\")
//!     .build()
//!     .await?;
//! 
//! server.run().await?;
//! # Ok(())
//! # }
//! ```

use std::net::SocketAddr;

pub mod error;
pub mod server;
pub mod config;
pub mod handlers;
pub mod auth;
pub mod metrics;
pub mod performance;
pub mod store;

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
        runtime.run().await
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