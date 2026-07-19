//! # OxiRS Fuseki - SPARQL HTTP Server
//!
//! [![Version](https://img.shields.io/badge/version-0.3.3-blue)](https://github.com/cool-japan/oxirs/releases)
//! [![docs.rs](https://docs.rs/oxirs-fuseki/badge.svg)](https://docs.rs/oxirs-fuseki)
//!
//! **Status**: Production Release (v0.3.3)
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
pub mod bind_values_enhanced_bind;
pub mod bind_values_enhanced_optim;
#[cfg(test)]
mod bind_values_enhanced_tests;
pub mod bind_values_enhanced_types;
pub mod bind_values_enhanced_values;
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
pub mod federated_query_optimizer_exec;
pub mod federated_query_optimizer_planner;
#[cfg(test)]
mod federated_query_optimizer_tests;
pub mod federated_query_optimizer_types;
// Backing implementation modules for the federated query optimizer facade.
pub mod federated_query_executor;
pub mod federated_query_planner;
pub mod federated_query_types;
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

// v0.3.0 API stability harness (mirrors core/oxirs-core/src/api_surface.rs)
pub mod api_surface;

pub use auth::cluster_auth::{
    ClusterAuthConfig, ClusterAuthError, ClusterAuthManager, ClusterNodeToken,
    NodeIdentity as ClusterNodeIdentity,
};
pub use auth::ldap_ha::{LdapHaError, LdapHaPool, LdapServer, LdapServerRole, ReadPolicy};
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

    /// Assemble the `AppState` that `run()` would eventually hand to
    /// `Runtime::build_app`, without binding a TCP listener or running
    /// `Runtime::initialize_services()`'s background service startup.
    ///
    /// This is the seam Task 4(b) needs: it exists purely so the
    /// `ServerBuilder::config()` -> `Server::config` -> `AppState::config`
    /// threading contract has direct, fast test coverage (see
    /// `server_builder_tests` below) instead of requiring a live network
    /// bind to observe. Uses the same `build_minimal_app_state` helper the
    /// production-router regression test uses, so the config field really is
    /// the one `ServerBuilder::build()` produced -- a revert of `build()`
    /// back to always using `ServerConfig::default()` makes the config this
    /// method exposes (and therefore the test asserting on it) go back to
    /// reporting no write protection.
    #[doc(hidden)]
    pub fn to_minimal_app_state(&self) -> server::AppState {
        server::build_minimal_app_state(self.store.clone(), self.config.clone())
    }
}

/// Server builder for configuration
pub struct ServerBuilder {
    port: u16,
    host: String,
    dataset_path: Option<String>,
    config: Option<config::ServerConfig>,
}

impl ServerBuilder {
    pub fn new() -> Self {
        ServerBuilder {
            port: 3030,
            host: "localhost".to_string(),
            dataset_path: None,
            config: None,
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

    /// Supply the fully-loaded [`ServerConfig`] (datasets, `read_only` flags,
    /// security, etc.). Without this, `build()` falls back to
    /// `ServerConfig::default()` — an empty datasets map — which is what
    /// silently dropped every `read_only` setting before: the loaded config
    /// never reached `AppState`, so write-protection could not be enforced.
    pub fn config(mut self, config: config::ServerConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub async fn build(self) -> Result<Server, Box<dyn std::error::Error>> {
        // Install the Pure Rust crypto provider as the process default before any
        // rustls-backed component runs (TLS termination, OAuth via reqwest, the
        // metrics push-gateway). Idempotent: a prior install elsewhere is fine.
        let _ = rustls::crypto::CryptoProvider::install_default((*oxitls::pure_provider()).clone());

        let addr: SocketAddr = format!("{}:{}", self.host, self.port).parse()?;
        let store = if let Some(path) = self.dataset_path {
            Store::open(path)?
        } else {
            Store::new()?
        };

        // Use the fully-loaded config when one was supplied (so `datasets[..].read_only`
        // reaches `AppState` and the update handlers can enforce it); otherwise fall
        // back to defaults. Keep the socket address the builder resolved (CLI/host/port
        // overrides win) rather than silently re-deriving it from the config.
        let config = self.config.unwrap_or_else(config::ServerConfig::default);

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

#[cfg(test)]
mod server_builder_tests {
    use super::*;
    use crate::config::{DatasetConfig, ServerConfig};

    fn read_only_dataset_config(name: &str) -> DatasetConfig {
        DatasetConfig {
            name: name.to_string(),
            location: String::new(),
            read_only: true,
            text_index: None,
            shacl_shapes: vec![],
            services: vec![],
            access_control: None,
            backup: None,
        }
    }

    /// Task 4(b) regression: `ServerBuilder::config()` must thread all the
    /// way into `AppState`. This test builds a `ServerConfig` with
    /// `datasets["default"].read_only = true`, drives it through the public
    /// `ServerBuilder` API (no port bind, no `initialize_services()`), and
    /// asserts the resulting `AppState` reports the default dataset
    /// read-only. A revert of `ServerBuilder::build()` back to always using
    /// `ServerConfig::default()` (dropping the caller-supplied config) makes
    /// this fail, since a fresh default config has no datasets at all.
    #[tokio::test]
    async fn test_server_builder_threads_read_only_config_into_app_state() {
        let mut config = ServerConfig::default();
        config
            .datasets
            .insert("default".to_string(), read_only_dataset_config("default"));

        // `.host("127.0.0.1")` is required because `ServerBuilder::new()`'s
        // default host is the *hostname* "localhost", and `build()` parses
        // `"{host}:{port}"` directly as a `SocketAddr` -- `FromStr` for
        // `SocketAddr` requires an IP literal and does not resolve hostnames,
        // so the default would fail to parse here regardless of `.config()`.
        // Irrelevant to what this test is verifying (config threading), so
        // pinned to a literal IP rather than exercised.
        let server = ServerBuilder::new()
            .host("127.0.0.1")
            .config(config)
            .build()
            .await
            .expect("build() must succeed without binding a port");

        let state = server.to_minimal_app_state();
        assert!(
            state.is_dataset_read_only("default"),
            "AppState must report the default dataset read-only when ServerBuilder \
             was given a config with datasets[\"default\"].read_only = true"
        );
    }

    /// Companion case: without an explicit `.config(..)` call, `build()`
    /// falls back to `ServerConfig::default()`, which has no datasets --
    /// nothing should be reported read-only.
    #[tokio::test]
    async fn test_server_builder_without_config_has_no_read_only_datasets() {
        let server = ServerBuilder::new()
            .host("127.0.0.1")
            .build()
            .await
            .expect("build() must succeed without binding a port");

        let state = server.to_minimal_app_state();
        assert!(!state.is_dataset_read_only("default"));
    }
}
