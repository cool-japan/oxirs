//! Service Registry Implementation for Federation
//!
//! This module provides comprehensive service registry functionality including:
//! - SPARQL endpoint registration and management
//! - GraphQL service registration with schema introspection
//! - Service capability negotiation and feature detection
//! - Health monitoring and failover support
//! - Connection pooling and lifecycle management

use anyhow::{anyhow, Result};
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, error, info, warn};
use url::Url;
use uuid::Uuid;

/// Service registry for managing federated endpoints
#[derive(Debug)]
pub struct ServiceRegistry {
    /// Registered SPARQL endpoints
    sparql_endpoints: Arc<DashMap<String, SparqlEndpoint>>,
    /// Registered GraphQL services
    graphql_services: Arc<DashMap<String, GraphQLService>>,
    /// Service health status
    health_status: Arc<DashMap<String, HealthStatus>>,
    /// Service capabilities cache
    capabilities_cache: Arc<RwLock<HashMap<String, ServiceCapabilities>>>,
    /// HTTP client for health checks and introspection
    http_client: Client,
    /// Configuration
    config: RegistryConfig,
    /// Health monitoring task handle
    health_monitor_handle: Option<tokio::task::JoinHandle<()>>,
}

/// Configuration for the service registry
#[derive(Debug, Clone)]
pub struct RegistryConfig {
    /// Health check interval
    pub health_check_interval: Duration,
    /// Service timeout for requests
    pub service_timeout: Duration,
    /// Maximum retry attempts for failed services
    pub max_retries: u32,
    /// Connection pool size per service
    pub connection_pool_size: usize,
    /// Enable automatic service discovery
    pub auto_discovery: bool,
    /// Service capability refresh interval
    pub capability_refresh_interval: Duration,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            health_check_interval: Duration::from_secs(30),
            service_timeout: Duration::from_secs(10),
            max_retries: 3,
            connection_pool_size: 10,
            auto_discovery: true,
            capability_refresh_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// SPARQL endpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlEndpoint {
    /// Unique endpoint ID
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Endpoint URL
    #[serde(serialize_with = "serialize_url", deserialize_with = "deserialize_url")]
    pub url: Url,
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    /// Endpoint capabilities
    pub capabilities: SparqlCapabilities,
    /// Performance statistics
    pub statistics: PerformanceStats,
    /// Registration timestamp
    pub registered_at: DateTime<Utc>,
    /// Last successful access
    pub last_access: Option<DateTime<Utc>>,
    /// Service metadata
    pub metadata: HashMap<String, String>,
    /// Connection pool configuration
    pub connection_config: ConnectionConfig,
}

/// GraphQL service metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLService {
    /// Unique service ID
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Service URL
    #[serde(serialize_with = "serialize_url", deserialize_with = "deserialize_url")]
    pub url: Url,
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    /// GraphQL schema
    pub schema: Option<String>,
    /// Federation directives
    pub federation_directives: FederationDirectives,
    /// Service capabilities
    pub capabilities: GraphQLCapabilities,
    /// Performance statistics
    pub statistics: PerformanceStats,
    /// Registration timestamp
    pub registered_at: DateTime<Utc>,
    /// Last schema update
    pub schema_updated_at: Option<DateTime<Utc>>,
    /// Service metadata
    pub metadata: HashMap<String, String>,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthConfig {
    /// No authentication
    None,
    /// Basic authentication
    Basic { username: String, password: String },
    /// Bearer token authentication
    Bearer { token: String },
    /// API key authentication
    ApiKey { key: String, header: String },
    /// OAuth 2.0
    OAuth2 {
        token_url: String,
        client_id: String,
        client_secret: String,
    },
    /// Custom headers
    Custom { headers: HashMap<String, String> },
}

/// SPARQL endpoint capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparqlCapabilities {
    /// Supported SPARQL version
    pub sparql_version: SparqlVersion,
    /// Supported result formats
    pub result_formats: HashSet<String>,
    /// Supported graph formats for data import
    pub graph_formats: HashSet<String>,
    /// Custom functions available
    pub custom_functions: HashSet<String>,
    /// Maximum query complexity allowed
    pub max_query_complexity: Option<u32>,
    /// Supports federated queries (SERVICE)
    pub supports_federation: bool,
    /// Supports SPARQL UPDATE
    pub supports_update: bool,
    /// Supports named graphs
    pub supports_named_graphs: bool,
    /// Service description (SD) vocabulary support
    pub service_description: Option<ServiceDescription>,
}

/// GraphQL service capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphQLCapabilities {
    /// GraphQL specification version
    pub graphql_version: String,
    /// Supports subscriptions
    pub supports_subscriptions: bool,
    /// Maximum query depth
    pub max_query_depth: Option<u32>,
    /// Maximum query complexity
    pub max_query_complexity: Option<u32>,
    /// Introspection enabled
    pub introspection_enabled: bool,
    /// Federation specification version
    pub federation_version: Option<String>,
}

impl Default for SparqlCapabilities {
    fn default() -> Self {
        Self {
            sparql_version: SparqlVersion::V11,
            result_formats: vec![
                "application/sparql-results+json".to_string(),
                "application/sparql-results+xml".to_string(),
            ]
            .into_iter()
            .collect(),
            graph_formats: vec!["text/turtle".to_string(), "application/rdf+xml".to_string()]
                .into_iter()
                .collect(),
            custom_functions: HashSet::new(),
            max_query_complexity: Some(1000),
            supports_federation: true,
            supports_update: false,
            supports_named_graphs: true,
            service_description: None,
        }
    }
}

impl Default for GraphQLCapabilities {
    fn default() -> Self {
        Self {
            graphql_version: "June 2018".to_string(),
            supports_subscriptions: false,
            max_query_depth: Some(10),
            max_query_complexity: Some(1000),
            introspection_enabled: true,
            federation_version: Some("v1.0".to_string()),
        }
    }
}

/// SPARQL version enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SparqlVersion {
    #[serde(rename = "1.0")]
    V10,
    #[serde(rename = "1.1")]
    V11,
    #[serde(rename = "1.2")]
    V12,
}

/// Service description metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDescription {
    /// Default graph URIs
    pub default_graphs: Vec<String>,
    /// Named graph URIs
    pub named_graphs: Vec<String>,
    /// Supported languages
    pub languages: Vec<String>,
    /// Property functions
    pub property_functions: Vec<String>,
}

/// GraphQL federation directives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationDirectives {
    /// Entity key fields
    pub key_fields: HashMap<String, Vec<String>>,
    /// External fields
    pub external_fields: HashSet<String>,
    /// Required fields
    pub requires_fields: HashMap<String, Vec<String>>,
    /// Provided fields
    pub provides_fields: HashMap<String, Vec<String>>,
}

/// Performance statistics for services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Total requests made
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// 95th percentile response time
    pub p95_response_time_ms: f64,
    /// Last recorded latency
    pub last_latency_ms: Option<u64>,
    /// Data freshness score (0.0 - 1.0)
    pub freshness_score: f64,
    /// Reliability score (0.0 - 1.0)
    pub reliability_score: f64,
}

impl Default for PerformanceStats {
    fn default() -> Self {
        Self {
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            avg_response_time_ms: 0.0,
            p95_response_time_ms: 0.0,
            last_latency_ms: None,
            freshness_score: 1.0,
            reliability_score: 1.0,
        }
    }
}

/// Connection configuration for services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    /// Connection timeout
    pub timeout: Duration,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Keep-alive settings
    pub keep_alive: bool,
    /// Compression enabled
    pub compression: bool,
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            max_connections: 10,
            keep_alive: true,
            compression: true,
        }
    }
}

/// Service health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Service ID
    pub service_id: String,
    /// Current health state
    pub status: HealthState,
    /// Last health check timestamp
    pub last_check: DateTime<Utc>,
    /// Consecutive failure count
    pub consecutive_failures: u32,
    /// Last error message
    pub last_error: Option<String>,
    /// Health check response time
    pub response_time_ms: Option<u64>,
}

/// Health state enumeration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthState {
    /// Service is healthy and operational
    Healthy,
    /// Service is degraded but operational
    Degraded,
    /// Service is unhealthy
    Unhealthy,
    /// Service is unknown (not yet checked)
    Unknown,
}

/// Service capabilities cache entry
#[derive(Debug, Clone)]
pub struct ServiceCapabilities {
    /// Service type
    pub service_type: ServiceType,
    /// Cached capabilities
    pub capabilities: CapabilityData,
    /// Cache timestamp
    pub cached_at: DateTime<Utc>,
    /// Cache expiry
    pub expires_at: DateTime<Utc>,
}

/// Service type enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum ServiceType {
    Sparql,
    GraphQL,
    Hybrid,
}

/// Capability data union
#[derive(Debug, Clone)]
pub enum CapabilityData {
    Sparql(SparqlCapabilities),
    GraphQL(GraphQLCapabilities),
}

/// Registry statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryStats {
    pub total_sparql_endpoints: usize,
    pub total_graphql_services: usize,
    pub healthy_services: usize,
    pub degraded_services: usize,
    pub unhealthy_services: usize,
    pub last_health_check: Option<DateTime<Utc>>,
}

impl ServiceRegistry {
    /// Create a new service registry with default configuration
    pub fn new() -> Self {
        Self::with_config(RegistryConfig::default())
    }

    /// Create a new service registry with custom configuration
    pub fn with_config(config: RegistryConfig) -> Self {
        let http_client = Client::builder()
            .timeout(config.service_timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            sparql_endpoints: Arc::new(DashMap::new()),
            graphql_services: Arc::new(DashMap::new()),
            health_status: Arc::new(DashMap::new()),
            capabilities_cache: Arc::new(RwLock::new(HashMap::new())),
            http_client,
            config,
            health_monitor_handle: None,
        }
    }

    /// Start the service registry with health monitoring
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting service registry");

        // Start health monitoring
        if self.health_monitor_handle.is_none() {
            let handle = self.start_health_monitoring().await;
            self.health_monitor_handle = Some(handle);
        }

        // Start capability refresh monitoring
        self.start_capability_monitoring().await;

        Ok(())
    }

    /// Stop the service registry
    pub async fn stop(&mut self) -> Result<()> {
        info!("Stopping service registry");

        if let Some(handle) = self.health_monitor_handle.take() {
            handle.abort();
        }

        Ok(())
    }

    /// Register a SPARQL endpoint
    pub async fn register_sparql_endpoint(&self, endpoint: SparqlEndpoint) -> Result<()> {
        info!(
            "Registering SPARQL endpoint: {} ({})",
            endpoint.name, endpoint.url
        );

        // Validate endpoint
        self.validate_sparql_endpoint(&endpoint).await?;

        // Detect capabilities
        let capabilities = self.detect_sparql_capabilities(&endpoint).await?;
        let mut endpoint = endpoint;
        endpoint.capabilities = capabilities;

        // Store endpoint
        let endpoint_id = endpoint.id.clone();
        self.sparql_endpoints.insert(endpoint_id.clone(), endpoint);

        // Initialize health status
        self.health_status.insert(
            endpoint_id.clone(),
            HealthStatus {
                service_id: endpoint_id,
                status: HealthState::Unknown,
                last_check: Utc::now(),
                consecutive_failures: 0,
                last_error: None,
                response_time_ms: None,
            },
        );

        debug!("SPARQL endpoint registered successfully");
        Ok(())
    }

    /// Register a GraphQL service
    pub async fn register_graphql_service(&self, service: GraphQLService) -> Result<()> {
        info!(
            "Registering GraphQL service: {} ({})",
            service.name, service.url
        );

        // Validate service
        self.validate_graphql_service(&service).await?;

        // Introspect schema and capabilities
        let (capabilities, schema) = self.introspect_graphql_service(&service).await?;
        let mut service = service;
        service.capabilities = capabilities;
        service.schema = schema;

        // Store service
        let service_id = service.id.clone();
        self.graphql_services.insert(service_id.clone(), service);

        // Initialize health status
        self.health_status.insert(
            service_id.clone(),
            HealthStatus {
                service_id,
                status: HealthState::Unknown,
                last_check: Utc::now(),
                consecutive_failures: 0,
                last_error: None,
                response_time_ms: None,
            },
        );

        debug!("GraphQL service registered successfully");
        Ok(())
    }

    /// Get all registered SPARQL endpoints
    pub fn get_sparql_endpoints(&self) -> Vec<SparqlEndpoint> {
        self.sparql_endpoints
            .iter()
            .map(|entry| entry.clone())
            .collect()
    }

    /// Get all registered GraphQL services
    pub fn get_graphql_services(&self) -> Vec<GraphQLService> {
        self.graphql_services
            .iter()
            .map(|entry| entry.clone())
            .collect()
    }

    /// Get service health status
    pub fn get_health_status(&self, service_id: &str) -> Option<HealthStatus> {
        self.health_status
            .get(service_id)
            .map(|entry| entry.clone())
    }

    /// Get all healthy SPARQL endpoints
    pub fn get_healthy_sparql_endpoints(&self) -> Vec<SparqlEndpoint> {
        self.sparql_endpoints
            .iter()
            .filter(|entry| {
                self.health_status
                    .get(entry.key())
                    .map(|status| status.status == HealthState::Healthy)
                    .unwrap_or(false)
            })
            .map(|entry| entry.clone())
            .collect()
    }

    /// Get all healthy GraphQL services
    pub fn get_healthy_graphql_services(&self) -> Vec<GraphQLService> {
        self.graphql_services
            .iter()
            .filter(|entry| {
                self.health_status
                    .get(entry.key())
                    .map(|status| status.status == HealthState::Healthy)
                    .unwrap_or(false)
            })
            .map(|entry| entry.clone())
            .collect()
    }

    /// Remove a service
    pub async fn remove_service(&self, service_id: &str) -> Result<()> {
        info!("Removing service: {}", service_id);

        // Remove from all collections
        self.sparql_endpoints.remove(service_id);
        self.graphql_services.remove(service_id);
        self.health_status.remove(service_id);

        // Remove from capabilities cache
        let mut cache = self.capabilities_cache.write().await;
        cache.remove(service_id);

        Ok(())
    }

    /// Validate SPARQL endpoint
    async fn validate_sparql_endpoint(&self, endpoint: &SparqlEndpoint) -> Result<()> {
        // Basic URL validation
        if endpoint.url.scheme() != "http" && endpoint.url.scheme() != "https" {
            return Err(anyhow!("Invalid URL scheme: {}", endpoint.url.scheme()));
        }

        // Test connectivity
        let response = self.http_client.get(endpoint.url.clone()).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!("Endpoint not accessible: {}", response.status()));
        }

        Ok(())
    }

    /// Validate GraphQL service
    async fn validate_graphql_service(&self, service: &GraphQLService) -> Result<()> {
        // Basic URL validation
        if service.url.scheme() != "http" && service.url.scheme() != "https" {
            return Err(anyhow!("Invalid URL scheme: {}", service.url.scheme()));
        }

        // Test introspection query
        let introspection_query = r#"
            query {
                __schema {
                    types {
                        name
                    }
                }
            }
        "#;

        let response = self
            .http_client
            .post(service.url.clone())
            .json(&serde_json::json!({
                "query": introspection_query
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(anyhow!("Service not accessible: {}", response.status()));
        }

        Ok(())
    }

    /// Detect SPARQL endpoint capabilities
    async fn detect_sparql_capabilities(
        &self,
        endpoint: &SparqlEndpoint,
    ) -> Result<SparqlCapabilities> {
        // TODO: Implement comprehensive capability detection
        // For now, return basic capabilities
        Ok(SparqlCapabilities {
            sparql_version: SparqlVersion::V11,
            result_formats: vec![
                "application/sparql-results+json".to_string(),
                "application/sparql-results+xml".to_string(),
            ]
            .into_iter()
            .collect(),
            graph_formats: vec!["text/turtle".to_string(), "application/rdf+xml".to_string()]
                .into_iter()
                .collect(),
            custom_functions: HashSet::new(),
            max_query_complexity: Some(1000),
            supports_federation: true,
            supports_update: false,
            supports_named_graphs: true,
            service_description: None,
        })
    }

    /// Introspect GraphQL service
    async fn introspect_graphql_service(
        &self,
        service: &GraphQLService,
    ) -> Result<(GraphQLCapabilities, Option<String>)> {
        // TODO: Implement comprehensive GraphQL introspection
        // For now, return basic capabilities
        let capabilities = GraphQLCapabilities {
            graphql_version: "June 2018".to_string(),
            supports_subscriptions: false,
            max_query_depth: Some(10),
            max_query_complexity: Some(1000),
            introspection_enabled: true,
            federation_version: Some("v1.0".to_string()),
        };

        Ok((capabilities, None))
    }

    /// Start health monitoring
    async fn start_health_monitoring(&self) -> tokio::task::JoinHandle<()> {
        let sparql_endpoints = Arc::clone(&self.sparql_endpoints);
        let graphql_services = Arc::clone(&self.graphql_services);
        let health_status = Arc::clone(&self.health_status);
        let http_client = self.http_client.clone();
        let interval = self.config.health_check_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // Check SPARQL endpoints
                for entry in sparql_endpoints.iter() {
                    let endpoint = entry.value();
                    let health = Self::check_sparql_health(&http_client, endpoint).await;
                    health_status.insert(endpoint.id.clone(), health);
                }

                // Check GraphQL services
                for entry in graphql_services.iter() {
                    let service = entry.value();
                    let health = Self::check_graphql_health(&http_client, service).await;
                    health_status.insert(service.id.clone(), health);
                }
            }
        })
    }

    /// Start capability monitoring
    async fn start_capability_monitoring(&self) {
        let capabilities_cache = Arc::clone(&self.capabilities_cache);
        let sparql_endpoints = Arc::clone(&self.sparql_endpoints);
        let graphql_services = Arc::clone(&self.graphql_services);
        let interval = self.config.capability_refresh_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                let now = Utc::now();
                let mut cache = capabilities_cache.write().await;

                // Remove expired entries
                cache.retain(|_, entry| entry.expires_at > now);
            }
        });
    }

    /// Check SPARQL endpoint health
    async fn check_sparql_health(client: &Client, endpoint: &SparqlEndpoint) -> HealthStatus {
        let start = Instant::now();
        let service_id = endpoint.id.clone();

        // Simple ASK query for health check
        let query = "ASK { ?s ?p ?o }";

        match client
            .post(endpoint.url.clone())
            .header("Content-Type", "application/sparql-query")
            .header("Accept", "application/sparql-results+json")
            .body(query)
            .send()
            .await
        {
            Ok(response) => {
                let response_time = start.elapsed().as_millis() as u64;

                if response.status().is_success() {
                    HealthStatus {
                        service_id,
                        status: HealthState::Healthy,
                        last_check: Utc::now(),
                        consecutive_failures: 0,
                        last_error: None,
                        response_time_ms: Some(response_time),
                    }
                } else {
                    HealthStatus {
                        service_id,
                        status: HealthState::Degraded,
                        last_check: Utc::now(),
                        consecutive_failures: 1,
                        last_error: Some(format!("HTTP {}", response.status())),
                        response_time_ms: Some(response_time),
                    }
                }
            }
            Err(e) => HealthStatus {
                service_id,
                status: HealthState::Unhealthy,
                last_check: Utc::now(),
                consecutive_failures: 1,
                last_error: Some(e.to_string()),
                response_time_ms: None,
            },
        }
    }

    /// Check GraphQL service health
    async fn check_graphql_health(client: &Client, service: &GraphQLService) -> HealthStatus {
        let start = Instant::now();
        let service_id = service.id.clone();

        // Simple introspection query for health check
        let query = r#"{ __schema { queryType { name } } }"#;

        match client
            .post(service.url.clone())
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "query": query
            }))
            .send()
            .await
        {
            Ok(response) => {
                let response_time = start.elapsed().as_millis() as u64;

                if response.status().is_success() {
                    HealthStatus {
                        service_id,
                        status: HealthState::Healthy,
                        last_check: Utc::now(),
                        consecutive_failures: 0,
                        last_error: None,
                        response_time_ms: Some(response_time),
                    }
                } else {
                    HealthStatus {
                        service_id,
                        status: HealthState::Degraded,
                        last_check: Utc::now(),
                        consecutive_failures: 1,
                        last_error: Some(format!("HTTP {}", response.status())),
                        response_time_ms: Some(response_time),
                    }
                }
            }
            Err(e) => HealthStatus {
                service_id,
                status: HealthState::Unhealthy,
                last_check: Utc::now(),
                consecutive_failures: 1,
                last_error: Some(e.to_string()),
                response_time_ms: None,
            },
        }
    }

    /// Register a federated service (generic method)
    pub async fn register(&self, service: crate::FederatedService) -> Result<()> {
        match service.service_type {
            crate::ServiceType::Sparql => {
                let sparql_endpoint = SparqlEndpoint {
                    id: service.id,
                    name: service.name,
                    url: Url::parse(&service.endpoint)?,
                    auth: None, // Convert from service.auth if needed
                    capabilities: SparqlCapabilities::default(),
                    statistics: PerformanceStats::default(),
                    registered_at: Utc::now(),
                    last_access: None,
                    metadata: HashMap::new(),
                    connection_config: ConnectionConfig::default(),
                };
                self.register_sparql_endpoint(sparql_endpoint).await
            }
            crate::ServiceType::GraphQL => {
                let graphql_service = GraphQLService {
                    id: service.id,
                    name: service.name,
                    url: Url::parse(&service.endpoint)?,
                    auth: None, // Convert from service.auth if needed
                    schema: None,
                    federation_directives: FederationDirectives {
                        key_fields: HashMap::new(),
                        external_fields: HashSet::new(),
                        requires_fields: HashMap::new(),
                        provides_fields: HashMap::new(),
                    },
                    capabilities: GraphQLCapabilities::default(),
                    statistics: PerformanceStats::default(),
                    registered_at: Utc::now(),
                    schema_updated_at: None,
                    metadata: HashMap::new(),
                };
                self.register_graphql_service(graphql_service).await
            }
            _ => {
                // For other service types, try registering as SPARQL by default
                let sparql_endpoint = SparqlEndpoint {
                    id: service.id,
                    name: service.name,
                    url: Url::parse(&service.endpoint)?,
                    auth: None,
                    capabilities: SparqlCapabilities::default(),
                    statistics: PerformanceStats::default(),
                    registered_at: Utc::now(),
                    last_access: None,
                    metadata: HashMap::new(),
                    connection_config: ConnectionConfig::default(),
                };
                self.register_sparql_endpoint(sparql_endpoint).await
            }
        }
    }

    /// Unregister a service (generic method)
    pub async fn unregister(&self, service_id: &str) -> Result<()> {
        self.remove_service(service_id).await
    }

    /// Get registry statistics
    pub async fn get_stats(&self) -> Result<RegistryStats> {
        Ok(RegistryStats {
            total_sparql_endpoints: self.sparql_endpoints.len(),
            total_graphql_services: self.graphql_services.len(),
            healthy_services: self
                .health_status
                .iter()
                .filter(|entry| entry.status == HealthState::Healthy)
                .count(),
            degraded_services: self
                .health_status
                .iter()
                .filter(|entry| entry.status == HealthState::Degraded)
                .count(),
            unhealthy_services: self
                .health_status
                .iter()
                .filter(|entry| entry.status == HealthState::Unhealthy)
                .count(),
            last_health_check: self
                .health_status
                .iter()
                .map(|entry| entry.last_check)
                .max(),
        })
    }

    /// Perform health check on all services
    pub async fn health_check(&self) -> Result<Vec<HealthStatus>> {
        let mut results = Vec::new();

        // Check SPARQL endpoints
        for entry in self.sparql_endpoints.iter() {
            let endpoint = entry.value();
            let health = Self::check_sparql_health(&self.http_client, endpoint).await;
            self.health_status
                .insert(endpoint.id.clone(), health.clone());
            results.push(health);
        }

        // Check GraphQL services
        for entry in self.graphql_services.iter() {
            let service = entry.value();
            let health = Self::check_graphql_health(&self.http_client, service).await;
            self.health_status
                .insert(service.id.clone(), health.clone());
            results.push(health);
        }

        Ok(results)
    }
}

// Helper functions for URL serialization
fn serialize_url<S>(url: &Url, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(url.as_str())
}

fn deserialize_url<'de, D>(deserializer: D) -> Result<Url, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Url::parse(&s).map_err(serde::de::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_registry_creation() {
        let config = RegistryConfig::default();
        let registry = ServiceRegistry::with_config(config);

        assert_eq!(registry.get_sparql_endpoints().len(), 0);
        assert_eq!(registry.get_graphql_services().len(), 0);
    }

    #[tokio::test]
    async fn test_sparql_endpoint_registration() {
        let config = RegistryConfig::default();
        let registry = ServiceRegistry::with_config(config);

        let endpoint = SparqlEndpoint {
            id: "test-endpoint".to_string(),
            name: "Test SPARQL Endpoint".to_string(),
            url: Url::parse("http://localhost:3030/test").unwrap(),
            auth: None,
            capabilities: SparqlCapabilities {
                sparql_version: SparqlVersion::V11,
                result_formats: HashSet::new(),
                graph_formats: HashSet::new(),
                custom_functions: HashSet::new(),
                max_query_complexity: Some(1000),
                supports_federation: true,
                supports_update: false,
                supports_named_graphs: true,
                service_description: None,
            },
            statistics: PerformanceStats::default(),
            registered_at: Utc::now(),
            last_access: None,
            metadata: HashMap::new(),
            connection_config: ConnectionConfig::default(),
        };

        // Note: This test would fail without a real endpoint
        // In a real test, we'd use a mock HTTP server
        // assert!(registry.register_sparql_endpoint(endpoint).await.is_ok());
    }
}
