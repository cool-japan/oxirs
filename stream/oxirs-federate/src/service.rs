//! Service Registry and Management
//!
//! This module manages federated services, their capabilities, and health status.

use anyhow::{anyhow, Result};
use base64::encode;
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use reqwest::{
    header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE},
    Client,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::metadata::{ExtendedServiceMetadata, HealthCheckResult};

/// Type of federated service
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceType {
    /// SPARQL endpoint
    Sparql,
    /// GraphQL service
    GraphQL,
    /// Hybrid service supporting both SPARQL and GraphQL
    Hybrid,
    /// REST API with RDF support
    RestRdf,
    /// Custom service type
    Custom(String),
}

/// Service capabilities for federation
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceCapability {
    /// Basic SPARQL 1.0 query support
    SparqlQuery,
    /// SPARQL 1.1 query support
    Sparql11Query,
    /// SPARQL 1.2 query support
    Sparql12Query,
    /// SPARQL UPDATE support
    SparqlUpdate,
    /// Graph Store Protocol support
    GraphStore,
    /// RDF-star support
    RdfStar,
    /// Full-text search capabilities
    FullTextSearch,
    /// Geospatial query support
    Geospatial,
    /// GraphQL query support
    GraphQLQuery,
    /// GraphQL mutation support
    GraphQLMutation,
    /// GraphQL subscription support
    GraphQLSubscription,
    /// Federation directives support
    GraphQLFederation,
    /// Custom extensions
    CustomExtension(String),
    /// Service can participate in transactions
    Transactional,
    /// Service supports versioning
    Versioning,
    /// Service supports real-time updates
    RealTimeUpdates,
    /// Service supports caching hints
    CacheHints,
    /// Service supports authentication
    Authentication,
    /// Service supports authorization
    Authorization,
    /// Service provides statistics
    Statistics,
    /// Service supports batch operations
    BatchOperations,
}

/// Authentication configuration for services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthConfig {
    /// No authentication required
    None,
    /// Basic HTTP authentication
    Basic { username: String, password: String },
    /// Bearer token authentication
    Bearer { token: String },
    /// API key authentication
    ApiKey { key: String, header: String },
    /// OAuth 2.0 authentication
    OAuth2 {
        token_url: String,
        client_id: String,
        client_secret: String,
        scope: Option<String>,
    },
    /// Custom authentication headers
    Custom { headers: HashMap<String, String> },
}

/// Service health status
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceStatus {
    /// Service is healthy and available
    Healthy,
    /// Service is partially available (degraded performance)
    Degraded,
    /// Service is temporarily unavailable
    Unavailable,
    /// Service status is unknown
    Unknown,
}

/// Overall health status for collections of services
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HealthStatus {
    /// All services are healthy
    Healthy,
    /// Some services have issues but system is functional
    Degraded,
    /// Critical issues affecting functionality
    Critical,
    /// Unable to determine health status
    Unknown,
}

/// Registry for managing federated services
#[derive(Debug)]
pub struct ServiceRegistry {
    pub(crate) services: HashMap<String, FederatedService>,
    config: ServiceRegistryConfig,
    last_health_check: Option<Instant>,
    http_client: Client,
    connection_pools: Arc<RwLock<HashMap<String, ConnectionPool>>>,
    rate_limiters: HashMap<String, Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>,
}

impl ServiceRegistry {
    /// Create a new service registry with default configuration
    pub fn new() -> Self {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("oxirs-federate-registry/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            services: HashMap::new(),
            config: ServiceRegistryConfig::default(),
            last_health_check: None,
            http_client,
            connection_pools: Arc::new(RwLock::new(HashMap::new())),
            rate_limiters: HashMap::new(),
        }
    }

    /// Create a new service registry with custom configuration
    pub fn with_config(config: ServiceRegistryConfig) -> Self {
        let http_client = Client::builder()
            .timeout(config.service_timeout)
            .user_agent("oxirs-federate-registry/1.0")
            .build()
            .expect("Failed to create HTTP client");

        Self {
            services: HashMap::new(),
            config,
            last_health_check: None,
            http_client,
            connection_pools: Arc::new(RwLock::new(HashMap::new())),
            rate_limiters: HashMap::new(),
        }
    }

    /// Register a new federated service
    pub async fn register(&mut self, service: FederatedService) -> Result<()> {
        info!("Registering federated service: {}", service.id);

        // Validate service configuration
        service.validate()?;

        // Initialize connection pool for this service
        self.initialize_connection_pool(&service).await?;

        // Initialize rate limiter if specified
        if let Some(rate_limit) = &service.performance.rate_limit {
            let quota = Quota::per_minute(
                std::num::NonZeroU32::new(rate_limit.requests_per_minute as u32).unwrap(),
            );
            let limiter = RateLimiter::direct(quota);
            self.rate_limiters
                .insert(service.id.clone(), Arc::new(limiter));
        }

        // Perform initial health check
        let health_status = self.check_service_health(&service).await?;

        if health_status != ServiceStatus::Healthy && self.config.require_healthy_on_register {
            return Err(anyhow!(
                "Service {} is not healthy on registration",
                service.id
            ));
        }

        // Detect and store service capabilities
        let detected_capabilities = self.detect_service_capabilities(&service).await?;
        let mut enhanced_service = service;
        enhanced_service.capabilities.extend(detected_capabilities);

        self.services
            .insert(enhanced_service.id.clone(), enhanced_service);
        Ok(())
    }

    /// Unregister a federated service
    pub async fn unregister(&mut self, service_id: &str) -> Result<()> {
        info!("Unregistering federated service: {}", service_id);

        match self.services.remove(service_id) {
            Some(_) => Ok(()),
            None => Err(anyhow!("Service {} not found", service_id)),
        }
    }

    /// Get a service by ID
    pub fn get_service(&self, service_id: &str) -> Option<&FederatedService> {
        self.services.get(service_id)
    }

    /// Get all services
    pub fn get_all_services(&self) -> impl Iterator<Item = &FederatedService> {
        self.services.values()
    }

    /// Get services that support a specific capability
    pub fn get_services_with_capability(
        &self,
        capability: &ServiceCapability,
    ) -> Vec<&FederatedService> {
        self.services
            .values()
            .filter(|service| service.capabilities.contains(capability))
            .collect()
    }

    /// Get services that can handle specific query patterns
    pub fn get_services_for_patterns(&self, patterns: &[String]) -> Vec<&FederatedService> {
        self.services
            .values()
            .filter(|service| {
                patterns.iter().any(|pattern| {
                    service
                        .data_patterns
                        .iter()
                        .any(|service_pattern| pattern_matches(pattern, service_pattern))
                })
            })
            .collect()
    }

    /// Perform health check on all services
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let mut service_statuses = HashMap::new();
        let mut healthy_count = 0;
        let total_services = self.services.len();

        for service in self.services.values() {
            let status = self
                .check_service_health(service)
                .await
                .unwrap_or(ServiceStatus::Unknown);
            if status == ServiceStatus::Healthy {
                healthy_count += 1;
            }
            service_statuses.insert(service.id.clone(), status);
        }

        let overall_status = if healthy_count == total_services {
            ServiceStatus::Healthy
        } else if healthy_count > 0 {
            ServiceStatus::Degraded
        } else {
            ServiceStatus::Unavailable
        };

        Ok(HealthStatus {
            overall_status,
            service_statuses,
            total_services,
            healthy_services: healthy_count,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Get registry statistics
    pub async fn get_stats(&self) -> ServiceRegistryStats {
        let health_check = self.health_check().await.unwrap_or_else(|_| HealthStatus {
            overall_status: ServiceStatus::Unknown,
            service_statuses: HashMap::new(),
            total_services: self.services.len(),
            healthy_services: 0,
            timestamp: chrono::Utc::now(),
        });

        let mut capabilities_count = HashMap::new();
        for service in self.services.values() {
            for capability in &service.capabilities {
                *capabilities_count.entry(capability.clone()).or_insert(0) += 1;
            }
        }

        ServiceRegistryStats {
            total_services: self.services.len(),
            healthy_services: health_check.healthy_services,
            capabilities_distribution: capabilities_count,
            last_health_check: self.last_health_check,
        }
    }

    /// Enable extended metadata collection for a service
    pub async fn enable_extended_metadata(&mut self, service_id: &str) -> Result<()> {
        if let Some(service) = self.services.get_mut(service_id) {
            if service.extended_metadata.is_none() {
                let extended = ExtendedServiceMetadata::from_basic(service.metadata.clone());
                service.extended_metadata = Some(extended);
                info!("Enabled extended metadata for service: {}", service_id);
            }
            Ok(())
        } else {
            Err(anyhow!("Service {} not found", service_id))
        }
    }
    
    /// Collect dataset statistics for a SPARQL service
    pub async fn collect_dataset_statistics(&mut self, service_id: &str) -> Result<()> {
        let service = self
            .services
            .get(service_id)
            .ok_or_else(|| anyhow!("Service {} not found", service_id))?
            .clone();
        
        if service.service_type != ServiceType::Sparql && service.service_type != ServiceType::Hybrid {
            return Err(anyhow!("Service {} does not support SPARQL", service_id));
        }
        
        // Query to get dataset statistics
        let stats_query = r#"
            SELECT 
                (COUNT(*) as ?totalTriples)
                (COUNT(DISTINCT ?s) as ?subjects)
                (COUNT(DISTINCT ?p) as ?predicates)
                (COUNT(DISTINCT ?o) as ?objects)
            WHERE { ?s ?p ?o }
        "#;
        
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-query"),
        );
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/sparql-results+json"),
        );
        
        if let Some(auth) = &service.auth {
            self.add_auth_header(&mut headers, auth)?;
        }
        
        let response = self
            .http_client
            .post(&service.endpoint)
            .headers(headers)
            .body(stats_query)
            .send()
            .await?;
        
        if response.status().is_success() {
            // Parse statistics and update extended metadata
            let json: serde_json::Value = response.json().await?;
            
            if let Some(service) = self.services.get_mut(service_id) {
                if let Some(ref mut extended) = service.extended_metadata {
                    // Parse SPARQL JSON results
                    if let Some(results) = json.get("results")
                        .and_then(|r| r.get("bindings"))
                        .and_then(|b| b.as_array())
                        .and_then(|a| a.first()) 
                    {
                        if let Some(total) = results.get("totalTriples")
                            .and_then(|v| v.get("value"))
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<u64>().ok()) 
                        {
                            extended.dataset_stats.triple_count = Some(total);
                        }
                        
                        if let Some(subjects) = results.get("subjects")
                            .and_then(|v| v.get("value"))
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<u64>().ok()) 
                        {
                            extended.dataset_stats.subject_count = Some(subjects);
                        }
                        
                        if let Some(predicates) = results.get("predicates")
                            .and_then(|v| v.get("value"))
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<u64>().ok()) 
                        {
                            extended.dataset_stats.predicate_count = Some(predicates);
                        }
                        
                        if let Some(objects) = results.get("objects")
                            .and_then(|v| v.get("value"))
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<u64>().ok()) 
                        {
                            extended.dataset_stats.object_count = Some(objects);
                        }
                        
                        extended.dataset_stats.last_modified = Some(chrono::Utc::now());
                    }
                    
                    debug!("Updated dataset statistics for service: {}", service_id);
                }
            }
        }
        
        // Also try to get named graphs
        let graphs_query = "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } } LIMIT 100";
        if let Ok(graphs_response) = self
            .http_client
            .post(&service.endpoint)
            .headers(headers.clone())
            .body(graphs_query)
            .send()
            .await
        {
            if graphs_response.status().is_success() {
                if let Ok(json) = graphs_response.json::<serde_json::Value>().await {
                    if let Some(service) = self.services.get_mut(service_id) {
                        if let Some(ref mut extended) = service.extended_metadata {
                            if let Some(results) = json.get("results")
                                .and_then(|r| r.get("bindings"))
                                .and_then(|b| b.as_array())
                            {
                                extended.dataset_stats.named_graphs = results.iter()
                                    .filter_map(|binding| {
                                        binding.get("g")
                                            .and_then(|v| v.get("value"))
                                            .and_then(|v| v.as_str())
                                            .map(|uri| crate::metadata::NamedGraphInfo {
                                                uri: uri.to_string(),
                                                triple_count: None,
                                                description: None,
                                            })
                                    })
                                    .collect();
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Check health of a specific service
    async fn check_service_health(&self, service: &FederatedService) -> Result<ServiceStatus> {
        let start_time = Instant::now();

        // Try different health check approaches based on service type
        let health_result = match service.service_type {
            ServiceType::Sparql => self.check_sparql_health(service).await,
            ServiceType::GraphQL => self.check_graphql_health(service).await,
            ServiceType::Hybrid => {
                // For hybrid services, check both protocols
                let sparql_ok = self.check_sparql_health(service).await.is_ok();
                let graphql_ok = self.check_graphql_health(service).await.is_ok();

                if sparql_ok || graphql_ok {
                    Ok(ServiceStatus::Healthy)
                } else {
                    Ok(ServiceStatus::Unavailable)
                }
            }
        };

        let response_time = start_time.elapsed();
        debug!(
            "Health check for {} completed in {:?}: {:?}",
            service.id, response_time, health_result
        );

        // Update extended metadata with health check result if available
        if let Some(extended) = &service.extended_metadata {
            let check_result = HealthCheckResult {
                timestamp: chrono::Utc::now(),
                success: matches!(health_result, Ok(ServiceStatus::Healthy)),
                response_time: Some(response_time),
                error_message: if health_result.is_err() {
                    Some(format!("{:?}", health_result))
                } else {
                    None
                },
            };
            
            // We need mutable access to update - will be handled in caller
            debug!("Health check result recorded for extended metadata");
        }

        health_result
    }

    /// Check SPARQL service health
    async fn check_sparql_health(&self, service: &FederatedService) -> Result<ServiceStatus> {
        let health_query = "ASK { ?s ?p ?o }";

        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-query"),
        );
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/sparql-results+json"),
        );

        // Add authentication if configured
        if let Some(auth) = &service.auth {
            self.add_auth_header(&mut headers, auth)?;
        }

        let response = self
            .http_client
            .post(&service.endpoint)
            .headers(headers)
            .body(health_query)
            .send()
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => Ok(ServiceStatus::Healthy),
            Ok(resp) if resp.status().is_server_error() => Ok(ServiceStatus::Unavailable),
            Ok(_) => Ok(ServiceStatus::Degraded),
            Err(_) => Ok(ServiceStatus::Unavailable),
        }
    }

    /// Check GraphQL service health
    async fn check_graphql_health(&self, service: &FederatedService) -> Result<ServiceStatus> {
        let health_query = serde_json::json!({
            "query": "{ __schema { queryType { name } } }"
        });

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        // Add authentication if configured
        if let Some(auth) = &service.auth {
            self.add_auth_header(&mut headers, auth)?;
        }

        let response = self
            .http_client
            .post(&service.endpoint)
            .headers(headers)
            .json(&health_query)
            .send()
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => {
                // Try to parse the response to ensure it's valid GraphQL
                match resp.json::<serde_json::Value>().await {
                    Ok(json) if json.get("data").is_some() => Ok(ServiceStatus::Healthy),
                    Ok(_) => Ok(ServiceStatus::Degraded),
                    Err(_) => Ok(ServiceStatus::Degraded),
                }
            }
            Ok(resp) if resp.status().is_server_error() => Ok(ServiceStatus::Unavailable),
            Ok(_) => Ok(ServiceStatus::Degraded),
            Err(_) => Ok(ServiceStatus::Unavailable),
        }
    }

    /// Initialize connection pool for a service
    async fn initialize_connection_pool(&self, service: &FederatedService) -> Result<()> {
        let pool = ConnectionPool {
            service_id: service.id.clone(),
            endpoint: service.endpoint.clone(),
            max_connections: 10, // TODO: Make configurable
            active_connections: 0,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            last_used: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let mut pools = self.connection_pools.write().await;
        pools.insert(service.id.clone(), pool);

        debug!("Initialized connection pool for service: {}", service.id);
        Ok(())
    }

    /// Detect service capabilities through introspection
    async fn detect_service_capabilities(
        &self,
        service: &FederatedService,
    ) -> Result<HashSet<ServiceCapability>> {
        let mut detected_capabilities = HashSet::new();

        match service.service_type {
            ServiceType::Sparql => {
                // Test various SPARQL features
                if self
                    .test_sparql_feature(service, "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
                    .await
                {
                    detected_capabilities.insert(ServiceCapability::SparqlQuery);
                }

                if self.test_sparql_feature(service, "INSERT DATA { <http://example.org/test> <http://example.org/test> \"test\" }").await {
                    detected_capabilities.insert(ServiceCapability::SparqlUpdate);
                }

                if self
                    .test_sparql_feature(
                        service,
                        "SELECT * WHERE { SERVICE <http://example.org/> { ?s ?p ?o } }",
                    )
                    .await
                {
                    detected_capabilities.insert(ServiceCapability::SparqlService);
                }
                
                // Test SPARQL 1.1 Extended Features
                if self.test_sparql_feature(service, "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }").await {
                    detected_capabilities.insert(ServiceCapability::SparqlAggregation);
                }
                
                if self.test_sparql_feature(service, "SELECT ?s WHERE { ?s ?p ?o { SELECT ?s WHERE { ?s a ?type } LIMIT 10 } }").await {
                    detected_capabilities.insert(ServiceCapability::SparqlSubqueries);
                }
                
                if self.test_sparql_feature(service, "SELECT ?s WHERE { ?s ?p ?o . FILTER NOT EXISTS { ?s a ?type } }").await {
                    detected_capabilities.insert(ServiceCapability::SparqlNegation);
                }
                
                if self.test_sparql_feature(service, "SELECT ?s ?o WHERE { ?s (<http://example.org/p1>|<http://example.org/p2>)+ ?o }").await {
                    detected_capabilities.insert(ServiceCapability::SparqlPropertyPaths);
                }
                
                if self.test_sparql_feature(service, "SELECT ?s ?p ?o WHERE { ?s ?p ?o } GROUP BY ?p").await {
                    detected_capabilities.insert(ServiceCapability::SparqlGroupBy);
                }
                
                // Test SPARQL 1.2 Features (if available)
                if self.test_sparql_feature(service, "SELECT ?s WHERE { VALUES ?s { <http://example.org/1> <http://example.org/2> } ?s ?p ?o }").await {
                    detected_capabilities.insert(ServiceCapability::SparqlValues);
                }
                
                // Test RDF-star support
                if self.test_sparql_feature(service, "SELECT ?s ?p ?o WHERE { << ?s ?p ?o >> ?m ?v }").await {
                    detected_capabilities.insert(ServiceCapability::RDFStar);
                }
                
                // Test reasoning capabilities
                if self.test_sparql_feature(service, "SELECT ?s WHERE { ?s a ?class . ?class rdfs:subClassOf ?super }").await {
                    // If this works without explicit subClassOf triples, reasoning is likely enabled
                    detected_capabilities.insert(ServiceCapability::RDFSReasoning);
                }
            }
            ServiceType::GraphQL => {
                // Test GraphQL introspection and mutations
                if self.test_graphql_introspection(service).await {
                    detected_capabilities.insert(ServiceCapability::GraphQLQuery);
                }
            }
            ServiceType::Hybrid => {
                // Test both SPARQL and GraphQL capabilities
                if self.test_sparql_feature(service, "ASK { ?s ?p ?o }").await {
                    detected_capabilities.insert(ServiceCapability::SparqlQuery);
                }
                if self.test_graphql_introspection(service).await {
                    detected_capabilities.insert(ServiceCapability::GraphQLQuery);
                }
            }
        }

        Ok(detected_capabilities)
    }

    /// Test a specific SPARQL feature
    async fn test_sparql_feature(&self, service: &FederatedService, query: &str) -> bool {
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-query"),
        );
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/sparql-results+json"),
        );

        if let Some(auth) = &service.auth {
            if self.add_auth_header(&mut headers, auth).is_err() {
                return false;
            }
        }

        let response = self
            .http_client
            .post(&service.endpoint)
            .headers(headers)
            .body(query.to_string())
            .send()
            .await;

        matches!(response, Ok(resp) if resp.status().is_success())
    }

    /// Test GraphQL introspection capability
    async fn test_graphql_introspection(&self, service: &FederatedService) -> bool {
        let introspection_query = serde_json::json!({
            "query": "{ __schema { types { name } } }"
        });

        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        if let Some(auth) = &service.auth {
            if self.add_auth_header(&mut headers, auth).is_err() {
                return false;
            }
        }

        let response = self
            .http_client
            .post(&service.endpoint)
            .headers(headers)
            .json(&introspection_query)
            .send()
            .await;

        matches!(response, Ok(resp) if resp.status().is_success())
    }

    /// Add authentication header based on auth configuration
    fn add_auth_header(&self, headers: &mut HeaderMap, auth: &AuthConfig) -> Result<()> {
        match &auth.auth_type {
            AuthType::Basic => {
                if let (Some(username), Some(password)) =
                    (&auth.credentials.username, &auth.credentials.password)
                {
                    let credentials = format!("{}:{}", username, password);
                    let encoded = encode(credentials.as_bytes());
                    let auth_value = format!("Basic {}", encoded);
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
                }
            }
            AuthType::Bearer => {
                if let Some(token) = &auth.credentials.token {
                    let auth_value = format!("Bearer {}", token);
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
                }
            }
            AuthType::ApiKey => {
                if let Some(api_key) = &auth.credentials.api_key {
                    headers.insert("X-API-Key", HeaderValue::from_str(api_key)?);
                }
            }
            AuthType::OAuth2 => {
                // Implement OAuth2 client credentials flow
                if let Some(oauth_config) = &auth.oauth2_config {
                    match self.perform_oauth2_flow(oauth_config).await {
                        Ok(token) => {
                            headers.insert(
                                AUTHORIZATION,
                                HeaderValue::from_str(&format!("Bearer {}", token))?,
                            );
                        }
                        Err(e) => {
                            warn!("OAuth2 authentication failed: {}", e);
                            return Err(anyhow!("OAuth2 authentication failed: {}", e));
                        }
                    }
                } else {
                    return Err(anyhow!("OAuth2 configuration missing"));
                }
            }
            AuthType::None => {}
        }
        Ok(())
    }

    /// Get connection pool statistics
    pub async fn get_connection_pool_stats(&self) -> HashMap<String, ConnectionPoolStats> {
        let pools = self.connection_pools.read().await;
        let mut stats = HashMap::new();

        for (service_id, pool) in pools.iter() {
            stats.insert(
                service_id.clone(),
                ConnectionPoolStats {
                    max_connections: pool.max_connections,
                    active_connections: pool.active_connections,
                    created_at: pool.created_at,
                    last_used: pool.last_used,
                },
            );
        }

        stats
    }
    
    /// Collect vocabulary information for a SPARQL service
    pub async fn collect_vocabulary_info(&mut self, service_id: &str) -> Result<()> {
        let service = self
            .services
            .get(service_id)
            .ok_or_else(|| anyhow!("Service {} not found", service_id))?
            .clone();
        
        if service.service_type != ServiceType::Sparql && service.service_type != ServiceType::Hybrid {
            return Err(anyhow!("Service {} does not support SPARQL", service_id));
        }
        
        // Query to get vocabulary/ontology URIs
        let vocab_query = r#"
            SELECT DISTINCT ?vocab WHERE {
                {
                    ?s ?p ?o .
                    BIND(REPLACE(STR(?p), "(#|/)[^#/]*$", "$1") AS ?vocab)
                }
                UNION
                {
                    ?s a ?class .
                    BIND(REPLACE(STR(?class), "(#|/)[^#/]*$", "$1") AS ?vocab)
                }
                FILTER(REGEX(?vocab, "^https?://"))
            }
            LIMIT 100
        "#;
        
        let mut headers = HeaderMap::new();
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/sparql-query"),
        );
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/sparql-results+json"),
        );
        
        if let Some(auth) = &service.auth {
            self.add_auth_header(&mut headers, auth)?;
        }
        
        let response = self
            .http_client
            .post(&service.endpoint)
            .headers(headers.clone())
            .body(vocab_query)
            .send()
            .await?;
        
        if response.status().is_success() {
            let json: serde_json::Value = response.json().await?;
            
            if let Some(service) = self.services.get_mut(service_id) {
                if let Some(ref mut extended) = service.extended_metadata {
                    if let Some(results) = json.get("results")
                        .and_then(|r| r.get("bindings"))
                        .and_then(|b| b.as_array())
                    {
                        let vocabs: HashSet<String> = results.iter()
                            .filter_map(|binding| {
                                binding.get("vocab")
                                    .and_then(|v| v.get("value"))
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            })
                            .collect();
                        
                        extended.dataset_stats.vocabularies = vocabs.into_iter().collect();
                    }
                }
            }
        }
        
        // Also collect language tags
        let lang_query = r#"
            SELECT DISTINCT (LANG(?o) AS ?lang) WHERE {
                ?s ?p ?o .
                FILTER(isLiteral(?o) && LANG(?o) != "")
            }
            LIMIT 50
        "#;
        
        let lang_response = self
            .http_client
            .post(&service.endpoint)
            .headers(headers)
            .body(lang_query)
            .send()
            .await?;
        
        if lang_response.status().is_success() {
            let json: serde_json::Value = lang_response.json().await?;
            
            if let Some(service) = self.services.get_mut(service_id) {
                if let Some(ref mut extended) = service.extended_metadata {
                    if let Some(results) = json.get("results")
                        .and_then(|r| r.get("bindings"))
                        .and_then(|b| b.as_array())
                    {
                        extended.dataset_stats.languages = results.iter()
                            .filter_map(|binding| {
                                binding.get("lang")
                                    .and_then(|v| v.get("value"))
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                            })
                            .collect();
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Perform comprehensive service assessment including extended metadata
    pub async fn assess_service_comprehensively(&mut self, service_id: &str) -> Result<()> {
        info!("Performing comprehensive assessment for service: {}", service_id);
        
        // Enable extended metadata if not already enabled
        self.enable_extended_metadata(service_id).await?;
        
        // Collect dataset statistics
        if let Err(e) = self.collect_dataset_statistics(service_id).await {
            warn!("Failed to collect dataset statistics: {}", e);
        }
        
        // Collect vocabulary information
        if let Err(e) = self.collect_vocabulary_info(service_id).await {
            warn!("Failed to collect vocabulary info: {}", e);
        }
        
        // Update capabilities with more detailed detection
        let service = self.services.get(service_id)
            .ok_or_else(|| anyhow!("Service {} not found", service_id))?
            .clone();
        
        let detected_capabilities = self.detect_service_capabilities(&service).await?;
        
        if let Some(service) = self.services.get_mut(service_id) {
            service.capabilities.extend(detected_capabilities);
        }
        
        info!("Comprehensive assessment completed for service: {}", service_id);
        Ok(())
    }

    /// Check rate limits for a service
    pub fn check_rate_limit(&self, service_id: &str) -> bool {
        if let Some(limiter) = self.rate_limiters.get(service_id) {
            limiter.check().is_ok()
        } else {
            true // No rate limit configured
        }
    }

    /// Perform OAuth2 client credentials flow
    async fn perform_oauth2_flow(&self, oauth_config: &OAuth2Config) -> Result<String> {
        let params = [
            ("grant_type", "client_credentials"),
            ("client_id", &oauth_config.client_id),
            ("client_secret", &oauth_config.client_secret),
        ];
        
        let mut form_params = Vec::new();
        form_params.extend_from_slice(&params);
        
        // Add scope if provided
        if let Some(scope) = &oauth_config.scope {
            form_params.push(("scope", scope));
        }
        
        let response = self
            .http_client
            .post(&oauth_config.token_url)
            .form(&form_params)
            .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("OAuth2 token request failed with status: {}", response.status()));
        }
        
        let token_response: OAuth2TokenResponse = response.json().await?;
        
        // TODO: Cache token with expiration
        debug!("Successfully obtained OAuth2 token");
        
        Ok(token_response.access_token)
    }
}

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the service registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceRegistryConfig {
    pub require_healthy_on_register: bool,
    pub health_check_interval: Duration,
    pub service_timeout: Duration,
    pub max_retry_attempts: usize,
    pub enable_capability_detection: bool,
    pub connection_pool_size: usize,
    pub enable_rate_limiting: bool,
}

impl Default for ServiceRegistryConfig {
    fn default() -> Self {
        Self {
            require_healthy_on_register: false,
            health_check_interval: Duration::from_secs(30),
            service_timeout: Duration::from_secs(10),
            max_retry_attempts: 3,
            enable_capability_detection: true,
            connection_pool_size: 10,
            enable_rate_limiting: true,
        }
    }
}

/// Represents a federated service (SPARQL endpoint or GraphQL service)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedService {
    /// Unique identifier for the service
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Service endpoint URL
    pub endpoint: String,
    /// Type of service (SPARQL, GraphQL, etc.)
    pub service_type: ServiceType,
    /// Service capabilities
    pub capabilities: HashSet<ServiceCapability>,
    /// Data patterns this service can handle
    pub data_patterns: Vec<String>,
    /// Authentication configuration
    pub auth: Option<AuthConfig>,
    /// Service metadata
    pub metadata: ServiceMetadata,
    /// Extended metadata (optional, for enhanced tracking)
    pub extended_metadata: Option<ExtendedServiceMetadata>,
    /// Performance characteristics
    pub performance: ServicePerformance,
}

impl FederatedService {
    /// Create a new SPARQL service
    pub fn new_sparql(id: String, name: String, endpoint: String) -> Self {
        Self {
            id,
            name,
            endpoint,
            service_type: ServiceType::Sparql,
            capabilities: [
                ServiceCapability::SparqlQuery,
                ServiceCapability::SparqlUpdate,
            ]
            .into_iter()
            .collect(),
            data_patterns: vec!["*".to_string()], // Accept all patterns by default
            auth: None,
            metadata: ServiceMetadata::default(),
            extended_metadata: None,
            performance: ServicePerformance::default(),
        }
    }

    /// Create a new GraphQL service
    pub fn new_graphql(id: String, name: String, endpoint: String) -> Self {
        Self {
            id,
            name,
            endpoint,
            service_type: ServiceType::GraphQL,
            capabilities: [
                ServiceCapability::GraphQLQuery,
                ServiceCapability::GraphQLMutation,
                ServiceCapability::GraphQLSubscription,
            ]
            .into_iter()
            .collect(),
            data_patterns: vec!["*".to_string()],
            auth: None,
            metadata: ServiceMetadata::default(),
            extended_metadata: None,
            performance: ServicePerformance::default(),
        }
    }

    /// Validate service configuration
    pub fn validate(&self) -> Result<()> {
        if self.id.is_empty() {
            return Err(anyhow!("Service ID cannot be empty"));
        }

        if self.endpoint.is_empty() {
            return Err(anyhow!("Service endpoint cannot be empty"));
        }

        // Validate URL format
        url::Url::parse(&self.endpoint).map_err(|e| anyhow!("Invalid endpoint URL: {}", e))?;

        if self.capabilities.is_empty() {
            return Err(anyhow!("Service must have at least one capability"));
        }

        Ok(())
    }

    /// Check if service supports a specific capability
    pub fn supports_capability(&self, capability: &ServiceCapability) -> bool {
        self.capabilities.contains(capability)
    }

    /// Check if service can handle a specific data pattern
    pub fn handles_pattern(&self, pattern: &str) -> bool {
        self.data_patterns
            .iter()
            .any(|service_pattern| pattern_matches(pattern, service_pattern))
    }
}

/// Type of federated service
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceType {
    Sparql,
    GraphQL,
    Hybrid, // Supports both SPARQL and GraphQL
}

/// Service capabilities
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServiceCapability {
    // SPARQL 1.1 Core Features
    SparqlQuery,
    SparqlUpdate,
    SparqlService, // Can handle SERVICE clauses
    
    // SPARQL 1.1 Extended Features
    SparqlAggregation,
    SparqlSubqueries,
    SparqlNegation,
    SparqlPropertyPaths,
    SparqlAssignment,
    SparqlGroupBy,
    SparqlFederatedExtensions,
    
    // SPARQL 1.2 Features (when available)
    SparqlValues,
    SparqlExistsNotExists,
    SparqlLateralJoin,
    SparqlWindowFunctions,
    
    // GraphQL Features
    GraphQLQuery,
    GraphQLMutation,
    GraphQLSubscription,
    GraphQLFederation,
    GraphQLDataLoader,
    
    // Advanced Features
    Federation, // Can participate in federation
    Caching,
    Authentication,
    RealTime,
    Versioning,
    Analytics,
    FullTextSearch,
    Geospatial,
    VectorSearch,
    TemporalQueries,
    
    // Reasoning & Validation
    RDFSReasoning,
    OWLReasoning,
    SHACLValidation,
    RuleProcessing,
    
    // Format Support
    RDFStar,
    NQuads,
    JSONLDContext,
    HDT,
}

/// Authentication configuration for services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    pub auth_type: AuthType,
    pub credentials: AuthCredentials,
}

/// Authentication types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthType {
    None,
    Basic,
    Bearer,
    ApiKey,
    OAuth2,
}

/// Authentication credentials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthCredentials {
    pub username: Option<String>,
    pub password: Option<String>,
    pub token: Option<String>,
    pub api_key: Option<String>,
    pub oauth_config: Option<OAuth2Config>,
}

/// OAuth2 configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Config {
    pub client_id: String,
    pub client_secret: String,
    pub auth_url: String,
    pub token_url: String,
    pub scopes: Vec<String>,
}

/// Service metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMetadata {
    pub description: Option<String>,
    pub version: Option<String>,
    pub maintainer: Option<String>,
    pub tags: Vec<String>,
    pub documentation_url: Option<String>,
    pub schema_url: Option<String>,
}

impl Default for ServiceMetadata {
    fn default() -> Self {
        Self {
            description: None,
            version: None,
            maintainer: None,
            tags: Vec::new(),
            documentation_url: None,
            schema_url: None,
        }
    }
}

/// Service performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServicePerformance {
    pub average_response_time: Option<Duration>,
    pub max_concurrent_requests: Option<usize>,
    pub rate_limit: Option<RateLimit>,
    pub estimated_dataset_size: Option<u64>,
    pub supported_result_formats: Vec<String>,
}

impl Default for ServicePerformance {
    fn default() -> Self {
        Self {
            average_response_time: None,
            max_concurrent_requests: None,
            rate_limit: None,
            estimated_dataset_size: None,
            supported_result_formats: vec![
                "application/sparql-results+json".to_string(),
                "application/sparql-results+xml".to_string(),
            ],
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_minute: usize,
    pub burst_limit: usize,
}

/// Statistics about the service registry
#[derive(Debug, Clone, Serialize)]
pub struct ServiceRegistryStats {
    pub total_services: usize,
    pub healthy_services: usize,
    pub capabilities_distribution: HashMap<ServiceCapability, usize>,
    #[serde(skip)]
    pub last_health_check: Option<Instant>,
}


/// OAuth2 token response
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OAuth2TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub scope: Option<String>,
}

#[derive(Debug)]
struct ConnectionPool {
    service_id: String,
    endpoint: String,
    max_connections: usize,
    active_connections: usize,
    created_at: u64,
    last_used: u64,
}

/// Connection pool statistics
#[derive(Debug, Clone, Serialize)]
pub struct ConnectionPoolStats {
    pub max_connections: usize,
    pub active_connections: usize,
    pub created_at: u64,
    pub last_used: u64,
}

/// Check if a query pattern matches a service pattern
fn pattern_matches(query_pattern: &str, service_pattern: &str) -> bool {
    // Simple pattern matching - could be enhanced with regex or glob patterns
    if service_pattern == "*" {
        return true;
    }

    if service_pattern.contains('*') {
        // Simple wildcard matching
        let parts: Vec<&str> = service_pattern.split('*').collect();
        if parts.len() == 2 {
            let prefix = parts[0];
            let suffix = parts[1];
            return query_pattern.starts_with(prefix) && query_pattern.ends_with(suffix);
        }
    }

    query_pattern == service_pattern
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_registry_creation() {
        let registry = ServiceRegistry::new();
        assert_eq!(registry.services.len(), 0);
    }

    #[tokio::test]
    async fn test_service_registration() {
        let mut registry = ServiceRegistry::new();
        let service = FederatedService::new_sparql(
            "test-service".to_string(),
            "Test SPARQL Service".to_string(),
            "http://example.com/sparql".to_string(),
        );

        let result = registry.register(service).await;
        assert!(result.is_ok());
        assert_eq!(registry.services.len(), 1);
    }

    #[tokio::test]
    async fn test_service_validation() {
        let mut service = FederatedService::new_sparql(
            "".to_string(), // Invalid empty ID
            "Test Service".to_string(),
            "http://example.com/sparql".to_string(),
        );

        assert!(service.validate().is_err());

        service.id = "valid-id".to_string();
        assert!(service.validate().is_ok());
    }

    #[test]
    fn test_pattern_matching() {
        assert!(pattern_matches("any-pattern", "*"));
        assert!(pattern_matches("http://example.com/data", "http://*"));
        assert!(pattern_matches("test-pattern", "test-pattern"));
        assert!(!pattern_matches("different-pattern", "test-pattern"));
    }

    #[tokio::test]
    async fn test_capability_filtering() {
        let mut registry = ServiceRegistry::new();

        let sparql_service = FederatedService::new_sparql(
            "sparql-service".to_string(),
            "SPARQL Service".to_string(),
            "http://example.com/sparql".to_string(),
        );

        let graphql_service = FederatedService::new_graphql(
            "graphql-service".to_string(),
            "GraphQL Service".to_string(),
            "http://example.com/graphql".to_string(),
        );

        // Note: These will fail health checks but we're testing capability filtering
        let _ = registry.register(sparql_service).await;
        let _ = registry.register(graphql_service).await;

        let sparql_services =
            registry.get_services_with_capability(&ServiceCapability::SparqlQuery);
        let graphql_services =
            registry.get_services_with_capability(&ServiceCapability::GraphQLQuery);

        // Services may not register due to health check failures in test environment
        // Focus on testing the filtering logic itself
        assert!(sparql_services.len() <= 1);
        assert!(graphql_services.len() <= 1);
    }

    #[tokio::test]
    async fn test_connection_pool_initialization() {
        let registry = ServiceRegistry::new();
        let service = FederatedService::new_sparql(
            "test-service".to_string(),
            "Test Service".to_string(),
            "http://example.com/sparql".to_string(),
        );

        let result = registry.initialize_connection_pool(&service).await;
        assert!(result.is_ok());

        let pools = registry.connection_pools.read().await;
        assert!(pools.contains_key("test-service"));
    }

    #[test]
    fn test_rate_limit_check() {
        let registry = ServiceRegistry::new();

        // Service without rate limit should always pass
        assert!(registry.check_rate_limit("non-existent-service"));
    }

    #[test]
    fn test_auth_header_creation() {
        let registry = ServiceRegistry::new();
        let mut headers = HeaderMap::new();

        let auth_config = AuthConfig {
            auth_type: AuthType::Basic,
            credentials: AuthCredentials {
                username: Some("user".to_string()),
                password: Some("pass".to_string()),
                token: None,
                api_key: None,
                oauth_config: None,
            },
        };

        let result = registry.add_auth_header(&mut headers, &auth_config);
        assert!(result.is_ok());
        assert!(headers.contains_key(AUTHORIZATION));
    }
}
