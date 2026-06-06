//! Federated Service Registry Core
//!
//! Implements the [`FederatedServiceRegistry`], which manages registration,
//! health checking, capability detection, connection pooling, rate limiting,
//! and extended metadata collection for federated services.

use crate::HealthStatus;
use anyhow::{anyhow, Result};
use base64::{engine::general_purpose, Engine as _};
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use reqwest::{
    header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE},
    Client,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::metadata::{ExtendedServiceMetadata, HealthCheckResult};
use crate::service_types::{
    pattern_matches, AuthType, ConnectionPool, ConnectionPoolStats, FederatedService,
    ServiceAuthConfig, ServiceCapability, ServiceRegistryConfig, ServiceRegistryStats,
    ServiceStatus, ServiceType,
};

/// Registry for managing federated services
#[derive(Debug, Clone)]
pub struct FederatedServiceRegistry {
    pub(crate) services: HashMap<String, FederatedService>,
    config: ServiceRegistryConfig,
    last_health_check: Option<Instant>,
    http_client: Client,
    connection_pools: Arc<RwLock<HashMap<String, ConnectionPool>>>,
    rate_limiters: HashMap<String, Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>,
}

impl FederatedServiceRegistry {
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

        // Check if service already exists
        if self.services.contains_key(&service.id) {
            return Err(anyhow!(
                "Service with ID '{}' already registered",
                service.id
            ));
        }

        // Validate service configuration
        service.validate()?;

        // Initialize connection pool for this service
        self.initialize_connection_pool(&service).await?;

        // Initialize rate limiter if specified
        if let Some(rate_limit) = &service.performance.rate_limit {
            let quota = Quota::per_minute(
                std::num::NonZeroU32::new(rate_limit.requests_per_minute as u32)
                    .expect("construction should succeed"),
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

        Ok(crate::HealthStatus {
            overall_status: match overall_status {
                ServiceStatus::Healthy => crate::ServiceStatus::Healthy,
                ServiceStatus::Degraded => crate::ServiceStatus::Degraded,
                ServiceStatus::Unavailable => crate::ServiceStatus::Unavailable,
                ServiceStatus::Unknown => crate::ServiceStatus::Unknown,
            },
            service_statuses: service_statuses
                .into_iter()
                .map(|(k, v)| {
                    let status = match v {
                        ServiceStatus::Healthy => crate::ServiceStatus::Healthy,
                        ServiceStatus::Degraded => crate::ServiceStatus::Degraded,
                        ServiceStatus::Unavailable => crate::ServiceStatus::Unavailable,
                        ServiceStatus::Unknown => crate::ServiceStatus::Unknown,
                    };
                    (k, status)
                })
                .collect(),
            total_services,
            healthy_services: healthy_count,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Get registry statistics
    pub async fn get_stats(&self) -> ServiceRegistryStats {
        let health_check = self
            .health_check()
            .await
            .unwrap_or_else(|_| crate::HealthStatus {
                overall_status: crate::ServiceStatus::Unknown,
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

        if service.service_type != ServiceType::Sparql
            && service.service_type != ServiceType::Hybrid
        {
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
            self.add_auth_header(&mut headers, auth).await?;
        }

        let response = self
            .http_client
            .post(&service.endpoint)
            .headers(headers.clone())
            .body(stats_query)
            .send()
            .await?;

        if response.status().is_success() {
            // Parse statistics and update extended metadata
            let json: serde_json::Value = response.json().await?;

            if let Some(service) = self.services.get_mut(service_id) {
                if let Some(ref mut extended) = service.extended_metadata {
                    // Parse SPARQL JSON results
                    if let Some(results) = json
                        .get("results")
                        .and_then(|r| r.get("bindings"))
                        .and_then(|b| b.as_array())
                        .and_then(|a| a.first())
                    {
                        if let Some(total) = results
                            .get("totalTriples")
                            .and_then(|v| v.get("value"))
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<u64>().ok())
                        {
                            extended.dataset_stats.triple_count = Some(total);
                        }

                        if let Some(subjects) = results
                            .get("subjects")
                            .and_then(|v| v.get("value"))
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<u64>().ok())
                        {
                            extended.dataset_stats.subject_count = Some(subjects);
                        }

                        if let Some(predicates) = results
                            .get("predicates")
                            .and_then(|v| v.get("value"))
                            .and_then(|v| v.as_str())
                            .and_then(|s| s.parse::<u64>().ok())
                        {
                            extended.dataset_stats.predicate_count = Some(predicates);
                        }

                        if let Some(objects) = results
                            .get("objects")
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
                            if let Some(results) = json
                                .get("results")
                                .and_then(|r| r.get("bindings"))
                                .and_then(|b| b.as_array())
                            {
                                extended.dataset_stats.named_graphs = results
                                    .iter()
                                    .filter_map(|binding| {
                                        binding
                                            .get("g")
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
            ServiceType::RestRdf => self.check_sparql_health(service).await, // REST-RDF typically uses SPARQL endpoints
            ServiceType::Custom(_) => self.check_sparql_health(service).await, // Default to SPARQL health check
        };

        let response_time = start_time.elapsed();
        debug!(
            "Health check for {} completed in {:?}: {:?}",
            service.id, response_time, health_result
        );

        // Update extended metadata with health check result if available
        if let Some(_extended) = &service.extended_metadata {
            let _check_result = HealthCheckResult {
                timestamp: chrono::Utc::now(),
                success: matches!(health_result, Ok(ServiceStatus::Healthy)),
                response_time: Some(response_time),
                error_message: if health_result.is_err() {
                    Some(format!("{health_result:?}"))
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
            self.add_auth_header(&mut headers, auth).await?;
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
            self.add_auth_header(&mut headers, auth).await?;
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
            max_connections: self.config.connection_pool_size,
            active_connections: 0,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("operation should succeed")
                .as_secs(),
            last_used: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("operation should succeed")
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
                if self
                    .test_sparql_feature(service, "SELECT (COUNT(*) as ?count) WHERE { ?s ?p ?o }")
                    .await
                {
                    detected_capabilities.insert(ServiceCapability::SparqlAggregation);
                }

                if self
                    .test_sparql_feature(
                        service,
                        "SELECT ?s WHERE { ?s ?p ?o { SELECT ?s WHERE { ?s a ?type } LIMIT 10 } }",
                    )
                    .await
                {
                    detected_capabilities.insert(ServiceCapability::SparqlSubqueries);
                }

                if self
                    .test_sparql_feature(
                        service,
                        "SELECT ?s WHERE { ?s ?p ?o . FILTER NOT EXISTS { ?s a ?type } }",
                    )
                    .await
                {
                    detected_capabilities.insert(ServiceCapability::SparqlNegation);
                }

                if self.test_sparql_feature(service, "SELECT ?s ?o WHERE { ?s (<http://example.org/p1>|<http://example.org/p2>)+ ?o }").await {
                    detected_capabilities.insert(ServiceCapability::SparqlPropertyPaths);
                }

                if self
                    .test_sparql_feature(service, "SELECT ?s ?p ?o WHERE { ?s ?p ?o } GROUP BY ?p")
                    .await
                {
                    detected_capabilities.insert(ServiceCapability::SparqlGroupBy);
                }

                // Test SPARQL 1.2 Features (if available)
                if self.test_sparql_feature(service, "SELECT ?s WHERE { VALUES ?s { <http://example.org/1> <http://example.org/2> } ?s ?p ?o }").await {
                    detected_capabilities.insert(ServiceCapability::SparqlValues);
                }

                // Test RDF-star support
                if self
                    .test_sparql_feature(service, "SELECT ?s ?p ?o WHERE { << ?s ?p ?o >> ?m ?v }")
                    .await
                {
                    detected_capabilities.insert(ServiceCapability::RDFStar);
                }

                // Test reasoning capabilities
                if self
                    .test_sparql_feature(
                        service,
                        "SELECT ?s WHERE { ?s a ?class . ?class rdfs:subClassOf ?super }",
                    )
                    .await
                {
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
            ServiceType::RestRdf => {
                // REST-RDF services typically support basic SPARQL capabilities
                if self
                    .test_sparql_feature(service, "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1")
                    .await
                {
                    detected_capabilities.insert(ServiceCapability::SparqlQuery);
                }
                // Usually support Graph Store Protocol
                detected_capabilities.insert(ServiceCapability::GraphStore);
            }
            ServiceType::Custom(_) => {
                // For custom services, test basic capabilities
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
            if self.add_auth_header(&mut headers, auth).await.is_err() {
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
            if self.add_auth_header(&mut headers, auth).await.is_err() {
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
    async fn add_auth_header(
        &self,
        headers: &mut HeaderMap,
        auth: &ServiceAuthConfig,
    ) -> Result<()> {
        match &auth.auth_type {
            AuthType::Basic => {
                if let (Some(username), Some(password)) =
                    (&auth.credentials.username, &auth.credentials.password)
                {
                    let credentials = format!("{username}:{password}");
                    let encoded = general_purpose::STANDARD.encode(credentials.as_bytes());
                    let auth_value = format!("Basic {encoded}");
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
                }
            }
            AuthType::Bearer => {
                if let Some(token) = &auth.credentials.token {
                    let auth_value = format!("Bearer {token}");
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
                }
            }
            AuthType::ApiKey => {
                if let Some(api_key) = &auth.credentials.api_key {
                    let header_name = auth
                        .credentials
                        .api_key_header
                        .as_deref()
                        .unwrap_or("X-API-Key");
                    headers.insert(
                        HeaderName::from_bytes(header_name.as_bytes())?,
                        HeaderValue::from_str(api_key)?,
                    );
                }
            }
            AuthType::OAuth2 => {
                // Simplified OAuth2 - in production would implement full flow
                if let Some(token) = &auth.credentials.token {
                    let auth_value = format!("Bearer {token}");
                    headers.insert(AUTHORIZATION, HeaderValue::from_str(&auth_value)?);
                } else {
                    warn!("OAuth2 token not available");
                    return Err(anyhow!("OAuth2 token not available"));
                }
            }
            AuthType::Custom => {
                if let Some(custom_headers) = &auth.credentials.custom_headers {
                    for (key, value) in custom_headers {
                        let header_name = HeaderName::try_from(key.clone())?;
                        headers.insert(header_name, HeaderValue::from_str(value)?);
                    }
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

        if service.service_type != ServiceType::Sparql
            && service.service_type != ServiceType::Hybrid
        {
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
            self.add_auth_header(&mut headers, auth).await?;
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
                    if let Some(results) = json
                        .get("results")
                        .and_then(|r| r.get("bindings"))
                        .and_then(|b| b.as_array())
                    {
                        let vocabs: HashSet<String> = results
                            .iter()
                            .filter_map(|binding| {
                                binding
                                    .get("vocab")
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
                    if let Some(results) = json
                        .get("results")
                        .and_then(|r| r.get("bindings"))
                        .and_then(|b| b.as_array())
                    {
                        extended.dataset_stats.languages = results
                            .iter()
                            .filter_map(|binding| {
                                binding
                                    .get("lang")
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
        info!(
            "Performing comprehensive assessment for service: {}",
            service_id
        );

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
        let service = self
            .services
            .get(service_id)
            .ok_or_else(|| anyhow!("Service {} not found", service_id))?
            .clone();

        let detected_capabilities = self.detect_service_capabilities(&service).await?;

        if let Some(service) = self.services.get_mut(service_id) {
            service.capabilities.extend(detected_capabilities);
        }

        info!(
            "Comprehensive assessment completed for service: {}",
            service_id
        );
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
}

impl Default for FederatedServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}
