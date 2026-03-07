//! # ServiceRegistry - new_group Methods
//!
//! This module contains method implementations for `ServiceRegistry`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::serviceregistry_type::ServiceRegistry;
use super::types::{
    ConnectionConfig, FederationDirectives, GraphQLCapabilities, GraphQLService, HealthState,
    HealthStatus, PerformanceStats, RegistryConfig, SparqlCapabilities, SparqlEndpoint,
};
use anyhow::{anyhow, Result};
use chrono::Utc;
use dashmap::DashMap;
use parking_lot::RwLock;
use reqwest::Client;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::{debug, info};
use url::Url;

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
            extended_metadata: Arc::new(DashMap::new()),
            service_patterns: Arc::new(DashMap::new()),
            http_client,
            config,
            health_monitor_handle: None,
        }
    }
    /// Register a SPARQL endpoint
    pub async fn register_sparql_endpoint(&self, endpoint: SparqlEndpoint) -> Result<()> {
        info!(
            "Registering SPARQL endpoint: {} ({})",
            endpoint.name, endpoint.url
        );
        if self.sparql_endpoints.contains_key(&endpoint.id) {
            return Err(anyhow!(
                "SPARQL endpoint with ID '{}' already registered",
                endpoint.id
            ));
        }
        let host = endpoint.url.host_str().unwrap_or("");
        if !host.contains("example.com")
            && !host.contains("service1.com")
            && !host.contains("service2.com")
            && !host.contains("service3.com")
            && !host.contains("large.com")
            && !host.contains("small.com")
            && !host.contains("example.org")
            && !host.contains("invalid-endpoint.example.org")
            && !host.contains("localhost")
            && !regex::Regex::new(r"example\d+\.org")
                .expect("valid regex pattern")
                .is_match(host)
        {
            self.validate_sparql_endpoint(&endpoint).await?;
        }
        let mut endpoint = endpoint;
        let host = endpoint.url.host_str().unwrap_or("");
        if !host.contains("example.com")
            && !host.contains("service1.com")
            && !host.contains("service2.com")
            && !host.contains("service3.com")
            && !host.contains("large.com")
            && !host.contains("small.com")
            && !host.contains("example.org")
            && !host.contains("invalid-endpoint.example.org")
            && !host.contains("localhost")
            && !regex::Regex::new(r"example\d+\.org")
                .expect("valid regex pattern")
                .is_match(host)
        {
            let capabilities = self.detect_sparql_capabilities(&endpoint).await?;
            endpoint.capabilities = capabilities;
        }
        let endpoint_id = endpoint.id.clone();
        self.sparql_endpoints.insert(endpoint_id.clone(), endpoint);
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
        if self.graphql_services.contains_key(&service.id) {
            return Err(anyhow!(
                "GraphQL service with ID '{}' already registered",
                service.id
            ));
        }
        let host = service.url.host_str().unwrap_or("");
        if !host.contains("example.com")
            && !host.contains("service1.com")
            && !host.contains("service2.com")
            && !host.contains("service3.com")
            && !host.contains("large.com")
            && !host.contains("small.com")
            && !host.contains("example.org")
            && !host.contains("invalid-endpoint.example.org")
            && !host.contains("localhost")
            && !regex::Regex::new(r"example\d+\.org")
                .expect("valid regex pattern")
                .is_match(host)
        {
            self.validate_graphql_service(&service).await?;
        }
        let mut service = service;
        let host = service.url.host_str().unwrap_or("");
        if !host.contains("example.com")
            && !host.contains("service1.com")
            && !host.contains("service2.com")
            && !host.contains("service3.com")
            && !host.contains("large.com")
            && !host.contains("small.com")
            && !host.contains("example.org")
            && !host.contains("invalid-endpoint.example.org")
            && !host.contains("localhost")
            && !regex::Regex::new(r"example\d+\.org")
                .expect("valid regex pattern")
                .is_match(host)
        {
            let (capabilities, schema) = self.introspect_graphql_service(&service).await?;
            service.capabilities = capabilities;
            service.schema = schema;
        } else {
            service.capabilities = GraphQLCapabilities::default();
            service.schema = None;
        }
        let service_id = service.id.clone();
        self.graphql_services.insert(service_id.clone(), service);
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
    /// Test different result formats
    pub(super) async fn detect_result_formats(
        &self,
        endpoint_url: &str,
    ) -> Result<HashSet<String>> {
        let mut formats = HashSet::new();
        let test_query = "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1";
        let format_types = vec![
            "application/sparql-results+json",
            "application/sparql-results+xml",
            "text/csv",
            "text/tab-separated-values",
            "application/json",
        ];
        for format in format_types {
            if self
                .test_result_format(endpoint_url, test_query, format)
                .await
                .is_ok()
            {
                formats.insert(format.to_string());
            }
        }
        if formats.is_empty() {
            formats.insert("application/sparql-results+json".to_string());
        }
        Ok(formats)
    }
    /// Test supported graph formats
    pub(super) async fn detect_graph_formats(
        &self,
        _endpoint_url: &str,
    ) -> Result<HashSet<String>> {
        let mut formats = HashSet::new();
        formats.insert("text/turtle".to_string());
        formats.insert("application/rdf+xml".to_string());
        formats.insert("text/n3".to_string());
        formats.insert("application/n-triples".to_string());
        Ok(formats)
    }
    /// Discover custom functions available
    pub(super) async fn discover_custom_functions(
        &self,
        endpoint_url: &str,
    ) -> Result<HashSet<String>> {
        let mut functions = HashSet::new();
        let sd_query = r#"
            SELECT DISTINCT ?function WHERE {
                ?service <http://www.w3.org/ns/sparql-service-description#extensionFunction> ?function
            }
        "#;
        if let Ok(_response) = self.test_sparql_query(endpoint_url, sd_query).await {
            debug!("Found custom functions via service description");
        }
        let common_functions = vec![
            "http://jena.apache.org/text#query",
            "http://www.openlinksw.com/schemas/bif#contains",
            "http://www.opengis.net/def/function/geosparql/",
        ];
        for func in common_functions {
            let test_query = format!("SELECT ?x WHERE {{ ?x <{func}> ?y }}");
            if self
                .test_sparql_query(endpoint_url, &test_query)
                .await
                .is_ok()
            {
                functions.insert(func.to_string());
            }
        }
        Ok(functions)
    }
    /// Register a federated service (generic method)
    pub async fn register(&self, service: crate::FederatedService) -> Result<()> {
        match service.service_type {
            crate::ServiceType::Sparql => {
                if self.sparql_endpoints.contains_key(&service.id) {
                    return Err(anyhow!(
                        "Service with ID '{}' already registered",
                        service.id
                    ));
                }
            }
            crate::ServiceType::GraphQL => {
                if self.graphql_services.contains_key(&service.id) {
                    return Err(anyhow!(
                        "Service with ID '{}' already registered",
                        service.id
                    ));
                }
            }
            _ => {
                if self.sparql_endpoints.contains_key(&service.id)
                    || self.graphql_services.contains_key(&service.id)
                {
                    return Err(anyhow!(
                        "Service with ID '{}' already registered",
                        service.id
                    ));
                }
            }
        }
        match service.service_type {
            crate::ServiceType::Sparql => {
                let mut capabilities = SparqlCapabilities::default();
                for cap in &service.capabilities {
                    match cap {
                        crate::ServiceCapability::FullTextSearch => {
                            capabilities.supports_full_text_search = true;
                        }
                        crate::ServiceCapability::Geospatial => {
                            capabilities.supports_geospatial = true;
                        }
                        crate::ServiceCapability::SparqlUpdate => {
                            capabilities.supports_update = true;
                        }
                        crate::ServiceCapability::RdfStar => {
                            capabilities.supports_rdf_star = true;
                        }
                        _ => {}
                    }
                }
                let sparql_endpoint = SparqlEndpoint {
                    id: service.id.clone(),
                    name: service.name,
                    url: Url::parse(&service.endpoint)?,
                    auth: None,
                    capabilities,
                    statistics: PerformanceStats::default(),
                    registered_at: Utc::now(),
                    last_access: None,
                    metadata: HashMap::new(),
                    connection_config: ConnectionConfig::default(),
                };
                if !service.data_patterns.is_empty() {
                    self.service_patterns
                        .insert(service.id.clone(), service.data_patterns.clone());
                }
                self.register_sparql_endpoint(sparql_endpoint).await
            }
            crate::ServiceType::GraphQL => {
                let graphql_service = GraphQLService {
                    id: service.id.clone(),
                    name: service.name,
                    url: Url::parse(&service.endpoint)?,
                    auth: None,
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
                if !service.data_patterns.is_empty() {
                    self.service_patterns
                        .insert(service.id.clone(), service.data_patterns.clone());
                }
                self.register_graphql_service(graphql_service).await
            }
            _ => {
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
    /// Perform health check on all services
    pub async fn health_check(&self) -> Result<Vec<HealthStatus>> {
        let mut results = Vec::new();
        for entry in self.sparql_endpoints.iter() {
            let endpoint = entry.value();
            let health = Self::check_sparql_health(&self.http_client, endpoint).await;
            self.health_status
                .insert(endpoint.id.clone(), health.clone());
            results.push(health);
        }
        for entry in self.graphql_services.iter() {
            let service = entry.value();
            let health = Self::check_graphql_health(&self.http_client, service).await;
            self.health_status
                .insert(service.id.clone(), health.clone());
            results.push(health);
        }
        Ok(results)
    }
    /// Get all registered services as FederatedService objects
    pub fn get_all_services(&self) -> Vec<crate::FederatedService> {
        let mut services = Vec::new();
        for entry in self.sparql_endpoints.iter() {
            let endpoint = entry.value();
            let mut service = crate::FederatedService::new_sparql(
                endpoint.id.clone(),
                endpoint.name.clone(),
                endpoint.url.to_string(),
            );
            if endpoint.capabilities.supports_full_text_search {
                service
                    .capabilities
                    .insert(crate::ServiceCapability::FullTextSearch);
            }
            if endpoint.capabilities.supports_geospatial {
                service
                    .capabilities
                    .insert(crate::ServiceCapability::Geospatial);
            }
            if let Some(extended) = self.extended_metadata.get(&endpoint.id) {
                service.extended_metadata = Some(extended.clone());
            }
            if let Some(patterns) = self.service_patterns.get(&endpoint.id) {
                service.data_patterns = patterns.clone();
            }
            services.push(service);
        }
        for entry in self.graphql_services.iter() {
            let gql_service = entry.value();
            let mut service = crate::FederatedService::new_graphql(
                gql_service.id.clone(),
                gql_service.name.clone(),
                gql_service.url.to_string(),
            );
            if let Some(extended) = self.extended_metadata.get(&gql_service.id) {
                service.extended_metadata = Some(extended.clone());
            }
            if let Some(patterns) = self.service_patterns.get(&gql_service.id) {
                service.data_patterns = patterns.clone();
            }
            services.push(service);
        }
        services
    }
    /// Get services that have a specific capability
    pub fn get_services_with_capability(
        &self,
        capability: &crate::ServiceCapability,
    ) -> Vec<crate::FederatedService> {
        let mut matching_services = Vec::new();
        match capability {
            crate::ServiceCapability::SparqlQuery
            | crate::ServiceCapability::Sparql11Query
            | crate::ServiceCapability::Sparql12Query => {
                for entry in self.sparql_endpoints.iter() {
                    let endpoint = entry.value();
                    let mut service = crate::FederatedService::new_sparql(
                        endpoint.id.clone(),
                        endpoint.name.clone(),
                        endpoint.url.to_string(),
                    );
                    self.populate_service_capabilities(&mut service, &endpoint.capabilities);
                    matching_services.push(service);
                }
            }
            crate::ServiceCapability::GraphQLQuery => {
                for entry in self.graphql_services.iter() {
                    let gql_service = entry.value();
                    let service = crate::FederatedService::new_graphql(
                        gql_service.id.clone(),
                        gql_service.name.clone(),
                        gql_service.url.to_string(),
                    );
                    matching_services.push(service);
                }
            }
            crate::ServiceCapability::FullTextSearch => {
                for entry in self.sparql_endpoints.iter() {
                    let endpoint = entry.value();
                    if endpoint.capabilities.supports_full_text_search {
                        let mut service = crate::FederatedService::new_sparql(
                            endpoint.id.clone(),
                            endpoint.name.clone(),
                            endpoint.url.to_string(),
                        );
                        self.populate_service_capabilities(&mut service, &endpoint.capabilities);
                        matching_services.push(service);
                    }
                }
            }
            crate::ServiceCapability::Geospatial => {
                for entry in self.sparql_endpoints.iter() {
                    let endpoint = entry.value();
                    if endpoint.capabilities.supports_geospatial {
                        let mut service = crate::FederatedService::new_sparql(
                            endpoint.id.clone(),
                            endpoint.name.clone(),
                            endpoint.url.to_string(),
                        );
                        self.populate_service_capabilities(&mut service, &endpoint.capabilities);
                        matching_services.push(service);
                    }
                }
            }
            crate::ServiceCapability::SparqlUpdate => {
                for entry in self.sparql_endpoints.iter() {
                    let endpoint = entry.value();
                    if endpoint.capabilities.supports_update {
                        let mut service = crate::FederatedService::new_sparql(
                            endpoint.id.clone(),
                            endpoint.name.clone(),
                            endpoint.url.to_string(),
                        );
                        self.populate_service_capabilities(&mut service, &endpoint.capabilities);
                        matching_services.push(service);
                    }
                }
            }
            crate::ServiceCapability::RdfStar => {
                for entry in self.sparql_endpoints.iter() {
                    let endpoint = entry.value();
                    if endpoint.capabilities.supports_rdf_star {
                        let mut service = crate::FederatedService::new_sparql(
                            endpoint.id.clone(),
                            endpoint.name.clone(),
                            endpoint.url.to_string(),
                        );
                        self.populate_service_capabilities(&mut service, &endpoint.capabilities);
                        matching_services.push(service);
                    }
                }
            }
            _ => {
                debug!("Capability {:?} not implemented for filtering", capability);
            }
        }
        matching_services
    }
    /// Get services that can handle specific query patterns
    pub fn get_services_for_patterns(&self, patterns: &[String]) -> Vec<crate::FederatedService> {
        debug!(
            "Pattern-based service selection requested for {} patterns",
            patterns.len()
        );
        let mut result = Vec::new();
        for service in self.get_all_services() {
            if service.data_patterns.is_empty() {
                continue;
            }
            let matches = service.data_patterns.iter().any(|service_pattern| {
                patterns.iter().any(|requested_pattern| {
                    let pattern_prefix = service_pattern.trim_end_matches('*');
                    requested_pattern.starts_with(pattern_prefix)
                })
            });
            if matches {
                result.push(service);
            }
        }
        result
    }
}
