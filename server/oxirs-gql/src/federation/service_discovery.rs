//! Dynamic Service Discovery for GraphQL Federation
//!
//! This module provides automatic service discovery capabilities for GraphQL federation,
//! including health monitoring, capability detection, and load balancing.

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};

use super::config::{RemoteEndpoint, RetryStrategy};
use crate::introspection::IntrospectionQuery;

/// Service discovery configuration
#[derive(Debug, Clone)]
pub struct ServiceDiscoveryConfig {
    pub discovery_interval: Duration,
    pub health_check_interval: Duration,
    pub service_timeout: Duration,
    pub max_concurrent_checks: usize,
    pub auto_register: bool,
    pub discovery_methods: Vec<DiscoveryMethod>,
}

impl Default for ServiceDiscoveryConfig {
    fn default() -> Self {
        Self {
            discovery_interval: Duration::from_secs(30),
            health_check_interval: Duration::from_secs(15),
            service_timeout: Duration::from_secs(10),
            max_concurrent_checks: 10,
            auto_register: true,
            discovery_methods: vec![
                DiscoveryMethod::Consul,
                DiscoveryMethod::Kubernetes,
                DiscoveryMethod::Static,
            ],
        }
    }
}

/// Service discovery methods
#[derive(Debug, Clone, PartialEq)]
pub enum DiscoveryMethod {
    Consul,
    Kubernetes,
    Etcd,
    Static,
    DNS,
    Custom(String),
}

/// Service metadata and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceInfo {
    pub id: String,
    pub name: String,
    pub url: String,
    pub version: Option<String>,
    pub capabilities: ServiceCapabilities,
    pub health_status: HealthStatus,
    pub metadata: HashMap<String, String>,
    pub last_seen: DateTime<Utc>,
    pub response_time: Option<Duration>,
    pub load_factor: f64,
    pub federation_version: Option<String>,
}

/// Service capabilities and features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceCapabilities {
    pub federation_enabled: bool,
    pub introspection_enabled: bool,
    pub subscriptions_enabled: bool,
    pub supported_directives: HashSet<String>,
    pub custom_scalars: HashSet<String>,
    pub query_complexity_limit: Option<usize>,
    pub schema_version: Option<String>,
    pub entity_types: HashSet<String>,
}

impl Default for ServiceCapabilities {
    fn default() -> Self {
        Self {
            federation_enabled: false,
            introspection_enabled: true,
            subscriptions_enabled: false,
            supported_directives: HashSet::new(),
            custom_scalars: HashSet::new(),
            query_complexity_limit: None,
            schema_version: None,
            entity_types: HashSet::new(),
        }
    }
}

/// Service health status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// Service discovery event
#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    ServiceRegistered(ServiceInfo),
    ServiceUpdated(ServiceInfo),
    ServiceDeregistered(String),
    HealthChanged {
        service_id: String,
        old_status: HealthStatus,
        new_status: HealthStatus,
    },
}

/// Service discovery trait for different backends
#[async_trait]
pub trait ServiceDiscoveryBackend: Send + Sync {
    async fn discover_services(&self) -> Result<Vec<ServiceInfo>>;
    async fn register_service(&self, service: &ServiceInfo) -> Result<()>;
    async fn deregister_service(&self, service_id: &str) -> Result<()>;
    async fn update_health(&self, service_id: &str, status: HealthStatus) -> Result<()>;
}

/// Dynamic service discovery engine
pub struct ServiceDiscovery {
    config: ServiceDiscoveryConfig,
    services: Arc<RwLock<HashMap<String, ServiceInfo>>>,
    backends: Vec<Box<dyn ServiceDiscoveryBackend>>,
    event_handlers: Arc<RwLock<Vec<Box<dyn ServiceDiscoveryEventHandler>>>>,
    http_client: reqwest::Client,
    health_checks: Arc<Mutex<HashMap<String, Instant>>>,
}

/// Event handler trait for service discovery events
#[async_trait]
pub trait ServiceDiscoveryEventHandler: Send + Sync {
    async fn handle_event(&self, event: DiscoveryEvent) -> Result<()>;
}

impl ServiceDiscovery {
    /// Create a new service discovery engine
    pub fn new(config: ServiceDiscoveryConfig) -> Self {
        let http_client = reqwest::Client::builder()
            .timeout(config.service_timeout)
            .build()
            .expect("Failed to create HTTP client");

        let discovery = Self {
            config,
            services: Arc::new(RwLock::new(HashMap::new())),
            backends: Vec::new(),
            event_handlers: Arc::new(RwLock::new(Vec::new())),
            http_client,
            health_checks: Arc::new(Mutex::new(HashMap::new())),
        };

        discovery
    }

    /// Add a service discovery backend
    pub fn add_backend(&mut self, backend: Box<dyn ServiceDiscoveryBackend>) {
        self.backends.push(backend);
    }

    /// Add an event handler
    pub async fn add_event_handler(&self, handler: Box<dyn ServiceDiscoveryEventHandler>) {
        let mut handlers = self.event_handlers.write().await;
        handlers.push(handler);
    }

    /// Start the service discovery process
    pub async fn start(&self) -> Result<()> {
        info!("Starting service discovery engine");

        // Start discovery loop
        self.start_discovery_loop().await;

        // Start health check loop
        self.start_health_check_loop().await;

        Ok(())
    }

    /// Get all discovered services
    pub async fn get_services(&self) -> Vec<ServiceInfo> {
        let services = self.services.read().await;
        services.values().cloned().collect()
    }

    /// Get a specific service by ID
    pub async fn get_service(&self, service_id: &str) -> Option<ServiceInfo> {
        let services = self.services.read().await;
        services.get(service_id).cloned()
    }

    /// Get healthy services only
    pub async fn get_healthy_services(&self) -> Vec<ServiceInfo> {
        let services = self.services.read().await;
        services
            .values()
            .filter(|service| service.health_status == HealthStatus::Healthy)
            .cloned()
            .collect()
    }

    /// Get services by capability
    pub async fn get_services_with_capability(
        &self,
        check: impl Fn(&ServiceCapabilities) -> bool,
    ) -> Vec<ServiceInfo> {
        let services = self.services.read().await;
        services
            .values()
            .filter(|service| check(&service.capabilities))
            .cloned()
            .collect()
    }

    /// Register a service manually
    pub async fn register_service(&self, service: ServiceInfo) -> Result<()> {
        info!("Manually registering service: {}", service.id);

        // Store locally
        {
            let mut services = self.services.write().await;
            services.insert(service.id.clone(), service.clone());
        }

        // Register with backends
        for backend in &self.backends {
            if let Err(e) = backend.register_service(&service).await {
                warn!(
                    "Failed to register service {} with backend: {}",
                    service.id, e
                );
            }
        }

        // Emit event
        self.emit_event(DiscoveryEvent::ServiceRegistered(service))
            .await;

        Ok(())
    }

    /// Deregister a service
    pub async fn deregister_service(&self, service_id: &str) -> Result<()> {
        info!("Deregistering service: {}", service_id);

        // Remove locally
        {
            let mut services = self.services.write().await;
            services.remove(service_id);
        }

        // Deregister from backends
        for backend in &self.backends {
            if let Err(e) = backend.deregister_service(service_id).await {
                warn!(
                    "Failed to deregister service {} from backend: {}",
                    service_id, e
                );
            }
        }

        // Emit event
        self.emit_event(DiscoveryEvent::ServiceDeregistered(service_id.to_string()))
            .await;

        Ok(())
    }

    /// Get service with best performance (lowest latency and load)
    pub async fn get_best_service(
        &self,
        predicate: impl Fn(&ServiceInfo) -> bool,
    ) -> Option<ServiceInfo> {
        let services = self.services.read().await;

        services
            .values()
            .filter(|service| service.health_status == HealthStatus::Healthy && predicate(service))
            .min_by(|a, b| {
                let a_score = a.load_factor
                    + a.response_time
                        .unwrap_or(Duration::from_secs(1))
                        .as_secs_f64();
                let b_score = b.load_factor
                    + b.response_time
                        .unwrap_or(Duration::from_secs(1))
                        .as_secs_f64();
                a_score
                    .partial_cmp(&b_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
    }

    /// Introspect a service to determine its capabilities
    pub async fn introspect_service(&self, url: &str) -> Result<ServiceCapabilities> {
        debug!("Introspecting service at: {}", url);

        let introspection_query = IntrospectionQuery::full_query();

        let response = self
            .http_client
            .post(url)
            .json(&serde_json::json!({
                "query": introspection_query,
                "variables": {}
            }))
            .send()
            .await
            .context("Failed to send introspection query")?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Introspection failed with status: {}",
                response.status()
            ));
        }

        let response_json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse introspection response")?;

        self.parse_capabilities(&response_json)
    }

    /// Parse service capabilities from introspection result
    fn parse_capabilities(&self, introspection: &serde_json::Value) -> Result<ServiceCapabilities> {
        let mut capabilities = ServiceCapabilities::default();

        if let Some(schema) = introspection.get("data").and_then(|d| d.get("__schema")) {
            // Check for federation directives
            if let Some(directives) = schema.get("directives").and_then(|d| d.as_array()) {
                for directive in directives {
                    if let Some(name) = directive.get("name").and_then(|n| n.as_str()) {
                        capabilities.supported_directives.insert(name.to_string());

                        // Check for federation-specific directives
                        if ["key", "external", "requires", "provides", "extends"].contains(&name) {
                            capabilities.federation_enabled = true;
                        }
                    }
                }
            }

            // Check for custom scalars
            if let Some(types) = schema.get("types").and_then(|t| t.as_array()) {
                for type_def in types {
                    if let Some(kind) = type_def.get("kind").and_then(|k| k.as_str()) {
                        if kind == "SCALAR" {
                            if let Some(name) = type_def.get("name").and_then(|n| n.as_str()) {
                                if !["String", "Int", "Float", "Boolean", "ID"].contains(&name) {
                                    capabilities.custom_scalars.insert(name.to_string());
                                }
                            }
                        }
                    }
                }
            }

            // Check for subscription support
            if let Some(subscription_type) = schema.get("subscriptionType") {
                if !subscription_type.is_null() {
                    capabilities.subscriptions_enabled = true;
                }
            }

            // Extract schema version from description
            if let Some(description) = schema.get("description").and_then(|d| d.as_str()) {
                capabilities.schema_version = self.extract_version_from_description(description);
            }
        }

        Ok(capabilities)
    }

    /// Extract version from description string
    fn extract_version_from_description(&self, description: &str) -> Option<String> {
        // Simple regex-based version extraction
        let version_patterns = vec![
            r"version\s*:?\s*([0-9]+\.[0-9]+\.[0-9]+)",
            r"v([0-9]+\.[0-9]+\.[0-9]+)",
            r"([0-9]+\.[0-9]+\.[0-9]+)",
        ];

        for pattern_str in &version_patterns {
            if let Ok(pattern) = regex::Regex::new(pattern_str) {
                if let Some(captures) = pattern.captures(description) {
                    if let Some(version_match) = captures.get(1) {
                        return Some(version_match.as_str().to_string());
                    }
                }
            }
        }

        None
    }

    /// Check service health
    pub async fn check_service_health(&self, service: &ServiceInfo) -> HealthStatus {
        let start = Instant::now();

        // Try a simple introspection query
        let simple_query = r#"
            query HealthCheck {
                __schema {
                    queryType {
                        name
                    }
                }
            }
        "#;

        let result = self
            .http_client
            .post(&service.url)
            .json(&serde_json::json!({
                "query": simple_query,
                "variables": {}
            }))
            .timeout(Duration::from_secs(5))
            .send()
            .await;

        match result {
            Ok(response) => {
                let duration = start.elapsed();

                if response.status().is_success() {
                    // Update response time
                    self.update_service_response_time(&service.id, duration)
                        .await;

                    if duration < Duration::from_millis(500) {
                        HealthStatus::Healthy
                    } else {
                        HealthStatus::Degraded
                    }
                } else {
                    HealthStatus::Unhealthy
                }
            }
            Err(_) => HealthStatus::Unhealthy,
        }
    }

    /// Update service response time
    async fn update_service_response_time(&self, service_id: &str, response_time: Duration) {
        let mut services = self.services.write().await;
        if let Some(service) = services.get_mut(service_id) {
            service.response_time = Some(response_time);
        }
    }

    /// Start the service discovery loop
    async fn start_discovery_loop(&self) {
        let services = Arc::clone(&self.services);
        let backends = self.backends.len(); // Capture count
        let discovery_interval = self.config.discovery_interval;

        tokio::spawn(async move {
            let mut interval = interval(discovery_interval);

            loop {
                interval.tick().await;

                debug!("Running service discovery sweep");

                // In a real implementation, you would iterate over backends
                // For now, this is a placeholder that logs the discovery attempt
                if backends > 0 {
                    debug!(
                        "Discovered {} services from {} backends",
                        services.read().await.len(),
                        backends
                    );
                }
            }
        });
    }

    /// Start the health check loop
    async fn start_health_check_loop(&self) {
        let services = Arc::clone(&self.services);
        let health_checks = Arc::clone(&self.health_checks);
        let health_check_interval = self.config.health_check_interval;
        let http_client = self.http_client.clone();
        let event_handlers = Arc::clone(&self.event_handlers);

        tokio::spawn(async move {
            let mut interval = interval(health_check_interval);

            loop {
                interval.tick().await;

                let service_list = {
                    let services_guard = services.read().await;
                    services_guard.values().cloned().collect::<Vec<_>>()
                };

                for service in service_list {
                    // Check if we need to health check this service
                    let should_check = {
                        let mut checks = health_checks.lock().await;
                        let last_check = checks
                            .get(&service.id)
                            .copied()
                            .unwrap_or_else(|| Instant::now() - health_check_interval);

                        if last_check.elapsed() >= health_check_interval {
                            checks.insert(service.id.clone(), Instant::now());
                            true
                        } else {
                            false
                        }
                    };

                    if should_check {
                        let service_clone = service.clone();
                        let services_clone = Arc::clone(&services);
                        let event_handlers_clone = Arc::clone(&event_handlers);
                        let http_client_clone = http_client.clone();

                        tokio::spawn(async move {
                            let new_status =
                                Self::perform_health_check(&http_client_clone, &service_clone)
                                    .await;

                            let old_status = {
                                let mut services_guard = services_clone.write().await;
                                if let Some(service_mut) = services_guard.get_mut(&service_clone.id)
                                {
                                    let old = service_mut.health_status.clone();
                                    service_mut.health_status = new_status.clone();
                                    service_mut.last_seen = Utc::now();
                                    old
                                } else {
                                    return;
                                }
                            };

                            // Emit health change event if status changed
                            if old_status != new_status {
                                let event = DiscoveryEvent::HealthChanged {
                                    service_id: service_clone.id.clone(),
                                    old_status,
                                    new_status,
                                };

                                let handlers = event_handlers_clone.read().await;
                                for handler in handlers.iter() {
                                    if let Err(e) = handler.handle_event(event.clone()).await {
                                        error!("Failed to handle discovery event: {}", e);
                                    }
                                }
                            }
                        });
                    }
                }
            }
        });
    }

    /// Perform health check for a service
    async fn perform_health_check(
        http_client: &reqwest::Client,
        service: &ServiceInfo,
    ) -> HealthStatus {
        let simple_query = r#"
            query HealthCheck {
                __schema {
                    queryType {
                        name
                    }
                }
            }
        "#;

        let start = Instant::now();
        let result = http_client
            .post(&service.url)
            .json(&serde_json::json!({
                "query": simple_query,
                "variables": {}
            }))
            .timeout(Duration::from_secs(5))
            .send()
            .await;

        match result {
            Ok(response) => {
                let duration = start.elapsed();

                if response.status().is_success() {
                    if duration < Duration::from_millis(500) {
                        HealthStatus::Healthy
                    } else {
                        HealthStatus::Degraded
                    }
                } else {
                    HealthStatus::Unhealthy
                }
            }
            Err(_) => HealthStatus::Unhealthy,
        }
    }

    /// Emit a discovery event to all handlers
    async fn emit_event(&self, event: DiscoveryEvent) {
        let handlers = self.event_handlers.read().await;
        for handler in handlers.iter() {
            if let Err(e) = handler.handle_event(event.clone()).await {
                error!("Failed to handle discovery event: {}", e);
            }
        }
    }
}

/// Static service discovery backend (for manually configured services)
pub struct StaticServiceDiscovery {
    services: Vec<ServiceInfo>,
}

impl StaticServiceDiscovery {
    pub fn new(services: Vec<ServiceInfo>) -> Self {
        Self { services }
    }
}

#[async_trait]
impl ServiceDiscoveryBackend for StaticServiceDiscovery {
    async fn discover_services(&self) -> Result<Vec<ServiceInfo>> {
        Ok(self.services.clone())
    }

    async fn register_service(&self, _service: &ServiceInfo) -> Result<()> {
        // Static discovery doesn't support dynamic registration
        Ok(())
    }

    async fn deregister_service(&self, _service_id: &str) -> Result<()> {
        // Static discovery doesn't support dynamic deregistration
        Ok(())
    }

    async fn update_health(&self, _service_id: &str, _status: HealthStatus) -> Result<()> {
        // Static discovery doesn't support health updates
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_service_discovery_creation() {
        let config = ServiceDiscoveryConfig::default();
        let discovery = ServiceDiscovery::new(config);

        let services = discovery.get_services().await;
        assert!(services.is_empty());
    }

    #[tokio::test]
    async fn test_service_registration() {
        let config = ServiceDiscoveryConfig::default();
        let discovery = ServiceDiscovery::new(config);

        let service = ServiceInfo {
            id: "test-service".to_string(),
            name: "Test Service".to_string(),
            url: "http://localhost:4000/graphql".to_string(),
            version: Some("1.0.0".to_string()),
            capabilities: ServiceCapabilities::default(),
            health_status: HealthStatus::Healthy,
            metadata: HashMap::new(),
            last_seen: Utc::now(),
            response_time: Some(Duration::from_millis(100)),
            load_factor: 0.5,
            federation_version: Some("2.0".to_string()),
        };

        discovery.register_service(service.clone()).await.unwrap();

        let services = discovery.get_services().await;
        assert_eq!(services.len(), 1);
        assert_eq!(services[0].id, "test-service");
    }

    #[tokio::test]
    async fn test_healthy_services_filter() {
        let config = ServiceDiscoveryConfig::default();
        let discovery = ServiceDiscovery::new(config);

        let healthy_service = ServiceInfo {
            id: "healthy-service".to_string(),
            name: "Healthy Service".to_string(),
            url: "http://localhost:4000/graphql".to_string(),
            version: Some("1.0.0".to_string()),
            capabilities: ServiceCapabilities::default(),
            health_status: HealthStatus::Healthy,
            metadata: HashMap::new(),
            last_seen: Utc::now(),
            response_time: Some(Duration::from_millis(100)),
            load_factor: 0.5,
            federation_version: Some("2.0".to_string()),
        };

        let unhealthy_service = ServiceInfo {
            id: "unhealthy-service".to_string(),
            name: "Unhealthy Service".to_string(),
            url: "http://localhost:4001/graphql".to_string(),
            version: Some("1.0.0".to_string()),
            capabilities: ServiceCapabilities::default(),
            health_status: HealthStatus::Unhealthy,
            metadata: HashMap::new(),
            last_seen: Utc::now(),
            response_time: None,
            load_factor: 1.0,
            federation_version: Some("2.0".to_string()),
        };

        discovery.register_service(healthy_service).await.unwrap();
        discovery.register_service(unhealthy_service).await.unwrap();

        let all_services = discovery.get_services().await;
        let healthy_services = discovery.get_healthy_services().await;

        assert_eq!(all_services.len(), 2);
        assert_eq!(healthy_services.len(), 1);
        assert_eq!(healthy_services[0].id, "healthy-service");
    }
}
