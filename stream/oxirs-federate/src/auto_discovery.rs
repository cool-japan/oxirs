//! Automatic Service Discovery
//!
//! This module implements automatic discovery protocols including mDNS/DNS-SD
//! for finding federated SPARQL and GraphQL services on the network.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio::time::interval;
#[cfg(feature = "service-discovery")]
use tracing::debug;
use tracing::{error, info, warn};

#[cfg(feature = "service-discovery")]
use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

use crate::{discovery::ServiceDiscovery, service::ServiceType, service_registry::ServiceRegistry};

/// Service discovery daemon for automatic discovery
pub struct AutoDiscovery {
    config: AutoDiscoveryConfig,
    #[cfg(feature = "service-discovery")]
    mdns_daemon: Option<ServiceDaemon>,
    discovered_services: Arc<RwLock<HashMap<String, DiscoveredEndpoint>>>,
    discovery_channel: Option<mpsc::Sender<DiscoveredEndpoint>>,
}

impl std::fmt::Debug for AutoDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoDiscovery")
            .field("config", &self.config)
            .field("discovered_services", &"<Arc<RwLock>>")
            .field("discovery_channel", &self.discovery_channel.is_some())
            .finish()
    }
}

impl AutoDiscovery {
    /// Create a new auto discovery instance
    pub fn new(config: AutoDiscoveryConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "service-discovery")]
            mdns_daemon: None,
            discovered_services: Arc::new(RwLock::new(HashMap::new())),
            discovery_channel: None,
        }
    }

    /// Start the auto discovery process
    pub async fn start(&mut self) -> Result<mpsc::Receiver<DiscoveredEndpoint>> {
        info!("Starting automatic service discovery");

        let (tx, rx) = mpsc::channel(100);
        self.discovery_channel = Some(tx.clone());

        // Check if any discovery method is enabled
        let any_enabled = self.config.enable_dns_discovery
            || self.config.enable_kubernetes_discovery
            || self.config.enable_mdns;

        if !any_enabled {
            info!("No discovery methods enabled, closing channel");
            // Drop the sender to signal that no discoveries will be sent
            drop(tx);
            return Ok(rx);
        }

        // Start mDNS/DNS-SD discovery if enabled
        #[cfg(feature = "service-discovery")]
        if self.config.enable_mdns {
            self.start_mdns_discovery(tx.clone()).await?;
        }

        // Start DNS-based discovery for known domains
        if self.config.enable_dns_discovery {
            self.start_dns_discovery(tx.clone()).await?;
        }

        // Start Kubernetes service discovery if configured
        if self.config.enable_kubernetes_discovery {
            self.start_kubernetes_discovery(tx.clone()).await?;
        }

        Ok(rx)
    }

    /// Stop the auto discovery process
    pub async fn stop(&mut self) {
        info!("Stopping automatic service discovery");

        #[cfg(feature = "service-discovery")]
        if let Some(daemon) = self.mdns_daemon.take() {
            if let Err(e) = daemon.shutdown() {
                error!("Error shutting down mDNS daemon: {}", e);
            }
        }

        self.discovery_channel = None;
    }

    /// Start mDNS/DNS-SD discovery
    #[cfg(feature = "service-discovery")]
    async fn start_mdns_discovery(&mut self, tx: mpsc::Sender<DiscoveredEndpoint>) -> Result<()> {
        info!("Starting mDNS/DNS-SD discovery");

        let daemon = ServiceDaemon::new()?;
        self.mdns_daemon = Some(daemon);

        // Browse for SPARQL services
        let sparql_service = "_sparql._tcp.local.";
        let graphql_service = "_graphql._tcp.local.";

        if let Some(ref daemon) = self.mdns_daemon {
            let receiver = daemon.browse(sparql_service)?;
            let tx_clone = tx.clone();

            // Spawn task to handle SPARQL service events
            tokio::spawn(async move {
                while let Ok(event) = receiver.recv() {
                    Self::handle_mdns_event(event, ServiceType::Sparql, tx_clone.clone()).await;
                }
            });

            // Browse for GraphQL services
            let receiver = daemon.browse(graphql_service)?;

            tokio::spawn(async move {
                while let Ok(event) = receiver.recv() {
                    Self::handle_mdns_event(event, ServiceType::GraphQL, tx.clone()).await;
                }
            });
        }

        Ok(())
    }

    /// Handle mDNS service event
    #[cfg(feature = "service-discovery")]
    async fn handle_mdns_event(
        event: ServiceEvent,
        service_type: ServiceType,
        tx: mpsc::Sender<DiscoveredEndpoint>,
    ) {
        match event {
            ServiceEvent::ServiceResolved(resolved) => {
                // ResolvedService has same interface as ServiceInfo
                debug!(
                    "Discovered {:?} service: {}",
                    service_type,
                    resolved.get_fullname()
                );

                let endpoint = Self::extract_endpoint_from_resolved(&resolved);
                if let Some(endpoint) = endpoint {
                    let discovered = DiscoveredEndpoint {
                        url: endpoint,
                        service_type,
                        discovery_method: DiscoveryMethod::MDNS,
                        metadata: Self::extract_resolved_metadata(&resolved),
                        timestamp: chrono::Utc::now(),
                    };

                    if let Err(e) = tx.send(discovered).await {
                        error!("Failed to send discovered endpoint: {}", e);
                    }
                }
            }
            ServiceEvent::ServiceRemoved(_, fullname) => {
                debug!("Service removed: {}", fullname);
            }
            _ => {}
        }
    }

    /// Extract endpoint URL from mDNS service info
    #[cfg(feature = "service-discovery")]
    #[allow(dead_code)]
    fn extract_endpoint_from_mdns(info: &ServiceInfo) -> Option<String> {
        let addresses: Vec<_> = info.get_addresses().iter().collect();
        if let Some(addr) = addresses.first() {
            let port = info.get_port();
            let path = info
                .get_properties()
                .get_property_val_str("path")
                .unwrap_or("");

            Some(format!("http://{}:{}{}", addr, port, path))
        } else {
            None
        }
    }

    /// Extract endpoint URL from resolved mDNS service
    #[cfg(feature = "service-discovery")]
    fn extract_endpoint_from_resolved(resolved: &mdns_sd::ResolvedService) -> Option<String> {
        let addresses: Vec<_> = resolved.get_addresses().iter().collect();
        if let Some(addr) = addresses.first() {
            let port = resolved.get_port();
            let path = resolved
                .get_properties()
                .get_property_val_str("path")
                .unwrap_or("");

            Some(format!("http://{}:{}{}", addr, port, path))
        } else {
            None
        }
    }

    /// Extract metadata from mDNS service info
    #[cfg(feature = "service-discovery")]
    #[allow(dead_code)]
    fn extract_mdns_metadata(info: &ServiceInfo) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        for property in info.get_properties().iter() {
            let key = property.key();
            let val_str = property.val_str();
            metadata.insert(key.to_string(), val_str.to_string());
        }

        metadata
    }

    /// Extract metadata from resolved mDNS service
    #[cfg(feature = "service-discovery")]
    fn extract_resolved_metadata(resolved: &mdns_sd::ResolvedService) -> HashMap<String, String> {
        let mut metadata = HashMap::new();

        for property in resolved.get_properties().iter() {
            let key = property.key();
            let val_str = property.val_str();
            metadata.insert(key.to_string(), val_str.to_string());
        }

        metadata
    }

    /// Start DNS-based discovery for known domains
    async fn start_dns_discovery(&self, tx: mpsc::Sender<DiscoveredEndpoint>) -> Result<()> {
        info!("Starting DNS-based service discovery");

        let domains = self.config.dns_domains.clone();
        let discovered = self.discovered_services.clone();

        tokio::spawn(async move {
            let mut discovery_interval = interval(Duration::from_secs(300)); // Check every 5 minutes

            loop {
                discovery_interval.tick().await;

                for domain in &domains {
                    // Try common SPARQL endpoint patterns
                    let sparql_endpoints = vec![
                        format!("https://{}/sparql", domain),
                        format!("https://sparql.{}", domain),
                        format!("https://{}/query", domain),
                        format!("https://query.{}", domain),
                    ];

                    for endpoint in sparql_endpoints {
                        if Self::test_endpoint(&endpoint, ServiceType::Sparql).await {
                            let discovered_endpoint = DiscoveredEndpoint {
                                url: endpoint.clone(),
                                service_type: ServiceType::Sparql,
                                discovery_method: DiscoveryMethod::DNS,
                                metadata: HashMap::new(),
                                timestamp: chrono::Utc::now(),
                            };

                            let mut services = discovered.write().await;
                            services.insert(endpoint.clone(), discovered_endpoint.clone());

                            if let Err(e) = tx.send(discovered_endpoint).await {
                                error!("Failed to send discovered endpoint: {}", e);
                            }
                        }
                    }

                    // Try common GraphQL endpoint patterns
                    let graphql_endpoints = vec![
                        format!("https://{}/graphql", domain),
                        format!("https://graphql.{}", domain),
                        format!("https://{}/api/graphql", domain),
                        format!("https://api.{}/graphql", domain),
                    ];

                    for endpoint in graphql_endpoints {
                        if Self::test_endpoint(&endpoint, ServiceType::GraphQL).await {
                            let discovered_endpoint = DiscoveredEndpoint {
                                url: endpoint.clone(),
                                service_type: ServiceType::GraphQL,
                                discovery_method: DiscoveryMethod::DNS,
                                metadata: HashMap::new(),
                                timestamp: chrono::Utc::now(),
                            };

                            let mut services = discovered.write().await;
                            services.insert(endpoint.clone(), discovered_endpoint.clone());

                            if let Err(e) = tx.send(discovered_endpoint).await {
                                error!("Failed to send discovered endpoint: {}", e);
                            }
                        }
                    }
                }
            }
        });

        Ok(())
    }

    /// Test if an endpoint is accessible
    async fn test_endpoint(endpoint: &str, service_type: ServiceType) -> bool {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        match service_type {
            ServiceType::Sparql => {
                // Try a simple ASK query
                let response = client
                    .post(endpoint)
                    .header("Content-Type", "application/sparql-query")
                    .header("Accept", "application/sparql-results+json")
                    .body("ASK { ?s ?p ?o }")
                    .send()
                    .await;

                matches!(response, Ok(resp) if resp.status().is_success())
            }
            ServiceType::GraphQL => {
                // Try introspection query
                let response = client
                    .post(endpoint)
                    .header("Content-Type", "application/json")
                    .json(&serde_json::json!({
                        "query": "{ __schema { queryType { name } } }"
                    }))
                    .send()
                    .await;

                matches!(response, Ok(resp) if resp.status().is_success())
            }
            _ => false,
        }
    }

    /// Start Kubernetes service discovery
    async fn start_kubernetes_discovery(&self, tx: mpsc::Sender<DiscoveredEndpoint>) -> Result<()> {
        #[cfg(feature = "kubernetes")]
        {
            info!("Starting Kubernetes service discovery");

            use crate::k8s_discovery::{K8sDiscoveryConfig, K8sServiceDiscovery};

            let k8s_config = K8sDiscoveryConfig {
                namespace: self.config.k8s_namespace.clone(),
                label_selectors: self.config.k8s_label_selectors.clone(),
                use_cluster_dns: self.config.k8s_use_cluster_dns,
                external_domain: self.config.k8s_external_domain.clone(),
                ..Default::default()
            };

            let mut k8s_discovery = K8sServiceDiscovery::new(k8s_config).await?;
            let mut k8s_rx = k8s_discovery.start().await?;

            // Spawn task to forward Kubernetes discoveries
            tokio::spawn(async move {
                while let Some(discovered) = k8s_rx.recv().await {
                    if let Err(e) = tx.send(discovered).await {
                        error!("Failed to forward Kubernetes discovery: {}", e);
                        break;
                    }
                }
            });
        }

        #[cfg(not(feature = "kubernetes"))]
        {
            let _ = tx; // Suppress unused variable warning when feature is disabled
            warn!("Kubernetes discovery requested but feature not enabled");
        }

        Ok(())
    }

    /// Get all discovered services
    pub async fn get_discovered_services(&self) -> Vec<DiscoveredEndpoint> {
        let services = self.discovered_services.read().await;
        services.values().cloned().collect()
    }

    /// Register discovered services with a service registry
    pub async fn register_discovered_services(
        &self,
        registry: &mut ServiceRegistry,
        discovery_service: &ServiceDiscovery,
    ) -> Result<usize> {
        let services = self.get_discovered_services().await;
        let mut registered_count = 0;

        for discovered in services {
            // Use the ServiceDiscovery to get full service details
            if let Ok(Some(service)) = discovery_service
                .discover_service_at_endpoint(&discovered.url)
                .await
            {
                // Enhance with discovery metadata
                let mut enhanced_service = service;
                enhanced_service.metadata.tags.push(format!(
                    "discovered:{}",
                    discovered.discovery_method.as_str()
                ));

                match registry.register(enhanced_service).await {
                    Err(e) => {
                        warn!(
                            "Failed to register discovered service {}: {}",
                            discovered.url, e
                        );
                    }
                    _ => {
                        registered_count += 1;
                        info!("Registered discovered service: {}", discovered.url);
                    }
                }
            }
        }

        Ok(registered_count)
    }
}

/// Configuration for automatic discovery
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoDiscoveryConfig {
    /// Enable mDNS/DNS-SD discovery
    pub enable_mdns: bool,

    /// Enable DNS-based discovery
    pub enable_dns_discovery: bool,

    /// Enable Kubernetes service discovery
    pub enable_kubernetes_discovery: bool,

    /// DNS domains to check for services
    pub dns_domains: Vec<String>,

    /// Service name patterns to search for
    pub service_patterns: Vec<String>,

    /// Discovery interval
    pub discovery_interval: Duration,

    /// Maximum concurrent discovery operations
    pub max_concurrent_discoveries: usize,

    /// Kubernetes namespace to watch (None for all namespaces)
    pub k8s_namespace: Option<String>,

    /// Kubernetes label selectors
    pub k8s_label_selectors: HashMap<String, String>,

    /// Use Kubernetes cluster DNS names
    pub k8s_use_cluster_dns: bool,

    /// External domain for Kubernetes services
    pub k8s_external_domain: Option<String>,
}

impl Default for AutoDiscoveryConfig {
    fn default() -> Self {
        let mut k8s_label_selectors = HashMap::new();
        k8s_label_selectors.insert("federation".to_string(), "enabled".to_string());

        Self {
            enable_mdns: true,
            enable_dns_discovery: true,
            enable_kubernetes_discovery: false,
            dns_domains: vec![
                "dbpedia.org".to_string(),
                "wikidata.org".to_string(),
                "data.gov".to_string(),
            ],
            service_patterns: vec![
                "_sparql._tcp".to_string(),
                "_graphql._tcp".to_string(),
                "_rdf._tcp".to_string(),
            ],
            discovery_interval: Duration::from_secs(300),
            max_concurrent_discoveries: 10,
            k8s_namespace: None,
            k8s_label_selectors,
            k8s_use_cluster_dns: true,
            k8s_external_domain: None,
        }
    }
}

/// Discovered endpoint information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredEndpoint {
    /// Endpoint URL
    pub url: String,

    /// Service type
    pub service_type: ServiceType,

    /// How the service was discovered
    pub discovery_method: DiscoveryMethod,

    /// Additional metadata from discovery
    pub metadata: HashMap<String, String>,

    /// When the service was discovered
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Method used to discover a service
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryMethod {
    /// mDNS/DNS-SD
    MDNS,
    /// DNS-based discovery
    DNS,
    /// Kubernetes service discovery
    Kubernetes,
    /// Manual registration
    Manual,
}

impl DiscoveryMethod {
    pub fn as_str(&self) -> &'static str {
        match self {
            DiscoveryMethod::MDNS => "mdns",
            DiscoveryMethod::DNS => "dns",
            DiscoveryMethod::Kubernetes => "kubernetes",
            DiscoveryMethod::Manual => "manual",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_discovery_config() {
        let config = AutoDiscoveryConfig::default();
        assert!(config.enable_mdns);
        assert!(config.enable_dns_discovery);
        assert!(!config.enable_kubernetes_discovery);
    }

    #[tokio::test]
    async fn test_auto_discovery_creation() {
        let config = AutoDiscoveryConfig::default();
        let discovery = AutoDiscovery::new(config);
        let services = discovery.get_discovered_services().await;
        assert!(services.is_empty());
    }

    #[test]
    fn test_discovery_method_string() {
        assert_eq!(DiscoveryMethod::MDNS.as_str(), "mdns");
        assert_eq!(DiscoveryMethod::DNS.as_str(), "dns");
        assert_eq!(DiscoveryMethod::Kubernetes.as_str(), "kubernetes");
        assert_eq!(DiscoveryMethod::Manual.as_str(), "manual");
    }
}
