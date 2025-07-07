//! Kubernetes Service Discovery for Federated Services
//!
//! This module implements automatic discovery of SPARQL and GraphQL services
//! running in Kubernetes clusters by watching for services with specific
//! labels and annotations.

use crate::FederatedService;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::warn;

#[cfg(feature = "kubernetes")]
use k8s_openapi::api::core::v1::{Service as K8sService, ServicePort};
#[cfg(feature = "kubernetes")]
use kube::{
    api::{Api, ListParams, WatchEvent, WatchParams},
    runtime::watcher::{watcher, Config as WatcherConfig, Event},
    Client,
};

use crate::DiscoveredEndpoint;

/// Kubernetes service discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct K8sDiscoveryConfig {
    /// Namespace to watch (None for all namespaces)
    pub namespace: Option<String>,

    /// Labels to filter services
    pub label_selectors: HashMap<String, String>,

    /// Annotations that identify service types
    pub service_type_annotations: ServiceTypeAnnotations,

    /// Whether to use cluster DNS names
    pub use_cluster_dns: bool,

    /// External domain for services (if exposed)
    pub external_domain: Option<String>,

    /// Retry configuration
    pub retry_interval: Duration,

    /// Maximum retry attempts
    pub max_retries: usize,
}

impl Default for K8sDiscoveryConfig {
    fn default() -> Self {
        let mut label_selectors = HashMap::new();
        label_selectors.insert("federation".to_string(), "enabled".to_string());

        Self {
            namespace: None,
            label_selectors,
            service_type_annotations: ServiceTypeAnnotations::default(),
            use_cluster_dns: true,
            external_domain: None,
            retry_interval: Duration::from_secs(30),
            max_retries: 3,
        }
    }
}

/// Annotations that identify service types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceTypeAnnotations {
    /// Annotation key for service type
    pub service_type_key: String,

    /// Annotation value for SPARQL services
    pub sparql_value: String,

    /// Annotation value for GraphQL services
    pub graphql_value: String,

    /// Annotation key for endpoint path
    pub endpoint_path_key: String,

    /// Annotation key for authentication type
    pub auth_type_key: String,
}

impl Default for ServiceTypeAnnotations {
    fn default() -> Self {
        Self {
            service_type_key: "oxirs.federate/service-type".to_string(),
            sparql_value: "sparql".to_string(),
            graphql_value: "graphql".to_string(),
            endpoint_path_key: "oxirs.federate/endpoint-path".to_string(),
            auth_type_key: "oxirs.federate/auth-type".to_string(),
        }
    }
}

/// Kubernetes service discovery implementation
#[cfg(feature = "kubernetes")]
pub struct K8sServiceDiscovery {
    config: K8sDiscoveryConfig,
    client: Client,
    discovery_tx: Option<mpsc::Sender<DiscoveredEndpoint>>,
}

#[cfg(feature = "kubernetes")]
impl K8sServiceDiscovery {
    /// Create a new Kubernetes service discovery instance
    pub async fn new(config: K8sDiscoveryConfig) -> Result<Self> {
        let client = Client::try_default().await?;

        Ok(Self {
            config,
            client,
            discovery_tx: None,
        })
    }

    /// Start watching for services
    pub async fn start(&mut self) -> Result<mpsc::Receiver<DiscoveredEndpoint>> {
        info!("Starting Kubernetes service discovery");

        let (tx, rx) = mpsc::channel(100);
        self.discovery_tx = Some(tx.clone());

        let client = self.client.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            if let Err(e) = Self::watch_services(client, config, tx).await {
                error!("Kubernetes service watcher error: {}", e);
            }
        });

        Ok(rx)
    }

    /// Watch for Kubernetes services
    async fn watch_services(
        client: Client,
        config: K8sDiscoveryConfig,
        tx: mpsc::Sender<DiscoveredEndpoint>,
    ) -> Result<()> {
        let services: Api<K8sService> = if let Some(ns) = &config.namespace {
            Api::namespaced(client, ns)
        } else {
            Api::all(client)
        };

        let mut list_params = ListParams::default();

        // Build label selector
        if !config.label_selectors.is_empty() {
            let selector = config
                .label_selectors
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(",");
            list_params = list_params.labels(&selector);
        }

        let watcher_config = WatcherConfig::default();
        let mut stream = watcher(services, watcher_config)
            .default_backoff()
            .try_for_each(|event| async {
                match event {
                    Event::Applied(service) => {
                        Self::handle_service_event(&service, &config, &tx, true).await;
                    }
                    Event::Deleted(service) => {
                        Self::handle_service_event(&service, &config, &tx, false).await;
                    }
                    Event::Restarted(services) => {
                        info!(
                            "Service watch restarted, processing {} services",
                            services.len()
                        );
                        for service in services {
                            Self::handle_service_event(&service, &config, &tx, true).await;
                        }
                    }
                }
                Ok(())
            });

        stream.await?;
        Ok(())
    }

    /// Handle a service event
    async fn handle_service_event(
        service: &K8sService,
        config: &K8sDiscoveryConfig,
        tx: &mpsc::Sender<DiscoveredEndpoint>,
        is_added: bool,
    ) {
        let service_name = service
            .metadata
            .name
            .as_ref()
            .unwrap_or(&"unknown".to_string())
            .clone();

        let namespace = service
            .metadata
            .namespace
            .as_ref()
            .unwrap_or(&"default".to_string())
            .clone();

        debug!(
            "Processing service event: {} in namespace {} (added: {})",
            service_name, namespace, is_added
        );

        if !is_added {
            // For now, we don't handle service deletions
            // In a full implementation, we'd notify about removed services
            return;
        }

        // Extract service type from annotations
        let annotations = service.metadata.annotations.as_ref();
        let service_type =
            Self::extract_service_type(annotations, &config.service_type_annotations);

        if service_type.is_none() {
            debug!(
                "Service {} does not have service type annotation, skipping",
                service_name
            );
            return;
        }

        let service_type = service_type.unwrap();

        // Extract endpoint information
        if let Some(endpoints) = Self::extract_endpoints(service, config, &service_type) {
            for endpoint in endpoints {
                let discovered = DiscoveredEndpoint {
                    url: endpoint.url,
                    service_type,
                    discovery_method: DiscoveryMethod::Kubernetes,
                    metadata: endpoint.metadata,
                    timestamp: chrono::Utc::now(),
                };

                if let Err(e) = tx.send(discovered).await {
                    error!("Failed to send discovered endpoint: {}", e);
                }
            }
        }
    }

    /// Extract service type from annotations
    fn extract_service_type(
        annotations: Option<&HashMap<String, String>>,
        type_annotations: &ServiceTypeAnnotations,
    ) -> Option<ServiceType> {
        annotations.and_then(|annots| {
            annots
                .get(&type_annotations.service_type_key)
                .and_then(|value| {
                    if value == &type_annotations.sparql_value {
                        Some(ServiceType::Sparql)
                    } else if value == &type_annotations.graphql_value {
                        Some(ServiceType::GraphQL)
                    } else {
                        None
                    }
                })
        })
    }

    /// Extract endpoints from a Kubernetes service
    fn extract_endpoints(
        service: &K8sService,
        config: &K8sDiscoveryConfig,
        service_type: &ServiceType,
    ) -> Option<Vec<K8sEndpoint>> {
        let service_name = service.metadata.name.as_ref()?;
        let namespace = service.metadata.namespace.as_ref()?;
        let spec = service.spec.as_ref()?;
        let ports = spec.ports.as_ref()?;

        let annotations = service.metadata.annotations.as_ref();
        let endpoint_path = annotations
            .and_then(|a| a.get(&config.service_type_annotations.endpoint_path_key))
            .map(|s| s.as_str())
            .unwrap_or_else(|| match service_type {
                ServiceType::Sparql => "/sparql",
                ServiceType::GraphQL => "/graphql",
                _ => "/",
            });

        let auth_type = annotations
            .and_then(|a| a.get(&config.service_type_annotations.auth_type_key))
            .cloned();

        let mut endpoints = Vec::new();

        for port in ports {
            // Skip if port doesn't have required fields
            if port.port == 0 {
                continue;
            }

            let port_name = port.name.as_ref().map(|s| s.as_str()).unwrap_or("http");

            // Determine protocol
            let protocol = if port_name.contains("https") || port.port == 443 {
                "https"
            } else {
                "http"
            };

            // Build URLs based on service type
            let urls = Self::build_service_urls(
                service,
                namespace,
                port.port,
                protocol,
                endpoint_path,
                config,
            );

            for url in urls {
                let mut metadata = HashMap::new();
                metadata.insert("k8s.namespace".to_string(), namespace.clone());
                metadata.insert("k8s.service".to_string(), service_name.clone());
                metadata.insert("k8s.port".to_string(), port.port.to_string());

                if let Some(ref auth) = auth_type {
                    metadata.insert("auth.type".to_string(), auth.clone());
                }

                // Add labels as metadata
                if let Some(labels) = &service.metadata.labels {
                    for (key, value) in labels {
                        metadata.insert(format!("label.{key}"), value.clone());
                    }
                }

                endpoints.push(K8sEndpoint { url, metadata });
            }
        }

        Some(endpoints)
    }

    /// Build service URLs based on configuration
    fn build_service_urls(
        service: &K8sService,
        namespace: &str,
        port: i32,
        protocol: &str,
        endpoint_path: &str,
        config: &K8sDiscoveryConfig,
    ) -> Vec<String> {
        let service_name = service
            .metadata
            .name
            .as_ref()
            .unwrap_or(&"unknown".to_string());

        let mut urls = Vec::new();

        // Cluster-internal URL
        if config.use_cluster_dns {
            let cluster_url = format!(
                "{}://{}.{}.svc.cluster.local:{}{}",
                protocol, service_name, namespace, port, endpoint_path
            );
            urls.push(cluster_url);
        }

        // External URL if LoadBalancer or NodePort with external domain
        if let Some(spec) = &service.spec {
            match spec.type_.as_deref() {
                Some("LoadBalancer") => {
                    if let Some(status) = &service.status {
                        if let Some(lb) = &status.load_balancer {
                            if let Some(ingress) = &lb.ingress {
                                for ing in ingress {
                                    if let Some(hostname) = &ing.hostname {
                                        urls.push(format!(
                                            "{}://{}:{}{}",
                                            protocol, hostname, port, endpoint_path
                                        ));
                                    }
                                    if let Some(ip) = &ing.ip {
                                        urls.push(format!(
                                            "{}://{}:{}{}",
                                            protocol, ip, port, endpoint_path
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                Some("NodePort") => {
                    if let (Some(domain), Some(ports)) = (&config.external_domain, &spec.ports) {
                        for svc_port in ports {
                            if svc_port.port == port {
                                if let Some(node_port) = svc_port.node_port {
                                    urls.push(format!(
                                        "{}://{}:{}{}",
                                        protocol, domain, node_port, endpoint_path
                                    ));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        urls
    }

    /// List all current federated services in the cluster
    pub async fn list_services(&self) -> Result<Vec<FederatedService>> {
        let services: Api<K8sService> = if let Some(ns) = &self.config.namespace {
            Api::namespaced(self.client.clone(), ns)
        } else {
            Api::all(self.client.clone())
        };

        let mut list_params = ListParams::default();

        // Build label selector
        if !self.config.label_selectors.is_empty() {
            let selector = self
                .config
                .label_selectors
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(",");
            list_params = list_params.labels(&selector);
        }

        let service_list = services.list(&list_params).await?;
        let mut federated_services = Vec::new();

        for k8s_service in service_list {
            let annotations = k8s_service.metadata.annotations.as_ref();
            let service_type =
                Self::extract_service_type(annotations, &self.config.service_type_annotations);

            if let Some(service_type) = service_type {
                if let Some(endpoints) =
                    Self::extract_endpoints(&k8s_service, &self.config, &service_type)
                {
                    for endpoint in endpoints {
                        // Try to discover full service details
                        let discovery = ServiceDiscovery::new();
                        if let Ok(Some(mut service)) =
                            discovery.discover_service_at_endpoint(&endpoint.url).await
                        {
                            // Enhance with Kubernetes metadata
                            for (key, value) in endpoint.metadata {
                                service.metadata.tags.push(format!("{key}:{value}"));
                            }
                            federated_services.push(service);
                        }
                    }
                }
            }
        }

        Ok(federated_services)
    }
}

/// Kubernetes endpoint information
struct K8sEndpoint {
    url: String,
    metadata: HashMap<String, String>,
}

/// Stub implementation when Kubernetes feature is not enabled
#[cfg(not(feature = "kubernetes"))]
pub struct K8sServiceDiscovery {
    config: K8sDiscoveryConfig,
}

#[cfg(not(feature = "kubernetes"))]
impl K8sServiceDiscovery {
    pub async fn new(config: K8sDiscoveryConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub async fn start(&mut self) -> Result<mpsc::Receiver<DiscoveredEndpoint>> {
        warn!("Kubernetes discovery requested but feature not enabled");
        let (_tx, rx) = mpsc::channel(1);
        Ok(rx)
    }

    pub async fn list_services(&self) -> Result<Vec<FederatedService>> {
        warn!("Kubernetes discovery requested but feature not enabled");
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k8s_config_default() {
        let config = K8sDiscoveryConfig::default();
        assert!(config.namespace.is_none());
        assert!(config.use_cluster_dns);
        assert_eq!(config.retry_interval, Duration::from_secs(30));
    }

    #[test]
    fn test_service_type_annotations() {
        let annotations = ServiceTypeAnnotations::default();
        assert_eq!(annotations.service_type_key, "oxirs.federate/service-type");
        assert_eq!(annotations.sparql_value, "sparql");
        assert_eq!(annotations.graphql_value, "graphql");
    }
}
