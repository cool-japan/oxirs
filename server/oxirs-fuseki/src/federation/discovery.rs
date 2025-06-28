//! Service discovery for federated SPARQL endpoints

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::{
    sync::{Notify, RwLock},
    time::interval,
};
use trust_dns_resolver::{config::*, Resolver};
use url::Url;

use crate::{
    error::{FusekiError, FusekiResult},
    federation::{
        FederationConfig, ServiceCapabilities, ServiceEndpoint, ServiceHealth, ServiceMetadata,
    },
};

/// Service discovery mechanisms
#[derive(Debug, Clone)]
pub enum DiscoveryMethod {
    /// Static configuration
    Static(Vec<ServiceRegistration>),
    /// DNS-based service discovery
    Dns { domain: String },
    /// Consul service discovery
    Consul { endpoint: Url },
    /// Kubernetes service discovery
    Kubernetes { namespace: String },
    /// SPARQL Service Description
    ServiceDescription,
}

/// Service registration information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceRegistration {
    pub id: String,
    pub url: Url,
    pub metadata: ServiceMetadata,
}

/// Service discovery component
pub struct ServiceDiscovery {
    config: FederationConfig,
    endpoints: Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
    discovery_methods: Vec<DiscoveryMethod>,
    http_client: Client,
    shutdown: Arc<Notify>,
}

impl ServiceDiscovery {
    /// Create a new service discovery instance
    pub fn new(
        config: FederationConfig,
        endpoints: Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
    ) -> Self {
        Self {
            config,
            endpoints,
            discovery_methods: Vec::new(),
            http_client: Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .unwrap(),
            shutdown: Arc::new(Notify::new()),
        }
    }

    /// Add a discovery method
    pub fn add_method(&mut self, method: DiscoveryMethod) {
        self.discovery_methods.push(method);
    }

    /// Start service discovery
    pub async fn start(&self) -> Result<()> {
        let shutdown = self.shutdown.clone();
        let config = self.config.clone();
        let endpoints = self.endpoints.clone();
        let methods = self.discovery_methods.clone();
        let client = self.http_client.clone();

        tokio::spawn(async move {
            let mut interval = interval(config.discovery_interval);

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = Self::discover_services(&methods, &endpoints, &client).await {
                            tracing::error!("Service discovery error: {}", e);
                        }
                    }
                    _ = shutdown.notified() => {
                        tracing::info!("Service discovery shutting down");
                        break;
                    }
                }
            }
        });

        // Run initial discovery
        Self::discover_services(&self.discovery_methods, &self.endpoints, &self.http_client)
            .await?;

        Ok(())
    }

    /// Stop service discovery
    pub async fn stop(&self) -> Result<()> {
        self.shutdown.notify_one();
        Ok(())
    }

    /// Discover services using configured methods
    async fn discover_services(
        methods: &[DiscoveryMethod],
        endpoints: &Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
        client: &Client,
    ) -> Result<()> {
        for method in methods {
            match method {
                DiscoveryMethod::Static(registrations) => {
                    Self::discover_static(registrations, endpoints).await?;
                }
                DiscoveryMethod::ServiceDescription => {
                    Self::discover_via_service_description(endpoints, client).await?;
                }
                DiscoveryMethod::Dns { domain } => {
                    Self::discover_via_dns(domain, endpoints, client).await?;
                }
                DiscoveryMethod::Consul { endpoint } => {
                    Self::discover_via_consul(endpoint, endpoints, client).await?;
                }
                DiscoveryMethod::Kubernetes { namespace } => {
                    Self::discover_via_kubernetes(namespace, endpoints, client).await?;
                }
            }
        }
        Ok(())
    }

    /// Discover services from static configuration
    async fn discover_static(
        registrations: &[ServiceRegistration],
        endpoints: &Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
    ) -> Result<()> {
        let mut eps = endpoints.write().await;

        for reg in registrations {
            let endpoint = ServiceEndpoint {
                url: reg.url.clone(),
                metadata: reg.metadata.clone(),
                health: ServiceHealth::Unknown,
                capabilities: ServiceCapabilities::default(),
            };

            eps.insert(reg.id.clone(), endpoint);
        }

        Ok(())
    }

    /// Discover services using SPARQL Service Description
    async fn discover_via_service_description(
        endpoints: &Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
        client: &Client,
    ) -> Result<()> {
        let eps = endpoints.read().await;
        let service_urls: Vec<_> = eps.values().map(|ep| ep.url.clone()).collect();
        drop(eps);

        for url in service_urls {
            if let Ok(capabilities) = Self::fetch_service_description(&url, client).await {
                let mut eps = endpoints.write().await;
                if let Some(ep) = eps.values_mut().find(|ep| ep.url == url) {
                    ep.capabilities = capabilities;
                }
            }
        }

        Ok(())
    }

    /// Fetch SPARQL Service Description
    async fn fetch_service_description(
        base_url: &Url,
        client: &Client,
    ) -> Result<ServiceCapabilities> {
        let query = r#"
            PREFIX sd: <http://www.w3.org/ns/sparql-service-description#>
            PREFIX void: <http://rdfs.org/ns/void#>
            
            SELECT ?feature ?triples WHERE {
                ?service a sd:Service ;
                    sd:supportedLanguage ?feature .
                OPTIONAL {
                    ?service sd:defaultDataset/void:triples ?triples
                }
            }
        "#;

        let response = client
            .get(base_url.as_str())
            .query(&[("query", query)])
            .header("Accept", "application/sparql-results+json")
            .send()
            .await
            .map_err(|e| Error::Custom(format!("Failed to fetch service description: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::Custom(format!(
                "Service description query failed: {}",
                response.status()
            )));
        }

        // Parse SPARQL results and build capabilities
        let mut capabilities = ServiceCapabilities::default();

        // For now, return basic capabilities
        // TODO: Parse actual SPARQL JSON results
        capabilities.sparql_features = vec![
            "SPARQL 1.1 Query".to_string(),
            "SPARQL 1.1 Update".to_string(),
        ];
        capabilities.result_formats = vec![
            "application/sparql-results+json".to_string(),
            "application/sparql-results+xml".to_string(),
        ];

        Ok(capabilities)
    }

    /// Discover services via DNS SRV records
    async fn discover_via_dns(
        domain: &str,
        endpoints: &Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
        client: &Client,
    ) -> Result<()> {
        tracing::info!(
            "Starting DNS-based service discovery for domain: {}",
            domain
        );

        // Create DNS resolver
        let resolver = Resolver::new(ResolverConfig::default(), ResolverOpts::default())
            .map_err(|e| Error::Custom(format!("Failed to create DNS resolver: {}", e)))?;

        // Look up SRV records for SPARQL services
        // Convention: _sparql._tcp.domain.com
        let srv_query = format!("_sparql._tcp.{}", domain);

        match resolver.srv_lookup(&srv_query).await {
            Ok(lookup) => {
                let mut eps = endpoints.write().await;
                let mut discovered_count = 0;

                for record in lookup.iter() {
                    let target = record.target().to_string();
                    let port = record.port();

                    // Construct service URL
                    let service_url = match Url::parse(&format!(
                        "http://{}:{}/sparql",
                        target.trim_end_matches('.'),
                        port
                    )) {
                        Ok(url) => url,
                        Err(e) => {
                            tracing::warn!("Invalid URL for SRV record {}:{}: {}", target, port, e);
                            continue;
                        }
                    };

                    // Create unique service ID
                    let service_id = format!("dns-{}:{}", target.trim_end_matches('.'), port);

                    // Check if service is reachable
                    match Self::check_service_health(&service_url, client).await {
                        Ok(health) => {
                            let endpoint = ServiceEndpoint {
                                url: service_url,
                                metadata: ServiceMetadata {
                                    name: format!(
                                        "SPARQL Service at {}:{}",
                                        target.trim_end_matches('.'),
                                        port
                                    ),
                                    description: Some(format!(
                                        "Discovered via DNS SRV record for {}",
                                        domain
                                    )),
                                    version: None,
                                    contact: None,
                                },
                                health,
                                capabilities: ServiceCapabilities::default(),
                            };

                            eps.insert(service_id.clone(), endpoint);
                            discovered_count += 1;

                            tracing::info!(
                                "Discovered SPARQL service: {} at {}:{}",
                                service_id,
                                target,
                                port
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Service at {}:{} is not reachable: {}",
                                target,
                                port,
                                e
                            );
                        }
                    }
                }

                tracing::info!(
                    "DNS discovery completed: {} services discovered for domain {}",
                    discovered_count,
                    domain
                );
            }
            Err(e) => {
                tracing::warn!("No SRV records found for {}: {}", srv_query, e);

                // Fallback: try common SPARQL service ports on the domain itself
                Self::discover_via_fallback_ports(domain, endpoints, client).await?;
            }
        }

        Ok(())
    }

    /// Discover services via Consul
    async fn discover_via_consul(
        consul_endpoint: &Url,
        endpoints: &Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
        client: &Client,
    ) -> Result<()> {
        tracing::info!(
            "Starting Consul-based service discovery from: {}",
            consul_endpoint
        );

        // Query Consul for services with "sparql" tag
        let consul_url = format!("{}/v1/health/service/sparql?passing=true", consul_endpoint);

        let response = client
            .get(&consul_url)
            .header("Accept", "application/json")
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| Error::Custom(format!("Failed to query Consul: {}", e)))?;

        if !response.status().is_success() {
            return Err(Error::Custom(format!(
                "Consul query failed with status: {}",
                response.status()
            )));
        }

        let consul_services: serde_json::Value = response
            .json()
            .await
            .map_err(|e| Error::Custom(format!("Failed to parse Consul response: {}", e)))?;

        let mut eps = endpoints.write().await;
        let mut discovered_count = 0;

        if let Some(services) = consul_services.as_array() {
            for service in services {
                if let (Some(service_obj), Some(checks)) = (
                    service.get("Service").and_then(|s| s.as_object()),
                    service.get("Checks").and_then(|c| c.as_array()),
                ) {
                    // Extract service information
                    let service_name = service_obj
                        .get("Service")
                        .and_then(|s| s.as_str())
                        .unwrap_or("unknown");
                    let service_id = service_obj
                        .get("ID")
                        .and_then(|id| id.as_str())
                        .unwrap_or(service_name);
                    let address = service_obj
                        .get("Address")
                        .and_then(|a| a.as_str())
                        .unwrap_or("localhost");
                    let port = service_obj
                        .get("Port")
                        .and_then(|p| p.as_u64())
                        .unwrap_or(8080) as u16;

                    // Check if all health checks are passing
                    let all_passing = checks.iter().all(|check| {
                        check
                            .get("Status")
                            .and_then(|s| s.as_str())
                            .map(|status| status == "passing")
                            .unwrap_or(false)
                    });

                    if !all_passing {
                        tracing::debug!("Skipping unhealthy Consul service: {}", service_id);
                        continue;
                    }

                    // Extract metadata from service tags
                    let tags = service_obj
                        .get("Tags")
                        .and_then(|t| t.as_array())
                        .map(|tags| tags.iter().filter_map(|t| t.as_str()).collect::<Vec<_>>())
                        .unwrap_or_default();

                    let sparql_path = tags
                        .iter()
                        .find(|tag| tag.starts_with("sparql-path="))
                        .map(|tag| tag.strip_prefix("sparql-path=").unwrap_or("/sparql"))
                        .unwrap_or("/sparql");

                    // Construct service URL
                    let service_url =
                        match Url::parse(&format!("http://{}:{}{}", address, port, sparql_path)) {
                            Ok(url) => url,
                            Err(e) => {
                                tracing::warn!(
                                    "Invalid URL for Consul service {}:{}:{}: {}",
                                    service_id,
                                    address,
                                    port,
                                    e
                                );
                                continue;
                            }
                        };

                    // Double-check service health
                    match Self::check_service_health(&service_url, client).await {
                        Ok(health) => {
                            let consul_service_id = format!("consul-{}", service_id);

                            let endpoint = ServiceEndpoint {
                                url: service_url.clone(),
                                metadata: ServiceMetadata {
                                    name: format!("Consul Service: {}", service_name),
                                    description: Some(format!(
                                        "Discovered via Consul from {}",
                                        consul_endpoint
                                    )),
                                    version: tags
                                        .iter()
                                        .find(|tag| tag.starts_with("version="))
                                        .map(|tag| {
                                            tag.strip_prefix("version=")
                                                .unwrap_or("unknown")
                                                .to_string()
                                        }),
                                    contact: None,
                                },
                                health,
                                capabilities: ServiceCapabilities::default(),
                            };

                            eps.insert(consul_service_id.clone(), endpoint);
                            discovered_count += 1;

                            tracing::info!(
                                "Discovered SPARQL service via Consul: {} at {}",
                                consul_service_id,
                                service_url
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Consul service {} at {} failed health check: {}",
                                service_id,
                                service_url,
                                e
                            );
                        }
                    }
                }
            }
        }

        tracing::info!(
            "Consul discovery completed: {} services discovered from {}",
            discovered_count,
            consul_endpoint
        );
        Ok(())
    }

    /// Discover services via Kubernetes
    async fn discover_via_kubernetes(
        namespace: &str,
        endpoints: &Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
        client: &Client,
    ) -> Result<()> {
        // TODO: Implement Kubernetes-based discovery
        tracing::debug!(
            "Kubernetes discovery not yet implemented for namespace: {}",
            namespace
        );
        Ok(())
    }

    /// Check if a SPARQL service is healthy and reachable
    async fn check_service_health(url: &Url, client: &Client) -> Result<ServiceHealth> {
        let health_check_query = "ASK { ?s ?p ?o }";

        let response = client
            .get(url.as_str())
            .query(&[("query", health_check_query)])
            .header("Accept", "application/sparql-results+json")
            .timeout(Duration::from_secs(5))
            .send()
            .await;

        match response {
            Ok(resp) if resp.status().is_success() => {
                tracing::debug!("Service at {} is healthy", url);
                Ok(ServiceHealth::Healthy)
            }
            Ok(resp) => {
                tracing::warn!("Service at {} returned status: {}", url, resp.status());
                Ok(ServiceHealth::Unhealthy)
            }
            Err(e) => {
                tracing::warn!("Failed to reach service at {}: {}", url, e);
                Err(Error::Custom(format!("Service health check failed: {}", e)))
            }
        }
    }

    /// Fallback discovery by trying common SPARQL ports
    async fn discover_via_fallback_ports(
        domain: &str,
        endpoints: &Arc<RwLock<HashMap<String, ServiceEndpoint>>>,
        client: &Client,
    ) -> Result<()> {
        let common_ports = [8080, 3030, 8000, 80, 443];
        let common_paths = ["/sparql", "/query", "/sparql/query"];

        let mut eps = endpoints.write().await;
        let mut discovered_count = 0;

        for port in &common_ports {
            for path in &common_paths {
                let scheme = if *port == 443 { "https" } else { "http" };
                let service_url =
                    match Url::parse(&format!("{}://{}:{}{}", scheme, domain, port, path)) {
                        Ok(url) => url,
                        Err(e) => {
                            tracing::debug!(
                                "Invalid fallback URL for {}:{}{}: {}",
                                domain,
                                port,
                                path,
                                e
                            );
                            continue;
                        }
                    };

                match Self::check_service_health(&service_url, client).await {
                    Ok(health) => {
                        let service_id = format!("fallback-{}:{}{}", domain, port, path);

                        let endpoint = ServiceEndpoint {
                            url: service_url.clone(),
                            metadata: ServiceMetadata {
                                name: format!("SPARQL Service at {}:{}{}", domain, port, path),
                                description: Some(
                                    "Discovered via fallback port scanning".to_string(),
                                ),
                                version: None,
                                contact: None,
                            },
                            health,
                            capabilities: ServiceCapabilities::default(),
                        };

                        eps.insert(service_id.clone(), endpoint);
                        discovered_count += 1;

                        tracing::info!(
                            "Discovered SPARQL service via fallback: {} at {}",
                            service_id,
                            service_url
                        );

                        // Only discover one service per port to avoid duplicates
                        break;
                    }
                    Err(_) => {
                        // Service not reachable, continue to next path/port
                        tracing::debug!("No SPARQL service found at {}:{}{}", domain, port, path);
                    }
                }
            }
        }

        tracing::info!(
            "Fallback discovery completed: {} services discovered for domain {}",
            discovered_count,
            domain
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_static_discovery() {
        let endpoints = Arc::new(RwLock::new(HashMap::new()));
        let config = FederationConfig::default();
        let mut discovery = ServiceDiscovery::new(config, endpoints.clone());

        let registrations = vec![ServiceRegistration {
            id: "test-service".to_string(),
            url: Url::parse("http://example.com/sparql").unwrap(),
            metadata: ServiceMetadata {
                name: "Test Service".to_string(),
                ..Default::default()
            },
        }];

        discovery.add_method(DiscoveryMethod::Static(registrations));
        discovery.start().await.unwrap();

        // Give it time to discover
        tokio::time::sleep(Duration::from_millis(100)).await;

        let eps = endpoints.read().await;
        assert_eq!(eps.len(), 1);
        assert!(eps.contains_key("test-service"));

        discovery.stop().await.unwrap();
    }
}
