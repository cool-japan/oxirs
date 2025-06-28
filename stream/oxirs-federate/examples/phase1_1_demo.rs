//! Phase 1.1 Service Registry and Discovery Demo
//!
//! This example demonstrates the core Phase 1.1 functionality:
//! - Service Registry with health monitoring and capability detection
//! - Automatic Service Discovery (mDNS, DNS, Kubernetes)
//! - Extended metadata collection and management
//! - Connection pooling and rate limiting

use anyhow::Result;
use oxirs_federate::{
    AuthConfig, AuthCredentials, AuthType, AutoDiscovery, AutoDiscoveryConfig, FederatedService,
    ServiceDiscovery, ServiceRegistry, ServiceType,
};
use std::time::Duration;
use tokio::time::sleep;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::init();

    info!("🚀 OxiRS Federation Engine - Phase 1.1 Demo");
    info!("===============================================");

    // Demo 1: Service Registry with Manual Registration
    demo_service_registry().await?;

    // Demo 2: Service Discovery and Capability Detection
    demo_service_discovery().await?;

    // Demo 3: Automatic Discovery
    demo_auto_discovery().await?;

    // Demo 4: Extended Metadata and Health Monitoring
    demo_extended_metadata().await?;

    info!("✅ Phase 1.1 demonstration completed successfully!");
    Ok(())
}

/// Demonstrate core service registry functionality
async fn demo_service_registry() -> Result<()> {
    info!("\n📋 Demo 1: Service Registry");
    info!("---------------------------");

    let mut registry = ServiceRegistry::new();

    // Create a sample SPARQL service
    let sparql_service = FederatedService::new_sparql(
        "dbpedia".to_string(),
        "DBpedia SPARQL Endpoint".to_string(),
        "https://dbpedia.org/sparql".to_string(),
    );

    // Create a sample GraphQL service with authentication
    let mut graphql_service = FederatedService::new_graphql(
        "countries-api".to_string(),
        "Countries GraphQL API".to_string(),
        "https://countries.trevorblades.com/".to_string(),
    );

    // Add authentication configuration
    graphql_service.auth = Some(AuthConfig {
        auth_type: AuthType::ApiKey,
        credentials: AuthCredentials {
            api_key: Some("example-api-key".to_string()),
            api_key_header: Some("X-API-Key".to_string()),
            ..Default::default()
        },
    });

    // Register services
    info!("Registering SPARQL service: {}", sparql_service.name);
    if let Err(e) = registry.register(sparql_service).await {
        warn!("Failed to register SPARQL service: {}", e);
    }

    info!("Registering GraphQL service: {}", graphql_service.name);
    if let Err(e) = registry.register(graphql_service).await {
        warn!("Failed to register GraphQL service: {}", e);
    }

    // Get registry statistics
    let stats = registry.get_stats().await;
    info!("📊 Registry Stats:");
    info!("  - Total services: {}", stats.total_services);
    info!("  - Healthy services: {}", stats.healthy_services);
    info!(
        "  - Capabilities distribution: {:?}",
        stats.capabilities_distribution
    );

    // Perform health check
    match registry.health_check().await {
        Ok(health) => {
            info!("🏥 Health Check Results:");
            info!("  - Overall status: {:?}", health.overall_status);
            info!("  - Service statuses: {:?}", health.service_statuses);
        }
        Err(e) => warn!("Health check failed: {}", e),
    }

    Ok(())
}

/// Demonstrate service discovery capabilities
async fn demo_service_discovery() -> Result<()> {
    info!("\n🔍 Demo 2: Service Discovery");
    info!("-----------------------------");

    let discovery = ServiceDiscovery::new();

    // Test endpoints for discovery
    let test_endpoints = vec![
        "https://dbpedia.org".to_string(),
        "https://query.wikidata.org".to_string(),
        "https://countries.trevorblades.com".to_string(),
    ];

    info!(
        "Discovering services from {} endpoints...",
        test_endpoints.len()
    );

    match discovery.discover_services(&test_endpoints).await {
        Ok(services) => {
            info!("✅ Discovered {} services:", services.len());
            for service in services {
                info!(
                    "  - {} ({}): {}",
                    service.name,
                    format!("{:?}", service.service_type),
                    service.endpoint
                );
                info!("    Capabilities: {:?}", service.capabilities);
            }
        }
        Err(e) => warn!("Service discovery failed: {}", e),
    }

    Ok(())
}

/// Demonstrate automatic discovery features
async fn demo_auto_discovery() -> Result<()> {
    info!("\n🤖 Demo 3: Automatic Discovery");
    info!("-------------------------------");

    let config = AutoDiscoveryConfig {
        enable_mdns: false, // Disabled for demo - requires mDNS feature
        enable_dns_discovery: true,
        enable_kubernetes_discovery: false, // Disabled for demo
        dns_domains: vec!["example.com".to_string(), "test.org".to_string()],
        discovery_interval: Duration::from_secs(10),
        ..Default::default()
    };

    let mut auto_discovery = AutoDiscovery::new(config);
    info!("Starting automatic discovery (running for 5 seconds)...");

    match auto_discovery.start().await {
        Ok(mut receiver) => {
            // Listen for discoveries for a short time
            let timeout = sleep(Duration::from_secs(5));
            tokio::pin!(timeout);

            loop {
                tokio::select! {
                    discovered = receiver.recv() => {
                        if let Some(endpoint) = discovered {
                            info!("🎯 Auto-discovered: {} ({:?}) via {:?}",
                                  endpoint.url,
                                  endpoint.service_type,
                                  endpoint.discovery_method);
                        }
                    }
                    _ = &mut timeout => {
                        info!("Auto-discovery demo timeout reached");
                        break;
                    }
                }
            }

            auto_discovery.stop().await;

            let discovered_services = auto_discovery.get_discovered_services().await;
            info!(
                "📈 Total services discovered: {}",
                discovered_services.len()
            );
        }
        Err(e) => warn!("Auto-discovery startup failed: {}", e),
    }

    Ok(())
}

/// Demonstrate extended metadata and monitoring
async fn demo_extended_metadata() -> Result<()> {
    info!("\n📊 Demo 4: Extended Metadata & Monitoring");
    info!("------------------------------------------");

    let mut registry = ServiceRegistry::new();

    // Create a service for metadata collection
    let mut service = FederatedService::new_sparql(
        "wikidata".to_string(),
        "Wikidata SPARQL Endpoint".to_string(),
        "https://query.wikidata.org/sparql".to_string(),
    );

    // Add some metadata tags
    service.metadata.tags.push("public".to_string());
    service.metadata.tags.push("linked-data".to_string());
    service.metadata.description = Some("Wikidata knowledge base SPARQL endpoint".to_string());

    if let Err(e) = registry.register(service).await {
        warn!("Failed to register service for metadata demo: {}", e);
        return Ok(());
    }

    // Enable extended metadata collection
    match registry.enable_extended_metadata("wikidata").await {
        Ok(_) => info!("✅ Extended metadata enabled for Wikidata service"),
        Err(e) => warn!("Failed to enable extended metadata: {}", e),
    }

    // Perform comprehensive assessment
    match registry.assess_service_comprehensively("wikidata").await {
        Ok(_) => info!("✅ Comprehensive service assessment completed"),
        Err(e) => warn!("Comprehensive assessment failed: {}", e),
    }

    // Collect dataset statistics
    match registry.collect_dataset_statistics("wikidata").await {
        Ok(_) => info!("✅ Dataset statistics collected"),
        Err(e) => warn!("Dataset statistics collection failed: {}", e),
    }

    // Collect vocabulary information
    match registry.collect_vocabulary_info("wikidata").await {
        Ok(_) => info!("✅ Vocabulary information collected"),
        Err(e) => warn!("Vocabulary collection failed: {}", e),
    }

    // Get connection pool statistics
    let pool_stats = registry.get_connection_pool_stats().await;
    info!("🔗 Connection Pool Stats:");
    for (service_id, stats) in pool_stats {
        info!(
            "  - {}: {}/{} connections",
            service_id, stats.active_connections, stats.max_connections
        );
    }

    Ok(())
}
