//! Federation Demo Example
//!
//! This example demonstrates how to set up GraphQL federation with multiple services
//! using the enhanced federation manager with real-time schema synchronization.

use anyhow::Result;
use oxirs_gql::{
    federation::{
        EnhancedFederationConfig, EnhancedFederationManager, LoadBalancingConfig,
        LoadBalancingAlgorithm, SyncConfig, ConflictResolution, ServiceInfo, HealthStatus,
    },
    GraphQLConfig, GraphQLServer, RdfStore,
};
use std::sync::Arc;
use std::time::Duration;
use tracing::{info, Level};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .init();

    info!("Starting OxiRS GraphQL Federation Demo");

    // Create local RDF store and schema
    let store = Arc::new(RdfStore::new()?);
    let local_schema = create_local_schema(&store).await?;

    // Configure federation with advanced features
    let federation_config = EnhancedFederationConfig {
        load_balancing: LoadBalancingConfig {
            algorithm: LoadBalancingAlgorithm::WeightedRoundRobin,
            health_check_weight: 0.4,
            response_time_weight: 0.3,
            load_weight: 0.3,
            circuit_breaker_enabled: true,
            circuit_breaker_failure_threshold: 5,
            circuit_breaker_timeout: Duration::from_secs(30),
            ..Default::default()
        },
        real_time_sync: SyncConfig {
            sync_interval: Duration::from_secs(30),
            conflict_resolution: ConflictResolution::LastWriterWins,
            enable_change_notifications: true,
            schema_version_tracking: true,
            ..Default::default()
        },
        service_discovery_enabled: true,
        distributed_tracing_enabled: true,
        query_plan_caching: true,
        ..Default::default()
    };

    // Create enhanced federation manager
    let federation_manager = EnhancedFederationManager::new(
        federation_config,
        local_schema,
    ).await?;

    // Register federated services
    register_services(&federation_manager).await?;

    info!("Starting federation manager...");
    federation_manager.start().await?;

    // Create main GraphQL server with federation
    let server_config = GraphQLConfig {
        enable_introspection: true,
        enable_playground: true,
        max_query_depth: Some(15), // Allow deeper queries for federation
        max_query_complexity: Some(5000), // Higher complexity for federated queries
        enable_query_validation: true,
        ..Default::default()
    };

    let server = GraphQLServer::new(store.clone())
        .with_config(server_config);

    info!("Federation setup complete!");
    info!("GraphQL Playground available at http://127.0.0.1:4000/playground");
    info!("Federated GraphQL endpoint at http://127.0.0.1:4000/graphql");
    
    // Example federated query
    demo_federated_queries(&federation_manager).await?;

    // Start the main server
    server.start("127.0.0.1:4000").await?;

    Ok(())
}

/// Create a local schema for the main service
async fn create_local_schema(store: &Arc<RdfStore>) -> Result<String> {
    // Load some sample data
    let mut store_mut = RdfStore::new()?;
    
    // Add sample organization data
    store_mut.insert_triple(
        "http://example.org/org/1",
        "http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        "http://example.org/Organization",
    )?;
    
    store_mut.insert_triple(
        "http://example.org/org/1",
        "http://example.org/name",
        "\"Tech Corp\"",
    )?;

    info!("Local schema and data created");

    // Return GraphQL schema definition
    Ok(r#"
type Query {
  organizations: [Organization]
  organization(id: ID!): Organization
}

type Organization {
  id: ID!
  name: String!
  employees: [Employee] # Federated from employee service
  projects: [Project]   # Federated from project service
}

type Employee @key(fields: "id") {
  id: ID!
  organizationId: ID!
}

type Project @key(fields: "id") {
  id: ID!
  organizationId: ID!
}
"#.to_string())
}

/// Register federated services
async fn register_services(manager: &EnhancedFederationManager) -> Result<()> {
    info!("Registering federated services...");

    // Register employee service
    let employee_service = ServiceInfo {
        id: "employee-service".to_string(),
        name: "Employee Service".to_string(),
        url: "http://localhost:4001/graphql".to_string(),
        health_status: HealthStatus::Healthy,
        capabilities: vec!["employees".to_string(), "employee-queries".to_string()],
        version: "1.0.0".to_string(),
        metadata: std::collections::HashMap::new(),
        last_health_check: chrono::Utc::now(),
        response_time_ms: 50.0,
        load_factor: 0.3,
    };

    manager.register_service(employee_service).await?;

    // Register project service
    let project_service = ServiceInfo {
        id: "project-service".to_string(),
        name: "Project Service".to_string(),
        url: "http://localhost:4002/graphql".to_string(),
        health_status: HealthStatus::Healthy,
        capabilities: vec!["projects".to_string(), "project-queries".to_string()],
        version: "1.0.0".to_string(),
        metadata: std::collections::HashMap::new(),
        last_health_check: chrono::Utc::now(),
        response_time_ms: 75.0,
        load_factor: 0.5,
    };

    manager.register_service(project_service).await?;

    // Register analytics service
    let analytics_service = ServiceInfo {
        id: "analytics-service".to_string(),
        name: "Analytics Service".to_string(),
        url: "http://localhost:4003/graphql".to_string(),
        health_status: HealthStatus::Healthy,
        capabilities: vec!["analytics".to_string(), "reports".to_string()],
        version: "1.2.0".to_string(),
        metadata: std::collections::HashMap::new(),
        last_health_check: chrono::Utc::now(),
        response_time_ms: 120.0,
        load_factor: 0.7,
    };

    manager.register_service(analytics_service).await?;

    info!("All federated services registered successfully");
    Ok(())
}

/// Demonstrate federated queries
async fn demo_federated_queries(manager: &EnhancedFederationManager) -> Result<()> {
    info!("Running federation demo queries...");

    // Example 1: Simple federated query
    let query1 = r#"
    query GetOrganizationWithEmployees {
        organization(id: "1") {
            id
            name
            employees {
                id
                name
                position
            }
        }
    }
    "#;

    info!("Executing simple federated query...");
    let result1 = manager.execute_query(
        &juniper::parse_query(query1).unwrap(),
        std::collections::HashMap::new(),
        ()
    ).await;

    match result1 {
        Ok(_) => info!("✓ Simple federated query executed successfully"),
        Err(e) => info!("✗ Simple federated query failed: {}", e),
    }

    // Example 2: Complex multi-service query
    let query2 = r#"
    query GetOrganizationAnalytics {
        organization(id: "1") {
            id
            name
            employees {
                id
                name
                projects {
                    id
                    name
                    status
                }
            }
            analytics {
                totalEmployees
                totalProjects
                averageProjectDuration
                productivityScore
            }
        }
    }
    "#;

    info!("Executing complex multi-service query...");
    let result2 = manager.execute_query(
        &juniper::parse_query(query2).unwrap(),
        std::collections::HashMap::new(),
        ()
    ).await;

    match result2 {
        Ok(_) => info!("✓ Complex multi-service query executed successfully"),
        Err(e) => info!("✗ Complex multi-service query failed: {}", e),
    }

    // Example 3: Batched federation query
    let query3 = r#"
    query BatchedOrganizationData {
        organizations {
            id
            name
            employees(limit: 5) {
                id
                name
            }
            projects(status: ACTIVE) {
                id
                name
                deadline
            }
        }
    }
    "#;

    info!("Executing batched federation query...");
    let result3 = manager.execute_query(
        &juniper::parse_query(query3).unwrap(),
        std::collections::HashMap::new(),
        ()
    ).await;

    match result3 {
        Ok(_) => info!("✓ Batched federation query executed successfully"),
        Err(e) => info!("✗ Batched federation query failed: {}", e),
    }

    // Display federation stats
    if let Some(stats) = manager.get_federation_stats().await {
        info!("Federation Statistics:");
        info!("  Active Services: {}", stats.active_services);
        info!("  Total Queries: {}", stats.total_queries);
        info!("  Average Response Time: {:.2}ms", stats.average_response_time);
        info!("  Cache Hit Rate: {:.2}%", stats.cache_hit_rate * 100.0);
    }

    Ok(())
}