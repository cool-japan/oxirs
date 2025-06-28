//! Federation manager for coordinating GraphQL and SPARQL federation

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use super::config::{FederationConfig, RemoteEndpoint};
use super::dataset_federation::{DatasetFederation, SparqlEndpoint};
use super::query_planner::{QueryPlan, QueryPlanner};
use super::schema_stitcher::SchemaStitcher;
use crate::types::Schema;

/// Main federation manager coordinating all federation activities
pub struct FederationManager {
    config: FederationConfig,
    schema_stitcher: Arc<SchemaStitcher>,
    query_planner: Arc<QueryPlanner>,
    dataset_federation: Arc<RwLock<DatasetFederation>>,
    merged_schema: Arc<RwLock<Option<Schema>>>,
}

impl FederationManager {
    pub fn new(local_schema: Arc<Schema>, config: FederationConfig) -> Self {
        let schema_stitcher = Arc::new(SchemaStitcher::new(local_schema));
        let query_planner = Arc::new(QueryPlanner::new(schema_stitcher.clone(), config.clone()));
        let dataset_federation = Arc::new(RwLock::new(DatasetFederation::new()));

        Self {
            config,
            schema_stitcher,
            query_planner,
            dataset_federation,
            merged_schema: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize federation by introspecting all remote endpoints
    pub async fn initialize(&self) -> Result<()> {
        tracing::info!(
            "Initializing federation manager with {} endpoints",
            self.config.endpoints.len()
        );

        // Introspect and merge all remote schemas
        let merged_schema = self
            .schema_stitcher
            .merge_schemas(&self.config.endpoints)
            .await
            .context("Failed to merge remote schemas")?;

        {
            let mut schema_guard = self.merged_schema.write().await;
            *schema_guard = Some(merged_schema);
        }

        // Initialize SPARQL endpoints for dataset federation
        {
            let mut dataset_federation = self.dataset_federation.write().await;
            for endpoint in &self.config.endpoints {
                let sparql_endpoint = SparqlEndpoint {
                    id: endpoint.id.clone(),
                    url: endpoint.url.clone(),
                    auth_header: endpoint.auth_header.clone(),
                    timeout_secs: endpoint.timeout_secs,
                    max_concurrent_queries: 10, // Default
                    supported_features: std::collections::HashSet::new(), // Will be populated
                    statistics: None,
                };
                dataset_federation.add_endpoint(sparql_endpoint);
            }
        }

        // Update endpoint statistics
        for endpoint in &self.config.endpoints {
            if let Err(e) = self.update_endpoint_statistics(&endpoint.id).await {
                tracing::warn!(
                    "Failed to update statistics for endpoint {}: {}",
                    endpoint.id,
                    e
                );
            }
        }

        tracing::info!("Federation manager initialized successfully");
        Ok(())
    }

    /// Get the merged schema
    pub async fn get_merged_schema(&self) -> Result<Schema> {
        let schema_guard = self.merged_schema.read().await;
        schema_guard
            .as_ref()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Federation not initialized"))
    }

    /// Execute a federated GraphQL query
    pub async fn execute_graphql_query(
        &self,
        query: &crate::ast::Document,
    ) -> Result<serde_json::Value> {
        let merged_schema = self.get_merged_schema().await?;

        // Plan the query execution
        let query_plan = self
            .query_planner
            .plan_query(query, &merged_schema)
            .await
            .context("Failed to plan federated query")?;

        tracing::debug!(
            "Executing federated query with {} steps",
            query_plan.steps.len()
        );

        // Execute the plan
        self.query_planner
            .execute_plan(&query_plan)
            .await
            .context("Failed to execute federated query plan")
    }

    /// Execute a federated SPARQL query
    pub async fn execute_sparql_query(&self, query: &str) -> Result<serde_json::Value> {
        let dataset_federation = self.dataset_federation.read().await;
        dataset_federation
            .federate_sparql_query(query)
            .await
            .context("Failed to execute federated SPARQL query")
    }

    /// Add a new remote endpoint dynamically
    pub async fn add_endpoint(&mut self, endpoint: RemoteEndpoint) -> Result<()> {
        tracing::info!("Adding new federation endpoint: {}", endpoint.id);

        // Introspect the new endpoint
        let remote_schema = self
            .schema_stitcher
            .introspect_remote(&endpoint)
            .await
            .context("Failed to introspect new endpoint")?;

        // Update merged schema
        let mut merged_schema = self.get_merged_schema().await?;
        self.schema_stitcher
            .merge_schema_into(&mut merged_schema, &remote_schema, &endpoint)
            .context("Failed to merge new endpoint schema")?;

        {
            let mut schema_guard = self.merged_schema.write().await;
            *schema_guard = Some(merged_schema);
        }

        // Add to dataset federation
        {
            let mut dataset_federation = self.dataset_federation.write().await;
            let sparql_endpoint = SparqlEndpoint {
                id: endpoint.id.clone(),
                url: endpoint.url.clone(),
                auth_header: endpoint.auth_header.clone(),
                timeout_secs: endpoint.timeout_secs,
                max_concurrent_queries: 10,
                supported_features: std::collections::HashSet::new(),
                statistics: None,
            };
            dataset_federation.add_endpoint(sparql_endpoint);
        }

        // Update configuration
        self.config.endpoints.push(endpoint);

        tracing::info!("Successfully added new federation endpoint");
        Ok(())
    }

    /// Remove an endpoint
    pub async fn remove_endpoint(&mut self, endpoint_id: &str) -> Result<()> {
        tracing::info!("Removing federation endpoint: {}", endpoint_id);

        // Remove from configuration
        self.config.endpoints.retain(|ep| ep.id != endpoint_id);

        // Rebuild merged schema without this endpoint
        let merged_schema = self
            .schema_stitcher
            .merge_schemas(&self.config.endpoints)
            .await
            .context("Failed to rebuild merged schema")?;

        {
            let mut schema_guard = self.merged_schema.write().await;
            *schema_guard = Some(merged_schema);
        }

        tracing::info!("Successfully removed federation endpoint: {}", endpoint_id);
        Ok(())
    }

    /// Update endpoint statistics
    pub async fn update_endpoint_statistics(&self, endpoint_id: &str) -> Result<()> {
        let mut dataset_federation = self.dataset_federation.write().await;
        dataset_federation
            .update_endpoint_statistics(endpoint_id)
            .await
            .context("Failed to update endpoint statistics")
    }

    /// Get federation health status
    pub async fn get_health_status(&self) -> Result<FederationHealthStatus> {
        let mut status = FederationHealthStatus {
            total_endpoints: self.config.endpoints.len(),
            healthy_endpoints: 0,
            endpoint_statuses: HashMap::new(),
            schema_cache_size: 0,
        };

        // Check each endpoint health
        for endpoint in &self.config.endpoints {
            let endpoint_healthy = self.check_endpoint_health(endpoint).await;
            if endpoint_healthy {
                status.healthy_endpoints += 1;
            }

            status.endpoint_statuses.insert(
                endpoint.id.clone(),
                EndpointHealthStatus {
                    healthy: endpoint_healthy,
                    last_check: chrono::Utc::now(),
                    response_time_ms: None, // Would be measured during health check
                },
            );
        }

        Ok(status)
    }

    /// Check if a specific endpoint is healthy
    async fn check_endpoint_health(&self, endpoint: &RemoteEndpoint) -> bool {
        if let Some(health_url) = &endpoint.health_check_url {
            let client = reqwest::Client::new();
            match client
                .get(health_url)
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await
            {
                Ok(response) => response.status().is_success(),
                Err(_) => false,
            }
        } else {
            // If no health check URL, try a simple introspection query
            self.schema_stitcher
                .introspect_remote(endpoint)
                .await
                .is_ok()
        }
    }

    /// Get federation configuration
    pub fn get_config(&self) -> &FederationConfig {
        &self.config
    }

    /// Update federation configuration
    pub async fn update_config(&mut self, new_config: FederationConfig) -> Result<()> {
        tracing::info!("Updating federation configuration");

        self.config = new_config;

        // Reinitialize with new configuration
        self.initialize()
            .await
            .context("Failed to reinitialize with new configuration")
    }

    /// Clear all caches
    pub async fn clear_caches(&self) -> Result<()> {
        tracing::info!("Clearing federation caches");

        // Clear schema cache
        {
            let mut schema_guard = self.merged_schema.write().await;
            *schema_guard = None;
        }

        // Reinitialize
        self.initialize()
            .await
            .context("Failed to reinitialize after clearing caches")
    }
}

/// Federation health status
#[derive(Debug, Clone)]
pub struct FederationHealthStatus {
    pub total_endpoints: usize,
    pub healthy_endpoints: usize,
    pub endpoint_statuses: HashMap<String, EndpointHealthStatus>,
    pub schema_cache_size: usize,
}

/// Individual endpoint health status
#[derive(Debug, Clone)]
pub struct EndpointHealthStatus {
    pub healthy: bool,
    pub last_check: chrono::DateTime<chrono::Utc>,
    pub response_time_ms: Option<u64>,
}
