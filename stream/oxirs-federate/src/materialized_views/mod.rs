//! Materialized Views Module
//!
//! This module provides comprehensive materialized view management for federated queries,
//! including creation, maintenance, cost analysis, and optimization.

pub mod cost_analysis;
pub mod maintenance;
pub mod query_rewriting;
pub mod types;

// Re-export commonly used types and functions
pub use cost_analysis::{MaintenanceCost, ViewBenefit, ViewCostAnalyzer, ViewCreationCost};
pub use maintenance::{CleanupResult, MaintenanceConfig, MaintenanceScheduler, ViewStorageCleaner};
pub use query_rewriting::{
    QueryRewriter, RewritingConfig, RewritingResult, RewritingStrategy, ViewUsage,
};
pub use types::*;

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info};
use uuid::Uuid;

use crate::{
    executor::{SparqlResults, SparqlResultsData},
    planner::planning::types::QueryInfo,
    service_registry::ServiceRegistry,
};

/// Main materialized view manager that coordinates all view operations
#[derive(Debug)]
pub struct MaterializedViewManager {
    config: MaterializedViewConfig,
    views: HashMap<String, MaterializedView>,
    view_statistics: HashMap<String, ViewStatistics>,
    /// Materialized result data for each view, produced by an actual refresh.
    view_data: HashMap<String, SparqlResults>,
    maintenance_scheduler: MaintenanceScheduler,
    cost_analyzer: ViewCostAnalyzer,
}

impl MaterializedViewManager {
    /// Create a new materialized view manager
    pub fn new() -> Self {
        Self {
            config: MaterializedViewConfig::default(),
            views: HashMap::new(),
            view_statistics: HashMap::new(),
            view_data: HashMap::new(),
            maintenance_scheduler: MaintenanceScheduler::new(),
            cost_analyzer: ViewCostAnalyzer::new(),
        }
    }

    /// Create a new materialized view manager with custom configuration
    pub fn with_config(config: MaterializedViewConfig) -> Self {
        Self {
            config,
            views: HashMap::new(),
            view_statistics: HashMap::new(),
            view_data: HashMap::new(),
            maintenance_scheduler: MaintenanceScheduler::new(),
            cost_analyzer: ViewCostAnalyzer::new(),
        }
    }

    /// Access the materialized result data for a view, if it has been refreshed.
    pub fn get_view_data(&self, view_id: &str) -> Option<&SparqlResults> {
        self.view_data.get(view_id)
    }

    /// Create a new materialized view
    pub async fn create_view(
        &mut self,
        definition: ViewDefinition,
        registry: &ServiceRegistry,
    ) -> Result<String> {
        debug!("Creating materialized view: {}", definition.name);

        // Validate the definition
        self.validate_view_definition(&definition)?;

        // Check if we've reached the maximum number of views
        if self.views.len() >= self.config.max_views {
            return Err(anyhow!(
                "Maximum number of views ({}) reached",
                self.config.max_views
            ));
        }

        // Estimate creation cost
        let creation_cost = self
            .cost_analyzer
            .estimate_creation_cost(&definition, registry)
            .await?;

        info!(
            "Estimated creation cost for view '{}': ${:.2}",
            definition.name, creation_cost.total_cost
        );

        // Create the view instance
        let view_id = format!("view_{}", Uuid::new_v4());
        let view = MaterializedView {
            id: view_id.clone(),
            definition,
            creation_time: chrono::Utc::now(),
            last_refresh: None,
            size_bytes: 0,
            row_count: 0,
            is_stale: true, // Newly created views start as stale
            refresh_in_progress: false,
            error_count: 0,
            last_error: None,
            access_count: 0,
            last_access: None,
            data_location: ViewDataLocation::Memory, // Default to memory storage
        };

        // Initialize statistics
        let statistics = ViewStatistics::new(view_id.clone());

        // Store the view and statistics
        self.views.insert(view_id.clone(), view.clone());
        self.view_statistics.insert(view_id.clone(), statistics);

        // Schedule initial refresh
        self.maintenance_scheduler.schedule_refresh(&view, true);

        info!("Created materialized view: {}", view_id);
        Ok(view_id)
    }

    /// Get a materialized view by ID
    pub fn get_view(&self, view_id: &str) -> Option<&MaterializedView> {
        self.views.get(view_id)
    }

    /// Get all materialized views
    pub fn get_all_views(&self) -> impl Iterator<Item = &MaterializedView> {
        self.views.values()
    }

    /// Get view statistics
    pub fn get_view_statistics(&self, view_id: &str) -> Option<&ViewStatistics> {
        self.view_statistics.get(view_id)
    }

    /// Update view statistics after a query
    pub fn record_view_access(&mut self, view_id: &str, was_hit: bool) {
        if let Some(view) = self.views.get_mut(view_id) {
            view.access_count += 1;
            view.last_access = Some(chrono::Utc::now());
        }

        if let Some(stats) = self.view_statistics.get_mut(view_id) {
            if was_hit {
                stats.record_hit();
            } else {
                stats.record_miss();
            }
        }
    }

    /// Find suitable views for a query
    pub async fn find_suitable_views(
        &self,
        query_info: &QueryInfo,
        registry: &ServiceRegistry,
    ) -> Result<Vec<ViewRecommendation>> {
        debug!(
            "Finding suitable views for query with {} patterns",
            query_info.patterns.len()
        );

        let mut recommendations = Vec::new();

        for view in self.views.values() {
            // Check if the view can support the query patterns
            if view.definition.supports_patterns(&query_info.patterns) {
                // Estimate the benefit of using this view
                let benefit = self
                    .cost_analyzer
                    .estimate_query_benefit(view, query_info, registry)
                    .await?;

                if benefit.cost_saving > 0.0 {
                    let recommendation = ViewRecommendation {
                        view_id: view.id.clone(),
                        reason: RecommendationReason::ImprovedCacheHitRatio,
                        estimated_benefit: benefit.cost_saving,
                        implementation_cost: 0.0, // Already implemented
                        confidence: benefit.cache_hit_probability,
                    };

                    recommendations.push(recommendation);
                }
            }
        }

        // Sort by estimated benefit
        recommendations.sort_by(|a, b| {
            b.estimated_benefit
                .partial_cmp(&a.estimated_benefit)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        debug!("Found {} suitable views", recommendations.len());
        Ok(recommendations)
    }

    /// Remove a materialized view
    pub async fn remove_view(&mut self, view_id: &str) -> Result<()> {
        if let Some(_view) = self.views.remove(view_id) {
            self.view_statistics.remove(view_id);
            self.view_data.remove(view_id);
            self.maintenance_scheduler
                .cancel_operations_for_view(view_id);

            info!("Removed materialized view: {}", view_id);
            Ok(())
        } else {
            Err(anyhow!("View not found: {}", view_id))
        }
    }

    /// Perform maintenance operations.
    ///
    /// The `registry` is required so that `Refresh` operations can actually
    /// execute each view's query against the underlying services and materialize
    /// real result data.
    pub async fn perform_maintenance(
        &mut self,
        registry: &ServiceRegistry,
    ) -> Result<MaintenanceReport> {
        let start_time = Instant::now();
        let mut operations_performed = 0;
        let mut errors = Vec::new();

        // Re-mark views stale when their refresh interval (TTL) has elapsed, so
        // that a view materialized long ago is refreshed again rather than being
        // trusted forever.
        self.mark_stale_by_ttl();

        // Process scheduled maintenance operations
        while let Some(operation) = self.maintenance_scheduler.get_next_operation() {
            let operation_id = self
                .maintenance_scheduler
                .start_operation(operation.clone());

            let result = match operation.operation {
                MaintenanceOperation::Refresh => {
                    self.refresh_view(&operation.view_id, registry).await
                }
                MaintenanceOperation::Cleanup => self.cleanup_view(&operation.view_id).await,
                MaintenanceOperation::Optimize => self.optimize_view(&operation.view_id).await,
                MaintenanceOperation::Validate => self.validate_view(&operation.view_id).await,
                MaintenanceOperation::Archive => self.archive_view(&operation.view_id).await,
            };

            match result {
                Ok(duration) => {
                    operations_performed += 1;
                    self.maintenance_scheduler.complete_operation(
                        &operation_id,
                        maintenance::OperationResult::Success {
                            duration,
                            details: format!(
                                "Successfully completed {:?} operation",
                                operation.operation
                            ),
                        },
                    );
                }
                Err(e) => {
                    errors.push(format!(
                        "Operation {:?} failed for view {}: {}",
                        operation.operation, operation.view_id, e
                    ));
                    self.maintenance_scheduler.complete_operation(
                        &operation_id,
                        maintenance::OperationResult::Failed {
                            error: e.to_string(),
                            retry_count: 0,
                        },
                    );
                }
            }
        }

        Ok(MaintenanceReport {
            operations_performed,
            errors,
            duration: start_time.elapsed(),
            scheduler_stats: self.maintenance_scheduler.get_statistics(),
        })
    }

    // Private helper methods

    fn validate_view_definition(&self, definition: &ViewDefinition) -> Result<()> {
        if definition.name.is_empty() {
            return Err(anyhow!("View name cannot be empty"));
        }

        if definition.query.is_empty() {
            return Err(anyhow!("View query cannot be empty"));
        }

        if definition.source_patterns.is_empty() {
            return Err(anyhow!("View must have at least one source pattern"));
        }

        // Check for duplicate view names
        if self
            .views
            .values()
            .any(|v| v.definition.name == definition.name)
        {
            return Err(anyhow!(
                "View with name '{}' already exists",
                definition.name
            ));
        }

        Ok(())
    }

    /// Re-mark views as stale when their refresh interval (TTL) has elapsed
    /// since the last successful refresh.
    fn mark_stale_by_ttl(&mut self) {
        let now = chrono::Utc::now();
        for view in self.views.values_mut() {
            if view.refresh_in_progress {
                continue;
            }
            let ttl = view.definition.estimate_freshness_requirement();
            if let Some(last) = view.last_refresh {
                let age = now.signed_duration_since(last);
                let ttl_ms = ttl.as_millis() as i64;
                if age.num_milliseconds() >= ttl_ms {
                    view.is_stale = true;
                }
            }
        }
    }

    /// Refresh a view by actually executing its query against the underlying
    /// services and storing the materialized result. On failure the view is left
    /// stale and the error is propagated (fail-loud) — a view is never marked
    /// fresh without real data behind it.
    async fn refresh_view(
        &mut self,
        view_id: &str,
        registry: &ServiceRegistry,
    ) -> Result<Duration> {
        let start_time = Instant::now();

        // Snapshot the data needed for execution without holding a mutable borrow.
        let (query, source_service_ids) = {
            let view = self
                .views
                .get(view_id)
                .ok_or_else(|| anyhow!("View not found: {}", view_id))?;
            let ids: Vec<String> = view
                .definition
                .source_patterns
                .iter()
                .map(|sp| sp.service_id.clone())
                .collect();
            (view.definition.query.clone(), ids)
        };

        if let Some(view) = self.views.get_mut(view_id) {
            view.refresh_in_progress = true;
        }

        let outcome = Self::execute_view_query(&query, &source_service_ids, registry).await;

        match outcome {
            Ok(results) => {
                let row_count = results.results.bindings.len() as u64;
                let size_bytes = serde_json::to_vec(&results)
                    .map(|b| b.len() as u64)
                    .unwrap_or(0);
                self.view_data.insert(view_id.to_string(), results);

                if let Some(view) = self.views.get_mut(view_id) {
                    view.refresh_in_progress = false;
                    view.last_refresh = Some(chrono::Utc::now());
                    view.is_stale = false;
                    view.row_count = row_count;
                    view.size_bytes = size_bytes;
                    view.last_error = None;
                }

                if let Some(stats) = self.view_statistics.get_mut(view_id) {
                    stats.record_refresh(start_time.elapsed());
                }

                Ok(start_time.elapsed())
            }
            Err(e) => {
                if let Some(view) = self.views.get_mut(view_id) {
                    view.refresh_in_progress = false;
                    view.error_count += 1;
                    view.last_error = Some(e.to_string());
                    // Deliberately leave `is_stale` untouched (stays stale): a
                    // failed refresh must not make the view appear fresh.
                }
                Err(e)
            }
        }
    }

    /// Execute a view's query against its source services and materialize the
    /// combined bindings. Fails loudly when no source SPARQL endpoint can be
    /// resolved or every source query fails.
    async fn execute_view_query(
        query: &str,
        source_service_ids: &[String],
        registry: &ServiceRegistry,
    ) -> Result<SparqlResults> {
        // Resolve source SPARQL endpoints from the registry.
        let mut endpoints: Vec<String> = Vec::new();
        for service_id in source_service_ids {
            if let Some(service) = registry.get_service(service_id) {
                if matches!(service.service_type, crate::ServiceType::Sparql) {
                    endpoints.push(service.endpoint.clone());
                }
            }
        }
        // De-duplicate while preserving order.
        endpoints.dedup();

        if endpoints.is_empty() {
            return Err(anyhow!(
                "Cannot refresh view: no resolvable SPARQL source endpoint among {:?}",
                source_service_ids
            ));
        }

        let client = reqwest::Client::new();
        let mut vars: Vec<String> = Vec::new();
        let mut all_bindings = Vec::new();
        let mut last_error: Option<anyhow::Error> = None;
        let mut any_success = false;

        for endpoint in &endpoints {
            match Self::query_endpoint(&client, endpoint, query).await {
                Ok(results) => {
                    any_success = true;
                    if vars.is_empty() {
                        vars = results.head.vars.clone();
                    }
                    all_bindings.extend(results.results.bindings);
                }
                Err(e) => {
                    last_error = Some(e);
                }
            }
        }

        if !any_success {
            return Err(last_error
                .unwrap_or_else(|| anyhow!("View refresh failed against all source endpoints")));
        }

        Ok(SparqlResults {
            head: crate::executor::SparqlHead { vars },
            results: SparqlResultsData {
                bindings: all_bindings,
            },
        })
    }

    /// Issue a single SPARQL query against an endpoint and parse the result set.
    async fn query_endpoint(
        client: &reqwest::Client,
        endpoint: &str,
        query: &str,
    ) -> Result<SparqlResults> {
        let response = client
            .post(endpoint)
            .header("Content-Type", "application/sparql-query")
            .header("Accept", "application/sparql-results+json")
            .body(query.to_string())
            .send()
            .await
            .map_err(|e| anyhow!("Request to '{}' failed: {}", endpoint, e))?;

        if !response.status().is_success() {
            return Err(anyhow!(
                "Endpoint '{}' returned status {}",
                endpoint,
                response.status()
            ));
        }

        let text = response
            .text()
            .await
            .map_err(|e| anyhow!("Failed to read response from '{}': {}", endpoint, e))?;
        let results: SparqlResults = serde_json::from_str(&text)
            .map_err(|e| anyhow!("Failed to parse SPARQL results from '{}': {}", endpoint, e))?;
        Ok(results)
    }

    /// Cleanup drops materialized data for stale views to free storage; the view
    /// bookkeeping remains so it can be refreshed again later.
    async fn cleanup_view(&mut self, view_id: &str) -> Result<Duration> {
        let start_time = Instant::now();
        debug!("Cleaning up view: {}", view_id);

        let is_stale = self
            .views
            .get(view_id)
            .map(|v| v.is_stale)
            .ok_or_else(|| anyhow!("View not found: {}", view_id))?;

        if is_stale && self.view_data.remove(view_id).is_some() {
            if let Some(view) = self.views.get_mut(view_id) {
                view.size_bytes = 0;
                view.row_count = 0;
            }
        }

        Ok(start_time.elapsed())
    }

    /// Optimize recomputes the stored size from the actual materialized data.
    async fn optimize_view(&mut self, view_id: &str) -> Result<Duration> {
        let start_time = Instant::now();
        debug!("Optimizing view: {}", view_id);

        let size = self
            .view_data
            .get(view_id)
            .and_then(|d| serde_json::to_vec(d).ok())
            .map(|b| b.len() as u64);
        if let (Some(size), Some(view)) = (size, self.views.get_mut(view_id)) {
            view.size_bytes = size;
        }

        Ok(start_time.elapsed())
    }

    /// Validate checks that a view claimed fresh actually has materialized data
    /// with a consistent row count. Fails loudly on inconsistency.
    async fn validate_view(&mut self, view_id: &str) -> Result<Duration> {
        let start_time = Instant::now();
        debug!("Validating view: {}", view_id);

        let view = self
            .views
            .get(view_id)
            .ok_or_else(|| anyhow!("View not found: {}", view_id))?;

        if !view.is_stale {
            match self.view_data.get(view_id) {
                None => {
                    if let Some(view) = self.views.get_mut(view_id) {
                        view.is_stale = true;
                    }
                    return Err(anyhow!(
                        "View '{}' is marked fresh but has no materialized data",
                        view_id
                    ));
                }
                Some(data) => {
                    let actual = data.results.bindings.len() as u64;
                    if actual != view.row_count {
                        return Err(anyhow!(
                            "View '{}' row_count ({}) disagrees with materialized data ({})",
                            view_id,
                            view.row_count,
                            actual
                        ));
                    }
                }
            }
        }

        Ok(start_time.elapsed())
    }

    /// Archive persists the materialized data to disk and records its location.
    async fn archive_view(&mut self, view_id: &str) -> Result<Duration> {
        let start_time = Instant::now();
        debug!("Archiving view: {}", view_id);

        let data = self
            .view_data
            .get(view_id)
            .ok_or_else(|| anyhow!("View '{}' has no materialized data to archive", view_id))?;

        let bytes = serde_json::to_vec(data)
            .map_err(|e| anyhow!("Failed to serialize view '{}': {}", view_id, e))?;
        let dir = std::env::temp_dir().join("oxirs_federate_views");
        std::fs::create_dir_all(&dir)
            .map_err(|e| anyhow!("Failed to create archive dir: {}", e))?;
        let path = dir.join(format!("{view_id}.json"));
        std::fs::write(&path, &bytes)
            .map_err(|e| anyhow!("Failed to write archive for view '{}': {}", view_id, e))?;

        if let Some(view) = self.views.get_mut(view_id) {
            view.data_location = ViewDataLocation::Disk {
                path: path.to_string_lossy().to_string(),
            };
        }

        Ok(start_time.elapsed())
    }
}

impl Default for MaterializedViewManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Report of maintenance operations performed
#[derive(Debug, Clone)]
pub struct MaintenanceReport {
    pub operations_performed: usize,
    pub errors: Vec<String>,
    pub duration: Duration,
    pub scheduler_stats: maintenance::MaintenanceStatistics,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::service_registry::ServiceRegistry;

    fn make_view(id: &str, service_id: &str) -> MaterializedView {
        MaterializedView {
            id: id.to_string(),
            definition: ViewDefinition {
                name: format!("view_{id}"),
                description: None,
                source_patterns: vec![ServicePattern {
                    service_id: service_id.to_string(),
                    patterns: vec![],
                    filters: vec![],
                    estimated_selectivity: 1.0,
                }],
                query: "SELECT * WHERE { ?s ?p ?o }".to_string(),
                refresh_interval: Some(Duration::from_secs(60)),
                supports_incremental: false,
                partitioning_key: None,
                dependencies: vec![],
            },
            creation_time: chrono::Utc::now(),
            last_refresh: None,
            size_bytes: 0,
            row_count: 0,
            is_stale: true,
            refresh_in_progress: false,
            error_count: 0,
            last_error: None,
            access_count: 0,
            last_access: None,
            data_location: ViewDataLocation::Memory,
        }
    }

    /// Regression: refresh must actually try to materialize data and fail loudly
    /// (leaving the view stale) when no source endpoint can be resolved — it must
    /// not just flip `is_stale` to false.
    #[tokio::test]
    async fn regression_refresh_fails_loud_without_endpoint() {
        let mut mgr = MaterializedViewManager::new();
        let registry = ServiceRegistry::new(); // empty registry: service missing
        let view = make_view("v1", "missing_service");
        mgr.views.insert("v1".to_string(), view);
        mgr.view_statistics
            .insert("v1".to_string(), ViewStatistics::new("v1".to_string()));

        let result = mgr.refresh_view("v1", &registry).await;
        assert!(result.is_err(), "refresh with no endpoint must fail loudly");
        assert!(
            mgr.views["v1"].is_stale,
            "a failed refresh must leave the view stale"
        );
        assert!(mgr.get_view_data("v1").is_none());
        assert_eq!(mgr.views["v1"].error_count, 1);
    }

    /// Regression: a view claiming freshness but with no materialized data behind
    /// it must be caught by validation and re-marked stale.
    #[tokio::test]
    async fn regression_validate_detects_fresh_without_data() {
        let mut mgr = MaterializedViewManager::new();
        let mut view = make_view("v2", "svc");
        view.is_stale = false; // claims fresh but no data stored
        mgr.views.insert("v2".to_string(), view);

        let result = mgr.validate_view("v2").await;
        assert!(
            result.is_err(),
            "fresh view without materialized data must fail validation"
        );
        assert!(
            mgr.views["v2"].is_stale,
            "validation must re-mark the view stale"
        );
    }
}
