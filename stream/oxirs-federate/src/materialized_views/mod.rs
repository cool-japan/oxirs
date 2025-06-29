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
pub use query_rewriting::{QueryRewriter, RewritingConfig, RewritingResult, RewritingStrategy, ViewUsage};
pub use types::*;

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::{planner::planning::types::QueryInfo, ServiceRegistry};

/// Main materialized view manager that coordinates all view operations
#[derive(Debug)]
pub struct MaterializedViewManager {
    config: MaterializedViewConfig,
    views: HashMap<String, MaterializedView>,
    view_statistics: HashMap<String, ViewStatistics>,
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
            maintenance_scheduler: MaintenanceScheduler::new(),
            cost_analyzer: ViewCostAnalyzer::new(),
        }
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
        if let Some(view) = self.views.remove(view_id) {
            self.view_statistics.remove(view_id);
            self.maintenance_scheduler
                .cancel_operations_for_view(view_id);

            info!("Removed materialized view: {}", view_id);
            Ok(())
        } else {
            Err(anyhow!("View not found: {}", view_id))
        }
    }

    /// Perform maintenance operations
    pub async fn perform_maintenance(&mut self) -> Result<MaintenanceReport> {
        let start_time = Instant::now();
        let mut operations_performed = 0;
        let mut errors = Vec::new();

        // Process scheduled maintenance operations
        while let Some(operation) = self.maintenance_scheduler.get_next_operation() {
            let operation_id = self
                .maintenance_scheduler
                .start_operation(operation.clone());

            let result = match operation.operation {
                MaintenanceOperation::Refresh => self.refresh_view(&operation.view_id).await,
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

    async fn refresh_view(&mut self, view_id: &str) -> Result<Duration> {
        let start_time = Instant::now();

        if let Some(view) = self.views.get_mut(view_id) {
            view.refresh_in_progress = true;
            view.last_refresh = Some(chrono::Utc::now());
            view.is_stale = false;
            view.refresh_in_progress = false;

            // Record refresh in statistics
            if let Some(stats) = self.view_statistics.get_mut(view_id) {
                stats.record_refresh(start_time.elapsed());
            }

            Ok(start_time.elapsed())
        } else {
            Err(anyhow!("View not found: {}", view_id))
        }
    }

    async fn cleanup_view(&mut self, view_id: &str) -> Result<Duration> {
        let start_time = Instant::now();

        // Perform cleanup operations
        debug!("Cleaning up view: {}", view_id);

        Ok(start_time.elapsed())
    }

    async fn optimize_view(&mut self, view_id: &str) -> Result<Duration> {
        let start_time = Instant::now();

        // Perform optimization operations
        debug!("Optimizing view: {}", view_id);

        Ok(start_time.elapsed())
    }

    async fn validate_view(&mut self, view_id: &str) -> Result<Duration> {
        let start_time = Instant::now();

        // Perform validation operations
        debug!("Validating view: {}", view_id);

        Ok(start_time.elapsed())
    }

    async fn archive_view(&mut self, view_id: &str) -> Result<Duration> {
        let start_time = Instant::now();

        // Perform archival operations
        debug!("Archiving view: {}", view_id);

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
