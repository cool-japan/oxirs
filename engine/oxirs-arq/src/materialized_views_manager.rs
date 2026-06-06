//! High-level orchestration for materialized views.
//!
//! [`MaterializedViewManager`] ties the storage, rewriter, scheduler, and
//! recommendation engine together.  It exposes the user-facing operations
//! (`create_view`, `rewrite_query`, `update_view`, …) and contains the
//! dependency-analysis, cost-estimation, and incremental/full update logic
//! delegated to from those operations.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

use anyhow::{anyhow, Result};
use tracing::{debug, info, span, Level};

use crate::algebra::{Algebra, Expression, Solution, Term, TriplePattern, Variable};
use crate::cost_model::{CostEstimate, CostModel};
use crate::executor::{Dataset, ExecutionStats, QueryExecutor};
use crate::materialized_views_types::{
    DeltaState, IncrementalState, JoinDependency, MaintenanceInfo, MaintenanceScheduler,
    MaintenanceStrategy, MaintenanceTaskType, MaterializedView, MaterializedViewConfig,
    MaterializedViewManager, QueryRewriter, SchedulerConfig, UsageRecord, ViewCostEstimates,
    ViewData, ViewDependencies, ViewMetadata, ViewRecommendation, ViewRecommendationEngine,
    ViewStorage, ViewUsageStatistics, ViewUsageStats,
};
use crate::statistics_collector::StatisticsCollector;

impl MaterializedViewManager {
    /// Create a new materialized view manager
    pub fn new(
        config: MaterializedViewConfig,
        cost_model: Arc<Mutex<CostModel>>,
        statistics_collector: Arc<StatisticsCollector>,
    ) -> Result<Self> {
        let views = Arc::new(RwLock::new(std::collections::HashMap::new()));
        let view_storage = Arc::new(RwLock::new(ViewStorage::new(config.max_memory_usage)));

        let rewriter = QueryRewriter::new()?;
        let maintenance_scheduler = MaintenanceScheduler::new(SchedulerConfig::default())?;
        let usage_statistics = Arc::new(RwLock::new(ViewUsageStatistics::default()));
        let recommendation_engine = ViewRecommendationEngine::new()?;

        Ok(Self {
            config,
            views,
            view_storage,
            rewriter,
            maintenance_scheduler,
            cost_model,
            statistics_collector,
            usage_statistics,
            recommendation_engine,
        })
    }

    /// Create a new materialized view
    pub fn create_view(
        &mut self,
        name: String,
        definition: Algebra,
        metadata: ViewMetadata,
        executor: &mut QueryExecutor,
        dataset: &dyn Dataset,
    ) -> Result<String> {
        let _span = span!(Level::INFO, "create_materialized_view").entered();

        let view_id = format!("view_{}", uuid::Uuid::new_v4().simple());

        info!("Creating materialized view: {} ({})", name, view_id);

        // Execute the view definition to materialize initial data
        let start_time = Instant::now();
        let (results, stats) = executor.execute(&definition, dataset)?;
        let materialization_time = start_time.elapsed();

        // Calculate data size and checksum
        let size_bytes = self.estimate_result_size(&results);
        let checksum = self.calculate_checksum(&results);

        let view_data = ViewData {
            results,
            size_bytes,
            row_count: stats.final_results,
            materialized_at: SystemTime::now(),
            checksum,
        };

        // Analyze dependencies
        let dependencies = self.analyze_dependencies(&definition)?;

        // Calculate cost estimates
        let cost_estimates = self.calculate_view_costs(&definition, &view_data, &stats)?;

        // Set up maintenance info
        let maintenance_info = MaintenanceInfo {
            last_updated: SystemTime::now(),
            next_maintenance: self.calculate_next_maintenance(&self.config.maintenance_strategy),
            strategy: self.config.maintenance_strategy.clone(),
            update_count: 0,
            total_maintenance_time: materialization_time,
            needs_update: false,
            incremental_state: if self.config.incremental_maintenance {
                Some(IncrementalState {
                    last_transaction_id: 0,
                    change_log: Vec::new(),
                    delta_state: DeltaState {
                        positive_delta: Vec::new(),
                        negative_delta: Vec::new(),
                        dirty_partitions: HashSet::new(),
                    },
                })
            } else {
                None
            },
        };

        let view = MaterializedView {
            id: view_id.clone(),
            name,
            definition: definition.clone(),
            data: view_data.clone(),
            metadata,
            maintenance_info,
            cost_estimates,
            dependencies,
        };

        // Store the view
        {
            let mut views = self.views.write().expect("lock poisoned");
            views.insert(view_id.clone(), view);
        }

        // Store the data
        {
            let mut storage = self.view_storage.write().expect("lock poisoned");
            storage.store_view_data(view_id.clone(), view_data)?;
        }

        // Update view index
        self.rewriter.update_view_index(&view_id, &definition)?;

        // Schedule maintenance if needed
        if let Some(next_maintenance) =
            self.calculate_next_maintenance(&self.config.maintenance_strategy)
        {
            self.maintenance_scheduler.schedule_maintenance(
                view_id.clone(),
                MaintenanceTaskType::StatisticsUpdate,
                next_maintenance,
                3, // Medium priority
            )?;
        }

        info!(
            "Created materialized view {} in {:?}",
            view_id, materialization_time
        );
        Ok(view_id)
    }

    /// Rewrite a query to use materialized views
    pub fn rewrite_query(&self, query: &Algebra) -> Result<(Algebra, Vec<String>)> {
        let _span = span!(Level::DEBUG, "rewrite_query").entered();

        self.rewriter
            .rewrite_query(query, &self.views, &self.cost_model)
    }

    /// Get view usage statistics
    pub fn get_usage_statistics(&self, view_id: &str) -> Result<Option<ViewUsageStats>> {
        let stats = self.usage_statistics.read().expect("lock poisoned");

        Ok(stats
            .access_counts
            .get(view_id)
            .map(|&access_count| ViewUsageStats {
                access_count,
                total_time_saved: stats.time_saved.get(view_id).copied().unwrap_or_default(),
                hit_rate: stats.hit_rates.get(view_id).copied().unwrap_or(0.0),
                cost_benefit: stats.cost_benefits.get(view_id).copied().unwrap_or(0.0),
            }))
    }

    /// Get view recommendations based on query patterns
    pub fn get_view_recommendations(&self) -> Result<Vec<ViewRecommendation>> {
        self.recommendation_engine.get_recommendations()
    }

    /// Update view with new data
    pub fn update_view(
        &mut self,
        view_id: &str,
        executor: &mut QueryExecutor,
        dataset: &dyn Dataset,
    ) -> Result<()> {
        let _span = span!(Level::INFO, "update_view").entered();

        let start_time = Instant::now();

        // Get view definition
        let _definition = {
            let views = self.views.read().expect("lock poisoned");
            let view = views
                .get(view_id)
                .ok_or_else(|| anyhow!("View not found: {}", view_id))?;
            view.definition.clone()
        };

        // Check if incremental update is possible
        let use_incremental = {
            let views = self.views.read().expect("lock poisoned");
            let view = views
                .get(view_id)
                .expect("view should exist for given view_id");
            self.config.incremental_maintenance
                && view.maintenance_info.incremental_state.is_some()
                && self.can_update_incrementally(&view.dependencies)
        };

        if use_incremental {
            self.update_view_incrementally(view_id, executor, dataset)?;
        } else {
            self.update_view_fully(view_id, executor, dataset)?;
        }

        let update_time = start_time.elapsed();

        // Update maintenance info
        {
            let mut views = self.views.write().expect("lock poisoned");
            if let Some(view) = views.get_mut(view_id) {
                view.maintenance_info.last_updated = SystemTime::now();
                view.maintenance_info.update_count += 1;
                view.maintenance_info.total_maintenance_time += update_time;
                view.maintenance_info.needs_update = false;
                view.maintenance_info.next_maintenance =
                    self.calculate_next_maintenance(&view.maintenance_info.strategy);
            }
        }

        info!("Updated view {} in {:?}", view_id, update_time);
        Ok(())
    }

    /// Record view usage for statistics
    pub fn record_view_usage(
        &self,
        view_id: &str,
        query_hash: u64,
        time_saved: Duration,
        cost_benefit: f64,
    ) -> Result<()> {
        let mut stats = self.usage_statistics.write().expect("lock poisoned");

        // Update access count
        *stats.access_counts.entry(view_id.to_string()).or_insert(0) += 1;

        // Update time saved
        *stats
            .time_saved
            .entry(view_id.to_string())
            .or_insert(Duration::ZERO) += time_saved;

        // Update cost benefit
        let current_benefit = stats
            .cost_benefits
            .entry(view_id.to_string())
            .or_insert(0.0);
        *current_benefit = (*current_benefit + cost_benefit) / 2.0; // Moving average

        // Add usage record
        let usage_record = UsageRecord {
            timestamp: SystemTime::now(),
            query_hash,
            time_saved,
            cost_benefit,
        };

        stats
            .usage_history
            .entry(view_id.to_string())
            .or_default()
            .push_back(usage_record);

        // Limit history size
        if let Some(history) = stats.usage_history.get_mut(view_id) {
            while history.len() > 1000 {
                history.pop_front();
            }
        }

        Ok(())
    }

    // Private helper methods

    fn estimate_result_size(&self, results: &Solution) -> usize {
        // Estimate size based on number of results and average binding size
        results.len() * 100 // Rough estimate: 100 bytes per result
    }

    fn calculate_checksum(&self, results: &Solution) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for result in results {
            format!("{result:?}").hash(&mut hasher);
        }
        hasher.finish()
    }

    fn analyze_dependencies(&self, algebra: &Algebra) -> Result<ViewDependencies> {
        let mut base_tables = Vec::new();
        let mut dependent_patterns = Vec::new();
        let mut dependent_variables = HashSet::new();
        let mut join_dependencies = Vec::new();

        self.analyze_algebra_dependencies(
            algebra,
            &mut base_tables,
            &mut dependent_patterns,
            &mut dependent_variables,
            &mut join_dependencies,
        )?;

        Ok(ViewDependencies {
            base_tables,
            dependent_patterns,
            dependent_variables,
            join_dependencies,
        })
    }

    #[allow(clippy::only_used_in_recursion)]
    fn analyze_algebra_dependencies(
        &self,
        algebra: &Algebra,
        base_tables: &mut Vec<String>,
        dependent_patterns: &mut Vec<TriplePattern>,
        dependent_variables: &mut HashSet<Variable>,
        join_dependencies: &mut Vec<JoinDependency>,
    ) -> Result<()> {
        match algebra {
            Algebra::Bgp(patterns) => {
                dependent_patterns.extend(patterns.iter().cloned());
                for pattern in patterns {
                    self.extract_variables_from_pattern(pattern, dependent_variables);
                }
            }
            Algebra::Join { left, right } => {
                self.analyze_algebra_dependencies(
                    left,
                    base_tables,
                    dependent_patterns,
                    dependent_variables,
                    join_dependencies,
                )?;
                self.analyze_algebra_dependencies(
                    right,
                    base_tables,
                    dependent_patterns,
                    dependent_variables,
                    join_dependencies,
                )?;

                // Analyze join dependency
                if let (Algebra::Bgp(left_patterns), Algebra::Bgp(right_patterns)) =
                    (left.as_ref(), right.as_ref())
                {
                    if let (Some(left_pattern), Some(right_pattern)) =
                        (left_patterns.first(), right_patterns.first())
                    {
                        let join_vars = self.find_common_variables(left_pattern, right_pattern);
                        if !join_vars.is_empty() {
                            join_dependencies.push(JoinDependency {
                                left_pattern: left_pattern.clone(),
                                right_pattern: right_pattern.clone(),
                                join_variables: join_vars,
                                selectivity: 0.1, // Default selectivity
                            });
                        }
                    }
                }
            }
            Algebra::Union { left, right } => {
                self.analyze_algebra_dependencies(
                    left,
                    base_tables,
                    dependent_patterns,
                    dependent_variables,
                    join_dependencies,
                )?;
                self.analyze_algebra_dependencies(
                    right,
                    base_tables,
                    dependent_patterns,
                    dependent_variables,
                    join_dependencies,
                )?;
            }
            Algebra::Filter { pattern, condition } => {
                self.analyze_algebra_dependencies(
                    pattern,
                    base_tables,
                    dependent_patterns,
                    dependent_variables,
                    join_dependencies,
                )?;
                self.extract_variables_from_expression(condition, dependent_variables);
            }
            _ => {
                // Handle other algebra types as needed
            }
        }
        Ok(())
    }

    fn extract_variables_from_pattern(
        &self,
        pattern: &TriplePattern,
        variables: &mut HashSet<Variable>,
    ) {
        if let Term::Variable(var) = &pattern.subject {
            variables.insert(var.clone());
        }
        if let Term::Variable(var) = &pattern.predicate {
            variables.insert(var.clone());
        }
        if let Term::Variable(var) = &pattern.object {
            variables.insert(var.clone());
        }
    }

    #[allow(clippy::only_used_in_recursion)]
    fn extract_variables_from_expression(
        &self,
        expression: &Expression,
        variables: &mut HashSet<Variable>,
    ) {
        match expression {
            Expression::Variable(var) => {
                variables.insert(var.clone());
            }
            Expression::Binary { left, right, .. } => {
                self.extract_variables_from_expression(left, variables);
                self.extract_variables_from_expression(right, variables);
            }
            Expression::Unary { operand, .. } => {
                self.extract_variables_from_expression(operand, variables);
            }
            Expression::Function { args, .. } => {
                for arg in args {
                    self.extract_variables_from_expression(arg, variables);
                }
            }
            _ => {}
        }
    }

    fn find_common_variables(&self, left: &TriplePattern, right: &TriplePattern) -> Vec<Variable> {
        let mut left_vars = HashSet::new();
        let mut right_vars = HashSet::new();

        self.extract_variables_from_pattern(left, &mut left_vars);
        self.extract_variables_from_pattern(right, &mut right_vars);

        left_vars.intersection(&right_vars).cloned().collect()
    }

    fn calculate_view_costs(
        &self,
        _definition: &Algebra,
        view_data: &ViewData,
        _stats: &ExecutionStats,
    ) -> Result<ViewCostEstimates> {
        // Simplified cost calculation
        let access_cost = CostEstimate::new(
            view_data.row_count as f64 * 0.1,    // CPU cost
            0.0,                                 // I/O cost (in memory)
            view_data.size_bytes as f64 * 0.001, // Memory cost
            0.0,                                 // Network cost
            view_data.row_count,
        );

        let maintenance_cost = CostEstimate::new(
            view_data.row_count as f64 * 0.5,    // CPU cost for maintenance
            view_data.row_count as f64 * 0.1,    // I/O cost
            view_data.size_bytes as f64 * 0.002, // Memory cost
            0.0,                                 // Network cost
            view_data.row_count,
        );

        Ok(ViewCostEstimates {
            access_cost,
            maintenance_cost,
            storage_cost: view_data.size_bytes as f64,
            benefit_ratio: 2.0, // Assume 2x benefit by default
            last_estimated: SystemTime::now(),
        })
    }

    fn calculate_next_maintenance(&self, strategy: &MaintenanceStrategy) -> Option<SystemTime> {
        match strategy {
            MaintenanceStrategy::Periodic(interval) => Some(SystemTime::now() + *interval),
            MaintenanceStrategy::CostBased => {
                Some(SystemTime::now() + Duration::from_secs(3600)) // 1 hour default
            }
            MaintenanceStrategy::Hybrid => {
                Some(SystemTime::now() + Duration::from_secs(1800)) // 30 minutes default
            }
            _ => None,
        }
    }

    fn can_update_incrementally(&self, _dependencies: &ViewDependencies) -> bool {
        // Simplified check - in practice would analyze if incremental update is feasible
        true
    }

    fn update_view_incrementally(
        &mut self,
        view_id: &str,
        _executor: &QueryExecutor,
        _dataset: &dyn Dataset,
    ) -> Result<()> {
        // Simplified incremental update - would implement delta computation
        debug!("Performing incremental update for view {}", view_id);
        Ok(())
    }

    fn update_view_fully(
        &mut self,
        view_id: &str,
        executor: &mut QueryExecutor,
        dataset: &dyn Dataset,
    ) -> Result<()> {
        debug!("Performing full update for view {}", view_id);

        // Get view definition
        let definition = {
            let views = self.views.read().expect("lock poisoned");
            let view = views
                .get(view_id)
                .ok_or_else(|| anyhow!("View not found: {}", view_id))?;
            view.definition.clone()
        };

        // Re-execute the view definition
        let (results, stats) = executor.execute(&definition, dataset)?;

        // Calculate new data properties
        let size_bytes = self.estimate_result_size(&results);
        let checksum = self.calculate_checksum(&results);

        let new_data = ViewData {
            results,
            size_bytes,
            row_count: stats.final_results,
            materialized_at: SystemTime::now(),
            checksum,
        };

        // Update view data
        {
            let mut views = self.views.write().expect("lock poisoned");
            if let Some(view) = views.get_mut(view_id) {
                view.data = new_data.clone();
            }
        }

        // Update storage
        {
            let mut storage = self.view_storage.write().expect("lock poisoned");
            storage.store_view_data(view_id.to_string(), new_data)?;
        }

        Ok(())
    }
}
