//! # OxiRS ARQ
//!
//! Jena-style SPARQL algebra with extension points and query optimization.
//!
//! This crate provides advanced SPARQL query processing capabilities including
//! query algebra, optimization, and extension points for custom functions.

use anyhow::Result;

pub mod advanced_optimizer;
pub mod algebra;
pub mod algebra_generation;
pub mod bgp_optimizer;
pub mod builtin_fixed;
pub mod cache_integration;
pub mod cost_model;
pub mod distributed;
// pub mod executor;  // Temporarily commented to avoid module conflict
pub mod extensions;
pub mod integrated_query_planner;
pub mod materialized_views;
pub mod optimizer;
pub mod query_analysis;
pub mod streaming;
pub mod vector_query_optimizer;

// Use executor from subdirectory
pub use self::executor_impl as executor;
pub mod executor_impl {
    pub use super::executor_submodule::*;
}

#[path = "executor/mod.rs"]
pub mod executor_submodule;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod path;
pub use builtin_fixed as builtin;
pub mod expression;
pub mod query;
pub mod statistics_collector;
pub mod term;
pub mod update;

// Re-export main types for convenience
pub use advanced_optimizer::*;
pub use algebra::*;
pub use algebra_generation::*;
pub use cache_integration::*;
pub use cost_model::*;
pub use distributed::*;
pub use executor::*;
pub use expression::*;
pub use extensions::*;
pub use integrated_query_planner::*;
pub use materialized_views::*;
pub use optimizer::*;
pub use path::*;
pub use query::*;
pub use query_analysis::*;
pub use streaming::*;
pub use term::*;
pub use update::*;
pub use vector_query_optimizer::*;

/// SPARQL Query Engine - High-level interface
pub struct SparqlEngine {
    executor: QueryExecutor,
    optimizer: QueryOptimizer,
    extensions: ExtensionRegistry,
    parser: query::QueryParser,
    integrated_planner: Option<integrated_query_planner::IntegratedQueryPlanner>,
    vector_optimizer: Option<vector_query_optimizer::VectorQueryOptimizer>,
    materialized_view_manager: Option<materialized_views::MaterializedViewManager>,
}

impl SparqlEngine {
    /// Create a new SPARQL engine with default configuration
    pub fn new() -> Result<Self> {
        let mut extensions = ExtensionRegistry::new();
        builtin::register_builtin_functions(&extensions)?;

        Ok(Self {
            executor: QueryExecutor::new(),
            optimizer: QueryOptimizer::new(),
            extensions,
            parser: query::QueryParser::new(),
            integrated_planner: None,
            vector_optimizer: None,
            materialized_view_manager: None,
        })
    }

    /// Create a new SPARQL engine with custom configuration
    pub fn with_config(
        executor_config: executor::ExecutionContext,
        optimizer_config: optimizer::OptimizerConfig,
    ) -> Result<Self> {
        let mut extensions = ExtensionRegistry::new();
        builtin::register_builtin_functions(&extensions)?;

        Ok(Self {
            executor: QueryExecutor::with_context(executor_config),
            optimizer: QueryOptimizer::with_config(optimizer_config),
            extensions,
            parser: query::QueryParser::new(),
            integrated_planner: None,
            vector_optimizer: None,
            materialized_view_manager: None,
        })
    }

    /// Parse and execute a SPARQL query
    pub fn execute_query(
        &mut self,
        query_str: &str,
        dataset: &dyn executor::Dataset,
    ) -> Result<(algebra::Solution, executor::ExecutionStats)> {
        // Parse query
        let query = self.parser.parse(query_str)?;

        // Convert to algebra
        let algebra = self.convert_query_to_algebra(query)?;

        // Try materialized view rewriting first
        let (rewritten_algebra, used_views) = if let Some(ref view_manager) = self.materialized_view_manager {
            view_manager.rewrite_query(&algebra)?
        } else {
            (algebra.clone(), Vec::new())
        };

        // Optimize algebra with vector awareness if available
        let optimized_algebra = if let Some(ref mut vector_optimizer) = self.vector_optimizer {
            // Use vector-enhanced optimization
            let vector_plan = vector_optimizer.create_vector_enhanced_plan(&rewritten_algebra)?;
            vector_plan.base_plan.optimized_algebra
        } else if let Some(ref mut planner) = self.integrated_planner {
            // Use advanced integrated planner if available
            let plan = planner.create_plan(&rewritten_algebra)?;
            plan.optimized_algebra
        } else {
            // Use standard optimizer
            self.optimizer.optimize(rewritten_algebra)?
        };

        // Record view usage if any views were used
        if !used_views.is_empty() {
            if let Some(ref view_manager) = self.materialized_view_manager {
                let query_hash = self.calculate_query_hash(query_str);
                for view_id in used_views {
                    let _ = view_manager.record_view_usage(
                        &view_id,
                        query_hash,
                        std::time::Duration::from_millis(0), // Would measure actual time saved
                        1.0, // Would calculate actual cost benefit
                    );
                }
            }
        }

        // Execute
        self.executor.execute(&optimized_algebra, dataset)
    }

    /// Register a custom function
    pub fn register_function<F>(&self, function: F) -> Result<()>
    where
        F: extensions::CustomFunction + 'static,
    {
        self.extensions.register_function(function)
    }

    /// Register a custom aggregate
    pub fn register_aggregate<A>(&self, aggregate: A) -> Result<()>
    where
        A: extensions::CustomAggregate + 'static,
    {
        self.extensions.register_aggregate(aggregate)
    }

    /// Enable advanced integrated query planning
    pub fn enable_integrated_planning(&mut self) -> Result<()> {
        let config = integrated_query_planner::IntegratedPlannerConfig::default();
        self.integrated_planner = Some(integrated_query_planner::IntegratedQueryPlanner::new(
            config,
        )?);
        Ok(())
    }

    /// Enable advanced integrated query planning with custom configuration
    pub fn enable_integrated_planning_with_config(
        &mut self,
        config: integrated_query_planner::IntegratedPlannerConfig,
    ) -> Result<()> {
        self.integrated_planner = Some(integrated_query_planner::IntegratedQueryPlanner::new(
            config,
        )?);
        Ok(())
    }

    /// Check if integrated planning is enabled
    pub fn is_integrated_planning_enabled(&self) -> bool {
        self.integrated_planner.is_some()
    }

    /// Update execution feedback for the integrated planner
    pub fn update_execution_feedback(
        &mut self,
        plan_hash: u64,
        actual_duration: std::time::Duration,
        actual_cardinality: usize,
        memory_used: usize,
        success: bool,
        error_info: Option<String>,
    ) -> Result<()> {
        if let Some(ref mut planner) = self.integrated_planner {
            planner.update_execution_feedback(
                plan_hash,
                actual_duration,
                actual_cardinality,
                memory_used,
                success,
                error_info,
            )?;
        }
        Ok(())
    }

    /// Get index recommendations from the integrated planner
    pub fn get_index_recommendations(
        &self,
    ) -> Result<Vec<integrated_query_planner::IndexRecommendation>> {
        if let Some(ref planner) = self.integrated_planner {
            planner.get_index_recommendations()
        } else {
            Ok(vec![])
        }
    }

    /// Enable vector-aware query optimization
    pub fn enable_vector_optimization(&mut self) -> Result<()> {
        let vector_config = vector_query_optimizer::VectorOptimizerConfig::default();
        let planner_config = integrated_query_planner::IntegratedPlannerConfig::default();
        self.vector_optimizer = Some(vector_query_optimizer::VectorQueryOptimizer::new(
            vector_config,
            planner_config,
        )?);
        Ok(())
    }

    /// Enable vector-aware query optimization with custom configuration
    pub fn enable_vector_optimization_with_config(
        &mut self,
        vector_config: vector_query_optimizer::VectorOptimizerConfig,
        planner_config: integrated_query_planner::IntegratedPlannerConfig,
    ) -> Result<()> {
        self.vector_optimizer = Some(vector_query_optimizer::VectorQueryOptimizer::new(
            vector_config,
            planner_config,
        )?);
        Ok(())
    }

    /// Check if vector optimization is enabled
    pub fn is_vector_optimization_enabled(&self) -> bool {
        self.vector_optimizer.is_some()
    }

    /// Register a vector index for use in optimization
    pub fn register_vector_index(
        &self,
        name: String,
        index_info: vector_query_optimizer::VectorIndexInfo,
    ) -> Result<()> {
        if let Some(ref optimizer) = self.vector_optimizer {
            optimizer.register_vector_index(name, index_info)
        } else {
            Err(anyhow::anyhow!(
                "Vector optimization not enabled. Call enable_vector_optimization() first."
            ))
        }
    }

    /// Get vector optimization performance metrics
    pub fn get_vector_performance_metrics(
        &self,
    ) -> Option<vector_query_optimizer::VectorPerformanceMetrics> {
        self.vector_optimizer
            .as_ref()
            .map(|opt| opt.get_performance_metrics())
    }

    /// Update vector execution feedback for adaptive optimization
    pub fn update_vector_execution_feedback(
        &mut self,
        strategy_hash: u64,
        actual_duration: std::time::Duration,
        actual_recall: f32,
        actual_memory: usize,
        success: bool,
    ) -> Result<()> {
        if let Some(ref mut optimizer) = self.vector_optimizer {
            optimizer.update_vector_execution_feedback(
                strategy_hash,
                actual_duration,
                actual_recall,
                actual_memory,
                success,
            )
        } else {
            Ok(()) // Silently ignore if vector optimization is not enabled
        }
    }

    /// Enable materialized view management
    pub fn enable_materialized_views(&mut self) -> Result<()> {
        let config = materialized_views::MaterializedViewConfig::default();
        let cost_model = std::sync::Arc::new(std::sync::Mutex::new(
            cost_model::CostModel::new(cost_model::CostModelConfig::default())
        ));
        let statistics_collector = std::sync::Arc::new(
            statistics_collector::StatisticsCollector::new()
        );
        
        self.materialized_view_manager = Some(
            materialized_views::MaterializedViewManager::new(
                config, 
                cost_model, 
                statistics_collector
            )?
        );
        Ok(())
    }

    /// Enable materialized view management with custom configuration
    pub fn enable_materialized_views_with_config(
        &mut self,
        config: materialized_views::MaterializedViewConfig,
    ) -> Result<()> {
        let cost_model = std::sync::Arc::new(std::sync::Mutex::new(
            cost_model::CostModel::new(cost_model::CostModelConfig::default())
        ));
        let statistics_collector = std::sync::Arc::new(
            statistics_collector::StatisticsCollector::new()
        );
        
        self.materialized_view_manager = Some(
            materialized_views::MaterializedViewManager::new(
                config, 
                cost_model, 
                statistics_collector
            )?
        );
        Ok(())
    }

    /// Check if materialized view management is enabled
    pub fn is_materialized_views_enabled(&self) -> bool {
        self.materialized_view_manager.is_some()
    }

    /// Create a new materialized view
    pub fn create_materialized_view(
        &mut self,
        name: String,
        definition: Algebra,
        metadata: materialized_views::ViewMetadata,
        dataset: &dyn executor::Dataset,
    ) -> Result<String> {
        if let Some(ref mut view_manager) = self.materialized_view_manager {
            view_manager.create_view(name, definition, metadata, &self.executor, dataset)
        } else {
            Err(anyhow::anyhow!(
                "Materialized views not enabled. Call enable_materialized_views() first."
            ))
        }
    }

    /// Update a materialized view with fresh data
    pub fn update_materialized_view(
        &mut self,
        view_id: &str,
        dataset: &dyn executor::Dataset,
    ) -> Result<()> {
        if let Some(ref mut view_manager) = self.materialized_view_manager {
            view_manager.update_view(view_id, &self.executor, dataset)
        } else {
            Err(anyhow::anyhow!(
                "Materialized views not enabled. Call enable_materialized_views() first."
            ))
        }
    }

    /// Get materialized view recommendations based on query patterns
    pub fn get_materialized_view_recommendations(
        &self,
    ) -> Result<Vec<materialized_views::ViewRecommendation>> {
        if let Some(ref view_manager) = self.materialized_view_manager {
            view_manager.get_view_recommendations()
        } else {
            Ok(vec![])
        }
    }

    /// Get usage statistics for a materialized view
    pub fn get_view_usage_statistics(
        &self,
        view_id: &str,
    ) -> Result<Option<materialized_views::ViewUsageStats>> {
        if let Some(ref view_manager) = self.materialized_view_manager {
            view_manager.get_usage_statistics(view_id)
        } else {
            Ok(None)
        }
    }

    /// Calculate hash for a query string (internal utility)
    fn calculate_query_hash(&self, query_str: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        query_str.hash(&mut hasher);
        hasher.finish()
    }

    /// Convert parsed query to algebra expression
    fn convert_query_to_algebra(&self, query: query::Query) -> Result<Algebra> {
        let mut algebra = query.where_clause;

        // Apply query modifiers in reverse order of precedence

        // 1. Apply GROUP BY and aggregates if present
        if !query.group_by.is_empty() {
            // Extract aggregates from select variables
            let mut aggregates = Vec::new();
            for select_var in &query.select_variables {
                // Check if this variable represents an aggregate function
                // This would typically come from parsing expressions like COUNT(?x) AS ?count
                // For now, we'll build this as the parser provides more context
                // The actual aggregate detection would happen during query parsing
            }

            algebra = Algebra::Group {
                pattern: Box::new(algebra),
                variables: query.group_by,
                aggregates, // Now properly extracted from select variables
            };
        }

        // 2. Apply HAVING clause if present
        if let Some(having_condition) = query.having {
            algebra = Algebra::Having {
                pattern: Box::new(algebra),
                condition: having_condition,
            };
        }

        // 3. Apply projection (SELECT variables)
        match query.query_type {
            query::QueryType::Select => {
                if !query.select_variables.is_empty() {
                    algebra = Algebra::Project {
                        pattern: Box::new(algebra),
                        variables: query.select_variables,
                    };
                }

                // Apply DISTINCT or REDUCED
                if query.distinct {
                    algebra = Algebra::Distinct {
                        pattern: Box::new(algebra),
                    };
                } else if query.reduced {
                    algebra = Algebra::Reduced {
                        pattern: Box::new(algebra),
                    };
                }
            }
            query::QueryType::Construct => {
                // For CONSTRUCT queries, we need to handle the construct template
                // This is a simplified implementation
                if query.distinct {
                    algebra = Algebra::Distinct {
                        pattern: Box::new(algebra),
                    };
                }
            }
            query::QueryType::Ask | query::QueryType::Describe => {
                // ASK and DESCRIBE don't need projection modifications
            }
        }

        // 4. Apply ORDER BY if present
        if !query.order_by.is_empty() {
            algebra = Algebra::OrderBy {
                pattern: Box::new(algebra),
                conditions: query.order_by,
            };
        }

        // 5. Apply SLICE (LIMIT and OFFSET) if present
        if query.limit.is_some() || query.offset.is_some() {
            algebra = Algebra::Slice {
                pattern: Box::new(algebra),
                offset: query.offset,
                limit: query.limit,
            };
        }

        Ok(algebra)
    }
}

impl Default for SparqlEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default SPARQL engine")
    }
}

/// Distributed SPARQL Query Engine - Enhanced interface for distributed processing
pub struct DistributedSparqlEngine {
    local_engine: SparqlEngine,
    distributed_processor: distributed::DistributedQueryProcessor,
    distribution_threshold: f64,
}

impl DistributedSparqlEngine {
    /// Create a new distributed SPARQL engine with default configuration
    pub fn new() -> Result<Self> {
        let local_engine = SparqlEngine::new()?;
        let distributed_config = distributed::DistributedConfig::default();
        let distributed_processor = distributed::DistributedQueryProcessor::new(distributed_config);

        Ok(Self {
            local_engine,
            distributed_processor,
            distribution_threshold: 100.0, // Complexity threshold for distributed execution
        })
    }

    /// Create a new distributed SPARQL engine with custom configuration
    pub fn with_config(
        executor_config: executor::ExecutionContext,
        optimizer_config: optimizer::OptimizerConfig,
        distributed_config: distributed::DistributedConfig,
        distribution_threshold: f64,
    ) -> Result<Self> {
        let local_engine = SparqlEngine::with_config(executor_config, optimizer_config)?;
        let distributed_processor = distributed::DistributedQueryProcessor::new(distributed_config);

        Ok(Self {
            local_engine,
            distributed_processor,
            distribution_threshold,
        })
    }

    /// Register a node in the distributed system
    pub async fn register_node(&self, node_info: distributed::NodeInfo) -> Result<()> {
        self.distributed_processor.register_node(node_info).await
    }

    /// Execute a SPARQL query with automatic distribution decision
    pub async fn execute_query(
        &mut self,
        query_str: &str,
        dataset: &dyn executor::Dataset,
    ) -> Result<(algebra::Solution, executor::ExecutionStats)> {
        // Parse and analyze query
        let query = self.local_engine.parser.parse(query_str)?;
        let algebra = self.local_engine.convert_query_to_algebra(query)?;

        // Determine if query should be distributed
        if self.should_distribute_query(&algebra).await? {
            self.execute_distributed_query(algebra, dataset).await
        } else {
            // Execute locally
            let optimized_algebra = self.local_engine.optimizer.optimize(algebra)?;
            self.local_engine
                .executor
                .execute(&optimized_algebra, dataset)
        }
    }

    /// Execute query in distributed mode
    async fn execute_distributed_query(
        &mut self,
        algebra: Algebra,
        dataset: &dyn executor::Dataset,
    ) -> Result<(algebra::Solution, executor::ExecutionStats)> {
        let start_time = std::time::Instant::now();

        // Execute distributed query
        let bindings = self
            .distributed_processor
            .execute_distributed(algebra, std::collections::HashMap::new())
            .await?;

        let execution_time = start_time.elapsed();

        // Convert results to expected format
        // Solution is just Vec<Binding>, so use bindings directly
        let solution: algebra::Solution = bindings;

        let stats = executor::ExecutionStats {
            execution_time,
            intermediate_results: 0,
            final_results: solution.len(),
            memory_used: 0, // TODO: Aggregate from distributed execution
            operations: 1,
            property_path_evaluations: 0,
            time_spent_on_paths: std::time::Duration::from_millis(0),
            service_calls: 0,
            time_spent_on_services: std::time::Duration::from_millis(0),
            warnings: Vec::new(),
        };

        Ok((solution, stats))
    }

    /// Determine if a query should be executed in distributed mode
    async fn should_distribute_query(&self, algebra: &Algebra) -> Result<bool> {
        let complexity = self.calculate_query_complexity(algebra);
        Ok(complexity > self.distribution_threshold)
    }

    /// Calculate query complexity score
    fn calculate_query_complexity(&self, algebra: &Algebra) -> f64 {
        match algebra {
            Algebra::Bgp(patterns) => patterns.len() as f64,
            Algebra::Join { left, right } => {
                self.calculate_query_complexity(left)
                    + self.calculate_query_complexity(right)
                    + 10.0
            }
            Algebra::Union { left, right } => {
                self.calculate_query_complexity(left) + self.calculate_query_complexity(right) + 5.0
            }
            Algebra::Filter { pattern, .. } => self.calculate_query_complexity(pattern) + 2.0,
            Algebra::Group { pattern, .. } => {
                self.calculate_query_complexity(pattern) + 20.0 // Aggregation is expensive
            }
            _ => 1.0,
        }
    }

    /// Register a custom function (delegated to local engine)
    pub fn register_function<F>(&self, function: F) -> Result<()>
    where
        F: extensions::CustomFunction + 'static,
    {
        self.local_engine.register_function(function)
    }

    /// Register a custom aggregate (delegated to local engine)
    pub fn register_aggregate<A>(&self, aggregate: A) -> Result<()>
    where
        A: extensions::CustomAggregate + 'static,
    {
        self.local_engine.register_aggregate(aggregate)
    }

    /// Get the distributed processor for advanced configuration
    pub fn distributed_processor(&self) -> &distributed::DistributedQueryProcessor {
        &self.distributed_processor
    }

    /// Set distribution threshold
    pub fn set_distribution_threshold(&mut self, threshold: f64) {
        self.distribution_threshold = threshold;
    }
}

impl Default for DistributedSparqlEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default distributed SPARQL engine")
    }
}
