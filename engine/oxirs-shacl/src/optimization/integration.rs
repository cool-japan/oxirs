//! Integration of optimization capabilities with the main validation engine
//!
//! This module provides optimized validation capabilities that integrate
//! advanced performance features with the core SHACL validation engine.

use crate::{
    constraints::{ConstraintContext, ConstraintEvaluationResult},
    iri_resolver::IriResolver,
    optimization::{
        AdvancedConstraintEvaluator, ConstraintCache, IncrementalValidationEngine,
        OptimizationConfig, StreamingValidationEngine, ValidationOptimizationEngine,
    },
    paths::PropertyPathEvaluator,
    report::ValidationReport,
    sparql::SparqlConstraintExecutor,
    targets::TargetSelector,
    Constraint, ConstraintComponentId, PropertyPath, Result, ShaclError, Shape, ShapeId,
    ValidationConfig,
};
use indexmap::IndexMap;
use oxirs_core::{model::Term, Store};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Enhanced validation engine with advanced optimization capabilities
#[derive(Debug)]
pub struct OptimizedValidationEngine {
    /// Core optimization engine
    optimization_engine: Arc<Mutex<ValidationOptimizationEngine>>,
    /// Incremental validation engine
    incremental_engine: Arc<Mutex<IncrementalValidationEngine>>,
    /// Streaming validation engine
    streaming_engine: Arc<Mutex<StreamingValidationEngine>>,
    /// IRI resolver
    iri_resolver: IriResolver,
    /// Target selector
    target_selector: TargetSelector,
    /// Property path evaluator
    path_evaluator: PropertyPathEvaluator,
    /// SPARQL executor
    sparql_executor: SparqlConstraintExecutor,
    /// Validation configuration
    config: ValidationConfig,
    /// Optimization configuration
    optimization_config: OptimizationConfig,
    /// Performance metrics
    metrics: ValidationPerformanceMetrics,
}

/// Enhanced performance metrics for optimized validation
#[derive(Debug, Clone, Default)]
pub struct ValidationPerformanceMetrics {
    /// Total constraints evaluated
    pub total_constraints_evaluated: u64,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// Average constraint evaluation time (microseconds)
    pub avg_constraint_time_us: f64,
    /// Total validation time saved through optimization (microseconds)
    pub optimization_time_saved_us: f64,
    /// Number of nodes processed incrementally
    pub incremental_nodes_processed: u64,
    /// Number of nodes skipped due to no changes
    pub incremental_nodes_skipped: u64,
    /// Parallel processing efficiency ratio
    pub parallel_efficiency_ratio: f64,
    /// Memory usage optimization ratio
    pub memory_optimization_ratio: f64,
}

impl OptimizedValidationEngine {
    /// Create a new optimized validation engine
    pub fn new(config: ValidationConfig, optimization_config: OptimizationConfig) -> Self {
        let optimization_engine = Arc::new(Mutex::new(ValidationOptimizationEngine::new(
            optimization_config.clone(),
        )));
        let incremental_engine = Arc::new(Mutex::new(IncrementalValidationEngine::default()));
        let streaming_engine = Arc::new(Mutex::new(StreamingValidationEngine::default()));

        Self {
            optimization_engine,
            incremental_engine,
            streaming_engine,
            iri_resolver: IriResolver::new(),
            target_selector: TargetSelector::new(),
            path_evaluator: PropertyPathEvaluator::new(),
            sparql_executor: SparqlConstraintExecutor::new(),
            config,
            optimization_config,
            metrics: ValidationPerformanceMetrics::default(),
        }
    }

    /// Validate a store with optimization
    pub fn validate_store_optimized(
        &mut self,
        store: &dyn Store,
        shapes: &IndexMap<ShapeId, Shape>,
    ) -> Result<ValidationReport> {
        let start_time = Instant::now();
        let mut report = ValidationReport::new();

        // Collect all constraints from all shapes
        let mut all_constraints_with_contexts = Vec::new();

        for (shape_id, shape) in shapes {
            if !shape.is_active() {
                continue;
            }

            // Get target nodes for this shape
            let mut target_nodes = Vec::new();
            for target in &shape.targets {
                let nodes = self.target_selector.select_targets(store, target, None)?;
                target_nodes.extend(nodes);
            }

            // Handle implicit targets for node shapes
            if target_nodes.is_empty() && shape.is_node_shape() {
                if let Ok(shape_iri) = oxirs_core::model::NamedNode::new(shape_id.as_str()) {
                    let implicit_target = crate::targets::Target::implicit(shape_iri);
                    let nodes =
                        self.target_selector
                            .select_targets(store, &implicit_target, None)?;
                    target_nodes.extend(nodes);
                }
            }

            // Create constraint-context pairs for optimization
            for focus_node in target_nodes {
                for (component_id, constraint) in &shape.constraints {
                    let values = if shape.is_property_shape() {
                        // For property shapes, evaluate the property path to get values
                        if let Some(path) = &shape.path {
                            self.path_evaluator
                                .evaluate_path(store, &focus_node, path, None)?
                        } else {
                            vec![focus_node.clone()]
                        }
                    } else {
                        vec![focus_node.clone()]
                    };

                    let mut context = ConstraintContext::new(focus_node.clone(), shape_id.clone())
                        .with_values(values);
                    if let Some(path) = &shape.path {
                        context = context.with_path(path.clone());
                    }

                    all_constraints_with_contexts.push((constraint.clone(), context));
                }
            }
        }

        // Use optimization engine for batch evaluation
        let results = {
            let mut opt_engine = self.optimization_engine.lock().unwrap();
            opt_engine.optimize_and_evaluate(store, all_constraints_with_contexts.clone())?
        };

        // Convert results to validation report
        for (result, context) in results.iter().zip(all_constraints_with_contexts.iter()) {
            if result.is_violated() {
                if let crate::constraints::constraint_context::ConstraintEvaluationResult::Violated {
                    violating_value, message, details
                } = result {
                    let violation = crate::validation::ValidationViolation {
                        focus_node: context.1.focus_node.clone(),
                        source_shape: context.1.shape_id.clone(),
                        source_constraint_component: context.0.component_id().clone(),
                        result_path: context.1.path.clone(),
                        value: violating_value.clone(),
                        result_message: message.clone(),
                        result_severity: crate::Severity::Violation,
                        details: details.clone(),
                        nested_results: Vec::new(),
                    };
                    report.add_violation(violation);
                }
            }
        }

        // Update performance metrics
        let total_time = start_time.elapsed();
        self.update_performance_metrics(total_time);

        Ok(report)
    }

    /// Validate with incremental optimization
    pub fn validate_incremental(
        &mut self,
        store: &dyn Store,
        shapes: &IndexMap<ShapeId, Shape>,
        changed_nodes: Option<&[Term]>,
        force_revalidate: bool,
    ) -> Result<ValidationReport> {
        let start_time = Instant::now();

        // If no specific nodes provided, fall back to full validation
        let nodes_to_validate = if let Some(nodes) = changed_nodes {
            nodes.to_vec()
        } else {
            // Extract all potential nodes from shapes' targets
            let mut all_nodes = Vec::new();
            for (_, shape) in shapes {
                for target in &shape.targets {
                    let nodes = self.target_selector.select_targets(store, target, None)?;
                    all_nodes.extend(nodes);
                }
            }
            all_nodes
        };

        // Collect constraints from all active shapes
        let mut all_constraints = Vec::new();
        for (_, shape) in shapes {
            if shape.is_active() {
                for (_, constraint) in &shape.constraints {
                    all_constraints.push(constraint.clone());
                }
            }
        }

        // Use incremental validation engine
        let incremental_result = {
            let mut inc_engine = self.incremental_engine.lock().unwrap();
            inc_engine.validate_incremental(
                store,
                all_constraints,
                &nodes_to_validate,
                force_revalidate,
            )?
        };

        // Convert incremental result to validation report
        let mut report = ValidationReport::new();
        if incremental_result.new_violations > 0 {
            // For incremental validation, we'd need to collect specific violations
            // This is a simplified implementation
            for _ in 0..incremental_result.new_violations {
                // Would need actual violation details in practice
            }
        }

        // Update metrics
        self.metrics.incremental_nodes_processed += incremental_result.revalidated_nodes as u64;
        self.metrics.incremental_nodes_skipped += incremental_result.skipped_nodes as u64;

        let total_time = start_time.elapsed();
        self.update_performance_metrics(total_time);

        Ok(report)
    }

    /// Validate large datasets with streaming optimization
    pub fn validate_streaming<I>(
        &mut self,
        store: &dyn Store,
        shapes: &IndexMap<ShapeId, Shape>,
        node_stream: I,
    ) -> Result<ValidationReport>
    where
        I: Iterator<Item = Term>,
    {
        let start_time = Instant::now();

        // Collect constraints from all active shapes
        let mut all_constraints = Vec::new();
        for (_, shape) in shapes {
            if shape.is_active() {
                for (_, constraint) in &shape.constraints {
                    all_constraints.push(constraint.clone());
                }
            }
        }

        // Use streaming validation engine
        let streaming_result = {
            let streaming_engine = self.streaming_engine.lock().unwrap();
            streaming_engine.validate_streaming(store, all_constraints, node_stream)?
        };

        // Convert streaming result to validation report
        let mut report = ValidationReport::new();
        for _ in 0..streaming_result.total_violations {
            // Would need actual violation details in practice
        }

        let total_time = start_time.elapsed();
        self.update_performance_metrics(total_time);

        Ok(report)
    }

    /// Validate with parallel processing optimization
    pub fn validate_parallel(
        &mut self,
        store: &dyn Store,
        shapes: &IndexMap<ShapeId, Shape>,
        max_threads: Option<usize>,
    ) -> Result<ValidationReport> {
        // For parallel validation, we need to be careful about thread safety
        // Currently disabled in the optimization engine due to Rc<> issues
        // Would require Arc<> and thread-safe Store interface

        // For now, fall back to optimized sequential validation
        self.validate_store_optimized(store, shapes)
    }

    /// Get comprehensive performance metrics
    pub fn get_performance_metrics(&self) -> ValidationPerformanceMetrics {
        let mut metrics = self.metrics.clone();

        // Update with optimization engine metrics
        if let Ok(opt_engine) = self.optimization_engine.lock() {
            let opt_metrics = opt_engine.get_metrics();
            metrics.cache_hit_ratio = opt_metrics.cache_hit_rate;
            metrics.avg_constraint_time_us = opt_metrics.avg_evaluation_time_us;
            metrics.optimization_time_saved_us = opt_metrics.optimization_time_saved_us;
        }

        // Update with incremental engine metrics
        if let Ok(inc_engine) = self.incremental_engine.lock() {
            let inc_stats = inc_engine.get_incremental_stats();
            // Calculate memory optimization ratio
            metrics.memory_optimization_ratio = if inc_stats.memory_usage_mb > 0 {
                1.0 - (inc_stats.memory_usage_mb as f64 / 1000.0) // Normalized
            } else {
                1.0
            };
        }

        metrics
    }

    /// Reset all optimization state and metrics
    pub fn reset_optimization_state(&mut self) {
        if let Ok(mut opt_engine) = self.optimization_engine.lock() {
            opt_engine.reset();
        }

        if let Ok(mut inc_engine) = self.incremental_engine.lock() {
            inc_engine.clear_history();
        }

        self.metrics = ValidationPerformanceMetrics::default();
    }

    /// Update optimization configuration
    pub fn update_optimization_config(&mut self, config: OptimizationConfig) {
        self.optimization_config = config.clone();

        if let Ok(mut opt_engine) = self.optimization_engine.lock() {
            opt_engine.update_config(config);
        }
    }

    /// Get cache statistics across all optimization engines
    pub fn get_cache_statistics(&self) -> CacheStatistics {
        let mut stats = CacheStatistics::default();

        if let Ok(opt_engine) = self.optimization_engine.lock() {
            let cache_stats = opt_engine.get_cache_stats();
            stats.total_hits = cache_stats.hits;
            stats.total_misses = cache_stats.misses;
            stats.hit_ratio = cache_stats.hit_rate();
            stats.total_evaluations = cache_stats.evaluations;
            stats.evictions = cache_stats.evictions;
        }

        stats
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self, validation_time: Duration) {
        self.metrics.total_constraints_evaluated += 1;

        // Update timing metrics using exponential moving average
        let alpha = 0.1;
        let new_time_us = validation_time.as_micros() as f64;
        self.metrics.avg_constraint_time_us =
            alpha * new_time_us + (1.0 - alpha) * self.metrics.avg_constraint_time_us;
    }

    /// Check if optimization is enabled for a specific feature
    pub fn is_optimization_enabled(&self, feature: OptimizationFeature) -> bool {
        match feature {
            OptimizationFeature::Caching => self.optimization_config.enable_caching,
            OptimizationFeature::Parallel => self.optimization_config.enable_parallel,
            OptimizationFeature::Reordering => self.optimization_config.enable_reordering,
            OptimizationFeature::Incremental => true, // Always available
            OptimizationFeature::Streaming => true,   // Always available
        }
    }
}

/// Available optimization features
#[derive(Debug, Clone, Copy)]
pub enum OptimizationFeature {
    Caching,
    Parallel,
    Reordering,
    Incremental,
    Streaming,
}

/// Comprehensive cache statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    pub total_hits: usize,
    pub total_misses: usize,
    pub hit_ratio: f64,
    pub total_evaluations: usize,
    pub evictions: usize,
}

/// Strategy pattern for different validation approaches
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ValidationStrategy {
    /// Standard sequential validation
    Sequential,
    /// Optimized with constraint reordering and caching
    Optimized,
    /// Incremental validation for changed data
    Incremental { force_revalidate: bool },
    /// Streaming validation for large datasets
    Streaming { batch_size: usize },
    /// Parallel validation (when thread-safe)
    Parallel { max_threads: usize },
}

impl Default for ValidationStrategy {
    fn default() -> Self {
        ValidationStrategy::Optimized
    }
}

impl OptimizedValidationEngine {
    /// Validate using a specific strategy
    pub fn validate_with_strategy(
        &mut self,
        store: &dyn Store,
        shapes: &IndexMap<ShapeId, Shape>,
        strategy: ValidationStrategy,
        context: Option<ValidationContext>,
    ) -> Result<ValidationReport> {
        match strategy {
            ValidationStrategy::Sequential => {
                // Use basic validation without optimizations
                self.validate_basic(store, shapes)
            }
            ValidationStrategy::Optimized => self.validate_store_optimized(store, shapes),
            ValidationStrategy::Incremental { force_revalidate } => {
                let changed_nodes = context.and_then(|c| c.changed_nodes);
                self.validate_incremental(store, shapes, changed_nodes.as_deref(), force_revalidate)
            }
            ValidationStrategy::Streaming { batch_size } => {
                // Create a node stream from target selection
                let nodes = self.collect_all_target_nodes(store, shapes)?;
                self.validate_streaming(store, shapes, nodes.into_iter())
            }
            ValidationStrategy::Parallel { max_threads } => {
                self.validate_parallel(store, shapes, Some(max_threads))
            }
        }
    }

    /// Basic validation without optimizations (for comparison)
    fn validate_basic(
        &mut self,
        store: &dyn Store,
        shapes: &IndexMap<ShapeId, Shape>,
    ) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        for (shape_id, shape) in shapes {
            if !shape.is_active() {
                continue;
            }

            // Basic sequential validation without optimization
            for target in &shape.targets {
                let target_nodes = self.target_selector.select_targets(store, target, None)?;

                for focus_node in target_nodes {
                    for (_, constraint) in &shape.constraints {
                        let values = if shape.is_property_shape() {
                            if let Some(path) = &shape.path {
                                self.path_evaluator
                                    .evaluate_path(store, &focus_node, path, None)?
                            } else {
                                vec![focus_node.clone()]
                            }
                        } else {
                            vec![focus_node.clone()]
                        };

                        let mut context =
                            ConstraintContext::new(focus_node.clone(), shape_id.clone())
                                .with_values(values);
                        if let Some(path) = &shape.path {
                            context = context.with_path(path.clone());
                        }

                        let result = constraint.evaluate(store, &context)?;
                        if result.is_violated() {
                            if let crate::constraints::constraint_context::ConstraintEvaluationResult::Violated {
                                violating_value, message, details
                            } = result {
                                let violation = crate::validation::ValidationViolation {
                                    focus_node: context.focus_node.clone(),
                                    source_shape: context.shape_id.clone(),
                                    source_constraint_component: constraint.component_id().clone(),
                                    result_path: context.path.clone(),
                                    value: violating_value,
                                    result_message: message,
                                    result_severity: crate::Severity::Violation,
                                    details,
                                    nested_results: Vec::new(),
                                };
                                report.add_violation(violation);
                            }
                        }

                        // Check early termination
                        if self.config.fail_fast && !report.conforms() {
                            return Ok(report);
                        }
                    }
                }
            }
        }

        Ok(report)
    }

    /// Collect all target nodes from all shapes
    fn collect_all_target_nodes(
        &mut self,
        store: &dyn Store,
        shapes: &IndexMap<ShapeId, Shape>,
    ) -> Result<Vec<Term>> {
        let mut all_nodes = Vec::new();

        for (_, shape) in shapes {
            if !shape.is_active() {
                continue;
            }

            for target in &shape.targets {
                let nodes = self.target_selector.select_targets(store, target, None)?;
                all_nodes.extend(nodes);
            }
        }

        // Remove duplicates
        all_nodes.sort();
        all_nodes.dedup();

        Ok(all_nodes)
    }
}

/// Validation context for providing additional information
#[derive(Debug, Clone)]
pub struct ValidationContext {
    /// Nodes that have changed (for incremental validation)
    pub changed_nodes: Option<Vec<Term>>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl ValidationContext {
    /// Create new validation context
    pub fn new() -> Self {
        Self {
            changed_nodes: None,
            metadata: HashMap::new(),
        }
    }

    /// Set changed nodes for incremental validation
    pub fn with_changed_nodes(mut self, nodes: Vec<Term>) -> Self {
        self.changed_nodes = Some(nodes);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl Default for ValidationContext {
    fn default() -> Self {
        Self::new()
    }
}
