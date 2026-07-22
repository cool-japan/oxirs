//! Core validation engine implementation

#![allow(dead_code)]

use indexmap::IndexMap;
use std::sync::Arc;
use std::time::Instant;

use oxirs_core::{
    model::{NamedNode, Term},
    Store,
};

use crate::{
    constraints::*, iri_resolver::*, optimization::*, paths::*, shapes::*, sparql::*, targets::*,
    Constraint, ConstraintComponentId, PropertyPath, Result, ShaclError, Shape, ShapeId, Target,
    ValidationConfig, ValidationReport,
};

use super::{
    cache::{ConstraintCache, InheritanceCache},
    stats::ValidationStats,
    ConstraintCacheKey, ConstraintEvaluationResult, ValidationViolation,
};

/// Core SHACL validation engine
///
/// The `ValidationEngine` is the core component responsible for validating RDF data
/// against SHACL shapes. It supports various validation strategies and optimization
/// techniques for different use cases.
///
/// ## Features
///
/// - **Multiple validation strategies**: Sequential, parallel, incremental, and streaming
/// - **Constraint caching**: Intelligent caching of constraint evaluation results
/// - **Performance optimization**: Advanced optimization engine for large datasets
/// - **Error recovery**: Graceful handling of malformed data and constraints
/// - **Extensibility**: Support for custom constraint components and functions
///
/// ## Example
///
/// ```rust
/// use oxirs_shacl::{ValidationEngine, ValidationConfig, Shape, ShapeId};
/// use indexmap::IndexMap;
/// use oxirs_core::Store;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Prepare shapes (normally loaded from RDF)
/// let shapes = IndexMap::new();
/// let config = ValidationConfig::default();
///
/// // Create validation engine
/// let engine = ValidationEngine::new(&shapes, config);
///
/// // Validate a store
/// // let store = Store::new()?;
/// // let report = engine.validate_store(&store)?;
/// # Ok(())
/// # }
/// ```
///
/// ## Validation Strategies
///
/// The engine supports different validation strategies optimized for different scenarios:
///
/// - **Sequential**: Single-threaded validation, best for small datasets
/// - **Parallel**: Multi-threaded validation, optimal for large datasets
/// - **Incremental**: Only validates changes, ideal for real-time applications
/// - **Streaming**: Memory-efficient validation for very large datasets
/// - **Optimized**: Uses advanced optimization techniques and caching
///
/// ## Performance Considerations
///
/// For optimal performance:
/// - Use parallel strategy for datasets > 10,000 triples
/// - Enable constraint caching for repeated validations
/// - Use incremental validation for real-time scenarios
/// - Configure memory limits for streaming validation
///
#[derive(Debug)]
pub struct ValidationEngine<'a> {
    /// Reference to shapes to validate against
    shapes: &'a IndexMap<ShapeId, Shape>,

    /// Validation configuration
    config: ValidationConfig,

    /// IRI resolver for namespace expansion and validation
    iri_resolver: IriResolver,

    /// Target selector for finding focus nodes
    target_selector: TargetSelector,

    /// Property path evaluator
    path_evaluator: PropertyPathEvaluator,

    /// SPARQL constraint executor
    sparql_executor: SparqlConstraintExecutor,

    /// Shape validator for validating shapes graphs
    shape_validator: ShapeValidator,

    /// Validation statistics
    stats: ValidationStats,

    /// Cache for constraint evaluation results
    constraint_cache: ConstraintCache,

    /// Cache for resolved inherited constraints
    inheritance_cache: InheritanceCache,

    /// Optimized validation engine for enhanced performance
    optimization_engine: Option<OptimizedValidationEngine>,

    /// Current validation strategy
    validation_strategy: ValidationStrategy,
}

impl<'a> ValidationEngine<'a> {
    /// Create a new validation engine
    pub fn new(shapes: &'a IndexMap<ShapeId, Shape>, config: ValidationConfig) -> Self {
        Self {
            shapes,
            config,
            iri_resolver: IriResolver::new(),
            target_selector: TargetSelector::new(),
            path_evaluator: PropertyPathEvaluator::new(),
            sparql_executor: SparqlConstraintExecutor::new(),
            shape_validator: ShapeValidator::new(),
            stats: ValidationStats::default(),
            constraint_cache: ConstraintCache::new(),
            inheritance_cache: InheritanceCache::new(),
            optimization_engine: None,
            validation_strategy: ValidationStrategy::default(),
        }
    }

    /// Create a new validation engine with custom IRI resolver
    pub fn with_iri_resolver(
        shapes: &'a IndexMap<ShapeId, Shape>,
        config: ValidationConfig,
        iri_resolver: IriResolver,
    ) -> Self {
        Self {
            shapes,
            config,
            iri_resolver,
            target_selector: TargetSelector::new(),
            path_evaluator: PropertyPathEvaluator::new(),
            sparql_executor: SparqlConstraintExecutor::new(),
            shape_validator: ShapeValidator::new(),
            stats: ValidationStats::default(),
            constraint_cache: ConstraintCache::new(),
            inheritance_cache: InheritanceCache::new(),
            optimization_engine: None,
            validation_strategy: ValidationStrategy::default(),
        }
    }

    /// Get a reference to the IRI resolver
    pub fn iri_resolver(&self) -> &IriResolver {
        &self.iri_resolver
    }

    /// Get a mutable reference to the IRI resolver
    pub fn iri_resolver_mut(&mut self) -> &mut IriResolver {
        &mut self.iri_resolver
    }

    /// Get validation statistics
    pub fn get_statistics(&self) -> &ValidationStats {
        &self.stats
    }

    /// Get cache hit rate
    pub fn get_cache_hit_rate(&self) -> f64 {
        self.constraint_cache.hit_rate()
    }

    /// Clear all validation caches
    pub fn clear_caches(&mut self) {
        self.constraint_cache.clear();
        self.inheritance_cache.clear();
    }

    /// Get the number of shapes in the engine
    ///
    /// Returns the total count of shapes that this engine is configured to validate against.
    pub fn shape_count(&self) -> usize {
        self.shapes.len()
    }

    /// Get a reference to all shapes
    ///
    /// Returns a reference to the IndexMap containing all shapes indexed by ShapeId.
    pub fn get_shapes(&self) -> &IndexMap<ShapeId, Shape> {
        self.shapes
    }

    /// Get a specific shape by ID
    ///
    /// Returns a reference to the shape with the given ID, or None if no such shape exists.
    pub fn get_shape(&self, shape_id: &ShapeId) -> Option<&Shape> {
        self.shapes.get(shape_id)
    }

    /// Get all shape IDs
    ///
    /// Returns an iterator over all shape IDs in the engine.
    pub fn shape_ids(&self) -> impl Iterator<Item = &ShapeId> {
        self.shapes.keys()
    }

    /// Check if a shape with the given ID exists
    ///
    /// Returns true if the engine contains a shape with the given ID.
    pub fn has_shape(&self, shape_id: &ShapeId) -> bool {
        self.shapes.contains_key(shape_id)
    }

    /// Validate that the shapes graph is well-formed according to SHACL specification
    ///
    /// This method validates the shapes themselves before they are used to validate data.
    /// It checks for structural integrity, correct SHACL syntax, valid targets,
    /// property paths, and constraint definitions.
    ///
    /// # Returns
    ///
    /// Returns a `ShapeValidationReport` containing any validation errors or warnings
    /// for the shapes graph. If the shapes graph is valid, the report will indicate
    /// conformance.
    ///
    /// # Example
    ///
    /// ```rust
    /// use oxirs_shacl::{ValidationEngine, ValidationConfig, Shape, ShapeId};
    /// use indexmap::IndexMap;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let shapes = IndexMap::new();
    /// let config = ValidationConfig::default();
    /// let engine = ValidationEngine::new(&shapes, config);
    ///
    /// // Validate that shapes are well-formed
    /// let shape_report = engine.validate_shapes_graph()?;
    /// if shape_report.is_valid() {
    ///     println!("Shapes graph is valid!");
    /// } else {
    ///     println!("Shape validation errors found:");
    ///     for error in shape_report.all_errors() {
    ///         println!("  - {}", error);
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn validate_shapes_graph(&self) -> Result<ShapeValidationReport> {
        let shapes_vec: Vec<Shape> = self.shapes.values().cloned().collect();
        self.shape_validator.validate_shapes(&shapes_vec)
    }

    /// Validate all data in a store against all loaded shapes
    ///
    /// This method first validates that the shapes graph is well-formed, then
    /// proceeds to validate the data against those shapes.
    ///
    /// For rapid validation cycles, consider using `validate_store_rapid()` instead.
    pub fn validate_store(&mut self, store: &dyn Store) -> Result<ValidationReport> {
        self.validate_store_internal(store, true)
    }

    /// Optimized validation for rapid validation cycles
    ///
    /// This method is designed for scenarios where the same shapes are validated
    /// repeatedly against potentially changing data. It skips redundant shape graph
    /// validation and uses aggressive caching for better performance.
    ///
    /// Use this method when:
    /// - Performing rapid validation cycles in real-time applications
    /// - Shapes don't change between validations
    /// - Performance is critical
    ///
    /// For the first call, use `prepare_for_rapid_validation()` to warm up caches.
    pub fn validate_store_rapid(&mut self, store: &dyn Store) -> Result<ValidationReport> {
        self.validate_store_internal(store, false)
    }

    /// Internal validation method with configurable shape validation
    ///
    /// Dispatches on [`Self::validation_strategy`] (see [`set_validation_strategy`](Self::set_validation_strategy)):
    /// each strategy performs a genuinely different validation execution path
    /// rather than silently falling back to one hard-coded behavior.
    fn validate_store_internal(
        &mut self,
        store: &dyn Store,
        validate_shapes: bool,
    ) -> Result<ValidationReport> {
        let start_time = Instant::now();

        // Only validate shapes graph if explicitly requested
        if validate_shapes {
            let shape_validation_report = self.validate_shapes_graph()?;
            if !shape_validation_report.is_valid() {
                return Err(ShaclError::ShapeValidation(format!(
                    "Shapes graph validation failed with {} error(s): {}",
                    shape_validation_report.error_count(),
                    shape_validation_report.all_errors().join("; ")
                )));
            }
        }

        // Collect active shapes for potential parallel/batched processing
        let active_shapes: Vec<&Shape> = self
            .shapes
            .values()
            .filter(|shape| shape.is_active())
            .collect();

        let strategy = self.validation_strategy.clone();

        let report = match strategy {
            ValidationStrategy::Sequential => {
                self.validate_shapes_sequential(store, &active_shapes)?
            }

            ValidationStrategy::Optimized => {
                if self.optimization_engine.is_some() && active_shapes.len() > 1 {
                    // Optimization enabled: use real parallel constraint
                    // evaluation (Rayon default thread count) instead of a
                    // sequential stand-in.
                    self.validate_shapes_engine_parallel(store, &active_shapes, 0)?
                } else {
                    self.validate_shapes_sequential(store, &active_shapes)?
                }
            }

            ValidationStrategy::Incremental {
                force_revalidate: _,
            } => {
                // `validate_store()` carries no prior-state / changed-node
                // information to diff against, so there is nothing to
                // incrementally skip here. Always perform a full, correct
                // revalidation (equivalent to `force_revalidate = true`)
                // rather than silently omitting nodes that might have
                // changed. Callers that need real incremental behavior
                // should use a changed-node-aware entry point instead.
                self.validate_shapes_sequential(store, &active_shapes)?
            }

            ValidationStrategy::Streaming { batch_size } => {
                self.validate_shapes_streaming(store, &active_shapes, batch_size.max(1))?
            }

            ValidationStrategy::Parallel { max_threads } => {
                self.validate_shapes_engine_parallel(store, &active_shapes, max_threads)?
            }
        };

        // Update statistics
        self.stats.total_validations += 1;
        self.stats.total_validation_time += start_time.elapsed();
        self.stats.last_validation_time = Some(start_time.elapsed());

        Ok(report)
    }

    /// Validate all active shapes sequentially (the baseline strategy every
    /// other strategy falls back to when it cannot do better).
    fn validate_shapes_sequential(
        &mut self,
        store: &dyn Store,
        shapes: &[&Shape],
    ) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        for shape in shapes {
            let shape_result = self.validate_shape(store, shape, None)?;
            report.merge_result(shape_result);

            if self.config.fail_fast && !report.conforms() {
                break;
            }

            if self.config.max_violations > 0
                && report.violation_count() >= self.config.max_violations
            {
                break;
            }
        }

        Ok(report)
    }

    /// Validate all active shapes in bounded-size batches (`ValidationStrategy::Streaming`).
    ///
    /// Unlike [`Self::validate_shapes_sequential`], this strategy bounds peak
    /// memory growth over a long validation run: the constraint-evaluation
    /// cache is cleared after every batch, trading some cache-hit rate for a
    /// cache size proportional to `batch_size` rather than to the total
    /// number of focus nodes across the whole store.
    fn validate_shapes_streaming(
        &mut self,
        store: &dyn Store,
        shapes: &[&Shape],
        batch_size: usize,
    ) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        'shapes: for shape in shapes {
            let target_nodes: Vec<Term> = if shape.targets.is_empty() && shape.is_node_shape() {
                let implicit_target =
                    Target::implicit(NamedNode::new(shape.id.as_str()).map_err(|e| {
                        ShaclError::TargetSelection(format!("Invalid shape IRI: {e}"))
                    })?);
                self.target_selector
                    .select_targets(store, &implicit_target, None)?
            } else {
                let mut nodes = Vec::new();
                for target in &shape.targets {
                    nodes.extend(self.target_selector.select_targets(store, target, None)?);
                }
                nodes
            };

            for batch in target_nodes.chunks(batch_size) {
                for node in batch {
                    let node_result = self.validate_node_against_shape(store, shape, node, None)?;
                    report.merge_result(node_result);

                    if self.should_stop_validation(&report) {
                        break 'shapes;
                    }
                }

                // Bound peak memory: this strategy explicitly trades
                // cache-hit rate for a cache footprint proportional to
                // `batch_size` rather than the whole store.
                self.constraint_cache.clear();
            }
        }

        Ok(report)
    }

    /// Validate all active shapes using real parallel constraint evaluation
    /// (`ValidationStrategy::Parallel`), via a Rayon thread pool.
    ///
    /// Target selection, property-path evaluation, and inherited-constraint
    /// resolution all require `&mut self` (they populate this engine's
    /// caches) and therefore run sequentially first to build a flat list of
    /// self-contained work items; the actual constraint *evaluation* — the
    /// expensive part — then runs concurrently across Rayon worker threads,
    /// since `Constraint::evaluate` only needs `&self` / `&dyn Store` (and
    /// `Store: Send + Sync`).
    fn validate_shapes_engine_parallel(
        &mut self,
        store: &dyn Store,
        shapes: &[&Shape],
        max_threads: usize,
    ) -> Result<ValidationReport> {
        let mut work_items = Vec::new();
        for shape in shapes {
            self.collect_shape_work_items(store, shape, None, &mut work_items)?;
        }

        let mut report = ValidationReport::new();
        if work_items.is_empty() {
            return Ok(report);
        }

        let violations = evaluate_work_items_parallel(store, &work_items, max_threads)?;

        // Note: because all constraint evaluation for this strategy already
        // happened concurrently above, `fail_fast` cannot abort in-flight
        // work; it is honored here as a post-hoc truncation of the
        // (already-fully-computed) violation list, same as `max_violations`.
        for violation in violations.into_iter().flatten() {
            report.add_violation(violation);

            if self.config.fail_fast && !report.conforms() {
                break;
            }
            if self.config.max_violations > 0
                && report.violation_count() >= self.config.max_violations
            {
                break;
            }
        }

        Ok(report)
    }

    /// Recursively collect self-contained constraint-evaluation work items for
    /// `shape` (and, for node shapes, any linked `sh:property` shapes),
    /// resolving targets/paths/inherited-constraints via `&mut self` so the
    /// resulting items can later be evaluated without further mutable access.
    fn collect_shape_work_items(
        &mut self,
        store: &dyn Store,
        shape: &Shape,
        focus_node_override: Option<&Term>,
        work_items: &mut Vec<ParallelWorkItem>,
    ) -> Result<()> {
        let resolved_constraints = self.resolve_inherited_constraints(&shape.id)?;
        let shapes_registry = Arc::new(self.shapes.clone());

        let focus_nodes: Vec<Term> = if let Some(node) = focus_node_override {
            vec![node.clone()]
        } else if shape.targets.is_empty() && shape.is_node_shape() {
            let implicit_target = Target::implicit(
                NamedNode::new(shape.id.as_str())
                    .map_err(|e| ShaclError::TargetSelection(format!("Invalid shape IRI: {e}")))?,
            );
            self.target_selector
                .select_targets(store, &implicit_target, None)?
        } else {
            let mut nodes = Vec::new();
            for target in &shape.targets {
                nodes.extend(self.target_selector.select_targets(store, target, None)?);
            }
            nodes
        };

        for focus_node in focus_nodes {
            if shape.is_node_shape() {
                let context = ConstraintContext::new(focus_node.clone(), shape.id.clone())
                    .with_values(vec![focus_node.clone()])
                    .with_shapes_registry(shapes_registry.clone());

                for (component_id, constraint) in &resolved_constraints {
                    work_items.push(ParallelWorkItem {
                        context: context.clone(),
                        component_id: component_id.clone(),
                        constraint: constraint.clone(),
                        shape_id: shape.id.clone(),
                        shape_severity: shape.severity,
                    });
                }

                for property_shape_id in &shape.property_shapes {
                    if let Some(property_shape) = self.shapes.get(property_shape_id).cloned() {
                        self.collect_shape_work_items(
                            store,
                            &property_shape,
                            Some(&focus_node),
                            work_items,
                        )?;
                    } else {
                        tracing::warn!(
                            "Property shape {} not found in shapes map",
                            property_shape_id.as_str()
                        );
                    }
                }
            } else if shape.is_property_shape() {
                let path = shape.path.as_ref().ok_or_else(|| {
                    ShaclError::ValidationEngine(
                        "Property shape must have a property path".to_string(),
                    )
                })?;
                let values = self
                    .path_evaluator
                    .evaluate_path(store, &focus_node, path, None)?;
                let context = ConstraintContext::new(focus_node.clone(), shape.id.clone())
                    .with_values(values)
                    .with_path(path.clone())
                    .with_shapes_registry(shapes_registry.clone());

                for (component_id, constraint) in &resolved_constraints {
                    work_items.push(ParallelWorkItem {
                        context: context.clone(),
                        component_id: component_id.clone(),
                        constraint: constraint.clone(),
                        shape_id: shape.id.clone(),
                        shape_severity: shape.severity,
                    });
                }
            }
        }

        Ok(())
    }

    /// Prepare the engine for rapid validation cycles
    ///
    /// This method pre-warms caches and validates the shapes graph once to optimize
    /// subsequent rapid validation calls.
    pub fn prepare_for_rapid_validation(&mut self, store: &dyn Store) -> Result<()> {
        // Validate shapes graph once and cache the result
        let _shape_validation_report = self.validate_shapes_graph()?;

        // Pre-warm constraint inheritance cache
        for shape_id in self.shapes.keys() {
            let _ = self.resolve_inherited_constraints(shape_id);
        }

        // Pre-warm target selection cache for common targets
        for shape in self.shapes.values() {
            for target in &shape.targets {
                let _ = self.target_selector.select_targets(store, target, None);
            }
        }

        Ok(())
    }

    /// Validate specific nodes against a specific shape
    pub fn validate_nodes(
        &mut self,
        store: &dyn Store,
        shape: &Shape,
        nodes: &[Term],
    ) -> Result<ValidationReport> {
        let start_time = Instant::now();
        let mut report = ValidationReport::new();

        for node in nodes {
            let node_result = self.validate_node_against_shape(store, shape, node, None)?;
            report.merge_result(node_result);

            // Check early termination conditions
            if self.config.fail_fast && !report.conforms() {
                break;
            }

            if self.config.max_violations > 0
                && report.violation_count() >= self.config.max_violations
            {
                break;
            }
        }

        // Update statistics
        self.stats.total_node_validations += nodes.len();
        self.stats.total_validation_time += start_time.elapsed();

        Ok(report)
    }

    /// Enable optimization with default configuration
    pub fn enable_optimization(&mut self) {
        self.enable_optimization_with_config(OptimizationConfig::default());
    }

    /// Enable optimization with custom configuration
    pub fn enable_optimization_with_config(&mut self, optimization_config: OptimizationConfig) {
        self.optimization_engine = Some(OptimizedValidationEngine::new(
            self.config.clone(),
            optimization_config,
        ));
        self.validation_strategy = ValidationStrategy::Optimized;
    }

    /// Disable optimization (use basic validation)
    pub fn disable_optimization(&mut self) {
        self.optimization_engine = None;
        self.validation_strategy = ValidationStrategy::Sequential;
    }

    /// Set validation strategy
    pub fn set_validation_strategy(&mut self, strategy: ValidationStrategy) {
        self.validation_strategy = strategy;
    }

    /// Get current validation strategy
    pub fn get_validation_strategy(&self) -> &ValidationStrategy {
        &self.validation_strategy
    }

    /// Check if optimization is enabled
    pub fn is_optimization_enabled(&self) -> bool {
        self.optimization_engine.is_some()
    }

    /// Validate a shape against its targets
    fn validate_shape(
        &mut self,
        store: &dyn Store,
        shape: &Shape,
        graph_name: Option<&str>,
    ) -> Result<ValidationReport> {
        tracing::trace!(
            "validating shape {} (type: {:?}) with {} targets",
            shape.id.as_str(),
            shape.shape_type,
            shape.targets.len()
        );
        let mut report = ValidationReport::new();

        // If no explicit targets, this might be an implicit target shape
        if shape.targets.is_empty() && shape.is_node_shape() {
            // Try using the shape IRI as an implicit class target
            let implicit_target = Target::implicit(
                NamedNode::new(shape.id.as_str())
                    .map_err(|e| ShaclError::TargetSelection(format!("Invalid shape IRI: {e}")))?,
            );
            let target_nodes =
                self.target_selector
                    .select_targets(store, &implicit_target, graph_name)?;

            for node in target_nodes {
                let node_result =
                    self.validate_node_against_shape(store, shape, &node, graph_name)?;
                report.merge_result(node_result);

                if self.should_stop_validation(&report) {
                    break;
                }
            }
        } else {
            // Validate against explicit targets
            for target in &shape.targets {
                tracing::trace!("processing target {target:?}");
                let target_nodes = self
                    .target_selector
                    .select_targets(store, target, graph_name)?;
                tracing::trace!("found {} target nodes", target_nodes.len());

                for (i, node) in target_nodes.iter().enumerate() {
                    tracing::trace!("validating target node[{i}] = {node:?}");
                    let node_result =
                        self.validate_node_against_shape(store, shape, node, graph_name)?;
                    tracing::trace!(
                        "node[{}] result: conforms={}, violations={}",
                        i,
                        node_result.conforms(),
                        node_result.violation_count()
                    );
                    report.merge_result(node_result);

                    if self.should_stop_validation(&report) {
                        break;
                    }
                }

                if self.should_stop_validation(&report) {
                    break;
                }
            }
        }

        Ok(report)
    }

    /// Validate a specific node against a shape
    pub fn validate_node_against_shape(
        &mut self,
        store: &dyn Store,
        shape: &Shape,
        focus_node: &Term,
        graph_name: Option<&str>,
    ) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();

        if shape.is_node_shape() {
            // For node shapes, validate constraints directly against the focus node
            let values = vec![focus_node.clone()];
            let constraint_results =
                self.validate_constraints(store, shape, focus_node, None, &values, graph_name)?;

            for violation in constraint_results.into_iter().flatten() {
                report.add_violation(violation);
            }

            // Also validate any property shapes linked via sh:property
            for property_shape_id in &shape.property_shapes {
                if let Some(property_shape) = self.shapes.get(property_shape_id) {
                    // Clone the property shape to avoid borrow issues
                    let property_shape = property_shape.clone();
                    let property_report = self.validate_node_against_shape(
                        store,
                        &property_shape,
                        focus_node,
                        graph_name,
                    )?;
                    report.merge_result(property_report);
                } else {
                    tracing::warn!(
                        "Property shape {} not found in shapes map",
                        property_shape_id.as_str()
                    );
                }
            }
        } else if shape.is_property_shape() {
            // For property shapes, evaluate the property path first
            if let Some(path) = &shape.path {
                let values = self
                    .path_evaluator
                    .evaluate_path(store, focus_node, path, graph_name)?;
                let constraint_results = self.validate_constraints(
                    store,
                    shape,
                    focus_node,
                    Some(path),
                    &values,
                    graph_name,
                )?;

                for violation in constraint_results.into_iter().flatten() {
                    report.add_violation(violation);
                }
            } else {
                return Err(ShaclError::ValidationEngine(
                    "Property shape must have a property path".to_string(),
                ));
            }
        }

        Ok(report)
    }

    /// Check if validation should stop early
    fn should_stop_validation(&self, report: &ValidationReport) -> bool {
        // Stop if fail_fast is enabled and we have violations
        if self.config.fail_fast && !report.conforms() {
            return true;
        }

        // Stop if we've reached the maximum violation limit
        if self.config.max_violations > 0 && report.violation_count() >= self.config.max_violations
        {
            return true;
        }

        false
    }

    /// Validate all constraints for a shape
    fn validate_constraints(
        &mut self,
        store: &dyn Store,
        shape: &Shape,
        focus_node: &Term,
        path: Option<&PropertyPath>,
        values: &[Term],
        graph_name: Option<&str>,
    ) -> Result<Vec<Option<ValidationViolation>>> {
        let mut results = Vec::new();

        // Resolve inherited constraints (includes shape's own constraints)
        let resolved_constraints = self.resolve_inherited_constraints(&shape.id)?;

        for (component_id, constraint) in &resolved_constraints {
            tracing::trace!(
                "validate_constraints: evaluating constraint {} for shape {}",
                component_id.as_str(),
                shape.id.as_str()
            );
            tracing::trace!("validate_constraints: constraint = {constraint:?}");
            tracing::trace!(
                "validate_constraints: focus_node = {focus_node:?}, values = {values:?}"
            );

            // Create constraint context
            let context = ConstraintContext::new(focus_node.clone(), shape.id.clone())
                .with_values(values.to_vec())
                .with_shapes_registry(Arc::new(self.shapes.clone()));

            let constraint_result =
                self.validate_constraint(store, constraint, &context, path, graph_name)?;
            tracing::trace!("validate_constraints: constraint result = {constraint_result:?}");

            match constraint_result {
                ConstraintEvaluationResult::Satisfied => {
                    results.push(None);
                }
                ConstraintEvaluationResult::SatisfiedWithNote { note: _ } => {
                    // Constraint is satisfied but has a note - treat as satisfied
                    results.push(None);
                }
                ConstraintEvaluationResult::Error { message } => {
                    // Fail-loud contract: a constraint that could not be evaluated
                    // (e.g. an unrecognized/misconfigured custom constraint
                    // component) must never be silently treated as satisfied,
                    // or a real violation could go undetected. Surface it as a
                    // hard error instead, matching error_recovery.rs's handling
                    // of the same variant.
                    return Err(ShaclError::ValidationEngine(format!(
                        "Constraint evaluation error for component '{}' on shape '{}': {}",
                        component_id.as_str(),
                        shape.id.as_str(),
                        message
                    )));
                }
                ConstraintEvaluationResult::Violated {
                    violating_value,
                    message,
                } => {
                    let violation = ValidationViolation::new(
                        focus_node.clone(),
                        shape.id.clone(),
                        component_id.clone(),
                        constraint.severity().unwrap_or(shape.severity),
                    )
                    .with_value(violating_value.unwrap_or_else(|| focus_node.clone()))
                    .with_message(message.unwrap_or_else(|| {
                        format!("Constraint {} violated", component_id.as_str())
                    }));

                    results.push(Some(violation));
                }
            }
        }

        Ok(results)
    }

    /// Resolve inherited constraints for a shape
    pub fn resolve_inherited_constraints(
        &mut self,
        shape_id: &ShapeId,
    ) -> Result<IndexMap<ConstraintComponentId, Constraint>> {
        // Check cache first
        if let Some(cached) = self.inheritance_cache.get(shape_id) {
            return Ok(cached);
        }

        // Implement proper inheritance resolution
        let mut constraints = IndexMap::new();
        let mut visited = std::collections::HashSet::new();

        // Recursively collect constraints from the inheritance chain
        self.collect_inherited_constraints(shape_id, &mut constraints, &mut visited)?;

        // Cache the result
        self.inheritance_cache
            .insert(shape_id.clone(), constraints.clone());

        Ok(constraints)
    }

    /// Recursively collect constraints from a shape and its parents
    fn collect_inherited_constraints(
        &self,
        shape_id: &ShapeId,
        constraints: &mut IndexMap<ConstraintComponentId, Constraint>,
        visited: &mut std::collections::HashSet<ShapeId>,
    ) -> Result<()> {
        // Prevent circular inheritance
        if visited.contains(shape_id) {
            return Ok(());
        }
        visited.insert(shape_id.clone());

        if let Some(shape) = self.shapes.get(shape_id) {
            // First collect constraints from parent shapes (lowest priority)
            for parent_id in &shape.extends {
                self.collect_inherited_constraints(parent_id, constraints, visited)?;
            }

            // Then add this shape's own constraints (higher priority - can override parent constraints)
            for (constraint_id, constraint) in &shape.constraints {
                constraints.insert(constraint_id.clone(), constraint.clone());
            }
        }

        Ok(())
    }

    /// Validate a single constraint
    fn validate_constraint(
        &mut self,
        store: &dyn Store,
        constraint: &Constraint,
        context: &ConstraintContext,
        path: Option<&PropertyPath>,
        _graph_name: Option<&str>,
    ) -> Result<ConstraintEvaluationResult> {
        // Check cache first
        let cache_key = ConstraintCacheKey {
            focus_node: context.focus_node.clone(),
            shape_id: context.shape_id.clone(),
            constraint_component_id: constraint.component_id(),
            property_path: path.cloned(),
        };

        // Check if we have a cached result
        if let Some(cached_result) = self.constraint_cache.get(&cache_key) {
            self.stats.record_cache_hit();
            return Ok(cached_result);
        }
        self.stats.record_cache_miss();

        // Evaluate the constraint using its own evaluate method
        let constraint_result = constraint.evaluate(store, context)?;

        // Convert from constraints::ConstraintEvaluationResult to validation::ConstraintEvaluationResult
        let result = convert_constraint_result(constraint_result);

        // Do not cache constraint evaluation errors: they may stem from
        // transient conditions (e.g. a store query failure) and must not
        // silently poison subsequent identical lookups with a stale "error"
        // verdict.
        if !matches!(result, ConstraintEvaluationResult::Error { .. }) {
            self.constraint_cache.insert(cache_key, result.clone());
        }

        Ok(result)
    }
}

/// Convert a `constraints::constraint_context::ConstraintEvaluationResult`
/// (the result type returned by `Constraint::evaluate`) into this module's
/// `ConstraintEvaluationResult`.
///
/// Error results are preserved as `Error` — never silently downgraded to
/// `SatisfiedWithNote` — so callers can react to evaluation failures per the
/// project's fail-loud contract instead of treating an unevaluated
/// constraint as a pass.
fn convert_constraint_result(
    result: crate::constraints::constraint_context::ConstraintEvaluationResult,
) -> ConstraintEvaluationResult {
    match result {
        crate::constraints::constraint_context::ConstraintEvaluationResult::Satisfied => {
            ConstraintEvaluationResult::satisfied()
        }
        crate::constraints::constraint_context::ConstraintEvaluationResult::Violated {
            violating_value,
            message,
            details: _,
        } => ConstraintEvaluationResult::violated(violating_value, message),
        crate::constraints::constraint_context::ConstraintEvaluationResult::Error {
            message,
            details: _,
        } => ConstraintEvaluationResult::Error { message },
    }
}

/// A single, self-contained unit of constraint-evaluation work built by
/// [`ValidationEngine::collect_shape_work_items`] for `ValidationStrategy::Parallel`.
///
/// Everything needed to evaluate one `(focus node, constraint)` pair is
/// captured by value so this can be evaluated from any Rayon worker thread
/// without further access to the (non-`Sync`) `ValidationEngine`.
#[derive(Debug, Clone)]
struct ParallelWorkItem {
    context: ConstraintContext,
    component_id: ConstraintComponentId,
    constraint: Constraint,
    shape_id: ShapeId,
    shape_severity: crate::Severity,
}

/// Evaluate every [`ParallelWorkItem`] concurrently across a Rayon thread pool
/// (Rayon's default pool when `max_threads == 0`, otherwise a dedicated pool
/// sized to `max_threads`), returning one `Option<ValidationViolation>` per
/// item in the same order as `work_items`.
///
/// Any constraint evaluation error is propagated as a hard `Err` (fail-loud
/// contract): an unevaluated constraint must never be silently treated as
/// satisfied just because it happened to run on a background thread.
fn evaluate_work_items_parallel(
    store: &dyn Store,
    work_items: &[ParallelWorkItem],
    max_threads: usize,
) -> Result<Vec<Option<ValidationViolation>>> {
    use rayon::prelude::*;

    let eval_one = |item: &ParallelWorkItem| -> Result<Option<ValidationViolation>> {
        let result = item.constraint.evaluate(store, &item.context)?;
        match result {
            crate::constraints::constraint_context::ConstraintEvaluationResult::Satisfied => {
                Ok(None)
            }
            crate::constraints::constraint_context::ConstraintEvaluationResult::Error {
                message,
                details: _,
            } => Err(ShaclError::ValidationEngine(format!(
                "Constraint evaluation error for component '{}' on shape '{}': {}",
                item.component_id.as_str(),
                item.shape_id.as_str(),
                message
            ))),
            crate::constraints::constraint_context::ConstraintEvaluationResult::Violated {
                violating_value,
                message,
                details: _,
            } => {
                let severity = item.constraint.severity().unwrap_or(item.shape_severity);
                let violation = ValidationViolation::new(
                    item.context.focus_node.clone(),
                    item.shape_id.clone(),
                    item.component_id.clone(),
                    severity,
                )
                .with_value(violating_value.unwrap_or_else(|| item.context.focus_node.clone()))
                .with_message(message.unwrap_or_else(|| {
                    format!("Constraint {} violated", item.component_id.as_str())
                }));
                Ok(Some(violation))
            }
        }
    };

    if max_threads > 0 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(max_threads)
            .build()
            .map_err(|e| {
                ShaclError::ValidationEngine(format!("Failed to build Rayon thread pool: {e}"))
            })?;
        pool.install(|| work_items.par_iter().map(eval_one).collect())
    } else {
        work_items.par_iter().map(eval_one).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_convert_constraint_result_preserves_error() {
        // A constraint evaluation failure (e.g. an unrecognized/misconfigured
        // custom constraint component) must be preserved as `Error`, never
        // silently downgraded to `SatisfiedWithNote` (which callers treat as
        // a pass, hiding the fact that the constraint was never evaluated).
        let raw = crate::constraints::constraint_context::ConstraintEvaluationResult::error(
            "unrecognized custom constraint component 'ex:RegxConstraintComponent'".to_string(),
        );

        let converted = convert_constraint_result(raw);

        assert!(
            matches!(converted, ConstraintEvaluationResult::Error { .. }),
            "an evaluation error must convert to Error, not SatisfiedWithNote"
        );
        assert!(
            !converted.is_satisfied(),
            "an evaluation error must never report as satisfied"
        );
    }

    #[test]
    fn regression_convert_constraint_result_preserves_satisfied_and_violated() {
        let satisfied = convert_constraint_result(
            crate::constraints::constraint_context::ConstraintEvaluationResult::Satisfied,
        );
        assert!(satisfied.is_satisfied());

        let violated = convert_constraint_result(
            crate::constraints::constraint_context::ConstraintEvaluationResult::Violated {
                violating_value: None,
                message: Some("bad value".to_string()),
                details: std::collections::HashMap::new(),
            },
        );
        assert!(violated.is_violated());
    }
}
