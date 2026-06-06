//! SHACL Optimization Strategies — Dependency Analysis, Streaming, and Incremental Validation
//!
//! Provides:
//! - `ConstraintDependencyAnalyzer` — cost/selectivity model for constraint ordering
//! - `AdvancedConstraintEvaluator` — ordered evaluation with early-termination
//! - `StreamingValidationEngine` — streaming validation for large datasets
//! - `IncrementalValidationEngine` — change-aware incremental validation
//! - All supporting change-detection types

use crate::{
    constraints::{Constraint, ConstraintContext, ConstraintEvaluationResult},
    Result, ShaclError, ShapeId,
};
use oxirs_core::{model::Term, RdfTerm, Store};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::core_engine::{BatchConstraintEvaluator, ConstraintCache};

// ─── ConstraintDependencyAnalyzer ────────────────────────────────────────────

/// Analyzes constraint dependencies and estimates cost/selectivity for optimal ordering.
#[derive(Debug)]
pub struct ConstraintDependencyAnalyzer {
    cost_estimates: HashMap<String, f64>,
}

impl Default for ConstraintDependencyAnalyzer {
    fn default() -> Self {
        let mut cost_estimates = HashMap::new();

        cost_estimates.insert("class".to_string(), 5.0);
        cost_estimates.insert("datatype".to_string(), 1.0);
        cost_estimates.insert("nodeKind".to_string(), 1.0);
        cost_estimates.insert("minCount".to_string(), 1.0);
        cost_estimates.insert("maxCount".to_string(), 1.0);
        cost_estimates.insert("pattern".to_string(), 3.0);
        cost_estimates.insert("sparql".to_string(), 10.0);
        cost_estimates.insert("qualifiedValueShape".to_string(), 8.0);
        cost_estimates.insert("closed".to_string(), 6.0);

        Self { cost_estimates }
    }
}

impl ConstraintDependencyAnalyzer {
    /// Reorder constraints for optimal evaluation (most selective first, then cheapest).
    pub fn optimize_constraint_order(&self, constraints: Vec<Constraint>) -> Vec<Constraint> {
        let mut constraint_info: Vec<_> = constraints
            .into_iter()
            .map(|c| {
                let cost = self.estimate_constraint_cost(&c);
                let selectivity = self.estimate_constraint_selectivity(&c);
                (c, selectivity, cost)
            })
            .collect();

        constraint_info.sort_by(|a, b| {
            let selectivity_cmp = a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal);
            if selectivity_cmp == std::cmp::Ordering::Equal {
                a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                selectivity_cmp
            }
        });

        constraint_info.into_iter().map(|(c, _, _)| c).collect()
    }

    /// Estimate the relative computational cost of evaluating a constraint.
    pub fn estimate_constraint_cost(&self, constraint: &Constraint) -> f64 {
        match constraint {
            Constraint::Class(_) => self.cost_estimates.get("class").copied().unwrap_or(5.0),
            Constraint::Datatype(_) => self.cost_estimates.get("datatype").copied().unwrap_or(1.0),
            Constraint::NodeKind(_) => self.cost_estimates.get("nodeKind").copied().unwrap_or(1.0),
            Constraint::MinCount(_) => self.cost_estimates.get("minCount").copied().unwrap_or(1.0),
            Constraint::MaxCount(_) => self.cost_estimates.get("maxCount").copied().unwrap_or(1.0),
            Constraint::Pattern(_) => self.cost_estimates.get("pattern").copied().unwrap_or(3.0),
            Constraint::Sparql(_) => self.cost_estimates.get("sparql").copied().unwrap_or(10.0),
            Constraint::QualifiedValueShape(_) => self
                .cost_estimates
                .get("qualifiedValueShape")
                .copied()
                .unwrap_or(8.0),
            Constraint::Closed(_) => self.cost_estimates.get("closed").copied().unwrap_or(6.0),
            Constraint::And(_) | Constraint::Or(_) | Constraint::Xone(_) => 7.0,
            _ => 3.0,
        }
    }

    /// Estimate how selective a constraint is (lower = more selective = fewer results pass).
    pub fn estimate_constraint_selectivity(&self, constraint: &Constraint) -> f64 {
        match constraint {
            Constraint::Class(_) => 0.8,
            Constraint::Datatype(_) => 0.6,
            Constraint::NodeKind(_) => 0.3,
            Constraint::HasValue(_) => 0.05,
            Constraint::In(_) => 0.15,
            Constraint::MinCount(_) | Constraint::MaxCount(_) => 0.1,
            Constraint::Pattern(_) => 0.5,
            Constraint::MinLength(_) | Constraint::MaxLength(_) => 0.6,
            Constraint::MinInclusive(_) | Constraint::MaxInclusive(_) => 0.7,
            Constraint::MinExclusive(_) | Constraint::MaxExclusive(_) => 0.7,
            Constraint::Sparql(_) => 0.8,
            Constraint::QualifiedValueShape(_) => 0.6,
            Constraint::Closed(_) => 0.4,
            Constraint::And(_) => 0.3,
            Constraint::Or(_) => 0.8,
            Constraint::Xone(_) => 0.5,
            Constraint::Not(_) => 0.9,
            _ => 0.5,
        }
    }

    /// Update cost estimate for a constraint type using exponential moving average.
    pub fn update_cost_estimate(&mut self, constraint_type: &str, actual_cost: f64) {
        let alpha = 0.1;
        let current_estimate = self
            .cost_estimates
            .get(constraint_type)
            .copied()
            .unwrap_or(3.0);
        let new_estimate = alpha * actual_cost + (1.0 - alpha) * current_estimate;
        self.cost_estimates
            .insert(constraint_type.to_string(), new_estimate);
    }
}

// ─── ConstraintPerformanceStats ──────────────────────────────────────────────

/// Performance statistics for constraint evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintPerformanceStats {
    pub cache_hit_rate: f64,
    pub total_evaluations: usize,
    pub avg_evaluation_time_us: f64,
    pub cache_evictions: usize,
}

// ─── AdvancedConstraintEvaluator ─────────────────────────────────────────────

/// Advanced constraint evaluation orchestrator with optional early termination.
#[derive(Debug)]
pub struct AdvancedConstraintEvaluator {
    pub(crate) batch_evaluator: BatchConstraintEvaluator,
    dependency_analyzer: ConstraintDependencyAnalyzer,
    enable_early_termination: bool,
}

impl Default for AdvancedConstraintEvaluator {
    fn default() -> Self {
        Self {
            batch_evaluator: BatchConstraintEvaluator::default(),
            dependency_analyzer: ConstraintDependencyAnalyzer::default(),
            enable_early_termination: true,
        }
    }
}

impl AdvancedConstraintEvaluator {
    /// Create new advanced evaluator with custom configuration.
    pub fn new(
        cache: ConstraintCache,
        parallel: bool,
        batch_size: usize,
        early_termination: bool,
    ) -> Self {
        Self {
            batch_evaluator: BatchConstraintEvaluator::new(cache, parallel, batch_size),
            dependency_analyzer: ConstraintDependencyAnalyzer::default(),
            enable_early_termination: early_termination,
        }
    }

    /// Evaluate constraints with dependency-ordered execution and optional early termination.
    pub fn evaluate_optimized(
        &self,
        store: &dyn Store,
        constraints: Vec<Constraint>,
        context: ConstraintContext,
    ) -> Result<Vec<ConstraintEvaluationResult>> {
        let optimized_constraints = self
            .dependency_analyzer
            .optimize_constraint_order(constraints);

        let constraints_with_contexts: Vec<_> = optimized_constraints
            .into_iter()
            .map(|c| (c, context.clone()))
            .collect();

        if self.enable_early_termination {
            let mut results = Vec::new();

            for (constraint, ctx) in constraints_with_contexts {
                let result = if let Some(cached) = self.batch_evaluator.cache.get(&constraint, &ctx)
                {
                    cached
                } else {
                    let start_time = Instant::now();
                    let result = constraint.evaluate(store, &ctx)?;
                    let evaluation_time = start_time.elapsed();
                    self.batch_evaluator.cache.put(
                        &constraint,
                        &ctx,
                        result.clone(),
                        evaluation_time,
                    );
                    result
                };

                results.push(result.clone());

                // Early termination marker; full evaluation continues for now
                if result.is_violated() {
                    // Reserved for future use: break on first violation if policy demands it
                }
            }

            Ok(results)
        } else {
            self.batch_evaluator
                .evaluate_batch(store, constraints_with_contexts)
        }
    }

    /// Get performance statistics from the underlying cache.
    pub fn get_performance_stats(&self) -> ConstraintPerformanceStats {
        let cache_stats = self.batch_evaluator.cache_stats();
        ConstraintPerformanceStats {
            cache_hit_rate: cache_stats.hit_rate(),
            total_evaluations: cache_stats.evaluations,
            avg_evaluation_time_us: cache_stats.avg_evaluation_time_us,
            cache_evictions: cache_stats.evictions,
        }
    }
}

// ─── StreamingValidationEngine ───────────────────────────────────────────────

/// Streaming validation engine for large datasets
#[derive(Debug)]
pub struct StreamingValidationEngine {
    batch_size: usize,
    /// Memory threshold in bytes; reserved for future spill-to-disk implementation
    #[allow(dead_code)]
    memory_threshold: usize,
    memory_monitoring: bool,
    evaluator: AdvancedConstraintEvaluator,
}

impl Default for StreamingValidationEngine {
    fn default() -> Self {
        Self::new(1000, 100 * 1024 * 1024, true)
    }
}

impl StreamingValidationEngine {
    /// Create a new streaming validation engine.
    pub fn new(batch_size: usize, memory_threshold: usize, memory_monitoring: bool) -> Self {
        let cache = ConstraintCache::new(10000, Duration::from_secs(300));
        let evaluator = AdvancedConstraintEvaluator::new(cache, true, batch_size / 4, true);

        Self {
            batch_size,
            memory_threshold,
            memory_monitoring,
            evaluator,
        }
    }

    /// Validate a large dataset in streaming fashion using the given node iterator.
    pub fn validate_streaming<I>(
        &self,
        store: &dyn Store,
        constraints: Vec<Constraint>,
        node_stream: I,
    ) -> Result<StreamingValidationResult>
    where
        I: Iterator<Item = Term>,
    {
        let mut result = StreamingValidationResult::new();
        let mut current_batch = Vec::new();
        let mut processed_count = 0;

        for node in node_stream {
            current_batch.push(node);

            if current_batch.len() >= self.batch_size {
                let batch_result = self.process_batch(store, &constraints, &current_batch)?;
                result.merge_batch_result(batch_result);

                processed_count += current_batch.len();
                current_batch.clear();

                if self.memory_monitoring && self.check_memory_pressure()? {
                    result.memory_pressure_events += 1;
                    self.evaluator.batch_evaluator.cache.clear();
                    tracing::warn!("Memory pressure detected, cleared cache");
                }

                if processed_count % (self.batch_size * 10) == 0 {
                    tracing::info!("Processed {} nodes", processed_count);
                }
            }
        }

        if !current_batch.is_empty() {
            let batch_result = self.process_batch(store, &constraints, &current_batch)?;
            result.merge_batch_result(batch_result);
        }

        result.total_nodes = processed_count + current_batch.len();
        Ok(result)
    }

    fn process_batch(
        &self,
        store: &dyn Store,
        constraints: &[Constraint],
        nodes: &[Term],
    ) -> Result<BatchValidationResult> {
        let mut batch_result = BatchValidationResult::new();
        let start_time = Instant::now();

        for node in nodes {
            let context = ConstraintContext::new(node.clone(), ShapeId::new("BatchValidation"));

            let constraint_results =
                self.evaluator
                    .evaluate_optimized(store, constraints.to_vec(), context)?;

            let violations = constraint_results
                .iter()
                .filter(|r| r.is_violated())
                .count();
            batch_result.violation_count += violations;
            batch_result.node_count += 1;
        }

        batch_result.processing_time = start_time.elapsed();
        Ok(batch_result)
    }

    fn check_memory_pressure(&self) -> Result<bool> {
        if !self.memory_monitoring {
            return Ok(false);
        }

        let stats = self.evaluator.get_performance_stats();
        Ok(stats.cache_evictions > 100 && stats.cache_hit_rate < 0.5)
    }
}

/// Result of streaming validation over an entire node stream
#[derive(Debug, Clone)]
pub struct StreamingValidationResult {
    pub total_nodes: usize,
    pub total_violations: usize,
    pub total_processing_time: Duration,
    pub memory_pressure_events: usize,
    pub batches_processed: usize,
}

impl StreamingValidationResult {
    fn new() -> Self {
        Self {
            total_nodes: 0,
            total_violations: 0,
            total_processing_time: Duration::ZERO,
            memory_pressure_events: 0,
            batches_processed: 0,
        }
    }

    fn merge_batch_result(&mut self, batch: BatchValidationResult) {
        self.total_violations += batch.violation_count;
        self.total_processing_time += batch.processing_time;
        self.batches_processed += 1;
    }
}

/// Result of processing a single batch of nodes
#[derive(Debug, Clone)]
struct BatchValidationResult {
    pub node_count: usize,
    pub violation_count: usize,
    pub processing_time: Duration,
}

impl BatchValidationResult {
    fn new() -> Self {
        Self {
            node_count: 0,
            violation_count: 0,
            processing_time: Duration::ZERO,
        }
    }
}

// ─── IncrementalValidationEngine ─────────────────────────────────────────────

/// Validation snapshot for incremental processing
#[derive(Debug, Clone)]
struct ValidationSnapshot {
    /// Stored for future validation-history APIs
    #[allow(dead_code)]
    node: Term,
    constraints_hash: u64,
    properties_hash: u64,
    /// Stored for future diff-based violation reporting
    #[allow(dead_code)]
    result: Vec<ConstraintEvaluationResult>,
    /// Stored for cache expiry and staleness detection
    #[allow(dead_code)]
    validated_at: Instant,
}

/// Level of change detection granularity for incremental validation
#[derive(Debug, Clone)]
pub enum ChangeDetectionLevel {
    NodeOnly,
    Properties,
    SubGraph,
}

/// Incremental validation engine — re-validates only changed nodes.
#[derive(Debug)]
pub struct IncrementalValidationEngine {
    previous_results: Arc<RwLock<HashMap<Term, ValidationSnapshot>>>,
    evaluator: AdvancedConstraintEvaluator,
    /// Reserved for SubGraph-level change detection in future versions
    #[allow(dead_code)]
    change_detection_level: ChangeDetectionLevel,
}

impl Default for IncrementalValidationEngine {
    fn default() -> Self {
        let cache = ConstraintCache::new(50000, Duration::from_secs(3600));
        let evaluator = AdvancedConstraintEvaluator::new(cache, true, 100, false);

        Self {
            previous_results: Arc::new(RwLock::new(HashMap::new())),
            evaluator,
            change_detection_level: ChangeDetectionLevel::Properties,
        }
    }
}

impl IncrementalValidationEngine {
    /// Validate only changed nodes since last validation run.
    pub fn validate_incremental(
        &mut self,
        store: &dyn Store,
        constraints: Vec<Constraint>,
        nodes: &[Term],
        force_revalidate: bool,
    ) -> Result<IncrementalValidationResult> {
        let mut result = IncrementalValidationResult::new();
        let start_time = Instant::now();

        let constraints_hash = self.hash_constraints(&constraints);

        for node in nodes {
            let properties_hash = self.hash_node_properties(store, node)?;

            let needs_validation = force_revalidate || {
                let previous_results = self
                    .previous_results
                    .read()
                    .expect("read lock should not be poisoned");
                match previous_results.get(node) {
                    Some(snapshot) => {
                        snapshot.constraints_hash != constraints_hash
                            || snapshot.properties_hash != properties_hash
                    }
                    None => true,
                }
            };

            if needs_validation {
                let context =
                    ConstraintContext::new(node.clone(), ShapeId::new("IncrementalValidation"));

                let constraint_results =
                    self.evaluator
                        .evaluate_optimized(store, constraints.clone(), context)?;

                let snapshot = ValidationSnapshot {
                    node: node.clone(),
                    constraints_hash,
                    properties_hash,
                    result: constraint_results.clone(),
                    validated_at: Instant::now(),
                };

                {
                    let mut previous_results = self
                        .previous_results
                        .write()
                        .expect("write lock should not be poisoned");
                    previous_results.insert(node.clone(), snapshot);
                }

                let violations = constraint_results
                    .iter()
                    .filter(|r| r.is_violated())
                    .count();
                result.revalidated_nodes += 1;
                result.new_violations += violations;
            } else {
                result.skipped_nodes += 1;
            }
        }

        result.total_processing_time = start_time.elapsed();
        Ok(result)
    }

    fn hash_constraints(&self, constraints: &[Constraint]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        for constraint in constraints {
            format!("{constraint:?}").hash(&mut hasher);
        }
        hasher.finish()
    }

    fn hash_node_properties(&self, store: &dyn Store, node: &Term) -> Result<u64> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hash;
        let mut hasher = DefaultHasher::new();

        match self.query_node_triples_as_subject(store, node) {
            Ok(subject_triples) => {
                for triple in subject_triples {
                    triple.subject().as_str().hash(&mut hasher);
                    triple.predicate().as_str().hash(&mut hasher);
                    triple.object().as_str().hash(&mut hasher);
                }
            }
            Err(_) => {
                node.as_str().hash(&mut hasher);
            }
        }

        if let Ok(object_triples) = self.query_node_triples_as_object(store, node) {
            for triple in object_triples {
                triple.subject().as_str().hash(&mut hasher);
                triple.predicate().as_str().hash(&mut hasher);
                triple.object().as_str().hash(&mut hasher);
            }
        }

        Ok(hasher.finish())
    }

    fn query_node_triples_as_subject(
        &self,
        store: &dyn Store,
        node: &Term,
    ) -> Result<Vec<oxirs_core::model::Triple>> {
        let mut triples = Vec::new();

        let subject = match node {
            Term::NamedNode(nn) => Some(oxirs_core::model::Subject::NamedNode(nn.clone())),
            Term::BlankNode(bn) => Some(oxirs_core::model::Subject::BlankNode(bn.clone())),
            Term::Variable(v) => Some(oxirs_core::model::Subject::Variable(v.clone())),
            _ => None,
        };
        let quads = match subject {
            Some(s) => store.find_quads(Some(&s), None, None, None)?,
            None => Vec::new(),
        };
        for quad in quads {
            let triple = oxirs_core::model::Triple::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
            );
            triples.push(triple);
        }

        Ok(triples)
    }

    fn query_node_triples_as_object(
        &self,
        store: &dyn Store,
        node: &Term,
    ) -> Result<Vec<oxirs_core::model::Triple>> {
        let mut triples = Vec::new();

        let object = match node {
            Term::NamedNode(nn) => Some(oxirs_core::model::Object::NamedNode(nn.clone())),
            Term::BlankNode(bn) => Some(oxirs_core::model::Object::BlankNode(bn.clone())),
            Term::Literal(lit) => Some(oxirs_core::model::Object::Literal(lit.clone())),
            Term::Variable(v) => Some(oxirs_core::model::Object::Variable(v.clone())),
            _ => None,
        };
        let quads = match object {
            Some(o) => store.find_quads(None, None, Some(&o), None)?,
            None => Vec::new(),
        };
        for quad in quads {
            let triple = oxirs_core::model::Triple::new(
                quad.subject().clone(),
                quad.predicate().clone(),
                quad.object().clone(),
            );
            triples.push(triple);
        }

        Ok(triples)
    }

    /// Clear all stored validation history.
    pub fn clear_history(&mut self) {
        self.previous_results
            .write()
            .expect("write lock should not be poisoned")
            .clear();
    }

    /// Return statistics about the number of cached validations.
    pub fn get_incremental_stats(&self) -> IncrementalValidationStats {
        let snapshots = self
            .previous_results
            .read()
            .expect("read lock should not be poisoned");
        IncrementalValidationStats {
            cached_validations: snapshots.len(),
            memory_usage_mb: snapshots.len() * std::mem::size_of::<ValidationSnapshot>()
                / (1024 * 1024),
        }
    }

    /// Compute a detailed change delta between current state and cached snapshots.
    pub fn compute_change_delta(
        &self,
        store: &dyn Store,
        current_constraints: &[Constraint],
        nodes: &[Term],
    ) -> Result<ChangesDelta> {
        let mut delta = ChangesDelta::new();
        let snapshots = self
            .previous_results
            .read()
            .expect("read lock should not be poisoned");

        for node in nodes {
            if let Some(previous_snapshot) = snapshots.get(node) {
                let current_property_hash = self.hash_node_properties(store, node)?;
                let current_constraint_hash = self.hash_constraints(current_constraints);

                if current_property_hash != previous_snapshot.properties_hash {
                    delta.nodes_with_property_changes.push(NodePropertyChange {
                        node: node.clone(),
                        previous_hash: previous_snapshot.properties_hash,
                        current_hash: current_property_hash,
                        property_changes: self.compute_property_changes(store, node)?,
                        detected_at: std::time::SystemTime::now(),
                    });
                }

                if current_constraint_hash != previous_snapshot.constraints_hash {
                    delta
                        .nodes_with_constraint_changes
                        .push(NodeConstraintChange {
                            node: node.clone(),
                            previous_constraints_hash: previous_snapshot.constraints_hash,
                            current_constraints_hash: current_constraint_hash,
                            changed_shapes: vec![],
                            detected_at: std::time::SystemTime::now(),
                        });
                }
            } else {
                delta.new_nodes.push(node.clone());
            }
        }

        let current_nodes: std::collections::HashSet<&Term> = nodes.iter().collect();
        for snapshot_node in snapshots.keys() {
            if !current_nodes.contains(snapshot_node) {
                delta.deleted_nodes.push(snapshot_node.clone());
            }
        }

        Ok(delta)
    }

    /// Generate change events for external system integration from a delta.
    pub fn generate_change_events(
        &self,
        delta: &ChangesDelta,
        _validation_results: &[crate::constraints::ConstraintEvaluationResult],
    ) -> Vec<ChangeEvent> {
        let mut events = Vec::new();
        let timestamp = std::time::SystemTime::now();

        for property_change in &delta.nodes_with_property_changes {
            let event_id = format!(
                "prop_change_{}_{}",
                property_change.node.as_str(),
                timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_millis()
            );

            let payload = serde_json::json!({
                "node": property_change.node.as_str(),
                "previous_hash": property_change.previous_hash,
                "current_hash": property_change.current_hash,
                "detected_at": property_change.detected_at
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_secs()
            });

            events.push(ChangeEvent {
                event_type: ChangeEventType::NodePropertiesChanged,
                node: property_change.node.clone(),
                shape_context: None,
                payload,
                timestamp,
                event_id,
            });
        }

        for constraint_change in &delta.nodes_with_constraint_changes {
            let event_id = format!(
                "constraint_change_{}_{}",
                constraint_change.node.as_str(),
                timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_millis()
            );

            let payload = serde_json::json!({
                "node": constraint_change.node.as_str(),
                "previous_constraints_hash": constraint_change.previous_constraints_hash,
                "current_constraints_hash": constraint_change.current_constraints_hash,
                "changed_shapes": constraint_change.changed_shapes
            });

            events.push(ChangeEvent {
                event_type: ChangeEventType::ShapeConstraintsChanged,
                node: constraint_change.node.clone(),
                shape_context: None,
                payload,
                timestamp,
                event_id,
            });
        }

        for new_node in &delta.new_nodes {
            let event_id = format!(
                "node_added_{}_{}",
                new_node.as_str(),
                timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_millis()
            );

            let payload = serde_json::json!({
                "node": new_node.as_str(),
                "detected_at": timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_secs()
            });

            events.push(ChangeEvent {
                event_type: ChangeEventType::NodeAdded,
                node: new_node.clone(),
                shape_context: None,
                payload,
                timestamp,
                event_id,
            });
        }

        for deleted_node in &delta.deleted_nodes {
            let event_id = format!(
                "node_removed_{}_{}",
                deleted_node.as_str(),
                timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_millis()
            );

            let payload = serde_json::json!({
                "node": deleted_node.as_str(),
                "detected_at": timestamp
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("operation should succeed")
                    .as_secs()
            });

            events.push(ChangeEvent {
                event_type: ChangeEventType::NodeRemoved,
                node: deleted_node.clone(),
                shape_context: None,
                payload,
                timestamp,
                event_id,
            });
        }

        events
    }

    fn compute_property_changes(
        &self,
        store: &dyn Store,
        node: &Term,
    ) -> Result<Vec<PropertyChange>> {
        let mut changes = Vec::new();
        let current_triples = self.query_node_triples_as_subject(store, node)?;

        if !current_triples.is_empty() {
            for triple in current_triples.iter().take(5) {
                if let oxirs_core::model::Predicate::NamedNode(predicate_nn) = triple.predicate() {
                    changes.push(PropertyChange {
                        subject: node.clone(),
                        property: predicate_nn.clone(),
                        change_type: PropertyChangeType::Modified,
                        old_value: None,
                        new_value: Some(match triple.object() {
                            oxirs_core::model::Object::NamedNode(nn) => Term::NamedNode(nn.clone()),
                            oxirs_core::model::Object::BlankNode(bn) => Term::BlankNode(bn.clone()),
                            oxirs_core::model::Object::Literal(lit) => Term::Literal(lit.clone()),
                            oxirs_core::model::Object::Variable(v) => Term::Variable(v.clone()),
                            _ => continue,
                        }),
                        timestamp: std::time::SystemTime::now(),
                    });
                }
            }
        }

        Ok(changes)
    }

    /// Reconstruct term from string key (simplified — supports NamedNode only).
    pub fn reconstruct_term_from_key(&self, key: &str) -> Result<Term> {
        if key.starts_with("NamedNode(") {
            let iri = key
                .trim_start_matches("NamedNode(\"")
                .trim_end_matches("\")");
            oxirs_core::model::NamedNode::new(iri)
                .map(Term::NamedNode)
                .map_err(|e| ShaclError::ValidationEngine(format!("Invalid IRI: {e}")))
        } else {
            Err(ShaclError::ValidationEngine(format!(
                "Unsupported term key format: {key}"
            )))
        }
    }
}

// ─── Result and delta types ───────────────────────────────────────────────────

/// Enhanced result of incremental validation with delta processing
#[derive(Debug, Clone)]
pub struct IncrementalValidationResult {
    pub revalidated_nodes: usize,
    pub skipped_nodes: usize,
    pub new_violations: usize,
    pub total_processing_time: Duration,
    pub change_delta: ChangesDelta,
    pub resolved_violations: Vec<crate::validation::ValidationViolation>,
    pub new_violation_details: Vec<crate::validation::ValidationViolation>,
    pub change_events: Vec<ChangeEvent>,
}

impl IncrementalValidationResult {
    fn new() -> Self {
        Self {
            revalidated_nodes: 0,
            skipped_nodes: 0,
            new_violations: 0,
            total_processing_time: Duration::ZERO,
            change_delta: ChangesDelta::new(),
            resolved_violations: Vec::new(),
            new_violation_details: Vec::new(),
            change_events: Vec::new(),
        }
    }

    pub fn efficiency_ratio(&self) -> f64 {
        let total_nodes = self.revalidated_nodes + self.skipped_nodes;
        if total_nodes == 0 {
            0.0
        } else {
            self.skipped_nodes as f64 / total_nodes as f64
        }
    }

    pub fn net_violation_change(&self) -> i32 {
        self.new_violation_details.len() as i32 - self.resolved_violations.len() as i32
    }

    pub fn improved_conformance(&self) -> bool {
        self.resolved_violations.len() > self.new_violation_details.len()
    }

    pub fn change_summary(&self) -> String {
        format!(
            "Incremental validation: {} nodes revalidated, {} skipped ({}% efficiency), {} net violation change",
            self.revalidated_nodes,
            self.skipped_nodes,
            (self.efficiency_ratio() * 100.0) as u32,
            self.net_violation_change()
        )
    }
}

/// Comprehensive delta information about detected changes
#[derive(Debug, Clone)]
pub struct ChangesDelta {
    pub nodes_with_property_changes: Vec<NodePropertyChange>,
    pub nodes_with_constraint_changes: Vec<NodeConstraintChange>,
    pub new_nodes: Vec<oxirs_core::model::Term>,
    pub deleted_nodes: Vec<oxirs_core::model::Term>,
    pub property_changes: Vec<PropertyChange>,
}

impl ChangesDelta {
    fn new() -> Self {
        Self {
            nodes_with_property_changes: Vec::new(),
            nodes_with_constraint_changes: Vec::new(),
            new_nodes: Vec::new(),
            deleted_nodes: Vec::new(),
            property_changes: Vec::new(),
        }
    }

    pub fn has_changes(&self) -> bool {
        !self.nodes_with_property_changes.is_empty()
            || !self.nodes_with_constraint_changes.is_empty()
            || !self.new_nodes.is_empty()
            || !self.deleted_nodes.is_empty()
            || !self.property_changes.is_empty()
    }

    pub fn total_changes(&self) -> usize {
        self.nodes_with_property_changes.len()
            + self.nodes_with_constraint_changes.len()
            + self.new_nodes.len()
            + self.deleted_nodes.len()
            + self.property_changes.len()
    }
}

/// Details about property changes for a specific node
#[derive(Debug, Clone)]
pub struct NodePropertyChange {
    pub node: oxirs_core::model::Term,
    pub previous_hash: u64,
    pub current_hash: u64,
    pub property_changes: Vec<PropertyChange>,
    pub detected_at: std::time::SystemTime,
}

/// Details about constraint changes affecting a node
#[derive(Debug, Clone)]
pub struct NodeConstraintChange {
    pub node: oxirs_core::model::Term,
    pub previous_constraints_hash: u64,
    pub current_constraints_hash: u64,
    pub changed_shapes: Vec<crate::ShapeId>,
    pub detected_at: std::time::SystemTime,
}

/// Specific property-level change information
#[derive(Debug, Clone)]
pub struct PropertyChange {
    pub subject: oxirs_core::model::Term,
    pub property: oxirs_core::model::NamedNode,
    pub change_type: PropertyChangeType,
    pub old_value: Option<oxirs_core::model::Term>,
    pub new_value: Option<oxirs_core::model::Term>,
    pub timestamp: std::time::SystemTime,
}

/// Type of property change
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyChangeType {
    Added,
    Modified,
    Deleted,
}

/// Change events for external system integration
#[derive(Debug, Clone)]
pub struct ChangeEvent {
    pub event_type: ChangeEventType,
    pub node: oxirs_core::model::Term,
    pub shape_context: Option<crate::ShapeId>,
    pub payload: serde_json::Value,
    pub timestamp: std::time::SystemTime,
    pub event_id: String,
}

/// Types of change events
#[derive(Debug, Clone, PartialEq)]
pub enum ChangeEventType {
    ValidationStatusChanged,
    ViolationAdded,
    ViolationResolved,
    NodePropertiesChanged,
    ShapeConstraintsChanged,
    NodeAdded,
    NodeRemoved,
}

/// Statistics for incremental validation
#[derive(Debug, Clone)]
pub struct IncrementalValidationStats {
    pub cached_validations: usize,
    pub memory_usage_mb: usize,
}
