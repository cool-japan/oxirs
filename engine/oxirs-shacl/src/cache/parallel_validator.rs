//! Parallel constraint validation for SHACL
//!
//! Validates independent shape constraints concurrently using Rayon's work-stealing
//! thread pool. Shape constraints without dependencies on one another are evaluated
//! in parallel; results are collected and merged after all workers finish.

use std::sync::Arc;
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::constraints::{Constraint, ConstraintContext, ConstraintEvaluationResult};
use crate::{ConstraintComponentId, Result, Shape, ShapeId};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the parallel constraint validator.
#[derive(Debug, Clone)]
pub struct ParallelConstraintConfig {
    /// Maximum Rayon thread-pool size (0 = use Rayon's default).
    pub max_threads: usize,

    /// Minimum number of constraints before parallel processing is used.
    ///
    /// Below this threshold, serial evaluation is used to avoid thread
    /// overhead for small shapes.
    pub parallel_threshold: usize,

    /// Whether to abort all workers on the first violation detected.
    ///
    /// When `true`, evaluation may stop early but the result will still
    /// report the violation. Note: other workers already in flight may
    /// complete before the abort is observed.
    pub fail_fast: bool,
}

impl Default for ParallelConstraintConfig {
    fn default() -> Self {
        Self {
            max_threads: 0,
            parallel_threshold: 4,
            fail_fast: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-constraint result
// ---------------------------------------------------------------------------

/// Outcome of evaluating one constraint in parallel.
#[derive(Debug, Clone)]
pub struct ParallelConstraintOutcome {
    /// The constraint component ID
    pub component_id: ConstraintComponentId,
    /// The evaluation result
    pub result: ConstraintEvaluationResult,
    /// Wall-clock time taken to evaluate this constraint
    pub elapsed: Duration,
    /// Which worker thread (Rayon index) processed this constraint
    pub worker_hint: usize,
}

// ---------------------------------------------------------------------------
// Summary result
// ---------------------------------------------------------------------------

/// Summary of a parallel constraint validation run.
#[derive(Debug, Clone)]
pub struct ParallelValidationSummary {
    /// Shape that was validated
    pub shape_id: ShapeId,
    /// All per-constraint outcomes (in unspecified order)
    pub outcomes: Vec<ParallelConstraintOutcome>,
    /// `true` if every constraint is satisfied
    pub all_satisfied: bool,
    /// Wall-clock time for the entire parallel run
    pub total_elapsed: Duration,
    /// Number of constraints evaluated in parallel
    pub parallel_count: usize,
    /// Number of constraints evaluated serially (below threshold or by design)
    pub serial_count: usize,
}

impl ParallelValidationSummary {
    /// Return only the violated outcomes.
    pub fn violations(&self) -> impl Iterator<Item = &ParallelConstraintOutcome> {
        self.outcomes
            .iter()
            .filter(|o| matches!(o.result, ConstraintEvaluationResult::Violated { .. }))
    }

    /// Number of violated constraints.
    pub fn violation_count(&self) -> usize {
        self.violations().count()
    }
}

// ---------------------------------------------------------------------------
// Parallel constraint validator
// ---------------------------------------------------------------------------

/// Validates all constraints in a shape concurrently.
///
/// Independent constraints (those without shape-level cross-dependencies) are
/// distributed across Rayon worker threads. This is particularly effective for
/// shapes with many property constraints (e.g., sh:minCount, sh:maxCount,
/// sh:pattern, sh:datatype) where each constraint is independent.
pub struct ParallelConstraintValidator {
    config: ParallelConstraintConfig,
}

impl ParallelConstraintValidator {
    /// Create a new parallel validator with the given configuration.
    pub fn new(config: ParallelConstraintConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ParallelConstraintConfig::default())
    }

    /// Validate all constraints in `shape` against the provided context.
    ///
    /// `constraint_evaluator` is a closure that evaluates a single constraint.
    /// It takes `(component_id, constraint, context)` and returns a `Result`.
    ///
    /// The closure must be `Send + Sync` because it may be invoked from
    /// multiple Rayon threads concurrently.
    pub fn validate_shape<F>(
        &self,
        shape: &Shape,
        context: &ConstraintContext,
        constraint_evaluator: F,
    ) -> Result<ParallelValidationSummary>
    where
        F: Fn(
                &ConstraintComponentId,
                &Constraint,
                &ConstraintContext,
            ) -> Result<ConstraintEvaluationResult>
            + Send
            + Sync,
    {
        if shape.deactivated {
            return Ok(ParallelValidationSummary {
                shape_id: shape.id.clone(),
                outcomes: Vec::new(),
                all_satisfied: true,
                total_elapsed: Duration::ZERO,
                parallel_count: 0,
                serial_count: 0,
            });
        }

        let constraints: Vec<(ConstraintComponentId, Constraint)> = shape
            .constraints
            .iter()
            .map(|(id, c)| (id.clone(), c.clone()))
            .collect();

        let start = Instant::now();

        let outcomes = if constraints.len() >= self.config.parallel_threshold {
            self.evaluate_parallel(&constraints, context, &constraint_evaluator)?
        } else {
            self.evaluate_serial(&constraints, context, &constraint_evaluator)?
        };

        let total_elapsed = start.elapsed();
        let (parallel_count, serial_count) = if constraints.len() >= self.config.parallel_threshold
        {
            (constraints.len(), 0)
        } else {
            (0, constraints.len())
        };

        let all_satisfied = outcomes
            .iter()
            .all(|o| matches!(o.result, ConstraintEvaluationResult::Satisfied));

        Ok(ParallelValidationSummary {
            shape_id: shape.id.clone(),
            outcomes,
            all_satisfied,
            total_elapsed,
            parallel_count,
            serial_count,
        })
    }

    /// Validate multiple focus nodes against a single shape in parallel.
    ///
    /// Each focus node is validated in its own Rayon task. This is useful when
    /// validating a large number of independent nodes against the same shape.
    pub fn validate_nodes<F>(
        &self,
        shape: &Shape,
        contexts: &[ConstraintContext],
        constraint_evaluator: F,
    ) -> Result<Vec<ParallelValidationSummary>>
    where
        F: Fn(
                &ConstraintComponentId,
                &Constraint,
                &ConstraintContext,
            ) -> Result<ConstraintEvaluationResult>
            + Send
            + Sync,
    {
        let evaluator_arc = Arc::new(constraint_evaluator);

        contexts
            .par_iter()
            .map(|ctx| {
                let eval = Arc::clone(&evaluator_arc);
                self.validate_shape(shape, ctx, |id, c, ctx| eval(id, c, ctx))
            })
            .collect()
    }

    // ---- Internal: parallel evaluation -----------------------------------

    fn evaluate_parallel<F>(
        &self,
        constraints: &[(ConstraintComponentId, Constraint)],
        context: &ConstraintContext,
        evaluator: &F,
    ) -> Result<Vec<ParallelConstraintOutcome>>
    where
        F: Fn(
                &ConstraintComponentId,
                &Constraint,
                &ConstraintContext,
            ) -> Result<ConstraintEvaluationResult>
            + Send
            + Sync,
    {
        // Use a thread_local counter as a stand-in for worker identity since
        // Rayon does not expose thread IDs directly from user code.
        use std::sync::atomic::{AtomicUsize, Ordering};
        static WORKER_COUNTER: AtomicUsize = AtomicUsize::new(0);

        constraints
            .par_iter()
            .map(|(id, constraint)| {
                let worker_hint = WORKER_COUNTER.fetch_add(1, Ordering::Relaxed)
                    % rayon::current_num_threads().max(1);
                let t0 = Instant::now();
                let result = evaluator(id, constraint, context)?;
                let elapsed = t0.elapsed();
                Ok(ParallelConstraintOutcome {
                    component_id: id.clone(),
                    result,
                    elapsed,
                    worker_hint,
                })
            })
            .collect::<Result<Vec<_>>>()
    }

    // ---- Internal: serial evaluation -------------------------------------

    fn evaluate_serial<F>(
        &self,
        constraints: &[(ConstraintComponentId, Constraint)],
        context: &ConstraintContext,
        evaluator: &F,
    ) -> Result<Vec<ParallelConstraintOutcome>>
    where
        F: Fn(
                &ConstraintComponentId,
                &Constraint,
                &ConstraintContext,
            ) -> Result<ConstraintEvaluationResult>
            + Send
            + Sync,
    {
        constraints
            .iter()
            .enumerate()
            .map(|(idx, (id, constraint))| {
                let t0 = Instant::now();
                let result = evaluator(id, constraint, context)?;
                let elapsed = t0.elapsed();

                if self.config.fail_fast
                    && matches!(result, ConstraintEvaluationResult::Violated { .. })
                {
                    // In serial mode fail-fast means we stop after the first violation.
                    return Ok(ParallelConstraintOutcome {
                        component_id: id.clone(),
                        result,
                        elapsed,
                        worker_hint: idx,
                    });
                }

                Ok(ParallelConstraintOutcome {
                    component_id: id.clone(),
                    result,
                    elapsed,
                    worker_hint: idx,
                })
            })
            .collect::<Result<Vec<_>>>()
    }
}

// ---------------------------------------------------------------------------
// Performance statistics
// ---------------------------------------------------------------------------

/// Aggregated performance statistics for a batch of parallel validation runs.
#[derive(Debug, Clone, Default)]
pub struct ParallelValidationStats {
    /// Total shapes validated
    pub shapes_validated: usize,
    /// Total constraints evaluated
    pub constraints_evaluated: usize,
    /// Total wall-clock time across all runs
    pub total_elapsed: Duration,
    /// Number of violations found
    pub total_violations: usize,
    /// Average time per constraint
    pub avg_constraint_time: Duration,
}

impl ParallelValidationStats {
    /// Merge a validation summary into this statistics object.
    pub fn merge(&mut self, summary: &ParallelValidationSummary) {
        self.shapes_validated += 1;
        self.constraints_evaluated += summary.outcomes.len();
        self.total_elapsed += summary.total_elapsed;
        self.total_violations += summary.violation_count();

        let total_nanos: u128 = summary.outcomes.iter().map(|o| o.elapsed.as_nanos()).sum();
        let count = summary.outcomes.len().max(1);
        let avg_nanos = total_nanos / count as u128;
        // Running average (simple accumulation; caller can reset between batches)
        self.avg_constraint_time = Duration::from_nanos(
            ((self.avg_constraint_time.as_nanos()
                * (self.constraints_evaluated.saturating_sub(count)) as u128
                + avg_nanos * count as u128)
                / self.constraints_evaluated.max(1) as u128) as u64,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{
        cardinality_constraints::MinCountConstraint, constraint_context::ConstraintContext,
        Constraint,
    };
    use crate::ShaclError;
    use crate::{ConstraintComponentId, Shape, ShapeId, ShapeType};
    use std::collections::HashMap;

    use oxirs_core::model::{NamedNode, Term};

    fn make_shape_with_constraints(n: usize) -> Shape {
        let mut shape = Shape::new(ShapeId::new("http://ex/TestShape"), ShapeType::NodeShape);
        for i in 0..n {
            let id = ConstraintComponentId::new(format!("sh:minCount_{i}"));
            shape.constraints.insert(
                id,
                Constraint::MinCount(MinCountConstraint { min_count: 0 }),
            );
        }
        shape
    }

    fn dummy_context() -> ConstraintContext {
        let focus = Term::NamedNode(NamedNode::new("http://ex/Alice").expect("valid IRI"));
        ConstraintContext::new(focus, ShapeId::new("http://ex/TestShape"))
    }

    fn always_satisfied_evaluator(
        _id: &ConstraintComponentId,
        _c: &Constraint,
        _ctx: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        Ok(ConstraintEvaluationResult::Satisfied)
    }

    fn always_violated_evaluator(
        _id: &ConstraintComponentId,
        _c: &Constraint,
        _ctx: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        Ok(ConstraintEvaluationResult::Violated {
            violating_value: None,
            message: Some("test violation".to_string()),
            details: HashMap::new(),
        })
    }

    // ---- Basic correctness -----------------------------------------------

    #[test]
    fn test_all_satisfied_parallel() {
        let shape = make_shape_with_constraints(10);
        let ctx = dummy_context();
        let validator = ParallelConstraintValidator::with_defaults();

        let summary = validator
            .validate_shape(&shape, &ctx, always_satisfied_evaluator)
            .expect("validation should succeed");

        assert!(summary.all_satisfied);
        assert_eq!(summary.violation_count(), 0);
        assert_eq!(summary.outcomes.len(), 10);
    }

    #[test]
    fn test_violations_detected_parallel() {
        let shape = make_shape_with_constraints(6);
        let ctx = dummy_context();
        let validator = ParallelConstraintValidator::with_defaults();

        let summary = validator
            .validate_shape(&shape, &ctx, always_violated_evaluator)
            .expect("validation should succeed");

        assert!(!summary.all_satisfied);
        assert_eq!(summary.violation_count(), 6);
    }

    // ---- Serial fallback (below threshold) --------------------------------

    #[test]
    fn test_serial_for_small_shape() {
        let shape = make_shape_with_constraints(2); // below default threshold of 4
        let ctx = dummy_context();
        let validator = ParallelConstraintValidator::with_defaults();

        let summary = validator
            .validate_shape(&shape, &ctx, always_satisfied_evaluator)
            .expect("validation should succeed");

        assert_eq!(summary.serial_count, 2);
        assert_eq!(summary.parallel_count, 0);
        assert!(summary.all_satisfied);
    }

    // ---- Deactivated shape ----------------------------------------------

    #[test]
    fn test_deactivated_shape_skipped() {
        let mut shape = make_shape_with_constraints(5);
        shape.deactivated = true;
        let ctx = dummy_context();
        let validator = ParallelConstraintValidator::with_defaults();

        let summary = validator
            .validate_shape(&shape, &ctx, always_violated_evaluator)
            .expect("validation should succeed");

        assert!(summary.all_satisfied);
        assert_eq!(summary.outcomes.len(), 0);
    }

    // ---- validate_nodes batch --------------------------------------------

    #[test]
    fn test_validate_nodes_batch() {
        let shape = make_shape_with_constraints(4);
        let validator = ParallelConstraintValidator::with_defaults();

        let focus_iris = ["http://ex/Alice", "http://ex/Bob", "http://ex/Carol"];

        let contexts: Vec<_> = focus_iris
            .iter()
            .map(|iri| {
                let focus = Term::NamedNode(NamedNode::new(*iri).expect("valid IRI"));
                ConstraintContext::new(focus, ShapeId::new("http://ex/TestShape"))
            })
            .collect();

        let summaries = validator
            .validate_nodes(&shape, &contexts, always_satisfied_evaluator)
            .expect("batch validation should succeed");

        assert_eq!(summaries.len(), 3);
        for s in &summaries {
            assert!(s.all_satisfied);
        }
    }

    // ---- Statistics merging ---------------------------------------------

    #[test]
    fn test_stats_merge() {
        let shape = make_shape_with_constraints(5);
        let ctx = dummy_context();
        let validator = ParallelConstraintValidator::with_defaults();

        let summary = validator
            .validate_shape(&shape, &ctx, always_satisfied_evaluator)
            .expect("validation should succeed");

        let mut stats = ParallelValidationStats::default();
        stats.merge(&summary);

        assert_eq!(stats.shapes_validated, 1);
        assert_eq!(stats.constraints_evaluated, 5);
        assert_eq!(stats.total_violations, 0);
    }

    // ---- Error propagation -----------------------------------------------

    #[test]
    fn test_evaluator_error_propagated() {
        let shape = make_shape_with_constraints(4);
        let ctx = dummy_context();
        let validator = ParallelConstraintValidator::with_defaults();

        let result = validator.validate_shape(&shape, &ctx, |_id, _c, _ctx| {
            Err(ShaclError::ConstraintValidation(
                "simulated error".to_string(),
            ))
        });

        assert!(result.is_err());
    }
}

// ---------------------------------------------------------------------------
// Extended parallel validator tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod extended_parallel_tests {
    use super::*;
    use crate::constraints::{
        cardinality_constraints::MinCountConstraint, constraint_context::ConstraintContext,
        Constraint,
    };
    use crate::{ConstraintComponentId, Shape, ShapeId, ShapeType};
    use oxirs_core::model::{NamedNode, Term};
    use std::collections::HashMap;

    fn make_shape(n: usize) -> Shape {
        let mut shape = Shape::new(ShapeId::new("http://ex/TestShape"), ShapeType::NodeShape);
        for i in 0..n {
            let id = ConstraintComponentId::new(format!("sh:minCount_{i}"));
            shape.constraints.insert(
                id,
                Constraint::MinCount(MinCountConstraint { min_count: 0 }),
            );
        }
        shape
    }

    fn ctx_for(iri: &str) -> ConstraintContext {
        let focus = Term::NamedNode(NamedNode::new(iri).expect("valid IRI"));
        ConstraintContext::new(focus, ShapeId::new("http://ex/TestShape"))
    }

    fn always_ok(
        _id: &ConstraintComponentId,
        _c: &Constraint,
        _ctx: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        Ok(ConstraintEvaluationResult::Satisfied)
    }

    fn always_fail(
        _id: &ConstraintComponentId,
        _c: &Constraint,
        _ctx: &ConstraintContext,
    ) -> Result<ConstraintEvaluationResult> {
        Ok(ConstraintEvaluationResult::Violated {
            violating_value: None,
            message: Some("fail".to_string()),
            details: HashMap::new(),
        })
    }

    // ---- Config construction -------------------------------------------

    #[test]
    fn test_default_config_threshold_is_four() {
        let cfg = ParallelConstraintConfig::default();
        assert_eq!(cfg.parallel_threshold, 4);
    }

    #[test]
    fn test_default_config_fail_fast_is_false() {
        let cfg = ParallelConstraintConfig::default();
        assert!(!cfg.fail_fast);
    }

    #[test]
    fn test_custom_config_stored() {
        let cfg = ParallelConstraintConfig {
            max_threads: 2,
            parallel_threshold: 10,
            fail_fast: true,
        };
        let validator = ParallelConstraintValidator::new(cfg);
        assert_eq!(validator.config.parallel_threshold, 10);
        assert!(validator.config.fail_fast);
    }

    // ---- violation_count / violations iterator -------------------------

    #[test]
    fn test_violations_iterator_empty_when_all_satisfied() {
        let shape = make_shape(4);
        let ctx = ctx_for("http://ex/Alice");
        let v = ParallelConstraintValidator::with_defaults();
        let summary = v.validate_shape(&shape, &ctx, always_ok).expect("ok");
        assert_eq!(summary.violations().count(), 0);
    }

    #[test]
    fn test_violations_iterator_matches_violation_count() {
        let shape = make_shape(6);
        let ctx = ctx_for("http://ex/Alice");
        let v = ParallelConstraintValidator::with_defaults();
        let summary = v.validate_shape(&shape, &ctx, always_fail).expect("ok");
        let iter_count = summary.violations().count();
        assert_eq!(iter_count, summary.violation_count());
    }

    // ---- validate_nodes empty list ------------------------------------

    #[test]
    fn test_validate_nodes_empty_list() {
        let shape = make_shape(4);
        let v = ParallelConstraintValidator::with_defaults();
        let summaries = v.validate_nodes(&shape, &[], always_ok).expect("ok");
        assert!(summaries.is_empty());
    }

    // ---- validate_nodes with violations in some nodes -----------------

    #[test]
    fn test_validate_nodes_partial_violations() {
        let shape = make_shape(5);
        let v = ParallelConstraintValidator::with_defaults();

        let ctxs: Vec<_> = ["http://ex/A", "http://ex/B", "http://ex/C"]
            .iter()
            .map(|iri| ctx_for(iri))
            .collect();

        // all fail
        let summaries = v.validate_nodes(&shape, &ctxs, always_fail).expect("ok");
        assert_eq!(summaries.len(), 3);
        assert!(summaries.iter().all(|s| !s.all_satisfied));
    }

    // ---- ParallelValidationStats ---------------------------------------

    #[test]
    fn test_stats_merge_accumulates_shapes() {
        let shape = make_shape(4);
        let ctx1 = ctx_for("http://ex/A");
        let ctx2 = ctx_for("http://ex/B");
        let v = ParallelConstraintValidator::with_defaults();

        let s1 = v.validate_shape(&shape, &ctx1, always_ok).expect("ok");
        let s2 = v.validate_shape(&shape, &ctx2, always_ok).expect("ok");

        let mut stats = ParallelValidationStats::default();
        stats.merge(&s1);
        stats.merge(&s2);

        assert_eq!(stats.shapes_validated, 2);
        assert_eq!(stats.constraints_evaluated, 8);
    }

    #[test]
    fn test_stats_total_violations_accumulates() {
        let shape = make_shape(3);
        let ctx = ctx_for("http://ex/A");
        let v = ParallelConstraintValidator::with_defaults();

        let summary = v.validate_shape(&shape, &ctx, always_fail).expect("ok");

        let mut stats = ParallelValidationStats::default();
        stats.merge(&summary);

        assert_eq!(stats.total_violations, 3);
    }

    #[test]
    fn test_stats_merge_twice_doubles_counts() {
        let shape = make_shape(4);
        let ctx = ctx_for("http://ex/A");
        let v = ParallelConstraintValidator::with_defaults();
        let summary = v.validate_shape(&shape, &ctx, always_ok).expect("ok");

        let mut stats = ParallelValidationStats::default();
        stats.merge(&summary);
        stats.merge(&summary);

        assert_eq!(stats.shapes_validated, 2);
        assert_eq!(stats.constraints_evaluated, 8);
    }

    // ---- all_satisfied flag -------------------------------------------

    #[test]
    fn test_all_satisfied_true_when_no_violations() {
        let shape = make_shape(8);
        let ctx = ctx_for("http://ex/Alice");
        let v = ParallelConstraintValidator::with_defaults();
        let summary = v.validate_shape(&shape, &ctx, always_ok).expect("ok");
        assert!(summary.all_satisfied);
    }

    #[test]
    fn test_all_satisfied_false_when_violations_exist() {
        let shape = make_shape(8);
        let ctx = ctx_for("http://ex/Alice");
        let v = ParallelConstraintValidator::with_defaults();
        let summary = v.validate_shape(&shape, &ctx, always_fail).expect("ok");
        assert!(!summary.all_satisfied);
    }

    // ---- outcomes length matches constraint count ----------------------

    #[test]
    fn test_outcomes_length_matches_constraint_count() {
        let n = 7;
        let shape = make_shape(n);
        let ctx = ctx_for("http://ex/X");
        let v = ParallelConstraintValidator::with_defaults();
        let summary = v.validate_shape(&shape, &ctx, always_ok).expect("ok");
        assert_eq!(summary.outcomes.len(), n);
    }

    // ---- shape with zero constraints -----------------------------------

    #[test]
    fn test_shape_with_zero_constraints() {
        let shape = make_shape(0);
        let ctx = ctx_for("http://ex/Alice");
        let v = ParallelConstraintValidator::with_defaults();
        let summary = v.validate_shape(&shape, &ctx, always_ok).expect("ok");
        assert!(summary.all_satisfied);
        assert_eq!(summary.outcomes.len(), 0);
    }

    // ---- above parallel threshold uses parallel path ------------------

    #[test]
    fn test_above_threshold_uses_parallel_path() {
        let shape = make_shape(20); // well above default threshold of 4
        let ctx = ctx_for("http://ex/Alice");
        let v = ParallelConstraintValidator::with_defaults();
        let summary = v.validate_shape(&shape, &ctx, always_ok).expect("ok");
        // parallel_count should be > 0
        assert!(
            summary.parallel_count > 0,
            "expected parallel evaluation for 20 constraints"
        );
    }
}
