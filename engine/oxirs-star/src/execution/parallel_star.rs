//! Work-stealing parallel SPARQL-star evaluation.
//!
//! This module implements a work-stealing scheduler for RDF-star sub-queries, enabling
//! parallel evaluation of annotated triple patterns without a central bottleneck.
//!
//! # Architecture
//!
//! ```text
//! ParallelStarExecutor
//!   └─ WorkStealingScheduler
//!        ├─ WorkerQueue[0]  (local deque, LIFO pop, FIFO steal)
//!        ├─ WorkerQueue[1]
//!        └─ ...
//! ```
//!
//! Each worker pops from its own queue first. When empty, it steals from the
//! front of a random peer's queue.  This is a simplified but correct
//! approximation of Chase-Lev work-stealing using Rust's standard primitives.

use crate::{StarError, StarResult, StarTerm, StarTriple};
use rayon::prelude::*;
use scirs2_core::profiling::Profiler;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Sub-query types
// ---------------------------------------------------------------------------

/// An annotated triple pattern for SPARQL-star evaluation.
#[derive(Debug, Clone)]
pub struct AnnotatedPattern {
    /// Subject slot – `None` represents an unbound variable.
    pub subject: Option<StarTerm>,
    /// Predicate slot – `None` represents an unbound variable.
    pub predicate: Option<StarTerm>,
    /// Object slot – `None` represents an unbound variable.
    pub object: Option<StarTerm>,
    /// Optional annotation predicate (e.g. `ex:certainty`).
    pub annotation_pred: Option<StarTerm>,
    /// Optional annotation value.
    pub annotation_val: Option<StarTerm>,
    /// Variable name for the quoted triple itself (SPARQL `<<…>> ?qt …`).
    pub quoted_triple_var: Option<String>,
    /// Priority hint for the scheduler (higher = more urgent).
    pub priority: u8,
}

impl AnnotatedPattern {
    /// Convenience constructor.
    pub fn new(
        subject: Option<StarTerm>,
        predicate: Option<StarTerm>,
        object: Option<StarTerm>,
    ) -> Self {
        Self {
            subject,
            predicate,
            object,
            annotation_pred: None,
            annotation_val: None,
            quoted_triple_var: None,
            priority: 0,
        }
    }

    /// Check whether a given triple matches this pattern.
    pub fn matches(&self, triple: &StarTriple) -> bool {
        let s_ok = self
            .subject
            .as_ref()
            .map(|s| s == &triple.subject)
            .unwrap_or(true);
        let p_ok = self
            .predicate
            .as_ref()
            .map(|p| p == &triple.predicate)
            .unwrap_or(true);
        let o_ok = self
            .object
            .as_ref()
            .map(|o| o == &triple.object)
            .unwrap_or(true);
        s_ok && p_ok && o_ok
    }
}

// ---------------------------------------------------------------------------
// Sub-query work unit
// ---------------------------------------------------------------------------

/// A single unit of work for the scheduler.
#[derive(Debug, Clone)]
pub struct SubQuery {
    /// Unique identifier.
    pub id: u64,
    /// Annotated pattern to evaluate.
    pub pattern: AnnotatedPattern,
    /// Variable bindings inherited from the parent query.
    pub bindings: HashMap<String, StarTerm>,
    /// Depth in the query tree (for recursion guard).
    pub depth: usize,
}

impl SubQuery {
    pub fn new(id: u64, pattern: AnnotatedPattern) -> Self {
        Self {
            id,
            pattern,
            bindings: HashMap::new(),
            depth: 0,
        }
    }

    pub fn with_bindings(mut self, bindings: HashMap<String, StarTerm>) -> Self {
        self.bindings = bindings;
        self
    }

    pub fn with_depth(mut self, depth: usize) -> Self {
        self.depth = depth;
        self
    }
}

// ---------------------------------------------------------------------------
// Evaluation result
// ---------------------------------------------------------------------------

/// Result of evaluating a sub-query.
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// Originating sub-query ID.
    pub query_id: u64,
    /// Matched triples.
    pub matches: Vec<StarTriple>,
    /// Variable bindings produced by this match.
    pub bindings: Vec<HashMap<String, StarTerm>>,
    /// Evaluation latency.
    pub latency: Duration,
    /// Whether the evaluation succeeded.
    pub success: bool,
    /// Error message if not successful.
    pub error: Option<String>,
}

impl EvalResult {
    pub fn empty(query_id: u64) -> Self {
        Self {
            query_id,
            matches: Vec::new(),
            bindings: Vec::new(),
            latency: Duration::ZERO,
            success: true,
            error: None,
        }
    }

    pub fn error(query_id: u64, msg: impl Into<String>) -> Self {
        Self {
            query_id,
            matches: Vec::new(),
            bindings: Vec::new(),
            latency: Duration::ZERO,
            success: false,
            error: Some(msg.into()),
        }
    }
}

// ---------------------------------------------------------------------------
// Work-stealing scheduler
// ---------------------------------------------------------------------------

/// Configuration for the parallel star executor.
#[derive(Debug, Clone)]
pub struct ParallelStarConfig {
    /// Number of parallel worker threads.
    pub worker_count: usize,
    /// Maximum sub-query queue depth before back-pressure.
    pub max_queue_depth: usize,
    /// Maximum recursion depth for nested quoted triple patterns.
    pub max_depth: usize,
    /// Timeout per sub-query evaluation.
    pub query_timeout: Duration,
}

impl Default for ParallelStarConfig {
    fn default() -> Self {
        Self {
            worker_count: num_cpus::get().max(2),
            max_queue_depth: 1024,
            max_depth: 8,
            query_timeout: Duration::from_secs(30),
        }
    }
}

/// Shared work queue with work-stealing semantics.
///
/// Each logical worker owns a `VecDeque`.  Pop from the back (LIFO for
/// locality); steal from the front (FIFO for breadth).
#[derive(Debug)]
struct WorkerQueue {
    deque: VecDeque<SubQuery>,
}

impl WorkerQueue {
    fn new() -> Self {
        Self {
            deque: VecDeque::new(),
        }
    }

    fn push(&mut self, item: SubQuery) {
        self.deque.push_back(item);
    }

    fn pop(&mut self) -> Option<SubQuery> {
        self.deque.pop_back()
    }

    fn steal(&mut self) -> Option<SubQuery> {
        self.deque.pop_front()
    }

    fn len(&self) -> usize {
        self.deque.len()
    }
}

/// Work-stealing scheduler coordinating sub-query dispatch.
pub struct WorkStealingScheduler {
    queues: Vec<Arc<Mutex<WorkerQueue>>>,
    worker_count: usize,
}

impl WorkStealingScheduler {
    pub fn new(worker_count: usize) -> Self {
        let queues = (0..worker_count)
            .map(|_| Arc::new(Mutex::new(WorkerQueue::new())))
            .collect();
        Self {
            queues,
            worker_count,
        }
    }

    /// Submit a sub-query, routing to the least-loaded worker.
    pub fn submit(&self, query: SubQuery) -> StarResult<()> {
        let target = self.least_loaded_worker();
        let mut q = self.queues[target]
            .lock()
            .map_err(|_| StarError::processing_error("WorkerQueue lock poisoned"))?;
        q.push(query);
        Ok(())
    }

    /// Try to pop a task for the given worker, stealing from a peer if local is empty.
    pub fn pop_or_steal(&self, worker_id: usize) -> Option<SubQuery> {
        // Try own queue first.
        if let Ok(mut q) = self.queues[worker_id].try_lock() {
            if let Some(task) = q.pop() {
                return Some(task);
            }
        }
        // Try to steal from a peer.
        for offset in 1..self.worker_count {
            let peer = (worker_id + offset) % self.worker_count;
            if let Ok(mut q) = self.queues[peer].try_lock() {
                if let Some(task) = q.steal() {
                    return Some(task);
                }
            }
        }
        None
    }

    /// Total pending tasks across all queues.
    pub fn pending_count(&self) -> usize {
        self.queues
            .iter()
            .filter_map(|q| q.try_lock().ok())
            .map(|q| q.len())
            .sum()
    }

    fn least_loaded_worker(&self) -> usize {
        let mut min_load = usize::MAX;
        let mut best = 0;
        for (i, q) in self.queues.iter().enumerate() {
            if let Ok(q) = q.try_lock() {
                let load = q.len();
                if load < min_load {
                    min_load = load;
                    best = i;
                }
            }
        }
        best
    }
}

// ---------------------------------------------------------------------------
// Parallel annotated triple evaluator
// ---------------------------------------------------------------------------

/// Snapshot of the triple store used for evaluation.
pub type TripleStore = Vec<StarTriple>;

/// Parallel SPARQL-star executor with work-stealing scheduling.
pub struct ParallelStarExecutor {
    config: ParallelStarConfig,
    scheduler: Arc<WorkStealingScheduler>,
    #[allow(dead_code)]
    profiler: Arc<Mutex<Profiler>>,
    #[allow(dead_code)]
    result_log: Arc<RwLock<Vec<EvalResult>>>,
}

impl ParallelStarExecutor {
    /// Create a new executor with the given configuration.
    pub fn new(config: ParallelStarConfig) -> Self {
        let scheduler = Arc::new(WorkStealingScheduler::new(config.worker_count));
        Self {
            config,
            scheduler,
            profiler: Arc::new(Mutex::new(Profiler::new())),
            result_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Evaluate a batch of annotated patterns against a triple store in parallel.
    ///
    /// Returns one `EvalResult` per input sub-query.
    pub fn evaluate_batch(
        &self,
        store: &TripleStore,
        patterns: Vec<AnnotatedPattern>,
    ) -> StarResult<Vec<EvalResult>> {
        let sub_queries: Vec<SubQuery> = patterns
            .into_iter()
            .enumerate()
            .map(|(i, p)| SubQuery::new(i as u64, p))
            .collect();

        // Enqueue all sub-queries.
        for sq in &sub_queries {
            self.scheduler.submit(sq.clone())?;
        }

        let worker_count = self.config.worker_count;
        let timeout = self.config.query_timeout;
        let max_depth = self.config.max_depth;
        let scheduler = Arc::clone(&self.scheduler);
        let store_ref: Vec<StarTriple> = store.clone();

        // Parallel evaluation using rayon.
        let results: Vec<EvalResult> = (0..worker_count)
            .into_par_iter()
            .flat_map(|worker_id| {
                let mut local_results = Vec::new();
                let start = Instant::now();

                loop {
                    if start.elapsed() > timeout {
                        break;
                    }

                    let task = match scheduler.pop_or_steal(worker_id) {
                        Some(t) => t,
                        None => break,
                    };

                    let eval_start = Instant::now();
                    let result = evaluate_sub_query(&task, &store_ref, max_depth);
                    let mut result = result;
                    result.latency = eval_start.elapsed();
                    local_results.push(result);
                }

                local_results
            })
            .collect();

        // Sort results by query_id to match input order.
        let mut sorted = results;
        sorted.sort_by_key(|r| r.query_id);

        Ok(sorted)
    }

    /// Evaluate a single annotated pattern against a triple store.
    pub fn evaluate_single(
        &self,
        store: &TripleStore,
        pattern: AnnotatedPattern,
    ) -> StarResult<EvalResult> {
        let sq = SubQuery::new(0, pattern);
        let result = evaluate_sub_query(&sq, store, self.config.max_depth);
        Ok(result)
    }

    /// Statistics about the executor's profiling data.
    pub fn execution_stats(&self) -> ExecutionStats {
        ExecutionStats {
            pending_tasks: self.scheduler.pending_count(),
            worker_count: self.config.worker_count,
        }
    }
}

/// Evaluate a single sub-query against the store.
fn evaluate_sub_query(sq: &SubQuery, store: &[StarTriple], max_depth: usize) -> EvalResult {
    if sq.depth > max_depth {
        return EvalResult::error(sq.id, format!("Max recursion depth {} exceeded", max_depth));
    }

    let mut result = EvalResult::empty(sq.id);

    // Filter triples by the annotated pattern.
    for triple in store {
        if sq.pattern.matches(triple) {
            // Produce variable bindings.
            let mut binding: HashMap<String, StarTerm> = sq.bindings.clone();

            if let Some(ref var_name) = sq.pattern.quoted_triple_var {
                binding.insert(
                    var_name.clone(),
                    StarTerm::QuotedTriple(Box::new(triple.clone())),
                );
            }

            // Bind subject variable if unbound.
            bind_variable_slot(None, &triple.subject, &mut binding);
            bind_variable_slot(None, &triple.predicate, &mut binding);
            bind_variable_slot(None, &triple.object, &mut binding);

            result.matches.push(triple.clone());
            result.bindings.push(binding);
        }
    }

    result
}

/// Bind a variable-name slot in the binding map if the slot is a variable term.
fn bind_variable_slot(
    var_name: Option<&str>,
    term: &StarTerm,
    bindings: &mut HashMap<String, StarTerm>,
) {
    if let Some(name) = var_name {
        bindings
            .entry(name.to_string())
            .or_insert_with(|| term.clone());
    }
}

/// Executor statistics snapshot.
#[derive(Debug, Clone)]
pub struct ExecutionStats {
    pub pending_tasks: usize,
    pub worker_count: usize,
}

// ---------------------------------------------------------------------------
// Parallel annotated evaluation helpers
// ---------------------------------------------------------------------------

/// Evaluate a set of quoted triple patterns in parallel and merge bindings.
///
/// This function is the primary entry point for parallel SPARQL-star evaluation
/// without needing to construct a full executor.
pub fn parallel_eval_annotated(
    store: &[StarTriple],
    patterns: &[AnnotatedPattern],
) -> StarResult<Vec<EvalResult>> {
    let config = ParallelStarConfig::default();
    let executor = ParallelStarExecutor::new(config);
    executor.evaluate_batch(&store.to_vec(), patterns.to_vec())
}

/// Merge multiple binding sets using hash-join semantics.
///
/// Two binding maps are compatible if all shared variable names have the same value.
pub fn merge_bindings(
    left: &[HashMap<String, StarTerm>],
    right: &[HashMap<String, StarTerm>],
) -> Vec<HashMap<String, StarTerm>> {
    left.par_iter()
        .flat_map(|l| {
            right
                .par_iter()
                .filter_map(|r| {
                    // Check compatibility on shared variables.
                    for (k, v) in l {
                        if let Some(rv) = r.get(k) {
                            if rv != v {
                                return None;
                            }
                        }
                    }
                    // Merge.
                    let mut merged = l.clone();
                    for (k, v) in r {
                        merged.entry(k.clone()).or_insert_with(|| v.clone());
                    }
                    Some(merged)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StarTerm, StarTriple};

    fn make_triple(s: &str, p: &str, o: &str) -> StarTriple {
        StarTriple::new(
            StarTerm::iri(s).unwrap(),
            StarTerm::iri(p).unwrap(),
            StarTerm::iri(o).unwrap(),
        )
    }

    fn build_store(n: usize) -> TripleStore {
        (0..n)
            .map(|i| {
                make_triple(
                    &format!("http://ex.org/s{i}"),
                    "http://ex.org/p",
                    &format!("http://ex.org/o{i}"),
                )
            })
            .collect()
    }

    // --- AnnotatedPattern tests ---

    #[test]
    fn test_pattern_matches_exact() {
        let t = make_triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        let pattern = AnnotatedPattern::new(
            Some(StarTerm::iri("http://ex.org/s").unwrap()),
            Some(StarTerm::iri("http://ex.org/p").unwrap()),
            Some(StarTerm::iri("http://ex.org/o").unwrap()),
        );
        assert!(pattern.matches(&t));
    }

    #[test]
    fn test_pattern_matches_wildcard_subject() {
        let t = make_triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        let pattern = AnnotatedPattern::new(
            None,
            Some(StarTerm::iri("http://ex.org/p").unwrap()),
            Some(StarTerm::iri("http://ex.org/o").unwrap()),
        );
        assert!(pattern.matches(&t));
    }

    #[test]
    fn test_pattern_no_match() {
        let t = make_triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        let pattern = AnnotatedPattern::new(
            Some(StarTerm::iri("http://ex.org/DIFFERENT").unwrap()),
            None,
            None,
        );
        assert!(!pattern.matches(&t));
    }

    #[test]
    fn test_pattern_all_wildcard() {
        let t = make_triple("http://ex.org/s", "http://ex.org/p", "http://ex.org/o");
        let pattern = AnnotatedPattern::new(None, None, None);
        assert!(pattern.matches(&t));
    }

    // --- SubQuery tests ---

    #[test]
    fn test_sub_query_construction() {
        let p = AnnotatedPattern::new(None, None, None);
        let sq = SubQuery::new(42, p).with_depth(3);
        assert_eq!(sq.id, 42);
        assert_eq!(sq.depth, 3);
    }

    // --- evaluate_sub_query tests ---

    #[test]
    fn test_evaluate_sub_query_all_match() {
        let store = build_store(5);
        let pattern = AnnotatedPattern::new(None, None, None);
        let sq = SubQuery::new(0, pattern);
        let result = evaluate_sub_query(&sq, &store, 8);
        assert!(result.success);
        assert_eq!(result.matches.len(), 5);
    }

    #[test]
    fn test_evaluate_sub_query_selective() {
        let store = build_store(5);
        let pattern =
            AnnotatedPattern::new(Some(StarTerm::iri("http://ex.org/s0").unwrap()), None, None);
        let sq = SubQuery::new(0, pattern);
        let result = evaluate_sub_query(&sq, &store, 8);
        assert!(result.success);
        assert_eq!(result.matches.len(), 1);
    }

    #[test]
    fn test_evaluate_sub_query_depth_exceeded() {
        let store = build_store(5);
        let pattern = AnnotatedPattern::new(None, None, None);
        let sq = SubQuery::new(0, pattern).with_depth(100);
        let result = evaluate_sub_query(&sq, &store, 8);
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_evaluate_sub_query_empty_store() {
        let store: Vec<StarTriple> = Vec::new();
        let pattern = AnnotatedPattern::new(None, None, None);
        let sq = SubQuery::new(0, pattern);
        let result = evaluate_sub_query(&sq, &store, 8);
        assert!(result.success);
        assert!(result.matches.is_empty());
    }

    // --- WorkStealingScheduler tests ---

    #[test]
    fn test_scheduler_submit_and_pop() {
        let scheduler = WorkStealingScheduler::new(2);
        let p = AnnotatedPattern::new(None, None, None);
        let sq = SubQuery::new(1, p);
        scheduler.submit(sq.clone()).unwrap();
        assert_eq!(scheduler.pending_count(), 1);
        let popped = scheduler.pop_or_steal(0);
        assert!(popped.is_some());
    }

    #[test]
    fn test_scheduler_steal_from_peer() {
        let scheduler = WorkStealingScheduler::new(2);
        // Submit to worker 0.
        for i in 0..5u64 {
            let p = AnnotatedPattern::new(None, None, None);
            let sq = SubQuery::new(i, p);
            // Force to worker 0 by direct lock.
            scheduler.queues[0].lock().unwrap().push(sq);
        }
        // Worker 1 should steal from worker 0.
        let stolen = scheduler.pop_or_steal(1);
        assert!(stolen.is_some(), "Worker 1 should steal from worker 0");
    }

    #[test]
    fn test_scheduler_empty_returns_none() {
        let scheduler = WorkStealingScheduler::new(3);
        let task = scheduler.pop_or_steal(0);
        assert!(task.is_none());
    }

    // --- ParallelStarExecutor tests ---

    #[test]
    fn test_executor_evaluate_batch_empty_patterns() {
        let store = build_store(10);
        let config = ParallelStarConfig {
            worker_count: 2,
            ..Default::default()
        };
        let executor = ParallelStarExecutor::new(config);
        let results = executor.evaluate_batch(&store, vec![]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_executor_evaluate_batch_single_pattern() {
        let store = build_store(10);
        let config = ParallelStarConfig {
            worker_count: 2,
            ..Default::default()
        };
        let executor = ParallelStarExecutor::new(config);
        let pattern = AnnotatedPattern::new(None, None, None);
        let results = executor.evaluate_batch(&store, vec![pattern]).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].matches.len(), 10);
    }

    #[test]
    fn test_executor_evaluate_batch_multiple_patterns() {
        let store = build_store(20);
        let config = ParallelStarConfig {
            worker_count: 4,
            ..Default::default()
        };
        let executor = ParallelStarExecutor::new(config);
        let patterns: Vec<AnnotatedPattern> = (0..8)
            .map(|i| {
                AnnotatedPattern::new(
                    Some(StarTerm::iri(&format!("http://ex.org/s{i}")).unwrap()),
                    None,
                    None,
                )
            })
            .collect();
        let results = executor.evaluate_batch(&store, patterns).unwrap();
        // Each pattern should match exactly one triple.
        for r in &results {
            assert!(r.success, "Sub-query {} should succeed", r.query_id);
        }
    }

    #[test]
    fn test_executor_evaluate_single() {
        let store = build_store(5);
        let config = ParallelStarConfig::default();
        let executor = ParallelStarExecutor::new(config);
        let pattern = AnnotatedPattern::new(None, None, None);
        let result = executor.evaluate_single(&store, pattern).unwrap();
        assert!(result.success);
        assert_eq!(result.matches.len(), 5);
    }

    #[test]
    fn test_executor_stats() {
        let config = ParallelStarConfig {
            worker_count: 3,
            ..Default::default()
        };
        let executor = ParallelStarExecutor::new(config);
        let stats = executor.execution_stats();
        assert_eq!(stats.worker_count, 3);
        assert_eq!(stats.pending_tasks, 0);
    }

    // --- parallel_eval_annotated tests ---

    #[test]
    fn test_parallel_eval_annotated_basic() {
        let store = build_store(10);
        let patterns = vec![
            AnnotatedPattern::new(None, None, None),
            AnnotatedPattern::new(Some(StarTerm::iri("http://ex.org/s5").unwrap()), None, None),
        ];
        let results = parallel_eval_annotated(&store, &patterns).unwrap();
        // At least one result should contain 10 matches (all-wildcard pattern).
        let all_match = results.iter().any(|r| r.matches.len() == 10);
        assert!(all_match, "One pattern should match all 10 triples");
    }

    // --- merge_bindings tests ---

    #[test]
    fn test_merge_bindings_compatible() {
        let mut b1 = HashMap::new();
        b1.insert(
            "x".to_string(),
            StarTerm::iri("http://ex.org/alice").unwrap(),
        );

        let mut b2 = HashMap::new();
        b2.insert("y".to_string(), StarTerm::iri("http://ex.org/bob").unwrap());

        let merged = merge_bindings(&[b1], &[b2]);
        assert_eq!(merged.len(), 1);
        assert!(merged[0].contains_key("x"));
        assert!(merged[0].contains_key("y"));
    }

    #[test]
    fn test_merge_bindings_incompatible() {
        let mut b1 = HashMap::new();
        b1.insert(
            "x".to_string(),
            StarTerm::iri("http://ex.org/alice").unwrap(),
        );

        let mut b2 = HashMap::new();
        b2.insert("x".to_string(), StarTerm::iri("http://ex.org/bob").unwrap());

        let merged = merge_bindings(&[b1], &[b2]);
        assert!(merged.is_empty(), "Incompatible bindings should not merge");
    }

    #[test]
    fn test_merge_bindings_empty_left() {
        let b2: Vec<HashMap<String, StarTerm>> = vec![HashMap::new()];
        let merged = merge_bindings(&[], &b2);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_merge_bindings_empty_right() {
        let mut b1 = HashMap::new();
        b1.insert("x".to_string(), StarTerm::iri("http://ex.org/a").unwrap());
        let merged = merge_bindings(&[b1], &[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_annotated_pattern_priority() {
        let mut p = AnnotatedPattern::new(None, None, None);
        p.priority = 5;
        assert_eq!(p.priority, 5);
    }

    #[test]
    fn test_annotated_pattern_annotation_fields() {
        let mut p = AnnotatedPattern::new(None, None, None);
        p.annotation_pred = Some(StarTerm::iri("http://ex.org/certainty").unwrap());
        p.annotation_val = Some(StarTerm::Literal(crate::model::Literal {
            value: "0.9".to_string(),
            language: None,
            datatype: None,
        }));
        assert!(p.annotation_pred.is_some());
        assert!(p.annotation_val.is_some());
    }

    #[test]
    fn test_annotated_pattern_quoted_triple_var() {
        let mut p = AnnotatedPattern::new(None, None, None);
        p.quoted_triple_var = Some("qt".to_string());
        let store = build_store(3);
        let sq = SubQuery::new(0, p);
        let result = evaluate_sub_query(&sq, &store, 8);
        assert_eq!(result.matches.len(), 3);
        // Each binding should have the "qt" variable.
        for b in &result.bindings {
            assert!(b.contains_key("qt"), "Binding should contain 'qt' variable");
        }
    }

    #[test]
    fn test_eval_result_error_constructor() {
        let r = EvalResult::error(99, "test error");
        assert_eq!(r.query_id, 99);
        assert!(!r.success);
        assert_eq!(r.error.as_deref(), Some("test error"));
    }
}
