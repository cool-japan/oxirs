//! Query Execution Engine
//!
//! This module provides the core query execution engine that evaluates
//! SPARQL algebra expressions and produces result bindings.

use crate::algebra::{
    Aggregate, Algebra, BinaryOperator, Binding, Expression, Iri, Literal, PropertyPath, PropertyPathPattern,
    Solution, Term as AlgebraTerm, TriplePattern, UnaryOperator, Variable,
};
use crate::term::{Term, BindingContext, NumericValue, xsd};
use crate::expression::ExpressionEvaluator;
use crate::path::{PropertyPath as PathPropertyPath, PropertyPathEvaluator, PathDataset, PathContext};
use crate::extensions::{Value, ValueType, ExtensionRegistry, ExecutionContext as ExtContext};
use crate::builtin::register_builtin_functions;
use anyhow::{anyhow, Result};
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};
use lazy_static::lazy_static;
use chrono::Utc;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Query execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    /// Maximum execution time
    pub timeout: Option<Duration>,
    /// Memory limit in bytes
    pub memory_limit: Option<usize>,
    /// Enable parallel execution
    pub parallel: bool,
    /// Parallel execution configuration
    pub parallel_config: ParallelConfig,
    /// Streaming configuration
    pub streaming: StreamingConfig,
    /// Statistics collection
    pub collect_stats: bool,
    /// Query complexity threshold for parallel execution
    pub parallel_threshold: usize,
    /// Enable query result caching
    pub enable_caching: bool,
    /// Extension registry for functions
    pub extension_registry: Arc<ExtensionRegistry>,
}

/// Parallel execution configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of threads to use
    pub max_threads: usize,
    /// Enable work-stealing
    pub work_stealing: bool,
    /// Chunk size for parallel processing
    pub chunk_size: usize,
    /// Threshold for enabling parallel execution
    pub parallel_threshold: usize,
    /// Thread pool configuration
    pub thread_pool_config: ThreadPoolConfig,
}

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Thread stack size
    pub stack_size: Option<usize>,
    /// Thread priority
    pub thread_priority: Option<i32>,
    /// Enable thread affinity
    pub thread_affinity: bool,
}

// Global function registry
lazy_static! {
    static ref FUNCTION_REGISTRY: Arc<ExtensionRegistry> = {
        let registry = Arc::new(ExtensionRegistry::new());
        register_builtin_functions(&registry).expect("Failed to register built-in functions");
        registry
    };
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            timeout: Some(Duration::from_secs(300)), // 5 minutes default timeout
            memory_limit: Some(1024 * 1024 * 1024),  // 1GB default limit
            parallel: true,
            parallel_config: ParallelConfig::default(),
            streaming: StreamingConfig::default(),
            collect_stats: false,
            parallel_threshold: 1000,
            enable_caching: true,
            extension_registry: FUNCTION_REGISTRY.clone(),
        }
    }
}

impl Default for ParallelConfig {
    fn default() -> Self {
        let num_cpus = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            max_threads: num_cpus,
            work_stealing: true,
            chunk_size: 1000,
            parallel_threshold: 10000,
            thread_pool_config: ThreadPoolConfig::default(),
        }
    }
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            stack_size: Some(8 * 1024 * 1024), // 8MB stack size
            thread_priority: None,
            thread_affinity: false,
        }
    }
}

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Buffer size for streaming results
    pub buffer_size: usize,
    /// Batch size for result processing
    pub batch_size: usize,
    /// Enable streaming mode
    pub enabled: bool,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            batch_size: 1000,
            enabled: false,
        }
    }
}

/// Join algorithm selection
#[derive(Debug, Clone, Copy, PartialEq)]
enum JoinAlgorithm {
    NestedLoop,
    Hash,
    SortMerge,
    IndexNestedLoop,
}

/// Query execution statistics
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Execution time
    pub execution_time: Duration,
    /// Number of intermediate results
    pub intermediate_results: usize,
    /// Number of final results
    pub final_results: usize,
    /// Memory used (estimated)
    pub memory_used: usize,
    /// Number of operations performed
    pub operations: usize,
    /// Number of property path evaluations
    pub property_path_evaluations: usize,
    /// Time spent on property path evaluations
    pub time_spent_on_paths: Duration,
    /// Number of service calls
    pub service_calls: usize,
    /// Time spent on service calls
    pub time_spent_on_services: Duration,
    /// Warnings during execution
    pub warnings: Vec<String>,
}

/// Dataset abstraction for query execution
pub trait Dataset: Send + Sync {
    /// Find all triples matching the given pattern
    fn find_triples(&self, pattern: &TriplePattern) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>>;

    /// Check if a triple exists in the dataset
    fn contains_triple(&self, subject: &AlgebraTerm, predicate: &AlgebraTerm, object: &AlgebraTerm) -> Result<bool>;

    /// Get all subjects in the dataset
    fn subjects(&self) -> Result<Vec<AlgebraTerm>>;

    /// Get all predicates in the dataset
    fn predicates(&self) -> Result<Vec<AlgebraTerm>>;

    /// Get all objects in the dataset
    fn objects(&self) -> Result<Vec<AlgebraTerm>>;
}

/// In-memory dataset implementation for testing
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    triples: Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>,
}

impl InMemoryDataset {
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
        }
    }

    pub fn add_triple(&mut self, subject: AlgebraTerm, predicate: AlgebraTerm, object: AlgebraTerm) {
        self.triples.push((subject, predicate, object));
    }

    pub fn from_triples(triples: Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>) -> Self {
        Self { triples }
    }
}

impl Dataset for InMemoryDataset {
    fn find_triples(&self, pattern: &TriplePattern) -> Result<Vec<(AlgebraTerm, AlgebraTerm, AlgebraTerm)>> {
        let results = self
            .triples
            .iter()
            .filter(|(s, p, o)| {
                matches_term(&pattern.subject, s)
                    && matches_term(&pattern.predicate, p)
                    && matches_term(&pattern.object, o)
            })
            .cloned()
            .collect();
        Ok(results)
    }

    fn contains_triple(&self, subject: &AlgebraTerm, predicate: &AlgebraTerm, object: &AlgebraTerm) -> Result<bool> {
        Ok(self
            .triples
            .iter()
            .any(|(s, p, o)| s == subject && p == predicate && o == object))
    }

    fn subjects(&self) -> Result<Vec<AlgebraTerm>> {
        let subjects: HashSet<_> = self.triples.iter().map(|(s, _, _)| s.clone()).collect();
        Ok(subjects.into_iter().collect())
    }

    fn predicates(&self) -> Result<Vec<AlgebraTerm>> {
        let predicates: HashSet<_> = self.triples.iter().map(|(_, p, _)| p.clone()).collect();
        Ok(predicates.into_iter().collect())
    }

    fn objects(&self) -> Result<Vec<AlgebraTerm>> {
        let objects: HashSet<_> = self.triples.iter().map(|(_, _, o)| o.clone()).collect();
        Ok(objects.into_iter().collect())
    }
}

impl Default for InMemoryDataset {
    fn default() -> Self {
        Self::new()
    }
}

fn matches_term(pattern: &AlgebraTerm, term: &AlgebraTerm) -> bool {
    match pattern {
        AlgebraTerm::Variable(_) => true, // Variables match any term
        _ => pattern == term,
    }
}

/// Adapter to make Dataset implement PathDataset
struct DatasetPathAdapter<'a> {
    dataset: &'a dyn Dataset,
}

impl<'a> PathDataset for DatasetPathAdapter<'a> {
    fn find_outgoing(&self, subject: &AlgebraTerm, predicate: &AlgebraTerm) -> Result<Vec<AlgebraTerm>> {
        let pattern = TriplePattern::new(
            subject.clone(),
            predicate.clone(),
            AlgebraTerm::Variable("?o".to_string()),
        );
        let triples = self.dataset.find_triples(&pattern)?;
        Ok(triples.into_iter().map(|(_, _, o)| o).collect())
    }

    fn find_incoming(&self, predicate: &AlgebraTerm, object: &AlgebraTerm) -> Result<Vec<AlgebraTerm>> {
        let pattern = TriplePattern::new(
            AlgebraTerm::Variable("?s".to_string()),
            predicate.clone(),
            object.clone(),
        );
        let triples = self.dataset.find_triples(&pattern)?;
        Ok(triples.into_iter().map(|(s, _, _)| s).collect())
    }

    fn find_predicates(&self, subject: &AlgebraTerm, object: &AlgebraTerm) -> Result<Vec<AlgebraTerm>> {
        let pattern = TriplePattern::new(
            subject.clone(),
            AlgebraTerm::Variable("?p".to_string()),
            object.clone(),
        );
        let triples = self.dataset.find_triples(&pattern)?;
        Ok(triples.into_iter().map(|(_, p, _)| p).collect())
    }

    fn get_predicates(&self) -> Result<Vec<AlgebraTerm>> {
        self.dataset.predicates()
    }

    fn contains_triple(&self, subject: &AlgebraTerm, predicate: &AlgebraTerm, object: &AlgebraTerm) -> Result<bool> {
        self.dataset.contains_triple(subject, predicate, object)
    }
}

/// Convert algebra PropertyPath to path module PropertyPath
fn convert_property_path(path: &PropertyPath) -> Result<PathPropertyPath> {
    match path {
        PropertyPath::Iri(iri) => Ok(PathPropertyPath::Direct(AlgebraTerm::Iri(iri.clone()))),
        PropertyPath::Variable(var) => Ok(PathPropertyPath::Direct(AlgebraTerm::Variable(var.clone()))),
        PropertyPath::Inverse(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::Inverse(Box::new(inner_path)))
        }
        PropertyPath::Sequence(left, right) => {
            let left_path = convert_property_path(left)?;
            let right_path = convert_property_path(right)?;
            Ok(PathPropertyPath::Sequence(Box::new(left_path), Box::new(right_path)))
        }
        PropertyPath::Alternative(left, right) => {
            let left_path = convert_property_path(left)?;
            let right_path = convert_property_path(right)?;
            Ok(PathPropertyPath::Alternative(Box::new(left_path), Box::new(right_path)))
        }
        PropertyPath::ZeroOrMore(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::ZeroOrMore(Box::new(inner_path)))
        }
        PropertyPath::OneOrMore(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::OneOrMore(Box::new(inner_path)))
        }
        PropertyPath::ZeroOrOne(inner) => {
            let inner_path = convert_property_path(inner)?;
            Ok(PathPropertyPath::ZeroOrOne(Box::new(inner_path)))
        }
        PropertyPath::NegatedPropertySet(paths) => {
            let mut terms = Vec::new();
            for p in paths {
                match p {
                    PropertyPath::Iri(iri) => terms.push(AlgebraTerm::Iri(iri.clone())),
                    PropertyPath::Variable(var) => terms.push(AlgebraTerm::Variable(var.clone())),
                    _ => return Err(anyhow!("Negated property set can only contain IRIs or variables")),
                }
            }
            Ok(PathPropertyPath::NegatedPropertySet(terms))
        }
    }
}

/// Query executor
pub struct QueryExecutor {
    context: ExecutionContext,
    function_registry: FunctionRegistry,
    parallel_executor: Option<Arc<ParallelExecutor>>,
    result_cache: Arc<RwLock<HashMap<String, CachedResult>>>,
}

/// Parallel execution engine
pub struct ParallelExecutor {
    config: ParallelConfig,
    thread_pool: Arc<Mutex<Option<ThreadPool>>>,
    work_queue: Arc<Mutex<Vec<WorkItem>>>,
    completed_work: Arc<AtomicUsize>,
    total_work: Arc<AtomicUsize>,
}

/// Work item for parallel execution
#[derive(Debug)]
pub struct WorkItem {
    pub id: usize,
    pub algebra: Algebra,
    pub context: ExecutionContext,
}

/// Thread pool for parallel execution
pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: std::sync::mpsc::Sender<Message>,
}

/// Worker thread
struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
}

/// Message for worker communication
enum Message {
    NewJob(Job),
    Terminate,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

/// Cached query result
#[derive(Debug, Clone)]
pub struct CachedResult {
    pub solution: Solution,
    pub stats: ExecutionStats,
    pub timestamp: Instant,
    pub expiry: Option<Instant>,
}

impl QueryExecutor {
    pub fn new() -> Self {
        let context = ExecutionContext::default();
        let parallel_executor = if context.parallel {
            Some(Arc::new(ParallelExecutor::new(
                context.parallel_config.clone(),
            )))
        } else {
            None
        };

        Self {
            context,
            function_registry: FunctionRegistry::new(),
            parallel_executor,
            result_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn with_context(context: ExecutionContext) -> Self {
        let parallel_executor = if context.parallel {
            Some(Arc::new(ParallelExecutor::new(
                context.parallel_config.clone(),
            )))
        } else {
            None
        };

        Self {
            context,
            function_registry: FunctionRegistry::new(),
            parallel_executor,
            result_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Enable parallel execution
    pub fn enable_parallel(&mut self, config: ParallelConfig) {
        self.context.parallel = true;
        self.context.parallel_config = config.clone();
        self.parallel_executor = Some(Arc::new(ParallelExecutor::new(config)));
    }

    /// Disable parallel execution
    pub fn disable_parallel(&mut self) {
        self.context.parallel = false;
        self.parallel_executor = None;
    }

    /// Clear result cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.result_cache.write() {
            cache.clear();
        }
    }

    /// Execute algebra expression against dataset
    pub fn execute(
        &self,
        algebra: &Algebra,
        dataset: &dyn Dataset,
    ) -> Result<(Solution, ExecutionStats)> {
        let start_time = Instant::now();
        let mut stats = ExecutionStats::default();

        // Check cache if enabled
        if self.context.enable_caching {
            let cache_key = format!("{:?}", algebra);
            if let Ok(cache) = self.result_cache.read() {
                if let Some(cached) = cache.get(&cache_key) {
                    // Check if cache entry is still valid
                    if cached.expiry.map_or(true, |exp| Instant::now() < exp) {
                        let mut cached_stats = cached.stats.clone();
                        cached_stats.execution_time = start_time.elapsed();
                        return Ok((cached.solution.clone(), cached_stats));
                    }
                }
            }
        }

        // Determine if parallel execution should be used
        let solution = if self.should_use_parallel_execution(algebra) {
            self.execute_parallel(algebra, dataset, &mut stats)?
        } else {
            self.execute_algebra(algebra, dataset, &mut stats)?
        };

        stats.execution_time = start_time.elapsed();
        stats.final_results = solution.len();

        // Cache result if enabled
        if self.context.enable_caching {
            let cache_key = format!("{:?}", algebra);
            let cached_result = CachedResult {
                solution: solution.clone(),
                stats: stats.clone(),
                timestamp: Instant::now(),
                expiry: Some(Instant::now() + Duration::from_secs(3600)), // 1 hour expiry
            };

            if let Ok(mut cache) = self.result_cache.write() {
                cache.insert(cache_key, cached_result);
            }
        }

        Ok((solution, stats))
    }

    /// Execute algebra expression in parallel
    fn execute_parallel(
        &self,
        algebra: &Algebra,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        if let Some(ref parallel_executor) = self.parallel_executor {
            parallel_executor.execute(algebra, dataset, stats)
        } else {
            // Fallback to sequential execution
            self.execute_algebra(algebra, dataset, stats)
        }
    }

    /// Determine if parallel execution should be used
    fn should_use_parallel_execution(&self, algebra: &Algebra) -> bool {
        if !self.context.parallel || self.parallel_executor.is_none() {
            return false;
        }

        // Use heuristics to determine complexity
        let complexity = self.estimate_complexity(algebra);
        complexity > self.context.parallel_threshold
    }

    /// Estimate complexity of algebra expression
    fn estimate_complexity(&self, algebra: &Algebra) -> usize {
        match algebra {
            Algebra::Bgp(patterns) => patterns.len() * 100,
            Algebra::Join { left, right } => {
                self.estimate_complexity(left) + self.estimate_complexity(right) + 500
            }
            Algebra::LeftJoin { left, right, .. } => {
                self.estimate_complexity(left) + self.estimate_complexity(right) + 750
            }
            Algebra::Union { left, right } => {
                std::cmp::max(
                    self.estimate_complexity(left),
                    self.estimate_complexity(right),
                ) + 300
            }
            Algebra::Filter { pattern, .. } => self.estimate_complexity(pattern) + 200,
            Algebra::Extend { pattern, .. } => self.estimate_complexity(pattern) + 150,
            Algebra::Minus { left, right } => {
                self.estimate_complexity(left) + self.estimate_complexity(right) + 600
            }
            Algebra::Service { pattern, .. } => self.estimate_complexity(pattern) + 2000, // High cost for remote
            Algebra::Graph { pattern, .. } => self.estimate_complexity(pattern) + 100,
            Algebra::Project { pattern, .. } => self.estimate_complexity(pattern) + 50,
            Algebra::Distinct { pattern } => self.estimate_complexity(pattern) + 400,
            Algebra::Reduced { pattern } => self.estimate_complexity(pattern) + 300,
            Algebra::Slice { pattern, .. } => self.estimate_complexity(pattern) + 50,
            Algebra::OrderBy { pattern, .. } => self.estimate_complexity(pattern) + 800,
            Algebra::Group { pattern, .. } => self.estimate_complexity(pattern) + 1000,
            Algebra::Having { pattern, .. } => self.estimate_complexity(pattern) + 300,
            Algebra::PropertyPath { path, .. } => path.complexity(),
            Algebra::Values { bindings, .. } => bindings.len() * 10,
            Algebra::Table => 1,
            Algebra::Zero => 0,
        }
    }

    /// Main algebra execution method
    fn execute_algebra(
        &self,
        algebra: &Algebra,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        stats.operations += 1;

        match algebra {
            Algebra::Bgp(patterns) => self.execute_bgp(patterns, dataset, stats),
            Algebra::Join { left, right } => self.execute_join(left, right, dataset, stats),
            Algebra::LeftJoin {
                left,
                right,
                filter,
            } => self.execute_left_join(left, right, filter.as_ref(), dataset, stats),
            Algebra::Union { left, right } => self.execute_union(left, right, dataset, stats),
            Algebra::Filter { pattern, condition } => {
                self.execute_filter(pattern, condition, dataset, stats)
            }
            Algebra::Extend {
                pattern,
                variable,
                expr,
            } => self.execute_extend(pattern, variable, expr, dataset, stats),
            Algebra::Minus { left, right } => self.execute_minus(left, right, dataset, stats),
            Algebra::Service {
                endpoint,
                pattern,
                silent,
            } => self.execute_service(endpoint, pattern, *silent, dataset, stats),
            Algebra::Graph { graph, pattern } => self.execute_graph(graph, pattern, dataset, stats),
            Algebra::Project { pattern, variables } => {
                self.execute_project(pattern, variables, dataset, stats)
            }
            Algebra::Distinct { pattern } => self.execute_distinct(pattern, dataset, stats),
            Algebra::Reduced { pattern } => self.execute_reduced(pattern, dataset, stats),
            Algebra::Slice {
                pattern,
                offset,
                limit,
            } => self.execute_slice(pattern, *offset, *limit, dataset, stats),
            Algebra::OrderBy {
                pattern,
                conditions,
            } => self.execute_order_by(pattern, conditions, dataset, stats),
            Algebra::Group {
                pattern,
                variables,
                aggregates,
            } => self.execute_group(pattern, variables, aggregates, dataset, stats),
            Algebra::Having { pattern, condition } => {
                self.execute_having(pattern, condition, dataset, stats)
            }
            Algebra::PropertyPath {
                subject,
                path,
                object,
            } => self.execute_property_path(subject, path, object, dataset, stats),
            Algebra::Values {
                variables,
                bindings,
            } => self.execute_values(variables, bindings, stats),
            Algebra::Table => Ok(vec![HashMap::new()]), // Single empty binding
            Algebra::Zero => Ok(vec![]),                // No bindings
        }
    }

    /// Execute Basic Graph Pattern with optimized triple pattern matching
    fn execute_bgp(
        &self,
        patterns: &[TriplePattern],
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        if patterns.is_empty() {
            return Ok(vec![HashMap::new()]);
        }

        // Start with the most selective pattern (heuristic: patterns with fewer variables)
        let mut sorted_patterns = patterns.to_vec();
        sorted_patterns.sort_by_key(|pattern| {
            let var_count = [&pattern.subject, &pattern.predicate, &pattern.object]
                .iter()
                .filter(|term| matches!(term, AlgebraTerm::Variable(_)))
                .count();
            var_count
        });

        let mut solution = vec![HashMap::new()];

        for pattern in &sorted_patterns {
            solution = self.join_with_pattern(solution, pattern, dataset, stats)?;
            if solution.is_empty() {
                break; // Early termination if no results
            }
        }

        Ok(solution)
    }

    /// Join current solution with a triple pattern
    fn join_with_pattern(
        &self,
        solution: Solution,
        pattern: &TriplePattern,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let mut result = Vec::new();

        for binding in solution {
            let instantiated_pattern = self.instantiate_pattern(pattern, &binding);
            let triples = dataset.find_triples(&instantiated_pattern)?;

            for (s, p, o) in triples {
                let term_s = Term::from_algebra_term(&s);
                let term_p = Term::from_algebra_term(&p);
                let term_o = Term::from_algebra_term(&o);
                if let Some(new_binding) = self.extend_binding(&binding, pattern, &term_s, &term_p, &term_o) {
                    result.push(new_binding);
                }
            }
        }

        stats.intermediate_results += result.len();
        Ok(result)
    }

    /// Instantiate pattern with current bindings
    fn instantiate_pattern(&self, pattern: &TriplePattern, binding: &Binding) -> TriplePattern {
        TriplePattern {
            subject: self.instantiate_term(&pattern.subject, binding),
            predicate: self.instantiate_term(&pattern.predicate, binding),
            object: self.instantiate_term(&pattern.object, binding),
        }
    }

    /// Instantiate term with binding
    fn instantiate_term(&self, term: &AlgebraTerm, binding: &Binding) -> AlgebraTerm {
        match term {
            AlgebraTerm::Variable(var) => binding.get(var).cloned().unwrap_or_else(|| term.clone()),
            _ => term.clone(),
        }
    }

    /// Extend binding with new triple match
    fn extend_binding(
        &self,
        binding: &Binding,
        pattern: &TriplePattern,
        s: &Term,
        p: &Term,
        o: &Term,
    ) -> Option<Binding> {
        let mut new_binding = binding.clone();
        
        let algebra_s = s.to_algebra_term();
        let algebra_p = p.to_algebra_term();
        let algebra_o = o.to_algebra_term();

        if !self.try_bind(&mut new_binding, &pattern.subject, &algebra_s)
            || !self.try_bind(&mut new_binding, &pattern.predicate, &algebra_p)
            || !self.try_bind(&mut new_binding, &pattern.object, &algebra_o)
        {
            return None;
        }

        Some(new_binding)
    }

    /// Try to bind variable to term
    fn try_bind(&self, binding: &mut Binding, pattern_term: &AlgebraTerm, actual_term: &AlgebraTerm) -> bool {
        match pattern_term {
            AlgebraTerm::Variable(var) => {
                if let Some(existing) = binding.get(var) {
                    existing == actual_term
                } else {
                    binding.insert(var.clone(), actual_term.clone());
                    true
                }
            }
            _ => pattern_term == actual_term,
        }
    }

    /// Execute join with optimized join algorithms
    fn execute_join(
        &self,
        left: &Algebra,
        right: &Algebra,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // Analyze patterns to determine best join strategy
        let join_strategy = self.select_join_algorithm(left, right, dataset);
        
        match join_strategy {
            JoinAlgorithm::IndexNestedLoop => {
                // Use index nested loop when one side has selective patterns
                self.index_nested_loop_join(left, right, dataset, stats)
            }
            JoinAlgorithm::SortMerge => {
                // Use sort-merge for large joins with high selectivity
                let left_result = self.execute_algebra(left, dataset, stats)?;
                let right_result = self.execute_algebra(right, dataset, stats)?;
                self.sort_merge_join(left_result, right_result, stats)
            }
            JoinAlgorithm::Hash => {
                // Use hash join for medium-sized joins
                let left_result = self.execute_algebra(left, dataset, stats)?;
                let right_result = self.execute_algebra(right, dataset, stats)?;
                self.hash_join(left_result, right_result, stats)
            }
            JoinAlgorithm::NestedLoop => {
                // Use nested loop for small joins
                let left_result = self.execute_algebra(left, dataset, stats)?;
                let right_result = self.execute_algebra(right, dataset, stats)?;
                self.nested_loop_join(left_result, right_result, stats)
            }
        }
    }

    /// Select the best join algorithm based on pattern analysis
    fn select_join_algorithm(
        &self,
        left: &Algebra,
        right: &Algebra,
        _dataset: &dyn Dataset,
    ) -> JoinAlgorithm {
        // Estimate cardinalities (simplified for now)
        let left_estimate = self.estimate_cardinality(left);
        let right_estimate = self.estimate_cardinality(right);
        
        // Check if either side is highly selective (good for index nested loop)
        if (left_estimate < 100 && right_estimate > 10000) ||
           (right_estimate < 100 && left_estimate > 10000) {
            return JoinAlgorithm::IndexNestedLoop;
        }
        
        // Small datasets - use nested loop
        if left_estimate < 1000 && right_estimate < 1000 {
            return JoinAlgorithm::NestedLoop;
        }
        
        // Large datasets with similar sizes - consider sort-merge
        if left_estimate > 100000 && right_estimate > 100000 {
            let ratio = if left_estimate > right_estimate {
                left_estimate as f64 / right_estimate as f64
            } else {
                right_estimate as f64 / left_estimate as f64
            };
            
            if ratio < 10.0 {
                return JoinAlgorithm::SortMerge;
            }
        }
        
        // Default to hash join
        JoinAlgorithm::Hash
    }

    /// Estimate cardinality of an algebra expression (simplified)
    fn estimate_cardinality(&self, algebra: &Algebra) -> usize {
        match algebra {
            Algebra::Bgp(patterns) => {
                // Rough estimate based on pattern count
                1000_usize.saturating_div(patterns.len().max(1))
            }
            Algebra::Filter { pattern, .. } => {
                // Filters typically reduce cardinality
                self.estimate_cardinality(pattern) / 2
            }
            Algebra::Join { left, right } => {
                // Join cardinality estimation (simplified)
                let left_card = self.estimate_cardinality(left);
                let right_card = self.estimate_cardinality(right);
                (left_card * right_card).saturating_div(100).max(1)
            }
            _ => 10000, // Default estimate
        }
    }

    /// Hash join implementation with proper join variable analysis
    fn hash_join(
        &self,
        left: Solution,
        right: Solution,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let mut result = Vec::new();

        // Analyze join variables
        let (join_vars, is_cartesian) = self.analyze_join_variables(&left, &right);
        
        if is_cartesian && self.context.collect_stats {
            stats.warnings.push("Cartesian product detected in join".to_string());
        }

        // Build hash map for smaller relation
        let (build_side, probe_side, swap_needed) = if left.len() <= right.len() {
            (&left, &right, false)
        } else {
            (&right, &left, true)
        };

        // Build hash table keyed by join variables
        let mut hash_table: HashMap<Vec<AlgebraTerm>, Vec<&Binding>> = HashMap::new();
        
        for binding in build_side {
            let key = join_vars.iter()
                .filter_map(|var| binding.get(var))
                .cloned()
                .collect::<Vec<_>>();
            hash_table.entry(key).or_insert_with(Vec::new).push(binding);
        }

        // Probe phase
        for probe_binding in probe_side {
            let probe_key = join_vars.iter()
                .filter_map(|var| probe_binding.get(var))
                .cloned()
                .collect::<Vec<_>>();

            if let Some(build_bindings) = hash_table.get(&probe_key) {
                for build_binding in build_bindings {
                    let (left_binding, right_binding) = if swap_needed {
                        (probe_binding, *build_binding)
                    } else {
                        (*build_binding, probe_binding)
                    };
                    
                    if let Some(joined) = self.merge_bindings(left_binding, right_binding) {
                        result.push(joined);
                    }
                }
            } else if join_vars.is_empty() {
                // Cartesian product case
                for build_binding in build_side {
                    if let Some(joined) = self.merge_bindings(build_binding, probe_binding) {
                        result.push(joined);
                    }
                }
            }
        }

        stats.intermediate_results += result.len();
        Ok(result)
    }

    /// Nested loop join implementation
    fn nested_loop_join(
        &self,
        left: Solution,
        right: Solution,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let mut result = Vec::new();

        for left_binding in &left {
            for right_binding in &right {
                if let Some(joined) = self.merge_bindings(left_binding, right_binding) {
                    result.push(joined);
                }
            }
        }

        stats.intermediate_results += result.len();
        Ok(result)
    }

    /// Merge two bindings, checking for consistency
    fn merge_bindings(&self, left: &Binding, right: &Binding) -> Option<Binding> {
        let mut result = left.clone();

        for (var, term) in right {
            if let Some(existing) = result.get(var) {
                if existing != term {
                    return None; // Inconsistent binding
                }
            } else {
                result.insert(var.clone(), term.clone());
            }
        }

        Some(result)
    }

    /// Analyze join variables between two solution sets
    fn analyze_join_variables(&self, left: &Solution, right: &Solution) -> (HashSet<Variable>, bool) {
        let mut left_vars = HashSet::new();
        let mut right_vars = HashSet::new();
        
        // Collect all variables from left and right
        for binding in left.iter().take(1) {
            left_vars.extend(binding.keys().cloned());
        }
        
        for binding in right.iter().take(1) {
            right_vars.extend(binding.keys().cloned());
        }
        
        // Find common variables
        let join_vars: HashSet<_> = left_vars.intersection(&right_vars).cloned().collect();
        let is_cartesian = join_vars.is_empty() && !left.is_empty() && !right.is_empty();
        
        (join_vars, is_cartesian)
    }

    /// Sort-merge join implementation
    fn sort_merge_join(
        &self,
        mut left: Solution,
        mut right: Solution,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let mut result = Vec::new();
        
        // Analyze join variables
        let (join_vars, is_cartesian) = self.analyze_join_variables(&left, &right);
        
        if is_cartesian {
            // Fall back to nested loop for cartesian products
            return self.nested_loop_join(left, right, stats);
        }
        
        // Sort both relations by join variables
        let join_vars_vec: Vec<_> = join_vars.iter().cloned().collect();
        
        left.sort_by(|a, b| {
            let a_key: Vec<_> = join_vars_vec.iter()
                .filter_map(|v| a.get(v))
                .collect();
            let b_key: Vec<_> = join_vars_vec.iter()
                .filter_map(|v| b.get(v))
                .collect();
            a_key.cmp(&b_key)
        });
        
        right.sort_by(|a, b| {
            let a_key: Vec<_> = join_vars_vec.iter()
                .filter_map(|v| a.get(v))
                .collect();
            let b_key: Vec<_> = join_vars_vec.iter()
                .filter_map(|v| b.get(v))
                .collect();
            a_key.cmp(&b_key)
        });
        
        // Merge phase
        let mut left_idx = 0;
        let mut right_idx = 0;
        
        while left_idx < left.len() && right_idx < right.len() {
            let left_binding = &left[left_idx];
            let right_binding = &right[right_idx];
            
            let left_key: Vec<_> = join_vars_vec.iter()
                .filter_map(|v| left_binding.get(v))
                .collect();
            let right_key: Vec<_> = join_vars_vec.iter()
                .filter_map(|v| right_binding.get(v))
                .collect();
            
            match left_key.cmp(&right_key) {
                std::cmp::Ordering::Less => left_idx += 1,
                std::cmp::Ordering::Greater => right_idx += 1,
                std::cmp::Ordering::Equal => {
                    // Find all matching right bindings
                    let mut right_match_idx = right_idx;
                    while right_match_idx < right.len() {
                        let right_match_binding = &right[right_match_idx];
                        let right_match_key: Vec<_> = join_vars_vec.iter()
                            .filter_map(|v| right_match_binding.get(v))
                            .collect();
                        
                        if right_match_key != right_key {
                            break;
                        }
                        
                        if let Some(joined) = self.merge_bindings(left_binding, right_match_binding) {
                            result.push(joined);
                        }
                        
                        right_match_idx += 1;
                    }
                    
                    left_idx += 1;
                }
            }
        }
        
        stats.intermediate_results += result.len();
        Ok(result)
    }

    /// Index nested loop join (uses dataset indexes for efficient lookup)
    fn index_nested_loop_join(
        &self,
        left: &Algebra,
        right: &Algebra,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // Execute left side
        let left_result = self.execute_algebra(left, dataset, stats)?;
        let mut result = Vec::new();
        
        // For each left binding, use it to constrain the right side execution
        for left_binding in left_result {
            // Create a modified right algebra with variable substitutions
            let constrained_right = self.substitute_variables(right, &left_binding);
            
            // Execute the constrained right side (this can use indexes)
            let right_results = self.execute_algebra(&constrained_right, dataset, stats)?;
            
            // Merge results
            for right_binding in right_results {
                if let Some(joined) = self.merge_bindings(&left_binding, &right_binding) {
                    result.push(joined);
                }
            }
        }
        
        stats.intermediate_results += result.len();
        Ok(result)
    }

    /// Substitute variables in algebra expression with bound values
    fn substitute_variables(&self, algebra: &Algebra, binding: &Binding) -> Algebra {
        match algebra {
            Algebra::Bgp(patterns) => {
                let substituted_patterns = patterns.iter()
                    .map(|pattern| TriplePattern {
                        subject: self.substitute_term(&pattern.subject, binding),
                        predicate: self.substitute_term(&pattern.predicate, binding),
                        object: self.substitute_term(&pattern.object, binding),
                    })
                    .collect();
                Algebra::Bgp(substituted_patterns)
            }
            Algebra::Join { left, right } => Algebra::Join {
                left: Box::new(self.substitute_variables(left, binding)),
                right: Box::new(self.substitute_variables(right, binding)),
            },
            Algebra::Filter { pattern, condition } => Algebra::Filter {
                pattern: Box::new(self.substitute_variables(pattern, binding)),
                condition: condition.clone(), // TODO: substitute in expressions too
            },
            // Add more cases as needed
            _ => algebra.clone(),
        }
    }

    /// Substitute a term if it's a variable with a binding
    fn substitute_term(&self, term: &AlgebraTerm, binding: &Binding) -> AlgebraTerm {
        match term {
            AlgebraTerm::Variable(var) => binding.get(var).cloned().unwrap_or_else(|| term.clone()),
            _ => term.clone(),
        }
    }

    /// Execute left join (OPTIONAL) with optional filter
    fn execute_left_join(
        &self,
        left: &Algebra,
        right: &Algebra,
        filter: Option<&Expression>,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let left_result = self.execute_algebra(left, dataset, stats)?;
        let right_result = self.execute_algebra(right, dataset, stats)?;

        let mut result = Vec::new();

        for left_binding in left_result {
            let mut found_match = false;

            for right_binding in &right_result {
                if let Some(joined) = self.merge_bindings(&left_binding, right_binding) {
                    // Apply filter if present
                    if let Some(filter_expr) = filter {
                        if self
                            .evaluate_expression(filter_expr, &joined)
                            .unwrap_or(false)
                        {
                            result.push(joined);
                            found_match = true;
                        }
                    } else {
                        result.push(joined);
                        found_match = true;
                    }
                }
            }

            // If no matches found, include left binding as-is (OPTIONAL semantics)
            if !found_match {
                result.push(left_binding);
            }
        }

        stats.intermediate_results += result.len();
        Ok(result)
    }

    /// Execute union operation
    fn execute_union(
        &self,
        left: &Algebra,
        right: &Algebra,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let mut left_result = self.execute_algebra(left, dataset, stats)?;
        let right_result = self.execute_algebra(right, dataset, stats)?;

        left_result.extend(right_result);
        stats.intermediate_results += left_result.len();
        Ok(left_result)
    }

    /// Execute filter operation
    fn execute_filter(
        &self,
        pattern: &Algebra,
        condition: &Expression,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let pattern_result = self.execute_algebra(pattern, dataset, stats)?;

        let filtered: Vec<_> = pattern_result
            .into_iter()
            .filter(|binding| {
                self.evaluate_expression(condition, binding)
                    .unwrap_or(false)
            })
            .collect();

        stats.intermediate_results += filtered.len();
        Ok(filtered)
    }


    /// Evaluate expression against binding returning a boolean
    fn evaluate_expression(&self, expr: &Expression, binding: &Binding) -> Result<bool> {
        let term = self.evaluate_expression_to_term(expr, binding)?;
        match &term {
            AlgebraTerm::Literal(lit) => {
                // Check if it's a boolean literal
                if let Some(datatype) = &lit.datatype {
                    if datatype.0 == "http://www.w3.org/2001/XMLSchema#boolean" {
                        return Ok(lit.value == "true" || lit.value == "1");
                    }
                }
                // Empty string is false, non-empty is true
                Ok(!lit.value.is_empty())
            }
            AlgebraTerm::Iri(_) => Ok(true),
            AlgebraTerm::BlankNode(_) => Ok(true),
            AlgebraTerm::Variable(_) => Ok(false), // Unbound variables are false
        }
    }

    /// Execute BIND operation
    fn execute_extend(
        &self,
        pattern: &Algebra,
        variable: &Variable,
        expr: &Expression,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let pattern_result = self.execute_algebra(pattern, dataset, stats)?;
        let mut result = Vec::new();

        for binding in pattern_result {
            let mut extended_binding = binding.clone();
            
            // Evaluate expression and bind to variable
            if let Ok(value) = self.evaluate_expression_to_term(expr, &binding) {
                extended_binding.insert(variable.clone(), value);
                result.push(extended_binding);
            } else {
                // Skip binding if expression evaluation fails
                continue;
            }
        }

        stats.intermediate_results += result.len();
        Ok(result)
    }

    fn execute_minus(
        &self,
        left: &Algebra,
        right: &Algebra,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let left_result = self.execute_algebra(left, dataset, stats)?;
        let right_result = self.execute_algebra(right, dataset, stats)?;
        
        let mut result = Vec::new();
        
        // For each left binding, check if it has a compatible binding in right
        for left_binding in left_result {
            let mut has_compatible = false;
            
            for right_binding in &right_result {
                if self.bindings_compatible(&left_binding, right_binding) {
                    has_compatible = true;
                    break;
                }
            }
            
            // Include left binding only if no compatible right binding found
            if !has_compatible {
                result.push(left_binding);
            }
        }
        
        stats.intermediate_results += result.len();
        Ok(result)
    }

    fn execute_service(
        &self,
        endpoint: &AlgebraTerm,
        pattern: &Algebra,
        silent: bool,
        _dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let start_time = Instant::now();
        
        // Extract endpoint URL
        let endpoint_url = match endpoint {
            AlgebraTerm::Iri(iri) => &iri.0,
            AlgebraTerm::Variable(var) => {
                return Err(anyhow!("SERVICE with variable endpoint not yet supported: ?{}", var));
            }
            _ => {
                return Err(anyhow!("SERVICE endpoint must be an IRI"));
            }
        };
        
        // Convert the algebra pattern to SPARQL query string
        let query = self.algebra_to_sparql(pattern)?;
        
        // Execute remote query (simplified implementation)
        // In a real implementation, this would:
        // 1. Use an HTTP client to send the query
        // 2. Parse the results in SPARQL Results format
        // 3. Convert back to our internal Solution format
        
        if silent {
            // In SILENT mode, errors are suppressed and empty results returned
            match self.execute_remote_query(endpoint_url, &query) {
                Ok(solution) => {
                    stats.service_calls += 1;
                    stats.time_spent_on_services += start_time.elapsed();
                    stats.intermediate_results += solution.len();
                    Ok(solution)
                }
                Err(_) => Ok(vec![]), // Silent failure
            }
        } else {
            let solution = self.execute_remote_query(endpoint_url, &query)?;
            stats.service_calls += 1;
            stats.time_spent_on_services += start_time.elapsed();
            stats.intermediate_results += solution.len();
            Ok(solution)
        }
    }

    fn execute_graph(
        &self,
        graph: &AlgebraTerm,
        pattern: &Algebra,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // The GRAPH clause executes a pattern within a specific named graph
        // For now, we'll implement a simplified version that:
        // 1. If graph is a variable, bind it to all named graphs and execute the pattern
        // 2. If graph is an IRI, execute the pattern only within that named graph
        
        match graph {
            AlgebraTerm::Variable(graph_var) => {
                // Find all named graphs and execute pattern in each
                // This requires dataset to support named graphs
                // For now, return empty as we don't have named graph support yet
                Ok(vec![])
            }
            AlgebraTerm::Iri(graph_iri) => {
                // Execute pattern within the specific named graph
                // This would require the dataset to support filtering by graph
                // For now, just execute the pattern normally
                self.execute_algebra(pattern, dataset, stats)
            }
            _ => {
                Err(anyhow!("GRAPH clause requires IRI or variable, not {:?}", graph))
            }
        }
    }

    fn execute_project(
        &self,
        pattern: &Algebra,
        variables: &[Variable],
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let pattern_result = self.execute_algebra(pattern, dataset, stats)?;

        let projected: Vec<_> = pattern_result
            .into_iter()
            .map(|binding| {
                let mut projected_binding = HashMap::new();
                for var in variables {
                    if let Some(term) = binding.get(var) {
                        projected_binding.insert(var.clone(), term.clone());
                    }
                }
                projected_binding
            })
            .collect();

        stats.intermediate_results += projected.len();
        Ok(projected)
    }

    fn execute_distinct(
        &self,
        pattern: &Algebra,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let pattern_result = self.execute_algebra(pattern, dataset, stats)?;
        let mut unique_results = Vec::new();
        let mut seen = HashSet::new();

        for binding in pattern_result {
            let key: Vec<_> = binding
                .iter()
                .map(|(var, term)| (var.clone(), term.clone()))
                .collect();

            if seen.insert(key) {
                unique_results.push(binding);
            }
        }

        stats.intermediate_results += unique_results.len();
        Ok(unique_results)
    }

    fn execute_reduced(
        &self,
        pattern: &Algebra,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // REDUCED is like DISTINCT but doesn't guarantee uniqueness - simplified implementation
        self.execute_distinct(pattern, dataset, stats)
    }

    fn execute_slice(
        &self,
        pattern: &Algebra,
        offset: Option<usize>,
        limit: Option<usize>,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let pattern_result = self.execute_algebra(pattern, dataset, stats)?;

        let start = offset.unwrap_or(0);
        let end = if let Some(limit) = limit {
            std::cmp::min(start + limit, pattern_result.len())
        } else {
            pattern_result.len()
        };

        let sliced: Vec<_> = pattern_result
            .into_iter()
            .skip(start)
            .take(end - start)
            .collect();
        stats.intermediate_results += sliced.len();
        Ok(sliced)
    }

    fn execute_order_by(
        &self,
        pattern: &Algebra,
        conditions: &[crate::algebra::OrderCondition],
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let mut pattern_result = self.execute_algebra(pattern, dataset, stats)?;
        
        // Sort by multiple conditions
        pattern_result.sort_by(|a, b| {
            for condition in conditions {
                let a_val = self.evaluate_expression_to_term(&condition.expr, a)
                    .unwrap_or_else(|_| AlgebraTerm::Literal(Literal {
                        value: String::new(),
                        language: None,
                        datatype: None,
                    }))
                    .to_string();
                let b_val = self.evaluate_expression_to_term(&condition.expr, b)
                    .unwrap_or_else(|_| AlgebraTerm::Literal(Literal {
                        value: String::new(),
                        language: None,
                        datatype: None,
                    }))
                    .to_string();
                
                let cmp = if condition.ascending {
                    a_val.cmp(&b_val)
                } else {
                    b_val.cmp(&a_val)
                };
                
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
        
        stats.intermediate_results += pattern_result.len();
        Ok(pattern_result)
    }

    fn execute_group(
        &self,
        pattern: &Algebra,
        variables: &[crate::algebra::GroupCondition],
        aggregates: &[(Variable, Aggregate)],
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let pattern_result = self.execute_algebra(pattern, dataset, stats)?;
        
        eprintln!("DEBUG execute_group: pattern_result.len() = {}", pattern_result.len());
        eprintln!("DEBUG execute_group: variables.len() = {}", variables.len());
        eprintln!("DEBUG execute_group: aggregates.len() = {}", aggregates.len());
        
        // Group bindings by group variables
        let mut groups: HashMap<Vec<AlgebraTerm>, Vec<Binding>> = HashMap::new();
        
        for binding in pattern_result {
            let mut group_key = Vec::new();
            
            for group_var in variables {
                let value = self.evaluate_expression_to_term(&group_var.expr, &binding)
                    .unwrap_or_else(|_| AlgebraTerm::Literal(Literal {
                        value: String::new(),
                        language: None,
                        datatype: None,
                    }));
                group_key.push(value);
            }
            
            eprintln!("DEBUG execute_group: Adding binding to group with key len = {}", group_key.len());
            groups.entry(group_key).or_insert_with(Vec::new).push(binding);
        }
        
        eprintln!("DEBUG execute_group: Number of groups = {}", groups.len());
        
        // Compute aggregates for each group
        let mut result = Vec::new();
        
        for (group_key, group_bindings) in groups {
            eprintln!("DEBUG execute_group: Processing group with {} bindings", group_bindings.len());
            let mut group_binding = HashMap::new();
            
            // Add group variables to binding
            for (i, group_var) in variables.iter().enumerate() {
                if let Some(alias) = &group_var.alias {
                    eprintln!("DEBUG execute_group: Adding group variable {} = {:?}", alias, group_key[i]);
                    group_binding.insert(alias.clone(), group_key[i].clone());
                }
            }
            
            // Compute aggregates
            for (agg_var, aggregate) in aggregates {
                eprintln!("DEBUG execute_group: Computing aggregate {:?} for variable {}", aggregate, agg_var);
                let agg_value = self.compute_aggregate(aggregate, &group_bindings)?;
                eprintln!("DEBUG execute_group: Aggregate result = {:?}", agg_value);
                group_binding.insert(agg_var.clone(), agg_value);
            }
            
            eprintln!("DEBUG execute_group: Final group_binding has {} entries", group_binding.len());
            result.push(group_binding);
        }
        
        eprintln!("DEBUG execute_group: Final result.len() = {}", result.len());
        stats.intermediate_results += result.len();
        Ok(result)
    }

    fn execute_having(
        &self,
        pattern: &Algebra,
        condition: &Expression,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let pattern_result = self.execute_algebra(pattern, dataset, stats)?;
        
        let filtered: Vec<_> = pattern_result
            .into_iter()
            .filter(|binding| {
                self.evaluate_expression(condition, binding)
                    .unwrap_or(false)
            })
            .collect();
        
        stats.intermediate_results += filtered.len();
        Ok(filtered)
    }

    fn execute_property_path(
        &self,
        subject: &AlgebraTerm,
        path: &PropertyPath,
        object: &AlgebraTerm,
        dataset: &dyn Dataset,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let adapter = DatasetPathAdapter { dataset };
        let evaluator = PropertyPathEvaluator::new();
        let path_path = convert_property_path(path)?;
        
        let start_time = Instant::now();
        
        // Determine which variables need to be bound
        let (start_var, start_term) = match subject {
            AlgebraTerm::Variable(var) => (Some(var), None),
            term => (None, Some(term)),
        };
        
        let (end_var, end_term) = match object {
            AlgebraTerm::Variable(var) => (Some(var), None),
            term => (None, Some(term)),
        };
        
        let solution = evaluator.evaluate_path_with_bindings(
            start_var,
            start_term,
            &path_path,
            end_var,
            end_term,
            &adapter,
        )?;
        
        stats.property_path_evaluations += 1;
        stats.intermediate_results += solution.len();
        stats.time_spent_on_paths += start_time.elapsed();
        
        Ok(solution)
    }

    fn execute_values(
        &self,
        variables: &[Variable],
        bindings: &[Binding],
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // VALUES clause returns the provided bindings
        let mut result = Vec::new();
        for binding in bindings {
            let mut filtered_binding = HashMap::new();
            for var in variables {
                if let Some(term) = binding.get(var) {
                    filtered_binding.insert(var.clone(), term.clone());
                }
            }
            result.push(filtered_binding);
        }

        stats.intermediate_results += result.len();
        Ok(result)
    }

    /// Evaluate expression against binding returning a term  
    fn evaluate_expression_to_term(&self, expr: &Expression, binding: &Binding) -> Result<AlgebraTerm> {
        match expr {
            Expression::Variable(var) => {
                binding.get(var)
                    .cloned()
                    .ok_or_else(|| anyhow!("Unbound variable: {}", var))
            }
            Expression::Literal(lit) => Ok(AlgebraTerm::Literal(lit.clone())),
            Expression::Iri(iri) => Ok(AlgebraTerm::Iri(iri.clone())),
            Expression::Function { name, args } => {
                self.evaluate_function(name, args, binding)
            }
            Expression::Binary { op, left, right } => {
                let left_term = self.evaluate_expression_to_term(left, binding)?;
                let right_term = self.evaluate_expression_to_term(right, binding)?;
                self.evaluate_binary_operation(op, &left_term, &right_term)
            }
            Expression::Unary { op, expr } => {
                let term = self.evaluate_expression_to_term(expr, binding)?;
                self.evaluate_unary_operation(op, &term)
            }
            Expression::Conditional { condition, then_expr, else_expr } => {
                if self.evaluate_expression(condition, binding)? {
                    self.evaluate_expression_to_term(then_expr, binding)
                } else {
                    self.evaluate_expression_to_term(else_expr, binding)
                }
            }
            Expression::Bound(var) => {
                Ok(AlgebraTerm::Literal(Literal {
                    value: binding.contains_key(var).to_string(),
                    language: None,
                    datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#boolean".to_string())),
                }))
            }
            Expression::Exists(_) => {
                // EXISTS/NOT EXISTS require dataset context and cannot be evaluated here
                Err(anyhow!("EXISTS requires dataset context - use Filter algebra node instead"))
            }
            Expression::NotExists(_) => {
                // EXISTS/NOT EXISTS require dataset context and cannot be evaluated here
                Err(anyhow!("NOT EXISTS requires dataset context - use Filter algebra node instead"))
            }
        }
    }

    /// Check if two bindings are compatible (share common variables with same values)
    fn bindings_compatible(&self, left: &Binding, right: &Binding) -> bool {
        for (var, left_term) in left {
            if let Some(right_term) = right.get(var) {
                if left_term != right_term {
                    return false;
                }
            }
        }
        true
    }

    /// Compute aggregate value for a group
    fn compute_aggregate(&self, aggregate: &Aggregate, bindings: &[Binding]) -> Result<AlgebraTerm> {
        eprintln!("DEBUG compute_aggregate: aggregate = {:?}, bindings.len() = {}", aggregate, bindings.len());
        match aggregate {
            Aggregate::Count { distinct, expr } => {
                let count = if let Some(expr) = expr {
                    eprintln!("DEBUG compute_aggregate: COUNT with expression");
                    let mut values = Vec::new();
                    for binding in bindings {
                        if let Ok(value) = self.evaluate_expression_to_term(expr, binding) {
                            if *distinct {
                                if !values.contains(&value) {
                                    values.push(value);
                                }
                            } else {
                                values.push(value);
                            }
                        }
                    }
                    values.len()
                } else {
                    eprintln!("DEBUG compute_aggregate: COUNT(*) - counting all bindings");
                    bindings.len()
                };
                eprintln!("DEBUG compute_aggregate: COUNT result = {}", count);
                Ok(AlgebraTerm::Literal(Literal {
                    value: (count as i64).to_string(),
                    language: None,
                    datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())),
                }))
            }
            Aggregate::Sum { distinct, expr } => {
                let mut sum = 0.0;
                let mut values = Vec::new();
                
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_to_term(expr, binding) {
                        if let AlgebraTerm::Literal(lit) = term {
                            // Check if it's a numeric literal by trying to parse
                            if let Ok(value) = lit.value.parse::<f64>() {
                                if *distinct {
                                    if !values.contains(&value) {
                                        values.push(value);
                                        sum += value;
                                    }
                                } else {
                                    sum += value;
                                }
                            }
                        }
                    }
                }
                Ok(AlgebraTerm::Literal(Literal {
                    value: sum.to_string(),
                    language: None,
                    datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#decimal".to_string())),
                }))
            }
            Aggregate::Min { distinct: _, expr } => {
                let mut min_val: Option<String> = None;
                
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_to_term(expr, binding) {
                        let val = term.to_string();
                        min_val = Some(match min_val {
                            None => val,
                            Some(current_min) => if val < current_min { val } else { current_min },
                        });
                    }
                }
                
                Ok(AlgebraTerm::Literal(Literal {
                    value: min_val.unwrap_or_default(),
                    language: None,
                    datatype: None,
                }))
            }
            Aggregate::Max { distinct: _, expr } => {
                let mut max_val: Option<String> = None;
                
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_to_term(expr, binding) {
                        let val = term.to_string();
                        max_val = Some(match max_val {
                            None => val,
                            Some(current_max) => if val > current_max { val } else { current_max },
                        });
                    }
                }
                
                Ok(AlgebraTerm::Literal(Literal {
                    value: max_val.unwrap_or_default(),
                    language: None,
                    datatype: None,
                }))
            }
            Aggregate::Avg { distinct, expr } => {
                let mut sum = 0.0;
                let mut count = 0;
                let mut values = Vec::new();
                
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_to_term(expr, binding) {
                        if let AlgebraTerm::Literal(lit) = term {
                            // Check if it's a numeric literal by trying to parse
                            if let Ok(value) = lit.value.parse::<f64>() {
                                if *distinct {
                                    if !values.contains(&value) {
                                        values.push(value);
                                        sum += value;
                                        count += 1;
                                    }
                                } else {
                                    sum += value;
                                    count += 1;
                                }
                            }
                        }
                    }
                }
                
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                Ok(AlgebraTerm::Literal(Literal::decimal(avg)))
            }
            Aggregate::Sample { distinct: _, expr } => {
                // Return first non-null value (simplified)
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_to_term(expr, binding) {
                        return Ok(term);
                    }
                }
                Ok(AlgebraTerm::Literal(Literal::string("")))
            }
            Aggregate::GroupConcat { distinct, expr, separator } => {
                let mut values = Vec::new();
                let sep = separator.as_deref().unwrap_or(" ");
                
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_to_term(expr, binding) {
                        let value = term.to_string();
                        if *distinct {
                            if !values.contains(&value) {
                                values.push(value);
                            }
                        } else {
                            values.push(value);
                        }
                    }
                }
                
                Ok(AlgebraTerm::Literal(Literal {
                    value: values.join(sep),
                    language: None,
                    datatype: None,
                }))
            }
        }
    }

    /// Evaluate function call using the function registry
    fn evaluate_function(&self, name: &str, args: &[Expression], binding: &Binding) -> Result<AlgebraTerm> {
        // First evaluate all arguments
        let arg_terms: Result<Vec<_>> = args.iter()
            .map(|arg| self.evaluate_expression_to_term(arg, binding))
            .collect();
        let arg_terms = arg_terms?;
        
        // Convert terms to values
        let arg_values: Result<Vec<_>> = arg_terms.iter()
            .map(|term| self.term_to_value(term))
            .collect();
        let arg_values = arg_values?;
        
        // Try to find function in registry
        let functions = self.context.extension_registry.functions.read()
            .map_err(|_| anyhow!("Failed to acquire read lock on functions"))?;
        
        if let Some(function) = functions.get(name) {
            // Create execution context for function
            let mut ext_context = ExtContext {
                variables: binding.clone(),
                namespaces: HashMap::new(),
                base_iri: None,
                dataset_context: None,
                query_time: Utc::now(),
                optimization_level: crate::extensions::OptimizationLevel::Basic,
                memory_limit: self.context.memory_limit,
                time_limit: self.context.timeout,
            };
            
            // Execute function
            let result = function.execute(&arg_values, &ext_context)?;
            
            // Convert result back to term
            self.value_to_term(&result)
        } else {
            // Fall back to hardcoded functions for backwards compatibility
            match name {
                "str" => {
                    if let Some(term) = arg_terms.first() {
                        Ok(AlgebraTerm::Literal(Literal {
                            value: term.to_string(),
                            language: None,
                            datatype: None,
                        }))
                    } else {
                        Err(anyhow!("str() requires one argument"))
                    }
                }
                "strlen" => {
                    if let Some(AlgebraTerm::Literal(lit)) = arg_terms.first() {
                        Ok(AlgebraTerm::Literal(Literal {
                            value: (lit.value.len() as i64).to_string(),
                            language: None,
                            datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())),
                        }))
                    } else {
                        Err(anyhow!("strlen() requires a literal argument"))
                    }
                }
            "ucase" => {
                if let Some(AlgebraTerm::Literal(lit)) = arg_terms.first() {
                    Ok(AlgebraTerm::Literal(Literal {
                        value: lit.value.to_uppercase(),
                        language: None,
                        datatype: None,
                    }))
                } else {
                    Err(anyhow!("ucase() requires a literal argument"))
                }
            }
            "lcase" => {
                if let Some(AlgebraTerm::Literal(lit)) = arg_terms.first() {
                    Ok(AlgebraTerm::Literal(Literal {
                        value: lit.value.to_lowercase(),
                        language: None,
                        datatype: None,
                    }))
                } else {
                    Err(anyhow!("lcase() requires a literal argument"))
                }
            }
            _ => Err(anyhow!("Unknown function: {}", name))
            }
        }
    }

    /// Evaluate binary operation with full SPARQL support
    fn evaluate_binary_operation(&self, op: &BinaryOperator, left: &AlgebraTerm, right: &AlgebraTerm) -> Result<AlgebraTerm> {
        // Convert terms to values for more flexible comparison
        let left_val = self.term_to_value(left)?;
        let right_val = self.term_to_value(right)?;
        
        let result = match op {
            // Arithmetic operations
            BinaryOperator::Add => self.apply_numeric_op(&left_val, &right_val, |a, b| a + b)?,
            BinaryOperator::Subtract => self.apply_numeric_op(&left_val, &right_val, |a, b| a - b)?,
            BinaryOperator::Multiply => self.apply_numeric_op(&left_val, &right_val, |a, b| a * b)?,
            BinaryOperator::Divide => {
                match (&left_val, &right_val) {
                    (Value::Integer(a), Value::Integer(b)) if *b != 0 => Value::Float(*a as f64 / *b as f64),
                    (Value::Float(a), Value::Float(b)) if *b != 0.0 => Value::Float(a / b),
                    (Value::Integer(a), Value::Float(b)) if *b != 0.0 => Value::Float(*a as f64 / b),
                    (Value::Float(a), Value::Integer(b)) if *b != 0 => Value::Float(a / *b as f64),
                    _ => return Err(anyhow!("Division by zero or non-numeric operands")),
                }
            }
            
            // Comparison operations  
            BinaryOperator::Equal => Value::Boolean(self.values_equal(&left_val, &right_val)?),
            BinaryOperator::NotEqual => Value::Boolean(!self.values_equal(&left_val, &right_val)?),
            BinaryOperator::Less => Value::Boolean(self.values_compare(&left_val, &right_val)? < 0),
            BinaryOperator::LessEqual => Value::Boolean(self.values_compare(&left_val, &right_val)? <= 0),
            BinaryOperator::Greater => Value::Boolean(self.values_compare(&left_val, &right_val)? > 0),
            BinaryOperator::GreaterEqual => Value::Boolean(self.values_compare(&left_val, &right_val)? >= 0),
            
            // Logical operations
            BinaryOperator::And => {
                match (&left_val, &right_val) {
                    (Value::Boolean(a), Value::Boolean(b)) => Value::Boolean(*a && *b),
                    _ => return Err(anyhow!("AND requires boolean operands")),
                }
            }
            BinaryOperator::Or => {
                match (&left_val, &right_val) {
                    (Value::Boolean(a), Value::Boolean(b)) => Value::Boolean(*a || *b),
                    _ => return Err(anyhow!("OR requires boolean operands")),
                }
            }
            // Other operations can be added here
            _ => return Err(anyhow!("Unsupported binary operation: {:?}", op)),
        };
        
        self.value_to_term(&result)
    }
    
    /// Apply numeric operation
    fn apply_numeric_op<F>(&self, left: &Value, right: &Value, op: F) -> Result<Value>
    where F: Fn(f64, f64) -> f64
    {
        match (left, right) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Float(op(*a as f64, *b as f64))),
            (Value::Float(a), Value::Float(b)) => Ok(Value::Float(op(*a, *b))),
            (Value::Integer(a), Value::Float(b)) => Ok(Value::Float(op(*a as f64, *b))),
            (Value::Float(a), Value::Integer(b)) => Ok(Value::Float(op(*a, *b as f64))),
            _ => Err(anyhow!("Numeric operation requires numeric operands")),
        }
    }
    
    /// Check if two values are equal according to SPARQL semantics
    fn values_equal(&self, left: &Value, right: &Value) -> Result<bool> {
        Ok(match (left, right) {
            // Same types
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Float(a), Value::Float(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Iri(a), Value::Iri(b)) => a == b,
            (Value::BlankNode(a), Value::BlankNode(b)) => a == b,
            (Value::DateTime(a), Value::DateTime(b)) => a == b,
            
            // Numeric type coercion
            (Value::Integer(a), Value::Float(b)) => *a as f64 == *b,
            (Value::Float(a), Value::Integer(b)) => *a == *b as f64,
            
            // Literal comparison
            (Value::Literal { value: v1, language: l1, datatype: d1 },
             Value::Literal { value: v2, language: l2, datatype: d2 }) => {
                v1 == v2 && l1 == l2 && d1 == d2
            }
            
            // Different types are not equal
            _ => false,
        })
    }
    
    /// Compare two values returning -1, 0, or 1
    fn values_compare(&self, left: &Value, right: &Value) -> Result<i32> {
        match (left, right) {
            // Numeric comparison
            (Value::Integer(a), Value::Integer(b)) => Ok(a.cmp(b) as i32),
            (Value::Float(a), Value::Float(b)) => Ok(a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) as i32),
            (Value::Integer(a), Value::Float(b)) => Ok((*a as f64).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) as i32),
            (Value::Float(a), Value::Integer(b)) => Ok(a.partial_cmp(&(*b as f64)).unwrap_or(std::cmp::Ordering::Equal) as i32),
            
            // String comparison
            (Value::String(a), Value::String(b)) => Ok(a.cmp(b) as i32),
            (Value::Literal { value: v1, .. }, Value::Literal { value: v2, .. }) => Ok(v1.cmp(v2) as i32),
            
            // DateTime comparison
            (Value::DateTime(a), Value::DateTime(b)) => Ok(a.cmp(b) as i32),
            
            // Other types
            (Value::Boolean(a), Value::Boolean(b)) => Ok(a.cmp(b) as i32),
            
            _ => Err(anyhow!("Cannot compare values of different types")),
        }
    }
    
    /// Convert value to string
    fn value_to_string(&self, value: &Value) -> Result<String> {
        Ok(match value {
            Value::String(s) => s.clone(),
            Value::Integer(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Boolean(b) => b.to_string(),
            Value::Iri(iri) => iri.clone(),
            Value::Literal { value, .. } => value.clone(),
            _ => return Err(anyhow!("Cannot convert {:?} to string", value)),
        })
    }
    
    /// Convert Term to Value for expression evaluation
    fn term_to_value(&self, term: &AlgebraTerm) -> Result<Value> {
        match term {
            AlgebraTerm::Iri(iri) => Ok(Value::Iri(iri.0.clone())),
            AlgebraTerm::BlankNode(id) => Ok(Value::BlankNode(id.clone())),
            AlgebraTerm::Variable(var) => Err(anyhow!("Cannot convert unbound variable {} to value", var)),
            AlgebraTerm::Literal(lit) => {
                // Try to parse typed literals
                if let Some(datatype) = &lit.datatype {
                    match datatype.0.as_str() {
                        "http://www.w3.org/2001/XMLSchema#integer" => {
                            if let Ok(i) = lit.value.parse::<i64>() {
                                Ok(Value::Integer(i))
                            } else {
                                Ok(Value::String(lit.value.clone()))
                            }
                        }
                        "http://www.w3.org/2001/XMLSchema#decimal" |
                        "http://www.w3.org/2001/XMLSchema#double" |
                        "http://www.w3.org/2001/XMLSchema#float" => {
                            if let Ok(f) = lit.value.parse::<f64>() {
                                Ok(Value::Float(f))
                            } else {
                                Ok(Value::String(lit.value.clone()))
                            }
                        }
                        "http://www.w3.org/2001/XMLSchema#boolean" => {
                            match lit.value.as_str() {
                                "true" | "1" => Ok(Value::Boolean(true)),
                                "false" | "0" => Ok(Value::Boolean(false)),
                                _ => Ok(Value::String(lit.value.clone())),
                            }
                        }
                        "http://www.w3.org/2001/XMLSchema#dateTime" => {
                            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(&lit.value) {
                                Ok(Value::DateTime(dt.with_timezone(&Utc)))
                            } else {
                                Ok(Value::String(lit.value.clone()))
                            }
                        }
                        _ => Ok(Value::Literal {
                            value: lit.value.clone(),
                            language: lit.language.clone(),
                            datatype: Some(datatype.0.clone()),
                        }),
                    }
                } else if lit.language.is_some() {
                    Ok(Value::Literal {
                        value: lit.value.clone(),
                        language: lit.language.clone(),
                        datatype: None,
                    })
                } else {
                    // Plain literal - treat as string
                    Ok(Value::String(lit.value.clone()))
                }
            }
        }
    }
    
    /// Convert Value back to Term
    fn value_to_term(&self, value: &Value) -> Result<AlgebraTerm> {
        match value {
            Value::String(s) => Ok(AlgebraTerm::Literal(Literal {
                value: s.clone(),
                language: None,
                datatype: None,
            })),
            Value::Integer(i) => Ok(AlgebraTerm::Literal(Literal {
                value: i.to_string(),
                language: None,
                datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())),
            })),
            Value::Float(f) => Ok(AlgebraTerm::Literal(Literal {
                value: f.to_string(),
                language: None,
                datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#double".to_string())),
            })),
            Value::Boolean(b) => Ok(AlgebraTerm::Literal(Literal {
                value: b.to_string(),
                language: None,
                datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#boolean".to_string())),
            })),
            Value::DateTime(dt) => Ok(AlgebraTerm::Literal(Literal {
                value: dt.to_rfc3339(),
                language: None,
                datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#dateTime".to_string())),
            })),
            Value::Iri(iri) => Ok(AlgebraTerm::Iri(Iri(iri.clone()))),
            Value::BlankNode(id) => Ok(AlgebraTerm::BlankNode(id.clone())),
            Value::Literal { value, language, datatype } => Ok(AlgebraTerm::Literal(Literal {
                value: value.clone(),
                language: language.clone(),
                datatype: datatype.as_ref().map(|dt| Iri(dt.clone())),
            })),
            _ => Err(anyhow!("Cannot convert value {:?} to term", value)),
        }
    }

    /// Evaluate unary operation with full SPARQL support
    fn evaluate_unary_operation(&self, op: &UnaryOperator, operand: &AlgebraTerm) -> Result<AlgebraTerm> {
        let val = self.term_to_value(operand)?;
        
        let result = match op {
            UnaryOperator::Not => {
                match val {
                    Value::Boolean(b) => Value::Boolean(!b),
                    _ => return Err(anyhow!("NOT operation requires boolean operand")),
                }
            }
            UnaryOperator::Plus => {
                match val {
                    Value::Integer(i) => Value::Integer(i),
                    Value::Float(f) => Value::Float(f),
                    _ => return Err(anyhow!("Unary plus requires numeric operand")),
                }
            }
            UnaryOperator::Minus => {
                match val {
                    Value::Integer(i) => Value::Integer(-i),
                    Value::Float(f) => Value::Float(-f),
                    _ => return Err(anyhow!("Unary minus requires numeric operand")),
                }
            }
            UnaryOperator::IsIri => {
                Value::Boolean(matches!(val, Value::Iri(_)))
            }
            UnaryOperator::IsBlank => {
                Value::Boolean(matches!(val, Value::BlankNode(_)))
            }
            UnaryOperator::IsLiteral => {
                Value::Boolean(matches!(val, Value::Literal { .. } | Value::String(_) | Value::Integer(_) | Value::Float(_) | Value::Boolean(_) | Value::DateTime(_)))
            }
            UnaryOperator::IsNumeric => {
                Value::Boolean(matches!(val, Value::Integer(_) | Value::Float(_)))
            }
        };
        
        self.value_to_term(&result)
    }
    
    /// Convert algebra to SPARQL query string (simplified)
    fn algebra_to_sparql(&self, algebra: &Algebra) -> Result<String> {
        // This is a simplified implementation
        // A full implementation would recursively convert the algebra tree
        match algebra {
            Algebra::Bgp(patterns) => {
                let mut query = String::from("SELECT * WHERE {\n");
                for pattern in patterns {
                    query.push_str(&format!("  {} {} {} .\n", 
                        self.term_to_sparql(&pattern.subject)?,
                        self.term_to_sparql(&pattern.predicate)?,
                        self.term_to_sparql(&pattern.object)?
                    ));
                }
                query.push_str("}");
                Ok(query)
            }
            _ => {
                // For now, just create a basic query
                Ok("SELECT * WHERE { ?s ?p ?o }".to_string())
            }
        }
    }
    
    /// Convert term to SPARQL syntax
    fn term_to_sparql(&self, term: &AlgebraTerm) -> Result<String> {
        match term {
            AlgebraTerm::Variable(var) => Ok(format!("?{}", var)),
            AlgebraTerm::Iri(iri) => Ok(format!("<{}>", iri.0)),
            AlgebraTerm::Literal(lit) => {
                if let Some(lang) = &lit.language {
                    Ok(format!("\"{}\"@{}", lit.value, lang))
                } else if let Some(datatype) = &lit.datatype {
                    Ok(format!("\"{}\"^^<{}>", lit.value, datatype.0))
                } else {
                    Ok(format!("\"{}\"", lit.value))
                }
            }
            AlgebraTerm::BlankNode(id) => Ok(format!("_:{}", id)),
        }
    }
    
    /// Execute a remote SPARQL query (stub implementation)
    fn execute_remote_query(&self, endpoint: &str, query: &str) -> Result<Solution> {
        // In a real implementation, this would:
        // 1. Create an HTTP client
        // 2. Send a POST request with the query
        // 3. Parse the SPARQL Results JSON/XML response
        // 4. Convert to our internal Solution format
        
        // For now, return empty results
        Ok(vec![])
    }
}

/// Function registry for custom SPARQL functions
pub struct FunctionRegistry {
    functions: HashMap<String, Box<dyn Fn(&[AlgebraTerm]) -> Result<AlgebraTerm> + Send + Sync>>,
}

impl FunctionRegistry {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }
}

impl Default for FunctionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel executor implementation placeholders
impl ParallelExecutor {
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            thread_pool: Arc::new(Mutex::new(None)),
            work_queue: Arc::new(Mutex::new(Vec::new())),
            completed_work: Arc::new(AtomicUsize::new(0)),
            total_work: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn execute(
        &self,
        _algebra: &Algebra,
        _dataset: &dyn Dataset,
        _stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        // TODO: Implement parallel execution
        Ok(vec![])
    }
}

impl Default for QueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::{Iri, Literal, Term, TriplePattern};

    #[test]
    fn test_in_memory_dataset() {
        let mut dataset = InMemoryDataset::new();
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/person1".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Literal(Literal {
                value: "John".to_string(),
                language: None,
                datatype: None,
            }),
        );

        let pattern = TriplePattern::new(
            Term::Variable("person".to_string()),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Variable("name".to_string()),
        );

        let triples = dataset.find_triples(&pattern).unwrap();
        assert_eq!(triples.len(), 1);
    }

    #[test]
    fn test_bgp_execution() {
        let mut dataset = InMemoryDataset::new();
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/person1".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Literal(Literal {
                value: "John".to_string(),
                language: None,
                datatype: None,
            }),
        );

        let pattern = TriplePattern::new(
            Term::Variable("person".to_string()),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Variable("name".to_string()),
        );

        let algebra = Algebra::Bgp(vec![pattern]);
        let executor = QueryExecutor::new();

        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();
        assert_eq!(solution.len(), 1);
        assert!(solution[0].contains_key("person"));
        assert!(solution[0].contains_key("name"));
    }
    
    #[test]
    fn test_property_path_execution() {
        let mut dataset = InMemoryDataset::new();
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/person1".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string())),
            Term::Iri(Iri("http://example.org/person2".to_string())),
        );
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/person2".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string())),
            Term::Iri(Iri("http://example.org/person3".to_string())),
        );
        
        // Test direct property path
        let path = PropertyPath::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string()));
        let algebra = Algebra::PropertyPath {
            subject: Term::Iri(Iri("http://example.org/person1".to_string())),
            path: path.clone(),
            object: Term::Variable("friend".to_string()),
        };
        
        let executor = QueryExecutor::new();
        let (solution, stats) = executor.execute(&algebra, &dataset).unwrap();
        
        // Should find person2 as a friend of person1
        assert_eq!(solution.len(), 1);
        assert_eq!(stats.property_path_evaluations, 1);
        assert!(solution[0].contains_key("friend"));
    }
    
    #[test]
    fn test_service_execution() {
        let dataset = InMemoryDataset::new();
        let executor = QueryExecutor::new();
        
        // Test SERVICE with endpoint
        let bgp = Algebra::Bgp(vec![TriplePattern::new(
            Term::Variable("s".to_string()),
            Term::Variable("p".to_string()),
            Term::Variable("o".to_string()),
        )]);
        
        let algebra = Algebra::Service {
            endpoint: Term::Iri(Iri("http://example.org/sparql".to_string())),
            pattern: Box::new(bgp),
            silent: false,
        };
        
        let (solution, stats) = executor.execute(&algebra, &dataset).unwrap();
        
        // Since execute_remote_query is a stub, it returns empty results
        assert_eq!(solution.len(), 0);
        assert_eq!(stats.service_calls, 1);
    }
    
    #[test]
    fn test_service_silent_mode() {
        let dataset = InMemoryDataset::new();
        let executor = QueryExecutor::new();
        
        // Test SERVICE with SILENT - errors should be suppressed
        let bgp = Algebra::Bgp(vec![TriplePattern::new(
            Term::Variable("s".to_string()),
            Term::Variable("p".to_string()),
            Term::Variable("o".to_string()),
        )]);
        
        let algebra = Algebra::Service {
            endpoint: Term::Iri(Iri("http://example.org/sparql".to_string())),
            pattern: Box::new(bgp),
            silent: true,
        };
        
        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();
        
        // Should not fail even if service is unavailable
        assert_eq!(solution.len(), 0);
    }
    
    #[test]
    fn test_graph_execution() {
        let mut dataset = InMemoryDataset::new();
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/s".to_string())),
            Term::Iri(Iri("http://example.org/p".to_string())),
            Term::Iri(Iri("http://example.org/o".to_string())),
        );
        
        let executor = QueryExecutor::new();
        
        // Test GRAPH with IRI
        let bgp = Algebra::Bgp(vec![TriplePattern::new(
            Term::Variable("s".to_string()),
            Term::Variable("p".to_string()),
            Term::Variable("o".to_string()),
        )]);
        
        let algebra = Algebra::Graph {
            graph: Term::Iri(Iri("http://example.org/graph1".to_string())),
            pattern: Box::new(bgp),
        };
        
        let (solution, _stats) = executor.execute(&algebra, &dataset).unwrap();
        
        // Since we don't have named graph support yet, it executes the pattern normally
        assert_eq!(solution.len(), 1);
    }
    
    #[test]
    fn test_term_to_sparql_conversion() {
        let executor = QueryExecutor::new();
        
        // Test variable conversion
        let var_term = Term::Variable("x".to_string());
        assert_eq!(executor.term_to_sparql(&var_term).unwrap(), "?x");
        
        // Test IRI conversion
        let iri_term = Term::Iri(Iri("http://example.org/test".to_string()));
        assert_eq!(executor.term_to_sparql(&iri_term).unwrap(), "<http://example.org/test>");
        
        // Test literal conversion
        let lit_term = Term::Literal(Literal {
            value: "Hello".to_string(),
            language: Some("en".to_string()),
            datatype: None,
        });
        assert_eq!(executor.term_to_sparql(&lit_term).unwrap(), "\"Hello\"@en");
        
        // Test typed literal
        let typed_lit = Term::Literal(Literal {
            value: "42".to_string(),
            language: None,
            datatype: Some(Iri("http://www.w3.org/2001/XMLSchema#integer".to_string())),
        });
        assert_eq!(executor.term_to_sparql(&typed_lit).unwrap(), "\"42\"^^<http://www.w3.org/2001/XMLSchema#integer>");
        
        // Test blank node
        let bnode = Term::BlankNode("b1".to_string());
        assert_eq!(executor.term_to_sparql(&bnode).unwrap(), "_:b1");
    }
    
    #[test]
    fn test_join_algorithms() {
        let mut dataset = InMemoryDataset::new();
        
        // Create test data with join patterns
        // person1 -> knows -> person2
        // person2 -> knows -> person3
        // person1 -> name -> "Alice"
        // person2 -> name -> "Bob"
        // person3 -> name -> "Charlie"
        
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/person1".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string())),
            Term::Iri(Iri("http://example.org/person2".to_string())),
        );
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/person2".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string())),
            Term::Iri(Iri("http://example.org/person3".to_string())),
        );
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/person1".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Literal(Literal::string("Alice")),
        );
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/person2".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Literal(Literal::string("Bob")),
        );
        dataset.add_triple(
            Term::Iri(Iri("http://example.org/person3".to_string())),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Literal(Literal::string("Charlie")),
        );
        
        // Test join: find people who know each other and their names
        let knows_pattern = TriplePattern::new(
            Term::Variable("person1".to_string()),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/knows".to_string())),
            Term::Variable("person2".to_string()),
        );
        
        let name_pattern1 = TriplePattern::new(
            Term::Variable("person1".to_string()),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Variable("name1".to_string()),
        );
        
        let name_pattern2 = TriplePattern::new(
            Term::Variable("person2".to_string()),
            Term::Iri(Iri("http://xmlns.com/foaf/0.1/name".to_string())),
            Term::Variable("name2".to_string()),
        );
        
        // Create join algebra
        let left_bgp = Algebra::Bgp(vec![knows_pattern]);
        let right_bgp = Algebra::Bgp(vec![name_pattern1, name_pattern2]);
        let join = Algebra::Join {
            left: Box::new(left_bgp),
            right: Box::new(right_bgp),
        };
        
        let executor = QueryExecutor::new();
        let (solution, stats) = executor.execute(&join, &dataset).unwrap();
        
        // Should find two relationships
        assert_eq!(solution.len(), 2);
        
        // Check for warnings about cartesian products
        assert!(stats.warnings.is_empty(), "No cartesian product warning expected");
        
        // Verify the results contain expected bindings
        for binding in &solution {
            assert!(binding.contains_key("person1"));
            assert!(binding.contains_key("person2"));
            assert!(binding.contains_key("name1"));
            assert!(binding.contains_key("name2"));
        }
    }
    
    #[test]
    fn test_sort_merge_join() {
        let executor = QueryExecutor::new();
        let mut stats = ExecutionStats::default();
        
        // Create test solutions for sort-merge join
        let left_solution: Solution = vec![
            [("x".to_string(), Term::Literal(Literal::string("1"))),
             ("y".to_string(), Term::Literal(Literal::string("a")))].into_iter().collect(),
            [("x".to_string(), Term::Literal(Literal::string("2"))),
             ("y".to_string(), Term::Literal(Literal::string("b")))].into_iter().collect(),
            [("x".to_string(), Term::Literal(Literal::string("3"))),
             ("y".to_string(), Term::Literal(Literal::string("c")))].into_iter().collect(),
        ];
        
        let right_solution: Solution = vec![
            [("x".to_string(), Term::Literal(Literal::string("1"))),
             ("z".to_string(), Term::Literal(Literal::string("foo")))].into_iter().collect(),
            [("x".to_string(), Term::Literal(Literal::string("2"))),
             ("z".to_string(), Term::Literal(Literal::string("bar")))].into_iter().collect(),
            [("x".to_string(), Term::Literal(Literal::string("2"))),
             ("z".to_string(), Term::Literal(Literal::string("baz")))].into_iter().collect(),
        ];
        
        let result = executor.sort_merge_join(left_solution, right_solution, &mut stats).unwrap();
        
        // Should produce 3 results (1 for x=1, 2 for x=2)
        assert_eq!(result.len(), 3);
        
        // Check all results have all three variables
        for binding in &result {
            assert!(binding.contains_key("x"));
            assert!(binding.contains_key("y"));
            assert!(binding.contains_key("z"));
        }
    }
    
    #[test]
    fn test_cartesian_product_warning() {
        let executor = QueryExecutor::new();
        let mut stats = ExecutionStats::default();
        
        // Create solutions with no common variables (cartesian product)
        let left_solution: Solution = vec![
            [("x".to_string(), Term::Literal(Literal::string("1")))].into_iter().collect(),
            [("x".to_string(), Term::Literal(Literal::string("2")))].into_iter().collect(),
        ];
        
        let right_solution: Solution = vec![
            [("y".to_string(), Term::Literal(Literal::string("a")))].into_iter().collect(),
            [("y".to_string(), Term::Literal(Literal::string("b")))].into_iter().collect(),
        ];
        
        let ctx = ExecutionContext {
            collect_stats: true,
            ..Default::default()
        };
        let executor_with_stats = QueryExecutor::with_context(ctx);
        
        let result = executor_with_stats.hash_join(left_solution, right_solution, &mut stats).unwrap();
        
        // Should produce cartesian product: 2 x 2 = 4 results
        assert_eq!(result.len(), 4);
        
        // Should have warning about cartesian product
        assert!(!stats.warnings.is_empty());
        assert!(stats.warnings[0].contains("Cartesian product"));
    }
    
    #[test]
    fn test_join_variable_analysis() {
        let executor = QueryExecutor::new();
        
        let solution1: Solution = vec![
            [("x".to_string(), Term::Literal(Literal::string("1"))),
             ("y".to_string(), Term::Literal(Literal::string("a")))].into_iter().collect(),
        ];
        
        let solution2: Solution = vec![
            [("y".to_string(), Term::Literal(Literal::string("a"))),
             ("z".to_string(), Term::Literal(Literal::string("foo")))].into_iter().collect(),
        ];
        
        let (join_vars, is_cartesian) = executor.analyze_join_variables(&solution1, &solution2);
        
        // Should identify 'y' as join variable
        assert_eq!(join_vars.len(), 1);
        assert!(join_vars.contains("y"));
        assert!(!is_cartesian);
        
        // Test cartesian product detection
        let solution3: Solution = vec![
            [("a".to_string(), Term::Literal(Literal::string("1")))].into_iter().collect(),
        ];
        
        let solution4: Solution = vec![
            [("b".to_string(), Term::Literal(Literal::string("2")))].into_iter().collect(),
        ];
        
        let (join_vars2, is_cartesian2) = executor.analyze_join_variables(&solution3, &solution4);
        assert!(join_vars2.is_empty());
        assert!(is_cartesian2);
    }
    
    #[test]
    fn test_comprehensive_filter_evaluation() {
        let executor = QueryExecutor::new();
        let dataset = InMemoryDataset::new();
        let mut stats = ExecutionStats::default();
        
        // Create test data
        let base_pattern = Algebra::Table;
        let mut binding = Binding::new();
        binding.insert("x".to_string(), Term::Literal(Literal::integer(10)));
        binding.insert("y".to_string(), Term::Literal(Literal::string("test")));
        binding.insert("z".to_string(), Term::Iri(Iri("http://example.org/resource".to_string())));
        
        // Test numeric comparison
        let numeric_filter = Expression::Binary {
            op: BinaryOperator::Greater,
            left: Box::new(Expression::Variable("x".to_string())),
            right: Box::new(Expression::Literal(Literal::integer(5))),
        };
        
        let result = executor.evaluate_expression(&numeric_filter, &binding).unwrap();
        assert!(result); // 10 > 5
        
        // Test string operations
        let string_expr = Expression::Function {
            name: "strlen".to_string(),
            args: vec![Expression::Variable("y".to_string())],
        };
        
        let str_result = executor.evaluate_expression_to_term(&string_expr, &binding).unwrap();
        if let Term::Literal(lit) = str_result {
            assert_eq!(lit.value, "4"); // "test" has length 4
        } else {
            panic!("Expected literal result");
        }
        
        // Test logical operations
        let complex_filter = Expression::Binary {
            op: BinaryOperator::And,
            left: Box::new(Expression::Binary {
                op: BinaryOperator::Greater,
                left: Box::new(Expression::Variable("x".to_string())),
                right: Box::new(Expression::Literal(Literal::integer(5))),
            }),
            right: Box::new(Expression::Bound("y".to_string())),
        };
        
        let complex_result = executor.evaluate_expression(&complex_filter, &binding).unwrap();
        assert!(complex_result); // x > 5 AND y is bound
        
        // TODO: IN and NOT IN expressions are not yet implemented
        // Test IN expression
        // let in_expr = Expression::In {
        //     expr: Box::new(Expression::Variable("x".to_string())),
        //     list: vec![
        //         Expression::Literal(Literal::integer(5)),
        //         Expression::Literal(Literal::integer(10)),
        //         Expression::Literal(Literal::integer(15)),
        //     ],
        // };
        // 
        // let in_result = executor.evaluate_expression(&in_expr, &binding).unwrap();
        // assert!(in_result); // x (10) is in the list
        // 
        // // Test NOT IN expression
        // let not_in_expr = Expression::NotIn {
        //     expr: Box::new(Expression::Variable("x".to_string())),
        //     list: vec![
        //         Expression::Literal(Literal::integer(1)),
        //         Expression::Literal(Literal::integer(2)),
        //         Expression::Literal(Literal::integer(3)),
        //     ],
        // };
        // 
        // let not_in_result = executor.evaluate_expression(&not_in_expr, &binding).unwrap();
        // assert!(not_in_result); // x (10) is not in the list
    }
    
    #[test]
    fn test_filter_with_functions() {
        let executor = QueryExecutor::new();
        let mut binding = Binding::new();
        binding.insert("name".to_string(), Term::Literal(Literal::string("alice")));
        binding.insert("email".to_string(), Term::Literal(Literal::string("alice@example.com")));
        
        // Test string functions
        let ucase_expr = Expression::Function {
            name: "ucase".to_string(),
            args: vec![Expression::Variable("name".to_string())],
        };
        
        let result = executor.evaluate_expression_to_term(&ucase_expr, &binding).unwrap();
        if let Term::Literal(lit) = result {
            assert_eq!(lit.value, "ALICE");
        } else {
            panic!("Expected literal result");
        }
        
        // Test function registry integration
        // This will use the built-in str function from the registry
        let str_expr = Expression::Function {
            name: "http://www.w3.org/2001/XMLSchema#string".to_string(),
            args: vec![Expression::Variable("email".to_string())],
        };
        
        let str_result = executor.evaluate_expression_to_term(&str_expr, &binding).unwrap();
        if let Term::Literal(lit) = str_result {
            assert_eq!(lit.value, "alice@example.com");
        } else {
            panic!("Expected literal result");
        }
    }
    
    #[test]
    fn test_filter_type_checking() {
        let executor = QueryExecutor::new();
        let mut binding = Binding::new();
        binding.insert("iri".to_string(), Term::Iri(Iri("http://example.org".to_string())));
        binding.insert("lit".to_string(), Term::Literal(Literal::string("value")));
        binding.insert("blank".to_string(), Term::BlankNode("b1".to_string()));
        
        // Test unary type checking operations
        let is_iri = Expression::Unary {
            op: UnaryOperator::IsIri,
            expr: Box::new(Expression::Variable("iri".to_string())),
        };
        
        let iri_result = executor.evaluate_expression(&is_iri, &binding).unwrap();
        assert!(iri_result);
        
        let is_literal = Expression::Unary {
            op: UnaryOperator::IsLiteral,
            expr: Box::new(Expression::Variable("lit".to_string())),
        };
        
        let lit_result = executor.evaluate_expression(&is_literal, &binding).unwrap();
        assert!(lit_result);
        
        let is_blank = Expression::Unary {
            op: UnaryOperator::IsBlank,
            expr: Box::new(Expression::Variable("blank".to_string())),
        };
        
        let blank_result = executor.evaluate_expression(&is_blank, &binding).unwrap();
        assert!(blank_result);
    }
}
