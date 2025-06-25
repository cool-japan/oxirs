//! Query Execution Engine
//!
//! This module provides the core query execution engine that evaluates
//! SPARQL algebra expressions and produces result bindings.

use crate::algebra::{
    Aggregate, Algebra, BinaryOperator, Binding, Expression, Literal, PropertyPath, PropertyPathPattern,
    Solution, Term, TriplePattern, UnaryOperator, Variable,
};
use anyhow::{anyhow, Result};
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

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
}

/// Dataset abstraction for query execution
pub trait Dataset: Send + Sync {
    /// Find all triples matching the given pattern
    fn find_triples(&self, pattern: &TriplePattern) -> Result<Vec<(Term, Term, Term)>>;

    /// Check if a triple exists in the dataset
    fn contains_triple(&self, subject: &Term, predicate: &Term, object: &Term) -> Result<bool>;

    /// Get all subjects in the dataset
    fn subjects(&self) -> Result<Vec<Term>>;

    /// Get all predicates in the dataset
    fn predicates(&self) -> Result<Vec<Term>>;

    /// Get all objects in the dataset
    fn objects(&self) -> Result<Vec<Term>>;
}

/// In-memory dataset implementation for testing
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    triples: Vec<(Term, Term, Term)>,
}

impl InMemoryDataset {
    pub fn new() -> Self {
        Self {
            triples: Vec::new(),
        }
    }

    pub fn add_triple(&mut self, subject: Term, predicate: Term, object: Term) {
        self.triples.push((subject, predicate, object));
    }

    pub fn from_triples(triples: Vec<(Term, Term, Term)>) -> Self {
        Self { triples }
    }
}

impl Dataset for InMemoryDataset {
    fn find_triples(&self, pattern: &TriplePattern) -> Result<Vec<(Term, Term, Term)>> {
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

    fn contains_triple(&self, subject: &Term, predicate: &Term, object: &Term) -> Result<bool> {
        Ok(self
            .triples
            .iter()
            .any(|(s, p, o)| s == subject && p == predicate && o == object))
    }

    fn subjects(&self) -> Result<Vec<Term>> {
        let subjects: HashSet<_> = self.triples.iter().map(|(s, _, _)| s.clone()).collect();
        Ok(subjects.into_iter().collect())
    }

    fn predicates(&self) -> Result<Vec<Term>> {
        let predicates: HashSet<_> = self.triples.iter().map(|(_, p, _)| p.clone()).collect();
        Ok(predicates.into_iter().collect())
    }

    fn objects(&self) -> Result<Vec<Term>> {
        let objects: HashSet<_> = self.triples.iter().map(|(_, _, o)| o.clone()).collect();
        Ok(objects.into_iter().collect())
    }
}

impl Default for InMemoryDataset {
    fn default() -> Self {
        Self::new()
    }
}

fn matches_term(pattern: &Term, term: &Term) -> bool {
    match pattern {
        Term::Variable(_) => true, // Variables match any term
        _ => pattern == term,
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
                .filter(|term| matches!(term, Term::Variable(_)))
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
                if let Some(new_binding) = self.extend_binding(&binding, pattern, &s, &p, &o) {
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
    fn instantiate_term(&self, term: &Term, binding: &Binding) -> Term {
        match term {
            Term::Variable(var) => binding.get(var).cloned().unwrap_or_else(|| term.clone()),
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

        if !self.try_bind(&mut new_binding, &pattern.subject, s)
            || !self.try_bind(&mut new_binding, &pattern.predicate, p)
            || !self.try_bind(&mut new_binding, &pattern.object, o)
        {
            return None;
        }

        Some(new_binding)
    }

    /// Try to bind variable to term
    fn try_bind(&self, binding: &mut Binding, pattern_term: &Term, actual_term: &Term) -> bool {
        match pattern_term {
            Term::Variable(var) => {
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
        let left_result = self.execute_algebra(left, dataset, stats)?;
        let right_result = self.execute_algebra(right, dataset, stats)?;

        // Choose join algorithm based on result sizes
        if left_result.len() < 1000 && right_result.len() < 1000 {
            self.nested_loop_join(left_result, right_result, stats)
        } else {
            self.hash_join(left_result, right_result, stats)
        }
    }

    /// Hash join implementation
    fn hash_join(
        &self,
        left: Solution,
        right: Solution,
        stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        let mut result = Vec::new();

        // Build hash map for smaller relation
        let (build_side, probe_side) = if left.len() <= right.len() {
            (left, right)
        } else {
            (right, left)
        };

        // Group by common variables
        let mut hash_table: HashMap<Vec<(Variable, Term)>, Vec<Binding>> = HashMap::new();

        for binding in build_side {
            let key: Vec<_> = binding
                .iter()
                .map(|(var, term)| (var.clone(), term.clone()))
                .collect();
            hash_table.entry(key).or_insert_with(Vec::new).push(binding);
        }

        // Probe phase
        for probe_binding in probe_side {
            let probe_key: Vec<_> = probe_binding
                .iter()
                .map(|(var, term)| (var.clone(), term.clone()))
                .collect();

            if let Some(build_bindings) = hash_table.get(&probe_key) {
                for build_binding in build_bindings {
                    if let Some(joined) = self.merge_bindings(build_binding, &probe_binding) {
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

    /// Evaluate expression against binding
    fn evaluate_expression(&self, expr: &Expression, binding: &Binding) -> Result<bool> {
        match expr {
            Expression::Variable(var) => Ok(binding.contains_key(var)),
            Expression::Literal(lit) => Ok(lit.is_boolean() && lit.value == "true"),
            Expression::Binary { op, left, right } => {
                let left_val = self.evaluate_expression(left, binding)?;
                let right_val = self.evaluate_expression(right, binding)?;

                match op {
                    BinaryOperator::And => Ok(left_val && right_val),
                    BinaryOperator::Or => Ok(left_val || right_val),
                    BinaryOperator::Equal => Ok(left_val == right_val),
                    BinaryOperator::NotEqual => Ok(left_val != right_val),
                    _ => Ok(false), // Simplified for basic operations
                }
            }
            Expression::Unary { op, expr } => {
                let val = self.evaluate_expression(expr, binding)?;
                match op {
                    UnaryOperator::Not => Ok(!val),
                    _ => Ok(val),
                }
            }
            Expression::Bound(var) => Ok(binding.contains_key(var)),
            _ => Ok(false), // Simplified for other expressions
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
        _endpoint: &Term,
        _pattern: &Algebra,
        _silent: bool,
        _dataset: &dyn Dataset,
        _stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        Ok(vec![]) // TODO: Implement SERVICE federation
    }

    fn execute_graph(
        &self,
        _graph: &Term,
        _pattern: &Algebra,
        _dataset: &dyn Dataset,
        _stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        Ok(vec![]) // TODO: Implement GRAPH operation
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
                    .unwrap_or_else(|_| Term::Literal(Literal::string("")))
                    .to_string();
                let b_val = self.evaluate_expression_to_term(&condition.expr, b)
                    .unwrap_or_else(|_| Term::Literal(Literal::string("")))
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
        
        // Group bindings by group variables
        let mut groups: HashMap<Vec<Term>, Vec<Binding>> = HashMap::new();
        
        for binding in pattern_result {
            let mut group_key = Vec::new();
            
            for group_var in variables {
                let value = self.evaluate_expression_to_term(&group_var.expr, &binding)
                    .unwrap_or_else(|_| Term::Literal(Literal::string("")));
                group_key.push(value);
            }
            
            groups.entry(group_key).or_insert_with(Vec::new).push(binding);
        }
        
        // Compute aggregates for each group
        let mut result = Vec::new();
        
        for (group_key, group_bindings) in groups {
            let mut group_binding = HashMap::new();
            
            // Add group variables to binding
            for (i, group_var) in variables.iter().enumerate() {
                if let Some(alias) = &group_var.alias {
                    group_binding.insert(alias.clone(), group_key[i].clone());
                }
            }
            
            // Compute aggregates
            for (agg_var, aggregate) in aggregates {
                let agg_value = self.compute_aggregate(aggregate, &group_bindings)?;
                group_binding.insert(agg_var.clone(), agg_value);
            }
            
            result.push(group_binding);
        }
        
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
        _subject: &Term,
        _path: &PropertyPath,
        _object: &Term,
        _dataset: &dyn Dataset,
        _stats: &mut ExecutionStats,
    ) -> Result<Solution> {
        Ok(vec![]) // TODO: Implement property path evaluation
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
    fn evaluate_expression_to_term(&self, expr: &Expression, binding: &Binding) -> Result<Term> {
        match expr {
            Expression::Variable(var) => {
                binding.get(var)
                    .cloned()
                    .ok_or_else(|| anyhow!("Unbound variable: {}", var))
            }
            Expression::Literal(lit) => Ok(Term::Literal(lit.clone())),
            Expression::Iri(iri) => Ok(Term::Iri(iri.clone())),
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
                Ok(Term::Literal(Literal::boolean(binding.contains_key(var))))
            }
            _ => Err(anyhow!("Unsupported expression type"))
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
    fn compute_aggregate(&self, aggregate: &Aggregate, bindings: &[Binding]) -> Result<Term> {
        match aggregate {
            Aggregate::Count { distinct, expr } => {
                let count = if let Some(expr) = expr {
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
                    bindings.len()
                };
                Ok(Term::Literal(Literal::integer(count as i64)))
            }
            Aggregate::Sum { distinct, expr } => {
                let mut sum = 0.0;
                let mut values = Vec::new();
                
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_to_term(expr, binding) {
                        if let Term::Literal(lit) = term {
                            if lit.is_numeric() {
                                let value: f64 = lit.value.parse().unwrap_or(0.0);
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
                Ok(Term::Literal(Literal::decimal(sum)))
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
                
                Ok(Term::Literal(Literal::string(min_val.unwrap_or_default())))
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
                
                Ok(Term::Literal(Literal::string(max_val.unwrap_or_default())))
            }
            Aggregate::Avg { distinct, expr } => {
                let mut sum = 0.0;
                let mut count = 0;
                let mut values = Vec::new();
                
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_to_term(expr, binding) {
                        if let Term::Literal(lit) = term {
                            if lit.is_numeric() {
                                let value: f64 = lit.value.parse().unwrap_or(0.0);
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
                Ok(Term::Literal(Literal::decimal(avg)))
            }
            Aggregate::Sample { distinct: _, expr } => {
                // Return first non-null value (simplified)
                for binding in bindings {
                    if let Ok(term) = self.evaluate_expression_to_term(expr, binding) {
                        return Ok(term);
                    }
                }
                Ok(Term::Literal(Literal::string("")))
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
                
                Ok(Term::Literal(Literal::string(values.join(sep))))
            }
        }
    }

    /// Evaluate function call
    fn evaluate_function(&self, name: &str, args: &[Expression], binding: &Binding) -> Result<Term> {
        let arg_terms: Result<Vec<_>> = args.iter()
            .map(|arg| self.evaluate_expression_to_term(arg, binding))
            .collect();
        let arg_terms = arg_terms?;

        match name {
            "str" => {
                if let Some(term) = arg_terms.first() {
                    Ok(Term::Literal(Literal::string(term.to_string())))
                } else {
                    Err(anyhow!("str() requires one argument"))
                }
            }
            "strlen" => {
                if let Some(Term::Literal(lit)) = arg_terms.first() {
                    Ok(Term::Literal(Literal::integer(lit.value.len() as i64)))
                } else {
                    Err(anyhow!("strlen() requires a literal argument"))
                }
            }
            "ucase" => {
                if let Some(Term::Literal(lit)) = arg_terms.first() {
                    Ok(Term::Literal(Literal::string(lit.value.to_uppercase())))
                } else {
                    Err(anyhow!("ucase() requires a literal argument"))
                }
            }
            "lcase" => {
                if let Some(Term::Literal(lit)) = arg_terms.first() {
                    Ok(Term::Literal(Literal::string(lit.value.to_lowercase())))
                } else {
                    Err(anyhow!("lcase() requires a literal argument"))
                }
            }
            _ => Err(anyhow!("Unknown function: {}", name))
        }
    }

    /// Evaluate binary operation
    fn evaluate_binary_operation(&self, op: &BinaryOperator, left: &Term, right: &Term) -> Result<Term> {
        match (left, right) {
            (Term::Literal(left_lit), Term::Literal(right_lit)) => {
                match op {
                    BinaryOperator::Add => {
                        if left_lit.is_numeric() && right_lit.is_numeric() {
                            let left_val: f64 = left_lit.value.parse().unwrap_or(0.0);
                            let right_val: f64 = right_lit.value.parse().unwrap_or(0.0);
                            Ok(Term::Literal(Literal::decimal(left_val + right_val)))
                        } else {
                            Err(anyhow!("Addition requires numeric operands"))
                        }
                    }
                    BinaryOperator::Subtract => {
                        if left_lit.is_numeric() && right_lit.is_numeric() {
                            let left_val: f64 = left_lit.value.parse().unwrap_or(0.0);
                            let right_val: f64 = right_lit.value.parse().unwrap_or(0.0);
                            Ok(Term::Literal(Literal::decimal(left_val - right_val)))
                        } else {
                            Err(anyhow!("Subtraction requires numeric operands"))
                        }
                    }
                    BinaryOperator::Multiply => {
                        if left_lit.is_numeric() && right_lit.is_numeric() {
                            let left_val: f64 = left_lit.value.parse().unwrap_or(0.0);
                            let right_val: f64 = right_lit.value.parse().unwrap_or(0.0);
                            Ok(Term::Literal(Literal::decimal(left_val * right_val)))
                        } else {
                            Err(anyhow!("Multiplication requires numeric operands"))
                        }
                    }
                    BinaryOperator::Divide => {
                        if left_lit.is_numeric() && right_lit.is_numeric() {
                            let left_val: f64 = left_lit.value.parse().unwrap_or(0.0);
                            let right_val: f64 = right_lit.value.parse().unwrap_or(1.0);
                            if right_val != 0.0 {
                                Ok(Term::Literal(Literal::decimal(left_val / right_val)))
                            } else {
                                Err(anyhow!("Division by zero"))
                            }
                        } else {
                            Err(anyhow!("Division requires numeric operands"))
                        }
                    }
                    BinaryOperator::Equal => {
                        Ok(Term::Literal(Literal::boolean(left_lit.value == right_lit.value)))
                    }
                    BinaryOperator::NotEqual => {
                        Ok(Term::Literal(Literal::boolean(left_lit.value != right_lit.value)))
                    }
                    BinaryOperator::Less => {
                        Ok(Term::Literal(Literal::boolean(left_lit.value < right_lit.value)))
                    }
                    BinaryOperator::LessEqual => {
                        Ok(Term::Literal(Literal::boolean(left_lit.value <= right_lit.value)))
                    }
                    BinaryOperator::Greater => {
                        Ok(Term::Literal(Literal::boolean(left_lit.value > right_lit.value)))
                    }
                    BinaryOperator::GreaterEqual => {
                        Ok(Term::Literal(Literal::boolean(left_lit.value >= right_lit.value)))
                    }
                    _ => Err(anyhow!("Unsupported binary operation"))
                }
            }
            _ => Err(anyhow!("Binary operations require literal operands"))
        }
    }

    /// Evaluate unary operation
    fn evaluate_unary_operation(&self, op: &UnaryOperator, operand: &Term) -> Result<Term> {
        match op {
            UnaryOperator::Not => {
                if let Term::Literal(lit) = operand {
                    if lit.is_boolean() {
                        let val = lit.value == "true";
                        Ok(Term::Literal(Literal::boolean(!val)))
                    } else {
                        Err(anyhow!("NOT operation requires boolean operand"))
                    }
                } else {
                    Err(anyhow!("NOT operation requires literal operand"))
                }
            }
            UnaryOperator::Plus => {
                if let Term::Literal(lit) = operand {
                    if lit.is_numeric() {
                        Ok(operand.clone())
                    } else {
                        Err(anyhow!("Plus operation requires numeric operand"))
                    }
                } else {
                    Err(anyhow!("Plus operation requires literal operand"))
                }
            }
            UnaryOperator::Minus => {
                if let Term::Literal(lit) = operand {
                    if lit.is_numeric() {
                        let val: f64 = lit.value.parse().unwrap_or(0.0);
                        Ok(Term::Literal(Literal::decimal(-val)))
                    } else {
                        Err(anyhow!("Minus operation requires numeric operand"))
                    }
                } else {
                    Err(anyhow!("Minus operation requires literal operand"))
                }
            }
            UnaryOperator::IsIri => {
                Ok(Term::Literal(Literal::boolean(matches!(operand, Term::Iri(_)))))
            }
            UnaryOperator::IsBlank => {
                Ok(Term::Literal(Literal::boolean(matches!(operand, Term::BlankNode(_)))))
            }
            UnaryOperator::IsLiteral => {
                Ok(Term::Literal(Literal::boolean(matches!(operand, Term::Literal(_)))))
            }
            UnaryOperator::IsNumeric => {
                if let Term::Literal(lit) = operand {
                    Ok(Term::Literal(Literal::boolean(lit.is_numeric())))
                } else {
                    Ok(Term::Literal(Literal::boolean(false)))
                }
            }
        }
    }
}

/// Function registry for custom SPARQL functions
pub struct FunctionRegistry {
    functions: HashMap<String, Box<dyn Fn(&[Term]) -> Result<Term> + Send + Sync>>,
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
}
